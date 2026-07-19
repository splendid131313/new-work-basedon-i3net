import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]


def resize(volume: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Resize a 3D volume in XY plane, keeping slice dimension unchanged.

    Args:
        volume: numpy array of shape [H, W, S]
        out_h/out_w: target spatial size

    Returns:
        numpy array of shape [out_h, out_w, S]
    """
    if volume.ndim != 3:
        raise ValueError(f"resize_volume_xy expects [H,W,S], got shape: {volume.shape}")
    h, w, s = volume.shape
    if h == out_h and w == out_w:
        return volume

    # [H,W,S] -> [S,1,H,W]
    t = torch.from_numpy(volume).unsqueeze(0)  # [1,H,W,S]
    t = t.permute(3, 0, 1, 2).contiguous()     # [S,1,H,W]
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    # [S,1,out_h,out_w] -> [out_h,out_w,S]
    t = t.permute(2, 3, 0, 1).contiguous().squeeze(-1)
    return t.cpu().numpy()

def normalize(x, return_stats: bool = False):
    """
    Min-max normalize to [0, 1].

    If return_stats=True, also returns (vmin, vmax) for denormalization.
    """
    eps = 1e-8
    if isinstance(x, np.ndarray):
        vmax = x.max()
        vmin = x.min()
        y = (x - vmin) / (vmax - vmin + eps)
        return (y, float(vmin), float(vmax)) if return_stats else y
    else:  # torch.Tensor
        vmax = x.max()
        vmin = x.min()
        y = (x - vmin) / (vmax - vmin + eps)
        return (y, vmin, vmax) if return_stats else y


def denormalize(x, vmin, vmax):
    """Inverse of normalize(): x in [0,1] -> original scale."""
    return x * (vmax - vmin) + vmin

class RandomCrop3d(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples)
        return transform(samples)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample.astype(np.float32))

def prepare(args,gpu_id=0,precision='half'):
    device = torch.device(f'cuda:{gpu_id}')
    def _prepare(tensor):
        if precision == 'half': tensor = tensor.half()
        return tensor.to(device)
        
    # return [_prepare(a) for a in args]
    return _prepare(args)

import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def pkload(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0
        self.stderr = 0
        self.median = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)
        self.stderr = self.std / np.sqrt(self.count)
        self.median = np.median(self.vals)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class register_model(nn.Module):
    def __init__(self, img_size=(256, 256), mode="bilinear"):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0]
        flow = x[1]
        out = self.spatial_trans(img, flow)
        return out
