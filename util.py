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

# def normalize(slice):
#     ma,mi=4095 , 0
#     slice = (slice - mi)/(ma - mi)
#
#     return slice


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

