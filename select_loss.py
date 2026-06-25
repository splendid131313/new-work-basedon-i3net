import torch
from torch import nn

class Select_Loss(nn.Module):
    def __init__(self,args):
        super(Select_Loss,self).__init__()
        self.args = args
        self.l1loss  = nn.L1Loss()

    def forward(self,sr,gt):
        l1loss = self.l1loss(sr,gt)
        loss = l1loss 
        return loss


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim = SSIM().to(pred.device, pred.dtype)
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss.mean()


import torch
from torch import nn
import torch.fft
import torch.nn.functional as F
from torch.cuda.amp import autocast

def _edge_aware_smoothness_single(flow, image):
    """flow/image: (B, C, H, W)"""
    flow_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    flow_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

    img_dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    img_dy = image[:, :, 1:, :] - image[:, :, :-1, :]

    weight_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True))

    loss_x = (flow_dx.abs() * weight_x).mean()
    loss_y = (flow_dy.abs() * weight_y).mean()

    return loss_x + loss_y


def edge_aware_smoothness(flow_list, lr):
    """Edge-aware flow smoothness for lr-1 consecutive flows.

    Args:
        flow_list: list of (B, 2, H, W), len = lr_slice - 1
        lr: (B, H, W, S) low-res volume, S = lr_slice_patch
    """
    if not isinstance(flow_list, (list, tuple)):
        flow_list = [flow_list]

    loss = 0.0
    for i, flow in enumerate(flow_list):
        # flow i: lr[..., i] -> lr[..., i+1]; use source slice as reference
        image = lr[..., i].unsqueeze(1)
        loss = loss + _edge_aware_smoothness_single(flow, image)

    return loss / max(len(flow_list), 1)

def gauss_kernel(channels=3, device=None):
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ]
    )
    kernel /= 256.0
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat(
        [x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(x.device)],
        dim=3,
    )
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat(
        [
            cc,
            torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2).to(x.device),
        ],
        dim=3,
    )
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr


class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=7, weights=[1.0, 0.5, 0.25], device=None):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)
        self.weights = weights

    def forward(self, input, target):
        loss = 0.0
        pyr_input = laplacian_pyramid(
            img=input, kernel=self.gauss_kernel, max_levels=self.max_levels
        )
        pyr_target = laplacian_pyramid(
            img=target, kernel=self.gauss_kernel, max_levels=self.max_levels
        )

        for i, (a, b) in enumerate(zip(pyr_input, pyr_target)):
            loss += self.weights[i] * F.l1_loss(a, b)
        return loss


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float()

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1]
        ]).float()

        self.register_buffer('weight_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('weight_y', sobel_y.view(1, 1, 3, 3))

    def gradient(self, x):
        B, C, H, W = x.shape
        weight_x = self.weight_x.to(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
        weight_y = self.weight_y.to(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)

        grad_x = F.conv2d(x, weight_x, padding=1, groups=C)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=C)
        return grad_x, grad_y

    def forward(self, pred, target):
        pred_grad_x, pred_grad_y = self.gradient(pred)
        target_grad_x, target_grad_y = self.gradient(target)

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return loss_x + loss_y


class TotalLoss(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()

        self.lambda_l1 = args.lambda_l1
        self.lambda_lap = args.lambda_lap
        self.lambda_gra = args.lambda_gra

        self.l1 = nn.L1Loss()
        self.lap_loss = LapLoss(channels=args.hr_slice_patch, device=device)
        self.grad_loss = GradientLoss()


    def forward(self, pred, target):
        with autocast(enabled=False):
            pred = pred.float().permute(0, 3, 1, 2).contiguous()
            target = target.float().permute(0, 3, 1, 2).contiguous()

            l1 = self.l1(pred, target) * self.lambda_l1
            freq = self.lap_loss(pred, target) * self.lambda_lap
            grad = self.grad_loss(pred, target) * self.lambda_gra
            return l1 + freq + grad
