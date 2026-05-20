import torch
from torch import nn
import torch.fft
import torch.nn.functional as F
from torch.cuda.amp import autocast


def _rfft2_magnitude(x):
    """x: (N, H, W) float32 contiguous. FFT on CPU to avoid cuFFT issues."""
    x_fft = torch.fft.rfft2(x.cpu(), norm='ortho')
    return torch.abs(x_fft).to(device=x.device, dtype=x.dtype)


# class Select_Loss(nn.Module):
#     def __init__(self, args):
#         super(Select_Loss, self).__init__()
#         self.args = args
#         self.l1loss = nn.L1Loss()

#     def forward(self, sr, gt):
#         l1loss = self.l1loss(sr, gt)
#         return l1loss


class HighFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self._weight_key = None
        self.register_buffer('_freq_weight', torch.empty(0), persistent=False)

    def _get_freq_weight(self, H, W, device, dtype):
        key = (H, W, device)
        if self._weight_key != key:
            y = torch.linspace(0, 1, H, device=device, dtype=dtype).view(H, 1)
            x = torch.linspace(0, 1, W // 2 + 1, device=device, dtype=dtype).view(1, W // 2 + 1)
            weight = torch.sqrt(x ** 2 + y ** 2)
            weight = weight / weight.max()
            self._freq_weight = weight.unsqueeze(0).unsqueeze(0)
            self._weight_key = key
        return self._freq_weight

    def forward(self, pred, target):
        """
        pred/target: [B,C,H,W]
        """
        pred = pred.float().contiguous()
        target = target.float().contiguous()

        B, C, H, W = pred.shape
        n = B * C

        pred_mag = _rfft2_magnitude(pred.view(n, H, W))
        target_mag = _rfft2_magnitude(target.view(n, H, W))

        weight = self._get_freq_weight(H, W, pred.device, pred.dtype)
        loss = torch.mean(weight * torch.abs(pred_mag - target_mag))

        return self.loss_weight * loss


class GradientLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

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
        return self.loss_weight * (loss_x + loss_y)


class TotalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lambda_l1 = args.lambda_l1
        lambda_fre = args.lambda_fre
        lambda_gra = args.lambda_gra

        self.l1 = nn.L1Loss()
        self.freq_loss = HighFrequencyLoss(loss_weight=lambda_fre)
        self.grad_loss = GradientLoss(loss_weight=lambda_gra)

    def forward(self, pred, target):
        with autocast(enabled=False):
            pred = pred.float().permute(0, 3, 1, 2).contiguous()
            target = target.float().permute(0, 3, 1, 2).contiguous()

            l1 = self.l1(pred, target) * self.lambda_l1
            freq = self.freq_loss(pred, target)
            grad = self.grad_loss(pred, target)
            return l1 + freq + grad
