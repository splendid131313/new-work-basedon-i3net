import torch
from torch import nn
import torch.nn.functional as F

class Select_Loss(nn.Module):
    def __init__(self, args):
        super(Select_Loss, self).__init__()
        self.args = args
        self.l1loss = nn.L1Loss()
        self.ssim = SSIM()
        self.ssim_weight = float(getattr(args, "ssim_loss_weight", 0.0))
        self.charbonnier_eps = float(getattr(args, "charbonnier_eps", 1e-3))
        self.near_anchor_weight = float(getattr(args, "near_anchor_loss_weight", 0.0))
        self.mask_prior_weight = float(getattr(args, "mask_prior_loss_weight", 0.0))

    def _charbonnier(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.charbonnier_eps ** 2).mean()

    def forward(self, sr, gt):
        aux = None
        if isinstance(sr, dict):
            aux = sr.get("aux")
            sr = sr["out"]

        loss = self._charbonnier(sr, gt)
        if self.ssim_weight > 0 and sr.ndim == 4:
            sr_nchw = sr.permute(0, 3, 1, 2).contiguous()
            gt_nchw = gt.permute(0, 3, 1, 2).contiguous()
            loss = loss + self.ssim_weight * self.ssim(sr_nchw, gt_nchw).mean()
        if self.near_anchor_weight > 0 and aux is not None and "near_pred" in aux:
            loss = loss + self.near_anchor_weight * self._charbonnier(
                aux["near_pred"], aux["near_target"]
            )
        if self.mask_prior_weight > 0 and aux is not None and "mask_prior" in aux:
            mp = aux["mask_prior"]
            loss = loss + self.mask_prior_weight * torch.abs(
                mp["mask_mean"] - mp["target_mask"]
            ).mean()
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

