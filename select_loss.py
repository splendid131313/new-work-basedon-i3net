import torch
from torch import nn
import torch.nn.functional as F

class Select_Loss(nn.Module):
    def __init__(self, args):
        super(Select_Loss, self).__init__()
        self.args = args
        self.l1loss = nn.L1Loss()
        self.ssim = SSIM()
        self.ssim_weight = float(getattr(args, "ssim_loss_weight", 0.1))
        self.charbonnier_eps = float(getattr(args, "charbonnier_eps", 1e-3))

    def forward(self, sr, gt):
        charbonnier = torch.sqrt((sr - gt) ** 2 + self.charbonnier_eps ** 2).mean()
        loss = charbonnier
        if self.ssim_weight > 0 and sr.ndim == 4:
            sr_nchw = sr.permute(0, 3, 1, 2).contiguous()
            gt_nchw = gt.permute(0, 3, 1, 2).contiguous()
            loss = loss + self.ssim_weight * self.ssim(sr_nchw, gt_nchw).mean()
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

def compute_consistency_loss(pred_local, pred_global):
    B, H, W, T = pred_local.shape

    loss_cons = F.l1_loss(pred_local[..., T//2], pred_global[..., T//2])

    return loss_cons
