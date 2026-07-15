import torch
from torch import nn

from model_zoo.flow_module import warp


class Select_Loss(nn.Module):
    def __init__(self, args):
        super(Select_Loss, self).__init__()
        self.args = args
        self.l1loss = nn.L1Loss()
        self.ssim = SSIM()
        self.ssim_weight = float(getattr(args, "ssim_loss_weight", 0.0))
        self.charbonnier_eps = float(getattr(args, "charbonnier_eps", 1e-3))
        self.near_anchor_weight = float(getattr(args, "near_anchor_loss_weight", 0.0))
        self.near_anchor_threshold = float(getattr(args, "near_anchor_threshold", 0.35))
        self.mask_prior_weight = float(getattr(args, "mask_prior_loss_weight", 0.0))

    def _charbonnier(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.charbonnier_eps ** 2).mean()

    def _mask_prior_loss(self, mask, cond):
        t = cond[:, :1]
        mask_mean = mask.mean(dim=(2, 3))
        target_mask = 1.0 - t
        return torch.abs(mask_mean - target_mask).mean()

    def _near_anchor_loss(self, pred, img0, img1, gt, cond, flow_fn):
        t = cond[:, 0]
        left_mask = t <= self.near_anchor_threshold
        right_mask = t >= (1.0 - self.near_anchor_threshold)

        near_pred, near_target = [], []

        if left_mask.any():
            pred_l = pred[left_mask]
            gt_l = gt[left_mask]
            img0_l = img0[left_mask]
            flow_t0, _ = flow_fn(gt_l, img0_l)
            near_pred.append(warp(pred_l, -flow_t0))
            near_target.append(img0_l)

        if right_mask.any():
            pred_r = pred[right_mask]
            gt_r = gt[right_mask]
            img1_r = img1[right_mask]
            flow_t1, _ = flow_fn(gt_r, img1_r)
            near_pred.append(warp(pred_r, -flow_t1))
            near_target.append(img1_r)

        if not near_pred:
            return None

        return self._charbonnier(
            torch.cat(near_pred, dim=0),
            torch.cat(near_target, dim=0),
        )

    def forward(self, sr, gt, cond=None, lr=None, flow_fn=None):
        mask = None
        if isinstance(sr, dict):
            mask = sr.get("mask")
            sr = sr["out"]

        loss = self._charbonnier(sr, gt)
        if self.ssim_weight > 0 and sr.ndim == 4:
            sr_nchw = sr.permute(0, 3, 1, 2).contiguous()
            gt_nchw = gt.permute(0, 3, 1, 2).contiguous()
            loss = loss + self.ssim_weight * self.ssim(sr_nchw, gt_nchw).mean()

        if self.mask_prior_weight > 0 and mask is not None and cond is not None:
            loss = loss + self.mask_prior_weight * self._mask_prior_loss(mask, cond)

        if (
            self.near_anchor_weight > 0
            and flow_fn is not None
            and lr is not None
            and cond is not None
        ):
            pred = sr.permute(0, 3, 1, 2).contiguous()
            gt_nchw = gt.permute(0, 3, 1, 2).contiguous()
            lr_nchw = lr.permute(0, 3, 1, 2).contiguous()
            near_loss = self._near_anchor_loss(
                pred,
                lr_nchw[:, 0:1],
                lr_nchw[:, 1:2],
                gt_nchw,
                cond,
                flow_fn,
            )
            if near_loss is not None:
                loss = loss + self.near_anchor_weight * near_loss

        return loss


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
