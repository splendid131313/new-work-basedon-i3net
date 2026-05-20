import torch
from torch import nn
import torch.fft
from pytorch_msssim import ssim

class Select_Loss(nn.Module):
    def __init__(self,args):
        super(Select_Loss,self).__init__()
        self.args = args
        self.l1loss  = nn.L1Loss()

    def forward(self,sr,gt):
        l1loss = self.l1loss(sr,gt)
        loss = l1loss 
        return loss

class MedLoss(nn.Module):
    def __init__(
        self, args
    ):
        super().__init__()

        self.args = args
        self.lambda_l1 = args.lambda_l1
        self.lambda_ssim = args.lambda_ssim
        self.lambda_freq = args.lambda_freq
        self.lambda_motion = args.lambda_motion

        self.l1 = nn.L1Loss()

    def frequency_loss(self, pred, gt):
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        gt_fft = torch.fft.rfft2(gt, dim=(-2, -1), norm="ortho")

        pred_amp = torch.abs(pred_fft)
        gt_amp = torch.abs(gt_fft)

        return self.l1(pred_amp, gt_amp)
    
    def sequence_ssim_loss(self, pred, gt):

        # pred: [B,T,H,W]

        B, T, H, W = pred.shape

        total = 0

        for t in range(1, T, self.args.upscale):

            total += 1 - ssim(
                pred[:, t:t+1],
                gt[:, t:t+1],
                data_range=1.0,
                size_average=True
            )

        return total / (self.args.lr_slice_patch - 1)

    def forward(self, pred, gt, motion_loss=None):
        pred = pred.permute(0, 3, 1, 2)
        gt = gt.permute(0, 3, 1, 2)

        # L1
        if self.lambda_l1 > 0:
            loss_l1 = self.l1(pred, gt)

        # SSIM
        if self.lambda_ssim > 0:    
            loss_ssim = self.sequence_ssim_loss(pred, gt)

        # Frequency
        if self.lambda_freq > 0:
            loss_freq = self.frequency_loss(pred, gt)

        total_loss = 0
        if self.lambda_l1 > 0:
            total_loss += self.lambda_l1 * loss_l1
        if self.lambda_ssim > 0:
            total_loss += self.lambda_ssim * loss_ssim
        if self.lambda_freq > 0:
            total_loss += self.lambda_freq * loss_freq

        if motion_loss is not None:
            total_loss += self.lambda_motion * motion_loss

        return total_loss