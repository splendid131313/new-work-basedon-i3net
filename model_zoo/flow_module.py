import torch
import torch.nn as nn
import torch.nn.functional as F
from .flowseek.core.flowseek import FlowSeek

def warp(x, flow):
    B, C, H, W = x.size()
    xx = torch.linspace(-1.0, 1.0, W, device=x.device)
    yy = torch.linspace(-1.0, 1.0, H, device=x.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).expand(B, -1, -1, -1)

    flow_norm = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        dim=1,
    )
    grid = grid + flow_norm.permute(0, 2, 3, 1)

    return F.grid_sample(x, grid, padding_mode="border", align_corners=True)


class FlowEstimator(nn.Module):
    def __init__(self, args,n_feats, kernel_size):
        super(FlowEstimator, self).__init__()
        self.cond_dim = 2
        self.flowseek = FlowSeek(args)
        pad = kernel_size // 2
        self.flow_refine = nn.Sequential(
            nn.Conv2d(2 + self.cond_dim, n_feats, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv2d(n_feats, 2, kernel_size, padding=pad),  # 输出: 光流残差修正量
        )

    def _vol_to_rgb(self, vol):  # vol: (B, 1, H, W) 或 (B, H, W)
        if vol.ndim == 3:
            vol = vol.unsqueeze(1)  # (B, 1, H, W)
        rgb = vol.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        return (rgb * 255.0).clamp(0.0, 255.0)
    
    def _get_base_flow(self, img0, img1):
        img0 = self._vol_to_rgb(img0)
        img1 = self._vol_to_rgb(img1)

        with torch.no_grad():
            flow01 = self.flowseek(img0, img1, test_mode=True)["final"]
            flow10 = self.flowseek(img1, img0, test_mode=True)["final"]
        
        return flow01, flow10
    
    def _get_continuous_flow(self, img0, flow01, flow10, cond):
        B, _, H, W = img0.shape
        t = cond[:, :1]

        t_tensor = t.view(B, 1, 1, 1).expand(-1, -1, H, W)
        cond_tensor = cond.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W)

        # 基础线性流
        base_flow0t = flow01 * t.view(B, 1, 1, 1)
        base_flow1t = flow10 * (1.0 - t.view(B, 1, 1, 1))

        # 非线性修正
        cond_1t = cond_tensor.clone()
        cond_1t[:, :1] = 1.0 - t_tensor
        feat_0t = torch.cat([base_flow0t, cond_tensor], dim=1)
        feat_1t = torch.cat([base_flow1t, cond_1t], dim=1)

        delta_flow0t = self.flow_refine(feat_0t)
        delta_flow1t = self.flow_refine(feat_1t)

        # 最终非线性对齐光流
        final_flow0t = base_flow0t + delta_flow0t
        final_flow1t = base_flow1t + delta_flow1t

        return final_flow0t, final_flow1t
    
    def forward(self, img0, img1, cond):
        flow01, flow10 = self._get_base_flow(img0, img1)
        final_flow0t, final_flow1t = self._get_continuous_flow(img0, flow01, flow10, cond)
        warped0 = warp(img0, -final_flow0t)
        warped1 = warp(img1, -final_flow1t)
        return warped0, warped1, final_flow0t, final_flow1t