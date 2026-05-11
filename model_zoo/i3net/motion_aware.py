import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from flowseek.core.flowseek import FlowSeek
from .basic_model import FeatureExtractor
from .frequency_aware import PreNormResidual, FeedForward


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


class FlowDeformAttn(nn.Module):
    def __init__(self, dim, num_heads=4, num_points=4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        in_q = dim + 2
        self.sampling_offsets = nn.Conv2d(in_q, num_heads * num_points * 2, 3, 1, 1)
        self.attention_weights = nn.Conv2d(in_q, num_heads * num_points, 3, 1, 1)
        self.value_proj = nn.Conv2d(dim, dim, 1)
        self.output_proj = nn.Conv2d(dim, dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (grid_init / (grid_init.abs().max(-1, keepdim=True)[0] + 1e-6)).view(
            self.num_heads, 1, 1, 2
        ).repeat(1, 1, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, 0, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.data = grid_init.reshape(-1).to(self.sampling_offsets.bias.dtype)

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    @staticmethod
    def _sample_value_per_point(vh, grid_hw):
        B, heads, d, H, W = vh.shape
        P = grid_hw.shape[2]
        v_flat = vh.reshape(B * heads, d, H, W)
        outs = []
        for p in range(P):
            g = grid_hw[:, :, p].contiguous().reshape(B * heads, H, W, 2)
            outs.append(F.grid_sample(v_flat, g, padding_mode="border", align_corners=True))
        stacked = torch.stack(outs, dim=1)
        return stacked.view(B, heads, P, d, H, W)

    def forward(self, x1, x2, flow):
        """
        x1: query
        x2: key/value
        flow: [B,2,H,W]
        """
        B, C, H, W = x1.shape
        wm = max(W - 1, 1)
        hm = max(H - 1, 1)

        flow_norm = torch.cat(
            [flow[:, 0:1] / wm, flow[:, 1:2] / hm],
            dim=1,
        )
        qf = torch.cat([x1, flow_norm], dim=1)
        offsets = self.sampling_offsets(qf).view(B, self.num_heads, self.num_points, 2, H, W)
        attn_logits = self.attention_weights(qf).view(B, self.num_heads, self.num_points, H, W)
        attn = torch.softmax(attn_logits, dim=2)

        v = self.value_proj(x2).view(B, self.num_heads, self.head_dim, H, W)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=x1.device, dtype=torch.float32),
            torch.arange(W, device=x1.device, dtype=torch.float32),
            indexing="ij",
        )
        base_grid = torch.stack((xx, yy), dim=0).float().unsqueeze(0).unsqueeze(0)
        flow_b = flow.unsqueeze(1).unsqueeze(2)
        sampling_pix = base_grid + flow_b + offsets

        sampling_n = sampling_pix.clone()
        sampling_n[..., 0] = 2 * sampling_n[..., 0] / wm - 1
        sampling_n[..., 1] = 2 * sampling_n[..., 1] / hm - 1

        grid_hw = sampling_n.permute(0, 1, 2, 4, 5, 3).contiguous()
        sampled_v = self._sample_value_per_point(v, grid_hw)

        out = (attn.unsqueeze(3) * sampled_v).sum(dim=2).reshape(B, C, H, W)
        return self.output_proj(out)


class SpatialPreNormResidualFFN(nn.Module):

    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        self.block = PreNormResidual(
            dim,
            FeedForward(dim, expansion_factor=expansion_factor, dropout=0.0, dense=nn.Linear),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.permute(0, 2, 3, 1).contiguous().reshape(b, h * w, c)
        t = self.block(t)
        return t.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class DeformMotionBlock(nn.Module):
    def __init__(self, n_feats, num_heads=4, ffn_expansion=2, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale

        self.norm1 = nn.LayerNorm(1, n_feats)
        self.deform1 = FlowDeformAttn(n_feats, num_heads)
        self.norm2 = nn.LayerNorm(1, n_feats)
        self.deform2 = FlowDeformAttn(n_feats, num_heads)
        self.ffn = SpatialPreNormResidualFFN(n_feats, expansion_factor=ffn_expansion)

        self.fuse = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 1),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )

    def forward(self, f1, f2, flow):
        f2_warp = warp(f2, flow)
        x = self.fuse(torch.cat([f1, f2_warp], dim=1))
        x = x + self.res_scale * self.deform1(self.norm1(x), f2, flow)
        x = x + self.res_scale * self.deform2(self.norm2(x), f2, flow)
        x = x + self.res_scale * (self.ffn(x) - x)
        return x

class MotionAwareGroup(nn.Module):
    def __init__(self, args, n_feats, kernel_size, head_num=1, lambda_flow=10.0, window_size=16, num_blocks=16, n_depth=2):
        super().__init__()
        self.args = args

        self.encoder = FeatureExtractor(in_channels=1, out_channels=n_feats)
        self.flowseek = FlowSeek(args)
        motion = [
            DeformMotionBlock(n_feats, head_num) for _ in range(num_blocks // 2)]
        self.motion = nn.ModuleList(motion)

        self.temporal_fuse = nn.Sequential(
            nn.Conv3d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(n_feats, n_feats, 3, padding=1),
        )
    
    def _vol_to_flowseek_rgb(self, vol):  # vol: (B, 1, H, W) 或 (B, H, W)
        if vol.ndim == 3:
            vol = vol.unsqueeze(1)  # (B, 1, H, W)
        rgb = vol.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        return (rgb * 255.0).clamp(0.0, 255.0)
    
    def _get_flow(self, i_start, i_end):
        finetune = getattr(self.args, "finetune_flowseek", False)
        if self.training and finetune:
            self.flowseek.train()
            self.flowseek.dav2.eval()
        else:
            self.flowseek.eval()

        B, T, H, W = i_start.shape

        flows = []
        grad_enabled = self.training and finetune
        for i in range(T):
            img0 = i_start[:, i, :, :]
            img1 = i_end[:, i, :, :]
            img0 = self._vol_to_flowseek_rgb(img0)
            img1 = self._vol_to_flowseek_rgb(img1)

            forward_iters = getattr(self.args, "flowseek_forward_iters", 0)
            call_kw = {"test_mode": True}
            if forward_iters and forward_iters > 0:
                call_kw["iters"] = int(forward_iters)

            with torch.set_grad_enabled(grad_enabled):
                flow01 = self.flowseek(img0, img1, **call_kw)["final"]
                flows.append(flow01)

        flows = torch.cat(flows, dim=1)
        return flows
    
    def flow_local(self, x):
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]
        flows = self._get_flow(i_start, i_end)
        return flows
    
    def forward(self, x):
        B, T, H, W = x.shape
        flows = self.flow_local(x)
             
        # 1. feature
        feats = self.encoder(x.view(B*T,1,H,W))
        _, C, Hf, Wf = feats.shape
        feats = feats.view(B, T, C, Hf, Wf)

        outs = []
        for i in range(T-1):
            flow = flows[:, i*2:i*2+2]
            flow = F.interpolate(flow, size=(Hf,Wf), mode='bilinear')
            flow = flow / (H / Hf)
            f1 = feats[:, i, :, :]
            f2 = feats[:, i+1, :, :]
            res = f1
            for id, motion_block in enumerate(self.motion):
                res = motion_block(res, f2, flow)
            outs.append(res)
        
        outs = torch.stack(outs, dim=1)
        outs = outs.permute(0,2,1,3,4)  # B,C,T,H,W

        motion_feat = self.temporal_fuse(outs)
        return motion_feat.mean(2)
