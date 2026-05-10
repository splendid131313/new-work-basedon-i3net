import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from flowseek.core.flowseek import FlowSeek
from .basic_model import window_partitions, window_reverses, FeatureExtractor

def _pad_spatial(x, window_size):
    """Pad H,W to multiples of window_size (bottom/right). Returns padded x and original (H, W)."""
    _, _, H, W = x.shape
    Hp = math.ceil(H / window_size) * window_size
    Wp = math.ceil(W / window_size) * window_size
    ph, pw = Hp - H, Wp - W
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (H, W)


def _crop_spatial(x, orig_hw):
    H, W = orig_hw
    return x[:, :, :H, :W]


def _roll_spatial(x, shift_h, shift_w):
    """Cyclic roll on dims H,W (dims 2,3). shift_h, shift_w >= 0 typical for inverse."""
    if shift_h == 0 and shift_w == 0:
        return x
    return torch.roll(x, shifts=(-shift_h, -shift_w), dims=(2, 3))


def _unroll_spatial(x, shift_h, shift_w):
    if shift_h == 0 and shift_w == 0:
        return x
    return torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))


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


class MotionAware(nn.Module):
    """
    Windowed cross-frame attention with flow bias (medium-weight vs full spatial N*N).
    Q from x1, K/V from x2; flow biases attention logits toward correspondences along optical flow.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        lambda_flow=10.0,
        window_size=16,
        shift_size=0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lambda_flow = lambda_flow
        self.window_size = window_size
        self.shift_size = shift_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)

        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x1, x2, flow):
        """
        x1: [B, C, H, W]
        x2: [B, C, H, W]
        flow: [B, 2, H, W]
        """
        B, C, H0, W0 = x1.shape
        if C != self.dim:
            raise ValueError(f"channel dim {C} != MotionAware.dim {self.dim}")
        ws = self.window_size
        M = ws * ws

        x1p, orig_hw = _pad_spatial(x1, ws)
        x2p, _ = _pad_spatial(x2, ws)
        flowp, _ = _pad_spatial(flow, ws)
        _, _, H, W = x1p.shape

        yy, xx = torch.meshgrid(
            torch.arange(H, device=x1.device, dtype=torch.float32),
            torch.arange(W, device=x1.device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]  (x_col, y_row)

        qf = self.q_proj(x1p)
        kf = self.k_proj(x2p)
        vf = self.v_proj(x2p)

        grid_hw = grid.permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]
        sh = sw = int(self.shift_size)
        if sh or sw:
            qf = _roll_spatial(qf, sh, sw)
            kf = _roll_spatial(kf, sh, sw)
            vf = _roll_spatial(vf, sh, sw)
            flowp = _roll_spatial(flowp, sh, sw)
            grid_hw = _roll_spatial(grid_hw, sh, sw)

        win_q = window_partitions(qf, ws)
        win_k = window_partitions(kf, ws)
        win_v = window_partitions(vf, ws)
        win_grid = window_partitions(grid_hw, ws)
        win_flow = window_partitions(flowp, ws)

        nW = win_q.shape[0]
        win_grid = win_grid.view(nW, 2, M).transpose(1, 2)  # [nW, M, 2]
        win_flow = win_flow.view(nW, 2, M).transpose(1, 2)  # [nW, M, 2]

        q = win_q.view(nW, C, M).transpose(1, 2).reshape(nW, M, self.num_heads, self.head_dim).transpose(1, 2)
        k = win_k.view(nW, C, M).transpose(1, 2).reshape(nW, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = win_v.view(nW, C, M).transpose(1, 2).reshape(nW, M, self.num_heads, self.head_dim).transpose(1, 2)
        # q,k,v: [nW, heads, M, Dh]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        target = win_grid + win_flow
        dist = (target.unsqueeze(2) - win_grid.unsqueeze(1)) ** 2
        dist = dist.sum(-1)
        bias = -self.lambda_flow * torch.sqrt(dist + 1e-6)
        bias = bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        attn = attn + bias
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(nW, M, C)
        out = out.transpose(1, 2).view(nW, C, ws, ws)
        out = window_reverses(out, ws, H, W)
        if sh or sw:
            out = _unroll_spatial(out, sh, sw)
        out = _crop_spatial(out, orig_hw)
        out = self.out_proj(out)

        return x1 + out

class MotionAwareBlock(nn.Module):
    def __init__(self, n_feats, head_num=1, lambda_flow=10.0, window_size=16, n_depth=2):
        super().__init__()

        body = [MotionAware(dim=n_feats, num_heads=head_num, lambda_flow=lambda_flow, window_size=window_size, shift_size=0) for _ in range(n_depth)]
        self.body = nn.ModuleList(body)

        body_shifted = [MotionAware(dim=n_feats, num_heads=head_num, lambda_flow=lambda_flow, window_size=window_size, shift_size=window_size // 2) for _ in range(n_depth)]
        self.body_shifted = nn.ModuleList(body_shifted)
    
    def forward(self, f1, f2, flow):
        res = self.body[0](f1, f2, flow)
        res = self.body_shifted[0](res, f2, flow)

        for layer, layer_shifted in zip(self.body[1:], self.body_shifted[1:]):
            res = layer(res, f2, flow)
            res = layer_shifted(res, f2, flow)

        return res

class MotionAwareGroup(nn.Module):
    def __init__(self, args, n_feats, kernel_size, head_num=1, lambda_flow=10.0, window_size=16, num_blocks=16, n_depth=2):
        super().__init__()
        self.args = args

        self.encoder = FeatureExtractor(in_channels=1, out_channels=n_feats)
        self.flowseek = FlowSeek(args)
        motion = [
            MotionAwareBlock(n_feats=n_feats, head_num=head_num,
            lambda_flow=lambda_flow, window_size=window_size, n_depth=n_depth) for _ in range(num_blocks // 2 - 2)]
        self.motion = nn.ModuleList(motion)

        self.temporal_fuse = nn.Sequential(
            nn.Conv3d(n_feats, n_feats, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(n_feats, n_feats, 3, padding=1),
        )
    
    def _vol_to_flowseek_rgb(self, vol):  # vol: (B, 1, H, W) 或 (B, H, W)
        if vol.ndim == 3:
            vol = vol.unsqueeze(1)  # (B, 1, H, W)
        rgb = vol.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        return (rgb * 255.0).clamp(0.0, 255.0)
    
    def _get_flow(self, i_start, i_end):
        self.flowseek.eval()
        B, T, H, W = i_start.shape

        flows = []
        for i in range(T):
            img0 = i_start[:, i, :, :]
            img1 = i_end[:, i, :, :]
            img0 = self._vol_to_flowseek_rgb(img0)
            img1 = self._vol_to_flowseek_rgb(img1)

            with torch.no_grad():
                flow01 = self.flowseek(img0, img1, test_mode=True)["final"]
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
        collect = []
        for i in range(T-1):
            col = []
            flow = flows[:, i*2:i*2+2]
            flow = F.interpolate(flow, size=(Hf,Wf), mode='bilinear')
            flow = flow / (H / Hf)
            f1 = feats[:, i, :, :]
            f2 = feats[:, i+1, :, :]
            res = f1
            for id, motion_block in enumerate(self.motion):
                res = motion_block(res, f2, flow)
                if id in [1, 3, 5]:
                    col.append(res)
            outs.append(res)
            col = torch.stack(col, dim=1)
            col = self.temporal_fuse(col.permute(0,2,1,3,4))
            collect.append(col.mean(2))
        
        outs = torch.stack(outs, dim=1)
        outs = outs.permute(0,2,1,3,4)  # B,C,T,H,W

        motion_feat = self.temporal_fuse(outs)
        collect.append(motion_feat.mean(2))
        collect = torch.stack(collect, dim=1)
        return collect
