import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
import torch

def sampling_generator(N: int, reverse: bool = False):
    samplings = [False, True] * (N // 2)
    samplings = samplings[:N]
    return list(reversed(samplings)) if reverse else samplings

class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        upsampling=False,
        act_norm=False,
        act_inplace=True,
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels,
                        out_channels * 4,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.PixelShuffle(2),
                ]
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        downsampling=False,
        upsampling=False,
        act_norm=True,
        act_inplace=True,
    ):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=kernel_size,
            stride=stride,
            upsampling=upsampling,
            padding=padding,
            act_norm=act_norm,
            act_inplace=act_inplace,
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class Encoder(nn.Module):
    """Spatial encoder with interleaved downsampling stages."""

    def __init__(
        self, c_in: int, c_hid: int, n_s: int, k: int, act_inplace: bool = False
    ):
        super().__init__()
        samplings = sampling_generator(n_s)
        self.enc = nn.Sequential(
            ConvSC(c_in, c_hid, k, downsampling=samplings[0], act_inplace=act_inplace),
            *[
                ConvSC(c_hid, c_hid, k, downsampling=s, act_inplace=act_inplace)
                for s in samplings[1:]
            ],
        )

    def forward(self, x: torch.Tensor):
        x0 = self.enc[0](x)
        z = x0
        for i in range(1, len(self.enc)):
            z = self.enc[i](z)
        return z, x0

class Decoder(nn.Module):
    """Spatial decoder with symmetric upsampling stages and a residual skip."""

    def __init__(
        self, c_hid: int, c_out: int, n_s: int, k: int, act_inplace: bool = False
    ):
        super().__init__()
        samplings = sampling_generator(n_s, reverse=True)
        self.dec = nn.Sequential(
            *[
                ConvSC(c_hid, c_hid, k, upsampling=s, act_inplace=act_inplace)
                for s in samplings[:-1]
            ],
            ConvSC(c_hid, c_hid, k, upsampling=samplings[-1], act_inplace=act_inplace),
        )
        self.readout = nn.Conv2d(c_hid, c_out, kernel_size=1)

    def forward(self, z: torch.Tensor, skip: torch.Tensor):
        for i in range(len(self.dec) - 1):
            z = self.dec[i](z)
        y = self.dec[-1](z + skip)
        return self.readout(y)
        

class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2 style)
    x: (B, C, H, W) -> y = x + gamma * (x / ||x||_2) + beta
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))  # learnable scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))  # learnable bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # L2 over spatial dims
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta


class _RepDWLite(nn.Module):
    """
    Lightweight re-parameterized depthwise composition:
    DW(1xK) -> DW(Kx1) + DW(3x3) + DW(1x1 ~ identity)
    """

    def __init__(self, dim: int, K: int, stride: int = 1):
        super().__init__()

        # separable large-kernel approximation
        self.dw_h = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, K),
            stride=(1, stride),
            padding=(0, K // 2),
            groups=dim,
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            dim,
            dim,
            kernel_size=(K, 1),
            stride=(stride, 1),
            padding=(K // 2, 0),
            groups=dim,
            bias=False,
        )

        # complementary 3x3 + identity-like DW(1x1)
        self.dw_s = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            groups=dim,
            bias=False,
        )
        self.dw_i = nn.Conv2d(
            dim, dim, kernel_size=1, stride=stride, groups=dim, bias=False
        )
        nn.init.dirac_(self.dw_i.weight)  # start as identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_v(self.dw_h(x)) + self.dw_s(x) + self.dw_i(x)


class PFGA(nn.Module):
    """
    Peripheral-Frequency Guided Aggregation (token mixer):
    - Multiple large-kernel depthwise branches (peripheral)
    - Pixel-wise frequency gating (Sobel/Laplacian/variance cues)
    - Optional center suppression
    """

    class Branch(nn.Module):
        def __init__(self, dim: int, K: int, center_suppress: bool = True):
            super().__init__()
            self.center_suppress = center_suppress

            # approximate KxK with DW(1xK) + DW(Kx1)
            self.dw_h = nn.Conv2d(
                dim,
                dim,
                kernel_size=(1, K),
                padding=(0, K // 2),
                groups=dim,
                bias=False,
            )
            self.dw_v = nn.Conv2d(
                dim,
                dim,
                kernel_size=(K, 1),
                padding=(K // 2, 0),
                groups=dim,
                bias=False,
            )

            # optional 3x3 center path for suppression
            if self.center_suppress:
                self.dw_c = nn.Conv2d(
                    dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
                )
                self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            else:
                self.register_parameter("beta", None)
                self.dw_c = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.dw_v(self.dw_h(x))
            if self.center_suppress:
                center = self.dw_c(x)
                y = y - torch.tanh(self.beta) * center  # explicit center suppression
            return y

    def __init__(
        self,
        dim: int,
        K_list=(9, 15, 31),
        use_grn: bool = False,
        center_suppress: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.K_list = K_list

        # multi-scale peripheral branches
        self.branches = nn.ModuleList(
            [PFGA.Branch(dim, K, center_suppress=center_suppress) for K in K_list]
        )

        # fixed frequency filters (buffers): Sobel x/y + Laplacian
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        laplace = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplace", laplace, persistent=False)

        # 1x1 conv to produce per-scale gating logits
        self.gate_head = nn.Conv2d(3, len(K_list), kernel_size=1, bias=True)

        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(dim)

    # depthwise apply fixed 3x3 kernels to all channels
    def _depthwise_filter(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = k.repeat(C, 1, 1, 1)
        return F.conv2d(x, w, padding=1, groups=C)

    # build frequency maps: gradient magnitude, Laplacian magnitude, local variance
    def _freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)

        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        mean = F.avg_pool2d(x, 3, 1, 1)
        mean2 = F.avg_pool2d(x * x, 3, 1, 1)
        var = torch.clamp(mean2 - mean * mean, min=0.0)

        f1 = grad_mag.mean(dim=1, keepdim=True)
        f2 = lap.abs().mean(dim=1, keepdim=True)
        f3 = var.mean(dim=1, keepdim=True)
        return torch.cat([f1, f2, f3], dim=1)  # (B,3,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # peripheral responses at multiple scales
        peris = [b(x) for b in self.branches]

        # per-pixel softmax over scales from frequency cues
        Freq = self._freq_maps(x)
        logits = self.gate_head(Freq)
        alpha = torch.softmax(logits, dim=1)  # (B,K,H,W)

        # pixel-wise fusion
        Y = 0.0
        for i, y in enumerate(peris):
            Y = Y + y * alpha[:, i : i + 1, :, :]

        if self.use_grn:
            Y = self.grn(Y)
        return Y


class MSInit(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_list=(3, 5, 7),
        stride: int = 1,
        use_gn: bool = True,
    ):
        super().__init__()
        # equal split for branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    _RepDWLite(in_ch, K=k, stride=stride),
                    nn.Conv2d(in_ch, out_ch // len(k_list), 1, bias=False),
                )
                for k in k_list
            ]
        )

        # handle remainder channels
        gap = out_ch - (out_ch // len(k_list)) * len(k_list)
        self.tail = (
            nn.Identity()
            if gap == 0
            else nn.Sequential(
                _RepDWLite(in_ch, K=k_list[0], stride=stride),
                nn.Conv2d(in_ch, gap, 1, bias=False),
            )
        )

        self.fuse = nn.Identity()
        self.norm = nn.GroupNorm(1, out_ch) if use_gn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.branches]
        if not isinstance(self.tail, nn.Identity):
            parts.append(self.tail(x))
        y = torch.cat(parts, dim=1)
        return self.act(self.norm(y))


class PFG(nn.Module):
    """
    Main PFG block:
    - Token mixing by PFGA (peripheral + frequency gating)
    - Channel mixing by GLU-like depthwise-MLP (1x1 -> DW -> 1x1)
    - LayerScale + DropPath keep identical to original
    """

    def __init__(
        self,
        dim: int,
        groups_pw: int = 1,
        layerscale_init: float = 1e-6,
        act_layer=nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        pfga_K=(9, 15, 31),
        mlp_ratio: float = 4.0,
        dw_kernel: int = 3,
    ):
        super().__init__()
        self.dim = dim

        # lightweight norms before token/channel mixers
        self.norm_dw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.norm_pw = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)

        # token mixing by PFGA
        self.tm = PFGA(dim, K_list=pfga_K, use_grn=False)

        # GRN after each stage
        self.grn_dw = GRN(dim)
        self.grn_pw = GRN(dim)

        self.mlp_ratio = mlp_ratio
        self.dw_kernel = dw_kernel

        # GLU-like channel mixing
        E = max(dim, int(dim * self.mlp_ratio))

        self.pw_in = nn.Conv2d(dim, 2 * E, kernel_size=1, bias=True, groups=groups_pw)
        self.dw_v = nn.Conv2d(
            E, E, kernel_size=self.dw_kernel, padding=1, groups=E, bias=False
        )
        self.pw_out = nn.Conv2d(E, dim, kernel_size=1, bias=True, groups=groups_pw)

        self.act = act_layer()

        # LayerScale (kept exactly as original)
        self.gamma_dw = nn.Parameter(torch.ones(dim) * layerscale_init)
        self.gamma_pw = nn.Parameter(torch.ones(dim) * layerscale_init)

        from timm.layers import DropPath

        self.dropout_dw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.dropout_pw = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self._init_params()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"gamma_dw", "gamma_pw"}

    def _init_params(self):
        # standard conv/norm init, identical to original
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) token mixing
        y = self.norm_dw(x)
        y = self.tm(y)
        y = self.act(y)
        y = self.grn_dw(y)
        y = self.dropout_dw(y)
        x = x + self.drop_path(y * self.gamma_dw.view(1, self.dim, 1, 1))

        # (2) channel mixing (GLU style)
        z = self.norm_pw(x)
        uv = self.pw_in(z)
        u, v = torch.chunk(uv, 2, dim=1)
        v = self.dw_v(v)
        z = F.silu(u) * v
        z = self.pw_out(z)

        z = self.grn_pw(z)
        z = self.dropout_pw(z)
        x = x + self.drop_path(z * self.gamma_pw.view(1, self.dim, 1, 1))
        return x


class MidPFG(nn.Module):
    """PFG translator operating on the flattened (T*C) channel space."""

    def __init__(
        self,
        in_ch: int,
        depth: int,
        groups_pw: int = 1,
        layerscale_init: float = 1e-6,
        cel_k=(3, 5, 7),
        drop: float = 0.0,
        drop_path: float = 0.0,
        pfga_K=(9, 15, 31),
    ):
        super().__init__()
        self.cel = MSInit(in_ch, in_ch, k_list=cel_k, use_gn=True)

        if drop_path > 0 and depth > 0:
            dpr = [x.item() for x in torch.linspace(1e-2, drop_path, depth // 2)]
        else:
            dpr = [0.0] * (depth // 2)

        self.blocks = nn.Sequential(
            *[
                PFG(
                    in_ch,
                    groups_pw=groups_pw,
                    layerscale_init=layerscale_init,
                    act_layer=nn.GELU,
                    drop=drop,
                    drop_path=(drop_path if i == depth // 2 - 1 else dpr[i]),
                    pfga_K=pfga_K,
                )
                for i in range(depth // 2)
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = z.shape
        x = z.view(b, t * c, h, w)
        x = self.cel(x)
        x = self.blocks(x)
        return x.view(b, t, c, h, w)