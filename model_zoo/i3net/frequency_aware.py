import torch.nn as nn
import torch
import einops
from functools import partial

from .dct_util import DCT2x, IDCT2x
from .basic_model import window_partitions, window_reverses, default_conv, CrossViewBlock


pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
    )


class FrequencyAware(nn.Module):
    def __init__(
        self,
        conv=nn.Conv2d,
        n_feats=64,
        bias=True,
        win_num_sqrt=16,
        window_size=16,
    ):
        super().__init__()

        self.window_size = window_size
        self.win_num_sqrt = win_num_sqrt

        self.dct = DCT2x()
        self.norm = nn.LayerNorm(n_feats)
        self.conv = nn.Sequential(
            conv(n_feats, n_feats, 1, bias=bias),
            conv(n_feats, n_feats, 3, 1, 1, bias=bias),
            conv(n_feats, n_feats, 3, 1, 1, bias=bias),
        )
        self.idct = IDCT2x()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.attn = nn.Sequential(
            PreNormResidual(
                dim=n_feats,
                fn=FeedForward(
                    dim=window_size**2, expansion_factor=1, dropout=0, dense=chan_first
                ),
            ),  # dim=num_patch
            PreNormResidual(
                dim=n_feats,
                fn=FeedForward(
                    dim=n_feats, expansion_factor=2, dropout=0, dense=chan_last
                ),
            ),  # dim=h*w*c
        )
        self.last_conv = conv(n_feats, n_feats, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x_dct = self.dct(x)
        x_dct = einops.rearrange(x_dct, "b c h w -> b (h w) c")
        x_dct = self.norm(x_dct)
        x_dct = einops.rearrange(x_dct, "b (h w) c -> b c h w", h=h, w=w)
        x_dct = self.conv(x_dct)

        x_dct_windows = window_partitions(
            x_dct, window_size=self.window_size
        )  # [b,c,h,w]

        bi, ci, hi, wi = x_dct_windows.shape
        x_dct_windows = einops.rearrange(x_dct_windows, "b c h w -> b (h w) c")
        x_dct_windows_attn = self.attn(x_dct_windows)
        x_dct_windows = x_dct_windows + x_dct_windows_attn
        x_dct_windows = einops.rearrange(
            x_dct_windows, "b (h w) c -> b c h w", h=hi, w=wi
        )

        x_dct_attn = window_reverses(
            x_dct_windows, window_size=self.window_size, H=h, W=w
        )

        x_dct_idct = self.idct(x_dct_attn)
        x_attn = self.last_conv(x_dct_idct)

        return x_attn

class FrequencyAwareBlock(nn.Module):
    def __init__(self, conv=nn.Conv2d, in_slice=4, n_feats=64, kernel_size=3, bias=True, 
                n_depth=2, head_num=1, win_num_sqrt=16, window_size=16):
        super().__init__()
        body = [FrequencyAware(conv=conv, n_feats=n_feats, bias=bias,
        win_num_sqrt=win_num_sqrt, window_size=window_size) for _ in range(n_depth)]
        self.body = nn.ModuleList(body)
    
    def forward(self, x):
        res = self.body[0](x)
        res = res + x
        for layer in self.body[1:]:
            res = layer(res)
            res = res + x

        return res

class FrequencyAwareGroup(nn.Module):
    def __init__(self, conv=default_conv, in_slice=4, n_feats=64, kernel_size=3, bias=True, 
                head_num=1, win_num_sqrt=16, window_size=16, num_blocks=16, n_depth=2, image_size=256):
        super().__init__()

        self.head = nn.Sequential(
            conv(in_slice, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size),
        )

        frequency = [
            FrequencyAwareBlock(conv=nn.Conv2d, in_slice=in_slice, n_feats=n_feats, kernel_size=kernel_size, bias=bias, 
            n_depth=n_depth, head_num=head_num, win_num_sqrt=win_num_sqrt, window_size=window_size)
            for _ in range(num_blocks // 2)
        ]
        self.frequency = nn.ModuleList(frequency)

        self.alignment = nn.ModuleList(
            [CrossViewBlock(n_feats, image_size=image_size) for _ in range(3)]
        )

        self.fuse_align = nn.Conv2d(3 * n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x_head = self.head(x)
        res = x_head
        align_list = []
        collect = []
        res = self.alignment[0](res) + res
        align_list.append(res)
        for id, layer in enumerate(self.frequency):
            res = layer(res)
            if id in [3, 7]:
                res = self.alignment[id // 4 + 1](res) + res
                align_list.append(res)
            if id in [1, 4, 7]:
                collect.append(res)

        res = torch.cat(align_list, 1)
        res = self.fuse_align(torch.cat(align_list, 1))
        res = res + x_head
        collect.append(res)
        collect = torch.stack(collect, dim=1)
        return collect