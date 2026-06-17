import contextlib

import torch.nn as nn
import torch
import einops
import torch.distributions as td
from functools import partial
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .dct_util import DCT2x,IDCT2x


#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


#####################################################################
class SliceAttentionModule(nn.Module):
    def __init__(self, in_features, n_feats=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, n_feats),
            nn.ReLU(),
            nn.Linear(n_feats, in_features)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        avg = torch.mean(x, dim=(2,3))      # (B, C)
        maxv = torch.amax(x, dim=(2,3))     # (B, C)

        att = self.mlp(avg) + self.mlp(maxv)

        att = torch.sigmoid(att).view(B, C, 1, 1)

        return x * att


class FreqSliceAttentionModule(nn.Module):
    """Slice attention in DCT domain; weights from spectral stats, output in pixel domain."""

    def __init__(self, in_features, n_feats=64):
        super().__init__()
        self.dct = DCT2x()
        self.idct = IDCT2x()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, n_feats),
            nn.ReLU(),
            nn.Linear(n_feats, in_features),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_dct = self.dct(x)

        avg = torch.mean(x_dct, dim=(2, 3))
        maxv = torch.amax(x_dct, dim=(2, 3))
        att = torch.sigmoid(self.mlp(avg) + self.mlp(maxv)).view(B, C, 1, 1)

        return self.idct(x_dct * att)


class IntraSliceBranch(nn.Module):
    def __init__(self,conv=nn.Conv2d,n_feat=64,kernel_size=3,bias=True,
                 head_num=1, win_num_sqrt=16, window_size=16):
        super().__init__()

        self.window_size = window_size
        self.win_num_sqrt = win_num_sqrt

        self.dct = DCT2x()
        self.norm = nn.LayerNorm(n_feat)
        self.conv = nn.Sequential(
            conv(n_feat,n_feat,1, bias=bias),
            conv(n_feat,n_feat,3,1,1, bias=bias),
            conv(n_feat,n_feat,3,1,1, bias=bias)
        )
        self.idct = IDCT2x()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.attn = nn.Sequential(
            PreNormResidual(dim=n_feat, fn=FeedForward(dim=window_size**2, expansion_factor=1, dropout=0, dense=chan_first)), # dim=num_patch
            PreNormResidual(dim=n_feat, fn=FeedForward(dim=n_feat, expansion_factor=2, dropout=0, dense=chan_last)) # dim=h*w*c
        )
        self.last_conv = conv(n_feat,n_feat,kernel_size=1,bias=bias)

    def forward(self,x):
        b,c,h,w = x.shape
        x_dct = self.dct(x)
        x_dct = einops.rearrange(x_dct,'b c h w -> b (h w) c')
        x_dct = self.norm(x_dct)
        x_dct = einops.rearrange(x_dct,'b (h w) c -> b c h w',h=h,w=w)
        x_dct = self.conv(x_dct)

        x_dct_windows = window_partitions(x_dct,window_size=self.window_size) # [b,c,h,w]
     
        bi,ci,hi,wi = x_dct_windows.shape
        x_dct_windows = einops.rearrange(x_dct_windows,'b c h w -> b (h w) c')
        x_dct_windows_attn = self.attn(x_dct_windows)
        x_dct_windows = x_dct_windows + x_dct_windows_attn
        x_dct_windows = einops.rearrange(x_dct_windows,'b (h w) c -> b c h w',h=hi,w=wi)
        
        x_dct_attn =  window_reverses(x_dct_windows,window_size=self.window_size,H=h,W=w)

        x_dct_idct = self.idct(x_dct_attn)
        x_attn = self.last_conv(x_dct_idct)

        return x_attn

class I2Block(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1, head_num=1, win_num_sqrt=16, window_size=16):
        super(I2Block, self).__init__()
        inter_slice_branch = [
            nn.PixelUnshuffle(2),
            nn.Conv2d(4 * n_feat, 4 * n_feat, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(4 * n_feat, 4 * n_feat, 3, 1, 1),  # +
            nn.PixelShuffle(2),  # +
            nn.Conv2d(n_feat, n_feat, 1, 1, 0)
        ]
        self.inter_slice_branch = nn.Sequential(*inter_slice_branch)

        self.res_scale = res_scale

        self.intra_slice_branch = IntraSliceBranch(conv=nn.Conv2d,n_feat=n_feat,kernel_size=kernel_size,bias=bias
                                  ,head_num=head_num,win_num_sqrt=win_num_sqrt,window_size=window_size)

    def forward(self, x):
        x_inter = self.inter_slice_branch(x).mul(self.res_scale)
        x_intra = self.intra_slice_branch(x)
        out = x_inter + x_intra + x
        return out

class I2Group(nn.Module):
    def __init__(
        self, conv, n_depth, n_feat, kernel_size,skip_connect=False,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,head_num=1,win_num_sqrt=16,window_size=16):
        super().__init__()

        body = [I2Block(conv, n_feat, kernel_size,
                        bias, bn, act, res_scale, head_num, win_num_sqrt, window_size) for _ in range(n_depth)]

        self.body = nn.ModuleList(body)
    def forward(self,x):
        res = x
        for block in self.body:
            res = block(res)
        out = res
        return out


class DWConv(nn.Sequential):
    def __init__(self,n_feat,expand=1):
        super().__init__(
            nn.Conv2d(n_feat,n_feat*expand,1,1,0),
            nn.Conv2d(n_feat*expand,n_feat*expand,3,1,1,groups=n_feat*expand),
            nn.Conv2d(n_feat*expand,n_feat,1,1,0)
        )

class NeighborFusion(nn.Module):
    def __init__(self, n_feats=16):
        super().__init__()

        self.weight_net = nn.Sequential(
            nn.Conv2d(2, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, 2, 3, padding=1),
        )

    def forward(self, x):
        B, N, H, W = x.shape

        out = x.clone()

        for i in range(1, N, 2):
            left = x[:, i - 1 : i, :, :]
            right = x[:, i + 1 : i + 2, :, :]

            if right.shape[1] == 0:
                right = left

            pair = torch.cat([left, right], dim=1)  # [B,2,H,W]

            w = self.weight_net(pair)  # [B,2,H,W]
            w = torch.softmax(w, dim=1)

            fused = w[:, 0:1] * left + w[:, 1:2] * right

            out[:, i : i + 1] = fused

        return out