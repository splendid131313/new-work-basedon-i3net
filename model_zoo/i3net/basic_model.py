import torch.nn as nn
import torch 
import einops
from functools import partial
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

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


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, upsampling=False,
                 act_norm=False, act_inplace=True, ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, stride=1, padding=padding,
                          dilation=dilation),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation)

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
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True,
                 act_inplace=True, ):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, upsampling=upsampling,
                                padding=padding, act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class SliceEncoder(nn.Module):
    """Multi-scale encoder for slice input; outputs align with net downsample stages."""

    def __init__(self, c_in: int, channels, k: int = 3, act_inplace: bool = False):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    ConvSC(c_in, c0, k, downsampling=False, act_inplace=act_inplace),
                ),
                nn.Sequential(
                    ConvSC(c0, c1, k, downsampling=True, act_inplace=act_inplace),
                    ConvSC(c1, c1, k, downsampling=False, act_inplace=act_inplace),
                ),
                nn.Sequential(
                    ConvSC(c1, c2, k, downsampling=True, act_inplace=act_inplace),
                    ConvSC(c2, c2, k, downsampling=False, act_inplace=act_inplace),
                ),
                nn.Sequential(
                    ConvSC(c2, c3, k, downsampling=True, act_inplace=act_inplace),
                    ConvSC(c3, c3, k, downsampling=False, act_inplace=act_inplace),
                ),
            ]
        )

    def forward(self, x: torch.Tensor):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats[0], feats[1], feats[2], feats[3]


class LocalMMF(nn.Module):
    def __init__(self, in_ch=96, win_size=7):
        super().__init__()

        self.win_size = win_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.BatchNorm2d(in_ch), nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.BatchNorm2d(in_ch), nn.ReLU(True)
        )

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = self.conv1(x)
        y = self.conv2(y)

        k, pad = self.win_size, self.win_size // 2
        unfold = lambda t: F.unfold(t, k, padding=pad).view(B, C, k * k, H * W)

        x_patch, y_patch = unfold(x), unfold(y)
        x_center = x.view(B, C, 1, H * W)

        attn = torch.softmax((x_center * y_patch).sum(dim=1), dim=1)
        out_x = (attn.unsqueeze(1) * x_patch).sum(dim=2).view(B, C, H, W)
        out_y = (attn.unsqueeze(1) * y_patch).sum(dim=2).view(B, C, H, W)

        return out_x + out_y + x + y


#####################################################################
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
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,head_num=1,win_num_sqrt=16,window_size=16):
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
                            bias, bn, act, res_scale,head_num,win_num_sqrt, window_size) for _ in range(n_depth)]

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