import torch.nn as nn
import einops
import torch.nn.functional as F
from typing import Sequence
from einops.layers.torch import Rearrange

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
    x = x.view(
        B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1]
    )
    windows = (
        x.permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(-1, C, window_size[0], window_size[1])
    )
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
    x = windows.view(
        -1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1]
    )
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

#####################################################################
def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


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

class DWConv(nn.Sequential):
    def __init__(self,n_feat,expand=1):
        super().__init__(
            nn.Conv2d(n_feat,n_feat*expand,1,1,0),
            nn.Conv2d(n_feat*expand,n_feat*expand,3,1,1,groups=n_feat*expand),
            nn.Conv2d(n_feat*expand,n_feat,1,1,0)
        )


class ConvBlock(nn.Module):
    """
    Same stacking as EMA-VFI `feature_extractor.ConvBlock`: repeated 3x3 conv + activation.
    """

    def __init__(self, in_dim, out_dim, depths=2, act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
            layers.append(act_layer(out_dim))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class  FeatureExtractor(nn.Module):

    def __init__(
        self,
        in_channels: int = 2,
        embed_dims: Sequence[int] = (64, 128, 256),
        depths: Sequence[int] = (2, 2, 2),
        out_channels: int = 64,
        act_layer=nn.PReLU,
    ):
        super().__init__()
        if len(embed_dims) != len(depths):
            raise ValueError("embed_dims and depths must have the same length")

        parts = nn.ModuleList()
        parts.append(ConvBlock(in_channels, embed_dims[0], depths[0], act_layer))
        for i in range(1, len(embed_dims)):
            parts.append(nn.Conv2d(embed_dims[i - 1], embed_dims[i], 3, 2, 1, bias=True))
            parts.append(act_layer(embed_dims[i]))
            parts.append(ConvBlock(embed_dims[i], embed_dims[i], depths[i], act_layer))
        self.stages = parts

        self.proj_out = nn.Conv2d(embed_dims[-1], out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        H, W = x.shape[-2:]
        for stage in self.stages:
            x = stage(x)
        x = self.proj_out(x)
        
        return x

class CrossViewBlock(nn.Module):
    def __init__(self, n_feat, image_size):
        super().__init__()
        self.image_size = image_size

        self.norm = nn.LayerNorm(n_feat)

        self.conv_sag = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, 1, 0),
            Rearrange("b c h w -> b h c w"),
            nn.PixelShuffle(2),
            nn.Conv2d(image_size // 4, n_feat, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat, image_size // 4, 3, 1, 1),
            nn.PixelUnshuffle(2),
            Rearrange("b h c w -> b c h w"),
        )

        self.conv_cor = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, 1, 0),
            Rearrange("b c h w -> b w c h"),
            nn.PixelShuffle(2),
            nn.Conv2d(image_size // 4, n_feat, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat, image_size // 4, 3, 1, 1),
            nn.PixelUnshuffle(2),
            Rearrange("b w c h -> b c h w"),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x_sag_f = self.conv_sag(x)  # b c h w
        x_cor_f = self.conv_cor(x)  # b c h w
        x_out = x_cor_f + x_sag_f
        return x_out