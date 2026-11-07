"""I3Net with explicit/implicit confidence fusion replacing CrossViewBlock.

The implicit head receives only original slices. The geometry head receives
both warped volumes. After the first fusion, the trunk is a mixed reconstruction
state; keeping it strictly implicit at every depth would require a second trunk.
"""

import torch
import torch.nn as nn

from .confidence_fusion import ConfidenceFusionBlock
from .flowseek.core.flowseek import FlowSeek
from .i3net.basic_model import default_conv, I2Group
from .i3net.flow_module import warp


def make_model(args):
    return I3Net(args)


class I3Net(nn.Module):
    def __init__(self, args=None, conv=default_conv):
        super().__init__()
        self.args = args
        n_feats = args.n_feats
        kernel_size = args.kernel_size
        in_slice = args.lr_slice_patch
        out_slice = args.hr_slice_patch

        if args.i_num_blocks // 2 < 8:
            raise ValueError("Three fusion stages at entry, body[3], body[7] require at least 8 I2Groups.")

        # Independent implicit and explicit encoders.
        self.head = nn.Sequential(
            conv(in_slice, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size),
        )
        self.geo_encoder = nn.Sequential(
            conv(2 * out_slice, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size),
        )
        self.flowseek = FlowSeek(args)
        self.body = nn.ModuleList([
            I2Group(
                conv,
                n_depth=2,
                n_feat=n_feats,
                kernel_size=kernel_size,
                act=nn.ReLU(True),
                res_scale=args.res_scale,
                head_num=args.head_num,
                win_num_sqrt=args.win_num_sqrt,
                window_size=args.image_size // args.win_num_sqrt,
            )
            for _ in range(args.i_num_blocks // 2)
        ])
        self.alignment = nn.ModuleList([
            ConfidenceFusionBlock(n_feats) for _ in range(3)
        ])
        self.fuse_align = nn.Conv2d(3 * n_feats, n_feats, 1)
        self.tail = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, 2 * out_slice, kernel_size),
        )
        # Keep the original output-head initialization.
        with torch.no_grad():
            nn.init.normal_(self.tail[-1].weight[out_slice:], mean=0.0, std=1e-4)
            if self.tail[-1].bias is not None:
                nn.init.zeros_(self.tail[-1].bias[:out_slice])

    def _vol_to_flowseek_rgb(self, vol):
        if vol.ndim == 3:
            vol = vol.unsqueeze(1)
        return (vol.repeat(1, 3, 1, 1) * 255.0).clamp(0.0, 255.0)

    def _get_align(self, i_start, i_end):
        """Keep the existing FlowSeek sampling convention for this experiment."""
        batch, intervals, height, width = i_start.shape
        w0_seq = i_start.new_zeros(batch, self.args.hr_slice_patch, height, width)
        w1_seq = i_start.new_zeros(batch, self.args.hr_slice_patch, height, width)

        for i in range(self.args.lr_slice_patch):
            idx = i * self.args.upscale
            original = i_start[:, i] if i < intervals else i_end[:, -1]
            w0_seq[:, idx] = original
            w1_seq[:, idx] = original

        for i in range(intervals):
            img0 = self._vol_to_flowseek_rgb(i_start[:, i])
            img1 = self._vol_to_flowseek_rgb(i_end[:, i])
            flow = self.flowseek(img0, img1, test_mode=True)["final"]
            for j in range(1, self.args.upscale):
                time = j / self.args.upscale
                idx = i * self.args.upscale + j
                img0t = warp(img0, flow * time)
                imgt1 = warp(img1, flow * (1 - time))
                w0_seq[:, idx] = img0t.mean(dim=1) / 255.0
                w1_seq[:, idx] = imgt1.mean(dim=1) / 255.0
        return w0_seq, w1_seq

    def forward(self, x, return_confidence=False):
        # x: [B, H, W, lr_slice_patch]
        x = x.permute(0, 3, 1, 2).contiguous()
        warped0, warped1 = self._get_align(x[:, :-1], x[:, 1:])

        x_head = self.head(x)
        geo_feat = self.geo_encoder(torch.cat([warped0, warped1], dim=1))
        res = x_head
        align_list = []
        confidence_list = []

        delta, confidence = self.alignment[0](res, geo_feat, return_confidence=True)
        res = res + delta
        align_list.append(res)
        if return_confidence:
            confidence_list.append(confidence)

        for idx, layer in enumerate(self.body):
            res = layer(res)
            if idx in (3, 7):
                k = idx // 4 + 1
                delta, confidence = self.alignment[k](res, geo_feat, return_confidence=True)
                res = res + delta
                align_list.append(res)
                if return_confidence:
                    confidence_list.append(confidence)

        res = self.fuse_align(torch.cat(align_list, dim=1)) + x_head
        raw_output = self.tail(res)
        mask = torch.sigmoid(raw_output[:, :self.args.hr_slice_patch])
        delta = raw_output[:, self.args.hr_slice_patch:]
        out = mask * warped0 + (1.0 - mask) * warped1 + delta
        out[:, ::self.args.upscale] = x
        out = out.permute(0, 2, 3, 1).contiguous()

        if return_confidence:
            return out, confidence_list
        return out
