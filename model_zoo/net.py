import torch
import torch.nn as nn
from .i3net.basic_model import default_conv, I2Group, ConvSC, LocalMMF
from .flowseek.core.flowseek import FlowSeek
from .i3net.flow_module import warp

# from i3net.basic_model import default_conv, I2Group, CrossViewBlock, ConvSC, LocalMMF
# from flowseek.core.flowseek import FlowSeek
# from i3net.flow_module import warp

def make_model(args):
    return Net(args)

class Net(nn.Module):
    def __init__(self, args=None, conv=default_conv):
        super(Net, self).__init__()
        self.args = args
        n_feats = args.n_feats  # 64
        kernel_size = args.kernel_size  # 3
        num_blocks = args.i_num_blocks  # 16
        act = nn.ReLU(True)
        res_scale = args.res_scale  # 1
        in_slice = args.lr_slice_patch * 1
        out_slice = args.hr_slice_patch
        self.time_list = args.lr_time_list[1:-1]
        channels = args.channels

        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        window_size = args.image_size // args.win_num_sqrt
        self.head = nn.Sequential(
            ConvSC(2 * out_slice, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size),
        )
        # self.encoder = SliceEncoder(2 * out_slice, channels, kernel_size, act_inplace=False)
        self.flowseek = FlowSeek(args)

        modules_body = [
            I2Group(
                conv,
                n_depth=2,
                n_feat=n_feats,
                kernel_size=kernel_size,
                act=act,
                res_scale=res_scale,
                head_num=head_num,
                win_num_sqrt=win_num_sqrt,
                window_size=window_size,
            )
            for _ in range(num_blocks // 2)
        ]
        self.body = nn.ModuleList(modules_body)

        self.fuse_align = nn.Conv2d(3 * n_feats, n_feats, 1, 1, 0)

        # self.align_local = nn.ModuleList([LocalMMF(in_ch=n_feats, win_size=3) for _ in range(2)])
        self.align = LocalMMF(in_ch=n_feats, win_size=5)

        modules_tail = [
            ConvSC(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, out_slice * 2, kernel_size),
        ]
        self.tail = nn.Sequential(*modules_tail)
        tail_last = self.tail[-1]
        with torch.no_grad():
            nn.init.normal_(tail_last.weight[out_slice:], mean=0.0, std=1e-4)
            if tail_last.bias is not None:
                nn.init.zeros_(tail_last.bias[:out_slice])

    def _vol_to_flowseek_rgb(self, vol):  # vol: (B, 1, H, W) 或 (B, H, W)
        if vol.ndim == 3:
            vol = vol.unsqueeze(1)  # (B, 1, H, W)
        rgb = vol.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        return (rgb * 255.0).clamp(0.0, 255.0)
    
    def _get_align(self, i_start, i_end):
        B, T, H, W = i_start.shape

        w0_seq = torch.zeros((B, self.args.hr_slice_patch, H, W)).to(i_start.device)
        w1_seq = torch.zeros((B, self.args.hr_slice_patch, H, W)).to(i_start.device)
        flow_list = []

        for i in range(self.args.lr_slice_patch):
            idx = i * self.args.upscale
            w0_seq[:, idx, :, :] = i_start[:, i, :, :] if i < i_start.shape[1] else i_end[:, -1, :, :]
            w1_seq[:, idx, :, :] = w0_seq[:, idx, :, :]

        for i in range(T):
            img0 = i_start[:, i, :, :]
            img1 = i_end[:, i, :, :]
            img0 = self._vol_to_flowseek_rgb(img0)
            img1 = self._vol_to_flowseek_rgb(img1)

            flow = self.flowseek(img0, img1, test_mode=True)["final"]
            flow_list.append(flow)

            # flow01 = self.flowseek(img0, img1, test_mode=True)["final"]
            # flow10 = self.flowseek(img1, img0, test_mode=True)["final"]

            for j in range(1, self.args.upscale):

                time = j / self.args.upscale
                curr_idx = i * self.args.upscale + j

                img0t = warp(img0, flow * time)
                imgt1 = warp(img1, flow * (1 - time))
                # img0t = warp(img0, -flow01 * time)
                # imgt1 = warp(img1, -flow10 * (1 - time))

                w0_seq[:, curr_idx, :, :] = torch.mean(img0t, 1) / 255.0
                w1_seq[:, curr_idx, :, :] = torch.mean(imgt1, 1) / 255.0

        return w0_seq, w1_seq, flow_list

    def forward_with_vis(self, x):
        """前向推理并返回中间特征，供 visualize_features.py 使用。"""
        x = x.permute(0, 3, 1, 2).contiguous()
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]

        warped0, warped1, flow_list = self._get_align(i_start, i_end)

        align_input = torch.cat([warped0, warped1], 1)
        x1 = self.head(align_input)

        res = x1
        align_list = []
        alignment_out = []
        alignment_res = []
        body_out = []

        a0 = self.alignment[0](res)
        alignment_out.append(a0)
        res = a0 + res
        alignment_res.append(res)
        align_list.append(res)

        for id, layer in enumerate(self.body):
            res = layer(res)
            body_out.append(res)
            if id in [3, 7]:
                ai = id // 4
                a_out = self.alignment[ai + 1](res)
                alignment_out.append(a_out)
                res = a_out + res
                alignment_res.append(res)
                align_list.append(res)

        fuse_align = self.fuse_align(torch.cat(align_list, 1))
        align_out = self.align(fuse_align, x1)

        raw_output = self.tail(align_out)
        mask = torch.sigmoid(raw_output[:, : self.args.hr_slice_patch, :, :])
        delta = torch.tanh(raw_output[:, self.args.hr_slice_patch :, :, :]) * 0.1
        out = mask * warped0 + (1 - mask) * warped1 + delta
        out[:, :: self.args.upscale] = x
        out = out.permute(0, 2, 3, 1).contiguous()

        vis = {
            "lr_input": x,
            "warped0": warped0,
            "warped1": warped1,
            "flow_list": flow_list,
            "head": x1,
            "alignment_out": alignment_out,
            "alignment_res": alignment_res,
            "body_out": body_out,
            "fuse_align": fuse_align,
            "align_out": align_out,
            "before_tail": align_out,
            "raw_tail": raw_output,
            "mask": mask,
            "delta": delta,
            "fused": mask * warped0 + (1 - mask) * warped1 + delta,
        }
        return out, vis

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]
        
        warped0, warped1, flow_list = self._get_align(i_start, i_end)

        align_input = torch.cat([warped0, warped1], 1)
        x1 = self.head(align_input)

        res = x1

        align_list = []
        align_list.append(res)

        for id, layer in enumerate(self.body):
            res = layer(res)
            if id in [3, 7]:
                align_list.append(res)

        res = self.fuse_align(torch.cat(align_list, 1))

        align = self.align(res, x1)

        raw_output = self.tail(align)  # [B, out_slice * 2, H, W]

        mask = torch.sigmoid(raw_output[:, : self.args.hr_slice_patch, :, :])
        delta = raw_output[:, self.args.hr_slice_patch :, :, :]

        out = mask * warped0 + (1 - mask) * warped1 + delta

        out[:, :: self.args.upscale] = x
        out = out.permute(0, 2, 3, 1).contiguous()

        return out, flow_list


if __name__ == "__main__":
    import argparse
    from flowseek.config.parser import parse_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="./model_zoo/flowseek/config/eval/flowseek-S.json"
    )

    args = parse_args(parser)
    args.upscale = 3
    args.n_feats = 64
    args.kernel_size = 3
    args.res_scale = 1
    args.i_num_blocks = 16
    args.lr_slice_patch = 4
    args.hr_slice_patch = (args.lr_slice_patch - 1) * args.upscale + 1
    args.head_num = 1
    args.win_num_sqrt = 16
    args.image_size = 256
    args.channels = [64, 128, 320, 512]

    gpy_id = 0
    model = Net(args).cuda(gpy_id)
    x = torch.ones(1, args.image_size, args.image_size, args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(1, args.image_size, args.image_size, args.hr_slice_patch).cuda(gpy_id)
    pred = model(x)
    print(pred.shape)
