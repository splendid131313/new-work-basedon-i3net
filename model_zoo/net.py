import torch
import torch.nn as nn
from .i3net.basic_model import default_conv, I2Group, CrossViewBlock
from .flowseek.core.flowseek import FlowSeek
from .i3net.flow_module import warp

def make_model(args):
    return I3Net(args)

class I3Net(nn.Module):
    def __init__(self, args=None, conv=default_conv):
        super(I3Net, self).__init__()
        self.args = args
        n_feats = args.n_feats  # 64
        kernel_size = args.kernel_size  # 3
        num_blocks = args.i_num_blocks  # 16
        act = nn.ReLU(True)
        res_scale = args.res_scale  # 1
        in_slice = args.lr_slice_patch * 1
        out_slice = args.hr_slice_patch
        self.time_list = args.lr_time_list[1:-1]

        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        window_size = args.image_size // args.win_num_sqrt
        self.head = nn.Sequential(
            conv(out_slice, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size),
        )
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

        self.alignment = nn.ModuleList([CrossViewBlock(n_feats, image_size=args.image_size) for _ in range(3)])

        self.fuse_align = nn.Conv2d(3 * n_feats, n_feats, 1, 1, 0)

        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, out_slice, kernel_size),
        ]
        self.tail = nn.Sequential(*modules_tail)

    def _vol_to_flowseek_rgb(self, vol):  # vol: (B, 1, H, W) 或 (B, H, W)
        if vol.ndim == 3:
            vol = vol.unsqueeze(1)  # (B, 1, H, W)
        rgb = vol.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        return (rgb * 255.0).clamp(0.0, 255.0)
    
    def _get_align(self, i_start, i_end):
        self.flowseek.eval()
        B, T, H, W = i_start.shape
        align_list = []
        align_list.append(torch.zeros((B, 1, H, W)).to(i_start.device))
        for i in range(T):
            img0 = i_start[:, i, :, :]
            img1 = i_end[:, i, :, :]
            img0 = self._vol_to_flowseek_rgb(img0)
            img1 = self._vol_to_flowseek_rgb(img1)

            with torch.no_grad():
                flow = self.flowseek(img0, img1, test_mode=True)["final"]
            for time in self.time_list:
                img0t = warp(img0, flow * time)
                imgt1 = warp(img1, flow * (1 - time))
                imgt = (img0t + imgt1) / 2
                imgt = torch.mean(imgt, dim=1, keepdim=True)
                imgt = imgt / 255.0
                align_list.append(imgt)
            align_list.append(torch.zeros((B, 1, H, W)).to(i_start.device))
        align_list = torch.cat(align_list, 1)

        return align_list

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]
        
        # B, T, H, W = x.shape
        # align = torch.zeros(B, 7, H, W).to(x.device)
        align = self._get_align(i_start, i_end)
        align[:, ::self.args.upscale, :, :] = x
        x_head = self.head(align)

        res = x_head

        align_list = []
        res = self.alignment[0](res) + res
        align_list.append(res)

        for id, layer in enumerate(self.body):
            res = layer(res)
            if id in [3, 7]:
                res = self.alignment[id // 4 + 1](res) + res
                align_list.append(res)

        res = self.fuse_align(torch.cat(align_list, 1))

        res += x_head

        out = self.tail(res)  # [bz,s,h,w]

        out[:, :: self.args.upscale] = x
        out = out.permute(0, 2, 3, 1).contiguous()

        return out


if __name__ == "__main__":
    import argparse
    from flowseek.config.parser import parse_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="./flowseek/config/eval/flowseek-S.json"
    )

    args = parse_args(parser)
    args.upscale = 2
    args.n_feats = 64
    args.kernel_size = 3
    args.res_scale = 1
    args.num_blocks = 16
    args.lr_slice_patch = 4
    args.hr_slice_patch = (args.lr_slice_patch - 1) * args.upscale + 1
    args.head_num = 1
    args.win_num_sqrt = 16
    args.image_size = 256
    args.lr_time_list = [0, 0.5, 1]

    gpy_id = 0
    model = I3Net(args).cuda(gpy_id)
    x = torch.ones(1, args.image_size, args.image_size, args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(1, args.image_size, args.image_size, args.hr_slice_patch).cuda(gpy_id)
    pred = model(x)
    print(pred.shape)
