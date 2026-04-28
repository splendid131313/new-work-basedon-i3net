import torch
import torch.nn as nn
from .i3net.basic_model import default_conv, I2Group, CrossViewBlock
from .flowseek.core.flowseek import FlowSeek
from .i3net.flow_module import warp

# from i3net.basic_model import default_conv, I2Group, CrossViewBlock
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

        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        window_size = args.image_size // args.win_num_sqrt
        self.head = nn.ModuleDict({
            'local': nn.Sequential(
                conv(in_slice + 2 * out_slice, n_feats, kernel_size),
                nn.ReLU(),
                conv(n_feats, n_feats, kernel_size),
            ),
            'global': nn.Sequential(
                conv(2 + 2 * out_slice, n_feats, kernel_size),
                nn.ReLU(),
                conv(n_feats, n_feats, kernel_size),
            ),
        })
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
    
    def _get_flow(self, i_start, i_end):
        self.flowseek.eval()
        B, T, H, W = i_start.shape

        w0_seq = torch.zeros((B, self.args.hr_slice_patch, H, W)).to(i_start.device)
        w1_seq = torch.zeros((B, self.args.hr_slice_patch, H, W)).to(i_start.device)

        for i in range(self.args.lr_slice_patch):
            idx = i * self.args.upscale
            w0_seq[:, idx, :, :] = i_start[:, i, :, :] if i < i_start.shape[1] else i_end[:, -1, :, :]
            w1_seq[:, idx, :, :] = w0_seq[:, idx, :, :]

        for i in range(T):
            img0 = i_start[:, i, :, :]
            img1 = i_end[:, i, :, :]
            img0 = self._vol_to_flowseek_rgb(img0)
            img1 = self._vol_to_flowseek_rgb(img1)

            with torch.no_grad():
                flow01 = self.flowseek(img0, img1, test_mode=True)["final"]
                flow10 = self.flowseek(img1, img0, test_mode=True)["final"]

            for j in range(1, self.args.upscale):
                time = j / self.args.upscale
                curr_idx = i * self.args.upscale + j

                imgt0 = warp(img0, -flow01 * time)
                imgt1 = warp(img1, -flow10 * (1 - time))

                w0_seq[:, curr_idx, :, :] = torch.mean(imgt0, 1) / 255.0
                w1_seq[:, curr_idx, :, :] = torch.mean(imgt1, 1) / 255.0

        return w0_seq, w1_seq

    def _get_global_flow(self, i_start, i_end):
        self.flowseek.eval()

        warped0_list = []
        warped1_list = []

        warped0_list.append(i_start)
        warped1_list.append(i_start)

        img0 = self._vol_to_flowseek_rgb(i_start)
        img1 = self._vol_to_flowseek_rgb(i_end)

        with torch.no_grad():
            flow01 = self.flowseek(img0, img1, test_mode=True)["final"]
            flow10 = self.flowseek(img1, img0, test_mode=True)["final"]

        times = self.args.hr_slice_patch - 1

        for j in range(1, times):
            t = j / times

            imgt0 = warp(img0, -flow01 * t)
            imgt1 = warp(img1, -flow10 * (1 - t))

            warped0_list.append(torch.mean(imgt0, 1) / 255.0)
            warped1_list.append(torch.mean(imgt1, 1) / 255.0)
        
        warped0_list.append(i_end)
        warped1_list.append(i_end)

        # shape: [B, times+2, H, W]
        warped0 = torch.stack(warped0_list, dim=1)
        warped1 = torch.stack(warped1_list, dim=1)

        return warped0, warped1
    
    def _refine(self, x, warped0, warped1, branch='local'):
        x = torch.cat([x, warped0, warped1], 1)
        x_head = self.head[branch](x)
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

        raw_output = self.tail(res)  # [B, out_slice * 2, H, W]
        mask = torch.sigmoid(raw_output[:, : self.args.hr_slice_patch, :, :])
        delta = raw_output[:, self.args.hr_slice_patch :, :, :]

        out = mask * warped0 + (1 - mask) * warped1 + delta

        return out
    
    def model_local(self, x):
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]
        warped0, warped1 = self._get_flow(i_start, i_end)
        return warped0, warped1
    
    def model_global(self, x):
        i_start = x[:, 0, :, :]
        i_end = x[:, -1, :, :]
        warped0, warped1 = self._get_global_flow(i_start, i_end)
        return warped0, warped1

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        warped0_local, warped1_local = self.model_local(x)
        warped0_global, warped1_global = self.model_global(x)

        out_local = self._refine(x, warped0_local, warped1_local, branch='local')
        x_global = torch.cat([x[:, 0:1, :, :], x[:, -1:, :, :]], 1)
        out_global = self._refine(x_global, warped0_global, warped1_global, branch='global')

        out_local[:, :: self.args.upscale] = x
        out_global[:, 0, :, :] = x[:, 0, :, :]
        out_global[:, -1, :, :] = x[:, -1, :, :]

        out_local = out_local.permute(0, 2, 3, 1).contiguous()
        out_global = out_global.permute(0, 2, 3, 1).contiguous()

        return out_local, out_global


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
    args.lr_slice_patch = 6
    args.hr_slice_patch = (args.lr_slice_patch - 1) * args.upscale + 1
    args.head_num = 1
    args.win_num_sqrt = 16
    args.image_size = 256

    gpy_id = 0
    model = Net(args).cuda(gpy_id)
    x = torch.ones(1, args.image_size, args.image_size, args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(1, args.image_size, args.image_size, args.hr_slice_patch).cuda(gpy_id)
    out_local, out_global = model(x)
    print(out_local.shape)
    print(out_global.shape)
