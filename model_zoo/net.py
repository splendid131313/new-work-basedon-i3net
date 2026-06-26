import torch
import torch.nn as nn
import torch.nn.functional as F
from .i3net.basic_model import default_conv, RDB, KernelGenerator, DynamicRefine

from .flowseek.core.flowseek import FlowSeek
from .i3net.flow_module import warp

# from i3net.basic_model import default_conv, I2Group, CrossViewBlock
# from flowseek.core.flowseek import FlowSeek
# from i3net.flow_module import warp

def make_model(args):
    return I3Net(args)

class I3Net(nn.Module):
    def __init__(self, args=None, conv=default_conv):
        super(I3Net, self).__init__()
        self.args = args
        n_feats = args.n_feats  # 64
        kernel_size = args.kernel_size  # 3
        num_blocks = args.i_num_blocks  # 16
        in_slice = args.lr_slice_patch * 1
        out_slice = args.hr_slice_patch
        self.D, C, G = {"A": (20, 6, 32), "B": (16, 8, 64), "C": (6, 8, 32)}[
            args.RDNconfig
        ]

        self.head = nn.Sequential(
            conv(in_slice + 2 * out_slice, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size),
        )
        self.flowseek = FlowSeek(args)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=n_feats, growRate=G, nConvLayers=C))
        
        # Global Feature Fusion
        self.GFF = nn.Sequential(
            *[
                nn.Conv2d(self.D * n_feats, n_feats, 1, padding=0, stride=1),
                nn.Conv2d(
                    n_feats,
                    n_feats,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    stride=1,
                ),
            ]
        )

        self.kernel_generator = KernelGenerator(in_c=4)
        self.dynamic_refine = DynamicRefine(n_feats)
        
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
    
    def _get_align(self, i_start, i_end):
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
                flow = self.flowseek(img0, img1, test_mode=True)["final"]

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

        return w0_seq, w1_seq, flow

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]
        
        # B, T, H, W = x.shape
        warped0, warped1, flow = self._get_align(i_start, i_end)

        u = flow[:, 0:1]
        v = flow[:, 1:2]  
        sobel_x = torch.tensor([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=torch.float32, device=flow.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32, device=flow.device).view(1,1,3,3)  

        ux = F.conv2d(u, sobel_x, padding=1)
        uy = F.conv2d(u, sobel_y, padding=1)

        vx = F.conv2d(v, sobel_x, padding=1)
        vy = F.conv2d(v, sobel_y, padding=1)

        jacobian = torch.cat([ux, uy, vx, vy], dim=1)

        # flow_mag = torch.norm(flow, dim=1, keepdim=True)
        
        ##### 第2种使用位置 ####
        # kernel = self.kernel_generator(flow_mag)
        # warped0 = warped0 + self.dynamic_refine(warped0, kernel)
        # warped1 = warped1 + self.dynamic_refine(warped1, kernel)

        align_input = torch.cat([x, warped0, warped1], 1)
        x_head = self.head(align_input)
        res = x_head
        RDBs_out = []
        for i in range(self.D):
            res = self.RDBs[i](res)
            RDBs_out.append(res)
        res = self.GFF(torch.cat(RDBs_out, 1))
        res += x_head

        ##### 第1种使用位置 #####
        kernel = self.kernel_generator(jacobian)
        res += self.dynamic_refine(res, kernel)

        raw_output = self.tail(res)  # [B, out_slice * 2, H, W]

        mask = torch.sigmoid(raw_output[:, : self.args.hr_slice_patch, :, :])
        delta = raw_output[:, self.args.hr_slice_patch :, :, :]

        out = mask * warped0 + (1 - mask) * warped1 + delta

        out[:, :: self.args.upscale] = x
        out = out.permute(0, 2, 3, 1).contiguous()

        return out


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
    args.lr_time_list = [0, 0.3, 0.6, 1]

    gpy_id = 0
    model = I3Net(args).cuda(gpy_id)
    x = torch.ones(1, args.image_size, args.image_size, args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(1, args.image_size, args.image_size, args.hr_slice_patch).cuda(gpy_id)
    pred = model(x)
    print(pred.shape)
