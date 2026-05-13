import torch
import torch.nn as nn
from .i3net.basic_model import default_conv, CrossViewBlock
from .i3net.pfg import MidPFG, Encoder, Decoder
from .flowseek.core.flowseek import FlowSeek
from .i3net.flow_module import warp

# from i3net.basic_model import default_conv, CrossViewBlock
# from i3net.pfg import MidPFG, Encoder, Decoder
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
        act = nn.ReLU(True)
        res_scale = args.res_scale  # 1
        in_slice = args.lr_slice_patch * 1
        out_slice = args.hr_slice_patch
        self.time_list = args.lr_time_list[1:-1]

        
        self.encoder = Encoder(c_in=1, c_hid=n_feats, n_s=4, k=kernel_size, act_inplace=False)
        self.decoder = Decoder(c_hid=n_feats, c_out=1, n_s=4, k=kernel_size, act_inplace=False)
        self.flowseek = FlowSeek(args)
        self.hid = MidPFG(in_ch=out_slice * n_feats, depth=num_blocks, groups_pw=1, layerscale_init=1e-6, cel_k=(3, 5, 7), drop=0.0, drop_path=0.0, pfga_K=(9, 15, 31))

        # modules_tail = [
        #     conv(out_slice, n_feats, kernel_size),
        #     nn.ReLU(),
        #     conv(n_feats, out_slice * 2, kernel_size),
        # ]
        # self.tail = nn.Sequential(*modules_tail)
        # tail_last = self.tail[-1]
        # with torch.no_grad():
        #     nn.init.normal_(tail_last.weight[out_slice:], mean=0.0, std=1e-4)
        #     if tail_last.bias is not None:
        #         nn.init.zeros_(tail_last.bias[:out_slice])

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
                # flow = self.flowseek(img0, img1, test_mode=True)["final"]
                flow01 = self.flowseek(img0, img1, test_mode=True)["final"]
                flow10 = self.flowseek(img1, img0, test_mode=True)["final"]

            for j in range(1, self.args.upscale):
                time = j / self.args.upscale
                curr_idx = i * self.args.upscale + j

                # img0t = warp(img0, flow * time)
                # imgt1 = warp(img1, flow * (1 - time))
                img0t = warp(img0, -flow01 * time)
                imgt1 = warp(img1, -flow10 * (1 - time))

                w0_seq[:, curr_idx, :, :] = torch.mean(img0t, 1) / 255.0
                w1_seq[:, curr_idx, :, :] = torch.mean(imgt1, 1) / 255.0

        return w0_seq, w1_seq

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]
        
        warped0, warped1 = self._get_align(i_start, i_end)
        B, T, H, W = warped0.shape

        x0 = warped0.view(B * T, -1, H, W)
        embed, skip = self.encoder(x0)
        _, c2, h2, w2 = embed.shape
        z = embed.view(B, T, c2, h2, w2)

        z = self.hid(z)
        hid = z.reshape(B * T, c2, h2, w2)

        y = self.decoder(hid, skip)
        y = y.view(-1, T, H, W)

        y[:, :: self.args.upscale] = x
        y = y.permute(0, 2, 3, 1).contiguous()

        return y

        # raw_output = self.tail(y)
        # mask = torch.sigmoid(raw_output[:, : self.args.hr_slice_patch, :, :])
        # delta = raw_output[:, self.args.hr_slice_patch :, :, :]

        # out = mask * warped0 + (1 - mask) * warped1 + delta

        # out[:, :: self.args.upscale] = x
        # out = out.permute(0, 2, 3, 1).contiguous()

        # return out


if __name__ == "__main__":
    import argparse
    from flowseek.config.parser import parse_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="./model_zoo/flowseek/config/eval/flowseek-S.json"
    )

    args = parse_args(parser)
    args.upscale = 2
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
