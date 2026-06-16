import torch
import torch.nn as nn
from .i3net.basic_model import default_conv, HighFrequencyAttention
from .i3net.pfg import MidPFG, Encoder, Decoder
from .flowseek.core.flowseek import FlowSeek
from .i3net.flow_module import warp

# from i3net.basic_model import default_conv
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

        
        # raw w0/w1/diff + HF residuals from slice_attn
        self.encoder = Encoder(c_in=5, c_hid=n_feats, n_s=4, k=kernel_size, act_inplace=False)
        self.decoder = Decoder(c_hid=n_feats, c_out=2, n_s=4, k=kernel_size, act_inplace=False)
        self.flowseek = FlowSeek(args)
        self.hid = MidPFG(in_ch=out_slice * n_feats, depth=num_blocks, groups_pw=1, layerscale_init=1e-6, cel_k=(3, 5, 7), drop=0.0, drop_path=0.0, pfga_K=(9, 15, 31))

        self.slice_attn = HighFrequencyAttention(out_slice, n_feats)
        modules_tail = [
            conv(out_slice * 2, n_feats, kernel_size),
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

    def _get_align_with_vis(self, i_start, i_end):
        """与 _get_align 相同，额外返回每对相邻 LR 切片的光流。"""
        self.flowseek.eval()
        B, T, H, W = i_start.shape

        w0_seq = torch.zeros((B, self.args.hr_slice_patch, H, W), device=i_start.device)
        w1_seq = torch.zeros((B, self.args.hr_slice_patch, H, W), device=i_start.device)
        flow01_list, flow10_list = [], []

        for i in range(self.args.lr_slice_patch):
            idx = i * self.args.upscale
            w0_seq[:, idx, :, :] = (
                i_start[:, i, :, :] if i < i_start.shape[1] else i_end[:, -1, :, :]
            )
            w1_seq[:, idx, :, :] = w0_seq[:, idx, :, :]

        for i in range(T):
            img0 = i_start[:, i, :, :]
            img1 = i_end[:, i, :, :]
            img0_rgb = self._vol_to_flowseek_rgb(img0)
            img1_rgb = self._vol_to_flowseek_rgb(img1)

            with torch.no_grad():
                flow01 = self.flowseek(img0_rgb, img1_rgb, test_mode=True)["final"]
                flow10 = self.flowseek(img1_rgb, img0_rgb, test_mode=True)["final"]
            flow01_list.append(flow01)
            flow10_list.append(flow10)

            for j in range(1, self.args.upscale):
                time = j / self.args.upscale
                curr_idx = i * self.args.upscale + j
                img0t = warp(img0_rgb, -flow01 * time)
                imgt1 = warp(img1_rgb, -flow10 * (1 - time))
                w0_seq[:, curr_idx, :, :] = torch.mean(img0t, 1) / 255.0
                w1_seq[:, curr_idx, :, :] = torch.mean(imgt1, 1) / 255.0

        return w0_seq, w1_seq, flow01_list, flow10_list

    def _enhance_warped(self, warped0, warped1):
        """HF-enhance warped volumes; keep raw copies for encoder input."""
        warped0_raw = warped0
        warped1_raw = warped1
        warped0_enh = self.slice_attn(warped0_raw)
        warped1_enh = self.slice_attn(warped1_raw)
        return warped0_raw, warped1_raw, warped0_enh, warped1_enh

    def _build_encoder_input(self, warped0_raw, warped1_raw, warped0_enh, warped1_enh):
        """Encoder sees raw warp + explicit HF residuals; fusion uses enhanced warps."""
        B, T, H, W = warped0_raw.shape
        w0 = warped0_raw.reshape(B * T, 1, H, W)
        w1 = warped1_raw.reshape(B * T, 1, H, W)
        hf0 = (warped0_enh - warped0_raw).reshape(B * T, 1, H, W)
        hf1 = (warped1_enh - warped1_raw).reshape(B * T, 1, H, W)
        return torch.cat([w0, w1, w0 - w1, hf0, hf1], dim=1)

    def forward_with_vis(self, x):
        """前向推理并返回中间特征，供可视化脚本使用。"""
        x = x.permute(0, 3, 1, 2).contiguous()
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]

        warped0, warped1, flow01_list, flow10_list = self._get_align_with_vis(
            i_start, i_end
        )
        B, T, H, W = warped0.shape

        warped0_raw, warped1_raw, warped0_enh, warped1_enh = self._enhance_warped(
            warped0, warped1
        )
        enc_in = self._build_encoder_input(
            warped0_raw, warped1_raw, warped0_enh, warped1_enh
        )
        embed, skip = self.encoder(enc_in)
        _, c2, h2, w2 = embed.shape
        z = embed.view(B, T, c2, h2, w2)

        b, t, c, h, w = z.shape
        x_hid = z.view(b, t * c, h, w)
        freq_shallow_feat = self.hid.cel(x_hid)
        first_blk = self.hid.blocks[0]
        z_shallow_bt = freq_shallow_feat.view(b * t, c, h, w)
        # _freq_maps 支持任意 C；norm_dw 需要 T*C 通道，故可视化时跳过 norm
        freq_shallow_maps = torch.cat(
            [first_blk.tm._freq_maps(z_shallow_bt[i : i + 1]) for i in range(b * t)],
            dim=0,
        )

        x_blk = freq_shallow_feat
        x_pre_last = x_blk
        for i, blk in enumerate(self.hid.blocks):
            if i == len(self.hid.blocks) - 1:
                x_pre_last = x_blk
            x_blk = blk(x_blk)
        freq_deep_feat = x_blk
        z = freq_deep_feat.view(b, t, c, h, w)
        z_pre_last_bt = x_pre_last.view(b * t, c, h, w)
        last_blk = self.hid.blocks[-1]
        freq_deep_maps = torch.cat(
            [last_blk.tm._freq_maps(z_pre_last_bt[i : i + 1]) for i in range(b * t)],
            dim=0,
        )

        freq_shallow_feat = z_shallow_bt
        freq_deep_feat = z.reshape(b * t, c, h, w)

        hid = z.reshape(B * T, c2, h2, w2)
        dec_z = hid
        n_dec = len(self.decoder.dec)
        shallow_idx = max(n_dec // 2 - 1, 0)
        for i in range(shallow_idx + 1):
            dec_z = self.decoder.dec[i](dec_z)
        decoder_shallow = dec_z
        for i in range(shallow_idx + 1, n_dec - 1):
            dec_z = self.decoder.dec[i](dec_z)
        decoder_deep = self.decoder.dec[-1](dec_z + skip)

        y = self.decoder.readout(decoder_deep)
        y = y.view(B, T * 2, H, W)

        raw_output = self.tail(y)
        mask = torch.sigmoid(raw_output[:, : self.args.hr_slice_patch, :, :])
        delta = raw_output[:, self.args.hr_slice_patch :, :, :]
        out = mask * warped0_enh + (1 - mask) * warped1_enh + delta
        out[:, :: self.args.upscale] = x
        out = out.permute(0, 2, 3, 1).contiguous()

        vis = {
            "lr_input": x,
            "warped0_raw": warped0_raw,
            "warped1_raw": warped1_raw,
            "warped0": warped0_enh,
            "warped1": warped1_enh,
            "hf0": warped0_enh - warped0_raw,
            "hf1": warped1_enh - warped1_raw,
            "flow01_list": flow01_list,
            "flow10_list": flow10_list,
            "encoder_input": enc_in,
            "encoder_shallow": skip,
            "encoder_deep": embed,
            "freq_shallow_feat": freq_shallow_feat,
            "freq_shallow_maps": freq_shallow_maps,
            "freq_deep_feat": freq_deep_feat,
            "freq_deep_maps": freq_deep_maps,
            "decoder_shallow": decoder_shallow,
            "decoder_deep": decoder_deep,
            "before_tail": y,
            "raw_tail": raw_output,
            "mask": mask,
            "delta": delta,
            "fused": mask * warped0_enh + (1 - mask) * warped1_enh + delta,
        }
        return out, vis

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        i_start = x[:, :-1, :, :]
        i_end = x[:, 1:, :, :]
        
        warped0, warped1 = self._get_align(i_start, i_end)
        B, T, H, W = warped0.shape

        ####### slice attntion #########
        # warped0 = self.slice_attn(warped0)
        # warped1 = self.slice_attn(warped1)
        # w0 = warped0.view(B * T, -1, H, W)
        # w1 = warped1.view(B * T, -1, H, W)
        # x0 = torch.cat([w0, w1, w0 - w1], dim=1)

        warped0_raw, warped1_raw, warped0_enh, warped1_enh = self._enhance_warped(
            warped0, warped1
        )
        x0 = self._build_encoder_input(
            warped0_raw, warped1_raw, warped0_enh, warped1_enh
        )

        embed, skip = self.encoder(x0)
        _, c2, h2, w2 = embed.shape
        z = embed.view(B, T, c2, h2, w2)

        z = self.hid(z)
        hid = z.reshape(B * T, c2, h2, w2)

        y = self.decoder(hid, skip)
        y = y.view(-1, T * 2, H, W)

        y = self.tail(y)
        mask = torch.sigmoid(y[:, :self.args.hr_slice_patch, :, :])
        delta = y[:, self.args.hr_slice_patch:, :, :]

        out = mask * warped0_enh + (1 - mask) * warped1_enh + delta

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
    args.lr_time_list = [0, 0.5, 1]

    gpy_id = 0
    model = I3Net(args).cuda(gpy_id)
    x = torch.ones(1, args.image_size, args.image_size, args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(1, args.image_size, args.image_size, args.hr_slice_patch).cuda(gpy_id)
    pred = model(x)
    print(pred.shape)
