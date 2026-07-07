import torch
import torch.nn as nn
# from .i3net.basic_model import default_conv, I2Group, CrossViewBlock, TimeConditionModulation
# from .flowseek.core.flowseek import FlowSeek
# from .i3net.flow_module import warp

from i3net.basic_model import default_conv, I2Group, CrossViewBlock, TimeConditionModulation
from flowseek.core.flowseek import FlowSeek
from i3net.flow_module import warp

def make_model(args):
    return ContinuousNet(args)

class ContinuousNet(nn.Module):
    def __init__(self, args=None, conv=default_conv):
        super(ContinuousNet, self).__init__()
        self.args = args
        n_feats = args.n_feats  # 64
        kernel_size = args.kernel_size  # 3
        num_blocks = args.i_num_blocks  # 16
        act = nn.ReLU(True)
        res_scale = args.res_scale  # 1

        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        window_size = args.image_size // args.win_num_sqrt
        self.head = nn.Sequential(
            conv(4, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size),
        )
        
        self.flowseek = FlowSeek(args)

        # 1. 引入时间发生器，将标量 t 映射为时空特征
        self.time_embed = nn.Sequential(
            nn.Linear(1, n_feats), nn.Linear(n_feats, n_feats)
        )

        # 2. 光流残差修正网络：应对大形变/大倍率导致的非线性轨迹
        self.flow_refine = nn.Sequential(
            conv(2 + 1, n_feats, kernel_size),  # 输入: 粗光流(2ch) + 时间t(1ch)
            nn.ReLU(),
            conv(n_feats, 2, kernel_size),  # 输出: 光流残差修正量
        )

        # 3. 特征调制层
        self.mod1 = TimeConditionModulation(n_feats)
        # self.mod2 = TimeConditionModulation(n_feats)

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

        # self.alignment = nn.ModuleList([CrossViewBlock(n_feats, image_size=args.image_size) for _ in range(3)])

        self.fuse_align = nn.Conv2d(3 * n_feats, n_feats, 1, 1, 0)

        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, 2, kernel_size),
        ]
        self.tail = nn.Sequential(*modules_tail)
        tail_last = self.tail[-1]
        with torch.no_grad():
            nn.init.normal_(tail_last.weight[1:], mean=0.0, std=1e-4)
            if tail_last.bias is not None:
                nn.init.zeros_(tail_last.bias[:1])

    def _vol_to_rgb(self, vol):  # vol: (B, 1, H, W) 或 (B, H, W)
        if vol.ndim == 3:
            vol = vol.unsqueeze(1)  # (B, 1, H, W)
        rgb = vol.repeat(1, 3, 1, 1)  # (B, 3, H, W)

        return (rgb * 255.0).clamp(0.0, 255.0)
    
    def _get_base_flow(self, img0, img1):
        img0 = self._vol_to_rgb(img0)
        img1 = self._vol_to_rgb(img1)

        with torch.no_grad():
            flow01 = self.flowseek(img0, img1, test_mode=True)["final"]
            flow10 = self.flowseek(img1, img0, test_mode=True)["final"]
        
        return flow01, flow10
    
    def _get_continuous_flow(self, img0, flow01, flow10, t):
        B, _, H, W = img0.shape

        t_tensor = t.view(B, 1, 1, 1).expand(-1, -1, H, W)

        # 基础线性流
        base_flow0t = flow01 * t.view(B, 1, 1, 1)
        base_flow1t = flow10 * (1.0 - t.view(B, 1, 1, 1))

        # 非线性修正
        feat_0t = torch.cat([base_flow0t, t_tensor], dim=1)
        feat_1t = torch.cat([base_flow1t, 1.0 - t_tensor], dim=1)

        delta_flow0t = self.flow_refine(feat_0t)
        delta_flow1t = self.flow_refine(feat_1t)

        # 最终非线性对齐光流
        final_flow0t = base_flow0t + delta_flow0t
        final_flow1t = base_flow1t + delta_flow1t

        return final_flow0t, final_flow1t
    
    def forward_single_t(self, img0, img1, flow01, flow10, t):
        # img0, img1: (B, 1, H, W)
        # t: (B, 1)

        flow0t, flow1t = self._get_continuous_flow(img0, flow01, flow10, t)

        warped0 = warp(img0, -flow0t)
        warped1 = warp(img1, -flow1t)

        x_in = torch.cat([img0, img1, warped0, warped1], dim=1)
        feat = self.head(x_in)

        feat = self.mod1(feat, t)
        res = feat

        align_list = []
        # res = self.alignment[0](res) + res
        align_list.append(res)

        for id, layer in enumerate(self.body):
            res = layer(res, t)
            if id in [3, 7]:
                # res = self.alignment[id // 4 + 1](res) + res
                align_list.append(res)

        # feat = self.mod2(feat, t)
        feat = self.fuse_align(torch.cat(align_list, 1))

        raw_output = self.tail(feat)
        mask = torch.sigmoid(raw_output[:, :1, :, :])
        delta = raw_output[:, 1:, :, :]

        out = mask * warped0 + (1 - mask) * warped1 + delta
        return out
    
    def forward(self, x, t_list=None):
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.training:
            flow01, flow10 = self._get_base_flow(x[:, 0:1], x[:, 1:2])
            out = self.forward_single_t(x[:, 0:1], x[:, 1:2], flow01, flow10, t_list)
            out = out.permute(0, 2, 3, 1).contiguous()
            return out
        else:
            B, T_in, H, W = x.shape
            out_volume = []

            for i in range(T_in - 1):
                img0 = x[:, i : i + 1]
                img1 = x[:, i + 1 : i + 2]
                img0 = self._vol_to_rgb(img0)
                img1 = self._vol_to_rgb(img1)
                flow01, flow10 = self._get_base_flow(img0, img1)
                out_volume.append(img0)

                for t_val in t_list:
                    t_tensor = torch.full((B, 1), t_val, device=x.device, dtype=x.dtype)
                    out_t = self.forward_single_t(img0, img1, flow01, flow10, t_tensor)
                    out_volume.append(out_t)
            out_volume.append(x[:, -1:])
            return torch.cat(out_volume, dim=1)


if __name__ == "__main__":
    import argparse
    import random

    from flowseek.config.parser import parse_args

    def mock_train_batch(batch_size, image_size, z_size=30, max_mid_slices=5, device="cuda"):
        """模拟 data.py __getitem__ 返回的 (lr, gt, t)。"""
        lr_list, gt_list, t_list = [], [], []
        for _ in range(batch_size):
            n_upper = min(z_size - 2, max_mid_slices)
            n_mid = random.randint(1, n_upper)
            span_len = n_mid + 2

            hr_span = torch.rand(image_size, image_size, span_len)
            lr = hr_span[:, :, [0, -1]]
            mid_idx = random.randint(1, n_mid)
            gt = hr_span[:, :, mid_idx : mid_idx + 1]
            t = torch.tensor([mid_idx / (n_mid + 1)], dtype=torch.float32)

            lr_list.append(lr)
            gt_list.append(gt)
            t_list.append(t)

        lr = torch.stack(lr_list, 0).to(device)
        gt = torch.stack(gt_list, 0).to(device)
        t = torch.stack(t_list, 0).to(device)
        return lr, gt, t

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="./model_zoo/flowseek/config/eval/flowseek-S.json"
    )

    args = parse_args(parser)
    args.n_feats = 64
    args.kernel_size = 3
    args.res_scale = 1
    args.i_num_blocks = 16
    args.hr_slice_patch = 1
    args.head_num = 1
    args.win_num_sqrt = 16
    args.image_size = 256
    args.batch_size = 2
    args.max_mid_slices = 5

    device = torch.device("cuda", 0)
    model = ContinuousNet(args).to(device)
    model.train()

    lr, gt, t = mock_train_batch(
        batch_size=args.batch_size,
        image_size=args.image_size,
        z_size=30,
        max_mid_slices=args.max_mid_slices,
        device=device,
    )
    out = model(lr, t)

    print(f"lr:  {lr.shape}")   # [B, H, W, 2]
    print(f"gt:  {gt.shape}")   # [B, H, W, 1]
    print(f"t:   {t.shape}")    # [B, 1]
    print(f"out: {out.shape}")  # [B, H, W, 1]
