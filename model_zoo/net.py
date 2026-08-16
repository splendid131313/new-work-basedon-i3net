import torch
import torch.nn as nn

from .basic_model import default_conv, I2Group, TimeConditionModulation, CrossViewBlock
from .flow_module import FlowEstimator

# from basic_model import default_conv, I2Group, TimeConditionModulation
# from flow_module import FlowEstimator


def make_model(args):
    return Net(args)


class Net(nn.Module):
    def __init__(self, args=None, conv=default_conv):
        super(Net, self).__init__()
        self.args = args
        self.cond_dim = 2
        n_feats = args.n_feats  # 64
        kernel_size = args.kernel_size  # 3
        num_blocks = args.i_num_blocks  # 16
        act = nn.ReLU(True)
        res_scale = args.res_scale  # 1

        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        window_size = args.image_size // args.win_num_sqrt

        self.head = nn.Sequential(conv(4, n_feats, kernel_size), nn.ReLU(), conv(n_feats, n_feats, kernel_size))
        self.flow_estimator = FlowEstimator(args, n_feats, kernel_size)
        # self.mod1 = TimeConditionModulation(n_feats, cond_dim=self.cond_dim)

        self.alignment = nn.ModuleList([CrossViewBlock(n_feats, image_size=args.image_size) for _ in range(3)])

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
                cond_dim=self.cond_dim,
            )
            for _ in range(num_blocks // 2)
        ]
        self.body = nn.ModuleList(modules_body)

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

    def forward_single_t(self, img0, img1, cond):
        # img0, img1: (B, 1, H, W)
        # cond: (B, 2)
        warped0, warped1, flow0t, flow1t = self.flow_estimator(img0, img1, cond)

        x_in = torch.cat([img0, img1, warped0, warped1], dim=1)
        feat = self.head(x_in)
        # feat = self.mod1(feat, cond)
        res = feat

        align_list = []
        res = self.alignment[0](res) + res
        align_list.append(res)
        for id, layer in enumerate(self.body):
            res = layer(res, cond)
            if id in [3, 7]:
                res = self.alignment[id // 4 + 1](res) + res
                align_list.append(res)

        feat = self.fuse_align(torch.cat(align_list, 1))
        raw_output = self.tail(feat)
        mask = torch.sigmoid(raw_output[:, :1, :, :])
        delta = raw_output[:, 1:, :, :]

        out = mask * warped0 + (1 - mask) * warped1 + delta
        return out, mask, warped0, warped1, flow0t, flow1t

    def forward(self, x, cond):
        x = x.permute(0, 3, 1, 2).contiguous()
        out, mask, warped0, warped1, flow0t, flow1t = self.forward_single_t(
            x[:, 0:1], x[:, 1:2], cond
        )
        out = out.permute(0, 2, 3, 1).contiguous()
        return {
            "out": out,
            "mask": mask,
            "warped0": warped0,
            "warped1": warped1,
            "flow0t": flow0t,
            "flow1t": flow1t,
            "img0": x[:, 0:1],
            "img1": x[:, 1:2],
        }

    def inference(self, x, cond, gap=None):
        x = x.permute(0, 3, 1, 2).contiguous()
        B, T_in, H, W = x.shape
        # A regular interpolation list contains gap - 1 target positions.
        # Allow an explicit gap for custom/non-dense query lists.
        if gap is None:
            gap = len(cond) + 1
        gap = float(gap)

        out_volume = []

        for i in range(T_in - 1):
            img0 = x[:, i : i + 1]
            img1 = x[:, i + 1 : i + 2]
            out_volume.append(img0)

            for t_val in cond:
                t_tensor = torch.tensor(
                    [t_val, gap], device=x.device, dtype=x.dtype
                ).view(1, self.cond_dim).expand(B, -1)
                out_t, *_ = self.forward_single_t(img0, img1, t_tensor)
                out_volume.append(out_t)
        out_volume.append(x[:, -1:])
        return torch.cat(out_volume, dim=1)


if __name__ == "__main__":
    import argparse
    import random
    import sys
    from pathlib import Path

    # allow `python model_zoo/net.py` style runs
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from flowseek.config.parser import parse_args

    def mock_train_batch(
        batch_size,
        image_size,
        train_gaps=(2, 3, 4),
        targets_per_span=2,
        device="cuda",
    ):
        lr_list, gt_list, t_list = [], [], []
        targets_per_span = max(1, int(targets_per_span))
        valid_gaps = [int(gap) for gap in train_gaps if int(gap) - 1 >= targets_per_span]

        for _ in range(batch_size):
            gap = random.choice(valid_gaps)
            n_mid = gap - 1
            span_len = gap + 1

            hr_span = torch.rand(image_size, image_size, span_len)
            lr_base = hr_span[:, :, [0, -1]]

            all_mid = list(range(1, n_mid + 1))
            if targets_per_span <= n_mid:
                mid_indices = random.sample(all_mid, targets_per_span)
            else:
                mid_indices = [random.choice(all_mid) for _ in range(targets_per_span)]

            for mid_idx in mid_indices:
                lr_list.append(lr_base)
                gt_list.append(hr_span[:, :, mid_idx : mid_idx + 1])
                t_list.append(
                    torch.tensor([mid_idx / gap, float(gap)], dtype=torch.float32)
                )

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
    args.head_num = 1
    args.win_num_sqrt = 16
    args.image_size = 256
    args.batch_size = 1
    args.train_gaps = [2, 3, 4]
    args.targets_per_span = 2

    device = torch.device("cuda", 0)
    model = Net(args).to(device)
    model.train()

    lr, gt, t = mock_train_batch(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_gaps=args.train_gaps,
        targets_per_span=args.targets_per_span,
        device=device,
    )
    out = model(lr, t)

    print(f"lr:  {lr.shape}")   # [B * targets_per_span, H, W, 2]
    print(f"gt:  {gt.shape}")   # [B * targets_per_span, H, W, 1]
    print(f"t:   {t.shape}")    # [B * targets_per_span, 2]
    print(t[:, :1])
    print(f"out: {out['out'].shape}")  # [B, H, W, 1]
    print(f"mask: {out['mask'].shape}")
