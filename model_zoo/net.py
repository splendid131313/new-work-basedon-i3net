import torch
import torch.nn as nn
import torch.nn.functional as F
# from .i3net.basic_model import default_conv
# from .i3net.frequency_aware import FrequencyAwareGroup
# from .i3net.motion_aware import MotionAwareGroup

from i3net.basic_model import default_conv
from i3net.frequency_aware import FrequencyAwareGroup
from i3net.motion_aware import MotionAwareGroup

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
        in_slice = args.lr_slice_patch * 1
        out_slice = args.hr_slice_patch

        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        window_size = args.image_size // args.win_num_sqrt
        lambda_flow = args.lambda_flow
        
        self.frequency = FrequencyAwareGroup(conv=default_conv, in_slice=in_slice, n_feats=n_feats, kernel_size=kernel_size, head_num=head_num, win_num_sqrt=win_num_sqrt, window_size=window_size, num_blocks=num_blocks)

        self.motion = MotionAwareGroup(args, n_feats=n_feats, kernel_size=kernel_size, head_num=head_num, lambda_flow=lambda_flow, window_size=window_size)

        self.motion_gate = nn.Sequential(
            nn.Conv2d(n_feats,n_feats,3,1,1),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(n_feats * 3, n_feats, 1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )

        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(),
            conv(n_feats, out_slice, kernel_size),
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()

        frequency = self.frequency(x)
        Hf, Wf = frequency.shape[-2:]

        motion = self.motion(x)
        motion = F.interpolate(motion, size=(Hf, Wf), mode="bilinear", align_corners=False)

        gate = self.motion_gate(motion)
        fused = motion * gate + frequency * (1 - gate)
        fused = self.fuse(torch.cat([frequency, motion, fused], dim=1))
        out = self.tail(fused)

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
    args.lambda_flow = 10.0
    args.finetune_flowseek = True
    args.flowseek_forward_iters = 6
    args.flowseek_finetune_scope = "minimal"
    args.flow_lr_ratio = 0.1

    gpy_id = 0
    model = Net(args).cuda(gpy_id)
    x = torch.ones(2, args.image_size, args.image_size, args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(2, args.image_size, args.image_size, args.hr_slice_patch).cuda(gpy_id)
    out = model(x)
