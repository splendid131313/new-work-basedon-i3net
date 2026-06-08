from i3net.basic_model import *

def make_model(args):
    return Net(args)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        n_feats = args.n_feats
        channels = args.channels
        in_slice = args.lr_slice_patch
        out_slice = args.hr_slice_patch
        kernel_size = args.kernel_size
        head_num = args.head_num
        win_num_sqrt = args.win_num_sqrt
        image_size = args.image_size 
        num_blocks = args.num_blocks

        self.encoder = SliceEncoder(c_in=in_slice, channels=channels, k=kernel_size)

        self.rfb1 = RFB_modified(channels[0], n_feats)
        self.rfb2 = RFB_modified(channels[1], n_feats)
        self.rfb3 = RFB_modified(channels[2], n_feats)
        self.rfb4 = RFB_modified(channels[3], n_feats)

        self.decoder = Decoder(channels, out_slice, kernel_size)

        self.down = nn.Conv2d(
            in_channels=3 * n_feats, out_channels=n_feats, kernel_size=1
        )

        self.high = nn.ModuleList(
            [
                IntraSliceBranch(
                    n_feat=n_feats,
                    kernel_size=kernel_size,
                    head_num=head_num,
                    win_num_sqrt=win_num_sqrt,
                    image_size=image_size,
                )
                for _ in range(num_blocks // 2)
            ]
        )

        self.low = nn.ModuleList(
            [
                IntraSliceBranch(
                    n_feat=n_feats,
                    kernel_size=kernel_size,
                    head_num=head_num,
                    win_num_sqrt=win_num_sqrt,
                    image_size=image_size,
                )
                for _ in range(num_blocks // 4)
            ]
        )

        self.align1 = LocalMMF(in_ch=n_feats, win_size=7)
        self.align2 = LocalMMF(in_ch=n_feats, win_size=7)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x1, x2, x3, x4 = self.encoder(x)

        res1, res2, res3, res4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)

        res1h, res1w = res1.size()[-2:]
        res2 = F.interpolate(res2, size=(res1h, res1w), mode="bicubic", align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode="bicubic", align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode="bicubic", align_corners=False)

        high = res1
        for block in self.high:
            high = block(high) + high

        low = torch.cat([res2, res3, res4], dim=1)
        low = self.down(low)
        for block in self.low:
            low = block(low) + low

        high_align = self.align1(high, res1)
        low_align = self.align2(low, res1)

        align = torch.cat([high_align, low_align], dim=1)

        out = self.decoder(align)
        out[:, :: self.args.upscale] = x
        out = out.permute(0, 2, 3, 1).contiguous()

        return out


if __name__ == "__main__":
    import argparse

    args = argparse.Namespace()
    args.upscale = 2
    args.n_feats = 64
    args.kernel_size = 3
    args.res_scale = 1
    args.num_blocks = 16
    args.lr_slice_patch = 4
    args.hr_slice_patch = (args.lr_slice_patch-1)*args.upscale + 1
    args.head_num = 1
    args.win_num_sqrt = 16
    args.image_size = 256
    args.channels = [64, 128, 320, 512]
    args.conv_depths = [2, 2, 2, 2]
    args.dropout = 0.1

    gpy_id = 0
    model = Net(args).cuda(gpy_id)
    x = torch.ones(1,args.image_size,args.image_size,args.lr_slice_patch).cuda(gpy_id)
    y = torch.ones(1,args.image_size,args.image_size,args.hr_slice_patch).cuda(gpy_id)
    pred=model(x)
    print(pred.shape)