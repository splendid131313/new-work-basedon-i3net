import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .models.VoxelMorph.model import VoxelMorph, SpatialTransformer
from .models.UNet.model import Unet3D_multi, Unet3D
from .models.feature_extract.model import FeatureExtract


def make_model(args):
    return UVINet(args)


class _FlowEstimatorCompat:
    """Expose `_get_base_flow` for Select_Loss near-anchor path."""

    def __init__(self, net):
        self._net = net

    def _get_base_flow(self, img0, img1):
        return self._net._get_base_flow(img0, img1)


class UVINet(nn.Module):
    """
    UVI-Net style pipeline adapted to I3Net's 2D slice + time-condition interface.

    - weight_cycle == 0: supervised interpolation at t vs GT
    - weight_cycle  > 0: unsupervised cycle path (UVI original)
    """

    def __init__(self, args=None):
        super(UVINet, self).__init__()
        self.args = args
        image_size = int(args.image_size)
        self.inshape = (image_size, image_size)
        self.feature_extract = bool(getattr(args, "feature_extract", True))
        self.weight_cycle = float(getattr(args, "weight_cycle", 0.0))

        self.flow_model = VoxelMorph(self.inshape)
        if self.feature_extract:
            self.feature_model = FeatureExtract(ndims=2)
            self.refinement_model = Unet3D_multi(self.inshape)
        else:
            self.feature_model = None
            self.refinement_model = Unet3D(self.inshape)

        self.flow_estimator = _FlowEstimatorCompat(self)
        self.transformer = SpatialTransformer(self.inshape)
        self.feat_transformers = nn.ModuleList(
            [
                SpatialTransformer(tuple(s // (2**idx) for s in self.inshape))
                for idx in range(3)
            ]
        )

    def _get_base_flow(self, img0, img1):
        i0_i1 = torch.cat([img0, img1], dim=1)
        _, _, flow_0_1, flow_1_0 = self.flow_model(i0_i1)
        return flow_0_1, flow_1_0

    def _warp(self, img, flow):
        return self.transformer(img, flow)

    def _warp_feat_lists(self, feat_list_0, feat_list_1, flow_0_t, flow_1_t):
        warped_0, warped_1 = [], []
        for idx in range(len(feat_list_0)):
            scale = 0.5**idx
            flow0 = F.interpolate(
                flow_0_t * scale, scale_factor=scale, mode="bilinear", align_corners=True
            )
            flow1 = F.interpolate(
                flow_1_t * scale, scale_factor=scale, mode="bilinear", align_corners=True
            )
            warped_0.append(self.feat_transformers[idx](feat_list_0[idx], flow0))
            warped_1.append(self.feat_transformers[idx](feat_list_1[idx], flow1))
        return warped_0, warped_1

    def interpolate_at_t(self, i0, i1, flow_0_1, flow_1_0, t):
        """Supervised temporal interpolation at t in [0, 1]. Returns (B, 1, H, W)."""
        B = i0.shape[0]
        t = t.view(B, 1, 1, 1)

        flow_0_t = flow_0_1 * t
        flow_1_t = flow_1_0 * (1.0 - t)
        warped0 = self._warp(i0, flow_0_t)
        warped1 = self._warp(i1, flow_1_t)
        combined = (1.0 - t) * warped0 + t * warped1

        if self.feature_extract:
            feat0 = self.feature_model(i0)
            feat1 = self.feature_model(i1)
            feat0_t, feat1_t = self._warp_feat_lists(feat0, feat1, flow_0_t, flow_1_t)
            residual = self.refinement_model(combined, feat0_t, feat1_t)
        else:
            residual = self.refinement_model(combined)

        return combined + residual

    def cycle_interpolation(self, flow_0_1, flow_1_0, i0, i1):
        alpha1 = random.uniform(-0.5, 0.0)
        alpha2 = random.uniform(0.0, 1.0)
        alpha3 = random.uniform(1.0, 1.5)

        flow_0_a1 = flow_0_1 * alpha1
        i_0_a1 = self._warp(i0, flow_0_a1)

        if alpha2 < 0.5:
            flow_0_a2 = flow_0_1 * alpha2
            i_unknown_a2 = self._warp(i0, flow_0_a2)
        else:
            flow_1_a2 = flow_1_0 * (1 - alpha2)
            i_unknown_a2 = self._warp(i1, flow_1_a2)

        flow_1_a3 = flow_1_0 * (1 - alpha3)
        i_1_a3 = self._warp(i1, flow_1_a3)

        ia1_ia2 = torch.cat((i_0_a1, i_unknown_a2), dim=1)
        ia2_ia3 = torch.cat((i_unknown_a2, i_1_a3), dim=1)

        _, _, flow_a1_a2, flow_a2_a1 = self.flow_model(ia1_ia2)
        _, _, flow_a2_a3, flow_a3_a2 = self.flow_model(ia2_ia3)

        alpha12 = (0 - alpha1) / (alpha2 - alpha1)
        alpha23 = (1 - alpha2) / (alpha3 - alpha2)

        flow_a1_0 = flow_a1_a2 * alpha12
        flow_a2_0 = flow_a2_a1 * (1 - alpha12)
        flow_a2_1 = flow_a2_a3 * alpha23
        flow_a3_1 = flow_a3_a2 * (1 - alpha23)

        i_a1_0 = self._warp(i_0_a1, flow_a1_0)
        i_a2_0 = self._warp(i_unknown_a2, flow_a2_0)
        i_a2_1 = self._warp(i_unknown_a2, flow_a2_1)
        i_a3_1 = self._warp(i_1_a3, flow_a3_1)

        i0_combined = (1 - alpha12) * i_a1_0 + alpha12 * i_a2_0
        i1_combined = (1 - alpha23) * i_a2_1 + alpha23 * i_a3_1

        if self.feature_extract:
            x_feat_a1_list = self.feature_model(i_0_a1)
            x_feat_a2_list = self.feature_model(i_unknown_a2)
            x_feat_a3_list = self.feature_model(i_1_a3)
            x_feat_a1_0_list, x_feat_a2_0_list = [], []
            x_feat_a2_1_list, x_feat_a3_1_list = [], []

            for feat_idx in range(len(x_feat_a1_list)):
                st = self.feat_transformers[feat_idx]
                scale = 0.5**feat_idx
                x_feat_a1_0_list.append(
                    st(
                        x_feat_a1_list[feat_idx],
                        F.interpolate(
                            flow_a1_0 * scale,
                            scale_factor=scale,
                            mode="bilinear",
                            align_corners=True,
                        ),
                    )
                )
                x_feat_a2_0_list.append(
                    st(
                        x_feat_a2_list[feat_idx],
                        F.interpolate(
                            flow_a2_0 * scale,
                            scale_factor=scale,
                            mode="bilinear",
                            align_corners=True,
                        ),
                    )
                )
                x_feat_a2_1_list.append(
                    st(
                        x_feat_a2_list[feat_idx],
                        F.interpolate(
                            flow_a2_1 * scale,
                            scale_factor=scale,
                            mode="bilinear",
                            align_corners=True,
                        ),
                    )
                )
                x_feat_a3_1_list.append(
                    st(
                        x_feat_a3_list[feat_idx],
                        F.interpolate(
                            flow_a3_1 * scale,
                            scale_factor=scale,
                            mode="bilinear",
                            align_corners=True,
                        ),
                    )
                )

            i0_out_diff = self.refinement_model(
                i0_combined, x_feat_a1_0_list, x_feat_a2_0_list
            )
            i1_out_diff = self.refinement_model(
                i1_combined, x_feat_a2_1_list, x_feat_a3_1_list
            )
        else:
            i0_out_diff = self.refinement_model(i0_combined)
            i1_out_diff = self.refinement_model(i1_combined)

        i0_out = i0_combined + i0_out_diff
        i1_out = i1_combined + i1_out_diff
        return i0_out, i1_out, i0_out_diff, i1_out_diff

    def forward(self, x, cond=None):
        """
        x: (B, H, W, 2) endpoints
        cond: (B, 2) with t = cond[:, 0] — required when weight_cycle == 0
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 2, H, W)
        i0, i1 = x[:, 0:1], x[:, 1:2]
        i_0_1, i_1_0, flow_0_1, flow_1_0 = self.flow_model(x)

        if self.weight_cycle == 0.0:
            if cond is None:
                raise ValueError(
                    "cond (time t) is required for supervised mode (weight_cycle=0)"
                )
            pred = self.interpolate_at_t(i0, i1, flow_0_1, flow_1_0, cond[:, :1])
            return {
                "out": pred,
                "i_0_1": i_0_1,
                "i_1_0": i_1_0,
                "flow_0_1": flow_0_1,
                "flow_1_0": flow_1_0,
            }

        i0_out, i1_out, i0_out_diff, i1_out_diff = self.cycle_interpolation(
            flow_0_1, flow_1_0, i0, i1
        )
        return (
            i_0_1,
            i_1_0,
            flow_0_1,
            flow_1_0,
            i0_out,
            i1_out,
            i0_out_diff,
            i1_out_diff,
        )


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    def mock_train_batch(batch_size, image_size, device="cpu"):
        lr_list, gt_list, t_list = [], [], []
        for _ in range(batch_size):
            lr_list.append(torch.rand(image_size, image_size, 2))
            gt_list.append(torch.rand(image_size, image_size, 1))
            t_list.append(torch.tensor([random.random(), 0.5], dtype=torch.float32))
        lr = torch.stack(lr_list, 0).to(device)
        gt = torch.stack(gt_list, 0).to(device)
        t = torch.stack(t_list, 0).to(device)
        return lr, gt, t

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    cli = parser.parse_args()

    args = argparse.Namespace(
        image_size=cli.image_size,
        feature_extract=True,
        weight_cycle=0.0,
    )
    device = torch.device(cli.device)
    model = UVINet(args).to(device)
    model.train()

    lr, gt, t = mock_train_batch(cli.batch_size, cli.image_size, device=device)
    out = model(lr, t)
    gt_nchw = gt.permute(0, 3, 1, 2)
    loss = F.l1_loss(out["out"], gt_nchw)
    loss.backward()

    print(f"lr:   {lr.shape}")
    print(f"gt:   {gt.shape}")
    print(f"t:    {t.shape}")
    print(f"out:  {out['out'].shape}")
    print(f"loss: {loss.item():.6f}")
    print("smoke ok")
