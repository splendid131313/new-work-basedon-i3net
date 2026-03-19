import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(x, flow):
    B, C, H, W = x.size()
    xx = torch.linspace(-1.0, 1.0, W, device=x.device)
    yy = torch.linspace(-1.0, 1.0, H, device=x.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).expand(B, -1, -1, -1)

    flow_norm = torch.cat(
        [
            flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
        ],
        dim=1,
    )
    grid = grid + flow_norm.permute(0, 2, 3, 1)

    return F.grid_sample(x, grid, padding_mode="border", align_corners=True)

class FlowEstimator(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_feat * 2 + 1, n_feat * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat * 4, n_feat * 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat * 4, n_feat * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat * 2, 2, 3, 1, 1),  # flow
        )

    def forward(self, fs, fe, t):
        B, C, H, W = fs.shape
        t_map = torch.ones(B, 1, H, W, device=fs.device) * t
        x = torch.cat([fs, fe, t_map], dim=1)
        flow = self.encoder(x)
        return flow


class RelationRefine(nn.Module):
    def __init__(self, n_feat):

        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(n_feat * 4 + 2, n_feat, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        )

    def forward(self, fs, fe, warps, warpe, flow):
        relation = torch.cat([fs, fe, warps, warpe, flow], dim=1)
        delta = self.refine(relation)
        return warps + warpe + delta


class FlowModule(nn.Module):
    def __init__(self, n_feat):

        super().__init__()

        self.flow_estimator = FlowEstimator(n_feat)

        # TODO: add mask net
        # self.mask_net = nn.Sequential(
        #     nn.Conv2d(in_channels * 2 + 2, 16, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid(),
        # )

        self.relation_refine = RelationRefine(n_feat)
        self.fusion = nn.Sequential(
            nn.Conv2d(n_feat * 3, n_feat, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        )

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(n_feat, 1, 3, 1, 1),
        # )

    def forward(self, fs, fe, t=0.5):

        flow = self.flow_estimator(fs, fe, t)

        warps = warp(fs, flow * t)
        warpe = warp(fe, -flow * (1 - t))

        # TODO: add mask net
        # combined = torch.cat([warps, warpe, -flow * (1 - t)], dim=1)
        # mask = self.mask_net(combined)

        relation = self.relation_refine(fs, fe, warps, warpe, flow)
        out = self.fusion(torch.cat([fs, fe, relation], dim=1))

        # out = self.decoder(relation)

        return out, flow

