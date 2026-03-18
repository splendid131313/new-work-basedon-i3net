import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(x, flow):

    B, C, H, W = x.size()

    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))

    grid = torch.stack((grid_x, grid_y), 2).to(x.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    flow = flow.permute(0, 2, 3, 1)

    new_grid = grid + flow

    out = F.grid_sample(x, new_grid, align_corners=True)

    return out

class TrajectoryFlowNet(nn.Module):
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

    def forward(self, f0, f3, t):

        B, C, H, W = f0.shape

        t_map = torch.ones(B, 1, H, W, device=f0.device) * t

        x = torch.cat([f0, f3, t_map], dim=1)

        flow = self.encoder(x)

        return flow


class SliceSynthesis(nn.Module):
    def __init__(self, n_feat):

        super().__init__()

        self.flow_net = TrajectoryFlowNet(n_feat)

        self.decoder = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feat, 1, 3, 1, 1),
        )

    def forward(self, fs, fe, t):

        flow = self.flow_net(fs, fe, t)

        warps = warp(fs, flow * t)
        warpe = warp(fe, -flow * (1 - t))

        x = torch.cat([warps, warpe], dim=1)

        out = self.decoder(x)

        return out

# class RelationRefine(nn.Module):
#     def __init__(self, n_feat):

#         super().__init__()

#         self.refine = nn.Sequential(
#             nn.Conv2d(n_feat * 3, n_feat, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(n_feat, n_feat, 3, 1, 1),
#         )

#     def forward(self, relation, f_mid, f0):

#         x = torch.cat([relation, f_mid, f0], dim=1)

#         delta = self.refine(x)

#         return relation + delta 