import torch
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


