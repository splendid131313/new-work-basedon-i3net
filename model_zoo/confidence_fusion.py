"""Residual fusion of implicit reconstruction and explicit geometry features."""

import torch
from torch import nn


class ConfidenceFusionBlock(nn.Module):
    """Return a residual update, to be used as ``res = res + block(res, geo)``.

    Inputs: implicit_feat and geometry_feat both have shape [B, C, H, W].
    Confidence channels: 0 = implicit, 1 = explicit; their sum is one.
    These are learned relative fusion weights, not calibrated probabilities.
    """

    def __init__(self, n_feats, implicit_bias=2.0):
        super().__init__()
        # Each insertion stage learns its own mapping into the current feature space.
        self.geo_adapter = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        # Normalization is used for confidence estimation only.
        self.implicit_norm = nn.GroupNorm(1, n_feats)
        self.explicit_norm = nn.GroupNorm(1, n_feats)
        self.conf_net = nn.Sequential(
            nn.Conv2d(3 * n_feats, n_feats, 1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_feats, 2, 1),
        )
        # Start near the implicit path while allowing both branches to learn.
        nn.init.zeros_(self.conf_net[-1].weight)
        with torch.no_grad():
            self.conf_net[-1].bias.copy_(torch.tensor([implicit_bias, 0.0]))

    def forward(self, implicit_feat, geometry_feat, return_confidence=False):
        if implicit_feat.shape != geometry_feat.shape:
            raise ValueError(
                "Implicit and geometry features must have the same shape; "
                f"got {tuple(implicit_feat.shape)} and {tuple(geometry_feat.shape)}"
            )

        explicit_feat = self.geo_adapter(geometry_feat)
        implicit_context = self.implicit_norm(implicit_feat)
        explicit_context = self.explicit_norm(explicit_feat)
        evidence = torch.cat(
            [implicit_context, explicit_context,
             torch.abs(implicit_context - explicit_context)],
            dim=1,
        )
        confidence = self.conf_net(evidence).softmax(dim=1)
        explicit_weight = confidence[:, 1:2]

        # implicit + delta = c_implicit * implicit + c_explicit * explicit.
        # Do not return the fused feature here: the caller adds the residual.
        delta = explicit_weight * (explicit_feat - implicit_feat)
        if return_confidence:
            return delta, confidence
        return delta
