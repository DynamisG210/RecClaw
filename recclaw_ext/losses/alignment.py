"""Auxiliary alignment loss stub for local LightGCN experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LayerAlignmentLoss(nn.Module):
    """Encourage propagation-layer embeddings to stay aligned."""

    def __init__(self, lambda_align: float = 1e-3) -> None:
        super().__init__()
        self.lambda_align = float(lambda_align)

    def forward(self, layer_embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(layer_embeddings) < 2 or self.lambda_align <= 0:
            device = layer_embeddings[0].device if layer_embeddings else torch.device("cpu")
            return torch.zeros((), device=device)

        target = F.normalize(layer_embeddings[-1].detach(), dim=-1)
        penalties = [
            F.mse_loss(F.normalize(layer, dim=-1), target)
            for layer in layer_embeddings[:-1]
        ]
        return self.lambda_align * torch.stack(penalties).mean()
