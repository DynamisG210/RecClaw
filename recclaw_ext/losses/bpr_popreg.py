"""Popularity-regularized BPR loss for local RecClaw experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class BPRPopularityRegularizedLoss(nn.Module):
    """BPR loss with a simple popularity-weighted negative-score penalty."""

    def __init__(self, lambda_pop: float = 1e-4) -> None:
        super().__init__()
        self.lambda_pop = float(lambda_pop)

    def forward(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        neg_popularity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_loss = -F.logsigmoid(pos_score - neg_score).mean()
        if neg_popularity is None or self.lambda_pop <= 0:
            return base_loss

        pop_weight = neg_popularity.to(device=neg_score.device, dtype=neg_score.dtype)
        pop_weight = pop_weight / pop_weight.mean().clamp_min(1e-12)
        pop_penalty = (pop_weight * F.softplus(neg_score)).mean()
        return base_loss + self.lambda_pop * pop_penalty
