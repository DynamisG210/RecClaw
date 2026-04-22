"""Rank-aware pairwise loss stub for local RecClaw experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class RankAwarePairwiseLoss(nn.Module):
    """Pairwise loss that can upweight rank-sensitive training pairs."""

    def __init__(self, rank_weight_alpha: float = 0.5) -> None:
        super().__init__()
        self.rank_weight_alpha = float(rank_weight_alpha)

    def forward(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        pair_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pair_loss = -F.logsigmoid(pos_score - neg_score)
        if pair_weight is None or self.rank_weight_alpha <= 0:
            return pair_loss.mean()

        weight = pair_weight.to(device=pair_loss.device, dtype=pair_loss.dtype)
        weight = weight / weight.mean().clamp_min(1e-12)
        weight = weight.pow(self.rank_weight_alpha)
        return (pair_loss * weight).mean()
