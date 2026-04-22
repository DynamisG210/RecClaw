"""Long-tail reweighted BPR loss stub for local RecClaw experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class BPRLongTailReweightLoss(nn.Module):
    """BPR loss with optional per-pair tail-aware weights."""

    def __init__(self, tail_weight_alpha: float = 0.5) -> None:
        super().__init__()
        self.tail_weight_alpha = float(tail_weight_alpha)

    def forward(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        item_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pair_loss = -F.logsigmoid(pos_score - neg_score)
        if item_weight is None or self.tail_weight_alpha <= 0:
            return pair_loss.mean()

        weight = item_weight.to(device=pair_loss.device, dtype=pair_loss.dtype)
        weight = weight / weight.mean().clamp_min(1e-12)
        weight = weight.pow(self.tail_weight_alpha)
        return (pair_loss * weight).mean()
