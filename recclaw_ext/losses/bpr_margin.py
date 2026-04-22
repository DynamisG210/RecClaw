"""Margin-aware BPR loss for local RecClaw experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BPRMarginLoss(nn.Module):
    """BPR loss with an explicit positive-negative score margin."""

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(pos_score - neg_score - self.margin).mean()
