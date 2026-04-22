"""Norm-constrained BPR loss stub for local RecClaw experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class BPRNormConstrainedLoss(nn.Module):
    """BPR loss with a soft embedding norm penalty."""

    def __init__(self, lambda_norm: float = 1e-4, max_norm: float = 1.0) -> None:
        super().__init__()
        self.lambda_norm = float(lambda_norm)
        self.max_norm = float(max_norm)

    def forward(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        user_embedding: Optional[torch.Tensor] = None,
        pos_embedding: Optional[torch.Tensor] = None,
        neg_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_loss = -F.logsigmoid(pos_score - neg_score).mean()
        embeddings = [
            emb
            for emb in (user_embedding, pos_embedding, neg_embedding)
            if emb is not None
        ]
        if not embeddings or self.lambda_norm <= 0:
            return base_loss

        penalty = torch.stack(
            [
                F.relu(emb.norm(dim=-1) - self.max_norm).pow(2).mean()
                for emb in embeddings
            ]
        ).mean()
        return base_loss + self.lambda_norm * penalty
