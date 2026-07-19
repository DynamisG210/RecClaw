"""Internal loss building blocks for model-entry RecClaw candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

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
