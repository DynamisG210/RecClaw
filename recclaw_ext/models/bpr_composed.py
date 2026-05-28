"""Composable BPR variants built from existing sampling and loss helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ._losses import BPRMarginLoss
from ._utils import config_float
from .bpr_regularized import _item_frequency
from .bpr_sampling import (
    BPRHardNegative,
    BPRPopularityAwareNegative,
    _repair_invalid_negatives,
)


class BPRHardNegativeMargin(BPRHardNegative):
    """BPR with mixed hard negatives and an explicit pairwise margin."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.margin = config_float(config, "margin", 0.2)
        self.margin_loss = BPRMarginLoss(margin=self.margin)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = self.negative_sampler.sample(pos_item.shape, device=pos_item.device)
        neg_item = _repair_invalid_negatives(neg_item.long(), pos_item.long(), self.n_items)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.margin_loss(pos_score, neg_score)


class BPRPopularityAwareMargin(BPRPopularityAwareNegative):
    """BPR with popularity-aware negatives and an explicit pairwise margin."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.margin = config_float(config, "margin", 0.2)
        self.margin_loss = BPRMarginLoss(margin=self.margin)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = self.negative_sampler.sample(pos_item.shape, device=pos_item.device)
        neg_item = _repair_invalid_negatives(neg_item.long(), pos_item.long(), self.n_items)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.margin_loss(pos_score, neg_score)


class BPRHardNegativeMarginTailReweight(BPRHardNegative):
    """BPR hard-negative margin objective with long-tail positive-item weights."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.margin = config_float(config, "margin", 0.2)
        self.tail_weight_alpha = config_float(config, "tail_weight_alpha", 0.2)
        item_frequency = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        tail_weight = item_frequency.rsqrt()
        tail_weight = tail_weight / tail_weight.mean().clamp_min(1e-12)
        self.register_buffer("item_tail_weight", tail_weight)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = self.negative_sampler.sample(pos_item.shape, device=pos_item.device)
        neg_item = _repair_invalid_negatives(neg_item.long(), pos_item.long(), self.n_items)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        pair_loss = -F.logsigmoid(pos_score - neg_score - self.margin)
        weights = self.item_tail_weight[pos_item].to(device=pair_loss.device, dtype=pair_loss.dtype)
        weights = weights / weights.mean().clamp_min(1e-12)
        if self.tail_weight_alpha > 0:
            weights = weights.pow(self.tail_weight_alpha)
        return (pair_loss * weights).mean()


class BPRRankAwareHardNegativeMargin(BPRHardNegative):
    """BPR hard-negative margin objective with a rank-aware pair weight."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.margin = config_float(config, "margin", 0.2)
        self.rank_weight_alpha = config_float(config, "rank_weight_alpha", 0.1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = self.negative_sampler.sample(pos_item.shape, device=pos_item.device)
        neg_item = _repair_invalid_negatives(neg_item.long(), pos_item.long(), self.n_items)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        pair_loss = -F.logsigmoid(pos_score - neg_score - self.margin)
        if self.rank_weight_alpha <= 0:
            return pair_loss.mean()
        pair_weight = (1.0 + torch.sigmoid(neg_score - pos_score)).detach()
        pair_weight = pair_weight / pair_weight.mean().clamp_min(1e-12)
        pair_weight = pair_weight.pow(self.rank_weight_alpha)
        return (pair_loss * pair_weight).mean()
