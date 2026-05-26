"""Composable BPR variants built from existing sampling and loss helpers."""

from __future__ import annotations

import torch

from ._losses import BPRMarginLoss
from ._utils import config_float
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
