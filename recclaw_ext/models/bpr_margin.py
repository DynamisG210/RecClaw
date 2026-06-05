"""BPR variant that actually wires RecClaw's margin-aware loss into training."""

from __future__ import annotations

import torch
from recbole.model.general_recommender.bpr import BPR

from ._losses import BPRMarginLoss


class BPRMargin(BPR):
    """RecBole BPR with a positive-negative score margin in calculate_loss."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.margin = float(config["margin"])
        self.loss = BPRMarginLoss(margin=self.margin)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)

        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.loss(pos_score, neg_score)
