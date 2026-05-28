"""Local LightGCN variants with model-level negative sampling."""

from __future__ import annotations

import torch
from recbole.model.general_recommender.lightgcn import LightGCN

from ._samplers import DebiasedNegativeSampler
from ._utils import config_float
from .bpr_sampling import _item_frequency, _repair_invalid_negatives


class LightGCNDebiasedNegative(LightGCN):
    """LightGCN that replaces sampled negatives with inverse-popularity negatives."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.debias_alpha = config_float(config, "debias_alpha", 0.5)
        popularity = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        self.negative_sampler = DebiasedNegativeSampler(
            popularity,
            alpha=self.debias_alpha,
            avoid_zero=True,
        )

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = self.negative_sampler.sample(pos_item.shape, device=pos_item.device)
        neg_item = _repair_invalid_negatives(neg_item.long(), pos_item.long(), self.n_items)

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )
        return mf_loss + self.reg_weight * reg_loss
