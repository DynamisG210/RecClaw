"""Residual LightGCN variants with soft embedding norm control."""

from __future__ import annotations

import torch

from ._utils import config_float, soft_l2_norm_penalty
from .lightgcn_edge_dropout_residual import LightGCNEdgeDropoutResidualMix
from .lightgcn_residual import LightGCNResidualMix


class _ResidualNormLossMixin:
    def _init_norm_control(self, config) -> None:
        self.lambda_norm = config_float(config, "lambda_norm", 1e-4)
        self.max_norm = config_float(config, "max_norm", 1.0)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

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
        norm_penalty = soft_l2_norm_penalty(
            u_embeddings,
            pos_embeddings,
            neg_embeddings,
            max_norm=self.max_norm,
            weight=self.lambda_norm,
        )
        return mf_loss + self.reg_weight * reg_loss + norm_penalty


class LightGCNResidualNormConstrained(_ResidualNormLossMixin, LightGCNResidualMix):
    """Residual LightGCN with a soft norm penalty on propagated embeddings."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_norm_control(config)


class LightGCNEdgeDropoutResidualNorm(_ResidualNormLossMixin, LightGCNEdgeDropoutResidualMix):
    """Edge-dropout residual LightGCN with soft propagated-embedding norm control."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_norm_control(config)
