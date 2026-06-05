"""Residual LightGCN variants with soft embedding norm control."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ._losses import LayerAlignmentLoss, RankAwarePairwiseLoss
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


class LightGCNResidualNormRankAlignment(_ResidualNormLossMixin, LightGCNResidualMix):
    """Residual-norm LightGCN with rank-aware and layer-alignment auxiliary signals."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._init_norm_control(config)
        self.lambda_align = config_float(config, "lambda_align", 1e-4)
        self.rank_weight_alpha = config_float(config, "rank_weight_alpha", 0.1)
        self.alignment_loss = LayerAlignmentLoss(lambda_align=self.lambda_align)
        self.rank_loss = RankAwarePairwiseLoss(rank_weight_alpha=self.rank_weight_alpha)

    def forward_with_layers(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        embeddings_per_layer = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_per_layer.append(all_embeddings)

        mean_embeddings = torch.stack(embeddings_per_layer, dim=0).mean(dim=0)
        residual_weight = max(0.0, min(1.0, self.residual_weight))
        aggregated_embeddings = (1.0 - residual_weight) * mean_embeddings + residual_weight * ego_embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            aggregated_embeddings,
            [self.n_users, self.n_items],
        )
        return user_all_embeddings, item_all_embeddings, embeddings_per_layer

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, layers = self.forward_with_layers()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        pair_weight = (1.0 + torch.sigmoid(neg_scores - pos_scores)).detach()
        mf_loss = self.rank_loss(pos_scores, neg_scores, pair_weight)

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
        node_index = torch.cat([user, pos_item + self.n_users, neg_item + self.n_users], dim=0)
        layer_batch = [layer.index_select(0, node_index) for layer in layers]
        return mf_loss + self.reg_weight * reg_loss + norm_penalty + self.alignment_loss(layer_batch)


class LightGCNEdgeDropoutResidualNormGated(LightGCNEdgeDropoutResidualNorm):
    """Edge-dropout residual-norm LightGCN with a lightweight residual gate."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.residual_gate_scale = config_float(config, "residual_gate_scale", 0.5)
        self.gate_dropout = max(0.0, min(0.95, config_float(config, "gate_dropout", 0.0)))

    def forward(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        embeddings_per_layer = [all_embeddings]
        propagation_matrix = self._propagation_matrix()

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(propagation_matrix, all_embeddings)
            embeddings_per_layer.append(all_embeddings)

        mean_embeddings = torch.stack(embeddings_per_layer, dim=0).mean(dim=0)
        residual_weight = max(0.0, min(1.0, self.residual_weight))
        norm_delta = ego_embeddings.norm(dim=1, keepdim=True) - mean_embeddings.norm(dim=1, keepdim=True)
        gate = torch.sigmoid(float(self.residual_gate_scale) * norm_delta)
        if self.training and self.gate_dropout > 0:
            gate = F.dropout(gate, p=self.gate_dropout, training=True)
        dynamic_residual = residual_weight * gate
        aggregated_embeddings = (1.0 - dynamic_residual) * mean_embeddings + dynamic_residual * ego_embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            aggregated_embeddings,
            [self.n_users, self.n_items],
        )
        return user_all_embeddings, item_all_embeddings
