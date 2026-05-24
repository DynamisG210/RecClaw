"""Local LightGCN objective variants for RecClaw experiments."""

from __future__ import annotations

import torch
from recbole.model.general_recommender.lightgcn import LightGCN

from ._losses import LayerAlignmentLoss, RankAwarePairwiseLoss


def _config_float(config, key: str, default: float) -> float:
    try:
        value = config[key]
    except Exception:  # noqa: BLE001 - RecBole config is mapping-like, not a plain dict.
        value = default
    if value is None or str(value).strip() == "":
        value = default
    return float(value)


class _LayerTrackingLightGCN(LightGCN):
    def forward_with_layers(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_per_layer = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_per_layer.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_per_layer, dim=1).mean(dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings,
            [self.n_users, self.n_items],
        )
        return user_all_embeddings, item_all_embeddings, embeddings_per_layer


class LightGCNAuxAlignment(_LayerTrackingLightGCN):
    """LightGCN with a lightweight layer-alignment auxiliary objective."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.lambda_align = _config_float(config, "lambda_align", 1e-3)
        self.alignment_loss = LayerAlignmentLoss(lambda_align=self.lambda_align)

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

        node_index = torch.cat([user, pos_item + self.n_users, neg_item + self.n_users], dim=0)
        layer_batch = [layer.index_select(0, node_index) for layer in layers]
        return mf_loss + self.reg_weight * reg_loss + self.alignment_loss(layer_batch)


class LightGCNRankAware(_LayerTrackingLightGCN):
    """LightGCN with rank-aware weighting on the pairwise objective."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.rank_weight_alpha = _config_float(config, "rank_weight_alpha", 0.5)
        self.rank_loss = RankAwarePairwiseLoss(rank_weight_alpha=self.rank_weight_alpha)

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward_with_layers()
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
        return mf_loss + self.reg_weight * reg_loss
