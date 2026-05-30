"""Best v4m LightGCN dual-path residual-norm candidate."""

from __future__ import annotations

import torch

from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss

from recclaw_ext.models._utils import config_float, soft_l2_norm_penalty


class LightGCNEdgeDropoutResidualNormDualPathBlend(LightGCN):
    """LightGCN with clean/dropout propagation paths, residual mix, and norm control."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.residual_weight = config_float(config, "residual_weight", 0.1)
        self.edge_dropout = config_float(config, "edge_dropout", 0.1)
        self.lambda_norm = config_float(config, "lambda_norm", 0.0)
        self.max_norm = config_float(config, "max_norm", 1.0)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.apply(xavier_uniform_initialization)

    def _edge_dropped_graph(self):
        if (not self.training) or self.edge_dropout <= 0.0:
            return self.norm_adj_matrix

        graph = self.norm_adj_matrix.coalesce()
        indices = graph.indices()
        values = graph.values()
        keep_prob = max(1.0 - self.edge_dropout, 1e-12)
        mask = torch.rand(values.size(), device=values.device) < keep_prob
        if mask.sum() == 0:
            mask[torch.randint(0, values.size(0), (1,), device=values.device)] = True

        dropped_indices = indices[:, mask]
        dropped_values = values[mask] / keep_prob
        return torch.sparse_coo_tensor(dropped_indices, dropped_values, graph.size()).coalesce()

    def computer(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        embeddings_per_layer = [all_embeddings]

        clean_graph = self.norm_adj_matrix
        dropped_graph = self._edge_dropped_graph()
        residual_weight = max(0.0, min(1.0, self.residual_weight))

        for _ in range(self.n_layers):
            clean_embeddings = torch.sparse.mm(clean_graph, all_embeddings)
            dropped_embeddings = torch.sparse.mm(dropped_graph, all_embeddings)
            propagated = 0.5 * (clean_embeddings + dropped_embeddings)
            all_embeddings = (1.0 - residual_weight) * propagated + residual_weight * ego_embeddings
            embeddings_per_layer.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_per_layer, dim=1).mean(dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings,
            [self.n_users, self.n_items],
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.computer()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        norm_penalty = soft_l2_norm_penalty(
            u_embeddings,
            pos_embeddings,
            neg_embeddings,
            max_norm=self.max_norm,
            weight=self.lambda_norm,
        )
        return mf_loss + self.reg_weight * reg_loss + norm_penalty
