"""Local residual-mix LightGCN scaffold for RecClaw experiments."""

from __future__ import annotations

import torch
from recbole.model.general_recommender.lightgcn import LightGCN


class LightGCNResidualMix(LightGCN):
    """LightGCN with a small residual contribution from ego embeddings."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.residual_weight = float(
            config["residual_weight"] if "residual_weight" in config else 0.2
        )

    def forward(self):
        ego_embeddings = self.get_ego_embeddings()
        all_embeddings = ego_embeddings
        embeddings_per_layer = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_per_layer.append(all_embeddings)

        mean_embeddings = torch.stack(embeddings_per_layer, dim=0).mean(dim=0)
        residual_weight = max(0.0, min(1.0, self.residual_weight))
        aggregated_embeddings = (
            (1.0 - residual_weight) * mean_embeddings
            + residual_weight * ego_embeddings
        )

        user_all_embeddings, item_all_embeddings = torch.split(
            aggregated_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings
