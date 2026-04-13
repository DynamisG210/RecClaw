"""Local LightGCN variants for method-level RecClaw experiments."""

from __future__ import annotations

import torch
from recbole.model.general_recommender.lightgcn import LightGCN


class LightGCNLW(LightGCN):
    """LightGCN with learnable layer-wise aggregation.

    Assumptions:
    - Layer 0 (ego embeddings before message passing) participates in the final mix.
    - This scaffold only changes aggregation; loss, sampling, and evaluation stay aligned
      with RecBole's LightGCN implementation.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.layer_logits = torch.nn.Parameter(torch.zeros(self.n_layers + 1))

    def get_layer_weights(self) -> torch.Tensor:
        """Return normalized layer weights for inspection or downstream logging."""
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_per_layer = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_per_layer.append(all_embeddings)

        layer_weights = self.get_layer_weights()
        stacked_embeddings = torch.stack(embeddings_per_layer, dim=0)
        aggregated_embeddings = torch.sum(
            layer_weights.view(-1, 1, 1) * stacked_embeddings, dim=0
        )

        user_all_embeddings, item_all_embeddings = torch.split(
            aggregated_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings
