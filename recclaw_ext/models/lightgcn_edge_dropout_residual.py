"""LightGCN residual variant with training-time sparse edge dropout."""

from __future__ import annotations

import torch

from .lightgcn_residual import LightGCNResidualMix


def _config_float(config, key: str, default: float) -> float:
    try:
        value = config[key]
    except Exception:  # noqa: BLE001 - RecBole config is mapping-like, not a plain dict.
        value = default
    if value is None or str(value).strip() == "":
        value = default
    return float(value)


class LightGCNEdgeDropoutResidualMix(LightGCNResidualMix):
    """Residual LightGCN with sparse adjacency dropout during training only."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.edge_dropout = max(0.0, min(0.95, _config_float(config, "edge_dropout", 0.1)))

    def _propagation_matrix(self) -> torch.Tensor:
        if not self.training or self.edge_dropout <= 0:
            return self.norm_adj_matrix
        adj = self.norm_adj_matrix.coalesce()
        values = adj.values()
        keep_prob = 1.0 - self.edge_dropout
        mask = torch.rand(values.shape, device=values.device) < keep_prob
        if not bool(mask.any()):
            return adj
        dropped_values = values[mask] / keep_prob
        return torch.sparse_coo_tensor(
            adj.indices()[:, mask],
            dropped_values,
            adj.shape,
            device=adj.device,
        ).coalesce()

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
        aggregated_embeddings = (1.0 - residual_weight) * mean_embeddings + residual_weight * ego_embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            aggregated_embeddings,
            [self.n_users, self.n_items],
        )
        return user_all_embeddings, item_all_embeddings
