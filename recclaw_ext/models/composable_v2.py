"""Package-owned bounded compositional BPR and LightGCN handlers."""

from __future__ import annotations

from typing import Iterable

import torch
from torch.nn import functional as F
from recbole.model.general_recommender.bpr import BPR
from recbole.model.general_recommender.lightgcn import LightGCN

from ._samplers import (
    DebiasedNegativeSampler,
    MixedNegativeSampler,
    PopularityAwareNegativeSampler,
)
from ._utils import config_float
from .bpr_sampling import _item_frequency, _repair_invalid_negatives


def _operators(config) -> frozenset[str]:
    try:
        value = config["composition_operators"]
    except Exception:  # noqa: BLE001 - RecBole Config is mapping-like.
        value = ()
    if value is None:
        return frozenset()
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return frozenset(str(item) for item in value)


def _pair_scores(model, interaction, *, negative_sampler=None):
    user = interaction[model.USER_ID]
    pos_item = interaction[model.ITEM_ID]
    if negative_sampler is None:
        neg_item = interaction[model.NEG_ITEM_ID]
    else:
        neg_item = negative_sampler.sample(
            pos_item.shape, device=pos_item.device
        )
        neg_item = _repair_invalid_negatives(
            neg_item.long(), pos_item.long(), model.n_items
        )
    return user, pos_item, neg_item


class BPRComposableV2(BPR):
    """BPR with at most two compatible typed operators."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.composition_operators = _operators(config)
        popularity = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        self.negative_sampler = None
        if "BPR_MIXED_NEGATIVE" in self.composition_operators:
            self.negative_sampler = MixedNegativeSampler(
                self.n_items,
                popularity=popularity,
                hard_negative_ratio=config_float(
                    config, "hard_negative_ratio", 0.5
                ),
                avoid_zero=True,
            )
        elif "BPR_POPULARITY_NEGATIVE" in self.composition_operators:
            self.negative_sampler = PopularityAwareNegativeSampler(
                popularity,
                alpha=config_float(config, "popularity_alpha", 0.5),
                avoid_zero=True,
            )
        self.margin = config_float(config, "margin", 0.2)
        self.rank_weight_alpha = config_float(
            config, "rank_weight_alpha", 0.2
        )
        self.tail_weight_alpha = config_float(
            config, "tail_weight_alpha", 0.2
        )
        self.lambda_norm = config_float(config, "lambda_norm", 1e-4)
        self.max_norm = config_float(config, "max_norm", 1.0)
        self.lambda_pop = config_float(config, "lambda_pop", 1e-4)
        tail_weight = popularity.rsqrt()
        self.register_buffer(
            "item_tail_weight",
            tail_weight / tail_weight.mean().clamp_min(1e-12),
        )
        item_popularity = torch.log1p(popularity)
        self.register_buffer(
            "item_popularity",
            item_popularity
            / item_popularity.mean().clamp_min(1e-12),
        )

    def calculate_loss(self, interaction):
        user, pos_item, neg_item = _pair_scores(
            self, interaction, negative_sampler=self.negative_sampler
        )
        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        difference = pos_score - neg_score
        if "BPR_MARGIN" in self.composition_operators:
            difference = difference - self.margin
        pair_loss = -F.logsigmoid(difference)
        if "BPR_RANK_AWARE" in self.composition_operators:
            weight = (1.0 + torch.sigmoid(-difference)).detach()
            weight = weight / weight.mean().clamp_min(1e-12)
            pair_loss = pair_loss * weight.pow(self.rank_weight_alpha)
        if "BPR_TAIL_REWEIGHT" in self.composition_operators:
            weight = self.item_tail_weight[pos_item].to(pair_loss)
            weight = weight / weight.mean().clamp_min(1e-12)
            pair_loss = pair_loss * weight.pow(self.tail_weight_alpha)
        loss = pair_loss.mean()
        if "BPR_NORM_CONSTRAINT" in self.composition_operators:
            excess = torch.cat(
                (user_e.norm(dim=1), pos_e.norm(dim=1), neg_e.norm(dim=1))
            )
            loss = loss + self.lambda_norm * F.relu(
                excess - self.max_norm
            ).square().mean()
        if "BPR_POPULARITY_REG" in self.composition_operators:
            loss = loss + self.lambda_pop * (
                neg_score.square() * self.item_popularity[neg_item]
            ).mean()
        return loss


class LightGCNComposableV2(LightGCN):
    """LightGCN with typed propagation, aggregation and loss operators."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.composition_operators = _operators(config)
        if "LGCN_SHALLOW" in self.composition_operators:
            self.n_layers = 1
        self.edge_dropout = config_float(config, "edge_dropout", 0.1)
        self.residual_weight = config_float(
            config, "residual_weight", 0.2
        )
        self.dual_path_weight = config_float(
            config, "dual_path_weight", 0.35
        )
        self.rank_weight_alpha = config_float(
            config, "rank_weight_alpha", 0.5
        )
        self.lambda_align = config_float(config, "lambda_align", 1e-3)
        self.lambda_norm = config_float(config, "lambda_norm", 1e-4)
        self.max_norm = config_float(config, "max_norm", 1.0)
        if "LGCN_LAYER_WEIGHTED" in self.composition_operators:
            self.layer_logits = torch.nn.Parameter(
                torch.zeros(self.n_layers + 1)
            )
        popularity = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        self.negative_sampler = (
            DebiasedNegativeSampler(
                popularity,
                alpha=config_float(config, "debias_alpha", 0.5),
                avoid_zero=True,
            )
            if "LGCN_DEBIASED_NEGATIVE"
            in self.composition_operators
            else None
        )
        self._composition_layers: tuple[torch.Tensor, ...] = ()

    def _adjacency(self) -> torch.Tensor:
        adjacency = self.norm_adj_matrix
        if (
            "LGCN_EDGE_DROPOUT" not in self.composition_operators
            or not self.training
            or self.edge_dropout <= 0
        ):
            return adjacency
        coalesced = adjacency.coalesce()
        values = F.dropout(
            coalesced.values(),
            p=self.edge_dropout,
            training=True,
        )
        return torch.sparse_coo_tensor(
            coalesced.indices(),
            values,
            coalesced.size(),
            device=values.device,
        ).coalesce()

    def forward(self):
        ego = self.get_ego_embeddings()
        current = ego
        layers = [current]
        adjacency = self._adjacency()
        for _ in range(self.n_layers):
            current = torch.sparse.mm(adjacency, current)
            layers.append(current)
        stacked = torch.stack(layers, dim=0)
        if "LGCN_LAYER_WEIGHTED" in self.composition_operators:
            weights = torch.softmax(self.layer_logits, dim=0)
            aggregated = torch.sum(
                weights.view(-1, 1, 1) * stacked, dim=0
            )
        else:
            aggregated = stacked.mean(dim=0)
        if "LGCN_RESIDUAL" in self.composition_operators:
            weight = max(0.0, min(1.0, self.residual_weight))
            aggregated = (1.0 - weight) * aggregated + weight * ego
        if "LGCN_DUAL_PATH" in self.composition_operators:
            weight = max(0.0, min(1.0, self.dual_path_weight))
            aggregated = (
                (1.0 - weight) * aggregated + weight * layers[-1]
            )
        self._composition_layers = tuple(layers)
        return torch.split(aggregated, [self.n_users, self.n_items])

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        user, pos_item, neg_item = _pair_scores(
            self, interaction, negative_sampler=self.negative_sampler
        )
        user_all, item_all = self.forward()
        user_e = user_all[user]
        pos_e = item_all[pos_item]
        neg_e = item_all[neg_item]
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        pair_loss = -F.logsigmoid(pos_score - neg_score)
        if "LGCN_RANK_AWARE" in self.composition_operators:
            weight = (
                1.0 + torch.sigmoid(neg_score - pos_score)
            ).detach()
            weight = weight / weight.mean().clamp_min(1e-12)
            pair_loss = pair_loss * weight.pow(self.rank_weight_alpha)
        ego_user = self.user_embedding(user)
        ego_pos = self.item_embedding(pos_item)
        ego_neg = self.item_embedding(neg_item)
        loss = pair_loss.mean() + self.reg_weight * self.reg_loss(
            ego_user,
            ego_pos,
            ego_neg,
            require_pow=self.require_pow,
        ).mean()
        if "LGCN_AUX_ALIGNMENT" in self.composition_operators:
            node_ids = torch.cat(
                (user, pos_item + self.n_users, neg_item + self.n_users)
            )
            adjacent = zip(
                self._composition_layers,
                self._composition_layers[1:],
            )
            alignment = [
                1.0
                - F.cosine_similarity(
                    left.index_select(0, node_ids),
                    right.index_select(0, node_ids),
                    dim=1,
                ).mean()
                for left, right in adjacent
            ]
            if alignment:
                loss = loss + self.lambda_align * torch.stack(
                    alignment
                ).mean()
        if "LGCN_NORM_CONSTRAINT" in self.composition_operators:
            excess = torch.cat(
                (user_e.norm(dim=1), pos_e.norm(dim=1), neg_e.norm(dim=1))
            )
            loss = loss + self.lambda_norm * F.relu(
                excess - self.max_norm
            ).square().mean()
        return loss


def composition_operator_ids(config) -> tuple[str, ...]:
    """Stable inspection hook used by runtime canaries."""

    return tuple(sorted(_operators(config)))


__all__ = [
    "BPRComposableV2",
    "LightGCNComposableV2",
    "composition_operator_ids",
]
