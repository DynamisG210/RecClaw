"""Local BPR variants with model-level negative sampling."""

from __future__ import annotations

import torch
from recbole.model.general_recommender.bpr import BPR

from ._samplers import MixedNegativeSampler, PopularityAwareNegativeSampler


def _config_float(config, key: str, default: float) -> float:
    try:
        value = config[key]
    except Exception:  # noqa: BLE001 - RecBole config is mapping-like, not a plain dict.
        value = default
    if value is None or str(value).strip() == "":
        value = default
    return float(value)


def _item_frequency(dataset, item_field: str, item_count: int) -> torch.Tensor:
    counts = torch.ones(item_count, dtype=torch.float32)
    try:
        item_ids = torch.as_tensor(dataset.inter_feat[item_field], dtype=torch.long)
        counts = torch.bincount(item_ids, minlength=item_count).float().clamp_min(1.0)
    except Exception:  # noqa: BLE001 - fall back to neutral sampling if dataset internals differ.
        pass
    if counts.numel() > 1:
        counts[0] = 0
    return counts


def _repair_invalid_negatives(neg_item: torch.Tensor, pos_item: torch.Tensor, item_count: int) -> torch.Tensor:
    if item_count <= 2:
        return neg_item.clamp_min(0)
    replacement = (pos_item % (item_count - 1)) + 1
    invalid = (neg_item <= 0) | (neg_item == pos_item)
    return torch.where(invalid, replacement.to(device=neg_item.device), neg_item)


class BPRHardNegative(BPR):
    """BPR that replaces RecBole's sampled negatives with a popularity-uniform mix."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.hard_negative_ratio = _config_float(config, "hard_negative_ratio", 0.5)
        popularity = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        self.negative_sampler = MixedNegativeSampler(
            self.n_items,
            popularity=popularity,
            hard_negative_ratio=self.hard_negative_ratio,
            avoid_zero=True,
        )

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = self.negative_sampler.sample(pos_item.shape, device=pos_item.device)
        neg_item = _repair_invalid_negatives(neg_item.long(), pos_item.long(), self.n_items)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.loss(pos_score, neg_score)


class BPRPopularityAwareNegative(BPR):
    """BPR that samples replacement negatives from a popularity-shaped distribution."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.popularity_alpha = _config_float(config, "popularity_alpha", 0.5)
        popularity = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        self.negative_sampler = PopularityAwareNegativeSampler(
            popularity,
            alpha=self.popularity_alpha,
            avoid_zero=True,
        )

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = self.negative_sampler.sample(pos_item.shape, device=pos_item.device)
        neg_item = _repair_invalid_negatives(neg_item.long(), pos_item.long(), self.n_items)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.loss(pos_score, neg_score)
