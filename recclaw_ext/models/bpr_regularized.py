"""Local BPR variants that wire existing RecClaw loss helpers."""

from __future__ import annotations

import torch
from recbole.model.general_recommender.bpr import BPR

from ._losses import (
    BPRLongTailReweightLoss,
    BPRNormConstrainedLoss,
    BPRPopularityRegularizedLoss,
)


def _config_float(config, key: str, default: float) -> float:
    try:
        value = config[key]
    except Exception:  # noqa: BLE001 - RecBole config behaves like a mapping but is not always plain dict.
        value = default
    if value is None or str(value).strip() == "":
        value = default
    return float(value)


def _item_frequency(dataset, item_field: str, item_count: int) -> torch.Tensor:
    counts = torch.ones(item_count, dtype=torch.float32)
    try:
        item_ids = dataset.inter_feat[item_field]
        item_ids = torch.as_tensor(item_ids, dtype=torch.long)
        counts = torch.bincount(item_ids, minlength=item_count).float()
        counts = counts.clamp_min(1.0)
    except Exception:  # noqa: BLE001 - fall back to neutral weights if dataset internals differ.
        pass
    return counts


class BPRLongTailReweight(BPR):
    """BPR with positive-item long-tail reweighting."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.tail_weight_alpha = _config_float(config, "tail_weight_alpha", 0.5)
        self.loss = BPRLongTailReweightLoss(tail_weight_alpha=self.tail_weight_alpha)
        item_frequency = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        tail_weight = item_frequency.rsqrt()
        tail_weight = tail_weight / tail_weight.mean().clamp_min(1e-12)
        self.register_buffer("item_tail_weight", tail_weight)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.loss(pos_score, neg_score, self.item_tail_weight[pos_item])


class BPRPopularityRegularized(BPR):
    """BPR with a popularity-weighted penalty on negative scores."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.lambda_pop = _config_float(config, "lambda_pop", 1e-4)
        self.loss = BPRPopularityRegularizedLoss(lambda_pop=self.lambda_pop)
        item_frequency = _item_frequency(dataset, self.ITEM_ID, self.n_items)
        popularity = torch.log1p(item_frequency)
        popularity = popularity / popularity.mean().clamp_min(1e-12)
        self.register_buffer("item_popularity", popularity)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.loss(pos_score, neg_score, self.item_popularity[neg_item])


class BPRNormConstrained(BPR):
    """BPR with a soft embedding norm penalty."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.lambda_norm = _config_float(config, "lambda_norm", 1e-4)
        self.max_norm = _config_float(config, "max_norm", 1.0)
        self.loss = BPRNormConstrainedLoss(lambda_norm=self.lambda_norm, max_norm=self.max_norm)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.loss(pos_score, neg_score, user_e, pos_e, neg_e)
