import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.general_recommender.bpr import BPR


class FreshCandidateModel(BPR):
    _STABILITY_WEIGHT = 0.1
    _EPS = 1e-12

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = int(config["embedding_size"])
        self._mechanism_enabled = True
        self.support_gate = nn.Linear(self.embedding_size * 2, 1, bias=False)
        self._build_user_histories(dataset)

    def set_mechanism_enabled(self, enabled: bool):
        self._mechanism_enabled = bool(enabled)
        return self

    def _user_embedding_layer(self):
        if hasattr(self, "user_embedding"):
            return self.user_embedding
        if hasattr(self, "embedding_user"):
            return self.embedding_user
        raise AttributeError("User embedding layer not found")

    def _item_embedding_layer(self):
        if hasattr(self, "item_embedding"):
            return self.item_embedding
        if hasattr(self, "embedding_item"):
            return self.embedding_item
        raise AttributeError("Item embedding layer not found")

    def _build_user_histories(self, dataset):
        uid_field = self.USER_ID
        iid_field = self.ITEM_ID
        user_raw = dataset.inter_feat[uid_field]
        item_raw = dataset.inter_feat[iid_field]
        user_np = np.asarray(user_raw.detach().cpu().numpy() if torch.is_tensor(user_raw) else user_raw, dtype=np.int64)
        item_np = np.asarray(item_raw.detach().cpu().numpy() if torch.is_tensor(item_raw) else item_raw, dtype=np.int64)
        histories = [[] for _ in range(self.n_users)]
        for u, i in zip(user_np.tolist(), item_np.tolist()):
            if 0 <= u < self.n_users and 0 <= i < self.n_items:
                histories[u].append(int(i))
        self._user_histories = [
            torch.tensor(items, dtype=torch.long) if items else torch.empty(0, dtype=torch.long)
            for items in histories
        ]

    def _parent_scores(self, users, items):
        user_e = self._user_embedding_layer()(users)
        item_e = self._item_embedding_layer()(items)
        return torch.sum(user_e * item_e, dim=-1)

    def _parent_calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_scores = self._parent_scores(users, pos_items)
        neg_scores = self._parent_scores(users, neg_items)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

    def _get_user_history(self, user_id: int, exclude_item: Optional[int] = None):
        history = self._user_histories[user_id]
        if history.numel() == 0 or exclude_item is None:
            return history
        mask = history.ne(int(exclude_item))
        return history[mask]

    def _history_summary(self, users, exclude_items: Optional[torch.Tensor] = None):
        device = self._user_embedding_layer().weight.device
        user_e = self._user_embedding_layer()(users.to(device))
        summaries = []
        weights_pack = []
        hist_pack = []
        for idx in range(users.size(0)):
            user_id = int(users[idx].item())
            exclude = None if exclude_items is None else int(exclude_items[idx].item())
            history = self._get_user_history(user_id, exclude)
            if history.numel() == 0:
                summaries.append(torch.zeros(self.embedding_size, device=device))
                weights_pack.append(None)
                hist_pack.append(None)
                continue
            hist = history.to(device)
            h_e = self._item_embedding_layer()(hist)
            u_e = user_e[idx].unsqueeze(0).expand_as(h_e)
            logits = self.support_gate(torch.cat([u_e, h_e], dim=-1)).squeeze(-1)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            weights = torch.softmax(logits, dim=0)
            summary = torch.sum(weights.unsqueeze(-1) * h_e, dim=0)
            summaries.append(summary)
            weights_pack.append(weights)
            hist_pack.append(h_e)
        return torch.stack(summaries, dim=0), weights_pack, hist_pack

    def _stability_penalty(self, users, user_e, pos_e, neg_e, exclude_items: Optional[torch.Tensor] = None):
        device = user_e.device
        penalties = []
        for idx in range(users.size(0)):
            user_id = int(users[idx].item())
            exclude = None if exclude_items is None else int(exclude_items[idx].item())
            history = self._get_user_history(user_id, exclude)
            if history.numel() <= 1:
                continue
            hist = history.to(device)
            h_e = self._item_embedding_layer()(hist)
            u_e = user_e[idx]
            p_e = pos_e[idx]
            n_e = neg_e[idx]
            u_expand = u_e.unsqueeze(0).expand_as(h_e)
            logits = self.support_gate(torch.cat([u_expand, h_e], dim=-1)).squeeze(-1)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            weights = torch.softmax(logits, dim=0)
            summary = torch.sum(weights.unsqueeze(-1) * h_e, dim=0)
            margin = torch.sum((u_e + summary) * (p_e - n_e))
            denom = (1.0 - weights).clamp_min(self._EPS)
            loo_summary = (summary.unsqueeze(0) - weights.unsqueeze(-1) * h_e) / denom.unsqueeze(-1)
            loo_margin = torch.sum((u_e.unsqueeze(0) + loo_summary) * (p_e - n_e).unsqueeze(0), dim=-1)
            penalties.append(torch.mean((loo_margin - margin) ** 2))
        if not penalties:
            return torch.zeros((), device=device)
        return torch.mean(torch.stack(penalties))

    def calculate_loss(self, interaction):
        if not self._mechanism_enabled:
            return self._parent_calculate_loss(interaction)

        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        device = self._user_embedding_layer().weight.device
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        user_e = self._user_embedding_layer()(users)
        pos_e = self._item_embedding_layer()(pos_items)
        neg_e = self._item_embedding_layer()(neg_items)

        summaries, _, _ = self._history_summary(users, exclude_items=pos_items)
        user_repr = user_e + summaries
        pos_scores = torch.sum(user_repr * pos_e, dim=-1)
        neg_scores = torch.sum(user_repr * neg_e, dim=-1)
        base_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        stability = self._stability_penalty(users, user_e, pos_e, neg_e, exclude_items=pos_items)
        loss = base_loss + self._STABILITY_WEIGHT * stability
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

    def predict(self, interaction):
        if not self._mechanism_enabled:
            users = interaction[self.USER_ID]
            items = interaction[self.ITEM_ID]
            scores = self._parent_scores(users, items)
            return torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=1e6)

        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        device = self._user_embedding_layer().weight.device
        users = users.to(device)
        items = items.to(device)
        user_e = self._user_embedding_layer()(users)
        item_e = self._item_embedding_layer()(items)
        summaries, _, _ = self._history_summary(users, exclude_items=items)
        scores = torch.sum((user_e + summaries) * item_e, dim=-1)
        return torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=1e6)

    def full_sort_predict(self, interaction):
        if not self._mechanism_enabled:
            users = interaction[self.USER_ID]
            device = self._user_embedding_layer().weight.device
            users = users.to(device)
            user_e = self._user_embedding_layer()(users)
            all_item_e = self._item_embedding_layer().weight
            scores = torch.matmul(user_e, all_item_e.t())
            return torch.nan_to_num(scores.view(-1), nan=0.0, posinf=1e6, neginf=1e6)

        users = interaction[self.USER_ID]
        device = self._user_embedding_layer().weight.device
        users = users.to(device)
        user_e = self._user_embedding_layer()(users)
        summaries, _, _ = self._history_summary(users, exclude_items=None)
        all_item_e = self._item_embedding_layer().weight
        scores = torch.matmul(user_e + summaries, all_item_e.t())
        return torch.nan_to_num(scores.view(-1), nan=0.0, posinf=1e6, neginf=1e6)
