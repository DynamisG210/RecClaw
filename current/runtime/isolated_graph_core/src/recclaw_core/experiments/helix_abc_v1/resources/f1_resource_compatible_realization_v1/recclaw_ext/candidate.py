"""Resource-compatible equivalent realization of sealed F1 candidate.

Only common-subexpression reuse and evaluation chunking differ from the sealed
source.  Model parameters, support selection, scores, loss, detach boundary,
diagnostics, and candidate-universe ordering are unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.general_recommender.bpr import BPR
from recbole.utils import InputType


class FreshCandidateModel(BPR):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.support_query = nn.Linear(
            self.embedding_size, self.embedding_size, bias=False
        )
        self.support_key = nn.Linear(
            self.embedding_size, self.embedding_size, bias=False
        )
        self.stability_weight = 0.2
        self.support_topk = 1
        self.score_chunk_size = 256
        self.full_sort_user_chunk_size = 64
        self.eps = 1e-12
        self.latest_diagnostics = {}

        self._init_history_buffers(dataset)
        self.apply(self._init_extra_parameters)

    def _init_extra_parameters(self, module):
        if module in (self.support_query, self.support_key):
            nn.init.xavier_uniform_(module.weight)

    def _init_history_buffers(self, dataset):
        user_tensor = dataset.inter_feat[self.USER_ID]
        item_tensor = dataset.inter_feat[self.ITEM_ID]

        if not isinstance(user_tensor, torch.Tensor):
            user_tensor = torch.as_tensor(user_tensor, dtype=torch.long)
        else:
            user_tensor = user_tensor.long()

        if not isinstance(item_tensor, torch.Tensor):
            item_tensor = torch.as_tensor(item_tensor, dtype=torch.long)
        else:
            item_tensor = item_tensor.long()

        histories = [[] for _ in range(self.n_users)]
        for user_id, item_id in zip(user_tensor.tolist(), item_tensor.tolist()):
            if 0 <= user_id < self.n_users and 0 <= item_id < self.n_items:
                histories[user_id].append(item_id)

        max_len = 1
        for items in histories:
            if len(items) > max_len:
                max_len = len(items)

        history_items = torch.full((self.n_users, max_len), -1, dtype=torch.long)
        history_mask = torch.zeros((self.n_users, max_len), dtype=torch.bool)
        history_len = torch.zeros(self.n_users, dtype=torch.long)

        for user_id, items in enumerate(histories):
            if not items:
                continue
            item_row = torch.tensor(items, dtype=torch.long)
            row_len = item_row.size(0)
            history_items[user_id, :row_len] = item_row
            history_mask[user_id, :row_len] = True
            history_len[user_id] = row_len

        self.register_buffer("history_items", history_items)
        self.register_buffer("history_mask", history_mask)
        self.register_buffer("history_len", history_len)
        self.register_buffer(
            "all_item_ids", torch.arange(self.n_items, dtype=torch.long)
        )

    def _gather_history(self, user_ids):
        history_items = self.history_items[user_ids]
        history_mask = self.history_mask[user_ids]
        if history_items.shape[0] > 0:
            batch_max_len = max(1, int(self.history_len[user_ids].max().item()))
            history_items = history_items[:, :batch_max_len]
            history_mask = history_mask[:, :batch_max_len]
        return history_items, history_mask

    def _history_embeddings(self, history_items):
        safe_history = history_items.clamp_min(0)
        return self.item_embedding(safe_history)

    def _shared_history(self, user_ids):
        history_items, base_mask = self._gather_history(user_ids)
        history_emb = self._history_embeddings(history_items)
        support_key = self.support_key(history_emb)
        return history_items, base_mask, history_emb, support_key

    def _support_attention(
        self, history_emb, support_key, target_emb, valid_mask
    ):
        query = self.support_query(target_emb).unsqueeze(1)
        logits = (query * support_key).sum(dim=-1) / math.sqrt(
            float(self.embedding_size)
        )
        logits = logits.masked_fill(~valid_mask, -1e9)

        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid_mask.float()
        weight_norm = weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weights = weights / weight_norm
        context = torch.sum(weights.unsqueeze(-1) * history_emb, dim=1)
        return context, weights

    def _pair_score_from_shared(
        self,
        user_emb,
        target_ids,
        history_items,
        base_mask,
        history_emb,
        support_key,
        removal_mask=None,
    ):
        valid_mask = base_mask & (history_items != target_ids.unsqueeze(1))
        if removal_mask is not None:
            valid_mask = valid_mask & (~removal_mask)
        target_emb = self.item_embedding(target_ids)
        context, weights = self._support_attention(
            history_emb, support_key, target_emb, valid_mask
        )
        score = torch.sum((user_emb + context) * target_emb, dim=-1)
        score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        return score, weights, valid_mask

    def _select_supports(self, weights, valid_mask):
        top_k = min(self.support_topk, weights.size(1))
        if top_k <= 0:
            return torch.zeros_like(valid_mask)

        safe_weights = weights.masked_fill(~valid_mask, -1.0)
        top_values, top_indices = torch.topk(safe_weights, k=top_k, dim=1)
        selected = torch.zeros_like(valid_mask)
        selected.scatter_(1, top_indices, top_values > 0)
        selected = selected & valid_mask
        return selected

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        user_emb = self.user_embedding(user)
        shared = self._shared_history(user)

        pos_score, pos_weights, pos_valid_mask = self._pair_score_from_shared(
            user_emb, pos_item, *shared
        )
        neg_score, _, _ = self._pair_score_from_shared(
            user_emb, neg_item, *shared
        )
        factual_margin = pos_score - neg_score

        support_mask = self._select_supports(pos_weights, pos_valid_mask)
        pos_score_cf, _, _ = self._pair_score_from_shared(
            user_emb, pos_item, *shared, support_mask
        )
        neg_score_cf, _, _ = self._pair_score_from_shared(
            user_emb, neg_item, *shared, support_mask
        )
        counterfactual_margin = pos_score_cf - neg_score_cf

        ranking_loss = self.loss(pos_score, neg_score)
        stability_loss = F.smooth_l1_loss(
            counterfactual_margin, factual_margin.detach()
        )
        total_loss = ranking_loss + self.stability_weight * stability_loss
        total_loss = torch.nan_to_num(
            total_loss, nan=0.0, posinf=1e6, neginf=1e6
        )

        with torch.no_grad():
            support_count = support_mask.float().sum(dim=1)
            score_drop = pos_score - pos_score_cf
            self.latest_diagnostics = {
                "support_count_mean": support_count.mean().item(),
                "support_weight_max_mean": pos_weights.max(dim=1)[0].mean().item(),
                "positive_score_drop_mean": score_drop.mean().item(),
                "margin_drop_mean": (
                    factual_margin - counterfactual_margin
                ).mean().item(),
            }

        return total_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_emb = self.user_embedding(user)
        score, _, _ = self._pair_score_from_shared(
            user_emb, item, *self._shared_history(user)
        )
        return score

    def _full_sort_scores_chunk(
        self,
        user_emb,
        history_emb,
        support_key,
        history_items,
        valid_mask,
        item_ids,
    ):
        item_emb = self.item_embedding(item_ids)
        query = self.support_query(item_emb)

        logits = torch.einsum("bld,nd->bln", support_key, query) / math.sqrt(
            float(self.embedding_size)
        )
        target_mask = history_items.unsqueeze(-1) != item_ids.view(1, 1, -1)
        combined_mask = valid_mask.unsqueeze(-1) & target_mask
        logits = logits.masked_fill(~combined_mask, -1e9)

        weights = torch.softmax(logits, dim=1)
        weights = weights * combined_mask.float()
        weight_norm = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        weights = weights / weight_norm

        context = torch.einsum("bln,bld->bnd", weights, history_emb)
        scores = torch.sum(
            (user_emb.unsqueeze(1) + context) * item_emb.unsqueeze(0), dim=-1
        )
        return torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        user_score_chunks = []
        for user_start in range(0, users.shape[0], self.full_sort_user_chunk_size):
            user_end = min(
                user_start + self.full_sort_user_chunk_size, users.shape[0]
            )
            user = users[user_start:user_end]
            user_emb = self.user_embedding(user)
            history_items, valid_mask, history_emb, support_key = (
                self._shared_history(user)
            )
            item_score_chunks = []
            for item_start in range(0, self.n_items, self.score_chunk_size):
                item_end = min(item_start + self.score_chunk_size, self.n_items)
                item_ids = self.all_item_ids[item_start:item_end]
                item_score_chunks.append(
                    self._full_sort_scores_chunk(
                        user_emb,
                        history_emb,
                        support_key,
                        history_items,
                        valid_mask,
                        item_ids,
                    )
                )
            user_score_chunks.append(torch.cat(item_score_chunks, dim=1))
        return torch.cat(user_score_chunks, dim=0).reshape(-1)
