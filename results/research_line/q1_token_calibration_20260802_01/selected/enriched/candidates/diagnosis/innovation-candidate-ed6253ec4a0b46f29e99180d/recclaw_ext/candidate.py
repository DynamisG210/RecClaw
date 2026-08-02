import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from recbole.model.general_recommender.bpr import BPR


class FreshCandidateModel(BPR):
    SUPPORT_FRACTION = 0.5
    STABILITY_WEIGHT = 0.1
    REG_WEIGHT = 1e-4
    FULL_SORT_CHUNK_SIZE = 512
    EPS = 1e-12

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.mechanism_enabled = True
        self._inv_sqrt_dim = 1.0 / math.sqrt(float(self.embedding_size))
        self._build_history_buffers(dataset)

    def set_mechanism_enabled(self, enabled: bool):
        self.mechanism_enabled = bool(enabled)

    def _build_history_buffers(self, dataset):
        csr = None
        for form in ("csr", "coo"):
            try:
                mat = dataset.inter_matrix(form=form)
                csr = mat.tocsr() if form != "csr" else mat
                break
            except Exception:
                pass
        if csr is None:
            try:
                csr = dataset.inter_matrix().tocsr()
            except Exception:
                csr = sp.csr_matrix((self.n_users, self.n_items), dtype=np.float32)

        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int64)
        lengths = indptr[1:] - indptr[:-1]
        max_len = int(lengths.max()) if lengths.size > 0 else 0
        max_len = max(max_len, 1)

        hist_items = np.full((self.n_users, max_len), -1, dtype=np.int64)
        hist_lengths = lengths.astype(np.int64)
        for u in range(self.n_users):
            start, end = int(indptr[u]), int(indptr[u + 1])
            items = indices[start:end]
            if items.size > 0:
                hist_items[u, : items.size] = items

        self.register_buffer("history_items", torch.from_numpy(hist_items).long(), persistent=False)
        self.register_buffer("history_lengths", torch.from_numpy(hist_lengths).long(), persistent=False)

    def _get_user_history(self, user_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_ids = user_ids.long()
        hist_items = self.history_items[user_ids]
        hist_mask = hist_items.ge(0)
        return hist_items, hist_mask

    def _bpr_base_score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_e = self.user_embedding(user_ids.long())
        item_e = self.item_embedding(item_ids.long())
        return torch.sum(user_e * item_e, dim=-1)

    def _router_scores(self, user_ids: torch.Tensor, item_ids: torch.Tensor):
        user_ids = user_ids.long()
        item_ids = item_ids.long()
        user_e = self.user_embedding(user_ids)
        item_e = self.item_embedding(item_ids)
        base_score = torch.sum(user_e * item_e, dim=-1)

        if not self.mechanism_enabled:
            return base_score, base_score, base_score

        hist_items, hist_valid = self._get_user_history(user_ids)
        hist_e = self.item_embedding(hist_items.clamp_min(0))
        exclude_target = hist_items.ne(item_ids.unsqueeze(1))
        valid = hist_valid & exclude_target
        valid_counts = valid.long().sum(dim=1)
        has_history = valid_counts.gt(0)

        factual_context = torch.zeros_like(user_e)
        support_context = torch.zeros_like(user_e)

        if has_history.any():
            rows = torch.nonzero(has_history, as_tuple=False).squeeze(-1)
            for r in rows.tolist():
                row_valid = valid[r]
                row_hist = hist_e[r]
                row_item = item_e[r]
                logits = torch.sum(row_hist * row_item.unsqueeze(0), dim=-1) * self._inv_sqrt_dim
                logits = logits.masked_fill(~row_valid, -1e9)
                weights = torch.softmax(logits, dim=0)
                factual_context[r] = torch.sum(weights.unsqueeze(-1) * row_hist, dim=0)

                k = int(math.ceil(float(valid_counts[r].item()) * self.SUPPORT_FRACTION))
                k = max(1, min(k, int(valid_counts[r].item())))
                topk_idx = torch.topk(logits.masked_fill(~row_valid, -1e9), k=k, largest=True).indices
                support_mask = torch.zeros_like(row_valid)
                support_mask[topk_idx] = True
                support_logits = logits.masked_fill(~support_mask, -1e9)
                support_weights = torch.softmax(support_logits, dim=0)
                support_context[r] = torch.sum(support_weights.unsqueeze(-1) * row_hist, dim=0)

        factual_score = torch.sum((user_e + factual_context) * item_e, dim=-1)
        support_score = torch.sum((user_e + support_context) * item_e, dim=-1)
        return factual_score, support_score, base_score

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        if not self.mechanism_enabled:
            pos_score = self._bpr_base_score(user, pos_item)
            neg_score = self._bpr_base_score(user, neg_item)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            reg = (
                self.user_embedding(user).norm(2).pow(2)
                + self.item_embedding(pos_item).norm(2).pow(2)
                + self.item_embedding(neg_item).norm(2).pow(2)
            ) / float(user.shape[0])
            return loss + self.REG_WEIGHT * reg

        pos_factual, pos_support, pos_base = self._router_scores(user, pos_item)
        neg_factual, neg_support, neg_base = self._router_scores(user, neg_item)

        margin_factual = pos_factual - neg_factual
        bpr_loss = -F.logsigmoid(margin_factual).mean()
        stability_loss = F.mse_loss(pos_factual - neg_factual, pos_support - neg_support)
        reg = (
            self.user_embedding(user).norm(2).pow(2)
            + self.item_embedding(pos_item).norm(2).pow(2)
            + self.item_embedding(neg_item).norm(2).pow(2)
        ) / float(user.shape[0])
        return bpr_loss + self.REG_WEIGHT * reg + self.STABILITY_WEIGHT * stability_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if not self.mechanism_enabled:
            return self._bpr_base_score(user, item)
        factual_score, _, _ = self._router_scores(user, item)
        return factual_score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].view(-1)
        device = user.device
        if not self.mechanism_enabled:
            user_e = self.user_embedding(user.long())
            item_e = self.item_embedding.weight
            scores = torch.matmul(user_e, item_e.transpose(0, 1))
            return scores.view(-1)

        all_item_ids = torch.arange(self.n_items, device=device)
        chunks = []
        batch_size = user.shape[0]
        for start in range(0, self.n_items, self.FULL_SORT_CHUNK_SIZE):
            end = min(start + self.FULL_SORT_CHUNK_SIZE, self.n_items)
            item_chunk = all_item_ids[start:end]
            repeated_user = user.unsqueeze(1).expand(-1, item_chunk.shape[0]).reshape(-1)
            repeated_item = item_chunk.unsqueeze(0).expand(batch_size, -1).reshape(-1)
            scores, _, _ = self._router_scores(repeated_user, repeated_item)
            chunks.append(scores.view(batch_size, -1))
        return torch.cat(chunks, dim=1).reshape(-1)
