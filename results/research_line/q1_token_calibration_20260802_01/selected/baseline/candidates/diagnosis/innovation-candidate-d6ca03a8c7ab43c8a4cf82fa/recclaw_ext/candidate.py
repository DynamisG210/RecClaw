import numpy as np
import torch
import torch.nn.functional as F
from recbole.model.general_recommender.bpr import BPR


class FreshCandidateModel(BPR):
    _HISTORY_WINDOW = 5

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._enabled = True
        history = self._build_recent_history(dataset)
        self.register_buffer('_user_recent_items', history)

    def set_mechanism_enabled(self, enabled: bool):
        self._enabled = bool(enabled)

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _build_recent_history(self, dataset):
        history = np.full((self.n_users, self._HISTORY_WINDOW), -1, dtype=np.int64)
        try:
            inter_feat = dataset.inter_feat
            users = self._to_numpy(inter_feat[self.USER_ID]).astype(np.int64, copy=False)
            items = self._to_numpy(inter_feat[self.ITEM_ID]).astype(np.int64, copy=False)
        except Exception:
            return torch.as_tensor(history, dtype=torch.long)

        order = np.arange(users.shape[0])
        time_field = getattr(self, 'TIME_FIELD', None)
        if time_field is not None:
            try:
                times = self._to_numpy(inter_feat[time_field])
                order = np.argsort(times, kind='stable')
            except Exception:
                pass

        per_user = [[] for _ in range(self.n_users)]
        for idx in order:
            u = int(users[idx])
            i = int(items[idx])
            if 0 <= u < self.n_users and 0 <= i < self.n_items:
                per_user[u].append(i)

        for u, seq in enumerate(per_user):
            tail = seq[-self._HISTORY_WINDOW:]
            if tail:
                history[u, -len(tail):] = np.asarray(tail, dtype=np.int64)

        return torch.as_tensor(history, dtype=torch.long)

    def _user_memory(self, user):
        hist = self._user_recent_items[user]
        valid = hist >= 0
        safe_hist = hist.clamp_min(0)
        hist_emb = self.item_embedding(safe_hist)
        hist_emb = hist_emb * valid.unsqueeze(-1).float()
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        memory = hist_emb.sum(dim=1) / count
        has_history = valid.any(dim=1, keepdim=True)
        return memory, has_history

    def _enabled_score(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        memory, has_history = self._user_memory(user)

        direct = (user_emb * item_emb).sum(dim=-1)
        residual = (memory * item_emb).sum(dim=-1)
        gate = torch.sigmoid((user_emb * memory).sum(dim=-1, keepdim=True))
        gate = torch.where(has_history, gate, torch.ones_like(gate))
        score = gate.squeeze(-1) * direct + (1.0 - gate.squeeze(-1)) * residual
        return torch.nan_to_num(score)

    def calculate_loss(self, interaction):
        if not self._enabled:
            return super().calculate_loss(interaction)
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        pos_score = self._enabled_score(user, pos_item)
        neg_score = self._enabled_score(user, neg_item)
        return -F.logsigmoid(pos_score - neg_score).mean()

    def predict(self, interaction):
        if not self._enabled:
            return super().predict(interaction)
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self._enabled_score(user, item)

    def full_sort_predict(self, interaction):
        if not self._enabled:
            return super().full_sort_predict(interaction)
        user = interaction[self.USER_ID]
        user_emb = self.user_embedding(user)
        memory, has_history = self._user_memory(user)
        item_emb = self.item_embedding.weight

        direct = torch.matmul(user_emb, item_emb.t())
        residual = torch.matmul(memory, item_emb.t())
        gate = torch.sigmoid((user_emb * memory).sum(dim=-1, keepdim=True))
        gate = torch.where(has_history, gate, torch.ones_like(gate))
        scores = gate * direct + (1.0 - gate) * residual
        scores = torch.nan_to_num(scores)
        return scores.view(-1)
