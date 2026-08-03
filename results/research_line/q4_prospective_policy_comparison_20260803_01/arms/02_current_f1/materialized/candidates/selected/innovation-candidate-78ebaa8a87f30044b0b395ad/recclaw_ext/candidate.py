import numpy as np
import torch
from torch import nn

from recbole.model.general_recommender.bpr import BPR


class FreshCandidateModel(BPR):
    RECENT_HISTORY_LEN = 5

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.embedding_size = int(config["embedding_size"])
        self.mechanism_enabled = True

        self.residual_memory = nn.Embedding(self.n_users, self.embedding_size)
        self.position_embedding = nn.Embedding(self.RECENT_HISTORY_LEN, self.embedding_size)
        self.intent_attention = nn.Linear(self.embedding_size, 1, bias=False)
        self.residual_gate = nn.Linear(self.embedding_size * 2, 1, bias=True)

        history_items, history_lengths = self._build_train_histories(dataset)
        self.register_buffer("user_history_items", history_items)
        self.register_buffer("user_history_lengths", history_lengths)

        self._reset_new_parameters()

    def _reset_new_parameters(self):
        nn.init.xavier_normal_(self.residual_memory.weight)
        nn.init.xavier_normal_(self.position_embedding.weight)
        nn.init.xavier_normal_(self.intent_attention.weight)
        nn.init.xavier_uniform_(self.residual_gate.weight)
        nn.init.zeros_(self.residual_gate.bias)
        with torch.no_grad():
            if self.item_embedding.weight.size(0) > 0:
                self.position_embedding.weight[0].mul_(1.0)
            if self.residual_memory.weight.size(0) > 0:
                self.residual_memory.weight[0].mul_(1.0)

    def set_mechanism_enabled(self, enabled: bool):
        self.mechanism_enabled = bool(enabled)

    def _build_train_histories(self, dataset):
        inter_feat = dataset.inter_feat
        user_np = inter_feat[self.USER_ID].cpu().numpy()
        item_np = inter_feat[self.ITEM_ID].cpu().numpy()

        time_field = getattr(dataset, "time_field", None)
        if time_field is not None and time_field in inter_feat:
            time_np = inter_feat[time_field].cpu().numpy()
            order = np.lexsort((np.arange(len(user_np)), time_np, user_np))
        else:
            order = np.argsort(user_np, kind="stable")

        history = np.zeros((self.n_users, self.RECENT_HISTORY_LEN), dtype=np.int64)
        lengths = np.zeros(self.n_users, dtype=np.int64)

        user_lists = [[] for _ in range(self.n_users)]
        for idx in order:
            u = int(user_np[idx])
            i = int(item_np[idx])
            if 0 <= u < self.n_users:
                user_lists[u].append(i)

        for u, items in enumerate(user_lists):
            if not items:
                continue
            tail = items[-self.RECENT_HISTORY_LEN :]
            lengths[u] = len(tail)
            history[u, -len(tail) :] = np.asarray(tail, dtype=np.int64)

        return torch.from_numpy(history), torch.from_numpy(lengths)

    def _encode_recent_intent(self, user):
        history_items = self.user_history_items[user]
        mask = history_items.gt(0)

        item_emb = self.item_embedding(history_items)
        pos_ids = torch.arange(
            self.RECENT_HISTORY_LEN, device=history_items.device, dtype=torch.long
        ).unsqueeze(0).expand_as(history_items)
        pos_emb = self.position_embedding(pos_ids)

        seq_hidden = item_emb + pos_emb
        attn_logits = self.intent_attention(seq_hidden).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1.0e9)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

        intent = torch.sum(attn.unsqueeze(-1) * item_emb, dim=1)
        intent = torch.nan_to_num(intent, nan=0.0, posinf=0.0, neginf=0.0)
        return intent

    def _user_representation(self, user):
        base_user = self.user_embedding(user)
        recent_intent = self._encode_recent_intent(user)
        combined_user = base_user + recent_intent

        if self.mechanism_enabled:
            residual = self.residual_memory(user)
            gate_input = torch.cat([base_user, recent_intent], dim=-1)
            gate = torch.sigmoid(self.residual_gate(gate_input))
            combined_user = combined_user + gate * residual

        combined_user = torch.nan_to_num(combined_user, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return combined_user

    def _score(self, user, item):
        user_e = self._user_representation(user)
        item_e = self.item_embedding(item)
        scores = torch.sum(user_e * item_e, dim=-1)
        return torch.nan_to_num(scores, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_score = self._score(user, pos_item)
        neg_score = self._score(user, neg_item)
        loss = self.loss(pos_score, neg_score)
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0e6, neginf=0.0)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self._score(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self._user_representation(user)
        all_item_e = self.item_embedding.weight
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return scores.view(-1)
