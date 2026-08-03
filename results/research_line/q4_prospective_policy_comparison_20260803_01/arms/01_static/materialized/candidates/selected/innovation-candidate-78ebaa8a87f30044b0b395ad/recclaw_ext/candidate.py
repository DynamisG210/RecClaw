import torch
from torch import nn

from recbole.model.general_recommender.bpr import BPR
from recbole.model.init import xavier_normal_initialization


class FreshCandidateModel(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.mechanism_enabled = True
        self.embedding_size = int(config["embedding_size"])
        self.history_length = 5

        self.residual_memory_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.gate_layer = nn.Linear(self.embedding_size * 2, self.embedding_size)

        history_items = self._build_user_recent_histories(dataset, self.history_length)
        self.register_buffer("user_recent_history", history_items)
        position_weight = torch.arange(1, self.history_length + 1, dtype=torch.float)
        self.register_buffer("history_position_weight", position_weight)

        self.apply(xavier_normal_initialization)

    def set_mechanism_enabled(self, enabled: bool):
        self.mechanism_enabled = bool(enabled)

    def _build_user_recent_histories(self, dataset, history_length):
        history = torch.zeros((self.n_users, history_length), dtype=torch.long)

        inter_feat = getattr(dataset, "inter_feat", None)
        if inter_feat is None:
            return history

        try:
            user_tensor = inter_feat[self.USER_ID].cpu()
            item_tensor = inter_feat[self.ITEM_ID].cpu()
        except Exception:
            return history

        time_tensor = None
        time_field = getattr(dataset, "time_field", None)
        if time_field:
            try:
                time_tensor = inter_feat[time_field].cpu()
            except Exception:
                time_tensor = None

        num_inter = int(user_tensor.shape[0])
        if num_inter == 0:
            return history

        user_lists = [[] for _ in range(self.n_users)]

        if time_tensor is None:
            for idx in range(num_inter):
                u = int(user_tensor[idx])
                i = int(item_tensor[idx])
                if 0 <= u < self.n_users:
                    user_lists[u].append(i)
        else:
            order = sorted(
                range(num_inter),
                key=lambda idx: (int(user_tensor[idx]), float(time_tensor[idx]), idx),
            )
            for idx in order:
                u = int(user_tensor[idx])
                i = int(item_tensor[idx])
                if 0 <= u < self.n_users:
                    user_lists[u].append(i)

        for u in range(self.n_users):
            seq = user_lists[u][-history_length:]
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            history[u, -len(seq) :] = seq_tensor

        return history

    def _recent_intent(self, user):
        history_items = self.user_recent_history[user]
        mask = history_items.gt(0).float()
        item_emb = self.item_embedding(history_items)

        weight = self.history_position_weight.view(1, -1, 1).to(item_emb.device)
        mask3 = mask.unsqueeze(-1)
        weighted_item_emb = item_emb * mask3 * weight
        denom = (mask.unsqueeze(-1) * weight).sum(dim=1).clamp_min(1.0)
        intent = weighted_item_emb.sum(dim=1) / denom
        return intent

    def _fused_user_representation(self, user):
        base_user = self.user_embedding(user)
        recent_intent = self._recent_intent(user)
        direct_plus_intent = base_user + recent_intent

        if self.mechanism_enabled:
            residual_memory = self.residual_memory_embedding(user)
            gate_input = torch.cat([base_user, recent_intent], dim=-1)
            gate = torch.sigmoid(self.gate_layer(gate_input))
            fused_user = direct_plus_intent + gate * residual_memory
        else:
            fused_user = direct_plus_intent

        return fused_user

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self._fused_user_representation(user)
        pos_e = self.item_embedding(pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_score = torch.sum(user_e * pos_e, dim=-1)
        neg_score = torch.sum(user_e * neg_e, dim=-1)
        loss = self.loss(pos_score, neg_score)
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self._fused_user_representation(user)
        item_e = self.item_embedding(item)
        score = torch.sum(user_e * item_e, dim=-1)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self._fused_user_representation(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        return score.view(-1)
