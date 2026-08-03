import numpy as np
import torch
from torch import nn

from recbole.model.general_recommender.bpr import BPR


class FreshCandidateModel(BPR):
    """Multi-State Transported Preference Routing recommender.

    Enabled mode:
    - each user has K persistent latent states
    - recent train-history items build a context vector from the last L items
    - context generates a transport delta applied to all states
    - candidate-conditioned routing mixes transported states for scoring

    Disabled mode:
    - matched single-state BPR-style user-item dot product using the parent user embedding
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_size = int(config["embedding_size"])
        self.num_states = 4
        self.history_len = 5
        self.transport_scale = 0.1
        self.routing_temperature = 1.0
        self.mechanism_enabled = True

        self.user_states = nn.Embedding(self.n_users, self.num_states * self.embedding_size)
        self.context_to_delta = nn.Linear(self.embedding_size, self.num_states * self.embedding_size, bias=True)
        self.context_gate = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.state_gate = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.apply(self._init_transport_modules)
        self._build_train_histories(dataset)
        self.latest_diagnostics = {
            "state_usage_entropy": 0.0,
            "transport_magnitude": 0.0,
            "mean_history_diversity": float(self.user_history_diversity.mean().item()) if self.user_history_diversity.numel() > 0 else 0.0,
        }

    def _init_transport_modules(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_mechanism_enabled(self, enabled: bool):
        self.mechanism_enabled = bool(enabled)

    def _to_numpy(self, value):
        if value is None:
            return None
        if hasattr(value, "cpu"):
            return value.cpu().numpy()
        return np.asarray(value)

    def _get_time_field(self, dataset):
        if hasattr(dataset, "time_field") and dataset.time_field is not None:
            return dataset.time_field
        inter_feat = getattr(dataset, "inter_feat", None)
        if inter_feat is None:
            return None
        for key in ["timestamp", "time", "ts", "TIME_FIELD"]:
            try:
                if key in inter_feat:
                    return key
            except Exception:
                pass
        try:
            cols = list(inter_feat)
        except Exception:
            cols = []
        for key in cols:
            low = str(key).lower()
            if "time" in low or "stamp" in low:
                return key
        return None

    def _build_train_histories(self, dataset):
        inter_feat = getattr(dataset, "inter_feat", None)
        histories = [[] for _ in range(self.n_users)]
        diversity = np.zeros(self.n_users, dtype=np.float32)

        if inter_feat is None:
            hist_items = torch.zeros((self.n_users, self.history_len), dtype=torch.long)
            hist_len = torch.zeros(self.n_users, dtype=torch.long)
            hist_div = torch.zeros(self.n_users, dtype=torch.float32)
            self.register_buffer("user_hist_items", hist_items)
            self.register_buffer("user_hist_len", hist_len)
            self.register_buffer("user_history_diversity", hist_div)
            return

        user_np = self._to_numpy(inter_feat[self.USER_ID])
        item_np = self._to_numpy(inter_feat[self.ITEM_ID])
        time_field = self._get_time_field(dataset)
        if time_field is not None:
            time_np = self._to_numpy(inter_feat[time_field])
            order = np.argsort(time_np, kind="stable")
            user_np = user_np[order]
            item_np = item_np[order]

        for u, i in zip(user_np, item_np):
            u = int(u)
            i = int(i)
            if 0 <= u < self.n_users and 0 <= i < self.n_items:
                histories[u].append(i)

        hist_items = np.zeros((self.n_users, self.history_len), dtype=np.int64)
        hist_len = np.zeros(self.n_users, dtype=np.int64)
        for u in range(self.n_users):
            seq = histories[u]
            if len(seq) == 0:
                continue
            tail = seq[-self.history_len :]
            hist_len[u] = len(tail)
            hist_items[u, : len(tail)] = np.asarray(tail, dtype=np.int64)
            diversity[u] = float(len(set(seq))) / float(max(len(seq), 1))

        self.register_buffer("user_hist_items", torch.from_numpy(hist_items))
        self.register_buffer("user_hist_len", torch.from_numpy(hist_len))
        self.register_buffer("user_history_diversity", torch.from_numpy(diversity))

    def _get_recent_context(self, user):
        hist_items = self.user_hist_items[user]
        hist_len = self.user_hist_len[user]
        item_emb = self.item_embedding(hist_items)
        mask = (hist_items > 0).float().unsqueeze(-1)
        denom = hist_len.clamp(min=1).float().unsqueeze(-1)
        context = (item_emb * mask).sum(dim=1) / denom
        return context

    def _get_transported_states(self, user):
        base_states = self.user_states(user).view(-1, self.num_states, self.embedding_size)
        context = self._get_recent_context(user)
        delta = torch.tanh(self.context_to_delta(context)).view(-1, self.num_states, self.embedding_size)
        transported = base_states + self.transport_scale * delta
        return transported, context, delta

    def _routing_logits(self, transported_states, context, item_e):
        state_item = (transported_states * item_e.unsqueeze(1)).sum(dim=-1)
        gated_context = self.context_gate(context)
        gated_states = self.state_gate(transported_states)
        context_affinity = (gated_states * gated_context.unsqueeze(1)).sum(dim=-1)
        return state_item + context_affinity

    def _enabled_score(self, user, item):
        transported, context, delta = self._get_transported_states(user)
        item_e = self.item_embedding(item)
        logits = self._routing_logits(transported, context, item_e)
        logits = logits / self.routing_temperature
        weights = torch.softmax(logits, dim=1)
        state_item = (transported * item_e.unsqueeze(1)).sum(dim=-1)
        score = (weights * state_item).sum(dim=1)

        with torch.no_grad():
            p = weights.clamp(min=1e-12)
            entropy = (-p * torch.log(p)).sum(dim=1).mean()
            transport_mag = delta.norm(dim=-1).mean()
            self.latest_diagnostics = {
                "state_usage_entropy": float(entropy.detach().cpu().item()),
                "transport_magnitude": float(transport_mag.detach().cpu().item()),
                "mean_history_diversity": float(self.user_history_diversity[user].float().mean().detach().cpu().item()) if user.numel() > 0 else 0.0,
            }

        return score

    def _disabled_score(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return (user_e * item_e).sum(dim=1)

    def _enabled_full_sort_score(self, user):
        transported, context, delta = self._get_transported_states(user)
        all_item_e = self.item_embedding.weight
        state_item = torch.einsum("bkd,nd->bkn", transported, all_item_e)
        gated_context = self.context_gate(context)
        gated_states = self.state_gate(transported)
        context_affinity = (gated_states * gated_context.unsqueeze(1)).sum(dim=-1).unsqueeze(-1)
        logits = (state_item + context_affinity) / self.routing_temperature
        logits = logits - logits.max(dim=1, keepdim=True).values
        weights = torch.softmax(logits, dim=1)
        scores = (weights * state_item).sum(dim=1)

        with torch.no_grad():
            p = weights.clamp(min=1e-12)
            entropy = (-p * torch.log(p)).sum(dim=1).mean()
            transport_mag = delta.norm(dim=-1).mean()
            self.latest_diagnostics = {
                "state_usage_entropy": float(entropy.detach().cpu().item()),
                "transport_magnitude": float(transport_mag.detach().cpu().item()),
                "mean_history_diversity": float(self.user_history_diversity[user].float().mean().detach().cpu().item()) if user.numel() > 0 else 0.0,
            }

        return scores

    def _disabled_full_sort_score(self, user):
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        return torch.matmul(user_e, all_item_e.transpose(0, 1))

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        if self.mechanism_enabled:
            pos_score = self._enabled_score(user, pos_item)
            neg_score = self._enabled_score(user, neg_item)
            reg = (
                self.user_states(user).pow(2).mean()
                + self.item_embedding(pos_item).pow(2).mean()
                + self.item_embedding(neg_item).pow(2).mean()
            ) * 1e-8
        else:
            pos_score = self._disabled_score(user, pos_item)
            neg_score = self._disabled_score(user, neg_item)
            reg = (
                self.user_embedding(user).pow(2).mean()
                + self.item_embedding(pos_item).pow(2).mean()
                + self.item_embedding(neg_item).pow(2).mean()
            ) * 1e-8

        loss = self.loss(pos_score, neg_score)
        loss = loss + reg
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.mechanism_enabled:
            score = self._enabled_score(user, item)
        else:
            score = self._disabled_score(user, item)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.mechanism_enabled:
            scores = self._enabled_full_sort_score(user)
        else:
            scores = self._disabled_full_sort_score(user)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores.view(-1)
