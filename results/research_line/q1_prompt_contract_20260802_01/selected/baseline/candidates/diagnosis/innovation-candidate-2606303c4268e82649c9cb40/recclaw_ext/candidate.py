import torch
from recbole.model.general_recommender.bpr import BPR
from recbole.utils import InputType


class FreshCandidateModel(BPR):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.mechanism_enabled = True
        self.rank_aware_weight = 0.1
        self.rank_aware_temperature = 1.0

    def set_mechanism_enabled(self, enabled: bool):
        self.mechanism_enabled = bool(enabled)

    def _safe_score(self, user_e, item_e):
        score = torch.mul(user_e, item_e).sum(dim=1)
        return torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    def _rank_aware_term(self, user_e, pos_e, neg_e):
        pos_score = self._safe_score(user_e, pos_e)
        neg_score = self._safe_score(user_e, neg_e)
        diff = (pos_score - neg_score) / self.rank_aware_temperature
        diff = torch.clamp(diff, min=-50.0, max=50.0)
        prob = torch.sigmoid(diff)
        uncertainty = prob * (1.0 - prob)
        term = self.rank_aware_weight * uncertainty
        return torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self.get_user_embedding(user)
        pos_e = self.get_item_embedding(pos_item)
        neg_e = self.get_item_embedding(neg_item)

        pos_item_score = self._safe_score(user_e, pos_e)
        neg_item_score = self._safe_score(user_e, neg_e)

        loss = self.loss(pos_item_score, neg_item_score)
        if self.mechanism_enabled:
            loss = loss + self._rank_aware_term(user_e, pos_e, neg_e).mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        score = self._safe_score(user_e, item_e)
        return score

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        return score.view(-1)
