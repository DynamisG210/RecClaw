"""NGCF author-code equations ported to the common RecBole model interface.

Reference: xiangwang1223/neural_graph_collaborative_filtering,
commit a718a4f2df7c3942ca0df6759926975762c61eed, NGCF/NGCF.py.
Variant: adj_type=norm, no pretraining, no node dropout. This is a PyTorch
adaptation, not a bitwise TensorFlow reproduction. See NGCF_FIDELITY.md.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class FreshCandidateModel(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.reg_weight = float(config["reg_weight"])
        self.reg_batch_size = int(config["train_batch_size"])
        widths = [int(config["embedding_size"])] + list(config["hidden_size_list"])
        self.message_dropout = list(config["message_dropout"])
        if len(self.message_dropout) != len(widths) - 1:
            raise ValueError("NGCF needs one message dropout probability per layer")
        self.user_embedding = nn.Embedding(self.n_users, widths[0])
        self.item_embedding = nn.Embedding(self.n_items, widths[0])
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.W_gc = nn.ParameterList()
        self.b_gc = nn.ParameterList()
        self.W_bi = nn.ParameterList()
        self.b_bi = nn.ParameterList()
        for left, right in zip(widths[:-1], widths[1:]):
            for group, shape in [(self.W_gc, (left, right)), (self.b_gc, (1, right)),
                                 (self.W_bi, (left, right)), (self.b_bi, (1, right))]:
                value = nn.Parameter(torch.empty(shape))
                nn.init.xavier_uniform_(value)
                group.append(value)
        # Author norm variant: row-normalize A+I, not symmetric LightGCN A.
        interactions = dataset.inter_matrix(form="coo").astype(np.float32)
        rows = np.concatenate([interactions.row, interactions.col + self.n_users])
        cols = np.concatenate([interactions.col + self.n_users, interactions.row])
        size = self.n_users + self.n_items
        adjacency = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                  shape=(size, size)).tocsr()
        adjacency.data[:] = 1  # author's train interaction matrix is binary
        adjacency = adjacency + sp.eye(size, dtype=np.float32, format="csr")
        degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        normalized = (sp.diags(1.0 / degrees) @ adjacency).tocoo()
        self.graph = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack([normalized.row, normalized.col]).astype(np.int64)),
            torch.from_numpy(normalized.data.astype(np.float32)), (size, size)
        ).coalesce().to(self.device)
        self.restore_user_e = self.restore_item_e = None
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def forward(self):
        hidden = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        outputs = [hidden]
        for index in range(len(self.W_gc)):
            neighbor = torch.sparse.mm(self.graph, hidden)
            summed = F.leaky_relu(neighbor @ self.W_gc[index] + self.b_gc[index], 0.2)
            interacted = F.leaky_relu((hidden * neighbor) @ self.W_bi[index] + self.b_bi[index], 0.2)
            hidden = F.dropout(summed + interacted, p=self.message_dropout[index], training=self.training)
            # TF l2_normalize uses sqrt(max(sum(x*x), epsilon)). Normalized
            # outputs are concatenated; the unnormalized hidden feeds next layer.
            outputs.append(hidden / hidden.square().sum(1, keepdim=True).clamp_min(1e-12).sqrt())
        combined = torch.cat(outputs, dim=1)
        return torch.split(combined, [self.n_users, self.n_items])

    def calculate_loss(self, interaction):
        self.restore_user_e = self.restore_item_e = None
        users, items = self.forward()
        u = users[interaction[self.USER_ID]]
        pos = items[interaction[self.ITEM_ID]]
        neg = items[interaction[self.NEG_ITEM_ID]]
        margin = (u * pos).sum(1) - (u * neg).sum(1)
        # Stable equivalent of author's mean(-log(sigmoid(margin))), not its
        # commented alternative that sums the softplus across the batch.
        recommendation = -F.logsigmoid(margin).mean()
        regularization = self.reg_weight * sum(x.square().sum() for x in (u, pos, neg)) / (2 * self.reg_batch_size)
        return recommendation, regularization

    def predict(self, interaction):
        users, items = self.forward()
        return (users[interaction[self.USER_ID]] * items[interaction[self.ITEM_ID]]).sum(1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        return (self.restore_user_e[interaction[self.USER_ID]] @ self.restore_item_e.T).reshape(-1)
