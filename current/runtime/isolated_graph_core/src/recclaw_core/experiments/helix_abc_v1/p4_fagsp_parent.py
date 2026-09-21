from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


FROZEN_FAGSP_PARAMETERS: Mapping[str, float | int] = {
    "pri_factor1": 256,
    "pri_factor2": 128,
    "alpha1": 0.3,
    "alpha2": 0.5,
    "order1": 12,
    "order2": 14,
    "q": 0.7,
}


def _normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_degree = matrix.sum(axis=1)
    col_degree = matrix.sum(axis=0)
    row_scale = np.zeros_like(row_degree)
    col_scale = np.zeros_like(col_degree)
    np.power(row_degree, -0.5, out=row_scale, where=row_degree > 0)
    np.power(col_degree, -0.5, out=col_scale, where=col_degree > 0)
    normalized = row_scale[:, None] * matrix * col_scale[None, :]
    col_inverse = np.zeros_like(col_scale)
    np.divide(1.0, col_scale, out=col_inverse, where=col_scale > 0)
    return normalized, col_scale, col_inverse


def _spectral_complement_action(
    left: np.ndarray,
    singular_values: np.ndarray,
    right_t: np.ndarray,
    ratings: np.ndarray,
    item_order: int,
    user_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    item_weight = 1.0 - np.power(1.0 - singular_values**2, item_order)
    user_weight = 1.0 - np.power(1.0 - singular_values**2, user_order)
    p11 = ((ratings @ right_t.T) * item_weight[None, :]) @ right_t
    p12 = (left * user_weight[None, :]) @ (left.T @ ratings)
    return p11, p12


def frozen_fagsp_scores(
    train: sp.csr_matrix, *, residual_inputs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Execute frozen ML-1M FaGSP and restore RecBole's padding column."""

    matrix = train.tocsr().astype(np.float32)
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    ratings = matrix[:, 1:].toarray().astype(np.float64, copy=False)
    params = FROZEN_FAGSP_PARAMETERS

    normalized, item_pre, item_post = _normalize(ratings)
    left, singular_values, right_t = np.linalg.svd(normalized, full_matrices=False)
    p11, p12 = _spectral_complement_action(
        left,
        singular_values,
        right_t,
        ratings,
        int(params["order1"]),
        int(params["order2"]),
    )

    low_right = right_t[-min(int(params["pri_factor2"]), right_t.shape[0]) :]
    p30 = ((ratings * item_pre[None, :]) @ low_right.T) @ low_right
    p30 *= item_post[None, :]
    thresholds = np.quantile(p30, q=float(params["q"]), axis=0, keepdims=True)
    mask = (p30 > thresholds) & (ratings >= 1.0)
    p3 = ratings + float(params["alpha2"]) * mask.astype(np.float64)

    normalized_p3, p3_pre, p3_post = _normalize(p3)
    _, _, right_t_p3 = np.linalg.svd(normalized_p3, full_matrices=False)
    high_right = right_t_p3[: min(int(params["pri_factor1"]), right_t_p3.shape[0])]
    p2 = ((p3 * p3_pre[None, :]) @ high_right.T) @ high_right
    p2 *= p3_post[None, :]

    unpadded = p11 + p12 + float(params["alpha1"]) * p2
    if not np.isfinite(unpadded).all():
        raise RuntimeError("frozen FaGSP parent produced non-finite scores")
    padded = np.zeros(matrix.shape, dtype=np.float64)
    padded[:, 1:] = unpadded
    if residual_inputs is not None:
        # Reuse the decomposition already evaluated by the frozen parent.
        # Padding belongs to the RecBole boundary, not the residual mechanism.
        padded_right_t = np.zeros((right_t.shape[0], matrix.shape[1]), dtype=right_t.dtype)
        padded_right_t[:, 1:] = right_t
        residual_inputs.update(
            train_csr=matrix,
            user_degree=np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel(),
            item_degree=np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel(),
            svd_left=left,
            svd_values=singular_values,
            svd_right_t=padded_right_t,
        )
    return padded


def frozen_fagsp_scores_from_dataset(dataset: Any) -> np.ndarray:
    return frozen_fagsp_scores(dataset.inter_matrix(form="csr"))


class FrozenFaGSPResidualModelBase(GeneralRecommender):
    """Immutable FaGSP execution path with a candidate-owned residual only."""

    input_type = InputType.PAIRWISE

    def __init__(self, config: Any, dataset: Any) -> None:
        super().__init__(config, dataset)
        inputs: dict[str, Any] = {}
        parent = torch.from_numpy(frozen_fagsp_scores(
            dataset.inter_matrix(form="csr"), residual_inputs=inputs,
        ))
        self.register_buffer("_recclaw_frozen_parent", parent, persistent=False)
        self._p4_train_csr = inputs.pop("train_csr")
        self._p4_seed = int(config["seed"])
        coo = self._p4_train_csr.tocoo()
        train_matrix = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
            torch.from_numpy(coo.data.astype(np.float64)),
            size=coo.shape,
        ).coalesce()
        self.register_buffer("_p4_train_matrix", train_matrix, persistent=False)
        for name, value in inputs.items():
            self.register_buffer("_p4_" + name, torch.from_numpy(value), persistent=False)
        self._recclaw_mechanism_enabled = True

    def p4_residual_inputs(self) -> dict[str, Any]:
        """Train-only inputs; tensor references follow the model's device."""
        return {
            "parent_scores": self._recclaw_frozen_parent,
            "train_matrix": self._p4_train_matrix,
            "train_csr": self._p4_train_csr,
            "user_degree": self._p4_user_degree,
            "item_degree": self._p4_item_degree,
            "svd_left": self._p4_svd_left,
            "svd_values": self._p4_svd_values,
            "svd_right_t": self._p4_svd_right_t,
            "seed": self._p4_seed,
        }

    def _mechanism_residual_scores(self) -> torch.Tensor:
        raise NotImplementedError

    def set_mechanism_enabled(self, enabled: bool) -> None:
        self._recclaw_mechanism_enabled = bool(enabled)

    def _score_matrix(self) -> torch.Tensor:
        parent = self._recclaw_frozen_parent
        if not self._recclaw_mechanism_enabled:
            return parent
        residual = self._mechanism_residual_scores()
        if residual.shape != parent.shape:
            raise RuntimeError("P4 residual does not match the frozen parent shape")
        if not torch.isfinite(residual).all().item():
            raise RuntimeError("P4 residual contains non-finite scores")
        return parent + residual.to(device=parent.device, dtype=parent.dtype)

    def calculate_loss(self, interaction: Any) -> torch.Tensor:
        scores = self._score_matrix()
        users = interaction[self.USER_ID]
        positive = scores[users, interaction[self.ITEM_ID]]
        negative = scores[users, interaction[self.NEG_ITEM_ID]]
        return -F.logsigmoid(positive - negative).mean()

    def predict(self, interaction: Any) -> torch.Tensor:
        return self._score_matrix()[interaction[self.USER_ID], interaction[self.ITEM_ID]]

    def full_sort_predict(self, interaction: Any) -> torch.Tensor:
        return self._score_matrix()[interaction[self.USER_ID]].reshape(-1)
