"""Post-training score adjustment candidates for local RecClaw experiments."""

from __future__ import annotations

import torch


class PopularityPenaltyReranker:
    """Subtract a normalized item-popularity penalty from scores."""

    def __init__(self, lambda_pop: float = 0.05) -> None:
        self.lambda_pop = float(lambda_pop)

    def adjust(self, scores: torch.Tensor, item_popularity: torch.Tensor) -> torch.Tensor:
        popularity = item_popularity.to(device=scores.device, dtype=scores.dtype)
        popularity = popularity / popularity.mean().clamp_min(1e-12)
        return scores - self.lambda_pop * popularity


class CoverageBoostReranker:
    """Boost scores for lower-exposure items with a simple inverse exposure signal."""

    def __init__(self, lambda_coverage: float = 0.05) -> None:
        self.lambda_coverage = float(lambda_coverage)

    def adjust(self, scores: torch.Tensor, item_exposure: torch.Tensor) -> torch.Tensor:
        exposure = item_exposure.to(device=scores.device, dtype=scores.dtype)
        inverse_exposure = 1.0 / (1.0 + exposure)
        inverse_exposure = inverse_exposure / inverse_exposure.mean().clamp_min(1e-12)
        return scores + self.lambda_coverage * inverse_exposure
