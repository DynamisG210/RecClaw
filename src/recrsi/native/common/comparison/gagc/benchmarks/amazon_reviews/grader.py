"""Metric computation for sequential recommendation evaluation.

Standalone module — no dependency on data loading or predict scripts.
All functions operate on pre-computed ranks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MetricAccumulator:
    """Accumulates HR and NDCG hits over a stream of ranked queries."""

    hr5: float = 0.0
    hr10: float = 0.0
    hr20: float = 0.0
    ndcg5: float = 0.0
    ndcg10: float = 0.0
    ndcg20: float = 0.0
    count: int = 0

    def update(self, rank: int) -> None:
        """Record one query result. rank is 0-indexed position of the positive."""
        self.hr5    += 1.0 if rank < 5 else 0.0
        self.hr10   += 1.0 if rank < 10 else 0.0
        self.hr20   += 1.0 if rank < 20 else 0.0
        self.ndcg5  += (1.0 / math.log2(rank + 2)) if rank < 5 else 0.0
        self.ndcg10 += (1.0 / math.log2(rank + 2)) if rank < 10 else 0.0
        self.ndcg20 += (1.0 / math.log2(rank + 2)) if rank < 20 else 0.0
        self.count  += 1

    def result(self) -> dict[str, float]:
        """Return averaged metrics dict. Raises if no queries were recorded."""
        if self.count == 0:
            raise RuntimeError("No queries evaluated — accumulator is empty.")
        n = self.count
        return {
            "HR@5":    round(self.hr5    / n, 4),
            "HR@10":   round(self.hr10   / n, 4),
            "HR@20":   round(self.hr20   / n, 4),
            "NDCG@5":  round(self.ndcg5  / n, 4),
            "NDCG@10": round(self.ndcg10 / n, 4),
            "NDCG@20": round(self.ndcg20 / n, 4),
        }


def rank_positive(scores: list[float]) -> int:
    """Return 0-indexed rank of candidates[0] (the positive) after descending sort.

    Ties are broken by original index order (lower index = better rank).
    """
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])
    for position, (orig_idx, _) in enumerate(indexed):
        if orig_idx == 0:
            return position
    raise ValueError("Positive item (index 0) not found in scores — should never happen.")


def aggregate_dataset_metrics(
    per_dataset: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Average per-dataset metric dicts into a single summary dict.

    Keys: "HR@10", "HR@20", "NDCG@10", "NDCG@20" and additionally
    one key per dataset "<dataset>/<metric>" for detailed inspection.
    """
    from gagc.benchmarks.amazon_reviews.task import ALL_METRICS

    agg: dict[str, float] = {m: 0.0 for m in ALL_METRICS}
    n = len(per_dataset)

    flat: dict[str, float] = {}
    for ds_name, ds_metrics in per_dataset.items():
        for metric, value in ds_metrics.items():
            agg[metric] = agg.get(metric, 0.0) + value
            flat[f"{ds_name}/{metric}"] = value

    if n > 0:
        for m in list(agg):
            agg[m] = round(agg[m] / n, 4)

    return {**agg, **flat}
