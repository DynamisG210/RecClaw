"""Amazon Reviews Sequential Recommendation — Task Specification.

This module defines the frozen task spec: datasets, metric targets,
candidate generation protocol, and evaluation constraints.
It is the single source of truth for what any MLE agent is expected to do.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

# ── Frozen dataset list ──────────────────────────────────────────────
DATASETS: Final[list[str]] = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]


def active_datasets() -> list[str]:
    """Return runtime datasets, optionally overridden by GAGC_DATASETS."""
    raw = os.environ.get("GAGC_DATASETS", "").strip()
    if not raw:
        return list(DATASETS)
    datasets = [part.strip() for part in raw.split(",") if part.strip()]
    return datasets or list(DATASETS)


def active_dataset_label() -> str:
    """Human-readable label for the currently configured Amazon dataset set."""
    datasets = active_datasets()
    if datasets == list(DATASETS):
        return "the default four Amazon Reviews datasets"
    if len(datasets) == 1:
        return f"Amazon Reviews dataset {datasets[0]}"
    return "Amazon Reviews datasets " + ", ".join(datasets)

# ── Evaluation protocol constants (FROZEN — agents must not change) ──
EVAL_SEED: Final[int] = 42
NUM_CANDIDATES: Final[int] = 99      # negatives per query; total = 100 (1 pos + 99 neg)
MAX_EVAL_USERS: Final[int] = 10_000
VAL_NUM_NEG: Final[int] = 99         # minimum negatives for valid val evaluation
VAL_FAST_USERS: Final[int] = 200     # cap for fast val feedback during iteration

# ── Primary metric for ranking / leaderboard ─────────────────────────
PRIMARY_METRIC: Final[str] = "HR@10"
METRIC_MAXIMIZE: Final[bool] = True

ALL_METRICS: Final[list[str]] = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of the Amazon Reviews SeqRec benchmark task."""

    name: str = "amazon-reviews-seqrec"
    description: str = field(default_factory=lambda: (
        f"Sequential recommendation on {active_dataset_label()}. "
        "Leave-one-out split; 1 positive + 99 random negatives as candidates by default. "
        "Maximize HR@10; report the configured ranking metrics for all active datasets."
    ))
    datasets: tuple[str, ...] = field(default_factory=lambda: tuple(active_datasets()))
    primary_metric: str = PRIMARY_METRIC
    metric_maximize: bool = METRIC_MAXIMIZE
    all_metrics: tuple[str, ...] = tuple(ALL_METRICS)
    num_candidates: int = NUM_CANDIDATES
    eval_seed: int = EVAL_SEED
    max_eval_users: int = MAX_EVAL_USERS

    # Data layout expected by the harness
    data_dir_key: str = "trainval"   # subdir containing *_train.txt, *_valid.txt
    test_dir_key: str = "test"       # subdir containing *_test.txt
    stats_filename: str = "dataset_stats.json"

    # Predict function contract
    predict_fn_name: str = "predict"
    predict_signature: str = (
        "def predict(user_id: int, history: list[int], candidates: list[int]) "
        "-> list[float]"
    )


# Singleton accessed by the rest of the package
TASK = TaskSpec()
