"""Amazon Reviews Sequential Recommendation — Benchmark Harness.

This is the **single entry point** for any MLE agent or the RecHarness framework
to evaluate a model on this benchmark.  It mirrors the call contract of the
existing ``eval_interface.run_evaluation`` so agents are not confused by a
new API, while adding:

  * Multi-dataset evaluation in one call
  * Val (fast) vs. Test (final) evaluation modes
  * Protocol guard (ProtocolViolation on contract breach)
  * Aggregate metrics across all four datasets
  * BenchmarkResult dataclass with per-dataset breakdown

Usage (agent code)
------------------
    import sys
    sys.path.insert(0, './input')

    from gagc.benchmarks.amazon_reviews.harness import evaluate

    # --- Validation mode (fast, call often during training) ---
    val_result = evaluate(
        predict_script  = './working/predict.py',
        data_dir        = './input/trainval',
        mode            = 'val',
        max_eval_users  = 200,
    )
    print(f"Val HR@10: {val_result.primary_metric:.4f}")

    # --- Test mode (call ONCE per dataset after all training) ---
    test_result = evaluate(
        predict_script  = './working/predict.py',
        data_dir        = './input/trainval',
        test_dir        = './input/test',
        mode            = 'test',
    )
    print(f"Test HR@10: {test_result.primary_metric:.4f}")
    print(test_result.summary_table())

Data interface dependency
-------------------------
The harness loads data using the same ``data_interface`` / ``eval_interface``
modules from ``Amazon_Reviews_Data/``.  Pass ``interface_dir`` to specify
where they live, or set the ``AMAZON_REVIEWS_INTERFACE_DIR`` environment variable.
"""
from __future__ import annotations

import inspect
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np

from gagc.benchmarks.amazon_reviews.grader import MetricAccumulator, rank_positive, aggregate_dataset_metrics
from gagc.benchmarks.amazon_reviews.protocol import (
    ProtocolViolation,
    check_no_trivial_scores,
    check_scores,
    load_predict_fn,
)
from gagc.benchmarks.amazon_reviews.task import (
    ALL_METRICS,
    DATASETS,
    EVAL_SEED,
    MAX_EVAL_USERS,
    NUM_CANDIDATES,
    PRIMARY_METRIC,
    TASK,
    VAL_FAST_USERS,
    VAL_NUM_NEG,
    active_datasets,
)


# ── Result dataclass ─────────────────────────────────────────────────

@dataclass
class DatasetResult:
    dataset: str
    metrics: dict[str, float]
    n_users: int
    mode: str  # 'val' or 'test'

    @property
    def hr10(self) -> float:
        return self.metrics.get("HR@10", self.metrics.get("Recall@10", 0.0))


@dataclass
class BenchmarkResult:
    """Aggregate result across all evaluated datasets."""

    mode: str                                           # 'val' or 'test'
    per_dataset: dict[str, DatasetResult] = field(default_factory=dict)
    aggregate: dict[str, float] = field(default_factory=dict)

    @property
    def primary_metric(self) -> float:
        """Average primary metric across all evaluated datasets."""
        if _full_ranking_enabled():
            return self.aggregate.get("Recall@10", self.aggregate.get("HR@10", 0.0))
        return self.aggregate.get(PRIMARY_METRIC, 0.0)

    def summary_table(self) -> str:
        """Return a human-readable markdown-style summary table."""
        if _full_ranking_enabled():
            header = f"{'Dataset':<35} {'R@5':>7} {'R@10':>7} {'N@5':>9} {'N@10':>9}"
        else:
            header = f"{'Dataset':<35} {'HR@5':>7} {'HR@10':>7} {'HR@20':>7} {'NDCG@5':>9} {'NDCG@10':>9} {'NDCG@20':>9}"
        sep = "-" * len(header)
        rows = [header, sep]
        for ds in active_datasets():
            if ds not in self.per_dataset:
                continue
            m = self.per_dataset[ds].metrics
            if _full_ranking_enabled():
                rows.append(
                    f"{ds:<35} {m.get('Recall@5', m.get('HR@5', 0)):.4f}  {m.get('Recall@10', m.get('HR@10', 0)):.4f}  "
                    f"{m.get('NDCG@5',0):.4f}    {m.get('NDCG@10',0):.4f}"
                )
            else:
                rows.append(
                    f"{ds:<35} {m.get('HR@5',0):.4f}  {m.get('HR@10',0):.4f}  {m.get('HR@20',0):.4f}  "
                    f"{m.get('NDCG@5',0):.4f}    {m.get('NDCG@10',0):.4f}    {m.get('NDCG@20',0):.4f}"
                )
        rows.append(sep)
        agg = self.aggregate
        if _full_ranking_enabled():
            rows.append(
                f"{'AVERAGE':<35} {agg.get('Recall@5', agg.get('HR@5', 0)):.4f}  {agg.get('Recall@10', agg.get('HR@10', 0)):.4f}  "
                f"{agg.get('NDCG@5',0):.4f}    {agg.get('NDCG@10',0):.4f}"
            )
        else:
            rows.append(
                f"{'AVERAGE':<35} {agg.get('HR@5',0):.4f}  {agg.get('HR@10',0):.4f}  {agg.get('HR@20',0):.4f}  "
                f"{agg.get('NDCG@5',0):.4f}    {agg.get('NDCG@10',0):.4f}    {agg.get('NDCG@20',0):.4f}"
            )
        return "\n".join(rows)


# ── Internal data helpers ─────────────────────────────────────────────

def _read_split(filepath: str) -> dict[int, list[int]]:
    user_items: dict[int, list[int]] = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, i = line.split()
            user_items[int(u)].append(int(i))
    return dict(user_items)


def _resolve_interface_dir(interface_dir: str | None) -> str | None:
    """Try to find the Amazon_Reviews_Data directory.

    Priority:
      1. Explicit ``interface_dir`` argument
      2. ``AMAZON_REVIEWS_INTERFACE_DIR`` env var
      3. Returns None (caller falls back to raw file reads)
    """
    if interface_dir:
        return interface_dir
    env = os.environ.get("AMAZON_REVIEWS_INTERFACE_DIR")
    if env:
        return env
    return None


def _full_ranking_enabled() -> bool:
    return os.environ.get("GAGC_FULL_RANKING", "").strip().lower() in {"1", "true", "yes", "on"}


def _with_recall_aliases(metrics: dict[str, float]) -> dict[str, float]:
    if not _full_ranking_enabled():
        return metrics
    aliased = dict(metrics)
    if "HR@5" in aliased:
        aliased["Recall@5"] = aliased["HR@5"]
    if "HR@10" in aliased:
        aliased["Recall@10"] = aliased["HR@10"]
    return {k: v for k, v in aliased.items() if k in {"Recall@5", "Recall@10", "NDCG@5", "NDCG@10"}}


def _load_data_raw(
    dataset_name: str,
    data_dir: str,
    test_dir: str | None,
) -> tuple[dict, dict, dict | None, int]:
    """Read train/val (and optionally test) interaction files directly."""
    train_path = os.path.join(data_dir, f"{dataset_name}_train.txt")
    valid_path = os.path.join(data_dir, f"{dataset_name}_valid.txt")

    user_train = _read_split(train_path)
    user_valid = _read_split(valid_path)
    user_test: dict | None = None

    if test_dir:
        test_path = os.path.join(test_dir, f"{dataset_name}_test.txt")
        if os.path.isfile(test_path):
            user_test = _read_split(test_path)

    # Prefer dataset_stats.json so itemnum covers all splits including test
    # candidates, preventing neg_pool from being too small and inflating HR@10.
    import json
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if dataset_name in stats:
            itemnum = stats[dataset_name]["itemnum"]
        else:
            itemnum = max(stats[ds]["itemnum"] for ds in stats)
    else:
        # Fallback: scan all available split files
        item_ids = [max(v) for v in user_train.values() if v] + \
                   [max(v) for v in user_valid.values() if v]
        if user_test:
            item_ids += [max(v) for v in user_test.values() if v]
        itemnum = max(item_ids) if item_ids else 0

    return user_train, user_valid, user_test, itemnum


# ── Core per-dataset evaluation loop ─────────────────────────────────

def _evaluate_one_dataset(
    dataset_name: str,
    predict_fn: Callable,
    data_dir: str,
    test_dir: str | None,
    mode: Literal["val", "test"],
    max_eval_users: int,
    seed: int,
    num_candidates: int,
) -> DatasetResult:
    """Run the evaluation loop for one dataset.

    Val mode: history = train sequence, positive = val item.
    Test mode: history = train + val sequence, positive = test item.
    """
    random.seed(seed)
    np.random.seed(seed)

    user_train, user_valid, user_test, itemnum = _load_data_raw(
        dataset_name, data_dir, test_dir
    )

    all_items = set(range(1, itemnum + 1))

    if mode == "val":
        # Build eval users from val split
        eval_users = [
            u for u in user_valid
            if u in user_train and len(user_valid[u]) > 0
        ]
    else:
        if user_test is None:
            raise ValueError(
                f"test_dir required for mode='test' but no test file found "
                f"for '{dataset_name}' in '{test_dir}'."
            )
        eval_users = [
            u for u in user_test
            if u in user_train and len(user_test[u]) > 0
        ]

    if len(eval_users) > max_eval_users:
        eval_users = random.sample(eval_users, max_eval_users)

    acc = MetricAccumulator()

    # Support optional dataset_name kwarg for per-dataset predict functions.
    # Check once before the loop to avoid per-user overhead.
    try:
        _predict_accepts_dataset = "dataset_name" in inspect.signature(predict_fn).parameters
    except (ValueError, TypeError):
        _predict_accepts_dataset = False

    for u in eval_users:
        train_seq = user_train.get(u, [])
        val_seq   = user_valid.get(u, [])

        if mode == "val":
            history  = train_seq
            pos_item = val_seq[0]
            interacted = set(train_seq) | set(val_seq) | {0}
        else:
            history  = train_seq + val_seq
            pos_item = user_test[u][0]  # type: ignore[index]
            interacted = set(train_seq) | set(val_seq) | {pos_item, 0}

        if _full_ranking_enabled():
            candidates = [pos_item] + sorted(all_items - interacted - {pos_item})
        elif num_candidates and num_candidates > 0:
            neg_pool = list(all_items - interacted - {pos_item})
            if len(neg_pool) >= num_candidates:
                negatives = random.sample(neg_pool, num_candidates)
            else:
                negatives = neg_pool
            candidates = [pos_item] + negatives
        else:
            candidates = [pos_item] + sorted(all_items - interacted - {pos_item})

        # Support optional dataset_name kwarg for per-dataset predict functions.
        # Falls back to positional call for legacy predict(user_id, history, candidates).
        if _predict_accepts_dataset:
            raw_scores = predict_fn(u, history, candidates, dataset_name=dataset_name)
        else:
            raw_scores = predict_fn(u, history, candidates)
        scores = check_scores(raw_scores, candidates, context=f"{dataset_name}/user={u}")

        rank = rank_positive(scores)
        acc.update(rank)

    return DatasetResult(
        dataset=dataset_name,
        metrics=_with_recall_aliases(acc.result()),
        n_users=acc.count,
        mode=mode,
    )


# ── Public API ────────────────────────────────────────────────────────

def evaluate(
    predict_script: str,
    data_dir: str,
    test_dir: Optional[str] = None,
    datasets: Optional[list[str]] = None,
    mode: Literal["val", "test"] = "test",
    num_candidates: int = NUM_CANDIDATES,
    max_eval_users: int = MAX_EVAL_USERS,
    seed: int = EVAL_SEED,
    interface_dir: Optional[str] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Evaluate a model on the Amazon Reviews SeqRec benchmark.

    This is the **main entry point** for RecHarness and other MLE agents.

    Parameters
    ----------
    predict_script:
        Path to a Python script that exposes:
            def predict(user_id, history, candidates) -> list[float]
    data_dir:
        Directory containing ``*_train.txt`` and ``*_valid.txt`` files.
    test_dir:
        Directory containing ``*_test.txt`` files.
        Required when ``mode='test'``. Defaults to ``data_dir``.
    datasets:
        List of dataset names to evaluate. Defaults to all four datasets.
    mode:
        ``'val'``  — evaluate on the validation split (fast, use often).
        ``'test'`` — evaluate on the test split (final, call once per run).
    num_candidates:
        Number of negative candidates per query.  **Must be 99** for fair
        comparison — changing this breaks comparability with the leaderboard.
    max_eval_users:
        Cap on users evaluated per dataset.
        Use 200 for fast val feedback, 10000 for final test.
    seed:
        Random seed for candidate sampling. **Must be 42** for fair comparison.
    interface_dir:
        Optional path to Amazon_Reviews_Data/ (for eval_interface fallback).
    verbose:
        Print per-dataset progress lines.

    Returns
    -------
    BenchmarkResult
        Contains per-dataset metrics and aggregate averages.
        Access ``.primary_metric`` for the aggregate HR@10.

    Raises
    ------
    ProtocolViolation
        If predict() violates the function contract.
    ValueError
        If mode='test' and test_dir / test files are not found.
    """
    if num_candidates != NUM_CANDIDATES:
        import warnings
        warnings.warn(
            f"num_candidates={num_candidates} deviates from the benchmark standard "
            f"({NUM_CANDIDATES}). Results will NOT be comparable to the leaderboard.",
            stacklevel=2,
        )

    if seed != EVAL_SEED:
        import warnings
        warnings.warn(
            f"seed={seed} deviates from the benchmark standard ({EVAL_SEED}). "
            "Results will NOT be comparable to the leaderboard.",
            stacklevel=2,
        )

    if test_dir is None:
        test_dir = data_dir

    target_datasets = datasets if datasets is not None else active_datasets()

    # Load predict function once — reused across all datasets
    predict_fn = load_predict_fn(predict_script)

    per_dataset: dict[str, DatasetResult] = {}

    for ds in target_datasets:
        if verbose:
            print(f"[benchmark] Evaluating '{ds}' (mode={mode}, users≤{max_eval_users}) ...", flush=True)

        ds_result = _evaluate_one_dataset(
            dataset_name  = ds,
            predict_fn    = predict_fn,
            data_dir      = data_dir,
            test_dir      = test_dir if mode == "test" else None,
            mode          = mode,
            max_eval_users= max_eval_users,
            seed          = seed,
            num_candidates= num_candidates,
        )
        per_dataset[ds] = ds_result

        if verbose:
            m = ds_result.metrics
            print(
                (
                    f"  Recall@5={m['Recall@5']:.4f}  Recall@10={m['Recall@10']:.4f}  "
                    f"NDCG@5={m['NDCG@5']:.4f}  NDCG@10={m['NDCG@10']:.4f}  "
                    if _full_ranking_enabled() else
                    f"  HR@10={m['HR@10']:.4f}  HR@20={m['HR@20']:.4f}  "
                    f"NDCG@10={m['NDCG@10']:.4f}  NDCG@20={m['NDCG@20']:.4f}  "
                ) + f"(n={ds_result.n_users})"
            )

    aggregate = aggregate_dataset_metrics({ds: r.metrics for ds, r in per_dataset.items()})

    result = BenchmarkResult(
        mode=mode,
        per_dataset=per_dataset,
        aggregate=aggregate,
    )

    # Protocol sanity check — warn if any dataset looks trivially perfect
    check_no_trivial_scores({ds: r.metrics for ds, r in per_dataset.items()})

    if verbose:
        print()
        print(result.summary_table())
        print(f"\nPrimary metric ({PRIMARY_METRIC}, avg): {result.primary_metric:.4f}")

    return result


def evaluate_val_fast(
    predict_script: str,
    data_dir: str,
    max_eval_users: int = 200,
    datasets: Optional[list[str]] = None,
    seed: int = EVAL_SEED,
    verbose: bool = True,
) -> BenchmarkResult:
    """Convenience wrapper for fast validation feedback during training.

    Equivalent to ``evaluate(..., mode='val', max_eval_users=200)``.
    Use this inside the training loop. Call ``evaluate(..., mode='test')``
    exactly once after all training is done.
    """
    return evaluate(
        predict_script  = predict_script,
        data_dir        = data_dir,
        mode            = "val",
        max_eval_users  = max_eval_users,
        datasets        = datasets,
        seed            = seed,
        verbose         = verbose,
    )
