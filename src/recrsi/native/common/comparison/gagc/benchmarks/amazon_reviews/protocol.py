"""Protocol guard — enforces the frozen evaluation rules.

Checks that a predict function and its outputs comply with the benchmark
contract before any metrics are recorded. This mirrors the anti-cheating
rules documented in amazon_reviews_description.md.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Callable


class ProtocolViolation(Exception):
    """Raised when an agent violates the evaluation protocol."""


def load_predict_fn(predict_script: str) -> Callable:
    """Import predict_script and return its `predict` callable.

    Each call uses a unique module name derived from the script path so that
    loading predict.py from multiple workspaces in the same process does not
    cause sys.modules collisions.

    Parameters
    ----------
    predict_script:
        Absolute or relative path to a Python script that exposes:
            def predict(user_id, history, candidates) -> list[float]

    Raises
    ------
    FileNotFoundError
        If the script does not exist.
    ProtocolViolation
        If the script does not define a callable named 'predict'.
    """
    path = os.path.abspath(predict_script)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"predict_script not found: {path}")

    # Unique module name: avoids sys.modules collision when scoring multiple
    # workspaces (each with their own predict.py) in one process.
    module_name = f"_predict_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "predict") or not callable(module.predict):
        raise ProtocolViolation(
            f"'{path}' must define a callable named 'predict'. "
            "Expected signature: predict(user_id, history, candidates) -> list[float]"
        )
    return module.predict


def check_scores(
    scores: object,
    candidates: list[int],
    *,
    context: str = "",
) -> list[float]:
    """Validate that `scores` is a proper float list matching len(candidates).

    Returns the validated list.

    Raises
    ------
    ProtocolViolation
        On type mismatch or length mismatch.
    """
    prefix = f"[{context}] " if context else ""

    try:
        scores_list = list(scores)  # accept numpy arrays, generators, etc.
    except TypeError as exc:
        raise ProtocolViolation(
            f"{prefix}predict() return value is not iterable: {type(scores)}"
        ) from exc

    if len(scores_list) != len(candidates):
        raise ProtocolViolation(
            f"{prefix}predict() returned {len(scores_list)} scores "
            f"but len(candidates)={len(candidates)}. "
            "Return value length MUST equal len(candidates)."
        )

    try:
        scores_float = [float(s) for s in scores_list]
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(
            f"{prefix}predict() scores must be numeric (float-castable). Got: {scores_list[:5]}"
        ) from exc

    return scores_float


def check_no_trivial_scores(
    per_dataset_metrics: dict[str, dict[str, float]],
    hr10_trivial_threshold: float = 0.99,
) -> None:
    """Warn (not raise) if HR@10 looks trivially perfect across any dataset.

    A model that uses item-level statistical features or leaks test labels
    often achieves HR@10 ≈ 1.0 from the first epoch.
    """
    import warnings

    for ds_name, metrics in per_dataset_metrics.items():
        if metrics.get("HR@10", 0.0) >= hr10_trivial_threshold:
            warnings.warn(
                f"\n*** PROTOCOL WARNING ***\n"
                f"Dataset '{ds_name}' has HR@10={metrics['HR@10']:.4f} ≥ {hr10_trivial_threshold}.\n"
                "This is almost certainly caused by one of:\n"
                "  1. num_neg < 99 in load_dataset() for validation\n"
                "  2. Item popularity / IPS / recency features used as model input\n"
                "  3. Test labels accessed directly\n"
                "The result will be flagged as INVALID if any of the above applies.\n"
                "*** END WARNING ***\n",
                stacklevel=3,
            )
