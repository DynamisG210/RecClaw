#!/usr/bin/env python3
"""Utilities for loading the RecClaw runtime action-space contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTION_SPACE_PATH = PROJECT_ROOT / "configs" / "action_space.yaml"


FALLBACK_PARAMETER_SPACE: dict[str, list[Any]] = {
    "embedding_size": [32, 64, 128],
    "learning_rate": [0.0001, 0.001, 0.005],
    "n_layers": [1, 2, 3],
    "reg_weight": [1e-6, 1e-5, 1e-4],
    "margin": [0.1, 0.2, 0.5],
    "residual_weight": [0.05, 0.1, 0.2, 0.3],
    "tail_weight_alpha": [0.2, 0.5, 1.0],
    "lambda_pop": [1e-4, 1e-3, 1e-2],
    "lambda_norm": [1e-5, 1e-4, 1e-3],
    "max_norm": [0.5, 1.0, 2.0],
    "hard_negative_ratio": [0.25, 0.5, 0.75],
    "popularity_alpha": [0.2, 0.5, 1.0],
    "debias_alpha": [0.1, 0.2, 0.5],
    "lambda_align": [1e-4, 1e-3, 1e-2],
    "rank_weight_alpha": [0.1, 0.2, 0.5],
    "lambda_coverage": [1e-4, 1e-3, 1e-2],
    "edge_dropout": [0.05, 0.1, 0.2],
    "gate_dropout": [0.0, 0.05, 0.1],
    "residual_gate_scale": [0.25, 0.5, 1.0],
    "cl_temperature": [0.1, 0.2, 0.5],
    "pareto_temperature": [0.2, 0.5, 1.0],
}


FALLBACK_PARAMETER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("residual_weight",),
    ("margin",),
    ("tail_weight_alpha",),
    ("hard_negative_ratio",),
    ("popularity_alpha",),
    ("lambda_pop",),
    ("lambda_norm", "max_norm"),
    ("lambda_align",),
    ("rank_weight_alpha",),
    ("lambda_coverage",),
    ("embedding_size", "n_layers"),
    ("residual_gate_scale", "gate_dropout"),
    ("learning_rate",),
    ("reg_weight",),
)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_action_space(path: str | Path = ACTION_SPACE_PATH) -> dict[str, Any]:
    payload = load_yaml(Path(path))
    return payload if isinstance(payload, dict) else {}


def parameter_space_from_action_space(action_space: dict[str, Any]) -> dict[str, list[Any]]:
    raw = action_space.get("parameter_space")
    if not isinstance(raw, dict):
        return dict(FALLBACK_PARAMETER_SPACE)
    parameter_space: dict[str, list[Any]] = {}
    for key, spec in raw.items():
        if isinstance(spec, dict):
            values = spec.get("values")
        else:
            values = spec
        if isinstance(values, list):
            parameter_space[str(key)] = values
    return parameter_space or dict(FALLBACK_PARAMETER_SPACE)


def parameter_groups_from_action_space(action_space: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    raw = action_space.get("parameter_groups")
    if not isinstance(raw, list):
        return FALLBACK_PARAMETER_GROUPS
    groups: list[tuple[str, ...]] = []
    for item in raw:
        if isinstance(item, list):
            group = tuple(str(value) for value in item if str(value))
            if group:
                groups.append(group)
    return tuple(groups) or FALLBACK_PARAMETER_GROUPS


def allowed_action_types(action_space: dict[str, Any]) -> set[str]:
    raw = action_space.get("action_types")
    if isinstance(raw, dict):
        return {str(key) for key in raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    return {
        "parameter_tuning",
        "local_loss",
        "aggregation",
        "regularization",
        "sampling_wrapper",
        "posthoc_rerank",
    }


def allowed_implementation_roots(action_space: dict[str, Any]) -> set[str]:
    roots = action_space.get("allowed_implementation_roots")
    if not isinstance(roots, list):
        return {"recclaw_ext/models/", "recclaw_ext/posthoc/"}
    return {str(root).replace("\\", "/") for root in roots if str(root).strip()}


def load_parameter_space(path: str | Path = ACTION_SPACE_PATH) -> dict[str, list[Any]]:
    return parameter_space_from_action_space(load_action_space(path))


def load_parameter_groups(path: str | Path = ACTION_SPACE_PATH) -> tuple[tuple[str, ...], ...]:
    return parameter_groups_from_action_space(load_action_space(path))

