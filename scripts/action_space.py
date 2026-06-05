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


def parameter_metadata(action_space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return full metadata for declared parameters."""
    raw = action_space.get("parameter_space")
    if not isinstance(raw, dict):
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    for key, spec in raw.items():
        if isinstance(spec, dict):
            metadata[str(key)] = dict(spec)
    return metadata


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def base_models(action_space: dict[str, Any]) -> list[str]:
    models = _string_list(action_space.get("base_models"))
    return models or ["BPR", "LightGCN"]


def compatible_models_for_parameter(param_name: str, action_space: dict[str, Any]) -> list[str]:
    """Return models a parameter supports; unspecified means all base models."""
    meta = parameter_metadata(action_space).get(str(param_name), {})
    explicit = meta.get("compatible_models", meta.get("model_compatibility"))
    models = _string_list(explicit)
    return models or base_models(action_space)


def is_parameter_compatible_with_model(param_name: str, model: str, action_space: dict[str, Any]) -> bool:
    if not str(model).strip():
        return True
    metadata = parameter_metadata(action_space)
    if str(param_name) not in metadata:
        return True
    return str(model) in compatible_models_for_parameter(str(param_name), action_space)


def conditional_validity_rules(action_space: dict[str, Any]) -> list[dict[str, Any]]:
    raw = action_space.get("conditional_validity_rules")
    rules: list[dict[str, Any]] = []
    if isinstance(raw, list):
        rules.extend(item for item in raw if isinstance(item, dict))
    for parameter, meta in parameter_metadata(action_space).items():
        rule = meta.get("conditional_validity")
        if isinstance(rule, dict):
            candidate_rule = {"parameter": parameter, **rule}
            if not any(
                str(existing.get("parameter") or "") == parameter
                and str(existing.get("requires") or "") == str(candidate_rule.get("requires") or "")
                for existing in rules
            ):
                rules.append(candidate_rule)
    return rules


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_condition(value: Any, condition: Any) -> bool:
    text = str(condition).strip()
    if not text:
        return True
    for op in (">=", "<=", "!=", "==", ">", "<"):
        if not text.startswith(op):
            continue
        expected = text[len(op) :].strip()
        numeric_value = _numeric_value(value)
        numeric_expected = _numeric_value(expected)
        if numeric_value is not None and numeric_expected is not None:
            if op == ">=":
                return numeric_value >= numeric_expected
            if op == "<=":
                return numeric_value <= numeric_expected
            if op == "!=":
                return numeric_value != numeric_expected
            if op == "==":
                return numeric_value == numeric_expected
            if op == ">":
                return numeric_value > numeric_expected
            if op == "<":
                return numeric_value < numeric_expected
        actual = str(value).strip()
        if op == "!=":
            return actual != expected
        if op == "==":
            return actual == expected
        return True
    numeric_value = _numeric_value(value)
    numeric_expected = _numeric_value(text)
    if numeric_value is not None and numeric_expected is not None:
        return numeric_value == numeric_expected
    return str(value).strip() == text


def validate_parameter_conditions(
    param_name: str,
    param_value: Any,
    all_params: dict[str, Any],
    action_space: dict[str, Any],
    *,
    strict_missing: bool = False,
) -> list[str]:
    """Validate one parameter against action_space conditional rules.

    Missing dependency parameters are ignored by default because tuning proposals
    often override only one knob while inheriting the rest from a parent config.
    Set strict_missing=True for callers that have the complete config context.
    """
    del param_value
    violations: list[str] = []
    for rule in conditional_validity_rules(action_space):
        if str(rule.get("parameter") or "") != str(param_name):
            continue
        base_model = str(all_params.get("base_model") or "").strip()
        incompatible_models = set(_string_list(rule.get("incompatible_with_models")))
        if base_model and base_model in incompatible_models:
            reason = str(rule.get("reason") or "").strip()
            message = f"{param_name} is incompatible with base_model {base_model}"
            violations.append(f"{message} ({reason})" if reason else message)
        requires = rule.get("requires")
        if not isinstance(requires, dict):
            continue
        for required_name, condition in requires.items():
            required_key = str(required_name)
            if required_key not in all_params or all_params.get(required_key) in (None, ""):
                if strict_missing:
                    violations.append(f"{param_name} requires {required_key} {condition}, but it is unset")
                continue
            if not _parse_condition(all_params.get(required_key), condition):
                reason = str(rule.get("reason") or "").strip()
                message = f"{param_name} requires {required_key} {condition}, got {all_params.get(required_key)}"
                violations.append(f"{message} ({reason})" if reason else message)
    return violations


def validate_proposal_parameter_compatibility(
    params: dict[str, Any],
    base_model_name: str,
    action_space: dict[str, Any],
    *,
    strict_missing: bool = False,
) -> list[str]:
    context = dict(params)
    context["base_model"] = base_model_name
    violations: list[str] = []
    for param_name, param_value in params.items():
        key = str(param_name)
        if not is_parameter_compatible_with_model(key, base_model_name, action_space):
            allowed = compatible_models_for_parameter(key, action_space)
            violations.append(f"{key} is not compatible with {base_model_name} (allowed: {allowed})")
        violations.extend(
            validate_parameter_conditions(
                key,
                param_value,
                context,
                action_space,
                strict_missing=strict_missing,
            )
        )
    return violations


def semantic_role_for_parameter(param_name: str, action_space: dict[str, Any]) -> str:
    meta = parameter_metadata(action_space).get(str(param_name), {})
    return str(meta.get("semantic_role") or "")


def typical_effect_for_parameter(param_name: str, action_space: dict[str, Any]) -> str:
    meta = parameter_metadata(action_space).get(str(param_name), {})
    return str(meta.get("typical_effect") or "")


def action_types_for_parameter(param_name: str, action_space: dict[str, Any]) -> list[str]:
    meta = parameter_metadata(action_space).get(str(param_name), {})
    return _string_list(meta.get("action_types"))


def parameter_schema_type(param_name: str, action_space: dict[str, Any]) -> list[str]:
    """Return JSON-schema type list for a parameter based on its declared values."""
    meta = parameter_metadata(action_space).get(str(param_name), {})
    values = meta.get("values", [])
    if not isinstance(values, list):
        return ["number", "null"]
    has_string = any(isinstance(v, str) for v in values)
    has_number = any(isinstance(v, (int, float)) for v in values)
    if has_string and has_number:
        return ["string", "number", "null"]
    if has_string:
        return ["string", "null"]
    return ["number", "null"]


def parameter_schema_map(action_space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a JSON-schema properties map for all declared parameters."""
    return {
        key: {"type": parameter_schema_type(key, action_space)}
        for key in parameter_space_from_action_space(action_space)
    }


def method_space_projection(action_space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = action_space.get("method_space_projection")
    return raw if isinstance(raw, dict) else {}


def load_parameter_space(path: str | Path = ACTION_SPACE_PATH) -> dict[str, list[Any]]:
    return parameter_space_from_action_space(load_action_space(path))


def load_parameter_groups(path: str | Path = ACTION_SPACE_PATH) -> tuple[tuple[str, ...], ...]:
    return parameter_groups_from_action_space(load_action_space(path))
