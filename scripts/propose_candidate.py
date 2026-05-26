#!/usr/bin/env python3
"""Generate RecClaw candidate proposals as stable JSONL.

This script is intentionally deterministic and serves as the stable heuristic
fallback/reference path. It supports three proposal modes:

- conservative: only immediately runnable parameter/config tuning proposals
- mixed: a blend of runnable tuning and new local-code algorithm ideas
- explore: algorithmic variants and research specs that expand the search space
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import yaml

try:
    from .action_space import (
        load_action_space,
        parameter_groups_from_action_space,
        parameter_space_from_action_space,
    )
except ImportError:
    from action_space import (
        load_action_space,
        parameter_groups_from_action_space,
        parameter_space_from_action_space,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "candidate_proposal_schema.yaml"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "notes" / "experiment_log.md"
MEMORY_PATH = PROJECT_ROOT / "results" / "agent_memory.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "candidate_proposals.jsonl"
ACTION_SPACE_PATH = PROJECT_ROOT / "configs" / "action_space.yaml"

DEFAULT_VALIDATION_SEEDS = [2026, 2027, 2028]
ACTION_SPACE = load_action_space(ACTION_SPACE_PATH)
DEFAULT_PARAMETER_SPACE: dict[str, list[Any]] = parameter_space_from_action_space(ACTION_SPACE)
DEFAULT_PARAMETER_GROUPS: tuple[tuple[str, ...], ...] = parameter_groups_from_action_space(ACTION_SPACE)

SIGNATURE_EXCLUDED_KEYS = {"seed", "reproducibility", "checkpoint_dir"}


def normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_signature_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_signature_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_signature_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

ALGORITHM_TEMPLATES: list[dict[str, Any]] = [
    {
        "proposal_type": "algorithmic_variant",
        "candidate_stub": "cand_bpr_hard_negative_margin",
        "parent_candidate_id": "cand_bpr_hard_negative_mix",
        "base_model": "BPR",
        "category": "Bias & Sample Construction",
        "action_type": "pairwise_loss",
        "hypothesis": (
            "Combining hard-negative sampling with a small pairwise margin may strengthen the "
            "early BPR ranking signal without changing the evaluation protocol."
        ),
        "runnable_level": "code_required",
        "runner_type": "model",
        "consumes": ["hard_negative_ratio", "margin"],
        "new_parameters": [
            {"name": "margin", "default": 0.2, "search_space": [0.1, 0.2, 0.5]},
        ],
        "implementation_plan": {
            "summary": "Reuse the local hard-negative sampler and margin loss helper in a composed BPR subclass.",
            "entrypoint": "recclaw_ext.models.bpr_composed:BPRHardNegativeMargin",
            "files": [
                "recclaw_ext/models/bpr_composed.py",
            ],
        },
        "allowed_files": ["recclaw_ext/models/"],
        "expected_effect": {
            "primary_metric": "ndcg@10",
            "direction": "increase",
            "rationale": "Historical first-50 results favored a hard-negative plus margin family.",
        },
        "risk": {
            "quality": "Hard negatives and margin can over-sharpen ranking if both are too strong.",
            "runtime": "Small overhead from replacement negative sampling.",
            "implementation": "Uses an existing local template class when available.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "ndcg@10 improves over the comparable BPR candidate and precision@10 does not drop sharply.",
            "revise_if": "ranking improves only for lower margin or lower hard_negative_ratio.",
            "discard_if": "candidate crashes or ranking metrics regress under comparable BPR budget.",
        },
        "evaluation_plan": {
            "primary_metric": "ndcg@10",
            "validation_seeds": DEFAULT_VALIDATION_SEEDS,
            "aggregation": "report mean and std over validation_seeds before claiming improvement",
        },
    },
    {
        "proposal_type": "algorithmic_variant",
        "candidate_stub": "cand_bpr_popularity_aware_margin",
        "parent_candidate_id": "cand_bpr_popularity_aware_negative",
        "base_model": "BPR",
        "category": "Bias & Sample Construction",
        "action_type": "pairwise_loss",
        "hypothesis": (
            "Popularity-aware negatives with a small pairwise margin may improve exposure-robust "
            "ranking without introducing post-hoc reranking."
        ),
        "runnable_level": "code_required",
        "runner_type": "model",
        "consumes": ["popularity_alpha", "margin"],
        "new_parameters": [
            {"name": "margin", "default": 0.2, "search_space": [0.1, 0.2, 0.5]},
        ],
        "implementation_plan": {
            "summary": "Reuse the local popularity-aware sampler and margin loss helper in a composed BPR subclass.",
            "entrypoint": "recclaw_ext.models.bpr_composed:BPRPopularityAwareMargin",
            "files": [
                "recclaw_ext/models/bpr_composed.py",
            ],
        },
        "allowed_files": ["recclaw_ext/models/"],
        "expected_effect": {
            "primary_metric": "ndcg@10",
            "direction": "increase",
            "rationale": "This tests whether exposure-aware sampling benefits from a clearer pairwise boundary.",
        },
        "risk": {
            "quality": "May over-penalize popular negatives and reduce precision@10.",
            "runtime": "Small overhead from popularity-shaped replacement sampling.",
            "implementation": "Uses an existing local template class when available.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "ndcg@10 improves and precision@10 does not drop sharply.",
            "revise_if": "coverage improves but ranking metrics are flat.",
            "discard_if": "candidate crashes or ranking metrics regress under comparable BPR budget.",
        },
        "evaluation_plan": {
            "primary_metric": "ndcg@10",
            "validation_seeds": DEFAULT_VALIDATION_SEEDS,
            "aggregation": "report mean and std over validation_seeds before claiming improvement",
        },
    },
    {
        "proposal_type": "algorithmic_variant",
        "candidate_stub": "cand_lightgcn_residual_norm_constrained",
        "parent_candidate_id": "cand_lightgcn_residual_layer_mix",
        "base_model": "LightGCN",
        "category": "Representation & Interaction",
        "action_type": "regularization",
        "hypothesis": (
            "Residual layer mixing plus soft norm control may reproduce the strongest historical "
            "LightGCNResidualNorm family and stabilize late-stage search."
        ),
        "runnable_level": "code_required",
        "runner_type": "model",
        "consumes": ["embedding_size", "n_layers", "residual_weight", "lambda_norm", "max_norm"],
        "new_parameters": [
            {"name": "lambda_norm", "default": 0.0001, "search_space": [1e-5, 1e-4, 1e-3]},
            {"name": "max_norm", "default": 1.0, "search_space": [0.5, 1.0, 2.0]},
        ],
        "implementation_plan": {
            "summary": "Reuse the local residual LightGCN path and add a soft propagated-embedding norm penalty.",
            "entrypoint": "recclaw_ext.models.lightgcn_residual_norm:LightGCNResidualNormConstrained",
            "files": [
                "recclaw_ext/models/lightgcn_residual_norm.py",
            ],
        },
        "allowed_files": ["recclaw_ext/models/"],
        "expected_effect": {
            "primary_metric": "ndcg@10",
            "direction": "increase",
            "rationale": "The historical best family used residual LightGCN with norm control.",
        },
        "risk": {
            "quality": "Overly tight norm constraints may suppress useful graph signal.",
            "runtime": "Small training overhead from one extra norm penalty.",
            "implementation": "Uses an existing local template class when available.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "ndcg@10 improves over comparable residual LightGCN candidates.",
            "revise_if": "result is near baseline but suggests lower lambda_norm or looser max_norm.",
            "discard_if": "norm control suppresses ranking quality or training becomes unstable.",
        },
        "evaluation_plan": {
            "primary_metric": "ndcg@10",
            "validation_seeds": DEFAULT_VALIDATION_SEEDS,
            "aggregation": "report mean and std over validation_seeds before claiming improvement",
        },
    },
    {
        "proposal_type": "algorithmic_variant",
        "candidate_stub": "cand_lightgcn_edge_dropout_residual_norm",
        "parent_candidate_id": "cand_lightgcn_edge_dropout_residual_mix",
        "base_model": "LightGCN",
        "category": "Representation & Interaction",
        "action_type": "graph_augmentation",
        "hypothesis": (
            "Edge-dropout residual propagation plus soft norm control may combine structural "
            "regularization with the historical residual-norm advantage."
        ),
        "runnable_level": "code_required",
        "runner_type": "model",
        "consumes": [
            "embedding_size",
            "n_layers",
            "residual_weight",
            "edge_dropout",
            "lambda_norm",
            "max_norm",
        ],
        "new_parameters": [
            {"name": "lambda_norm", "default": 0.0001, "search_space": [1e-5, 1e-4, 1e-3]},
            {"name": "max_norm", "default": 1.0, "search_space": [0.5, 1.0, 2.0]},
        ],
        "implementation_plan": {
            "summary": "Reuse the edge-dropout residual LightGCN path with the residual-norm loss mixin.",
            "entrypoint": "recclaw_ext.models.lightgcn_residual_norm:LightGCNEdgeDropoutResidualNorm",
            "files": [
                "recclaw_ext/models/lightgcn_residual_norm.py",
            ],
        },
        "allowed_files": ["recclaw_ext/models/"],
        "expected_effect": {
            "primary_metric": "ndcg@10",
            "direction": "increase",
            "rationale": "This is the strongest safe composition now supported by local LightGCN extensions.",
        },
        "risk": {
            "quality": "Dropout and norm control can over-regularize if both are too strong.",
            "runtime": "Small training overhead from sparse edge dropout and norm penalty.",
            "implementation": "Uses an existing local template class when available.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "ndcg@10 improves over residual-norm and edge-dropout parents.",
            "revise_if": "only one regularizer appears beneficial.",
            "discard_if": "combined regularization lowers ranking quality or destabilizes training.",
        },
        "evaluation_plan": {
            "primary_metric": "ndcg@10",
            "validation_seeds": DEFAULT_VALIDATION_SEEDS,
            "aggregation": "report mean and std over validation_seeds before claiming improvement",
        },
    },
    {
        "proposal_type": "research_spec",
        "candidate_stub": "cand_posthoc_popularity_coverage_pareto_rerank",
        "parent_candidate_id": "cand_rerank_coverage_boost",
        "base_model": "LightGCN",
        "category": "Result Distribution Quality",
        "action_type": "posthoc_rerank",
        "hypothesis": (
            "A Pareto-style posthoc reranker may improve coverage and popularity balance with "
            "a bounded NDCG tradeoff."
        ),
        "runnable_level": "spec_only",
        "runner_type": "posthoc",
        "consumes": ["lambda_coverage", "lambda_pop", "pareto_temperature"],
        "new_parameters": [
            {"name": "lambda_pop", "default": 0.001, "search_space": [1e-4, 1e-3, 1e-2]},
            {"name": "pareto_temperature", "default": 0.5, "search_space": [0.2, 0.5, 1.0]},
        ],
        "implementation_plan": {
            "summary": "Define a trained-score adjustment flow after run_candidate supports posthoc execution.",
            "entrypoint": "recclaw_ext.posthoc.adjustments:ParetoPopularityCoverageReranker",
            "files": [
                "recclaw_ext/posthoc/adjustments.py",
            ],
        },
        "allowed_files": ["recclaw_ext/posthoc/"],
        "expected_effect": {
            "primary_metric": "itemcoverage@10",
            "direction": "increase",
            "rationale": "Explicit Pareto scoring can explore quality-diversity tradeoffs after model scoring.",
        },
        "risk": {
            "quality": "May reduce ndcg@10 if reranking pressure is too strong.",
            "runtime": "Inference-time reranking adds scoring overhead.",
            "implementation": "Blocked until the posthoc runner path is implemented.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "coverage improves with bounded ndcg@10 loss.",
            "revise_if": "coverage improves but precision loss is too large.",
            "discard_if": "reranking cannot preserve baseline ranking quality.",
        },
        "evaluation_plan": {
            "primary_metric": "itemcoverage@10",
            "validation_seeds": DEFAULT_VALIDATION_SEEDS,
            "aggregation": "report mean and std over validation_seeds before claiming improvement",
        },
    },
]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit > 0:
        lines = lines[-limit:]
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def normalize_base_model(raw: Any) -> str:
    text = str(raw or "")
    if "LightGCN" in text:
        return "LightGCN"
    if "BPR" in text:
        return "BPR"
    return text


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "candidate"


def params_signature(params: dict[str, Any]) -> str:
    normalized = {
        str(key): normalize_signature_value(value)
        for key, value in params.items()
        if str(key) not in SIGNATURE_EXCLUDED_KEYS
    }
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def parent_param_signature(parent_id: str, params: dict[str, Any]) -> str:
    return f"{parent_id}::{params_signature(params)}"


def extract_parent_and_params(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    parent_id = str(row.get("parent_candidate_id") or row.get("candidate_id") or "")
    params = row.get("parameter_overrides")
    if not isinstance(params, dict):
        params = row.get("params")
    if not parent_id or not isinstance(params, dict) or not params:
        return None
    return parent_id, params


def used_parent_param_signatures(memory: list[dict[str, Any]]) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for row in memory:
        extracted = extract_parent_and_params(row)
        if extracted is None:
            continue
        parent_id, params = extracted
        used.setdefault(parent_id, set()).add(params_signature(params))
    return used


def merge_used_signatures(*sources: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for source in sources:
        for parent_id, signatures in source.items():
            merged.setdefault(parent_id, set()).update(signatures)
    return merged


def parameter_groups_for_parent(parent: dict[str, Any]) -> list[tuple[str, ...]]:
    consumes = {str(item) for item in parent.get("consumes") or []}
    groups: list[tuple[str, ...]] = []
    covered: set[str] = set()
    for group in DEFAULT_PARAMETER_GROUPS:
        if all(key in consumes and DEFAULT_PARAMETER_SPACE.get(key) for key in group):
            groups.append(group)
            covered.update(group)
    for key in sorted(consumes - covered):
        if DEFAULT_PARAMETER_SPACE.get(key):
            groups.append((key,))
    return groups


def iter_param_overrides(group: tuple[str, ...]) -> list[dict[str, Any]]:
    value_lists = [DEFAULT_PARAMETER_SPACE.get(key) or [] for key in group]
    if any(not values for values in value_lists):
        return []
    return [dict(zip(group, values)) for values in product(*value_lists)]


def choose_param_overrides(
    parent: dict[str, Any],
    used_signatures: dict[str, set[str]],
) -> dict[str, Any] | None:
    parent_id = str(parent.get("candidate_id") or "")
    already_used = used_signatures.get(parent_id, set())
    for group in parameter_groups_for_parent(parent):
        for params in iter_param_overrides(group):
            if params_signature(params) not in already_used:
                return params
    return None


def default_evaluation_plan(primary_metric: str = "ndcg@10") -> dict[str, Any]:
    return {
        "primary_metric": primary_metric,
        "primary_seed": DEFAULT_VALIDATION_SEEDS[0],
        "validation_seeds": DEFAULT_VALIDATION_SEEDS,
        "aggregation": "report mean and std over validation_seeds before claiming improvement",
        "promote_if": (
            f"mean {primary_metric} improves over the comparable baseline and the result "
            "is not explained by a single lucky seed"
        ),
    }


def override_slug(params: dict[str, Any]) -> str:
    chunks = []
    for key, value in sorted(params.items()):
        chunks.append(f"{slugify(str(key))}_{slugify(str(value))}")
    return "_".join(chunks) or "params"


def describe_overrides(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def primary_override_key(params: dict[str, Any]) -> str:
    if "residual_weight" in params:
        return "residual_weight"
    if "margin" in params:
        return "margin"
    for key in sorted(params):
        if key not in {"embedding_size", "n_layers"}:
            return key
    return sorted(params)[0] if params else "parameter"


def primary_metric_for_param(param_name: str) -> str:
    return "ndcg@10"


def rationale_for_param(parent: dict[str, Any], param_name: str) -> str:
    if param_name in {"embedding_size", "n_layers", "learning_rate", "reg_weight"}:
        return "This is a RecBole-native knob already consumed by the wired parent."
    return "The parent candidate already exposes this local parameter through consumes."


def parent_signature_set(
    rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for row in rows:
        extracted = extract_parent_and_params(row)
        if extracted is None:
            continue
        parent_id, params = extracted
        used.setdefault(parent_id, set()).add(params_signature(params))
    return used


def score_parent(parent: dict[str, Any], memory: list[dict[str, Any]], experiment_log: str) -> float:
    parent_id = str(parent.get("candidate_id") or "")
    score = 0.0
    if parent.get("wired"):
        score += 5.0
    if str(parent.get("status") or "") == "implemented":
        score += 2.0
    if str(parent.get("runner_type") or "") in {"config_only", "model"}:
        score += 1.0
    if parent.get("consumes"):
        score += 1.0
    if parent_id and parent_id in experiment_log:
        score += 0.5
    recent_runs = [row for row in memory if row.get("candidate_id") == parent_id]
    score -= min(len(recent_runs), 5) * 0.25
    if recent_runs and str(recent_runs[-1].get("decision") or "") in {"crash", "discard"}:
        score -= 0.5
    return score


def build_parameter_proposal(
    parent: dict[str, Any],
    parameter_overrides: dict[str, Any],
    sequence: int,
    stamp: str,
) -> dict[str, Any]:
    parent_id = str(parent.get("candidate_id") or "")
    base_model = normalize_base_model(parent.get("base_model"))
    param_name = primary_override_key(parameter_overrides)
    candidate_id = f"{parent_id}_proposal_{override_slug(parameter_overrides)}_{stamp}_{sequence:02d}"
    primary_metric = primary_metric_for_param(param_name)
    rationale = rationale_for_param(parent, param_name)
    override_text = describe_overrides(parameter_overrides)
    return {
        "proposal_type": "tuning",
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_id,
        "base_model": base_model,
        "category": parent.get("category") or "Uncategorized",
        "action_type": "parameter_tuning",
        "hypothesis": (
            f"Tuning {override_text} may improve {primary_metric} "
            f"without changing the executable candidate path."
        ),
        "runnable_level": "parameter_only",
        "runner_type": parent.get("runner_type") or "config_only",
        "consumes": sorted(parameter_overrides),
        "parameter_overrides": parameter_overrides,
        "parameter_signature": parent_param_signature(parent_id, parameter_overrides),
        "expected_effect": {
            "primary_metric": primary_metric,
            "direction": "increase",
            "rationale": rationale,
        },
        "risk": {
            "quality": "Could overfit this single parameter or trade off secondary metrics.",
            "runtime": "No material runtime change expected unless the parameter changes model size.",
            "implementation": "No code change required.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "Primary metric improves over a comparable baseline and no major secondary metric regresses.",
            "revise_if": "Primary metric is near baseline but secondary metrics suggest a nearby value may help.",
            "discard_if": "Run crashes or primary metric drops below the comparable baseline.",
        },
        "evaluation_plan": default_evaluation_plan(primary_metric),
        "promotion_requirements": [
            "validator status is accepted",
            "run through scripts/run_candidate.py with parameter_overrides",
            "repeat accepted improvements over evaluation_plan.validation_seeds before claiming a stable gain",
            "record result in results/agent_memory.jsonl",
        ],
    }


def generate_tuning_proposals(
    *,
    registry: list[dict[str, Any]],
    experiment_log: str,
    memory: list[dict[str, Any]],
    count: int,
    stamp: str,
    proposal_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    used_signatures = merge_used_signatures(
        used_parent_param_signatures(memory),
        parent_signature_set(proposal_history or []),
    )
    parents = [
        item
        for item in registry
        if bool(item.get("wired"))
        and str(item.get("runner_type") or "") in {"config_only", "model"}
        and normalize_base_model(item.get("base_model")) in {"BPR", "LightGCN"}
        and item.get("consumes")
    ]
    parents.sort(key=lambda item: score_parent(item, memory, experiment_log), reverse=True)

    proposals: list[dict[str, Any]] = []
    for parent in parents:
        picked = choose_param_overrides(parent, used_signatures)
        if picked is None:
            continue
        parent_id = str(parent.get("candidate_id") or "")
        used_signatures.setdefault(parent_id, set()).add(params_signature(picked))
        proposals.append(build_parameter_proposal(parent, picked, len(proposals) + 1, stamp))
        if len(proposals) >= count:
            break
    return proposals


def existing_candidate_ids(registry: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("candidate_id")) for item in registry if item.get("candidate_id")}


def wired_candidate_ids(registry: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("candidate_id"))
        for item in registry
        if item.get("candidate_id")
        and bool(item.get("wired"))
        and str(item.get("status") or "") == "implemented"
    }


def template_parent_exists(template: dict[str, Any], registry_ids: set[str]) -> bool:
    return str(template.get("parent_candidate_id") or "") in registry_ids


def build_algorithmic_proposal(template: dict[str, Any], sequence: int, stamp: str) -> dict[str, Any]:
    proposal = dict(template)
    stub = str(proposal.pop("candidate_stub"))
    proposal["candidate_id"] = f"{stub}_{stamp}_{sequence:02d}"
    proposal.setdefault("promotion_requirements", [])
    proposal["promotion_requirements"] = [
        *proposal["promotion_requirements"],
        "validate proposal before implementation",
        "implement only under allowed_files",
        "add candidate config and registry entry with wired=false first",
        "set wired=true only after dry-run/import check passes",
    ]
    return proposal


def generate_algorithmic_proposals(
    *,
    registry: list[dict[str, Any]],
    count: int,
    stamp: str,
    include_spec_only: bool = False,
) -> list[dict[str, Any]]:
    registry_ids = existing_candidate_ids(registry)
    wired_ids = wired_candidate_ids(registry)
    proposals: list[dict[str, Any]] = []
    for template in ALGORITHM_TEMPLATES:
        if str(template.get("runnable_level") or "") == "spec_only" and not include_spec_only:
            continue
        if str(template.get("candidate_stub") or "") in wired_ids:
            continue
        if not template_parent_exists(template, registry_ids):
            continue
        proposals.append(build_algorithmic_proposal(template, len(proposals) + 1, stamp))
        if len(proposals) >= count:
            break
    return proposals


def generate_proposals(
    *,
    registry_path: Path,
    experiment_log_path: Path,
    memory_path: Path,
    count: int,
    memory_limit: int,
    mode: str,
    proposal_history_path: Path | None = None,
) -> list[dict[str, Any]]:
    registry = load_yaml(registry_path).get("candidates", [])
    if not isinstance(registry, list):
        raise ValueError(f"registry candidates must be a list: {registry_path}")
    experiment_log = experiment_log_path.read_text(encoding="utf-8", errors="replace") if experiment_log_path.exists() else ""
    memory = load_jsonl(memory_path, memory_limit)
    proposal_history = load_jsonl(proposal_history_path, 0) if proposal_history_path else []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == "conservative":
        return generate_tuning_proposals(
            registry=registry,
            experiment_log=experiment_log,
            memory=memory,
            count=count,
            stamp=stamp,
            proposal_history=proposal_history,
        )

    if mode == "explore":
        return generate_algorithmic_proposals(registry=registry, count=count, stamp=stamp)

    tuning_count = max(1, count // 2)
    algorithmic_count = max(0, count - tuning_count)
    proposals = generate_tuning_proposals(
        registry=registry,
        experiment_log=experiment_log,
        memory=memory,
        count=tuning_count,
        stamp=stamp,
        proposal_history=proposal_history,
    )
    proposals.extend(generate_algorithmic_proposals(registry=registry, count=algorithmic_count, stamp=stamp))
    if len(proposals) < count:
        more = generate_algorithmic_proposals(registry=registry, count=count - len(proposals), stamp=stamp)
        used_ids = {item["candidate_id"] for item in proposals}
        proposals.extend(item for item in more if item["candidate_id"] not in used_ids)
    return proposals[:count]


def write_jsonl(path: Path, proposals: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for proposal in proposals:
            handle.write(json.dumps(proposal, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RecClaw candidate proposals as JSONL.")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="Path to candidate_registry.yaml")
    parser.add_argument("--experiment-log", default=str(EXPERIMENT_LOG_PATH), help="Path to experiment_log.md")
    parser.add_argument("--memory", default=str(MEMORY_PATH), help="Path to agent_memory.jsonl")
    parser.add_argument("--schema", default=str(SCHEMA_PATH), help="Path to candidate_proposal_schema.yaml")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSONL path")
    parser.add_argument("--count", type=int, default=4, help="Number of proposals to emit")
    parser.add_argument(
        "--mode",
        choices=("conservative", "mixed", "explore"),
        default="mixed",
        help="Proposal generation mode",
    )
    parser.add_argument("--memory-limit", type=int, default=2000, help="Read only latest N memory rows; 0 means all")
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwriting it")
    args = parser.parse_args()

    schema = load_yaml(Path(args.schema))
    if not schema.get("required_fields"):
        raise ValueError(f"proposal schema is missing required_fields: {args.schema}")

    proposals = generate_proposals(
        registry_path=Path(args.registry),
        experiment_log_path=Path(args.experiment_log),
        memory_path=Path(args.memory),
        count=max(1, args.count),
        memory_limit=max(0, args.memory_limit),
        mode=args.mode,
        proposal_history_path=Path(args.output),
    )
    write_jsonl(Path(args.output), proposals, args.append)
    print(
        json.dumps(
            {
                "output": args.output,
                "mode": args.mode,
                "proposal_count": len(proposals),
                "by_status_expectation": {
                    "runnable_tuning": sum(1 for item in proposals if item.get("proposal_type") == "tuning"),
                    "implementation_queue": sum(
                        1 for item in proposals if item.get("proposal_type") != "tuning"
                    ),
                },
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
