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
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "candidate_proposal_schema.yaml"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "notes" / "experiment_log.md"
MEMORY_PATH = PROJECT_ROOT / "results" / "agent_memory.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "candidate_proposals.jsonl"

DEFAULT_PARAMETER_SPACE: dict[str, list[Any]] = {
    "embedding_size": [32, 64, 128],
    "learning_rate": [0.0001, 0.001, 0.005],
    "n_layers": [1, 2, 3],
    "reg_weight": [1e-6, 1e-5, 1e-4],
    "margin": [0.1, 0.2, 0.5],
    "residual_weight": [0.05, 0.1, 0.2, 0.3],
    "tail_weight_alpha": [0.2, 0.5, 1.0],
}

ALGORITHM_TEMPLATES: list[dict[str, Any]] = [
    {
        "proposal_type": "algorithmic_variant",
        "candidate_stub": "cand_bpr_adaptive_popularity_margin",
        "parent_candidate_id": "cand_bpr_margin_loss",
        "base_model": "BPR",
        "category": "Objective & Optimization",
        "hypothesis": (
            "Adaptive margins based on negative-item popularity may reduce head-item dominance "
            "while preserving pairwise ranking strength."
        ),
        "runnable_level": "code_required",
        "runner_type": "model",
        "consumes": ["margin", "popularity_alpha"],
        "new_parameters": [
            {"name": "popularity_alpha", "default": 0.3, "search_space": [0.1, 0.3, 0.5]},
        ],
        "implementation_plan": {
            "summary": "Add a local BPR subclass that scales pairwise margin by negative-item popularity.",
            "entrypoint": "recclaw_ext.models:BPRAdaptivePopularityMargin",
            "files": [
                "recclaw_ext/models/bpr_adaptive_popularity_margin.py",
                "recclaw_ext/models/__init__.py",
                "configs/candidates/cand_bpr_adaptive_popularity_margin.yaml",
                "configs/candidate_registry.yaml",
            ],
        },
        "allowed_files": ["recclaw_ext/models/", "configs/candidates/", "configs/candidate_registry.yaml"],
        "expected_effect": {
            "primary_metric": "ndcg@10",
            "direction": "increase",
            "rationale": "Combines BPR's margin signal with popularity-aware pressure against head dominance.",
        },
        "risk": {
            "quality": "May over-penalize popular negatives and reduce precision@10.",
            "runtime": "Small overhead from item popularity lookup.",
            "implementation": "Requires local model code and registry wiring.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "ndcg@10 improves and precision@10 does not drop sharply.",
            "revise_if": "coverage improves but ranking metrics are flat.",
            "discard_if": "candidate crashes or ranking metrics regress under comparable BPR budget.",
        },
    },
    {
        "proposal_type": "algorithmic_variant",
        "candidate_stub": "cand_lightgcn_edge_dropout_residual_mix",
        "parent_candidate_id": "cand_lightgcn_residual_layer_mix",
        "base_model": "LightGCN",
        "category": "Representation & Interaction",
        "hypothesis": (
            "Applying light edge dropout before residual layer mixing may reduce over-smoothing "
            "and improve top-k generalization."
        ),
        "runnable_level": "code_required",
        "runner_type": "model",
        "consumes": ["embedding_size", "n_layers", "residual_weight", "edge_dropout"],
        "new_parameters": [
            {"name": "edge_dropout", "default": 0.1, "search_space": [0.05, 0.1, 0.2]},
        ],
        "implementation_plan": {
            "summary": "Extend the local residual LightGCN path with sparse edge dropout during training.",
            "entrypoint": "recclaw_ext.models:LightGCNEdgeDropoutResidualMix",
            "files": [
                "recclaw_ext/models/lightgcn_edge_dropout_residual.py",
                "recclaw_ext/models/__init__.py",
                "configs/candidates/cand_lightgcn_edge_dropout_residual_mix.yaml",
                "configs/candidate_registry.yaml",
            ],
        },
        "allowed_files": ["recclaw_ext/models/", "configs/candidates/", "configs/candidate_registry.yaml"],
        "expected_effect": {
            "primary_metric": "ndcg@10",
            "direction": "increase",
            "rationale": "Graph perturbation can regularize LightGCN without changing RecBole core.",
        },
        "risk": {
            "quality": "Too much dropout may damage high-order collaborative signal.",
            "runtime": "Small training overhead from sampled adjacency masking.",
            "implementation": "Needs careful local handling of sparse graph tensors.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "ndcg@10 and recall@10 improve without latency regression at inference.",
            "revise_if": "ranking improves only at low dropout values.",
            "discard_if": "training becomes unstable or graph propagation metrics regress.",
        },
    },
    {
        "proposal_type": "algorithmic_variant",
        "candidate_stub": "cand_lightgcn_contrastive_layer_alignment",
        "parent_candidate_id": "cand_lightgcn_layer_weighted_agg",
        "base_model": "LightGCN",
        "category": "Objective & Optimization",
        "hypothesis": (
            "A small contrastive alignment term between shallow and final embeddings may improve "
            "layer consistency without replacing LightGCN propagation."
        ),
        "runnable_level": "code_required",
        "runner_type": "model",
        "consumes": ["embedding_size", "n_layers", "lambda_align", "cl_temperature"],
        "new_parameters": [
            {"name": "lambda_align", "default": 0.01, "search_space": [0.005, 0.01, 0.03]},
            {"name": "cl_temperature", "default": 0.2, "search_space": [0.1, 0.2, 0.5]},
        ],
        "implementation_plan": {
            "summary": "Add a local LightGCN subclass that augments BPR loss with layer-level contrastive alignment.",
            "entrypoint": "recclaw_ext.models:LightGCNContrastiveLayerAlignment",
            "files": [
                "recclaw_ext/models/lightgcn_contrastive_alignment.py",
                "recclaw_ext/models/_losses.py",
                "recclaw_ext/models/__init__.py",
                "configs/candidates/cand_lightgcn_contrastive_layer_alignment.yaml",
                "configs/candidate_registry.yaml",
            ],
        },
        "allowed_files": ["recclaw_ext/models/", "configs/candidates/", "configs/candidate_registry.yaml"],
        "expected_effect": {
            "primary_metric": "ndcg@10",
            "direction": "increase",
            "rationale": "Layer alignment may improve representation stability for top-k ranking.",
        },
        "risk": {
            "quality": "Auxiliary loss can conflict with pairwise ranking if overweighted.",
            "runtime": "Moderate training overhead from contrastive similarity computation.",
            "implementation": "Requires local loss helper and model subclass only.",
            "recbole_core_change_required": False,
        },
        "decision_rule": {
            "keep_if": "ndcg@10 improves and training runtime remains acceptable.",
            "revise_if": "recall improves while precision drops mildly.",
            "discard_if": "auxiliary loss destabilizes training.",
        },
    },
    {
        "proposal_type": "research_spec",
        "candidate_stub": "cand_posthoc_popularity_coverage_pareto_rerank",
        "parent_candidate_id": "cand_rerank_coverage_boost",
        "base_model": "LightGCN",
        "category": "Result Distribution Quality",
        "hypothesis": (
            "A Pareto-style posthoc reranker may improve coverage and popularity balance with "
            "a bounded NDCG tradeoff."
        ),
        "runnable_level": "spec_only",
        "runner_type": "posthoc",
        "consumes": ["lambda_coverage", "lambda_pop", "pareto_temperature"],
        "new_parameters": [
            {"name": "lambda_pop", "default": 0.1, "search_space": [0.05, 0.1, 0.2]},
            {"name": "pareto_temperature", "default": 0.5, "search_space": [0.2, 0.5, 1.0]},
        ],
        "implementation_plan": {
            "summary": "Define a trained-score adjustment flow after run_candidate supports posthoc execution.",
            "entrypoint": "recclaw_ext.posthoc:ParetoPopularityCoverageReranker",
            "files": [
                "recclaw_ext/posthoc/adjustments.py",
                "recclaw_ext/posthoc/__init__.py",
                "configs/candidates/cand_posthoc_popularity_coverage_pareto_rerank.yaml",
                "configs/candidate_registry.yaml",
            ],
        },
        "allowed_files": ["recclaw_ext/posthoc/", "configs/candidates/", "configs/candidate_registry.yaml"],
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
    return json.dumps(params, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def used_parent_param_signatures(memory: list[dict[str, Any]]) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for row in memory:
        candidate_id = str(row.get("candidate_id") or "")
        params = row.get("params")
        if candidate_id and isinstance(params, dict):
            used.setdefault(candidate_id, set()).add(params_signature(params))
    return used


def choose_param_override(
    parent: dict[str, Any],
    used_signatures: dict[str, set[str]],
) -> tuple[str, Any] | None:
    parent_id = str(parent.get("candidate_id") or "")
    for key in parent.get("consumes") or []:
        key = str(key)
        values = DEFAULT_PARAMETER_SPACE.get(key) or []
        if not values:
            continue
        already_used = used_signatures.get(parent_id, set())
        for value in values:
            if params_signature({key: value}) not in already_used:
                return key, value
        return key, values[0]
    return None


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
    param_name: str,
    param_value: Any,
    sequence: int,
    stamp: str,
) -> dict[str, Any]:
    parent_id = str(parent.get("candidate_id") or "")
    base_model = normalize_base_model(parent.get("base_model"))
    candidate_id = f"{parent_id}_proposal_{slugify(param_name)}_{stamp}_{sequence:02d}"
    primary_metric = "ndcg@10"
    if param_name in {"embedding_size", "n_layers"}:
        rationale = "This is a RecBole-native knob already consumed by the wired parent."
    else:
        rationale = "The parent candidate already exposes this local parameter through consumes."
    return {
        "proposal_type": "tuning",
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_id,
        "base_model": base_model,
        "category": parent.get("category") or "Uncategorized",
        "hypothesis": (
            f"Tuning {param_name} to {param_value} may improve {primary_metric} "
            f"without changing the executable candidate path."
        ),
        "runnable_level": "parameter_only",
        "runner_type": parent.get("runner_type") or "config_only",
        "consumes": [param_name],
        "parameter_overrides": {param_name: param_value},
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
        "promotion_requirements": [
            "validator status is accepted",
            "run through scripts/run_candidate.py with parameter_overrides",
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
) -> list[dict[str, Any]]:
    used_signatures = used_parent_param_signatures(memory)
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
    seen_parent_param: set[tuple[str, str]] = set()
    for parent in parents:
        picked = choose_param_override(parent, used_signatures)
        if picked is None:
            continue
        param_name, param_value = picked
        key = (str(parent.get("candidate_id") or ""), param_name)
        if key in seen_parent_param:
            continue
        seen_parent_param.add(key)
        proposals.append(build_parameter_proposal(parent, param_name, param_value, len(proposals) + 1, stamp))
        if len(proposals) >= count:
            break
    return proposals


def existing_candidate_ids(registry: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("candidate_id")) for item in registry if item.get("candidate_id")}


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
) -> list[dict[str, Any]]:
    registry_ids = existing_candidate_ids(registry)
    proposals: list[dict[str, Any]] = []
    for template in ALGORITHM_TEMPLATES:
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
) -> list[dict[str, Any]]:
    registry = load_yaml(registry_path).get("candidates", [])
    if not isinstance(registry, list):
        raise ValueError(f"registry candidates must be a list: {registry_path}")
    experiment_log = experiment_log_path.read_text(encoding="utf-8", errors="replace") if experiment_log_path.exists() else ""
    memory = load_jsonl(memory_path, memory_limit)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == "conservative":
        return generate_tuning_proposals(
            registry=registry,
            experiment_log=experiment_log,
            memory=memory,
            count=count,
            stamp=stamp,
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
