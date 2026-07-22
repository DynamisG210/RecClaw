#!/usr/bin/env python3
"""RecClaw candidate orchestration agent.

Execution chain:
Observe -> Plan -> Act -> Evaluate -> Reflect -> Remember
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import socket
import ssl
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml

try:
    from .action_space import (
        allowed_action_types,
        load_action_space,
        parameter_space_from_action_space,
    )
    from .collect_result import load_result_source
    from .compare_runs import compare_results
    from .research_line import (
        annotate_proposals,
        build_producer_directives,
        build_search_memory,
        load_yaml as load_research_yaml,
        meta_research_update,
        order_proposals_by_route,
        producer_directives_payload,
        route_candidate_option,
        route_proposals,
    )
except ImportError:
    from action_space import (
        allowed_action_types,
        load_action_space,
        parameter_space_from_action_space,
    )
    from collect_result import load_result_source
    from compare_runs import compare_results
    from research_line import (
        annotate_proposals,
        build_producer_directives,
        build_search_memory,
        load_yaml as load_research_yaml,
        meta_research_update,
        order_proposals_by_route,
        producer_directives_payload,
        route_candidate_option,
        route_proposals,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
MEMORY_PATH = PROJECT_ROOT / "results" / "agent_memory.jsonl"
STATE_SUMMARY_PATH = PROJECT_ROOT / "results" / "agent_state_summary.json"
NOTES_DIR = PROJECT_ROOT / "notes"
PROPOSAL_PATH = PROJECT_ROOT / "results" / "candidate_proposals.jsonl"
PROPOSAL_SCHEMA_PATH = PROJECT_ROOT / "configs" / "candidate_proposal_schema.yaml"
ACTION_SPACE_PATH = PROJECT_ROOT / "configs" / "action_space.yaml"
SEARCH_POLICY_PATH = PROJECT_ROOT / "configs" / "search_policy.yaml"
CANDIDATE_TREE_PATH = PROJECT_ROOT / "results" / "candidate_search_tree.json"
CANDIDATE_TREE_MD_PATH = PROJECT_ROOT / "results" / "candidate_search_tree.md"
CANDIDATE_TREE_MMD_PATH = PROJECT_ROOT / "results" / "candidate_search_tree.mmd"
EXPERIENCE_SUMMARY_PATH = PROJECT_ROOT / "results" / "experience_summary.md"
EXPERIENCE_SUMMARY_JSON_PATH = PROJECT_ROOT / "results" / "experience_summary.json"
REFLECTION_MEMORY_PATH = PROJECT_ROOT / "results" / "reflection_memory.jsonl"
AUTO_PROMOTABLE_PARENT_IDS = {
    "cand_bpr_long_tail_reweight",
    "cand_bpr_popularity_regularized",
    "cand_bpr_norm_constrained",
    "cand_bpr_hard_negative_mix",
    "cand_bpr_popularity_aware_negative",
    "cand_lightgcn_edge_dropout_residual_mix",
    "cand_lightgcn_aux_alignment_loss",
    "cand_lightgcn_rank_aware_loss",
}

REQUIRED_MULTI_METRICS = (
    "ndcg@10",
    "recall@10",
    "mrr@10",
    "hit@10",
    "precision@10",
    "itemcoverage@10",
    "latency_ms",
)
DEFAULT_METRIC_DIRECTIONS: dict[str, int] = {
    "run_time": -1,
    "latency_ms": -1,
}
EVALUATION_DIMENSIONS = (
    "ndcg@10",
    "recall@10",
    "mrr@10",
    "hit@10",
    "precision@10",
    "itemcoverage@10",
    "latency_ms",
    "run_time",
)

PRIORITY_SCORE = {"high": 1.0, "medium": 0.5, "low": 0.2}
STATUS_SCORE = {"implemented": 1.0, "implement-ready": 0.7, "spec-ready": 0.3, "idea": 0.1}
SIGNATURE_EXCLUDED_KEYS = {"seed", "reproducibility", "checkpoint_dir"}
TERMINAL_IMPLEMENTATION_SKIP_REASONS = {
    "known parent was auto-promoted and is runnable",
    "dry_run",
}
DEFAULT_VALIDATION_SEEDS = [2026, 2027, 2028]
MAX_EXPERIMENT_DIRECTIVE_CHARS = 4000
IMPLEMENTATION_RUNNABLE_STATUSES = {
    "implemented_and_runnable",
    "implemented_and_importable",
    "implemented_and_smoke_passed",
}
IMPLEMENTATION_TERMINAL_FAILURE_STATUSES = {
    "implementation_failed",
    "implementation_rejected",
    "implemented_but_smoke_failed",
}
LOOP_MODE_POLICIES = {
    "tuning": {
        "proposal_mode": "conservative",
        "auto_promote": False,
        "auto_implement": False,
        "llm_plan": False,
    },
    "mixed": {
        "proposal_mode": "mixed",
        "auto_promote": True,
        "auto_implement": False,
        "llm_plan": False,
    },
    "explore": {
        "proposal_mode": "explore",
        "auto_promote": True,
        "auto_implement": True,
        "llm_plan": False,
    },
    "auto": {
        "proposal_mode": "algorithm_first",
        "auto_promote": True,
        "auto_implement": True,
        "llm_plan": True,
    },
}
AUTO_PLANNER_ACTIONS = {
    "run_available",
    "run_algorithm_variant",
    "propose_tuning",
    "propose_mixed",
    "propose_explore",
    "propose_algorithm",
    "implement_needs_review",
    "implement_algorithm",
    "tune_after_algorithm_success",
    "multi_seed_verify",
    "report",
}
ACTION_SPACE = load_action_space(ACTION_SPACE_PATH)
DEFAULT_PARAMETER_SPACE = parameter_space_from_action_space(ACTION_SPACE)
PROPOSAL_PARAMETER_KEYS = tuple(DEFAULT_PARAMETER_SPACE)
PROPOSAL_ACTION_TYPES = tuple(sorted(allowed_action_types(ACTION_SPACE)))


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
PROPOSAL_PARAMETER_SCHEMA = {key: {"type": ["number", "null"]} for key in PROPOSAL_PARAMETER_KEYS}
SELF_REG_LOSS_PATTERN = re.compile(r"\bself\.reg_loss\s*\(")
LIGHTGCN_BASE_CLASS_PATTERN = re.compile(r"class\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*LightGCN[^)]*\)")
REG_LOSS_GUARD_PATTERN = re.compile(r"hasattr\s*\(\s*self\s*,\s*[\"']reg_weight[\"']\s*\)")
SOFT_L2_POSITIONAL_MAX_NORM_PATTERN = re.compile(
    r"soft_l2_norm_penalty\s*\([^)\n]+,\s*(?:self\.)?(?:max_norm|lambda_norm)\b"
)

PROPOSAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "proposal_type": {"type": "string", "enum": ["tuning", "algorithmic_variant", "research_spec"]},
                    "candidate_id": {"type": "string"},
                    "parent_candidate_id": {"type": "string"},
                    "base_model": {"type": "string", "enum": ["BPR", "LightGCN"]},
                    "category": {"type": "string"},
                    "action_type": {"type": "string", "enum": list(PROPOSAL_ACTION_TYPES)},
                    "mechanism": {"type": "string"},
                    "mechanism_composition": {"type": "array", "items": {"type": "string"}},
                    "novelty_claim": {"type": "string"},
                    "expected_failure_mode": {"type": "string"},
                    "ablation_parent": {"type": "string"},
                    "implementation_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "hypothesis": {"type": "string"},
                    "runnable_level": {
                        "type": "string",
                        "enum": ["parameter_only", "config_only", "spec_only", "code_required"],
                    },
                    "runner_type": {"type": "string", "enum": ["config_only", "model", "posthoc"]},
                    "consumes": {"type": "array", "items": {"type": "string"}},
                    "parameter_overrides": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": PROPOSAL_PARAMETER_SCHEMA,
                        "required": list(PROPOSAL_PARAMETER_KEYS),
                    },
                    "parameter_signature": {"type": "string"},
                    "expected_effect": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "primary_metric": {"type": "string"},
                            "direction": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["primary_metric", "direction", "rationale"],
                    },
                    "risk": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "quality": {"type": "string"},
                            "runtime": {"type": "string"},
                            "implementation": {"type": "string"},
                            "recbole_core_change_required": {"type": "boolean"},
                        },
                        "required": ["quality", "runtime", "implementation", "recbole_core_change_required"],
                    },
                    "decision_rule": {"type": "string"},
                    "evaluation_plan": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "primary_metric": {"type": "string"},
                            "validation_seeds": {"type": "array", "items": {"type": "integer"}},
                            "aggregation": {"type": "string"},
                            "promote_if": {"type": "string"},
                        },
                        "required": ["primary_metric", "validation_seeds", "aggregation", "promote_if"],
                    },
                    "implementation_plan": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "summary": {"type": "string"},
                            "entrypoint": {"type": "string"},
                            "files": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["summary", "entrypoint", "files"],
                    },
                    "allowed_files": {"type": "array", "items": {"type": "string"}},
                    "new_parameters": {"type": "array", "items": {"type": "string"}},
                    "promotion_requirements": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "proposal_type",
                    "candidate_id",
                    "parent_candidate_id",
                    "base_model",
                    "category",
                    "action_type",
                    "mechanism",
                    "mechanism_composition",
                    "novelty_claim",
                    "expected_failure_mode",
                    "ablation_parent",
                    "implementation_complexity",
                    "hypothesis",
                    "runnable_level",
                    "runner_type",
                    "consumes",
                    "expected_effect",
                    "risk",
                    "decision_rule",
                    "parameter_overrides",
                    "parameter_signature",
                    "evaluation_plan",
                    "implementation_plan",
                    "allowed_files",
                    "new_parameters",
                    "promotion_requirements",
                ],
            },
        }
    },
    "required": ["proposals"],
}

PLANNER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": sorted(AUTO_PLANNER_ACTIONS)},
        "reason": {"type": "string"},
        "proposal_count": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    "required": ["action", "reason", "proposal_count"],
}


@dataclass
class AgentConfig:
    rounds: int = 5
    start_round: int = 1
    metric: str = "ndcg@10"
    multi_metrics: bool = False
    min_keep_delta: float = 1e-5
    revisit_penalty: float = 0.1
    crash_penalty: float = 0.2
    novelty_weight: float = 0.45
    priority_weight: float = 0.25
    status_weight: float = 0.20
    recent_schedule_penalty: float = 0.15
    seed: int = 42
    dry_run: bool = False
    memory_read_limit: int = 2000
    prompt_memory_tail: int = 120
    use_experiment_directive: bool = False
    experiment_directive: str = ""
    static_experiment_directive: str = ""
    memory_path: Path = MEMORY_PATH
    state_summary_path: Path = STATE_SUMMARY_PATH
    results_csv: Path = RESULTS_CSV
    baseline_dir: Path = BASELINE_DIR
    registry_path: Path = REGISTRY_PATH
    candidate_tree_path: Path = CANDIDATE_TREE_PATH
    candidate_tree_md_path: Path = CANDIDATE_TREE_MD_PATH
    candidate_tree_mmd_path: Path = CANDIDATE_TREE_MMD_PATH
    experience_summary_path: Path = EXPERIENCE_SUMMARY_PATH
    experience_summary_json_path: Path = EXPERIENCE_SUMMARY_JSON_PATH
    reflection_memory_path: Path = REFLECTION_MEMORY_PATH
    search_policy_path: Path = SEARCH_POLICY_PATH
    research_route_path: Path = Path(os.environ.get("RECCLAW_RESEARCH_ROUTES", str(PROJECT_ROOT / "results" / "research_routes.jsonl")))
    research_line_enabled: bool = True
    refresh_experience_every: int = 0
    loop_mode: str = "mixed"
    enable_candidate_proposals: bool = True
    proposal_path: Path = PROPOSAL_PATH
    proposal_mode: str = "mixed"
    proposal_count: int = 5
    proposal_every: int = 3
    proposal_bonus: float = 0.35
    proposal_source: str = "llm"
    proposal_schema_path: Path = PROPOSAL_SCHEMA_PATH
    auto_promote_needs_review: bool = True
    auto_implement_code_required: bool = False
    max_pending_implemented: int = 3
    max_implement_per_round: int = 1
    implementation_smoke_run: bool = True
    max_failed_implementation_attempts: int = 2
    search_intensity: str = "balanced"
    algorithm_budget_per_window: int = 3
    algorithm_first_explore_rounds: int = 20
    seed_validation_min_metric: float = 0.0
    anti_plateau_enabled: bool = True
    plateau_window_metric_rows: int = 30
    plateau_min_global_improvement: float = 0.0005
    plateau_family_overuse_window: int = 20
    max_same_family_repair_streak: int = 8
    plateau_weak_family_ceiling: float = 0.280
    anti_plateau_anchor_bonus: float = 0.85
    anti_plateau_local_family_penalty: float = 1.25
    anchor_families: list[str] = field(
        default_factory=lambda: [
            "cand_bpr_hard_negative_margin",
            "cand_lightgcn_shallow_layers",
            "cand_lightgcn_residual_norm_constrained",
            "cand_lightgcn_edge_dropout_residual_norm",
        ]
    )
    tuned_lightgcn_mean: float = 0.274367
    exploitation_window_rounds: int = 8
    post_validation_structured_followup_window: int = 20
    post_validation_sibling_churn_limit: int = 6
    post_validation_min_followup_improvement: float = 0.0005
    global_overrides: list[str] = field(default_factory=list)
    checkpoint_dir: str = ""
    checkpoint_policy: str = "none"
    enable_seed_validation: bool = False
    validation_seeds: list[int] = field(default_factory=lambda: list(DEFAULT_VALIDATION_SEEDS))
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_temperature: float = 0.2
    llm_timeout: int = 120
    llm_max_tokens: int = 4096
    llm_retries: int = 2
    allow_llm_fallback: bool = False
    metrics_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ndcg@10": 0.2,
            "recall@10": 0.2,
            "mrr@10": 0.15,
            "hit@10": 0.15,
            "precision@10": 0.1,
            "itemcoverage@10": 0.1,
            "latency_ms": 0.1,
        }
    )
    metric_directions: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_METRIC_DIRECTIONS))
    parameter_space: dict[str, list[Any]] = field(default_factory=lambda: dict(DEFAULT_PARAMETER_SPACE))


@dataclass
class TrialRecord:
    round_id: int
    candidate_id: str
    params: dict[str, Any]
    run_id: str
    status: str
    result: dict[str, Any]
    compare_baseline: dict[str, Any]
    compare_history_best: dict[str, Any]
    dimension_report: dict[str, Any]
    decision: str
    reason: str
    next_action: str
    parent_candidate_id: str = ""
    proposal_id: str = ""
    parameter_signature: str = ""
    execution_signature: str = ""
    proposal_source: str = ""
    producer_id: str = ""
    producer_role: str = ""
    route_decision: dict[str, Any] = field(default_factory=dict)
    is_baseline_improvement: bool = False
    is_history_best: bool = False
    seed_validation: dict[str, Any] = field(default_factory=dict)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_weights(weights_json: str) -> dict[str, float]:
    raw = json.loads(weights_json)
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("--metrics-weights-json must be an object")
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        metric = str(key).lower().strip()
        if metric not in REQUIRED_MULTI_METRICS:
            raise argparse.ArgumentTypeError(f"unsupported metric in weights: {key}")
        normalized[metric] = float(value)
    total = sum(normalized.values())
    if total <= 0:
        raise argparse.ArgumentTypeError("sum(weights) must be > 0")
    return {k: v / total for k, v in normalized.items()}


def resolve_experiment_directive(
    directive: str = "",
    directive_file: str | Path = "",
    *,
    disabled: bool = False,
) -> str:
    if disabled:
        return ""
    parts: list[str] = []
    if directive_file:
        path = Path(directive_file)
        if not path.exists():
            raise FileNotFoundError(f"experiment directive file does not exist: {path}")
        file_text = path.read_text(encoding="utf-8", errors="replace").strip()
        if file_text:
            parts.append(file_text)
    directive_text = str(directive or "").strip()
    if directive_text:
        parts.append(directive_text)
    text = "\n\n".join(parts).strip()
    if len(text) <= MAX_EXPERIMENT_DIRECTIVE_CHARS:
        return text
    return text[:MAX_EXPERIMENT_DIRECTIVE_CHARS].rstrip() + "\n[truncated by RecClaw]"


def reject_runtime_lablog_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if "recclaw_lablog" in text:
        raise ValueError(f"agent runtime refuses LabLog path: {path}")


class RecClawAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.memory: list[dict[str, Any]] = []
        self.registry: list[dict[str, Any]] = []
        self.baselines_by_model: dict[str, dict[str, Any]] = {}
        self.history_by_candidate: dict[str, list[dict[str, Any]]] = {}
        self.best_by_model: dict[str, dict[str, Any]] = {}
        self.results_rows: list[dict[str, Any]] = []
        self.candidate_proposals: list[dict[str, Any]] = []
        self.proposal_validation_report: dict[str, Any] = {}
        self.scheduled_candidate_ids: set[str] = set()
        self.scheduled_param_signatures: set[str] = set()
        self.scheduled_execution_signatures: set[str] = set()
        self.recorded_proposal_events: set[str] = set()
        self.last_planner_action: dict[str, Any] = {}
        self.agent_state_summary: dict[str, Any] = {}
        self.candidate_health_issues: dict[str, list[str]] = {}
        self.quarantined_candidate_ids: set[str] = set()
        self.search_policy: dict[str, Any] = {}
        self.search_memory: dict[str, Any] = {}
        self.current_proposal_source: str = config.proposal_source
        self.skip_current_round: bool = False
        self.skip_proposal_generation: bool = False
        self.force_proposal_refresh: bool = False
        self.last_experiment_directive_digest: str = ""

    # ===== Observe =====
    def observe(self) -> None:
        self.memory = []
        self.registry = []
        self.baselines_by_model = {}
        self.history_by_candidate = {}
        self.best_by_model = {}
        self.results_rows = []
        self.scheduled_param_signatures = set()
        self.scheduled_execution_signatures = set()
        self.recorded_proposal_events = set()
        payload = load_yaml(self.config.registry_path)
        self.registry = payload.get("candidates", [])
        if not isinstance(self.registry, list):
            raise ValueError("registry candidates must be a list")
        self.search_policy = load_research_yaml(self.config.search_policy_path)

        if self.config.memory_path.exists():
            lines = self.config.memory_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if self.config.memory_read_limit > 0:
                lines = lines[-self.config.memory_read_limit :]
            skipped_memory_lines = 0
            for line_no, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.memory.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    skipped_memory_lines += 1
                    print(
                        f"[Observe] skip invalid memory line {line_no}: {exc}",
                        file=sys.stderr,
                    )
            if skipped_memory_lines:
                print(
                    f"[Observe] skipped_invalid_memory_lines={skipped_memory_lines}",
                    file=sys.stderr,
                )

        if self.config.results_csv.exists():
            with self.config.results_csv.open("r", encoding="utf-8", newline="") as handle:
                self.results_rows = list(csv.DictReader(handle))

        for item in self.memory:
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id:
                self.history_by_candidate.setdefault(candidate_id, []).append(item)

        for log_file in self.config.baseline_dir.glob("*.log"):
            parsed = load_result_source(log_file)
            model = self._normalize_model_key(parsed.get("model"))
            if model:
                existing = self.baselines_by_model.get(model)
                if existing is not None:
                    print(f"Warning: duplicate baseline for model '{model}', keeping best-scoring baseline")
                if existing is None or self._should_replace_result(existing, parsed):
                    self.baselines_by_model[model] = parsed

        for row in self.results_rows:
            model = self._normalize_model_key(row.get("model"))
            status = str(row.get("status") or "").lower()
            if not model or status != "success":
                continue
            if model not in self.best_by_model:
                self.best_by_model[model] = dict(row)
                continue
            if self._should_replace_result(self.best_by_model[model], row):
                self.best_by_model[model] = dict(row)
        self.candidate_health_issues = self._build_candidate_health_issues()
        self.quarantined_candidate_ids = {
            candidate_id for candidate_id, issues in self.candidate_health_issues.items() if issues
        }
        self.search_policy = load_research_yaml(self.config.search_policy_path)
        self._refresh_search_memory()
        self._refresh_search_memory()
        self.agent_state_summary = self._build_agent_state_summary()
        self._write_agent_state_summary(self.agent_state_summary)

    def _refresh_search_memory(self) -> None:
        if not self.config.research_line_enabled:
            self.search_memory = {}
            return
        self.search_memory = build_search_memory(
            self.memory,
            metric=self.config.metric,
            anchor_families=list(self.config.anchor_families),
        )

    def _reload_registry(self) -> None:
        payload = load_yaml(self.config.registry_path)
        registry = payload.get("candidates", [])
        if not isinstance(registry, list):
            raise ValueError("registry candidates must be a list")
        self.registry = registry
        self.candidate_health_issues = self._build_candidate_health_issues()
        self.quarantined_candidate_ids = {
            candidate_id for candidate_id, issues in self.candidate_health_issues.items() if issues
        }

    def _build_candidate_health_issues(self) -> dict[str, list[str]]:
        crash_counts: dict[str, int] = {}
        for row in self._trial_memory_rows():
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id and (str(row.get("decision") or "") == "crash" or str(row.get("status") or "") == "crash"):
                crash_counts[candidate_id] = crash_counts.get(candidate_id, 0) + 1

        issues: dict[str, list[str]] = {}
        for candidate in self.registry:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or not bool(candidate.get("wired")):
                continue
            candidate_issues: list[str] = []
            candidate_config_path = PROJECT_ROOT / "configs" / "candidates" / f"{candidate_id}.yaml"
            if not candidate_config_path.exists():
                candidate_issues.append(f"candidate config is missing: {candidate_config_path.relative_to(PROJECT_ROOT)}")
            entrypoint = str(candidate.get("entrypoint") or "")
            module = entrypoint.split(":", 1)[0]
            if module in {"recclaw_ext.models", "recclaw_ext.posthoc"}:
                candidate_issues.append("package-level local entrypoint is not concrete")
            elif module.startswith("recclaw_ext.models."):
                source_path = PROJECT_ROOT / (module.replace(".", "/") + ".py")
                if not source_path.exists():
                    candidate_issues.append(f"local model source is missing: {source_path.relative_to(PROJECT_ROOT)}")
                else:
                    text = source_path.read_text(encoding="utf-8", errors="replace")
                    if "config.get" in text:
                        candidate_issues.append("local model source uses RecBole-unsafe config.get")
                    if (
                        SELF_REG_LOSS_PATTERN.search(text)
                        and "def reg_loss" not in text
                        and "self.reg_loss =" not in text
                        and not LIGHTGCN_BASE_CLASS_PATTERN.search(text)
                        and not REG_LOSS_GUARD_PATTERN.search(text)
                    ):
                        candidate_issues.append("local model source calls undefined self.reg_loss")
                    if SOFT_L2_POSITIONAL_MAX_NORM_PATTERN.search(text):
                        candidate_issues.append("local model source passes max_norm/lambda_norm positionally to soft_l2_norm_penalty")
            if crash_counts.get(candidate_id, 0) >= 2:
                candidate_issues.append(f"candidate has {crash_counts[candidate_id]} crash records in agent memory")
            if candidate_issues:
                issues[candidate_id] = candidate_issues
        return issues

    @staticmethod
    def _normalize_model_key(raw: Any) -> str:
        text = str(raw or "").strip()
        lowered = text.lower()
        if "lightgcn" in lowered:
            return "LightGCN"
        if "bpr" in lowered:
            return "BPR"
        return text

    def _should_replace_result(self, current: dict[str, Any], candidate: dict[str, Any]) -> bool:
        current_success = str(current.get("status") or "").lower() == "success"
        candidate_success = str(candidate.get("status") or "").lower() == "success"
        if candidate_success != current_success:
            return candidate_success
        return self._score(candidate) > self._score(current)

    def _update_history_best(self, model_key: str, result: dict[str, Any]) -> None:
        key = self._normalize_model_key(model_key or result.get("model"))
        if not key or str(result.get("status") or "").lower() != "success":
            return
        self.results_rows.append(dict(result))
        current_best = self.best_by_model.get(key)
        if current_best is None or self._should_replace_result(current_best, result):
            self.best_by_model[key] = dict(result)

    def _load_optional_context(self, candidate: dict[str, Any], force: bool = False) -> dict[str, str]:
        context: dict[str, str] = {}
        if not force:
            return context
        cid = str(candidate.get("candidate_id") or "")
        if not cid:
            return context
        cid_pattern = re.compile(rf"\b{re.escape(cid)}\b", flags=re.IGNORECASE)
        for name in ("candidate_library.md", "method_change_space.md"):
            path = NOTES_DIR / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            match = cid_pattern.search(text)
            if match is None:
                continue
            start = max(0, match.start() - 600)
            end = min(len(text), match.end() + 1200)
            context[name] = text[start:end]
        return context

    @staticmethod
    def _params_signature(params: dict[str, Any]) -> str:
        if not params:
            return "{}"
        normalized = {
            str(key): normalize_signature_value(value)
            for key, value in params.items()
            if str(key) not in SIGNATURE_EXCLUDED_KEYS
        }
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def _parameter_signature(self, parent_id: str, params: dict[str, Any]) -> str:
        if not parent_id or not params:
            return ""
        return f"{parent_id}::{self._params_signature(params)}"

    def _canonical_parameter_signature_text(self, raw: Any) -> str:
        text = str(raw or "").strip()
        if "::" not in text:
            return text
        parent_id, raw_params = text.split("::", 1)
        try:
            params = json.loads(raw_params)
        except json.JSONDecodeError:
            return text
        if not isinstance(params, dict):
            return text
        return self._parameter_signature(parent_id, params)

    def _execution_signature_from_values(
        self,
        *,
        model_key: str,
        runner_type: str,
        entrypoint: str,
        params: dict[str, Any],
    ) -> str:
        normalized_model = self._normalize_model_key(model_key)
        normalized_params = self._params_signature(params)
        normalized_entrypoint = str(entrypoint or "").strip()
        normalized_runner = str(runner_type or "").strip()
        lowered_entrypoint = normalized_entrypoint.lower()
        if normalized_model == "LightGCN" and (
            normalized_runner == "config_only" or "general_recommender.lightgcn" in lowered_entrypoint
        ):
            executable = "LightGCN/native"
        elif normalized_entrypoint:
            executable = normalized_entrypoint
        elif normalized_model:
            executable = normalized_model
        else:
            return ""
        return f"{executable}::{normalized_params}"

    def _execution_signature(self, candidate: dict[str, Any], params: dict[str, Any]) -> str:
        return self._execution_signature_from_values(
            model_key=str(candidate.get("base_model") or candidate.get("model") or ""),
            runner_type=str(candidate.get("runner_type") or ""),
            entrypoint=str(candidate.get("entrypoint") or ""),
            params=params,
        )

    def _used_parent_param_signatures(self) -> set[str]:
        signatures: set[str] = set()
        for row in self.memory:
            explicit = str(row.get("parameter_signature") or "")
            if explicit:
                signatures.add(self._canonical_parameter_signature_text(explicit))
                continue
            parent_id = str(row.get("parent_candidate_id") or row.get("candidate_id") or "")
            params = row.get("params") or row.get("parameter_overrides")
            if parent_id and isinstance(params, dict) and params:
                signatures.add(self._parameter_signature(parent_id, params))
        return signatures

    def _used_execution_signatures(self) -> set[str]:
        signatures: set[str] = set()
        for row in self.memory:
            explicit = str(row.get("execution_signature") or "")
            if explicit:
                signatures.add(explicit)
                continue
            params = row.get("params")
            if not isinstance(params, dict):
                continue
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            model_key = str(result.get("model") or row.get("base_model") or "")
            fallback = self._execution_signature_from_values(
                model_key=model_key,
                runner_type="config_only" if self._normalize_model_key(model_key) == "LightGCN" else "",
                entrypoint="",
                params=dict(params),
            )
            if fallback:
                signatures.add(fallback)
        return signatures

    def _used_param_signatures(self, candidate_id: str) -> set[str]:
        signatures: set[str] = set()
        for row in self.history_by_candidate.get(candidate_id, []):
            params = row.get("params")
            if isinstance(params, dict):
                signatures.add(self._params_signature(params))
        return signatures

    def _choose_params(self, candidate: dict[str, Any]) -> dict[str, Any]:
        consumes = candidate.get("consumes") or []
        param_values: list[tuple[str, list[Any]]] = []
        params: dict[str, Any] = {}
        for key in consumes:
            key_text = str(key)
            values = self.config.parameter_space.get(key_text, [])
            if values:
                param_values.append((key_text, values))
                params[key_text] = self.rng.choice(values)

        if not param_values:
            return params

        used = self._used_param_signatures(str(candidate.get("candidate_id") or ""))
        current_sig = self._params_signature(params)
        if current_sig not in used:
            return params

        max_tries = max(20, 4 * len(param_values))
        for _ in range(max_tries):
            proposal = {k: self.rng.choice(v) for k, v in param_values}
            if self._params_signature(proposal) not in used:
                return proposal
        return params

    def _choose_execution_unique_params(
        self,
        candidate: dict[str, Any],
        used_execution_signatures: set[str],
    ) -> dict[str, Any]:
        params = self._choose_params(candidate)
        execution_signature = self._execution_signature(candidate, params)
        if not execution_signature or (
            execution_signature not in used_execution_signatures
            and execution_signature not in self.scheduled_execution_signatures
        ):
            return params

        consumes = candidate.get("consumes") or []
        param_values: list[tuple[str, list[Any]]] = []
        for key in consumes:
            values = self.config.parameter_space.get(str(key), [])
            if values:
                param_values.append((str(key), values))
        max_tries = max(30, 6 * len(param_values))
        for _ in range(max_tries):
            proposal = {key: self.rng.choice(values) for key, values in param_values}
            execution_signature = self._execution_signature(candidate, proposal)
            if execution_signature and (
                execution_signature in used_execution_signatures
                or execution_signature in self.scheduled_execution_signatures
            ):
                continue
            if self._params_signature(proposal) in self._used_param_signatures(str(candidate.get("candidate_id") or "")):
                continue
            return proposal
        return params

    # ===== Plan =====
    def _find_registry_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        for candidate in self.registry:
            if str(candidate.get("candidate_id") or "") == candidate_id:
                return candidate
        return None

    def _proposal_status_by_id(self) -> dict[str, dict[str, Any]]:
        results = self.proposal_validation_report.get("results", [])
        if not isinstance(results, list):
            return {}
        return {str(item.get("candidate_id") or ""): item for item in results if isinstance(item, dict)}

    def _accepted_proposal_options(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        statuses = self._proposal_status_by_id()
        options: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for proposal in self.candidate_proposals:
            proposal_id = str(proposal.get("candidate_id") or "")
            status = statuses.get(proposal_id, {})
            status_name = str(status.get("status") or "")
            proposal_type = str(proposal.get("proposal_type") or "")
            runnable_level = str(proposal.get("runnable_level") or "")
            parent_id = str(proposal.get("parent_candidate_id") or "")
            accepted_tuning = (
                status_name == "accepted"
                and proposal_type == "tuning"
                and runnable_level in {"parameter_only", "config_only"}
            )
            promoted_code = (
                status_name == "needs_review"
                and proposal_type == "algorithmic_variant"
                and runnable_level == "code_required"
                and parent_id in AUTO_PROMOTABLE_PARENT_IDS
            )
            if not accepted_tuning and not promoted_code:
                continue
            if proposal_id in self.scheduled_candidate_ids:
                continue
            if self.history_by_candidate.get(proposal_id):
                continue
            parent = self._find_registry_candidate(parent_id)
            if parent is None or not bool(parent.get("wired")):
                continue
            if parent_id in self.quarantined_candidate_ids:
                continue
            if str(parent.get("runner_type") or "") not in {"config_only", "model"}:
                continue
            overrides = proposal.get("parameter_overrides") or {}
            if not isinstance(overrides, dict):
                continue
            parameter_signature = str(status.get("parameter_signature") or "")
            if not parameter_signature and overrides:
                parameter_signature = self._parameter_signature(parent_id, dict(overrides))
            if parameter_signature and (
                parameter_signature in self._used_parent_param_signatures()
                or parameter_signature in self.scheduled_param_signatures
            ):
                continue
            execution_signature = self._execution_signature(parent, dict(overrides))
            if execution_signature and (
                execution_signature in self._used_execution_signatures()
                or execution_signature in self.scheduled_execution_signatures
            ):
                continue
            if promoted_code and not overrides:
                continue
            parent_consumes = {str(item) for item in (parent.get("consumes") or [])}
            if promoted_code and not set(map(str, overrides.keys())).issubset(parent_consumes):
                continue
            candidate = {
                **parent,
                "candidate_id": proposal_id,
                "run_candidate_id": parent_id,
                "parent_candidate_id": parent_id,
                "proposal_type": proposal.get("proposal_type"),
                "runnable_level": runnable_level,
                "hypothesis": proposal.get("hypothesis") or parent.get("hypothesis"),
                "consumes": proposal.get("consumes") or parent.get("consumes") or [],
                "proposal_id": proposal_id,
                "proposal_source": self.current_proposal_source,
                "producer_id": proposal.get("producer_id", ""),
                "producer_role": proposal.get("producer_role", ""),
                "research_line": proposal.get("research_line", {}),
                "parameter_signature": parameter_signature,
                "execution_signature": execution_signature,
            }
            options.append((candidate, dict(overrides)))
        return options

    def _has_candidate_run(self, candidate_id: str) -> bool:
        if self.history_by_candidate.get(candidate_id):
            return True
        for row in self.memory:
            if row.get("event"):
                continue
            if str(row.get("parent_candidate_id") or "") == candidate_id:
                return True
        return False

    def _pending_implemented_candidate_ids(self) -> set[str]:
        implemented_ids = {
            str(row.get("candidate_id") or "")
            for row in self.memory
            if row.get("event") == "implementation_result"
            and str(row.get("status") or "") in IMPLEMENTATION_RUNNABLE_STATUSES
            and row.get("candidate_id")
        }
        pending: set[str] = set()
        for candidate_id in implemented_ids:
            if candidate_id in self.quarantined_candidate_ids:
                continue
            if candidate_id in self.scheduled_candidate_ids or self._has_candidate_run(candidate_id):
                continue
            candidate = self._find_registry_candidate(candidate_id)
            if candidate is None:
                continue
            if not bool(candidate.get("wired")):
                continue
            if str(candidate.get("runner_type") or "") not in {"config_only", "model"}:
                continue
            pending.add(candidate_id)
        return pending

    def _implemented_algorithm_candidate_ids(self) -> set[str]:
        return {
            str(row.get("candidate_id") or "")
            for row in self.memory
            if row.get("event") == "implementation_result"
            and str(row.get("status") or "") in IMPLEMENTATION_RUNNABLE_STATUSES
            and row.get("candidate_id")
        }

    def _failed_implementation_attempts(self, proposal_id: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self.memory
            if row.get("event") == "implementation_result"
            and str(row.get("proposal_id") or "") == proposal_id
            and str(row.get("status") or "") in IMPLEMENTATION_TERMINAL_FAILURE_STATUSES
        ]

    def _formal_trial_count(self) -> int:
        return len(self._trial_memory_rows())

    def _candidate_config_params(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return {}
        config_path = PROJECT_ROOT / "configs" / "candidates" / f"{candidate_id}.yaml"
        config = load_yaml(config_path)
        if not config:
            return {}
        consumes = {str(item) for item in (candidate.get("consumes") or [])}
        params: dict[str, Any] = {}
        for key in consumes:
            if key in config and key in self.config.parameter_space:
                params[key] = config[key]
        return params

    def _candidate_plan_score(self, candidate: dict[str, Any]) -> float:
        cid = str(candidate.get("candidate_id") or "")
        runs = self.history_by_candidate.get(cid, [])
        explored = len(runs)
        crashes = sum(1 for r in runs if str(r.get("decision")) == "crash")
        recently_scheduled = bool(self.memory) and str(self.memory[-1].get("candidate_id") or "") == cid
        priority = PRIORITY_SCORE.get(str(candidate.get("priority", "")).lower(), 0.0)
        status = STATUS_SCORE.get(str(candidate.get("status", "")).lower(), 0.0)
        novelty = 1.0 / (1.0 + explored)
        proposal_bonus = self.config.proposal_bonus if candidate.get("parent_candidate_id") else 0.0
        family_credit = self._family_plan_credit(str(candidate.get("parent_candidate_id") or cid))
        return (
            self.config.novelty_weight * novelty
            + self.config.priority_weight * priority
            + self.config.status_weight * status
            + proposal_bonus
            + family_credit
            - self.config.revisit_penalty * explored
            - self.config.crash_penalty * crashes
            - (self.config.recent_schedule_penalty if recently_scheduled else 0.0)
        )

    def _family_plan_credit(self, candidate_id: str) -> float:
        keeps = 0
        revises = 0
        discards = 0
        crashes = 0
        collapses = 0
        for row in self._trial_memory_rows():
            if self._family_key(row) != candidate_id and str(row.get("candidate_id") or "") != candidate_id:
                continue
            decision = str(row.get("decision") or "")
            keeps += int(decision == "keep")
            revises += int(decision == "revise")
            discards += int(decision == "discard")
            crashes += int(decision == "crash")
            collapses += int(self._is_extreme_quality_collapse(row))
        return (
            min(0.45, 0.2 * keeps + 0.04 * revises)
            - min(0.45, 0.15 * crashes + 0.04 * discards + 0.18 * collapses)
        )

    @staticmethod
    def _family_matches(candidate_family: str, target_family: str) -> bool:
        candidate_family = str(candidate_family or "")
        target_family = str(target_family or "")
        return (
            candidate_family == target_family
            or candidate_family.startswith(f"{target_family}_")
            or target_family.startswith(f"{candidate_family}_")
        )

    @staticmethod
    def _coarse_family_key(family: str) -> str:
        family = str(family or "")
        known_prefixes = (
            "cand_lightgcn_shallow_alignment_rankaware_gate",
            "cand_lightgcn_shallow_rankaware_lastlayeralign",
            "cand_lightgcn_edge_dropout_residual_norm",
            "cand_lightgcn_residual_norm_constrained",
            "cand_bpr_hard_negative_margin",
            "cand_lightgcn_shallow_layers",
        )
        for prefix in known_prefixes:
            if family == prefix or family.startswith(f"{prefix}_"):
                return prefix
        parts = family.split("_")
        return "_".join(parts[:4]) if len(parts) >= 4 else family

    def _metric_trial_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(self._trial_memory_rows()):
            if str(row.get("status") or "").lower() != "success":
                continue
            value = self._to_float((row.get("result") or {}).get(self.config.metric))
            if value is None:
                continue
            rows.append(
                {
                    "index": index,
                    "round_id": int(row.get("round_id") or index + 1),
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "family": self._family_key(row),
                    "value": value,
                    "decision": str(row.get("decision") or ""),
                }
            )
        return rows

    def _plateau_state(self) -> dict[str, Any]:
        rows = self._metric_trial_rows()
        window = max(1, self.config.plateau_window_metric_rows)
        overuse_window = max(1, self.config.plateau_family_overuse_window)
        min_improvement = max(0.0, self.config.plateau_min_global_improvement)
        default_forced = list(self.config.anchor_families)
        if not self.config.anti_plateau_enabled or len(rows) <= window:
            return {
                "enabled": bool(self.config.anti_plateau_enabled),
                "plateau_detected": False,
                "metric_rows": len(rows),
                "window_metric_rows": window,
                "forced_exploration_families": default_forced,
                "local_repair_capped_families": [],
            }

        values = [float(row["value"]) for row in rows]
        best_value = max(values)
        best_index = values.index(best_value)
        best_row = rows[best_index]
        prior_best = max(values[:-window])
        recent_rows = rows[-window:]
        recent_best = max(float(row["value"]) for row in recent_rows)
        recent_improvement = recent_best - prior_best
        rows_since_best = len(rows) - best_index - 1

        recent_family_counts: dict[str, int] = {}
        recent_cluster_counts: dict[str, int] = {}
        for row in rows[-overuse_window:]:
            family = str(row["family"])
            recent_family_counts[family] = recent_family_counts.get(family, 0) + 1
            cluster = self._coarse_family_key(family)
            recent_cluster_counts[cluster] = recent_cluster_counts.get(cluster, 0) + 1

        capped_families: list[str] = []
        for family, count in sorted(recent_cluster_counts.items(), key=lambda item: (-item[1], item[0])):
            overused = count >= self.config.max_same_family_repair_streak
            weak_recent_frontier = recent_best <= self.config.plateau_weak_family_ceiling
            stale_global_best = rows_since_best >= window
            is_anchor = any(self._family_matches(family, anchor) for anchor in self.config.anchor_families)
            if overused and stale_global_best and (weak_recent_frontier or not is_anchor):
                capped_families.append(family)

        plateau_detected = (
            rows_since_best >= window
            and recent_improvement < min_improvement
        ) or bool(capped_families)
        if plateau_detected and rows_since_best >= window:
            stale_best_family = self._coarse_family_key(str(best_row.get("family") or ""))
            stale_best_is_anchor = any(
                self._family_matches(stale_best_family, anchor)
                for anchor in self.config.anchor_families
            )
            if stale_best_family and not stale_best_is_anchor and stale_best_family not in capped_families:
                capped_families.append(stale_best_family)
        forced = [
            family
            for family in self.config.anchor_families
            if not any(self._family_matches(family, capped) for capped in capped_families)
        ] or default_forced
        return {
            "enabled": True,
            "plateau_detected": bool(plateau_detected),
            "metric_rows": len(rows),
            "window_metric_rows": window,
            "rows_since_best": rows_since_best,
            "best_metric": round(best_value, 6),
            "best_round_id": best_row.get("round_id"),
            "best_family": best_row.get("family"),
            "recent_best": round(recent_best, 6),
            "recent_improvement": round(recent_improvement, 6),
            "local_repair_capped_families": capped_families[:10],
            "recent_family_counts": dict(sorted(recent_family_counts.items(), key=lambda item: (-item[1], item[0]))[:10]),
            "recent_family_clusters": dict(sorted(recent_cluster_counts.items(), key=lambda item: (-item[1], item[0]))[:10]),
            "forced_exploration_families": forced,
            "instruction": (
                "Plateau detected: cap local repairs on overused families and force cross-family "
                "algorithm proposals around residual-norm, edge-drop residual-norm, hard-negative margin, "
                "or other architecture-level mechanisms."
            )
            if plateau_detected
            else "",
        }

    def _algorithm_first_score_adjustment(self, candidate: dict[str, Any]) -> float:
        if self.config.search_intensity != "algorithm_first":
            return 0.0
        if str(candidate.get("candidate_id") or "") in self._pending_implemented_candidate_ids():
            return 0.0
        adjustment = 0.0
        family = str(candidate.get("parent_candidate_id") or candidate.get("candidate_id") or "")
        plateau = self._plateau_state()
        if plateau.get("plateau_detected"):
            capped = [str(item) for item in plateau.get("local_repair_capped_families", [])]
            forced = [str(item) for item in plateau.get("forced_exploration_families", [])]
            if any(self._family_matches(family, item) for item in capped):
                adjustment -= self.config.anti_plateau_local_family_penalty
            if any(self._family_matches(family, item) for item in forced):
                adjustment += self.config.anti_plateau_anchor_bonus
        if not self._has_strong_algorithm_signal():
            return adjustment
        if candidate.get("parent_candidate_id"):
            return adjustment
        if self._normalize_model_key(candidate.get("base_model")) == "BPR":
            adjustment -= 1.0
        return adjustment

    def _plateau_allows_family(self, family: str) -> bool:
        if self.config.search_intensity != "algorithm_first":
            return True
        plateau = self._plateau_state()
        if not plateau.get("plateau_detected"):
            return True
        family = str(family or "")
        capped = [str(item) for item in plateau.get("local_repair_capped_families", [])]
        forced = [str(item) for item in plateau.get("forced_exploration_families", [])]
        if any(self._family_matches(family, item) for item in forced):
            return True
        return not any(self._family_matches(family, item) for item in capped)

    def plan(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        plan_options: list[tuple[dict[str, Any], dict[str, Any] | None]] = [
            (item, None)
            for item in self.registry
            if bool(item.get("wired")) and str(item.get("runner_type") or "") in {"config_only", "model"}
            and str(item.get("candidate_id") or "") not in self.quarantined_candidate_ids
        ]
        plan_options.extend(self._accepted_proposal_options())
        if not plan_options:
            raise RuntimeError("no runnable candidates found in registry")
        pending_ids = self._pending_implemented_candidate_ids()
        if len(pending_ids) >= self.config.max_pending_implemented:
            pending_options = [
                (candidate, params)
                for candidate, params in plan_options
                if str(candidate.get("candidate_id") or "") in pending_ids
            ]
            if pending_options:
                plan_options = pending_options
        if (
            self.config.search_intensity == "algorithm_first"
            and self._formal_trial_count() < self.config.algorithm_first_explore_rounds
        ):
            implemented_algorithm_ids = self._implemented_algorithm_candidate_ids()
            algorithm_options = [
                (candidate, params)
                for candidate, params in plan_options
                if str(candidate.get("candidate_id") or "") in implemented_algorithm_ids
                or str(candidate.get("proposal_type") or "") == "algorithmic_variant"
                or str(candidate.get("runnable_level") or "") == "code_required"
            ]
            if algorithm_options:
                plan_options = algorithm_options
        if str(self.last_planner_action.get("action") or "") == "tune_after_algorithm_success":
            validated_focus = self._best_validated_keep()
            if validated_focus:
                focused_options = [
                    (candidate, params)
                    for candidate, params in plan_options
                    if self._candidate_matches_validated_focus(candidate, validated_focus)
                ]
                if focused_options:
                    plan_options = focused_options

        scored_options = []
        for candidate, params_override in plan_options:
            pending_bonus = 1.0 if str(candidate.get("candidate_id") or "") in pending_ids else 0.0
            algorithm_bonus = self._algorithm_first_score_adjustment(candidate)
            route = route_candidate_option(
                candidate,
                base_score=self._candidate_plan_score(candidate),
                search_memory=self.search_memory,
                pending_bonus=pending_bonus,
                algorithm_bonus=algorithm_bonus,
            )
            candidate["route_decision"] = route.as_dict()
            scored_options.append((route.score, candidate, params_override))
        scored_options.sort(key=lambda item: item[0], reverse=True)
        used_execution_signatures = self._used_execution_signatures()
        fallback_choice: tuple[dict[str, Any], dict[str, Any]] | None = None
        chosen, params = scored_options[0][1], scored_options[0][2] or {}
        for _, candidate, params_override in scored_options:
            config_params = self._candidate_config_params(candidate)
            candidate_params = (
                params_override
                if params_override is not None
                else (
                    config_params
                    if str(candidate.get("candidate_id") or "") in pending_ids and config_params
                    else self._choose_execution_unique_params(candidate, used_execution_signatures)
                )
            )
            if fallback_choice is None:
                fallback_choice = (candidate, candidate_params)
            execution_signature = self._execution_signature(candidate, candidate_params)
            if execution_signature and (
                execution_signature in used_execution_signatures
                or execution_signature in self.scheduled_execution_signatures
            ):
                continue
            chosen, params = candidate, candidate_params
            chosen["execution_signature"] = execution_signature
            break
        else:
            chosen, params = fallback_choice or (scored_options[0][1], scored_options[0][2] or {})
            chosen["execution_signature"] = self._execution_signature(chosen, params)
        need_context = not self.history_by_candidate.get(str(chosen.get("candidate_id") or ""))
        context = self._load_optional_context(chosen, force=need_context)
        return chosen, params, context

    def _write_research_route(self, round_id: int, candidate: dict[str, Any], params: dict[str, Any]) -> None:
        if not self.config.research_line_enabled:
            return
        try:
            self.config.research_route_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "round_id": round_id,
                "candidate_id": candidate.get("candidate_id", ""),
                "parent_candidate_id": candidate.get("parent_candidate_id", ""),
                "producer_id": candidate.get("producer_id", ""),
                "producer_role": candidate.get("producer_role", ""),
                "params": params,
                "route_decision": candidate.get("route_decision", {}),
            }
            with self.config.research_route_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        except OSError as exc:
            print(f"[ResearchLine] route trace write skipped: {exc}", file=sys.stderr)

    # ===== Act =====
    def act(self, candidate: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id"))
        run_candidate_id = str(candidate.get("run_candidate_id") or candidate_id)
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_candidate.py"), run_candidate_id]
        cmd.extend(["--registry-path", str(self.config.registry_path)])
        cmd.extend(["--results-csv", str(self.config.results_csv)])
        run_root = self.config.results_csv.parent
        cmd.extend(["--result-dir", str(run_root / "candidates")])
        cmd.extend(["--override-dir", str(run_root / "overrides")])
        if self.config.checkpoint_dir:
            cmd.extend(["--checkpoint-dir", self.config.checkpoint_dir])
        if self.config.checkpoint_policy != "none":
            cmd.append("--checkpoint-per-run")
        for item in self.config.global_overrides:
            cmd.extend(["--set", item])
        for key, value in params.items():
            cmd.extend(["--set", f"{key}={value}"])

        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        summary = self._extract_last_json(completed.stdout)
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "summary": summary,
            "run_candidate_id": run_candidate_id,
        }

    def _checkpoint_root(self) -> Path:
        if self.config.checkpoint_dir:
            return Path(self.config.checkpoint_dir).expanduser().resolve()
        return PROJECT_ROOT / "results" / "checkpoints"

    def cleanup_run_checkpoints(self, run_ids: list[str]) -> None:
        if self.config.checkpoint_policy == "none":
            return
        root = self._checkpoint_root()
        for run_id in run_ids:
            if not run_id:
                continue
            path = root / run_id
            try:
                resolved_path = path.resolve()
                resolved_root = root.resolve()
                resolved_path.relative_to(resolved_root)
                if resolved_path != resolved_root and resolved_path.exists():
                    shutil.rmtree(resolved_path)
            except OSError as exc:
                print(f"[Checkpoint] cleanup skipped for {run_id}: {exc}", file=sys.stderr)
            except ValueError:
                print(f"[Checkpoint] refuse cleanup outside checkpoint root: {path}", file=sys.stderr)

    def retain_checkpoints_for_result(
        self,
        *,
        decision: str,
        run_id: str,
        seed_validation: dict[str, Any],
    ) -> None:
        policy = self.config.checkpoint_policy
        if policy == "none":
            return
        validation_run_ids = [
            str(item)
            for item in seed_validation.get("run_ids", [])
            if isinstance(item, str) and item
        ]
        all_run_ids = [run_id, *validation_run_ids]
        validation_status = str(seed_validation.get("status") or "")
        retained_run_ids: set[str] = set()
        if policy == "keep_any":
            if decision == "keep":
                retained_run_ids.update(item for item in all_run_ids if item)
        elif policy == "keep_validated":
            if decision == "keep" and validation_status == "passed":
                best_run_id = self._best_seed_validation_run_id(seed_validation)
                if best_run_id:
                    retained_run_ids.add(best_run_id)
            elif decision == "keep" and not self.config.enable_seed_validation and run_id:
                retained_run_ids.add(run_id)
        for item in sorted(retained_run_ids):
            print(f"[Checkpoint] retained={item}")
        self.cleanup_run_checkpoints(
            [item for item in all_run_ids if item not in retained_run_ids],
        )

    def _best_seed_validation_run_id(self, seed_validation: dict[str, Any]) -> str:
        runs = seed_validation.get("runs")
        if not isinstance(runs, list):
            return ""
        scored: list[tuple[float, str]] = []
        direction = self.config.metric_directions.get(self.config.metric, 1)
        for item in runs:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id") or "")
            value = self._to_float(item.get("value"))
            if not run_id or value is None:
                continue
            scored.append((direction * value, run_id))
        if not scored:
            return ""
        return max(scored, key=lambda item: item[0])[1]

    @staticmethod
    def _extract_last_json(text: str) -> dict[str, Any]:
        lines = [line for line in text.splitlines() if line.strip()]
        for idx in range(len(lines) - 1, -1, -1):
            chunk = "\n".join(lines[idx:])
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _load_candidate_proposals(self) -> list[dict[str, Any]]:
        if not self.config.proposal_path.exists():
            return []
        proposals: list[dict[str, Any]] = []
        for line in self.config.proposal_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                proposals.append(parsed)
        return proposals

    def _notes_excerpt(self, names: tuple[str, ...], limit: int = 12000) -> dict[str, str]:
        excerpts: dict[str, str] = {}
        for name in names:
            path = NOTES_DIR / name
            if path.exists():
                excerpts[name] = path.read_text(encoding="utf-8", errors="replace")[-limit:]
        return excerpts

    def _trial_memory_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.memory if not row.get("event") and "decision" in row]

    @staticmethod
    def _family_key(row: dict[str, Any]) -> str:
        return str(row.get("parent_candidate_id") or row.get("candidate_id") or "unknown")

    def _unverified_keep_rows(self) -> list[dict[str, Any]]:
        completed_targets = self._seed_validation_completed_target_keys()
        return [
            row
            for row in self._trial_memory_rows()
            if row.get("decision") == "keep"
            and not row.get("seed_validation")
            and self._seed_validation_eligible(row)
            and self._seed_validation_target_key(row) not in completed_targets
        ]

    def _seed_validation_eligible(self, row: dict[str, Any]) -> bool:
        if self.config.seed_validation_min_metric <= 0:
            return True
        metric_value = self._to_float((row.get("result") or {}).get(self.config.metric))
        if metric_value is None:
            return False
        return metric_value >= self.config.seed_validation_min_metric

    def _seed_validation_target_key(self, row: dict[str, Any]) -> str:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            return f"candidate::{candidate_id}"
        signature = self._canonical_parameter_signature_text(row.get("parameter_signature"))
        if signature:
            return f"signature::{signature}"
        parent_id = str(row.get("parent_candidate_id") or row.get("candidate_id") or "").strip()
        params = row.get("params") or row.get("parameter_overrides")
        if parent_id and isinstance(params, dict) and params:
            return f"signature::{self._parameter_signature(parent_id, params)}"
        return ""

    def _seed_validated_target_keys(self) -> set[str]:
        validated: set[str] = set()
        for row in self.memory:
            validation = row.get("seed_validation")
            if not isinstance(validation, dict) or not validation:
                continue
            if str(validation.get("status") or "").lower() != "passed":
                continue
            key = self._seed_validation_target_key(row)
            if key:
                validated.add(key)
            signature = self._canonical_parameter_signature_text(row.get("parameter_signature"))
            if signature:
                validated.add(f"signature::{signature}")
        return validated

    def _seed_validation_completed_target_keys(self) -> set[str]:
        completed: set[str] = set()
        for row in self.memory:
            validation = row.get("seed_validation")
            if not isinstance(validation, dict) or not validation:
                continue
            if str(validation.get("status") or "").lower() not in {"passed", "failed"}:
                continue
            key = self._seed_validation_target_key(row)
            if key:
                completed.add(key)
            signature = self._canonical_parameter_signature_text(row.get("parameter_signature"))
            if signature:
                completed.add(f"signature::{signature}")
        return completed

    def _best_validated_keep(self) -> dict[str, Any]:
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        validation_by_key: dict[str, dict[str, Any]] = {}
        for row in self.memory:
            validation = row.get("seed_validation")
            if not isinstance(validation, dict) or validation.get("status") != "passed":
                continue
            key = self._seed_validation_target_key(row)
            if key:
                validation_by_key[key] = validation
            signature = self._canonical_parameter_signature_text(row.get("parameter_signature"))
            if signature:
                validation_by_key[f"signature::{signature}"] = validation
        for idx, row in enumerate(self._trial_memory_rows()):
            validation = row.get("seed_validation")
            if not isinstance(validation, dict) or validation.get("status") != "passed":
                keys = [self._seed_validation_target_key(row)]
                signature = self._canonical_parameter_signature_text(row.get("parameter_signature"))
                if signature:
                    keys.append(f"signature::{signature}")
                validation = next((validation_by_key[key] for key in keys if key in validation_by_key), {})
            if not isinstance(validation, dict) or validation.get("status") != "passed":
                continue
            value = self._to_float(validation.get("mean"))
            if value is None:
                value = self._to_float((row.get("result") or {}).get(self.config.metric))
            if value is None:
                continue
            candidates.append((value, idx, {**row, "seed_validation": validation}))
        if not candidates:
            return {}
        return max(candidates, key=lambda item: item[0])[2]

    @staticmethod
    def _candidate_matches_validated_focus(candidate: dict[str, Any], validated: dict[str, Any]) -> bool:
        focus_id = str(validated.get("candidate_id") or "").strip()
        if not focus_id:
            return False
        candidate_ids = {
            str(candidate.get("candidate_id") or "").strip(),
            str(candidate.get("run_candidate_id") or "").strip(),
            str(candidate.get("proposal_id") or "").strip(),
        }
        if focus_id in candidate_ids:
            return True
        # Post-success refinement should stay on the validated algorithm or
        # direct repair/ablation children, not fall through to unrelated
        # runnable baselines when a proposal is rejected or smoke fails.
        return str(candidate.get("parent_candidate_id") or "").strip() == focus_id

    def _post_validation_followup_state(self) -> dict[str, Any]:
        validated = self._best_validated_keep()
        if not validated:
            return {"enabled": False}

        focus_id = str(validated.get("candidate_id") or "").strip()
        if not focus_id:
            return {"enabled": False}

        validation_metric = self._to_float((validated.get("seed_validation") or {}).get("mean"))
        if validation_metric is None:
            validation_metric = self._to_float((validated.get("result") or {}).get(self.config.metric))
        validation_round = int(validated.get("round_id") or 0)
        target_key = self._seed_validation_target_key(validated)
        for row in self.memory:
            validation = row.get("seed_validation")
            if not isinstance(validation, dict) or validation.get("status") != "passed":
                continue
            if target_key and self._seed_validation_target_key(row) != target_key:
                continue
            try:
                validation_round = max(validation_round, int(row.get("round_id") or 0))
            except (TypeError, ValueError):
                pass

        trials = self._trial_memory_rows()
        latest_round = max([int(row.get("round_id") or 0) for row in trials] or [validation_round])
        window = max(1, self.config.post_validation_structured_followup_window)
        followups: list[dict[str, Any]] = []
        for row in trials:
            try:
                round_id = int(row.get("round_id") or 0)
            except (TypeError, ValueError):
                continue
            if round_id <= validation_round:
                continue
            candidate_id = str(row.get("candidate_id") or "").strip()
            parent_id = str(row.get("parent_candidate_id") or "").strip()
            if candidate_id == focus_id or parent_id == focus_id:
                followups.append(row)
        recent_followups = followups[-window:]
        values = [
            value
            for row in recent_followups
            if (value := self._to_float((row.get("result") or {}).get(self.config.metric))) is not None
        ]
        best_followup = max(values) if values else None
        improvement = (
            best_followup - validation_metric
            if best_followup is not None and validation_metric is not None
            else None
        )
        needs_structured = (
            len(recent_followups) >= max(1, self.config.post_validation_sibling_churn_limit)
            and improvement is not None
            and improvement < self.config.post_validation_min_followup_improvement
        )
        return {
            "enabled": True,
            "candidate_id": focus_id,
            "parent_candidate_id": validated.get("parent_candidate_id") or focus_id,
            "validation_round_id": validation_round,
            "latest_round_id": latest_round,
            "validation_metric": round(validation_metric, 6) if validation_metric is not None else None,
            "recent_followup_trials": len(recent_followups),
            "best_followup_metric": round(best_followup, 6) if best_followup is not None else None,
            "followup_improvement": round(improvement, 6) if improvement is not None else None,
            "needs_structured_followup": bool(needs_structured),
            "guidance": (
                "Validated best has enough non-improving local siblings; switch to one-axis ablation, "
                "seed stability checks, or parameter-region refinement instead of generating more nearby code variants."
            )
            if needs_structured
            else "Validated best can still accept focused follow-up, but prefer diagnostic variants over free-form sibling churn.",
        }

    def _recent_parameter_signatures(self, limit: int = 120) -> list[str]:
        signatures: list[str] = []
        seen: set[str] = set()
        for row in reversed(self.memory):
            signature = str(row.get("parameter_signature") or "")
            signature = self._canonical_parameter_signature_text(signature)
            if not signature or signature in seen:
                continue
            signatures.append(signature)
            seen.add(signature)
            if len(signatures) >= limit:
                break
        return list(reversed(signatures))

    @staticmethod
    def _crash_taxonomy(row: dict[str, Any]) -> str:
        text = " ".join(
            str(value)
            for value in (
                row.get("reason"),
                (row.get("compare_baseline") or {}).get("explanation")
                if isinstance(row.get("compare_baseline"), dict)
                else "",
            )
        ).lower()
        if "config.get" in text or "has no attribute 'get'" in text:
            return "recbole_config_get"
        if "reg_loss" in text:
            return "missing_reg_loss"
        if "entrypoint" in text or "import" in text or "attributeerror" in text:
            return "implementation_wiring"
        if "timeout" in text:
            return "timeout"
        return "other"

    def _is_extreme_quality_collapse(self, row: dict[str, Any]) -> bool:
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        value = self._to_float(result.get(self.config.metric))
        if value is None:
            return False
        compare = row.get("compare_baseline") if isinstance(row.get("compare_baseline"), dict) else {}
        baseline = self._to_float(compare.get("baseline_metric"))
        if baseline is None:
            return value < 0.05
        return value < max(0.05, baseline * 0.60)

    def _family_stats_summary(self) -> list[dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for row in self._trial_memory_rows():
            family = self._family_key(row)
            item = stats.setdefault(
                family,
                {
                    "family": family,
                    "trials": 0,
                    "success": 0,
                    "keep": 0,
                    "revise": 0,
                    "discard": 0,
                    "crash": 0,
                    "collapse": 0,
                    "baseline_win": 0,
                    "best_metric": None,
                    "avg_metric": None,
                    "_metric_sum": 0.0,
                    "_metric_count": 0,
                    "last_error_type": "",
                },
            )
            item["trials"] += 1
            decision = str(row.get("decision") or "")
            if decision in {"keep", "revise", "discard", "crash"}:
                item[decision] += 1
            if str(row.get("status") or "").lower() == "success":
                item["success"] += 1
            if decision == "crash":
                item["last_error_type"] = self._crash_taxonomy(row)
            if self._is_extreme_quality_collapse(row):
                item["collapse"] += 1
            value = self._to_float((row.get("result") or {}).get(self.config.metric))
            if value is not None:
                item["_metric_sum"] += value
                item["_metric_count"] += 1
                compare = row.get("compare_baseline") if isinstance(row.get("compare_baseline"), dict) else {}
                delta = self._to_float(compare.get("delta"))
                if delta is not None and delta > self.config.min_keep_delta:
                    item["baseline_win"] += 1
                if item["best_metric"] is None or value > item["best_metric"]:
                    item["best_metric"] = round(value, 6)
        summaries = []
        for item in stats.values():
            if item["_metric_count"]:
                item["avg_metric"] = round(item["_metric_sum"] / item["_metric_count"], 6)
            item.pop("_metric_sum", None)
            item.pop("_metric_count", None)
            summaries.append(item)
        return sorted(summaries, key=lambda item: (item.get("best_metric") is not None, item.get("best_metric") or -1), reverse=True)

    def _best_trial_metric(self) -> float | None:
        values: list[float] = []
        for row in self._trial_memory_rows():
            if str(row.get("status") or "").lower() != "success":
                continue
            value = self._to_float((row.get("result") or {}).get(self.config.metric))
            if value is not None:
                values.append(value)
        return max(values) if values else None

    def _has_strong_algorithm_signal(self) -> bool:
        best = self._best_trial_metric()
        return best is not None and best >= self.config.tuned_lightgcn_mean

    def _algorithm_fallback_action(self) -> str:
        pending_count = len(self._pending_implemented_candidate_ids())
        if pending_count:
            return "run_algorithm_variant" if self.config.search_intensity == "algorithm_first" else "run_available"
        if self._has_pending_code_required_review():
            return "implement_algorithm" if self.config.search_intensity == "algorithm_first" else "implement_needs_review"
        if self.config.search_intensity == "algorithm_first" and self._plateau_state().get("plateau_detected"):
            return "propose_algorithm"
        return "propose_algorithm" if self.config.search_intensity == "algorithm_first" else "propose_mixed"

    def _build_agent_state_summary(self) -> dict[str, Any]:
        trials = self._trial_memory_rows()
        decision_counts: dict[str, int] = {}
        for row in trials:
            decision = str(row.get("decision") or "unknown")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        scored_trials: list[tuple[float, dict[str, Any]]] = []
        for row in trials:
            if str(row.get("status") or "").lower() != "success":
                continue
            value = self._to_float((row.get("result") or {}).get(self.config.metric))
            if value is not None:
                scored_trials.append((value, row))
        top_frontier = [
            {
                "candidate_id": row.get("candidate_id", ""),
                "parent_candidate_id": row.get("parent_candidate_id", ""),
                "params": row.get("params", {}),
                "run_id": row.get("run_id", ""),
                "decision": row.get("decision", ""),
                self.config.metric: round(value, 6),
            }
            for value, row in sorted(scored_trials, key=lambda item: item[0], reverse=True)[:10]
        ]
        best_by_model_summary = {
            model: {
                "run_id": row.get("run_id", ""),
                "model": row.get("model", model),
                self.config.metric: self._to_float(row.get(self.config.metric)),
                "status": row.get("status", ""),
            }
            for model, row in sorted(self.best_by_model.items())
        }

        validated_best = self._best_validated_keep()
        focused_context: dict[str, Any] = {}
        if validated_best:
            round_id = int(validated_best.get("round_id") or 0)
            latest_round = max([int(row.get("round_id") or 0) for row in trials] or [0])
            if latest_round - round_id <= self.config.exploitation_window_rounds:
                focused_context = {
                    "parent_candidate_id": validated_best.get("parent_candidate_id") or validated_best.get("candidate_id", ""),
                    "candidate_id": validated_best.get("candidate_id", ""),
                    "params": validated_best.get("params", {}),
                    "seed_validation": validated_best.get("seed_validation", {}),
                    "guidance": "Prioritize local exploitation around this validated best while keeping at least one genuinely novel proposal.",
                }

        proposal_rejections = [
            row
            for row in self.memory
            if row.get("event") == "proposal_rejected"
        ][-30:]
        duplicate_rejections = [
            row.get("parameter_signature")
            for row in proposal_rejections
            if any("already run" in str(error) for error in (row.get("errors") or []))
        ]
        family_stats = self._family_stats_summary()
        anchor_family_set = set(self.config.anchor_families)
        anchor_family_stats = [
            item for item in family_stats if str(item.get("family") or "") in anchor_family_set
        ]
        bad_families = [
            item
            for item in family_stats
            if (
                item.get("crash", 0) >= 2
                or item.get("collapse", 0) >= 2
                or (
                    str(item.get("family") or "") not in anchor_family_set
                    and item.get("trials", 0) >= 4
                    and item.get("keep", 0) == 0
                    and item.get("baseline_win", 0) == 0
                )
            )
        ][:10]
        pending_implemented = sorted(self._pending_implemented_candidate_ids())
        best_metric = self._best_trial_metric()
        plateau_state = self._plateau_state()
        search_memory_summary = {
            "trial_count": self.search_memory.get("trial_count", 0),
            "best_metric": self.search_memory.get("best_metric"),
            "best_family": self.search_memory.get("best_family", ""),
            "promising_families": self.search_memory.get("promising_families", [])[:10],
            "frozen_families": self.search_memory.get("frozen_families", [])[:10],
            "duplicate_signatures": self.search_memory.get("duplicate_signatures", [])[:20],
            "blocker_counts": dict(list((self.search_memory.get("blocker_counts") or {}).items())[:10]),
            "producer_stats": list((self.search_memory.get("producer_stats") or {}).values())[:10],
        }
        meta_update = meta_research_update(self.search_memory) if self.config.research_line_enabled else {}
        post_validation_followup = self._post_validation_followup_state()
        if focused_context and post_validation_followup.get("needs_structured_followup"):
            focused_context["guidance"] = post_validation_followup["guidance"]
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "metric": self.config.metric,
            "search_intensity": self.config.search_intensity,
            "best_metric_so_far": round(best_metric, 6) if best_metric is not None else None,
            "experiment_directive": self._experiment_directive_context(),
            "anti_plateau": plateau_state,
            "research_line": {
                "enabled": bool(self.config.research_line_enabled),
                "search_memory": search_memory_summary,
                "meta_research_advisory": meta_update,
            },
            "trial_count": len(trials),
            "decision_counts": decision_counts,
            "validated_best": {
                "candidate_id": validated_best.get("candidate_id", ""),
                "parent_candidate_id": validated_best.get("parent_candidate_id", ""),
                "params": validated_best.get("params", {}),
                "run_id": validated_best.get("run_id", ""),
                "seed_validation": validated_best.get("seed_validation", {}),
            }
            if validated_best
            else {},
            "focused_exploitation": focused_context,
            "post_validation_followup": post_validation_followup,
            "top_frontier": top_frontier,
            "best_by_model": best_by_model_summary,
            "family_stats": family_stats[:20],
            "anchor_family_stats": anchor_family_stats[:20],
            "bad_families": bad_families,
            "quarantined_candidates": [
                {"candidate_id": candidate_id, "issues": issues}
                for candidate_id, issues in sorted(self.candidate_health_issues.items())
            ][:50],
            "pending_implemented_candidates": pending_implemented[:50],
            "recent_forbidden_parameter_signatures": self._recent_parameter_signatures(limit=120),
            "recent_duplicate_rejections": [item for item in duplicate_rejections if item][-30:],
            "planner_gates": {
                "unverified_keep_count": len(self._unverified_keep_rows()),
                "pending_implemented_count": len(pending_implemented),
                "quarantined_count": len(self.quarantined_candidate_ids),
                "max_pending_implemented": self.config.max_pending_implemented,
            },
            "algorithm_first_targets": {
                "enabled": self.config.search_intensity == "algorithm_first",
                "tuned_lightgcn_mean": self.config.tuned_lightgcn_mean,
                "old_recclaw_first50_best": 0.2765,
                "anchor_families": list(self.config.anchor_families),
                "algorithm_budget_per_window": self.config.algorithm_budget_per_window,
                "algorithm_first_explore_rounds": self.config.algorithm_first_explore_rounds,
                "seed_validation_min_metric": self.config.seed_validation_min_metric,
                "guidance": (
                    "If best_metric_so_far is below tuned_lightgcn_mean, prioritize propose_algorithm, "
                    "implement_algorithm, and run_algorithm_variant over parameter-only tuning. "
                    "Planner-triggered seed validation is reserved for keeps meeting seed_validation_min_metric. "
                    "If anti_plateau.plateau_detected is true, cap local repairs on "
                    "anti_plateau.local_repair_capped_families and force cross-family algorithm exploration "
                    "around anti_plateau.forced_exploration_families. "
                    "If post_validation_followup.needs_structured_followup is true, stop adding nearby code "
                    "siblings for the validated best and switch to one-axis ablation, stability replication, "
                    "or parameter-region refinement."
                ),
            },
        }
        return summary

    def _write_agent_state_summary(self, summary: dict[str, Any]) -> None:
        try:
            self.config.state_summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.state_summary_path.write_text(
                json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[StateSummary] write skipped: {exc}", file=sys.stderr)

    def remember_event(self, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
        self.memory.append(event)

    def _active_experiment_directive(self) -> str:
        if not self.config.use_experiment_directive:
            return ""
        return str(self.config.experiment_directive or "").strip()

    def _experiment_directive_context(self) -> dict[str, Any]:
        directive = self._active_experiment_directive()
        if not directive:
            return {"enabled": False}
        return {
            "enabled": True,
            "priority": "highest user steering instruction below hard_constraints and schema validity",
            "text": directive,
        }

    def remember_experiment_directive(self) -> None:
        directive = self._active_experiment_directive()
        if not directive:
            return
        if directive == self.last_experiment_directive_digest:
            return
        self.remember_event(
            {
                "event": "experiment_directive",
                "priority": "highest_user_steering_below_hard_constraints",
                "text": directive,
            }
        )
        self.last_experiment_directive_digest = directive

    def _refresh_experience_artifacts(self, round_id: int, *, reason: str) -> bool:
        if self.config.refresh_experience_every <= 0:
            return False
        if not self.config.use_experiment_directive:
            return False
        if self.config.dry_run:
            print(f"[Reflection] dry-run skip refresh round={round_id} reason={reason}")
            return False

        runtime_paths = (
            self.config.registry_path,
            self.config.proposal_path,
            self.config.memory_path,
            self.config.results_csv,
            self.config.candidate_tree_path,
            self.config.candidate_tree_md_path,
            self.config.candidate_tree_mmd_path,
            self.config.experience_summary_path,
            self.config.experience_summary_json_path,
            self.config.reflection_memory_path,
        )
        for path in runtime_paths:
            reject_runtime_lablog_path(path)

        tree_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analysis" / "build_candidate_search_tree.py"),
            "--registry",
            str(self.config.registry_path),
            "--proposals",
            str(self.config.proposal_path),
            "--memory",
            str(self.config.memory_path),
            "--results",
            str(self.config.results_csv),
            "--metric",
            self.config.metric,
            "--out-json",
            str(self.config.candidate_tree_path),
            "--out-md",
            str(self.config.candidate_tree_md_path),
            "--out-mmd",
            str(self.config.candidate_tree_mmd_path),
        ]
        summary_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_experience_summary.py"),
            "--memory",
            str(self.config.memory_path),
            "--results",
            str(self.config.results_csv),
            "--tree",
            str(self.config.candidate_tree_path),
            "--out-md",
            str(self.config.experience_summary_path),
            "--out-json",
            str(self.config.experience_summary_json_path),
            "--out-jsonl",
            str(self.config.reflection_memory_path),
        ]

        print(f"[Reflection] refresh round={round_id} reason={reason}")
        for cmd in (tree_cmd, summary_cmd):
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.stdout.strip():
                print(completed.stdout.strip())
            if completed.returncode != 0:
                if completed.stderr.strip():
                    print(completed.stderr.strip(), file=sys.stderr)
                raise RuntimeError(f"experience refresh failed: {' '.join(cmd)}")

        self.config.experiment_directive = resolve_experiment_directive(
            directive=self.config.static_experiment_directive,
            directive_file=self.config.experience_summary_path,
            disabled=False,
        )
        self.config.use_experiment_directive = bool(self.config.experiment_directive)
        self.remember_event(
            {
                "event": "experience_refresh",
                "round_id": round_id,
                "reason": reason,
                "tree": str(self.config.candidate_tree_path),
                "summary": str(self.config.experience_summary_path),
            }
        )
        self.remember_experiment_directive()
        return True

    def _llm_proposal_context(self) -> dict[str, Any]:
        schema = load_yaml(self.config.proposal_schema_path)
        self.agent_state_summary = self._build_agent_state_summary()
        self._write_agent_state_summary(self.agent_state_summary)
        tail_limit = max(0, min(self.config.memory_read_limit or len(self.memory), self.config.prompt_memory_tail))
        memory = self.memory[-tail_limit:] if tail_limit > 0 else []
        results_tail = self.results_rows[-50:]
        pending_implemented = sorted(self._pending_implemented_candidate_ids())
        producer_directives = build_producer_directives(
            search_policy=self.search_policy,
            search_memory=self.search_memory,
            proposal_count=self.config.proposal_count,
            mode=self.config.proposal_mode,
        )
        return {
            "task": "Generate RecClaw candidate proposals for algorithm discovery.",
            "loop_mode": self.config.loop_mode,
            "proposal_mode": self.config.proposal_mode,
            "proposal_count": self.config.proposal_count,
            "search_intensity": self.config.search_intensity,
            "algorithm_budget_per_window": self.config.algorithm_budget_per_window,
            "anchor_families": list(self.config.anchor_families),
            "tuned_lightgcn_mean": self.config.tuned_lightgcn_mean,
            "steering_priority": [
                "hard_constraints and schema validity",
                "experiment_directive when enabled",
                "agent_state_summary and memory",
                "default proposal/search policy",
            ],
            "experiment_directive": self._experiment_directive_context(),
            "pending_implemented_count": len(pending_implemented),
            "pending_implemented_candidates": pending_implemented[:50],
            "candidate_producers": producer_directives_payload(producer_directives),
            "action_space": ACTION_SPACE,
            "parameter_space": self.config.parameter_space,
            "schema": schema,
            "registry": self.registry,
            "agent_state_summary": self.agent_state_summary,
            "recent_agent_memory": memory,
            "recent_results": results_tail,
            "notes": self._notes_excerpt(
                (
                    "candidate_proposal_workflow.md",
                    "experiment_log.md",
                )
            ),
            "hard_constraints": [
                "Return only JSON, no Markdown.",
                "Return an object with a proposals list, or a JSON array.",
                "Each proposal must satisfy configs/candidate_proposal_schema.yaml.",
                "Do not require RecBole core changes.",
                "If experiment_directive.enabled is true, follow experiment_directive.text as the highest-priority user steering instruction unless it conflicts with these hard constraints.",
                "For mixed mode, include both runnable tuning proposals and code_required/spec_only exploration when useful.",
                "For algorithm_first mode, prioritize algorithmic mechanisms, mechanism compositions, and code_required local extensions; parameter-only proposals are only sanity checks or local refinement after a strong algorithm signal.",
                "Treat candidate_producers as separate proposal agents. Fill producer_id and producer_role when possible; otherwise RecClaw will annotate them after generation.",
                "Runnable tuning proposals must use parameter_overrides and an existing wired parent candidate.",
                "Runnable tuning parameter_overrides must use values from parameter_space.",
                "Every proposal must include mechanism, mechanism_composition, novelty_claim, expected_failure_mode, ablation_parent, and implementation_complexity.",
                "Every proposal must stay inside action_space; use action_space.method_space_projection as the compact method-space guide.",
                "Do not propose exact parameter signatures listed in agent_state_summary.recent_forbidden_parameter_signatures.",
                "If agent_state_summary.anti_plateau.plateau_detected is true, do not spend the next proposal set on minor variants of agent_state_summary.anti_plateau.local_repair_capped_families; include cross-family algorithmic proposals from agent_state_summary.anti_plateau.forced_exploration_families.",
                "If agent_state_summary.post_validation_followup.needs_structured_followup is true, do not propose more nearby code_required siblings for that validated best; propose a one-axis ablation, seed-stability replication, or parameter-region refinement using the validated parent/family.",
                "If agent_state_summary.focused_exploitation is non-empty, include focused local variants around it unless the planner explicitly asks only for reporting.",
                "Do not propose or run candidates listed in agent_state_summary.quarantined_candidates unless explicitly proposing a repair.",
                "Avoid families listed in agent_state_summary.bad_families unless the proposal is an explicit minimal repair or ablation.",
                "For code_required proposals, include a concrete implementation_plan and only local extension files.",
                "For code_required proposals, do not list configs, registry files, notes, or any __init__.py in implementation_plan.files or allowed_files.",
                "For code_required/spec_only proposals, every consumed parameter not exposed by the parent must be declared in new_parameters.",
                "Use concrete local entrypoints like recclaw_ext.models.my_model:MyModel instead of package-level recclaw_ext.models:MyModel.",
                "Do not change configs/task_ml1m.yaml, base model configs, data split, or evaluation protocol.",
            ],
        }

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    @classmethod
    def _parse_json_loose(cls, text: str) -> Any:
        stripped = cls._strip_json_fence(text)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        starts = [idx for idx, char in enumerate(stripped) if char in "[{"]
        for start in starts:
            stack = ["}" if stripped[start] == "{" else "]"]
            in_string = False
            escape = False
            for idx in range(start + 1, len(stripped)):
                char = stripped[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in "{[":
                    stack.append("}" if char == "{" else "]")
                elif stack and char == stack[-1]:
                    stack.pop()
                    if not stack:
                        try:
                            return json.loads(stripped[start : idx + 1])
                        except json.JSONDecodeError:
                            break
        raise ValueError("LLM output does not contain a parseable JSON object or array")

    def _parse_llm_proposals(self, text: str) -> list[dict[str, Any]]:
        if not text.strip():
            return []
        try:
            parsed = self._parse_json_loose(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("proposals"), list):
                parsed = parsed["proposals"]
            elif isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                raise ValueError("LLM proposal output must be a JSON object, array, or object with proposals")
            proposals = parsed
        except ValueError:
            stripped = self._strip_json_fence(text)
            proposals = []
            for line_no, line in enumerate(stripped.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    proposals.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid LLM JSONL at line {line_no}: {exc}") from exc
        invalid = [idx for idx, item in enumerate(proposals, start=1) if not isinstance(item, dict)]
        if invalid:
            raise ValueError(f"LLM proposals must be objects; invalid rows: {invalid}")
        registry_ids = {str(item.get("candidate_id") or "") for item in self.registry if item.get("candidate_id")}
        registry_by_id = {str(item.get("candidate_id") or ""): item for item in self.registry if item.get("candidate_id")}
        used_ids = set(registry_ids)
        for proposal in proposals:
            overrides = proposal.get("parameter_overrides")
            if isinstance(overrides, dict):
                proposal["parameter_overrides"] = {
                    key: value for key, value in overrides.items() if value is not None and str(value).strip() != ""
                }
                parent_id = str(proposal.get("parent_candidate_id") or "")
                if parent_id:
                    proposal["parameter_signature"] = self._parameter_signature(parent_id, proposal["parameter_overrides"])
            self._repair_llm_proposal_metadata(proposal, registry_by_id, used_ids)
        return proposals[: max(1, self.config.proposal_count)]

    def _repair_llm_proposal_metadata(
        self,
        proposal: dict[str, Any],
        registry_by_id: dict[str, dict[str, Any]],
        used_ids: set[str],
    ) -> None:
        candidate_id = str(proposal.get("candidate_id") or "").strip()
        if candidate_id:
            base_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("_.-") or "cand_llm_proposal"
            next_id = base_id
            suffix = 1
            while next_id in used_ids:
                next_id = f"{base_id}_r{suffix:02d}"
                suffix += 1
            proposal["candidate_id"] = next_id
            used_ids.add(next_id)

        if not proposal.get("action_type"):
            proposal["action_type"] = self._infer_action_type(proposal)
        parent_id = str(proposal.get("parent_candidate_id") or "")
        mechanism = str(proposal.get("mechanism") or "").strip()
        if not mechanism:
            mechanism = str(proposal.get("category") or proposal.get("action_type") or "algorithmic mechanism")
            proposal["mechanism"] = mechanism
        composition = proposal.get("mechanism_composition")
        if not isinstance(composition, list) or not composition:
            tokens = [
                str(item)
                for item in (
                    proposal.get("consumes")
                    if isinstance(proposal.get("consumes"), list)
                    else []
                )
                if str(item).strip()
            ]
            proposal["mechanism_composition"] = tokens[:6] or [mechanism]
        proposal.setdefault(
            "novelty_claim",
            "Local RecClaw extension that changes the recommendation mechanism while preserving the fixed protocol.",
        )
        proposal.setdefault(
            "expected_failure_mode",
            "May underperform if the added mechanism over-regularizes embeddings or destabilizes pairwise training.",
        )
        proposal.setdefault("ablation_parent", parent_id)
        complexity = str(proposal.get("implementation_complexity") or "").strip().lower()
        if complexity not in {"low", "medium", "high"}:
            proposal["implementation_complexity"] = "medium"

        runnable_level = str(proposal.get("runnable_level") or "")
        if runnable_level not in {"code_required", "spec_only"}:
            return
        parent = registry_by_id.get(parent_id)
        if parent is None:
            return
        consumes = [str(item) for item in (proposal.get("consumes") or [])]
        parent_consumes = {str(item) for item in (parent.get("consumes") or [])}
        new_names: set[str] = set()
        new_parameters = proposal.get("new_parameters")
        if isinstance(new_parameters, list):
            for item in new_parameters:
                if isinstance(item, str):
                    new_names.add(item)
                elif isinstance(item, dict) and item.get("name"):
                    new_names.add(str(item["name"]))
        else:
            proposal["new_parameters"] = []
            new_parameters = proposal["new_parameters"]
        missing = sorted(set(consumes) - parent_consumes - new_names)
        if missing and isinstance(new_parameters, list):
            new_parameters.extend(missing)

    @staticmethod
    def _infer_action_type(proposal: dict[str, Any]) -> str:
        proposal_type = str(proposal.get("proposal_type") or "")
        runner_type = str(proposal.get("runner_type") or "")
        category = str(proposal.get("category") or "").lower()
        consumes = {str(item) for item in (proposal.get("consumes") or [])}
        if proposal_type == "tuning":
            return "parameter_tuning"
        if runner_type == "posthoc":
            return "posthoc_rerank"
        if consumes.intersection({"hard_negative_ratio", "popularity_alpha", "debias_alpha"}):
            return "negative_sampling"
        if consumes.intersection({"margin", "tail_weight_alpha"}):
            return "pairwise_loss"
        if consumes.intersection({"lambda_norm", "max_norm", "reg_weight"}):
            return "regularization"
        if consumes.intersection({"lambda_align"}):
            return "auxiliary_loss"
        if consumes.intersection({"rank_weight_alpha"}):
            return "rank_aware_loss"
        if consumes.intersection({"edge_dropout"}):
            return "graph_augmentation"
        if "representation" in category or consumes.intersection({"residual_weight"}):
            return "aggregation"
        return "local_loss"

    @staticmethod
    def _find_action_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            action = str(value.get("action") or "").strip()
            if action:
                return value
            for nested in value.values():
                found = RecClawAgent._find_action_payload(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = RecClawAgent._find_action_payload(item)
                if found:
                    return found
        return {}

    def _sanitize_planner_payload(self, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        ignored_fields = sorted(set(payload) - {"action", "reason", "proposal_count"})
        sanitized = {
            "action": str(payload.get("action") or self._algorithm_fallback_action()).strip(),
            "reason": str(payload.get("reason") or "").strip(),
            "proposal_count": payload.get("proposal_count", self.config.proposal_count),
        }
        try:
            sanitized["proposal_count"] = max(1, int(sanitized["proposal_count"]))
        except (TypeError, ValueError):
            sanitized["proposal_count"] = self.config.proposal_count
        return sanitized, ignored_fields

    def _recent_planner_actions(self, limit: int = 4) -> list[str]:
        actions: list[str] = []
        for row in reversed(self.memory):
            if row.get("event") != "planner_action":
                continue
            actions.append(str(row.get("action") or ""))
            if len(actions) >= limit:
                break
        return list(reversed(actions))

    def _recent_trial_decisions(self, limit: int = 4) -> list[str]:
        decisions: list[str] = []
        for row in reversed(self.memory):
            if row.get("event") or "decision" not in row:
                continue
            decisions.append(str(row.get("decision") or ""))
            if len(decisions) >= limit:
                break
        return list(reversed(decisions))

    def _has_pending_code_required_review(self) -> bool:
        handled = {
            str(row.get("proposal_id") or "")
            for row in self.memory
            if row.get("event") == "implementation_result"
        }
        handled.update(
            str(row.get("proposal_id") or "")
            for row in self.memory
            if row.get("event") == "implementation_skipped"
            and str(row.get("reason") or "") in TERMINAL_IMPLEMENTATION_SKIP_REASONS
        )
        for row in reversed(self.memory):
            if row.get("event") != "proposal_needs_review":
                continue
            proposal_id = str(row.get("proposal_id") or "")
            if not proposal_id or proposal_id in handled:
                continue
            if row.get("next_action") == "promote_to_implementation_queue":
                parent_id = str(row.get("parent_candidate_id") or "")
                if not self._plateau_allows_family(parent_id or proposal_id):
                    continue
                return True
        return False

    def _maybe_override_auto_action(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        action = str(payload.get("action") or "")
        if action == "multi_seed_verify" and not self._unverified_keep_rows():
            updated = dict(payload)
            updated["action"] = self._algorithm_fallback_action()
            updated["reason"] = (
                f"auto override because multi_seed_verify has no unverified keep; "
                f"{payload.get('reason', '')}"
            ).strip()
            return updated, "no_unverified_keep_for_seed_verify"
        if action in {"multi_seed_verify", "report"}:
            return payload, ""
        if action in {"implement_needs_review", "implement_algorithm"} and not self._has_pending_code_required_review():
            updated = dict(payload)
            updated["action"] = self._algorithm_fallback_action()
            updated["reason"] = (
                f"auto override because no actionable code_required review is pending; "
                f"{payload.get('reason', '')}"
            ).strip()
            return updated, "no_actionable_code_review_pending"
        pending_count = len(self._pending_implemented_candidate_ids())
        if pending_count >= self.config.max_pending_implemented and (
            action.startswith("propose") or action in {"implement_needs_review", "implement_algorithm"}
        ):
            updated = dict(payload)
            updated["action"] = self._algorithm_fallback_action()
            updated["reason"] = (
                f"auto override to run pending implemented candidates before adding more code; "
                f"pending_implemented_count={pending_count}; {payload.get('reason', '')}"
            ).strip()
            return updated, "pending_implemented_backlog"
        post_validation = self._post_validation_followup_state()
        if (
            self.config.search_intensity == "algorithm_first"
            and post_validation.get("needs_structured_followup")
            and action in {"propose_algorithm", "propose_explore", "propose_mixed", "implement_needs_review", "implement_algorithm"}
        ):
            updated = dict(payload)
            updated["action"] = "tune_after_algorithm_success"
            updated["reason"] = (
                "post-validation override: the validated best has enough non-improving local siblings, "
                "so the next step should be one-axis ablation, stability replication, or parameter-region "
                f"refinement instead of more nearby code variants; {payload.get('reason', '')}"
            ).strip()
            return updated, "post_validation_structured_followup"
        plateau = self._plateau_state()
        if self.config.search_intensity == "algorithm_first" and plateau.get("plateau_detected"):
            local_actions = {
                "propose_tuning",
                "propose_mixed",
                "propose_explore",
                "run_available",
                "tune_after_algorithm_success",
            }
            if action in local_actions:
                updated = dict(payload)
                updated["action"] = self._algorithm_fallback_action()
                updated["reason"] = (
                    "anti-plateau override: global best has not improved recently, so the next step must "
                    f"force cross-family algorithm exploration; {payload.get('reason', '')}"
                ).strip()
                return updated, "anti_plateau_force_algorithm_exploration"
        if self.config.search_intensity == "algorithm_first" and not self._has_strong_algorithm_signal():
            if action in {"propose_tuning", "propose_mixed", "propose_explore", "tune_after_algorithm_success"}:
                updated = dict(payload)
                updated["action"] = self._algorithm_fallback_action()
                updated["reason"] = (
                    f"algorithm-first override while best_metric is below tuned LightGCN; "
                    f"{payload.get('reason', '')}"
                ).strip()
                return updated, "algorithm_first_no_strong_signal"
        recent_actions = self._recent_planner_actions(limit=4)
        recent_decisions = self._recent_trial_decisions(limit=4)
        repeated_mixed = len(recent_actions) >= 3 and all(item == "propose_mixed" for item in recent_actions[-3:])
        no_recent_keep = recent_decisions and all(item != "keep" for item in recent_decisions)
        if repeated_mixed and self._has_pending_code_required_review():
            updated = dict(payload)
            updated["action"] = "implement_algorithm"
            updated["reason"] = (
                f"auto override after repeated propose_mixed actions; {payload.get('reason', '')}"
            ).strip()
            return updated, "repeated_mixed_pending_code_required"
        if repeated_mixed and no_recent_keep:
            updated = dict(payload)
            updated["action"] = "propose_algorithm" if self.config.search_intensity == "algorithm_first" else "propose_explore"
            updated["reason"] = (
                f"auto override to broaden exploration after repeated mixed/discard rounds; {payload.get('reason', '')}"
            ).strip()
            return updated, "repeated_mixed_no_recent_keep"
        return payload, ""

    def reset_round_policy(self) -> None:
        self.skip_current_round = False
        self.skip_proposal_generation = False
        self.force_proposal_refresh = False
        if self.config.loop_mode != "auto":
            return
        policy = LOOP_MODE_POLICIES[self.config.loop_mode]
        self.config.enable_candidate_proposals = True
        self.config.proposal_mode = str(policy["proposal_mode"])
        self.config.auto_promote_needs_review = bool(policy["auto_promote"])
        self.config.auto_implement_code_required = bool(policy["auto_implement"])

    @staticmethod
    def _https_context() -> ssl.SSLContext:
        context = ssl.create_default_context()
        ignore_unexpected_eof = getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0)
        if ignore_unexpected_eof:
            context.options |= ignore_unexpected_eof
        return context

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str = "",
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        model = self.config.llm_model or os.environ.get("DEEPSEEK_MODEL", "") or os.environ.get("OPENAI_MODEL", "")
        if not model:
            raise ValueError("LLM proposal source requires --llm-model or OPENAI_MODEL")
        api_key = os.environ.get(self.config.llm_api_key_env, "")
        if not api_key:
            raise ValueError(f"missing LLM API key env var: {self.config.llm_api_key_env}")

        base_url = (
            self.config.llm_base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).rstrip("/")
        if not base_url:
            base_url = "https://api.deepseek.com/v1"
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.llm_max_tokens,
        }
        if self.config.llm_provider == "openai" and response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or "recclaw_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw = ""
        max_attempts = max(1, self.config.llm_retries + 1)
        for attempt in range(1, max_attempts + 1):
            try:
                with urlrequest.urlopen(
                    req,
                    timeout=self.config.llm_timeout,
                    context=self._https_context(),
                ) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                break
            except urlerror.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
            except (urlerror.URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
                if attempt >= max_attempts:
                    raise RuntimeError(f"LLM API request failed after {attempt} attempts: {exc}") from exc
                time.sleep(min(2.0 * attempt, 5.0))

        parsed = json.loads(raw)
        choices = parsed.get("choices") or []
        if not choices:
            raise ValueError("LLM API response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item.get("text") or item.get("content") or ""))
                else:
                    chunks.append(str(item))
            content = "".join(chunks)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM API response has empty message content")
        return content

    def generate_llm_candidate_proposals(self) -> dict[str, Any]:
        context = self._llm_proposal_context()
        system_prompt = (
            "You are RecClaw's candidate proposal generator. "
            "Generate valid recommender-system candidate proposals only. "
            "Honor experiment_directive as the highest-priority user steering below hard constraints. "
            "Do not modify RecBole core. Return strict JSON only."
        )
        user_prompt = (
            "Generate candidate proposals from this context. "
            "Return a JSON object shaped as {\"proposals\": [...]}.\n\n"
            + json.dumps(context, ensure_ascii=True)
        )
        content = self._chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema_name="recclaw_candidate_proposals",
            response_schema=PROPOSAL_RESPONSE_SCHEMA,
        )
        proposals = self._parse_llm_proposals(content)
        directives = build_producer_directives(
            search_policy=self.search_policy,
            search_memory=self.search_memory,
            proposal_count=self.config.proposal_count,
            mode=self.config.proposal_mode,
        )
        proposals = annotate_proposals(proposals, directives=directives, default_source="llm")
        proposals = order_proposals_by_route(
            proposals,
            search_memory=self.search_memory,
            search_policy=self.search_policy,
            limit=max(1, self.config.proposal_count),
        )
        self.config.proposal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.proposal_path.open("w", encoding="utf-8") as handle:
            for proposal in proposals:
                handle.write(json.dumps(proposal, ensure_ascii=True, sort_keys=True) + "\n")
        return {
            "output": str(self.config.proposal_path),
            "mode": self.config.proposal_mode,
            "proposal_source": "llm",
            "proposal_count": len(proposals),
        }

    def apply_auto_planner(self, round_id: int) -> None:
        self.skip_current_round = False
        self.skip_proposal_generation = False
        if self.config.loop_mode != "auto":
            return
        context = {
            **self._llm_proposal_context(),
            "planner_actions": sorted(AUTO_PLANNER_ACTIONS),
            "instruction": (
                "Choose exactly one next action for this RecClaw round. "
                "Prefer propose_algorithm, implement_algorithm, or run_algorithm_variant while the best metric is "
                "below tuned_lightgcn_mean. Prefer tune_after_algorithm_success only after an algorithmic candidate "
                "has crossed tuned_lightgcn_mean. Prefer multi_seed_verify only when there is an unverified keep "
                "that meets agent_state_summary.algorithm_first_targets.seed_validation_min_metric. "
                "If agent_state_summary.anti_plateau.plateau_detected is true, choose propose_algorithm, "
                "implement_algorithm, or run_algorithm_variant; do not choose tune_after_algorithm_success "
                "for local repair unless it is explicitly cross-family and architecture-level. "
                "If agent_state_summary.post_validation_followup.needs_structured_followup is true, choose "
                "tune_after_algorithm_success so the next round performs one-axis ablation, stability replication, "
                "or parameter-region refinement instead of more nearby code siblings. "
                "Return strict JSON with action, reason, and optional proposal_count."
            ),
        }
        try:
            content = self._chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are RecClaw's loop planner. "
                            "Honor experiment_directive as the highest-priority user steering below hard constraints. "
                            "Return strict JSON only."
                        ),
                    },
                    {"role": "user", "content": json.dumps(context, ensure_ascii=True)},
                ],
                schema_name="recclaw_planner_action",
                response_schema=PLANNER_RESPONSE_SCHEMA,
            )
            parsed = self._parse_json_loose(content)
            if not isinstance(parsed, dict):
                parsed = self._find_action_payload(parsed)
            else:
                parsed = self._find_action_payload(parsed) or parsed
            if not isinstance(parsed, dict):
                raise ValueError("planner output must be a JSON object")
            parsed, ignored_fields = self._sanitize_planner_payload(parsed)
            action = str(parsed.get("action") or "").strip()
            if action not in AUTO_PLANNER_ACTIONS:
                raise ValueError(f"unsupported planner action: {action}")
        except Exception as exc:  # noqa: BLE001
            if not self.config.allow_llm_fallback:
                raise RuntimeError(f"auto planner LLM failed: {type(exc).__name__}: {exc}") from exc
            parsed = {
                "action": self._algorithm_fallback_action(),
                "reason": f"planner fallback after {type(exc).__name__}: {exc}",
                "proposal_count": self.config.proposal_count,
            }
            ignored_fields = []
        parsed, override_reason = self._maybe_override_auto_action(parsed)
        action = str(parsed.get("action") or self._algorithm_fallback_action())
        self.last_planner_action = parsed
        event = {
            "event": "planner_action",
            "round_id": round_id,
            "action": action,
            "reason": parsed.get("reason", ""),
        }
        if ignored_fields:
            event["ignored_fields"] = ignored_fields
        if override_reason:
            event["override_reason"] = override_reason
        self.remember_event(event)
        print(f"[Round {round_id}] planner_action={json.dumps(parsed, ensure_ascii=True, sort_keys=True)}")

        if action == "multi_seed_verify":
            self.verify_last_keep(round_id)
            self.skip_current_round = True
            return
        if action == "report":
            self.remember_event(
                {
                    "event": "planner_report",
                    "round_id": round_id,
                    "reason": parsed.get("reason", ""),
                    "memory_rows": len(self.memory),
                    "registered_candidates": len(self.registry),
                    "proposal_rows": len(self.candidate_proposals),
                }
            )
            self.skip_current_round = True
            return

        if action == "propose_tuning":
            self.config.enable_candidate_proposals = True
            self.config.proposal_mode = "conservative"
            self.config.auto_promote_needs_review = False
            self.config.auto_implement_code_required = False
        elif action in {"propose_algorithm", "propose_explore", "implement_needs_review", "implement_algorithm"}:
            self.config.enable_candidate_proposals = True
            self.config.proposal_mode = "algorithm_first" if action in {"propose_algorithm", "implement_algorithm"} else "explore"
            self.config.auto_promote_needs_review = True
            self.config.auto_implement_code_required = True
            self.skip_proposal_generation = action in {"implement_needs_review", "implement_algorithm"}
        elif action in {"run_available", "run_algorithm_variant"}:
            self.config.enable_candidate_proposals = False
        elif action == "tune_after_algorithm_success":
            self.config.enable_candidate_proposals = True
            self.config.proposal_mode = "mixed"
            self.config.auto_promote_needs_review = True
            self.config.auto_implement_code_required = False
        else:
            self.config.enable_candidate_proposals = True
            self.config.proposal_mode = (
                "algorithm_first" if self.config.search_intensity == "algorithm_first" else "mixed"
            )
            self.config.auto_promote_needs_review = True
            self.config.auto_implement_code_required = True
        if action.startswith("propose"):
            self.force_proposal_refresh = True
        if parsed.get("proposal_count") is not None:
            try:
                self.config.proposal_count = max(1, int(parsed["proposal_count"]))
            except (TypeError, ValueError):
                pass

    def verify_last_keep(self, round_id: int) -> None:
        if self.config.dry_run:
            self.remember_event(
                {
                    "event": "seed_validation_skipped",
                    "round_id": round_id,
                    "reason": "dry_run",
                }
            )
            return
        unverified_keeps = self._unverified_keep_rows()
        target = unverified_keeps[-1] if unverified_keeps else None
        if target is None:
            self.remember_event(
                {
                    "event": "seed_validation_skipped",
                    "round_id": round_id,
                    "reason": "no unverified keep found",
                }
            )
            return
        candidate_id = str(target.get("candidate_id") or "")
        parent_id = str(target.get("parent_candidate_id") or "")
        base = self._find_registry_candidate(parent_id or candidate_id)
        if base is None:
            self.remember_event(
                {
                    "event": "seed_validation_skipped",
                    "round_id": round_id,
                    "candidate_id": candidate_id,
                    "reason": "candidate not found in registry",
                }
            )
            return
        candidate = dict(base)
        if parent_id:
            candidate["candidate_id"] = candidate_id
            candidate["run_candidate_id"] = parent_id
            candidate["parent_candidate_id"] = parent_id
        params = target.get("params") if isinstance(target.get("params"), dict) else {}
        parameter_signature = self._canonical_parameter_signature_text(target.get("parameter_signature"))
        if not parameter_signature and parent_id and params:
            parameter_signature = self._parameter_signature(parent_id, dict(params))
        compare_baseline = target.get("compare_baseline") if isinstance(target.get("compare_baseline"), dict) else {}
        validation = self.run_seed_validation(candidate, dict(params), self._to_float(compare_baseline.get("baseline_metric")))
        self.remember_event(
            {
                "event": "seed_validation",
                "round_id": round_id,
                "candidate_id": candidate_id,
                "parent_candidate_id": parent_id,
                "params": params,
                "parameter_signature": parameter_signature,
                "seed_validation": validation,
            }
        )

    def _run_json_command(self, cmd: list[str], *, allow_json_failure: bool = False) -> dict[str, Any]:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if allow_json_failure and completed.stdout.strip():
            try:
                parsed = json.loads(completed.stdout)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                parsed["_exit_code"] = completed.returncode
                if completed.stderr:
                    parsed["_stderr"] = completed.stderr
                return parsed
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(cmd)}\n{message}")
        parsed = json.loads(completed.stdout)
        if not isinstance(parsed, dict):
            raise ValueError(f"command did not return a JSON object: {' '.join(cmd)}")
        return parsed

    def remember_event(self, payload: dict[str, Any]) -> None:
        self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        event = {"timestamp": datetime.now().isoformat(timespec="seconds"), **payload}
        with self.config.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        self.memory.append(event)
        self._refresh_search_memory()
        self.agent_state_summary = self._build_agent_state_summary()
        self._write_agent_state_summary(self.agent_state_summary)

    def refresh_candidate_proposals(self, round_id: int) -> None:
        if not self.config.enable_candidate_proposals:
            return
        every = max(1, self.config.proposal_every)
        should_generate = (
            not self.skip_proposal_generation
            and (
                self.force_proposal_refresh
                or round_id == 1
                or (round_id - 1) % every == 0
                or not self.config.proposal_path.exists()
            )
        )
        if should_generate:
            try:
                self.skip_proposal_generation = False
                if self.config.proposal_source == "llm":
                    generated = self.generate_llm_candidate_proposals()
                else:
                    generated = self.generate_heuristic_candidate_proposals()
            except Exception as exc:  # noqa: BLE001
                if self.config.proposal_source == "llm" and not self.config.allow_llm_fallback:
                    raise
                self.remember_event(
                    {
                        "event": "proposal_generation_failed",
                        "round_id": round_id,
                        "proposal_source": self.config.proposal_source,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[Round {round_id}] proposal_generation_failed={type(exc).__name__}: {exc}")
                generated = self.generate_heuristic_candidate_proposals()
                generated["proposal_source"] = "heuristic_fallback"
            print(
                f"[Round {round_id}] proposals_generated="
                f"{generated.get('proposal_count')} mode={generated.get('mode')} "
                f"source={generated.get('proposal_source', self.config.proposal_source)}"
            )
            self.current_proposal_source = str(generated.get("proposal_source") or self.config.proposal_source)
            self.remember_event(
                {
                    "event": "proposal_generated",
                    "round_id": round_id,
                    "proposal_source": self.current_proposal_source,
                    "proposal_mode": generated.get("mode", self.config.proposal_mode),
                    "proposal_count": generated.get("proposal_count", 0),
                }
            )

        self.candidate_proposals = self._load_candidate_proposals()
        if not self.candidate_proposals:
            self.proposal_validation_report = {}
            print(f"[Round {round_id}] proposals_available=0")
            return

        self.proposal_validation_report = self._run_json_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "validate_candidate_proposal.py"),
                "--proposals",
                str(self.config.proposal_path),
                "--registry",
                str(self.config.registry_path),
                "--schema",
                str(self.config.proposal_schema_path),
                "--memory",
                str(self.config.memory_path),
            ],
            allow_json_failure=True,
        )
        summary = self.proposal_validation_report.get("summary", {})
        next_actions = self.proposal_validation_report.get("next_actions", {})
        print(
            f"[Round {round_id}] proposals_validation="
            f"{json.dumps(summary, ensure_ascii=True, sort_keys=True)} "
            f"next_actions={json.dumps(next_actions, ensure_ascii=True, sort_keys=True)}"
        )
        self.record_proposal_routes(round_id)
        if not self.config.auto_promote_needs_review:
            return

        promotion_report = self._run_json_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "promote_candidate_proposal.py"),
                "--proposals",
                str(self.config.proposal_path),
                "--registry",
                str(self.config.registry_path),
                "--schema",
                str(self.config.proposal_schema_path),
                "--memory",
                str(self.config.memory_path),
            ]
            + (["--dry-run"] if self.config.dry_run else []),
            allow_json_failure=True,
        )
        promoted = promotion_report.get("promoted_parent_candidate_ids", [])
        if promoted:
            self._reload_registry()
            print(
                f"[Round {round_id}] proposals_promoted="
                f"{json.dumps(promoted, ensure_ascii=True, sort_keys=True)}"
            )
        blocked = promotion_report.get("blocked", {})
        if blocked:
            print(
                f"[Round {round_id}] proposals_promotion_blocked="
                f"{json.dumps(blocked, ensure_ascii=True, sort_keys=True)}"
            )
        pending_implemented = sorted(self._pending_implemented_candidate_ids())
        if (
            self.config.auto_implement_code_required
            and len(pending_implemented) >= self.config.max_pending_implemented
        ):
            self.remember_event(
                {
                    "event": "implementation_skipped",
                    "round_id": round_id,
                    "reason": "pending_implemented_backlog",
                    "pending_implemented_count": len(pending_implemented),
                    "pending_implemented_candidates": pending_implemented[:20],
                }
            )
            print(
                f"[Round {round_id}] implementation_skipped="
                f"pending_implemented_backlog count={len(pending_implemented)}"
            )
            return
        if self.config.auto_implement_code_required and self.config.dry_run:
            self.remember_event(
                {
                    "event": "implementation_skipped",
                    "round_id": round_id,
                    "reason": "dry_run",
                }
            )
        elif self.config.auto_implement_code_required:
            self.implement_needs_review_proposals(round_id)

    def generate_heuristic_candidate_proposals(self) -> dict[str, Any]:
        return self._run_json_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "propose_candidate.py"),
                "--mode",
                self.config.proposal_mode,
                "--count",
                str(max(1, self.config.proposal_count)),
                "--output",
                str(self.config.proposal_path),
                "--memory",
                str(self.config.memory_path),
                "--memory-limit",
                str(max(0, self.config.memory_read_limit)),
                "--search-policy",
                str(self.config.search_policy_path),
            ]
        )

    def record_proposal_routes(self, round_id: int) -> None:
        by_id = {str(item.get("candidate_id") or ""): item for item in self.candidate_proposals}
        route_by_id = {
            item.candidate_id: item.as_dict()
            for item in route_proposals(
                self.candidate_proposals,
                search_memory=self.search_memory,
                search_policy=self.search_policy,
                validation_results=self.proposal_validation_report.get("results", []),
            )
        }
        for row in self.proposal_validation_report.get("results", []):
            if not isinstance(row, dict):
                continue
            proposal_id = str(row.get("candidate_id") or "")
            status = str(row.get("status") or "")
            event_key = f"{proposal_id}:{status}"
            if not proposal_id or event_key in self.recorded_proposal_events:
                continue
            if status not in {"rejected", "needs_review"}:
                continue
            proposal = by_id.get(proposal_id, {})
            self.recorded_proposal_events.add(event_key)
            mechanism = proposal.get("mechanism")
            if not mechanism:
                composition = proposal.get("mechanism_composition")
                if isinstance(composition, list):
                    mechanism = "+".join(str(item) for item in composition if str(item))
            self.remember_event(
                {
                    "event": f"proposal_{status}",
                    "round_id": round_id,
                    "proposal_id": proposal_id,
                    "parent_candidate_id": proposal.get("parent_candidate_id", ""),
                    "action_type": proposal.get("action_type", ""),
                    "proposal_type": proposal.get("proposal_type", ""),
                    "runnable_level": proposal.get("runnable_level", ""),
                    "mechanism": mechanism or "",
                    "novelty_claim": proposal.get("novelty_claim", ""),
                    "expected_failure_mode": proposal.get("expected_failure_mode", ""),
                    "ablation_parent": proposal.get("ablation_parent", ""),
                    "implementation_complexity": proposal.get("implementation_complexity", ""),
                    "parameter_signature": row.get("parameter_signature", ""),
                    "proposal_source": self.current_proposal_source,
                    "producer_id": proposal.get("producer_id", ""),
                    "producer_role": proposal.get("producer_role", ""),
                    "route_decision": route_by_id.get(proposal_id, {}),
                    "errors": row.get("errors", []),
                    "review_reasons": row.get("review_reasons", []),
                    "next_action": row.get("next_action", ""),
                }
            )

    def implement_needs_review_proposals(self, round_id: int) -> None:
        implemented_this_round = 0
        for row in self.proposal_validation_report.get("results", []):
            if not isinstance(row, dict):
                continue
            proposal_id = str(row.get("candidate_id") or "")
            if row.get("status") != "needs_review" or row.get("runnable_level") != "code_required":
                continue
            if implemented_this_round >= self.config.max_implement_per_round:
                self.remember_event(
                    {
                        "event": "implementation_skipped",
                        "round_id": round_id,
                        "proposal_id": proposal_id,
                        "reason": "max_implement_per_round",
                        "max_implement_per_round": self.config.max_implement_per_round,
                    }
                )
                continue
            if not proposal_id or proposal_id in self.scheduled_candidate_ids:
                continue
            failed_attempts = self._failed_implementation_attempts(proposal_id)
            if len(failed_attempts) >= max(1, self.config.max_failed_implementation_attempts):
                self.remember_event(
                    {
                        "event": "implementation_skipped",
                        "round_id": round_id,
                        "proposal_id": proposal_id,
                        "reason": "prior_failed_implementation_attempts",
                        "failed_attempt_count": len(failed_attempts),
                        "last_status": failed_attempts[-1].get("status", ""),
                    }
                )
                continue
            proposal = next(
                (
                    item
                    for item in self.candidate_proposals
                    if str(item.get("candidate_id") or "") == proposal_id
                ),
                {},
            )
            parent_id = str(proposal.get("parent_candidate_id") or "")
            if not self._plateau_allows_family(parent_id or proposal_id):
                self.remember_event(
                    {
                        "event": "implementation_skipped",
                        "round_id": round_id,
                        "proposal_id": proposal_id,
                        "parent_candidate_id": parent_id,
                        "reason": "anti_plateau_capped_family",
                    }
                )
                continue
            parent = self._find_registry_candidate(parent_id)
            proposal_consumes = {str(item) for item in (proposal.get("consumes") or [])}
            parent_consumes = {str(item) for item in ((parent or {}).get("consumes") or [])}
            if (
                parent_id in AUTO_PROMOTABLE_PARENT_IDS
                and parent is not None
                and bool(parent.get("wired"))
                and str(parent.get("status") or "") == "implemented"
                and proposal_consumes.issubset(parent_consumes)
            ):
                self.remember_event(
                    {
                        "event": "implementation_skipped",
                        "round_id": round_id,
                        "proposal_id": proposal_id,
                        "parent_candidate_id": parent_id,
                        "reason": "known parent was auto-promoted and is runnable",
                    }
                )
                continue
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "implement_candidate_proposal.py"),
                "--proposal-id",
                proposal_id,
                "--proposals",
                str(self.config.proposal_path),
                "--registry",
                str(self.config.registry_path),
                "--llm-provider",
                self.config.llm_provider,
                "--llm-base-url",
                self.config.llm_base_url,
                "--llm-model",
                self.config.llm_model,
                "--llm-api-key-env",
                self.config.llm_api_key_env,
                "--llm-temperature",
                str(self.config.llm_temperature),
                "--llm-timeout",
                str(self.config.llm_timeout),
                "--llm-max-tokens",
                str(self.config.llm_max_tokens),
                "--llm-retries",
                str(self.config.llm_retries),
            ]
            directive = self._active_experiment_directive()
            if directive:
                command.extend(["--experiment-directive", directive])
            if not self.config.implementation_smoke_run:
                command.append("--skip-smoke-run")
            command.extend(
                item
                for override in self.config.global_overrides
                for item in ("--smoke-set", override)
            )
            if self.config.dry_run:
                command.append("--dry-run")
            report = self._run_json_command(command, allow_json_failure=True)
            review_errors = report.get("review_errors", [])
            if not isinstance(review_errors, list):
                review_errors = []
            self.remember_event(
                {
                    "event": "implementation_result",
                    "round_id": round_id,
                    "proposal_id": proposal_id,
                    "status": report.get("status", "implementation_failed"),
                    "candidate_id": report.get("candidate_id", ""),
                    "reason": report.get("reason", ""),
                    "entrypoint": report.get("entrypoint", ""),
                    "source": report.get("source", ""),
                    "files": report.get("files", []),
                    "smoke": report.get("smoke", {}),
                    "review_errors": review_errors[-5:],
                }
            )
            print(f"[Round {round_id}] implementation_result={json.dumps(report, ensure_ascii=True, sort_keys=True)}")
            implemented_this_round += 1
            if str(report.get("status") or "") in IMPLEMENTATION_RUNNABLE_STATUSES:
                self._reload_registry()

    # ===== Evaluate =====
    def evaluate(self, candidate: dict[str, Any], action_out: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        summary = action_out.get("summary", {})
        run_id = str(summary.get("run_id") or "")
        model_name = self._normalize_model_key(summary.get("model") or candidate.get("base_model") or "")
        if "/" in model_name:
            model_name = model_name.split("/", 1)[0].strip()

        candidate_result = {}
        result_json_path = summary.get("result_json_path")
        if result_json_path:
            candidate_result = load_result_source(result_json_path)
        elif run_id:
            candidate_result = self._load_result_from_csv(run_id)

        baseline = self.baselines_by_model.get(model_name, {})
        history_best = self.best_by_model.get(model_name, baseline)
        if not baseline:
            baseline = {"status": "missing"}
        if not history_best:
            history_best = baseline

        compare_baseline = compare_results(
            baseline,
            candidate_result,
            self.config.metric,
            multi_metrics=self.config.multi_metrics,
            metrics_weights=self.config.metrics_weights,
            metric_directions=self.config.metric_directions,
        )
        compare_history = compare_results(
            history_best,
            candidate_result,
            self.config.metric,
            multi_metrics=self.config.multi_metrics,
            metrics_weights=self.config.metrics_weights,
            metric_directions=self.config.metric_directions,
        )
        compare_baseline = self._normalize_compare_payload(compare_baseline)
        compare_history = self._normalize_compare_payload(compare_history)
        dimension_report = self._build_dimension_report(baseline, history_best, candidate_result)
        return candidate_result, compare_baseline, compare_history, dimension_report

    @staticmethod
    def _normalize_compare_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload or {})
        normalized.setdefault("delta", None)
        normalized.setdefault("decision", "crash")
        normalized.setdefault("explanation", "invalid comparison payload")
        return normalized

    def _load_result_from_csv(self, run_id: str) -> dict[str, Any]:
        for row in reversed(self.results_rows):
            if row.get("run_id") == run_id:
                return dict(row)
        return {}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_dimension_report(
        self,
        baseline: dict[str, Any],
        history_best: dict[str, Any],
        candidate_result: dict[str, Any],
    ) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for metric in EVALUATION_DIMENSIONS:
            cand = self._to_float(candidate_result.get(metric))
            base = self._to_float(baseline.get(metric))
            hist = self._to_float(history_best.get(metric))
            direction = -1 if int(self.config.metric_directions.get(metric, 1)) < 0 else 1
            report[metric] = {
                "candidate": cand,
                "baseline": base,
                "history_best": hist,
                "delta_vs_baseline": None if cand is None or base is None else round(cand - base, 6),
                "delta_vs_history_best": None if cand is None or hist is None else round(cand - hist, 6),
                "direction": "lower_is_better" if direction < 0 else "higher_is_better",
            }
        return report

    # ===== Reflect =====
    def reflect(
        self,
        candidate: dict[str, Any],
        params: dict[str, Any],
        action_out: dict[str, Any],
        candidate_result: dict[str, Any],
        compare_baseline: dict[str, Any],
        compare_history: dict[str, Any],
    ) -> tuple[str, str, str]:
        status = str(candidate_result.get("status") or "").lower()
        if action_out.get("exit_code", 1) != 0 or status != "success":
            return "crash", "candidate run crashed or result status is not success", "check candidate wiring and dependencies"
        if compare_baseline.get("delta") is None:
            return "revise", "baseline comparison is missing or incomplete", "complete same-protocol baseline before drawing conclusions"
        if self._is_extreme_quality_collapse({"result": candidate_result, "compare_baseline": compare_baseline}):
            return (
                "discard",
                "candidate shows extreme early-quality collapse versus baseline",
                "quarantine this parameter region and prefer simpler ablations",
            )

        baseline_delta = float(compare_baseline.get("delta") or 0.0)
        history_delta = float(compare_history.get("delta") or 0.0)
        if baseline_delta > self.config.min_keep_delta and history_delta > self.config.min_keep_delta:
            metric_value = self._to_float(candidate_result.get(self.config.metric))
            promotion_floor = self.config.seed_validation_min_metric
            if (
                self.config.search_intensity == "algorithm_first"
                and promotion_floor > 0
                and metric_value is not None
                and metric_value < promotion_floor
            ):
                return (
                    "revise",
                    "candidate improves local baseline/history but is below algorithm-first promotion floor",
                    self._revise_suggestion(candidate, params),
                )
            return "keep", "candidate improves both baseline and current history best", "schedule follow-up around this candidate"
        if baseline_delta >= 0 and history_delta <= self.config.min_keep_delta:
            return "revise", "candidate improves baseline but does not beat history best", self._revise_suggestion(candidate, params)
        if -self.config.min_keep_delta < baseline_delta < 0:
            return "revise", "candidate is slightly below baseline and needs parameter refinement", self._revise_suggestion(candidate, params)
        return "discard", "candidate does not beat baseline under current settings", "de-prioritize this configuration and explore another candidate"

    def _revise_suggestion(self, candidate: dict[str, Any], params: dict[str, Any]) -> str:
        consumes = candidate.get("consumes") or []
        if consumes:
            key = str(consumes[0])
            current = params.get(key)
            values = self.config.parameter_space.get(key, [])
            if values and current in values:
                idx = values.index(current)
                neighbors = []
                if idx > 0:
                    neighbors.append(values[idx - 1])
                if idx + 1 < len(values):
                    neighbors.append(values[idx + 1])
                if neighbors:
                    return f"try {key} in {neighbors} (current={current})"
            return f"adjust {key} to another value in configured parameter_space"
        return "refine implementation details or move to a related candidate"

    def run_seed_validation(
        self,
        candidate: dict[str, Any],
        params: dict[str, Any],
        baseline_metric: float | None,
    ) -> dict[str, Any]:
        if baseline_metric is None:
            return {
                "status": "inconclusive",
                "reason": "baseline metric is missing",
                "metric": self.config.metric,
                "seeds": self.config.validation_seeds,
            }
        values: list[float] = []
        run_ids: list[str] = []
        seed_runs: list[dict[str, Any]] = []
        crashes = 0
        for seed in self.config.validation_seeds:
            seed_params = {**params, "seed": seed}
            action_out = self.act(candidate, seed_params)
            candidate_result, _, _, _ = self.evaluate(candidate, action_out)
            run_id = str((action_out.get("summary") or {}).get("run_id") or "")
            if run_id:
                run_ids.append(run_id)
            value = self._to_float(candidate_result.get(self.config.metric))
            if action_out.get("exit_code", 1) != 0 or str(candidate_result.get("status") or "").lower() != "success" or value is None:
                crashes += 1
                seed_runs.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "status": str(candidate_result.get("status") or "crash"),
                        "value": None,
                    }
                )
                continue
            values.append(value)
            seed_runs.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "status": "success",
                    "value": round(value, 6),
                }
            )

        if not values:
            status = "inconclusive"
            mean_value = None
            std_value = None
            wins = 0
        else:
            mean_value = statistics.mean(values)
            std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
            wins = sum(1 for value in values if value > baseline_metric + self.config.min_keep_delta)
            required_wins = max(1, (2 * len(self.config.validation_seeds) + 2) // 3)
            status = (
                "passed"
                if mean_value > baseline_metric + self.config.min_keep_delta and wins >= required_wins
                else "failed"
            )
            if crashes:
                status = "inconclusive"
        required_wins = max(1, (2 * len(self.config.validation_seeds) + 2) // 3)

        return {
            "status": status,
            "metric": self.config.metric,
            "seeds": self.config.validation_seeds,
            "values": values,
            "mean": None if mean_value is None else round(mean_value, 6),
            "std": None if std_value is None else round(std_value, 6),
            "baseline_reference": "single_seed",
            "baseline_metric": baseline_metric,
            "wins": wins,
            "required_wins": required_wins,
            "total": len(self.config.validation_seeds),
            "crashes": crashes,
            "run_ids": run_ids,
            "runs": seed_runs,
        }

    # ===== Remember =====
    def remember(self, record: TrialRecord) -> None:
        self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "round_id": record.round_id,
            "candidate_id": record.candidate_id,
            "params": record.params,
            "run_id": record.run_id,
            "status": record.status,
            "result": record.result,
            "compare_baseline": record.compare_baseline,
            "compare_history_best": record.compare_history_best,
            "dimension_report": record.dimension_report,
            "decision": record.decision,
            "reason": record.reason,
            "next_action": record.next_action,
            "is_baseline_improvement": record.is_baseline_improvement,
            "is_history_best": record.is_history_best,
        }
        if record.parent_candidate_id:
            payload["parent_candidate_id"] = record.parent_candidate_id
        if record.proposal_id:
            payload["proposal_id"] = record.proposal_id
        if record.parameter_signature:
            payload["parameter_signature"] = record.parameter_signature
        if record.execution_signature:
            payload["execution_signature"] = record.execution_signature
        if record.proposal_source:
            payload["proposal_source"] = record.proposal_source
        if record.producer_id:
            payload["producer_id"] = record.producer_id
        if record.producer_role:
            payload["producer_role"] = record.producer_role
        if record.route_decision:
            payload["route_decision"] = record.route_decision
        if record.seed_validation:
            payload["seed_validation"] = record.seed_validation
        with self.config.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.memory.append(payload)
        self.history_by_candidate.setdefault(record.candidate_id, []).append(payload)
        self._refresh_search_memory()
        self.agent_state_summary = self._build_agent_state_summary()
        self._write_agent_state_summary(self.agent_state_summary)

    def _score(self, item: dict[str, Any]) -> float:
        if self.config.multi_metrics:
            score = 0.0
            for metric, weight in self.config.metrics_weights.items():
                direction = -1.0 if int(self.config.metric_directions.get(metric, 1)) < 0 else 1.0
                score += float(item.get(metric) or 0.0) * weight * direction
            return score
        return float(item.get(self.config.metric) or 0.0)

    def run(self) -> None:
        start_round = max(1, self.config.start_round)
        end_round = max(start_round, self.config.rounds)
        self.observe()
        if start_round == 1:
            self._refresh_experience_artifacts(0, reason="initial")
        self.remember_experiment_directive()
        for round_id in range(start_round, end_round + 1):
            try:
                self.reset_round_policy()
                self.apply_auto_planner(round_id)
                if self.skip_current_round:
                    print(f"[Round {round_id}] planner requested no candidate run")
                    continue
                self.refresh_candidate_proposals(round_id)
                candidate, params, context = self.plan()
                candidate_id = str(candidate.get("candidate_id"))
                self._write_research_route(round_id, candidate, params)
                self.scheduled_candidate_ids.add(candidate_id)
                parameter_signature = str(candidate.get("parameter_signature") or "")
                if parameter_signature:
                    self.scheduled_param_signatures.add(parameter_signature)
                execution_signature = str(candidate.get("execution_signature") or self._execution_signature(candidate, params))
                if execution_signature:
                    self.scheduled_execution_signatures.add(execution_signature)
                print(f"\n[Round {round_id}] candidate={candidate_id} params={json.dumps(params, ensure_ascii=True)}")
                if candidate.get("parent_candidate_id"):
                    print(
                        f"[Round {round_id}] proposal_parent={candidate.get('parent_candidate_id')} "
                        f"run_candidate_id={candidate.get('run_candidate_id')}"
                    )
                if context:
                    print(f"[Round {round_id}] optional_context_loaded={list(context.keys())}")
                if self.config.dry_run:
                    continue

                action_out = self.act(candidate, params)
                candidate_result, compare_baseline, compare_history, dimension_report = self.evaluate(candidate, action_out)
                decision, reason, next_action = self.reflect(
                    candidate, params, action_out, candidate_result, compare_baseline, compare_history
                )
                run_id = str((action_out.get("summary") or {}).get("run_id") or "")
                baseline_delta = self._to_float(compare_baseline.get("delta"))
                history_delta = self._to_float(compare_history.get("delta"))
                is_baseline_improvement = baseline_delta is not None and baseline_delta > self.config.min_keep_delta
                is_history_best = history_delta is not None and history_delta > self.config.min_keep_delta
                seed_validation: dict[str, Any] = {}
                if self.config.enable_seed_validation and decision == "keep":
                    seed_validation = self.run_seed_validation(
                        candidate,
                        params,
                        self._to_float(compare_baseline.get("baseline_metric")),
                    )
                    validation_status = str(seed_validation.get("status") or "inconclusive")
                    if validation_status != "passed":
                        decision = "revise"
                        reason = f"{reason}; seed validation {validation_status}"
                        next_action = f"revise because seed validation {validation_status}"
                self.retain_checkpoints_for_result(
                    decision=decision,
                    run_id=run_id,
                    seed_validation=seed_validation,
                )
                record = TrialRecord(
                    round_id=round_id,
                    candidate_id=candidate_id,
                    params=params,
                    run_id=run_id,
                    status=str(candidate_result.get("status") or "unknown"),
                    result=candidate_result,
                    compare_baseline=compare_baseline,
                    compare_history_best=compare_history,
                    dimension_report=dimension_report,
                    decision=decision,
                    reason=reason,
                    next_action=next_action,
                    parent_candidate_id=str(candidate.get("parent_candidate_id") or ""),
                    proposal_id=str(candidate.get("proposal_id") or ""),
                    parameter_signature=parameter_signature,
                    execution_signature=execution_signature,
                    proposal_source=str(candidate.get("proposal_source") or ""),
                    producer_id=str(candidate.get("producer_id") or ""),
                    producer_role=str(candidate.get("producer_role") or ""),
                    route_decision=candidate.get("route_decision") if isinstance(candidate.get("route_decision"), dict) else {},
                    is_baseline_improvement=is_baseline_improvement,
                    is_history_best=is_history_best,
                    seed_validation=seed_validation,
                )
                self.remember(record)
                self._update_history_best(candidate.get("base_model") or candidate_result.get("model"), candidate_result)
                if self.config.refresh_experience_every > 0 and round_id % self.config.refresh_experience_every == 0:
                    self._refresh_experience_artifacts(round_id, reason=f"round_{round_id}")
                print(
                    json.dumps(
                        {
                            "round": round_id,
                            "candidate_id": candidate_id,
                            "run_id": run_id,
                            "decision": decision,
                            "reason": reason,
                            "next_action": next_action,
                            "latency_ms": candidate_result.get("latency_ms", ""),
                            "seed_validation": seed_validation.get("status", ""),
                        },
                        ensure_ascii=True,
                        indent=2,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                candidate_id = "unknown"
                run_id = f"agent_internal_round_{round_id}_{int(datetime.now().timestamp())}"
                reason = f"{type(exc).__name__}: {exc}"
                fallback_record = TrialRecord(
                    round_id=round_id,
                    candidate_id=candidate_id,
                    params={},
                    run_id=run_id,
                    status="crash",
                    result={},
                    compare_baseline={"delta": None, "decision": "crash", "explanation": reason},
                    compare_history_best={"delta": None, "decision": "crash", "explanation": reason},
                    dimension_report={},
                    decision="crash",
                    reason=reason,
                    next_action="inspect agent pipeline/runtime environment and retry",
                )
                self.remember(fallback_record)
                if self.config.refresh_experience_every > 0 and round_id % self.config.refresh_experience_every == 0:
                    self._refresh_experience_artifacts(round_id, reason=f"round_{round_id}_crash")
                print(f"[Round {round_id}] crash: {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RecClaw Observe->Plan->Act->Evaluate->Reflect->Remember agent")
    parser.add_argument("--rounds", type=int, default=5, help="Number of scheduling rounds")
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="First scheduling round id; use with --rounds as the final round id for safe continuation runs",
    )
    parser.add_argument(
        "--loop-mode",
        choices=("tuning", "mixed", "explore", "auto"),
        default="mixed",
        help="Agent loop mode controlling proposal type and implementation autonomy",
    )
    parser.add_argument("--metric", default="ndcg@10", help="Primary metric for single-objective mode")
    parser.add_argument("--multi-metrics", action="store_true", help="Enable weighted multi-metric scoring")
    parser.add_argument("--metrics-weights-json", help="Weights JSON for multi-metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run Observe+Plan only")
    parser.add_argument(
        "--set",
        dest="global_overrides",
        action="append",
        default=[],
        help="Append a temporary run_candidate.py override key=value to every run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("RECCLAW_CHECKPOINT_DIR", ""),
        help="Directory passed to run_candidate.py for RecBole checkpoints",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=("none", "cleanup_all", "keep_validated", "keep_any"),
        default=os.environ.get("RECCLAW_CHECKPOINT_POLICY", "none"),
        help=(
            "Agent-level checkpoint retention policy. keep_validated keeps checkpoints only "
            "for final keep decisions that pass seed validation."
        ),
    )
    parser.add_argument("--memory-path", default=str(MEMORY_PATH), help="Agent memory jsonl path")
    parser.add_argument("--results-csv", default=str(RESULTS_CSV), help="Candidate results.csv path used for evaluation")
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR), help="Directory containing baseline log files")
    parser.add_argument("--registry-path", default=str(REGISTRY_PATH), help="Candidate registry yaml path")
    parser.add_argument("--memory-read-limit", type=int, default=2000, help="Read only latest N memory rows (0 means all)")
    parser.add_argument(
        "--state-summary-path",
        default=str(STATE_SUMMARY_PATH),
        help="Write compact agent state summary for LLM planning/proposal context",
    )
    parser.add_argument(
        "--prompt-memory-tail",
        type=int,
        default=120,
        help="Maximum recent memory rows included verbatim in LLM prompts; summary still sees loaded memory",
    )
    parser.add_argument(
        "--experiment-directive",
        default=os.environ.get("RECCLAW_EXPERIMENT_DIRECTIVE", ""),
        help=(
            "Optional run-level user steering prompt. When provided, it is injected into "
            "planner/proposal prompts below hard safety constraints."
        ),
    )
    parser.add_argument(
        "--experiment-directive-file",
        default=os.environ.get("RECCLAW_EXPERIMENT_DIRECTIVE_FILE", ""),
        help="Optional UTF-8 text file containing the run-level experiment directive",
    )
    parser.add_argument(
        "--disable-experiment-directive",
        action="store_true",
        help="Ignore --experiment-directive, --experiment-directive-file, and directive environment variables",
    )
    parser.add_argument(
        "--refresh-experience-every",
        type=int,
        default=0,
        help="Refresh candidate tree and experience summary every N completed rounds when directives are enabled",
    )
    parser.add_argument("--candidate-tree-path", default=str(CANDIDATE_TREE_PATH), help="Output JSON path for candidate search tree")
    parser.add_argument(
        "--candidate-tree-md-path",
        default=str(CANDIDATE_TREE_MD_PATH),
        help="Output Markdown path for candidate search tree",
    )
    parser.add_argument(
        "--candidate-tree-mmd-path",
        default=str(CANDIDATE_TREE_MMD_PATH),
        help="Output Mermaid path for candidate search tree",
    )
    parser.add_argument("--experience-summary-path", default=str(EXPERIENCE_SUMMARY_PATH), help="Output Markdown path for experience summary")
    parser.add_argument(
        "--experience-summary-json-path",
        default=str(EXPERIENCE_SUMMARY_JSON_PATH),
        help="Output JSON path for experience summary",
    )
    parser.add_argument("--reflection-memory-path", default=str(REFLECTION_MEMORY_PATH), help="Append-only reflection memory JSONL path")
    parser.add_argument("--search-policy-path", default=str(SEARCH_POLICY_PATH), help="Runtime research-line search policy path")
    parser.add_argument(
        "--research-route-path",
        default=os.environ.get("RECCLAW_RESEARCH_ROUTES", str(PROJECT_ROOT / "results" / "research_routes.jsonl")),
        help="Append-only JSONL trace of Research Router selections",
    )
    parser.add_argument("--disable-research-line", action="store_true", help="Disable explicit producer/router research line")
    parser.add_argument("--recent-schedule-penalty", type=float, default=0.15, help="Penalty if candidate was scheduled in previous round")
    parser.add_argument(
        "--enable-candidate-proposals",
        action="store_true",
        help="Generate/validate candidate proposals and include accepted tuning proposals in planning",
    )
    parser.add_argument(
        "--disable-candidate-proposals",
        action="store_true",
        help="Disable proposal generation/validation and only schedule already-runnable registry candidates",
    )
    parser.add_argument("--proposal-path", default=str(PROPOSAL_PATH), help="Candidate proposal JSONL path")
    parser.add_argument(
        "--proposal-mode",
        choices=("conservative", "mixed", "explore", "algorithm_first"),
        default=None,
        help="Override the proposal generation mode selected by --loop-mode",
    )
    parser.add_argument(
        "--proposal-source",
        choices=("heuristic", "llm"),
        default=os.environ.get("RECCLAW_PROPOSAL_SOURCE", "llm"),
        help="Proposal source: heuristic fallback or direct LLM API from agent.py",
    )
    parser.add_argument("--proposal-count", type=int, default=5, help="Number of proposals to generate per refresh")
    parser.add_argument("--proposal-every", type=int, default=3, help="Refresh proposals every N rounds")
    parser.add_argument("--proposal-bonus", type=float, default=0.35, help="Planning score bonus for accepted proposals")
    parser.add_argument(
        "--auto-promote-needs-review",
        action="store_true",
        help="Promote known implementation-ready needs_review proposals into runnable registry candidates",
    )
    parser.add_argument(
        "--max-pending-implemented",
        type=int,
        default=3,
        help="Pause new auto-implementations and prioritize running implemented candidates at this backlog size",
    )
    parser.add_argument(
        "--max-implement-per-round",
        type=int,
        default=1,
        help="Maximum code_required proposals to auto-implement in one round",
    )
    parser.add_argument(
        "--search-intensity",
        choices=("balanced", "algorithm_first"),
        default=os.environ.get("RECCLAW_SEARCH_INTENSITY", "balanced"),
        help="Search posture for auto planner overrides and LLM proposal context",
    )
    parser.add_argument(
        "--algorithm-budget-per-window",
        type=int,
        default=int(os.environ.get("RECCLAW_ALGORITHM_BUDGET_PER_WINDOW", "3")),
        help="Target code_required algorithmic budget per short search window",
    )
    parser.add_argument(
        "--algorithm-first-explore-rounds",
        type=int,
        default=int(os.environ.get("RECCLAW_ALGORITHM_FIRST_EXPLORE_ROUNDS", "20")),
        help="In algorithm-first mode, prefer runnable algorithmic candidates over parameter-only proposals for this many formal trials.",
    )
    parser.add_argument(
        "--seed-validation-min-metric",
        type=float,
        default=float(os.environ.get("RECCLAW_SEED_VALIDATION_MIN_METRIC", "0")),
        help="Only planner-triggered multi-seed verification keeps whose primary metric is at least this value.",
    )
    parser.add_argument(
        "--disable-anti-plateau",
        action="store_true",
        help="Disable the algorithm-first anti-plateau planner guard.",
    )
    parser.add_argument(
        "--plateau-window-metric-rows",
        type=int,
        default=int(os.environ.get("RECCLAW_PLATEAU_WINDOW_METRIC_ROWS", "30")),
        help="Metric-bearing rows without global-best refresh before anti-plateau steering activates.",
    )
    parser.add_argument(
        "--plateau-min-global-improvement",
        type=float,
        default=float(os.environ.get("RECCLAW_PLATEAU_MIN_GLOBAL_IMPROVEMENT", "0.0005")),
        help="Minimum recent running-best improvement that counts as escaping a plateau.",
    )
    parser.add_argument(
        "--plateau-family-overuse-window",
        type=int,
        default=int(os.environ.get("RECCLAW_PLATEAU_FAMILY_OVERUSE_WINDOW", "20")),
        help="Recent metric-bearing rows used to detect over-concentrated local family repairs.",
    )
    parser.add_argument(
        "--max-same-family-repair-streak",
        type=int,
        default=int(os.environ.get("RECCLAW_MAX_SAME_FAMILY_REPAIR_STREAK", "8")),
        help="Maximum recent rows from one family before anti-plateau caps local repair.",
    )
    parser.add_argument(
        "--plateau-weak-family-ceiling",
        type=float,
        default=float(os.environ.get("RECCLAW_PLATEAU_WEAK_FAMILY_CEILING", "0.280")),
        help="Recent-best ceiling below which overused local repairs are treated as weak plateau behavior.",
    )
    parser.add_argument(
        "--anchor-families",
        default=os.environ.get(
            "RECCLAW_ANCHOR_FAMILIES",
            "cand_bpr_hard_negative_margin,cand_lightgcn_shallow_layers,"
            "cand_lightgcn_residual_norm_constrained,cand_lightgcn_edge_dropout_residual_norm",
        ),
        help="Comma-separated high-potential family anchors protected from ordinary discard freezing",
    )
    parser.add_argument(
        "--disable-implementation-smoke",
        action="store_true",
        help="Skip short smoke training after auto-implementing code_required proposals",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("deepseek", "openai", "compatible"),
        default=os.environ.get("RECCLAW_LLM_PROVIDER", "deepseek"),
        help="LLM provider. openai enables strict JSON schema responses for planner/proposals.",
    )
    parser.add_argument("--llm-model", default="", help="Model for --proposal-source llm")
    parser.add_argument(
        "--llm-base-url",
        default="",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--llm-api-key-env", default="", help="Environment variable containing API key")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM proposal sampling temperature")
    parser.add_argument("--llm-timeout", type=int, default=120, help="LLM API timeout in seconds")
    parser.add_argument("--llm-max-tokens", type=int, default=4096, help="Maximum LLM output tokens")
    parser.add_argument("--llm-retries", type=int, default=2, help="Retry count for transient LLM API transport errors")
    parser.add_argument(
        "--allow-llm-fallback",
        action="store_true",
        help="Allow heuristic fallback if DeepSeek-compatible LLM planning/proposal generation fails",
    )
    parser.add_argument("--enable-seed-validation", action="store_true", help="Run multi-seed validation for keep decisions")
    parser.add_argument(
        "--validation-seeds",
        default="2026,2027,2028",
        help="Comma-separated seeds for keep validation",
    )
    parser.add_argument(
        "--metric-lower-is-better",
        action="append",
        default=[],
        help="Metric names with lower-is-better semantics; repeatable",
    )
    args = parser.parse_args()

    metrics_weights = AgentConfig().metrics_weights
    if args.metrics_weights_json:
        metrics_weights = normalize_weights(args.metrics_weights_json)
    policy = LOOP_MODE_POLICIES[args.loop_mode]
    selected_proposal_mode = args.proposal_mode or str(policy["proposal_mode"])
    enable_candidate_proposals = not args.disable_candidate_proposals and (
        args.enable_candidate_proposals or args.loop_mode in LOOP_MODE_POLICIES
    )
    validation_seeds = [int(item.strip()) for item in str(args.validation_seeds).split(",") if item.strip()]
    anchor_families = [item.strip() for item in str(args.anchor_families).split(",") if item.strip()]
    llm_provider = str(args.llm_provider)
    if llm_provider == "openai":
        default_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
        default_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        default_key_env = "OPENAI_API_KEY"
    elif llm_provider == "deepseek":
        default_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        default_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        default_key_env = "DEEPSEEK_API_KEY"
    else:
        default_model = os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL", "")
        default_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", "")
        default_key_env = os.environ.get("RECCLAW_LLM_API_KEY_ENV", "OPENAI_API_KEY")
    llm_model = args.llm_model or default_model
    llm_base_url = args.llm_base_url or default_base_url
    llm_api_key_env = args.llm_api_key_env or default_key_env
    experiment_directive = resolve_experiment_directive(
        directive=args.experiment_directive,
        directive_file=args.experiment_directive_file,
        disabled=bool(args.disable_experiment_directive),
    )

    config = AgentConfig(
        rounds=max(max(1, args.rounds), max(1, args.start_round)),
        start_round=max(1, args.start_round),
        metric=args.metric,
        multi_metrics=args.multi_metrics,
        metrics_weights=metrics_weights,
        seed=args.seed,
        dry_run=args.dry_run,
        loop_mode=args.loop_mode,
        memory_path=Path(args.memory_path),
        state_summary_path=Path(args.state_summary_path),
        results_csv=Path(args.results_csv),
        baseline_dir=Path(args.baseline_dir),
        registry_path=Path(args.registry_path),
        memory_read_limit=max(0, args.memory_read_limit),
        prompt_memory_tail=max(0, args.prompt_memory_tail),
        use_experiment_directive=bool(experiment_directive),
        experiment_directive=experiment_directive,
        static_experiment_directive="" if args.disable_experiment_directive else str(args.experiment_directive or ""),
        candidate_tree_path=Path(args.candidate_tree_path),
        candidate_tree_md_path=Path(args.candidate_tree_md_path),
        candidate_tree_mmd_path=Path(args.candidate_tree_mmd_path),
        experience_summary_path=Path(args.experience_summary_path),
        experience_summary_json_path=Path(args.experience_summary_json_path),
        reflection_memory_path=Path(args.reflection_memory_path),
        search_policy_path=Path(args.search_policy_path),
        research_route_path=Path(args.research_route_path),
        research_line_enabled=not bool(args.disable_research_line),
        refresh_experience_every=max(0, args.refresh_experience_every),
        recent_schedule_penalty=max(0.0, args.recent_schedule_penalty),
        enable_candidate_proposals=enable_candidate_proposals,
        proposal_path=Path(args.proposal_path),
        proposal_mode=selected_proposal_mode,
        proposal_count=max(1, args.proposal_count),
        proposal_every=max(1, args.proposal_every),
        proposal_bonus=max(0.0, args.proposal_bonus),
        proposal_source=args.proposal_source,
        auto_promote_needs_review=bool(policy["auto_promote"] or args.auto_promote_needs_review),
        auto_implement_code_required=bool(policy["auto_implement"]),
        max_pending_implemented=max(1, args.max_pending_implemented),
        max_implement_per_round=max(0, args.max_implement_per_round),
        implementation_smoke_run=not bool(args.disable_implementation_smoke),
        search_intensity=str(args.search_intensity),
        algorithm_budget_per_window=max(0, args.algorithm_budget_per_window),
        algorithm_first_explore_rounds=max(0, args.algorithm_first_explore_rounds),
        seed_validation_min_metric=max(0.0, args.seed_validation_min_metric),
        anti_plateau_enabled=not bool(args.disable_anti_plateau),
        plateau_window_metric_rows=max(1, args.plateau_window_metric_rows),
        plateau_min_global_improvement=max(0.0, args.plateau_min_global_improvement),
        plateau_family_overuse_window=max(1, args.plateau_family_overuse_window),
        max_same_family_repair_streak=max(1, args.max_same_family_repair_streak),
        plateau_weak_family_ceiling=max(0.0, args.plateau_weak_family_ceiling),
        anchor_families=anchor_families or AgentConfig().anchor_families,
        global_overrides=list(args.global_overrides),
        checkpoint_dir=str(args.checkpoint_dir or ""),
        checkpoint_policy=str(args.checkpoint_policy),
        enable_seed_validation=bool(args.enable_seed_validation),
        validation_seeds=validation_seeds or list(DEFAULT_VALIDATION_SEEDS),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key_env=llm_api_key_env,
        llm_temperature=max(0.0, args.llm_temperature),
        llm_timeout=max(1, args.llm_timeout),
        llm_max_tokens=max(256, args.llm_max_tokens),
        llm_retries=max(0, args.llm_retries),
        allow_llm_fallback=bool(args.allow_llm_fallback),
        metric_directions={
            **DEFAULT_METRIC_DIRECTIONS,
            **{str(name).lower(): -1 for name in args.metric_lower_is_better},
        },
    )
    needs_llm = (
        (config.enable_candidate_proposals and config.proposal_source == "llm")
        or config.loop_mode == "auto"
        or (config.auto_implement_code_required and not config.dry_run)
    )
    if needs_llm and not config.allow_llm_fallback and not os.environ.get(config.llm_api_key_env, ""):
        raise SystemExit(f"missing LLM API key env var: {config.llm_api_key_env}")
    RecClawAgent(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
