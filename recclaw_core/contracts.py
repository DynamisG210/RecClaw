"""Shared contracts for RecClaw Candidate Foundry v0.1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml


NATIVE_BPR_RANKCUT_MEMORY_ID = "mem_bpr_rankcut_activation_metric_mismatch"

FORBIDDEN_CLAIM_FLAGS = {
    "search_quality_supported",
    "metric_improvement_supported",
    "formal_success_supported",
    "M5_ready",
    "runtime_kernel_ready",
}

CANDIDATE_REQUIRED_FIELDS = {
    "candidate_id",
    "family",
    "candidate_role",
    "parent_anchor",
    "cited_memory_ids",
    "cited_mechanism_ids",
    "mechanism_hypothesis",
    "minimal_intervention",
    "activation_probe",
    "expected_metric_signature",
    "falsification_rule",
    "claim_ceiling",
    "policy_features",
}

POLICY_FEATURES = {
    "opportunity_score",
    "positive_memory_similarity",
    "activation_observability",
    "information_value",
    "implementation_feasibility",
    "metric_smoke_plausibility",
    "duplicate_or_negative_collision",
    "implementation_cost",
    "strong_baseline_risk",
    "protocol_risk",
}

DEFAULT_POLICY_WEIGHTS = {
    "opportunity_score": 0.25,
    "positive_memory_similarity": 0.20,
    "activation_observability": 0.15,
    "information_value": 0.15,
    "implementation_feasibility": 0.10,
    "metric_smoke_plausibility": 0.15,
    "duplicate_or_negative_collision": -0.25,
    "implementation_cost": -0.10,
    "strong_baseline_risk": -0.10,
    "protocol_risk": -0.10,
}


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(paths: Iterable[Path], *, relative_to: Path | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if relative_to is not None:
            try:
                display_path = path.relative_to(relative_to).as_posix()
            except ValueError:
                display_path = path.as_posix()
        else:
            display_path = path.as_posix()
        rows.append({"path": display_path, "sha256": sha256_file(path)})
    return rows
