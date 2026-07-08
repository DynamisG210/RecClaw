"""Claim-boundary verification for Research Core v0.1 outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from recclaw_core.contracts import FORBIDDEN_CLAIM_FLAGS, load_yaml


def verify_claim_boundary(report: Mapping[str, Any]) -> Dict[str, Any]:
    issues = []
    for key in sorted(FORBIDDEN_CLAIM_FLAGS):
        if report.get(key) is not False:
            issues.append(f"{key}_must_be_false")
    if report.get("supported_scope") != "candidate_foundry_fixture_replay_only":
        issues.append("supported_scope_must_be_fixture_replay_only")
    if report.get("selected_candidate_queue_written") is not True:
        issues.append("selected_candidate_queue_written_missing")
    return {
        "status": "claim_boundary_passed" if not issues else "claim_boundary_failed",
        "issues": issues,
    }


def verify_claim_boundary_file(path: Path) -> Dict[str, Any]:
    return verify_claim_boundary(load_yaml(path))
