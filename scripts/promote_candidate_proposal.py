#!/usr/bin/env python3
"""Promote implementation-ready candidate proposals into runnable registry entries.

This is deliberately conservative: it only promotes known local templates that
already have code/config support in this repository. Unknown code_required
proposals remain in needs_review for human or separate implementation work.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from action_space import load_action_space
    from validate_candidate_proposal import (
        NEEDS_REVIEW,
        load_jsonl,
        load_memory_param_signatures,
        load_yaml,
        validate_one,
    )
except ImportError:
    from .action_space import load_action_space
    from .validate_candidate_proposal import (
        NEEDS_REVIEW,
        load_jsonl,
        load_memory_param_signatures,
        load_yaml,
        validate_one,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "candidate_proposal_schema.yaml"
PROPOSAL_PATH = PROJECT_ROOT / "results" / "candidate_proposals.jsonl"
MEMORY_PATH = PROJECT_ROOT / "results" / "agent_memory.jsonl"
ACTION_SPACE_PATH = PROJECT_ROOT / "configs" / "action_space.yaml"

AUTO_PROMOTABLE_PARENTS: dict[str, dict[str, Any]] = {
    "cand_bpr_long_tail_reweight": {
        "entrypoint": "recclaw_ext.models.bpr_regularized:BPRLongTailReweight",
        "config": "configs/candidates/cand_bpr_long_tail_reweight.yaml",
    },
    "cand_bpr_popularity_regularized": {
        "entrypoint": "recclaw_ext.models.bpr_regularized:BPRPopularityRegularized",
        "config": "configs/candidates/cand_bpr_popularity_regularized.yaml",
    },
    "cand_bpr_norm_constrained": {
        "entrypoint": "recclaw_ext.models.bpr_regularized:BPRNormConstrained",
        "config": "configs/candidates/cand_bpr_norm_constrained.yaml",
    },
}


def import_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"entrypoint must use module:attribute format: {spec}")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def validate_proposals(
    *,
    proposals_path: Path,
    registry_path: Path,
    schema_path: Path,
    memory_path: Path,
    action_space_path: Path = ACTION_SPACE_PATH,
) -> list[dict[str, Any]]:
    schema = load_yaml(schema_path)
    action_space = load_action_space(action_space_path)
    registry = load_yaml(registry_path).get("candidates", [])
    if not isinstance(registry, list):
        raise ValueError(f"registry candidates must be a list: {registry_path}")

    registry_by_id = {str(item.get("candidate_id")): item for item in registry if item.get("candidate_id")}
    registry_ids = set(registry_by_id)
    seen_ids: set[str] = set()
    seen_param_signatures: set[str] = set()
    memory_param_signatures = load_memory_param_signatures(memory_path)
    results: list[dict[str, Any]] = []

    for line_no, proposal, parse_error in load_jsonl(proposals_path):
        if parse_error or proposal is None:
            continue
        result = validate_one(
            proposal,
            line_no=line_no,
            schema=schema,
            registry_by_id=registry_by_id,
            registry_ids=registry_ids,
            seen_ids=seen_ids,
            seen_param_signatures=seen_param_signatures,
            memory_param_signatures=memory_param_signatures,
            action_space=action_space,
        )
        result["proposal"] = proposal
        results.append(result)
    return results


def parent_candidate_ids_for_promotion(validation_rows: list[dict[str, Any]]) -> set[str]:
    parent_ids: set[str] = set()
    for row in validation_rows:
        if row.get("status") != NEEDS_REVIEW:
            continue
        if row.get("runnable_level") != "code_required":
            continue
        proposal = row.get("proposal") or {}
        if not isinstance(proposal, dict):
            continue
        parent_id = str(proposal.get("parent_candidate_id") or "")
        if parent_id in AUTO_PROMOTABLE_PARENTS:
            parent_ids.add(parent_id)
    return parent_ids


def check_parent_ready(parent_id: str) -> list[str]:
    errors: list[str] = []
    meta = AUTO_PROMOTABLE_PARENTS[parent_id]
    try:
        import_object(str(meta["entrypoint"]))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"entrypoint import failed for {parent_id}: {type(exc).__name__}: {exc}")

    config_path = PROJECT_ROOT / str(meta["config"])
    if not config_path.exists():
        errors.append(f"candidate config is missing for {parent_id}: {config_path}")
    else:
        config = load_yaml(config_path)
        if not config.get("model"):
            errors.append(f"candidate config is missing model for {parent_id}: {config_path}")
    return errors


def promote_registry_entries(registry_path: Path, parent_ids: set[str]) -> list[str]:
    payload = load_yaml(registry_path)
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"registry candidates must be a list: {registry_path}")

    promoted: list[str] = []
    for candidate in candidates:
        parent_id = str(candidate.get("candidate_id") or "")
        if parent_id not in parent_ids:
            continue
        meta = AUTO_PROMOTABLE_PARENTS[parent_id]
        already_runnable = bool(candidate.get("wired")) and str(candidate.get("status") or "") == "implemented"
        candidate["status"] = "implemented"
        candidate["wired"] = True
        candidate["runner_type"] = "model"
        candidate["entrypoint"] = meta["entrypoint"]
        if not already_runnable:
            promoted.append(parent_id)

    if promoted:
        registry_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote known needs_review candidate proposals.")
    parser.add_argument("--proposals", default=str(PROPOSAL_PATH), help="Candidate proposal JSONL path")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="candidate_registry.yaml path")
    parser.add_argument("--schema", default=str(SCHEMA_PATH), help="candidate_proposal_schema.yaml path")
    parser.add_argument("--action-space", default=str(ACTION_SPACE_PATH), help="action_space.yaml path")
    parser.add_argument("--memory", default=str(MEMORY_PATH), help="agent_memory.jsonl path")
    parser.add_argument("--dry-run", action="store_true", help="Report promotable parents without editing registry")
    args = parser.parse_args()

    proposals_path = Path(args.proposals)
    if not proposals_path.exists():
        raise FileNotFoundError(f"proposal file does not exist: {proposals_path}")

    validation_rows = validate_proposals(
        proposals_path=proposals_path,
        registry_path=Path(args.registry),
        schema_path=Path(args.schema),
        memory_path=Path(args.memory),
        action_space_path=Path(args.action_space),
    )
    parent_ids = parent_candidate_ids_for_promotion(validation_rows)

    checked: dict[str, list[str]] = {parent_id: check_parent_ready(parent_id) for parent_id in sorted(parent_ids)}
    ready_parent_ids = {parent_id for parent_id, errors in checked.items() if not errors}
    promoted = [] if args.dry_run else promote_registry_entries(Path(args.registry), ready_parent_ids)

    report = {
        "promoted_count": len(promoted),
        "ready_count": len(ready_parent_ids),
        "blocked_count": sum(1 for errors in checked.values() if errors),
        "promoted_parent_candidate_ids": promoted,
        "ready_parent_candidate_ids": sorted(ready_parent_ids),
        "blocked": {parent_id: errors for parent_id, errors in checked.items() if errors},
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if not report["blocked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
