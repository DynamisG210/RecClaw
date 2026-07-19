"""Read-only post-run audit for the AB-002 integration Canary."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from recclaw_phase1.ab002_launcher import (
    CONTRACT,
    canonical_write,
    sha256_file,
)
from recclaw_phase1.ab002_preflight import exact_file_identities, tree_identity


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _arm_artifacts(root: Path, arm: str) -> dict[str, Any]:
    repetition = root / "runs" / arm / "search_seed_9001"
    pilot = repetition / "outputs" / "pilot"
    required = {
        "memory": pilot / "agent_memory.jsonl",
        "proposals": pilot / "candidate_proposals.jsonl",
        "results": pilot / "results.csv",
        "tree": pilot / "candidate_search_tree.json",
        "summary": pilot / "experience_summary.json",
        "budget": repetition / "candidate_execution_budget.jsonl",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"{arm} Canary artifacts missing: {missing}")
    memory = _jsonl(required["memory"])
    proposals = _jsonl(required["proposals"])
    budget = _jsonl(required["budget"])
    with required["results"].open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))
    implementation_events = [row for row in memory if row.get("event") == "implementation_result"]
    purposes = sorted({str(row.get("purpose") or "") for row in budget})
    if not proposals or not implementation_events:
        raise ValueError(f"{arm} Canary did not exercise proposal and implementation")
    if "smoke" not in purposes or "ordinary" not in purposes:
        raise ValueError(f"{arm} Canary did not exercise smoke and ordinary training")
    if not results:
        raise ValueError(f"{arm} Canary produced no candidate result rows")
    if len(budget) > 6:
        raise ValueError(f"{arm} Canary exceeded candidate execution budget")
    return {
        "artifact_paths": {name: str(path) for name, path in required.items()},
        "artifact_sha256": {name: sha256_file(path) for name, path in required.items()},
        "proposal_count": len(proposals),
        "implementation_event_count": len(implementation_events),
        "candidate_execution_count": len(budget),
        "execution_purposes": purposes,
        "result_row_count": len(results),
        "successful_result_count": sum(
            str(row.get("status") or "").lower() == "success" for row in results
        ),
    }


def _broker_audit(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = list(
            connection.execute(
                "SELECT * FROM requests WHERE pair_id='AB002-SEED-9001' "
                "ORDER BY request_sha256, match_index, arm"
            )
        )
    if not rows:
        raise ValueError("Canary broker contains no request records")
    per_arm = {
        arm: sum(row["arm"] == arm for row in rows) for arm in ("control", "treatment")
    }
    if not all(per_arm.values()) or any(count > 60 for count in per_arm.values()):
        raise ValueError("Canary broker per-arm request budget/coverage failed")
    grouped: dict[tuple[str, int], dict[str, sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault((row["request_sha256"], row["match_index"]), {})[row["arm"]] = row
    shared = []
    for key, arms in grouped.items():
        if set(arms) != {"control", "treatment"}:
            continue
        control = arms["control"]
        treatment = arms["treatment"]
        if control["canonical_request"] != treatment["canonical_request"]:
            raise ValueError("equal broker hash has unequal canonical request bytes")
        if control["response_body"] != treatment["response_body"]:
            raise ValueError("identical paired request received unequal response bytes")
        shared.append({"request_sha256": key[0], "match_index": key[1]})
    if not shared:
        raise ValueError("Canary broker proved no identical-response replay")
    incomplete = [
        (row["arm"], row["sequence_index"])
        for row in rows
        if row["state"] != "COMPLETE"
        or not row["response_sha256"]
        or not row["provider_request_id"]
        or not row["returned_model"]
        or not row["usage_json"]
    ]
    if incomplete:
        raise ValueError(f"Canary broker metadata incomplete: {incomplete}")
    return {
        "request_count_by_arm": per_arm,
        "identical_replay_count": len(shared),
        "identical_replays": shared,
        "db_sha256": sha256_file(db_path),
    }


def audit_canary(
    *, runtime_root: Path, preflight_path: Path, probes_path: Path
) -> dict[str, Any]:
    root = runtime_root.expanduser().resolve()
    preflight = _json(preflight_path)
    probes = _json(probes_path)
    pair_record = _json(root / "execution_records" / "pair_search_seed_9001.json")
    if pair_record.get("status") != "LOCAL_COMPLETE" or pair_record.get("exit_codes") != {
        "control": 0,
        "treatment": 0,
    }:
        raise ValueError("Canary pair execution did not complete both arms")
    control_manifest = _json(
        root / "runs" / "control" / "search_seed_9001" / "materialization_manifest.json"
    )
    treatment_manifest = _json(
        root / "runs" / "treatment" / "search_seed_9001" / "materialization_manifest.json"
    )
    if control_manifest["s0_source_tree_sha256"] != treatment_manifest["s0_source_tree_sha256"]:
        raise ValueError("Canary arms did not share exact S0")
    if control_manifest.get("control_guard_material_count") != 0:
        raise ValueError("Control loaded Guard material")

    arms = {arm: _arm_artifacts(root, arm) for arm in ("control", "treatment")}
    feedback_path = (
        root
        / "runs"
        / "treatment"
        / "search_seed_9001"
        / "outputs"
        / "pilot"
        / "evidence_guard_feedback.jsonl"
    )
    if not feedback_path.is_file():
        raise ValueError("Treatment Guard feedback was not persisted")
    feedback = _jsonl(feedback_path)
    phases = {str(row.get("phase") or "") for row in feedback}
    if "PRECHECK" not in phases or not phases.intersection({"POSTCHECK", "SEED_VALIDATION_POSTCHECK"}):
        raise ValueError("Treatment did not persist both Guard pre-check and post-check events")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    recbole_root = Path(contract["execution_environment"]["recbole_root"])
    recbole_after = exact_file_identities(
        recbole_root, contract["execution_environment"]["recbole_runtime_files_sha256"]
    )
    dataset_root = recbole_root / "dataset" / contract["protocol"]["dataset"]
    dataset_after, dataset_files_after = tree_identity(dataset_root)
    before = preflight["shared_state_before"]
    if dataset_after != before["dataset_tree_sha256"]:
        raise ValueError("shared dataset changed during Canary")
    if recbole_after != before["recbole_runtime_files_sha256"]:
        raise ValueError("shared RecBole changed during Canary")
    if not probes.get("fail_open_probe_passed") or not probes.get("quarantine_probe_passed"):
        raise ValueError("controlled fail-open/quarantine probes did not pass")

    return {
        "record_type": "RECCLAW_PHASE1_AB002_CANARY_REPORT",
        "schema_version": "recclaw.phase1.ab002.canary_report.v1",
        "status": "LOCAL_COMPLETE",
        "gate_status": "NOT_STARTED",
        "independent_canary_review_status": "NOT_STARTED",
        "canary_search_seed": 9001,
        "excluded_from_primary_outcomes": True,
        "same_s0": True,
        "control_guard_material_count": 0,
        "treatment_guard_feedback": {
            "event_count": len(feedback),
            "phases": sorted(phases),
            "sha256": sha256_file(feedback_path),
        },
        "arms": arms,
        "broker": _broker_audit(root / "broker" / "paired_llm.sqlite3"),
        "shared_state_after": {
            "dataset_tree_sha256": dataset_after,
            "dataset_files_sha256": dataset_files_after,
            "recbole_runtime_files_sha256": recbole_after,
        },
        "controlled_probes": {
            "record_sha256": sha256_file(probes_path),
            "fail_open_passed": True,
            "quarantine_passed": True,
            "scope": "CONTROLLED_INTEGRATION_PROBES_NOT_LIVE_OUTCOME_CLAIMS",
        },
        "canary_go_recommendation": "READY_FOR_INDEPENDENT_READ_ONLY_REVIEW",
        "full_ab_authorized": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only AB-002 Canary audit")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--preflight-record", type=Path, required=True)
    parser.add_argument("--controlled-probes-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = audit_canary(
        runtime_root=args.runtime_root,
        preflight_path=args.preflight_record,
        probes_path=args.controlled_probes_record,
    )
    canonical_write(args.output.expanduser().resolve(), record)
    print(json.dumps(record, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
