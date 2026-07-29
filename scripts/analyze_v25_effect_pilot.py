#!/usr/bin/env python3
"""Run the frozen read-only V25 chain and development-effect analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.effect_pilot_analysis import (  # noqa: E402
    arm_trajectory_metrics,
    effect_pilot_verdict,
    evidence_guard_visibility,
    research_capability_visibility,
)
from run_v13_pilot import _collect_analysis_rows  # noqa: E402


DEFAULT_CONTRACT = (
    ROOT
    / "docs/research_line/continuous_program/"
    "V25_FROZEN_EFFECT_PILOT_CONTRACT.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _open_snapshot(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _verify_digest(value: Mapping[str, Any], field: str) -> bool:
    preimage = dict(value)
    expected = preimage.pop(field, None)
    return bool(expected) and sha256_digest(preimage) == expected


def _checkpoint_checks(
    output_root: Path,
    state_snapshot: Path,
) -> tuple[dict[str, bool], int]:
    connection = _open_snapshot(state_snapshot)
    try:
        artifacts = list(
            connection.execute(
                "SELECT artifact_type, relative_path, sha256 "
                "FROM artifact_index WHERE artifact_type IN "
                "('INCREMENTAL_NEUTRAL_BEHAVIORAL_CHECKPOINT_V1',"
                "'EFFECT_PILOT_READ_ONLY_CHECKPOINT_V1')"
            )
        )
    finally:
        connection.close()
    artifact_root = output_root / "runtime/neutral/artifacts"
    neutral = [
        row
        for row in artifacts
        if row["artifact_type"]
        == "INCREMENTAL_NEUTRAL_BEHAVIORAL_CHECKPOINT_V1"
    ]
    effect = [
        row
        for row in artifacts
        if row["artifact_type"]
        == "EFFECT_PILOT_READ_ONLY_CHECKPOINT_V1"
    ]
    artifact_hashes_valid = all(
        (
            path := artifact_root / str(row["relative_path"])
        ).is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest()
        == str(row["sha256"])
        and _verify_digest(_read(path), "checkpoint_digest")
        for row in artifacts
    )

    private_paths = sorted(
        (output_root / "runtime/instances").glob(
            "*/registry/behavioral_checkpoints/9227/"
            "*.arm-private.v1.json"
        )
    )
    seed_mismatches = 0
    private_digests_valid = True
    for path in private_paths:
        record = _read(path)
        private_digests_valid &= _verify_digest(
            record,
            "checkpoint_digest",
        )
        binding = record["execution_seed_binding"]
        execution_count = int(
            record["round_result"]["ordinary_execution_count"]
        )
        if binding == "ABSENT_NO_EXECUTION":
            seed_mismatches += int(execution_count != 0)
            continue
        seed_mismatches += int(
            execution_count != 1
            or int(binding["execution_seed"])
            != int(binding["required_seed_or_control"])
            or binding["candidate_instance_id"]
            != record["integrated_round"]["candidate_instance_id"]
        )
    return (
        {
            "effect_checkpoint_count_exact": len(effect) == 3,
            "incremental_neutral_checkpoint_count_exact": (
                len(neutral) == 50
            ),
            "checkpoint_artifact_hashes_valid": artifact_hashes_valid,
            "private_checkpoint_count_exact": len(private_paths) == 150,
            "private_checkpoint_digests_valid": private_digests_valid,
        },
        seed_mismatches,
    )


def _guard_metrics(
    *,
    arm_c_rows: list[dict[str, Any]],
    backend_audit: Mapping[str, Any],
    eligibility: Mapping[str, str],
    immutable_bundle: Mapping[str, Any],
    output_root: Path,
    seed_binding_mismatch_count: int,
) -> dict[str, int]:
    guard_snapshot = output_root / "audit_snapshots/guard_state.audit.sqlite3"
    connection = _open_snapshot(guard_snapshot)
    try:
        calls = [
            (str(row["phase"]), str(row["candidate_id"]), json.loads(row["full_event_json"]))
            for row in connection.execute(
                "SELECT phase, candidate_id, full_event_json FROM guard_calls"
            )
        ]
    finally:
        connection.close()
    pre_legal = {
        candidate_id
        for phase, candidate_id, event in calls
        if phase == "PRE"
        and event["action_legality"]["development_verdict"] == "LEGAL"
    }
    post_candidates = {
        candidate_id
        for phase, candidate_id, _event in calls
        if phase == "POST"
    }
    nontrivial = sum(
        phase == "POST"
        and (
            event["evidence_admissibility"]["development_disposition"]
            != "NOT_EVALUATED"
            or event["router_directive"] != "NO_CHANGE"
        )
        for phase, _candidate_id, event in calls
    )
    c_no_execution = {
        str(row["candidate_id"])
        for row in arm_c_rows
        if int(row["ordinary_execution_count"]) == 0
        and row.get("candidate_id") is not None
    }
    legal_suppression = len(
        (pre_legal - post_candidates) & c_no_execution
    )
    mapping = _read(output_root / "sealed/TREATMENT_MAPPING.json")["mapping"]
    queues = _read(
        output_root / "audit_snapshots/RESEARCH_TASK_QUEUE_AUDIT.json"
    )
    completed_validations = sum(
        task["task_type"] == "VALIDATE_SAME_CANDIDATE"
        and task["task_status"] == "COMPLETED"
        for task in queues[mapping["C"]]
    )
    safety = immutable_bundle["m6i_safety_projection"]
    challenge = backend_audit["guard_challenge_suite"]
    preliminary_confirmed = sum(
        "CONFIRMED" in disposition
        for disposition in eligibility.values()
    )
    return {
        "completed_validation_count": completed_validations,
        "cross_arm_contamination_count": int(
            safety["cross_arm_physical_identities"]
            + safety["integrated_cross_arm_reads"]
        ),
        "false_allow_count": int(challenge["false_allow_count"]),
        "false_block_count": int(challenge["false_block_count"]),
        "guard_private_input_leak_count": int(
            safety["guard_private_context_token_count"]
        ),
        "legal_candidate_permanent_suppression_count": legal_suppression,
        "nontrivial_intervention_count": nontrivial,
        "preliminary_marked_confirmed_count": preliminary_confirmed,
        "search_memory_pollution_count": int(
            safety["guard_private_context_token_count"]
        ),
        "seed_binding_mismatch_count": seed_binding_mismatch_count,
        "successful_challenge_case_count": int(
            challenge["successful_challenge_case_count"]
        ),
    }


def analyze(contract_path: Path) -> dict[str, Any]:
    contract = _read(contract_path)
    output_root = Path(contract["output_root"])
    criteria = _read(Path(contract["analysis"]["criteria_path"]))
    backend_audit = _read(
        Path(contract["backend_qualification"]["audit_path"])
    )
    readiness = _read(output_root / "V25_PILOT_READINESS_REPORT.json")
    immutable_bundle = _read(output_root / "IMMUTABLE_AUDIT_BUNDLE.json")
    meta = _read(output_root / "sealed/META_V25_AUDIT.json")
    state_snapshot = output_root / "audit_snapshots/neutral_state.audit.sqlite3"
    rows, eligibility = _collect_analysis_rows(
        state_snapshot,
        output_root / "runtime/neutral/artifacts",
    )
    mapping = _read(output_root / "sealed/TREATMENT_MAPPING.json")["mapping"]
    by_arm = {
        arm: [
            row
            for row in rows
            if row["opaque_instance_id"] == mapping[arm]
        ]
        for arm in ("A", "B", "C")
    }
    budgets = criteria["frozen_analysis_budgets_per_arm"]
    research_criteria = criteria["research_capability"]
    arm_metrics = {
        arm: arm_trajectory_metrics(
            arm_rows,
            eligibility_by_observation=eligibility,
            rounds_per_arm=int(budgets["rounds"]),
            token_budget_cap=int(budgets["tokens"]),
            gpu_cost_budget_cap=int(budgets["gpu_cost_microunits"]),
            useful_signal_delta=float(
                research_criteria["useful_signal_delta"]
            ),
            best_tolerance=float(research_criteria["best_tolerance"]),
        )
        for arm, arm_rows in by_arm.items()
    }
    producer_counts = Counter(
        item["decision"]["selected_producer_role"]
        for item in meta["producer_opportunity_decisions"]
        if item["arm"] == "B"
    )
    checkpoint_checks, seed_mismatches = _checkpoint_checks(
        output_root,
        state_snapshot,
    )
    guard_metrics = _guard_metrics(
        arm_c_rows=by_arm["C"],
        backend_audit=backend_audit,
        eligibility=eligibility,
        immutable_bundle=immutable_bundle,
        output_root=output_root,
        seed_binding_mismatch_count=seed_mismatches,
    )
    research = research_capability_visibility(
        arm_a=arm_metrics["A"],
        arm_b=arm_metrics["B"],
        criteria=criteria,
        producer_role_counts=producer_counts,
    )
    guard = evidence_guard_visibility(
        arm_b=arm_metrics["B"],
        arm_c=arm_metrics["C"],
        criteria=criteria,
        guard_metrics=guard_metrics,
    )
    state = _open_snapshot(state_snapshot)
    try:
        closed_barriers = int(
            state.execute(
                "SELECT COUNT(*) FROM triplet_barrier "
                "WHERE closed_bitmap=7 AND next_index_authorized=1"
            ).fetchone()[0]
        )
        open_rounds = int(
            state.execute(
                "SELECT COUNT(*) FROM rounds WHERE status='OPEN'"
            ).fetchone()[0]
        )
    finally:
        state.close()
    safety = immutable_bundle["m6i_safety_projection"]
    snapshot_verification = immutable_bundle["verification"]
    chain_checks = {
        **checkpoint_checks,
        "all_150_analysis_rows_present": len(rows) == 150,
        "all_50_triplet_barriers_closed": closed_barriers == 50,
        "generic_readiness_preconditions_pass": (
            readiness["automated_preconditions_pass"] is True
        ),
        "immutable_snapshots_verified": all(
            item["immutable_open"]
            and item["integrity_check"] == "ok"
            and item["sha256_match"]
            and item["sidecars_created"] == 0
            for item in snapshot_verification.values()
        ),
        "no_cross_arm_runtime_identity": (
            safety["cross_arm_physical_identities"] == 0
            and safety["integrated_cross_arm_reads"] == 0
        ),
        "no_open_rounds": open_rounds == 0,
        "runtime_round_count_exact": (
            safety["integrated_round_count"] == 150
            and safety["integrated_triplet_barrier_count"] == 50
        ),
        "source_unchanged": readiness["engineering_checks"][
            "source_unchanged"
        ],
    }
    verdict = effect_pilot_verdict(
        chain_checks=chain_checks,
        research_visibility=research,
        guard_visibility=guard,
        criteria_digest=criteria["criteria_digest"],
    )
    report = {
        **verdict,
        "arm_metrics": arm_metrics,
        "guard_metrics": guard_metrics,
        "intent_to_treat_row_count": len(rows),
        "primary_estimands": {
            "B_MINUS_A": research["deltas_b_minus_a"],
            "C_MINUS_B": guard["c_minus_b"],
        },
        "producer_role_counts_b": dict(sorted(producer_counts.items())),
        "record_schema": "recclaw.v25-effect-pilot-analysis.v1",
        "secondary_only": {
            "C_MINUS_A": {
                "best": (
                    arm_metrics["C"]["best_search_eligible_ndcg_at_10"]
                    - arm_metrics["A"]["best_search_eligible_ndcg_at_10"]
                ),
                "round_auc": (
                    arm_metrics["C"]["round_auc"]
                    - arm_metrics["A"]["round_auc"]
                ),
            }
        },
    }
    report_path = output_root / "V25_EFFECT_PILOT_ANALYSIS.json"
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    report = analyze(args.contract.resolve())
    print(
        json.dumps(
            {
                "analysis_digest": report["analysis_digest"],
                "chain_line": report["chain_line"],
                "effect_line": report["effect_line"],
                "verdict": report["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["chain_line"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
