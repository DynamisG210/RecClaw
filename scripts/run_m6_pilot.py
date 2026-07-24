#!/usr/bin/env python3
"""Execute the exact frozen M6 development Pilot once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CodexCliCanaryBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.pilot_analysis import (  # noqa: E402
    four_axis_frontiers,
    pilot_readiness,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
    environment_preflight,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    PILOT_ROUNDS_PER_ARM,
    RealPilotOrchestratorV1,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    path.chmod(mode)


def verify_contract(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    content = dict(contract)
    expected = content.pop("content_digest")
    if sha256_digest(content) != expected:
        raise RuntimeError("Pilot contract content digest mismatch")
    for relative, expected_hash in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f"Pilot source identity mismatch: {relative}")
    exact_files = {
        Path(contract["broker"]["response_schema_path"]): contract["broker"][
            "response_schema_sha256"
        ],
        Path(contract["bl_icf"]["template_fixture_path"]): contract["bl_icf"][
            "template_fixture_sha256"
        ],
        Path(contract["training"]["profile_path"]): contract["training"][
            "profile_sha256"
        ],
        Path(contract["broker"]["codex_executable"]): contract["broker"][
            "codex_executable_sha256"
        ],
    }
    for path, expected_hash in exact_files.items():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Pilot external identity mismatch: {path}")
    if contract["status"] != "FROZEN_PRE_OUTCOME":
        raise RuntimeError("Pilot contract is not frozen")
    return contract


def source_snapshot(contract: Mapping[str, Any]) -> dict[str, str]:
    return {
        relative: file_sha256(ROOT / relative)
        for relative in sorted(contract["source"]["files"])
    }


def broker_export(db_path: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT logical_call_id, request_digest, response_digest,
                       input_tokens, cached_input_tokens, output_tokens,
                       total_tokens, latency_ms, returned_model, status,
                       error_type
                FROM calls ORDER BY logical_call_id
                """
            )
        ]
    finally:
        connection.close()


def resource_audit(
    db_path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT arm_code, round_index, dimension, SUM(quantity)
            FROM resource_ledger JOIN rounds USING(round_id)
            GROUP BY arm_code, round_index, dimension
            ORDER BY arm_code, round_index, dimension
            """
        ).fetchall()
        budget_digests = connection.execute(
            "SELECT DISTINCT budget_snapshot_digest FROM rounds"
        ).fetchall()
    finally:
        connection.close()
    ceilings = contract["budget_per_arm_round"]
    mapping = {
        "BILLED_TOKEN_DEBIT": "total_billed_token_debit",
        "COMMON_VALIDATION": "common_validation_count",
        "GPU_COST_MICROUNITS": "gpu_cost_microunits",
        "GPU_DEVICE_TIME_MS": "gpu_device_time_ms",
        "INPUT_TOKEN": "total_input_tokens",
        "ORDINARY_EXECUTION": "ordinary_executions",
        "OUTPUT_TOKEN": "total_output_tokens",
        "PROPOSAL": "total_proposal_count",
        "PROPOSAL_ATTEMPT": "proposal_attempt_debit",
        "RETRY": "retry_debit",
        "WALL_TIME_MS": "wall_time_ms",
    }
    violations = []
    for arm, round_index, dimension, quantity in rows:
        if dimension in mapping and int(quantity) > int(ceilings[mapping[dimension]]):
            violations.append(
                {
                    "arm": arm,
                    "ceiling": int(ceilings[mapping[dimension]]),
                    "dimension": dimension,
                    "quantity": int(quantity),
                    "round_index": int(round_index),
                }
            )
    return {
        "budget_snapshot_identity_count": len(budget_digests),
        "closed": not violations and len(budget_digests) == 1,
        "rows_digest": sha256_digest([list(row) for row in rows]),
        "violations": violations,
    }


def collect_rows(db_path: Path, artifact_root: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rounds = connection.execute(
            """
            SELECT r.round_id, r.arm_code, r.arm_instance_id, r.round_index,
                   a.relative_path
            FROM rounds r
            JOIN artifact_index a USING(round_id)
            WHERE a.artifact_type='RAW_RESULT_ENVELOPE_V1'
            ORDER BY r.round_index, r.arm_code
            """
        ).fetchall()
        debits = {
            (str(row[0]), str(row[1])): int(row[2])
            for row in connection.execute(
                """
                SELECT round_id, dimension, SUM(quantity)
                FROM resource_ledger
                GROUP BY round_id, dimension
                """
            )
        }
    finally:
        connection.close()
    rows = []
    for row in rounds:
        envelope = json.loads(
            (artifact_root / str(row["relative_path"])).read_text(
                encoding="utf-8"
            )
        )
        round_id = str(row["round_id"])
        metrics = dict(envelope["normalized_metrics"])
        rows.append(
            {
                "arm_code": str(row["arm_code"]),
                "billed_tokens": debits.get(
                    (round_id, "BILLED_TOKEN_DEBIT"), 0
                ),
                "candidate_id": str(envelope["candidate_id"]),
                "gpu_cost_microunits": debits.get(
                    (round_id, "GPU_COST_MICROUNITS"), 0
                ),
                "gpu_device_time_ms": debits.get(
                    (round_id, "GPU_DEVICE_TIME_MS"), 0
                ),
                "ndcg": metrics.get("ndcg"),
                "opaque_instance_id": str(row["arm_instance_id"]),
                "ordinary_execution_count": debits.get(
                    (round_id, "ORDINARY_EXECUTION"), 0
                ),
                "physical_call_count": debits.get(
                    (round_id, "PHYSICAL_LLM_CALL"), 0
                ),
                "round_id": round_id,
                "round_index": int(row["round_index"]),
                "run_status": str(envelope["exit_status"]),
                "wall_time_ms": debits.get((round_id, "WALL_TIME_MS"), 0),
            }
        )
    return rows


def pilot_environment_preflight(
    contract: Mapping[str, Any]
) -> dict[str, Any]:
    preflight = environment_preflight(contract)
    recbole_root = Path(contract["runtime"]["recbole_root"])
    commit = subprocess.run(
        ["git", "-C", str(recbole_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(recbole_root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(recbole_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if (
        commit != contract["runtime"]["recbole_git_commit"]
        or tree != contract["runtime"]["recbole_git_tree"]
        or status
    ):
        raise RuntimeError("Pilot RecBole runtime is not the exact clean worktree")
    preflight["recbole_clean_commit"] = commit
    preflight["recbole_clean_tree"] = tree
    return preflight


def execute(contract_path: Path, output_root: Path) -> int:
    if output_root.exists():
        raise RuntimeError("Pilot output root already exists")
    output_root.mkdir(parents=True)
    contract = verify_contract(contract_path)
    write_json(
        output_root / "FROZEN_CONTRACT_IDENTITY.json",
        {
            "contract_content_digest": contract["content_digest"],
            "contract_sha256": file_sha256(contract_path),
        },
    )
    preflight = pilot_environment_preflight(contract)
    write_json(output_root / "ENVIRONMENT_PREFLIGHT.json", preflight)
    before = source_snapshot(contract)
    upstream = CodexCliCanaryBrokerV1(
        output_root / "broker_private",
        schema_path=Path(contract["broker"]["response_schema_path"]),
        codex_executable=Path(contract["broker"]["codex_executable"]),
        model=contract["broker"]["model"],
        reasoning_effort=contract["broker"]["reasoning_effort"],
        service_tier=contract["broker"]["service_tier"],
        max_total_tokens_per_call=int(
            contract["broker"]["max_total_tokens_per_call"]
        ),
    )
    broker = RealCanaryProposalBrokerV1.create(
        upstream=upstream,
        template_path=Path(contract["bl_icf"]["template_fixture_path"]),
        call_prefix="pilot-",
        phase_name="Pilot",
        adaptive_memory=True,
    )
    try:
        with RealPilotOrchestratorV1(
            output_root / "runtime",
            broker=broker,
            project_root=ROOT,
            recbole_root=Path(contract["runtime"]["recbole_root"]),
            data_path=Path(contract["dataset"]["root"]).parent,
            python_executable=Path(contract["runtime"]["python"]),
        ) as orchestrator:
            rounds = orchestrator.run_pilot()
            audit = orchestrator.pilot_audit()
            neutral = [
                orchestrator.neutral_audit_projection(triplet)
                for triplet in rounds
            ]
            private_mapping = {
                arm.value: opaque
                for arm, opaque in orchestrator.assignment.arm_to_instance
            }
            write_json(
                output_root / "ROUND_RESULTS.json",
                [
                    [result.to_dict() for result in triplet]
                    for triplet in rounds
                ],
            )
            write_json(output_root / "NEUTRAL_AUDIT.json", neutral)
            write_json(
                output_root / "sealed" / "TREATMENT_MAPPING.json",
                {
                    "assignment_commitment": orchestrator.assignment.commitment,
                    "mapping": private_mapping,
                    "nonce_digest": orchestrator.assignment.nonce_digest,
                },
                mode=0o600,
            )
            state_db = orchestrator.store.db_path
            resource = resource_audit(state_db, contract)
            rows = collect_rows(
                state_db,
                output_root / "runtime" / "neutral" / "artifacts",
            )
        upstream.close()
        calls = broker_export(output_root / "broker_private" / "broker.sqlite3")
        after = source_snapshot(contract)
        readiness = pilot_readiness(
            rows,
            expected_instances=3,
            expected_rounds_per_instance=PILOT_ROUNDS_PER_ARM,
            guard_call_count=int(audit["guard_call_count"]),
            expected_guard_call_count=2 * PILOT_ROUNDS_PER_ARM,
            meta_versions=audit["meta_versions"],
        )
        expected_rounds = 3 * PILOT_ROUNDS_PER_ARM
        expected_calls = int(contract["broker"]["expected_upstream_calls"])
        gates = {
            "analysis_readiness": readiness["verdict"],
            "barriers_closed": bool(audit["barriers_closed"]),
            "broker_call_count": len(calls),
            "broker_failures": sum(row["status"] != "SUCCESS" for row in calls),
            "budget_accounting_closed": bool(resource["closed"]),
            "cross_arm_mutation_count": 0,
            "execution_count": int(audit["execution_count"]),
            "feedback_count": int(audit["feedback_count"]),
            "guard_call_count": int(audit["guard_call_count"]),
            "identity_mismatch_count": 0,
            "initial_research_identity": audit["initial_research_identity"],
            "no_source_mutation": before == after,
            "round_count": int(audit["round_count"]),
            "state_store_integrity": (
                audit["state_store_integrity"]["integrity_check"] == "ok"
                and not audit["state_store_integrity"]["foreign_key_violations"]
            ),
        }
        passed = (
            gates["analysis_readiness"] == "GO"
            and gates["barriers_closed"]
            and gates["broker_call_count"] == expected_calls
            and gates["broker_failures"] == 0
            and gates["budget_accounting_closed"]
            and gates["execution_count"] == expected_rounds
            and gates["feedback_count"] == expected_rounds
            and gates["guard_call_count"] == 2 * PILOT_ROUNDS_PER_ARM
            and gates["no_source_mutation"]
            and gates["round_count"] == expected_rounds
            and gates["state_store_integrity"]
        )
        write_json(output_root / "BROKER_CALL_AUDIT.json", calls)
        write_json(output_root / "PILOT_ITT_ROWS.json", rows)
        write_json(
            output_root / "FOUR_AXIS_FRONTIERS.json",
            four_axis_frontiers(rows),
        )
        write_json(output_root / "PILOT_READINESS.json", readiness)
        result = {
            "authority": "NONE",
            "contract_content_digest": contract["content_digest"],
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "gates": gates,
            "resource_audit": resource,
            "source_snapshot_digest": sha256_digest(after),
            "verdict": "GO" if passed else readiness["verdict"],
        }
        write_json(output_root / "PILOT_EXECUTION_RESULT_V1.json", result)
        return 0 if passed else 2
    except Exception as error:
        try:
            upstream.close()
        except Exception:
            pass
        write_json(
            output_root / "PILOT_FAILURE.json",
            {
                "authority": "NONE",
                "error_type": type(error).__name__,
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
                "reason": str(error)[:500],
                "verdict": "NOT_READY",
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "docs"
        / "research_line"
        / "m6"
        / "DEVELOPMENT_PILOT_CONTRACT_V1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.contract.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
