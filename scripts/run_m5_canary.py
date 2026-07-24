#!/usr/bin/env python3
"""Execute the exact frozen M5 development Canary once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any


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
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryOrchestratorV1,
    RealCanaryProposalBrokerV1,
    environment_preflight,
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
        raise RuntimeError("Canary contract content digest mismatch")
    for relative, expected_hash in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f"Canary source identity mismatch: {relative}")
    if (
        file_sha256(Path(contract["broker"]["response_schema_path"]))
        != contract["broker"]["response_schema_sha256"]
    ):
        raise RuntimeError("Canary response schema identity mismatch")
    if (
        file_sha256(Path(contract["bl_icf"]["template_fixture_path"]))
        != contract["bl_icf"]["template_fixture_sha256"]
    ):
        raise RuntimeError("Canary template identity mismatch")
    if (
        file_sha256(Path(contract["broker"]["codex_executable"]))
        != contract["broker"]["codex_executable_sha256"]
    ):
        raise RuntimeError("Codex executable identity mismatch")
    if contract["status"] != "FROZEN_PRE_OUTCOME":
        raise RuntimeError("Canary contract is not frozen")
    return contract


def source_snapshot(contract: dict[str, Any]) -> dict[str, str]:
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


def resource_audit(db_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
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
    finally:
        connection.close()
    by_arm_round: dict[tuple[str, int], dict[str, int]] = {}
    for arm, round_index, dimension, quantity in rows:
        by_arm_round.setdefault((str(arm), int(round_index)), {})[
            str(dimension)
        ] = int(quantity)
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
    for key, debits in by_arm_round.items():
        for dimension, ceiling_field in mapping.items():
            if debits.get(dimension, 0) > int(ceilings[ceiling_field]):
                violations.append(
                    {
                        "arm_round": list(key),
                        "ceiling": int(ceilings[ceiling_field]),
                        "dimension": dimension,
                        "quantity": debits.get(dimension, 0),
                    }
                )
    paired_equal = all(
        by_arm_round.get(("B", index), {})
        == by_arm_round.get(("C", index), {})
        for index in range(1, int(contract["canary"]["rounds_per_arm"]) + 1)
    )
    return {
        "budget_rows_digest": sha256_digest([list(item) for item in rows]),
        "closed": not violations,
        "paired_b_c_actual_debits_equal": paired_equal,
        "violations": violations,
    }


def execute(contract_path: Path, output_root: Path) -> int:
    if output_root.exists():
        raise RuntimeError("Canary output root already exists")
    output_root.mkdir(parents=True)
    contract = verify_contract(contract_path)
    write_json(
        output_root / "FROZEN_CONTRACT_IDENTITY.json",
        {
            "contract_content_digest": contract["content_digest"],
            "contract_sha256": file_sha256(contract_path),
        },
    )
    preflight = environment_preflight(contract)
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
    )
    try:
        with RealCanaryOrchestratorV1(
            output_root / "runtime", broker=broker
        ) as orchestrator:
            rounds = orchestrator.run_canary()
            audit = orchestrator.canary_audit()
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
        upstream.close()
        calls = broker_export(output_root / "broker_private" / "broker.sqlite3")
        after = source_snapshot(contract)
        no_source_mutation = before == after
        expected_rounds = 3 * int(contract["canary"]["rounds_per_arm"])
        expected_calls = (
            int(contract["broker"]["upstream_calls_per_round"])
            * int(contract["canary"]["rounds_per_arm"])
        )
        gates = {
            "b_c_controller_identity_equal": audit[
                "bc_controller_identity_equal"
            ],
            "barriers_closed": all(
                row[1:] == [7, 1] for row in audit["barriers"]
            ),
            "broker_call_count": audit["broker_successful_upstream_calls"],
            "budget_accounting_closed": resource["closed"],
            "cross_arm_mutation_count": 0,
            "execution_count": audit["execution_count"],
            "feedback_count": audit["feedback_count"],
            "guard_call_count": audit["guard_call_count"],
            "identity_mismatch_count": 0,
            "no_source_mutation": no_source_mutation,
            "paired_b_c_actual_debits_equal": resource[
                "paired_b_c_actual_debits_equal"
            ],
            "round_count": audit["round_count"],
            "state_store_integrity": (
                audit["state_store_integrity"]["integrity_check"] == "ok"
                and not audit["state_store_integrity"][
                    "foreign_key_violations"
                ]
            ),
        }
        passed = (
            gates["b_c_controller_identity_equal"]
            and gates["barriers_closed"]
            and gates["broker_call_count"] == expected_calls
            and gates["budget_accounting_closed"]
            and gates["execution_count"] == expected_rounds
            and gates["feedback_count"] == expected_rounds
            and gates["guard_call_count"]
            == 2 * int(contract["canary"]["rounds_per_arm"])
            and gates["no_source_mutation"]
            and gates["paired_b_c_actual_debits_equal"]
            and gates["round_count"] == expected_rounds
            and gates["state_store_integrity"]
        )
        result = {
            "authority": "NONE",
            "broker_calls_digest": sha256_digest(calls),
            "contract_content_digest": contract["content_digest"],
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "gates": gates,
            "resource_audit": resource,
            "source_snapshot_digest": sha256_digest(after),
            "verdict": (
                "READY_FOR_PILOT_REVIEW"
                if passed
                else "BLOCKED_NEEDS_HUMAN_DECISION"
            ),
        }
        write_json(output_root / "BROKER_CALL_AUDIT.json", calls)
        write_json(output_root / "CANARY_EXECUTION_RESULT_V1.json", result)
        return 0 if passed else 2
    except Exception as error:
        try:
            upstream.close()
        except Exception:
            pass
        write_json(
            output_root / "CANARY_FAILURE.json",
            {
                "authority": "NONE",
                "error_type": type(error).__name__,
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
                "reason": str(error)[:500],
                "verdict": "BLOCKED_NEEDS_HUMAN_DECISION",
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
        / "m5"
        / "DEVELOPMENT_CANARY_CONTRACT_V1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.contract.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
