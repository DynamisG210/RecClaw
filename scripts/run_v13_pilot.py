#!/usr/bin/env python3
"""Run the frozen V13 Pilot once.

This file is an entrypoint only. Importing it or verifying the contract does
not create roots, call the laboratory API, or start training.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from freeze_v13_pilot_contract import (  # noqa: E402
    DEFAULT_LLM_CONFIG,
    DEFAULT_OUTPUT,
    file_sha256,
    verify_v13_pilot_contract,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v13 import (  # noqa: E402
    V13_PILOT_ROUNDS_PER_ARM,
    V13_PILOT_SEARCH_SEED,
    V13PilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    LabApiCanaryBrokerV1,
    load_lab_api_credentials,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    MetaV18CampaignRuntimeV1,
)
from recclaw_core.experiments.helix_abc_v1.pilot_analysis import (  # noqa: E402
    FrontierProjectionV13,
    analysis_observation_key,
    four_axis_frontiers,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    campaign_training_runtime_release,
)


def _write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    path.chmod(mode)


def _open_snapshot(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?immutable=1", uri=True)


def _collect_analysis_rows(
    state_db: Path,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    connection = _open_snapshot(state_db)
    connection.row_factory = sqlite3.Row
    try:
        rounds = connection.execute(
            """
            SELECT r.round_id, r.arm_instance_id, r.round_index,
                   r.terminal_class, a.relative_path, e.payload_json
            FROM rounds r
            LEFT JOIN artifact_index a
              ON a.round_id=r.round_id
             AND a.artifact_type='RAW_RESULT_ENVELOPE_V2'
            JOIN round_events e
              ON e.round_id=r.round_id
             AND e.event_type='ROUND_FEEDBACK'
            WHERE r.status='CLOSED'
            ORDER BY r.round_index, r.arm_instance_id
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
    rows: list[dict[str, Any]] = []
    eligibility: dict[str, str] = {}
    for row in rounds:
        envelope = (
            json.loads(
                (artifact_root / str(row["relative_path"])).read_text(
                    encoding="utf-8"
                )
            )
            if row["relative_path"] is not None
            else None
        )
        payload = json.loads(str(row["payload_json"]))
        feedback = dict(payload["feedback"])
        round_id = str(row["round_id"])
        metrics = (
            dict(envelope["normalized_metrics"])
            if envelope is not None
            else {}
        )
        item = {
            "analysis_row_id": sha256_digest(
                {
                    "arm_instance_id": str(row["arm_instance_id"]),
                    "round_id": round_id,
                }
            ),
            "billed_tokens": debits.get(
                (round_id, "BILLED_TOKEN_DEBIT"), 0
            ),
            "candidate_id": (
                str(envelope["candidate_id"])
                if envelope is not None
                else None
            ),
            "gpu_cost_microunits": debits.get(
                (round_id, "GPU_COST_MICROUNITS"), 0
            ),
            "ndcg": metrics.get("ndcg"),
            "observation_seed": (
                str(envelope.get("seed", "NOT_EXECUTED"))
                if envelope is not None
                else "NOT_EXECUTED"
            ),
            "opaque_instance_id": str(row["arm_instance_id"]),
            "ordinary_execution_count": debits.get(
                (round_id, "ORDINARY_EXECUTION"), 0
            ),
            "round_index": int(row["round_index"]),
            "run_status": (
                str(envelope["exit_status"])
                if envelope is not None
                else str(row["terminal_class"])
            ),
        }
        rows.append(item)
        eligibility[analysis_observation_key(item)] = str(
            feedback["frontier_eligibility"]
        )
    return rows, eligibility


def _resource_audit(
    state_db: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    connection = _open_snapshot(state_db)
    try:
        rows = connection.execute(
            """
            SELECT r.round_id, r.round_index, l.dimension, SUM(l.quantity)
            FROM resource_ledger l JOIN rounds r USING(round_id)
            GROUP BY r.round_id, r.round_index, l.dimension
            ORDER BY r.round_index, r.round_id, l.dimension
            """
        ).fetchall()
        budget_identity_count = int(
            connection.execute(
                "SELECT COUNT(DISTINCT budget_snapshot_digest) FROM rounds"
            ).fetchone()[0]
        )
    finally:
        connection.close()
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
    ceilings = contract["budget_per_arm_round"]
    violations = [
        {
            "ceiling": int(ceilings[mapping[str(row[2])]]),
            "dimension": str(row[2]),
            "quantity": int(row[3]),
            "round_id": str(row[0]),
            "round_index": int(row[1]),
        }
        for row in rows
        if str(row[2]) in mapping
        and int(row[3]) > int(ceilings[mapping[str(row[2])]])
    ]
    return {
        "budget_snapshot_identity_count": budget_identity_count,
        "closed": not violations and budget_identity_count == 1,
        "rows_digest": sha256_digest([list(row) for row in rows]),
        "violations": violations,
    }


def _preflight(
    contract: Mapping[str, Any], llm_api_config: Path
) -> dict[str, Any]:
    base_url, api_key = load_lab_api_credentials(llm_api_config)
    if sha256_digest({"base_url": base_url.rstrip("/")}) != contract[
        "broker"
    ]["endpoint_digest"]:
        raise RuntimeError("V13 laboratory endpoint identity changed")
    if not api_key:
        raise RuntimeError("V13 laboratory API credential is unavailable")
    return {
        "broker_model": contract["broker"]["model"],
        "broker_release_digest": contract["broker"]["release_digest"],
        "credential_available": True,
        "executable_profile_digest": contract["common_substrate"][
            "executable_profile_digest"
        ],
        "meta_policy_bundle_digest": contract["meta"][
            "policy_bundle_digest"
        ],
        "training_release_digest": contract["training"]["release_digest"],
        "verdict": "PASS",
    }


def execute_campaign_pilot(
    contract_path: Path,
    llm_api_config: Path,
    *,
    verify_contract: Any,
    meta_runtime_class: Any,
    orchestrator_class: Any,
    search_seed: int,
    rounds_per_arm: int,
    version_label: str,
) -> int:
    contract = verify_contract(contract_path)
    output_root = Path(contract["output_root"])
    output_root.mkdir(parents=True)
    _write_json(
        output_root / "FROZEN_CONTRACT_IDENTITY.json",
        {
            "contract_content_digest": contract["content_digest"],
            "contract_sha256": file_sha256(contract_path),
        },
    )
    _write_json(
        output_root / "ENVIRONMENT_PREFLIGHT.json",
        _preflight(contract, llm_api_config),
    )
    meta_runtime = meta_runtime_class(
        checkpoint_path=Path(contract["meta"]["checkpoint_path"]),
        experiment_id=contract["pilot"]["experiment_id"],
        search_seed=search_seed,
        scheduled_rounds=rounds_per_arm,
        task_scale=float(contract["meta"]["task_context"]["task_scale"]),
        task_density=float(
            contract["meta"]["task_context"]["task_density"]
        ),
    )
    upstream = LabApiCanaryBrokerV1(
        output_root / "broker_private",
        schema_path=Path(contract["broker"]["response_schema_path"]),
        config_path=llm_api_config,
        model=contract["broker"]["model"],
        max_total_tokens_per_call=int(
            contract["broker"]["max_total_tokens_per_call"]
        ),
        timeout_ms=int(contract["broker"]["timeout_ms"]),
        release_manifest_path=Path(
            contract["broker"]["release_manifest_path"]
        ),
    )
    broker = RealCanaryProposalBrokerV1.create_v13(
        upstream=upstream,
        template_path=(
            ROOT
            / "src"
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "resources"
            / "campaign_anchor_programs_v1.json"
        ),
        repository_root=ROOT,
        search_seed=search_seed,
        call_prefix=f"campaign-{version_label.lower()}-",
        phase_name=f"Campaign Pilot {version_label}",
        adaptive_memory=True,
        campaign_meta_runtime=meta_runtime,
    )
    try:
        with orchestrator_class(
            output_root / "runtime",
            broker=broker,
            meta_runtime=meta_runtime,
            project_root=ROOT,
            recbole_root=Path(contract["training"]["recbole_root"]),
            data_path=Path(contract["dataset"]["search_parent"]),
            python_executable=Path(contract["training"]["python"]),
        ) as orchestrator:
            rounds = orchestrator.run_pilot()
            audit_bundle = orchestrator.immutable_audit_bundle(
                output_root / "audit_snapshots"
            )
            mapping = {
                arm.value: opaque
                for arm, opaque in orchestrator.assignment.arm_to_instance
            }
            result_rows = [
                [item.to_dict() for item in triplet]
                for triplet in rounds
            ]
            _write_json(
                output_root / "sealed" / "ROUND_RESULTS.json",
                result_rows,
                mode=0o600,
            )
            _write_json(
                output_root / "sealed" / "TREATMENT_MAPPING.json",
                {
                    "assignment_commitment": (
                        orchestrator.assignment.commitment
                    ),
                    "mapping": mapping,
                    "nonce_digest": orchestrator.assignment.nonce_digest,
                },
                mode=0o600,
            )
            _write_json(
                output_root / "sealed" / f"META_{version_label}_AUDIT.json",
                meta_runtime.audit_projection(),
                mode=0o600,
            )
            queue_audit = {
                orchestrator.assignment.mapping[arm]: [
                    task.to_dict()
                    for task in orchestrator.research_task_queues[arm].tasks
                ]
                for arm in orchestrator.research_task_queues
            }
            _write_json(
                output_root
                / "audit_snapshots"
                / "RESEARCH_TASK_QUEUE_AUDIT.json",
                queue_audit,
            )
    finally:
        upstream.close()

    state_db = (
        output_root
        / "audit_snapshots"
        / "neutral_state.audit.sqlite3"
    )
    rows, eligibility = _collect_analysis_rows(
        state_db,
        output_root / "runtime" / "neutral" / "artifacts",
    )
    observed = four_axis_frontiers(
        rows,
        frontier_projection=FrontierProjectionV13.OBSERVED,
    )
    search_eligible = four_axis_frontiers(
        rows,
        frontier_projection=FrontierProjectionV13.SEARCH_ELIGIBLE,
        eligibility_by_observation=eligibility,
    )
    resource = _resource_audit(state_db, contract)
    completed_validation_tasks = sum(
        1
        for tasks in queue_audit.values()
        for task in tasks
        if task["task_type"] == "VALIDATE_SAME_CANDIDATE"
        and task["task_status"] == "COMPLETED"
    )
    per_instance = {
        opaque: [row for row in rows if row["opaque_instance_id"] == opaque]
        for opaque in sorted(
            {str(row["opaque_instance_id"]) for row in rows}
        )
    }
    scientific_readiness = {
        "at_least_one_exact_validation_task_lifecycle": (
            completed_validation_tasks >= 1
        ),
        "completed_validation_task_count": completed_validation_tasks,
        "per_opaque_instance": {
            opaque: {
                "closed_result_count": len(instance_rows),
                "distinct_executed_candidate_count": len(
                    {
                        row["candidate_id"]
                        for row in instance_rows
                        if row["candidate_id"] is not None
                    }
                ),
                "execution_failure_count": sum(
                    row["ordinary_execution_count"] == 1
                    and row["run_status"] != "SUCCESS"
                    for row in instance_rows
                ),
                "no_execution_count": sum(
                    row["ordinary_execution_count"] == 0
                    for row in instance_rows
                ),
            }
            for opaque, instance_rows in per_instance.items()
        },
    }
    engineering = {
        "all_planned_round_rows_closed": len(rows)
        == rounds_per_arm * 3,
        "one_row_per_instance_round": len(
            {
                (
                    row["opaque_instance_id"],
                    row["round_index"],
                )
                for row in rows
            }
        )
        == rounds_per_arm * 3,
        "budget_closed": resource["closed"],
        "source_unchanged": all(
            file_sha256(ROOT / relative) == digest
            for relative, digest in contract["source"]["files"].items()
        ),
        "training_release_unchanged": (
            campaign_training_runtime_release().digest
            == contract["training"]["release_digest"]
        ),
    }
    automated_preconditions_pass = all(engineering.values()) and all(
        (
            scientific_readiness[
                "at_least_one_exact_validation_task_lifecycle"
            ],
            resource["closed"],
        )
    )
    report = {
        "analysis_class": "DEVELOPMENT_PILOT_READINESS_ONLY",
        "authority": "NONE",
        "automated_preconditions_pass": automated_preconditions_pass,
        "confirmed_frontier": "NOT_COMPUTED",
        "engineering_checks": engineering,
        "frontiers": {
            "observed": observed,
            "search_eligible": search_eligible,
        },
        "formal_inference": False,
        "go": "PENDING_INDEPENDENT_AUDIT",
        "main_authorized": False,
        "resource_audit": resource,
        "scientific_readiness": scientific_readiness,
        "treatment_effect": "NOT_AUTHORIZED",
    }
    _write_json(
        output_root / f"{version_label}_PILOT_READINESS_REPORT.json",
        report,
    )
    _write_json(output_root / "IMMUTABLE_AUDIT_BUNDLE.json", audit_bundle)
    return 0 if automated_preconditions_pass else 2


def execute(contract_path: Path, llm_api_config: Path) -> int:
    return execute_campaign_pilot(
        contract_path,
        llm_api_config,
        verify_contract=verify_v13_pilot_contract,
        meta_runtime_class=MetaV18CampaignRuntimeV1,
        orchestrator_class=V13PilotOrchestratorV1,
        search_seed=V13_PILOT_SEARCH_SEED,
        rounds_per_arm=V13_PILOT_ROUNDS_PER_ARM,
        version_label="V13",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--llm-api-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG,
    )
    args = parser.parse_args()
    return execute(args.contract.resolve(), args.llm_api_config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
