#!/usr/bin/env python3
"""Run a fixed-candidate full-recipe V22 backend qualification canary."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.mechanism_space import compile_program  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.common_execution_guard import (  # noqa: E402
    CommonExecutionGuardV1,
)
from recclaw_core.experiments.helix_abc_v1.campaign_dataset import (  # noqa: E402
    campaign_development_protocol,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (  # noqa: E402
    campaign_runtime_profile,
    campaign_training_profile,
    executable_mechanism,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.materialization import (  # noqa: E402
    DeterministicMaterializerV1,
    build_binding_v2,
    classify_execution_trust,
    development_execution_gate,
    register_materialization_artifacts,
)
from recclaw_core.experiments.helix_abc_v1.pilot_training import (  # noqa: E402
    PilotTrainingLauncherV1,
    pilot_training_profile_digest,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (  # noqa: E402
    common_release_projection_digest,
    development_protocol,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    RealPilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    CloseRoundCommand,
    OpenRoundCommand,
    ResourceDebitV1,
)
from recclaw_core.experiments.helix_abc_v1.training_execution_guard import (  # noqa: E402
    CommonTrainingExecutionGuardV1,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (  # noqa: E402
    build_training_filesystem_capability,
)
from recclaw_core.experiments.helix_abc_v1.training_materialization import (  # noqa: E402
    build_training_binding_v3,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (  # noqa: E402
    TrainingCompatibilityStatusV1,
    TrainingExecutionPurposeV1,
    TrainingRuntimeCompatibilityFixtureV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    CAMPAIGN_TRAINING_RUNNER_ABI,
    TRAINING_RUNNER_ABI,
    build_training_runtime_binding,
    training_runtime_compatibility_preflight,
    training_runtime_component_abis,
)
from recclaw_core.experiments.helix_abc_v1.training_state_store import (  # noqa: E402
    ClaimTrainingExecutionCommandV1,
    TrainingSingleWriterExperimentStoreV1,
)
from recclaw_core.experiments.helix_abc_v1.store_audit import (  # noqa: E402
    experiment_store_audit_port,
)


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
CANARY_SEARCH_SEED = 9301


@dataclass(frozen=True, slots=True)
class FixedCanaryStoreContractV1:
    experiment_id: str
    arm_policies: tuple[Any, Any, Any]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str = "M6R-FIXED-TRAINING-CANARY-V1",
        search_seed: int = CANARY_SEARCH_SEED,
    ) -> "FixedCanaryStoreContractV1":
        base = default_experiment_contract()
        payload = {
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": experiment_id,
            "formal_acceptance": False,
            "ordinary_execution_seed": 2026,
            "scheduled_slots_per_arm_seed": 1,
            "search_seeds": [search_seed],
        }
        return cls(
            experiment_id=str(payload["experiment_id"]),
            arm_policies=base.arm_policies,
            search_seeds=(search_seed,),
            scheduled_slots_per_arm_seed=1,
            ordinary_execution_seed=2026,
            identity_digest=sha256_digest(payload),
        )


def canary_budget() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=0,
        total_output_tokens=0,
        total_billed_token_debit=0,
        total_proposal_count=0,
        wall_time_ms=7_200_000,
        retry_debit=0,
        proposal_attempt_debit=0,
        ordinary_executions=1,
        common_validation_count=4,
        gpu_device_time_ms=7_200_000,
        gpu_cost_microunits=2_000_000,
    )


def fixed_program(anchor_name: str = "BPR_MF") -> dict[str, Any]:
    document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for row in document["fixtures"]:
        if row["anchor_name"] == anchor_name:
            return copy.deepcopy(row["program"])
    raise RuntimeError(f"{anchor_name} fixture is missing")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def execute(
    *,
    output_root: Path,
    python_executable: Path,
    recbole_root: Path,
    data_path: Path,
    anchor_name: str = "BPR_MF",
    experiment_id: str = "M6R-FIXED-TRAINING-CANARY-V1",
    lineage_partition: str = "FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN",
    result_filename: str = "M6R_FIXED_TRAINING_CANARY_RESULT.json",
    search_seed: int = CANARY_SEARCH_SEED,
    expected_runtime_failure: bool = False,
    full_triplet_rehearsal: bool = False,
    campaign_mechanism_id: str | None = None,
) -> int:
    if output_root.exists():
        raise RuntimeError("M6R fixed canary output root already exists")
    output_root.mkdir(parents=True)
    contract = FixedCanaryStoreContractV1.create(
        experiment_id=experiment_id,
        search_seed=search_seed,
    )
    budget = canary_budget()
    runtime_root = output_root / "runtime"
    arm_root = runtime_root / "arm-a"
    purpose = TrainingExecutionPurposeV1.PILOT.value
    campaign_mode = campaign_mechanism_id is not None
    runner_abi = (
        CAMPAIGN_TRAINING_RUNNER_ABI
        if campaign_mode
        else TRAINING_RUNNER_ABI
    )
    common_protocol = (
        campaign_development_protocol()
        if campaign_mode
        else development_protocol()
    )
    common_projection_digest = (
        campaign_runtime_profile()["profile_digest"]
        if campaign_mode
        else common_release_projection_digest()
    )
    active_training_profile_digest = (
        sha256_digest(campaign_training_profile())
        if campaign_mode
        else pilot_training_profile_digest()
    )
    protocol_digest = sha256_digest(
        {
            "base_protocol_digest": common_protocol.digest,
            "protocol_id": "M6R_FIXED_TRAINING_CANARY_PROTOCOL_V1",
        }
    )
    seed_policy_digest = sha256_digest(
        {
            "ordinary_execution_seed": contract.ordinary_execution_seed,
            "search_seed": search_seed,
        }
    )
    training_config_budget_digest = sha256_digest(
        {
            "budget": budget.to_dict(),
            "training_profile_digest": active_training_profile_digest,
        }
    )
    preflight_run_root = arm_root / "pilot_runs" / "M6R_PREFLIGHT_RUN"
    preflight = training_runtime_compatibility_preflight(
        data_path=data_path,
        fixture=TrainingRuntimeCompatibilityFixtureV1(
            {
                "accepted_evidence_eligibility": (
                    "NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE"
                ),
                "arm_common_projection_digest": common_projection_digest,
                "budget_digest": sha256_digest(budget.to_dict()),
                "candidate_id": "M6R_PREFLIGHT_FIXED_CANDIDATE",
                "checkpoint_root": str(preflight_run_root / "checkpoints"),
                "component_runner_abis": training_runtime_component_abis(
                    runner_abi=runner_abi
                ),
                "evaluation_purpose": purpose,
                "execution_purpose": purpose,
                "experiment_id": contract.experiment_id,
                "frontier_eligibility": "NOT_ELIGIBLE_FOR_MAIN_FRONTIER",
                "gpu_cost_ceiling_microunits": budget.gpu_cost_microunits,
                "gpu_device_time_ceiling_ms": budget.gpu_device_time_ms,
                "implementation_digest": sha256_digest(
                    {"preflight": "fixed-candidate-implementation"}
                ),
                "instance_private_root": str(arm_root),
                "lineage_digest": sha256_digest(
                    {
                        "experiment_contract_digest": contract.identity_digest,
                        "partition": lineage_partition,
                    }
                ),
                "opaque_arm_instance_id": "M6R_PREFLIGHT_ARM_A",
                "partition_purpose": (
                    lineage_partition
                ),
                "protocol_digest": protocol_digest,
                "protocol_profile_ref": (
                    "M6R_FIXED_TRAINING_CANARY_PROTOCOL_V1"
                ),
                "result_root": str(preflight_run_root),
                "round_id": "M6R_PREFLIGHT_ROUND",
                "run_id": "M6R_PREFLIGHT_RUN",
                "search_memory_eligibility": (
                    "NOT_ELIGIBLE_FOR_MAIN_SEARCH_MEMORY"
                ),
                "seed_policy_digest": seed_policy_digest,
                "training_config_budget_digest": (
                    training_config_budget_digest
                ),
            }
        ),
        python_executable=python_executable,
        recbole_root=recbole_root,
        runner_abi=runner_abi,
    )
    _write_json(output_root / "TRAINING_RUNTIME_PREFLIGHT.json", preflight.to_dict())
    if preflight.status != TrainingCompatibilityStatusV1.COMPATIBLE.value:
        raise RuntimeError(
            f"training runtime preflight failed: {tuple(preflight.failure_codes)}"
        )

    store = TrainingSingleWriterExperimentStoreV1(
        output_root / "neutral" / "experiment.sqlite3",
        output_root / "neutral" / "artifacts",
    )
    try:
        arm_ids = store.initialize_experiment(contract)
        genesis = sha256_digest(
            {
                "experiment_contract_digest": contract.identity_digest,
                "state": "GENESIS",
            }
        )
        opened = store.open_round(
            OpenRoundCommand(
                experiment_id=contract.experiment_id,
                arm_instance_id=arm_ids[ArmCode.A],
                arm_code=ArmCode.A,
                search_seed=search_seed,
                round_index=1,
                budget_snapshot=budget,
                controller_state_before_digest=genesis,
                idempotency_key="m6r-fixed:open",
            )
        )
        program = (
            copy.deepcopy(
                dict(
                    executable_mechanism(
                        str(campaign_mechanism_id)
                    ).mechanism_program
                )
            )
            if campaign_mode
            else fixed_program(anchor_name)
        )
        compiled = compile_program(program)
        common_guard = CommonExecutionGuardV1()
        plan, eligible = common_guard.plan_check(
            program=program,
            caller_compile_report=compiled,
            protocol=common_protocol,
            budget=budget,
        )
        if eligible is None:
            raise RuntimeError(f"fixed candidate failed common plan: {plan.reason_codes}")
        report = DeterministicMaterializerV1().materialize(
            eligible,
            program=program,
            arm_runtime_root=arm_root,
        )
        trust = classify_execution_trust(report, arm_runtime_root=arm_root)
        base_binding = build_binding_v2(
            eligible=eligible,
            report=report,
            trust=trust,
            opaque_arm_instance_id=arm_ids[ArmCode.A],
            arm_private_root=arm_root,
            round_id=str(opened["round_id"]),
            search_seed=2026,
        )
        materialization_artifacts = register_materialization_artifacts(
            store,
            binding=base_binding,
            report=report,
        )
        gate = development_execution_gate(
            binding=base_binding,
            eligible=eligible,
            report=report,
            trust=trust,
            task_authorization_ref=(
                "RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md#M1"
            ),
        )
        pre, base_permit = common_guard.pre_execute(
            eligible=eligible,
            report=report,
            trust=trust,
            binding=base_binding,
            gate=gate,
        )
        if base_permit is None:
            raise RuntimeError(
                f"fixed candidate failed common PRE: {tuple(pre.reason_codes)}"
            )
        run_root = arm_root / "pilot_runs" / str(base_binding.run_id)
        checkpoint_root = run_root / "checkpoints"
        filesystem_capability = build_training_filesystem_capability(
            instance_private_root=arm_root,
            result_root=run_root,
            checkpoint_root=checkpoint_root,
            project_root=ROOT,
            recbole_root=recbole_root,
            dataset_root=data_path / "ml-1m",
        )
        runtime_binding = build_training_runtime_binding(
            accepted_evidence_eligibility=(
                "NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE"
            ),
            arm_common_projection_digest=common_projection_digest,
            budget_digest=str(base_binding.budget_digest),
            candidate_id=str(base_binding.candidate_id),
            checkpoint_root=str(checkpoint_root),
            evaluation_purpose=purpose,
            execution_purpose=purpose,
            experiment_id=contract.experiment_id,
            frontier_eligibility="NOT_ELIGIBLE_FOR_MAIN_FRONTIER",
            gpu_cost_ceiling_microunits=budget.gpu_cost_microunits,
            gpu_device_time_ceiling_ms=budget.gpu_device_time_ms,
            implementation_digest=str(base_binding.implementation_digest),
            instance_private_root=str(arm_root),
            lineage_digest=sha256_digest(
                {
                    "experiment_contract_digest": contract.identity_digest,
                    "partition": lineage_partition,
                }
            ),
            opaque_arm_instance_id=str(base_binding.opaque_arm_instance_id),
            partition_purpose=lineage_partition,
            protocol_digest=protocol_digest,
            protocol_profile_ref="M6R_FIXED_TRAINING_CANARY_PROTOCOL_V1",
            result_root=str(run_root),
            round_id=str(base_binding.round_id),
            run_id=str(base_binding.run_id),
            search_memory_eligibility=(
                "NOT_ELIGIBLE_FOR_MAIN_SEARCH_MEMORY"
            ),
            seed_policy_digest=seed_policy_digest,
            training_config_budget_digest=training_config_budget_digest,
            filesystem_capability_digest=(
                filesystem_capability.capability_digest
            ),
            runner_abi=runner_abi,
        )
        binding = build_training_binding_v3(
            base_binding=base_binding,
            runtime_binding=runtime_binding,
        )
        training_guard = CommonTrainingExecutionGuardV1()
        training_plan = training_guard.plan_check(
            base_plan=plan,
            runtime_binding=runtime_binding,
        )
        permit = training_guard.pre_execute(
            base_permit=base_permit,
            base_binding=base_binding,
            binding=binding,
            runtime_binding=runtime_binding,
            training_plan=training_plan,
        )
        store.claim_training_execution(
            ClaimTrainingExecutionCommandV1(
                round_id=str(binding.round_id),
                experiment_id=contract.experiment_id,
                candidate_id=str(binding.candidate_id),
                run_id=str(binding.run_id),
                permit_digest=permit.digest,
                binding_digest=binding.digest,
                budget_digest=str(binding.budget_digest),
                runtime_release_digest=str(binding.runtime_release_digest),
                runtime_binding_digest=runtime_binding.digest,
                runner_abi=str(binding.runner_abi),
                execution_purpose=str(binding.execution_purpose),
                metric_contract_digest=str(
                    runtime_binding.metric_contract_digest
                ),
                resource_contract_digest=str(
                    runtime_binding.resource_contract_digest
                ),
                idempotency_key="m6r-fixed:claim",
            )
        )
        raw_output, envelope = PilotTrainingLauncherV1(
            store,
            project_root=ROOT,
            recbole_root=recbole_root,
            data_path=data_path,
            experiment_writable_root=output_root,
            python_executable=python_executable,
        ).launch(
            permit=permit,
            binding=binding,
            runtime_binding=runtime_binding,
            materialization_artifacts=materialization_artifacts,
            force_failure=expected_runtime_failure,
        )
        closed = store.close_round(
            CloseRoundCommand(
                round_id=str(binding.round_id),
                terminal_class="COMPLETED",
                feedback_payload={
                    "authority": "NONE",
                    "evidence_class": "DEVELOPMENT_ONLY",
                    "formal_acceptance": False,
                    "raw_output_digest": raw_output.digest,
                    "raw_result_envelope_digest": envelope.digest,
                },
                controller_state_after_digest=sha256_digest(
                    {
                        "fixed_canary": "CLOSED",
                        "raw_result_envelope_digest": envelope.digest,
                    }
                ),
                resource_debits=(
                    ResourceDebitV1("COMMON_VALIDATION", 1),
                    ResourceDebitV1(
                        "GPU_DEVICE_TIME_MS",
                        int(raw_output.gpu_device_time_ms),
                    ),
                    ResourceDebitV1(
                        "GPU_COST_MICROUNITS",
                        int(raw_output.gpu_cost_microunits),
                    ),
                    ResourceDebitV1("WALL_TIME_MS", int(raw_output.wall_time_ms)),
                ),
                idempotency_key="m6r-fixed:close",
            )
        )
        if full_triplet_rehearsal:
            for arm in (ArmCode.B, ArmCode.C):
                companion = store.open_round(
                    OpenRoundCommand(
                        experiment_id=contract.experiment_id,
                        arm_instance_id=arm_ids[arm],
                        arm_code=arm,
                        search_seed=search_seed,
                        round_index=1,
                        budget_snapshot=budget,
                        controller_state_before_digest=genesis,
                        idempotency_key=f"m6e-rehearsal:{arm.value}:open",
                    )
                )
                store.close_round(
                    CloseRoundCommand(
                        round_id=str(companion["round_id"]),
                        terminal_class="COMPLETED",
                        feedback_payload={
                            "authority": "NONE",
                            "evidence_class": "DEVELOPMENT_ONLY",
                            "formal_acceptance": False,
                            "run_status": "NOT_EXECUTED_FIXED_REHEARSAL_COMPANION",
                        },
                        controller_state_after_digest=sha256_digest(
                            {
                                "arm": arm.value,
                                "fixed_rehearsal": "COMPANION_CLOSED",
                            }
                        ),
                        resource_debits=(),
                        idempotency_key=f"m6e-rehearsal:{arm.value}:close",
                    )
                )
        claim = store.get_execution_claim(str(binding.round_id))
        integrity = store.integrity_report()
        connection = sqlite3.connect(store.db_path)
        try:
            ledger = {
                str(dimension): int(quantity)
                for dimension, quantity in connection.execute(
                    """
                    SELECT dimension, SUM(quantity)
                    FROM resource_ledger
                    WHERE round_id = ?
                    GROUP BY dimension
                    """,
                    (binding.round_id,),
                )
            }
            tables = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                )
            ]
        finally:
            connection.close()
        authoritative_audit = None
        readiness_input_packet = None
        if full_triplet_rehearsal:
            rehearsal = RealPilotOrchestratorV1.__new__(RealPilotOrchestratorV1)
            rehearsal.store = store
            rehearsal.store_audit_port = experiment_store_audit_port(store)
            rehearsal.guard_ledger = SimpleNamespace(count=lambda: 0)
            rehearsal.initial_research_identity = sha256_digest(
                {"fixed_rehearsal": "NO_RESEARCH_BROKER"}
            )
            rehearsal.broker = SimpleNamespace(
                research_controllers={
                    arm: SimpleNamespace(policy=SimpleNamespace(version=1))
                    for arm in (ArmCode.B, ArmCode.C)
                }
            )
            authoritative_audit = RealPilotOrchestratorV1.pilot_audit(rehearsal)
            readiness_row = {
                "ndcg": raw_output.normalized_metrics.get("ndcg"),
                "opaque_instance_id": arm_ids[ArmCode.A],
                "round_index": 1,
                "run_status": (
                    "SUCCESS"
                    if raw_output.exit_status == "SUCCESS"
                    else "RUNTIME_FAILURE"
                ),
            }
            readiness_input_packet = {
                "rows": [readiness_row],
                "source": "M6E_FIXED_NO_LLM_REHEARSAL",
                "success_count": int(readiness_row["run_status"] == "SUCCESS"),
                "failure_count": int(
                    readiness_row["run_status"] == "RUNTIME_FAILURE"
                ),
            }
            readiness_input_packet["packet_digest"] = sha256_digest(
                readiness_input_packet
            )
        metric_closed = (
            raw_output.exit_status == "RUNTIME_FAILURE"
            and raw_output.termination_class == "CRASH_OR_RUNTIME_FAILURE"
            and not raw_output.normalized_metrics
        ) if expected_runtime_failure else (
            raw_output.exit_status == "SUCCESS"
            and "ndcg" in raw_output.normalized_metrics
        )
        gates = {
            "claim_closed": (
                claim["claim_state"] == "FINISHED"
                and claim["attempt_state"] == "START_CONFIRMED"
                and claim["execution_debited"] == 1
            ),
            "common_result_closed": bool(
                envelope.common_result_closure_digest
            ),
            "exact_eight_tables": len(tables) == 8,
            "identity_closed": (
                permit.runtime_release_digest
                == binding.runtime_release_digest
                == runtime_binding.release_digest
                == claim["runtime_release_digest"]
                == envelope.runtime_release_digest
            ),
            "ledger_closed": (
                ledger.get("ORDINARY_EXECUTION") == 1
                and ledger.get("GPU_DEVICE_TIME_MS")
                == raw_output.gpu_device_time_ms
                and ledger.get("GPU_COST_MICROUNITS")
                == raw_output.gpu_cost_microunits
                and ledger.get("WALL_TIME_MS") == raw_output.wall_time_ms
            ),
            "metric_closed": metric_closed,
            "round_closed": closed["terminal_class"] == "COMPLETED",
            "store_integrity": (
                integrity["integrity_check"] == "ok"
                and not integrity["foreign_key_violations"]
            ),
            "training_started": raw_output.training_backend_started is True,
        }
        if full_triplet_rehearsal:
            gates["authoritative_pilot_audit"] = bool(
                authoritative_audit["barriers_closed"]
                and authoritative_audit["round_count"] == 3
                and authoritative_audit["feedback_count"] == 3
                and authoritative_audit["execution_count"] == 1
                and authoritative_audit["state_store_integrity"]["sqlite_integrity"]
                == "ok"
            )
            gates["readiness_input_classification"] = (
                readiness_input_packet["success_count"]
                == int(not expected_runtime_failure)
                and readiness_input_packet["failure_count"]
                == int(expected_runtime_failure)
            )
        passed = all(gates.values())
        record = {
            "anchor_name": anchor_name,
            "campaign_mechanism_id": campaign_mechanism_id,
            "authoritative_pilot_audit": authoritative_audit,
            "authority": "NONE",
            "candidate_id": binding.candidate_id,
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "expected_runtime_failure": expected_runtime_failure,
            "gates": gates,
            "ledger": ledger,
            "normalized_metrics": raw_output.to_dict()["normalized_metrics"],
            "raw_output_digest": raw_output.digest,
            "raw_result_envelope_digest": envelope.digest,
            "readiness_input_packet": readiness_input_packet,
            "round_id": binding.round_id,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": binding.runtime_release_digest,
            "search_seed": search_seed,
            "verdict": "PASS" if passed else "FAIL",
        }
        _write_json(output_root / result_filename, record)
        return 0 if passed else 2
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/projects/RecClaw_m6_training_runtime_v2/bin/python"),
    )
    parser.add_argument(
        "--recbole-root",
        type=Path,
        default=Path("/root/projects/RecBole_m6_runtime"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/root/projects/RecBole/dataset"),
    )
    parser.add_argument("--anchor-name", default="BPR_MF")
    parser.add_argument("--campaign-mechanism-id")
    parser.add_argument(
        "--experiment-id",
        default="M6R-FIXED-TRAINING-CANARY-V1",
    )
    parser.add_argument(
        "--lineage-partition",
        default="FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN",
    )
    parser.add_argument(
        "--result-filename",
        default="M6R_FIXED_TRAINING_CANARY_RESULT.json",
    )
    parser.add_argument("--search-seed", type=int, default=CANARY_SEARCH_SEED)
    parser.add_argument("--expected-runtime-failure", action="store_true")
    parser.add_argument("--full-triplet-rehearsal", action="store_true")
    args = parser.parse_args()
    return execute(
        output_root=args.output_root.resolve(),
        python_executable=args.python.absolute(),
        recbole_root=args.recbole_root.resolve(),
        data_path=args.data_path.resolve(),
        anchor_name=args.anchor_name,
        experiment_id=args.experiment_id,
        lineage_partition=args.lineage_partition,
        result_filename=args.result_filename,
        search_seed=args.search_seed,
        expected_runtime_failure=args.expected_runtime_failure,
        full_triplet_rehearsal=args.full_triplet_rehearsal,
        campaign_mechanism_id=args.campaign_mechanism_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
