#!/usr/bin/env python3
"""Run the no-LLM fixed-candidate M6R training-runtime canary."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
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
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    CloseRoundCommand,
    OpenRoundCommand,
    ResourceDebitV1,
)
from recclaw_core.experiments.helix_abc_v1.training_execution_guard import (  # noqa: E402
    CommonTrainingExecutionGuardV1,
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
    build_training_runtime_binding,
    training_runtime_compatibility_preflight,
    training_runtime_component_abis,
)
from recclaw_core.experiments.helix_abc_v1.training_state_store import (  # noqa: E402
    ClaimTrainingExecutionCommandV1,
    TrainingSingleWriterExperimentStoreV1,
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
    def create(cls) -> "FixedCanaryStoreContractV1":
        base = default_experiment_contract()
        payload = {
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": "M6R-FIXED-TRAINING-CANARY-V1",
            "formal_acceptance": False,
            "ordinary_execution_seed": 2026,
            "scheduled_slots_per_arm_seed": 1,
            "search_seeds": [CANARY_SEARCH_SEED],
        }
        return cls(
            experiment_id=str(payload["experiment_id"]),
            arm_policies=base.arm_policies,
            search_seeds=(CANARY_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=1,
            ordinary_execution_seed=2026,
            identity_digest=sha256_digest(payload),
        )


def canary_budget() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=0,
        total_output_tokens=0,
        total_billed_token_debit=0,
        total_proposal_count=1,
        wall_time_ms=1_500_000,
        retry_debit=0,
        proposal_attempt_debit=0,
        ordinary_executions=1,
        common_validation_count=1,
        gpu_device_time_ms=900_000,
        gpu_cost_microunits=250_000,
    )


def fixed_program() -> dict[str, Any]:
    document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for row in document["fixtures"]:
        if row["anchor_name"] == "BPR_MF":
            return copy.deepcopy(row["program"])
    raise RuntimeError("BPR_MF fixture is missing")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def execute(
    *,
    output_root: Path,
    python_executable: Path,
    recbole_root: Path,
    data_path: Path,
) -> int:
    if output_root.exists():
        raise RuntimeError("M6R fixed canary output root already exists")
    output_root.mkdir(parents=True)
    contract = FixedCanaryStoreContractV1.create()
    budget = canary_budget()
    runtime_root = output_root / "runtime"
    arm_root = runtime_root / "arm-a"
    purpose = TrainingExecutionPurposeV1.FIXED_CANARY.value
    protocol_digest = sha256_digest(
        {
            "base_protocol_digest": development_protocol().digest,
            "protocol_id": "M6R_FIXED_TRAINING_CANARY_PROTOCOL_V1",
        }
    )
    seed_policy_digest = sha256_digest(
        {
            "ordinary_execution_seed": contract.ordinary_execution_seed,
            "search_seed": CANARY_SEARCH_SEED,
        }
    )
    training_config_budget_digest = sha256_digest(
        {
            "budget": budget.to_dict(),
            "training_profile_digest": pilot_training_profile_digest(),
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
                "arm_common_projection_digest": common_release_projection_digest(),
                "budget_digest": sha256_digest(budget.to_dict()),
                "candidate_id": "M6R_PREFLIGHT_FIXED_CANDIDATE",
                "checkpoint_root": str(preflight_run_root / "checkpoints"),
                "component_runner_abis": training_runtime_component_abis(),
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
                        "partition": "FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN",
                    }
                ),
                "opaque_arm_instance_id": "M6R_PREFLIGHT_ARM_A",
                "partition_purpose": (
                    "FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN"
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
                search_seed=CANARY_SEARCH_SEED,
                round_index=1,
                budget_snapshot=budget,
                controller_state_before_digest=genesis,
                idempotency_key="m6r-fixed:open",
            )
        )
        program = fixed_program()
        compiled = compile_program(program)
        common_guard = CommonExecutionGuardV1()
        plan, eligible = common_guard.plan_check(
            program=program,
            caller_compile_report=compiled,
            protocol=development_protocol(),
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
        runtime_binding = build_training_runtime_binding(
            accepted_evidence_eligibility=(
                "NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE"
            ),
            arm_common_projection_digest=common_release_projection_digest(),
            budget_digest=str(base_binding.budget_digest),
            candidate_id=str(base_binding.candidate_id),
            checkpoint_root=str(
                arm_root
                / "pilot_runs"
                / str(base_binding.run_id)
                / "checkpoints"
            ),
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
                    "partition": "FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN",
                }
            ),
            opaque_arm_instance_id=str(base_binding.opaque_arm_instance_id),
            partition_purpose="FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN",
            protocol_digest=protocol_digest,
            protocol_profile_ref="M6R_FIXED_TRAINING_CANARY_PROTOCOL_V1",
            result_root=str(
                arm_root / "pilot_runs" / str(base_binding.run_id)
            ),
            round_id=str(base_binding.round_id),
            run_id=str(base_binding.run_id),
            search_memory_eligibility=(
                "NOT_ELIGIBLE_FOR_MAIN_SEARCH_MEMORY"
            ),
            seed_policy_digest=seed_policy_digest,
            training_config_budget_digest=training_config_budget_digest,
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
            python_executable=python_executable,
        ).launch(
            permit=permit,
            binding=binding,
            runtime_binding=runtime_binding,
            materialization_artifacts=materialization_artifacts,
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
            "metric_closed": (
                raw_output.exit_status == "SUCCESS"
                and "ndcg" in raw_output.normalized_metrics
            ),
            "round_closed": closed["terminal_class"] == "COMPLETED",
            "store_integrity": (
                integrity["integrity_check"] == "ok"
                and not integrity["foreign_key_violations"]
            ),
            "training_started": raw_output.training_backend_started is True,
        }
        passed = all(gates.values())
        record = {
            "authority": "NONE",
            "candidate_id": binding.candidate_id,
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "gates": gates,
            "ledger": ledger,
            "normalized_metrics": raw_output.to_dict()["normalized_metrics"],
            "raw_output_digest": raw_output.digest,
            "raw_result_envelope_digest": envelope.digest,
            "round_id": binding.round_id,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": binding.runtime_release_digest,
            "verdict": "PASS" if passed else "FAIL",
        }
        _write_json(output_root / "M6R_FIXED_TRAINING_CANARY_RESULT.json", record)
        return 0 if passed else 2
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/recbole/bin/python"),
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
    args = parser.parse_args()
    return execute(
        output_root=args.output_root.resolve(),
        python_executable=args.python.resolve(),
        recbole_root=args.recbole_root.resolve(),
        data_path=args.data_path.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
