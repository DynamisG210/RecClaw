from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ArmCode,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (
    common_release_projection_digest,
    development_protocol,
    runtime_release_digest,
    source_manifest_digest,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (
    ConservativeRecoveryCommand,
    IdempotencyConflict,
    InvariantViolation,
    OpenRoundCommand,
    RegisterArtifactCommand,
    SingleWriterExperimentStoreV1,
)
from recclaw_core.experiments.helix_abc_v1.runtime_contracts import (
    CandidateExecutionBindingV2,
    CommonDecision,
    CommonExecutionPermitV1,
    CommonPlanDecisionV1,
)
from recclaw_core.experiments.helix_abc_v1.training_execution_guard import (
    CommonTrainingExecutionGuardV1,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
    build_training_filesystem_capability,
)
from recclaw_core.experiments.helix_abc_v1.training_materialization import (
    build_training_binding_v3,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (
    CandidateExecutionBindingV3,
    ExecutionStartConfirmationV1,
    ExecutionStartReceiptV2,
    TrainingClosureDecisionV1,
    TrainingCompatibilityStatusV1,
    TrainingExecutionPurposeV1,
    TrainingRawRunOutputV2,
    TrainingResourceAccountingV1,
    TrainingRuntimeCompatibilityFixtureV1,
)
from recclaw_core.experiments.helix_abc_v1.pilot_training import (
    classify_training_termination,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
    PreCanaryInvariantError,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (
    FreshPilotOrchestratorV2,
    RealPilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    TRAINING_RUNNER_ABI,
    build_training_runtime_binding,
    resolve_runtime_release,
    training_runtime_compatibility_preflight,
    training_runtime_component_abis,
    training_runtime_release,
    validate_campaign_training_runtime_release,
    validate_training_runtime_release,
)
from recclaw_core.experiments.helix_abc_v1.training_state_store import (
    ClaimTrainingExecutionCommandV1,
    MarkTrainingExecutionStartedCommandV1,
    PrepareTrainingAttemptCommandV1,
    TrainingSingleWriterExperimentStoreV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path("/root/projects/RecClaw_m6_training_runtime_v2/bin/python")
RECBOLE = Path("/root/projects/RecBole_m6_runtime")
DATA = Path("/root/projects/RecBole/dataset")


def budget() -> ResourceCeilingsV1:
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


def runtime_binding_for(
    base_binding: CandidateExecutionBindingV2,
    *,
    purpose: str = TrainingExecutionPurposeV1.FIXED_CANARY.value,
    result_root: Path | None = None,
) -> Any:
    arm_root = Path(str(base_binding.arm_private_root))
    run_root = result_root or (
        arm_root / "pilot_runs" / str(base_binding.run_id)
    )
    capability = build_training_filesystem_capability(
        instance_private_root=arm_root,
        result_root=run_root,
        checkpoint_root=run_root / "checkpoints",
        project_root=PROJECT_ROOT,
        recbole_root=RECBOLE,
        dataset_root=DATA / "ml-1m",
    )
    return build_training_runtime_binding(
        accepted_evidence_eligibility="NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE",
        arm_common_projection_digest=common_release_projection_digest(),
        budget_digest=str(base_binding.budget_digest),
        candidate_id=str(base_binding.candidate_id),
        checkpoint_root=str(run_root / "checkpoints"),
        evaluation_purpose=purpose,
        execution_purpose=purpose,
        experiment_id="M6R-TEST-EXPERIMENT",
        frontier_eligibility="NOT_ELIGIBLE_FOR_MAIN_FRONTIER",
        gpu_cost_ceiling_microunits=250_000,
        gpu_device_time_ceiling_ms=900_000,
        implementation_digest=str(base_binding.implementation_digest),
        instance_private_root=str(arm_root),
        lineage_digest="8" * 64,
        opaque_arm_instance_id=str(base_binding.opaque_arm_instance_id),
        partition_purpose="FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN",
        protocol_digest="b" * 64,
        protocol_profile_ref="M6R_TEST_PROTOCOL_V1",
        result_root=str(run_root),
        round_id=str(base_binding.round_id),
        run_id=str(base_binding.run_id),
        search_memory_eligibility="NOT_ELIGIBLE_FOR_MAIN_SEARCH_MEMORY",
        seed_policy_digest="9" * 64,
        training_config_budget_digest="a" * 64,
        filesystem_capability_digest=capability.capability_digest,
    )


def _test_plan(runtime_binding: Any) -> CommonPlanDecisionV1:
    return CommonPlanDecisionV1(
        {
            "candidate_id": runtime_binding.candidate_id,
            "compile_report_digest": "1" * 64,
            "decision": CommonDecision.PASS.value,
            "mechanism_program_digest": "2" * 64,
            "mechanism_semantics_digest": "3" * 64,
            "profile_digest": "4" * 64,
            "protocol_digest": development_protocol().digest,
            "reason_codes": [],
            "release_projection_digest": common_release_projection_digest(),
            "subchecks": [],
        }
    )


def compatibility_fixture(
    root: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> TrainingRuntimeCompatibilityFixtureV1:
    run_root = root / "arm-a" / "pilot_runs" / "preflight-run"
    purpose = TrainingExecutionPurposeV1.FIXED_CANARY.value
    return TrainingRuntimeCompatibilityFixtureV1(
        {
            "accepted_evidence_eligibility": (
                "NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE"
            ),
            "arm_common_projection_digest": common_release_projection_digest(),
            "budget_digest": sha256_digest(budget().to_dict()),
            "candidate_id": "preflight-candidate",
            "checkpoint_root": str(run_root / "checkpoints"),
            "component_runner_abis": training_runtime_component_abis(overrides),
            "evaluation_purpose": purpose,
            "execution_purpose": purpose,
            "experiment_id": "M6R-PREFLIGHT-TEST",
            "frontier_eligibility": "NOT_ELIGIBLE_FOR_MAIN_FRONTIER",
            "gpu_cost_ceiling_microunits": budget().gpu_cost_microunits,
            "gpu_device_time_ceiling_ms": budget().gpu_device_time_ms,
            "implementation_digest": "1" * 64,
            "instance_private_root": str(root / "arm-a"),
            "lineage_digest": "2" * 64,
            "opaque_arm_instance_id": "opaque-a",
            "partition_purpose": (
                "FIXED_CANARY_EXCLUDED_FROM_PILOT_AND_MAIN"
            ),
            "protocol_digest": "3" * 64,
            "protocol_profile_ref": "M6R_TEST_PROTOCOL_V1",
            "result_root": str(run_root),
            "round_id": "preflight-round",
            "run_id": "preflight-run",
            "search_memory_eligibility": (
                "NOT_ELIGIBLE_FOR_MAIN_SEARCH_MEMORY"
            ),
            "seed_policy_digest": "4" * 64,
            "training_config_budget_digest": "5" * 64,
        }
    )


def base_binding_fixture(root: Path) -> CandidateExecutionBindingV2:
    return CandidateExecutionBindingV2(
        {
            "arm_private_root": str(root),
            "budget_digest": sha256_digest(budget().to_dict()),
            "candidate_id": "candidate-fixed",
            "implementation_digest": "2" * 64,
            "materialization_digest": "3" * 64,
            "mechanism_program_digest": "4" * 64,
            "mechanism_semantics_digest": "5" * 64,
            "opaque_arm_instance_id": "opaque-a",
            "profile_digest": "6" * 64,
            "round_id": "round-fixed",
            "run_id": "run-fixed",
            "runner_abi": "recclaw.fake-non-training-runner.v1",
            "runtime_release_digest": runtime_release_digest(),
            "search_seed": 2026,
            "trust_classification_digest": "7" * 64,
        }
    )


def base_permit_for(
    base_binding: CandidateExecutionBindingV2,
) -> CommonExecutionPermitV1:
    return CommonExecutionPermitV1(
        {
            "backend_digest": "8" * 64,
            "binding_digest": base_binding.digest,
            "budget_digest": base_binding.budget_digest,
            "candidate_id": base_binding.candidate_id,
            "gate_decision_digest": "9" * 64,
            "ordinary_launch_attempt_ordinal": 1,
            "pre_execution_decision_digest": "a" * 64,
            "round_id": base_binding.round_id,
            "run_id": base_binding.run_id,
            "runner_abi": base_binding.runner_abi,
        }
    )


def close_chain(
    root: Path,
    *,
    gpu_cost: int = 1,
    gpu_time: int = 10,
    metrics: dict[str, float] | None = None,
    termination_class: str = "SUCCESS",
    return_code: int = 0,
    exit_status: str = "SUCCESS",
) -> tuple[Any, ...]:
    base_binding = base_binding_fixture(root)
    runtime_binding = runtime_binding_for(base_binding)
    binding = build_training_binding_v3(
        base_binding=base_binding,
        runtime_binding=runtime_binding,
    )
    guard = CommonTrainingExecutionGuardV1()
    training_plan = guard.plan_check(
        base_plan=_test_plan(runtime_binding),
        runtime_binding=runtime_binding,
    )
    permit = guard.pre_execute(
        base_permit=base_permit_for(base_binding),
        base_binding=base_binding,
        binding=binding,
        runtime_binding=runtime_binding,
        training_plan=training_plan,
    )
    claim_id = "claim-fixed"
    confirmation = ExecutionStartConfirmationV1(
        {
            "binding_digest": binding.digest,
            "claim_id": claim_id,
            "execution_purpose": binding.execution_purpose,
            "ordinary_launch_attempt_ordinal": 1,
            "permit_digest": permit.digest,
            "pid": 123,
            "round_id": binding.round_id,
            "run_id": binding.run_id,
            "runner_abi": binding.runner_abi,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": binding.runtime_release_digest,
            "start_status": "START_CONFIRMED",
        }
    )
    receipt = ExecutionStartReceiptV2(
        {
            "binding_digest": binding.digest,
            "claim_id": claim_id,
            "execution_purpose": binding.execution_purpose,
            "ordinary_launch_attempt_ordinal": 1,
            "permit_digest": permit.digest,
            "round_id": binding.round_id,
            "run_id": binding.run_id,
            "runner_abi": binding.runner_abi,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": binding.runtime_release_digest,
            "start_confirmation_digest": confirmation.digest,
            "start_status": "STARTED",
        }
    )
    raw_output = TrainingRawRunOutputV2(
        {
            "binding_digest": binding.digest,
            "budget_digest": binding.budget_digest,
            "candidate_id": binding.candidate_id,
            "epochs_requested": 1,
            "environment_lock_digest": runtime_binding.environment_lock_digest,
            "execution_purpose": binding.execution_purpose,
            "filesystem_capability_digest": (
                runtime_binding.filesystem_capability_digest
            ),
            "filesystem_confinement_status": "PASS",
            "experiment_id": runtime_binding.experiment_id,
            "exit_status": exit_status,
            "gpu_cost_microunits": gpu_cost,
            "gpu_device_time_ms": gpu_time,
            "launcher_return_code": return_code,
            "lineage_digest": runtime_binding.lineage_digest,
            "metric_contract_digest": runtime_binding.metric_contract_digest,
            "model": "BPR",
            "normalized_metrics": (
                {"ndcg": 0.1} if metrics is None else metrics
            ),
            "permit_digest": permit.digest,
            "partition_purpose": runtime_binding.partition_purpose,
            "protocol_digest": runtime_binding.protocol_digest,
            "round_id": binding.round_id,
            "run_id": binding.run_id,
            "runner_abi": binding.runner_abi,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": binding.runtime_release_digest,
            "seed": 2026,
            "side_effect_audit_digest": "c" * 64,
            "training_backend_started": True,
            "termination_class": termination_class,
            "training_config_budget_digest": (
                runtime_binding.training_config_budget_digest
            ),
            "wall_time_ms": gpu_time,
            "worker_result_digest": "b" * 64,
        }
    )
    accounting = TrainingResourceAccountingV1(
        {
            "binding_digest": binding.digest,
            "claim_id": claim_id,
            "debits": [
                {"dimension": "GPU_COST_MICROUNITS", "quantity": gpu_cost},
                {"dimension": "GPU_DEVICE_TIME_MS", "quantity": gpu_time},
                {"dimension": "ORDINARY_EXECUTION", "quantity": 1},
                {"dimension": "WALL_TIME_MS", "quantity": gpu_time},
            ],
            "permit_digest": permit.digest,
            "raw_output_digest": raw_output.digest,
            "round_id": binding.round_id,
            "run_id": binding.run_id,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": binding.runtime_release_digest,
        }
    )
    claim = {
        "attempt_state": "START_CONFIRMED",
        "binding_digest": binding.digest,
        "claim_id": claim_id,
        "claim_state": "FINISHED",
        "execution_debited": 1,
        "execution_purpose": binding.execution_purpose,
        "permit_digest": permit.digest,
        "round_id": binding.round_id,
        "runner_abi": binding.runner_abi,
        "runtime_binding_digest": runtime_binding.digest,
        "runtime_release_digest": binding.runtime_release_digest,
    }
    artifacts = []
    for artifact_type, record in (
        ("EXECUTION_START_CONFIRMATION_V1", confirmation),
        ("EXECUTION_START_RECEIPT_V2", receipt),
        ("TRAINING_RAW_RUN_OUTPUT_V2", raw_output),
        ("TRAINING_RESOURCE_ACCOUNTING_V1", accounting),
    ):
        artifacts.append(
            {
                "artifact_type": artifact_type,
                "round_id": binding.round_id,
                "sha256": bytes_sha256(
                    canonical_json_bytes(record.to_dict())
                ),
            }
        )
    return (
        guard,
        permit,
        binding,
        runtime_binding,
        claim,
        confirmation,
        receipt,
        raw_output,
        accounting,
        artifacts,
    )


class BrokerTrap:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs: Any) -> None:
        self.calls += 1
        raise AssertionError("broker must not be called before preflight")


class M6RTrainingRuntimeTest(unittest.TestCase):
    def test_fake_release_identity_is_unchanged(self) -> None:
        self.assertEqual(
            runtime_release_digest(),
            "0a616ee205f494a161e7b10424181f37247347de8c5d714328dfa2f3f1ffb41d",
        )
        self.assertEqual(
            source_manifest_digest(),
            "038ebf3186ff0c67d7ebe7f5d799ea2d0f4aa6ee141f6a42d9767a05fb01a5fd",
        )
        self.assertEqual(
            common_release_projection_digest(),
            "97c4247af6f0a2fb2fbccd64663e2a6ed25cc1bc98aab2f5630e6d41d6f67347",
        )

    @unittest.skipUnless(
        PYTHON.is_file() and RECBOLE.is_dir() and DATA.is_dir(),
        "package training runtime is unavailable",
    )
    def test_historical_v2_is_not_reissued_after_v3_successor(self) -> None:
        historical_failures = validate_training_runtime_release(
            data_path=DATA,
            python_executable=PYTHON,
            recbole_root=RECBOLE,
        )
        self.assertTrue(
            any(
                item.startswith("TRAINING_SOURCE_MISMATCH:")
                for item in historical_failures
            )
        )
        self.assertEqual(
            validate_campaign_training_runtime_release(
                data_path=Path(
                    "/root/projects/RecClaw_campaign_dataset_v1/search"
                ),
                python_executable=PYTHON,
                recbole_root=RECBOLE,
            ),
            (),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            positive = training_runtime_compatibility_preflight(
                data_path=DATA,
                fixture=compatibility_fixture(root),
                python_executable=PYTHON,
                recbole_root=RECBOLE,
            )
            self.assertEqual(
                positive.status,
                TrainingCompatibilityStatusV1.INCOMPATIBLE.value,
            )
            self.assertTrue(
                any(
                    item.startswith("TRAINING_SOURCE_MISMATCH:")
                    for item in positive.failure_codes
                )
            )
            negative = training_runtime_compatibility_preflight(
                data_path=DATA,
                fixture=compatibility_fixture(
                    root,
                    overrides={
                        "launcher": (
                            "recclaw.package-owned-pilot-training-runner.v1"
                        ),
                        "state_store_claim": (
                            "recclaw.fake-non-training-runner.v1"
                        ),
                    },
                ),
                python_executable=PYTHON,
                recbole_root=RECBOLE,
            )
            self.assertEqual(
                negative.status,
                TrainingCompatibilityStatusV1.INCOMPATIBLE.value,
            )
            self.assertIn(
                "TRAINING_COMPONENT_ABI_MISMATCH:launcher",
                negative.failure_codes,
            )
            self.assertIn(
                "TRAINING_COMPONENT_ABI_MISMATCH:state_store_claim",
                negative.failure_codes,
            )

    def test_runtime_resolver_is_closed(self) -> None:
        fake = resolve_runtime_release("recclaw.fake-non-training-runner.v1")
        training = resolve_runtime_release(TRAINING_RUNNER_ABI)
        self.assertNotEqual(fake["release_digest"], training["release_digest"])
        self.assertEqual(training["release_digest"], training_runtime_release().digest)
        with self.assertRaises(ValueError):
            resolve_runtime_release("recclaw.unregistered-runner.v1")

    def test_real_pilot_rejects_before_any_broker_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            broker = BrokerTrap()
            with self.assertRaises(PreCanaryInvariantError):
                RealPilotOrchestratorV1(
                    Path(raw),
                    broker=broker,  # type: ignore[arg-type]
                    project_root=Path(__file__).resolve().parents[3],
                    recbole_root=RECBOLE,
                    data_path=DATA,
                    python_executable=Path(raw) / "missing-python",
                )
            self.assertEqual(broker.calls, 0)

    def test_fresh_pilot_rejects_before_any_broker_call(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            broker = BrokerTrap()
            with self.assertRaises(PreCanaryInvariantError):
                FreshPilotOrchestratorV2(
                    Path(raw),
                    broker=broker,  # type: ignore[arg-type]
                    project_root=Path(__file__).resolve().parents[3],
                    recbole_root=RECBOLE,
                    data_path=DATA,
                    python_executable=Path(raw) / "missing-python",
                )
            self.assertEqual(broker.calls, 0)

    def test_fake_training_cross_use_and_purpose_substitution_fail(self) -> None:
        base_binding = CandidateExecutionBindingV2(
            {
                "arm_private_root": "/tmp/m6r-binding",
                "budget_digest": "1" * 64,
                "candidate_id": "candidate-fixed",
                "implementation_digest": "2" * 64,
                "materialization_digest": "3" * 64,
                "mechanism_program_digest": "4" * 64,
                "mechanism_semantics_digest": "5" * 64,
                "opaque_arm_instance_id": "opaque-a",
                "profile_digest": "6" * 64,
                "round_id": "round-fixed",
                "run_id": "run-fixed",
                "runner_abi": "recclaw.fake-non-training-runner.v1",
                "runtime_release_digest": runtime_release_digest(),
                "search_seed": 2026,
                "trust_classification_digest": "7" * 64,
            }
        )
        base_permit = CommonExecutionPermitV1(
            {
                "backend_digest": "8" * 64,
                "binding_digest": base_binding.digest,
                "budget_digest": base_binding.budget_digest,
                "candidate_id": base_binding.candidate_id,
                "gate_decision_digest": "9" * 64,
                "ordinary_launch_attempt_ordinal": 1,
                "pre_execution_decision_digest": "a" * 64,
                "round_id": base_binding.round_id,
                "run_id": base_binding.run_id,
                "runner_abi": base_binding.runner_abi,
            }
        )
        runtime_binding = runtime_binding_for(base_binding)
        with self.assertRaises(ValueError):
            runtime_binding_for(
                base_binding,
                purpose=TrainingExecutionPurposeV1.MAIN.value,
            )
        binding = build_training_binding_v3(
            base_binding=base_binding,
            runtime_binding=runtime_binding,
        )
        training_guard = CommonTrainingExecutionGuardV1()
        training_plan = training_guard.plan_check(
            base_plan=_test_plan(runtime_binding),
            runtime_binding=runtime_binding,
        )
        permit = training_guard.pre_execute(
            base_permit=base_permit,
            base_binding=base_binding,
            binding=binding,
            runtime_binding=runtime_binding,
            training_plan=training_plan,
        )
        self.assertEqual(permit.runtime_release_digest, runtime_binding.release_digest)
        with self.assertRaises(ValueError):
            training_guard.pre_execute(
                base_permit=base_permit,
                base_binding=base_binding,
                binding=CandidateExecutionBindingV3(
                    {
                        **binding.to_dict(),
                        "runner_abi": "recclaw.fake-non-training-runner.v1",
                    }
                ),
                runtime_binding=runtime_binding,
                training_plan=training_plan,
            )
        with self.assertRaises(ValueError):
            training_guard.pre_execute(
                base_permit=base_permit,
                base_binding=base_binding,
                binding=CandidateExecutionBindingV3(
                    {
                        **binding.to_dict(),
                        "execution_purpose": (
                            TrainingExecutionPurposeV1.PILOT.value
                        ),
                    }
                ),
                runtime_binding=runtime_binding,
                training_plan=training_plan,
            )

    def test_training_claim_is_release_bound_and_replay_safe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = TrainingSingleWriterExperimentStoreV1(
                root / "state.sqlite3",
                root / "artifacts",
            )
            try:
                contract = default_experiment_contract()
                arm_ids = store.initialize_experiment(contract)
                opened = store.open_round(
                    OpenRoundCommand(
                        experiment_id=contract.experiment_id,
                        arm_instance_id=arm_ids[ArmCode.A],
                        arm_code=ArmCode.A,
                        search_seed=42,
                        round_index=1,
                        budget_snapshot=budget(),
                        controller_state_before_digest=sha256_digest(
                            {
                                "experiment_contract_digest": (
                                    contract.identity_digest
                                ),
                                "state": "GENESIS",
                            }
                        ),
                        idempotency_key="m6r-test:open",
                    )
                )
                release = training_runtime_release()
                arbitrary = ClaimTrainingExecutionCommandV1(
                    round_id=str(opened["round_id"]),
                    experiment_id=contract.experiment_id,
                    candidate_id="evil-candidate",
                    run_id="evil-run",
                    permit_digest="1" * 64,
                    binding_digest="2" * 64,
                    budget_digest=sha256_digest(budget().to_dict()),
                    runtime_release_digest="3" * 64,
                    runtime_binding_digest="4" * 64,
                    runner_abi="evil.runner.v1",
                    execution_purpose="EVIL_PURPOSE",
                    metric_contract_digest="5" * 64,
                    resource_contract_digest="6" * 64,
                    idempotency_key="m6r-test:arbitrary-claim",
                )
                with self.assertRaises(InvariantViolation):
                    store.claim_training_execution(arbitrary)
                command = ClaimTrainingExecutionCommandV1(
                    round_id=str(opened["round_id"]),
                    experiment_id=contract.experiment_id,
                    candidate_id="candidate-fixed",
                    run_id="run-fixed",
                    permit_digest="1" * 64,
                    binding_digest="2" * 64,
                    budget_digest=sha256_digest(budget().to_dict()),
                    runtime_release_digest=release.digest,
                    runtime_binding_digest="4" * 64,
                    runner_abi=TRAINING_RUNNER_ABI,
                    execution_purpose="DEVELOPMENT_FIXED_TRAINING_CANARY",
                    metric_contract_digest=sha256_digest(
                        release.metric_contract
                    ),
                    resource_contract_digest=sha256_digest(
                        release.resource_contract
                    ),
                    idempotency_key="m6r-test:claim",
                )
                first = store.claim_training_execution(command)
                replay = store.claim_training_execution(command)
                self.assertEqual(first, replay)
                with self.assertRaises(IdempotencyConflict):
                    store.claim_training_execution(
                        replace(
                            command,
                            runtime_binding_digest="7" * 64,
                        )
                    )
                prepared = store.prepare_training_attempt(
                    PrepareTrainingAttemptCommandV1(
                        round_id=command.round_id,
                        claim_id=str(first["claim_id"]),
                        permit_digest=command.permit_digest,
                        binding_digest=command.binding_digest,
                        runtime_release_digest=command.runtime_release_digest,
                        runtime_binding_digest=command.runtime_binding_digest,
                        runner_abi=command.runner_abi,
                        execution_purpose=command.execution_purpose,
                    )
                )
                self.assertEqual(prepared["attempt_state"], "PREPARED")
                with self.assertRaises(InvariantViolation):
                    store.prepare_training_attempt(
                        PrepareTrainingAttemptCommandV1(
                            round_id=command.round_id,
                            claim_id=str(first["claim_id"]),
                            permit_digest=command.permit_digest,
                            binding_digest=command.binding_digest,
                            runtime_release_digest="8" * 64,
                            runtime_binding_digest=command.runtime_binding_digest,
                            runner_abi=command.runner_abi,
                            execution_purpose=command.execution_purpose,
                        )
                    )

                forged_confirmation = ExecutionStartConfirmationV1(
                    {
                        "binding_digest": command.binding_digest,
                        "claim_id": first["claim_id"],
                        "execution_purpose": command.execution_purpose,
                        "ordinary_launch_attempt_ordinal": 1,
                        "permit_digest": command.permit_digest,
                        "pid": 123,
                        "round_id": command.round_id,
                        "run_id": "run-forged",
                        "runner_abi": command.runner_abi,
                        "runtime_binding_digest": command.runtime_binding_digest,
                        "runtime_release_digest": "7" * 64,
                        "start_status": "START_CONFIRMED",
                    }
                )
                forged_receipt = ExecutionStartReceiptV2(
                    {
                        "binding_digest": command.binding_digest,
                        "claim_id": first["claim_id"],
                        "execution_purpose": command.execution_purpose,
                        "ordinary_launch_attempt_ordinal": 1,
                        "permit_digest": command.permit_digest,
                        "round_id": command.round_id,
                        "run_id": "run-forged",
                        "runner_abi": command.runner_abi,
                        "runtime_binding_digest": command.runtime_binding_digest,
                        "runtime_release_digest": "7" * 64,
                        "start_confirmation_digest": forged_confirmation.digest,
                        "start_status": "STARTED",
                    }
                )
                confirmation_artifact = store.register_artifact(
                    RegisterArtifactCommand(
                        round_id=command.round_id,
                        artifact_type="EXECUTION_START_CONFIRMATION_V1",
                        relative_path="forged/confirmation.json",
                        producer="test",
                        idempotency_key="m6r-test:forged-confirmation",
                    ),
                    canonical_json_bytes(forged_confirmation.to_dict()),
                )
                self.assertEqual(
                    confirmation_artifact,
                    store.register_artifact(
                        RegisterArtifactCommand(
                            round_id=command.round_id,
                            artifact_type="EXECUTION_START_CONFIRMATION_V1",
                            relative_path="forged/confirmation.json",
                            producer="test",
                            idempotency_key="m6r-test:forged-confirmation",
                        ),
                        canonical_json_bytes(forged_confirmation.to_dict()),
                    ),
                )
                receipt_artifact = store.register_artifact(
                    RegisterArtifactCommand(
                        round_id=command.round_id,
                        artifact_type="EXECUTION_START_RECEIPT_V2",
                        relative_path="forged/receipt.json",
                        producer="test",
                        idempotency_key="m6r-test:forged-receipt",
                    ),
                    canonical_json_bytes(forged_receipt.to_dict()),
                )
                with self.assertRaises(InvariantViolation):
                    store.mark_training_execution_started(
                        MarkTrainingExecutionStartedCommandV1(
                            round_id=command.round_id,
                            claim_id=str(first["claim_id"]),
                            receipt_artifact_id=str(receipt_artifact["artifact_id"]),
                            confirmation_artifact_id=str(
                                confirmation_artifact["artifact_id"]
                            ),
                            idempotency_key="m6r-test:forged-start",
                        )
                    )
                confirmation = ExecutionStartConfirmationV1(
                    {
                        **forged_confirmation.to_dict(),
                        "runtime_release_digest": command.runtime_release_digest,
                    }
                )
                receipt = ExecutionStartReceiptV2(
                    {
                        **forged_receipt.to_dict(),
                        "runtime_release_digest": command.runtime_release_digest,
                        "start_confirmation_digest": confirmation.digest,
                    }
                )
                exact_confirmation = store.register_artifact(
                    RegisterArtifactCommand(
                        round_id=command.round_id,
                        artifact_type="EXECUTION_START_CONFIRMATION_V1",
                        relative_path="exact/confirmation.json",
                        producer="test",
                        idempotency_key="m6r-test:exact-confirmation",
                    ),
                    canonical_json_bytes(confirmation.to_dict()),
                )
                exact_receipt = store.register_artifact(
                    RegisterArtifactCommand(
                        round_id=command.round_id,
                        artifact_type="EXECUTION_START_RECEIPT_V2",
                        relative_path="exact/receipt.json",
                        producer="test",
                        idempotency_key="m6r-test:exact-receipt",
                    ),
                    canonical_json_bytes(receipt.to_dict()),
                )
                start_command = MarkTrainingExecutionStartedCommandV1(
                    round_id=command.round_id,
                    claim_id=str(first["claim_id"]),
                    receipt_artifact_id=str(exact_receipt["artifact_id"]),
                    confirmation_artifact_id=str(
                        exact_confirmation["artifact_id"]
                    ),
                    idempotency_key="m6r-test:exact-start",
                )
                started = store.mark_training_execution_started(start_command)
                replayed = store.mark_training_execution_started(start_command)
                self.assertEqual(started, replayed)
                self.assertEqual(started["attempt_state"], "START_CONFIRMED")
                self.assertEqual(started["execution_debited"], 1)
            finally:
                store.close()

    def test_training_close_result_success_crash_timeout_and_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)

            def close(chain: tuple[Any, ...]) -> tuple[Any, Any]:
                (
                    guard,
                    permit,
                    binding,
                    runtime_binding,
                    claim,
                    confirmation,
                    receipt,
                    raw_output,
                    accounting,
                    artifacts,
                ) = chain
                return guard.close_result(
                    permit=permit,
                    binding=binding,
                    runtime_binding=runtime_binding,
                    claim=claim,
                    confirmation=confirmation,
                    receipt=receipt,
                    raw_output=raw_output,
                    resource_accounting=accounting,
                    artifact_closure=artifacts,
                    seed=2026,
                )

            success, success_envelope = close(close_chain(root / "success"))
            self.assertEqual(
                success.decision,
                TrainingClosureDecisionV1.CLOSED.value,
            )
            self.assertIsNotNone(success_envelope)

            crash, crash_envelope = close(
                close_chain(
                    root / "crash",
                    metrics={},
                    termination_class="CRASH_OR_RUNTIME_FAILURE",
                    return_code=1,
                    exit_status="RUNTIME_FAILURE",
                )
            )
            self.assertEqual(
                crash.decision,
                TrainingClosureDecisionV1.CLOSED.value,
            )
            self.assertIsNotNone(crash_envelope)

            timeout, timeout_envelope = close(
                close_chain(
                    root / "timeout",
                    metrics={},
                    termination_class="TIMEOUT",
                    return_code=124,
                    exit_status="RUNTIME_FAILURE",
                )
            )
            self.assertEqual(
                timeout.decision,
                TrainingClosureDecisionV1.CLOSED.value,
            )
            self.assertIsNotNone(timeout_envelope)

            over_budget, envelope = close(
                close_chain(root / "budget", gpu_time=900_001)
            )
            self.assertEqual(
                over_budget.decision,
                TrainingClosureDecisionV1.REJECTED.value,
            )
            self.assertIsNotNone(envelope)
            self.assertEqual(
                envelope.exit_status,
                "COMMON_EXECUTION_FAILURE",
            )
            self.assertEqual(envelope.normalized_metrics, {})
            self.assertIn(
                "TRAINING_RESOURCE_CEILING_EXCEEDED",
                over_budget.reason_codes,
            )

            missing_metric, envelope = close(
                close_chain(root / "metric", metrics={})
            )
            self.assertEqual(
                missing_metric.decision,
                TrainingClosureDecisionV1.REJECTED.value,
            )
            self.assertIsNone(envelope)

            chain = list(close_chain(root / "fake-raw"))
            chain[7] = TrainingRawRunOutputV2(
                {**chain[7].to_dict(), "runner_abi": "recclaw.fake-non-training-runner.v1"}
            )
            fake_raw, envelope = close(tuple(chain))
            self.assertEqual(
                fake_raw.decision,
                TrainingClosureDecisionV1.REJECTED.value,
            )
            self.assertIsNone(envelope)

            chain = list(close_chain(root / "fake-claim"))
            chain[4] = {
                **chain[4],
                "runner_abi": "recclaw.fake-non-training-runner.v1",
                "runtime_release_digest": runtime_release_digest(),
            }
            fake_claim, envelope = close(tuple(chain))
            self.assertEqual(
                fake_claim.decision,
                TrainingClosureDecisionV1.REJECTED.value,
            )
            self.assertIsNone(envelope)

    def test_termination_classification_and_nonfinite_metric_fail_closed(self) -> None:
        self.assertEqual(
            classify_training_termination(
                return_code=0,
                timed_out=False,
                worker_status="SUCCESS",
            ),
            ("SUCCESS", "SUCCESS"),
        )
        self.assertEqual(
            classify_training_termination(
                return_code=124,
                timed_out=True,
                worker_status=None,
            ),
            ("RUNTIME_FAILURE", "TIMEOUT"),
        )
        self.assertEqual(
            classify_training_termination(
                return_code=2,
                timed_out=False,
                worker_status="RUNTIME_FAILURE",
            ),
            ("RUNTIME_FAILURE", "CRASH_OR_RUNTIME_FAILURE"),
        )
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                close_chain(Path(raw), metrics={"ndcg": float("nan")})

    def test_training_common_release_is_arm_equal_and_roots_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bindings = []
            for arm in ("a", "b", "c"):
                base = base_binding_fixture(root / f"arm-{arm}")
                base = CandidateExecutionBindingV2(
                    {
                        **base.to_dict(),
                        "arm_private_root": str(root / f"arm-{arm}"),
                        "opaque_arm_instance_id": f"opaque-{arm}",
                    }
                )
                bindings.append(runtime_binding_for(base))
            self.assertEqual(
                {item.release_digest for item in bindings},
                {training_runtime_release().digest},
            )
            self.assertEqual(
                {item.arm_common_projection_digest for item in bindings},
                {common_release_projection_digest()},
            )
            self.assertEqual(
                {item.frontier_eligibility for item in bindings},
                {"NOT_ELIGIBLE_FOR_MAIN_FRONTIER"},
            )
            self.assertEqual(len({item.result_root for item in bindings}), 3)
            with self.assertRaises(ValueError):
                runtime_binding_for(
                    base_binding_fixture(root / "arm-a"),
                    result_root=root / "arm-b" / "pilot_runs" / "escaped",
                )

    def test_training_prepared_claim_recovers_start_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = TrainingSingleWriterExperimentStoreV1(
                root / "state.sqlite3",
                root / "artifacts",
            )
            contract = default_experiment_contract()
            arm_ids = store.initialize_experiment(contract)
            opened = store.open_round(
                OpenRoundCommand(
                    experiment_id=contract.experiment_id,
                    arm_instance_id=arm_ids[ArmCode.A],
                    arm_code=ArmCode.A,
                    search_seed=42,
                    round_index=1,
                    budget_snapshot=budget(),
                    controller_state_before_digest=sha256_digest(
                        {
                            "experiment_contract_digest": contract.identity_digest,
                            "state": "GENESIS",
                        }
                    ),
                    idempotency_key="m6r-ambiguous:open",
                )
            )
            release = training_runtime_release()
            command = ClaimTrainingExecutionCommandV1(
                round_id=str(opened["round_id"]),
                experiment_id=contract.experiment_id,
                candidate_id="candidate-ambiguous",
                run_id="run-ambiguous",
                permit_digest="1" * 64,
                binding_digest="2" * 64,
                budget_digest=sha256_digest(budget().to_dict()),
                runtime_release_digest=release.digest,
                runtime_binding_digest="3" * 64,
                runner_abi=TRAINING_RUNNER_ABI,
                execution_purpose=TrainingExecutionPurposeV1.FIXED_CANARY.value,
                metric_contract_digest=sha256_digest(release.metric_contract),
                resource_contract_digest=sha256_digest(release.resource_contract),
                idempotency_key="m6r-ambiguous:claim",
            )
            claim = store.claim_training_execution(command)
            store.prepare_training_attempt(
                PrepareTrainingAttemptCommandV1(
                    round_id=command.round_id,
                    claim_id=str(claim["claim_id"]),
                    permit_digest=command.permit_digest,
                    binding_digest=command.binding_digest,
                    runtime_release_digest=command.runtime_release_digest,
                    runtime_binding_digest=command.runtime_binding_digest,
                    runner_abi=command.runner_abi,
                    execution_purpose=command.execution_purpose,
                )
            )
            store.close()
            recovered = TrainingSingleWriterExperimentStoreV1(
                root / "state.sqlite3",
                root / "artifacts",
            )
            try:
                report = recovered.conservative_recovery(
                    ConservativeRecoveryCommand(
                        experiment_id=contract.experiment_id,
                        search_seed=42,
                        current_round_index=1,
                        idempotency_key="m6r-ambiguous:recovery",
                    )
                )
                self.assertEqual(
                    report["recovered_rounds"][0]["terminal_class"],
                    "ABORTED_RECOVERY_START_AMBIGUOUS",
                )
                claim_after = recovered.get_execution_claim(command.round_id)
                self.assertEqual(claim_after["claim_state"], "START_AMBIGUOUS")
                self.assertEqual(claim_after["execution_debited"], 1)
            finally:
                recovered.close()

    def test_training_store_refuses_historical_v1_migration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = SingleWriterExperimentStoreV1(
                root / "state.sqlite3",
                root / "artifacts",
            )
            base.close()
            with self.assertRaises(InvariantViolation):
                TrainingSingleWriterExperimentStoreV1(
                    root / "state.sqlite3",
                    root / "training-artifacts",
                )


if __name__ == "__main__":
    unittest.main()
