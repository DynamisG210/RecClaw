"""M6 development Pilot with real broker calls and bounded RecBole training."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.helix.contracts import (
    CandidateEnvelope,
    GuardContext,
    RawResultEnvelope,
)

from .canonical import sha256_digest
from .contracts import ArmCode, ResourceCeilingsV1, default_experiment_contract
from .pilot_training import (
    PilotTrainingLauncherV1,
    pilot_training_profile,
    pilot_training_profile_digest,
)
from .precanary_orchestration import (
    ArmRoundResultV1,
    PreCanaryInvariantError,
    ThreeArmPreCanaryOrchestratorV1,
)
from .research_capability import (
    DISCOVERY_PRODUCERS,
    VersionedMetaPolicyUpdaterV1,
)
from .research_contracts import DevelopmentalMechanismBeliefV1
from .research_controller import ResearchLineControllerV1
from .real_canary import RealCanaryProposalBrokerV1
from .runtime_contracts import CommonDecision, GateStatus
from .runtime_release import common_release_projection_digest
from .training_execution_guard import CommonTrainingExecutionGuardV1
from .training_materialization import build_training_binding_v3
from .training_runtime_contracts import (
    TrainingCompatibilityStatusV1,
    TrainingExecutionPurposeV1,
    TrainingRuntimeBindingV1,
    TrainingRuntimeCompatibilityFixtureV1,
)
from .training_runtime_release import (
    build_training_runtime_binding,
    training_runtime_compatibility_preflight,
    training_runtime_component_abis,
)
from .training_state_store import (
    ClaimTrainingExecutionCommandV1,
    TrainingSingleWriterExperimentStoreV1,
)


PILOT_SEARCH_SEED = 9203
PILOT_ROUNDS_PER_ARM = 3


def pilot_budget() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=60_000,
        total_output_tokens=20_000,
        total_billed_token_debit=80_000,
        total_proposal_count=4,
        wall_time_ms=1_500_000,
        retry_debit=0,
        proposal_attempt_debit=4,
        ordinary_executions=1,
        common_validation_count=4,
        gpu_device_time_ms=900_000,
        gpu_cost_microunits=250_000,
    )


def pilot_protocol() -> dict[str, Any]:
    profile = pilot_training_profile()
    return {
        "protocol_id": "PROTO-ML1M-FULL-001",
        "profile_family": "OFFLINE_TOPN",
        "dataset": "ml-1m",
        "dataset_snapshot": "ml-1m-snapshot-001",
        "split": {
            "strategy": "random_user_holdout",
            "ratio": [0.8, 0.1, 0.1],
        },
        "training_sampling": {"mode": "uniform_negative"},
        "evaluation_candidate_universe": {"mode": "full_sort"},
        "candidate_policy": {"seen_items": "exclude"},
        "metric": {"name": "ndcg", "cutoff": 10},
        "training_procedure": {
            "optimizer": "adam",
            "max_epochs": int(profile["max_epochs"]),
        },
    }


def pilot_guard_context() -> GuardContext:
    return GuardContext(
        claim={
            "claim_id": "CLAIM-M6-PILOT-001",
            "protocol_id": "PROTO-ML1M-FULL-001",
            "claim_kind": "LOCAL_IMPROVEMENT",
            "target_model": "CandidateModel",
            "comparator": "LightGCN",
            "metric": "ndcg",
            "required_seed_count": 3,
            "scope": {"dataset": "ml-1m"},
        },
        protocol=pilot_protocol(),
        current_evidence={
            "snapshot_id": "M6-PILOT-EMPTY",
            "claim_id": "CLAIM-M6-PILOT-001",
            "protocol_id": "PROTO-ML1M-FULL-001",
            "observation_ids": [],
        },
    )


def pilot_common_gate_allows(gate_decision: str, pre_execution_decision: str) -> bool:
    return (
        gate_decision == GateStatus.ALLOW.value
        and pre_execution_decision == CommonDecision.PASS.value
    )


@dataclass(frozen=True, slots=True)
class PilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[Any, Any, Any]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "PilotStoreContractV1":
        base = default_experiment_contract()
        payload = {
            "arm_policies": [item.to_dict() for item in base.arm_policies],
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": "HELIX-ABC-DEVELOPMENT-PILOT-9203-V3",
            "formal_acceptance": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": PILOT_ROUNDS_PER_ARM,
            "search_seeds": [PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=str(payload["experiment_id"]),
            arm_policies=base.arm_policies,
            search_seeds=(PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=PILOT_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


class RealPilotOrchestratorV1(ThreeArmPreCanaryOrchestratorV1):
    def _create_store(
        self, db_path: Path, artifact_root: Path
    ) -> TrainingSingleWriterExperimentStoreV1:
        return TrainingSingleWriterExperimentStoreV1(db_path, artifact_root)

    def __init__(
        self,
        root: Path,
        *,
        broker: RealCanaryProposalBrokerV1,
        project_root: Path,
        recbole_root: Path,
        data_path: Path,
        python_executable: Path,
    ) -> None:
        super().__init__(
            root,
            assignment_nonce="M6-PILOT-9203-OPAQUE-V3",
            broker=broker,
            contract=PilotStoreContractV1.create(),
            resource_ceilings=pilot_budget(),
            guard_context=pilot_guard_context(),
        )
        purpose = TrainingExecutionPurposeV1.PILOT.value
        protocol_digest = sha256_digest(pilot_protocol())
        seed_policy_digest = sha256_digest(
            {
                "ordinary_execution_seed": self.contract.ordinary_execution_seed,
                "search_seeds": list(self.contract.search_seeds),
            }
        )
        training_config_budget_digest = sha256_digest(
            {
                "budget": self.resource_ceilings.to_dict(),
                "training_profile_digest": pilot_training_profile_digest(),
            }
        )
        preflight_arm_id = self.assignment.mapping[ArmCode.A]
        preflight_arm_root = self.layout.arm(preflight_arm_id).namespace("runtime")
        preflight_run_root = (
            preflight_arm_root / "pilot_runs" / "M6R_PREFLIGHT_RUN"
        )
        self.training_preflight = training_runtime_compatibility_preflight(
            data_path=data_path,
            fixture=TrainingRuntimeCompatibilityFixtureV1(
                {
                    "accepted_evidence_eligibility": (
                        "NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE"
                    ),
                    "arm_common_projection_digest": (
                        common_release_projection_digest()
                    ),
                    "budget_digest": sha256_digest(
                        self.resource_ceilings.to_dict()
                    ),
                    "candidate_id": "M6R_PREFLIGHT_PILOT_CANDIDATE",
                    "checkpoint_root": str(preflight_run_root / "checkpoints"),
                    "component_runner_abis": training_runtime_component_abis(),
                    "evaluation_purpose": purpose,
                    "execution_purpose": purpose,
                    "experiment_id": self.contract.experiment_id,
                    "frontier_eligibility": (
                        "NOT_ELIGIBLE_FOR_MAIN_FRONTIER"
                    ),
                    "gpu_cost_ceiling_microunits": (
                        self.resource_ceilings.gpu_cost_microunits
                    ),
                    "gpu_device_time_ceiling_ms": (
                        self.resource_ceilings.gpu_device_time_ms
                    ),
                    "implementation_digest": sha256_digest(
                        {"preflight": "pilot-implementation-schema"}
                    ),
                    "instance_private_root": str(preflight_arm_root),
                    "lineage_digest": sha256_digest(
                        {
                            "experiment_contract_digest": (
                                self.contract.identity_digest
                            ),
                            "partition": "PILOT_EXCLUDED_FROM_MAIN",
                        }
                    ),
                    "opaque_arm_instance_id": preflight_arm_id,
                    "partition_purpose": "PILOT_EXCLUDED_FROM_MAIN",
                    "protocol_digest": protocol_digest,
                    "protocol_profile_ref": "PROTO-ML1M-FULL-001",
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
        if (
            self.training_preflight.status
            != TrainingCompatibilityStatusV1.COMPATIBLE.value
        ):
            raise PreCanaryInvariantError(
                "training runtime preflight failed before broker use: "
                f"{tuple(self.training_preflight.failure_codes)}"
            )
        self.training_launcher = PilotTrainingLauncherV1(
            self.store,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
        )
        self.initial_research_identity = broker.bc_controller_identity_digest

    def _planned_guard_protocol(self) -> Mapping[str, Any]:
        return pilot_protocol()

    def _execute_selected(
        self,
        *,
        permit: Any,
        binding: Any,
        runtime_context: Any | None,
        gate: Any,
        pre_execution: Any,
        materialization_artifacts: tuple[dict[str, Any], ...],
    ) -> tuple[Any, Any]:
        if not pilot_common_gate_allows(gate.decision, pre_execution.decision):
            raise PreCanaryInvariantError(
                "Pilot training requires common ALLOW/PASS decisions"
            )
        if not isinstance(runtime_context, TrainingRuntimeBindingV1):
            raise PreCanaryInvariantError("Pilot training runtime binding is missing")
        return self.training_launcher.launch(
            permit=permit,
            binding=binding,
            runtime_binding=runtime_context,
            materialization_artifacts=materialization_artifacts,
        )

    def _prepare_runtime_execution(
        self,
        *,
        base_plan: Any,
        base_permit: Any,
        base_binding: Any,
        eligible: Any,
    ) -> tuple[Any, Any, TrainingRuntimeBindingV1]:
        del eligible
        purpose = TrainingExecutionPurposeV1.PILOT.value
        budget = self.resource_ceilings
        arm_root = Path(str(base_binding.arm_private_root))
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
            experiment_id=self.contract.experiment_id,
            frontier_eligibility="NOT_ELIGIBLE_FOR_MAIN_FRONTIER",
            gpu_cost_ceiling_microunits=budget.gpu_cost_microunits,
            gpu_device_time_ceiling_ms=budget.gpu_device_time_ms,
            implementation_digest=str(base_binding.implementation_digest),
            instance_private_root=str(arm_root),
            lineage_digest=sha256_digest(
                {
                    "experiment_contract_digest": self.contract.identity_digest,
                    "partition": "PILOT_EXCLUDED_FROM_MAIN",
                }
            ),
            opaque_arm_instance_id=str(base_binding.opaque_arm_instance_id),
            partition_purpose="PILOT_EXCLUDED_FROM_MAIN",
            protocol_digest=sha256_digest(pilot_protocol()),
            protocol_profile_ref="PROTO-ML1M-FULL-001",
            result_root=str(
                arm_root / "pilot_runs" / str(base_binding.run_id)
            ),
            round_id=str(base_binding.round_id),
            run_id=str(base_binding.run_id),
            search_memory_eligibility=(
                "NOT_ELIGIBLE_FOR_MAIN_SEARCH_MEMORY"
            ),
            seed_policy_digest=sha256_digest(
                {
                    "ordinary_execution_seed": (
                        self.contract.ordinary_execution_seed
                    ),
                    "search_seeds": list(self.contract.search_seeds),
                }
            ),
            training_config_budget_digest=sha256_digest(
                {
                    "budget": budget.to_dict(),
                    "training_profile_digest": pilot_training_profile_digest(),
                }
            ),
        )
        binding = build_training_binding_v3(
            base_binding=base_binding,
            runtime_binding=runtime_binding,
        )
        training_guard = CommonTrainingExecutionGuardV1()
        training_plan = training_guard.plan_check(
            base_plan=base_plan,
            runtime_binding=runtime_binding,
        )
        permit = training_guard.pre_execute(
            base_permit=base_permit,
            base_binding=base_binding,
            binding=binding,
            runtime_binding=runtime_binding,
            training_plan=training_plan,
        )
        return permit, binding, runtime_binding

    def _claim_runtime_execution(
        self,
        *,
        permit: Any,
        binding: Any,
        runtime_context: Any | None,
    ) -> None:
        if not isinstance(runtime_context, TrainingRuntimeBindingV1):
            raise PreCanaryInvariantError("Pilot claim is missing runtime binding")
        self.store.claim_training_execution(
            ClaimTrainingExecutionCommandV1(
                round_id=str(binding.round_id),
                experiment_id=self.contract.experiment_id,
                candidate_id=str(binding.candidate_id),
                run_id=str(binding.run_id),
                permit_digest=permit.digest,
                binding_digest=binding.digest,
                budget_digest=str(binding.budget_digest),
                runtime_release_digest=str(binding.runtime_release_digest),
                runtime_binding_digest=runtime_context.digest,
                runner_abi=str(binding.runner_abi),
                execution_purpose=str(binding.execution_purpose),
                metric_contract_digest=str(
                    runtime_context.metric_contract_digest
                ),
                resource_contract_digest=str(
                    runtime_context.resource_contract_digest
                ),
                idempotency_key=f"m6r:claim:{binding.round_id}",
            )
        )

    def _execution_resource_projection(
        self, raw_output: Any
    ) -> tuple[int, int, int]:
        return (
            int(raw_output.gpu_device_time_ms),
            int(raw_output.gpu_cost_microunits),
            int(raw_output.wall_time_ms),
        )

    def _build_helix_raw(
        self,
        *,
        selected: CandidateEnvelope,
        opaque_instance_id: str,
        common_result: Any,
    ) -> RawResultEnvelope:
        metrics = {
            str(key): float(value)
            for key, value in dict(common_result.normalized_metrics).items()
        }
        return RawResultEnvelope(
            candidate_id=selected.candidate_id,
            opaque_arm_instance_id=opaque_instance_id,
            raw_result_digest=str(common_result.raw_output_digest),
            common_result_closure_digest=str(
                common_result.common_result_closure_digest
            ),
            observed_protocol=pilot_protocol(),
            target_model="CandidateModel",
            comparator="LightGCN",
            seed_runs=(
                {
                    "seed_id": "2026",
                    "run_id": str(common_result.run_id),
                    "artifact_sha256": str(common_result.raw_output_digest),
                },
            ),
            observation_kind="METRIC_EVALUATION",
            run_status=str(common_result.exit_status),
            artifact_identity_status="EXACT",
            normalized_metrics=metrics,
        )

    def _research_belief(
        self,
        *,
        selected: CandidateEnvelope,
        feedback: Mapping[str, Any],
    ) -> DevelopmentalMechanismBeliefV1:
        return DevelopmentalMechanismBeliefV1(
            hypothesis_id=f"m6-{selected.candidate_id}",
            mechanism_axis="pilot_observation",
            competing_hypotheses=("runtime_failure", "weak_local_signal"),
            predicted_outcome_signature="bounded Pilot metric or runtime blocker",
            evidence_for=(str(feedback["raw_search_feedback_digest"]),),
            evidence_against=(),
            unresolved_confounds=("three_epoch_budget", "single_training_seed"),
            next_discriminative_test="next frozen Pilot round",
        )

    def _after_research_close(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        controller: ResearchLineControllerV1,
        feedback_projection: Mapping[str, Any],
    ) -> None:
        outcome = dict(feedback_projection["search_outcome"])
        success = outcome["run_status"] == "SUCCESS"
        controller.apply_meta_update(
            updater=VersionedMetaPolicyUpdaterV1(),
            completed_round_index=round_index,
            aggregate={
                "calibration_error": 0.05 if success else 0.2,
                "mechanism_axis_gaps": (
                    "objective",
                    "propagation",
                    "self_supervision",
                ),
                "producer_useful_rates": {
                    role: 0.8 if success else 0.2
                    for role in DISCOVERY_PRODUCERS
                },
            },
        )
        if arm not in {ArmCode.B, ArmCode.C}:
            raise PreCanaryInvariantError("Pilot Meta update escaped Research Arms")

    def _validate_triplet_results(
        self, results: tuple[ArmRoundResultV1, ...] | list[ArmRoundResultV1]
    ) -> None:
        expected_ids = set(self.assignment.mapping.values())
        if len(results) != 3 or {item.opaque_instance_id for item in results} != expected_ids:
            raise PreCanaryInvariantError("Pilot result set is not the opaque triplet")
        for item in results:
            if (
                item.terminal_class != "COMPLETED"
                or item.ordinary_execution_count != 1
                or not item.training_backend_started
                or item.input_tokens > self.resource_ceilings.total_input_tokens
                or item.output_tokens > self.resource_ceilings.total_output_tokens
                or item.billed_tokens
                > self.resource_ceilings.total_billed_token_debit
                or item.gpu_device_time_ms
                > self.resource_ceilings.gpu_device_time_ms
                or item.gpu_cost_microunits
                > self.resource_ceilings.gpu_cost_microunits
            ):
                raise PreCanaryInvariantError("Pilot triplet fails a frozen ceiling")

    def run_pilot(self) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        return tuple(
            self.run_fake_triplet(
                search_seed=PILOT_SEARCH_SEED,
                round_index=round_index,
                drafts=(),
            )
            for round_index in range(1, PILOT_ROUNDS_PER_ARM + 1)
        )

    def pilot_audit(self) -> dict[str, Any]:
        connection = sqlite3.connect(self.store.db_path)
        try:
            round_count = int(
                connection.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
            )
            feedback_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM round_events "
                    "WHERE event_type='ROUND_CLOSED'"
                ).fetchone()[0]
            )
            execution_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM execution_claims "
                    "WHERE execution_debited=1"
                ).fetchone()[0]
            )
            barriers = connection.execute(
                "SELECT round_index, closed_bitmap, next_index_authorized "
                "FROM triplet_barrier ORDER BY round_index"
            ).fetchall()
        finally:
            connection.close()
        return {
            "barriers_closed": all(
                int(bitmap) == 7 and int(authorized) == 1
                for _round, bitmap, authorized in barriers
            ),
            "execution_count": execution_count,
            "feedback_count": feedback_count,
            "guard_call_count": self.guard_ledger.count(),
            "initial_research_identity": self.initial_research_identity,
            "meta_versions": {
                arm.value: self.broker.research_controllers[arm].policy.version
                for arm in (ArmCode.B, ArmCode.C)
            },
            "round_count": round_count,
            "state_store_integrity": self.store.integrity_check(),
        }


__all__ = [
    "PILOT_ROUNDS_PER_ARM",
    "PILOT_SEARCH_SEED",
    "PilotStoreContractV1",
    "RealPilotOrchestratorV1",
    "pilot_budget",
    "pilot_common_gate_allows",
    "pilot_guard_context",
    "pilot_protocol",
]
