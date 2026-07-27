"""M6 development Pilot with real broker calls and bounded RecBole training."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.helix.contracts import (
    CandidateEnvelope,
    GuardContext,
    RawResultEnvelope,
)
from recclaw_core.helix.scientific_attribution import (
    FusedSearchFeedbackV2,
    NOT_AVAILABLE,
    SearchUtilityEventV2,
)
from .campaign_dataset import campaign_development_protocol
from .campaign_runtime import (
    campaign_runtime_profile,
    campaign_training_profile,
)

from .canonical import canonical_value, sha256_digest
from .audit_snapshot import (
    association_free_neutral_audit,
    create_immutable_audit_snapshot,
    open_immutable_snapshot,
    verify_immutable_snapshot,
)
from .contracts import ArmCode, ResourceCeilingsV1, default_experiment_contract
from .m6e_conformance import require_m6e_conformance_packet
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
from .research_contracts import (
    CandidateProposalV2,
    CandidateProposalV3,
    DevelopmentalMechanismBeliefV1,
)
from .research_controller import ResearchLineControllerV1
from .real_canary import RealCanaryProposalBrokerV1
from .runtime_contracts import CommonDecision, GateStatus
from .runtime_release import common_release_projection_digest
from .store_audit import experiment_store_audit_port
from .training_filesystem import build_training_filesystem_capability
from .training_execution_guard import CommonTrainingExecutionGuardV1
from .training_materialization import build_training_binding_v3
from .training_runtime_contracts import (
    TrainingCompatibilityStatusV1,
    TrainingExecutionPurposeV1,
    TrainingRuntimeBindingV2,
    TrainingRuntimeCompatibilityFixtureV1,
)
from .training_runtime_release import (
    CAMPAIGN_TRAINING_RUNNER_ABI,
    TRAINING_RUNNER_ABI,
    build_training_runtime_binding,
    training_runtime_compatibility_preflight,
    training_runtime_component_abis,
)
from .training_state_store import (
    ClaimTrainingExecutionCommandV1,
    TrainingSingleWriterExperimentStoreV1,
)


PILOT_SEARCH_SEED = 9203
FRESH_PILOT_SEARCH_SEED = 9204
FRESH_PILOT_V5_SEARCH_SEED = 9205
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


def campaign_pilot_protocol() -> dict[str, Any]:
    profile = campaign_training_profile()
    protocol = campaign_development_protocol()
    return {
        "protocol_id": protocol.protocol_id,
        "profile_family": protocol.profile_family,
        "dataset": "ml-1m",
        "dataset_snapshot": protocol.dataset_snapshot_digest,
        "split": {
            "strategy": "sha256_seeded_within_user",
            "ratio": [0.8, 0.1, 0.1],
            "online_partition": "DEVELOPMENT_VALIDATION",
            "heldout_access": "POST_SELECTION_ONLY",
        },
        "training_sampling": {"mode": "mechanism_program_defined"},
        "evaluation_candidate_universe": {"mode": "full_sort"},
        "candidate_policy": {"seen_items": "exclude"},
        "metric": {
            "name": "ndcg",
            "cutoff": 10,
            "source": "BEST_VALID_RESULT",
        },
        "training_procedure": {
            "optimizer": str(profile["optimizer"]),
            "max_epochs": int(profile["max_epochs"]),
            "early_stopping_patience": int(profile["stopping_step"]),
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


@dataclass(frozen=True, slots=True)
class PilotStoreContractV2:
    experiment_id: str
    arm_policies: tuple[Any, Any, Any]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "PilotStoreContractV2":
        base = default_experiment_contract()
        payload = {
            "arm_policies": [item.to_dict() for item in base.arm_policies],
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": "HELIX-ABC-DEVELOPMENT-PILOT-9204-V4",
            "formal_acceptance": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": PILOT_ROUNDS_PER_ARM,
            "search_seeds": [FRESH_PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=str(payload["experiment_id"]),
            arm_policies=base.arm_policies,
            search_seeds=(FRESH_PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=PILOT_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


@dataclass(frozen=True, slots=True)
class PilotStoreContractV3:
    experiment_id: str
    arm_policies: tuple[Any, Any, Any]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "PilotStoreContractV3":
        base = default_experiment_contract()
        payload = {
            "arm_policies": [item.to_dict() for item in base.arm_policies],
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": "HELIX-ABC-DEVELOPMENT-PILOT-9205-V5",
            "formal_acceptance": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": PILOT_ROUNDS_PER_ARM,
            "search_seeds": [FRESH_PILOT_V5_SEARCH_SEED],
        }
        return cls(
            experiment_id=str(payload["experiment_id"]),
            arm_policies=base.arm_policies,
            search_seeds=(FRESH_PILOT_V5_SEARCH_SEED,),
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
        _contract: (
            PilotStoreContractV1
            | PilotStoreContractV2
            | PilotStoreContractV3
            | None
        ) = None,
        _assignment_nonce: str | None = None,
        _guard_context: GuardContext | None = None,
        _training_runner_abi: str = TRAINING_RUNNER_ABI,
    ) -> None:
        contract = _contract or PilotStoreContractV1.create()
        self._training_project_root = project_root.resolve()
        self._training_recbole_root = recbole_root.resolve()
        self._training_data_path = data_path.resolve()
        self._training_runner_abi = _training_runner_abi
        self._campaign_training = (
            _training_runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI
        )
        super().__init__(
            root,
            assignment_nonce=(
                _assignment_nonce or "M6-PILOT-9203-OPAQUE-V3"
            ),
            broker=broker,
            contract=contract,
            resource_ceilings=pilot_budget(),
            guard_context=_guard_context or pilot_guard_context(),
        )
        self.store_audit_port = experiment_store_audit_port(self.store)
        self.store_audit_preflight = self.store_audit_port.audit_store()
        if not self.store_audit_preflight.passed:
            raise PreCanaryInvariantError(
                "Pilot store audit capability failed before broker use"
            )
        purpose = TrainingExecutionPurposeV1.PILOT.value
        protocol_digest = sha256_digest(self._active_pilot_protocol())
        seed_policy_digest = sha256_digest(
            {
                "ordinary_execution_seed": self.contract.ordinary_execution_seed,
                "search_seeds": list(self.contract.search_seeds),
            }
        )
        training_config_budget_digest = sha256_digest(
            {
                "budget": self.resource_ceilings.to_dict(),
                "training_profile_digest": (
                    sha256_digest(campaign_training_profile())
                    if self._campaign_training
                    else pilot_training_profile_digest()
                ),
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
                        campaign_runtime_profile()["profile_digest"]
                        if self._campaign_training
                        else common_release_projection_digest()
                    ),
                    "budget_digest": sha256_digest(
                        self.resource_ceilings.to_dict()
                    ),
                    "candidate_id": "M6R_PREFLIGHT_PILOT_CANDIDATE",
                    "checkpoint_root": str(preflight_run_root / "checkpoints"),
                    "component_runner_abis": training_runtime_component_abis(
                        runner_abi=self._training_runner_abi
                    ),
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
            runner_abi=self._training_runner_abi,
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
            experiment_writable_root=root,
            python_executable=python_executable,
        )
        self.initial_research_identity = broker.bc_controller_identity_digest

    def _planned_guard_protocol(self) -> Mapping[str, Any]:
        return self._active_pilot_protocol()

    def _common_execution_protocol(self) -> Any:
        return (
            campaign_development_protocol()
            if self._campaign_training
            else super()._common_execution_protocol()
        )

    def _active_pilot_protocol(self) -> Mapping[str, Any]:
        return (
            campaign_pilot_protocol()
            if self._campaign_training
            else pilot_protocol()
        )

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
        if not isinstance(runtime_context, TrainingRuntimeBindingV2):
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
    ) -> tuple[Any, Any, TrainingRuntimeBindingV2]:
        del eligible
        purpose = TrainingExecutionPurposeV1.PILOT.value
        budget = self.resource_ceilings
        arm_root = Path(str(base_binding.arm_private_root))
        run_root = arm_root / "pilot_runs" / str(base_binding.run_id)
        checkpoint_root = run_root / "checkpoints"
        filesystem_capability = build_training_filesystem_capability(
            instance_private_root=arm_root,
            result_root=run_root,
            checkpoint_root=checkpoint_root,
            project_root=self._training_project_root,
            recbole_root=self._training_recbole_root,
            dataset_root=(
                self._training_data_path
                / str(pilot_training_profile()["dataset"])
            ),
        )
        runtime_binding = build_training_runtime_binding(
            accepted_evidence_eligibility=(
                "NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE"
            ),
            arm_common_projection_digest=(
                campaign_runtime_profile()["profile_digest"]
                if self._campaign_training
                else common_release_projection_digest()
            ),
            budget_digest=str(base_binding.budget_digest),
            candidate_id=str(base_binding.candidate_id),
            checkpoint_root=str(checkpoint_root),
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
            protocol_digest=sha256_digest(self._active_pilot_protocol()),
            protocol_profile_ref=str(
                self._active_pilot_protocol()["protocol_id"]
            ),
            result_root=str(run_root),
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
                    "training_profile_digest": (
                        sha256_digest(campaign_training_profile())
                        if self._campaign_training
                        else pilot_training_profile_digest()
                    ),
                }
            ),
            filesystem_capability_digest=(
                filesystem_capability.capability_digest
            ),
            runner_abi=self._training_runner_abi,
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
        if not isinstance(runtime_context, TrainingRuntimeBindingV2):
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
        observation_seed: str,
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
            observed_protocol=self._active_pilot_protocol(),
            target_model=selected.target_model,
            comparator=selected.comparator,
            seed_runs=(
                {
                    "seed_id": observation_seed,
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
        event: SearchUtilityEventV2,
        proposal: CandidateProposalV2 | CandidateProposalV3,
    ) -> DevelopmentalMechanismBeliefV1:
        payload = deep_thaw(proposal.mechanism_program)["program_payload"]
        failure_modes = tuple(
            str(item) for item in payload.get("failure_modes", ())
        )
        expected_effects = canonical_value(payload.get("expected_effects", {}))
        comparator_delta = (
            None
            if event.comparator_delta == NOT_AVAILABLE
            else float(event.comparator_delta)
        )
        observation = "development_observation:" + event.digest
        evidence_for = (
            (observation,)
            if comparator_delta is not None and comparator_delta > 1e-4
            else ()
        )
        evidence_against = (
            (observation,)
            if (
                event.runnable_observation != "RUNNABLE"
                or (
                    comparator_delta is not None
                    and comparator_delta < -1e-4
                )
            )
            else ()
        )
        competing = (
            (proposal.competing_hypothesis,)
            if isinstance(proposal, CandidateProposalV3)
            else failure_modes
        )
        predicted = (
            proposal.predicted_outcome_signature
            if isinstance(proposal, CandidateProposalV3)
            else (
                f"{proposal.proposal_intent.value}:"
                f"{proposal.mechanism_axis}:"
                f"{expected_effects}"
            )
        )
        return DevelopmentalMechanismBeliefV1(
            hypothesis_id=str(selected.candidate_id),
            mechanism_axis=str(proposal.mechanism_axis),
            competing_hypotheses=(
                competing
                or (
                    "runtime_failure",
                    "weak_local_signal",
                    "confounded_anchor_effect",
                )
            ),
            predicted_outcome_signature=predicted,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            unresolved_confounds=(
                "single_training_seed",
                (
                    "matched_comparator_not_available"
                    if comparator_delta is None
                    else "matched_comparator_single_seed"
                ),
                f"run_status={event.common_outcome_class}",
            ),
            next_discriminative_test=(
                f"same-protocol {proposal.mechanism_axis} ablation "
                "against the frozen comparator"
            ),
        )

    def _after_research_close(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        controller: ResearchLineControllerV1,
        feedback: FusedSearchFeedbackV2,
        source_proposal_candidate_id: str,
    ) -> None:
        del source_proposal_candidate_id
        event = feedback.search_utility_event
        if event is None:
            raise PreCanaryInvariantError(
                "Meta-authorized feedback lacks SearchUtilityEventV2"
            )
        success = event.runnable_observation == "RUNNABLE"
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
                item.terminal_class
                not in {"COMPLETED", "NO_EXECUTION"}
                or item.ordinary_execution_count not in {0, 1}
                or (
                    item.ordinary_execution_count == 1
                    and not item.training_backend_started
                )
                or (
                    item.ordinary_execution_count == 0
                    and item.training_backend_started
                )
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
        by_arm = {
            next(
                arm
                for arm, opaque in self.assignment.mapping.items()
                if opaque == item.opaque_instance_id
            ): item
            for item in results
        }
        if any(
            by_arm[arm].ordinary_execution_count != 1
            for arm in (ArmCode.A, ArmCode.B)
        ):
            raise PreCanaryInvariantError(
                "A/B cannot lose the ordinary execution opportunity"
            )

    def immutable_audit_bundle(self, snapshot_root: Path) -> dict[str, Any]:
        snapshot_root = snapshot_root.resolve()
        snapshot_root.mkdir(parents=True, exist_ok=False)
        state_path = snapshot_root / "neutral_state.audit.sqlite3"
        broker_path = snapshot_root / "broker_state.audit.sqlite3"
        guard_path = snapshot_root / "guard_state.audit.sqlite3"
        with self.store._lock:
            state_manifest = create_immutable_audit_snapshot(
                writer_connection=self.store._connection,
                source_db_path=self.store.db_path,
                snapshot_path=state_path,
                source_schema_identity=self.store.migration_sha256,
                audit_purpose="M6F_NEUTRAL_STATE_AUDIT",
            )
        upstream = self.broker.upstream
        create_broker_snapshot = getattr(upstream, "create_audit_snapshot", None)
        if create_broker_snapshot is None:
            raise PreCanaryInvariantError(
                "Pilot Broker lacks immutable snapshot capability"
            )
        broker_manifest = create_broker_snapshot(
            broker_path, audit_purpose="M6F_BROKER_STATE_AUDIT"
        )
        guard_manifest = create_immutable_audit_snapshot(
            writer_connection=self.guard_ledger._connection,
            source_db_path=self.guard_ledger.db_path,
            snapshot_path=guard_path,
            source_schema_identity=sha256_digest(
                {"guard_calls_table": "EVIDENCE_GUARD_LEDGER_V1"}
            ),
            audit_purpose="M6F_GUARD_STATE_AUDIT",
        )
        manifests = {
            "broker": broker_manifest,
            "guard": guard_manifest,
            "state": state_manifest,
        }
        return {
            "manifests": {
                name: manifest.to_dict()
                for name, manifest in manifests.items()
            },
            "neutral_projection": association_free_neutral_audit(
                state_snapshot=state_path,
                broker_snapshot=broker_path,
                guard_snapshot=guard_path,
            ),
            "verification": {
                "broker": verify_immutable_snapshot(
                    broker_path, broker_manifest
                ),
                "guard": verify_immutable_snapshot(guard_path, guard_manifest),
                "state": verify_immutable_snapshot(state_path, state_manifest),
            },
        }

    def run_pilot(self) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        search_seed = int(self.contract.search_seeds[0])
        return tuple(
            self.run_fake_triplet(
                search_seed=search_seed,
                round_index=round_index,
                drafts=(),
            )
            for round_index in range(1, PILOT_ROUNDS_PER_ARM + 1)
        )

    def pilot_audit(
        self,
        state_snapshot: Path | None = None,
        guard_snapshot: Path | None = None,
    ) -> dict[str, Any]:
        legacy_temp: tempfile.TemporaryDirectory[str] | None = None
        legacy_store_integrity: dict[str, Any] | None = None
        if state_snapshot is None:
            legacy_temp = tempfile.TemporaryDirectory()
            state_snapshot = Path(legacy_temp.name) / "state.audit.sqlite3"
            with self.store._lock:
                create_immutable_audit_snapshot(
                    writer_connection=self.store._connection,
                    source_db_path=self.store.db_path,
                    snapshot_path=state_snapshot,
                    source_schema_identity=self.store.migration_sha256,
                    audit_purpose="M6E_PRESEAL_AUDIT_REHEARSAL",
                )
            legacy_store_integrity = (
                self.store_audit_port.audit_store().to_dict()
            )
        connection = open_immutable_snapshot(state_snapshot)
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
            integrity_check = str(
                connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
            foreign_key_violations = [
                list(row)
                for row in connection.execute("PRAGMA foreign_key_check").fetchall()
            ]
        finally:
            connection.close()
        if guard_snapshot is None:
            guard_call_count = int(self.guard_ledger.count())
        else:
            guard_connection = open_immutable_snapshot(guard_snapshot)
            try:
                guard_call_count = int(
                    guard_connection.execute(
                        "SELECT COUNT(*) FROM guard_calls"
                    ).fetchone()[0]
                )
            finally:
                guard_connection.close()
        result = {
            "barriers_closed": all(
                int(bitmap) == 7 and int(authorized) == 1
                for _round, bitmap, authorized in barriers
            ),
            "execution_count": execution_count,
            "feedback_count": feedback_count,
            "guard_call_count": guard_call_count,
            "initial_research_identity": self.initial_research_identity,
            "meta_versions": {
                arm.value: self.broker.research_controllers[arm].policy.version
                for arm in (ArmCode.B, ArmCode.C)
            },
            "round_count": round_count,
            "state_store_integrity": legacy_store_integrity
            or {
                    "foreign_key_violations": foreign_key_violations,
                    "integrity_check": integrity_check,
                },
        }
        if legacy_temp is not None:
            legacy_temp.cleanup()
        return result


def fresh_pilot_guard_context_v2() -> GuardContext:
    context = pilot_guard_context()
    claim = canonical_value(context.claim)
    protocol = canonical_value(context.protocol)
    current_evidence = canonical_value(context.current_evidence)
    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6-PILOT-9204-V4",
        },
        protocol=protocol,
        current_evidence={
            **current_evidence,
            "snapshot_id": "M6-PILOT-9204-V4-EMPTY",
            "claim_id": "CLAIM-M6-PILOT-9204-V4",
        },
    )


class FreshPilotOrchestratorV2(RealPilotOrchestratorV1):
    """Fresh post-M6R Pilot entrypoint; never reuses the sealed V1/V2/V3 state."""

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
        try:
            require_m6e_conformance_packet(project_root)
        except RuntimeError as error:
            raise PreCanaryInvariantError(
                "fresh Pilot is blocked before Broker use until M6E PASS"
            ) from error
        super().__init__(
            root,
            broker=broker,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=PilotStoreContractV2.create(),
            _assignment_nonce="M6-PILOT-9204-OPAQUE-V4",
            _guard_context=fresh_pilot_guard_context_v2(),
        )


def fresh_pilot_guard_context_v3() -> GuardContext:
    context = pilot_guard_context()
    claim = canonical_value(context.claim)
    protocol = canonical_value(context.protocol)
    current_evidence = canonical_value(context.current_evidence)
    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6-PILOT-9205-V5",
        },
        protocol=protocol,
        current_evidence={
            **current_evidence,
            "snapshot_id": "M6-PILOT-9205-V5-EMPTY",
            "claim_id": "CLAIM-M6-PILOT-9205-V5",
        },
    )


class FreshPilotOrchestratorV3(RealPilotOrchestratorV1):
    """Single authorized post-M6E Pilot entrypoint with a new V5 identity."""

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
        try:
            require_m6e_conformance_packet(project_root)
        except RuntimeError as error:
            raise PreCanaryInvariantError(
                "Pilot V5 is blocked before Broker use until M6E PASS"
            ) from error
        super().__init__(
            root,
            broker=broker,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=PilotStoreContractV3.create(),
            _assignment_nonce="M6-PILOT-9205-OPAQUE-V5",
            _guard_context=fresh_pilot_guard_context_v3(),
        )


__all__ = [
    "FRESH_PILOT_SEARCH_SEED",
    "FRESH_PILOT_V5_SEARCH_SEED",
    "FreshPilotOrchestratorV2",
    "FreshPilotOrchestratorV3",
    "PILOT_ROUNDS_PER_ARM",
    "PILOT_SEARCH_SEED",
    "PilotStoreContractV1",
    "PilotStoreContractV2",
    "PilotStoreContractV3",
    "RealPilotOrchestratorV1",
    "fresh_pilot_guard_context_v2",
    "fresh_pilot_guard_context_v3",
    "pilot_budget",
    "pilot_common_gate_allows",
    "pilot_guard_context",
    "pilot_protocol",
]
