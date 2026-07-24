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


PILOT_SEARCH_SEED = 9201
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
            "experiment_id": "HELIX-ABC-DEVELOPMENT-PILOT-9201-V1",
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
            assignment_nonce="M6-PILOT-9201-OPAQUE-V1",
            broker=broker,
            contract=PilotStoreContractV1.create(),
            resource_ceilings=pilot_budget(),
            guard_context=pilot_guard_context(),
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
        gate: Any,
        pre_execution: Any,
        materialization_artifacts: tuple[dict[str, Any], ...],
    ) -> tuple[Any, Any]:
        if gate.decision != "ALLOW" or pre_execution.decision != "PASS":
            raise PreCanaryInvariantError(
                "Pilot training requires common ALLOW/PASS decisions"
            )
        return self.training_launcher.launch(
            permit=permit,
            binding=binding,
            materialization_artifacts=materialization_artifacts,
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
    "pilot_guard_context",
    "pilot_protocol",
]
