"""Fresh V16 chain Pilot after resource-rejection round closure recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recclaw_core.helix.contracts import GuardContext

from .campaign_pilot_v15 import (
    V15_EXECUTABLE_PROFILE_DIGEST,
    v15_arm_policies,
)
from .campaign_runtime import campaign_runtime_profile
from .canonical import canonical_value, sha256_digest
from .contracts import ArmPolicyV1, default_experiment_contract
from .meta_vnext_campaign import MetaV19CampaignRuntimeV1
from .meta_vnext_pilot import MetaV17PilotOrchestratorV1
from .original_main import PinnedOriginalMainAdapterV1
from .precanary_orchestration import (
    ArmRoundResultV1,
    PreCanaryInvariantError,
)
from .real_canary import RealCanaryProposalBrokerV1
from .real_pilot import campaign_pilot_protocol, pilot_guard_context
from .training_runtime_release import CAMPAIGN_TRAINING_RUNNER_ABI


V16_PILOT_SEARCH_SEED = 9218
V16_PILOT_ROUNDS_PER_ARM = 5
V16_PILOT_EXPERIMENT_ID = (
    "HELIX-ABC-DEVELOPMENT-CAMPAIGN-PILOT-9218-V16"
)
V16_PILOT_ASSIGNMENT_NONCE = "M6G-V16-PILOT-9218-OPAQUE-V1"
V16_EXECUTABLE_PROFILE_DIGEST = V15_EXECUTABLE_PROFILE_DIGEST


def v16_arm_policies() -> tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]:
    return v15_arm_policies()


@dataclass(frozen=True, slots=True)
class V16PilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "V16PilotStoreContractV1":
        base = default_experiment_contract()
        policies = v16_arm_policies()
        profile = campaign_runtime_profile()
        if (
            profile["profile_id"] != "BL_ICF_EXECUTABLE_PROFILE_V2"
            or profile["executable_profile_digest"]
            != V16_EXECUTABLE_PROFILE_DIGEST
            or int(profile["executable_mechanism_count"]) != 66
        ):
            raise PreCanaryInvariantError(
                "V16 runtime profile is not the frozen 66-semantics release"
            )
        payload = {
            "arm_policies": [item.to_dict() for item in policies],
            "authority": "NONE",
            "campaign_runtime": "BL_ICF_EXECUTABLE_PROFILE_V2",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": V16_PILOT_EXPERIMENT_ID,
            "formal_acceptance": False,
            "main_eligibility": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": V16_PILOT_ROUNDS_PER_ARM,
            "search_seeds": [V16_PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=V16_PILOT_EXPERIMENT_ID,
            arm_policies=policies,
            search_seeds=(V16_PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=V16_PILOT_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


def v16_pilot_guard_context() -> GuardContext:
    base = pilot_guard_context()
    protocol = campaign_pilot_protocol()
    claim = canonical_value(base.claim)
    evidence = canonical_value(base.current_evidence)
    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6G-V16-PILOT-9218",
            "protocol_id": protocol["protocol_id"],
            "target_model": "CANDIDATE_SPECIFIC",
            "comparator": "FROZEN_ARM_MATCHED_PARENT",
        },
        protocol=protocol,
        current_evidence={
            **evidence,
            "snapshot_id": "M6G-V16-PILOT-9218-EMPTY",
            "claim_id": "CLAIM-M6G-V16-PILOT-9218",
            "protocol_id": protocol["protocol_id"],
        },
    )


class V16PilotOrchestratorV1(MetaV17PilotOrchestratorV1):
    def __init__(
        self,
        root: Path,
        *,
        broker: RealCanaryProposalBrokerV1,
        meta_runtime: MetaV19CampaignRuntimeV1,
        project_root: Path,
        recbole_root: Path,
        data_path: Path,
        python_executable: Path,
    ) -> None:
        if (
            not broker.v13_mode
            or not isinstance(
                broker.original_controller,
                PinnedOriginalMainAdapterV1,
            )
        ):
            raise PreCanaryInvariantError(
                "V16 Pilot requires the direct pinned Main Original path"
            )
        if broker.campaign_meta_runtime is not meta_runtime:
            raise PreCanaryInvariantError(
                "V16 Broker and orchestrator require the same V19 runtime"
            )
        super().__init__(
            root,
            broker=broker,
            meta_runtime=meta_runtime,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=V16PilotStoreContractV1.create(),
            _assignment_nonce=V16_PILOT_ASSIGNMENT_NONCE,
            _guard_context=v16_pilot_guard_context(),
            _training_runner_abi=CAMPAIGN_TRAINING_RUNNER_ABI,
        )

    def run_round_triplet(
        self,
        *,
        round_index: int,
    ) -> tuple[ArmRoundResultV1, ...]:
        return super().run_fake_triplet(
            search_seed=V16_PILOT_SEARCH_SEED,
            round_index=round_index,
            drafts=(),
        )

    def run_pilot(
        self,
    ) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        return tuple(
            self.run_round_triplet(round_index=round_index)
            for round_index in range(1, V16_PILOT_ROUNDS_PER_ARM + 1)
        )


__all__ = [
    "V16_EXECUTABLE_PROFILE_DIGEST",
    "V16_PILOT_ASSIGNMENT_NONCE",
    "V16_PILOT_EXPERIMENT_ID",
    "V16_PILOT_ROUNDS_PER_ARM",
    "V16_PILOT_SEARCH_SEED",
    "V16PilotOrchestratorV1",
    "V16PilotStoreContractV1",
    "v16_arm_policies",
    "v16_pilot_guard_context",
]
