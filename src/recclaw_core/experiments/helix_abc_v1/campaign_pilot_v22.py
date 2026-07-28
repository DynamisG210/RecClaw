"""Fresh post-M6I V22 Pilot on the qualified gpu35 backend closure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from recclaw_core.helix.contracts import GuardContext

from .campaign_pilot_v16 import (
    V16_EXECUTABLE_PROFILE_DIGEST,
    v16_arm_policies,
)
from .campaign_runtime import campaign_runtime_profile
from .canonical import canonical_value, sha256_digest
from .contracts import (
    ArmPolicyV1,
    ResourceCeilingsV1,
    default_experiment_contract,
)
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


V22_PILOT_SEARCH_SEED = 9224
V22_PILOT_ROUNDS_PER_ARM = 5
V22_PILOT_EXPERIMENT_ID = (
    "HELIX-ABC-DEVELOPMENT-CAMPAIGN-PILOT-9224-V22"
)
V22_PILOT_ASSIGNMENT_NONCE = "M6I-V22-PILOT-9224-OPAQUE-V1"
V22_EXECUTABLE_PROFILE_DIGEST = V16_EXECUTABLE_PROFILE_DIGEST
V22_RESOURCE_ENVELOPE_RESOURCE = (
    "pilot_v22_gpu35_resource_envelope.json"
)


def v22_arm_policies() -> tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]:
    """Return the exact frozen V16 A/B/C treatment policies."""

    return v16_arm_policies()


def v22_resource_ceilings() -> ResourceCeilingsV1:
    payload = json.loads(
        resources.files(
            "recclaw_core.experiments.helix_abc_v1.resources"
        )
        .joinpath(V22_RESOURCE_ENVELOPE_RESOURCE)
        .read_bytes()
    )
    return ResourceCeilingsV1(**payload["resource_ceilings"])


@dataclass(frozen=True, slots=True)
class V22PilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "V22PilotStoreContractV1":
        base = default_experiment_contract()
        policies = v22_arm_policies()
        profile = campaign_runtime_profile()
        if (
            profile["profile_id"] != "BL_ICF_EXECUTABLE_PROFILE_V2"
            or profile["executable_profile_digest"]
            != V22_EXECUTABLE_PROFILE_DIGEST
            or int(profile["executable_mechanism_count"]) != 66
        ):
            raise PreCanaryInvariantError(
                "V22 runtime profile is not the frozen 66-semantics release"
            )
        payload = {
            "arm_policies": [item.to_dict() for item in policies],
            "authority": "NONE",
            "campaign_runtime": "BL_ICF_EXECUTABLE_PROFILE_V2",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": V22_PILOT_EXPERIMENT_ID,
            "formal_acceptance": False,
            "main_eligibility": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": V22_PILOT_ROUNDS_PER_ARM,
            "search_seeds": [V22_PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=V22_PILOT_EXPERIMENT_ID,
            arm_policies=policies,
            search_seeds=(V22_PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=V22_PILOT_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


def v22_pilot_guard_context() -> GuardContext:
    base = pilot_guard_context()
    protocol = campaign_pilot_protocol()
    claim = canonical_value(base.claim)
    evidence = canonical_value(base.current_evidence)
    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6I-V22-PILOT-9224",
            "protocol_id": protocol["protocol_id"],
            "target_model": "CANDIDATE_SPECIFIC",
            "comparator": "FROZEN_ARM_MATCHED_PARENT",
        },
        protocol=protocol,
        current_evidence={
            **evidence,
            "snapshot_id": "M6I-V22-PILOT-9224-EMPTY",
            "claim_id": "CLAIM-M6I-V22-PILOT-9224",
            "protocol_id": protocol["protocol_id"],
        },
    )


class V22PilotOrchestratorV1(MetaV17PilotOrchestratorV1):
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
                "V22 Pilot requires the direct pinned Main Original path"
            )
        if broker.campaign_meta_runtime is not meta_runtime:
            raise PreCanaryInvariantError(
                "V22 Broker and orchestrator require the same V19 runtime"
            )
        expected_ceilings = v22_resource_ceilings()
        super().__init__(
            root,
            broker=broker,
            meta_runtime=meta_runtime,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=V22PilotStoreContractV1.create(),
            _assignment_nonce=V22_PILOT_ASSIGNMENT_NONCE,
            _guard_context=v22_pilot_guard_context(),
            _training_runner_abi=CAMPAIGN_TRAINING_RUNNER_ABI,
            _resource_ceilings=expected_ceilings,
        )
        if self.resource_ceilings != expected_ceilings:
            raise PreCanaryInvariantError(
                "V22 runtime resource envelope binding mismatch"
            )

    def run_round_triplet(
        self,
        *,
        round_index: int,
    ) -> tuple[ArmRoundResultV1, ...]:
        return super().run_fake_triplet(
            search_seed=V22_PILOT_SEARCH_SEED,
            round_index=round_index,
            drafts=(),
        )

    def run_pilot(
        self,
    ) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        return tuple(
            self.run_round_triplet(round_index=round_index)
            for round_index in range(1, V22_PILOT_ROUNDS_PER_ARM + 1)
        )


__all__ = [
    "V22_EXECUTABLE_PROFILE_DIGEST",
    "V22_PILOT_ASSIGNMENT_NONCE",
    "V22_PILOT_EXPERIMENT_ID",
    "V22_PILOT_ROUNDS_PER_ARM",
    "V22_PILOT_SEARCH_SEED",
    "V22PilotOrchestratorV1",
    "V22PilotStoreContractV1",
    "v22_arm_policies",
    "v22_pilot_guard_context",
    "v22_resource_ceilings",
]
