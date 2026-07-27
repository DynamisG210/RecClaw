"""Frozen-shape V13 Pilot runtime; construction does not start the Pilot."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from recclaw_core.helix.contracts import GuardContext

from .campaign_runtime import campaign_runtime_profile
from .canonical import canonical_value, sha256_digest
from .contracts import (
    ArmCode,
    ArmPolicyV1,
    MetaPolicyModeV1,
    default_experiment_contract,
)
from .meta_vnext_campaign import (
    MetaV18CampaignRuntimeV1,
    meta_v18_research_control_policy,
)
from .meta_vnext_pilot import MetaV17PilotOrchestratorV1
from .original_main import PinnedOriginalMainAdapterV1
from .precanary_orchestration import (
    ArmRoundResultV1,
    PreCanaryInvariantError,
)
from .real_canary import RealCanaryProposalBrokerV1
from .real_pilot import campaign_pilot_protocol, pilot_guard_context
from .training_runtime_release import CAMPAIGN_TRAINING_RUNNER_ABI


V13_PILOT_SEARCH_SEED = 9215
V13_PILOT_ROUNDS_PER_ARM = 5
V13_PILOT_EXPERIMENT_ID = "HELIX-ABC-DEVELOPMENT-CAMPAIGN-PILOT-9215-V13"
V13_PILOT_ASSIGNMENT_NONCE = "M6G-V13-PILOT-9215-OPAQUE-V1"
V13_EXECUTABLE_PROFILE_DIGEST = (
    "d483faa471c3f26d321daa89c6946ab67a11a95d459d2d5c095c9da419ee5da1"
)


def v13_arm_policies() -> tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]:
    policies = []
    meta_policy_digest = meta_v18_research_control_policy().digest
    for policy in default_experiment_contract().arm_policies:
        updates: dict[str, Any] = {
            "bl_icf_search_space_ref": "BL_ICF_EXECUTABLE_PROFILE_V2",
            "bl_icf_search_space_digest": V13_EXECUTABLE_PROFILE_DIGEST,
        }
        if policy.arm in {ArmCode.B, ArmCode.C}:
            updates.update(
                meta_policy_mode=MetaPolicyModeV1.VERSIONED_POLICY_UPDATE,
                controller_policy_digest=meta_policy_digest,
            )
        policies.append(replace(policy, **updates))
    result = tuple(policies)
    b_policy = next(item for item in result if item.arm is ArmCode.B)
    c_policy = next(item for item in result if item.arm is ArmCode.C)
    if b_policy.non_guard_projection() != c_policy.non_guard_projection():
        raise PreCanaryInvariantError(
            "V13 B/C non-Guard policies must be identical"
        )
    if {
        item.bl_icf_search_space_digest for item in result
    } != {V13_EXECUTABLE_PROFILE_DIGEST}:
        raise PreCanaryInvariantError(
            "V13 A/B/C must bind the same executable profile"
        )
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class V13PilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "V13PilotStoreContractV1":
        base = default_experiment_contract()
        policies = v13_arm_policies()
        profile = campaign_runtime_profile()
        if (
            profile["profile_id"] != "BL_ICF_EXECUTABLE_PROFILE_V2"
            or profile["executable_profile_digest"]
            != V13_EXECUTABLE_PROFILE_DIGEST
            or int(profile["executable_mechanism_count"]) != 66
        ):
            raise PreCanaryInvariantError(
                "V13 runtime profile is not the qualified 66-semantics release"
            )
        payload = {
            "arm_policies": [item.to_dict() for item in policies],
            "authority": "NONE",
            "campaign_runtime": "BL_ICF_EXECUTABLE_PROFILE_V2",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": V13_PILOT_EXPERIMENT_ID,
            "formal_acceptance": False,
            "main_eligibility": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": V13_PILOT_ROUNDS_PER_ARM,
            "search_seeds": [V13_PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=V13_PILOT_EXPERIMENT_ID,
            arm_policies=policies,
            search_seeds=(V13_PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=V13_PILOT_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


def v13_pilot_guard_context() -> GuardContext:
    base = pilot_guard_context()
    protocol = campaign_pilot_protocol()
    claim = canonical_value(base.claim)
    evidence = canonical_value(base.current_evidence)
    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6G-V13-PILOT-9215",
            "protocol_id": protocol["protocol_id"],
            "target_model": "CANDIDATE_SPECIFIC",
            "comparator": "FROZEN_ARM_MATCHED_PARENT",
        },
        protocol=protocol,
        current_evidence={
            **evidence,
            "snapshot_id": "M6G-V13-PILOT-9215-EMPTY",
            "claim_id": "CLAIM-M6G-V13-PILOT-9215",
            "protocol_id": protocol["protocol_id"],
        },
    )


class V13PilotOrchestratorV1(MetaV17PilotOrchestratorV1):
    def __init__(
        self,
        root: Path,
        *,
        broker: RealCanaryProposalBrokerV1,
        meta_runtime: MetaV18CampaignRuntimeV1,
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
                "V13 Pilot requires the direct pinned Main Original path"
            )
        if broker.campaign_meta_runtime is not meta_runtime:
            raise PreCanaryInvariantError(
                "V13 Broker and orchestrator require the same V18 runtime"
            )
        super().__init__(
            root,
            broker=broker,
            meta_runtime=meta_runtime,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=V13PilotStoreContractV1.create(),
            _assignment_nonce=V13_PILOT_ASSIGNMENT_NONCE,
            _guard_context=v13_pilot_guard_context(),
            _training_runner_abi=CAMPAIGN_TRAINING_RUNNER_ABI,
        )

    def run_round_triplet(
        self,
        *,
        round_index: int,
    ) -> tuple[ArmRoundResultV1, ...]:
        """Run one real Broker/training SearchRound for each opaque Arm."""

        return super().run_fake_triplet(
            search_seed=V13_PILOT_SEARCH_SEED,
            round_index=round_index,
            drafts=(),
        )

    def run_pilot(
        self,
    ) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        return tuple(
            self.run_round_triplet(round_index=round_index)
            for round_index in range(1, V13_PILOT_ROUNDS_PER_ARM + 1)
        )


__all__ = [
    "V13_EXECUTABLE_PROFILE_DIGEST",
    "V13_PILOT_ASSIGNMENT_NONCE",
    "V13_PILOT_EXPERIMENT_ID",
    "V13_PILOT_ROUNDS_PER_ARM",
    "V13_PILOT_SEARCH_SEED",
    "V13PilotOrchestratorV1",
    "V13PilotStoreContractV1",
    "v13_arm_policies",
    "v13_pilot_guard_context",
]
