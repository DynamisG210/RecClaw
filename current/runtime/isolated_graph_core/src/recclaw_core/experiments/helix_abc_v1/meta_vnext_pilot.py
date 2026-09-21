"""Fresh five-round development Pilot for promoted Meta VNext V17."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.helix.contracts import GuardContext

from .canonical import canonical_value, sha256_digest
from .contracts import (
    ArmCode,
    ArmPolicyV1,
    MetaPolicyModeV1,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from .meta_vnext_campaign import (
    MetaV17CampaignRuntimeV1,
    meta_v17_research_control_policy,
)
from .precanary_orchestration import ArmRoundResultV1, PreCanaryInvariantError
from .real_canary import RealCanaryProposalBrokerV1
from .real_pilot import (
    RealPilotOrchestratorV1,
    pilot_guard_context,
)
from .training_runtime_release import TRAINING_RUNNER_ABI


META_V17_PILOT_SEARCH_SEED = 9212
META_V17_PILOT_ROUNDS_PER_ARM = 5
META_V17_PILOT_EXPERIMENT_ID = (
    "HELIX-ABC-DEVELOPMENT-META-V17-PILOT-9212-V10"
)
META_V17_PILOT_ASSIGNMENT_NONCE = "M6-META-V17-PILOT-9212-OPAQUE-V10"


def meta_v17_arm_policies() -> tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]:
    policies = []
    for policy in default_experiment_contract().arm_policies:
        if policy.arm in {ArmCode.B, ArmCode.C}:
            policy = replace(
                policy,
                meta_policy_mode=MetaPolicyModeV1.VERSIONED_POLICY_UPDATE,
                controller_policy_digest=(
                    meta_v17_research_control_policy().digest
                ),
            )
        policies.append(policy)
    result = tuple(policies)
    b_policy = next(item for item in result if item.arm is ArmCode.B)
    c_policy = next(item for item in result if item.arm is ArmCode.C)
    if b_policy.non_guard_projection() != c_policy.non_guard_projection():
        raise PreCanaryInvariantError("V17 B/C non-Guard policies must be identical")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class MetaV17PilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "MetaV17PilotStoreContractV1":
        base = default_experiment_contract()
        policies = meta_v17_arm_policies()
        payload = {
            "arm_policies": [item.to_dict() for item in policies],
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": META_V17_PILOT_EXPERIMENT_ID,
            "formal_acceptance": False,
            "main_eligibility": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": META_V17_PILOT_ROUNDS_PER_ARM,
            "search_seeds": [META_V17_PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=META_V17_PILOT_EXPERIMENT_ID,
            arm_policies=policies,
            search_seeds=(META_V17_PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=META_V17_PILOT_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


def meta_v17_pilot_guard_context() -> GuardContext:
    context = pilot_guard_context()
    return GuardContext(
        claim={
            **canonical_value(context.claim),
            "claim_id": "CLAIM-M6-META-V17-PILOT-9212-V10",
        },
        protocol=canonical_value(context.protocol),
        current_evidence={
            **canonical_value(context.current_evidence),
            "snapshot_id": "M6-META-V17-PILOT-9212-V10-EMPTY",
            "claim_id": "CLAIM-M6-META-V17-PILOT-9212-V10",
        },
    )


class MetaV17PilotOrchestratorV1(RealPilotOrchestratorV1):
    def __init__(
        self,
        root: Path,
        *,
        broker: RealCanaryProposalBrokerV1,
        meta_runtime: MetaV17CampaignRuntimeV1,
        project_root: Path,
        recbole_root: Path,
        data_path: Path,
        python_executable: Path,
        _contract: Any | None = None,
        _assignment_nonce: str | None = None,
        _guard_context: GuardContext | None = None,
        _training_runner_abi: str = TRAINING_RUNNER_ABI,
        _resource_ceilings: ResourceCeilingsV1 | None = None,
    ) -> None:
        if broker.campaign_meta_runtime is not meta_runtime:
            raise PreCanaryInvariantError(
                "Pilot Broker and orchestrator require the same V17 runtime"
            )
        self.meta_runtime = meta_runtime
        super().__init__(
            root,
            broker=broker,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=_contract or MetaV17PilotStoreContractV1.create(),
            _assignment_nonce=(
                _assignment_nonce or META_V17_PILOT_ASSIGNMENT_NONCE
            ),
            _guard_context=_guard_context or meta_v17_pilot_guard_context(),
            _training_runner_abi=_training_runner_abi,
            _resource_ceilings=_resource_ceilings,
        )
        meta_runtime.bind_instances(
            dict(self.assignment.arm_to_instance)
        )

    def run_pilot(
        self,
    ) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        return tuple(
            self.run_fake_triplet(
                search_seed=META_V17_PILOT_SEARCH_SEED,
                round_index=round_index,
                drafts=(),
            )
            for round_index in range(
                1,
                META_V17_PILOT_ROUNDS_PER_ARM + 1,
            )
        )


__all__ = [
    "META_V17_PILOT_ASSIGNMENT_NONCE",
    "META_V17_PILOT_EXPERIMENT_ID",
    "META_V17_PILOT_ROUNDS_PER_ARM",
    "META_V17_PILOT_SEARCH_SEED",
    "MetaV17PilotOrchestratorV1",
    "MetaV17PilotStoreContractV1",
    "meta_v17_arm_policies",
    "meta_v17_pilot_guard_context",
]
