"""Fresh Pilot entrypoint for the Main-equivalent Research Campaign runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recclaw_core.helix.contracts import GuardContext

from .canonical import canonical_value, sha256_digest
from .contracts import ArmCode, default_experiment_contract
from .meta_vnext_campaign import MetaV17CampaignRuntimeV1
from .meta_vnext_pilot import (
    MetaV17PilotOrchestratorV1,
    meta_v17_arm_policies,
)
from .precanary_orchestration import ArmRoundResultV1
from .real_canary import RealCanaryProposalBrokerV1
from .real_pilot import campaign_pilot_protocol, pilot_guard_context
from .training_runtime_release import CAMPAIGN_TRAINING_RUNNER_ABI


CAMPAIGN_PILOT_SEARCH_SEED = 9214
CAMPAIGN_PILOT_ROUNDS_PER_ARM = 5
CAMPAIGN_PILOT_EXPERIMENT_ID = (
    "HELIX-ABC-DEVELOPMENT-CAMPAIGN-PILOT-9214-V12"
)
CAMPAIGN_PILOT_ASSIGNMENT_NONCE = (
    "M6-CAMPAIGN-PILOT-9214-OPAQUE-V12"
)


@dataclass(frozen=True, slots=True)
class CampaignPilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[Any, Any, Any]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "CampaignPilotStoreContractV1":
        base = default_experiment_contract()
        policies = meta_v17_arm_policies()
        payload = {
            "arm_policies": [item.to_dict() for item in policies],
            "authority": "NONE",
                "campaign_runtime": "BL_ICF_EXECUTABLE_PROFILE_V2",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": CAMPAIGN_PILOT_EXPERIMENT_ID,
            "formal_acceptance": False,
            "main_eligibility": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": (
                CAMPAIGN_PILOT_ROUNDS_PER_ARM
            ),
            "search_seeds": [CAMPAIGN_PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=CAMPAIGN_PILOT_EXPERIMENT_ID,
            arm_policies=policies,
            search_seeds=(CAMPAIGN_PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=(
                CAMPAIGN_PILOT_ROUNDS_PER_ARM
            ),
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


def campaign_pilot_guard_context() -> GuardContext:
    base = pilot_guard_context()
    protocol = campaign_pilot_protocol()
    claim = canonical_value(base.claim)
    evidence = canonical_value(base.current_evidence)
    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6-CAMPAIGN-PILOT-9214-V12",
            "protocol_id": protocol["protocol_id"],
            "target_model": "CANDIDATE_SPECIFIC",
            "comparator": "FROZEN_ARM_MATCHED_PARENT",
        },
        protocol=protocol,
        current_evidence={
            **evidence,
            "snapshot_id": "M6-CAMPAIGN-PILOT-9214-V12-EMPTY",
            "claim_id": "CLAIM-M6-CAMPAIGN-PILOT-9214-V12",
            "protocol_id": protocol["protocol_id"],
        },
    )


class CampaignPilotOrchestratorV1(MetaV17PilotOrchestratorV1):
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
    ) -> None:
        super().__init__(
            root,
            broker=broker,
            meta_runtime=meta_runtime,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=CampaignPilotStoreContractV1.create(),
            _assignment_nonce=CAMPAIGN_PILOT_ASSIGNMENT_NONCE,
            _guard_context=campaign_pilot_guard_context(),
            _training_runner_abi=CAMPAIGN_TRAINING_RUNNER_ABI,
        )

    def run_pilot(
        self,
    ) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        return tuple(
            self.run_fake_triplet(
                search_seed=CAMPAIGN_PILOT_SEARCH_SEED,
                round_index=round_index,
                drafts=(),
            )
            for round_index in range(
                1, CAMPAIGN_PILOT_ROUNDS_PER_ARM + 1
            )
        )


__all__ = [
    "CAMPAIGN_PILOT_ASSIGNMENT_NONCE",
    "CAMPAIGN_PILOT_EXPERIMENT_ID",
    "CAMPAIGN_PILOT_ROUNDS_PER_ARM",
    "CAMPAIGN_PILOT_SEARCH_SEED",
    "CampaignPilotOrchestratorV1",
    "CampaignPilotStoreContractV1",
    "campaign_pilot_guard_context",
]
