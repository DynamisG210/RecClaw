"""Fresh 50-round post-V24 Effect Pilot on one common gpu35 backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from recclaw_core.helix.contracts import GuardContext

from .campaign_pilot_v16 import v16_arm_policies
from .campaign_runtime import campaign_runtime_profile
from .canonical import canonical_json_bytes, canonical_value, sha256_digest
from .contracts import (
    ArmPolicyV1,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from .meta_vnext_campaign import MetaV20CampaignRuntimeV1
from .meta_vnext_pilot import MetaV17PilotOrchestratorV1
from .original_main import PinnedOriginalMainAdapterV1
from .precanary_orchestration import (
    ArmRoundResultV1,
    PreCanaryInvariantError,
)
from .real_canary import RealCanaryProposalBrokerV1
from .real_pilot import campaign_pilot_protocol, pilot_guard_context
from .state_store import RegisterArtifactCommand
from .training_runtime_release import CAMPAIGN_TRAINING_RUNNER_ABI


V25_PILOT_SEARCH_SEED = 9227
V25_PILOT_ROUNDS_PER_ARM = 50
V25_PILOT_CHECKPOINTS = (10, 20, 50)
V25_PILOT_EXPERIMENT_ID = (
    "HELIX-ABC-DEVELOPMENT-EFFECT-PILOT-9227-V25"
)
V25_PILOT_ASSIGNMENT_NONCE = "M6I-V25-EFFECT-PILOT-9227-OPAQUE-V1"
V25_EXECUTABLE_PROFILE_DIGEST = (
    "952ba027cb8dbfd8831fca6d33ce39db1aabd4dbc41534b089250e0dc1253b15"
)
V25_RESOURCE_ENVELOPE_RESOURCE = (
    "pilot_v25_gpu35_resource_envelope.json"
)


def v25_arm_policies() -> tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]:
    """Preserve the exact frozen V16 A/B/C treatment definitions."""

    return v16_arm_policies()


def v25_resource_ceilings() -> ResourceCeilingsV1:
    payload = json.loads(
        resources.files(
            "recclaw_core.experiments.helix_abc_v1.resources"
        )
        .joinpath(V25_RESOURCE_ENVELOPE_RESOURCE)
        .read_bytes()
    )
    return ResourceCeilingsV1(**payload["resource_ceilings"])


@dataclass(frozen=True, slots=True)
class V25PilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "V25PilotStoreContractV1":
        base = default_experiment_contract()
        policies = v25_arm_policies()
        profile = campaign_runtime_profile()
        if (
            profile["profile_id"] != "BL_ICF_EXECUTABLE_PROFILE_V2"
            or profile["executable_profile_digest"]
            != V25_EXECUTABLE_PROFILE_DIGEST
            or int(profile["executable_mechanism_count"]) != 66
        ):
            raise PreCanaryInvariantError(
                "V25 runtime profile is not the repaired 66-semantics release"
            )
        payload = {
            "arm_policies": [item.to_dict() for item in policies],
            "authority": "NONE",
            "campaign_runtime": "BL_ICF_EXECUTABLE_PROFILE_V2",
            "checkpoints": list(V25_PILOT_CHECKPOINTS),
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": V25_PILOT_EXPERIMENT_ID,
            "formal_acceptance": False,
            "main_eligibility": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": V25_PILOT_ROUNDS_PER_ARM,
            "search_seeds": [V25_PILOT_SEARCH_SEED],
        }
        return cls(
            experiment_id=V25_PILOT_EXPERIMENT_ID,
            arm_policies=policies,
            search_seeds=(V25_PILOT_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=V25_PILOT_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


def v25_pilot_guard_context() -> GuardContext:
    base = pilot_guard_context()
    protocol = campaign_pilot_protocol()
    claim = canonical_value(base.claim)
    evidence = canonical_value(base.current_evidence)
    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6I-V25-EFFECT-PILOT-9227",
            "protocol_id": protocol["protocol_id"],
            "target_model": "CANDIDATE_SPECIFIC",
            "comparator": "FROZEN_ARM_MATCHED_PARENT",
        },
        protocol=protocol,
        current_evidence={
            **evidence,
            "snapshot_id": "M6I-V25-EFFECT-PILOT-9227-EMPTY",
            "claim_id": "CLAIM-M6I-V25-EFFECT-PILOT-9227",
            "protocol_id": protocol["protocol_id"],
        },
    )


class V25PilotOrchestratorV1(MetaV17PilotOrchestratorV1):
    def __init__(
        self,
        root: Path,
        *,
        broker: RealCanaryProposalBrokerV1,
        meta_runtime: MetaV20CampaignRuntimeV1,
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
                "V25 Effect Pilot requires the direct pinned Main Original path"
            )
        if broker.campaign_meta_runtime is not meta_runtime:
            raise PreCanaryInvariantError(
                "V25 Broker and orchestrator require the same V20 runtime"
            )
        expected_ceilings = v25_resource_ceilings()
        super().__init__(
            root,
            broker=broker,
            meta_runtime=meta_runtime,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=V25PilotStoreContractV1.create(),
            _assignment_nonce=V25_PILOT_ASSIGNMENT_NONCE,
            _guard_context=v25_pilot_guard_context(),
            _training_runner_abi=CAMPAIGN_TRAINING_RUNNER_ABI,
            _resource_ceilings=expected_ceilings,
        )
        if self.resource_ceilings != expected_ceilings:
            raise PreCanaryInvariantError(
                "V25 runtime resource envelope binding mismatch"
            )

    def run_round_triplet(
        self,
        *,
        round_index: int,
    ) -> tuple[ArmRoundResultV1, ...]:
        return super().run_fake_triplet(
            search_seed=V25_PILOT_SEARCH_SEED,
            round_index=round_index,
            drafts=(),
        )

    def _persist_read_only_checkpoint(self, round_index: int) -> None:
        connection = self.store._connection
        counts = {
            "closed_triplet_barriers": int(
                connection.execute(
                    "SELECT COUNT(*) FROM triplet_barrier "
                    "WHERE closed_bitmap=7 AND next_index_authorized=1"
                ).fetchone()[0]
            ),
            "execution_claims_finished": int(
                connection.execute(
                    "SELECT COUNT(*) FROM execution_claims "
                    "WHERE claim_state='FINISHED'"
                ).fetchone()[0]
            ),
            "open_rounds": int(
                connection.execute(
                    "SELECT COUNT(*) FROM rounds WHERE status='OPEN'"
                ).fetchone()[0]
            ),
            "terminal_rounds": int(
                connection.execute(
                    "SELECT COUNT(*) FROM rounds "
                    "WHERE status IN ('CLOSED','ABORTED')"
                ).fetchone()[0]
            ),
        }
        expected = {
            "closed_triplet_barriers": round_index,
            "open_rounds": 0,
            "terminal_rounds": round_index * 3,
        }
        if any(counts[key] != value for key, value in expected.items()):
            raise PreCanaryInvariantError(
                "V25 read-only checkpoint encountered an incomplete barrier"
            )
        payload = canonical_value(
            {
                "association_free": True,
                "authority": "NONE",
                "checkpoint_round": round_index,
                "counts": counts,
                "evidence_class": "DEVELOPMENT_ONLY",
                "experiment_id": self.contract.experiment_id,
                "formal_acceptance": False,
                "integrated_state_digest": sha256_digest(
                    self.integrated_state.audit_projection()
                ),
                "schema": (
                    "recclaw.v25-effect-pilot-read-only-checkpoint.v1"
                ),
                "search_seed": V25_PILOT_SEARCH_SEED,
                "store_integrity_digest": sha256_digest(
                    self.store.integrity_report()
                ),
            }
        )
        record = {
            **payload,
            "checkpoint_digest": sha256_digest(payload),
        }
        self.store.register_artifact(
            RegisterArtifactCommand(
                round_id=None,
                artifact_type=(
                    "EFFECT_PILOT_READ_ONLY_CHECKPOINT_V1"
                ),
                relative_path=(
                    "effect_checkpoints/"
                    f"{round_index:04d}.read-only.v1.json"
                ),
                producer="V25PilotOrchestratorV1",
                idempotency_key=(
                    f"v25:effect-read-only-checkpoint:{round_index}"
                ),
            ),
            canonical_json_bytes(record) + b"\n",
        )

    def run_pilot(
        self,
    ) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        results = []
        for round_index in range(1, V25_PILOT_ROUNDS_PER_ARM + 1):
            results.append(
                self.run_round_triplet(round_index=round_index)
            )
            if round_index in V25_PILOT_CHECKPOINTS:
                self._persist_read_only_checkpoint(round_index)
        return tuple(results)

    def immutable_audit_bundle(self, snapshot_root: Path) -> dict[str, Any]:
        bundle = super().immutable_audit_bundle(snapshot_root)
        integrated = self.integrated_state.audit_projection()
        sharing = self.broker.call_sharing_audit()
        encoded_sharing = json.dumps(
            sharing["records"],
            sort_keys=True,
            separators=(",", ":"),
        ).lower()
        forbidden_guard_private_tokens = (
            "evidence_guard",
            "evidence_root",
            "guard_ledger",
            "guard_private",
        )
        bundle["m6i_safety_projection"] = canonical_value(
            {
                "call_sharing_policy": sharing["policy"],
                "candidate_instance_identities": sharing[
                    "candidate_instance_identities"
                ],
                "consumer_logical_identities": sharing[
                    "consumer_logical_identities"
                ],
                "cross_arm_physical_identities": sharing[
                    "cross_arm_physical_identities"
                ],
                "guard_private_context_token_count": sum(
                    token in encoded_sharing
                    for token in forbidden_guard_private_tokens
                ),
                "integrated_cross_arm_reads": integrated[
                    "cross_arm_reads"
                ],
                "integrated_round_count": len(integrated["rounds"]),
                "integrated_triplet_barrier_count": len(
                    integrated["triplet_barriers"]
                ),
                "physical_call_identities": sharing[
                    "physical_call_identities"
                ],
            }
        )
        return canonical_value(bundle)


__all__ = [
    "V25_EXECUTABLE_PROFILE_DIGEST",
    "V25_PILOT_ASSIGNMENT_NONCE",
    "V25_PILOT_CHECKPOINTS",
    "V25_PILOT_EXPERIMENT_ID",
    "V25_PILOT_ROUNDS_PER_ARM",
    "V25_PILOT_SEARCH_SEED",
    "V25PilotOrchestratorV1",
    "V25PilotStoreContractV1",
    "v25_arm_policies",
    "v25_pilot_guard_context",
    "v25_resource_ceilings",
]
