"""Shared immutable contracts visible to EvidencePort and Helix composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from recclaw_core.mechanism_space.canonical import deep_freeze, deep_thaw, snapshot_json

from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value, sha256_digest
from recclaw_core.experiments.helix_abc_v1.canonical import validate_sha256


class PortStage(str, Enum):
    PRE = "PRE"
    POST = "POST"


class PortStatus(str, Enum):
    NOT_ADJUDICATED = "NOT_ADJUDICATED"
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ADJUDICATED = "ADJUDICATED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class GuardContext:
    claim: Mapping[str, Any]
    protocol: Mapping[str, Any]
    current_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("claim", "protocol", "current_evidence"):
            object.__setattr__(
                self, name, deep_freeze(snapshot_json(dict(getattr(self, name))))
            )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": deep_thaw(self.claim),
            "protocol": deep_thaw(self.protocol),
            "current_evidence": deep_thaw(self.current_evidence),
        }


@dataclass(frozen=True, slots=True)
class CandidateEnvelope:
    candidate_id: str
    opaque_arm_instance_id: str
    common_status: str
    mechanism_program_digest: str
    common_plan_digest: str
    action_family: str
    planned_protocol: Mapping[str, Any]
    target_model: str
    comparator: str
    seed_ids: tuple[str, ...]
    purpose: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.opaque_arm_instance_id or not self.seed_ids:
            raise ValueError("CandidateEnvelope requires candidate and seed identities")
        if self.common_status != "COMMON_PASS":
            raise ValueError("EvidencePort accepts only COMMON_PASS candidates")
        validate_sha256(
            self.mechanism_program_digest, field_name="mechanism_program_digest"
        )
        validate_sha256(self.common_plan_digest, field_name="common_plan_digest")
        object.__setattr__(
            self,
            "planned_protocol",
            deep_freeze(snapshot_json(dict(self.planned_protocol))),
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_id": self.candidate_id,
                "opaque_arm_instance_id": self.opaque_arm_instance_id,
                "common_status": self.common_status,
                "mechanism_program_digest": self.mechanism_program_digest,
                "common_plan_digest": self.common_plan_digest,
                "action_family": self.action_family,
                "planned_protocol": deep_thaw(self.planned_protocol),
                "target_model": self.target_model,
                "comparator": self.comparator,
                "seed_ids": self.seed_ids,
                "purpose": self.purpose,
            }
        )


@dataclass(frozen=True, slots=True)
class RawResultEnvelope:
    candidate_id: str
    opaque_arm_instance_id: str
    raw_result_digest: str
    common_result_closure_digest: str
    observed_protocol: Mapping[str, Any]
    target_model: str
    comparator: str
    seed_runs: tuple[Mapping[str, Any], ...]
    observation_kind: str
    run_status: str
    artifact_identity_status: str
    normalized_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.opaque_arm_instance_id:
            raise ValueError("RawResultEnvelope requires opaque Arm identity")
        validate_sha256(self.raw_result_digest, field_name="raw_result_digest")
        validate_sha256(
            self.common_result_closure_digest,
            field_name="common_result_closure_digest",
        )
        object.__setattr__(
            self,
            "observed_protocol",
            deep_freeze(snapshot_json(dict(self.observed_protocol))),
        )
        object.__setattr__(
            self,
            "seed_runs",
            tuple(deep_freeze(snapshot_json(dict(item))) for item in self.seed_runs),
        )
        object.__setattr__(
            self,
            "normalized_metrics",
            deep_freeze(snapshot_json(dict(self.normalized_metrics))),
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_id": self.candidate_id,
                "opaque_arm_instance_id": self.opaque_arm_instance_id,
                "raw_result_digest": self.raw_result_digest,
                "common_result_closure_digest": self.common_result_closure_digest,
                "observed_protocol": deep_thaw(self.observed_protocol),
                "target_model": self.target_model,
                "comparator": self.comparator,
                "seed_runs": [deep_thaw(item) for item in self.seed_runs],
                "observation_kind": self.observation_kind,
                "run_status": self.run_status,
                "artifact_identity_status": self.artifact_identity_status,
                "normalized_metrics": deep_thaw(self.normalized_metrics),
            }
        )


@dataclass(frozen=True, slots=True)
class PortAdjudication:
    candidate_id: str
    stage: PortStage
    status: PortStatus
    protocol_status: str
    outcome_class: str
    claim_ceiling: str
    reason_codes: tuple[str, ...]
    comparator_delta: float | None
    evidence_use: str
    recommended_validation: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class CompactFeedback:
    candidate_id: str
    protocol_status: str
    outcome_class: str
    claim_ceiling: str
    reason_codes: tuple[str, ...]
    comparator_delta: float | None
    evidence_use: str
    recommended_validation: str

    def __post_init__(self) -> None:
        if len(self.__dataclass_fields__) != 8:
            raise ValueError("CompactFeedback must retain exactly eight fields")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)
