"""Typed Search-Utility contracts for the M2 Research Capability Line."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping

from recclaw_core.mechanism_space.canonical import deep_freeze, deep_thaw, snapshot_json

from .canonical import canonical_value, sha256_digest
from .contracts import (
    AUTHORITY,
    EVIDENCE_CLASS,
    ProducerExecutionModeV1,
    validate_no_research_evidence_authority_fields,
)


DISCOVERY_PRODUCERS = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)


class ProposalIntentV1(str, Enum):
    DISCOVERY = "DISCOVERY"
    FALSIFICATION = "FALSIFICATION"
    CONTROL = "CONTROL"
    REPAIR = "REPAIR"


class DiscoveryCreditV1(str, Enum):
    DISCOVERY = "DISCOVERY"
    NON_DISCOVERY_CONTROL = "NON_DISCOVERY_CONTROL"
    NON_DISCOVERY_REPAIR = "NON_DISCOVERY_REPAIR"


class RouterHardGateReasonV1(str, Enum):
    ALLOW = "ALLOW"
    BL_COMPILE_FAILED = "BL_COMPILE_FAILED"
    SEMANTIC_DUPLICATE = "SEMANTIC_DUPLICATE"
    RUNNABLE_BELOW_FLOOR = "RUNNABLE_BELOW_FLOOR"
    UTILITY_BELOW_FLOOR = "UTILITY_BELOW_FLOOR"
    BLOCKER_RISK_ABOVE_CEILING = "BLOCKER_RISK_ABOVE_CEILING"
    COST_ABOVE_CEILING = "COST_ABOVE_CEILING"
    SLATE_CEILING = "SLATE_CEILING"


class AgentizationVerdictV1(str, Enum):
    PASS_INDEPENDENT_MULTI_AGENT = "PASS_INDEPENDENT_MULTI_AGENT"
    PASS_BATCHED_ONLY = "PASS_BATCHED_ONLY"
    FAIL = "FAIL"


class MetaVerdictV1(str, Enum):
    PASS_VERSIONED_META = "PASS_VERSIONED_META"
    PASS_STATIC_ONLY = "PASS_STATIC_ONLY"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class SearchUtilityFeaturesV1:
    runnable_probability: float
    useful_signal: float
    frontier_potential: float
    information_gain: float
    cost: float
    blocker_risk: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{item.name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{item.name} must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class CandidateProposalV2:
    candidate_id: str
    producer_id: str
    producer_role: str
    proposal_intent: ProposalIntentV1
    discovery_credit: DiscoveryCreditV1
    mechanism_axis: str
    mechanism_program: Mapping[str, Any]
    utility_features: SearchUtilityFeaturesV1
    parent_candidate_id: str | None
    assigned_before_call: bool
    post_hoc_relabel: bool

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand-"):
            raise ValueError("candidate_id must use the closed cand- namespace")
        if not self.assigned_before_call or self.post_hoc_relabel:
            raise ValueError("Producer identity and role must be assigned before the call")
        if self.proposal_intent in {
            ProposalIntentV1.CONTROL,
            ProposalIntentV1.REPAIR,
        } and self.discovery_credit is DiscoveryCreditV1.DISCOVERY:
            raise ValueError("control/repair proposals cannot receive discovery credit")
        if self.proposal_intent is ProposalIntentV1.FALSIFICATION:
            if self.producer_role not in {"falsification_designer", "neutral"}:
                raise ValueError("falsification proposals require a preassigned slot")
        object.__setattr__(
            self,
            "mechanism_program",
            deep_freeze(snapshot_json(dict(self.mechanism_program))),
        )
        validate_no_research_evidence_authority_fields(self.search_owned_projection())

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_id": self.candidate_id,
                "producer_id": self.producer_id,
                "producer_role": self.producer_role,
                "proposal_intent": self.proposal_intent,
                "discovery_credit": self.discovery_credit,
                "mechanism_axis": self.mechanism_axis,
                "mechanism_program": deep_thaw(self.mechanism_program),
                "utility_features": self.utility_features,
                "parent_candidate_id": self.parent_candidate_id,
                "assigned_before_call": self.assigned_before_call,
                "post_hoc_relabel": self.post_hoc_relabel,
            }
        )

    def search_owned_projection(self) -> dict[str, Any]:
        """Exclude the opaque shared BL program from Research-owned field audit."""

        value = self.to_dict()
        value.pop("mechanism_program")
        return value


@dataclass(frozen=True, slots=True)
class ProducerCallRecordV1:
    session_id: str
    mode: ProducerExecutionModeV1
    physical_call_id: str
    producer_id: str
    producer_role: str
    request_digest: str
    response_digest: str
    context_digest: str
    memory_digest: str
    prompt_digest: str
    rng_digest: str
    candidate_ids: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    billed_tokens: int
    latency_ms: int

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ProducerSessionResultV1:
    session_id: str
    mode: ProducerExecutionModeV1
    calls: tuple[ProducerCallRecordV1, ...]
    proposals: tuple[CandidateProposalV2, ...]
    total_resource_envelope_digest: str
    base_model_ref: str
    bl_projection_digest: str
    candidate_schema_ref: str
    proposal_count: int
    physical_call_count: int
    input_tokens: int
    output_tokens: int
    billed_tokens: int
    session_latency_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "calls", tuple(self.calls))
        object.__setattr__(self, "proposals", tuple(self.proposals))
        if self.proposal_count != len(self.proposals):
            raise ValueError("proposal_count does not match proposals")
        if self.physical_call_count != len(self.calls):
            raise ValueError("physical_call_count does not match calls")
        if any(item.session_id != self.session_id or item.mode is not self.mode for item in self.calls):
            raise ValueError("call lineage does not match its ProposalGenerationSession")
        emitted = tuple(
            candidate_id for call in self.calls for candidate_id in call.candidate_ids
        )
        if set(emitted) != {item.candidate_id for item in self.proposals}:
            raise ValueError("physical calls and proposal lineage are inconsistent")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "session_id": self.session_id,
                "mode": self.mode,
                "calls": [item.to_dict() for item in self.calls],
                "proposals": [item.to_dict() for item in self.proposals],
                "total_resource_envelope_digest": self.total_resource_envelope_digest,
                "base_model_ref": self.base_model_ref,
                "bl_projection_digest": self.bl_projection_digest,
                "candidate_schema_ref": self.candidate_schema_ref,
                "proposal_count": self.proposal_count,
                "physical_call_count": self.physical_call_count,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "billed_tokens": self.billed_tokens,
                "session_latency_ms": self.session_latency_ms,
            }
        )


@dataclass(frozen=True, slots=True)
class RouterHardGateDecisionV1:
    candidate_id: str
    allowed: bool
    reason: RouterHardGateReasonV1
    compile_report_digest: str | None
    mechanism_semantics_digest: str | None
    feature_digest: str
    policy_digest: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class RouteTraceV1:
    pool_digest: str
    ordered_candidate_ids: tuple[str, ...]
    ranked_candidate_ids: tuple[str, ...]
    decisions: tuple[RouterHardGateDecisionV1, ...]
    selected_candidate_id: str | None
    selection_score: float | None
    policy_digest: str

    def __post_init__(self) -> None:
        decision_by_id = {item.candidate_id: item for item in self.decisions}
        if len(decision_by_id) != len(self.decisions):
            raise ValueError("RouteTrace candidate decisions must be unique")
        if set(decision_by_id) != set(self.ordered_candidate_ids):
            raise ValueError("RouteTrace decisions must cover the complete proposal pool")
        if any(
            candidate_id not in decision_by_id
            or not decision_by_id[candidate_id].allowed
            for candidate_id in self.ranked_candidate_ids
        ):
            raise ValueError("RouteTrace ranking may contain only allowed candidates")
        expected_selected = (
            self.ranked_candidate_ids[0] if self.ranked_candidate_ids else None
        )
        if self.selected_candidate_id != expected_selected:
            raise ValueError("RouteTrace selection must be the first ranked candidate")
        if (self.selected_candidate_id is None) != (self.selection_score is None):
            raise ValueError("RouteTrace selection and score must be present together")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class DevelopmentalMechanismBeliefV1:
    """Exactly the eight Search-Memory fields frozen by M2."""

    hypothesis_id: str
    mechanism_axis: str
    competing_hypotheses: tuple[str, ...]
    predicted_outcome_signature: str
    evidence_for: tuple[str, ...]
    evidence_against: tuple[str, ...]
    unresolved_confounds: tuple[str, ...]
    next_discriminative_test: str

    authority = AUTHORITY
    evidence_class = EVIDENCE_CLASS

    def __post_init__(self) -> None:
        if tuple(item.name for item in fields(self)) != (
            "hypothesis_id",
            "mechanism_axis",
            "competing_hypotheses",
            "predicted_outcome_signature",
            "evidence_for",
            "evidence_against",
            "unresolved_confounds",
            "next_discriminative_test",
        ):
            raise ValueError("Mechanism Belief must retain exactly eight fields")
        validate_no_research_evidence_authority_fields(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)
