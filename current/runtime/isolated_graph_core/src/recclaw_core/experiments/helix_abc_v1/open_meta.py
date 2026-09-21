"""Pre-learning Open Meta interfaces for Research Line vNext.

This module deliberately implements only deterministic, static policy
fixtures and a canonical replay-file boundary.  It does not train, promote,
or update a policy, and it does not call a Provider, runner, evaluator, GPU,
held-out split, or outcome-bearing interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from .vnext_contracts import (
    AcquisitionDecisionV1,
    AcquisitionDispositionV1,
    AcquisitionStageV1,
    NEXT_FRESH_CAMPAIGN,
    TypedResearchEpisodeV1,
)


RESEARCH_STATIC_VNEXT = "RESEARCH_STATIC_VNEXT"
NEXT_ROUND = "NEXT_ROUND"

STATIC_IDEA_POLICY_VERSION_V1 = "research-static-idea-v1.0.0"
STATIC_EXPERIMENT_POLICY_VERSION_V1 = "research-static-experiment-v1.0.0"
STATIC_IDEA_POLICY_REF_V1 = "policy:research-static-vnext:idea:v1"
STATIC_EXPERIMENT_POLICY_REF_V1 = "policy:research-static-vnext:experiment:v1"

IDEA_FALLBACK_SCOPE_V1 = "OPEN_HIGH_CHANGE_IDEA_PRE_LEARNING"
EXPERIMENT_FALLBACK_SCOPE_V1 = "FROZEN_CAPABILITY_SLATE_PRE_LEARNING"


class OpenMetaContractError(ValueError):
    """Raised when an F0 policy or replay invariant is violated."""


class PolicySupportStatusV1(str, Enum):
    IN_SUPPORT = "IN_SUPPORT"
    OUT_OF_SUPPORT = "OUT_OF_SUPPORT"


class OpenMetaRecordV1:
    schema: ClassVar[str]
    contract_version: ClassVar[str] = "1.0.0"
    identity_namespace: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "contract_version": self.contract_version,
                **self.to_dict(),
            }
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_dict())

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())

    @property
    def record_id(self) -> str:
        return content_id(self.identity_namespace, self.canonical_dict())


def _text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OpenMetaContractError(
            f"{field_name} must be a non-empty, whitespace-normalized string"
        )
    return value


def _digest(value: str, *, field_name: str) -> str:
    try:
        return validate_sha256(str(value), field_name=field_name)
    except ValueError as exc:
        raise OpenMetaContractError(str(exc)) from exc


def _identity(ref: str, digest: str, *, prefix: str) -> tuple[str, str]:
    return (
        _text(ref, field_name=f"{prefix}_ref"),
        _digest(digest, field_name=f"{prefix}_digest"),
    )


def _sorted_records(
    values: Sequence[OpenMetaRecordV1],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[Any, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise OpenMetaContractError(f"{field_name} must be non-empty")
    if len({item.record_id for item in normalized}) != len(normalized):
        raise OpenMetaContractError(f"{field_name} identities must be unique")
    return tuple(sorted(normalized, key=lambda item: item.record_id))


def _require_record_type(
    values: Sequence[object],
    expected_type: type,
    *,
    field_name: str,
) -> None:
    if any(not isinstance(item, expected_type) for item in values):
        raise OpenMetaContractError(
            f"{field_name} must contain only {expected_type.__name__}"
        )


def _require_unique_refs(
    values: Sequence[object],
    *,
    ref_field: str,
    field_name: str,
) -> None:
    refs = tuple(str(getattr(item, ref_field)) for item in values)
    if len(set(refs)) != len(refs):
        raise OpenMetaContractError(f"{field_name} refs must be unique")


@dataclass(frozen=True, slots=True)
class PolicyIdentityV1(OpenMetaRecordV1):
    ref: str
    digest_value: str

    schema = "recclaw.research-line.vnext.open-meta.policy-identity.v1"
    identity_namespace = "recclaw-open-meta-policy-identity-v1"

    def __post_init__(self) -> None:
        ref, digest = _identity(self.ref, self.digest_value, prefix="identity")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "digest_value", digest)


@dataclass(frozen=True, slots=True)
class TypedEpisodePolicySummaryV1(OpenMetaRecordV1):
    """Identity-only historical Episode projection.

    Outcome, interpretation, cost, held-out/split, producer, origin, and source
    labels are intentionally absent.  The Episode digest keeps audit linkage,
    while static F0 policies never inspect it as a ranking feature.
    """

    episode_ref: str
    episode_digest: str
    campaign_id: str
    executable_capability_ref: str
    executable_capability_digest: str
    executable_profile_ref: str
    executable_profile_digest: str
    protocol_ref: str
    protocol_digest: str

    schema = "recclaw.research-line.vnext.open-meta.typed-episode-summary.v1"
    identity_namespace = "recclaw-open-meta-episode-summary-v1"

    def __post_init__(self) -> None:
        _text(self.campaign_id, field_name="campaign_id")
        for prefix in (
            "episode",
            "executable_capability",
            "executable_profile",
            "protocol",
        ):
            ref, digest = _identity(
                getattr(self, f"{prefix}_ref"),
                getattr(self, f"{prefix}_digest"),
                prefix=prefix,
            )
            object.__setattr__(self, f"{prefix}_ref", ref)
            object.__setattr__(self, f"{prefix}_digest", digest)

    @classmethod
    def from_episode(
        cls,
        episode: TypedResearchEpisodeV1,
    ) -> "TypedEpisodePolicySummaryV1":
        return cls(
            episode_ref=episode.episode_id,
            episode_digest=episode.digest,
            campaign_id=episode.campaign_id,
            executable_capability_ref=episode.executable_capability_ref,
            executable_capability_digest=episode.executable_capability_digest,
            executable_profile_ref=episode.executable_profile_ref,
            executable_profile_digest=episode.executable_profile_digest,
            protocol_ref=episode.protocol_ref,
            protocol_digest=episode.protocol_digest,
        )


@dataclass(frozen=True, slots=True)
class IdeaCandidateIdentityV1(OpenMetaRecordV1):
    research_spec_ref: str
    research_spec_digest: str
    direction_ref: str
    direction_digest: str
    high_change: bool
    current_profile_expressible: bool

    schema = "recclaw.research-line.vnext.open-meta.idea-candidate-identity.v1"
    identity_namespace = "recclaw-open-meta-idea-candidate-v1"

    def __post_init__(self) -> None:
        for prefix in ("research_spec", "direction"):
            ref, digest = _identity(
                getattr(self, f"{prefix}_ref"),
                getattr(self, f"{prefix}_digest"),
                prefix=prefix,
            )
            object.__setattr__(self, f"{prefix}_ref", ref)
            object.__setattr__(self, f"{prefix}_digest", digest)


@dataclass(frozen=True, slots=True)
class CapabilityIdentityV1(OpenMetaRecordV1):
    capability_ref: str
    capability_digest: str

    schema = "recclaw.research-line.vnext.open-meta.capability-identity.v1"
    identity_namespace = "recclaw-open-meta-capability-identity-v1"

    def __post_init__(self) -> None:
        ref, digest = _identity(
            self.capability_ref,
            self.capability_digest,
            prefix="capability",
        )
        object.__setattr__(self, "capability_ref", ref)
        object.__setattr__(self, "capability_digest", digest)


@dataclass(frozen=True, slots=True)
class IdeaBudgetV1(OpenMetaRecordV1):
    ideation_slots: int
    implementation_slots: int
    qualification_slots: int

    schema = "recclaw.research-line.vnext.open-meta.idea-budget.v1"
    identity_namespace = "recclaw-open-meta-idea-budget-v1"

    def __post_init__(self) -> None:
        values = (
            self.ideation_slots,
            self.implementation_slots,
            self.qualification_slots,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            raise OpenMetaContractError(
                "Idea budget slots must be non-negative integers"
            )


@dataclass(frozen=True, slots=True)
class ExperimentBudgetV1(OpenMetaRecordV1):
    experiment_slots: int

    schema = "recclaw.research-line.vnext.open-meta.experiment-budget.v1"
    identity_namespace = "recclaw-open-meta-experiment-budget-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.experiment_slots, bool)
            or not isinstance(self.experiment_slots, int)
            or self.experiment_slots < 0
        ):
            raise OpenMetaContractError(
                "Experiment budget slots must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class IdeaPolicyInputV1(OpenMetaRecordV1):
    research_context_ref: str
    research_context_digest: str
    protocol_ref: str
    protocol_digest: str
    current_profile_ref: str
    current_profile_digest: str
    candidates: tuple[IdeaCandidateIdentityV1, ...]
    budget: IdeaBudgetV1
    historical_episode_summaries: tuple[TypedEpisodePolicySummaryV1, ...] = ()

    schema = "recclaw.research-line.vnext.open-meta.idea-policy-input.v1"
    identity_namespace = "recclaw-open-meta-idea-input-v1"

    def __post_init__(self) -> None:
        for prefix in ("research_context", "protocol", "current_profile"):
            ref, digest = _identity(
                getattr(self, f"{prefix}_ref"),
                getattr(self, f"{prefix}_digest"),
                prefix=prefix,
            )
            object.__setattr__(self, f"{prefix}_ref", ref)
            object.__setattr__(self, f"{prefix}_digest", digest)
        if not isinstance(self.budget, IdeaBudgetV1):
            raise OpenMetaContractError("IdeaPolicyInputV1 requires IdeaBudgetV1")
        _require_record_type(
            self.candidates,
            IdeaCandidateIdentityV1,
            field_name="candidates",
        )
        _require_unique_refs(
            self.candidates,
            ref_field="research_spec_ref",
            field_name="candidates",
        )
        _require_record_type(
            self.historical_episode_summaries,
            TypedEpisodePolicySummaryV1,
            field_name="historical_episode_summaries",
        )
        object.__setattr__(
            self,
            "candidates",
            _sorted_records(self.candidates, field_name="candidates"),
        )
        object.__setattr__(
            self,
            "historical_episode_summaries",
            _sorted_records(
                self.historical_episode_summaries,
                field_name="historical_episode_summaries",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentPolicyInputV1(OpenMetaRecordV1):
    research_context_ref: str
    research_context_digest: str
    protocol_ref: str
    protocol_digest: str
    current_profile_ref: str
    current_profile_digest: str
    frozen_slate_ref: str
    frozen_slate_digest: str
    frozen_slate_capabilities: tuple[CapabilityIdentityV1, ...]
    acquisition_subjects: tuple[CapabilityIdentityV1, ...]
    budget: ExperimentBudgetV1
    historical_episode_summaries: tuple[TypedEpisodePolicySummaryV1, ...] = ()

    schema = "recclaw.research-line.vnext.open-meta.experiment-policy-input.v1"
    identity_namespace = "recclaw-open-meta-experiment-input-v1"

    def __post_init__(self) -> None:
        for prefix in (
            "research_context",
            "protocol",
            "current_profile",
            "frozen_slate",
        ):
            ref, digest = _identity(
                getattr(self, f"{prefix}_ref"),
                getattr(self, f"{prefix}_digest"),
                prefix=prefix,
            )
            object.__setattr__(self, f"{prefix}_ref", ref)
            object.__setattr__(self, f"{prefix}_digest", digest)
        if not isinstance(self.budget, ExperimentBudgetV1):
            raise OpenMetaContractError(
                "ExperimentPolicyInputV1 requires ExperimentBudgetV1"
            )
        for field_name, values in (
            ("frozen_slate_capabilities", self.frozen_slate_capabilities),
            ("acquisition_subjects", self.acquisition_subjects),
        ):
            _require_record_type(
                values,
                CapabilityIdentityV1,
                field_name=field_name,
            )
            _require_unique_refs(
                values,
                ref_field="capability_ref",
                field_name=field_name,
            )
        _require_record_type(
            self.historical_episode_summaries,
            TypedEpisodePolicySummaryV1,
            field_name="historical_episode_summaries",
        )
        object.__setattr__(
            self,
            "frozen_slate_capabilities",
            _sorted_records(
                self.frozen_slate_capabilities,
                field_name="frozen_slate_capabilities",
            ),
        )
        object.__setattr__(
            self,
            "acquisition_subjects",
            _sorted_records(
                self.acquisition_subjects,
                field_name="acquisition_subjects",
            ),
        )
        object.__setattr__(
            self,
            "historical_episode_summaries",
            _sorted_records(
                self.historical_episode_summaries,
                field_name="historical_episode_summaries",
                allow_empty=True,
            ),
        )


_IDEA_UNITS = {"IDEATION_SLOT", "IMPLEMENTATION_SLOT", "QUALIFICATION_SLOT"}
_EXPERIMENT_UNITS = {"EXPERIMENT_SLOT"}


@dataclass(frozen=True, slots=True)
class PolicyBudgetAllocationV1(OpenMetaRecordV1):
    stage: AcquisitionStageV1
    target_kind: str
    target_ref: str
    units: int
    unit_name: str
    reason_code: str

    schema = "recclaw.research-line.vnext.open-meta.budget-allocation.v1"
    identity_namespace = "recclaw-open-meta-budget-allocation-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.stage, AcquisitionStageV1):
            raise OpenMetaContractError("budget allocation stage is invalid")
        _text(self.target_kind, field_name="target_kind")
        _text(self.target_ref, field_name="target_ref")
        _text(self.reason_code, field_name="reason_code")
        if (
            isinstance(self.units, bool)
            or not isinstance(self.units, int)
            or self.units <= 0
        ):
            raise OpenMetaContractError("allocated units must be a positive integer")
        expected_units = (
            _IDEA_UNITS
            if self.stage is AcquisitionStageV1.IDEA
            else _EXPERIMENT_UNITS
        )
        if self.unit_name not in expected_units:
            raise OpenMetaContractError(
                f"{self.stage.value} allocation cannot use {self.unit_name}"
            )


@dataclass(frozen=True, slots=True)
class OpenMetaPolicyDecisionV1(OpenMetaRecordV1):
    stage: AcquisitionStageV1
    policy_ref: str
    policy_digest: str
    policy_version: str
    policy_mode: str
    policy_input_ref: str
    policy_input_digest: str
    support_status: PolicySupportStatusV1
    support_reason_codes: tuple[str, ...]
    fallback_scope: str
    fallback_reason_codes: tuple[str, ...]
    budget_allocations: tuple[PolicyBudgetAllocationV1, ...]
    acquisition_decisions: tuple[AcquisitionDecisionV1, ...]
    explanation: str

    schema = "recclaw.research-line.vnext.open-meta.policy-decision.v1"
    identity_namespace = "recclaw-open-meta-policy-decision-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.stage, AcquisitionStageV1):
            raise OpenMetaContractError("policy decision stage is invalid")
        for prefix in ("policy", "policy_input"):
            ref, digest = _identity(
                getattr(self, f"{prefix}_ref"),
                getattr(self, f"{prefix}_digest"),
                prefix=prefix,
            )
            object.__setattr__(self, f"{prefix}_ref", ref)
            object.__setattr__(self, f"{prefix}_digest", digest)
        _text(self.policy_version, field_name="policy_version")
        _text(self.explanation, field_name="explanation")
        if self.policy_mode != RESEARCH_STATIC_VNEXT:
            raise OpenMetaContractError(
                "F0 supports only the explicit RESEARCH_STATIC_VNEXT baseline"
            )
        if not isinstance(self.support_status, PolicySupportStatusV1):
            raise OpenMetaContractError("support_status is invalid")
        support = tuple(sorted(set(self.support_reason_codes)))
        fallback = tuple(sorted(set(self.fallback_reason_codes)))
        if not support or not fallback:
            raise OpenMetaContractError(
                "support and fallback reasons must both be explicit"
            )
        object.__setattr__(self, "support_reason_codes", support)
        object.__setattr__(self, "fallback_reason_codes", fallback)
        expected_scope = (
            IDEA_FALLBACK_SCOPE_V1
            if self.stage is AcquisitionStageV1.IDEA
            else EXPERIMENT_FALLBACK_SCOPE_V1
        )
        if self.fallback_scope != expected_scope:
            raise OpenMetaContractError("fallback scope crosses the policy domain")
        _require_record_type(
            self.budget_allocations,
            PolicyBudgetAllocationV1,
            field_name="budget_allocations",
        )
        if any(
            not isinstance(item, AcquisitionDecisionV1)
            for item in self.acquisition_decisions
        ):
            raise OpenMetaContractError(
                "acquisition_decisions must contain only AcquisitionDecisionV1"
            )
        allocations = tuple(
            sorted(self.budget_allocations, key=lambda item: item.record_id)
        )
        if any(item.stage is not self.stage for item in allocations):
            raise OpenMetaContractError("budget allocation crosses the policy domain")
        decisions = tuple(
            sorted(self.acquisition_decisions, key=lambda item: item.decision_id)
        )
        if any(item.stage is not self.stage for item in decisions):
            raise OpenMetaContractError(
                "acquisition decision crosses the policy domain"
            )
        if any(
            item.policy_ref != self.policy_ref
            or item.policy_digest != self.policy_digest
            for item in decisions
        ):
            raise OpenMetaContractError(
                "acquisition decisions must bind the same frozen policy"
            )
        if self.support_status is PolicySupportStatusV1.OUT_OF_SUPPORT:
            if allocations:
                raise OpenMetaContractError(
                    "out-of-support decisions cannot allocate budget"
                )
            if any(
                item.disposition is not AcquisitionDispositionV1.DEFER
                for item in decisions
            ):
                raise OpenMetaContractError(
                    "out-of-support acquisitions must be explicit DEFER decisions"
                )
        elif not allocations or not decisions:
            raise OpenMetaContractError(
                "in-support decisions require allocations and acquisitions"
            )
        object.__setattr__(self, "budget_allocations", allocations)
        object.__setattr__(self, "acquisition_decisions", decisions)

    @property
    def decision_id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True)
class OpenMetaPolicyActivationV1(OpenMetaRecordV1):
    stage: AcquisitionStageV1
    policy_ref: str
    policy_digest: str
    policy_version: str
    policy_mode: str
    decision_ref: str
    decision_digest: str
    current_policy_ref: str
    current_policy_digest: str
    activation_boundary: str
    policy_frozen: bool
    promotion_authorized: bool
    replaces_current_policy: bool

    schema = "recclaw.research-line.vnext.open-meta.policy-activation.v1"
    identity_namespace = "recclaw-open-meta-policy-activation-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.stage, AcquisitionStageV1):
            raise OpenMetaContractError("activation stage is invalid")
        for prefix in ("policy", "decision", "current_policy"):
            ref, digest = _identity(
                getattr(self, f"{prefix}_ref"),
                getattr(self, f"{prefix}_digest"),
                prefix=prefix,
            )
            object.__setattr__(self, f"{prefix}_ref", ref)
            object.__setattr__(self, f"{prefix}_digest", digest)
        _text(self.policy_version, field_name="policy_version")
        if self.policy_mode != RESEARCH_STATIC_VNEXT:
            raise OpenMetaContractError("F0 activation is static-policy-only")
        expected = (
            NEXT_FRESH_CAMPAIGN
            if self.stage is AcquisitionStageV1.IDEA
            else NEXT_ROUND
        )
        if self.activation_boundary != expected:
            raise OpenMetaContractError(
                f"{self.stage.value} policy activates only at {expected}"
            )
        if not self.policy_frozen:
            raise OpenMetaContractError("only a frozen policy can be scheduled")
        if self.promotion_authorized:
            raise OpenMetaContractError("F0 does not implement policy promotion")
        if self.replaces_current_policy:
            raise OpenMetaContractError(
                "an unpromoted policy cannot replace the current policy"
            )

    @property
    def activation_id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True)
class OpenMetaReplayRecordV1(OpenMetaRecordV1):
    stage: AcquisitionStageV1
    policy_ref: str
    policy_digest: str
    policy_version: str
    policy_mode: str
    policy_input_ref: str
    policy_input_digest: str
    decision_ref: str
    decision_digest: str
    budget_allocations: tuple[PolicyBudgetAllocationV1, ...]
    acquisition_decisions: tuple[AcquisitionDecisionV1, ...]
    activation_ref: str
    activation_digest: str
    activation_boundary: str

    schema = "recclaw.research-line.vnext.open-meta.replay-record.v1"
    identity_namespace = "recclaw-open-meta-replay-record-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.stage, AcquisitionStageV1):
            raise OpenMetaContractError("replay stage is invalid")
        for prefix in ("policy", "policy_input", "decision", "activation"):
            ref, digest = _identity(
                getattr(self, f"{prefix}_ref"),
                getattr(self, f"{prefix}_digest"),
                prefix=prefix,
            )
            object.__setattr__(self, f"{prefix}_ref", ref)
            object.__setattr__(self, f"{prefix}_digest", digest)
        _text(self.policy_version, field_name="policy_version")
        if self.policy_mode != RESEARCH_STATIC_VNEXT:
            raise OpenMetaContractError("replay mode must be RESEARCH_STATIC_VNEXT")
        expected = (
            NEXT_FRESH_CAMPAIGN
            if self.stage is AcquisitionStageV1.IDEA
            else NEXT_ROUND
        )
        if self.activation_boundary != expected:
            raise OpenMetaContractError("replay activation boundary is invalid")
        _require_record_type(
            self.budget_allocations,
            PolicyBudgetAllocationV1,
            field_name="budget_allocations",
        )
        if any(
            not isinstance(item, AcquisitionDecisionV1)
            for item in self.acquisition_decisions
        ):
            raise OpenMetaContractError(
                "acquisition_decisions must contain only AcquisitionDecisionV1"
            )
        allocations = tuple(
            sorted(self.budget_allocations, key=lambda item: item.record_id)
        )
        decisions = tuple(
            sorted(self.acquisition_decisions, key=lambda item: item.decision_id)
        )
        if any(item.stage is not self.stage for item in allocations) or any(
            item.stage is not self.stage for item in decisions
        ):
            raise OpenMetaContractError("replay record crosses policy domains")
        object.__setattr__(self, "budget_allocations", allocations)
        object.__setattr__(self, "acquisition_decisions", decisions)


@dataclass(frozen=True, slots=True)
class OpenMetaReplayDatasetV1(OpenMetaRecordV1):
    dataset_version: str
    records: tuple[OpenMetaReplayRecordV1, ...]

    schema = "recclaw.research-line.vnext.open-meta.replay-dataset.v1"
    identity_namespace = "recclaw-open-meta-replay-dataset-v1"

    def __post_init__(self) -> None:
        if self.dataset_version != "1.0.0":
            raise OpenMetaContractError("unsupported replay dataset version")
        _require_record_type(
            self.records,
            OpenMetaReplayRecordV1,
            field_name="records",
        )
        object.__setattr__(
            self,
            "records",
            _sorted_records(self.records, field_name="records"),
        )


_IDEA_FEATURE_SCHEMA = {
    "schema": "recclaw.research-line.vnext.open-meta.idea-features.v1",
    "features": (
        "opaque_spec_identity",
        "direction_identity",
        "high_change",
        "current_profile_expressible",
    ),
    "origin_features": (),
    "outcome_features": (),
    "held_out_features": (),
}
_IDEA_BUDGET_SCHEMA = {
    "schema": "recclaw.research-line.vnext.open-meta.idea-budget-schema.v1",
    "units": tuple(sorted(_IDEA_UNITS)),
}
_IDEA_ELIGIBILITY_SCHEMA = {
    "schema": "recclaw.research-line.vnext.open-meta.idea-eligibility.v1",
    "rule": "high_change_and_not_currently_expressible",
}
_EXPERIMENT_FEATURE_SCHEMA = {
    "schema": "recclaw.research-line.vnext.open-meta.experiment-features.v1",
    "features": ("opaque_capability_identity", "frozen_slate_membership"),
    "origin_features": (),
    "outcome_features": (),
    "held_out_features": (),
}
_EXPERIMENT_BUDGET_SCHEMA = {
    "schema": "recclaw.research-line.vnext.open-meta.experiment-budget-schema.v1",
    "units": tuple(sorted(_EXPERIMENT_UNITS)),
}
_EXPERIMENT_ELIGIBILITY_SCHEMA = {
    "schema": "recclaw.research-line.vnext.open-meta.experiment-eligibility.v1",
    "rule": "subject_is_in_frozen_slate",
}

STATIC_IDEA_POLICY_DIGEST_V1 = sha256_digest(
    {
        "policy_ref": STATIC_IDEA_POLICY_REF_V1,
        "policy_version": STATIC_IDEA_POLICY_VERSION_V1,
        "policy_mode": RESEARCH_STATIC_VNEXT,
        "support_scope": IDEA_FALLBACK_SCOPE_V1,
        "ordering": "CANONICAL_OPAQUE_IDENTITY",
        "learner": None,
        "promotion": None,
        "catalog_fallback": None,
        "configuration_tuning_fallback": None,
    }
)
STATIC_EXPERIMENT_POLICY_DIGEST_V1 = sha256_digest(
    {
        "policy_ref": STATIC_EXPERIMENT_POLICY_REF_V1,
        "policy_version": STATIC_EXPERIMENT_POLICY_VERSION_V1,
        "policy_mode": RESEARCH_STATIC_VNEXT,
        "support_scope": EXPERIMENT_FALLBACK_SCOPE_V1,
        "ordering": "CANONICAL_OPAQUE_IDENTITY",
        "learner": None,
        "promotion": None,
        "catalog_fallback": None,
        "configuration_tuning_fallback": None,
    }
)


def _schema_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    return str(payload["schema"]), sha256_digest(payload)


def _acquisition_decision(
    *,
    stage: AcquisitionStageV1,
    subject_ref: str,
    subject_digest: str,
    policy_ref: str,
    policy_digest: str,
    context_ref: str,
    context_digest: str,
    protocol_ref: str,
    protocol_digest: str,
    feature_schema: Mapping[str, Any],
    feature_snapshot: Mapping[str, Any],
    budget_schema: Mapping[str, Any],
    budget_snapshot_digest: str,
    eligibility_schema: Mapping[str, Any],
    eligibility_snapshot: Mapping[str, Any],
    disposition: AcquisitionDispositionV1,
    reason_codes: tuple[str, ...],
) -> AcquisitionDecisionV1:
    feature_ref, feature_digest = _schema_identity(feature_schema)
    budget_ref, budget_digest = _schema_identity(budget_schema)
    eligibility_ref, eligibility_digest = _schema_identity(eligibility_schema)
    return AcquisitionDecisionV1(
        stage=stage,
        subject_kind=(
            "OPEN_RESEARCH_SPEC"
            if stage is AcquisitionStageV1.IDEA
            else "EXECUTABLE_CAPABILITY"
        ),
        subject_ref=subject_ref,
        subject_digest=subject_digest,
        policy_ref=policy_ref,
        policy_digest=policy_digest,
        context_ref=context_ref,
        context_digest=context_digest,
        protocol_ref=protocol_ref,
        protocol_digest=protocol_digest,
        feature_schema_ref=feature_ref,
        feature_schema_digest=feature_digest,
        feature_snapshot_digest=sha256_digest(feature_snapshot),
        budget_schema_ref=budget_ref,
        budget_schema_digest=budget_digest,
        budget_snapshot_digest=budget_snapshot_digest,
        eligibility_schema_ref=eligibility_ref,
        eligibility_schema_digest=eligibility_digest,
        eligibility_snapshot_digest=sha256_digest(eligibility_snapshot),
        disposition=disposition,
        reason_codes=reason_codes,
        cross_domain_schema_reuse=False,
    )


def run_static_idea_policy(
    policy_input: IdeaPolicyInputV1,
) -> OpenMetaPolicyDecisionV1:
    """Allocate open-idea budget without learning or fixed-catalog fallback."""

    if not isinstance(policy_input, IdeaPolicyInputV1):
        raise OpenMetaContractError("static Idea policy requires IdeaPolicyInputV1")
    reasons: list[str] = []
    if policy_input.budget.ideation_slots <= 0:
        reasons.append("IDEATION_BUDGET_ZERO")
    if policy_input.budget.implementation_slots <= 0:
        reasons.append("IMPLEMENTATION_BUDGET_ZERO")
    if policy_input.budget.qualification_slots <= 0:
        reasons.append("QUALIFICATION_BUDGET_ZERO")
    if any(not candidate.high_change for candidate in policy_input.candidates):
        reasons.append("PARAMETER_OR_CONFIG_ONLY_OUT_OF_SUPPORT")
    if any(
        candidate.current_profile_expressible
        for candidate in policy_input.candidates
    ):
        reasons.append("CURRENT_PROFILE_SEARCH_BELONGS_TO_EXPERIMENT_POLICY")
    in_support = not reasons
    support_status = (
        PolicySupportStatusV1.IN_SUPPORT
        if in_support
        else PolicySupportStatusV1.OUT_OF_SUPPORT
    )
    support_reasons = (
        ("STATIC_OPEN_HIGH_CHANGE_SCOPE_MATCH",)
        if in_support
        else tuple(reasons)
    )
    selections = min(
        len(policy_input.candidates),
        policy_input.budget.implementation_slots,
        policy_input.budget.qualification_slots,
    )
    selected_refs = {
        candidate.research_spec_ref
        for candidate in policy_input.candidates[:selections]
    } if in_support else set()
    decisions = tuple(
        _acquisition_decision(
            stage=AcquisitionStageV1.IDEA,
            subject_ref=candidate.research_spec_ref,
            subject_digest=candidate.research_spec_digest,
            policy_ref=STATIC_IDEA_POLICY_REF_V1,
            policy_digest=STATIC_IDEA_POLICY_DIGEST_V1,
            context_ref=policy_input.research_context_ref,
            context_digest=policy_input.research_context_digest,
            protocol_ref=policy_input.protocol_ref,
            protocol_digest=policy_input.protocol_digest,
            feature_schema=_IDEA_FEATURE_SCHEMA,
            feature_snapshot={
                "research_spec_ref": candidate.research_spec_ref,
                "research_spec_digest": candidate.research_spec_digest,
                "direction_ref": candidate.direction_ref,
                "direction_digest": candidate.direction_digest,
                "high_change": candidate.high_change,
                "current_profile_expressible": (
                    candidate.current_profile_expressible
                ),
            },
            budget_schema=_IDEA_BUDGET_SCHEMA,
            budget_snapshot_digest=policy_input.budget.digest,
            eligibility_schema=_IDEA_ELIGIBILITY_SCHEMA,
            eligibility_snapshot={
                "in_support": in_support,
                "support_reason_codes": support_reasons,
            },
            disposition=(
                AcquisitionDispositionV1.SELECT
                if candidate.research_spec_ref in selected_refs
                else AcquisitionDispositionV1.DEFER
            ),
            reason_codes=(
                ("STATIC_HIGH_CHANGE_IDENTITY_ORDER",)
                if candidate.research_spec_ref in selected_refs
                else (
                    ("STATIC_IDEA_BUDGET_EXHAUSTED",)
                    if in_support
                    else tuple(reasons)
                )
            ),
        )
        for candidate in policy_input.candidates
    )
    allocations: list[PolicyBudgetAllocationV1] = []
    if in_support:
        directions = sorted(
            {
                (candidate.direction_ref, candidate.direction_digest)
                for candidate in policy_input.candidates
            }
        )
        for index in range(policy_input.budget.ideation_slots):
            direction_ref, _direction_digest = directions[index % len(directions)]
            allocations.append(
                PolicyBudgetAllocationV1(
                    stage=AcquisitionStageV1.IDEA,
                    target_kind="RESEARCH_DIRECTION",
                    target_ref=direction_ref,
                    units=1,
                    unit_name="IDEATION_SLOT",
                    reason_code="STATIC_DIRECTION_ROUND_ROBIN",
                )
            )
        for candidate in policy_input.candidates[:selections]:
            for unit_name in ("IMPLEMENTATION_SLOT", "QUALIFICATION_SLOT"):
                allocations.append(
                    PolicyBudgetAllocationV1(
                        stage=AcquisitionStageV1.IDEA,
                        target_kind="OPEN_RESEARCH_SPEC",
                        target_ref=candidate.research_spec_ref,
                        units=1,
                        unit_name=unit_name,
                        reason_code="STATIC_HIGH_CHANGE_IDENTITY_ORDER",
                    )
                )
    return OpenMetaPolicyDecisionV1(
        stage=AcquisitionStageV1.IDEA,
        policy_ref=STATIC_IDEA_POLICY_REF_V1,
        policy_digest=STATIC_IDEA_POLICY_DIGEST_V1,
        policy_version=STATIC_IDEA_POLICY_VERSION_V1,
        policy_mode=RESEARCH_STATIC_VNEXT,
        policy_input_ref=policy_input.record_id,
        policy_input_digest=policy_input.digest,
        support_status=support_status,
        support_reason_codes=support_reasons,
        fallback_scope=IDEA_FALLBACK_SCOPE_V1,
        fallback_reason_codes=("PRE_LEARNING_NO_PROMOTED_IDEA_POLICY",),
        budget_allocations=tuple(allocations),
        acquisition_decisions=decisions,
        explanation=(
            "Static high-change Idea allocation over opaque identities; "
            "no learner, catalog fallback, or configuration-tuning fallback."
        ),
    )


def run_static_experiment_policy(
    policy_input: ExperimentPolicyInputV1,
) -> OpenMetaPolicyDecisionV1:
    """Allocate experiment slots only inside the frozen current slate."""

    if not isinstance(policy_input, ExperimentPolicyInputV1):
        raise OpenMetaContractError(
            "static Experiment policy requires ExperimentPolicyInputV1"
        )
    slate = {
        (item.capability_ref, item.capability_digest)
        for item in policy_input.frozen_slate_capabilities
    }
    outside = tuple(
        item
        for item in policy_input.acquisition_subjects
        if (item.capability_ref, item.capability_digest) not in slate
    )
    reasons: list[str] = []
    if policy_input.budget.experiment_slots <= 0:
        reasons.append("EXPERIMENT_BUDGET_ZERO")
    if outside:
        reasons.append("SUBJECT_OUTSIDE_FROZEN_SLATE")
    in_support = not reasons
    support_status = (
        PolicySupportStatusV1.IN_SUPPORT
        if in_support
        else PolicySupportStatusV1.OUT_OF_SUPPORT
    )
    support_reasons = (
        ("STATIC_FROZEN_SLATE_SCOPE_MATCH",)
        if in_support
        else tuple(reasons)
    )
    selections = (
        min(
            len(policy_input.acquisition_subjects),
            policy_input.budget.experiment_slots,
        )
        if in_support
        else 0
    )
    selected_refs = {
        candidate.capability_ref
        for candidate in policy_input.acquisition_subjects[:selections]
    }
    decisions = tuple(
        _acquisition_decision(
            stage=AcquisitionStageV1.EXPERIMENT,
            subject_ref=candidate.capability_ref,
            subject_digest=candidate.capability_digest,
            policy_ref=STATIC_EXPERIMENT_POLICY_REF_V1,
            policy_digest=STATIC_EXPERIMENT_POLICY_DIGEST_V1,
            context_ref=policy_input.research_context_ref,
            context_digest=policy_input.research_context_digest,
            protocol_ref=policy_input.protocol_ref,
            protocol_digest=policy_input.protocol_digest,
            feature_schema=_EXPERIMENT_FEATURE_SCHEMA,
            feature_snapshot={
                "capability_ref": candidate.capability_ref,
                "capability_digest": candidate.capability_digest,
                "frozen_slate_ref": policy_input.frozen_slate_ref,
                "frozen_slate_digest": policy_input.frozen_slate_digest,
                "in_frozen_slate": (
                    candidate.capability_ref,
                    candidate.capability_digest,
                )
                in slate,
            },
            budget_schema=_EXPERIMENT_BUDGET_SCHEMA,
            budget_snapshot_digest=policy_input.budget.digest,
            eligibility_schema=_EXPERIMENT_ELIGIBILITY_SCHEMA,
            eligibility_snapshot={
                "in_support": in_support,
                "support_reason_codes": support_reasons,
            },
            disposition=(
                AcquisitionDispositionV1.SELECT
                if candidate.capability_ref in selected_refs
                else AcquisitionDispositionV1.DEFER
            ),
            reason_codes=(
                ("STATIC_FROZEN_SLATE_IDENTITY_ORDER",)
                if candidate.capability_ref in selected_refs
                else (
                    ("STATIC_EXPERIMENT_BUDGET_EXHAUSTED",)
                    if in_support
                    else tuple(reasons)
                )
            ),
        )
        for candidate in policy_input.acquisition_subjects
    )
    allocations = tuple(
        PolicyBudgetAllocationV1(
            stage=AcquisitionStageV1.EXPERIMENT,
            target_kind="EXECUTABLE_CAPABILITY",
            target_ref=candidate.capability_ref,
            units=1,
            unit_name="EXPERIMENT_SLOT",
            reason_code="STATIC_FROZEN_SLATE_IDENTITY_ORDER",
        )
        for candidate in policy_input.acquisition_subjects[:selections]
    )
    return OpenMetaPolicyDecisionV1(
        stage=AcquisitionStageV1.EXPERIMENT,
        policy_ref=STATIC_EXPERIMENT_POLICY_REF_V1,
        policy_digest=STATIC_EXPERIMENT_POLICY_DIGEST_V1,
        policy_version=STATIC_EXPERIMENT_POLICY_VERSION_V1,
        policy_mode=RESEARCH_STATIC_VNEXT,
        policy_input_ref=policy_input.record_id,
        policy_input_digest=policy_input.digest,
        support_status=support_status,
        support_reason_codes=support_reasons,
        fallback_scope=EXPERIMENT_FALLBACK_SCOPE_V1,
        fallback_reason_codes=("PRE_LEARNING_NO_PROMOTED_EXPERIMENT_POLICY",),
        budget_allocations=allocations,
        acquisition_decisions=decisions,
        explanation=(
            "Static Experiment allocation inside the frozen slate; "
            "no learner, outcome feature, or capability injection."
        ),
    )


def schedule_static_policy_activation(
    decision: OpenMetaPolicyDecisionV1,
    *,
    current_policy: PolicyIdentityV1,
) -> OpenMetaPolicyActivationV1:
    """Schedule a frozen static decision without replacing current policy."""

    if not isinstance(decision, OpenMetaPolicyDecisionV1):
        raise OpenMetaContractError("activation requires OpenMetaPolicyDecisionV1")
    if not isinstance(current_policy, PolicyIdentityV1):
        raise OpenMetaContractError("activation requires current policy identity")
    return OpenMetaPolicyActivationV1(
        stage=decision.stage,
        policy_ref=decision.policy_ref,
        policy_digest=decision.policy_digest,
        policy_version=decision.policy_version,
        policy_mode=decision.policy_mode,
        decision_ref=decision.decision_id,
        decision_digest=decision.digest,
        current_policy_ref=current_policy.ref,
        current_policy_digest=current_policy.digest_value,
        activation_boundary=(
            NEXT_FRESH_CAMPAIGN
            if decision.stage is AcquisitionStageV1.IDEA
            else NEXT_ROUND
        ),
        policy_frozen=True,
        promotion_authorized=False,
        replaces_current_policy=False,
    )


def build_open_meta_replay_record(
    *,
    decision: OpenMetaPolicyDecisionV1,
    activation: OpenMetaPolicyActivationV1,
) -> OpenMetaReplayRecordV1:
    if activation.decision_ref != decision.decision_id:
        raise OpenMetaContractError("activation does not bind the decision")
    if activation.decision_digest != decision.digest:
        raise OpenMetaContractError("activation decision digest is inconsistent")
    if activation.stage is not decision.stage:
        raise OpenMetaContractError("activation crosses the policy domain")
    return OpenMetaReplayRecordV1(
        stage=decision.stage,
        policy_ref=decision.policy_ref,
        policy_digest=decision.policy_digest,
        policy_version=decision.policy_version,
        policy_mode=decision.policy_mode,
        policy_input_ref=decision.policy_input_ref,
        policy_input_digest=decision.policy_input_digest,
        decision_ref=decision.decision_id,
        decision_digest=decision.digest,
        budget_allocations=decision.budget_allocations,
        acquisition_decisions=decision.acquisition_decisions,
        activation_ref=activation.activation_id,
        activation_digest=activation.digest,
        activation_boundary=activation.activation_boundary,
    )


def write_open_meta_replay_dataset(
    path: str | Path,
    dataset: OpenMetaReplayDatasetV1,
) -> str:
    """Write one canonical JSON replay dataset and return its byte digest."""

    if not isinstance(dataset, OpenMetaReplayDatasetV1):
        raise OpenMetaContractError("writer requires OpenMetaReplayDatasetV1")
    payload = dataset.canonical_bytes() + b"\n"
    Path(path).write_bytes(payload)
    return bytes_sha256(payload)


def _allocation_from_dict(value: Mapping[str, Any]) -> PolicyBudgetAllocationV1:
    return PolicyBudgetAllocationV1(
        stage=AcquisitionStageV1(value["stage"]),
        target_kind=str(value["target_kind"]),
        target_ref=str(value["target_ref"]),
        units=int(value["units"]),
        unit_name=str(value["unit_name"]),
        reason_code=str(value["reason_code"]),
    )


def _acquisition_from_dict(value: Mapping[str, Any]) -> AcquisitionDecisionV1:
    return AcquisitionDecisionV1(
        stage=AcquisitionStageV1(value["stage"]),
        subject_kind=str(value["subject_kind"]),
        subject_ref=str(value["subject_ref"]),
        subject_digest=str(value["subject_digest"]),
        policy_ref=str(value["policy_ref"]),
        policy_digest=str(value["policy_digest"]),
        context_ref=str(value["context_ref"]),
        context_digest=str(value["context_digest"]),
        protocol_ref=str(value["protocol_ref"]),
        protocol_digest=str(value["protocol_digest"]),
        feature_schema_ref=str(value["feature_schema_ref"]),
        feature_schema_digest=str(value["feature_schema_digest"]),
        feature_snapshot_digest=str(value["feature_snapshot_digest"]),
        budget_schema_ref=str(value["budget_schema_ref"]),
        budget_schema_digest=str(value["budget_schema_digest"]),
        budget_snapshot_digest=str(value["budget_snapshot_digest"]),
        eligibility_schema_ref=str(value["eligibility_schema_ref"]),
        eligibility_schema_digest=str(value["eligibility_schema_digest"]),
        eligibility_snapshot_digest=str(value["eligibility_snapshot_digest"]),
        disposition=AcquisitionDispositionV1(value["disposition"]),
        reason_codes=tuple(value["reason_codes"]),
        cross_domain_schema_reuse=bool(value["cross_domain_schema_reuse"]),
    )


def _replay_record_from_dict(value: Mapping[str, Any]) -> OpenMetaReplayRecordV1:
    return OpenMetaReplayRecordV1(
        stage=AcquisitionStageV1(value["stage"]),
        policy_ref=str(value["policy_ref"]),
        policy_digest=str(value["policy_digest"]),
        policy_version=str(value["policy_version"]),
        policy_mode=str(value["policy_mode"]),
        policy_input_ref=str(value["policy_input_ref"]),
        policy_input_digest=str(value["policy_input_digest"]),
        decision_ref=str(value["decision_ref"]),
        decision_digest=str(value["decision_digest"]),
        budget_allocations=tuple(
            _allocation_from_dict(item) for item in value["budget_allocations"]
        ),
        acquisition_decisions=tuple(
            _acquisition_from_dict(item)
            for item in value["acquisition_decisions"]
        ),
        activation_ref=str(value["activation_ref"]),
        activation_digest=str(value["activation_digest"]),
        activation_boundary=str(value["activation_boundary"]),
    )


def read_open_meta_replay_dataset(path: str | Path) -> OpenMetaReplayDatasetV1:
    """Read and byte-check one canonical F0 replay dataset."""

    payload = Path(path).read_bytes()
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OpenMetaContractError("replay dataset is not UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise OpenMetaContractError("replay dataset root must be an object")
    if raw.get("schema") != OpenMetaReplayDatasetV1.schema:
        raise OpenMetaContractError("replay dataset schema is invalid")
    if raw.get("contract_version") != OpenMetaReplayDatasetV1.contract_version:
        raise OpenMetaContractError("replay contract version is invalid")
    records = raw.get("records")
    if not isinstance(records, list):
        raise OpenMetaContractError("replay records must be a list")
    try:
        dataset = OpenMetaReplayDatasetV1(
            dataset_version=str(raw["dataset_version"]),
            records=tuple(_replay_record_from_dict(item) for item in records),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenMetaContractError("replay dataset content is invalid") from exc
    expected = dataset.canonical_bytes() + b"\n"
    if payload != expected:
        raise OpenMetaContractError("replay dataset bytes are not canonical")
    return dataset


__all__ = [
    "EXPERIMENT_FALLBACK_SCOPE_V1",
    "IDEA_FALLBACK_SCOPE_V1",
    "NEXT_ROUND",
    "RESEARCH_STATIC_VNEXT",
    "STATIC_EXPERIMENT_POLICY_DIGEST_V1",
    "STATIC_EXPERIMENT_POLICY_REF_V1",
    "STATIC_EXPERIMENT_POLICY_VERSION_V1",
    "STATIC_IDEA_POLICY_DIGEST_V1",
    "STATIC_IDEA_POLICY_REF_V1",
    "STATIC_IDEA_POLICY_VERSION_V1",
    "CapabilityIdentityV1",
    "ExperimentBudgetV1",
    "ExperimentPolicyInputV1",
    "IdeaBudgetV1",
    "IdeaCandidateIdentityV1",
    "IdeaPolicyInputV1",
    "OpenMetaContractError",
    "OpenMetaPolicyActivationV1",
    "OpenMetaPolicyDecisionV1",
    "OpenMetaReplayDatasetV1",
    "OpenMetaReplayRecordV1",
    "PolicyBudgetAllocationV1",
    "PolicyIdentityV1",
    "PolicySupportStatusV1",
    "TypedEpisodePolicySummaryV1",
    "build_open_meta_replay_record",
    "read_open_meta_replay_dataset",
    "run_static_experiment_policy",
    "run_static_idea_policy",
    "schedule_static_policy_activation",
    "write_open_meta_replay_dataset",
]
