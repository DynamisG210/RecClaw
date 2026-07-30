"""Profile- and slate-bound adapter for the existing fixed Search Lane.

The adapter is deliberately upstream of the unchanged Router and downstream
of the accepted A0/BC contracts.  It has no mutable campaign state and does
not materialize, implement, qualify, run, or evaluate a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, Sequence

from recclaw_core.mechanism_space.canonical import deep_thaw

from .campaign_runtime import (
    CampaignRuntimeError,
    bl_icf_executable_profile_v2,
    executable_mechanisms,
    execution_recipe_for_program,
)
from .canonical import (
    canonical_json_bytes,
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from .capability_admission import VersionedCapabilityRegistry
from .compilation_cache import compile_campaign_program as compile_program
from .open_spec import (
    frozen_search_resolver_environment,
    project_candidate_proposal_v4,
    project_open_producer_draft,
    resolve_capability,
)
from .research_capability import StrongStaticRouterV1
from .research_contracts import CandidateProposalV4, RouteTraceV1
from .vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    AcquisitionDecisionV1,
    AcquisitionDispositionV1,
    AcquisitionStageV1,
    CapabilityResolutionResultV1,
    CapabilityResolutionV1,
    ExecutableProfileVNext,
    OpenResearchSpecV1,
)


class SearchAdapterError(ValueError):
    """Raised when a Search candidate crosses an E0 boundary incorrectly."""


class SearchProfileEntryOriginV1(str, Enum):
    FIXED_66 = "FIXED_66"
    QUALIFIED_REGISTRY = "QUALIFIED_REGISTRY"


class SearchProfileActivationV1(str, Enum):
    CURRENT_FROZEN_CAMPAIGN = "CURRENT_FROZEN_CAMPAIGN"
    NEXT_FRESH_CAMPAIGN = "NEXT_FRESH_CAMPAIGN"


class IdeaRouteV1(str, Enum):
    SEARCH = "SEARCH"
    INNOVATION = "INNOVATION"
    BLOCKED = "BLOCKED"


_FIXED_ENTRY_ACTIVATION = "CURRENT_AND_FUTURE_FRESH_CAMPAIGNS"

_IDEA_POLICY = {
    "policy": "E0_STATIC_IDEA_ACQUISITION_V1",
    "search": ("SEARCH_READY",),
    "innovation": ("INNOVATION_REQUIRED",),
    "defer": ("DEFERRED_PROTOCOL_CHANGE",),
    "reject": ("UNSUPPORTED", "INVALID_SPEC"),
}
_IDEA_FEATURE_SCHEMA = {
    "schema": "recclaw.research-line.vnext.idea-features.e0.v1",
    "fields": (
        "producer_role",
        "current_profile_expressibility_claim",
        "capability_diff",
    ),
}
_IDEA_BUDGET_SCHEMA = {
    "schema": "recclaw.research-line.vnext.idea-budget.e0.v1",
    "fields": ("required_budget",),
}
_IDEA_ELIGIBILITY_SCHEMA = {
    "schema": "recclaw.research-line.vnext.idea-eligibility.e0.v1",
    "fields": (
        "resolution",
        "protocol_compatible",
        "dependency_compatible",
        "budget_compatible",
    ),
}
_EXPERIMENT_FEATURE_SCHEMA = {
    "schema": "recclaw.research-line.vnext.experiment-features.e0.v1",
    "fields": ("utility_features", "feature_evidence"),
}
_EXPERIMENT_BUDGET_SCHEMA = {
    "schema": "recclaw.research-line.vnext.experiment-budget.e0.v1",
    "fields": ("frozen_budget_snapshot",),
}
_EXPERIMENT_ELIGIBILITY_SCHEMA = {
    "schema": "recclaw.research-line.vnext.experiment-eligibility.e0.v1",
    "fields": (
        "active_profile_ref",
        "active_profile_digest",
        "frozen_slate_ref",
        "router_hard_gate",
    ),
}


def _schema_identity(value: Mapping[str, Any]) -> tuple[str, str]:
    return str(value["schema"]), sha256_digest(value)


def _normalized_string(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise SearchAdapterError(
            f"{field_name} must be a normalized non-empty string"
        )
    return value


def _normalized_entrypoint(value: str) -> str:
    normalized = _normalized_string(
        value,
        field_name="executable_entrypoint",
    )
    module_name, separator, attribute = normalized.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute
        or any(character.isspace() for character in normalized)
    ):
        raise SearchAdapterError(
            "executable_entrypoint must use module.path:Attribute"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class SearchExecutableEntryV1:
    """One executable capability exposed to an active Search campaign."""

    capability_ref: str
    capability_digest: str
    executable_entrypoint: str
    semantic_identity_ref: str
    semantic_identity_digest: str
    origin: SearchProfileEntryOriginV1
    qualification_receipt_ref: str | None
    qualification_receipt_digest: str | None
    activation_boundary: str

    schema: ClassVar[str] = (
        "recclaw.research-line.vnext.search-executable-entry.e0.v1"
    )

    def __post_init__(self) -> None:
        for field_name in ("capability_ref", "semantic_identity_ref"):
            _normalized_string(
                getattr(self, field_name),
                field_name=field_name,
            )
        for field_name in (
            "capability_digest",
            "semantic_identity_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        object.__setattr__(
            self,
            "executable_entrypoint",
            _normalized_entrypoint(self.executable_entrypoint),
        )
        if not isinstance(self.origin, SearchProfileEntryOriginV1):
            raise SearchAdapterError("entry origin is outside the E0 domain")
        receipt_paired = (
            self.qualification_receipt_ref is not None
            and self.qualification_receipt_digest is not None
        )
        if (self.qualification_receipt_ref is None) != (
            self.qualification_receipt_digest is None
        ):
            raise SearchAdapterError(
                "qualification receipt ref and digest must be paired"
            )
        if self.origin is SearchProfileEntryOriginV1.FIXED_66:
            if receipt_paired or self.activation_boundary != _FIXED_ENTRY_ACTIVATION:
                raise SearchAdapterError(
                    "fixed entries cannot claim qualification or next-only activation"
                )
        else:
            if not receipt_paired or self.activation_boundary != NEXT_FRESH_CAMPAIGN:
                raise SearchAdapterError(
                    "registry entries require qualification and next-fresh activation"
                )
            _normalized_string(
                str(self.qualification_receipt_ref),
                field_name="qualification_receipt_ref",
            )
            object.__setattr__(
                self,
                "qualification_receipt_digest",
                validate_sha256(
                    str(self.qualification_receipt_digest),
                    field_name="qualification_receipt_digest",
                ),
            )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "capability_ref": self.capability_ref,
                "capability_digest": self.capability_digest,
                "executable_entrypoint": self.executable_entrypoint,
                "semantic_identity_ref": self.semantic_identity_ref,
                "semantic_identity_digest": self.semantic_identity_digest,
                "origin": self.origin,
                "qualification_receipt_ref": self.qualification_receipt_ref,
                "qualification_receipt_digest": (
                    self.qualification_receipt_digest
                ),
                "activation_boundary": self.activation_boundary,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class SearchExecutableProfileV1:
    """Immutable consumer view of the profile active in one campaign."""

    campaign_id: str
    profile_ref: str
    profile_digest: str
    protocol_ref: str
    protocol_digest: str
    activation: SearchProfileActivationV1
    predecessor_campaign_id: str | None
    predecessor_profile_ref: str | None
    predecessor_profile_digest: str | None
    entries: tuple[SearchExecutableEntryV1, ...]

    schema: ClassVar[str] = (
        "recclaw.research-line.vnext.search-executable-profile.e0.v1"
    )
    identity_namespace: ClassVar[str] = (
        "recclaw-search-executable-profile-e0-v1"
    )

    def __post_init__(self) -> None:
        for field_name in ("campaign_id", "profile_ref", "protocol_ref"):
            _normalized_string(
                getattr(self, field_name),
                field_name=field_name,
            )
        for field_name in ("profile_digest", "protocol_digest"):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        if not isinstance(self.activation, SearchProfileActivationV1):
            raise SearchAdapterError(
                "profile activation is outside the E0 domain"
            )
        predecessor_values = (
            self.predecessor_campaign_id,
            self.predecessor_profile_ref,
            self.predecessor_profile_digest,
        )
        if self.activation is SearchProfileActivationV1.CURRENT_FROZEN_CAMPAIGN:
            if any(value is not None for value in predecessor_values):
                raise SearchAdapterError(
                    "initial frozen profile cannot declare a predecessor campaign"
                )
        else:
            if any(value is None for value in predecessor_values):
                raise SearchAdapterError(
                    "fresh activation requires predecessor campaign and profile"
                )
            if self.predecessor_campaign_id == self.campaign_id:
                raise SearchAdapterError(
                    "next profile requires a distinct fresh campaign"
                )
            _normalized_string(
                str(self.predecessor_campaign_id),
                field_name="predecessor_campaign_id",
            )
            _normalized_string(
                str(self.predecessor_profile_ref),
                field_name="predecessor_profile_ref",
            )
            object.__setattr__(
                self,
                "predecessor_profile_digest",
                validate_sha256(
                    str(self.predecessor_profile_digest),
                    field_name="predecessor_profile_digest",
                ),
            )
        if not isinstance(self.entries, (tuple, list)) or not self.entries:
            raise SearchAdapterError(
                "active Search profile requires executable entries"
            )
        by_ref: dict[str, SearchExecutableEntryV1] = {}
        for entry in self.entries:
            if not isinstance(entry, SearchExecutableEntryV1):
                raise SearchAdapterError(
                    "profile entries must be SearchExecutableEntryV1"
                )
            existing = by_ref.get(entry.capability_ref)
            if existing is not None:
                raise SearchAdapterError(
                    "active capability references must be unique"
                )
            by_ref[entry.capability_ref] = entry
        object.__setattr__(
            self,
            "entries",
            tuple(sorted(by_ref.values(), key=lambda item: item.capability_ref)),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "profile_ref": self.profile_ref,
                "profile_digest": self.profile_digest,
                "protocol_ref": self.protocol_ref,
                "protocol_digest": self.protocol_digest,
                "activation": self.activation,
                "predecessor_campaign_id": self.predecessor_campaign_id,
                "predecessor_profile_ref": self.predecessor_profile_ref,
                "predecessor_profile_digest": (
                    self.predecessor_profile_digest
                ),
                "entries": tuple(
                    entry.canonical_dict() for entry in self.entries
                ),
            }
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_dict())

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())

    @property
    def adapter_profile_id(self) -> str:
        return content_id(self.identity_namespace, self.canonical_dict())

    def entry(self, capability_ref: str) -> SearchExecutableEntryV1:
        matches = tuple(
            entry
            for entry in self.entries
            if entry.capability_ref == capability_ref
        )
        if len(matches) != 1:
            raise SearchAdapterError(
                "capability is not in the active executable profile"
            )
        return matches[0]


def adapt_current_search_profile(
    *, campaign_id: str
) -> SearchExecutableProfileV1:
    """Mechanically expose the exact legacy 66 as the current profile."""

    source_profile = bl_icf_executable_profile_v2()
    environment = frozen_search_resolver_environment()
    by_semantics = {
        str(entry["semantics_digest"]): entry
        for entry in environment["current_capabilities"]
    }
    entries = []
    for mechanism in executable_mechanisms():
        capability = by_semantics.get(
            mechanism.mechanism_semantics_digest
        )
        if capability is None:
            raise SearchAdapterError(
                "resolver environment omitted a fixed executable mechanism"
            )
        entries.append(
            SearchExecutableEntryV1(
                capability_ref=str(capability["capability_ref"]),
                capability_digest=str(capability["capability_digest"]),
                executable_entrypoint=mechanism.entrypoint,
                semantic_identity_ref=(
                    f"bl-icf-mechanism:{mechanism.mechanism_id}"
                ),
                semantic_identity_digest=(
                    mechanism.mechanism_semantics_digest
                ),
                origin=SearchProfileEntryOriginV1.FIXED_66,
                qualification_receipt_ref=None,
                qualification_receipt_digest=None,
                activation_boundary=_FIXED_ENTRY_ACTIVATION,
            )
        )
    if len(entries) != 66:
        raise SearchAdapterError(
            "current Search adapter requires the exact fixed 66"
        )
    return SearchExecutableProfileV1(
        campaign_id=campaign_id,
        profile_ref=str(source_profile["profile_id"]),
        profile_digest=str(source_profile["profile_digest"]),
        protocol_ref=str(environment["protocol_ref"]),
        protocol_digest=str(environment["protocol_digest"]),
        activation=SearchProfileActivationV1.CURRENT_FROZEN_CAMPAIGN,
        predecessor_campaign_id=None,
        predecessor_profile_ref=None,
        predecessor_profile_digest=None,
        entries=tuple(entries),
    )


def predecessor_executable_entries(
    profile: SearchExecutableProfileV1,
) -> tuple[tuple[str, str, str], ...]:
    """Project an active Search profile into the accepted BC builder shape."""

    if not isinstance(profile, SearchExecutableProfileV1):
        raise SearchAdapterError(
            "profile must be SearchExecutableProfileV1"
        )
    return tuple(
        (
            entry.capability_ref,
            entry.capability_digest,
            entry.executable_entrypoint,
        )
        for entry in profile.entries
    )


def activate_next_fresh_search_profile(
    *,
    predecessor: SearchExecutableProfileV1,
    next_profile: ExecutableProfileVNext,
    registry: VersionedCapabilityRegistry,
    fresh_campaign_id: str,
) -> SearchExecutableProfileV1:
    """Activate an accepted successor only for a distinct fresh campaign."""

    if not isinstance(predecessor, SearchExecutableProfileV1):
        raise SearchAdapterError(
            "predecessor must be SearchExecutableProfileV1"
        )
    if not isinstance(next_profile, ExecutableProfileVNext):
        raise SearchAdapterError(
            "next_profile must be ExecutableProfileVNext"
        )
    if not isinstance(registry, VersionedCapabilityRegistry):
        raise SearchAdapterError(
            "registry must be VersionedCapabilityRegistry"
        )
    _normalized_string(fresh_campaign_id, field_name="fresh_campaign_id")
    if fresh_campaign_id == predecessor.campaign_id:
        raise SearchAdapterError(
            "next profile cannot activate in the current campaign"
        )
    if (
        next_profile.current_campaign_eligible
        or next_profile.activation_boundary != NEXT_FRESH_CAMPAIGN
    ):
        raise SearchAdapterError(
            "successor profile lacks the next-fresh activation boundary"
        )
    if (
        next_profile.predecessor_profile_ref != predecessor.profile_ref
        or next_profile.predecessor_profile_digest
        != predecessor.profile_digest
    ):
        raise SearchAdapterError("successor predecessor identity drift")
    if (
        next_profile.registry_ref != registry.registry_id
        or next_profile.registry_digest != registry.digest
    ):
        raise SearchAdapterError("successor registry identity drift")
    if (
        next_profile.protocol_ref != predecessor.protocol_ref
        or next_profile.protocol_digest != predecessor.protocol_digest
        or registry.protocol_ref != predecessor.protocol_ref
        or registry.protocol_digest != predecessor.protocol_digest
    ):
        raise SearchAdapterError("successor protocol identity drift")

    predecessor_entries = {
        (
            entry.capability_ref,
            entry.capability_digest,
            entry.executable_entrypoint,
        )
        for entry in predecessor.entries
    }
    registry_entries = {
        (
            capability.capability_id,
            capability.digest,
            capability.executable_entrypoint,
        )
        for capability in registry.capabilities
    }
    actual_entries = set(next_profile.executable_entries)
    if actual_entries != predecessor_entries | registry_entries:
        raise SearchAdapterError(
            "successor executable entries do not equal predecessor plus registry"
        )

    entries = list(predecessor.entries)
    for capability in registry.capabilities:
        entries.append(
            SearchExecutableEntryV1(
                capability_ref=capability.capability_id,
                capability_digest=capability.digest,
                executable_entrypoint=capability.executable_entrypoint,
                semantic_identity_ref=capability.semantic_identity_ref,
                semantic_identity_digest=(
                    capability.semantic_identity_digest
                ),
                origin=SearchProfileEntryOriginV1.QUALIFIED_REGISTRY,
                qualification_receipt_ref=(
                    capability.qualification_receipt_ref
                ),
                qualification_receipt_digest=(
                    capability.qualification_receipt_digest
                ),
                activation_boundary=capability.activation_boundary,
            )
        )
    return SearchExecutableProfileV1(
        campaign_id=fresh_campaign_id,
        profile_ref=next_profile.profile_id,
        profile_digest=next_profile.digest,
        protocol_ref=next_profile.protocol_ref,
        protocol_digest=next_profile.protocol_digest,
        activation=SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN,
        predecessor_campaign_id=predecessor.campaign_id,
        predecessor_profile_ref=predecessor.profile_ref,
        predecessor_profile_digest=predecessor.profile_digest,
        entries=tuple(entries),
    )


@dataclass(frozen=True, slots=True)
class IdeaAcquisitionResultV1:
    spec: OpenResearchSpecV1
    resolution: CapabilityResolutionV1
    decision: AcquisitionDecisionV1
    route: IdeaRouteV1
    search_proposal: CandidateProposalV4 | None
    budget_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.route, IdeaRouteV1):
            raise SearchAdapterError("Idea route is outside the E0 domain")
        if (
            self.route is IdeaRouteV1.SEARCH
            and self.resolution.resolution
            is not CapabilityResolutionResultV1.SEARCH_READY
        ):
            raise SearchAdapterError(
                "only SEARCH_READY may enter the Search route"
            )
        if (
            self.route is IdeaRouteV1.SEARCH
            and self.search_proposal is None
        ):
            raise SearchAdapterError(
                "Search acquisition requires the existing Router proposal"
            )
        if (
            self.route is not IdeaRouteV1.SEARCH
            and self.search_proposal is not None
        ):
            raise SearchAdapterError(
                "non-Search Idea routes cannot carry a Router proposal"
            )
        object.__setattr__(
            self,
            "budget_snapshot",
            canonical_value(dict(self.budget_snapshot)),
        )


def _idea_result(
    *,
    spec: OpenResearchSpecV1,
    resolution: CapabilityResolutionV1,
    resolution_facts: Mapping[str, Any],
    proposal: CandidateProposalV4 | None,
) -> IdeaAcquisitionResultV1:
    result = resolution.resolution
    if result is CapabilityResolutionResultV1.SEARCH_READY:
        route = IdeaRouteV1.SEARCH
        disposition = AcquisitionDispositionV1.SELECT
    elif result is CapabilityResolutionResultV1.INNOVATION_REQUIRED:
        route = IdeaRouteV1.INNOVATION
        disposition = AcquisitionDispositionV1.SELECT
    elif result is CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE:
        route = IdeaRouteV1.BLOCKED
        disposition = AcquisitionDispositionV1.DEFER
    else:
        route = IdeaRouteV1.BLOCKED
        disposition = AcquisitionDispositionV1.REJECT

    feature_ref, feature_digest = _schema_identity(_IDEA_FEATURE_SCHEMA)
    budget_ref, budget_digest = _schema_identity(_IDEA_BUDGET_SCHEMA)
    eligibility_ref, eligibility_digest = _schema_identity(
        _IDEA_ELIGIBILITY_SCHEMA
    )
    budget_snapshot = canonical_value(
        dict(resolution_facts.get("required_budget", {}))
    )
    decision = AcquisitionDecisionV1(
        stage=AcquisitionStageV1.IDEA,
        subject_kind="OPEN_RESEARCH_SPEC",
        subject_ref=spec.spec_id,
        subject_digest=spec.digest,
        policy_ref=str(_IDEA_POLICY["policy"]),
        policy_digest=sha256_digest(_IDEA_POLICY),
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        feature_schema_ref=feature_ref,
        feature_schema_digest=feature_digest,
        feature_snapshot_digest=sha256_digest(
            {
                "producer_role": spec.producer_role,
                "current_profile_expressibility_claim": (
                    spec.current_profile_expressibility_claim
                ),
                "capability_diff": resolution.capability_diff,
            }
        ),
        budget_schema_ref=budget_ref,
        budget_schema_digest=budget_digest,
        budget_snapshot_digest=sha256_digest(budget_snapshot),
        eligibility_schema_ref=eligibility_ref,
        eligibility_schema_digest=eligibility_digest,
        eligibility_snapshot_digest=sha256_digest(
            {
                "resolution": result,
                "protocol_compatible": resolution.protocol_compatible,
                "dependency_compatible": resolution.dependency_compatible,
                "budget_compatible": resolution.budget_compatible,
            }
        ),
        disposition=disposition,
        reason_codes=resolution.reason_codes,
        cross_domain_schema_reuse=False,
    )
    return IdeaAcquisitionResultV1(
        spec=spec,
        resolution=resolution,
        decision=decision,
        route=route,
        search_proposal=(
            proposal if route is IdeaRouteV1.SEARCH else None
        ),
        budget_snapshot=budget_snapshot,
    )


def acquire_candidate_idea(
    proposal: CandidateProposalV4,
    *,
    bindings: Mapping[str, Any],
    environment: Mapping[str, Any],
    required_dependencies: Sequence[str] = (),
    required_budget: Mapping[str, int] | None = None,
) -> IdeaAcquisitionResultV1:
    """Acquire one existing Producer proposal for Search or block it."""

    spec, facts = project_candidate_proposal_v4(
        proposal,
        bindings=bindings,
        required_dependencies=required_dependencies,
        required_budget=required_budget,
    )
    resolution = resolve_capability(
        spec,
        resolution_facts=facts,
        environment=environment,
    )
    return _idea_result(
        spec=spec,
        resolution=resolution,
        resolution_facts=facts,
        proposal=proposal,
    )


def acquire_open_idea(
    draft: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> IdeaAcquisitionResultV1:
    """Acquire an open Producer draft for Innovation, defer, or rejection."""

    spec, facts = project_open_producer_draft(draft, bindings=bindings)
    resolution = resolve_capability(
        spec,
        resolution_facts=facts,
        environment=environment,
    )
    return _idea_result(
        spec=spec,
        resolution=resolution,
        resolution_facts=facts,
        proposal=None,
    )


@dataclass(frozen=True, slots=True)
class SearchCandidateBindingV1:
    """Search-owned proposal bound to one active executable entry."""

    proposal: CandidateProposalV4
    capability_ref: str
    capability_digest: str
    executable_entrypoint: str
    entry_origin: SearchProfileEntryOriginV1
    mechanism_semantics_digest: str

    schema: ClassVar[str] = (
        "recclaw.research-line.vnext.search-candidate-binding.e0.v1"
    )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "proposal": self.proposal.to_dict(),
                "capability_ref": self.capability_ref,
                "capability_digest": self.capability_digest,
                "executable_entrypoint": self.executable_entrypoint,
                "entry_origin": self.entry_origin,
                "mechanism_semantics_digest": (
                    self.mechanism_semantics_digest
                ),
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())


def bind_search_candidate(
    *,
    profile: SearchExecutableProfileV1,
    proposal: CandidateProposalV4,
    capability_ref: str,
) -> SearchCandidateBindingV1:
    """Bind only an active profile capability to the unchanged Router input."""

    if not isinstance(profile, SearchExecutableProfileV1):
        raise SearchAdapterError(
            "profile must be SearchExecutableProfileV1"
        )
    if not isinstance(proposal, CandidateProposalV4):
        raise SearchAdapterError(
            "Search consumer requires CandidateProposalV4"
        )
    entry = profile.entry(capability_ref)
    report = compile_program(deep_thaw(proposal.mechanism_program))
    if not report.is_valid or report.mechanism_semantics_digest is None:
        raise SearchAdapterError(
            "Search proposal mechanism program is not BL-ICF valid"
        )
    if entry.origin is SearchProfileEntryOriginV1.FIXED_66:
        try:
            recipe = execution_recipe_for_program(
                deep_thaw(proposal.mechanism_program)
            )
        except CampaignRuntimeError as error:
            raise SearchAdapterError(
                "fixed-profile proposal is outside the exact 66"
            ) from error
        if (
            recipe["mechanism_id"] != proposal.mechanism_id
            or recipe["mechanism_semantics_digest"]
            != entry.semantic_identity_digest
            or recipe["entrypoint"] != entry.executable_entrypoint
        ):
            raise SearchAdapterError(
                "fixed-profile proposal identity drift"
            )
    else:
        if (
            profile.activation
            is not SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN
        ):
            raise SearchAdapterError(
                "qualified capability is unavailable before fresh activation"
            )
        try:
            execution_recipe_for_program(
                deep_thaw(proposal.mechanism_program)
            )
        except CampaignRuntimeError:
            pass
        else:
            raise SearchAdapterError(
                "qualified capability cannot silently fall back to the fixed 66"
            )
    return SearchCandidateBindingV1(
        proposal=proposal,
        capability_ref=entry.capability_ref,
        capability_digest=entry.capability_digest,
        executable_entrypoint=entry.executable_entrypoint,
        entry_origin=entry.origin,
        mechanism_semantics_digest=str(
            report.mechanism_semantics_digest
        ),
    )


@dataclass(frozen=True, slots=True)
class FrozenExperimentSlateV1:
    campaign_id: str
    profile_ref: str
    profile_digest: str
    bindings: tuple[SearchCandidateBindingV1, ...]
    budget_schema_ref: str
    budget_schema_digest: str
    budget_snapshot: Mapping[str, Any]

    schema: ClassVar[str] = (
        "recclaw.research-line.vnext.frozen-experiment-slate.e0.v1"
    )
    identity_namespace: ClassVar[str] = (
        "recclaw-frozen-experiment-slate-e0-v1"
    )

    def __post_init__(self) -> None:
        for field_name in (
            "campaign_id",
            "profile_ref",
            "budget_schema_ref",
        ):
            _normalized_string(
                getattr(self, field_name),
                field_name=field_name,
            )
        for field_name in ("profile_digest", "budget_schema_digest"):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        if not isinstance(self.bindings, (tuple, list)) or not self.bindings:
            raise SearchAdapterError(
                "frozen experiment slate requires candidate bindings"
            )
        if len(
            {binding.proposal.candidate_id for binding in self.bindings}
        ) != len(self.bindings):
            raise SearchAdapterError(
                "frozen slate candidate IDs must be unique"
            )
        object.__setattr__(
            self,
            "budget_snapshot",
            canonical_value(dict(self.budget_snapshot)),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "profile_ref": self.profile_ref,
                "profile_digest": self.profile_digest,
                "bindings": tuple(
                    binding.canonical_dict() for binding in self.bindings
                ),
                "budget_schema_ref": self.budget_schema_ref,
                "budget_schema_digest": self.budget_schema_digest,
                "budget_snapshot": self.budget_snapshot,
            }
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_dict())

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())

    @property
    def slate_id(self) -> str:
        return content_id(self.identity_namespace, self.canonical_dict())


def freeze_experiment_slate(
    *,
    profile: SearchExecutableProfileV1,
    bindings: Sequence[SearchCandidateBindingV1],
    budget_snapshot: Mapping[str, Any],
) -> FrozenExperimentSlateV1:
    """Freeze the only proposal sequence that the Router may consume."""

    if not isinstance(profile, SearchExecutableProfileV1):
        raise SearchAdapterError(
            "profile must be SearchExecutableProfileV1"
        )
    normalized_bindings = tuple(bindings)
    for binding in normalized_bindings:
        if not isinstance(binding, SearchCandidateBindingV1):
            raise SearchAdapterError(
                "slate entries must be SearchCandidateBindingV1"
            )
        rebuilt = bind_search_candidate(
            profile=profile,
            proposal=binding.proposal,
            capability_ref=binding.capability_ref,
        )
        if rebuilt != binding:
            raise SearchAdapterError(
                "candidate binding identity drift"
            )
    budget_ref, budget_digest = _schema_identity(
        _EXPERIMENT_BUDGET_SCHEMA
    )
    return FrozenExperimentSlateV1(
        campaign_id=profile.campaign_id,
        profile_ref=profile.profile_ref,
        profile_digest=profile.profile_digest,
        bindings=normalized_bindings,
        budget_schema_ref=budget_ref,
        budget_schema_digest=budget_digest,
        budget_snapshot=budget_snapshot,
    )


@dataclass(frozen=True, slots=True)
class ExperimentAcquisitionResultV1:
    slate_ref: str
    slate_digest: str
    route_trace: RouteTraceV1
    decisions: tuple[AcquisitionDecisionV1, ...]
    selected_binding: SearchCandidateBindingV1 | None


def route_frozen_experiment_slate(
    *,
    profile: SearchExecutableProfileV1,
    slate: FrozenExperimentSlateV1,
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None = None,
) -> ExperimentAcquisitionResultV1:
    """Invoke the unchanged current Router with only the frozen slate."""

    if not isinstance(profile, SearchExecutableProfileV1):
        raise SearchAdapterError(
            "profile must be SearchExecutableProfileV1"
        )
    if not isinstance(slate, FrozenExperimentSlateV1):
        raise SearchAdapterError(
            "slate must be FrozenExperimentSlateV1"
        )
    if not isinstance(router, StrongStaticRouterV1):
        raise SearchAdapterError(
            "router must be the existing StrongStaticRouterV1"
        )
    if (
        slate.campaign_id != profile.campaign_id
        or slate.profile_ref != profile.profile_ref
        or slate.profile_digest != profile.profile_digest
    ):
        raise SearchAdapterError(
            "experiment slate is not bound to the active campaign profile"
        )
    for binding in slate.bindings:
        if (
            bind_search_candidate(
                profile=profile,
                proposal=binding.proposal,
                capability_ref=binding.capability_ref,
            )
            != binding
        ):
            raise SearchAdapterError(
                "experiment slate binding drift"
            )

    trace = router.route(
        tuple(binding.proposal for binding in slate.bindings),
        policy_projection=policy_projection,
    )
    hard_gate_by_candidate = {
        decision.candidate_id: decision
        for decision in trace.decisions
    }
    feature_ref, feature_digest = _schema_identity(
        _EXPERIMENT_FEATURE_SCHEMA
    )
    eligibility_ref, eligibility_digest = _schema_identity(
        _EXPERIMENT_ELIGIBILITY_SCHEMA
    )
    decisions = []
    for binding in slate.bindings:
        hard_gate = hard_gate_by_candidate[binding.proposal.candidate_id]
        decisions.append(
            AcquisitionDecisionV1(
                stage=AcquisitionStageV1.EXPERIMENT,
                subject_kind="EXECUTABLE_CAPABILITY",
                subject_ref=binding.capability_ref,
                subject_digest=binding.capability_digest,
                policy_ref=(
                    "recclaw.search-router:StrongStaticRouterV1"
                ),
                policy_digest=trace.policy_digest,
                context_ref=slate.slate_id,
                context_digest=slate.digest,
                protocol_ref=profile.protocol_ref,
                protocol_digest=profile.protocol_digest,
                feature_schema_ref=feature_ref,
                feature_schema_digest=feature_digest,
                feature_snapshot_digest=sha256_digest(
                    {
                        "utility_features": (
                            binding.proposal.utility_features
                        ),
                        "feature_evidence": (
                            binding.proposal.feature_evidence
                        ),
                    }
                ),
                budget_schema_ref=slate.budget_schema_ref,
                budget_schema_digest=slate.budget_schema_digest,
                budget_snapshot_digest=sha256_digest(
                    slate.budget_snapshot
                ),
                eligibility_schema_ref=eligibility_ref,
                eligibility_schema_digest=eligibility_digest,
                eligibility_snapshot_digest=sha256_digest(
                    {
                        "active_profile_ref": profile.profile_ref,
                        "active_profile_digest": profile.profile_digest,
                        "frozen_slate_ref": slate.slate_id,
                        "allowed": hard_gate.allowed,
                        "reason": hard_gate.reason,
                    }
                ),
                disposition=(
                    AcquisitionDispositionV1.SELECT
                    if binding.proposal.candidate_id
                    == trace.selected_candidate_id
                    else AcquisitionDispositionV1.DEFER
                    if hard_gate.allowed
                    else AcquisitionDispositionV1.REJECT
                ),
                reason_codes=(
                    f"ROUTER_{hard_gate.reason.value}",
                    f"ORIGIN_{binding.entry_origin.value}",
                    (
                        "ROUTER_SELECTED"
                        if binding.proposal.candidate_id
                        == trace.selected_candidate_id
                        else "ROUTER_ALLOWED_NOT_SELECTED"
                        if hard_gate.allowed
                        else "ROUTER_REJECTED"
                    ),
                ),
                cross_domain_schema_reuse=False,
            )
        )
    selected = next(
        (
            binding
            for binding in slate.bindings
            if binding.proposal.candidate_id
            == trace.selected_candidate_id
        ),
        None,
    )
    return ExperimentAcquisitionResultV1(
        slate_ref=slate.slate_id,
        slate_digest=slate.digest,
        route_trace=trace,
        decisions=tuple(decisions),
        selected_binding=selected,
    )


__all__ = [
    "ExperimentAcquisitionResultV1",
    "FrozenExperimentSlateV1",
    "IdeaAcquisitionResultV1",
    "IdeaRouteV1",
    "SearchAdapterError",
    "SearchCandidateBindingV1",
    "SearchExecutableEntryV1",
    "SearchExecutableProfileV1",
    "SearchProfileActivationV1",
    "SearchProfileEntryOriginV1",
    "acquire_candidate_idea",
    "acquire_open_idea",
    "activate_next_fresh_search_profile",
    "adapt_current_search_profile",
    "bind_search_candidate",
    "freeze_experiment_slate",
    "predecessor_executable_entries",
    "route_frozen_experiment_slate",
]
