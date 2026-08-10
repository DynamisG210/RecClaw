"""Profile- and slate-bound adapter for the existing fixed Search Lane.

The adapter is deliberately upstream of the unchanged Router and downstream
of the accepted A0/BC contracts.  It has no mutable campaign state and does
not materialize, implement, qualify, run, or evaluate a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from .research_contracts import (
    CandidateProposalV4,
    RouteTraceV1,
    RouterHardGateDecisionV1,
    RouterHardGateReasonV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
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


_PORTFOLIO_PUBLIC_NAMES = frozenset(
    {
        "AttemptFailureScopeV2",
        "ParentValidationStateV2",
        "PortfolioAttemptFailureV2",
        "PortfolioCandidateV2",
        "PortfolioControlStateV2",
        "PortfolioEligibilityV2",
        "ResourceAdmissionStateV2",
        "rank_candidate_portfolio_v2",
    }
)


def _load_portfolio_symbols() -> None:
    """Load the opt-in portfolio path after this compatibility module is ready.

    ``recclaw_core.research_line`` exports ``runtime``, and ``runtime`` imports
    this adapter.  Eagerly importing the new module here would therefore make
    the package initialization path circular.  The portfolio path is opt-in,
    so loading it on first use preserves the legacy import graph and keeps the
    new controls isolated from the unchanged Research Line runtime.
    """

    if "PortfolioCandidateV2" in globals():
        return
    from recclaw_core.research_line import portfolio as portfolio_module

    for name in _PORTFOLIO_PUBLIC_NAMES:
        globals()[name] = getattr(portfolio_module, name)


def __getattr__(name: str) -> Any:
    if name in _PORTFOLIO_PUBLIC_NAMES:
        _load_portfolio_symbols()
        return globals()[name]
    raise AttributeError(name)


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
_NEXT_DEVELOPMENT_SEED = "NEXT_DEVELOPMENT_SEED"


def _pending_new_seed_matches(
    pending_task: Mapping[str, Any], observation_seed: str
) -> bool:
    required = pending_task.get("required_seed_or_control")
    if required == observation_seed:
        return True
    evidence_present = pending_task.get("evidence_present", ())
    if isinstance(evidence_present, str):
        evidence_present = (evidence_present,)
    return (
        required == _NEXT_DEVELOPMENT_SEED
        and isinstance(evidence_present, (tuple, list))
        and observation_seed not in evidence_present
    )
_REPEAT_UTILITY_PENALTY = 0.25
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


def open_spec_realization_identity(
    spec: OpenResearchSpecV1,
    *,
    candidate_package_ref: str,
    candidate_package_digest: str,
    candidate_root_ref: str,
    candidate_root_digest: str,
    source_tree_digest: str,
    executable_entrypoint: str,
    execution_contract: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the canonical semantic ref/digest for one qualified realization."""

    if not isinstance(spec, OpenResearchSpecV1):
        raise SearchAdapterError("realization identity requires OpenResearchSpecV1")
    package_ref = _normalized_string(
        candidate_package_ref,
        field_name="candidate_package_ref",
    )
    root_ref = _normalized_string(
        candidate_root_ref,
        field_name="candidate_root_ref",
    )
    package_digest = validate_sha256(
        candidate_package_digest,
        field_name="candidate_package_digest",
    )
    root_digest = validate_sha256(
        candidate_root_digest,
        field_name="candidate_root_digest",
    )
    source_digest = validate_sha256(
        source_tree_digest,
        field_name="source_tree_digest",
    )
    entrypoint = _normalized_entrypoint(executable_entrypoint)
    if spec.execution_contract is None:
        raise SearchAdapterError(
            "realization identity requires a resolver execution_contract"
        )
    if not isinstance(execution_contract, Mapping):
        raise SearchAdapterError("execution_contract must be a mapping")
    normalized_contract = canonical_value(dict(execution_contract))
    if normalized_contract != canonical_value(dict(spec.execution_contract)):
        raise SearchAdapterError(
            "realization execution_contract is not bound to its spec"
        )
    identity = canonical_value(
        {
            "schema": "recclaw-open-spec-realization-semantics-e0-v1",
            "research_spec_digest": spec.digest,
            "candidate_package_ref": package_ref,
            "candidate_package_digest": package_digest,
            "candidate_root_ref": root_ref,
            "candidate_root_digest": root_digest,
            "source_tree_digest": source_digest,
            "executable_entrypoint": entrypoint,
            "execution_contract": normalized_contract,
        }
    )
    digest = sha256_digest(identity)
    return (
        content_id("recclaw-open-spec-realization-v1", identity),
        digest,
    )


@dataclass(frozen=True, slots=True)
class OpenSpecSearchCandidateV1:
    """A qualified OpenSpec realization exposed to the next-fresh Search lane."""

    spec: OpenResearchSpecV1
    capability_ref: str
    capability_digest: str
    candidate_package_ref: str
    candidate_package_digest: str
    candidate_root_ref: str
    candidate_root_digest: str
    source_tree_digest: str
    executable_entrypoint: str
    execution_contract: Mapping[str, Any]
    semantic_identity_ref: str
    semantic_identity_digest: str
    qualification_receipt_ref: str
    qualification_receipt_digest: str
    mechanism_axis: str
    utility_features: SearchUtilityFeaturesV1
    feature_evidence: RouterFeatureEvidenceV1
    realization_semantics_digest: str = ""

    schema: ClassVar[str] = (
        "recclaw.research-line.vnext.open-spec-search-candidate.e0.v1"
    )
    identity_namespace: ClassVar[str] = (
        "recclaw-open-spec-search-candidate-e0-v1"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, OpenResearchSpecV1):
            raise SearchAdapterError("OpenSpec candidate requires OpenResearchSpecV1")
        for field_name in (
            "capability_ref",
            "candidate_package_ref",
            "candidate_root_ref",
            "semantic_identity_ref",
            "qualification_receipt_ref",
            "mechanism_axis",
        ):
            _normalized_string(
                getattr(self, field_name),
                field_name=field_name,
            )
        for field_name in (
            "capability_digest",
            "candidate_package_digest",
            "candidate_root_digest",
            "source_tree_digest",
            "semantic_identity_digest",
            "qualification_receipt_digest",
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
        if not isinstance(self.utility_features, SearchUtilityFeaturesV1):
            raise SearchAdapterError(
                "OpenSpec candidate utility_features have the wrong type"
            )
        if not isinstance(self.feature_evidence, RouterFeatureEvidenceV1):
            raise SearchAdapterError(
                "OpenSpec candidate feature_evidence have the wrong type"
            )
        if self.spec.execution_contract is None:
            raise SearchAdapterError(
                "OpenSpec candidate requires a resolver execution_contract"
            )
        if not isinstance(self.execution_contract, Mapping):
            raise SearchAdapterError("execution_contract must be a mapping")
        execution_contract = canonical_value(dict(self.execution_contract))
        if execution_contract != canonical_value(
            dict(self.spec.execution_contract)
        ):
            raise SearchAdapterError(
                "OpenSpec candidate execution_contract is not bound to its spec"
            )
        object.__setattr__(self, "execution_contract", execution_contract)
        expected_ref, expected_semantics = open_spec_realization_identity(
            self.spec,
            candidate_package_ref=self.candidate_package_ref,
            candidate_package_digest=self.candidate_package_digest,
            candidate_root_ref=self.candidate_root_ref,
            candidate_root_digest=self.candidate_root_digest,
            source_tree_digest=self.source_tree_digest,
            executable_entrypoint=self.executable_entrypoint,
            execution_contract=self.execution_contract,
        )
        if self.realization_semantics_digest:
            provided = validate_sha256(
                self.realization_semantics_digest,
                field_name="realization_semantics_digest",
            )
            if provided != expected_semantics:
                raise SearchAdapterError(
                    "OpenSpec realization semantic identity drift"
                )
        if self.semantic_identity_digest != expected_semantics:
            raise SearchAdapterError(
                "OpenSpec semantic identity must equal realization semantics"
            )
        if self.semantic_identity_ref != expected_ref:
            raise SearchAdapterError(
                "OpenSpec semantic identity ref is not the realization ref"
            )
        object.__setattr__(
            self,
            "realization_semantics_digest",
            expected_semantics,
        )

    def _identity_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "spec": self.spec.to_dict(),
                "capability_ref": self.capability_ref,
                "capability_digest": self.capability_digest,
                "candidate_package_ref": self.candidate_package_ref,
                "candidate_package_digest": self.candidate_package_digest,
                "candidate_root_ref": self.candidate_root_ref,
                "candidate_root_digest": self.candidate_root_digest,
                "source_tree_digest": self.source_tree_digest,
                "executable_entrypoint": self.executable_entrypoint,
                "execution_contract": self.execution_contract,
                "semantic_identity_ref": self.semantic_identity_ref,
                "semantic_identity_digest": self.semantic_identity_digest,
                "qualification_receipt_ref": self.qualification_receipt_ref,
                "qualification_receipt_digest": self.qualification_receipt_digest,
                "mechanism_axis": self.mechanism_axis,
                "utility_features": self.utility_features,
                "feature_evidence": self.feature_evidence,
                "realization_semantics_digest": self.realization_semantics_digest,
            }
        )

    @property
    def candidate_id(self) -> str:
        return f"cand-open-{sha256_digest(self._identity_dict())}"

    @property
    def mechanism_id(self) -> str:
        return f"OPEN_{self.realization_semantics_digest[:16].upper()}"

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                **self._identity_dict(),
                "candidate_id": self.candidate_id,
            }
        )


@dataclass(frozen=True, slots=True)
class SearchCandidateBindingV1:
    """Search-owned proposal bound to one active executable entry."""

    proposal: CandidateProposalV4 | OpenSpecSearchCandidateV1
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
    proposal: CandidateProposalV4 | OpenSpecSearchCandidateV1,
    capability_ref: str,
) -> SearchCandidateBindingV1:
    """Bind a legacy proposal or qualified OpenSpec to an active entry."""

    if not isinstance(profile, SearchExecutableProfileV1):
        raise SearchAdapterError(
            "profile must be SearchExecutableProfileV1"
        )
    if not isinstance(
        proposal,
        (CandidateProposalV4, OpenSpecSearchCandidateV1),
    ):
        raise SearchAdapterError(
            "Search consumer requires CandidateProposalV4 or OpenSpecSearchCandidateV1"
        )
    entry = profile.entry(capability_ref)
    if isinstance(proposal, OpenSpecSearchCandidateV1):
        if entry.origin is not SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
            raise SearchAdapterError(
                "OpenSpec candidate requires a qualified registry entry"
            )
        if profile.activation is not SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN:
            raise SearchAdapterError(
                "qualified capability is unavailable before fresh activation"
            )
        if (
            proposal.capability_ref != entry.capability_ref
            or proposal.capability_digest != entry.capability_digest
            or proposal.executable_entrypoint != entry.executable_entrypoint
            or proposal.semantic_identity_ref != entry.semantic_identity_ref
            or proposal.semantic_identity_digest != entry.semantic_identity_digest
            or proposal.qualification_receipt_ref
            != entry.qualification_receipt_ref
            or proposal.qualification_receipt_digest
            != entry.qualification_receipt_digest
        ):
            raise SearchAdapterError(
                "OpenSpec candidate qualified admission identity drift"
            )
        if (
            proposal.spec.protocol_ref != profile.protocol_ref
            or proposal.spec.protocol_digest != profile.protocol_digest
        ):
            raise SearchAdapterError(
                "OpenSpec candidate protocol identity drift"
            )
        return SearchCandidateBindingV1(
            proposal=proposal,
            capability_ref=entry.capability_ref,
            capability_digest=entry.capability_digest,
            executable_entrypoint=entry.executable_entrypoint,
            entry_origin=entry.origin,
            mechanism_semantics_digest=proposal.realization_semantics_digest,
        )
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


def _effective_router_policy_digest(
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None,
) -> str:
    return sha256_digest(
        {
            "hard_gate_policy_digest": router.policy_digest,
            "versioned_policy": policy_projection,
        }
    )


def _normalize_repeat_ranking_inputs(
    *,
    executed_semantic_seed_pairs: Sequence[tuple[str, str]],
    current_observation_seed: str | None,
    pending_task: Mapping[str, Any] | None,
) -> tuple[tuple[tuple[str, str], ...], str | None, Mapping[str, Any] | None]:
    pairs: list[tuple[str, str]] = []
    for pair in executed_semantic_seed_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise SearchAdapterError(
                "executed semantic history must contain semantic/seed pairs"
            )
        semantic_digest = validate_sha256(
            pair[0],
            field_name="executed_candidate_semantic_digest",
        )
        seed = _normalized_string(pair[1], field_name="executed_observation_seed")
        normalized_pair = (semantic_digest, seed)
        if normalized_pair not in pairs:
            pairs.append(normalized_pair)
    normalized_seed = (
        _normalized_string(
            current_observation_seed,
            field_name="current_observation_seed",
        )
        if current_observation_seed is not None
        else None
    )
    normalized_task = (
        canonical_value(dict(pending_task))
        if pending_task is not None
        else None
    )
    if normalized_task is not None and not isinstance(normalized_task, Mapping):
        raise SearchAdapterError("pending_task must be a mapping")
    return tuple(pairs), normalized_seed, normalized_task


def _effective_repeat_policy_projection(
    policy_projection: Mapping[str, Any] | None,
    *,
    executed_semantic_seed_pairs: tuple[tuple[str, str], ...],
    current_observation_seed: str | None,
    pending_task: Mapping[str, Any] | None,
    mechanism_axis_effects: Mapping[str, float],
) -> Mapping[str, Any] | None:
    if (
        not executed_semantic_seed_pairs
        and current_observation_seed is None
        and pending_task is None
        and not mechanism_axis_effects
    ):
        return policy_projection
    effective = dict(policy_projection or {})
    effective["research_line_repeat_ranking"] = {
        "executed_semantic_seed_pairs": executed_semantic_seed_pairs,
        "current_observation_seed": current_observation_seed,
        "pending_task": pending_task,
    }
    if mechanism_axis_effects:
        effective["research_line_axis_effects"] = dict(
            sorted(mechanism_axis_effects.items())
        )
    return canonical_value(effective)


def _axis_effect_adjustment(
    mechanism_axis: str,
    mechanism_axis_effects: Mapping[str, float],
) -> float:
    """Use observed axis effects as a soft ranking prior, never a gate."""

    effect = float(mechanism_axis_effects.get(mechanism_axis, 0.0))
    return round(max(-0.15, min(0.10, effect)), 12)


def _is_repeat_semantics(
    *,
    semantics_digest: str,
    executed_semantic_seed_pairs: tuple[tuple[str, str], ...],
    current_observation_seed: str | None,
    pending_task: Mapping[str, Any] | None,
) -> bool:
    if not any(semantic == semantics_digest for semantic, _seed in executed_semantic_seed_pairs):
        return False
    if current_observation_seed is None:
        return True
    executed_pair = (semantics_digest, current_observation_seed)
    task_type = (
        pending_task.get("task_type") if pending_task is not None else None
    )
    task_status = (
        pending_task.get("task_status") if pending_task is not None else None
    )
    if getattr(task_type, "value", task_type) == "VALIDATE_SAME_CANDIDATE":
        if (
            task_status == "PENDING"
            and pending_task.get("candidate_semantic_digest") == semantics_digest
            and _pending_new_seed_matches(pending_task, current_observation_seed)
            and executed_pair not in executed_semantic_seed_pairs
        ):
            return False
    return True


def _decision_with_reason(
    decision: RouterHardGateDecisionV1,
    *,
    allowed: bool,
    reason: RouterHardGateReasonV1,
    policy_digest: str | None = None,
) -> RouterHardGateDecisionV1:
    return RouterHardGateDecisionV1(
        candidate_id=decision.candidate_id,
        allowed=allowed,
        reason=reason,
        compile_report_digest=decision.compile_report_digest,
        mechanism_semantics_digest=decision.mechanism_semantics_digest,
        feature_digest=decision.feature_digest,
        policy_digest=(
            decision.policy_digest
            if policy_digest is None
            else policy_digest
        ),
    )


def _route_mixed_slate(
    *,
    slate: FrozenExperimentSlateV1,
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None,
    executed_semantic_seed_pairs: Sequence[tuple[str, str]] = (),
    current_observation_seed: str | None = None,
    pending_task: Mapping[str, Any] | None = None,
    mechanism_axis_effects: Mapping[str, float] | None = None,
) -> RouteTraceV1:
    """Route a legacy or mixed slate with history-aware soft utility ranking."""

    (
        executed_semantic_seed_pairs,
        current_observation_seed,
        pending_task,
    ) = _normalize_repeat_ranking_inputs(
        executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        current_observation_seed=current_observation_seed,
        pending_task=pending_task,
    )
    normalized_axis_effects = {
        str(axis): float(effect)
        for axis, effect in (mechanism_axis_effects or {}).items()
    }
    policy_projection = _effective_repeat_policy_projection(
        policy_projection,
        executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        current_observation_seed=current_observation_seed,
        pending_task=pending_task,
        mechanism_axis_effects=normalized_axis_effects,
    )

    policy_digest = _effective_router_policy_digest(
        router,
        policy_projection,
    )
    records: list[dict[str, Any]] = []
    for index, binding in enumerate(slate.bindings):
        proposal = binding.proposal
        if isinstance(proposal, CandidateProposalV4):
            # Calling the real router for a singleton preserves its compiler,
            # hard gates, feature digest, and policy scoring.  The mixed
            # slate applies semantic de-duplication and the slate ceiling
            # after both candidate kinds are in the same pool.
            single_trace = router.route(
                (proposal,),
                policy_projection=policy_projection,
            )
            original = single_trace.decisions[0]
            base_allowed = original.reason in {
                RouterHardGateReasonV1.ALLOW,
                RouterHardGateReasonV1.SLATE_CEILING,
            }
            base_reason = (
                RouterHardGateReasonV1.ALLOW
                if base_allowed
                else original.reason
            )
            decision = _decision_with_reason(
                original,
                allowed=base_allowed,
                reason=base_reason,
            )
            semantics_digest = original.mechanism_semantics_digest
            score = router.score(
                proposal.utility_features,
                policy_projection,
            )
        else:
            features = proposal.utility_features
            reason = RouterHardGateReasonV1.ALLOW
            if features.runnable_probability < router.runnable_floor:
                reason = RouterHardGateReasonV1.RUNNABLE_BELOW_FLOOR
            elif features.blocker_risk > router.blocker_ceiling:
                reason = RouterHardGateReasonV1.BLOCKER_RISK_ABOVE_CEILING
            elif features.cost > router.cost_ceiling:
                reason = RouterHardGateReasonV1.COST_ABOVE_CEILING
            score = router.score(features, policy_projection)
            if reason is RouterHardGateReasonV1.ALLOW and score < router.utility_floor:
                reason = RouterHardGateReasonV1.UTILITY_BELOW_FLOOR
            base_allowed = reason is RouterHardGateReasonV1.ALLOW
            decision = RouterHardGateDecisionV1(
                candidate_id=proposal.candidate_id,
                allowed=base_allowed,
                reason=reason,
                compile_report_digest=None,
                mechanism_semantics_digest=(
                    proposal.realization_semantics_digest
                ),
                feature_digest=sha256_digest(features),
                policy_digest=policy_digest,
            )
            semantics_digest = proposal.realization_semantics_digest

        score += _axis_effect_adjustment(
            proposal.mechanism_axis,
            normalized_axis_effects,
        )

        if semantics_digest is None:
            base_allowed = False
            decision = _decision_with_reason(
                decision,
                allowed=False,
                reason=RouterHardGateReasonV1.BL_COMPILE_FAILED,
            )
        repeat = (
            semantics_digest is not None
            and _is_repeat_semantics(
                semantics_digest=str(semantics_digest),
                executed_semantic_seed_pairs=executed_semantic_seed_pairs,
                current_observation_seed=current_observation_seed,
                pending_task=pending_task,
            )
        )
        if repeat:
            score -= _REPEAT_UTILITY_PENALTY
        records.append(
            {
                "index": index,
                "binding": binding,
                "candidate_id": proposal.candidate_id,
                "decision": decision,
                "base_allowed": base_allowed,
                "semantics_digest": semantics_digest,
                "score": score,
                "repeat": repeat,
            }
        )

    eligible = [
        record
        for record in records
        if record["base_allowed"] and record["semantics_digest"] is not None
    ]
    eligible.sort(key=lambda record: (-record["score"], record["index"]))
    unique: list[dict[str, Any]] = []
    duplicate_ids: set[str] = set()
    seen_semantics: set[str] = set()
    for record in eligible:
        semantics_digest = str(record["semantics_digest"])
        if semantics_digest in seen_semantics:
            duplicate_ids.add(record["candidate_id"])
            continue
        seen_semantics.add(semantics_digest)
        unique.append(record)

    keep = unique[: max(0, router.slate_ceiling)]
    keep_ids = {record["candidate_id"] for record in keep}
    decisions: list[RouterHardGateDecisionV1] = []
    for record in records:
        decision = record["decision"]
        candidate_id = record["candidate_id"]
        if candidate_id in duplicate_ids:
            decision = _decision_with_reason(
                decision,
                allowed=False,
                reason=RouterHardGateReasonV1.SEMANTIC_DUPLICATE,
            )
        elif record["base_allowed"] and candidate_id not in keep_ids:
            decision = _decision_with_reason(
                decision,
                allowed=False,
                reason=RouterHardGateReasonV1.SLATE_CEILING,
            )
        decisions.append(decision)

    proposal_pool = [
        binding.proposal.to_dict() for binding in slate.bindings
    ]
    return RouteTraceV1(
        pool_digest=sha256_digest(proposal_pool),
        ordered_candidate_ids=tuple(
            record["candidate_id"] for record in records
        ),
        ranked_candidate_ids=tuple(
            record["candidate_id"] for record in keep
        ),
        decisions=tuple(decisions),
        selected_candidate_id=(
            keep[0]["candidate_id"] if keep else None
        ),
        selection_score=(keep[0]["score"] if keep else None),
        policy_digest=policy_digest,
    )


def route_frozen_experiment_slate(
    *,
    profile: SearchExecutableProfileV1,
    slate: FrozenExperimentSlateV1,
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None = None,
    executed_semantic_seed_pairs: Sequence[tuple[str, str]] = (),
    current_observation_seed: str | None = None,
    pending_task: Mapping[str, Any] | None = None,
    mechanism_axis_effects: Mapping[str, float] | None = None,
    portfolio_candidates: (
        Sequence[PortfolioCandidateV2]
        | Mapping[str, PortfolioCandidateV2 | Mapping[str, Any]]
        | None
    ) = None,
    portfolio_control: PortfolioControlStateV2 | Mapping[str, Any] | None = None,
    attempt_failures: Sequence[
        PortfolioAttemptFailureV2 | Mapping[str, Any] | str
    ] = (),
    failed_attempts: Sequence[
        PortfolioAttemptFailureV2 | Mapping[str, Any] | str
    ] | None = None,
    failed_candidate_ids: Sequence[str] = (),
    first_window_size: int | None = None,
) -> ExperimentAcquisitionResultV1:
    """Route a frozen slate through the legacy or explicit portfolio path.

    All new arguments are optional.  Omitting them preserves the historical
    StrongStaticRouterV1 behavior.  Supplying portfolio state enables a full
    eligible-pool ranking with a bounded first window (four by default), so a
    sealed candidate-local failure can be removed and the remaining pool
    re-ranked without another Provider call.
    """

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

    effective_failed_attempts = tuple(
        attempt_failures
    ) + tuple(failed_attempts or ())
    portfolio_mode = any(
        (
            portfolio_candidates is not None,
            portfolio_control is not None,
            bool(effective_failed_attempts),
            bool(failed_candidate_ids),
            first_window_size is not None,
        )
    )
    if portfolio_mode:
        trace = _route_portfolio_slate(
            slate=slate,
            router=router,
            policy_projection=policy_projection,
            portfolio_candidates=portfolio_candidates,
            portfolio_control=portfolio_control,
            attempt_failures=effective_failed_attempts,
            failed_candidate_ids=failed_candidate_ids,
            first_window_size=first_window_size,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            current_observation_seed=current_observation_seed,
            pending_task=pending_task,
        )
    else:
        trace = _route_mixed_slate(
            slate=slate,
            router=router,
            policy_projection=policy_projection,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            current_observation_seed=current_observation_seed,
            pending_task=pending_task,
            mechanism_axis_effects=mechanism_axis_effects,
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
    "AttemptFailureScopeV2",
    "ExperimentAcquisitionResultV1",
    "FrozenExperimentSlateV1",
    "IdeaAcquisitionResultV1",
    "IdeaRouteV1",
    "OpenSpecSearchCandidateV1",
    "ParentValidationStateV2",
    "PortfolioAttemptFailureV2",
    "PortfolioCandidateV2",
    "PortfolioControlStateV2",
    "PortfolioEligibilityV2",
    "ResourceAdmissionStateV2",
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
    "open_spec_realization_identity",
    "predecessor_executable_entries",
    "route_frozen_experiment_slate",
]

def _portfolio_candidate_for_binding(
    binding: SearchCandidateBindingV1,
    override: PortfolioCandidateV2 | None = None,
) -> PortfolioCandidateV2:
    """Validate the explicit outcome-blind profile for one binding."""

    proposal = binding.proposal
    if override is None:
        raise SearchAdapterError(
            "portfolio/failover routing requires an explicit PortfolioCandidateV2 "
            "profile for every frozen Search candidate"
        )
    if (
        override.candidate_id != proposal.candidate_id
        or override.semantic_digest != binding.mechanism_semantics_digest
    ):
        raise SearchAdapterError(
            "portfolio candidate identity is not bound to the Search binding"
        )
    return override


def _portfolio_override_map(
    value: (
        Sequence[PortfolioCandidateV2]
        | Mapping[str, PortfolioCandidateV2 | Mapping[str, Any]]
        | None
    ),
) -> dict[str, PortfolioCandidateV2]:
    if value is None:
        return {}
    items: Sequence[Any]
    if isinstance(value, Mapping):
        if "candidates" in value:
            raw = value["candidates"]
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise SearchAdapterError(
                    "portfolio candidates must be a sequence"
                )
            items = raw
        else:
            items = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        raise SearchAdapterError(
            "portfolio candidates must be PortfolioCandidateV2 records"
        )
    result: dict[str, PortfolioCandidateV2] = {}
    for item in items:
        candidate = (
            item
            if isinstance(item, PortfolioCandidateV2)
            else PortfolioCandidateV2.from_mapping(item)
        )
        if candidate.candidate_id in result:
            raise SearchAdapterError(
                "portfolio candidate IDs must be unique"
            )
        result[candidate.candidate_id] = candidate
    return result


def _portfolio_failure(
    value: PortfolioAttemptFailureV2 | Mapping[str, Any] | str,
) -> PortfolioAttemptFailureV2:
    if isinstance(value, PortfolioAttemptFailureV2):
        return value
    if isinstance(value, Mapping):
        return PortfolioAttemptFailureV2(**dict(value))
    return PortfolioAttemptFailureV2(candidate_id=value)


def _portfolio_control(
    value: PortfolioControlStateV2 | Mapping[str, Any] | None,
    *,
    attempt_failures: Sequence[
        PortfolioAttemptFailureV2 | Mapping[str, Any] | str
    ],
    failed_candidate_ids: Sequence[str],
    first_window_size: int | None,
) -> PortfolioControlStateV2:
    if value is None:
        state = PortfolioControlStateV2()
    elif isinstance(value, PortfolioControlStateV2):
        state = value
    elif isinstance(value, Mapping):
        payload = dict(value)
        payload["attempt_failures"] = tuple(
            _portfolio_failure(item)
            for item in payload.get("attempt_failures", ())
        )
        state = PortfolioControlStateV2(**payload)
    else:
        raise SearchAdapterError(
            "portfolio control must be PortfolioControlStateV2"
        )
    for failure in attempt_failures:
        state = state.with_attempt_failure(_portfolio_failure(failure))
    if failed_candidate_ids:
        state = replace(
            state,
            failed_candidate_ids=(
                *state.failed_candidate_ids,
                *(str(item) for item in failed_candidate_ids),
            ),
        )
    if first_window_size is not None:
        state = replace(state, first_window_size=first_window_size)
    return state


def _portfolio_hard_gate(
    binding: SearchCandidateBindingV1,
    *,
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None,
) -> RouterHardGateDecisionV1:
    proposal = binding.proposal
    if isinstance(proposal, CandidateProposalV4):
        trace = router.route((proposal,), policy_projection=policy_projection)
        return trace.decisions[0]
    features = proposal.utility_features
    reason = RouterHardGateReasonV1.ALLOW
    if features.runnable_probability < router.runnable_floor:
        reason = RouterHardGateReasonV1.RUNNABLE_BELOW_FLOOR
    elif features.blocker_risk > router.blocker_ceiling:
        reason = RouterHardGateReasonV1.BLOCKER_RISK_ABOVE_CEILING
    elif features.cost > router.cost_ceiling:
        reason = RouterHardGateReasonV1.COST_ABOVE_CEILING
    score = router.score(features, policy_projection)
    if reason is RouterHardGateReasonV1.ALLOW and score < router.utility_floor:
        reason = RouterHardGateReasonV1.UTILITY_BELOW_FLOOR
    return RouterHardGateDecisionV1(
        candidate_id=proposal.candidate_id,
        allowed=reason is RouterHardGateReasonV1.ALLOW,
        reason=reason,
        compile_report_digest=None,
        mechanism_semantics_digest=binding.mechanism_semantics_digest,
        feature_digest=sha256_digest(features),
        policy_digest=_effective_router_policy_digest(
            router,
            policy_projection,
        ),
    )


def _portfolio_router_reason(
    eligibility: PortfolioEligibilityV2,
) -> RouterHardGateReasonV1:
    if eligibility is PortfolioEligibilityV2.NOT_RESOURCE_ADMITTED:
        return RouterHardGateReasonV1.RUNNABLE_BELOW_FLOOR
    if eligibility in {
        PortfolioEligibilityV2.UNVERIFIED_PARENT,
        PortfolioEligibilityV2.FAILED_PARENT,
        PortfolioEligibilityV2.CORRELATED_COMPUTE_PATTERN,
        PortfolioEligibilityV2.LINEAGE_RISK,
    }:
        return RouterHardGateReasonV1.BLOCKER_RISK_ABOVE_CEILING
    return RouterHardGateReasonV1.UTILITY_BELOW_FLOOR


def _route_portfolio_slate(
    *,
    slate: FrozenExperimentSlateV1,
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None,
    portfolio_candidates: (
        Sequence[PortfolioCandidateV2]
        | Mapping[str, PortfolioCandidateV2 | Mapping[str, Any]]
        | None
    ),
    portfolio_control: PortfolioControlStateV2 | Mapping[str, Any] | None,
    attempt_failures: Sequence[
        PortfolioAttemptFailureV2 | Mapping[str, Any] | str
    ],
    failed_candidate_ids: Sequence[str],
    first_window_size: int | None,
    executed_semantic_seed_pairs: Sequence[tuple[str, str]],
    current_observation_seed: str | None,
    pending_task: Mapping[str, Any] | None,
) -> RouteTraceV1:
    _load_portfolio_symbols()
    if portfolio_candidates is None:
        raise SearchAdapterError(
            "portfolio/failover routing requires explicit PortfolioCandidateV2 "
            "profiles for every frozen Search candidate"
        )
    overrides = _portfolio_override_map(portfolio_candidates)
    expected_ids = {binding.proposal.candidate_id for binding in slate.bindings}
    missing_overrides = expected_ids - set(overrides)
    unknown_overrides = set(overrides) - expected_ids
    if missing_overrides:
        raise SearchAdapterError(
            "portfolio candidate profiles are missing for frozen Search candidates: "
            + ", ".join(sorted(missing_overrides))
        )
    if unknown_overrides:
        raise SearchAdapterError(
            "portfolio candidate is outside the frozen Search slate: "
            + ", ".join(sorted(unknown_overrides))
        )
    control = _portfolio_control(
        portfolio_control,
        attempt_failures=attempt_failures,
        failed_candidate_ids=failed_candidate_ids,
        first_window_size=first_window_size,
    )
    raw_pairs = tuple(executed_semantic_seed_pairs)
    normalized_pairs, normalized_seed, normalized_task = (
        _normalize_repeat_ranking_inputs(
            executed_semantic_seed_pairs=raw_pairs,
            current_observation_seed=current_observation_seed,
            pending_task=pending_task,
        )
    )
    if raw_pairs:
        counts = dict(control.repeat_counts)
        for semantic, _seed in raw_pairs:
            counts[semantic] = counts.get(semantic, 0) + 1
        # An explicit, unexecuted new-seed task is information-seeking work,
        # not a blind repeat.  Keep the candidate routable without using its
        # unseen result.
        if (
            normalized_seed is not None
            and normalized_task is not None
            and normalized_task.get("task_status") == "PENDING"
            and normalized_task.get("task_type") == "VALIDATE_SAME_CANDIDATE"
            and normalized_task.get("candidate_semantic_digest")
            and _pending_new_seed_matches(normalized_task, normalized_seed)
            and (normalized_task["candidate_semantic_digest"], normalized_seed)
            not in normalized_pairs
        ):
            counts[normalized_task["candidate_semantic_digest"]] = 0
        control = replace(control, repeat_counts=tuple(counts.items()))

    hard_decisions: dict[str, RouterHardGateDecisionV1] = {}
    candidates: list[PortfolioCandidateV2] = []
    for binding in slate.bindings:
        decision = _portfolio_hard_gate(
            binding,
            router=router,
            policy_projection=policy_projection,
        )
        hard_decisions[binding.proposal.candidate_id] = decision
        candidates.append(
            _portfolio_candidate_for_binding(
                binding,
                overrides.get(binding.proposal.candidate_id),
            )
        )
    ranking = rank_candidate_portfolio_v2(
        candidates,
        control=control,
        first_window_size=first_window_size,
    )
    record_by_id = {
        item.candidate.candidate_id: item
        for item in ranking.records
    }
    hard_allowed = {
        candidate_id
        for candidate_id, decision in hard_decisions.items()
        if decision.allowed or decision.reason is RouterHardGateReasonV1.SLATE_CEILING
    }
    full_order = tuple(
        candidate_id
        for candidate_id in ranking.ranked_candidate_ids
        if candidate_id in hard_allowed
    )
    # A durable fresh-seed confirmation task is executable by the same
    # semantic candidate under the requested current seed.  When that exact
    # candidate remains portfolio-eligible, place it ahead of unrelated
    # exploration; resource/lineage/hard-gate exclusions still win.  Other
    # operations (for example matched controls) require an explicit control
    # binding and are not guessed from the intervention identity.
    if (
        normalized_seed is not None
        and normalized_task is not None
        and normalized_task.get("task_status") == "PENDING"
        and normalized_task.get("task_type") in {
            "VALIDATE_SAME_CANDIDATE",
            "RUN_MATCHED_CONTROL",
            "RUN_ABLATION",
        }
        and normalized_task.get("required_seed_or_control") == normalized_seed
    ):
        task_semantic = normalized_task.get("candidate_semantic_digest")
        task_candidate_id = normalized_task.get("candidate_id")
        task_candidates = tuple(
            candidate.candidate_id
            for candidate in candidates
            if candidate.semantic_digest == task_semantic
            and (
                task_candidate_id is None
                or candidate.candidate_id == task_candidate_id
            )
            and candidate.candidate_id in full_order
        )
        if task_candidates:
            full_order = (
                *task_candidates,
                *(item for item in full_order if item not in task_candidates),
            )
    window = control.first_window_size
    first_window = full_order[:window]
    selected = first_window[0] if first_window else None
    policy_digest = sha256_digest(
        {
            "legacy_router_policy": _effective_router_policy_digest(
                router,
                policy_projection,
            ),
            "portfolio_policy": ranking.policy_digest,
            "executed_semantic_seed_pairs": normalized_pairs,
            "current_observation_seed": normalized_seed,
            "pending_task": normalized_task,
        }
    )
    decisions: list[RouterHardGateDecisionV1] = []
    for binding in slate.bindings:
        candidate_id = binding.proposal.candidate_id
        base = hard_decisions[candidate_id]
        if candidate_id not in hard_allowed:
            decisions.append(
                _decision_with_reason(
                    base,
                    allowed=False,
                    reason=base.reason,
                    policy_digest=policy_digest,
                )
            )
            continue
        record = record_by_id[candidate_id]
        if not record.eligible:
            decisions.append(
                _decision_with_reason(
                    base,
                    allowed=False,
                    reason=_portfolio_router_reason(record.eligibility),
                    policy_digest=policy_digest,
                )
            )
            continue
        decisions.append(
            _decision_with_reason(
                base,
                allowed=True,
                reason=RouterHardGateReasonV1.ALLOW,
                policy_digest=policy_digest,
            )
        )
    selection_score = None
    if selected is not None:
        selection_score = next(
            item.score
            for item in ranking.records
            if item.candidate.candidate_id == selected
        )
    return RouteTraceV1(
        pool_digest=sha256_digest(
            [item.proposal.to_dict() for item in slate.bindings]
        ),
        ordered_candidate_ids=tuple(
            item.proposal.candidate_id for item in slate.bindings
        ),
        ranked_candidate_ids=first_window,
        decisions=tuple(decisions),
        selected_candidate_id=selected,
        selection_score=selection_score,
        policy_digest=policy_digest,
    )
