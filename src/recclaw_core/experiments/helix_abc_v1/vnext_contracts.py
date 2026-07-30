"""RC0 common contracts for the open-capability Research Line vNext.

This module freezes shared semantics, identity bindings, references, and
canonical serialization only.  It deliberately contains no Resolver,
Implementer, qualification runner, registry, profile builder, learner, or
experiment orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar

from .canonical import (
    canonical_json_bytes,
    canonical_value,
    content_id,
    sha256_digest,
    validate_relative_artifact_path,
    validate_sha256,
)
from .contracts import EVIDENCE_CLASS
from .research_contracts import DISCOVERY_PRODUCERS


NEXT_FRESH_CAMPAIGN = "NEXT_FRESH_CAMPAIGN"
NO_MECHANISM_BELIEF_AUTHORITY = "NONE"


class VNextContractError(ValueError):
    """Raised when an RC0 Research Line vNext invariant is violated."""


class CapabilityResolutionResultV1(str, Enum):
    SEARCH_READY = "SEARCH_READY"
    INNOVATION_REQUIRED = "INNOVATION_REQUIRED"
    DEFERRED_PROTOCOL_CHANGE = "DEFERRED_PROTOCOL_CHANGE"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_SPEC = "INVALID_SPEC"


class CurrentProfileExpressibilityV1(str, Enum):
    EXPRESSIBLE = "EXPRESSIBLE"
    NOT_EXPRESSIBLE = "NOT_EXPRESSIBLE"
    UNRESOLVED = "UNRESOLVED"


class QualificationStageV1(str, Enum):
    STATIC_VALIDATION = "STATIC_VALIDATION"
    CONSTRUCTION = "CONSTRUCTION"
    API_CONTRACT = "API_CONTRACT"
    UNIT = "UNIT"
    ONE_EPOCH_SMOKE = "ONE_EPOCH_SMOKE"


class QualificationCheckStatusV1(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


class QualificationStatusV1(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class QualificationFailureClassV1(str, Enum):
    NONE = "NONE"
    IMPLEMENTATION = "IMPLEMENTATION"
    INTERFACE = "INTERFACE"
    RUNTIME = "RUNTIME"
    PROTOCOL = "PROTOCOL"
    RESOURCE = "RESOURCE"
    INCONCLUSIVE = "INCONCLUSIVE"


class CapabilityKindV1(str, Enum):
    OPERATOR = "OPERATOR"
    COMPLETE_MODEL = "COMPLETE_MODEL"
    INTERACTION_HEAD = "INTERACTION_HEAD"
    PROPAGATION_MECHANISM = "PROPAGATION_MECHANISM"
    COMPOSITE_MODULE = "COMPOSITE_MODULE"
    RECIPE_PACKAGE = "RECIPE_PACKAGE"


class ResearchFailureClassV1(str, Enum):
    NONE = "NONE"
    MECHANISM = "MECHANISM"
    IMPLEMENTATION = "IMPLEMENTATION"
    INTERFACE = "INTERFACE"
    RUNTIME = "RUNTIME"
    PROTOCOL = "PROTOCOL"
    RESOURCE = "RESOURCE"
    INCONCLUSIVE = "INCONCLUSIVE"


class EpisodeEvidenceClassV1(str, Enum):
    ENGINEERING_ONLY = "ENGINEERING_ONLY"
    DEVELOPMENT_EXPERIMENT = "DEVELOPMENT_EXPERIMENT"
    FORMAL_EXPERIMENT = "FORMAL_EXPERIMENT"
    INCONCLUSIVE_EXPERIMENT = "INCONCLUSIVE_EXPERIMENT"


class AcquisitionStageV1(str, Enum):
    IDEA = "IDEA"
    EXPERIMENT = "EXPERIMENT"


class AcquisitionDispositionV1(str, Enum):
    SELECT = "SELECT"
    REJECT = "REJECT"
    DEFER = "DEFER"


class _CanonicalVNextContract:
    schema: ClassVar[str]
    identity_namespace: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value({"schema": self.schema, **self.to_dict()})

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_dict())

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())

    @property
    def record_id(self) -> str:
        return content_id(self.identity_namespace, self.canonical_dict())


def _nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VNextContractError(
            f"{field_name} must be a non-empty, whitespace-normalized string"
        )
    return value


def _normalize_digest_fields(value: object) -> None:
    for item in fields(value):
        if not item.name.endswith(("_digest", "_hash")):
            continue
        observed = getattr(value, item.name)
        if observed is None:
            continue
        normalized = validate_sha256(str(observed), field_name=item.name)
        object.__setattr__(value, item.name, normalized)


def _required_ref_digest(
    ref: str,
    digest: str,
    *,
    ref_field: str,
    digest_field: str,
) -> None:
    _nonempty(ref, field_name=ref_field)
    validate_sha256(digest, field_name=digest_field)


def _optional_ref_digest(
    ref: str | None,
    digest: str | None,
    *,
    ref_field: str,
    digest_field: str,
) -> None:
    if (ref is None) != (digest is None):
        raise VNextContractError(
            f"{ref_field} and {digest_field} must be present or absent together"
        )
    if ref is not None and digest is not None:
        _required_ref_digest(
            ref,
            digest,
            ref_field=ref_field,
            digest_field=digest_field,
        )


def _sorted_unique_strings(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(
        _nonempty(str(item), field_name=f"{field_name}[{index}]")
        for index, item in enumerate(values)
    )
    if not allow_empty and not normalized:
        raise VNextContractError(f"{field_name} must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise VNextContractError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def _normalize_ref_digest_pairs(
    values: tuple[tuple[str, str], ...],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for index, pair in enumerate(values):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise VNextContractError(
                f"{field_name}[{index}] must be a (ref, digest) pair"
            )
        ref = _nonempty(str(pair[0]), field_name=f"{field_name}[{index}].ref")
        digest = validate_sha256(
            str(pair[1]),
            field_name=f"{field_name}[{index}].digest",
        )
        normalized.append((ref, digest))
    if not allow_empty and not normalized:
        raise VNextContractError(f"{field_name} must be non-empty")
    if len({ref for ref, _digest in normalized}) != len(normalized):
        raise VNextContractError(f"{field_name} refs must be unique")
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _entrypoint(value: str, *, field_name: str = "executable_entrypoint") -> str:
    normalized = _nonempty(value, field_name=field_name)
    module, separator, attribute = normalized.partition(":")
    if (
        separator != ":"
        or not module
        or not attribute
        or any(character.isspace() for character in normalized)
    ):
        raise VNextContractError(
            f"{field_name} must use the canonical module.path:Attribute form"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class OpenResearchSpecV1(_CanonicalVNextContract):
    hypothesis: str
    mechanism_change: str
    competing_explanation: str
    matched_control_requirement: str
    implementation_requirements: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    falsifier: str
    compatibility_requirements: tuple[str, ...]
    protocol_ref: str
    protocol_digest: str
    context_ref: str
    context_digest: str
    current_profile_ref: str
    current_profile_digest: str
    producer_role: str
    high_change_justification: str
    current_profile_expressibility_claim: CurrentProfileExpressibilityV1

    schema = "recclaw.research-line.vnext.open-research-spec.v1"
    identity_namespace = "recclaw-open-research-spec-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        for field_name in (
            "hypothesis",
            "mechanism_change",
            "competing_explanation",
            "matched_control_requirement",
            "falsifier",
            "high_change_justification",
        ):
            _nonempty(getattr(self, field_name), field_name=field_name)
        for ref_field, digest_field in (
            ("protocol_ref", "protocol_digest"),
            ("context_ref", "context_digest"),
            ("current_profile_ref", "current_profile_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        if self.producer_role not in DISCOVERY_PRODUCERS:
            raise VNextContractError("producer_role must use a frozen Research Producer role")
        if not isinstance(
            self.current_profile_expressibility_claim,
            CurrentProfileExpressibilityV1,
        ):
            raise VNextContractError(
                "current_profile_expressibility_claim is outside the closed domain"
            )
        object.__setattr__(
            self,
            "implementation_requirements",
            _sorted_unique_strings(
                self.implementation_requirements,
                field_name="implementation_requirements",
            ),
        )
        object.__setattr__(
            self,
            "expected_evidence",
            _sorted_unique_strings(
                self.expected_evidence,
                field_name="expected_evidence",
            ),
        )
        object.__setattr__(
            self,
            "compatibility_requirements",
            _sorted_unique_strings(
                self.compatibility_requirements,
                field_name="compatibility_requirements",
            ),
        )

    @property
    def spec_id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True)
class CapabilityResolutionV1(_CanonicalVNextContract):
    research_spec_ref: str
    research_spec_digest: str
    current_profile_ref: str
    current_profile_digest: str
    resolution: CapabilityResolutionResultV1
    current_profile_match: bool
    resolved_current_capability_ref: str | None
    resolved_current_capability_digest: str | None
    capability_diff: tuple[str, ...]
    protocol_compatible: bool
    dependency_compatible: bool
    budget_compatible: bool
    reason_codes: tuple[str, ...]
    no_silent_fallback: bool
    catalog_fallback_used: bool

    schema = "recclaw.research-line.vnext.capability-resolution.v1"
    identity_namespace = "recclaw-capability-resolution-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        for ref_field, digest_field in (
            ("research_spec_ref", "research_spec_digest"),
            ("current_profile_ref", "current_profile_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        _optional_ref_digest(
            self.resolved_current_capability_ref,
            self.resolved_current_capability_digest,
            ref_field="resolved_current_capability_ref",
            digest_field="resolved_current_capability_digest",
        )
        if not isinstance(self.resolution, CapabilityResolutionResultV1):
            raise VNextContractError("resolution is outside the frozen five-state domain")
        object.__setattr__(
            self,
            "capability_diff",
            _sorted_unique_strings(
                self.capability_diff,
                field_name="capability_diff",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _sorted_unique_strings(self.reason_codes, field_name="reason_codes"),
        )
        if not self.no_silent_fallback or self.catalog_fallback_used:
            raise VNextContractError(
                "Capability Resolution may not silently fall back to the fixed catalog"
            )
        if self.resolution is CapabilityResolutionResultV1.SEARCH_READY:
            if (
                not self.current_profile_match
                or self.resolved_current_capability_ref is None
                or self.capability_diff
            ):
                raise VNextContractError(
                    "SEARCH_READY requires an exact current-profile capability match"
                )
        else:
            if self.current_profile_match:
                raise VNextContractError(
                    "non-SEARCH_READY resolution cannot claim a current-profile match"
                )
            if self.resolved_current_capability_ref is not None:
                raise VNextContractError(
                    "non-SEARCH_READY resolution cannot substitute a catalog capability"
                )
        if self.resolution is CapabilityResolutionResultV1.INNOVATION_REQUIRED:
            if (
                not self.capability_diff
                or not self.protocol_compatible
                or not self.dependency_compatible
                or not self.budget_compatible
            ):
                raise VNextContractError(
                    "INNOVATION_REQUIRED needs a feasible, protocol-compatible capability diff"
                )
        if (
            self.resolution
            is CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE
            and self.protocol_compatible
        ):
            raise VNextContractError(
                "DEFERRED_PROTOCOL_CHANGE requires a protocol incompatibility"
            )

    @property
    def resolution_id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True)
class CandidatePackageV1(_CanonicalVNextContract):
    research_spec_ref: str
    research_spec_digest: str
    protocol_ref: str
    protocol_digest: str
    source_tree_digest: str
    candidate_root_ref: str
    candidate_root_digest: str
    executable_entrypoint: str
    allowed_files: tuple[str, ...]
    dependency_identity_ref: str
    dependency_identity_digest: str
    runtime_identity_ref: str
    runtime_identity_digest: str
    implementation_receipt_ref: str
    implementation_receipt_digest: str
    origin_blind_projection_digest: str

    schema = "recclaw.research-line.vnext.candidate-package.v1"
    identity_namespace = "recclaw-candidate-package-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        for ref_field, digest_field in (
            ("research_spec_ref", "research_spec_digest"),
            ("protocol_ref", "protocol_digest"),
            ("candidate_root_ref", "candidate_root_digest"),
            ("dependency_identity_ref", "dependency_identity_digest"),
            ("runtime_identity_ref", "runtime_identity_digest"),
            ("implementation_receipt_ref", "implementation_receipt_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        object.__setattr__(
            self,
            "executable_entrypoint",
            _entrypoint(self.executable_entrypoint),
        )
        files = tuple(
            validate_relative_artifact_path(str(path)) for path in self.allowed_files
        )
        if not files or len(set(files)) != len(files):
            raise VNextContractError("allowed_files must be unique and non-empty")
        object.__setattr__(self, "allowed_files", tuple(sorted(files)))

    @property
    def package_id(self) -> str:
        return self.record_id


_QUALIFICATION_STAGE_FIELDS = (
    (QualificationStageV1.STATIC_VALIDATION, "static_result"),
    (QualificationStageV1.CONSTRUCTION, "construction_result"),
    (QualificationStageV1.API_CONTRACT, "api_contract_result"),
    (QualificationStageV1.UNIT, "unit_result"),
    (QualificationStageV1.ONE_EPOCH_SMOKE, "smoke_result"),
)


@dataclass(frozen=True, slots=True)
class QualificationReceiptV1(_CanonicalVNextContract):
    candidate_package_ref: str
    candidate_package_digest: str
    research_spec_ref: str
    research_spec_digest: str
    candidate_root_ref: str
    candidate_root_digest: str
    source_tree_digest: str
    runtime_identity_ref: str
    runtime_identity_digest: str
    protocol_ref: str
    protocol_digest: str
    stage: QualificationStageV1
    status: QualificationStatusV1
    failure_class: QualificationFailureClassV1
    static_result: QualificationCheckStatusV1
    construction_result: QualificationCheckStatusV1
    api_contract_result: QualificationCheckStatusV1
    unit_result: QualificationCheckStatusV1
    smoke_result: QualificationCheckStatusV1
    failure_detail_ref: str | None
    failure_detail_digest: str | None
    evidence_class: str = EVIDENCE_CLASS
    mechanism_belief_authority: str = NO_MECHANISM_BELIEF_AUTHORITY
    current_campaign_effect_evidence: bool = False

    schema = "recclaw.research-line.vnext.qualification-receipt.v1"
    identity_namespace = "recclaw-qualification-receipt-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        for ref_field, digest_field in (
            ("candidate_package_ref", "candidate_package_digest"),
            ("research_spec_ref", "research_spec_digest"),
            ("candidate_root_ref", "candidate_root_digest"),
            ("runtime_identity_ref", "runtime_identity_digest"),
            ("protocol_ref", "protocol_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        _optional_ref_digest(
            self.failure_detail_ref,
            self.failure_detail_digest,
            ref_field="failure_detail_ref",
            digest_field="failure_detail_digest",
        )
        if not isinstance(self.stage, QualificationStageV1):
            raise VNextContractError("stage is outside the qualification stage domain")
        if not isinstance(self.status, QualificationStatusV1):
            raise VNextContractError("status is outside the qualification status domain")
        if not isinstance(self.failure_class, QualificationFailureClassV1):
            raise VNextContractError(
                "failure_class is outside the qualification failure domain"
            )
        stage_index = next(
            index
            for index, (stage, _field_name) in enumerate(_QUALIFICATION_STAGE_FIELDS)
            if stage is self.stage
        )
        results = tuple(
            getattr(self, field_name) for _stage, field_name in _QUALIFICATION_STAGE_FIELDS
        )
        if any(not isinstance(item, QualificationCheckStatusV1) for item in results):
            raise VNextContractError("qualification checks use an unknown status")
        expected_terminal = (
            QualificationCheckStatusV1.PASS
            if self.status is QualificationStatusV1.PASS
            else QualificationCheckStatusV1.FAIL
        )
        if (
            any(
                item is not QualificationCheckStatusV1.PASS
                for item in results[:stage_index]
            )
            or results[stage_index] is not expected_terminal
            or any(
                item is not QualificationCheckStatusV1.NOT_RUN
                for item in results[stage_index + 1 :]
            )
        ):
            raise VNextContractError(
                "qualification results must form an ordered PASS prefix and one terminal stage"
            )
        if self.status is QualificationStatusV1.PASS:
            if (
                self.failure_class is not QualificationFailureClassV1.NONE
                or self.failure_detail_ref is not None
            ):
                raise VNextContractError("passing qualification cannot carry a failure")
        elif (
            self.failure_class is QualificationFailureClassV1.NONE
            or self.failure_detail_ref is None
        ):
            raise VNextContractError(
                "failed qualification requires a non-mechanism failure class and detail"
            )
        if (
            self.evidence_class != EVIDENCE_CLASS
            or self.mechanism_belief_authority != NO_MECHANISM_BELIEF_AUTHORITY
            or self.current_campaign_effect_evidence
        ):
            raise VNextContractError(
                "qualification is development-only and has no mechanism-belief authority"
            )

    @property
    def receipt_id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True)
class QualifiedCapabilityV1(_CanonicalVNextContract):
    capability_kind: CapabilityKindV1
    capability_version: str
    semantic_identity_ref: str
    semantic_identity_digest: str
    executable_entrypoint: str
    candidate_package_ref: str
    candidate_package_digest: str
    source_tree_digest: str
    qualification_receipt_ref: str
    qualification_receipt_digest: str
    qualification_stage: QualificationStageV1
    qualification_status: QualificationStatusV1
    protocol_ref: str
    protocol_digest: str
    compatibility_requirements: tuple[str, ...]
    predecessor_capability_ref: str | None
    predecessor_capability_digest: str | None
    current_campaign_ineligible: bool
    activation_boundary: str

    schema = "recclaw.research-line.vnext.qualified-capability.v1"
    identity_namespace = "recclaw-qualified-capability-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        if not isinstance(self.capability_kind, CapabilityKindV1):
            raise VNextContractError("capability_kind is outside the closed domain")
        _nonempty(self.capability_version, field_name="capability_version")
        for ref_field, digest_field in (
            ("semantic_identity_ref", "semantic_identity_digest"),
            ("candidate_package_ref", "candidate_package_digest"),
            ("qualification_receipt_ref", "qualification_receipt_digest"),
            ("protocol_ref", "protocol_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        _optional_ref_digest(
            self.predecessor_capability_ref,
            self.predecessor_capability_digest,
            ref_field="predecessor_capability_ref",
            digest_field="predecessor_capability_digest",
        )
        object.__setattr__(
            self,
            "executable_entrypoint",
            _entrypoint(self.executable_entrypoint),
        )
        object.__setattr__(
            self,
            "compatibility_requirements",
            _sorted_unique_strings(
                self.compatibility_requirements,
                field_name="compatibility_requirements",
            ),
        )
        if (
            self.qualification_stage is not QualificationStageV1.ONE_EPOCH_SMOKE
            or self.qualification_status is not QualificationStatusV1.PASS
        ):
            raise VNextContractError(
                "QualifiedCapability requires a passing one-epoch qualification receipt"
            )
        if (
            not self.current_campaign_ineligible
            or self.activation_boundary != NEXT_FRESH_CAMPAIGN
        ):
            raise VNextContractError(
                "QualifiedCapability cannot alter the current campaign"
            )

    @property
    def capability_id(self) -> str:
        return self.record_id


_ENGINEERING_FAILURES = frozenset(
    {
        ResearchFailureClassV1.IMPLEMENTATION,
        ResearchFailureClassV1.INTERFACE,
        ResearchFailureClassV1.RUNTIME,
        ResearchFailureClassV1.PROTOCOL,
        ResearchFailureClassV1.RESOURCE,
    }
)
_SCIENTIFIC_EVIDENCE_CLASSES = frozenset(
    {
        EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT,
        EpisodeEvidenceClassV1.FORMAL_EXPERIMENT,
    }
)


@dataclass(frozen=True, slots=True)
class TypedResearchEpisodeV1(_CanonicalVNextContract):
    campaign_id: str
    context_ref: str
    context_digest: str
    hypothesis: str
    executable_capability_ref: str
    executable_capability_digest: str
    executable_profile_ref: str
    executable_profile_digest: str
    experiment_binding_ref: str
    experiment_binding_digest: str
    comparator_ref: str | None
    comparator_digest: str | None
    outcome_ref: str
    outcome_digest: str
    cost_ref: str
    cost_digest: str
    protocol_ref: str
    protocol_digest: str
    evidence_class: EpisodeEvidenceClassV1
    experiment_executed: bool
    mechanism_interpretation: str
    competing_explanation: str
    failure_class: ResearchFailureClassV1
    mechanism_negative_evidence: bool
    next_discriminative_test: str
    qualification_receipt_ref: str | None
    qualification_receipt_digest: str | None
    qualification_evidence_used_as_scientific: bool

    schema = "recclaw.research-line.vnext.typed-research-episode.v1"
    identity_namespace = "recclaw-typed-research-episode-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        for field_name in (
            "campaign_id",
            "hypothesis",
            "mechanism_interpretation",
            "competing_explanation",
            "next_discriminative_test",
        ):
            _nonempty(getattr(self, field_name), field_name=field_name)
        for ref_field, digest_field in (
            ("context_ref", "context_digest"),
            ("executable_capability_ref", "executable_capability_digest"),
            ("executable_profile_ref", "executable_profile_digest"),
            ("experiment_binding_ref", "experiment_binding_digest"),
            ("outcome_ref", "outcome_digest"),
            ("cost_ref", "cost_digest"),
            ("protocol_ref", "protocol_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        _optional_ref_digest(
            self.comparator_ref,
            self.comparator_digest,
            ref_field="comparator_ref",
            digest_field="comparator_digest",
        )
        _optional_ref_digest(
            self.qualification_receipt_ref,
            self.qualification_receipt_digest,
            ref_field="qualification_receipt_ref",
            digest_field="qualification_receipt_digest",
        )
        if not isinstance(self.evidence_class, EpisodeEvidenceClassV1):
            raise VNextContractError("evidence_class is outside the episode domain")
        if not isinstance(self.failure_class, ResearchFailureClassV1):
            raise VNextContractError("failure_class is outside the research failure domain")
        if self.qualification_evidence_used_as_scientific:
            raise VNextContractError(
                "QualificationReceipt cannot be used as scientific effect evidence"
            )
        if self.failure_class in _ENGINEERING_FAILURES:
            if (
                self.evidence_class is not EpisodeEvidenceClassV1.ENGINEERING_ONLY
                or self.mechanism_interpretation != "NOT_ADJUDICATED"
                or self.mechanism_negative_evidence
            ):
                raise VNextContractError(
                    "engineering/interface/runtime failures cannot become mechanism evidence"
                )
        elif self.failure_class is ResearchFailureClassV1.MECHANISM:
            if (
                not self.experiment_executed
                or self.evidence_class not in _SCIENTIFIC_EVIDENCE_CLASSES
                or self.comparator_ref is None
                or not self.mechanism_negative_evidence
            ):
                raise VNextContractError(
                    "mechanism-negative evidence requires a real compared experiment"
                )
        elif self.failure_class is ResearchFailureClassV1.NONE:
            if (
                not self.experiment_executed
                or self.evidence_class not in _SCIENTIFIC_EVIDENCE_CLASSES
                or self.comparator_ref is None
                or self.mechanism_negative_evidence
            ):
                raise VNextContractError(
                    "successful scientific episodes require a real comparator outcome"
                )
        elif self.failure_class is ResearchFailureClassV1.INCONCLUSIVE:
            if (
                not self.experiment_executed
                or self.evidence_class
                is not EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT
                or self.comparator_ref is None
                or self.mechanism_interpretation != "NOT_ADJUDICATED"
                or self.mechanism_negative_evidence
            ):
                raise VNextContractError(
                    "inconclusive evidence must remain non-mechanistic"
                )

    @property
    def episode_id(self) -> str:
        return self.record_id


def _normalize_executable_entries(
    values: tuple[tuple[str, str, str], ...],
) -> tuple[tuple[str, str, str], ...]:
    normalized: list[tuple[str, str, str]] = []
    for index, entry in enumerate(values):
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise VNextContractError(
                f"executable_entries[{index}] must be a (ref, digest, entrypoint) tuple"
            )
        ref = _nonempty(str(entry[0]), field_name=f"executable_entries[{index}].ref")
        digest = validate_sha256(
            str(entry[1]),
            field_name=f"executable_entries[{index}].digest",
        )
        entrypoint = _entrypoint(
            str(entry[2]),
            field_name=f"executable_entries[{index}].entrypoint",
        )
        normalized.append((ref, digest, entrypoint))
    if not normalized:
        raise VNextContractError("executable_entries must be non-empty")
    if len({ref for ref, _digest, _entrypoint in normalized}) != len(normalized):
        raise VNextContractError("executable capability refs must be unique")
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True, slots=True)
class ExecutableProfileVNext(_CanonicalVNextContract):
    profile_version: str
    predecessor_profile_ref: str
    predecessor_profile_digest: str
    registry_ref: str
    registry_digest: str
    executable_entries: tuple[tuple[str, str, str], ...]
    protocol_ref: str
    protocol_digest: str
    compatibility_requirements: tuple[str, ...]
    current_campaign_eligible: bool
    activation_boundary: str

    schema = "recclaw.research-line.vnext.executable-profile.rc0"
    identity_namespace = "recclaw-executable-profile-vnext"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        _nonempty(self.profile_version, field_name="profile_version")
        for ref_field, digest_field in (
            ("predecessor_profile_ref", "predecessor_profile_digest"),
            ("registry_ref", "registry_digest"),
            ("protocol_ref", "protocol_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        object.__setattr__(
            self,
            "executable_entries",
            _normalize_executable_entries(self.executable_entries),
        )
        object.__setattr__(
            self,
            "compatibility_requirements",
            _sorted_unique_strings(
                self.compatibility_requirements,
                field_name="compatibility_requirements",
            ),
        )
        if self.current_campaign_eligible or self.activation_boundary != NEXT_FRESH_CAMPAIGN:
            raise VNextContractError(
                "a vNext profile activates only in a next fresh campaign"
            )

    @property
    def profile_id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True)
class AcquisitionDecisionV1(_CanonicalVNextContract):
    stage: AcquisitionStageV1
    subject_kind: str
    subject_ref: str
    subject_digest: str
    policy_ref: str
    policy_digest: str
    context_ref: str
    context_digest: str
    protocol_ref: str
    protocol_digest: str
    feature_schema_ref: str
    feature_schema_digest: str
    feature_snapshot_digest: str
    budget_schema_ref: str
    budget_schema_digest: str
    budget_snapshot_digest: str
    eligibility_schema_ref: str
    eligibility_schema_digest: str
    eligibility_snapshot_digest: str
    disposition: AcquisitionDispositionV1
    reason_codes: tuple[str, ...]
    cross_domain_schema_reuse: bool

    schema = "recclaw.research-line.vnext.acquisition-decision.v1"
    identity_namespace = "recclaw-acquisition-decision-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        if not isinstance(self.stage, AcquisitionStageV1):
            raise VNextContractError("stage must be IDEA or EXPERIMENT")
        if not isinstance(self.disposition, AcquisitionDispositionV1):
            raise VNextContractError("disposition is outside the acquisition domain")
        expected_subject = (
            "OPEN_RESEARCH_SPEC"
            if self.stage is AcquisitionStageV1.IDEA
            else "EXECUTABLE_CAPABILITY"
        )
        if self.subject_kind != expected_subject:
            raise VNextContractError(
                f"{self.stage.value} acquisition requires subject_kind={expected_subject}"
            )
        for ref_field, digest_field in (
            ("subject_ref", "subject_digest"),
            ("policy_ref", "policy_digest"),
            ("context_ref", "context_digest"),
            ("protocol_ref", "protocol_digest"),
            ("feature_schema_ref", "feature_schema_digest"),
            ("budget_schema_ref", "budget_schema_digest"),
            ("eligibility_schema_ref", "eligibility_schema_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        if len(
            {
                self.feature_schema_digest,
                self.budget_schema_digest,
                self.eligibility_schema_digest,
            }
        ) != 3:
            raise VNextContractError(
                "feature, budget, and eligibility schemas are distinct domains"
            )
        object.__setattr__(
            self,
            "reason_codes",
            _sorted_unique_strings(self.reason_codes, field_name="reason_codes"),
        )
        if self.cross_domain_schema_reuse:
            raise VNextContractError(
                "Idea and Experiment acquisition may not reuse the other domain's schema"
            )

    @property
    def decision_id(self) -> str:
        return self.record_id


@dataclass(frozen=True, slots=True)
class ProfileBuildReceiptV1(_CanonicalVNextContract):
    predecessor_profile_ref: str
    predecessor_profile_hash: str
    registry_ref: str
    registry_digest: str
    qualified_registry_refs: tuple[tuple[str, str], ...]
    new_profile_ref: str
    new_profile_hash: str
    build_policy_ref: str
    build_policy_digest: str
    protocol_ref: str
    protocol_digest: str
    current_profile_unchanged: bool
    deterministic_rebuild: bool
    activation_boundary: str

    schema = "recclaw.research-line.vnext.profile-build-receipt.v1"
    identity_namespace = "recclaw-profile-build-receipt-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        for ref_field, digest_field in (
            ("predecessor_profile_ref", "predecessor_profile_hash"),
            ("registry_ref", "registry_digest"),
            ("new_profile_ref", "new_profile_hash"),
            ("build_policy_ref", "build_policy_digest"),
            ("protocol_ref", "protocol_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        object.__setattr__(
            self,
            "qualified_registry_refs",
            _normalize_ref_digest_pairs(
                self.qualified_registry_refs,
                field_name="qualified_registry_refs",
            ),
        )
        if self.predecessor_profile_hash == self.new_profile_hash:
            raise VNextContractError(
                "a profile growth receipt must bind a distinct successor profile"
            )
        if (
            not self.current_profile_unchanged
            or not self.deterministic_rebuild
            or self.activation_boundary != NEXT_FRESH_CAMPAIGN
        ):
            raise VNextContractError(
                "profile construction must preserve the current profile and activate later"
            )

    @property
    def build_input_digest(self) -> str:
        return sha256_digest(
            {
                "predecessor_profile_hash": self.predecessor_profile_hash,
                "registry_digest": self.registry_digest,
                "qualified_registry_refs": self.qualified_registry_refs,
                "build_policy_digest": self.build_policy_digest,
                "protocol_digest": self.protocol_digest,
            }
        )

    @property
    def receipt_id(self) -> str:
        return self.record_id


__all__ = [
    "AcquisitionDecisionV1",
    "AcquisitionDispositionV1",
    "AcquisitionStageV1",
    "CandidatePackageV1",
    "CapabilityKindV1",
    "CapabilityResolutionResultV1",
    "CapabilityResolutionV1",
    "CurrentProfileExpressibilityV1",
    "EpisodeEvidenceClassV1",
    "ExecutableProfileVNext",
    "NEXT_FRESH_CAMPAIGN",
    "OpenResearchSpecV1",
    "ProfileBuildReceiptV1",
    "QualificationCheckStatusV1",
    "QualificationFailureClassV1",
    "QualificationReceiptV1",
    "QualificationStageV1",
    "QualificationStatusV1",
    "QualifiedCapabilityV1",
    "ResearchFailureClassV1",
    "TypedResearchEpisodeV1",
    "VNextContractError",
]
