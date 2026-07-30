"""Pure D0 closure semantics for scientific comparisons and diagnostics.

``TypedResearchEpisodeV1`` remains the sole scientific Episode contract.  This
module closes either an executed comparison into that Episode lane or a
non-scientific terminal path into a diagnostic receipt.  It does not write
memory, invoke an interpreter, or own runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from .canonical import canonical_json_bytes, canonical_value, content_id, sha256_digest
from .vnext_contracts import (
    EpisodeEvidenceClassV1,
    QualificationReceiptV1,
    QualificationStageV1,
    QualificationStatusV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
    VNextContractError,
    _nonempty,
    _normalize_digest_fields,
    _optional_ref_digest,
    _required_ref_digest,
)


class EpisodeClosureStatusV1(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


class EpisodeMemoryLaneV1(str, Enum):
    MECHANISM_MEMORY = "MECHANISM_MEMORY"
    ENGINEERING_DIAGNOSTIC = "ENGINEERING_DIAGNOSTIC"
    NONE = "NONE"


class _CanonicalEpisodeCoreContract:
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


@dataclass(frozen=True, slots=True)
class FrozenComparisonIdentityV1(_CanonicalEpisodeCoreContract):
    """The minimum pre-execution identity needed to close one comparison."""

    campaign_id: str
    context_ref: str
    context_digest: str
    executable_capability_ref: str
    executable_capability_digest: str
    executable_profile_ref: str
    executable_profile_digest: str
    experiment_binding_ref: str
    experiment_binding_digest: str
    comparator_ref: str
    comparator_digest: str
    protocol_ref: str
    protocol_digest: str

    schema = "recclaw.research-line.vnext.frozen-comparison-identity.v1"
    identity_namespace = "recclaw-frozen-comparison-identity-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        _nonempty(self.campaign_id, field_name="campaign_id")
        for ref_field, digest_field in (
            ("context_ref", "context_digest"),
            ("executable_capability_ref", "executable_capability_digest"),
            ("executable_profile_ref", "executable_profile_digest"),
            ("experiment_binding_ref", "experiment_binding_digest"),
            ("comparator_ref", "comparator_digest"),
            ("protocol_ref", "protocol_digest"),
        ):
            _required_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )

    @property
    def identity_id(self) -> str:
        return self.record_id


_STAGE_FIELDS = (
    "identity_result",
    "protocol_result",
    "package_result",
    "interface_result",
    "execution_result",
    "outcome_result",
    "comparator_result",
    "evidence_result",
)

_TERMINAL_STAGE_BY_FAILURE = {
    ResearchFailureClassV1.IDENTITY_DRIFT: 0,
    ResearchFailureClassV1.PROTOCOL: 1,
    ResearchFailureClassV1.IMPLEMENTATION: 2,
    ResearchFailureClassV1.PACKAGE: 2,
    ResearchFailureClassV1.INTERFACE: 3,
    ResearchFailureClassV1.RUNTIME: 4,
    ResearchFailureClassV1.RESOURCE: 4,
    ResearchFailureClassV1.PROVIDER: 4,
    ResearchFailureClassV1.OUTCOME_MISSING: 5,
}

_SCIENTIFIC_TERMINALS = frozenset(
    {
        ResearchFailureClassV1.NONE,
        ResearchFailureClassV1.MECHANISM,
        ResearchFailureClassV1.INCONCLUSIVE,
    }
)

_MECHANISM_MEMORY_FAILURE_CLASSES = frozenset(
    {
        ResearchFailureClassV1.NONE,
        ResearchFailureClassV1.MECHANISM,
    }
)

_MECHANISM_MEMORY_EVIDENCE_CLASSES = frozenset(
    {
        EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT,
        EpisodeEvidenceClassV1.FORMAL_EXPERIMENT,
    }
)


@dataclass(frozen=True, slots=True)
class ScientificEpisodeClosureV1(_CanonicalEpisodeCoreContract):
    comparison_identity_ref: str
    comparison_identity_digest: str
    failure_class: ResearchFailureClassV1
    evidence_class: EpisodeEvidenceClassV1
    episode_ref: str | None
    episode_digest: str | None
    qualification_receipt_ref: str | None
    qualification_receipt_digest: str | None
    failure_detail_ref: str | None
    failure_detail_digest: str | None
    outcome_ref: str | None
    outcome_digest: str | None
    identity_result: EpisodeClosureStatusV1
    protocol_result: EpisodeClosureStatusV1
    package_result: EpisodeClosureStatusV1
    interface_result: EpisodeClosureStatusV1
    execution_result: EpisodeClosureStatusV1
    outcome_result: EpisodeClosureStatusV1
    comparator_result: EpisodeClosureStatusV1
    evidence_result: EpisodeClosureStatusV1
    memory_lane: EpisodeMemoryLaneV1
    mechanism_memory_allowed: bool
    engineering_diagnostic_allowed: bool
    reason_codes: tuple[str, ...]

    schema = "recclaw.research-line.vnext.scientific-episode-closure.v1"
    identity_namespace = "recclaw-scientific-episode-closure-v1"

    def __post_init__(self) -> None:
        _normalize_digest_fields(self)
        _required_ref_digest(
            self.comparison_identity_ref,
            self.comparison_identity_digest,
            ref_field="comparison_identity_ref",
            digest_field="comparison_identity_digest",
        )
        for ref_field, digest_field in (
            ("episode_ref", "episode_digest"),
            ("qualification_receipt_ref", "qualification_receipt_digest"),
            ("failure_detail_ref", "failure_detail_digest"),
            ("outcome_ref", "outcome_digest"),
        ):
            _optional_ref_digest(
                getattr(self, ref_field),
                getattr(self, digest_field),
                ref_field=ref_field,
                digest_field=digest_field,
            )
        if not isinstance(self.failure_class, ResearchFailureClassV1):
            raise VNextContractError("failure_class is outside the D0 failure domain")
        if not isinstance(self.evidence_class, EpisodeEvidenceClassV1):
            raise VNextContractError("evidence_class is outside the D0 evidence domain")
        if not isinstance(self.memory_lane, EpisodeMemoryLaneV1):
            raise VNextContractError("memory_lane is outside the D0 memory domain")
        stage_results = tuple(getattr(self, name) for name in _STAGE_FIELDS)
        if any(not isinstance(item, EpisodeClosureStatusV1) for item in stage_results):
            raise VNextContractError("closure stages use an unknown status")
        terminal_index = _TERMINAL_STAGE_BY_FAILURE.get(self.failure_class)
        if terminal_index is None:
            if any(item is not EpisodeClosureStatusV1.PASS for item in stage_results):
                raise VNextContractError(
                    "a scientific terminal requires every closure stage to pass"
                )
        elif (
            any(
                item is not EpisodeClosureStatusV1.PASS
                for item in stage_results[:terminal_index]
            )
            or stage_results[terminal_index] is not EpisodeClosureStatusV1.FAIL
            or any(
                item is not EpisodeClosureStatusV1.NOT_RUN
                for item in stage_results[terminal_index + 1 :]
            )
        ):
            raise VNextContractError(
                "diagnostic closure requires a PASS prefix, one FAIL, and later NOT_RUN"
            )
        if (
            not self.reason_codes
            or tuple(sorted(set(self.reason_codes))) != self.reason_codes
        ):
            raise VNextContractError(
                "reason_codes must be sorted, unique, and non-empty"
            )
        scientific = self.failure_class in _SCIENTIFIC_TERMINALS
        if scientific:
            if (
                self.episode_ref is None
                or self.outcome_ref is None
                or self.failure_detail_ref is not None
                or self.evidence_class is EpisodeEvidenceClassV1.ENGINEERING_ONLY
            ):
                raise VNextContractError(
                    "scientific closure requires an Episode and observed outcome only"
                )
            if (
                self.failure_class in _MECHANISM_MEMORY_FAILURE_CLASSES
                and self.evidence_class
                not in _MECHANISM_MEMORY_EVIDENCE_CLASSES
            ) or (
                self.failure_class is ResearchFailureClassV1.INCONCLUSIVE
                and self.evidence_class
                is not EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT
            ):
                raise VNextContractError(
                    "scientific terminal and evidence class disagree"
                )
        elif (
            self.episode_ref is not None
            or self.outcome_ref is not None
            or self.failure_detail_ref is None
            or self.evidence_class is not EpisodeEvidenceClassV1.ENGINEERING_ONLY
        ):
            raise VNextContractError(
                "diagnostic closure preserves missing outcome and carries no Episode"
            )
        expected_mechanism_permission = (
            self.failure_class in _MECHANISM_MEMORY_FAILURE_CLASSES
            and self.evidence_class in _MECHANISM_MEMORY_EVIDENCE_CLASSES
        )
        if (
            self.mechanism_memory_allowed != expected_mechanism_permission
            or self.engineering_diagnostic_allowed != (not scientific)
        ):
            raise VNextContractError(
                "memory permission flags disagree with closure semantics"
            )
        expected_lane = (
            EpisodeMemoryLaneV1.MECHANISM_MEMORY
            if expected_mechanism_permission
            else (
                EpisodeMemoryLaneV1.ENGINEERING_DIAGNOSTIC
                if not scientific
                else EpisodeMemoryLaneV1.NONE
            )
        )
        if self.memory_lane is not expected_lane:
            raise VNextContractError("memory lane disagrees with closure semantics")

    @property
    def closure_id(self) -> str:
        return self.record_id


def _identity_mismatches(
    expected: FrozenComparisonIdentityV1,
    episode: TypedResearchEpisodeV1,
) -> tuple[str, ...]:
    mismatches = []
    for field_name in (
        "campaign_id",
        "context_ref",
        "context_digest",
        "executable_capability_ref",
        "executable_capability_digest",
        "executable_profile_ref",
        "executable_profile_digest",
        "experiment_binding_ref",
        "experiment_binding_digest",
        "comparator_ref",
        "comparator_digest",
        "protocol_ref",
        "protocol_digest",
    ):
        if getattr(expected, field_name) != getattr(episode, field_name):
            mismatches.append(field_name)
    return tuple(mismatches)


def _stage_results(
    failure_class: ResearchFailureClassV1,
) -> dict[str, EpisodeClosureStatusV1]:
    terminal_index = _TERMINAL_STAGE_BY_FAILURE.get(failure_class)
    if terminal_index is None:
        return {name: EpisodeClosureStatusV1.PASS for name in _STAGE_FIELDS}
    return {
        name: (
            EpisodeClosureStatusV1.PASS
            if index < terminal_index
            else (
                EpisodeClosureStatusV1.FAIL
                if index == terminal_index
                else EpisodeClosureStatusV1.NOT_RUN
            )
        )
        for index, name in enumerate(_STAGE_FIELDS)
    }


def _reason_codes(
    failure_class: ResearchFailureClassV1,
    memory_lane: EpisodeMemoryLaneV1,
) -> tuple[str, ...]:
    reasons = {
        (
            "SCIENTIFIC_COMPARISON_COMPLETE"
            if failure_class is ResearchFailureClassV1.NONE
            else (
                "MECHANISM_NEGATIVE_COMPARISON_COMPLETE"
                if failure_class is ResearchFailureClassV1.MECHANISM
                else (
                    "INCONCLUSIVE_SCIENTIFIC_EVIDENCE"
                    if failure_class is ResearchFailureClassV1.INCONCLUSIVE
                    else f"{failure_class.value}_FAILURE"
                )
            )
        )
    }
    if memory_lane is EpisodeMemoryLaneV1.MECHANISM_MEMORY:
        reasons.add("MECHANISM_MEMORY_ALLOWED")
    elif memory_lane is EpisodeMemoryLaneV1.ENGINEERING_DIAGNOSTIC:
        reasons.add("ENGINEERING_DIAGNOSTIC_ONLY")
    else:
        reasons.add("MEMORY_WRITE_DENIED")
    return tuple(sorted(reasons))


def close_scientific_episode(
    *,
    comparison_identity: FrozenComparisonIdentityV1,
    failure_class: ResearchFailureClassV1,
    episode: TypedResearchEpisodeV1 | None,
    observed_outcome_ref: str | None,
    observed_outcome_digest: str | None,
    qualification_receipt: QualificationReceiptV1 | None = None,
    failure_detail_ref: str | None = None,
    failure_detail_digest: str | None = None,
) -> ScientificEpisodeClosureV1:
    """Close one scientific comparison attempt without writing any memory."""

    if not isinstance(comparison_identity, FrozenComparisonIdentityV1):
        raise VNextContractError(
            "comparison_identity must use FrozenComparisonIdentityV1"
        )
    if not isinstance(failure_class, ResearchFailureClassV1):
        raise VNextContractError("failure_class is outside the D0 failure domain")
    _optional_ref_digest(
        observed_outcome_ref,
        observed_outcome_digest,
        ref_field="observed_outcome_ref",
        digest_field="observed_outcome_digest",
    )
    _optional_ref_digest(
        failure_detail_ref,
        failure_detail_digest,
        ref_field="failure_detail_ref",
        digest_field="failure_detail_digest",
    )

    scientific = failure_class in _SCIENTIFIC_TERMINALS
    if scientific:
        if episode is None:
            raise VNextContractError(
                "scientific closure requires TypedResearchEpisodeV1"
            )
        if episode.failure_class is not failure_class:
            raise VNextContractError("closure and Episode failure classes must match")
        mismatches = _identity_mismatches(comparison_identity, episode)
        if mismatches:
            raise VNextContractError(
                "scientific comparison identity drift: " + ", ".join(mismatches)
            )
        if (
            observed_outcome_ref is None
            or observed_outcome_ref != episode.outcome_ref
            or observed_outcome_digest != episode.outcome_digest
        ):
            raise VNextContractError("observed outcome identity drift")
        if failure_detail_ref is not None:
            raise VNextContractError(
                "scientific Episode carries interpretation, not diagnostic "
                "failure detail"
            )
        evidence_class = episode.evidence_class
        episode_ref = episode.episode_id
        episode_digest = episode.digest
        qualification_ref = episode.qualification_receipt_ref
        qualification_digest = episode.qualification_receipt_digest
        if qualification_ref is not None:
            if qualification_receipt is None:
                raise VNextContractError(
                    "referenced qualification receipt must be supplied for closure"
                )
            if (
                qualification_ref != qualification_receipt.receipt_id
                or qualification_digest != qualification_receipt.digest
            ):
                raise VNextContractError("Episode qualification receipt identity drift")
        elif qualification_receipt is not None:
            raise VNextContractError(
                "qualification receipt was supplied but the Episode does not "
                "reference it"
            )
        if qualification_receipt is not None and (
            qualification_receipt.status is not QualificationStatusV1.PASS
            or qualification_receipt.stage is not QualificationStageV1.ONE_EPOCH_SMOKE
        ):
            raise VNextContractError(
                "scientific closure may cite only a passing one-epoch qualification"
            )
        mechanism_allowed = (
            failure_class in _MECHANISM_MEMORY_FAILURE_CLASSES
            and evidence_class in _MECHANISM_MEMORY_EVIDENCE_CLASSES
        )
        memory_lane = (
            EpisodeMemoryLaneV1.MECHANISM_MEMORY
            if mechanism_allowed
            else EpisodeMemoryLaneV1.NONE
        )
    else:
        if failure_class not in _TERMINAL_STAGE_BY_FAILURE:
            raise VNextContractError("failure class has no diagnostic terminal stage")
        if episode is not None:
            raise VNextContractError(
                "non-scientific failures cannot create TypedResearchEpisodeV1"
            )
        if observed_outcome_ref is not None:
            raise VNextContractError(
                "diagnostic closure must preserve the outcome as missing"
            )
        if failure_detail_ref is None:
            raise VNextContractError("diagnostic closure requires failure detail")
        evidence_class = EpisodeEvidenceClassV1.ENGINEERING_ONLY
        episode_ref = None
        episode_digest = None
        qualification_ref = (
            qualification_receipt.receipt_id
            if qualification_receipt is not None
            else None
        )
        qualification_digest = (
            qualification_receipt.digest
            if qualification_receipt is not None
            else None
        )
        mechanism_allowed = False
        memory_lane = EpisodeMemoryLaneV1.ENGINEERING_DIAGNOSTIC

    return ScientificEpisodeClosureV1(
        comparison_identity_ref=comparison_identity.identity_id,
        comparison_identity_digest=comparison_identity.digest,
        failure_class=failure_class,
        evidence_class=evidence_class,
        episode_ref=episode_ref,
        episode_digest=episode_digest,
        qualification_receipt_ref=qualification_ref,
        qualification_receipt_digest=qualification_digest,
        failure_detail_ref=failure_detail_ref,
        failure_detail_digest=failure_detail_digest,
        outcome_ref=observed_outcome_ref,
        outcome_digest=observed_outcome_digest,
        memory_lane=memory_lane,
        mechanism_memory_allowed=mechanism_allowed,
        engineering_diagnostic_allowed=not scientific,
        reason_codes=_reason_codes(failure_class, memory_lane),
        **_stage_results(failure_class),
    )


__all__ = [
    "EpisodeClosureStatusV1",
    "EpisodeMemoryLaneV1",
    "FrozenComparisonIdentityV1",
    "ScientificEpisodeClosureV1",
    "close_scientific_episode",
]
