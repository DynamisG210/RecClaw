"""D1 mechanical adapter from closed vNext Episodes to existing beliefs.

The adapter has no writer capability.  It revalidates the D0 memory boundary
and returns the existing eight-field ``DevelopmentalMechanismBeliefV1`` that
the established Search Memory consumer already accepts.
"""

from __future__ import annotations

import re
from enum import Enum

from .research_contracts import DevelopmentalMechanismBeliefV1
from .scientific_episode import (
    EpisodeClosureStatusV1,
    EpisodeMemoryLaneV1,
    FrozenComparisonIdentityV1,
    ScientificEpisodeClosureV1,
)
from .vnext_contracts import (
    EpisodeEvidenceClassV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)


class ScientificEpisodeAdapterReasonV1(str, Enum):
    INVALID_INPUT_TYPE = "INVALID_INPUT_TYPE"
    NOT_MECHANISM_MEMORY = "NOT_MECHANISM_MEMORY"
    MECHANISM_MEMORY_PERMISSION_DENIED = (
        "MECHANISM_MEMORY_PERMISSION_DENIED"
    )
    SCIENTIFIC_FAILURE_NOT_ADMITTED = "SCIENTIFIC_FAILURE_NOT_ADMITTED"
    EPISODE_NOT_EXECUTED = "EPISODE_NOT_EXECUTED"
    EPISODE_IDENTITY_MISMATCH = "EPISODE_IDENTITY_MISMATCH"
    COMPARISON_IDENTITY_MISMATCH = "COMPARISON_IDENTITY_MISMATCH"
    OUTCOME_IDENTITY_MISMATCH = "OUTCOME_IDENTITY_MISMATCH"
    EVIDENCE_IDENTITY_MISMATCH = "EVIDENCE_IDENTITY_MISMATCH"
    QUALIFICATION_EVIDENCE_MISUSED = "QUALIFICATION_EVIDENCE_MISUSED"
    CLOSURE_STAGE_INCOMPLETE = "CLOSURE_STAGE_INCOMPLETE"
    MECHANISM_AXIS_INVALID = "MECHANISM_AXIS_INVALID"


class ScientificEpisodeAdapterError(ValueError):
    def __init__(
        self,
        reason_code: ScientificEpisodeAdapterReasonV1,
        detail: str,
    ) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code.value}: {detail}")


_SCIENTIFIC_FAILURES = frozenset(
    {
        ResearchFailureClassV1.NONE,
        ResearchFailureClassV1.MECHANISM,
    }
)

_SCIENTIFIC_EVIDENCE = frozenset(
    {
        EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT,
        EpisodeEvidenceClassV1.FORMAL_EXPERIMENT,
    }
)

_CLOSURE_STAGE_FIELDS = (
    "identity_result",
    "protocol_result",
    "package_result",
    "interface_result",
    "execution_result",
    "outcome_result",
    "comparator_result",
    "evidence_result",
)

_COMPARISON_FIELDS = (
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
)

_MECHANISM_AXIS_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ORIGIN_AXIS_TOKENS = frozenset(
    {
        "arm",
        "instance",
        "origin",
        "owner",
        "producer",
        "source",
    }
)


def _reject(
    reason_code: ScientificEpisodeAdapterReasonV1,
    detail: str,
) -> None:
    raise ScientificEpisodeAdapterError(reason_code, detail)


def _validate_input_types(
    comparison_identity: object,
    closure: object,
    episode: object,
) -> None:
    expected = (
        (comparison_identity, FrozenComparisonIdentityV1),
        (closure, ScientificEpisodeClosureV1),
        (episode, TypedResearchEpisodeV1),
    )
    invalid = tuple(
        expected_type.__name__
        for value, expected_type in expected
        if not isinstance(value, expected_type)
    )
    if invalid:
        _reject(
            ScientificEpisodeAdapterReasonV1.INVALID_INPUT_TYPE,
            "required=" + ",".join(invalid),
        )


def _validate_mechanism_axis(mechanism_axis: str) -> str:
    if (
        not isinstance(mechanism_axis, str)
        or _MECHANISM_AXIS_PATTERN.fullmatch(mechanism_axis) is None
        or not _ORIGIN_AXIS_TOKENS.isdisjoint(mechanism_axis.split("_"))
    ):
        _reject(
            ScientificEpisodeAdapterReasonV1.MECHANISM_AXIS_INVALID,
            "axis must be normalized and source/origin blind",
        )
    return mechanism_axis


def project_episode_to_mechanism_belief(
    *,
    comparison_identity: FrozenComparisonIdentityV1,
    closure: ScientificEpisodeClosureV1,
    episode: TypedResearchEpisodeV1,
    mechanism_axis: str,
) -> DevelopmentalMechanismBeliefV1:
    """Return one existing Search-Memory belief or fail with a reason code."""

    _validate_input_types(comparison_identity, closure, episode)

    if closure.memory_lane is not EpisodeMemoryLaneV1.MECHANISM_MEMORY:
        _reject(
            ScientificEpisodeAdapterReasonV1.NOT_MECHANISM_MEMORY,
            f"observed_lane={closure.memory_lane.value}",
        )
    if not closure.mechanism_memory_allowed:
        _reject(
            ScientificEpisodeAdapterReasonV1.MECHANISM_MEMORY_PERMISSION_DENIED,
            "D0 closure did not grant mechanism memory",
        )
    if (
        closure.failure_class not in _SCIENTIFIC_FAILURES
        or episode.failure_class not in _SCIENTIFIC_FAILURES
        or closure.failure_class is not episode.failure_class
    ):
        _reject(
            ScientificEpisodeAdapterReasonV1.SCIENTIFIC_FAILURE_NOT_ADMITTED,
            (
                f"closure={closure.failure_class.value};"
                f"episode={episode.failure_class.value}"
            ),
        )
    if not episode.experiment_executed:
        _reject(
            ScientificEpisodeAdapterReasonV1.EPISODE_NOT_EXECUTED,
            "TypedResearchEpisode must bind a real execution",
        )
    if (
        closure.episode_ref != episode.episode_id
        or closure.episode_digest != episode.digest
    ):
        _reject(
            ScientificEpisodeAdapterReasonV1.EPISODE_IDENTITY_MISMATCH,
            "closure Episode ref/digest does not match the supplied Episode",
        )
    if (
        closure.comparison_identity_ref != comparison_identity.identity_id
        or closure.comparison_identity_digest != comparison_identity.digest
    ):
        _reject(
            ScientificEpisodeAdapterReasonV1.COMPARISON_IDENTITY_MISMATCH,
            "closure does not bind the supplied frozen comparison identity",
        )
    comparison_mismatches = tuple(
        field_name
        for field_name in _COMPARISON_FIELDS
        if getattr(comparison_identity, field_name)
        != getattr(episode, field_name)
    )
    if comparison_mismatches:
        _reject(
            ScientificEpisodeAdapterReasonV1.COMPARISON_IDENTITY_MISMATCH,
            "fields=" + ",".join(comparison_mismatches),
        )
    if (
        closure.outcome_ref != episode.outcome_ref
        or closure.outcome_digest != episode.outcome_digest
    ):
        _reject(
            ScientificEpisodeAdapterReasonV1.OUTCOME_IDENTITY_MISMATCH,
            "closure outcome ref/digest does not match the supplied Episode",
        )
    if (
        closure.evidence_class not in _SCIENTIFIC_EVIDENCE
        or episode.evidence_class not in _SCIENTIFIC_EVIDENCE
        or closure.evidence_class is not episode.evidence_class
    ):
        _reject(
            ScientificEpisodeAdapterReasonV1.EVIDENCE_IDENTITY_MISMATCH,
            (
                f"closure={closure.evidence_class.value};"
                f"episode={episode.evidence_class.value}"
            ),
        )
    if episode.qualification_evidence_used_as_scientific:
        _reject(
            ScientificEpisodeAdapterReasonV1.QUALIFICATION_EVIDENCE_MISUSED,
            "QualificationReceipt cannot supply scientific evidence",
        )
    incomplete_stages = tuple(
        field_name
        for field_name in _CLOSURE_STAGE_FIELDS
        if getattr(closure, field_name) is not EpisodeClosureStatusV1.PASS
    )
    if incomplete_stages:
        _reject(
            ScientificEpisodeAdapterReasonV1.CLOSURE_STAGE_INCOMPLETE,
            "fields=" + ",".join(incomplete_stages),
        )
    mechanism_axis = _validate_mechanism_axis(mechanism_axis)

    observation = (
        "typed_episode_outcome:"
        + episode.outcome_digest
        + ":closure:"
        + closure.digest
    )
    mechanism_negative = (
        episode.failure_class is ResearchFailureClassV1.MECHANISM
    )
    unresolved_confounds = (
        ("development_experiment_claim_ceiling",)
        if episode.evidence_class
        is EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT
        else ()
    )
    return DevelopmentalMechanismBeliefV1(
        hypothesis_id=episode.episode_id,
        mechanism_axis=mechanism_axis,
        competing_hypotheses=(
            episode.hypothesis,
            episode.competing_explanation,
        ),
        predicted_outcome_signature=episode.hypothesis,
        evidence_for=() if mechanism_negative else (observation,),
        evidence_against=(observation,) if mechanism_negative else (),
        unresolved_confounds=unresolved_confounds,
        next_discriminative_test=episode.next_discriminative_test,
    )


__all__ = [
    "ScientificEpisodeAdapterError",
    "ScientificEpisodeAdapterReasonV1",
    "project_episode_to_mechanism_belief",
]
