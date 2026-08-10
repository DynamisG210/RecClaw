"""The minimal four-Producer edge for one Research Line round."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias

from recclaw_core.experiments.helix_abc_v1.open_spec import (
    OpenSpecProjectionError,
    project_candidate_proposal_v4,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV4,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import OpenResearchSpecV1

from .interfaces import ProducerOutcome, ResearchContext


ProducerResult: TypeAlias = CandidateProposalV4 | Mapping[str, Any]


class ResearchProducer(Protocol):
    """Injectable external Producer boundary; Provider invocation stays outside."""

    def __call__(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
    ) -> ProducerResult:
        ...


_BINDING_FIELDS = (
    "context_ref",
    "context_digest",
    "protocol_ref",
    "protocol_digest",
    "current_profile_ref",
    "current_profile_digest",
)


def _validate_bindings(
    context: ResearchContext,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping")

    supplied = dict(bindings)
    expected = {
        "context_ref": context.context_ref,
        "context_digest": context.digest,
        "protocol_ref": context.protocol_ref,
        "protocol_digest": context.protocol_digest,
        "current_profile_ref": context.active_profile_ref,
        "current_profile_digest": context.active_profile_digest,
    }
    missing = [field for field in _BINDING_FIELDS if field not in supplied]
    if missing:
        raise ValueError(
            "bindings missing required OpenSpec identity fields: "
            + ", ".join(missing)
        )
    mismatched = [
        field for field in _BINDING_FIELDS if supplied[field] != expected[field]
    ]
    if mismatched:
        raise ValueError(
            "bindings must match ResearchContext for: "
            + ", ".join(mismatched)
        )
    return supplied


def _project_result(
    producer_role: str,
    result: ProducerResult,
    *,
    bindings: Mapping[str, Any],
) -> tuple[OpenResearchSpecV1, dict[str, Any], CandidateProposalV4 | None]:
    if isinstance(result, CandidateProposalV4):
        if result.producer_role != producer_role:
            raise OpenSpecProjectionError(
                "CandidateProposalV4 producer_role does not match assigned role"
            )
        spec, facts = project_candidate_proposal_v4(
            result,
            bindings=bindings,
        )
        return spec, facts, result

    if isinstance(result, Mapping):
        if result.get("producer_role") != producer_role:
            raise OpenSpecProjectionError(
                "open Producer draft producer_role does not match assigned role"
            )
        spec, facts = project_open_producer_draft(
            result,
            bindings=bindings,
            strict_resolution_contract=True,
        )
        return (
            spec,
            facts,
            None,
        )

    raise OpenSpecProjectionError(
        "Producer must return CandidateProposalV4 or an open Producer draft mapping"
    )


def _failure(
    context: ResearchContext,
    producer_role: str,
    *,
    failure_code: str,
    error: Exception,
) -> ProducerOutcome:
    return ProducerOutcome(
        producer_role=producer_role,
        context_ref=context.context_ref,
        context_digest=context.digest,
        spec=None,
        resolution_facts={
            "failure_code": failure_code,
            "producer_role": producer_role,
        },
        provenance={
            "producer_role": producer_role,
            "context_ref": context.context_ref,
            "context_digest": context.digest,
            "status": "CALL_OR_PROJECTION_FAILURE",
        },
        failure_code=failure_code,
        failure_detail=f"{type(error).__name__}: {error}",
    )


def produce_research_specs(
    context: ResearchContext,
    producer: ResearchProducer,
    bindings: Mapping[str, Any],
) -> tuple[ProducerOutcome, ...]:
    """Call each discovery Producer once and preserve every role's outcome."""

    if not isinstance(context, ResearchContext):
        raise TypeError("context must be ResearchContext")
    normalized_bindings = _validate_bindings(context, bindings)

    outcomes: list[ProducerOutcome] = []
    for producer_role in DISCOVERY_PRODUCERS:
        context_view = context.producer_view(producer_role)
        try:
            result = producer(producer_role, context_view)
        except Exception as error:
            outcomes.append(
                _failure(
                    context,
                    producer_role,
                    failure_code="PRODUCER_CALL_FAILED",
                    error=error,
                )
            )
            continue

        try:
            spec, resolution_facts, source_proposal = _project_result(
                producer_role,
                result,
                bindings=normalized_bindings,
            )
        except OpenSpecProjectionError as error:
            outcomes.append(
                _failure(
                    context,
                    producer_role,
                    failure_code="OPEN_SPEC_PROJECTION_FAILED",
                    error=error,
                )
            )
            continue

        provenance = {
            "producer_role": producer_role,
            "context_ref": context.context_ref,
            "context_digest": context.digest,
            "spec_id": spec.spec_id,
            "spec_digest": spec.digest,
            "source_proposal_digest": (
                source_proposal.digest if source_proposal is not None else None
            ),
            "status": "PRODUCED",
        }
        outcomes.append(
            ProducerOutcome(
                producer_role=producer_role,
                context_ref=context.context_ref,
                context_digest=context.digest,
                spec=spec,
                resolution_facts=resolution_facts,
                source_proposal=source_proposal,
                provenance=provenance,
            )
        )

    return tuple(outcomes)


__all__ = ["ResearchProducer", "produce_research_specs"]
