"""Offline proposal-only replay for the Research Line Meta shadow lane."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.meta_control import (
    ProposalOnlyShadowMetricsV1,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemorySnapshotV1,
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV4,
)

from .interfaces import ProducerOutcome, ResearchContext
from .producers import ResearchProducer, produce_research_specs


class OfflineReplayError(ValueError):
    """Raised when a proposal-only replay cannot preserve its input contract."""


def _replay_context(
    context: ResearchContext,
    policy: VersionedResearchPolicyV1,
    search_memory: SearchMemorySnapshotV1,
) -> ResearchContext:
    memory = {
        **context.scientific_memory,
        "search_memory_head": search_memory.to_dict(),
    }
    return replace(
        context,
        policy=policy.to_dict(),
        scientific_memory=memory,
    )


def _replay_bindings(
    bindings: Mapping[str, Any],
    context: ResearchContext,
) -> dict[str, Any]:
    """Reuse one binding schema while rebinding only the replay Context identity."""

    result = dict(bindings)
    result["context_ref"] = context.context_ref
    result["context_digest"] = context.digest
    return result


def _lineage_root(
    proposal: CandidateProposalV4,
    proposals_by_id: Mapping[str, CandidateProposalV4],
) -> str:
    current = proposal
    visited: set[str] = set()
    while current.parent_candidate_id is not None:
        parent_id = current.parent_candidate_id
        if parent_id in visited:
            return parent_id
        visited.add(current.candidate_id)
        parent = proposals_by_id.get(parent_id)
        if parent is None:
            return parent_id
        current = parent
    return current.candidate_id


def _eligible_outcomes(
    outcomes: tuple[ProducerOutcome, ...],
) -> tuple[ProducerOutcome, ...]:
    return tuple(
        outcome
        for outcome in outcomes
        if outcome.spec is not None
    )


def _semantic_identity(outcome: ProducerOutcome) -> str:
    requested = outcome.resolution_facts.get(
        "requested_current_semantics_digest"
    )
    if isinstance(requested, str) and requested:
        return requested
    if outcome.spec is None:
        raise OfflineReplayError("eligible outcome is missing its OpenSpec")
    return sha256_digest(
        {
            "mechanism_change": outcome.spec.mechanism_change,
        }
    )


def _axis_identity(outcome: ProducerOutcome) -> str:
    if isinstance(outcome.source_proposal, CandidateProposalV4):
        return outcome.source_proposal.mechanism_axis
    return "OPEN_MECHANISM"


def _lineage_identity(
    outcome: ProducerOutcome,
    proposals_by_id: Mapping[str, CandidateProposalV4],
) -> str:
    if isinstance(outcome.source_proposal, CandidateProposalV4):
        return "candidate:" + _lineage_root(
            outcome.source_proposal,
            proposals_by_id,
        )
    if outcome.spec is not None and outcome.spec.closest_parent:
        return "open-parent:" + outcome.spec.closest_parent
    return "open-semantic:" + _semantic_identity(outcome)


def _has_control(outcome: ProducerOutcome) -> bool:
    if isinstance(outcome.source_proposal, CandidateProposalV4):
        return outcome.source_proposal.matched_control_plan.plan_status in {
            "MATCHED_COMPARATOR_AVAILABLE",
            "QUEUE_MATCHED_CONTROL",
        }
    return bool(
        outcome.spec is not None
        and outcome.spec.matched_control_requirement.strip()
    )


def _metrics(
    outcomes: tuple[ProducerOutcome, ...],
    *,
    equal_replay_token_charge: int,
) -> ProposalOnlyShadowMetricsV1:
    if len(outcomes) != len(DISCOVERY_PRODUCERS) or {
        outcome.producer_role for outcome in outcomes
    } != set(DISCOVERY_PRODUCERS):
        raise OfflineReplayError(
            "proposal-only replay must preserve exactly one outcome per Producer role"
        )

    eligible = _eligible_outcomes(outcomes)
    proposals_by_id = {
        outcome.source_proposal.candidate_id: outcome.source_proposal
        for outcome in eligible
        if isinstance(outcome.source_proposal, CandidateProposalV4)
    }
    semantics = {_semantic_identity(outcome) for outcome in eligible}
    axes = {_axis_identity(outcome) for outcome in eligible}
    lineage_roots = {
        _lineage_identity(outcome, proposals_by_id) for outcome in eligible
    }
    control_count = sum(_has_control(outcome) for outcome in eligible)
    common_eligible_count = len(eligible)
    return ProposalOnlyShadowMetricsV1(
        proposal_count=len(outcomes),
        common_eligible_count=common_eligible_count,
        unique_semantics_count=len(semantics),
        mechanism_axis_count=len(axes),
        lineage_root_count=len(lineage_roots),
        control_count=control_count,
        semantic_collision_count=max(0, common_eligible_count - len(semantics)),
        # The frozen metric field is retained for runtime compatibility; this
        # value is a caller-supplied equal replay charge, not observed usage.
        billed_tokens=equal_replay_token_charge,
    )


@dataclass(frozen=True, slots=True)
class OfflineProducerReplayV1:
    """Proposal-only replay of four Producers with an equal caller charge.

    The charge is copied into the frozen ``billed_tokens`` metric field; it is
    not observed Provider usage.
    """

    producer: ResearchProducer
    producer_bindings: Mapping[str, Any]
    equal_replay_token_charge: int
    deterministic_directive_replay: bool

    def __post_init__(self) -> None:
        if not callable(self.producer):
            raise OfflineReplayError("producer must be callable")
        if (
            isinstance(self.equal_replay_token_charge, bool)
            or not isinstance(self.equal_replay_token_charge, int)
            or self.equal_replay_token_charge <= 0
        ):
            raise OfflineReplayError(
                "equal_replay_token_charge must be a positive integer"
            )
        if not isinstance(self.deterministic_directive_replay, bool):
            raise OfflineReplayError("deterministic_directive_replay must be boolean")
        if not isinstance(self.producer_bindings, Mapping):
            raise OfflineReplayError("producer_bindings must be a mapping")
        object.__setattr__(
            self,
            "producer_bindings",
            canonical_value(dict(self.producer_bindings)),
        )

    def _run_policy(
        self,
        *,
        context: ResearchContext,
        policy: VersionedResearchPolicyV1,
        search_memory: SearchMemorySnapshotV1,
        replay_arm: str,
    ) -> tuple[ProducerOutcome, ...]:
        replay_context = _replay_context(context, policy, search_memory)
        namespaced_call = getattr(self.producer, "call_with_namespace", None)
        producer = self.producer
        if callable(namespaced_call):
            namespace = f"offline-replay:{replay_arm}:{policy.digest}"

            def producer(role: str, view: Mapping[str, Any]) -> Any:
                return namespaced_call(
                    role,
                    view,
                    logical_namespace=namespace,
                )

        return produce_research_specs(
            replay_context,
            producer,
            _replay_bindings(self.producer_bindings, replay_context),
        )

    def __call__(
        self,
        *,
        context: ResearchContext,
        champion_policy: VersionedResearchPolicyV1,
        challenger_policy: VersionedResearchPolicyV1,
        search_memory: SearchMemorySnapshotV1,
    ) -> Mapping[str, Any]:
        if not isinstance(context, ResearchContext):
            raise OfflineReplayError("context must be ResearchContext")
        if not isinstance(champion_policy, VersionedResearchPolicyV1):
            raise OfflineReplayError("champion_policy must be VersionedResearchPolicyV1")
        if not isinstance(challenger_policy, VersionedResearchPolicyV1):
            raise OfflineReplayError("challenger_policy must be VersionedResearchPolicyV1")
        if not isinstance(search_memory, SearchMemorySnapshotV1):
            raise OfflineReplayError(
                "search_memory must be SearchMemorySnapshotV1"
            )

        champion_outcomes = self._run_policy(
            context=context,
            policy=champion_policy,
            search_memory=search_memory,
            replay_arm="champion",
        )
        challenger_outcomes = self._run_policy(
            context=context,
            policy=challenger_policy,
            search_memory=search_memory,
            replay_arm="challenger",
        )
        return {
            "source_context_digest": context.digest,
            "source_search_memory_digest": search_memory.digest,
            "champion_policy_digest": champion_policy.digest,
            "challenger_policy_digest": challenger_policy.digest,
            "champion": _metrics(
                champion_outcomes,
                equal_replay_token_charge=self.equal_replay_token_charge,
            ),
            "challenger": _metrics(
                challenger_outcomes,
                equal_replay_token_charge=self.equal_replay_token_charge,
            ),
            "same_model_prompt_schema_and_contexts": True,
            "deterministic_directive_replay": self.deterministic_directive_replay,
        }


__all__ = [
    "OfflineProducerReplayV1",
    "OfflineReplayError",
]
