"""Standalone M2 Research Line controller with no Evidence Guard dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest
from .contracts import ProducerExecutionModeV1, ResourceCeilingsV1
from .meta_control import (
    MetaControlUpdateProposalV1,
    build_meta_update_proposal,
)
from .research_capability import (
    FixtureProducerBrokerV1,
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedResearchPolicyV1,
    VersionedMetaPolicyUpdaterV1,
)
from .research_contracts import DevelopmentalMechanismBeliefV1


@dataclass(frozen=True, slots=True)
class ResearchRoundPlanV1:
    round_index: int
    proposal_session_digest: str
    route_trace_digest: str
    selected_candidate_id: str | None
    physical_call_count: int
    proposal_count: int
    ordinary_execution_opportunities: int
    plan_status: str
    policy_digest: str


@dataclass(slots=True)
class ResearchLineControllerV1:
    producer_mode: ProducerExecutionModeV1
    policy: VersionedResearchPolicyV1
    broker: FixtureProducerBrokerV1
    router: StrongStaticRouterV1
    memory_writer: SearchMemoryWriterV1

    @property
    def identity_digest(self) -> str:
        return sha256_digest(
            {
                "controller": "ResearchLineControllerV1",
                "producer_mode": self.producer_mode.value,
                "policy_digest": self.policy.digest,
                "broker": self.broker,
                "router_policy_digest": self.router.policy_digest,
                "memory_namespace": "DEVELOPMENT_ONLY/SEARCH_MEMORY",
            }
        )

    def plan_round(
        self,
        *,
        round_index: int,
        session_id: str,
        drafts: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        role_memory: Mapping[str, Mapping[str, Any]],
        seed: int,
        ceilings: ResourceCeilingsV1,
    ) -> ResearchRoundPlanV1:
        session = self.broker.dispatch(
            session_id=session_id,
            mode=self.producer_mode,
            drafts=drafts,
            context=context,
            role_memory=role_memory,
            seed=seed,
            ceilings=ceilings,
            policy_projection=self.policy.to_dict(),
        )
        route = self.router.route(
            session.proposals, policy_projection=self.policy.to_dict()
        )
        return ResearchRoundPlanV1(
            round_index=round_index,
            proposal_session_digest=session.digest,
            route_trace_digest=route.digest,
            selected_candidate_id=route.selected_candidate_id,
            physical_call_count=session.physical_call_count,
            proposal_count=session.proposal_count,
            ordinary_execution_opportunities=1 if route.selected_candidate_id else 0,
            plan_status=(
                "SELECTED"
                if route.selected_candidate_id
                else "ALL_COMMON_ELIGIBLE_REJECTED"
            ),
            policy_digest=self.policy.digest,
        )

    def close_round(
        self,
        *,
        plan: ResearchRoundPlanV1,
        feedback_projection: Mapping[str, Any],
        beliefs: Sequence[DevelopmentalMechanismBeliefV1],
    ) -> Mapping[str, Any]:
        predecessor = self.memory_writer.head.digest if self.memory_writer.head else None
        snapshot = self.memory_writer.commit(
            round_index=plan.round_index,
            expected_predecessor_digest=predecessor,
            beliefs=beliefs,
            route_trace_digest=plan.route_trace_digest,
            feedback_projection=feedback_projection,
        )
        return {
            "feedback_consumption_count": 1,
            "memory_snapshot_digest": snapshot.digest,
            "search_memory_projection": canonical_value(
                {
                    "namespace": snapshot.namespace,
                    "round_index": snapshot.round_index,
                    "snapshot_digest": snapshot.digest,
                    "beliefs": [item.to_dict() for item in snapshot.beliefs],
                    "feedback": feedback_projection,
                }
            ),
            "ordinary_execution_opportunities": plan.ordinary_execution_opportunities,
            "round_transition_digest": sha256_digest(
                {
                    "plan": plan,
                    "feedback_projection": feedback_projection,
                    "memory": snapshot.digest,
                }
            ),
        }

    def close_round_v13(
        self,
        *,
        plan: ResearchRoundPlanV1,
        feedback: FusedSearchFeedbackV2,
        beliefs: Sequence[DevelopmentalMechanismBeliefV1],
    ) -> Mapping[str, Any]:
        """Consume only the closed V13 search projection.

        Guard-private adjudication never reaches this controller.  A feedback
        class that disallows Search Memory updates is a literal no-write.
        """

        from recclaw_core.helix.scientific_attribution import (
            PromptFeedbackProjectionV2,
        )

        if not feedback.controller_update_allowed:
            if beliefs:
                raise ValueError(
                    "state-preserving V13 feedback cannot carry beliefs"
                )
            return {
                "feedback_consumption_count": 0,
                "memory_snapshot_digest": (
                    self.memory_writer.head.digest
                    if self.memory_writer.head is not None
                    else None
                ),
                "search_memory_projection": None,
                "ordinary_execution_opportunities": (
                    plan.ordinary_execution_opportunities
                ),
                "round_transition_digest": None,
                "state_changed": False,
            }
        if not feedback.search_memory_update_allowed:
            raise ValueError(
                "controller update requires the V13 Search Memory update"
            )
        if feedback.search_utility_event is None and beliefs:
            raise ValueError(
                "task-only V13 feedback cannot carry mechanism beliefs"
            )
        prompt_projection = PromptFeedbackProjectionV2.from_fused(feedback)
        predecessor = (
            self.memory_writer.head.digest
            if self.memory_writer.head is not None
            else None
        )
        snapshot = self.memory_writer.commit(
            round_index=plan.round_index,
            expected_predecessor_digest=predecessor,
            beliefs=beliefs,
            route_trace_digest=plan.route_trace_digest,
            feedback_projection=prompt_projection.to_dict(),
        )
        public_projection = canonical_value(
            {
                "namespace": snapshot.namespace,
                "round_index": snapshot.round_index,
                "snapshot_digest": snapshot.digest,
                "beliefs": [item.to_dict() for item in snapshot.beliefs],
                "prompt_feedback_projection": prompt_projection.to_dict(),
            }
        )
        transition_digest = sha256_digest(
            {
                "plan": plan,
                "fused_search_feedback_digest": feedback.digest,
                "memory": snapshot.digest,
            }
        )
        return {
            "feedback_consumption_count": 1,
            "memory_snapshot_digest": snapshot.digest,
            "search_memory_projection": public_projection,
            "ordinary_execution_opportunities": (
                plan.ordinary_execution_opportunities
            ),
            "round_transition_digest": transition_digest,
            "state_changed": True,
        }

    def apply_meta_update(
        self,
        *,
        updater: VersionedMetaPolicyUpdaterV1,
        completed_round_index: int,
        aggregate: Mapping[str, Any],
    ) -> VersionedResearchPolicyV1:
        if (
            self.memory_writer.head is None
            or self.memory_writer.head.round_index != completed_round_index
        ):
            raise ValueError("Meta activation requires the matching closed round boundary")
        self.policy = updater.update(
            self.policy,
            completed_round_index=completed_round_index,
            aggregate=aggregate,
        )
        return self.policy

    def propose_next_campaign_policy(
        self,
    ) -> MetaControlUpdateProposalV1:
        """Create a challenger without mutating the active campaign policy."""

        if self.memory_writer.head is None:
            raise ValueError(
                "Meta proposal requires at least one closed SearchRound"
            )
        return build_meta_update_proposal(
            policy=self.policy,
            search_memory=self.memory_writer.head,
        )
