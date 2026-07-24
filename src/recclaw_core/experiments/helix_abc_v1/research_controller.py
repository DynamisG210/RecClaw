"""Standalone M2 Research Line controller with no Evidence Guard dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical import sha256_digest
from .contracts import ProducerExecutionModeV1, ResourceCeilingsV1
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
            "ordinary_execution_opportunities": plan.ordinary_execution_opportunities,
            "round_transition_digest": sha256_digest(
                {
                    "plan": plan,
                    "feedback_projection": feedback_projection,
                    "memory": snapshot.digest,
                }
            ),
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
