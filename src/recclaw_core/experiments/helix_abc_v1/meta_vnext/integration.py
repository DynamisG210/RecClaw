"""Shadow-only integration with the current four-Producer ResearchRound path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..canonical import validate_sha256
from ..contracts import ProducerExecutionModeV1
from ..research_capability import StrongStaticRouterV1
from ..research_contracts import DISCOVERY_PRODUCERS, ProducerSessionResultV1
from .contracts import (
    CandidatePoolVNextV1,
    MetaVNextRecord,
    ResearchContextV1,
)
from .features import materialize_candidate_pool
from .learning import PairwiseSlowPolicyV1
from .routing import (
    FastResidualStateV1,
    MetaVNextRouteDecisionV1,
    MetaVNextRouterV1,
    initialize_fast_residual,
    static_champion_candidate,
)


class MetaVNextIntegrationError(ValueError):
    """Raised when a shadow round is bound to inconsistent policy identities."""


@dataclass(frozen=True, slots=True)
class MetaVNextArmBindingV1(MetaVNextRecord):
    slow_policy_digest: str
    arm_b_instance_digest: str
    arm_c_instance_digest: str
    search_seed_digest: str
    arm_b_mode: str
    arm_c_mode: str
    arm_c_initial_fast_state_digest: str

    schema = "recclaw.meta-vnext.arm-binding.v1"

    def __post_init__(self) -> None:
        for name in (
            "slow_policy_digest",
            "arm_b_instance_digest",
            "arm_c_instance_digest",
            "search_seed_digest",
            "arm_c_initial_fast_state_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if self.arm_b_instance_digest == self.arm_c_instance_digest:
            raise MetaVNextIntegrationError("B/C require distinct private instances")
        if self.arm_b_mode != "SLOW_ONLY" or self.arm_c_mode != "SLOW_PLUS_FAST":
            raise MetaVNextIntegrationError("B/C modes are not the frozen ablation")


def bind_three_arm_shadow(
    *,
    policy: PairwiseSlowPolicyV1,
    arm_b_instance_digest: str,
    arm_c_instance_digest: str,
    search_seed_digest: str,
) -> tuple[MetaVNextArmBindingV1, FastResidualStateV1]:
    fast_state = initialize_fast_residual(
        opaque_arm_instance_digest=arm_c_instance_digest,
        search_seed_digest=search_seed_digest,
        policy=policy,
    )
    binding = MetaVNextArmBindingV1(
        slow_policy_digest=policy.digest,
        arm_b_instance_digest=arm_b_instance_digest,
        arm_c_instance_digest=arm_c_instance_digest,
        search_seed_digest=search_seed_digest,
        arm_b_mode="SLOW_ONLY",
        arm_c_mode="SLOW_PLUS_FAST",
        arm_c_initial_fast_state_digest=fast_state.digest,
    )
    return binding, fast_state


@dataclass(frozen=True, slots=True)
class MetaVNextShadowRoundResultV1(MetaVNextRecord):
    producer_session_digest: str
    candidate_pool: CandidatePoolVNextV1
    static_champion_candidate_id: str
    slow_decision: MetaVNextRouteDecisionV1
    fast_decision: MetaVNextRouteDecisionV1 | None
    ordinary_execution_opportunities: int
    runtime_authority: str

    schema = "recclaw.meta-vnext.shadow-round-result.v1"

    def __post_init__(self) -> None:
        validate_sha256(
            self.producer_session_digest, field_name="producer_session_digest"
        )
        if self.ordinary_execution_opportunities != 1:
            raise MetaVNextIntegrationError(
                "Meta VNext preserves one ordinary execution per round"
            )
        if self.runtime_authority != "NONE":
            raise MetaVNextIntegrationError("shadow result cannot activate runtime")
        if not self.slow_decision.shadow_mode:
            raise MetaVNextIntegrationError("slow decision must remain shadow-only")
        if self.fast_decision is not None and not self.fast_decision.shadow_mode:
            raise MetaVNextIntegrationError("fast decision must remain shadow-only")


@dataclass(frozen=True, slots=True)
class MetaVNextShadowRuntimeV1:
    static_router: StrongStaticRouterV1
    meta_router: MetaVNextRouterV1

    def plan_shadow_round(
        self,
        *,
        session: ProducerSessionResultV1,
        parent_programs: Mapping[str, Mapping[str, Any]],
        research_context: ResearchContextV1,
        pre_round_state_digest: str,
        candidate_order_policy_digest: str,
        policy: PairwiseSlowPolicyV1,
        policy_projection: Mapping[str, Any] | None = None,
        lineage_depths: Mapping[str, int] | None = None,
        fast_state: FastResidualStateV1 | None = None,
    ) -> MetaVNextShadowRoundResultV1:
        if (
            session.mode
            is not ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
            or session.physical_call_count != 4
            or {item.producer_id for item in session.calls}
            != set(DISCOVERY_PRODUCERS)
        ):
            raise MetaVNextIntegrationError(
                "Meta VNext requires the four independent Producer calls"
            )
        if (
            fast_state is not None
            and fast_state.slow_policy_digest != policy.digest
        ):
            raise MetaVNextIntegrationError(
                "shadow fast state is bound to another slow checkpoint"
            )
        pool = materialize_candidate_pool(
            pool_id=f"meta-vnext:{session.session_id}",
            proposals=session.proposals,
            parent_programs=parent_programs,
            research_context=research_context,
            producer_invocation_digests=tuple(item.digest for item in session.calls),
            pre_round_state_digest=pre_round_state_digest,
            candidate_order_policy_digest=candidate_order_policy_digest,
            static_router=self.static_router,
            policy_projection=policy_projection,
            lineage_depths=lineage_depths,
        )
        champion = static_champion_candidate(pool)
        slow = self.meta_router.route(pool, policy, shadow_mode=True)
        fast = (
            self.meta_router.route(
                pool,
                policy,
                fast_state=fast_state,
                shadow_mode=True,
            )
            if fast_state is not None
            else None
        )
        return MetaVNextShadowRoundResultV1(
            producer_session_digest=session.digest,
            candidate_pool=pool,
            static_champion_candidate_id=champion.candidate_id,
            slow_decision=slow,
            fast_decision=fast,
            ordinary_execution_opportunities=1,
            runtime_authority="NONE",
        )
