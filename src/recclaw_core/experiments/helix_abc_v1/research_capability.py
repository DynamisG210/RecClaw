"""Deterministic M2 Producer, Router, Meta, and Search Memory implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw

from .canonical import canonical_value, sha256_digest
from .contracts import (
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
    validate_no_research_evidence_authority_fields,
)
from .research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV2,
    CandidateProposalV3,
    CandidateProposalV4,
    DevelopmentalMechanismBeliefV1,
    DevelopmentalMechanismBeliefV2,
    DiscoveryCreditV1,
    ProducerCallRecordV1,
    ProducerSessionResultV1,
    ProposalIntentV1,
    RouteTraceV1,
    RouterHardGateDecisionV1,
    RouterHardGateReasonV1,
    SearchUtilityFeaturesV1,
)


class ResearchCapabilityError(ValueError):
    pass


def _candidate(
    *,
    session_id: str,
    producer_id: str,
    producer_role: str,
    slot_index: int,
    draft: Mapping[str, Any],
) -> CandidateProposalV2:
    intent = ProposalIntentV1(draft["proposal_intent"])
    credit = (
        DiscoveryCreditV1.NON_DISCOVERY_CONTROL
        if intent is ProposalIntentV1.CONTROL
        else DiscoveryCreditV1.NON_DISCOVERY_REPAIR
        if intent is ProposalIntentV1.REPAIR
        else DiscoveryCreditV1.DISCOVERY
    )
    preimage = {
        "session_id": session_id,
        "producer_id": producer_id,
        "producer_role": producer_role,
        "slot_index": slot_index,
        "mechanism_axis": draft["mechanism_axis"],
        "mechanism_program": draft["mechanism_program"],
        "proposal_intent": intent.value,
    }
    return CandidateProposalV2(
        candidate_id=f"cand-{sha256_digest(preimage)[:24]}",
        producer_id=producer_id,
        producer_role=producer_role,
        proposal_intent=intent,
        discovery_credit=credit,
        mechanism_axis=str(draft["mechanism_axis"]),
        mechanism_program=dict(draft["mechanism_program"]),
        utility_features=SearchUtilityFeaturesV1(**draft["utility_features"]),
        parent_candidate_id=draft.get("parent_candidate_id"),
        assigned_before_call=True,
        post_hoc_relabel=False,
    )


@dataclass(frozen=True, slots=True)
class FixtureProducerBrokerV1:
    """A no-LLM broker that exercises the exact physical-call contracts."""

    base_model_ref: str = "fixture-model/m2-v1"
    bl_projection_digest: str = (
        "c823daa22cf2007e679a81300c3ebf1d80e35f7918690aaadce770dba8a3dc65"
    )
    candidate_schema_ref: str = "recclaw.candidate-proposal.v2"
    total_input_tokens: int = 200
    total_output_tokens: int = 200

    @staticmethod
    def _allocate(total: int, weights: Sequence[float]) -> tuple[int, ...]:
        raw = [total * float(weight) / sum(weights) for weight in weights]
        base = [int(item) for item in raw]
        for index in sorted(
            range(len(raw)), key=lambda item: (-(raw[item] - base[item]), item)
        )[: total - sum(base)]:
            base[index] += 1
        return tuple(base)

    def dispatch(
        self,
        *,
        session_id: str,
        mode: ProducerExecutionModeV1,
        drafts: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        role_memory: Mapping[str, Mapping[str, Any]],
        seed: int,
        ceilings: ResourceCeilingsV1,
        policy_projection: Mapping[str, Any] | None = None,
    ) -> ProducerSessionResultV1:
        if len(drafts) != 4 or ceilings.total_proposal_count < 4:
            raise ResearchCapabilityError("M2 fixture sessions require four proposal slots")
        validate_no_research_evidence_authority_fields(context)
        validate_no_research_evidence_authority_fields(role_memory)
        if ceilings.total_input_tokens < self.total_input_tokens:
            raise ResearchCapabilityError("input-token ceiling is below the frozen session plan")
        if ceilings.total_output_tokens < self.total_output_tokens:
            raise ResearchCapabilityError("output-token ceiling is below the frozen session plan")

        if mode is ProducerExecutionModeV1.BATCHED_ROLE_PORTFOLIO_V1:
            call_groups = ((tuple(range(4)), "batched_portfolio", "batched"),)
        elif mode is ProducerExecutionModeV1.NEUTRAL_MULTISAMPLE_CONTROL_V1:
            call_groups = tuple(((index,), f"neutral_sample_{index}", "neutral") for index in range(4))
        elif mode is ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1:
            call_groups = tuple(
                ((index,), role, role) for index, role in enumerate(DISCOVERY_PRODUCERS)
            )
        else:
            raise ResearchCapabilityError("unsupported Producer execution mode")

        if policy_projection is not None:
            validate_no_research_evidence_authority_fields(policy_projection)
        allocation = (
            dict(policy_projection["producer_token_allocation"])
            if policy_projection is not None
            and mode
            is ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
            else {}
        )
        weights = [
            float(allocation.get(role, 1.0 / len(call_groups)))
            for _slots, _producer, role in call_groups
        ]
        input_allocations = self._allocate(self.total_input_tokens, weights)
        output_allocations = self._allocate(self.total_output_tokens, weights)
        proposals: list[CandidateProposalV2] = []
        calls: list[ProducerCallRecordV1] = []
        for call_index, (slot_indexes, producer_id, role) in enumerate(call_groups):
            memory_view = (
                role_memory.get(role, {})
                if mode is ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
                else role_memory.get("neutral", {})
                if role == "neutral"
                else {}
            )
            context_view = {"common": context, "producer_role": role}
            prompt = {
                "contract": "typed-bl-icf-proposal-v2",
                "mode": mode.value,
                "producer_role": role,
                "mechanism_axis_targeting": (
                    policy_projection.get("mechanism_axis_targeting", ())
                    if policy_projection is not None
                    else ()
                ),
                "memory_retrieval_policy": (
                    policy_projection.get("memory_retrieval_policy", "STATIC_GATE_FIXTURE")
                    if policy_projection is not None
                    else "STATIC_GATE_FIXTURE"
                ),
            }
            rng_digest = sha256_digest(
                {"seed": seed, "mode": mode.value, "call_index": call_index}
            )
            call_proposals = [
                _candidate(
                    session_id=session_id,
                    producer_id=producer_id,
                    producer_role=(
                        DISCOVERY_PRODUCERS[slot]
                        if mode is ProducerExecutionModeV1.BATCHED_ROLE_PORTFOLIO_V1
                        else role
                    ),
                    slot_index=slot,
                    draft=drafts[slot],
                )
                for slot in slot_indexes
            ]
            request = {
                "base_model_ref": self.base_model_ref,
                "context_digest": sha256_digest(context_view),
                "memory_digest": sha256_digest(memory_view),
                "prompt_digest": sha256_digest(prompt),
                "rng_digest": rng_digest,
                "slot_indexes": slot_indexes,
                "versioned_policy_digest": (
                    sha256_digest(policy_projection)
                    if policy_projection is not None
                    else None
                ),
            }
            response = [item.to_dict() for item in call_proposals]
            physical_call_id = (
                f"fixture-call-{sha256_digest({'session': session_id, 'request': request})[:20]}"
            )
            calls.append(
                ProducerCallRecordV1(
                    session_id=session_id,
                    mode=mode,
                    physical_call_id=physical_call_id,
                    producer_id=producer_id,
                    producer_role=role,
                    request_digest=sha256_digest(request),
                    response_digest=sha256_digest(response),
                    context_digest=sha256_digest(context_view),
                    memory_digest=sha256_digest(memory_view),
                    prompt_digest=sha256_digest(prompt),
                    rng_digest=rng_digest,
                    candidate_ids=tuple(item.candidate_id for item in call_proposals),
                    input_tokens=input_allocations[call_index],
                    output_tokens=output_allocations[call_index],
                    billed_tokens=(
                        input_allocations[call_index] + output_allocations[call_index]
                    ),
                    latency_ms=20 * len(slot_indexes),
                )
            )
            proposals.extend(call_proposals)

        result = ProducerSessionResultV1(
            session_id=session_id,
            mode=mode,
            calls=tuple(calls),
            proposals=tuple(proposals),
            total_resource_envelope_digest=sha256_digest(ceilings),
            base_model_ref=self.base_model_ref,
            bl_projection_digest=self.bl_projection_digest,
            candidate_schema_ref=self.candidate_schema_ref,
            proposal_count=len(proposals),
            physical_call_count=len(calls),
            input_tokens=sum(item.input_tokens for item in calls),
            output_tokens=sum(item.output_tokens for item in calls),
            billed_tokens=sum(item.billed_tokens for item in calls),
            session_latency_ms=max(item.latency_ms for item in calls),
        )
        if (
            result.input_tokens != self.total_input_tokens
            or result.output_tokens != self.total_output_tokens
        ):
            raise ResearchCapabilityError("fixture broker did not consume the frozen total")
        return result


@dataclass(frozen=True, slots=True)
class ControlAblationBuilderV1:
    def build(self, parent: CandidateProposalV2) -> Mapping[str, Any]:
        return {
            "parent_candidate_id": parent.candidate_id,
            "service": "control_ablation_builder",
            "discovery_credit": DiscoveryCreditV1.NON_DISCOVERY_CONTROL.value,
            "mechanism_axis": parent.mechanism_axis,
            "program_digest": sha256_digest(parent.mechanism_program),
        }


@dataclass(frozen=True, slots=True)
class RepairEngineerV1:
    def build(self, parent: CandidateProposalV2, blocker_code: str) -> Mapping[str, Any]:
        return {
            "parent_candidate_id": parent.candidate_id,
            "service": "repair_engineer",
            "discovery_credit": DiscoveryCreditV1.NON_DISCOVERY_REPAIR.value,
            "blocker_code": blocker_code,
            "program_digest": sha256_digest(parent.mechanism_program),
        }


@dataclass(frozen=True, slots=True)
class StrongStaticRouterV1:
    runnable_floor: float = 0.45
    utility_floor: float = 0.35
    blocker_ceiling: float = 0.75
    cost_ceiling: float = 0.80
    slate_ceiling: int = 3

    @property
    def policy_digest(self) -> str:
        return sha256_digest(canonical_value(self))

    @staticmethod
    def score(
        features: SearchUtilityFeaturesV1,
        policy_projection: Mapping[str, Any] | None = None,
    ) -> float:
        priors = (
            dict(policy_projection.get("router_priors", ()))
            if policy_projection is not None
            else {}
        )
        acquisition = (
            dict(policy_projection.get("acquisition_parameters", ()))
            if policy_projection is not None
            else {}
        )
        return round(
            0.23
            * float(priors.get("runnable_probability", 0.5))
            / 0.5
            * features.runnable_probability
            + 0.21
            * float(priors.get("useful_signal", 0.5))
            / 0.5
            * features.useful_signal
            + 0.19
            * float(acquisition.get("exploration_weight", 0.5))
            / 0.5
            * features.frontier_potential
            + 0.19 * features.information_gain
            - 0.08 * features.cost
            - 0.10 * features.blocker_risk,
            12,
        )

    def route(
        self,
        proposals: Sequence[
            CandidateProposalV2 | CandidateProposalV3 | CandidateProposalV4
        ],
        policy_projection: Mapping[str, Any] | None = None,
    ) -> RouteTraceV1:
        effective_policy_digest = sha256_digest(
            {
                "hard_gate_policy_digest": self.policy_digest,
                "versioned_policy": policy_projection,
            }
        )
        decisions: list[RouterHardGateDecisionV1] = []
        eligible: list[
            tuple[
                float,
                int,
                CandidateProposalV2 | CandidateProposalV3 | CandidateProposalV4,
                str,
                str,
            ]
        ] = []
        for index, proposal in enumerate(proposals):
            compile_digest = None
            semantics_digest = None
            reason = RouterHardGateReasonV1.ALLOW
            try:
                report = compile_program(deep_thaw(proposal.mechanism_program))
                compile_digest = sha256_digest(report.to_dict())
                if not report.is_valid:
                    reason = RouterHardGateReasonV1.BL_COMPILE_FAILED
                else:
                    semantics_digest = report.mechanism_semantics_digest
            except Exception:
                reason = RouterHardGateReasonV1.BL_COMPILE_FAILED
            feature = proposal.utility_features
            if reason is RouterHardGateReasonV1.ALLOW and feature.runnable_probability < self.runnable_floor:
                reason = RouterHardGateReasonV1.RUNNABLE_BELOW_FLOOR
            elif reason is RouterHardGateReasonV1.ALLOW and feature.blocker_risk > self.blocker_ceiling:
                reason = RouterHardGateReasonV1.BLOCKER_RISK_ABOVE_CEILING
            elif reason is RouterHardGateReasonV1.ALLOW and feature.cost > self.cost_ceiling:
                reason = RouterHardGateReasonV1.COST_ABOVE_CEILING
            score = self.score(feature, policy_projection)
            if (
                reason is RouterHardGateReasonV1.ALLOW
                and score < self.utility_floor
            ):
                reason = RouterHardGateReasonV1.UTILITY_BELOW_FLOOR
            allowed = reason is RouterHardGateReasonV1.ALLOW
            decisions.append(
                RouterHardGateDecisionV1(
                    candidate_id=proposal.candidate_id,
                    allowed=allowed,
                    reason=reason,
                    compile_report_digest=compile_digest,
                    mechanism_semantics_digest=semantics_digest,
                    feature_digest=sha256_digest(feature),
                    policy_digest=effective_policy_digest,
                )
            )
            if allowed and semantics_digest is not None and compile_digest is not None:
                eligible.append(
                    (
                        score,
                        index,
                        proposal,
                        semantics_digest,
                        compile_digest,
                    )
                )

        # Choose the strongest representative of each executable semantic
        # program. Producer call order must not decide which duplicate survives.
        eligible.sort(key=lambda item: (-item[0], item[1]))
        ranked_unique: list[
            tuple[
                float,
                int,
                CandidateProposalV2 | CandidateProposalV3 | CandidateProposalV4,
                str,
                str,
            ]
        ] = []
        seen_semantics: set[str] = set()
        duplicate_ids: set[str] = set()
        for item in eligible:
            semantics_digest = item[3]
            if semantics_digest in seen_semantics:
                duplicate_ids.add(item[2].candidate_id)
                continue
            seen_semantics.add(semantics_digest)
            ranked_unique.append(item)

        keep = ranked_unique[: max(0, self.slate_ceiling)]
        keep_ids = {item[2].candidate_id for item in keep}
        for index, decision in enumerate(decisions):
            if decision.candidate_id in duplicate_ids:
                reason = RouterHardGateReasonV1.SEMANTIC_DUPLICATE
            elif decision.allowed and decision.candidate_id not in keep_ids:
                reason = RouterHardGateReasonV1.SLATE_CEILING
            else:
                continue
            decisions[index] = RouterHardGateDecisionV1(
                candidate_id=decision.candidate_id,
                allowed=False,
                reason=reason,
                compile_report_digest=decision.compile_report_digest,
                mechanism_semantics_digest=decision.mechanism_semantics_digest,
                feature_digest=decision.feature_digest,
                policy_digest=decision.policy_digest,
            )
        selected = keep[0] if keep else None
        return RouteTraceV1(
            pool_digest=sha256_digest([item.to_dict() for item in proposals]),
            ordered_candidate_ids=tuple(item.candidate_id for item in proposals),
            ranked_candidate_ids=tuple(item[2].candidate_id for item in keep),
            decisions=tuple(decisions),
            selected_candidate_id=selected[2].candidate_id if selected else None,
            selection_score=selected[0] if selected else None,
            policy_digest=effective_policy_digest,
        )


@dataclass(frozen=True, slots=True)
class VersionedResearchPolicyV1:
    version: int
    producer_token_allocation: tuple[tuple[str, float], ...]
    mechanism_axis_targeting: tuple[str, ...]
    memory_retrieval_policy: str
    router_priors: tuple[tuple[str, float], ...]
    acquisition_parameters: tuple[tuple[str, float], ...]
    predecessor_digest: str | None
    meta_router_policy_digest: str | None = None
    meta_router_promotion_decision_digest: str | None = None
    promotion_decision_digest: str | None = None
    activation_boundary: str = "NONE"
    control_mode: str = "STATIC_INITIAL_V1"

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def initial_research_policy() -> VersionedResearchPolicyV1:
    return VersionedResearchPolicyV1(
        version=1,
        producer_token_allocation=tuple((role, 0.25) for role in DISCOVERY_PRODUCERS),
        mechanism_axis_targeting=(
            "architecture",
            "geometry",
            "message_transform",
            "objective",
            "propagation",
            "sampling",
            "self_supervision",
        ),
        memory_retrieval_policy="ROLE_SCOPED_PRIOR_ROUND_V1",
        router_priors=(("runnable_probability", 0.5), ("useful_signal", 0.5)),
        acquisition_parameters=(("exploration_weight", 0.5), ("cost_weight", 0.5)),
        predecessor_digest=None,
    )


@dataclass(frozen=True, slots=True)
class VersionedMetaPolicyUpdaterV1:
    def update(
        self,
        policy: VersionedResearchPolicyV1,
        *,
        completed_round_index: int,
        aggregate: Mapping[str, Any],
    ) -> VersionedResearchPolicyV1:
        if completed_round_index < 1:
            raise ResearchCapabilityError("Meta updates occur only after a round boundary")
        validate_no_research_evidence_authority_fields(aggregate)
        if any("candidate" in str(key).lower() for key in aggregate):
            raise ResearchCapabilityError("Meta aggregate cannot contain Candidate identity")
        useful = aggregate.get("producer_useful_rates", {})
        if set(useful) != set(DISCOVERY_PRODUCERS):
            raise ResearchCapabilityError("Meta requires aggregate rates for every Producer")
        raw = {role: max(0.15, float(useful[role])) for role in DISCOVERY_PRODUCERS}
        total = sum(raw.values())
        allocation = tuple((role, round(raw[role] / total, 12)) for role in DISCOVERY_PRODUCERS)
        axis_gaps = aggregate.get("mechanism_axis_gaps", ())
        targeting = tuple(dict.fromkeys(str(item) for item in axis_gaps)) or policy.mechanism_axis_targeting
        calibration_error = float(aggregate.get("calibration_error", 0.0))
        return VersionedResearchPolicyV1(
            version=policy.version + 1,
            producer_token_allocation=allocation,
            mechanism_axis_targeting=targeting,
            memory_retrieval_policy=policy.memory_retrieval_policy,
            router_priors=(
                ("runnable_probability", round(0.5 + min(calibration_error, 0.2), 12)),
                ("useful_signal", round(0.5 - min(calibration_error, 0.2), 12)),
            ),
            acquisition_parameters=policy.acquisition_parameters,
            predecessor_digest=policy.digest,
            meta_router_policy_digest=policy.meta_router_policy_digest,
            meta_router_promotion_decision_digest=(
                policy.meta_router_promotion_decision_digest
            ),
            promotion_decision_digest=policy.promotion_decision_digest,
            activation_boundary=policy.activation_boundary,
            control_mode=policy.control_mode,
        )


@dataclass(frozen=True, slots=True)
class SearchMemorySnapshotV1:
    namespace: str
    round_index: int
    predecessor_digest: str | None
    beliefs: tuple[
        DevelopmentalMechanismBeliefV1 | DevelopmentalMechanismBeliefV2,
        ...,
    ]
    route_trace_digest: str
    feedback_projection_digest: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


class SearchMemoryWriterV1:
    def __init__(self, namespace: str) -> None:
        if namespace != "DEVELOPMENT_ONLY/SEARCH_MEMORY":
            raise ResearchCapabilityError("Search Memory requires its dedicated namespace")
        self._namespace = namespace
        self._head: SearchMemorySnapshotV1 | None = None

    @property
    def head(self) -> SearchMemorySnapshotV1 | None:
        return self._head

    def commit(
        self,
        *,
        round_index: int,
        expected_predecessor_digest: str | None,
        beliefs: Sequence[
            DevelopmentalMechanismBeliefV1 | DevelopmentalMechanismBeliefV2
        ],
        route_trace_digest: str,
        feedback_projection: Mapping[str, Any],
    ) -> SearchMemorySnapshotV1:
        actual = self._head.digest if self._head else None
        if actual != expected_predecessor_digest:
            raise ResearchCapabilityError("Search Memory predecessor mismatch")
        validate_no_research_evidence_authority_fields(feedback_projection)
        prior_beliefs = self._head.beliefs if self._head is not None else ()
        merged: dict[
            str,
            DevelopmentalMechanismBeliefV1
            | DevelopmentalMechanismBeliefV2,
        ] = {
            item.hypothesis_id: item for item in prior_beliefs
        }
        for belief in beliefs:
            merged[belief.hypothesis_id] = belief
        snapshot = SearchMemorySnapshotV1(
            namespace=self._namespace,
            round_index=round_index,
            predecessor_digest=actual,
            beliefs=tuple(merged.values())[-32:],
            route_trace_digest=route_trace_digest,
            feedback_projection_digest=sha256_digest(feedback_projection),
        )
        self._head = snapshot
        return snapshot
