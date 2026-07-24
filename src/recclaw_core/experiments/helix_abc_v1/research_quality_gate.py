"""Outcome-masked, deterministic M2 Research Capability quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw

from .canonical import canonical_value, sha256_digest
from .contracts import ProducerExecutionModeV1
from .research_capability import (
    ControlAblationBuilderV1,
    RepairEngineerV1,
    StrongStaticRouterV1,
    VersionedMetaPolicyUpdaterV1,
    VersionedResearchPolicyV1,
)
from .research_contracts import (
    AgentizationVerdictV1,
    MetaVerdictV1,
    ProducerSessionResultV1,
    ProposalIntentV1,
)


@dataclass(frozen=True, slots=True)
class AgentizationMetricsV1:
    schema_valid_rate: float
    bl_compile_rate: float
    semantic_uniqueness: float
    mechanism_axis_coverage: float
    falsification_completeness: float
    parent_ablation_completeness: float
    duplicate_rate: float
    static_router_topk_utility: float
    latency: int
    physical_call_count: int
    token_cost: int

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class AgentizationQualityGateV1:
    mode_metrics: tuple[tuple[str, AgentizationMetricsV1], ...]
    selected_mode: ProducerExecutionModeV1 | None
    verdict: AgentizationVerdictV1
    fixture_lineage_digest: str
    static_router_digest: str
    total_resource_envelope_digest: str
    outcome_masked: bool

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MetaQualityReportV1:
    selected_mode: ProducerExecutionModeV1
    baseline_policy_digest: str
    updated_policy_digest: str
    deterministic_replay: bool
    round_boundary_only: bool
    field_allowlist_valid: bool
    calibration_valid: bool
    candidate_id_leakage: bool
    single_producer_collapse: bool
    single_family_collapse: bool
    parameter_tuning_collapse: bool
    activated_policy_effect: bool
    verdict: MetaVerdictV1

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ResearchCapabilityQualityGateV1:
    agentization_gate_digest: str
    agentization_verdict: AgentizationVerdictV1
    selected_mode: ProducerExecutionModeV1 | None
    meta_report_digest: str | None
    meta_verdict: MetaVerdictV1
    progression_status: str
    producer_lineage_complete: float
    post_hoc_relabel_count: int
    falsification_slot_present: bool
    control_repair_credit_separated: bool
    complete_pool_and_route_trace: bool
    search_utility_only: bool

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ResearchLineStandaloneReadinessV1:
    research_source_digest: str
    shared_contract_digest: str
    common_release_projection_digest: str
    null_port_digest: str
    component_digests: tuple[tuple[str, str], ...]
    test_evidence_digest: str
    guard_package_absent: bool
    direct_import_boundary_passed: bool
    fixed_seed_replay_passed: bool
    multi_round_null_port_e2e_passed: bool
    verdict: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def measure_agentization_mode(
    session: ProducerSessionResultV1,
    router: StrongStaticRouterV1,
) -> AgentizationMetricsV1:
    proposals = session.proposals
    valid = 0
    semantics: list[str] = []
    for proposal in proposals:
        try:
            report = compile_program(deep_thaw(proposal.mechanism_program))
        except Exception:
            continue
        if report.is_valid:
            valid += 1
            if report.mechanism_semantics_digest is not None:
                semantics.append(report.mechanism_semantics_digest)
    route = router.route(proposals)
    control_records = tuple(
        ControlAblationBuilderV1().build(item)
        for item in proposals
        if item.proposal_intent
        in {ProposalIntentV1.DISCOVERY, ProposalIntentV1.FALSIFICATION}
    )
    unique = len(set(semantics))
    total = max(1, len(proposals))
    return AgentizationMetricsV1(
        schema_valid_rate=len(proposals) / total,
        bl_compile_rate=valid / total,
        semantic_uniqueness=unique / total,
        mechanism_axis_coverage=len({item.mechanism_axis for item in proposals}) / total,
        falsification_completeness=float(
            any(item.proposal_intent is ProposalIntentV1.FALSIFICATION for item in proposals)
        ),
        parent_ablation_completeness=float(len(control_records) == len(proposals)),
        duplicate_rate=(len(semantics) - unique) / total,
        static_router_topk_utility=float(route.selection_score or 0.0),
        latency=session.session_latency_ms,
        physical_call_count=session.physical_call_count,
        token_cost=session.billed_tokens,
    )


def run_agentization_gate(
    sessions: Mapping[ProducerExecutionModeV1, ProducerSessionResultV1],
    *,
    router: StrongStaticRouterV1,
    fixture_lineage_digest: str,
) -> AgentizationQualityGateV1:
    expected = set(ProducerExecutionModeV1)
    if set(sessions) != expected:
        raise ValueError("agentization gate requires all three Producer modes")
    envelope_digests = {
        item.total_resource_envelope_digest for item in sessions.values()
    }
    totals = {
        (item.proposal_count, item.input_tokens, item.output_tokens, item.billed_tokens)
        for item in sessions.values()
    }
    controlled_contracts = {
        (item.base_model_ref, item.bl_projection_digest, item.candidate_schema_ref)
        for item in sessions.values()
    }
    if len(envelope_digests) != 1 or len(totals) != 1 or len(controlled_contracts) != 1:
        raise ValueError("agentization modes do not share one total resource envelope")
    metrics = {
        mode: measure_agentization_mode(session, router)
        for mode, session in sessions.items()
    }
    independent = metrics[
        ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
    ]
    batched = metrics[ProducerExecutionModeV1.BATCHED_ROLE_PORTFOLIO_V1]
    neutral = metrics[ProducerExecutionModeV1.NEUTRAL_MULTISAMPLE_CONTROL_V1]
    independent_session = sessions[
        ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
    ]
    independent_call_contract = (
        len(independent_session.calls) == 4
        and len({item.physical_call_id for item in independent_session.calls}) == 4
        and tuple(item.producer_role for item in independent_session.calls)
        == (
            "mechanism_composer",
            "lineage_refiner",
            "falsification_designer",
            "frontier_architect",
        )
        and all(len(item.candidate_ids) == 1 for item in independent_session.calls)
        and len({item.prompt_digest for item in independent_session.calls}) == 4
        and len({item.context_digest for item in independent_session.calls}) == 4
        and len({item.memory_digest for item in independent_session.calls}) == 4
        and len({item.rng_digest for item in independent_session.calls}) == 4
    )
    independent_pass = (
        independent.schema_valid_rate == 1.0
        and independent.bl_compile_rate == 1.0
        and independent.semantic_uniqueness >= 0.75
        and independent.mechanism_axis_coverage >= 0.75
        and independent.falsification_completeness == 1.0
        and independent.parent_ablation_completeness == 1.0
        and independent.duplicate_rate <= 0.25
        and independent.semantic_uniqueness > batched.semantic_uniqueness
        and independent.mechanism_axis_coverage >= neutral.mechanism_axis_coverage
        and independent.static_router_topk_utility >= batched.static_router_topk_utility
        and independent.physical_call_count == 4
        and independent_call_contract
    )
    batched_pass = (
        batched.schema_valid_rate == 1.0
        and batched.bl_compile_rate == 1.0
        and batched.falsification_completeness == 1.0
    )
    if independent_pass:
        verdict = AgentizationVerdictV1.PASS_INDEPENDENT_MULTI_AGENT
        selected = ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
    elif batched_pass:
        verdict = AgentizationVerdictV1.PASS_BATCHED_ONLY
        selected = ProducerExecutionModeV1.BATCHED_ROLE_PORTFOLIO_V1
    else:
        verdict = AgentizationVerdictV1.FAIL
        selected = None
    return AgentizationQualityGateV1(
        mode_metrics=tuple(
            (mode.value, metrics[mode]) for mode in sorted(metrics, key=lambda item: item.value)
        ),
        selected_mode=selected,
        verdict=verdict,
        fixture_lineage_digest=fixture_lineage_digest,
        static_router_digest=router.policy_digest,
        total_resource_envelope_digest=next(iter(envelope_digests)),
        outcome_masked=True,
    )


def run_meta_gate(
    *,
    selected_mode: ProducerExecutionModeV1,
    baseline: VersionedResearchPolicyV1,
    updater: VersionedMetaPolicyUpdaterV1,
    aggregate: Mapping[str, Any],
    completed_round_index: int,
) -> MetaQualityReportV1:
    first = updater.update(
        baseline,
        completed_round_index=completed_round_index,
        aggregate=aggregate,
    )
    second = updater.update(
        baseline,
        completed_round_index=completed_round_index,
        aggregate=aggregate,
    )
    deterministic = first.to_dict() == second.to_dict()
    allowed_changes = {
        "version",
        "producer_token_allocation",
        "mechanism_axis_targeting",
        "memory_retrieval_policy",
        "router_priors",
        "acquisition_parameters",
        "predecessor_digest",
    }
    changed = {
        key
        for key, value in first.to_dict().items()
        if baseline.to_dict().get(key) != value
    }
    allowlist_valid = changed <= allowed_changes
    allocations = dict(first.producer_token_allocation)
    single_producer = (
        len(allocations) != 4
        or min(allocations.values(), default=0.0) < 0.10
        or max(allocations.values(), default=1.0) > 0.60
    )
    single_family = len(set(first.mechanism_axis_targeting)) < 3
    parameter_only = all("parameter" in item.lower() for item in first.mechanism_axis_targeting)
    candidate_leakage = "candidate" in str(aggregate).lower()
    calibration_valid = 0.0 <= float(aggregate.get("calibration_error", 1.0)) <= 0.2
    activated_policy_effect = (
        first.producer_token_allocation != baseline.producer_token_allocation
        and first.router_priors != baseline.router_priors
        and first.mechanism_axis_targeting != baseline.mechanism_axis_targeting
    )
    passed = (
        deterministic
        and completed_round_index >= 1
        and allowlist_valid
        and calibration_valid
        and not candidate_leakage
        and not single_producer
        and not single_family
        and not parameter_only
        and activated_policy_effect
    )
    return MetaQualityReportV1(
        selected_mode=selected_mode,
        baseline_policy_digest=baseline.digest,
        updated_policy_digest=first.digest,
        deterministic_replay=deterministic,
        round_boundary_only=completed_round_index >= 1,
        field_allowlist_valid=allowlist_valid,
        calibration_valid=calibration_valid,
        candidate_id_leakage=candidate_leakage,
        single_producer_collapse=single_producer,
        single_family_collapse=single_family,
        parameter_tuning_collapse=parameter_only,
        activated_policy_effect=activated_policy_effect,
        verdict=(
            MetaVerdictV1.PASS_VERSIONED_META if passed else MetaVerdictV1.FAIL
        ),
    )


def combine_research_quality_gate(
    *,
    agentization: AgentizationQualityGateV1,
    meta: MetaQualityReportV1 | None,
    sessions: Mapping[ProducerExecutionModeV1, ProducerSessionResultV1],
) -> ResearchCapabilityQualityGateV1:
    selected_session = (
        sessions.get(agentization.selected_mode)
        if agentization.selected_mode is not None
        else None
    )
    proposals = selected_session.proposals if selected_session else ()
    selected_calls = selected_session.calls if selected_session else ()
    call_candidate_ids = {
        candidate_id for call in selected_calls for candidate_id in call.candidate_ids
    }
    route = StrongStaticRouterV1().route(proposals)
    route_trace_complete = (
        route.ordered_candidate_ids == tuple(item.candidate_id for item in proposals)
        and len(route.decisions) == len(proposals)
        and {item.candidate_id for item in route.decisions}
        == {item.candidate_id for item in proposals}
    )
    lineage_complete = (
        sum(
            bool(
                item.producer_id
                and item.producer_role
                and item.assigned_before_call
                and item.candidate_id in call_candidate_ids
            )
            for item in proposals
        )
        / max(1, len(proposals))
    )
    post_hoc = sum(item.post_hoc_relabel for item in proposals)
    falsification = any(
        item.proposal_intent is ProposalIntentV1.FALSIFICATION for item in proposals
    )
    control_records = tuple(ControlAblationBuilderV1().build(item) for item in proposals)
    repair_records = tuple(
        RepairEngineerV1().build(item, "SYNTHETIC_BLOCKER") for item in proposals
    )
    credit_separated = (
        len(control_records) == len(proposals)
        and len(repair_records) == len(proposals)
        and all(
            item["discovery_credit"] == "NON_DISCOVERY_CONTROL"
            for item in control_records
        )
        and all(
            item["discovery_credit"] == "NON_DISCOVERY_REPAIR"
            for item in repair_records
        )
    )
    meta_verdict = meta.verdict if meta else MetaVerdictV1.FAIL
    pass_all = (
        agentization.verdict is AgentizationVerdictV1.PASS_INDEPENDENT_MULTI_AGENT
        and meta_verdict is MetaVerdictV1.PASS_VERSIONED_META
        and lineage_complete == 1.0
        and post_hoc == 0
        and falsification
        and credit_separated
        and route_trace_complete
    )
    return ResearchCapabilityQualityGateV1(
        agentization_gate_digest=agentization.digest,
        agentization_verdict=agentization.verdict,
        selected_mode=agentization.selected_mode,
        meta_report_digest=meta.digest if meta else None,
        meta_verdict=meta_verdict,
        progression_status="PASS" if pass_all else "FAIL",
        producer_lineage_complete=lineage_complete,
        post_hoc_relabel_count=post_hoc,
        falsification_slot_present=falsification,
        control_repair_credit_separated=credit_separated,
        complete_pool_and_route_trace=route_trace_complete,
        search_utility_only=True,
    )
