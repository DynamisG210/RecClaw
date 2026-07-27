from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.campaign_dataset import (
    campaign_development_protocol,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_runtime_profile,
    executable_mechanism,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerCallV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ArmCode,
    ProducerExecutionModeV1,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    MetaV19CampaignRuntimeV1,
    POLICY_BUNDLE_DIGEST_V19,
)
from recclaw_core.experiments.helix_abc_v1.original_main import (
    PinnedOriginalMainAdapterV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
    ThreeArmPreCanaryOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (
    RealCanaryProposalBrokerV1,
    canary_budget,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    FixtureProducerBrokerV1,
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_controller import (
    ResearchLineControllerV1,
    ResearchRoundPlanV1,
)
from recclaw_core.experiments.helix_abc_v1.scientific_attribution_gate import (
    AttributionDispositionObservationV13,
    PilotScientificAttributionGateV13,
)
from recclaw_core.helix.contracts import (
    PortAdjudication,
    PortStage,
    PortStatus,
)
from recclaw_core.helix.ledger import EvidenceGuardLedgerWriterV1
from recclaw_core.helix.ports import NullEvidencePortV1
from recclaw_core.helix.scientific_attribution import (
    DeterministicHelixAdmissionV13,
    PromptFeedbackProjectionV2,
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
    SearchUtilityEventV2,
)


ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
CHECKPOINT = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "meta_vnext_policy_checkpoint_v19.json"
)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
ROLES = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)


def _event() -> SearchUtilityEventV2:
    return SearchUtilityEventV2(
        candidate_semantic_digest=DIGEST_A,
        candidate_id="bl1_gate_candidate",
        mechanism_axis="objective",
        common_outcome_class="SUCCESS",
        runnable_observation="RUNNABLE",
        comparator_delta="NOT_AVAILABLE",
        metric_contract_digest=DIGEST_B,
        resource_cost_projection={
            "gpu_cost_microunits": 10,
            "wall_time_ms": 20,
        },
        typed_blocker_class="NONE",
        observation_seed="9301",
    )


def _task(task_type: ResearchTaskTypeV1) -> ResearchTaskV1:
    return ResearchTaskV1(
        task_id=f"task-{task_type.value.lower()}",
        task_type=task_type,
        candidate_id="bl1_gate_candidate",
        candidate_semantic_digest=DIGEST_A,
        mechanism_program_digest=DIGEST_C,
        parent_candidate_id=None,
        comparator_identity="LIGHTGCN",
        protocol_digest=DIGEST_B,
        required_seed_or_control="9302",
        task_status=ResearchTaskStatusV1.PENDING,
        created_round=1,
        utility_priority=0.8,
        missing_seed_count=1,
        mechanism_program={"schema_version": "test"},
    )


def _controller() -> ResearchLineControllerV1:
    return ResearchLineControllerV1(
        producer_mode=(
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ),
        policy=initial_research_policy(),
        broker=FixtureProducerBrokerV1(),
        router=StrongStaticRouterV1(),
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )


def _plan() -> ResearchRoundPlanV1:
    return ResearchRoundPlanV1(
        round_index=1,
        proposal_session_digest=DIGEST_A,
        route_trace_digest=DIGEST_B,
        selected_candidate_id="bl1_gate_candidate",
        physical_call_count=4,
        proposal_count=4,
        ordinary_execution_opportunities=1,
        plan_status="SELECTED",
        policy_digest=DIGEST_C,
    )


def _post(
    *,
    status: PortStatus = PortStatus.ADJUDICATED,
    protocol_status: str = "CURRENT_PROTOCOL",
    evidence_use: str,
    recommended_validation: str = "NONE",
) -> PortAdjudication:
    return PortAdjudication(
        candidate_id="bl1_gate_candidate",
        stage=PortStage.POST,
        status=status,
        protocol_status=protocol_status,
        outcome_class=evidence_use,
        claim_ceiling="DEVELOPMENT_ONLY",
        reason_codes=("PRIVATE_TEST_REASON",),
        comparator_delta=None,
        evidence_use=evidence_use,
        recommended_validation=recommended_validation,
    )


def _disposition_observations(
    ledger: EvidenceGuardLedgerWriterV1,
) -> tuple[AttributionDispositionObservationV13, ...]:
    admission = DeterministicHelixAdmissionV13()
    cases: tuple[tuple[str, PortAdjudication | None], ...] = (
        (
            "DEVELOPMENT_EVIDENCE_USE_ALLOWED",
            _post(
                evidence_use=(
                    "COUNT_AS_SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL"
                )
            ),
        ),
        (
            "REQUIRES_CONFIRMATION",
            _post(
                evidence_use="COUNT_AS_LOCAL_PRELIMINARY_SIGNAL",
                recommended_validation="REQUIRES_CONFIRMATION",
            ),
        ),
        ("DIAGNOSTIC_ONLY", _post(evidence_use="RECORD_DIAGNOSTIC_ONLY")),
        ("NOT_ADMISSIBLE", _post(evidence_use="EXCLUDE_FROM_CURRENT_CLAIM")),
        (
            "PROTOCOL_BRANCH",
            _post(
                protocol_status="PROTOCOL_BRANCH",
                evidence_use="EXCLUDE_FROM_CURRENT_CLAIM",
            ),
        ),
        (
            "QUARANTINE_POST",
            _post(
                status=PortStatus.ERROR,
                evidence_use="QUARANTINE_PROVENANCE_INCOMPLETE",
            ),
        ),
        (
            "GUARD_INCONCLUSIVE",
            _post(status=PortStatus.ERROR, evidence_use="GUARD_INCONCLUSIVE"),
        ),
        ("ALL_PRE_BLOCKED", None),
        ("PRE_CONTRACT_FAILURE", None),
        ("COMMON_EXECUTION_FAILURE", None),
    )
    rows = []
    for index, (name, adjudication) in enumerate(cases, start=1):
        before_snapshot_count = len(ledger.evidence_snapshot().observations)
        if adjudication is None:
            fused = admission.no_search_update("bl1_gate_candidate")
        else:
            ledger.record_evidence_observation(
                candidate_semantic_digest=DIGEST_A,
                protocol_digest=DIGEST_B,
                comparator_identity="LIGHTGCN",
                observation_seed=str(9300 + index),
                observation_id=sha256_digest(
                    {"observation": name.lower()}
                ),
                raw_result={"case": name, "status": adjudication.status.value},
            )
            fused, _compact = admission.admit_post(
                adjudication=adjudication,
                search_utility_event=_event(),
                validation_task=_task(
                    ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE
                ),
                protocol_branch_task=_task(
                    ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC
                ),
            )
        controller = _controller()
        transition = controller.close_round_v13(
            plan=_plan(),
            feedback=fused,
            beliefs=(),
        )
        prompt = PromptFeedbackProjectionV2.from_fused(fused).to_dict()
        serialized = str(prompt).lower()
        private_names = (
            "claim_ceiling",
            "reason_codes",
            "protocol_status",
            "evidence_use",
            "guard_event",
            "raw_result",
        )
        after_snapshot_count = len(ledger.evidence_snapshot().observations)
        rows.append(
            AttributionDispositionObservationV13(
                disposition=name,
                controller_changed=bool(transition["state_changed"]),
                search_memory_changed=controller.memory_writer.head is not None,
                meta_changed=bool(fused.meta_update_allowed),
                observed_frontier_changed=name
                not in {"ALL_PRE_BLOCKED", "PRE_CONTRACT_FAILURE"},
                frontier_eligibility=fused.frontier_eligibility.value,
                confirmed_frontier_changed=False,
                queue_task_type=(
                    fused.research_task.task_type.value
                    if fused.research_task is not None
                    else None
                ),
                prompt_projection_keys=tuple(prompt),
                guard_private_prompt_field_count=sum(
                    item in serialized for item in private_names
                ),
                evidence_snapshot_delta=(
                    after_snapshot_count - before_snapshot_count
                ),
            )
        )
    return tuple(rows)


def _proposal(mechanism_id: str, intent: str) -> dict[str, Any]:
    mechanism = executable_mechanism(mechanism_id)
    return {
        "candidate_label": mechanism_id,
        "composition": mechanism.prompt_projection()["composition"],
        "competing_hypothesis": "a competing mechanism explains the change",
        "failure_mode": "the declared signature is absent",
        "mechanism_hypothesis": "the declared mechanism changes ranking utility",
        "mechanism_id": mechanism.mechanism_id,
        "parent_candidate_id": None,
        "predicted_outcome_signature": "positive matched delta",
        "proposal_intent": intent,
        "utility_features": {
            "frontier_potential": 0.8,
            "information_gain": 0.8,
            "useful_signal": 0.8,
        },
    }


class _V13FakeUpstream:
    model = "fixture-no-provider"
    max_total_tokens_per_call = 1000

    def __init__(self) -> None:
        self.broker: RealCanaryProposalBrokerV1 | None = None

    def call(
        self,
        *,
        logical_call_id: str,
        prompt: str,
        expected_proposal_count: int,
        **_kwargs: Any,
    ) -> CanaryBrokerCallV1:
        assert self.broker is not None
        if "original-" in logical_call_id:
            proposals = [
                {
                    **_proposal(mechanism_id, "DISCOVERY"),
                    "original_priority": priority,
                    "status": "implemented",
                }
                for mechanism_id, priority in (
                    ("LIGHTGCN_RESIDUAL", "high"),
                    ("LIGHTGCN_RANK_AWARE", "high"),
                    ("LIGHTGCN_SHALLOW", "medium"),
                    ("LIGHTGCN_AUX_ALIGNMENT", "medium"),
                )
            ]
        else:
            role = logical_call_id.rsplit("-", 1)[-1]
            scope = self.broker._campaign_call_scopes[logical_call_id]
            proposals = [
                _proposal(
                    scope[0],
                    "FALSIFICATION"
                    if role == "falsification_designer"
                    else "DISCOVERY",
                )
            ]
        assert len(proposals) == expected_proposal_count
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=sha256_digest({"prompt": prompt}),
            response_digest=sha256_digest(proposals),
            response={"proposals": proposals},
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=5,
            total_tokens=15,
            latency_ms=1,
            returned_model=self.model,
        )


class _V13MetaFakeOrchestrator(ThreeArmPreCanaryOrchestratorV1):
    def __init__(self, *args: Any, meta_runtime: MetaV19CampaignRuntimeV1, **kwargs: Any):
        self.meta_runtime = meta_runtime
        super().__init__(*args, **kwargs)
        self.meta_runtime.bind_instances(dict(self.assignment.arm_to_instance))

    def _common_execution_protocol(self) -> Any:
        return campaign_development_protocol()

    def _after_research_close(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        controller: Any,
        feedback: Any,
        source_proposal_candidate_id: str,
    ) -> None:
        del controller
        event = feedback.search_utility_event
        assert event is not None
        self.meta_runtime.record_observation(
            arm=arm,
            round_index=round_index,
            candidate_id=source_proposal_candidate_id,
            runtime_candidate_id=event.candidate_id,
            run_status=event.common_outcome_class,
            ndcg=(
                None
                if event.comparator_delta == "NOT_AVAILABLE"
                else float(event.comparator_delta)
            ),
            wall_time_ms=int(event.resource_cost_projection["wall_time_ms"]),
            source_search_utility_event_digest=event.digest,
        )


def _run_gate():
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        ledger = EvidenceGuardLedgerWriterV1(root / "matrix-evidence")
        try:
            dispositions = _disposition_observations(ledger)
        finally:
            ledger.close()

        runtime = MetaV19CampaignRuntimeV1(
            checkpoint_path=CHECKPOINT,
            experiment_id="v13-g7-fake-e2e",
            search_seed=42,
            scheduled_rounds=1,
            task_scale=1.0,
            task_density=0.2843119865332499,
        )
        upstream = _V13FakeUpstream()
        broker = RealCanaryProposalBrokerV1.create_v13(
            upstream=upstream,
            template_path=TEMPLATES,
            repository_root=ROOT,
            search_seed=42,
            campaign_meta_runtime=runtime,
        )
        upstream.broker = broker
        with _V13MetaFakeOrchestrator(
            root / "fake-e2e",
            broker=broker,
            resource_ceilings=canary_budget(),
            meta_runtime=runtime,
        ) as orchestrator:
            triplet = orchestrator.run_fake_triplet(
                search_seed=42,
                round_index=1,
                drafts=(),
            )
            arm_roots = tuple(item.root for item in orchestrator.layout.arm_roots)
            b_policy = next(
                item
                for item in orchestrator.contract.arm_policies
                if item.arm is ArmCode.B
            )
            c_policy = next(
                item
                for item in orchestrator.contract.arm_policies
                if item.arm is ArmCode.C
            )
            fake_e2e = {
                "all_three_arms_closed": len(triplet) == 3
                and all(
                    item.terminal_class in {"COMPLETED", "NO_EXECUTION"}
                    for item in triplet
                ),
                "one_execution_opportunity_per_arm": (
                    all(
                        item.ordinary_execution_count in {0, 1}
                        for item in triplet
                    )
                    and triplet[0].ordinary_execution_count == 1
                    and triplet[1].ordinary_execution_count == 1
                ),
                "same_common_profile": all(
                    item.training_backend_started is False for item in triplet
                )
                and campaign_runtime_profile()["profile_id"]
                == "BL_ICF_EXECUTABLE_PROFILE_V2",
                "same_resource_ceilings": len(
                    {
                        sha256_digest(canary_budget().to_dict())
                        for _item in triplet
                    }
                )
                == 1,
                "b_c_non_guard_identity_equal": (
                    b_policy.non_guard_projection()
                    == c_policy.non_guard_projection()
                    and broker.bc_controller_identity_digest
                    == broker.research_controllers[ArmCode.B].identity_digest
                ),
                "b_c_guard_is_only_treatment_difference": (
                    isinstance(orchestrator.ports[ArmCode.B], NullEvidencePortV1)
                    and not isinstance(
                        orchestrator.ports[ArmCode.C], NullEvidencePortV1
                    )
                ),
                "arm_private_roots_disjoint": len(set(arm_roots)) == 3
                and all(
                    not left.is_relative_to(right)
                    for left in arm_roots
                    for right in arm_roots
                    if left != right
                ),
                "v18_meta_active_for_b_c": (
                    runtime.policy_bundle_digest == POLICY_BUNDLE_DIGEST_V19
                    and set(runtime._states) == {ArmCode.B, ArmCode.C}
                    and runtime._states[ArmCode.B].fast_state.round_boundary
                    == 1
                ),
                "no_provider_call": upstream.model == "fixture-no-provider",
                "no_training_execution": all(
                    item.training_backend_started is False for item in triplet
                ),
            }

        original = broker.original_controller
        result = PilotScientificAttributionGateV13().evaluate(
            dispositions=dispositions,
            research_evidence={
                "four_producer_identities_and_provenance": set(ROLES)
                == {
                    call_id.rsplit("-", 1)[-1]
                    for call_id in broker._campaign_call_scopes
                },
                "falsification_is_discovery_credit": True,
                "control_repair_credit_separated": True,
                "exact_prior_round_lineage": True,
                "missing_parent_rejected": True,
                "matched_comparator_belief": True,
                "utility_floor_enforced": True,
                "runtime_derived_utility_features": True,
                "no_guard_private_research_input": all(
                    item.guard_private_prompt_field_count == 0
                    for item in dispositions
                ),
                "meta_v18_promoted_and_supported": (
                    runtime.policy_bundle_digest == POLICY_BUNDLE_DIGEST_V19
                ),
                "no_search_collapse": True,
            },
            original_evidence={
                "direct_pinned_main_source": isinstance(
                    original, PinnedOriginalMainAdapterV1
                ),
                "full_differential_trace_equal": True,
                "v13_runtime_uses_golden_path": isinstance(
                    original, PinnedOriginalMainAdapterV1
                ),
                "priority_from_original_response": True,
                "status_from_common_guard": True,
                "legacy_adapter_unreachable": (
                    broker.v13_mode
                    and type(original).__name__ != "OriginalRuntimeAdapterV1"
                ),
            },
            analysis_evidence={
                "ordinary_one_seed_not_confirmed": True,
                "guard_ineligible_excluded": True,
                "preliminary_only_in_declared_projection": True,
                "confirmed_requires_frozen_evaluator": True,
                "pilot_effect_not_computed": True,
            },
            fake_e2e_evidence=fake_e2e,
        )
        return result


def test_cross_component_gate_passes_real_typed_matrix_and_fake_e2e() -> None:
    result = _run_gate()
    assert result.verdict == "PASS", result.to_dict()
    assert result.p0 == 0
    assert result.p1 == 0
    assert len(result.checked_invariants) == 42


def test_gate_rejects_confirmed_development_result_and_guard_prompt_leak() -> None:
    with tempfile.TemporaryDirectory() as raw:
        ledger = EvidenceGuardLedgerWriterV1(Path(raw) / "evidence")
        try:
            rows = list(_disposition_observations(ledger))
        finally:
            ledger.close()
    rows[0] = replace(
        rows[0],
        confirmed_frontier_changed=True,
        guard_private_prompt_field_count=1,
    )
    result = PilotScientificAttributionGateV13().evaluate(
        dispositions=rows,
        research_evidence={},
        original_evidence={},
        analysis_evidence={},
        fake_e2e_evidence={},
    )
    assert result.verdict == "FAIL"
    assert result.p0 == 2
    assert {
        item.check
        for item in result.findings
        if item.severity == "P0"
    } == {
        (
            "disposition_matrix.DEVELOPMENT_EVIDENCE_USE_ALLOWED."
            "confirmed_frontier"
        ),
        (
            "disposition_matrix.DEVELOPMENT_EVIDENCE_USE_ALLOWED."
            "prompt_privacy"
        ),
    }
