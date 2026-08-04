from __future__ import annotations

import pytest
import jsonschema

from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerError,
    research_canary_prompt,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ProducerExecutionModeV1,
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
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DevelopmentalMechanismBeliefV1,
)
from recclaw_core.helix.contracts import PortAdjudication, PortStage, PortStatus
from recclaw_core.helix.guard_adapter import _recommended_validation
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
    INLINE_RESEARCH_TASK_TYPES,
)
from recclaw_core.helix.scientific_attribution import (
    DeterministicHelixAdmissionV13,
    FrontierEligibilityV2,
    PromptFeedbackProjectionV2,
    ResearchTaskQueueV1,
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
    SearchFeedbackClassV2,
    SearchUtilityEventV2,
    schema_for,
)


def test_executability_only_result_does_not_trigger_seed_confirmation() -> None:
    result = {
        "affected_claim_scope": {"protocol_branch_required": False},
        "evidence_admissibility": {
            "development_disposition": "RECORD_EXECUTABILITY_ONLY"
        },
    }
    assert _recommended_validation(result) == "NONE"


def test_seed_confirmation_is_not_an_inline_search_task() -> None:
    assert ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE not in INLINE_RESEARCH_TASK_TYPES


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _event() -> SearchUtilityEventV2:
    return SearchUtilityEventV2(
        candidate_semantic_digest=DIGEST_A,
        candidate_id="bl1_test",
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
        observation_seed="2026",
    )


def _task(
    task_type: ResearchTaskTypeV1 = ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE,
) -> ResearchTaskV1:
    return ResearchTaskV1(
        task_id="task-v13-test",
        task_type=task_type,
        candidate_id="bl1_test",
        candidate_semantic_digest=DIGEST_A,
        mechanism_program_digest=DIGEST_C,
        parent_candidate_id="bl1_parent",
        comparator_identity="BPR_MF",
        protocol_digest=DIGEST_B,
        required_seed_or_control="2027",
        task_status=ResearchTaskStatusV1.PENDING,
        created_round=1,
        utility_priority=0.7,
        missing_seed_count=2,
        mechanism_program={"schema_version": "test"},
    )


def _belief() -> DevelopmentalMechanismBeliefV1:
    return DevelopmentalMechanismBeliefV1(
        hypothesis_id="bl1_test",
        mechanism_axis="objective",
        competing_hypotheses=("null",),
        predicted_outcome_signature="positive delta",
        evidence_for=(DIGEST_A,),
        evidence_against=(),
        unresolved_confounds=("single_seed",),
        next_discriminative_test="matched control",
    )


def _post(
    *,
    status: PortStatus = PortStatus.ADJUDICATED,
    protocol_status: str = "CURRENT_PROTOCOL",
    evidence_use: str,
    recommended_validation: str = "NONE",
) -> PortAdjudication:
    return PortAdjudication(
        candidate_id="bl1_test",
        stage=PortStage.POST,
        status=status,
        protocol_status=protocol_status,
        outcome_class=evidence_use,
        claim_ceiling="DEVELOPMENT_ONLY",
        reason_codes=("PRIVATE_REASON",),
        comparator_delta=None,
        evidence_use=evidence_use,
        recommended_validation=recommended_validation,
    )


def test_null_post_is_baseline_search_event() -> None:
    fused, compact = DeterministicHelixAdmissionV13().admit_post(
        adjudication=_post(
            status=PortStatus.NOT_ADJUDICATED,
            evidence_use="NOT_ADJUDICATED",
        ),
        search_utility_event=_event(),
    )
    assert compact is None
    assert fused.search_feedback_class is SearchFeedbackClassV2.BASELINE_RESULT
    assert fused.frontier_eligibility is FrontierEligibilityV2.SEARCH_ELIGIBLE
    assert fused.meta_update_allowed


def test_preliminary_post_creates_public_validation_task_without_guard_fields() -> None:
    fused, compact = DeterministicHelixAdmissionV13().admit_post(
        adjudication=_post(
            evidence_use="COUNT_AS_LOCAL_PRELIMINARY_SIGNAL",
            recommended_validation="REQUIRES_CONFIRMATION",
        ),
        search_utility_event=_event(),
        validation_task=_task(),
    )
    assert compact is not None
    assert (
        fused.search_feedback_class
        is SearchFeedbackClassV2.PRELIMINARY_SEARCH_SIGNAL
    )
    assert (
        fused.frontier_eligibility
        is FrontierEligibilityV2.SEARCH_ELIGIBLE_PRELIMINARY
    )
    prompt = PromptFeedbackProjectionV2.from_fused(fused).to_dict()
    serialized = str(prompt).lower()
    for forbidden in (
        "claim_ceiling",
        "reason_codes",
        "protocol_status",
        "evidence_use",
        "guard",
        "fusion",
        "raw_result",
        "artifact",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "evidence_use",
    (
        "RECORD_DIAGNOSTIC_ONLY",
        "RECORD_EXECUTABILITY_ONLY",
        "RECORD_RUNTIME_BLOCKER_ONLY",
        "QUARANTINE_METRIC_MISSING",
        "QUARANTINE_PROVENANCE_INCOMPLETE",
    ),
)
def test_diagnostic_dispositions_update_memory_but_not_frontier_or_meta(
    evidence_use: str,
) -> None:
    fused, _compact = DeterministicHelixAdmissionV13().admit_post(
        adjudication=_post(evidence_use=evidence_use),
        search_utility_event=_event(),
    )
    assert fused.frontier_eligibility is FrontierEligibilityV2.EXCLUDED
    assert fused.search_utility_event is None
    assert not fused.meta_update_allowed
    assert fused.controller_update_allowed
    assert fused.search_memory_update_allowed
    prompt = PromptFeedbackProjectionV2.from_fused(fused).to_dict()
    assert prompt["diagnostic_slot"] == {
        "candidate_id": fused.candidate_id,
        "search_feedback_class": "DIAGNOSTIC_ONLY",
    }


def test_cross_protocol_result_cannot_update_current_search_state() -> None:
    fused, _compact = DeterministicHelixAdmissionV13().admit_post(
        adjudication=_post(evidence_use="EXCLUDE_FROM_CURRENT_CLAIM"),
        search_utility_event=_event(),
    )
    assert fused.frontier_eligibility is FrontierEligibilityV2.EXCLUDED
    assert fused.search_utility_event is None
    assert not fused.controller_update_allowed
    assert not fused.meta_update_allowed
    assert not fused.search_memory_update_allowed


def test_protocol_branch_has_only_a_generic_task() -> None:
    fused, compact = DeterministicHelixAdmissionV13().admit_post(
        adjudication=_post(
            protocol_status="PROTOCOL_BRANCH",
            evidence_use="EXCLUDE_FROM_CURRENT_CLAIM",
        ),
        search_utility_event=_event(),
        protocol_branch_task=_task(
            ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC
        ),
    )
    assert compact is not None
    assert fused.search_utility_event is None
    assert not fused.meta_update_allowed
    assert fused.research_task is not None
    assert set(fused.research_task.prompt_projection()) == {
        "task_type",
        "candidate_semantic_digest",
        "required_seed_or_control",
        "task_status",
    }


def test_no_search_update_contract_is_state_preserving() -> None:
    fused = DeterministicHelixAdmissionV13.no_search_update("bl1_test")
    assert fused.search_feedback_class is SearchFeedbackClassV2.NO_SEARCH_UPDATE
    assert fused.frontier_eligibility is FrontierEligibilityV2.EXCLUDED
    assert not fused.controller_update_allowed
    assert not fused.meta_update_allowed
    assert not fused.search_memory_update_allowed


def test_queue_is_deterministic_and_identity_bound() -> None:
    queue = ResearchTaskQueueV1()
    task = queue.enqueue(_task())
    assert queue.select_next() == task
    assert queue.activate(task.task_id).task_status is ResearchTaskStatusV1.ACTIVE
    assert queue.complete(task.task_id).task_status is ResearchTaskStatusV1.COMPLETED
    substituted = ResearchTaskV1(
        **{
            **_task().to_dict(),
            "candidate_semantic_digest": DIGEST_B,
            "task_status": ResearchTaskStatusV1.PENDING,
            "task_type": ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE,
        }
    )
    with pytest.raises(ValueError, match="identity substitution"):
        queue.enqueue(substituted)


def _controller() -> ResearchLineControllerV1:
    return ResearchLineControllerV1(
        producer_mode=(
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ),
        policy=initial_research_policy(),
        broker=FixtureProducerBrokerV1(),
        router=StrongStaticRouterV1(),
        memory_writer=SearchMemoryWriterV1(
            "DEVELOPMENT_ONLY/SEARCH_MEMORY"
        ),
    )


def _plan() -> ResearchRoundPlanV1:
    return ResearchRoundPlanV1(
        round_index=1,
        proposal_session_digest=DIGEST_A,
        route_trace_digest=DIGEST_B,
        selected_candidate_id="bl1_test",
        physical_call_count=4,
        proposal_count=4,
        ordinary_execution_opportunities=1,
        plan_status="SELECTED",
        policy_digest=DIGEST_C,
    )


def test_controller_consumes_only_closed_prompt_projection() -> None:
    fused, _compact = DeterministicHelixAdmissionV13().admit_post(
        adjudication=_post(
            status=PortStatus.NOT_ADJUDICATED,
            evidence_use="NOT_ADJUDICATED",
        ),
        search_utility_event=_event(),
    )
    controller = _controller()
    transition = controller.close_round_v13(
        plan=_plan(), feedback=fused, beliefs=()
    )
    projection = transition["search_memory_projection"]
    assert set(projection) == {
        "namespace",
        "round_index",
        "snapshot_digest",
        "beliefs",
        "prompt_feedback_projection",
    }
    assert projection["prompt_feedback_projection"] == (
        PromptFeedbackProjectionV2.from_fused(fused).to_dict()
    )


def test_no_update_does_not_advance_search_memory() -> None:
    controller = _controller()
    transition = controller.close_round_v13(
        plan=_plan(),
        feedback=DeterministicHelixAdmissionV13.no_search_update(
            "bl1_test"
        ),
        beliefs=(),
    )
    assert transition["state_changed"] is False
    assert transition["search_memory_projection"] is None
    assert controller.memory_writer.head is None


def test_task_only_feedback_cannot_smuggle_a_mechanism_belief() -> None:
    fused, _compact = DeterministicHelixAdmissionV13().admit_post(
        adjudication=_post(
            protocol_status="PROTOCOL_BRANCH",
            evidence_use="EXCLUDE_FROM_CURRENT_CLAIM",
        ),
        search_utility_event=_event(),
        protocol_branch_task=_task(
            ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC
        ),
    )
    with pytest.raises(ValueError, match="cannot carry mechanism beliefs"):
        _controller().close_round_v13(
            plan=_plan(),
            feedback=fused,
            beliefs=(_belief(),),
        )


def test_prompt_rejects_any_non_closed_feedback_projection() -> None:
    with pytest.raises(
        CanaryBrokerError,
        match="closed PromptFeedbackProjectionV2",
    ):
        research_canary_prompt(
            role="mechanism_composer",
            round_index=1,
            search_seed=1,
            memory_summary={
                "common_search_utility_slot": "ABSENT",
                "research_task_slot": "ABSENT",
                "guard_compact_feedback": {"reason_codes": ["PRIVATE"]},
            },
        )


def test_generated_schema_is_closed_and_validates_typed_payload() -> None:
    schema = schema_for(SearchUtilityEventV2)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(_event().to_dict(), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {**_event().to_dict(), "guard_reason_codes": []},
            schema,
        )
