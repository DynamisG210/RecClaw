from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.helix.scientific_attribution import NOT_AVAILABLE
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_EVALUATOR,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    project_candidate_proposal_v4,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchProfileEntryOriginV1,
    adapt_current_search_profile,
    activate_next_fresh_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    EpisodeEvidenceClassV1,
    ResearchFailureClassV1,
)
from recclaw_core.research_line.execution import (
    execution_recipe_for_search_binding,
    project_common_execution_feedback,
)
from recclaw_core.research_line.interfaces import ProducerOutcome, ResearchContext


_E0_TEST_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "helix_abc_v1"
sys.path.insert(0, str(_E0_TEST_ROOT))

from test_e0_search_adapter import (  # noqa: E402
    _next_profile,
    _outside_66_program,
    _proposal,
    _qualified_capability,
)


def _context(profile: Any) -> ResearchContext:
    return ResearchContext(
        campaign_id=profile.campaign_id,
        round_index=1,
        knowledge_base={},
        frozen_goal={},
        frontier={},
        scientific_memory={},
        unresolved_questions=(),
        policy={},
        budget={},
        active_profile_ref=profile.profile_ref,
        active_profile_digest=profile.profile_digest,
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
    )


def _fixed_binding(
    profile: Any,
    mechanism_id: str,
) -> tuple[Any, ProducerOutcome, ResearchContext]:
    mechanism = next(
        item for item in executable_mechanisms() if item.mechanism_id == mechanism_id
    )
    proposal = _proposal(
        candidate_id=f"cand-execution-{mechanism_id.lower()}",
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        protocol_digest=profile.protocol_digest,
        selected=True,
    )
    entry = next(
        item
        for item in profile.entries
        if item.semantic_identity_digest == mechanism.mechanism_semantics_digest
    )
    binding = bind_search_candidate(
        profile=profile,
        proposal=proposal,
        capability_ref=entry.capability_ref,
    )
    context = _context(profile)
    bindings = frozen_search_bindings(
        context_ref=context.context_ref,
        context_digest=context.digest,
    )
    spec, facts = project_candidate_proposal_v4(
        proposal,
        bindings=bindings,
        required_budget={"implementation_tokens": 100},
    )
    outcome = ProducerOutcome(
        producer_role=spec.producer_role,
        context_ref=context.context_ref,
        context_digest=context.digest,
        spec=spec,
        resolution_facts=facts,
        source_proposal=proposal,
    )
    return binding, outcome, context


def _qualified_binding() -> tuple[Any, Any, Any]:
    current = adapt_current_search_profile(campaign_id="campaign:execution-current")
    fixed_binding, _outcome, _context_value = _fixed_binding(current, "BPR_MF")
    slate = freeze_experiment_slate(
        profile=current,
        bindings=(fixed_binding,),
        budget_snapshot={},
    )
    capability = _qualified_capability(
        profile=current,
        program=_outside_66_program(),
    )
    _registry, next_profile, _receipt = _next_profile(
        current_profile=current,
        current_slate=slate,
        capability=capability,
    )
    active = activate_next_fresh_search_profile(
        predecessor=current,
        next_profile=next_profile,
        registry=_registry,
        fresh_campaign_id="campaign:execution-next",
    )
    proposal = _proposal(
        candidate_id="cand-execution-qualified",
        mechanism_id="NGCF_EXECUTION_QUALIFIED",
        mechanism_axis="message_transform",
        mechanism_program=_outside_66_program(),
        protocol_digest=active.protocol_digest,
        selected=True,
    )
    binding = bind_search_candidate(
        profile=active,
        proposal=proposal,
        capability_ref=capability.capability_id,
    )
    return active, binding, capability


def _qualified_execution(capability: Any) -> dict[str, Any]:
    return {
        "capability_family": "NGCF_CUSTOM",
        "model": "E0QualifiedModel",
        "base_model_config": "LightGCN",
        "config": {"custom_operator": "message_transform"},
        "entrypoint_source_sha256": sha256_digest({"source": "qualified"}),
        "candidate_package_ref": capability.candidate_package_ref,
        "candidate_package_digest": capability.candidate_package_digest,
        "candidate_root_ref": "candidate-root:execution-qualified",
        "candidate_root_digest": sha256_digest({"root": "qualified"}),
        "candidate_source_tree_digest": capability.source_tree_digest,
    }


def test_fixed_catalog_bindings_keep_distinct_models_on_one_common_contract() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:execution-fixed")
    bpr_binding, _bpr_outcome, _context_value = _fixed_binding(profile, "BPR_MF")
    lightgcn_binding, _lightgcn_outcome, _context_value = _fixed_binding(
        profile,
        "LIGHTGCN",
    )

    bpr_recipe = execution_recipe_for_search_binding(
        bpr_binding,
        profile=profile,
    )
    lightgcn_recipe = execution_recipe_for_search_binding(
        lightgcn_binding,
        profile=profile,
    )

    assert bpr_recipe["model"] == "BPR"
    assert lightgcn_recipe["model"] != bpr_recipe["model"]
    assert bpr_recipe["dataset"] == lightgcn_recipe["dataset"] == "ml-1m"
    assert (
        bpr_recipe["split"]
        == lightgcn_recipe["split"]
        == "train/dev/heldout"
    )
    assert bpr_recipe["evaluator"] == lightgcn_recipe["evaluator"]
    assert bpr_recipe["execution_role"] == lightgcn_recipe["execution_role"] == "CANDIDATE"


def test_qualified_binding_requires_explicit_non_bpr_execution_inputs() -> None:
    profile, binding, capability = _qualified_binding()

    with pytest.raises(ValueError, match="requires explicit qualified execution"):
        execution_recipe_for_search_binding(binding, profile=profile)

    recipe = execution_recipe_for_search_binding(
        binding,
        profile=profile,
        qualified_execution=_qualified_execution(capability),
    )
    assert binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
    assert recipe["model"] == "E0QualifiedModel"
    assert recipe["model"] != "BPR"
    assert recipe["base_model_config"] == "LightGCN"
    assert recipe["capability_family"] == "NGCF_CUSTOM"

    invalid = _qualified_execution(capability)
    invalid["inferred_bpr_default"] = True
    with pytest.raises(ValueError, match="exactly the explicit execution inputs"):
        execution_recipe_for_search_binding(
            binding,
            profile=profile,
            qualified_execution=invalid,
        )


def _feedback_inputs() -> tuple[Any, ProducerOutcome, Any]:
    profile = adapt_current_search_profile(campaign_id="campaign:execution-feedback")
    binding, outcome, context = _fixed_binding(profile, "BPR_MF")
    return binding, outcome, context


def _candidate_run(*, status: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_status": status,
        "metrics": metrics,
        "experiment_binding_ref": "recclaw.research-line.experiment-binding.v1:candidate",
        "experiment_binding_digest": sha256_digest({"binding": "candidate"}),
        "wall_time_ms": 123,
        "resource_telemetry": {"gpu_cost_microunits": 7},
    }


def _incumbent() -> dict[str, Any]:
    return {
        "comparator_ref": "recclaw.research-line.experiment-binding.v1:comparator",
        "comparator_digest": sha256_digest({"binding": "comparator"}),
        "frozen_ndcg@10": 0.40,
    }


def _project_feedback(
    *,
    candidate_run: dict[str, Any],
) -> tuple[Any, ...]:
    binding, outcome, context = _feedback_inputs()
    return project_common_execution_feedback(
        context=context,
        selected_outcome=outcome,
        binding=binding,
        candidate_run=candidate_run,
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="seed:execution-1",
        next_discriminative_test="repeat the matched comparison",
    )


def test_success_projects_numeric_utility_but_inconclusive_episode() -> None:
    event, identity, episode, closure, failure_detail = _project_feedback(
        candidate_run=_candidate_run(
            status="SUCCESS",
            metrics={"ndcg@10": 0.45},
        )
    )

    assert event.comparator_delta == pytest.approx(0.05)
    assert event.runnable_observation == "RUNNABLE"
    assert identity.comparator_ref.endswith(":comparator")
    assert episode is not None
    assert episode.failure_class is ResearchFailureClassV1.INCONCLUSIVE
    assert episode.evidence_class is EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT
    assert episode.mechanism_interpretation == "NOT_ADJUDICATED"
    assert episode.mechanism_negative_evidence is False
    assert closure.failure_class is ResearchFailureClassV1.INCONCLUSIVE
    assert closure.mechanism_memory_allowed is False
    assert failure_detail is None


def test_cost_projection_keeps_bounded_reservation_evidence_without_aliasing_wall_time() -> None:
    reservation = {
        "schema": "recclaw.gpu-reservation-evidence.v1",
        "reservation_ref": "gpu-reservation:fixture",
        "identity": {
            "host": "gpu-fixture",
            "physical_gpu_id": "2",
            "cuda_visible_devices": "2",
            "reservation_owner_ref": "candidate-run:fixture",
            "device_uuid": "GPU-fixture",
            "exclusive": True,
            "scope": "CANDIDATE_PROCESS_TREE",
        },
        "identity_digest": sha256_digest({"identity": "fixture"}),
        "reservation_digest": sha256_digest({"reservation": "fixture"}),
    }
    candidate_run = _candidate_run(
        status="SUCCESS",
        metrics={"ndcg@10": 0.45},
    )
    candidate_run.update(
        {
            "parent_process_interval": {
                "parent_process_started_monotonic_ns": 100,
                "parent_process_ended_monotonic_ns": 2_000_000_100,
                "parent_process_interval_seconds": 2.0,
                "parent_process_interval_wall_time_ms": 2000,
            },
            "resource_telemetry_sha256": sha256_digest(
                candidate_run["resource_telemetry"]
            ),
            "resource_prediction": {
                "estimated_total_wall_time_seconds": 17.0,
            },
            "cuda_visible_devices": "2",
            "gpu_reservation_evidence": reservation,
            "gpu_reservation_status": "MEASURED_EXCLUSIVE_RESERVATION_INTERVAL",
            "reserved_gpu_worker_seconds": 2.0,
            "reserved_gpu_worker_seconds_semantics": (
                "EXCLUSIVE_RESERVATION_PARENT_PROCESS_INTERVAL;NOT_GPU_ACTIVE_TIME"
            ),
            "training_device_evidence": {
                "cuda_available": True,
                "current_device": 0,
            },
            "device_evidence_validation": "AVAILABLE_AND_CONSISTENT",
        }
    )

    event, *_ = _project_feedback(candidate_run=candidate_run)
    cost = event.resource_cost_projection

    assert cost["reserved_gpu_worker_seconds"] == pytest.approx(2.0)
    assert cost["wall_time_ms"] == 123
    assert cost["resource_prediction"]["estimated_total_wall_time_seconds"] == 17.0
    assert cost["gpu_reservation_evidence"] == reservation
    assert cost["parent_process_interval"]["parent_process_interval_seconds"] == 2.0
    assert "resource_telemetry" not in cost


@pytest.mark.parametrize(
    ("status", "metrics", "failure_class"),
    (
        ("RESOURCE_CENSORED", {}, ResearchFailureClassV1.RESOURCE),
        ("SUCCESS", {}, ResearchFailureClassV1.OUTCOME_MISSING),
    ),
)
def test_resource_and_missing_metric_results_are_engineering_diagnostics(
    status: str,
    metrics: dict[str, Any],
    failure_class: ResearchFailureClassV1,
) -> None:
    event, _identity, episode, closure, failure_detail = _project_feedback(
        candidate_run=_candidate_run(status=status, metrics=metrics)
    )

    assert event.comparator_delta == NOT_AVAILABLE
    assert event.runnable_observation == "NOT_RUNNABLE"
    assert episode is None
    assert closure.failure_class is failure_class
    assert closure.engineering_diagnostic_allowed is True
    assert closure.mechanism_memory_allowed is False
    assert closure.outcome_ref is None
    assert failure_detail is not None
    assert failure_detail["failure_class"] == failure_class.value
