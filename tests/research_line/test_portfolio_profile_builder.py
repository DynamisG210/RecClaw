from __future__ import annotations

from copy import deepcopy
import math

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.research_line.portfolio import (
    ParentValidationStateV2,
    PortfolioCandidateV2,
    ResourceAdmissionStateV2,
)
from recclaw_core.research_line.portfolio_profile_builder import (
    PortfolioProfileError,
    build_portfolio_candidate_profile_v2,
    build_portfolio_candidate_v2,
)


def _sha(label: str) -> str:
    return sha256_digest({"fixture": label})


def _mandate(
    *,
    producer_role: str = "frontier_architect",
    created_round: int = 4,
    frozen: bool = True,
    source_digest: str | None = None,
    exploration_floor_authorized: bool | None = None,
) -> dict[str, object]:
    mandate: dict[str, object] = {
        "kind": "EXPLORATION",
        "frozen": frozen,
        "source_digest": source_digest or _sha("exploration-mandate"),
        "source_ref": "mandate:test",
        "created_round": created_round,
        "priority": 0.6,
        "producer_role": producer_role,
    }
    if exploration_floor_authorized is not None:
        mandate["exploration_floor_authorized"] = exploration_floor_authorized
    return mandate


def _case(*, parent_id: str | None = "parent-1") -> dict[str, object]:
    semantic = _sha("candidate-semantic")
    package = _sha("candidate-package")
    source = _sha("candidate-source")
    candidate_identity: dict[str, object] = {
        "candidate_id": "candidate-new",
        "semantic_digest": semantic,
        "family_id": "family-a",
        "compute_pattern": "dense-v1",
        "resource_candidate_ref": "capability-new",
        "candidate_package_digest": package,
        "candidate_source_sha256": source,
    }
    binding: dict[str, object] = {
        "candidate_id": "candidate-new",
        "capability_ref": "capability-new",
        "mechanism_semantics_digest": semantic,
        "proposal": {
            "candidate_id": "candidate-new",
            "semantic_identity_digest": semantic,
            "producer_role": "frontier_architect",
        },
    }
    if parent_id is not None:
        candidate_identity["parent_id"] = parent_id
        binding["parent_candidate_id"] = parent_id
        binding["proposal"] = {
            **binding["proposal"],
            "parent_candidate_id": parent_id,
        }
    resolution: dict[str, object] = {
        "resolution": "SEARCH_READY",
        "resolved_current_capability_ref": "capability-new",
        "research_spec_digest": _sha("research-spec"),
    }
    candidate_identity["binding_digest"] = sha256_digest(binding)
    candidate_identity["resolution_digest"] = sha256_digest(resolution)

    return {
        "current_context": {
            "campaign_id": "research-test",
            "round_index": 5,
            "frozen_goal": {"metric": "NDCG@10", "direction": "maximize"},
            "active_profile_ref": "profile:test",
            "active_profile_digest": _sha("active-profile"),
            "protocol_ref": "protocol:test",
            "protocol_digest": _sha("protocol"),
        },
        "candidate_identity": candidate_identity,
        "binding": binding,
        "resolution": resolution,
        "task_queue": {
            "tasks": [
                {
                    "task_id": "task-new",
                    "candidate_id": "candidate-new",
                    "candidate_semantic_digest": semantic,
                    "parent_candidate_id": parent_id,
                    "priority": 0.75,
                    "created_round": 3,
                    "status": "PENDING",
                }
            ]
        },
        "prior_attempts": (),
        "calibration_prior": {
            "alpha": 2.0,
            "beta": 3.0,
            "frozen": True,
            "source_digest": _sha("pilot-calibration-prior"),
            "source_ref": "pilot-calibration:v1",
        },
        "family_history": (),
        "parent_history": (),
        "frontier_history": (),
        "resource_profile": {
            "profile_digest": _sha("resource-profile"),
            "candidate_ref": "capability-new",
            "candidate_package_digest": package,
            "candidate_source_sha256": source,
            "completion_probability": 0.8,
            "prediction_interval_seconds": [8.0, 12.0],
            "full_run_budget_after_probes_seconds": 100.0,
            "status": "RESOURCE_ADMITTED",
            "outcome_fields_consumed": [],
            "effect_fields_consumed": [],
            "held_out_reads": 0,
            "probe_process": {
                "process_isolated": True,
                "start_method": "spawn",
                "status": "RESULT",
                "exit_code": 0,
            },
            "schedule": [{"arm": "capability-new", "probability": 0.8}],
            "prediction": {
                "identity": {
                    "candidate_id": "candidate-new",
                    "candidate_ref": "capability-new",
                    "semantic_digest": semantic,
                    "candidate_package_digest": package,
                    "candidate_source_sha256": source,
                },
                "model": "FIXED_BATCH_THROUGHPUT_LINEAR_EXTRAPOLATION_V3",
                "completion_probability": 0.8,
                "prediction_interval_seconds": [8.0, 12.0],
                "estimated_total_wall_time_seconds": 10.0,
                "peak_memory_prediction_mib": 1000.0,
                "predicted_gpu_worker_seconds": 9.0,
            },
        },
    }


def _build(case: dict[str, object] | None = None):
    return build_portfolio_candidate_profile_v2(**(case or _case()))


def _worker_ceiling_risk_case() -> dict[str, object]:
    case = _case()
    profile = deepcopy(case["resource_profile"])
    prediction = profile["prediction"]
    prediction.update(
        {
            "prediction_interval_seconds": [2095.0, 7200.0],
            "estimated_total_wall_time_seconds": 3476.0,
            "prediction_interval_exceeds_worker_ceiling": True,
            "worker_ceiling_seconds": 3600.0,
            "training_only_lower_bound_seconds": 3300.0,
        }
    )
    profile.update(
        {
            "prediction_interval_seconds": [2095.0, 7200.0],
            "full_run_budget_after_probes_seconds": 3600.0,
            "prediction": prediction,
        }
    )
    case["resource_profile"] = profile
    return case


def test_builds_complete_candidate_with_identity_bound_worker_prediction() -> None:
    case = _case()
    first = _build(case)
    second = _build(deepcopy(case))

    assert isinstance(first.candidate, PortfolioCandidateV2)
    assert first.to_dict() == second.to_dict()
    assert first.candidate.predicted_gpu_seconds == 9.0
    assert first.candidate.resource_admission_state is ResourceAdmissionStateV2.RESOURCE_ADMITTED
    assert first.candidate.task_priority == 0.75
    assert first.candidate.age_rounds == 2
    assert first.candidate.repeat_count == 0
    assert first.evidence["selection_kind"] == "TASK"
    assert first.evidence["task"]["task_id"] == "task-new"
    assert first.evidence["exploration_mandate"] is None
    assert "task_record" in first.source_digests
    assert "exploration_mandate" not in first.source_digests
    assert first.source_digests["calibration_prior"] == _sha(
        "pilot-calibration-prior"
    )
    assert first.evidence["resource_admission"]["predicted_gpu_seconds_paths"] == [
        "prediction.predicted_gpu_worker_seconds",
    ]
    assert first.evidence["outcome_fields_consumed"] == []
    assert first.evidence["held_out_reads"] == 0


def test_convenience_contract_returns_only_the_existing_candidate_type() -> None:
    candidate = build_portfolio_candidate_v2(**_case())

    assert isinstance(candidate, PortfolioCandidateV2)
    assert candidate.candidate_id == "candidate-new"


def test_exact_task_selection_keeps_existing_priority_order() -> None:
    case = _case()
    semantic = case["candidate_identity"]["semantic_digest"]
    case["task_queue"] = {
        "tasks": [
            {
                "task_id": "lower-priority-task",
                "candidate_id": "candidate-new",
                "candidate_semantic_digest": semantic,
                "parent_candidate_id": "parent-1",
                "priority": 0.4,
                "created_round": 2,
                "status": "PENDING",
            },
            {
                "task_id": "higher-priority-task",
                "candidate_id": "candidate-new",
                "candidate_semantic_digest": semantic,
                "parent_candidate_id": "parent-1",
                "priority": 0.9,
                "created_round": 4,
                "status": "ACTIVE",
            },
        ],
    }

    result = _build(case)

    assert result.evidence["selection_kind"] == "TASK"
    assert result.evidence["task"]["task_id"] == "higher-priority-task"
    assert result.candidate.task_priority == 0.9


def test_empty_queue_with_valid_exploration_mandate_builds_without_fabricating_task() -> None:
    case = _case(parent_id=None)
    case["task_queue"] = {"tasks": []}
    case["exploration_mandate"] = _mandate()

    result = _build(case)

    assert result.candidate.task_priority == 0.6
    assert result.candidate.age_rounds == 1
    assert result.evidence["selection_kind"] == "EXPLORATION_MANDATE"
    assert result.evidence["task"] is None
    assert result.evidence["exploration_mandate"]["kind"] == "EXPLORATION"
    assert result.evidence["exploration_mandate"]["mandate_digest"] == (
        result.source_digests["exploration_mandate"]
    )
    assert "task_record" not in result.source_digests


def test_empty_queue_without_exploration_mandate_still_fails_fast() -> None:
    case = _case(parent_id=None)
    case["task_queue"] = {"tasks": []}

    with pytest.raises(PortfolioProfileError, match="exploration_mandate"):
        _build(case)


@pytest.mark.parametrize(
    "mandate",
    (
        _mandate(producer_role="wrong-role"),
        _mandate(source_digest="not-a-digest"),
        _mandate(created_round=6),
        _mandate(frozen=False),
    ),
)
def test_exploration_mandate_role_source_time_and_frozen_gates_are_strict(
    mandate: dict[str, object],
) -> None:
    case = _case(parent_id=None)
    case["task_queue"] = {"tasks": []}
    case["exploration_mandate"] = mandate

    with pytest.raises(PortfolioProfileError):
        _build(case)


def test_queued_confirmation_requires_explicit_exploration_floor_authorization() -> None:
    case = _case(parent_id=None)
    case["task_queue"] = {
        "tasks": [
            {
                "task_id": "confirmation-task",
                "candidate_id": "candidate-other",
                "candidate_semantic_digest": _sha("candidate-other"),
                "parent_candidate_id": None,
                "priority": 0.9,
                "created_round": 4,
                "status": "PENDING",
            }
        ]
    }
    case["exploration_mandate"] = _mandate()

    with pytest.raises(PortfolioProfileError, match="exploration_floor_authorized"):
        _build(case)


def test_queued_confirmation_and_floor_mandate_allow_explicit_exploration() -> None:
    case = _case(parent_id=None)
    case["task_queue"] = {
        "tasks": [
            {
                "task_id": "confirmation-task",
                "candidate_id": "candidate-other",
                "candidate_semantic_digest": _sha("candidate-other"),
                "parent_candidate_id": None,
                "priority": 0.9,
                "created_round": 4,
                "status": "ACTIVE",
            }
        ]
    }
    case["exploration_mandate"] = _mandate(
        exploration_floor_authorized=True,
    )

    result = _build(case)

    assert result.evidence["selection_kind"] == "EXPLORATION_MANDATE"
    assert result.evidence["exploration_mandate"][
        "queue_has_other_pending_or_active"
    ] is True
    assert result.evidence["exploration_mandate"][
        "exploration_floor_authorized"
    ] is True


def test_sparse_new_candidate_uses_explicit_beta_evidence_prior() -> None:
    result = _build(_case(parent_id=None))

    assert result.candidate.valid_seal_probability == pytest.approx(0.4)
    assert result.candidate.family_delta == 0.0
    assert result.candidate.parent_delta == 0.0
    assert result.candidate.frontier_gain == 0.0
    assert result.candidate.parent_state is ParentValidationStateV2.INDEPENDENT
    assert result.candidate.information_value == 0.75
    calibration = result.evidence["calibration"]
    assert calibration["calibration_prior"]["alpha"] == 2.0
    assert calibration["calibration_prior"]["beta"] == 3.0
    assert calibration["calibration_prior"]["source_digest"] == _sha(
        "pilot-calibration-prior"
    )
    assert calibration["exploration_uncertainty"] == 1.0
    assert calibration["levels"]["exact"]["attempt_count"] == 0


def test_missing_calibration_prior_is_rejected_when_no_sealed_history_exists() -> None:
    case = _case(parent_id=None)
    case.pop("calibration_prior")

    with pytest.raises(PortfolioProfileError, match="calibration_prior"):
        _build(case)


@pytest.mark.parametrize(
    "prior",
    (
        {
            "alpha": 0.0,
            "beta": 1.0,
            "frozen": True,
            "source_digest": _sha("invalid-alpha"),
        },
        {
            "alpha": 1.0,
            "beta": float("inf"),
            "frozen": True,
            "source_digest": _sha("invalid-beta"),
        },
        {
            "alpha": 1.0,
            "beta": 1.0,
            "frozen": False,
            "source_digest": _sha("unfrozen"),
        },
        {
            "alpha": 1.0,
            "beta": 1.0,
            "frozen": True,
            "source_digest": "not-a-digest",
        },
    ),
)
def test_calibration_prior_requires_frozen_finite_positive_beta_contract(
    prior: dict[str, object],
) -> None:
    case = _case(parent_id=None)
    case["calibration_prior"] = prior

    with pytest.raises(PortfolioProfileError):
        _build(case)


def test_hierarchical_calibration_is_sealed_and_shrunk_at_each_level() -> None:
    case = _case()
    case["calibration_prior"] = None
    semantic = case["candidate_identity"]["semantic_digest"]
    case["prior_attempts"] = [
        {
            "round": 1,
            "candidate_id": "candidate-other-1",
            "semantic_digest": _sha("other-1"),
            "family_id": "family-b",
            "compute_pattern": "dense-v1",
            "sealed": True,
            "sealed_valid_seal": True,
        },
        {
            "round": 2,
            "candidate_id": "candidate-other-2",
            "semantic_digest": _sha("other-2"),
            "family_id": "family-a",
            "compute_pattern": "dense-v1",
            "sealed": True,
            "sealed_valid_seal": False,
        },
        {
            "round": 3,
            "candidate_id": "candidate-other-3",
            "semantic_digest": _sha("other-3"),
            "family_id": "family-c",
            "compute_pattern": "sparse-v1",
            "sealed": True,
            "sealed_valid_seal": True,
        },
        {
            "round": 4,
            "candidate_id": "candidate-new",
            "semantic_digest": semantic,
            "family_id": "family-a",
            "compute_pattern": "dense-v1",
            "sealed": True,
            "sealed_valid_seal": True,
            "sealed_resource_admitted": True,
        },
        {
            "round": 4,
            "candidate_id": "candidate-new",
            "semantic_digest": semantic,
            "family_id": "family-a",
            "compute_pattern": "dense-v1",
            "sealed": False,
        },
    ]

    result = _build(case)
    levels = result.evidence["calibration"]["levels"]

    assert [levels[name]["attempt_count"] for name in ("global", "compute", "family", "exact")] == [4, 3, 2, 1]
    assert [levels[name]["valid_seal_count"] for name in ("global", "compute", "family", "exact")] == [3, 2, 1, 1]
    assert result.candidate.valid_seal_probability == pytest.approx(0.7333333333333334)
    assert result.candidate.repeat_count == 2


def test_explicit_prior_cannot_overpower_sufficient_sealed_history() -> None:
    case = _case(parent_id=None)
    case["calibration_prior"] = {
        "alpha": 1.0,
        "beta": 1.0,
        "frozen": True,
        "source_digest": _sha("weak-prior"),
    }
    attempts = []
    for index in range(100):
        attempts.append(
            {
                "round": (index % 4) + 1,
                "candidate_id": f"candidate-history-{index}",
                "semantic_digest": _sha(f"history-{index}"),
                "family_id": "family-history",
                "compute_pattern": "dense-history",
                "sealed": True,
                "sealed_valid_seal": index < 90,
            }
        )
    case["prior_attempts"] = attempts

    result = _build(case)

    assert result.candidate.valid_seal_probability == pytest.approx(
        (1.0 + 90.0) / (1.0 + 1.0 + 100.0),
        abs=0.02,
    )
    assert result.candidate.valid_seal_probability > 0.85
    assert result.evidence["calibration"]["levels"]["global"]["source"] == (
        "BETA_PRIOR_PLUS_SEALED_GLOBAL_POSTERIOR"
    )


def test_family_parent_frontier_priors_use_stable_history_with_shrinkage() -> None:
    case = _case()
    semantic = case["candidate_identity"]["semantic_digest"]
    case["family_history"] = [
        {"round": 1, "family_id": "family-a", "stable": True, "stable_delta": 0.4},
        {"round": 2, "family_id": "family-a", "stable": True, "stable_delta": 0.2},
        {"round": 3, "family_id": "family-b", "stable": True, "stable_delta": -0.4},
        {"round": 4, "family_id": "family-a", "stable": False},
    ]
    case["parent_history"] = [
        {
            "round": 1,
            "parent_id": "parent-1",
            "stable": True,
            "stable_validation": "VALIDATED",
            "stable_lineage_risk": 0.2,
            "stable_delta": 0.4,
        },
        {
            "round": 2,
            "parent_id": "parent-1",
            "stable": True,
            "stable_validation": "FAILED",
            "stable_lineage_risk": 0.8,
            "stable_delta": -0.2,
        },
        {
            "round": 3,
            "parent_id": "parent-other",
            "stable": True,
            "stable_validation": "VALIDATED",
            "stable_lineage_risk": 0.0,
            "stable_delta": 0.0,
        },
    ]
    case["frontier_history"] = [
        {
            "round": 1,
            "family_id": "family-a",
            "candidate_id": "candidate-new",
            "semantic_digest": semantic,
            "stable": True,
            "stable_frontier_gain": 0.6,
        },
        {
            "round": 2,
            "family_id": "family-a",
            "candidate_id": "candidate-other",
            "semantic_digest": _sha("frontier-other"),
            "stable": True,
            "stable_frontier_gain": 0.2,
        },
        {
            "round": 3,
            "family_id": "family-b",
            "candidate_id": "candidate-b",
            "semantic_digest": _sha("frontier-b"),
            "stable": True,
            "stable_frontier_gain": -0.1,
        },
    ]

    result = _build(case)

    assert result.candidate.family_delta == pytest.approx(0.18333333333333338)
    assert result.candidate.parent_delta == pytest.approx(0.08333333333333333)
    assert result.candidate.frontier_gain == pytest.approx(0.4111111111111111)
    assert result.candidate.parent_state is ParentValidationStateV2.FAILED
    assert 0.0 < result.candidate.lineage_risk < 1.0
    assert result.evidence["priors"]["family"]["stable_only"] is True
    assert result.evidence["priors"]["parent"]["parent_delta"]["stable_only"] is True


def test_temporal_cutoff_rejects_current_round_history() -> None:
    case = _case()
    row = {
        "round": 5,
        "candidate_id": "candidate-other",
        "semantic_digest": _sha("future"),
        "family_id": "family-a",
        "compute_pattern": "dense-v1",
        "sealed": True,
        "sealed_valid_seal": True,
    }
    case["prior_attempts"] = [row]

    with pytest.raises(PortfolioProfileError, match="temporal cutoff"):
        _build(case)


@pytest.mark.parametrize("field", ("current_outcome", "current_effect", "metric"))
def test_current_outcome_or_effect_fields_are_rejected(field: str) -> None:
    case = _case()
    case["current_context"] = {
        **case["current_context"],
        field: {"value": 1.0},
    }

    with pytest.raises(PortfolioProfileError, match="outcome/effect"):
        _build(case)


def test_history_and_resource_identity_mismatches_fail_closed() -> None:
    mismatch = _case()
    mismatch["binding"] = {
        **mismatch["binding"],
        "mechanism_semantics_digest": _sha("different-semantic"),
    }
    with pytest.raises(PortfolioProfileError, match="identity mismatch"):
        _build(mismatch)

    resource_mismatch = _case()
    resource_mismatch["resource_profile"] = {
        **resource_mismatch["resource_profile"],
        "candidate_ref": "capability-other",
    }
    with pytest.raises(PortfolioProfileError, match="candidate_ref"):
        _build(resource_mismatch)

    missing_gpu = _case()
    missing_prediction = deepcopy(missing_gpu["resource_profile"]["prediction"])
    del missing_prediction["predicted_gpu_worker_seconds"]
    missing_gpu["resource_profile"] = {
        **missing_gpu["resource_profile"],
        "prediction": missing_prediction,
    }
    with pytest.raises(PortfolioProfileError, match="not substituted"):
        _build(missing_gpu)


@pytest.mark.parametrize(
    ("location", "value"),
    (
        ("completion_probability", float("nan")),
        ("prediction_interval_seconds", [8.0, float("inf")]),
        ("estimated_total_wall_time_seconds", float("inf")),
        ("peak_memory_prediction_mib", float("nan")),
        ("predicted_gpu_worker_seconds", float("nan")),
        ("full_run_budget_after_probes_seconds", float("inf")),
    ),
)
def test_nonfinite_resource_prediction_or_budget_is_rejected(
    location: str,
    value: object,
) -> None:
    case = _case()
    profile = deepcopy(case["resource_profile"])
    if location == "full_run_budget_after_probes_seconds":
        profile[location] = value
    elif location == "completion_probability":
        profile[location] = value
        profile["prediction"][location] = value
    elif location == "prediction_interval_seconds":
        profile["prediction"][location] = value
        profile[location] = value
    else:
        profile["prediction"][location] = value
    case["resource_profile"] = profile

    with pytest.raises(PortfolioProfileError):
        _build(case)


def test_zero_resource_budget_remains_fail_closed() -> None:
    case = _case()
    profile = deepcopy(case["resource_profile"])
    profile["full_run_budget_after_probes_seconds"] = 0.0
    case["resource_profile"] = profile

    with pytest.raises(PortfolioProfileError, match="finite budget"):
        _build(case)


def test_production_shape_worker_ceiling_risk_interval_is_admitted_without_clamping() -> None:
    result = _build(_worker_ceiling_risk_case())

    assert result.candidate.resource_admission_state is ResourceAdmissionStateV2.RESOURCE_ADMITTED
    resource = result.evidence["resource_admission"]
    assert resource["prediction_interval_seconds"] == [2095.0, 7200.0]
    assert resource["estimated_total_wall_time_seconds"] == 3476.0
    assert resource["full_run_budget_after_probes_seconds"] == 3600.0


def test_worker_ceiling_may_be_stricter_than_finite_campaign_budget() -> None:
    case = _worker_ceiling_risk_case()
    profile = case["resource_profile"]
    profile["full_run_budget_after_probes_seconds"] = 7200.0
    profile["prediction"]["prediction_interval_seconds"] = [2095.0, 11769.0]
    profile["prediction_interval_seconds"] = [2095.0, 11769.0]

    result = _build(case)

    resource = result.evidence["resource_admission"]
    assert resource["prediction_interval_seconds"] == [2095.0, 11769.0]
    assert resource["full_run_budget_after_probes_seconds"] == 7200.0


@pytest.mark.parametrize(
    "failure",
    (
        "missing_flag",
        "false_flag",
        "ceiling_exceeds_budget",
        "point_over_budget",
        "missing_training_lower",
        "nonfinite_training_lower",
        "training_lower_over_budget",
    ),
)
def test_worker_ceiling_risk_contract_is_fail_closed(failure: str) -> None:
    case = _worker_ceiling_risk_case()
    prediction = case["resource_profile"]["prediction"]
    if failure == "missing_flag":
        prediction.pop("prediction_interval_exceeds_worker_ceiling")
    elif failure == "false_flag":
        prediction["prediction_interval_exceeds_worker_ceiling"] = False
    elif failure == "ceiling_exceeds_budget":
        prediction["worker_ceiling_seconds"] = 3601.0
    elif failure == "point_over_budget":
        prediction["estimated_total_wall_time_seconds"] = 3601.0
    elif failure == "missing_training_lower":
        prediction.pop("training_only_lower_bound_seconds")
    elif failure == "nonfinite_training_lower":
        prediction["training_only_lower_bound_seconds"] = float("nan")
    elif failure == "training_lower_over_budget":
        prediction["training_only_lower_bound_seconds"] = 3601.0
    else:  # pragma: no cover - protects this focused contract if extended.
        raise AssertionError(failure)

    with pytest.raises(PortfolioProfileError):
        _build(case)


def test_deferred_resource_profile_is_quarantined_without_becoming_runnable() -> None:
    case = _case()
    profile = deepcopy(case["resource_profile"])
    profile["status"] = "RESOURCE_DEFERRED"
    profile["completion_probability"] = 0.0
    profile["prediction"]["completion_probability"] = 0.0
    profile.pop("probe_process")
    profile["schedule"] = []
    case["resource_profile"] = profile

    result = _build(case)

    assert result.candidate.resource_admission_state is ResourceAdmissionStateV2.QUARANTINED
    assert result.candidate.valid_seal_probability == pytest.approx(0.4)
    assert result.evidence["resource_admission"]["admission_evidence"] == (
        "EXPLICIT_DEFERRED_OR_INFEASIBLE_STATUS"
    )


def test_current_outcome_is_not_consumed_from_history_or_resource_profile() -> None:
    case = _case()
    case["prior_attempts"] = [
        {
            "round": 1,
            "candidate_id": "candidate-other",
            "semantic_digest": _sha("other"),
            "family_id": "family-a",
            "compute_pattern": "dense-v1",
            "sealed": True,
            "sealed_valid_seal": True,
            "current_result": 0.99,
        }
    ]
    with pytest.raises(PortfolioProfileError, match="outcome/effect"):
        _build(case)

    case = _case()
    case["resource_profile"] = {
        **case["resource_profile"],
        "observed_effect": {"delta": 1.0},
    }
    with pytest.raises(PortfolioProfileError, match="outcome/effect"):
        _build(case)


def test_explicit_gpu_seconds_field_is_not_inferred_from_wall_time() -> None:
    case = _case()
    profile = deepcopy(case["resource_profile"])
    prediction = profile["prediction"]
    del prediction["predicted_gpu_worker_seconds"]
    profile["prediction"] = prediction
    profile["predicted_gpu_seconds"] = 7.5
    # An explicitly named field is consumed as supplied; arbitrary wall-time
    # remains rejected when no such field exists.
    result = _build({**case, "resource_profile": profile})
    assert math.isclose(result.candidate.predicted_gpu_seconds, 7.5)
