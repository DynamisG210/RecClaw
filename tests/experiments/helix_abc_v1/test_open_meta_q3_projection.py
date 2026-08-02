from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.fresh_f1 import (  # noqa: E402
    F1_ROOT,
    R2_EXTERNAL_ROOT,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (  # noqa: E402
    R1_EXTERNAL_ROOT,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_f1 import (  # noqa: E402
    build_f1_replay_dataset,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_q3 import (  # noqa: E402
    HEAD_AUTHORITY_MATRIX,
    build_q3_acquisition_manifest,
    build_q3_denominator_projection,
    build_q3_policy_activation,
    consume_q3_active_policy,
    evaluate_q3_development_activation,
    fit_q3_three_head_policy,
    predict_q3_heads,
    project_q2_evidence_row,
    project_resource_receipt_rows,
    run_group_aware_offline_replay,
    shadow_compare_q3_policy,
)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _real_projection() -> dict[str, object]:
    docs = ROOT / "docs/research_line/vnext"
    q2 = ROOT / "results/research_line/q2_mechanism_characterization_20260802_01"
    closure_root = ROOT.parent / "RecClaw_f1_gpu35_closure"
    closure = closure_root / (
        "docs/research_line/vnext/"
        "F1_RESOURCE_COMPATIBLE_EXECUTION_CANONICAL_RECEIPT.json"
    )
    required = (
        R1_EXTERNAL_ROOT / "R1_CANONICAL_RECEIPT.json",
        R2_EXTERNAL_ROOT / "R2_CANONICAL_RECEIPT.json",
        F1_ROOT / "F1_CANONICAL_RECEIPT.json",
        closure,
    )
    assert all(path.is_file() for path in required), required
    f1_replay = build_f1_replay_dataset(
        r1_root=R1_EXTERNAL_ROOT, r2_root=R2_EXTERNAL_ROOT
    )
    resource_names = (
        ("Q0", "Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json"),
        ("Q0R", "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"),
        (
            "Q0R_FIXED_BATCH",
            "Q0R_FIXED_BATCH_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
        ),
        (
            "Q0R_TYPE_PRESERVING",
            "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
        ),
        ("Q0R2", "Q0R2_RESOURCE_ADMISSION_CANONICAL_RECEIPT.json"),
    )
    return build_q3_denominator_projection(
        f1_replay=f1_replay,
        resource_receipts=tuple(
            (stage, _read(docs / name)) for stage, name in resource_names
        ),
        f1_receipt=_read(F1_ROOT / "F1_CANONICAL_RECEIPT.json"),
        f1_closure_receipt=_read(closure),
        q3_package=_read(q2 / "Q3_MECHANISM_EVIDENCE_PACKAGE.json"),
        q2_result=_read(q2 / "Q2_PHYSICAL_RESULT.json"),
        q2_resource_result=_read(q2 / "RESOURCE_PROBE_RESULT.json"),
    )


def test_authority_matrix_file_matches_runtime_contract() -> None:
    matrix = _read(
        ROOT / "docs/research_line/vnext/Q3_OUTCOME_AWARE_HEAD_AUTHORITY_MATRIX.json"
    )
    assert matrix == HEAD_AUTHORITY_MATRIX
    assert matrix["held_out_reads"] == 0
    assert matrix["q2_binding"] == {
        "effect_update": False,
        "feasibility_update": (
            "RESOURCE_CENSORED_AND_PROTOCOL_NO_CHECKPOINT_COMPONENTS_ONLY"
        ),
        "mechanism_information_update": (
            "NON_IDENTIFIABLE_LABEL_WITH_GENERAL_PROBE_DESIGN_FEATURES"
        ),
        "resource_status_changes_mechanism_effect": False,
    }


def test_censored_projection_updates_only_feasibility() -> None:
    receipt = {
        "held_out_reads": 0,
        "probe_runs": {
            "arbitrary-arm": {
                "exit_status": "RESOURCE_CENSORED",
                "wall_time_ms": 300_000,
                "resource_deadline_seconds": 300,
                "mechanism_effect_update_allowed": False,
            }
        },
    }
    row = project_resource_receipt_rows("RESOURCE_TEST", receipt)[0]
    assert row["head_authority"] == {
        "effect": {
            "allowed": False,
            "reason": "RESOURCE_ONLY_EVIDENCE_CANNOT_UPDATE_EFFECT",
        },
        "feasibility": {
            "allowed": True,
            "reason": "RESOURCE_OR_COMPLETION_AUTHORITY_ONLY",
        },
        "mechanism_information": {
            "allowed": False,
            "reason": "RESOURCE_STATUS_HAS_NO_MECHANISM_AUTHORITY",
        },
    }
    assert row["head_inputs"]["feasibility"]["completion_label"] == 0
    assert row["head_inputs"]["effect"] is None
    assert row["head_inputs"]["mechanism_information"] is None


def test_q2_non_identifiable_updates_mechanism_but_never_effect() -> None:
    q2 = ROOT / "results/research_line/q2_mechanism_characterization_20260802_01"
    package = _read(q2 / "Q3_MECHANISM_EVIDENCE_PACKAGE.json")
    result = _read(q2 / "Q2_PHYSICAL_RESULT.json")
    resource = _read(q2 / "RESOURCE_PROBE_RESULT.json")
    row = project_q2_evidence_row(package, result, resource)
    assert row["head_authority"]["feasibility"]["allowed"] is True
    assert row["head_authority"]["mechanism_information"]["allowed"] is True
    assert row["head_authority"]["effect"]["allowed"] is False
    assert row["head_inputs"]["mechanism_information"] == {
        "authority": "REAL_Q2_PROBE_NON_EFFECT",
        "identifiable_information_label": 0,
        "mechanism_state": "NON_IDENTIFIABLE",
        "probe_cost": {
            "resource_probe_executions": 1,
            "wall_time_ms": 300246,
        },
        "probe_design_features": {
            "checkpoint_available_for_discriminative_probe": False,
            "declared_parent_surface_equivalence_verified": False,
            "full_ablation_executed": False,
            "structural_routing_participation_observed": True,
            "target_conditioning_observed": False,
        },
    }
    assert row["head_inputs"]["effect"] is None


def test_effect_field_mutation_cannot_change_other_head_inputs() -> None:
    q2 = ROOT / "results/research_line/q2_mechanism_characterization_20260802_01"
    package = _read(q2 / "Q3_MECHANISM_EVIDENCE_PACKAGE.json")
    result = _read(q2 / "Q2_PHYSICAL_RESULT.json")
    resource = _read(q2 / "RESOURCE_PROBE_RESULT.json")
    baseline = project_q2_evidence_row(package, result, resource)
    mutated_package = copy.deepcopy(package)
    mutated_package["ndcg@10"] = 1.0
    mutated_package["effect"] = 999.0
    mutated_result = copy.deepcopy(result)
    mutated_result["candidate_minus_parent"] = 999.0
    mutated = project_q2_evidence_row(mutated_package, mutated_result, resource)
    assert (
        baseline["head_inputs"]["feasibility"]
        == mutated["head_inputs"]["feasibility"]
    )
    assert (
        baseline["head_inputs"]["mechanism_information"]
        == mutated["head_inputs"]["mechanism_information"]
    )
    assert mutated["head_inputs"]["effect"] is None


def test_real_full_denominator_projection_preserves_negative_evidence() -> None:
    projection = _real_projection()
    assert projection["held_out_reads"] == 0
    assert projection["row_count"] == 46
    assert projection["source_counts"] == {
        "F1_INITIAL": 2,
        "F1_ORIGINAL_SEALED_DISPOSITION": 1,
        "F1_RESOURCE_COMPATIBLE": 2,
        "Q0": 4,
        "Q0R": 4,
        "Q0R2": 4,
        "Q0R_FIXED_BATCH": 4,
        "Q0R_TYPE_PRESERVING": 4,
        "Q2": 1,
        "R1": 16,
        "R2": 4,
    }
    assert projection["negative_evidence_preserved"][
        "resource_censored_count"
    ] >= 4
    assert projection["negative_evidence_preserved"][
        "resource_deferred_count"
    ] >= 3
    assert projection["negative_evidence_preserved"][
        "q2_non_identifiable_count"
    ] == 1
    q2_row = next(row for row in projection["rows"] if row["source_stage"] == "Q2")
    assert q2_row["head_inputs"]["effect"] is None
    effect_rows = [
        row for row in projection["rows"] if row["head_authority"]["effect"]["allowed"]
    ]
    assert effect_rows
    assert all(
        row["head_inputs"]["effect"]["comparability"]
        == "FULL_MATCHED_FRESH_DEVELOPMENT_EPISODE"
        for row in effect_rows
    )
    assert any(
        row["row_id"] == "f1-resource-compatible/resource_compatible_realization"
        and row["head_inputs"]["effect"]["parent_relative_effect"]
        == 0.1806 - 0.2053
        for row in effect_rows
    )


def test_real_three_head_learning_and_group_aware_replay_are_separate() -> None:
    projection = _real_projection()
    parent_policy = _read(F1_ROOT / "policy/versioned_policy.json")
    policy = fit_q3_three_head_policy(
        projection, parent_policy_digest=parent_policy["policy_digest"]
    )
    assert set(policy["heads"]) == {
        "feasibility",
        "mechanism_information",
        "effect",
    }
    assert policy["heads"]["feasibility"]["input_count"] == projection[
        "head_update_counts"
    ]["feasibility"]
    assert policy["heads"]["mechanism_information"]["state_counts"] == {
        "NON_IDENTIFIABLE": 1
    }
    assert policy["heads"]["effect"]["input_count"] == projection[
        "head_update_counts"
    ]["effect"]
    assert policy["acquisition_rules"]["IDEA"]["uses_effect_head"] is False
    assert policy["acquisition_rules"]["EXPERIMENT"]["uses_effect_head"] is False
    replay = run_group_aware_offline_replay(
        projection, parent_policy_digest=parent_policy["policy_digest"]
    )
    assert replay["row_count"] == projection["row_count"]
    assert replay["group_count"] > 1
    assert replay["official_held_out_reads"] == 0
    assert all(row["excluded_from_training"] for row in replay["predictions"])
    assert replay["metrics"]["mechanism_information"]["scored_count"] == 1
    assert replay["metrics"]["effect"]["scored_count"] == policy["heads"][
        "effect"
    ]["input_count"]


def test_effect_mutation_cannot_change_fitted_non_effect_heads() -> None:
    projection = _real_projection()
    parent_policy = _read(F1_ROOT / "policy/versioned_policy.json")
    baseline = fit_q3_three_head_policy(
        projection, parent_policy_digest=parent_policy["policy_digest"]
    )
    mutated = copy.deepcopy(projection)
    for row in mutated["rows"]:
        if row["head_inputs"]["effect"] is not None:
            row["head_inputs"]["effect"]["parent_relative_effect"] += 100.0
    refit = fit_q3_three_head_policy(
        mutated, parent_policy_digest=parent_policy["policy_digest"]
    )
    assert baseline["heads"]["feasibility"] == refit["heads"]["feasibility"]
    assert (
        baseline["heads"]["mechanism_information"]
        == refit["heads"]["mechanism_information"]
    )
    assert baseline["heads"]["effect"] != refit["heads"]["effect"]


def test_q2_general_probe_features_reduce_similar_mechanism_prediction() -> None:
    projection = _real_projection()
    parent_policy = _read(F1_ROOT / "policy/versioned_policy.json")
    policy = fit_q3_three_head_policy(
        projection, parent_policy_digest=parent_policy["policy_digest"]
    )
    q2 = next(row for row in projection["rows"] if row["source_stage"] == "Q2")
    similar = predict_q3_heads(
        policy,
        {
            "resource_stage": "Q2_MECHANISM_PROBE_RESOURCE",
            "probe_design_features": q2["head_inputs"]["mechanism_information"][
                "probe_design_features"
            ],
        },
    )
    verified = predict_q3_heads(
        policy,
        {
            "resource_stage": "Q2_MECHANISM_PROBE_RESOURCE",
            "probe_design_features": {
                name: not bool(value)
                for name, value in q2["head_inputs"]["mechanism_information"][
                    "probe_design_features"
                ].items()
            },
        },
    )
    assert similar["mechanism_information"]["posterior_mean"] < verified[
        "mechanism_information"
    ]["posterior_mean"]
    assert similar["effect"] == verified["effect"]


def test_shadow_activation_and_real_q1_pool_consumer_close_the_loop(
    tmp_path: Path,
) -> None:
    projection = _real_projection()
    parent_policy = _read(F1_ROOT / "policy/versioned_policy.json")
    q1_pool = _read(
        ROOT
        / "results/research_line/q1_prompt_contract_20260802_01/"
        "FROZEN_SELECTION_BEFORE_IMPLEMENTATION.json"
    )
    policy = fit_q3_three_head_policy(
        projection, parent_policy_digest=parent_policy["policy_digest"]
    )
    replay = run_group_aware_offline_replay(
        projection, parent_policy_digest=parent_policy["policy_digest"]
    )
    shadow = shadow_compare_q3_policy(
        projection=projection,
        q1_pool=q1_pool,
        parent_policy=parent_policy,
        q3_policy=policy,
    )
    assert {row["task_type"] for row in shadow["comparisons"]} == {
        "IDEA",
        "EXPERIMENT",
        "REPLICATION",
    }
    assert all(
        row["budget"] == 1 and row["candidate_count"] > 0
        for row in shadow["comparisons"]
    )
    promotion = evaluate_q3_development_activation(
        projection, replay, shadow, policy
    )
    assert promotion["status"] == "DEVELOPMENT_ONLY_ACTIVATION_GATE_PASS"
    assert all(promotion["gates"].values())
    activation = build_q3_policy_activation(
        policy,
        promotion,
        projection_digest=projection["projection_digest"],
        activation_id="q3-test-activation",
    )
    direct_manifest = build_q3_acquisition_manifest(
        policy=policy,
        activation=activation,
        frozen_pool=q1_pool,
        task_type="IDEA",
        random_seed=56031,
    )
    policy_path = tmp_path / "policy.json"
    activation_path = tmp_path / "activation.json"
    pool_path = tmp_path / "pool.json"
    for path, value in (
        (policy_path, policy),
        (activation_path, activation),
        (pool_path, q1_pool),
    ):
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    manifest = consume_q3_active_policy(
        policy_path=policy_path,
        activation_path=activation_path,
        frozen_pool_path=pool_path,
        task_type="IDEA",
        random_seed=56031,
    )
    assert manifest["selected_candidate_id"] == direct_manifest[
        "selected_candidate_id"
    ]
    assert manifest["consumer_input"]["read_active_policy_from_disk"] is True
    assert manifest["pool_provider_origin"] == "REAL_Q1_FROZEN_PRE_OUTCOME_POOL"
    assert manifest["pool_static_fixture"] is False
    assert manifest["candidate_count"] == 4
    assert manifest["selection_budget"] == 1
    assert manifest["exploration_probability"] == 0.15
    assert manifest["selection_probabilities_sum"] == 1.0
    assert sum(row["selected"] for row in manifest["candidates"]) == 1
    assert all(
        set(row["head_predictions"])
        == {"feasibility", "mechanism_information", "effect"}
        for row in manifest["candidates"]
    )
    assert all(
        "effect" not in row["selection_score_terms"]
        for row in manifest["candidates"]
    )
    experiment = consume_q3_active_policy(
        policy_path=policy_path,
        activation_path=activation_path,
        frozen_pool_path=pool_path,
        task_type="EXPERIMENT",
        random_seed=56032,
    )
    replication = consume_q3_active_policy(
        policy_path=policy_path,
        activation_path=activation_path,
        frozen_pool_path=pool_path,
        task_type="REPLICATION",
        random_seed=56033,
    )
    assert len({row["selection_score"] for row in experiment["candidates"]}) > 1
    assert all(
        "effect" not in row["selection_score_terms"]
        for row in experiment["candidates"]
    )
    assert all(
        set(row["selection_score_terms"])
        == {"effect_interval_width", "comparability", "reproduction_value"}
        for row in replication["candidates"]
    )
