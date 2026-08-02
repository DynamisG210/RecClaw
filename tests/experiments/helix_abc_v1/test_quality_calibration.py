from __future__ import annotations

from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (
    build_active_r2_profile,
    build_r1_registry,
    load_registered_r1_artifacts,
    public_active_profile_catalog,
)
from recclaw_core.experiments.helix_abc_v1.quality_calibration import (
    NULL_NDCG_ABS_TOLERANCE,
    Q0_PROPOSAL_SEEDS,
    _evaluate_q0,
    audit_prior_evidence,
    build_prefrozen_manifest,
    render_q0_producer_prompt,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"


def _active_inputs() -> tuple[object, tuple[dict[str, str], ...]]:
    artifacts, _receipt = load_registered_r1_artifacts(ROOT)
    registry = build_r1_registry(artifacts)
    _current, _manifest, _next, _build, active = build_active_r2_profile(registry)
    catalog = public_active_profile_catalog(
        active,
        artifacts,
        seed=Q0_PROPOSAL_SEEDS["parent_equivalent_null"],
    )
    return active, catalog


def test_q0_prior_audit_consumes_exact_episode_and_resource_denominators() -> None:
    audit = audit_prior_evidence(ROOT)

    assert audit["typed_episode_count"] == 8
    assert audit["r1_typed_episode_count"] == 7
    assert audit["r2_typed_episode_count"] == 1
    assert audit["resource_censored_count"] == 6
    assert audit["resource_censored_mechanism_effect_updates"] == 0
    assert audit["completed_candidate_delta_summary"]["all_negative"] is True
    assert audit["implementation_drift"]["source_trees_equal"] is False
    assert all(
        row["mechanism_effect_update"] is False
        for row in audit["resource_censored_rows"]
    )


def test_q0_manifest_freezes_three_real_consumer_arms_before_outcomes() -> None:
    active, catalog = _active_inputs()
    manifest = build_prefrozen_manifest(active=active, catalog=catalog)

    assert manifest["selection_frozen_before_provider_or_runtime_outcomes"] is True
    assert manifest["arm_order"] == [
        "parent_equivalent_null",
        "known_good_reference",
        "frontier_candidate",
    ]
    assert manifest["common_runtime"]["epochs"] == 100
    assert manifest["common_runtime"]["held_out_reads"] == 0
    assert manifest["arms"]["parent_equivalent_null"]["registered_as_capability"] is False
    assert manifest["arms"]["known_good_reference"]["historical_canary_ndcg_at_10"] == 0.1108
    assert manifest["arms"]["frontier_candidate"]["selection_rule"].startswith(
        "one preassigned origin-blind"
    )


def test_q0_producer_prompt_is_origin_blind_and_arm_exact() -> None:
    active, catalog = _active_inputs()
    manifest = build_prefrozen_manifest(active=active, catalog=catalog)
    targets = {
        arm: row["target_semantics_digest"]
        for arm, row in manifest["arms"].items()
        if row["target_semantics_digest"] is not None
    }
    prompt = render_q0_producer_prompt(
        (RESOURCE_ROOT / "quality_calibration_producer_prompt_v1.txt").read_text(
            encoding="utf-8"
        ),
        arm="frontier_candidate",
        role="frontier_architect",
        seed=Q0_PROPOSAL_SEEDS["frontier_candidate"],
        active=active,
        catalog=catalog,
        targets=targets,
    )

    assert "one NOT_EXPRESSIBLE structural frontier" in prompt
    assert "side_a" not in prompt
    assert "side_b" not in prompt
    assert "outcome_digest" not in prompt
    assert "episode_digest" not in prompt


def test_q0_decision_table_requires_normal_instruments_before_h3() -> None:
    manifest = {
        "arm_order": [
            "parent_equivalent_null",
            "known_good_reference",
            "frontier_candidate",
        ]
    }
    prior = {"schema": "test-prior"}
    baseline = {
        "exit_status": "SUCCESS",
        "metrics": {"ndcg@10": 0.205},
        "wall_time_ms": 300_000,
    }
    records = {
        "parent_equivalent_null": {
            "qualification_status": "PASS",
            "resolution": "SEARCH_READY",
            "fresh_provider_implementation": True,
            "manual_candidate_patches": 0,
            "static_source_reuse": False,
            "training_run": {
                "exit_status": "SUCCESS",
                "metrics": {"ndcg@10": 0.205 + NULL_NDCG_ABS_TOLERANCE / 2},
                "wall_time_ms": 305_000,
            },
        },
        "known_good_reference": {
            "qualification_status": "PASS",
            "resolution": "SEARCH_READY",
            "fresh_provider_implementation": True,
            "manual_candidate_patches": 0,
            "static_source_reuse": False,
            "training_run": {
                "exit_status": "SUCCESS",
                "metrics": {"ndcg@10": 0.15},
                "wall_time_ms": 500_000,
            },
        },
        "frontier_candidate": {
            "qualification_status": "PASS",
            "resolution": "INNOVATION_REQUIRED",
            "fresh_provider_implementation": True,
            "manual_candidate_patches": 0,
            "static_source_reuse": False,
            "training_run": {
                "exit_status": "SUCCESS",
                "metrics": {"ndcg@10": 0.19},
                "wall_time_ms": 450_000,
            },
        },
    }

    result = _evaluate_q0(manifest, prior, records, baseline)

    assert result["null_normal"] is True
    assert result["known_good_normal"] is True
    assert result["frontier_negative"] is True
    assert result["decision"] == "H3_IDEA_QUALITY_AND_MECHANISM_JUDGMENT"
    assert result["all_core_gates_pass"] is True
