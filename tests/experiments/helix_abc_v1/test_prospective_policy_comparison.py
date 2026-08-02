from __future__ import annotations

import json
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.prospective_policy_comparison import (
    classify_mechanism_probe,
    compute_four_metrics,
    select_current_f1,
    select_outcome_aware,
    select_static,
)


DIRECTIONS = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)


def _pool() -> dict[str, object]:
    rows = []
    for index, role in enumerate(DIRECTIONS, 1):
        spec = {
            "schema": "recclaw.research-line.vnext.open-research-spec.v1",
            "protocol_ref": "protocol:test",
            "protocol_digest": "1" * 64,
            "current_profile_ref": "profile:test",
            "current_profile_digest": "2" * 64,
            "falsifier": f"falsifier-{index}",
            "matched_control_requirement": "same parent",
            "expected_evidence": ["distinguishing evidence"],
            "competing_explanation": "capacity",
            "mechanism_off_definition": "parent",
        }
        candidate_id = sha256_digest(spec)
        rows.append(
            {
                "slot": f"slot-{index:02d}",
                "producer_role": role,
                "research_spec": spec,
                "resolution": {
                    "resolution": "INNOVATION_REQUIRED",
                    "resolved_current_capability_ref": None,
                },
                "resolution_facts": {
                    "high_change_dimensions": ["MODEL_STRUCTURE"],
                    "required_budget": {"implementation_slots": 1},
                },
                "preoutcome_score": {
                    "spec_digest": candidate_id,
                    "features": {
                        "scientific_testability": 1,
                        "discriminative_value": 1,
                    },
                },
                "stage": "OPENSPEC_FROZEN",
            }
        )
    return {
        "schema": "recclaw.research-line.q4-prospective-shared-pool.v1",
        "candidate_pools": {"shared": rows},
        "selection_rule": "NONE_POOL_ONLY_POLICIES_SELECT_AFTER_BYTE_FREEZE",
        "implementation_or_qualification_outcomes_present_when_written": 0,
        "outcome_fields_consumed": [],
        "held_out_reads": 0,
    }


def _outcome_policy() -> tuple[dict[str, object], dict[str, object]]:
    root = (
        ROOT
        / "results/research_line/q4_multiround_soak_20260803_01/round_03/policy_update"
    )
    return (
        json.loads((root / "versioned_policy.json").read_text()),
        json.loads((root / "active_policy.json").read_text()),
    )


def test_three_consumers_use_one_pool_without_outcome_features() -> None:
    pool = _pool()
    pool_digest = sha256_digest(pool)
    f1_policy = {
        "policy_ref": "policy:research-open-meta-vnext:f1:v1",
        "policy_version": "research-open-meta-v1.0.0",
        "policy_digest": "3" * 64,
        "direction_order": [
            "falsification_designer",
            "lineage_refiner",
            "mechanism_composer",
            "frontier_architect",
        ],
    }
    outcome_policy, outcome_activation = _outcome_policy()

    static = select_static(pool, pool_digest=pool_digest)
    f1 = select_current_f1(pool, pool_digest=pool_digest, policy=f1_policy)
    outcome = select_outcome_aware(
        pool,
        pool_digest=pool_digest,
        policy=outcome_policy,
        activation=outcome_activation,
        random_seed=56331,
    )

    assert static["candidate_count"] == f1["candidate_count"] == 4
    assert static["pool_digest"] == f1["pool_digest"] == pool_digest
    assert f1["selected_candidate_id"] == next(
        row["preoutcome_score"]["spec_digest"]
        for row in pool["candidate_pools"]["shared"]
        if row["producer_role"] == "falsification_designer"
    )
    assert outcome["exploration_probability"] == 0.15
    assert outcome["random_seed"] == 56331
    for selection in (static, f1, outcome):
        assert selection["held_out_reads"] == 0
        assert abs(
            sum(
                float(row["selection_probability"])
                for row in selection["candidates"]
            )
            - 1.0
        ) < 1e-9
        assert sum(row["selected"] is True for row in selection["candidates"]) == 1


def test_resource_failure_has_no_mechanism_authority() -> None:
    result = classify_mechanism_probe(
        qualification={
            "status": "QUALIFICATION_PASS",
            "behavioral_evidence": {
                "probe_status": "PASS_STRUCTURAL_BEHAVIOR_ACTIVE",
                "behavioral_loss_max_abs_delta": 1.0,
                "mechanism_off_execution": "PASS",
            },
        },
        admission={"status": "RESOURCE_DEFERRED", "resource_probe": None},
    )
    assert result["mechanism_state"] == "NOT_ASSESSED"
    assert result["evidence"] == {}


def test_real_active_probe_requires_parent_equivalent_mechanism_off() -> None:
    qualification = {
        "status": "QUALIFICATION_PASS",
        "behavioral_evidence": {
            "probe_status": "PASS_STRUCTURAL_BEHAVIOR_ACTIVE",
            "behavioral_loss_max_abs_delta": 0.1,
            "behavioral_score_max_abs_delta": 0.2,
            "overridden_behavioral_methods": ["calculate_loss", "full_sort_predict"],
            "mechanism_off_execution": "PASS",
            "mechanism_off_full_sort_max_abs_delta": 0.0,
            "mechanism_off_loss_abs_delta": 0.0,
            "mechanism_off_predict_max_abs_delta": 0.0,
        },
    }
    admission = {
        "status": "RESOURCE_ADMITTED",
        "resource_probe": {"exit_status": "SUCCESS", "wall_time_ms": 50},
    }
    assert classify_mechanism_probe(
        qualification=qualification, admission=admission
    )["mechanism_state"] == "ACTIVE_SUPPORTED"
    qualification["behavioral_evidence"]["mechanism_off_execution"] = (
        "MATCHED_PARENT_PACKAGE_REQUIRED"
    )
    assert classify_mechanism_probe(
        qualification=qualification, admission=admission
    )["mechanism_state"] == "NON_IDENTIFIABLE"


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_resource_seed_matches_byte_bound_q0r2_contract() -> None:
    runner = _load_script(
        "prospective_policy_runner", "scripts/run_prospective_policy_comparison.py"
    )
    prefix = json.loads(
        (
            ROOT
            / "results/research_line/q0r_type_preserving_resource_scheduling_20260802_01"
            / "FIXED_BATCH_PREFIX_CONTRACT.json"
        ).read_text()
    )
    assert runner.COMMON_EXECUTION["resource_probe_seed"] == 54102
    assert prefix["seed"] == 54102


def test_v2_upstream_reader_rejects_byte_drift(tmp_path: Path) -> None:
    stage = _load_script("multiround_stage", "scripts/run_multiround_soak_stage.py")
    upstream = tmp_path / "v1"
    current = tmp_path / "v2"
    upstream.mkdir()
    current.mkdir()
    artifact = upstream / "MATERIALIZE_QUALIFIER_RECEIPT.json"
    artifact.write_text('{"status":"QUALIFICATION_PASS"}\n')
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (current / "SEALED_UPSTREAM_BINDING.json").write_text(
        json.dumps(
            {
                "sealed_artifacts": {
                    artifact.name: {"path": str(artifact), "sha256": digest}
                }
            }
        )
    )
    args = Namespace(round_root=current, upstream_root=upstream)
    assert stage._read_upstream(args, artifact.name)["status"] == "QUALIFICATION_PASS"
    artifact.write_text('{"status":"DRIFT"}\n')
    try:
        stage._read_upstream(args, artifact.name)
    except RuntimeError as error:
        assert "SHA drift" in str(error)
    else:
        raise AssertionError("sealed upstream byte drift was accepted")


def test_four_metrics_use_real_episode_metric_shape_and_split_cost_roots(
    tmp_path: Path,
) -> None:
    upstream = tmp_path / "upstream"
    arm = tmp_path / "arm"
    (upstream / "stage_costs").mkdir(parents=True)
    (arm / "stage_costs").mkdir(parents=True)
    for root, costs in (
        (upstream, {"implementer": 11, "materialize-qualifier": 12}),
        (
            arm,
            {
                "resource-admission": 13,
                "mechanism-probe": 14,
                "matched-execution": 15,
            },
        ),
    ):
        for stage_name, wall_time in costs.items():
            (root / "stage_costs" / f"{stage_name}.json").write_text(
                json.dumps({"wall_time_ms": wall_time})
            )
    (arm / "EPISODE_RECEIPT.json").write_text(
        json.dumps(
            {
                "status": "EPISODE_CREATED",
                "outcome_summary": {
                    "candidate_metrics": {"ndcg@10": 0.2},
                    "baseline_metrics": {"ndcg@10": 0.21},
                },
            }
        )
    )
    (arm / "MECHANISM_PROBE_RECEIPT.json").write_text(
        json.dumps({"mechanism_state": "NON_IDENTIFIABLE"})
    )
    metrics = compute_four_metrics(
        arm_root=arm,
        upstream_arm_root=upstream,
        full_pool_count=4,
        shared_pool_provider_wall_time_ms=10,
    )
    assert metrics["A_best_parent_relative_development_effect_ndcg_at_10"] == -0.01
    assert metrics["B_mechanism_identifiable_episode_count"] == 0
    assert metrics["C_cost_per_informative_episode_wall_time_ms"] == 75
    assert metrics["D_full_episode_completion_rate_main_selected_denominator"] == 1.0
