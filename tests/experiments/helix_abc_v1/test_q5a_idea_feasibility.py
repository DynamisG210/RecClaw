from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.q5a_idea_feasibility import (  # noqa: E402
    Q5A_POLICIES,
    Q5A_STAGES,
    Q5AIdeaFeasibilityError,
    build_q5a_pool_manifest,
    build_q5a_prefreeze_manifest,
    build_q5a_realization_union,
    build_q5a_selection_manifest,
    build_q5a_shared_exploration,
    build_q5a_stage_denominator,
)


def _row(index: int, *, prefix: str = "") -> dict[str, object]:
    digest = f"{index:064x}"
    return {
        "slot": f"slot-{index:02d}",
        "producer_role": "mechanism_composer",
        "research_spec": {"protocol_digest": "a" * 64},
        "preoutcome_score": {"spec_digest": prefix + digest},
        "resolution": {"resolution": "INNOVATION_REQUIRED"},
        "stage": "OPENSPEC_FROZEN",
        "manual_candidate_patches": 0,
        "outcome_fields_consumed": [],
    }


def _pool(index: int) -> dict[str, object]:
    rows = [_row(index * 100 + offset) for offset in range(1, 9)]
    return {"candidate_pools": {"shared": rows}}


def _prefreeze() -> dict[str, object]:
    return build_q5a_prefreeze_manifest(
        campaign_id="q5a-test",
        foundation_commit="8" * 40,
        foundation_package_digest="9" * 64,
        source_tree_digest="a" * 64,
        common_execution={"retries": 0, "held_out_reads": 0},
        pool_seeds=(101, 102, 103),
    )


def test_prefreeze_freezes_three_pools_eight_slots_and_no_retry_boundary() -> None:
    manifest = _prefreeze()

    assert manifest["pool_count"] == 3
    assert manifest["pool_size"] == 8
    assert manifest["policy_selection_budget"] == 2
    assert manifest["shared_random_exploration_per_pool"] == 1
    assert manifest["stage_order"] == list(Q5A_STAGES)
    assert manifest["held_out_reads"] == 0
    assert manifest["retries"] == 0


def test_pool_manifest_rejects_incomplete_provider_denominator() -> None:
    with pytest.raises(Q5AIdeaFeasibilityError, match="exactly eight"):
        build_q5a_pool_manifest(
            pool_index=1,
            pool_seed=101,
            pool={"candidate_pools": {"shared": [_row(1)]}},
            provider_usage={"physical_calls": 1, "retries": 0},
        )

    frozen = build_q5a_pool_manifest(
        pool_index=1,
        pool_seed=101,
        pool=_pool(1),
        provider_usage={"physical_calls": 8, "retries": 0},
    )
    assert frozen["provider_denominator"] == 8
    assert frozen["origin_blind"] is True


def test_selection_requires_two_unique_specs_and_records_probability_tie() -> None:
    pool = _pool(1)
    pool_manifest = build_q5a_pool_manifest(
        pool_index=1,
        pool_seed=101,
        pool=pool,
        provider_usage={"physical_calls": 8, "retries": 0},
    )
    candidate_ids = [
        row["preoutcome_score"]["spec_digest"] for row in pool["candidate_pools"]["shared"]
    ]
    selection = build_q5a_selection_manifest(
        pool=pool_manifest,
        pool_digest=pool_manifest["pool_digest"],
        policy_name="STATIC",
        selected_candidate_ids=candidate_ids[:2],
        candidate_probabilities={
            candidate_id: 1.0 if candidate_id in candidate_ids[:2] else 0.0
            for candidate_id in candidate_ids
        },
        tie_set=candidate_ids[:2],
        selected_by={candidate_ids[0]: "UNIFORM_TIE", candidate_ids[1]: "UNIFORM_TIE"},
        selection_seed=101,
    )

    assert selection["selection_budget"] == 2
    assert selection["selection_probability_sum"] == 2.0
    assert selection["top_tie_set"] == candidate_ids[:2]

    with pytest.raises(Q5AIdeaFeasibilityError, match="two unique"):
        build_q5a_selection_manifest(
            pool=pool_manifest,
            pool_digest=pool_manifest["pool_digest"],
            policy_name="STATIC",
            selected_candidate_ids=[candidate_ids[0], candidate_ids[0]],
            candidate_probabilities={candidate_id: 0.0 for candidate_id in candidate_ids},
            tie_set=(),
            selected_by={},
        )


def test_union_deduplicates_realization_and_keeps_policy_exploration_attribution() -> None:
    prefreeze = _prefreeze()
    pools = [
        build_q5a_pool_manifest(
            pool_index=index,
            pool_seed=100 + index,
            pool=_pool(index),
            provider_usage={"physical_calls": 8, "retries": 0},
        )
        for index in range(1, 4)
    ]
    selections = {}
    for policy_index, policy in enumerate(Q5A_POLICIES):
        selections[policy] = []
        for pool in pools:
            ids = [
                row["preoutcome_score"]["spec_digest"]
                for row in pool["candidate_pools"]["shared"]
            ]
            selections[policy].append(
                build_q5a_selection_manifest(
                    pool=pool,
                    pool_digest=pool["pool_digest"],
                    policy_name=policy,
                    selected_candidate_ids=ids[policy_index : policy_index + 2],
                    candidate_probabilities={
                        candidate_id: 1.0 if candidate_id in ids[policy_index : policy_index + 2] else 0.0
                        for candidate_id in ids
                    },
                    tie_set=ids[policy_index : policy_index + 2],
                    selected_by={
                        ids[policy_index]: "LEARNED_SCORE",
                        ids[policy_index + 1]: "LEARNED_SCORE",
                    },
                )
            )
    explorations = [
        build_q5a_shared_exploration(
            pool=pool,
            pool_digest=pool["pool_digest"],
            random_seed=700 + index,
        )
        for index, pool in enumerate(pools, 1)
    ]
    union = build_q5a_realization_union(
        prefreeze=prefreeze,
        pools=pools,
        selections=selections,
        explorations=explorations,
    )

    expected_union = {
        candidate_id
        for policy_rows in selections.values()
        for selection in policy_rows
        for candidate_id in selection["selected_candidate_ids"]
    }
    expected_union.update(
        exploration["selected_candidate_id"] for exploration in explorations
    )
    assert union["selected_unique_spec_count"] == len(expected_union)
    assert union["implementations_per_unique_openspec"] == 1
    assert set(union["shared_exploration_by_pool"]) == {
        pool["pool_digest"] for pool in pools
    }


def test_stage_denominator_outputs_all_five_stages_and_preserves_missingness() -> None:
    denominator = build_q5a_stage_denominator(
        realization_rows=[
            {"stage": "MATERIALIZE", "status": "SUCCESS", "cost_ms": 10},
            {"stage": "CONSTRUCT", "status": "FAILURE", "cost_ms": 20},
        ]
    )

    assert denominator["stage_order"] == list(Q5A_STAGES)
    assert len(denominator["stage_stats"]) == 5
    assert denominator["stage_stats"][0]["empirical_completion_rate"] == 1.0
    assert denominator["stage_stats"][1]["empirical_completion_rate"] == 0.0
    for row in denominator["stage_stats"][2:]:
        assert row["observed_count"] == 0
        assert row["empirical_completion_rate"] is None
        assert row["missingness_count"] == 2
    assert denominator["missingness_is_not_zero"] is True


def test_stage_denominator_can_bind_flattened_rows_to_unique_realization_denominator() -> None:
    rows = [
        {"research_spec_digest": "a" * 64, "stage": "MATERIALIZE", "status": "SUCCESS", "cost_ms": 10},
        {"research_spec_digest": "b" * 64, "stage": "MATERIALIZE", "status": "FAILURE", "cost_ms": 11},
        {"research_spec_digest": "a" * 64, "stage": "CONSTRUCT", "status": "SUCCESS", "cost_ms": 12},
    ]
    denominator = build_q5a_stage_denominator(
        realization_rows=rows,
        realization_denominator=2,
    )

    assert all(row["denominator"] == 2 for row in denominator["stage_stats"])
    assert denominator["stage_stats"][0]["observed_count"] == 2
    assert denominator["stage_stats"][0]["success_count"] == 1
    assert denominator["stage_stats"][0]["missingness_count"] == 0
    assert denominator["stage_stats"][1]["observed_count"] == 1
    assert denominator["stage_stats"][1]["missingness_count"] == 1
    assert denominator["stage_stats"][2]["missingness_count"] == 2
