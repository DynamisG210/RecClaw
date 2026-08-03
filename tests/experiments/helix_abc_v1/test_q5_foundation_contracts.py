from __future__ import annotations

import pytest

from recclaw_core.experiments.helix_abc_v1.prospective_policy_comparison import (
    ProspectivePolicyComparisonError,
    bind_observation_to_realization,
    build_shared_package_seed_outcome_ledger,
    build_shared_realization_contract,
    build_shared_realization_pool,
    classify_realization_authority,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_q3 import (
    OpenMetaQ3Error,
    project_stage_conditional_feasibility,
)


def _contract(
    *,
    spec: str = "1" * 64,
    package: str = "2" * 64,
    source: str = "3" * 64,
    equivalence: str = "4" * 64,
    realization_class: str = "SEMANTICS_EQUIVALENT_REALIZATION",
) -> dict[str, object]:
    return build_shared_realization_contract(
        research_spec_ref=f"openspec:{spec}",
        research_spec_digest=spec,
        candidate_package_ref=f"package:{package}",
        candidate_package_digest=package,
        source_tree_ref=f"tree:{source}",
        source_tree_digest=source,
        equivalence_ref=f"equivalence:{equivalence}",
        equivalence_digest=equivalence,
        protocol_ref="protocol:q5",
        protocol_digest="5" * 64,
        realization_class=realization_class,
    )


def test_shared_realization_pool_deduplicates_implementation_and_keeps_attribution() -> None:
    contract = _contract()
    pool = build_shared_realization_pool(
        policy_selections={
            "STATIC": {"selected_candidate_id": "1" * 64},
            "OUTCOME_AWARE": {"selected_candidate_id": "1" * 64},
        },
        realizations=[contract, contract],
        protocol_ref="protocol:q5",
        protocol_digest="5" * 64,
    )

    assert pool["realization_count"] == 1
    assert pool["implementations_per_open_spec"] == 1
    assert pool["policy_attribution"]["STATIC"] == pool["policy_attribution"]["OUTCOME_AWARE"]


def test_shared_realization_pool_rejects_cross_policy_byte_drift() -> None:
    with pytest.raises(ProspectivePolicyComparisonError, match="more than one"):
        build_shared_realization_pool(
            policy_selections={"STATIC": {"selected_candidate_id": "1" * 64}},
            realizations=[
                _contract(),
                _contract(package="6" * 64, source="7" * 64),
            ],
            protocol_ref="protocol:q5",
            protocol_digest="5" * 64,
        )


def test_old_receipt_without_realization_binding_cannot_masquerade() -> None:
    contract = _contract()
    with pytest.raises(ProspectivePolicyComparisonError, match="realization"):
        bind_observation_to_realization(
            {
                "candidate_package_digest": contract["candidate_package_digest"],
                "source_tree_digest": contract["source_tree_digest"],
                "research_spec_digest": contract["research_spec_digest"],
                "protocol_digest": contract["protocol_digest"],
            },
            contract,
            observation_kind="qualification_receipt",
        )


def test_package_seed_outcome_is_shared_and_policy_attribution_is_only_a_reference() -> None:
    contract = _contract()
    pool = build_shared_realization_pool(
        policy_selections={
            "STATIC": {"selected_candidate_id": "1" * 64},
            "CURRENT_F1": {"selected_candidate_id": "1" * 64},
        },
        realizations=[contract],
        protocol_ref="protocol:q5",
        protocol_digest="5" * 64,
    )
    outcome = {
        "candidate_package_digest": contract["candidate_package_digest"],
        "source_tree_digest": contract["source_tree_digest"],
        "research_spec_digest": contract["research_spec_digest"],
        "protocol_digest": contract["protocol_digest"],
        "realization_digest": contract["realization_digest"],
        "seed": 17,
        "status": "OUTCOME_CREATED",
    }

    ledger = build_shared_package_seed_outcome_ledger(
        realization_pool=pool,
        outcomes=[outcome],
    )

    assert ledger["outcome_count"] == 1
    key = ledger["outcomes"][0]["outcome_key"]
    assert ledger["policy_attribution"]["STATIC"] == [key]
    assert ledger["policy_attribution"]["CURRENT_F1"] == [key]
    with pytest.raises(ProspectivePolicyComparisonError, match="more than once"):
        build_shared_package_seed_outcome_ledger(
            realization_pool=pool,
            outcomes=[outcome, outcome],
        )


def _qualification(*, complete: bool = True, include_all_equivalence: bool = True) -> dict[str, object]:
    behavior: dict[str, object] = {
        "probe_status": "PASS_STRUCTURAL_BEHAVIOR_ACTIVE",
        "behavioral_loss_max_abs_delta": 0.2,
        "behavioral_score_max_abs_delta": 0.3,
        "overridden_behavioral_methods": ["calculate_loss", "full_sort_predict"],
        "mechanism_off_execution": "PASS",
        "mechanism_off_parent_equivalence": "PASS",
        "mechanism_off_loss_abs_delta": 0.0,
        "mechanism_off_predict_max_abs_delta": 0.0,
        "mechanism_off_full_sort_max_abs_delta": 0.0,
        "mechanism_off_protocol_equivalence": {
            "dataset": True,
            "evaluator": True,
            "seed": True,
            "optimizer": True,
            "batch": True,
        },
    }
    if include_all_equivalence:
        behavior.update(
            {
                "mechanism_off_gradients_max_abs_delta": 0.0,
                "mechanism_off_checkpoint_load_max_abs_delta": 0.0,
            }
        )
    return {
        "status": "QUALIFICATION_PASS" if complete else "QUALIFICATION_FAILURE",
        "behavioral_evidence": behavior,
    }


def test_effect_only_non_nested_cannot_enter_mechanism_information() -> None:
    result = classify_realization_authority(
        realization={
            "realization_class": "NEW_CANDIDATE",
            "realization_typing": "EFFECT_ONLY_NON_NESTED",
        },
        qualification=_qualification(),
        admission={"status": "RESOURCE_ADMITTED"},
    )

    assert result["mechanism_state"] == "NOT_ASSESSED"
    assert result["mechanism_information_authority"] == "NOT_ASSESSED"
    assert result["mechanism_information_input_allowed"] is False
    assert result["effect_authority"] == "NOT_ASSESSED"
    assert result["effect_input_allowed"] is False
    assert result["effect_eligibility"] == "ELIGIBLE_ONLY_AFTER_FULL_MATCHED_EPISODE"


@pytest.mark.parametrize(
    ("qualification", "admission"),
    [
        (_qualification(complete=False), {"status": "RESOURCE_ADMITTED"}),
        (_qualification(), {"status": "RESOURCE_DEFERRED"}),
        (_qualification(), {"status": "RESOURCE_ADMITTED"}),
    ],
    ids=["qualification-failure", "resource-deferred", "no-matched-episode"],
)
def test_pre_outcome_authority_never_admits_effect_input(
    qualification: dict[str, object], admission: dict[str, str]
) -> None:
    result = classify_realization_authority(
        realization={
            "realization_class": "NEW_CANDIDATE",
            "realization_typing": "EFFECT_ONLY_NON_NESTED",
        },
        qualification=qualification,
        admission=admission,
    )

    assert result["effect_authority"] == "NOT_ASSESSED"
    assert result["effect_input_allowed"] is False
    assert result["effect_eligibility"] == "ELIGIBLE_ONLY_AFTER_FULL_MATCHED_EPISODE"


def test_nested_gate_requires_gradients_and_checkpoint_load_equivalence() -> None:
    result = classify_realization_authority(
        realization={
            "realization_class": "SEMANTICS_EQUIVALENT_REALIZATION",
            "realization_typing": "NESTED_MECHANISM",
        },
        qualification=_qualification(include_all_equivalence=False),
        admission={"status": "RESOURCE_ADMITTED"},
    )

    assert result["mechanism_state"] == "NOT_ASSESSED"
    assert result["no_off_switch_is_not_negative_mechanism_evidence"] is True


def test_nested_parent_equivalent_realization_can_enter_mechanism_information() -> None:
    result = classify_realization_authority(
        realization={
            "realization_class": "SEMANTICS_EQUIVALENT_REALIZATION",
            "realization_typing": "NESTED_MECHANISM",
        },
        qualification=_qualification(),
        admission={"status": "RESOURCE_ADMITTED"},
    )

    assert result["mechanism_state"] == "ACTIVE_SUPPORTED"
    assert result["mechanism_information_authority"] == "MECHANISM_INFORMATION"


def test_stage_conditional_feasibility_uses_existing_lane_and_shrinks_sparse_stage() -> None:
    projection = project_stage_conditional_feasibility(
        [
            {
                "stages": {
                    "MATERIALIZE": {
                        "completion_label": 1,
                        "features_visible_before_stage": ["spec_digest"],
                    },
                    "CONSTRUCT": {
                        "completion_label": 1,
                        "features_visible_before_stage": ["package_digest"],
                    },
                }
            },
            {
                "stages": {
                    "MATERIALIZE": {
                        "completion_label": 1,
                        "features_visible_before_stage": ["spec_digest"],
                    },
                    "CONSTRUCT": {
                        "completion_label": 0,
                        "features_visible_before_stage": ["package_digest"],
                    },
                }
            },
        ]
    )

    assert projection["authority_lane"] == "FEASIBILITY_COMPLETION_HEAD"
    assert projection["factorization"].startswith("P(FULL_EPISODE)=")
    assert [row["stage"] for row in projection["stage_stats"]] == [
        "MATERIALIZE",
        "CONSTRUCT",
        "QUALIFY",
        "RESOURCE_ADMITTED",
        "FULL_EPISODE",
    ]
    assert len(projection["stage_stats"]) == 5
    materialize = projection["stage_stats"][0]
    assert materialize["stage"] == "MATERIALIZE"
    assert materialize["posterior"]["support_weight"] == 0.5
    assert materialize["conditional_probability"] == 0.625
    assert materialize["features_visible_before_stage"] == ["spec_digest"]
    for row in projection["stage_stats"][2:]:
        assert row["input_count"] == 0
        assert row["observed"] is False
        assert row["observation_status"] == "FROZEN_PRIOR_NO_OBSERVATION"
        assert row["posterior"]["support_weight"] == 0.0
        assert row["conditional_probability"] == 0.5
    assert projection["full_episode_probability"] == 0.0390625


def test_stage_conditional_feasibility_rejects_held_out_input() -> None:
    with pytest.raises(OpenMetaQ3Error, match="held-out"):
        project_stage_conditional_feasibility(
            [
                {
                    "held_out_reads": 1,
                    "stages": {
                        "MATERIALIZE": {"completion_label": 1},
                    },
                }
            ]
        )
