from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.fresh_f1 import (  # noqa: E402
    R2_EXTERNAL_ROOT,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (  # noqa: E402
    R1_EXTERNAL_ROOT,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_f1 import (  # noqa: E402
    F1_POLICY_VERSION,
    build_f1_replay_dataset,
    evaluate_development_promotion,
    fit_open_meta_policy,
    rank_search_ready_records,
    shadow_evaluate_open_meta,
)


def _dataset() -> dict[str, object]:
    directions = (
        "mechanism_composer",
        "lineage_refiner",
        "falsification_designer",
        "frontier_architect",
    )
    rows = []
    episode_budget = 8
    for index in range(20):
        direction = directions[index % 4]
        split = "SEARCH_TRAIN" if index < 8 else "DEVELOPMENT_VALIDATION"
        qualification_observed = index < 16 or index == 17
        qualification_pass = (
            direction == "falsification_designer"
            or (direction != "frontier_architect" and index % 3 != 0)
        ) if qualification_observed else None
        episode = episode_budget > 0 and qualification_pass is True
        if episode:
            episode_budget -= 1
        rows.append(
            {
                "audit_ref": f"row-{index}",
                "campaign_family": "R1" if index < 16 else "R2",
                "replay_split": split,
                "direction": direction,
                "high_change_dimensions": ("MODEL_STRUCTURE",),
                "required_implementation_tokens": 20_000,
                "required_qualification_gpu_minutes": 10,
                "required_qualification_wall_minutes": 30,
                "resolved_as_search_ready": index >= 16,
                "qualification_observed": qualification_observed,
                "qualification_pass": qualification_pass,
                "qualification_failure_class": (
                    "NONE" if qualification_pass else "INTERFACE"
                ) if qualification_observed else None,
                "qualification_missing_reason": (
                    None
                    if qualification_observed
                    else "NOT_SELECTED_BY_PREFROZEN_SLOT_ORDER"
                ),
                "experiment_observed": qualification_pass is True,
                "experiment_closed": episode,
                "runtime_failure": qualification_pass is True and not episode,
                "proposal_billed_tokens": 4_000,
                "implementation_billed_tokens": 9_000 if qualification_observed else 0,
                "candidate_runtime_wall_ms": 500_000 if qualification_pass else None,
                "ndcg_delta_audit_only": -0.02 if episode else None,
                "capability_ref": f"capability:{index % 5}" if qualification_pass else None,
                "selected_for_experiment": qualification_pass is True,
                "episode_observed": episode,
                "evidence_class": "INCONCLUSIVE_EXPERIMENT" if episode else None,
                "failure_class": "INCONCLUSIVE" if episode else None,
                "mechanism_interpretation": "NOT_ADJUDICATED" if episode else None,
                "mechanism_negative_evidence": False if episode else None,
            }
        )
    # Keep the contract denominator explicit even though this synthetic owner
    # test is not used as campaign evidence.
    payload = {
        "schema": "recclaw.research-line.vnext.open-meta.f1-replay.v1",
        "dataset_version": "1.0.0",
        "rows": rows,
        "row_count": 20,
        "episode_count": 8,
        "held_out_reads": 0,
        "scientific_semantics": {
            "evidence_class": "INCONCLUSIVE_EXPERIMENT",
            "mechanism_interpretation": "NOT_ADJUDICATED",
            "mechanism_negative_evidence": False,
            "outcome_usage": "AUDIT_AND_UNCERTAINTY_ONLY_NOT_MECHANISM_REWARD",
        },
    }
    payload["dataset_digest"] = sha256_digest(payload)
    return payload


def test_learner_shadow_and_engineering_promotion_do_not_claim_science() -> None:
    dataset = _dataset()
    shadow_policy = fit_open_meta_policy(
        dataset,
        splits=("SEARCH_TRAIN",),
        policy_version=F1_POLICY_VERSION + "-shadow",
    )
    shadow = shadow_evaluate_open_meta(dataset, shadow_policy)
    final_policy = fit_open_meta_policy(
        dataset,
        splits=("SEARCH_TRAIN", "DEVELOPMENT_VALIDATION"),
        policy_version=F1_POLICY_VERSION,
    )
    promotion = evaluate_development_promotion(dataset, shadow, final_policy)

    assert final_policy["training_row_count"] == 20
    assert final_policy["decision_rule"]["scientific_metric_reward"] is False
    assert shadow["uncertainty"]["policy_superiority_established"] is False
    assert shadow["uncertainty"]["scientific_effect_established"] is False
    assert promotion["status"] == "DEVELOPMENT_ONLY_PROMOTION_PASS"
    assert promotion["policy_superiority_claim"] is False
    assert promotion["scientific_effect_claim"] is False


def test_experiment_policy_prefers_missing_information_not_candidate_id() -> None:
    dataset = _dataset()
    policy = fit_open_meta_policy(
        dataset,
        splits=("SEARCH_TRAIN", "DEVELOPMENT_VALIDATION"),
        policy_version=F1_POLICY_VERSION,
    )
    records = (
        {
            "logical_slot_id": "slot-01",
            "resolution": "SEARCH_READY",
            "resolved_capability_ref": "capability:0",
        },
        {
            "logical_slot_id": "slot-02",
            "resolution": "SEARCH_READY",
            "resolved_capability_ref": "capability:unseen",
        },
    )
    forward = rank_search_ready_records(policy, records)
    reverse = rank_search_ready_records(policy, tuple(reversed(records)))

    assert forward == reverse
    assert forward[0]["resolved_capability_ref"] == "capability:unseen"


def test_real_r1_r2_replay_ingests_full_denominator_when_available() -> None:
    r1 = R1_EXTERNAL_ROOT
    r2 = R2_EXTERNAL_ROOT
    if not r1.is_dir() or not r2.is_dir():
        pytest.skip("accepted external R1/R2 evidence is not installed")
    dataset = build_f1_replay_dataset(r1_root=r1, r2_root=r2)

    assert dataset["row_count"] == 20
    assert dataset["episode_count"] == 8
    assert dataset["held_out_reads"] == 0
    assert dataset["qualification_missingness"] == {
        "NOT_SELECTED_BY_PREFROZEN_SLOT_ORDER": 3,
        "R1_QUALIFICATION_FAILURE": 5,
    }
    assert all(
        row["mechanism_negative_evidence"] is False
        for row in dataset["rows"]
        if row["episode_observed"]
    )
