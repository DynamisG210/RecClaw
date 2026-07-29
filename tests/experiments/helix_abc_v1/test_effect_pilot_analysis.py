from __future__ import annotations

import unittest

from recclaw_core.experiments.helix_abc_v1.effect_pilot_analysis import (
    EffectPilotAnalysisError,
    arm_trajectory_metrics,
    effect_pilot_verdict,
    evidence_guard_visibility,
    research_capability_visibility,
)
from recclaw_core.experiments.helix_abc_v1.pilot_analysis import (
    analysis_observation_key,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    DISCOVERY_PRODUCERS,
)


def _rows(values: list[float], *, semantics: list[str]) -> list[dict]:
    return [
        {
            "analysis_row_id": f"row-{index}",
            "billed_tokens": 10,
            "candidate_id": f"candidate-{index}",
            "gpu_cost_microunits": 20,
            "mechanism_semantics_digest": semantic,
            "ndcg": value,
            "observation_seed": "2026",
            "opaque_instance_id": "opaque",
            "ordinary_execution_count": 1,
            "round_index": index,
            "run_status": "SUCCESS",
        }
        for index, (value, semantic) in enumerate(
            zip(values, semantics, strict=True),
            1,
        )
    ]


def _eligibility(rows: list[dict]) -> dict[str, str]:
    return {
        analysis_observation_key(row): "SEARCH_ELIGIBLE"
        for row in rows
    }


CRITERIA = {
    "research_capability": {
        "best_noninferiority": -0.0005,
        "best_superiority": 0.002,
        "maximum_blocker_rate": 0.25,
        "maximum_duplicate_rate": 0.5,
        "minimum_distinct_executed_semantics": 4,
        "minimum_efficiency_ratio": 1.05,
        "minimum_selected_per_producer_role": 1,
        "required_producer_roles": list(DISCOVERY_PRODUCERS),
        "round_auc_noninferiority": -0.0005,
        "round_auc_superiority": 0.0015,
        "useful_signal_rate_gain": 0.05,
    },
    "evidence_guard": {
        "best_non_suppression": -0.003,
        "minimum_completed_validations": 2,
        "minimum_nontrivial_interventions": 2,
        "minimum_successful_challenge_cases": 8,
        "round_auc_non_suppression": -0.003,
    },
}


class EffectPilotAnalysisTests(unittest.TestCase):
    def test_arm_metrics_bind_round_cost_and_search_eligibility(self) -> None:
        rows = _rows(
            [0.20, 0.21, 0.205, 0.22],
            semantics=["a", "b", "b", "c"],
        )
        metrics = arm_trajectory_metrics(
            rows,
            eligibility_by_observation=_eligibility(rows),
            rounds_per_arm=4,
            token_budget_cap=100,
            gpu_cost_budget_cap=200,
            useful_signal_delta=0.0005,
            best_tolerance=0.0005,
        )
        self.assertEqual(metrics["best_search_eligible_ndcg_at_10"], 0.22)
        self.assertEqual(metrics["useful_signal_count"], 3)
        self.assertEqual(metrics["distinct_executed_semantics"], 3)
        self.assertEqual(metrics["duplicate_rate"], 0.25)
        self.assertEqual(metrics["budget_to_best"]["round"], 4)

    def test_missing_eligibility_and_rounds_are_rejected(self) -> None:
        rows = _rows([0.2], semantics=["a"])
        with self.assertRaises(EffectPilotAnalysisError):
            arm_trajectory_metrics(
                rows,
                eligibility_by_observation={},
                rounds_per_arm=1,
                token_budget_cap=10,
                gpu_cost_budget_cap=20,
                useful_signal_delta=0.0005,
                best_tolerance=0.0005,
            )
        with self.assertRaises(EffectPilotAnalysisError):
            arm_trajectory_metrics(
                rows,
                eligibility_by_observation=_eligibility(rows),
                rounds_per_arm=2,
                token_budget_cap=20,
                gpu_cost_budget_cap=40,
                useful_signal_delta=0.0005,
                best_tolerance=0.0005,
            )

    def test_research_visibility_requires_effect_branch_and_coverage(self) -> None:
        arm_a = {
            "best_search_eligible_ndcg_at_10": 0.21,
            "blocker_rate": 0.1,
            "distinct_executed_semantics": 4,
            "duplicate_rate": 0.2,
            "gpu_cost_auc": 0.15,
            "round_auc": 0.20,
            "token_auc": 0.15,
            "useful_signal_rate_per_execution": 0.10,
        }
        arm_b = {
            **arm_a,
            "best_search_eligible_ndcg_at_10": 0.212,
            "distinct_executed_semantics": 8,
            "round_auc": 0.202,
            "useful_signal_rate_per_execution": 0.16,
        }
        result = research_capability_visibility(
            arm_a=arm_a,
            arm_b=arm_b,
            criteria=CRITERIA,
            producer_role_counts={role: 2 for role in DISCOVERY_PRODUCERS},
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(result["branch_pass"]["efficacy"])

    def test_guard_visibility_requires_activity_safety_and_non_suppression(
        self,
    ) -> None:
        arm_b = {
            "best_search_eligible_ndcg_at_10": 0.21,
            "round_auc": 0.20,
        }
        arm_c = {
            "best_search_eligible_ndcg_at_10": 0.209,
            "round_auc": 0.198,
        }
        guard = {
            "completed_validation_count": 2,
            "cross_arm_contamination_count": 0,
            "false_allow_count": 0,
            "false_block_count": 0,
            "guard_private_input_leak_count": 0,
            "legal_candidate_permanent_suppression_count": 0,
            "nontrivial_intervention_count": 2,
            "preliminary_marked_confirmed_count": 0,
            "search_memory_pollution_count": 0,
            "seed_binding_mismatch_count": 0,
            "successful_challenge_case_count": 8,
        }
        result = evidence_guard_visibility(
            arm_b=arm_b,
            arm_c=arm_c,
            criteria=CRITERIA,
            guard_metrics=guard,
        )
        self.assertEqual(result["verdict"], "PASS")
        guard["seed_binding_mismatch_count"] = 1
        result = evidence_guard_visibility(
            arm_b=arm_b,
            arm_c=arm_c,
            criteria=CRITERIA,
            guard_metrics=guard,
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_final_verdict_requires_both_lines_and_chain(self) -> None:
        passed = {"verdict": "PASS"}
        result = effect_pilot_verdict(
            chain_checks={"closed": True, "isolated": True},
            research_visibility=passed,
            guard_visibility=passed,
            criteria_digest="f" * 64,
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(len(result["analysis_digest"]), 64)


if __name__ == "__main__":
    unittest.main()
