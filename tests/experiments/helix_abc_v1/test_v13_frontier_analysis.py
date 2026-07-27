from __future__ import annotations

import unittest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.pilot_analysis import (
    FrontierProjectionV13,
    analysis_observation_key,
    descriptive_effect_summary,
    four_axis_frontiers,
)


def _row(
    *,
    round_index: int,
    ndcg: float,
    seed: str,
    confirmation_status: str | None = None,
    evaluator_digest: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "billed_tokens": 10,
        "candidate_id": f"candidate-{round_index}",
        "gpu_cost_microunits": 2,
        "ndcg": ndcg,
        "observation_seed": seed,
        "opaque_instance_id": "instance-c",
        "ordinary_execution_count": 1,
        "raw_result_digest": sha256_digest(
            {"round_index": round_index, "seed": seed}
        ),
        "round_index": round_index,
    }
    if confirmation_status is not None:
        row["post_selection_status"] = confirmation_status
    if evaluator_digest is not None:
        row["post_selection_evaluator_digest"] = evaluator_digest
    return row


class V13FrontierAnalysisTest(unittest.TestCase):
    def test_observed_search_eligible_and_confirmed_are_distinct(self) -> None:
        evaluator = sha256_digest({"post_selection": "frozen-v1"})
        rows = [
            _row(round_index=1, ndcg=0.40, seed="2026"),
            _row(round_index=2, ndcg=0.60, seed="2027"),
            _row(
                round_index=3,
                ndcg=0.50,
                seed="held-out-1",
                confirmation_status="CONFIRMED",
                evaluator_digest=evaluator,
            ),
        ]
        eligibility = {
            analysis_observation_key(rows[0]): (
                "SEARCH_ELIGIBLE_PRELIMINARY"
            ),
            analysis_observation_key(rows[1]): "EXCLUDED",
            analysis_observation_key(rows[2]): "SEARCH_ELIGIBLE",
        }

        observed = four_axis_frontiers(
            rows,
            frontier_projection=FrontierProjectionV13.OBSERVED,
        )["instance-c"]
        search_eligible = four_axis_frontiers(
            rows,
            frontier_projection=FrontierProjectionV13.SEARCH_ELIGIBLE,
            eligibility_by_observation=eligibility,
        )["instance-c"]
        confirmed = four_axis_frontiers(
            rows,
            frontier_projection=FrontierProjectionV13.CONFIRMED,
            eligibility_by_observation=eligibility,
            post_selection_evaluator_digest=evaluator,
        )["instance-c"]

        self.assertEqual(observed[-1]["frontier_ndcg"], 0.60)
        self.assertEqual(search_eligible[0]["frontier_ndcg"], 0.40)
        self.assertEqual(search_eligible[1]["frontier_ndcg"], 0.40)
        self.assertEqual(search_eligible[-1]["frontier_ndcg"], 0.50)
        self.assertIsNone(confirmed[0]["frontier_ndcg"])
        self.assertIsNone(confirmed[1]["frontier_ndcg"])
        self.assertEqual(confirmed[-1]["frontier_ndcg"], 0.50)

    def test_one_seed_never_becomes_confirmed_without_frozen_evaluator(self) -> None:
        row = _row(round_index=1, ndcg=0.40, seed="2026")
        eligibility = {
            analysis_observation_key(row): (
                "SEARCH_ELIGIBLE_PRELIMINARY"
            )
        }
        with self.assertRaisesRegex(
            ValueError,
            "frozen post-selection evaluator",
        ):
            four_axis_frontiers(
                [row],
                frontier_projection=FrontierProjectionV13.CONFIRMED,
                eligibility_by_observation=eligibility,
            )
        confirmed = four_axis_frontiers(
            [row],
            frontier_projection=FrontierProjectionV13.CONFIRMED,
            eligibility_by_observation=eligibility,
            post_selection_evaluator_digest=sha256_digest(
                {"post_selection": "frozen-v1"}
            ),
        )
        self.assertIsNone(confirmed["instance-c"][0]["frontier_ndcg"])

    def test_guard_ineligible_result_cannot_enter_search_frontier(self) -> None:
        row = _row(round_index=1, ndcg=0.90, seed="2026")
        key = analysis_observation_key(row)
        frontier = four_axis_frontiers(
            [row],
            frontier_projection=FrontierProjectionV13.SEARCH_ELIGIBLE,
            eligibility_by_observation={key: "EXCLUDED"},
        )
        self.assertIsNone(frontier["instance-c"][0]["frontier_ndcg"])

    def test_pilot_summary_does_not_compute_effect_or_confirmation(self) -> None:
        summary = descriptive_effect_summary(
            [_row(round_index=1, ndcg=0.40, seed="2026")]
        )
        self.assertEqual(summary["confirmed_frontier"], "NOT_COMPUTED")
        self.assertEqual(summary["treatment_effect"], "NOT_AUTHORIZED")
        self.assertFalse(summary["formal_inference"])


if __name__ == "__main__":
    unittest.main()
