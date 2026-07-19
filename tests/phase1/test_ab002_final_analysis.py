from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_final_analysis import SEARCH_SEEDS, TRAINING_SEEDS, analyze
from recclaw_phase1.neutral_outcome_auditor import audit_outcome


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTCOME_POLICY = json.loads(
    (PROJECT_ROOT / "configs/phase1/ab002/outcome_audit_policy.json").read_text()
)


def audit(seed: int, value: float) -> dict[str, object]:
    return {
        "eligible_for_performance_analysis": True,
        "observed_seed": seed,
        "metrics": {
            "ndcg@10": value,
            "recall@10": 0.2,
            "mrr@10": 0.3,
            "hit@10": 0.4,
            "precision@10": 0.1,
        },
    }


class AB002FinalAnalysisTests(unittest.TestCase):
    def build_runtime(self, root: Path, *, omit_seed: int | None = None) -> None:
        baseline = {
            "runs": [
                {"training_seed": seed, "neutral_audit": audit(seed, 0.267 + index * 0.0001)}
                for index, seed in enumerate(TRAINING_SEEDS)
            ]
        }
        (root / "three_seed_baseline_report.json").parent.mkdir(parents=True, exist_ok=True)
        (root / "three_seed_baseline_report.json").write_text(json.dumps(baseline))
        inputs = root / "pair_inputs"
        inputs.mkdir()
        for index, search_seed in enumerate(SEARCH_SEEDS):
            if search_seed == omit_seed:
                continue
            control_value = 0.270 + index * 0.0001
            treatment_value = control_value + 0.001
            pair = {
                "search_seed": search_seed,
                "arms": {
                    "control": {
                        "new_candidate_discovered": True,
                        "candidate_id": f"candidate-c-{search_seed}",
                        "confirmation_runs": [
                            {"training_seed": seed, "neutral_audit": audit(seed, control_value)}
                            for seed in TRAINING_SEEDS
                        ],
                        "running_best_valid_curve": [control_value] * 20,
                        "mechanism": {
                            "invalid_feedback_entering_primary_memory_rate": 0.2,
                            "valid_informative_signal_rate": 0.2,
                        },
                    },
                    "treatment": {
                        "new_candidate_discovered": True,
                        "candidate_id": f"candidate-t-{search_seed}",
                        "confirmation_runs": [
                            {"training_seed": seed, "neutral_audit": audit(seed, treatment_value)}
                            for seed in TRAINING_SEEDS
                        ],
                        "running_best_valid_curve": [treatment_value] * 20,
                        "mechanism": {
                            "invalid_feedback_entering_primary_memory_rate": 0.05,
                            "valid_informative_signal_rate": 0.4,
                            "guard_reviewed_valid_action_count": 10,
                            "valid_action_false_block_count": 0,
                            "critical_valid_action_false_block_count": 0,
                        },
                    },
                },
            }
            (inputs / f"search_seed_{search_seed}.json").write_text(json.dumps(pair))
        (root / "simple_rule_comparison.json").write_text(json.dumps({"same_benefit": False}))

    def test_six_pair_analysis_reports_exact_statistics_and_strong_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runtime"
            output = Path(temporary) / "analysis"
            self.build_runtime(root)
            result = analyze(root, output)
            decision = audit_outcome(result, OUTCOME_POLICY)
            self.assertEqual(6, result["complete_pair_count"])
            self.assertAlmostEqual(0.001, result["aggregate"]["paired_mean_primary_difference"])
            self.assertEqual(64, result["aggregate"]["exact_paired_sign_flip"]["enumerated_assignments"])
            self.assertEqual("STRONG_POSITIVE", decision["experiment_outcome"])
            self.assertTrue((output / "paired_analysis.json").is_file())
            self.assertTrue((output / "running_best_curves.svg").is_file())

    def test_missing_pair_is_reported_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runtime"
            output = Path(temporary) / "analysis"
            self.build_runtime(root, omit_seed=47)
            result = analyze(root, output)
            decision = audit_outcome(result, OUTCOME_POLICY)
            self.assertEqual("IN_PROGRESS", result["status"])
            self.assertEqual(5, result["complete_pair_count"])
            self.assertEqual("INCONCLUSIVE", decision["experiment_outcome"])
            self.assertIn("INSUFFICIENT_COMPLETE_PAIRS", decision["reason_codes"])


if __name__ == "__main__":
    unittest.main()
