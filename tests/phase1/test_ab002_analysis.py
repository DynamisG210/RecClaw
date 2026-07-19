from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_analysis import SEEDS, analyze
from recclaw_phase1.neutral_outcome_auditor import audit_outcome


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_results(path: Path, rows: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "ndcg@10"])
        writer.writeheader()
        for run_id, status, metric in rows:
            writer.writerow({"run_id": run_id, "status": status, "ndcg@10": metric})


class AB002AnalysisTests(unittest.TestCase):
    def test_analysis_uses_only_guard_admitted_treatment_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runtime"
            for seed in SEEDS:
                control = root / f"runs/control/search_seed_{seed}/outputs/pilot/results.csv"
                treatment_dir = root / f"runs/treatment/search_seed_{seed}/outputs/pilot"
                write_results(control, [(f"c{seed}a", "success", 0.280), (f"c{seed}b", "success", 0.281)])
                write_results(
                    treatment_dir / "results.csv",
                    [(f"t{seed}a", "success", 0.290), (f"t{seed}b", "success", 0.310)],
                )
                feedback = [
                    {
                        "phase": "POSTCHECK",
                        "guard_latency_ms": 2.0,
                        "original_result_snapshot": {"run_id": f"t{seed}a"},
                        "original_trial_memory_disposition": "ADMIT_ORIGINAL_TRIAL",
                    },
                    {
                        "phase": "POSTCHECK",
                        "guard_latency_ms": 3.0,
                        "original_result_snapshot": {"run_id": f"t{seed}b"},
                        "original_trial_memory_disposition": "QUARANTINE_ORIGINAL_TRIAL",
                    },
                ]
                (treatment_dir / "evidence_guard_feedback.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in feedback), encoding="utf-8"
                )
            output = Path(temporary) / "analysis"
            result = analyze(root, output, 0.2671)
            self.assertEqual(6, result["complete_pair_count"])
            self.assertAlmostEqual(0.290, result["aggregate"]["treatment_mean_final_best"])
            self.assertAlmostEqual(0.281, result["aggregate"]["control_mean_final_best"])
            self.assertEqual(6, result["aggregate"]["quarantined_original_trial_count"])
            self.assertTrue((output / "ab002_running_best.svg").is_file())

    def test_neutral_auditor_requires_independent_false_block_input(self) -> None:
        policy = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/outcome_audit_policy.json").read_text()
        )
        analysis = {
            "assigned_pair_count": 6,
            "complete_pair_count": 6,
            "aggregate": {
                "paired_mean_primary_difference": 0.001,
                "paired_median_primary_difference": 0.001,
                "treatment_primary_win_count": 4,
                "valid_action_false_block_rate": None,
                "invalid_feedback_relative_reduction": 0.5,
                "valid_informative_signal_rate_delta": 0.1,
                "critical_valid_action_false_block_count": 0,
                "simple_rule_same_benefit": False,
            },
            "pairs": [],
        }
        first = audit_outcome(analysis, policy)
        self.assertEqual("INCONCLUSIVE", first["experiment_outcome"])
        analysis["aggregate"]["valid_action_false_block_rate"] = 0.0
        second = audit_outcome(analysis, policy)
        self.assertEqual("STRONG_POSITIVE", second["experiment_outcome"])

    def test_neutral_auditor_reports_negative_without_reinterpretation(self) -> None:
        policy = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/outcome_audit_policy.json").read_text()
        )
        analysis = {
            "assigned_pair_count": 6,
            "complete_pair_count": 6,
            "aggregate": {
                "paired_mean_primary_difference": -0.002,
                "paired_median_primary_difference": -0.002,
                "treatment_primary_win_count": 1,
                "valid_action_false_block_rate": 0.0,
                "invalid_feedback_relative_reduction": 0.5,
                "valid_informative_signal_rate_delta": 0.1,
                "critical_valid_action_false_block_count": 0,
                "simple_rule_same_benefit": False,
            },
            "pairs": [],
        }
        result = audit_outcome(analysis, policy)
        self.assertEqual("NEGATIVE", result["experiment_outcome"])


if __name__ == "__main__":
    unittest.main()
