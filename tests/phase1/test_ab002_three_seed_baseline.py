from __future__ import annotations

import unittest

from recclaw_phase1.ab002_three_seed_baseline import summarize_metric


class AB002ThreeSeedBaselineTests(unittest.TestCase):
    def test_summary_is_fixed_to_three_seeds(self) -> None:
        result = summarize_metric([0.2, 0.3, 0.4])
        self.assertAlmostEqual(0.3, result["mean"])
        self.assertAlmostEqual(0.006666666666666667, result["population_variance"])
        self.assertEqual({"2026", "2027", "2028"}, set(result["values_by_seed"]))

    def test_summary_rejects_missing_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "three finite"):
            summarize_metric([0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
