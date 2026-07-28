from __future__ import annotations

import unittest

from recclaw_core.experiments.helix_abc_v1.m6i_synthetic import (
    run_synthetic_schedule,
)


class M6ISyntheticScheduleTest(unittest.TestCase):
    def test_one_ten_round_schedule_closes_every_triplet(self) -> None:
        result = run_synthetic_schedule(schedule_seed=7, rounds_per_arm=10)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["store_counts"]["terminal_rounds"], 30)
        self.assertEqual(result["store_counts"]["open_rounds"], 0)
        self.assertEqual(result["store_counts"]["closed_barriers"], 10)
        self.assertEqual(
            result["store_counts"]["execution_claims"],
            result["fake_training"]["executions"],
        )
        self.assertEqual(
            result["store_counts"]["finished_execution_claims"],
            result["fake_training"]["executions"],
        )
        self.assertEqual(result["store_counts"]["duplicate_execution_claims"], 0)
        self.assertEqual(result["store_counts"]["duplicate_round_feedback"], 0)
        self.assertEqual(result["meta_boundary_events"], 20)
        self.assertEqual(
            result["fake_broker"]["sharing"][
                "cross_arm_physical_identities"
            ],
            0,
        )
        self.assertEqual(
            set(result["fault_rejections"]),
            {
                "IDEMPOTENT_OPEN_REPLAY",
                "IDEMPOTENT_CLOSE_REPLAY",
                "IDEMPOTENT_EXECUTION_CLAIM_REPLAY",
                "CLOSE_IDENTITY_SUBSTITUTION",
                "EXECUTION_CLAIM_IDENTITY_SUBSTITUTION",
                "CROSS_ARM_STATE_READ",
                "FOREIGN_PARENT",
                "STORE_ARM_OWNER_MISMATCH",
            },
        )


if __name__ == "__main__":
    unittest.main()
