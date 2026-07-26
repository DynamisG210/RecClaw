from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_m6s_static_diagnostic_pilot import (  # noqa: E402
    STATIC_DIAGNOSTIC_SEARCH_SEED,
    StaticDiagnosticPilotStoreContractV1,
    descriptive_effect_summary,
    static_diagnostic_readiness,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
    MetaPolicyModeV1,
)


class StaticDiagnosticPilotTests(unittest.TestCase):
    def test_store_contract_uses_new_seed_and_static_research_policy(self) -> None:
        contract = StaticDiagnosticPilotStoreContractV1.create()
        self.assertEqual(contract.search_seeds, (STATIC_DIAGNOSTIC_SEARCH_SEED,))
        self.assertEqual(contract.scheduled_slots_per_arm_seed, 1)
        policies = {policy.arm: policy for policy in contract.arm_policies}
        self.assertIsNone(policies[ArmCode.A].meta_policy_mode)
        self.assertEqual(
            policies[ArmCode.B].meta_policy_mode,
            MetaPolicyModeV1.STATIC_RESEARCH_ROUTER,
        )
        self.assertEqual(
            policies[ArmCode.C].meta_policy_mode,
            MetaPolicyModeV1.STATIC_RESEARCH_ROUTER,
        )
        self.assertEqual(
            policies[ArmCode.B].non_guard_projection(),
            policies[ArmCode.C].non_guard_projection(),
        )

    def test_static_readiness_requires_version_one_and_complete_triplet(self) -> None:
        rows = [
            {
                "opaque_instance_id": instance,
                "run_status": "SUCCESS",
                "ndcg": 0.1 + index / 100,
            }
            for index, instance in enumerate(("opaque-a", "opaque-b", "opaque-c"))
        ]
        result = static_diagnostic_readiness(
            rows,
            expected_instance_ids={"opaque-a", "opaque-b", "opaque-c"},
            guard_call_count=2,
            meta_versions={"B": 1, "C": 1},
        )
        self.assertEqual(result["verdict"], "CHAIN_PASS")
        self.assertTrue(result["checks"]["meta_not_activated"])

    def test_descriptive_effect_summary_is_not_formal_evidence(self) -> None:
        rows = [
            {
                "candidate_id": f"candidate-{arm}",
                "ndcg": ndcg,
                "opaque_instance_id": f"opaque-{arm}",
                "run_status": "SUCCESS",
            }
            for arm, ndcg in (("a", 0.10), ("b", 0.13), ("c", 0.12))
        ]
        result = descriptive_effect_summary(
            rows,
            {"A": "opaque-a", "B": "opaque-b", "C": "opaque-c"},
        )
        self.assertAlmostEqual(result["contrasts"]["B_minus_A"], 0.03)
        self.assertAlmostEqual(result["contrasts"]["C_minus_B"], -0.01)
        self.assertFalse(result["main_or_meta_evidence"])
        self.assertEqual(
            result["interpretation"], "DESCRIPTIVE_SINGLE_ROUND_ONLY"
        )


if __name__ == "__main__":
    unittest.main()
