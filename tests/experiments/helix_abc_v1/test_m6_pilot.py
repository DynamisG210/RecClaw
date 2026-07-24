from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerCallV1,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.pilot_analysis import (  # noqa: E402
    four_axis_frontiers,
    pilot_readiness,
)
from recclaw_core.experiments.helix_abc_v1.pilot_training import (  # noqa: E402
    training_model_for_primitives,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    FRESH_PILOT_SEARCH_SEED,
    FRESH_PILOT_V5_SEARCH_SEED,
    PILOT_SEARCH_SEED,
    PilotStoreContractV1,
    PilotStoreContractV2,
    PilotStoreContractV3,
    fresh_pilot_guard_context_v2,
    fresh_pilot_guard_context_v3,
    pilot_common_gate_allows,
)


TEMPLATES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"


class FakeUpstream:
    def __init__(self):
        self.calls = {}

    def call(self, *, logical_call_id, prompt, expected_proposal_count):
        if logical_call_id not in self.calls:
            self.calls[logical_call_id] = CanaryBrokerCallV1(
                logical_call_id=logical_call_id,
                request_digest="1" * 64,
                response_digest=("%064x" % (len(self.calls) + 1)),
                response={"proposals": []},
                input_tokens=10,
                cached_input_tokens=0,
                output_tokens=5,
                total_tokens=15,
                latency_ms=1,
                returned_model="fake",
            )
        return self.calls[logical_call_id]


class PilotTrainingProfileTests(unittest.TestCase):
    def test_fresh_v4_identity_is_additive_and_does_not_reuse_v3(self):
        historical = PilotStoreContractV1.create()
        fresh = PilotStoreContractV2.create()
        self.assertEqual(historical.search_seeds, (PILOT_SEARCH_SEED,))
        self.assertEqual(fresh.search_seeds, (FRESH_PILOT_SEARCH_SEED,))
        self.assertEqual(FRESH_PILOT_SEARCH_SEED, 9204)
        self.assertNotEqual(historical.experiment_id, fresh.experiment_id)
        self.assertNotEqual(historical.identity_digest, fresh.identity_digest)
        context = fresh_pilot_guard_context_v2()
        self.assertEqual(context.claim["claim_id"], "CLAIM-M6-PILOT-9204-V4")
        self.assertEqual(
            context.current_evidence["snapshot_id"],
            "M6-PILOT-9204-V4-EMPTY",
        )

    def test_fresh_v5_identity_is_additive_and_uses_next_seed(self):
        sealed = PilotStoreContractV2.create()
        fresh = PilotStoreContractV3.create()
        self.assertEqual(sealed.search_seeds, (9204,))
        self.assertEqual(fresh.search_seeds, (FRESH_PILOT_V5_SEARCH_SEED,))
        self.assertEqual(FRESH_PILOT_V5_SEARCH_SEED, 9205)
        self.assertNotEqual(sealed.experiment_id, fresh.experiment_id)
        self.assertNotEqual(sealed.identity_digest, fresh.identity_digest)
        context = fresh_pilot_guard_context_v3()
        self.assertEqual(context.claim["claim_id"], "CLAIM-M6-PILOT-9205-V5")
        self.assertEqual(
            context.current_evidence["snapshot_id"],
            "M6-PILOT-9205-V5-EMPTY",
        )

    def test_v5_model_catalog_projection_is_package_owned(self):
        path = (
            ROOT
            / "src"
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "resources"
            / "pilot_v5_model_catalog_snapshot.json"
        )
        catalog = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            catalog["record_schema"],
            "recclaw.frozen-selected-model-catalog-projection.v1",
        )
        self.assertEqual(
            [item["slug"] for item in catalog["models"]],
            ["gpt-5.4"],
        )

    def test_pilot_common_gate_uses_frozen_common_decision_value(self):
        self.assertTrue(
            pilot_common_gate_allows(
                "ALLOW_DEVELOPMENT_FAKE_RUN",
                "COMMON_PASS",
            )
        )
        self.assertFalse(pilot_common_gate_allows("ALLOW", "COMMON_PASS"))
        self.assertFalse(
            pilot_common_gate_allows("ALLOW_DEVELOPMENT_FAKE_RUN", "PASS")
        )

    def test_supported_model_mapping_and_geometry_rejection(self):
        self.assertEqual(
            training_model_for_primitives(
                ["encoder.none_mf", "objective.bpr", "score.dot_product"]
            ),
            "BPR",
        )
        self.assertEqual(
            training_model_for_primitives(
                [
                    "encoder.explicit_message_passing",
                    "message.identity",
                    "propagation.symmetric_normalization",
                ]
            ),
            "LightGCN",
        )
        self.assertEqual(
            training_model_for_primitives(
                [
                    "encoder.explicit_message_passing",
                    "message.linear_transform",
                ]
            ),
            "NGCF",
        )
        self.assertEqual(
            training_model_for_primitives(
                ["ssl.objective.info_nce", "ssl.view.edge_dropout"]
            ),
            "SGL",
        )
        with self.assertRaises(ValueError):
            training_model_for_primitives(
                [
                    "encoder.none_mf",
                    "objective.alignment_uniformity",
                    "regularizer.alignment",
                    "regularizer.uniformity",
                ]
            )

    def test_adaptive_broker_shares_equal_memory_then_separates(self):
        upstream = FakeUpstream()
        broker = RealCanaryProposalBrokerV1.create(
            upstream=upstream,
            template_path=TEMPLATES,
            call_prefix="pilot-",
            phase_name="Pilot",
            adaptive_memory=True,
        )
        broker._research_upstream_calls(
            arm=ArmCode.B, round_index=1, search_seed=9201
        )
        broker._research_upstream_calls(
            arm=ArmCode.C, round_index=1, search_seed=9201
        )
        self.assertEqual(len(upstream.calls), 4)
        broker.record_search_feedback(
            ArmCode.B, {"search_outcome": {"run_status": "SUCCESS"}}
        )
        broker.record_search_feedback(
            ArmCode.C,
            {"fusion_instruction": {"memory_target": "VALIDATION_ROUTER"}},
        )
        broker._research_upstream_calls(
            arm=ArmCode.B, round_index=2, search_seed=9201
        )
        broker._research_upstream_calls(
            arm=ArmCode.C, round_index=2, search_seed=9201
        )
        self.assertEqual(len(upstream.calls), 12)
        self.assertTrue(
            all(call_id.rsplit("-", 1)[-1] for call_id in upstream.calls)
        )


class PilotAnalysisTests(unittest.TestCase):
    def rows(self):
        rows = []
        for instance in ("i1", "i2", "i3"):
            for round_index, ndcg in ((1, 0.1), (2, 0.2), (3, 0.15)):
                rows.append(
                    {
                        "billed_tokens": 100,
                        "gpu_cost_microunits": 10,
                        "ndcg": ndcg,
                        "opaque_instance_id": instance,
                        "ordinary_execution_count": 1,
                        "round_index": round_index,
                        "run_status": "SUCCESS",
                    }
                )
        return rows

    def test_four_axes_carry_forward_and_readiness_ignore_effect_size(self):
        rows = self.rows()
        frontiers = four_axis_frontiers(rows)
        self.assertEqual(
            [point["frontier_ndcg"] for point in frontiers["i1"]],
            [0.1, 0.2, 0.2],
        )
        first = pilot_readiness(
            rows,
            expected_instances=3,
            expected_rounds_per_instance=3,
            guard_call_count=6,
            expected_guard_call_count=6,
            meta_versions={"B": 4, "C": 4},
        )
        for row in rows:
            row["ndcg"] = -100.0
        second = pilot_readiness(
            rows,
            expected_instances=3,
            expected_rounds_per_instance=3,
            guard_call_count=6,
            expected_guard_call_count=6,
            meta_versions={"B": 4, "C": 4},
        )
        self.assertEqual(first["verdict"], "GO")
        self.assertEqual(second["verdict"], "GO")

    def test_failure_and_incomplete_packets_do_not_go(self):
        rows = self.rows()
        rows[0]["run_status"] = "RUNTIME_FAILURE"
        rows[1]["run_status"] = "RUNTIME_FAILURE"
        rows[2]["run_status"] = "RUNTIME_FAILURE"
        not_ready = pilot_readiness(
            rows,
            expected_instances=3,
            expected_rounds_per_instance=3,
            guard_call_count=6,
            expected_guard_call_count=6,
            meta_versions={"B": 4, "C": 4},
        )
        self.assertEqual(not_ready["verdict"], "NOT_READY")
        incomplete = pilot_readiness(
            rows[:-1],
            expected_instances=3,
            expected_rounds_per_instance=3,
            guard_call_count=6,
            expected_guard_call_count=6,
            meta_versions={"B": 4, "C": 4},
        )
        self.assertEqual(incomplete["verdict"], "INSUFFICIENT_INFORMATION")


if __name__ == "__main__":
    unittest.main()
