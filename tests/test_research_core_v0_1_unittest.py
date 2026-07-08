import tempfile
import unittest
from pathlib import Path

from recclaw_core.contracts import NATIVE_BPR_RANKCUT_MEMORY_ID, load_yaml
from recclaw_core.foundry import replay_fixture
from recclaw_core.memory import ingest_bpr_rankcut_memory, retrieve_memory
from recclaw_core.policy import rank_candidates, select_candidate_queue
from recclaw_core.verifier import verify_claim_boundary


FIXTURE = Path("artifacts/research_core/golden_foundry_fixture")


class ResearchCoreV01Test(unittest.TestCase):
    def test_memory_ingestion_and_retrieval(self):
        memory = ingest_bpr_rankcut_memory(FIXTURE / "phase9_5")
        self.assertEqual(memory["memory_id"], NATIVE_BPR_RANKCUT_MEMORY_ID)
        self.assertIs(memory["source_policy"]["uses_codex_chat"], False)
        self.assertTrue(retrieve_memory("future BPR rank surrogate using batch local cutoff", [memory]))
        self.assertEqual(retrieve_memory("LightGCN residual norm depth variance candidate", [memory]), [])

    def test_policy_changes_top_candidate(self):
        cards = load_yaml(FIXTURE / "candidate_cards.yaml")["candidate_cards"]
        memory = ingest_bpr_rankcut_memory(FIXTURE / "phase9_5")
        without_memory = rank_candidates(cards, [])
        with_memory = rank_candidates(cards, [memory])
        self.assertEqual(
            without_memory["ranked_candidates"][0]["candidate_id"],
            "fixture_bpr_batch_local_rankcut_repeat",
        )
        self.assertEqual(
            with_memory["ranked_candidates"][0]["candidate_id"],
            "fixture_lightgcn_residual_norm_depth_variance",
        )
        queue = select_candidate_queue(with_memory["ranked_candidates"])
        selected = {row["candidate_id"] for row in queue["selected_candidates"]}
        self.assertIn("fixture_lightgcn_residual_norm_depth_variance", selected)
        self.assertNotIn("fixture_bpr_batch_local_rankcut_repeat", selected)

    def test_foundry_replay_and_claim_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = replay_fixture(FIXTURE, Path(temp_dir))
            self.assertEqual(result["status"], "candidate_foundry_fixture_replayed")
            self.assertEqual(result["claim_boundary_status"], "claim_boundary_passed")
            report = load_yaml(Path(temp_dir) / "claim_boundary_report.yaml")
            self.assertFalse(report["search_quality_supported"])
            self.assertFalse(report["metric_improvement_supported"])
            self.assertFalse(report["formal_success_supported"])

    def test_claim_boundary_rejects_search_quality_claim(self):
        report = {
            "supported_scope": "candidate_foundry_fixture_replay_only",
            "selected_candidate_queue_written": True,
            "search_quality_supported": True,
            "metric_improvement_supported": False,
            "formal_success_supported": False,
            "M5_ready": False,
            "runtime_kernel_ready": False,
        }
        result = verify_claim_boundary(report)
        self.assertEqual(result["status"], "claim_boundary_failed")
        self.assertIn("search_quality_supported_must_be_false", result["issues"])


if __name__ == "__main__":
    unittest.main()
