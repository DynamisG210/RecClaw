from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.neutral_outcome_auditor import audit_raw_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads(
    (PROJECT_ROOT / "configs/phase1/ab002/neutral_run_audit_policy.json").read_text()
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_log(run_id: str, *, mode: str = "full") -> str:
    return f"""01 Jan 00:00 INFO ['run_candidate.py', '--model=CandidateModel', '--dataset=ml-1m']
seed = 2026
reproducibility = True
epochs = 100
train_batch_size = 2048
learner = adam
learning_rate = 0.001
train_neg_sample_args = {{'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0}}
eval_step = 1
stopping_step = 10
eval_args = {{'split': {{'RS': [0.8, 0.1, 0.1]}}, 'order': 'RO', 'group_by': 'user', 'mode': {{'valid': '{mode}', 'test': '{mode}'}}}}
repeatable = False
metrics = ['Recall', 'NDCG', 'MRR', 'Hit', 'Precision']
topk = [10]
valid_metric = NDCG@10
valid_metric_bigger = True
eval_batch_size = 65536
worker = 8
valid_neg_sample_args = {{'distribution': 'uniform', 'sample_num': 'none'}}
test_neg_sample_args = {{'distribution': 'uniform', 'sample_num': 'none'}}
01 Jan 00:01 INFO test result: {{'ndcg@10': 0.28}}
"""


class NeutralRawRunAuditorTests(unittest.TestCase):
    def envelope(self, root: Path, *, mode: str = "full") -> dict[str, object]:
        run_id = "candidate_blind_001"
        log = root / f"{run_id}.log"
        result = root / f"{run_id}.json"
        config = root / "candidate.yaml"
        source = root / "candidate.py"
        log.write_text(valid_log(run_id, mode=mode), encoding="utf-8")
        result.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "success",
                    "ndcg@10": 0.28,
                    "recall@10": 0.2,
                    "mrr@10": 0.3,
                    "hit@10": 0.4,
                    "precision@10": 0.1,
                }
            ),
            encoding="utf-8",
        )
        config.write_text("model: CandidateModel\n", encoding="utf-8")
        source.write_text("class CandidateModel: pass\n", encoding="utf-8")
        snapshot = POLICY["dataset_snapshot_sha256"]
        return {
            "anonymous_run_id": "BLIND-001",
            "log_path": str(log),
            "result_json_path": str(result),
            "planned_training_seed": 2026,
            "dataset_snapshot_sha256": snapshot,
            "shared_state_integrity": {
                "dataset_before_sha256": snapshot,
                "dataset_after_sha256": snapshot,
                "recbole_files_before": dict(POLICY["recbole_runtime_files_sha256"]),
                "recbole_files_after": dict(POLICY["recbole_runtime_files_sha256"]),
            },
            "candidate_artifacts": [
                {"role": "candidate_config", "path": str(config), "sha256": digest(config)},
                {"role": "candidate_source", "path": str(source), "sha256": digest(source)},
            ],
        }

    def test_valid_blinded_full_sort_run_is_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_raw_run(self.envelope(Path(temporary)), POLICY)
        self.assertTrue(result["eligible_for_performance_analysis"])
        self.assertEqual([], result["diagnostics"])
        self.assertFalse(result["arm_label_read"])
        self.assertFalse(result["guard_classification_read"])

    def test_single_material_sampled_evaluation_flip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_raw_run(self.envelope(Path(temporary), mode="uni100"), POLICY)
        self.assertFalse(result["eligible_for_performance_analysis"])
        self.assertIn("PROTOCOL_MISMATCH:eval_args", result["diagnostics"])

    def test_arm_bearing_input_is_rejected_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            envelope = self.envelope(Path(temporary))
            envelope["arm"] = "treatment"
            result = audit_raw_run(envelope, POLICY)
        self.assertFalse(result["eligible_for_performance_analysis"])
        self.assertIn("AUDITOR_INPUT_NOT_ARM_BLIND", result["diagnostics"])

    def test_single_recbole_identity_fault_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            envelope = self.envelope(Path(temporary))
            shared = envelope["shared_state_integrity"]
            assert isinstance(shared, dict)
            before = shared["recbole_files_before"]
            assert isinstance(before, dict)
            before["run_recbole.py"] = "0" * 64
            result = audit_raw_run(envelope, POLICY)
        self.assertFalse(result["eligible_for_performance_analysis"])
        self.assertIn("RECBOLE_BEFORE_IDENTITY_MISMATCH", result["diagnostics"])
        self.assertIn("RECBOLE_SHARED_STATE_CHANGED", result["diagnostics"])

    def test_non_arm_evidence_guard_root_and_norm_control_candidate_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="RecClaw_evidence_guard_ab_002_") as temporary:
            envelope = self.envelope(Path(temporary))
            envelope["candidate_artifacts"][0]["path"] = str(
                Path(temporary) / "norm_control_candidate.yaml"
            )
            candidate = Path(envelope["candidate_artifacts"][0]["path"])
            candidate.write_text("model: CandidateModel\n", encoding="utf-8")
            envelope["candidate_artifacts"][0]["sha256"] = digest(candidate)
            result = audit_raw_run(envelope, POLICY)
        self.assertTrue(result["eligible_for_performance_analysis"])
        self.assertNotIn("AUDITOR_INPUT_NOT_ARM_BLIND", result["diagnostics"])


if __name__ == "__main__":
    unittest.main()
