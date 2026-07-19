from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_candidate_confirmation import select_seed_2026_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads(
    (PROJECT_ROOT / "configs/phase1/ab002/neutral_run_audit_policy.json").read_text()
)


def log_text(model: str, score: float) -> str:
    return f"""INFO ['run_candidate.py', '--model={model}', '--dataset=ml-1m']
seed = 2026
reproducibility = True
epochs = 100
train_batch_size = 2048
learner = adam
learning_rate = 0.001
train_neg_sample_args = {{'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0}}
eval_step = 1
stopping_step = 10
eval_args = {{'split': {{'RS': [0.8, 0.1, 0.1]}}, 'order': 'RO', 'group_by': 'user', 'mode': {{'valid': 'full', 'test': 'full'}}}}
repeatable = False
metrics = ['Recall', 'NDCG', 'MRR', 'Hit', 'Precision']
topk = [10]
valid_metric = NDCG@10
valid_metric_bigger = True
eval_batch_size = 65536
worker = 8
valid_neg_sample_args = {{'distribution': 'uniform', 'sample_num': 'none'}}
test_neg_sample_args = {{'distribution': 'uniform', 'sample_num': 'none'}}
INFO test result: {{'ndcg@10': {score}}}
"""


class AB002CandidateConfirmationTests(unittest.TestCase):
    def test_selection_uses_only_proposed_neutral_eligible_candidates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="RecClaw_evidence_guard_ab_002_") as temporary:
            packet = Path(temporary) / "blind" / "AB002-BLIND-X7Q9"
            source = packet / "source"
            pilot = packet / "pilot"
            (source / "configs/candidates").mkdir(parents=True)
            (source / "recclaw_ext/models").mkdir(parents=True)
            (pilot / "candidates").mkdir(parents=True)
            entries = []
            proposals = []
            result_rows = []
            for index, score in enumerate((0.27, 0.29), start=1):
                candidate_id = f"cand_new_{index}"
                model = f"Candidate{index}"
                run_id = f"candidate_{candidate_id}_fixed"
                entries.append(
                    {
                        "candidate_id": candidate_id,
                        "base_model": "LightGCN",
                        "runner_type": "model",
                        "entrypoint": f"recclaw_ext.models.custom{index}:{model}",
                    }
                )
                proposals.append({"candidate_id": candidate_id})
                (source / f"configs/candidates/{candidate_id}.yaml").write_text(
                    f"model: {model}\n", encoding="utf-8"
                )
                (source / f"recclaw_ext/models/custom{index}.py").write_text(
                    f"class {model}: pass\n", encoding="utf-8"
                )
                log = pilot / "candidates" / f"{run_id}.log"
                result = pilot / "candidates" / f"{run_id}.json"
                log.write_text(log_text(model, score), encoding="utf-8")
                result.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": "success",
                            "ndcg@10": score,
                            "recall@10": 0.2,
                            "mrr@10": 0.3,
                            "hit@10": 0.4,
                            "precision@10": 0.1,
                        }
                    ),
                    encoding="utf-8",
                )
                result_rows.append(
                    {
                        "run_id": run_id,
                        "status": "success",
                        "ndcg@10": score,
                        "notes": f"candidate_id={candidate_id}",
                    }
                )
            (source / "configs/candidate_registry.yaml").write_text(
                __import__("yaml").safe_dump({"candidates": entries}), encoding="utf-8"
            )
            (pilot / "candidate_proposals.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in proposals), encoding="utf-8"
            )
            with (pilot / "results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "ndcg@10", "notes"])
                writer.writeheader()
                writer.writerows(result_rows)
            selection = select_seed_2026_candidate(
                packet_root=packet,
                blind_id="AB002-BLIND-X7Q9",
                policy=POLICY,
                recbole_files=dict(POLICY["recbole_runtime_files_sha256"]),
            )
        self.assertEqual("SELECTED_NEW_CANDIDATE", selection["selection_status"])
        self.assertEqual("cand_new_2", selection["selected"]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
