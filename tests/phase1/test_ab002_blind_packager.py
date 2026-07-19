from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_blind_packager import create_blind_packet


class AB002BlindPackagerTests(unittest.TestCase):
    def test_blind_packet_excludes_agent_and_guard_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "control" / "search_seed_42"
            source = run / "source"
            pilot = run / "outputs" / "pilot"
            for relative in ("configs", "recclaw_ext"):
                (source / relative).mkdir(parents=True, exist_ok=True)
                (source / relative / "keep.txt").write_text("ok", encoding="utf-8")
            (source / "scripts").mkdir(parents=True)
            for name in ("run_candidate.py", "collect_result.py", "agent.py"):
                (source / "scripts" / name).write_text(name, encoding="utf-8")
            pilot.mkdir(parents=True)
            for name in ("candidate_proposals.jsonl", "results.csv"):
                (pilot / name).write_text("x", encoding="utf-8")
            for name in ("candidates", "overrides"):
                (pilot / name).mkdir()
                (pilot / name / "keep.txt").write_text("ok", encoding="utf-8")
            (pilot / "evidence_guard_feedback.jsonl").write_text("secret arm state", encoding="utf-8")
            (run / "candidate_execution_budget.jsonl").write_text("{}\n", encoding="utf-8")
            packet_root = root / "blind" / "AB002-BLIND-X7Q9"
            record = create_blind_packet(
                run_root=run,
                blind_root=packet_root,
                blind_id="AB002-BLIND-X7Q9",
                mapping_output=root / "mapping.json",
            )
            self.assertTrue((packet_root / "source/scripts/run_candidate.py").is_file())
            self.assertFalse((packet_root / "source/scripts/agent.py").exists())
            self.assertFalse((packet_root / "pilot/evidence_guard_feedback.jsonl").exists())
            self.assertFalse(any("control" in row["path"] for row in record["file_rows"]))


if __name__ == "__main__":
    unittest.main()
