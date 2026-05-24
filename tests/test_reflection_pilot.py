from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_reflection_pilot as pilot  # noqa: E402


class ReflectionPilotTests(unittest.TestCase):
    def test_choose_proposal_source_defaults_to_heuristic_without_key(self) -> None:
        self.assertEqual(pilot.choose_proposal_source(None, env={}), "heuristic")
        self.assertEqual(pilot.choose_proposal_source("llm", env={}), "llm")

    def test_dry_run_prints_complete_commands_without_creating_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = pilot.main(
                    [
                        "--dry-run",
                        "--pilot-root",
                        tmp,
                        "--stamp",
                        "unit",
                        "--rounds",
                        "50",
                        "--gpu-id",
                        "2",
                        "--proposal-source",
                        "heuristic",
                    ]
                )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertIn("lint", payload["commands"])
            self.assertIn("initial_tree", payload["commands"])
            self.assertIn("initial_summary", payload["commands"])
            self.assertIn("agent", payload["commands"])
            self.assertIn("--rounds 50", payload["commands"]["agent"])
            self.assertIn("--refresh-experience-every 10", payload["commands"]["agent"])
            self.assertIn("--set gpu_id=2", payload["commands"]["agent"])
            self.assertFalse((Path(tmp) / "unit").exists())


if __name__ == "__main__":
    unittest.main()
