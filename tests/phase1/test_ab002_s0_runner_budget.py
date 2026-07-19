from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "phase1/s0/ab002/source/scripts/run_candidate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("ab002_s0_run_candidate", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load S0 runner")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class AB002CandidateExecutionBudgetTests(unittest.TestCase):
    def test_single_fault_second_reservation_exceeds_fixed_budget(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "candidate_execution_budget.jsonl"
            environment = {
                "RECCLAW_CANDIDATE_EXECUTION_BUDGET": "1",
                "RECCLAW_CANDIDATE_EXECUTION_LEDGER": str(ledger),
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                first = runner.reserve_candidate_execution(
                    candidate_id="candidate-a",
                    run_id="run-a",
                    purpose="ordinary",
                )
                self.assertEqual(0, first["sequence_index"])
                with self.assertRaisesRegex(
                    runner.CandidateExecutionBudgetExhausted, "budget exhausted"
                ):
                    runner.reserve_candidate_execution(
                        candidate_id="candidate-b",
                        run_id="run-b",
                        purpose="smoke",
                    )
            self.assertEqual(1, len(ledger.read_text().splitlines()))


if __name__ == "__main__":
    unittest.main()
