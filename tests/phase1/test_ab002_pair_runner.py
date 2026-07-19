from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recclaw_phase1.ab002_pair_runner import run_pair
from recclaw_phase1.ab002_s0 import BASELINE_INPUTS


class AB002PairRunnerTests(unittest.TestCase):
    def test_canary_materializes_both_arms_and_runs_contemporaneous_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            baseline.mkdir()
            for item in BASELINE_INPUTS:
                (baseline / item["filename"]).write_text("fixture")
            runtime = root / "runtime"
            environment = {
                "RECCLAW_AB002_START_AUTHORIZED": "YES",
                "RECCLAW_BROKER_CLIENT_TOKEN": "test-token",
            }
            with (
                mock.patch.dict("os.environ", environment, clear=False),
                mock.patch(
                    "recclaw_phase1.ab002_pair_runner.validate_baseline_inputs",
                    return_value=[],
                ),
                mock.patch(
                    "recclaw_phase1.ab002_pair_runner.broker_health",
                    return_value={"status_code": 200, "response": {"status": "ok"}},
                ),
                mock.patch(
                    "recclaw_phase1.ab002_pair_runner._execute",
                    return_value=0,
                ) as execute,
            ):
                record = run_pair(
                    runtime_root=runtime,
                    search_seed=9001,
                    broker_url="http://127.0.0.1:18080",
                    baseline_dir=baseline,
                    python_executable="python3",
                    client_token_env="RECCLAW_BROKER_CLIENT_TOKEN",
                )
            self.assertEqual("LOCAL_COMPLETE", record["status"])
            self.assertEqual(2, execute.call_count)
            self.assertEqual(0, record["plans"]["control"]["gpu_id"])
            self.assertEqual(1, record["plans"]["treatment"]["gpu_id"])
            self.assertEqual(3, record["plans"]["control"]["round_budget"])

    def test_full_pair_requires_external_canary_go_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "RECCLAW_AB002_START_AUTHORIZED": "YES",
                "RECCLAW_BROKER_CLIENT_TOKEN": "test-token",
            }
            with mock.patch.dict("os.environ", environment, clear=True):
                with self.assertRaisesRegex(ValueError, "external Canary GO"):
                    run_pair(
                        runtime_root=Path(temporary) / "runtime",
                        search_seed=42,
                        broker_url="http://127.0.0.1:18080",
                        baseline_dir=Path(temporary) / "baseline",
                        python_executable="python3",
                        client_token_env="RECCLAW_BROKER_CLIENT_TOKEN",
                    )


if __name__ == "__main__":
    unittest.main()
