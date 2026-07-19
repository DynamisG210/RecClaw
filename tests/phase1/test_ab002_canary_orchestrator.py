from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recclaw_phase1.ab002_canary_orchestrator import run_canary


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class AB002CanaryOrchestratorTests(unittest.TestCase):
    def test_preflight_and_pair_share_ephemeral_client_token_scope(self) -> None:
        observed: list[str] = []

        def fake_preflight(**_: object) -> dict[str, object]:
            observed.append(os.environ.get("RECCLAW_BROKER_CLIENT_TOKEN", ""))
            self.assertEqual("lab-key", os.environ["RECCLAW_LAB_LLM_API_KEY"])
            return {"status": "LOCAL_COMPLETE"}

        def fake_pair(**_: object) -> dict[str, object]:
            observed.append(os.environ.get("RECCLAW_BROKER_CLIENT_TOKEN", ""))
            return {"status": "LOCAL_COMPLETE"}

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            with mock.patch.dict(
                os.environ,
                {
                    "RECCLAW_LAB_LLM_API_KEY": "lab-key",
                    "RECCLAW_LAB_LLM_BASE_URL": "https://lab.invalid/v1",
                },
                clear=True,
            ), mock.patch(
                "recclaw_phase1.ab002_canary_orchestrator.subprocess.Popen",
                return_value=_FakeProcess(),
            ), mock.patch(
                "recclaw_phase1.ab002_canary_orchestrator._wait_for_broker"
            ), mock.patch(
                "recclaw_phase1.ab002_canary_orchestrator.build_preflight",
                side_effect=fake_preflight,
            ), mock.patch(
                "recclaw_phase1.ab002_canary_orchestrator.run_pair",
                side_effect=fake_pair,
            ), mock.patch(
                "recclaw_phase1.ab002_canary_orchestrator.run_probes",
                return_value={"status": "LOCAL_COMPLETE"},
            ), mock.patch(
                "recclaw_phase1.ab002_canary_orchestrator.audit_canary",
                return_value={"status": "LOCAL_COMPLETE"},
            ):
                record = run_canary(
                    runtime_root=runtime,
                    baseline_dir=Path(temporary) / "baseline",
                    expected_tag="phase1-ab002-pre-canary-test",
                    release_identity_path=None,
                    port=18082,
                )
                self.assertNotIn("RECCLAW_BROKER_CLIENT_TOKEN", os.environ)

        self.assertEqual("LOCAL_COMPLETE", record["status"])
        self.assertEqual(2, len(observed))
        self.assertTrue(observed[0])
        self.assertEqual(observed[0], observed[1])


if __name__ == "__main__":
    unittest.main()
