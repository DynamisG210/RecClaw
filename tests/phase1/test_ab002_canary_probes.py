from __future__ import annotations

import unittest

from recclaw_phase1.ab002_canary_probes import run_probes


class AB002CanaryProbeTests(unittest.TestCase):
    def test_frozen_controlled_probes_pass(self) -> None:
        record = run_probes()
        self.assertEqual("LOCAL_COMPLETE", record["status"])
        self.assertTrue(record["fail_open_probe_passed"])
        self.assertTrue(record["quarantine_probe_passed"])


if __name__ == "__main__":
    unittest.main()
