"""Run the fixed controlled fail-open/quarantine probes for AB-002 Canary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from recclaw_phase1.ab002_launcher import canonical_write


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBES = (
    "tests.exploration.test_original_recclaw_adapter.OriginalRecClawAdapterTests.test_guard_exception_fails_open_and_preserves_original_decision",
    "tests.exploration.test_original_recclaw_adapter.OriginalRecClawAdapterTests.test_live_hook_material_sampled_flip_routes_branch_and_quarantines",
    "tests.exploration.test_original_recclaw_treatment_overlay.OriginalRecClawTreatmentOverlayTests.test_feedback_persistence_failure_fails_open_in_all_treatment_paths",
    "tests.exploration.test_original_recclaw_treatment_overlay.OriginalRecClawTreatmentOverlayTests.test_ordinary_raw_trial_and_frontier_are_guarded_before_write",
)


def run_probes() -> dict[str, object]:
    results = []
    for test_id in PROBES:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", test_id, "-v"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        results.append(
            {
                "test_id": test_id,
                "returncode": completed.returncode,
                "passed": completed.returncode == 0,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
    passed = all(item["passed"] for item in results)
    return {
        "record_type": "RECCLAW_PHASE1_AB002_CANARY_CONTROLLED_PROBES",
        "schema_version": "recclaw.phase1.ab002.canary_controlled_probes.v1",
        "status": "LOCAL_COMPLETE" if passed else "STOPPED",
        "fail_open_probe_passed": results[0]["passed"] and results[2]["passed"],
        "quarantine_probe_passed": results[1]["passed"] and results[3]["passed"],
        "probe_results": results,
        "live_canary_outcome_claim": "NOT_ESTABLISHED_BY_CONTROLLED_PROBES",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed AB-002 Canary integration probes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = run_probes()
    canonical_write(args.output.expanduser().resolve(), record)
    print(json.dumps(record, ensure_ascii=True, sort_keys=True))
    return 0 if record["status"] == "LOCAL_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
