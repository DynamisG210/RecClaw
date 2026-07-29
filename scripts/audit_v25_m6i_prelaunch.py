#!/usr/bin/env python3
"""Mechanically audit the final M6I evidence before V25 contract freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_v25_gpu35_closure import (  # noqa: E402
    GIT_EXECUTABLE,
    activate_original_git_tool,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)


DOCS = ROOT / "docs/research_line/continuous_program"
DEFAULT_OUTPUT = DOCS / "M6I_V25_FINAL_INDEPENDENT_AUDIT_V2.json"
REPORT_PATHS = {
    "arm_order": DOCS / "M6I_ARM_ORDER_INVARIANCE_REPORT.json",
    "call_sharing": DOCS / "M6I_CALL_SHARING_CONFORMANCE.json",
    "exact_100x50": DOCS / "M6I_V20_EXACT_100X50_REPORT.json",
    "mutable_inventory": DOCS / "M6I_MUTABLE_STATE_INVENTORY.json",
    "ownership": DOCS / "M6I_STATE_OWNERSHIP_MAP.json",
    "provider_probe": DOCS / "M6I_V25_PROVIDER_ISOLATION_PROBE.json",
    "state_machine": DOCS / "M6I_STATE_MACHINE_COVERAGE.json",
    "synthetic_100x50": DOCS / "M6I_V20_SYNTHETIC_100X50_REPORT.json",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git_head() -> str:
    activate_original_git_tool()
    return subprocess.check_output(
        [
            str(GIT_EXECUTABLE),
            f"--git-dir={ROOT / '.git'}",
            f"--work-tree={ROOT}",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def _is_pass(report: dict[str, Any]) -> bool:
    return (
        report.get("status", report.get("verdict")) == "PASS"
        and report.get("p0") == 0
        and report.get("p1") == 0
    )


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"audit output already exists: {output}")
    reports = {name: _read(path) for name, path in REPORT_PATHS.items()}
    checks: dict[str, bool] = {
        f"{name}_pass": _is_pass(report)
        for name, report in reports.items()
    }

    inventory = reports["mutable_inventory"]
    ownership = reports["ownership"]
    state_machine = reports["state_machine"]
    sharing = reports["call_sharing"]
    order = reports["arm_order"]
    exact = reports["exact_100x50"]
    synthetic = reports["synthetic_100x50"]
    probe = reports["provider_probe"]

    checks.update(
        {
            "no_forbidden_global_mutable": (
                inventory["forbidden_global_mutable_count"] == 0
            ),
            "all_treatment_state_owned": (
                ownership["unowned_treatment_dependent_objects"] == 0
            ),
            "active_call_sharing_is_arm_private": (
                sharing["active_runtime_policy"] == "ARM_PRIVATE"
            ),
            "v19_v20_v21_fixtures_present": (
                set(
                    state_machine["permanent_regression_fixtures"]["sha256"]
                )
                == {
                    "V19_FAILURE_RECORD.json",
                    "V20_FAILURE_RECORD.json",
                    "V21_CROSS_ARM_CONTAMINATION_HARD_STOP.json",
                    "V21_FAILURE_RECORD.json",
                }
            ),
            "all_six_orders_and_randomized_schedules": (
                order["full_scheduler"]["orders_executed"] == 6
                and order["full_scheduler"]["invariant"] is True
                and order["randomized_canonical_core"]["schedule_seeds"]
                == 500
                and not order["randomized_canonical_core"]["failures"]
            ),
            "exact_100x50_terminal": (
                exact["rounds_per_arm"] == 50
                and exact["randomized_schedule_seeds"] == 100
                and exact["totals"]["terminal_arm_rounds"] == 15_000
                and exact["totals"]["closed_triplet_barriers"] == 5_000
                and exact["totals"]["open_arm_rounds"] == 0
                and exact["totals"]["unfinished_execution_claims"] == 0
                and exact["totals"]["cross_arm_physical_identities"] == 0
                and exact["real_provider_calls"] == 0
                and exact["real_training_executions"] == 0
            ),
            "synthetic_100x50_terminal": (
                synthetic["rounds_per_arm"] == 50
                and synthetic["randomized_schedule_seeds"] == 100
                and synthetic["totals"]["terminal_arm_rounds"] == 15_000
                and synthetic["totals"]["closed_triplet_barriers"] == 5_000
                and synthetic["totals"]["open_arm_rounds"] == 0
                and synthetic["totals"]["successful_cross_arm_reads"] == 0
                and synthetic["totals"]["successful_cross_arm_writes"] == 0
                and synthetic["totals"][
                    "cross_arm_physical_call_identities"
                ]
                == 0
                and synthetic["totals"]["confirmed_frontier_entries"] == 0
            ),
            "provider_probe_fresh_and_isolated": (
                probe["probe_experiment_id"]
                == "M6I-PROVIDER-ISOLATION-PROBE-V6"
                and probe["successful_provider_calls"] == 12
                and probe["training_executions"] == 0
                and all(probe["checks"].values())
            ),
            "all_reports_deny_authority": all(
                report.get("authority") == "NONE"
                for report in reports.values()
            ),
            "no_report_claims_formal_acceptance": all(
                report.get("formal_acceptance", False) is False
                for report in reports.values()
            ),
        }
    )

    source_projection = exact["source_projection"]
    checks["exact_source_projection_digest"] = (
        sha256_digest(source_projection)
        == exact["source_projection_digest"]
        and exact["source_projection_unchanged"] is True
    )
    drift = [
        relative
        for relative, digest in source_projection.items()
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        != digest
    ]
    checks["exact_source_projection_current"] = not drift

    failures = sorted(name for name, passed in checks.items() if not passed)
    evidence = {
        name: {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in REPORT_PATHS.items()
    }
    payload = {
        "authority": "NONE",
        "checks": checks,
        "evidence": evidence,
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "failures": failures,
        "formal_acceptance": False,
        "main_eligibility": False,
        "p0": len(failures),
        "p1": 0,
        "record_schema": "recclaw.m6i-v25-final-independent-audit.v2",
        "source_head": _git_head(),
        "source_projection_drift": drift,
        "status": "PASS" if not failures else "FAIL",
    }
    report = {**payload, "audit_digest": sha256_digest(payload)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.output.resolve())
    print(
        json.dumps(
            {
                "audit_digest": report["audit_digest"],
                "p0": report["p0"],
                "p1": report["p1"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
