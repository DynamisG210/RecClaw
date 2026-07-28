#!/usr/bin/env python3
"""Run and record the bounded M6I synthetic qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.m6i_synthetic import (  # noqa: E402
    run_synthetic_qualification,
)


DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "M6I_SYNTHETIC_50R_REPORT.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomized-seeds", type=int, default=100)
    parser.add_argument("--rounds-per-arm", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_synthetic_qualification(
        randomized_seeds=args.randomized_seeds,
        rounds_per_arm=args.rounds_per_arm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "p0": report["p0"],
                "p1": report["p1"],
                "totals": report["totals"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
