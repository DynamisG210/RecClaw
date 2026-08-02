#!/usr/bin/env python3
"""Run Q0R DEVELOPMENT_ONLY resource calibration on gpu35."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (  # noqa: E402
    run_resource_scheduling,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q0-external-receipt", type=Path, required=True)
    parser.add_argument(
        "--canonical-receipt",
        type=Path,
        default=(
            ROOT
            / "docs/research_line/vnext/"
            "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
        ),
    )
    args = parser.parse_args()
    result = run_resource_scheduling(
        ROOT,
        q0_external_receipt_path=args.q0_external_receipt,
        canonical_receipt_path=args.canonical_receipt.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
