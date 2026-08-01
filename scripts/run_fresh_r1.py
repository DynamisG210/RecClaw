#!/usr/bin/env python3
"""Run the exact accepted formal fresh R1 once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (  # noqa: E402
    run_formal_fresh_r1,
    verify_formal_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-receipt",
        type=Path,
        default=ROOT / "docs/research_line/vnext/R1_FRESH_CANONICAL_RECEIPT.json",
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = verify_formal_identity(ROOT)
        print(json.dumps(result, sort_keys=True))
        return 0
    receipt = run_formal_fresh_r1(
        ROOT,
        canonical_receipt_path=args.canonical_receipt.resolve(),
    )
    print(
        json.dumps(
            {
                "canonical_receipt": str(args.canonical_receipt.resolve()),
                "episode_counts": receipt["episode_counts"],
                "gate_pass": receipt["gate"]["pass"],
                "status": receipt["status"],
                "wall_time_ms": receipt["wall_time_ms"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "R1_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
