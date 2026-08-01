#!/usr/bin/env python3
"""Run the unique fresh R2 registry-consumer architecture-effect campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (  # noqa: E402
    offline_registry_consumer_check,
    run_formal_fresh_r2,
    verify_r2_source_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-receipt",
        type=Path,
        default=(
            ROOT
            / "docs/research_line/vnext/"
            "R2_FRESH_REGISTRY_EFFECT_CANONICAL_RECEIPT.json"
        ),
    )
    parser.add_argument("--offline-consumer-check", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(
            json.dumps(
                verify_r2_source_identity(ROOT, require_fresh_root=True),
                sort_keys=True,
            )
        )
        return 0
    if args.offline_consumer_check:
        print(
            json.dumps(
                offline_registry_consumer_check(ROOT),
                sort_keys=True,
            )
        )
        return 0
    receipt = run_formal_fresh_r2(
        ROOT,
        canonical_receipt_path=args.canonical_receipt.resolve(),
    )
    print(
        json.dumps(
            {
                "canonical_receipt": str(args.canonical_receipt.resolve()),
                "held_out_reads": receipt["held_out_reads"],
                "selected_capability_ref": receipt["selection"][
                    "selected_capability_ref"
                ],
                "status": receipt["status"],
                "wall_time_ms": receipt["wall_time_ms"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "R2_ARCHITECTURE_EFFECT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
