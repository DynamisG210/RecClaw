#!/usr/bin/env python3
"""Run the Q0 DEVELOPMENT_ONLY common-mode calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.quality_calibration import (  # noqa: E402
    offline_q0_check,
    run_quality_calibration,
    verify_q0_source_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-receipt",
        type=Path,
        default=(
            ROOT
            / "docs/research_line/vnext/"
            "Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json"
        ),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--delegated-source-identity", type=Path)
    parser.add_argument("--runtime-environment-identity", type=Path)
    args = parser.parse_args()
    if args.preflight:
        result = verify_q0_source_identity(ROOT, require_fresh_root=True)
    elif args.offline:
        result = offline_q0_check(ROOT)
    else:
        result = run_quality_calibration(
            ROOT,
            canonical_receipt_path=args.canonical_receipt.resolve(),
            delegated_source_identity=(
                json.loads(args.delegated_source_identity.read_text(encoding="utf-8"))
                if args.delegated_source_identity
                else None
            ),
            runtime_environment_identity=(
                json.loads(
                    args.runtime_environment_identity.read_text(encoding="utf-8")
                )
                if args.runtime_environment_identity
                else None
            ),
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
