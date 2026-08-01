#!/usr/bin/env python3
"""Run the fresh F1 open-Meta architecture-effect campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.fresh_f1 import (  # noqa: E402
    offline_f1_check,
    run_f1_runtime_recovery,
    run_formal_fresh_f1,
    verify_f1_source_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-receipt",
        type=Path,
        default=(
            ROOT
            / "docs/research_line/vnext/"
            "F1_OPEN_META_CANONICAL_RECEIPT.json"
        ),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime-recovery", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = verify_f1_source_identity(ROOT, require_fresh_root=True)
    elif args.offline:
        result = offline_f1_check(ROOT)
    elif args.runtime_recovery:
        recovery_receipt = (
            ROOT
            / "docs/research_line/vnext/"
            "F1_OPEN_META_RUNTIME_RECOVERY_V2_CANONICAL_RECEIPT.json"
        )
        if args.canonical_receipt != parser.get_default("canonical_receipt"):
            recovery_receipt = args.canonical_receipt
        result = run_f1_runtime_recovery(
            ROOT,
            canonical_receipt_path=recovery_receipt.resolve(),
        )
    else:
        result = run_formal_fresh_f1(
            ROOT,
            canonical_receipt_path=args.canonical_receipt.resolve(),
        )
    print(json.dumps(result, sort_keys=True))
    if args.preflight or args.offline:
        return 0
    return 0 if result["status"] == "F1_ARCHITECTURE_EFFECT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
