#!/usr/bin/env python3
"""Build the repository Q0R receipt from an immutable physical HARD_BLOCK."""

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
    finalize_hard_block_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-receipt", type=Path, required=True)
    parser.add_argument("--external-receipt-ref")
    parser.add_argument(
        "--canonical-receipt",
        type=Path,
        default=(
            ROOT
            / "docs/research_line/vnext/"
            "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
        ),
    )
    args = parser.parse_args()
    result = finalize_hard_block_receipt(
        args.external_receipt,
        canonical_receipt_path=args.canonical_receipt.resolve(),
        external_receipt_ref=args.external_receipt_ref,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
