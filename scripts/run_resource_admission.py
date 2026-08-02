#!/usr/bin/env python3
"""Plan or execute Q0R2 first-principles resource admission on gpu35."""

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
    execute_resource_admission,
    plan_resource_admission,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "execute"), required=True)
    parser.add_argument("--q0-external-receipt", type=Path, required=True)
    args = parser.parse_args()
    function = (
        plan_resource_admission if args.phase == "plan" else execute_resource_admission
    )
    result = function(
        ROOT,
        campaign_root=args.campaign_root,
        q0_external_receipt_path=args.q0_external_receipt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
