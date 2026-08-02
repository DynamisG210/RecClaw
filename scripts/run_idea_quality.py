#!/usr/bin/env python3
"""Run the one-shot Q1 DEVELOPMENT_ONLY Idea/OpenSpec quality chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.idea_quality import (  # noqa: E402
    build_q1_ab_contract,
    build_research_context,
    run_idea_quality,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        result = {
            "ab_contract": build_q1_ab_contract(),
            "research_context": build_research_context(ROOT),
            "held_out_reads": 0,
        }
    else:
        if args.run_root is None:
            parser.error("--run-root is required unless --offline is used")
        result = run_idea_quality(ROOT, run_root=args.run_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
