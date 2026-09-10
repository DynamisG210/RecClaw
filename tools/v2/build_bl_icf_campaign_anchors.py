#!/usr/bin/env python3
"""Refresh the proven BL-ICF campaign anchors against the active space identity.

The 11 anchor programs are copied byte-semantically from the V2R4 predecessor;
only the family/search-space identity fields are migrated.  This preserves the
66-entry executable bootstrap while allowing the expanded Provider language to
carry its own exact digest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.mechanism_space import space_identity  # noqa: E402


RESOURCE_ROOT = (
    SRC / "recclaw_core" / "experiments" / "helix_abc_v1" / "resources"
)
SOURCE = RESOURCE_ROOT / "campaign_anchor_programs_v1.json"
OUTPUT = RESOURCE_ROOT / "campaign_anchor_programs_bl_icf_v2r4.json"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build() -> dict[str, Any]:
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 11:
        raise ValueError("the proven V2R4 campaign anchor set must contain 11 fixtures")
    identity = space_identity("BL_ICF_MECHANISM_SPACE_V1")
    for fixture in fixtures:
        program = fixture.get("program")
        if not isinstance(program, dict):
            raise TypeError("campaign anchor fixture is missing its program")
        if program.get("family_id") != identity.family_id:
            raise ValueError("campaign anchor family drifted before identity migration")
        program["family_version"] = identity.family_version
        program["search_space_id"] = identity.search_space_id
        program["search_space_digest"] = identity.search_space_digest
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = _canonical(build())
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != payload:
            raise SystemExit("campaign anchor resource is stale")
        print("PASS campaign_anchors=11")
        return 0
    OUTPUT.write_bytes(payload)
    print("WROTE campaign_anchors=11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
