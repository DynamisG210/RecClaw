#!/usr/bin/env python3
"""Generate V13 scientific-attribution schemas from the typed source."""

from __future__ import annotations

import json
from pathlib import Path

from recclaw_core.helix.scientific_attribution import (
    V13_SCHEMA_TYPES,
    schema_for,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "docs"
    / "research_line"
    / "v13_requalification"
    / "schemas"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for record_type in V13_SCHEMA_TYPES:
        path = OUTPUT / f"{record_type.__name__}.schema.json"
        path.write_text(
            json.dumps(
                schema_for(record_type),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
