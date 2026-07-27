#!/usr/bin/env python3
"""Freeze the V14 laboratory API release after strict-schema recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiBrokerReleaseV1,
    load_lab_api_credentials,
)


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
)
SCHEMA = RESOURCE_ROOT / "campaign_proposal_response_v1.schema.json"
CONFIG = Path("/root/projects/RecClaw_v2_0_Final_Reference/llm_api.md")
OUTPUT = RESOURCE_ROOT / "lab_api_broker_release_v1_v14_schema_v5.json"


def main() -> None:
    base_url, _api_key = load_lab_api_credentials(CONFIG)
    release = LabApiBrokerReleaseV1.create(
        base_url=base_url,
        model="gpt-5.4",
        response_schema_digest=hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
        max_total_tokens_per_call=20_000,
        timeout_ms=900_000,
    )
    OUTPUT.write_bytes(canonical_json_bytes(release.to_dict()) + b"\n")
    print(
        json.dumps(
            {
                "release_digest": release.release_digest,
                "release_path": OUTPUT.as_posix(),
                "response_schema_digest": (
                    release.response_schema_digest
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
