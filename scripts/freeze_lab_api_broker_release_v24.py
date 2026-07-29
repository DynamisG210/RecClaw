#!/usr/bin/env python3
"""Freeze the V24 laboratory API release after strict-schema recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiBrokerReleaseV1,
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
SCHEMA = RESOURCE_ROOT / "campaign_proposal_response_v2.schema.json"
BASE_RELEASE = (
    RESOURCE_ROOT / "lab_api_broker_release_v1_v14_schema_v5.json"
)
OUTPUT = RESOURCE_ROOT / "lab_api_broker_release_v1_v24_schema_v6.json"


def main() -> None:
    base = json.loads(BASE_RELEASE.read_text(encoding="utf-8"))
    payload = {
        "endpoint_digest": base["endpoint_digest"],
        "max_total_tokens_per_call": 20_000,
        "model": "gpt-5.4",
        "request_mode": "SINGLE_JSON_SCHEMA_NO_TOOLS",
        "response_schema_digest": hashlib.sha256(
            SCHEMA.read_bytes()
        ).hexdigest(),
        "retry_count": 0,
        "temperature": 0.0,
        "timeout_ms": 900_000,
        "transport": "HTTPS_CHAT_COMPLETIONS_V1",
    }
    release = LabApiBrokerReleaseV1(
        **payload,
        release_digest=sha256_digest(payload),
    )
    release.verify()
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
