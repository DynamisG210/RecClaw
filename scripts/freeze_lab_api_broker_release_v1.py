#!/usr/bin/env python3
"""Freeze the public laboratory API transport identity for a Campaign schema."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    LabApiBrokerReleaseV1,
    load_lab_api_credentials,
)


RESOURCE_ROOT = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
DEFAULT_OUTPUT = (
    RESOURCE_ROOT / "lab_api_broker_release_v1_campaign_schema_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-api-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    schema = RESOURCE_ROOT / "campaign_proposal_response_v1.schema.json"
    base_url, _api_key = load_lab_api_credentials(
        args.llm_api_config.resolve()
    )
    release = LabApiBrokerReleaseV1.create(
        base_url=base_url,
        model="gpt-5.4",
        response_schema_digest=hashlib.sha256(
            schema.read_bytes()
        ).hexdigest(),
        max_total_tokens_per_call=20_000,
        timeout_ms=900_000,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        canonical_json_bytes(release.to_dict()) + b"\n"
    )
    print(release.release_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
