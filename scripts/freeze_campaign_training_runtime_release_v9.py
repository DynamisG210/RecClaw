#!/usr/bin/env python3
"""Freeze Campaign runtime V9 after resource-rejection round closure recovery."""

from __future__ import annotations

import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_runtime_profile,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
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


def _refresh(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    refreshed: list[dict[str, str]] = []
    for row in rows:
        relative = str(row["path"])
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        refreshed.append(
            {"path": relative, "sha256": bytes_sha256(path.read_bytes())}
        )
    return refreshed


def main() -> None:
    base = json.loads(
        (RESOURCE_ROOT / "training_runtime_release_v8.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_profile = campaign_runtime_profile()
    base["source_manifest"] = _refresh(base["source_manifest"])
    base["training_config"] = {
        **base["training_config"],
        "files": _refresh(base["training_config"]["files"]),
    }
    base["release_id"] = "TRAINING_RUNTIME_RELEASE_V9"
    base["campaign_runtime_profile_digest"] = runtime_profile[
        "profile_digest"
    ]
    base["package_owned_handler_registry_digest"] = sha256_digest(
        {
            "campaign_runtime_profile_digest": runtime_profile[
                "profile_digest"
            ],
            "closed_abis": [
                "recclaw.fake-non-training-runner.v1",
                "recclaw.package-owned-search-training-runner.v1",
                "recclaw.package-owned-search-training-runner.v2",
            ],
            "training_release_id": "TRAINING_RUNTIME_RELEASE_V9",
            "training_release_resource": "training_runtime_release_v9.json",
        }
    )
    output = RESOURCE_ROOT / "training_runtime_release_v9.json"
    output.write_bytes(canonical_json_bytes(base) + b"\n")
    print(
        json.dumps(
            {
                "release_digest": sha256_digest(base),
                "release_path": output.as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
