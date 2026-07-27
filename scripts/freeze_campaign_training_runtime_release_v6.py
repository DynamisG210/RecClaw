#!/usr/bin/env python3
"""Freeze the V14 content-bound Campaign training runtime release V6."""

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


def _refresh_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    refreshed = []
    for row in rows:
        path = ROOT / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        refreshed.append(
            {
                "path": str(row["path"]),
                "sha256": bytes_sha256(path.read_bytes()),
            }
        )
    return refreshed


def main() -> None:
    base = json.loads(
        (RESOURCE_ROOT / "training_runtime_release_v5.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_profile = campaign_runtime_profile()
    base["campaign_runtime_profile_digest"] = runtime_profile[
        "profile_digest"
    ]
    base["release_id"] = "TRAINING_RUNTIME_RELEASE_V6"
    base["source_manifest"] = _refresh_rows(base["source_manifest"])
    base["environment_lock"]["files"] = _refresh_rows(
        base["environment_lock"]["files"]
    )
    base["training_config"]["files"] = _refresh_rows(
        base["training_config"]["files"]
    )
    sources = {
        row["path"]: row["sha256"] for row in base["source_manifest"]
    }
    base["launcher_source_digest"] = sources[
        "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py"
    ]
    base["runner_entrypoint_digest"] = sources[
        "scripts/campaign_train_worker.py"
    ]
    base["runner_source_digest"] = sources[
        "scripts/campaign_train_worker.py"
    ]
    base["close_result_policy_digest"] = sha256_digest(
        {
            "close_contract": base["close_contract"],
            "guard_source": sources[
                "src/recclaw_core/experiments/helix_abc_v1/"
                "training_execution_guard.py"
            ],
        }
    )
    base["common_guard_policy_digest"] = sha256_digest(
        {
            "guard": base["close_contract"]["common_guard"],
            "source": sources[
                "src/recclaw_core/experiments/helix_abc_v1/"
                "training_execution_guard.py"
            ],
        }
    )
    base["metric_parser_digest"] = sha256_digest(
        {
            "metric_contract": base["metric_contract"],
            "worker_source": sources["scripts/campaign_train_worker.py"],
        }
    )
    base["config_schema_digest"] = sha256_digest(base["training_config"])
    base["python_environment_lock_digest"] = sha256_digest(
        base["environment_lock"]
    )
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
            "training_release_id": "TRAINING_RUNTIME_RELEASE_V6",
            "training_release_resource": "training_runtime_release_v6.json",
        }
    )
    output = RESOURCE_ROOT / "training_runtime_release_v6.json"
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
