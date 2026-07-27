#!/usr/bin/env python3
"""Validate the active Campaign training release against one backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    campaign_training_runtime_release,
    validate_campaign_training_runtime_release,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--recbole-root", type=Path, required=True)
    args = parser.parse_args()
    release = campaign_training_runtime_release()
    failures = validate_campaign_training_runtime_release(
        data_path=args.data_path,
        python_executable=args.python,
        recbole_root=args.recbole_root,
    )
    print(
        json.dumps(
            {
                "failure_codes": list(failures),
                "release_digest": release.digest,
                "release_id": release.release_id,
                "verdict": "PASS" if not failures else "FAIL",
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
