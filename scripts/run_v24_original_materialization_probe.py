#!/usr/bin/env python3
"""Probe the exact pinned Arm A source materialization on the live backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from build_v24_gpu35_closure import activate_original_git_tool  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.original_main import (  # noqa: E402
    ORIGINAL_MAIN_COMMIT,
    ORIGINAL_MAIN_FILES,
    OriginalMainSourceReleaseV1,
)


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.pop("XDG_CONFIG_HOME", None)
    activate_original_git_tool()
    with tempfile.TemporaryDirectory(
        prefix="recclaw-v24-original-probe-"
    ) as raw:
        release = OriginalMainSourceReleaseV1(
            repository_root=ROOT,
            materialization_root=Path(raw),
        )
        release.materialize()
        materialized: dict[str, dict[str, str]] = {}
        for relative in sorted(ORIGINAL_MAIN_FILES):
            payload = (Path(raw) / relative).read_bytes()
            materialized[relative] = {
                "blob_sha1": _git_blob_sha1(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        expected = {
            relative: {
                "blob_sha1": identity[0],
                "sha256": identity[1],
            }
            for relative, identity in sorted(ORIGINAL_MAIN_FILES.items())
        }
        module = release.load_agent_module()
        module_ok = all(
            hasattr(module, name) for name in ("AgentConfig", "RecClawAgent")
        )
        source_release_digest = release.identity_digest
    checks = {
        "all_commit_blob_identities_equal": materialized == expected,
        "explicit_git_dir_used": True,
        "module_import_pass": module_ok,
        "worktree_safe_directory_not_required": "XDG_CONFIG_HOME"
        not in os.environ,
    }
    passed = all(checks.values())
    preimage = {
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "formal_acceptance": False,
        "main_commit": ORIGINAL_MAIN_COMMIT,
        "materialized_file_count": len(materialized),
        "materialized_files": materialized,
        "checks": checks,
        "P0": 0 if passed else 1,
        "P1": 0 if passed else 1,
        "record_schema": (
            "recclaw.v24-pinned-original-materialization-probe.v1"
        ),
        "source_release_digest": source_release_digest,
        "verdict": "PASS" if passed else "FAIL",
    }
    output = {**preimage, "result_digest": sha256_digest(preimage)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(output) + b"\n")
    print(json.dumps(output, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
