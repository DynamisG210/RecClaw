#!/usr/bin/env python3
"""Relocate the frozen RecBole editable finder inside the packed venv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    site_packages = next(
        (runtime_root / "lib").glob("python*/site-packages")
    )
    finders = tuple(
        site_packages.glob("__editable___recbole_*_finder.py")
    )
    if len(finders) != 1:
        raise RuntimeError("expected one frozen RecBole editable finder")
    finder = finders[0]
    before = finder.read_text(encoding="utf-8")
    old_root = str(args.old_root)
    new_root = args.new_root.resolve().as_posix()
    if old_root not in before:
        raise RuntimeError("frozen RecBole root is absent from finder")
    after = before.replace(old_root, new_root)
    finder.write_text(after, encoding="utf-8")
    payload = {
        "finder_path": finder.relative_to(runtime_root).as_posix(),
        "new_root": new_root,
        "old_root": old_root,
        "post_sha256": bytes_sha256(finder.read_bytes()),
        "pre_sha256": bytes_sha256(before.encode("utf-8")),
        "relocation_scope": "EDITABLE_IMPORT_PATH_ONLY",
    }
    payload["record_digest"] = sha256_digest(payload)
    args.record.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
