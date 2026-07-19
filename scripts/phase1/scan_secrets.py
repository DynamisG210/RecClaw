#!/usr/bin/env python3
"""High-confidence secret scanner used by the Phase-1 clean-tree check."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "openai_style_token": re.compile(rb"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "slack_token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential_url": re.compile(rb"[a-z][a-z0-9+.-]*://[^\s/@:]{2,}:[^\s/@]{8,}@", re.I),
}
SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks"}


def git_paths(root: Path, include_untracked: bool) -> list[Path]:
    command = ["git", "-C", str(root), "ls-files", "-z"]
    if include_untracked:
        command.extend(["--cached", "--others", "--exclude-standard"])
    raw = subprocess.check_output(command)
    return [root / value.decode("utf-8", errors="surrogateescape") for value in raw.split(b"\0") if value]


def scan(root: Path, include_untracked: bool) -> dict[str, object]:
    hits = []
    files = git_paths(root, include_untracked)
    for path in files:
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name.lower() in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
            hits.append({"path": rel, "rule": "sensitive_filename"})
            continue
        if path.stat().st_size > 8 * 1024 * 1024:
            continue
        data = path.read_bytes()
        for name, pattern in PATTERNS.items():
            if pattern.search(data):
                hits.append({"path": rel, "rule": name})
    return {
        "record_type": "RECCLAW_PHASE1_SECRET_SCAN",
        "status": "LOCAL_COMPLETE" if not hits else "STOPPED",
        "scanned_file_count": len(files),
        "finding_count": len(hits),
        "findings": hits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan the Phase-1 source tree for high-confidence secrets")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include-untracked", action="store_true")
    args = parser.parse_args(argv)
    result = scan(args.root.expanduser().resolve(), bool(args.include_untracked))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["finding_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
