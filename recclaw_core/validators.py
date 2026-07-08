"""Small validation helpers for RecClaw Core v0.1."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE),
    re.compile(r"OPENAI_API_KEY\s*[:=]\s*[A-Za-z0-9._/-]{20,}"),
    re.compile(r"OPENAI_BASE_URL\s*[:=]\s*https?://\S+"),
]

FORBIDDEN_RUNTIME_PATHS = [
    "results/results.csv",
    "configs/action_space.yaml",
    "configs/candidate_registry.yaml",
    "scripts/run_candidate.py",
    "RecBole",
    "recbole",
]


def secret_scan(root: Path) -> Dict[str, Any]:
    matches: List[Dict[str, str]] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                matches.append({"path": path.as_posix(), "pattern": pattern.pattern})
    return {
        "status": "secret_scan_passed" if not matches else "secret_scan_failed",
        "scanned_file_count": scanned,
        "secret_value_match_count": len(matches),
        "matches": matches,
    }


def forbidden_path_check(project_root: Path) -> Dict[str, Any]:
    args = ["git", "status", "--short", "--", *FORBIDDEN_RUNTIME_PATHS]
    try:
        output = subprocess.check_output(args, cwd=project_root, text=True, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001 - diagnostic helper.
        return {"status": "forbidden_path_check_error", "error": type(exc).__name__, "dirty_entries": []}
    dirty = [line for line in output.splitlines() if line.strip()]
    return {
        "status": "forbidden_path_check_passed" if not dirty else "forbidden_path_check_failed",
        "dirty_entries": dirty,
    }
