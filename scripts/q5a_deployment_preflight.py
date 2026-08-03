#!/usr/bin/env python3
"""Build or execute the Q5-A no-call deployment preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recclaw_core.experiments.helix_abc_v1.q5a_deployment import (  # noqa: E402
    build_q5a_deployment_manifest,
    run_q5a_comprehensive_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--repo-root", type=Path, default=ROOT)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--projects-root", type=Path, required=True)
    manifest.add_argument("--search-data-root", type=Path, required=True)
    manifest.add_argument("--recbole-root", type=Path, required=True)
    manifest.add_argument("--python-executable", type=Path, required=True)
    manifest.add_argument("--api-config", type=Path, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--repo-root", type=Path, default=ROOT)
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        value = build_q5a_deployment_manifest(
            repo_root=args.repo_root,
            projects_root=args.projects_root,
            search_data_root=args.search_data_root,
            recbole_root=args.recbole_root,
            python_executable=args.python_executable,
            api_config=args.api_config,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        print(json.dumps({"status": "MANIFEST_BUILT", "deployment_digest": value["deployment_digest"]}, sort_keys=True))
        return 0
    value = run_q5a_comprehensive_preflight(
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        output_path=args.output,
    )
    print(json.dumps({"status": value["status"], "preflight_digest": value["preflight_digest"]}, sort_keys=True))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
