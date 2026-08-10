#!/usr/bin/env python3
"""Prepare or execute one complete Research Line round."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1 import fresh_r1  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)
from recclaw_core.research_line.single_round import (  # noqa: E402
    compose_single_round,
    execute_single_round,
    recover_incomplete_single_round,
    recover_single_round,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--recover", action="store_true")
    mode.add_argument("--recover-incomplete", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--api-config",
        type=Path,
        default=fresh_r1.API_CONFIG,
    )
    parser.add_argument(
        "--campaign-id",
        default="research-line-single-round-20260805-01",
    )
    parser.add_argument("--seed", type=int, default=54303)
    parser.add_argument("--epochs", type=int, default=fresh_r1.EXPERIMENT_EPOCHS)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--watchdog-seconds", type=int, default=1800)
    parser.add_argument("--incumbent-receipt", type=Path)
    parser.add_argument("--source-run-root", type=Path)
    args = parser.parse_args()

    composition = compose_single_round(
        repo_root=args.repo_root,
        run_root=args.run_root,
        api_config_path=args.api_config,
        campaign_id=args.campaign_id,
        seed=args.seed,
        epochs=args.epochs,
        timeout_seconds=args.timeout_seconds,
        watchdog_seconds=args.watchdog_seconds,
        incumbent_receipt_path=args.incumbent_receipt,
    )
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_READY_NO_CALL",
                    "context_ref": composition.context.context_ref,
                    "context_digest": composition.context.digest,
                    "profile_entry_count": len(composition.profile.entries),
                    "manifest_digest": sha256_digest(composition.manifest),
                    "run_root": str(composition.run_root),
                    "provider_calls": 0,
                    "experiment_calls": 0,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.recover_incomplete:
        if args.source_run_root is None:
            parser.error("--recover-incomplete requires --source-run-root")
        source = compose_single_round(
            repo_root=args.repo_root,
            run_root=args.source_run_root,
            api_config_path=args.api_config,
            campaign_id=args.campaign_id,
            seed=args.seed,
            epochs=args.epochs,
            timeout_seconds=args.timeout_seconds,
            watchdog_seconds=args.watchdog_seconds,
            incumbent_receipt_path=args.incumbent_receipt,
        )
        summary = recover_incomplete_single_round(source, composition)
    elif args.recover:
        summary = recover_single_round(composition)
    else:
        summary = execute_single_round(composition)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] in {
        "COMPLETE",
        "RECOVERED_EPISODE_INCOMPLETE_ARCHITECTURE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
