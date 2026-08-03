#!/usr/bin/env python3
"""Freeze one Q4 round and initialize its sealed-stage ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.multiround_soak import (  # noqa: E402
    create_stage_ledger,
    freeze_round_manifest,
)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _write_new(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-plan", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--previous-projection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    plan = _read(args.campaign_plan.resolve())
    if plan.get("held_out_reads") != 0 or plan.get("development_only") is not True:
        raise RuntimeError("campaign plan authority boundary drift")
    round_plan = plan["rounds"][str(args.round_index)]
    common = plan["frozen_execution_common"]
    frozen_execution = canonical_value(
        {
            **common,
            **round_plan,
            "previous_projection_sha256": bytes_sha256(
                args.previous_projection.resolve().read_bytes()
            ),
        }
    )
    round_root = args.output_root.resolve() / f"round_{args.round_index:02d}"
    round_root.mkdir(parents=True, exist_ok=False)
    manifest_path = round_root / "ROUND_MANIFEST.json"
    manifest = freeze_round_manifest(
        output_path=manifest_path,
        campaign_id=str(plan["campaign_id"]),
        round_index=args.round_index,
        policy_path=args.policy.resolve(),
        activation_path=args.activation.resolve(),
        full_pool_path=args.pool.resolve(),
        acquisition_seeds=round_plan["acquisition_seeds"],
        execution_task_type=str(plan["execution_task_type"]),
        frozen_execution=frozen_execution,
        previous_round_activation_digest=(
            str(plan["base_activation_digest"])
            if args.round_index == 1
            else str(_read(args.activation.resolve())["activation_digest"])
        ),
    )
    ledger = create_stage_ledger(
        manifest_path=manifest_path,
        ledger_path=round_root / "STAGE_LEDGER.json",
    )
    receipt_payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-round-preparation.v1",
            "campaign_plan_sha256": bytes_sha256(args.campaign_plan.resolve().read_bytes()),
            "round_index": args.round_index,
            "manifest_digest": manifest["manifest_digest"],
            "manifest_sha256": bytes_sha256(manifest_path.read_bytes()),
            "ledger_digest": ledger["ledger_digest"],
            "input_policy_sha256": bytes_sha256(args.policy.resolve().read_bytes()),
            "input_activation_sha256": bytes_sha256(args.activation.resolve().read_bytes()),
            "input_pool_sha256": bytes_sha256(args.pool.resolve().read_bytes()),
            "previous_projection_sha256": bytes_sha256(
                args.previous_projection.resolve().read_bytes()
            ),
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    receipt = {**receipt_payload, "receipt_digest": sha256_digest(receipt_payload)}
    _write_new(round_root / "ROUND_PREPARATION_RECEIPT.json", receipt)
    print(
        json.dumps(
            {
                "round_root": str(round_root),
                "manifest_digest": manifest["manifest_digest"],
                "selected_candidate_id": manifest["selected_candidate"]["candidate_id"],
                "exploration_selected": manifest["selected_candidate"]["exploration_selected"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
