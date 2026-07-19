"""Pre-registered neutral outcome classification for AB-002 development results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase1/ab002/outcome_audit_policy.json"


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def audit_outcome(analysis: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    aggregate = analysis.get("aggregate") if isinstance(analysis.get("aggregate"), dict) else {}
    complete_pairs = int(analysis.get("complete_pair_count") or 0)
    control = _number(aggregate.get("control_mean_final_best"))
    treatment = _number(aggregate.get("treatment_mean_final_best"))
    false_block = _number(aggregate.get("valid_action_false_block_rate"))
    guard_latency = _number(aggregate.get("mean_guard_latency_ms"))
    pair_rows = analysis.get("pairs") if isinstance(analysis.get("pairs"), list) else []
    total_runtime_ms = 0.0
    for pair in pair_rows:
        if isinstance(pair, dict):
            value = _number(pair.get("mean_control_candidate_runtime_ms"))
            if value is not None:
                total_runtime_ms += value
    latency_fraction = (
        guard_latency / total_runtime_ms if guard_latency is not None and total_runtime_ms > 0 else None
    )
    reasons = []
    outcome = "INCONCLUSIVE"
    primary_delta = treatment - control if treatment is not None and control is not None else None
    minimum_pairs = int(policy["minimum_complete_pairs"])
    if complete_pairs < minimum_pairs:
        reasons.append("INSUFFICIENT_COMPLETE_PAIRS")
    if false_block is None:
        reasons.append("INDEPENDENT_FALSE_BLOCK_ADJUDICATION_MISSING")
    if primary_delta is None:
        reasons.append("PRIMARY_OUTCOME_MISSING")
    if not reasons:
        noninferiority = primary_delta >= -float(policy["primary_noninferiority_margin"])
        false_block_ok = false_block <= float(policy["maximum_valid_action_false_block_rate"])
        latency_ok = latency_fraction is None or latency_fraction <= float(policy["maximum_guard_latency_fraction"])
        mechanism_change = (
            int(aggregate.get("quarantined_original_trial_count") or 0) > 0
            or int(aggregate.get("protocol_mismatch_count") or 0) > 0
        )
        if not false_block_ok or primary_delta < -float(policy["primary_noninferiority_margin"]):
            outcome = "NEGATIVE"
        elif primary_delta >= float(policy["positive_final_best_delta"]) and latency_ok:
            outcome = "POSITIVE"
        elif noninferiority and false_block_ok and latency_ok and (
            mechanism_change or not bool(policy["neutral_requires_mechanism_change"])
        ):
            outcome = "NEUTRAL_USEFUL"
        else:
            outcome = "INCONCLUSIVE"
            reasons.append("PRE_REGISTERED_OUTCOME_CONJUNCTION_NOT_MET")
    return {
        "record_type": "RECCLAW_PHASE1_AB002_NEUTRAL_OUTCOME_AUDIT",
        "schema_version": "recclaw.phase1.ab002.neutral_outcome_audit.v1",
        "status": "LOCAL_COMPLETE",
        "experiment_outcome": outcome,
        "reason_codes": reasons,
        "complete_pair_count": complete_pairs,
        "primary_final_best_delta": primary_delta,
        "valid_action_false_block_rate": false_block,
        "guard_latency_fraction": latency_fraction,
        "policy_id": policy["policy_id"],
        "decision_effect": "DEVELOPMENT_REPORT_ONLY",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the neutral AB-002 outcome rule")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    result = audit_outcome(analysis, policy)
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise SystemExit("outcome audit output must be outside the source checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
