"""Pre-registered paired analysis for the AB-002 clean-start experiment."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_SEEDS = (42, 43, 44, 45, 46, 47)
TRAINING_SEEDS = (2026, 2027, 2028)
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_DRAWS = 20000


def _outside_source(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved
    raise ValueError("AB-002 final analysis inputs and outputs must be outside the source checkout")


def _finite(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_interval(differences: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def exact_sign_flip_result(differences: list[float]) -> dict[str, Any]:
    observed = statistics.fmean(differences)
    values = [
        statistics.fmean(sign * value for sign, value in zip(signs, differences))
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in values)
    return {
        "statistic": observed,
        "two_sided_p_value": extreme / len(values),
        "enumerated_assignments": len(values),
    }


def _confirmation_frontier(
    arm: dict[str, Any], *, baseline_mean: float
) -> dict[str, Any]:
    if arm.get("new_candidate_discovered") is False:
        return {
            "status": "LOCAL_COMPLETE",
            "candidate_id": None,
            "frontier_source": "FIXED_LIGHTGCN_BASELINE",
            "three_seed_values": [],
            "three_seed_mean_ndcg_at_10": baseline_mean,
            "three_seed_variance": 0.0,
            "delta_vs_lightgcn": 0.0,
            "reason": str(arm.get("no_candidate_reason") or "NO_NEW_PROTOCOL_VALID_CANDIDATE"),
        }
    candidate_id = str(arm.get("candidate_id") or "")
    runs = arm.get("confirmation_runs")
    if not candidate_id or not isinstance(runs, list):
        return {"status": "INCONCLUSIVE", "reason": "CANDIDATE_OR_CONFIRMATION_RUNS_MISSING"}
    by_seed: dict[int, float] = {}
    diagnostics = []
    for row in runs:
        if not isinstance(row, dict):
            diagnostics.append("CONFIRMATION_ROW_INVALID")
            continue
        seed = row.get("training_seed")
        audit = row.get("neutral_audit")
        if seed not in TRAINING_SEEDS:
            diagnostics.append("CONFIRMATION_SEED_INVALID")
            continue
        if not isinstance(audit, dict) or audit.get("eligible_for_performance_analysis") is not True:
            diagnostics.append(f"NEUTRAL_AUDIT_INELIGIBLE:{seed}")
            continue
        metrics = audit.get("metrics") if isinstance(audit.get("metrics"), dict) else {}
        value = _finite(metrics.get("ndcg@10"))
        if value is None:
            diagnostics.append(f"CONFIRMATION_METRIC_MISSING:{seed}")
            continue
        by_seed[int(seed)] = value
    if set(by_seed) != set(TRAINING_SEEDS):
        diagnostics.append("THREE_SEED_CONFIRMATION_INCOMPLETE")
    if diagnostics:
        return {
            "status": "INCONCLUSIVE",
            "candidate_id": candidate_id,
            "reason": "CONFIRMATION_NOT_ELIGIBLE",
            "diagnostics": sorted(set(diagnostics)),
        }
    values = [by_seed[seed] for seed in TRAINING_SEEDS]
    mean = statistics.fmean(values)
    return {
        "status": "LOCAL_COMPLETE",
        "candidate_id": candidate_id,
        "frontier_source": "NEW_PROTOCOL_VALID_CANDIDATE",
        "three_seed_values": [
            {"training_seed": seed, "ndcg@10": by_seed[seed]} for seed in TRAINING_SEEDS
        ],
        "three_seed_mean_ndcg_at_10": mean,
        "three_seed_variance": statistics.pvariance(values),
        "delta_vs_lightgcn": mean - baseline_mean,
    }


def _baseline(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("runs") if isinstance(report.get("runs"), list) else []
    values = {}
    diagnostics = []
    for row in rows:
        if not isinstance(row, dict) or row.get("training_seed") not in TRAINING_SEEDS:
            diagnostics.append("BASELINE_ROW_INVALID")
            continue
        audit = row.get("neutral_audit")
        if not isinstance(audit, dict) or audit.get("eligible_for_performance_analysis") is not True:
            diagnostics.append(f"BASELINE_NEUTRAL_AUDIT_INELIGIBLE:{row.get('training_seed')}")
            continue
        metrics = audit.get("metrics") if isinstance(audit.get("metrics"), dict) else {}
        metric = _finite(metrics.get("ndcg@10"))
        if metric is None:
            diagnostics.append(f"BASELINE_METRIC_MISSING:{row.get('training_seed')}")
            continue
        values[int(row["training_seed"])] = metric
    if set(values) != set(TRAINING_SEEDS):
        diagnostics.append("BASELINE_THREE_SEED_SET_INCOMPLETE")
    if diagnostics:
        return {"status": "INCONCLUSIVE", "diagnostics": sorted(set(diagnostics))}
    ordered = [values[seed] for seed in TRAINING_SEEDS]
    return {
        "status": "LOCAL_COMPLETE",
        "values": [{"training_seed": seed, "ndcg@10": values[seed]} for seed in TRAINING_SEEDS],
        "mean_ndcg_at_10": statistics.fmean(ordered),
        "variance": statistics.pvariance(ordered),
    }


def _curve(value: object, baseline: float) -> list[float]:
    if not isinstance(value, list):
        return []
    output = []
    running = baseline
    for item in value[:20]:
        metric = _finite(item)
        if metric is None:
            output.append(running)
            continue
        running = max(running, metric)
        output.append(running)
    return output


def _curve_summary(pairs: list[dict[str, Any]], baseline: float) -> dict[str, Any]:
    result: dict[str, Any] = {"individual": [], "aggregate": {}}
    for pair in pairs:
        for arm_name in ("control", "treatment"):
            curve = pair[arm_name].get("running_best_valid_curve", [])
            result["individual"].append(
                {"search_seed": pair["search_seed"], "arm": arm_name, "values": curve}
            )
    for arm_name in ("control", "treatment"):
        curves = [
            item[arm_name].get("running_best_valid_curve", [])
            for item in pairs
            if item[arm_name].get("running_best_valid_curve")
        ]
        points = []
        for index in range(20):
            values = [curve[index] if index < len(curve) else curve[-1] for curve in curves]
            if not values:
                points.append({"budget": index + 1, "mean": baseline, "lower": None, "upper": None})
                continue
            points.append(
                {
                    "budget": index + 1,
                    "mean": statistics.fmean(values),
                    "lower": min(values),
                    "upper": max(values),
                    "interval_role": "observed_repetition_range",
                }
            )
        result["aggregate"][arm_name] = points
    return result


def _svg(curves: dict[str, Any]) -> str:
    width, height = 920, 520
    left, top, plot_width, plot_height = 70, 35, 810, 420
    all_values = [
        value for item in curves["individual"] for value in item.get("values", [])
    ]
    ymin = min(all_values, default=0.26)
    ymax = max(all_values, default=0.30)
    if ymax <= ymin:
        ymax = ymin + 0.001

    def xy(index: int, value: float) -> tuple[float, float]:
        x = left + plot_width * (index / 20)
        y = top + plot_height * (1 - (value - ymin) / (ymax - ymin))
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
    ]
    for item in curves["individual"]:
        color = "#2563eb" if item["arm"] == "control" else "#dc2626"
        points = " ".join(
            f"{xy(index, value)[0]:.2f},{xy(index, value)[1]:.2f}"
            for index, value in enumerate(item["values"], start=1)
        )
        if points:
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1" opacity="0.3"/>'
            )
    for arm, color in (("control", "#2563eb"), ("treatment", "#dc2626")):
        points = " ".join(
            f"{xy(row['budget'], row['mean'])[0]:.2f},{xy(row['budget'], row['mean'])[1]:.2f}"
            for row in curves["aggregate"][arm]
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def analyze(runtime_root: Path, output_dir: Path) -> dict[str, Any]:
    root = _outside_source(runtime_root)
    output = _outside_source(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = root / "three_seed_baseline_report.json"
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {}
    baseline = _baseline(baseline_report)
    if baseline.get("status") != "LOCAL_COMPLETE":
        result = {
            "record_type": "RECCLAW_PHASE1_AB002_PAIRED_ANALYSIS",
            "schema_version": "recclaw.phase1.ab002.paired_analysis.v2",
            "status": "IN_PROGRESS",
            "assigned_pair_count": 6,
            "complete_pair_count": 0,
            "diagnostics": ["THREE_SEED_BASELINE_INCOMPLETE"],
            "baseline": baseline,
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
        }
        (output / "paired_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    baseline_mean = float(baseline["mean_ndcg_at_10"])
    pairs = []
    for seed in SEARCH_SEEDS:
        path = root / "pair_inputs" / f"search_seed_{seed}.json"
        if not path.is_file():
            pairs.append({"search_seed": seed, "status": "INCONCLUSIVE", "reason": "PAIR_INPUT_MISSING"})
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        if source.get("search_seed") != seed:
            pairs.append({"search_seed": seed, "status": "INCONCLUSIVE", "reason": "PAIR_SEED_MISMATCH"})
            continue
        arms = source.get("arms") if isinstance(source.get("arms"), dict) else {}
        pair = {"search_seed": seed, "status": "LOCAL_COMPLETE"}
        for arm_name in ("control", "treatment"):
            arm = arms.get(arm_name) if isinstance(arms.get(arm_name), dict) else {}
            frontier = _confirmation_frontier(arm, baseline_mean=baseline_mean)
            pair[arm_name] = {
                **frontier,
                "running_best_valid_curve": _curve(arm.get("running_best_valid_curve"), baseline_mean),
                "mechanism": arm.get("mechanism") if isinstance(arm.get("mechanism"), dict) else {},
                "events": arm.get("events") if isinstance(arm.get("events"), list) else [],
            }
            if arm_name == "treatment":
                guard_applied = (
                    arm.get("guard_applied_secondary")
                    if isinstance(arm.get("guard_applied_secondary"), dict)
                    else {}
                )
                pair[arm_name]["guard_applied_secondary"] = (
                    _confirmation_frontier(guard_applied, baseline_mean=baseline_mean)
                    if guard_applied
                    else {"status": "INCONCLUSIVE", "reason": "GUARD_APPLIED_SUBSET_MISSING"}
                )
            if frontier.get("status") != "LOCAL_COMPLETE":
                pair["status"] = "INCONCLUSIVE"
        if pair["status"] == "LOCAL_COMPLETE":
            pair["primary_paired_difference"] = (
                pair["treatment"]["delta_vs_lightgcn"] - pair["control"]["delta_vs_lightgcn"]
            )
        pairs.append(pair)

    complete = [pair for pair in pairs if pair.get("status") == "LOCAL_COMPLETE"]
    differences = [float(pair["primary_paired_difference"]) for pair in complete]
    guard_secondary_pairs = [
        pair
        for pair in complete
        if pair["treatment"]["guard_applied_secondary"].get("status") == "LOCAL_COMPLETE"
    ]
    guard_secondary_differences = [
        float(pair["treatment"]["guard_applied_secondary"]["delta_vs_lightgcn"])
        - float(pair["control"]["delta_vs_lightgcn"])
        for pair in guard_secondary_pairs
    ]
    control_invalid = [
        _finite(pair["control"]["mechanism"].get("invalid_feedback_entering_primary_memory_rate"))
        for pair in complete
    ]
    treatment_invalid = [
        _finite(pair["treatment"]["mechanism"].get("invalid_feedback_entering_primary_memory_rate"))
        for pair in complete
    ]
    control_signal = [
        _finite(pair["control"]["mechanism"].get("valid_informative_signal_rate"))
        for pair in complete
    ]
    treatment_signal = [
        _finite(pair["treatment"]["mechanism"].get("valid_informative_signal_rate"))
        for pair in complete
    ]
    c_invalid = statistics.fmean(value for value in control_invalid if value is not None) if all(value is not None for value in control_invalid) and control_invalid else None
    t_invalid = statistics.fmean(value for value in treatment_invalid if value is not None) if all(value is not None for value in treatment_invalid) and treatment_invalid else None
    c_signal = statistics.fmean(value for value in control_signal if value is not None) if all(value is not None for value in control_signal) and control_signal else None
    t_signal = statistics.fmean(value for value in treatment_signal if value is not None) if all(value is not None for value in treatment_signal) and treatment_signal else None
    reviewed = sum(int(pair["treatment"]["mechanism"].get("guard_reviewed_valid_action_count") or 0) for pair in complete)
    false_blocks = sum(int(pair["treatment"]["mechanism"].get("valid_action_false_block_count") or 0) for pair in complete)
    critical_false_blocks = sum(int(pair["treatment"]["mechanism"].get("critical_valid_action_false_block_count") or 0) for pair in complete)
    simple_path = root / "simple_rule_comparison.json"
    simple_rule = json.loads(simple_path.read_text(encoding="utf-8")) if simple_path.is_file() else {}
    aggregate = {
        "paired_mean_primary_difference": statistics.fmean(differences) if differences else None,
        "paired_median_primary_difference": statistics.median(differences) if differences else None,
        "paired_bootstrap_95_interval": paired_bootstrap_interval(differences) if len(differences) == 6 else None,
        "exact_paired_sign_flip": exact_sign_flip_result(differences) if len(differences) == 6 else None,
        "treatment_primary_win_count": sum(value > 0 for value in differences),
        "invalid_feedback_relative_reduction": ((c_invalid - t_invalid) / c_invalid if c_invalid is not None and t_invalid is not None and c_invalid > 0 else (0.0 if c_invalid == t_invalid == 0 else None)),
        "valid_informative_signal_rate_delta": (t_signal - c_signal if t_signal is not None and c_signal is not None else None),
        "valid_action_false_block_rate": false_blocks / reviewed if reviewed else None,
        "critical_valid_action_false_block_count": critical_false_blocks,
        "simple_rule_same_benefit": simple_rule.get("same_benefit") if isinstance(simple_rule.get("same_benefit"), bool) else None,
    }
    curves = _curve_summary(complete, baseline_mean)
    result = {
        "record_type": "RECCLAW_PHASE1_AB002_PAIRED_ANALYSIS",
        "schema_version": "recclaw.phase1.ab002.paired_analysis.v2",
        "status": "LOCAL_COMPLETE" if len(complete) == 6 else "IN_PROGRESS",
        "experiment_run_status": "LOCAL_COMPLETE" if len(complete) == 6 else "IN_PROGRESS",
        "assigned_pair_count": 6,
        "complete_pair_count": len(complete),
        "intention_to_treat": True,
        "baseline": baseline,
        "pairs": pairs,
        "aggregate": aggregate,
        "guard_applied_secondary_analysis": {
            "complete_pair_count": len(guard_secondary_pairs),
            "paired_differences": guard_secondary_differences,
            "paired_mean_primary_difference": (
                statistics.fmean(guard_secondary_differences)
                if guard_secondary_differences
                else None
            ),
            "interpretation": "SECONDARY_NON_ITT_GUARD_SUCCESS_SUBSET",
        },
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "paired_search_seed"},
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    (output / "paired_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "curve_data.json").write_text(json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "running_best_curves.svg").write_text(_svg(curves), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen AB-002 paired final analysis")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.runtime_root, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
