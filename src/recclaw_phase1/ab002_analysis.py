"""Deterministic AB-002 analysis and dependency-free SVG plotting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)


def _outside_source(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved
    raise ValueError("analysis inputs and outputs must be outside the source checkout")


def _float(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def guard_dispositions(feedback_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    by_run: dict[str, str] = {}
    rows = read_jsonl(feedback_path)
    latencies = []
    revision_count = 0
    protocol_mismatch_count = 0
    quarantined = 0
    admitted = 0
    runtime_blockers = 0
    for row in rows:
        latency = _float(row.get("guard_latency_ms"))
        if latency is not None:
            latencies.append(latency)
        if row.get("phase") == "PRECHECK" and row.get("next_iteration_effect") in {
            "REQUEST_PROPOSAL_REVISION",
            "ROUTE_TO_SEPARATE_PROTOCOL_BRANCH",
        }:
            revision_count += 1
        binding = row.get("live_protocol_binding")
        if isinstance(binding, dict) and binding.get("material_differences"):
            protocol_mismatch_count += 1
        disposition = str(row.get("original_trial_memory_disposition") or "")
        if disposition == "QUARANTINE_ORIGINAL_TRIAL":
            quarantined += 1
        elif disposition == "ADMIT_ORIGINAL_TRIAL":
            admitted += 1
        if row.get("postcheck_outcome_classification") == "RUNTIME_BLOCKER":
            runtime_blockers += 1
        snapshot = row.get("original_result_snapshot")
        if isinstance(snapshot, dict):
            run_id = str(snapshot.get("run_id") or "")
            if run_id:
                by_run[run_id] = disposition
    metrics = {
        "feedback_event_count": len(rows),
        "revision_or_branch_count": revision_count,
        "protocol_mismatch_count": protocol_mismatch_count,
        "quarantined_original_trial_count": quarantined,
        "admitted_original_trial_count": admitted,
        "runtime_blocker_count": runtime_blockers,
        "mean_guard_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p95_guard_latency_ms": (
            sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)]
            if latencies
            else None
        ),
    }
    return by_run, metrics


def result_curve(
    results_path: Path,
    *,
    baseline: float,
    admitted_by_run: dict[str, str] | None,
) -> dict[str, Any]:
    if not results_path.is_file():
        return {"status": "NOT_STARTED", "points": [], "final_best": None, "auc": None}
    points = []
    best = baseline
    successful = 0
    admitted = 0
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        for budget, row in enumerate(csv.DictReader(handle), start=1):
            status = str(row.get("status") or "").strip().lower()
            metric = _float(row.get("ndcg@10"))
            run_id = str(row.get("run_id") or "")
            success = status == "success" and metric is not None
            if success:
                successful += 1
            is_admitted = success and (
                admitted_by_run is None
                or admitted_by_run.get(run_id) == "ADMIT_ORIGINAL_TRIAL"
            )
            if is_admitted:
                admitted += 1
                best = max(best, float(metric))
            points.append({"budget": budget, "running_best": best, "run_id": run_id})
    auc = statistics.fmean(point["running_best"] for point in points) if points else baseline
    return {
        "status": "LOCAL_COMPLETE",
        "points": points,
        "row_count": len(points),
        "successful_result_count": successful,
        "admitted_result_count": admitted,
        "final_best": best,
        "auc": auc,
    }


def broker_metrics(db_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        return {"status": "NOT_STARTED", "request_count": 0}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT pair_relation, state, COUNT(*) FROM requests GROUP BY pair_relation, state"
        ).fetchall()
    counts = {f"{relation}:{state}": count for relation, state, count in rows}
    return {
        "status": "LOCAL_COMPLETE",
        "request_count": sum(counts.values()),
        "counts": counts,
        "divergent_request_count": sum(
            value for key, value in counts.items() if key.startswith("DIVERGENT_REQUEST:")
        ),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _svg(pairs: list[dict[str, Any]]) -> str:
    width, height = 920, 520
    left, top, plot_width, plot_height = 70, 35, 810, 420
    all_points = [
        point["running_best"]
        for pair in pairs
        for arm in ("control", "treatment")
        for point in pair[arm].get("points", [])
    ]
    ymin = min(all_points, default=0.26)
    ymax = max(all_points, default=0.30)
    if ymax <= ymin:
        ymax = ymin + 0.001
    max_budget = max(
        (len(pair[arm].get("points", [])) for pair in pairs for arm in ("control", "treatment")),
        default=1,
    )

    def xy(index: int, value: float) -> tuple[float, float]:
        x = left + plot_width * (index / max(1, max_budget))
        y = top + plot_height * (1 - (value - ymin) / (ymax - ymin))
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        '<text x="460" y="505" text-anchor="middle" font-family="sans-serif" font-size="14">candidate-result budget</text>',
        '<text x="18" y="250" transform="rotate(-90 18 250)" text-anchor="middle" font-family="sans-serif" font-size="14">admissible running-best NDCG@10</text>',
    ]
    for pair in pairs:
        for arm, color in (("control", "#2563eb"), ("treatment", "#dc2626")):
            points = pair[arm].get("points", [])
            if not points:
                continue
            coords = " ".join(
                f"{xy(i, float(point['running_best']))[0]:.2f},{xy(i, float(point['running_best']))[1]:.2f}"
                for i, point in enumerate(points, start=1)
            )
            lines.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.55"/>'
            )
    lines.extend(
        [
            '<line x1="690" y1="20" x2="720" y2="20" stroke="#2563eb" stroke-width="3"/><text x="728" y="25" font-family="sans-serif" font-size="13">Control</text>',
            '<line x1="790" y1="20" x2="820" y2="20" stroke="#dc2626" stroke-width="3"/><text x="828" y="25" font-family="sans-serif" font-size="13">Treatment</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(runtime_root: Path, output_dir: Path, baseline: float) -> dict[str, Any]:
    root = _outside_source(runtime_root)
    output = _outside_source(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for seed in SEEDS:
        control_dir = root / "runs/control" / f"search_seed_{seed}" / "outputs/pilot"
        treatment_dir = root / "runs/treatment" / f"search_seed_{seed}" / "outputs/pilot"
        dispositions, guard = guard_dispositions(
            treatment_dir / "evidence_guard_feedback.jsonl"
        )
        pairs.append(
            {
                "search_seed": seed,
                "control": result_curve(
                    control_dir / "results.csv", baseline=baseline, admitted_by_run=None
                ),
                "treatment": result_curve(
                    treatment_dir / "results.csv",
                    baseline=baseline,
                    admitted_by_run=dispositions,
                ),
                "guard_mechanisms": guard,
            }
        )
    complete_pairs = [
        pair
        for pair in pairs
        if pair["control"]["status"] == "LOCAL_COMPLETE"
        and pair["treatment"]["status"] == "LOCAL_COMPLETE"
    ]
    result = {
        "record_type": "RECCLAW_PHASE1_AB002_ANALYSIS",
        "schema_version": "recclaw.phase1.ab002.analysis.v1",
        "status": "LOCAL_COMPLETE" if len(complete_pairs) == len(SEEDS) else "IN_PROGRESS",
        "experiment_run_status": "NOT_STARTED" if not complete_pairs else "IN_PROGRESS",
        "complete_pair_count": len(complete_pairs),
        "pairs": pairs,
        "aggregate": {
            "control_mean_final_best": _mean(pair["control"].get("final_best") for pair in complete_pairs),
            "treatment_mean_final_best": _mean(pair["treatment"].get("final_best") for pair in complete_pairs),
            "control_mean_auc": _mean(pair["control"].get("auc") for pair in complete_pairs),
            "treatment_mean_auc": _mean(pair["treatment"].get("auc") for pair in complete_pairs),
            "mean_guard_latency_ms": _mean(
                pair["guard_mechanisms"].get("mean_guard_latency_ms") for pair in complete_pairs
            ),
            "quarantined_original_trial_count": sum(
                int(pair["guard_mechanisms"]["quarantined_original_trial_count"])
                for pair in complete_pairs
            ),
            "protocol_mismatch_count": sum(
                int(pair["guard_mechanisms"]["protocol_mismatch_count"])
                for pair in complete_pairs
            ),
            "valid_action_false_block_rate": None,
            "valid_action_false_block_oracle": "INDEPENDENT_ADJUDICATION_REQUIRED",
        },
        "broker": broker_metrics(root / "broker/paired_llm.sqlite3"),
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    (output / "ab002_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "ab002_running_best.svg").write_text(_svg(pairs), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze AB-002 external outputs")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=float, default=0.2671)
    args = parser.parse_args(argv)
    result = analyze(args.runtime_root, args.output_dir, args.baseline)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
