#!/usr/bin/env python3
"""Compare two RecClaw run results and emit keep/discard/crash."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    # Package import path (e.g. import scripts.compare_runs)
    from .collect_result import load_result_source
except ImportError:
    # Script execution path (e.g. py scripts/compare_runs.py)
    from collect_result import load_result_source


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_csv_result(csv_path: str | Path, row_number: int | None, run_id: str | None) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if run_id:
        for row in rows:
            if row.get("run_id") == run_id:
                return row
        raise ValueError(f"run_id not found in CSV: {run_id}")

    if row_number is None:
        raise ValueError("row_number or run_id must be provided for CSV loading")

    if row_number < 1 or row_number > len(rows):
        raise ValueError(
            f"row_number {row_number} is out of range for CSV with {len(rows)} data rows"
        )
    return rows[row_number - 1]


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric_name: str,
    multi_metrics: bool = False,
    metrics_weights: dict[str, float] = None,
) -> dict[str, Any]:
    candidate_status = str(candidate.get("status", "")).lower()

    if candidate_status != "success":
        decision = "crash"
        explanation = (
            f"candidate status is {candidate.get('status') or 'unknown'}, so the run is treated as crash."
        )
        return {
            "baseline_metric_name": metric_name,
            "baseline_metric": None,
            "candidate_metric_name": metric_name,
            "candidate_metric": None,
            "delta": None,
            "decision": decision,
            "explanation": explanation,
        }

    if multi_metrics and metrics_weights:
        # Multi-objective optimization using weighted sum
        baseline_score = 0.0
        candidate_score = 0.0
        missing_metrics = []

        for metric, weight in metrics_weights.items():
            metric_key = metric.lower()
            baseline_metric = to_float(baseline.get(metric_key))
            candidate_metric = to_float(candidate.get(metric_key))

            if baseline_metric is None or candidate_metric is None:
                missing_metrics.append(metric)
            else:
                baseline_score += baseline_metric * weight
                candidate_score += candidate_metric * weight

        if missing_metrics:
            decision = "crash"
            explanation = f"missing metrics: {', '.join(missing_metrics)}, so the comparison is not reliable."
        elif candidate_score > baseline_score:
            decision = "keep"
            explanation = f"candidate weighted score improved over baseline."
        else:
            decision = "discard"
            explanation = f"candidate weighted score did not improve over baseline."

        delta = round(candidate_score - baseline_score, 6)

        return {
            "baseline_metric_name": "weighted_score",
            "baseline_metric": round(baseline_score, 6),
            "candidate_metric_name": "weighted_score",
            "candidate_metric": round(candidate_score, 6),
            "delta": delta,
            "decision": decision,
            "explanation": explanation,
        }
    else:
        # Single metric comparison (original behavior)
        metric_key = metric_name.lower()
        baseline_metric = to_float(baseline.get(metric_key))
        candidate_metric = to_float(candidate.get(metric_key))

        if baseline_metric is None or candidate_metric is None:
            decision = "crash"
            explanation = f"missing {metric_name} value, so the comparison is not reliable."
        elif candidate_metric > baseline_metric:
            decision = "keep"
            explanation = f"candidate {metric_name} improved over baseline."
        else:
            decision = "discard"
            explanation = f"candidate {metric_name} did not improve over baseline."

        delta = None
        if baseline_metric is not None and candidate_metric is not None:
            delta = round(candidate_metric - baseline_metric, 6)

        return {
            "baseline_metric_name": metric_name,
            "baseline_metric": baseline_metric,
            "candidate_metric_name": metric_name,
            "candidate_metric": candidate_metric,
            "delta": delta,
            "decision": decision,
            "explanation": explanation,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two RecClaw runs.")
    parser.add_argument("baseline", nargs="?", help="Baseline log path or JSON path/string")
    parser.add_argument("candidate", nargs="?", help="Candidate log path or JSON path/string")
    parser.add_argument("--csv", help="Load both runs from a CSV file")
    parser.add_argument("--baseline-row", type=int, help="1-based baseline data row index in CSV")
    parser.add_argument("--candidate-row", type=int, help="1-based candidate data row index in CSV")
    parser.add_argument("--baseline-id", help="Baseline run_id in CSV")
    parser.add_argument("--candidate-id", help="Candidate run_id in CSV")
    parser.add_argument("--metric", default="ndcg@10", help="Primary metric used for the decision rule")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    args = parser.parse_args()

    if args.csv:
        baseline = load_csv_result(args.csv, args.baseline_row, args.baseline_id)
        candidate = load_csv_result(args.csv, args.candidate_row, args.candidate_id)
    else:
        if not args.baseline or not args.candidate:
            parser.error("provide two inputs or use --csv with row ids")
        baseline = load_result_source(args.baseline)
        candidate = load_result_source(args.candidate)

    result = compare_results(baseline, candidate, args.metric)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=True))
    else:
        print(f"baseline {args.metric}: {result['baseline_metric']}")
        print(f"candidate {args.metric}: {result['candidate_metric']}")
        print(f"delta: {result['delta']}")
        print(f"decision: {result['decision']}")
        print(f"reason: {result['explanation']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
