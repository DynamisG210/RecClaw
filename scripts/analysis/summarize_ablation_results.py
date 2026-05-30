#!/usr/bin/env python3
"""Summarize RecClaw ablation results by variant.

The formal results CSV intentionally stays close to run_candidate output and may
not include one-off analysis tags such as ablation_variant. This script joins
each result row back to the generated override YAML named in config_change.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _parse_simple_yaml_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        lhs, rhs = stripped.split(":", 1)
        if lhs.strip() == key:
            value = rhs.strip().strip("'\"")
            return value or None
    return None


def _override_path(row: dict[str, str], override_dir: Path) -> Path | None:
    config_change = row.get("config_change", "")
    for part in config_change.split("+"):
        part = part.strip()
        if part.startswith("candidate_") and part.endswith(".yaml"):
            return override_dir / part
    run_id = row.get("run_id", "").strip()
    if run_id:
        candidate = override_dir / f"{run_id}.yaml"
        if candidate.exists():
            return candidate
    return None


def _variant(row: dict[str, str], override_dir: Path) -> str:
    notes = row.get("notes", "") or ""
    for part in notes.split("|"):
        if "ablation_variant=" in part:
            return part.split("=", 1)[1].strip() or "unknown"
    path = _override_path(row, override_dir)
    if path:
        value = _parse_simple_yaml_value(path, "ablation_variant")
        if value:
            return value
    return "unknown"


def _metric(row: dict[str, str], metric: str) -> float | None:
    for key in (metric, metric.lower(), metric.upper()):
        if key in row and row[key] not in ("", None):
            try:
                return float(row[key])
            except ValueError:
                return None
    return None


def summarize(results_csv: Path, override_dir: Path, metric: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(results_csv.open(newline="", encoding="utf-8")))
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _metric(row, metric)
        if value is None:
            continue
        grouped.setdefault(_variant(row, override_dir), []).append(value)

    summary = []
    for variant, values in sorted(grouped.items()):
        summary.append(
            {
                "variant": variant,
                "n": len(values),
                f"mean_{metric}": sum(values) / len(values),
                f"std_pop_{metric}": statistics.pstdev(values) if len(values) > 1 else 0.0,
                f"min_{metric}": min(values),
                f"max_{metric}": max(values),
                "values": values,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--override-dir", required=True, type=Path)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    summary = summarize(args.results_csv, args.override_dir, args.metric)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "variant",
            "n",
            f"mean_{args.metric}",
            f"std_pop_{args.metric}",
            f"min_{args.metric}",
            f"max_{args.metric}",
            "values",
        ]
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary:
                out = dict(row)
                out["values"] = json.dumps(out["values"])
                writer.writerow(out)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
