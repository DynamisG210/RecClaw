#!/usr/bin/env python3
"""Parse RecBole logs and optionally append a row to results.csv."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "run_id",
    "model",
    "dataset",
    "config_change",
    "ndcg@10",
    "recall@10",
    "mrr@10",
    "hit@10",
    "precision@10",
    "itemcoverage@10",
    "valid_metric",
    "run_time",
    "status",
    "log_path",
    "notes",
]

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TIMESTAMP_RE = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2} \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2})\s+"
)
MODEL_CLI_RE = re.compile(r"--model(?:=|\s+)(?P<value>[A-Za-z0-9_.+-]+)")
DATASET_CLI_RE = re.compile(r"--dataset(?:=|\s+)(?P<value>[A-Za-z0-9_.-]+)")
MODEL_BLOCK_RE = re.compile(r"INFO\s+(?P<value>[A-Za-z0-9_]+)\($")
CONFIG_VALUE_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_@.+-]+)\s*=\s*(?P<value>.+?)\s*$")
INLINE_METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_@.+-]+)\s*:\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
BEST_VALID_LINE_RE = re.compile(r"best valid\s*:\s*(?P<blob>.+)$", re.IGNORECASE)
TEST_RESULT_LINE_RE = re.compile(r"test result\s*:\s*(?P<blob>.+)$", re.IGNORECASE)


def coerce_float(value: Any) -> float | None:
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


def parse_metric_blob(blob: str) -> dict[str, float]:
    text = blob.strip()
    metrics: dict[str, float] = {}

    if text.startswith("OrderedDict(") and text.endswith(")"):
        inner = text[len("OrderedDict(") : -1]
        try:
            pairs = ast.literal_eval(inner)
            if isinstance(pairs, list):
                for key, value in pairs:
                    number = coerce_float(value)
                    if number is not None:
                        metrics[str(key).lower()] = number
                if metrics:
                    return metrics
        except (SyntaxError, ValueError):
            pass

    if text.startswith("{") and text.endswith("}"):
        try:
            raw_dict = ast.literal_eval(text)
            if isinstance(raw_dict, dict):
                for key, value in raw_dict.items():
                    number = coerce_float(value)
                    if number is not None:
                        metrics[str(key).lower()] = number
                if metrics:
                    return metrics
        except (SyntaxError, ValueError):
            pass

    for match in INLINE_METRIC_RE.finditer(text):
        number = coerce_float(match.group("value"))
        if number is not None:
            metrics[match.group("key").lower()] = number
    return metrics


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("ts"), "%a %d %b %Y %H:%M:%S")
    except ValueError:
        return None


def parse_recbole_log(log_path: str | Path) -> dict[str, Any]:
    path = Path(log_path)
    warnings: list[str] = []
    model = ""
    dataset = ""
    valid_metric = ""
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    test_metrics: dict[str, float] = {}
    best_valid_metrics: dict[str, float] = {}
    last_valid_metrics: dict[str, float] = {}
    waiting_for_valid_metrics = False
    saw_error_marker = False

    try:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {
            "model": "",
            "dataset": "",
            "ndcg@10": "",
            "recall@10": "",
            "mrr@10": "",
            "hit@10": "",
            "precision@10": "",
            "itemcoverage@10": "",
            "valid_metric": "",
            "run_time": "",
            "status": "crash",
            "metric_source": "",
            "warnings": [f"failed to read log file: {exc}"],
        }

    for raw_line in raw_lines:
        line = ANSI_ESCAPE_RE.sub("", raw_line)
        ts = parse_timestamp(line)
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        if not model:
            cli_model = MODEL_CLI_RE.search(line)
            if cli_model:
                model = cli_model.group("value")
        if not dataset:
            cli_dataset = DATASET_CLI_RE.search(line)
            if cli_dataset:
                dataset = cli_dataset.group("value")

        config_match = CONFIG_VALUE_RE.match(line)
        if config_match:
            key = config_match.group("key")
            value = config_match.group("value")
            if key == "model" and not model:
                model = value
            elif key == "dataset" and not dataset:
                dataset = value
            elif key == "valid_metric" and not valid_metric:
                valid_metric = value

        if not model:
            model_block = MODEL_BLOCK_RE.search(line)
            if model_block:
                model = model_block.group("value")

        if "valid result:" in line.lower():
            waiting_for_valid_metrics = True
            continue

        if waiting_for_valid_metrics:
            candidate_metrics = parse_metric_blob(line)
            if candidate_metrics:
                last_valid_metrics = candidate_metrics
                waiting_for_valid_metrics = False

        best_valid_match = BEST_VALID_LINE_RE.search(line)
        if best_valid_match:
            best_valid_metrics = parse_metric_blob(best_valid_match.group("blob"))

        test_result_match = TEST_RESULT_LINE_RE.search(line)
        if test_result_match:
            test_metrics = parse_metric_blob(test_result_match.group("blob"))

        lowered = line.lower()
        if "traceback" in lowered or "error" in lowered or "exception" in lowered:
            saw_error_marker = True

    metric_source = ""
    chosen_metrics: dict[str, float] = {}
    if test_metrics:
        metric_source = "test_result"
        chosen_metrics = test_metrics
    elif best_valid_metrics:
        metric_source = "best_valid"
        chosen_metrics = best_valid_metrics
        warnings.append("test result not found; fell back to best valid metrics")
    elif last_valid_metrics:
        metric_source = "last_valid"
        chosen_metrics = last_valid_metrics
        warnings.append("test result not found; fell back to last valid metrics")
    else:
        warnings.append("no metric block could be parsed from the log")

    run_time_seconds: float | str = ""
    if first_ts is not None and last_ts is not None:
        run_time_seconds = round((last_ts - first_ts).total_seconds(), 2)
    else:
        warnings.append("run time could not be inferred from log timestamps")

    if not model:
        warnings.append("model could not be inferred from the log")
    if not dataset:
        warnings.append("dataset could not be inferred from the log")
    if not valid_metric:
        warnings.append("valid_metric could not be inferred from the log")

    if test_metrics:
        status = "success"
    elif saw_error_marker:
        status = "crash"
    else:
        status = "incomplete"

    return {
        "model": model,
        "dataset": dataset,
        "ndcg@10": chosen_metrics.get("ndcg@10", ""),
        "recall@10": chosen_metrics.get("recall@10", ""),
        "mrr@10": chosen_metrics.get("mrr@10", ""),
        "hit@10": chosen_metrics.get("hit@10", ""),
        "precision@10": chosen_metrics.get("precision@10", ""),
        "itemcoverage@10": chosen_metrics.get("itemcoverage@10", ""),
        "valid_metric": valid_metric,
        "run_time": run_time_seconds,
        "status": status,
        "metric_source": metric_source,
        "warnings": warnings,
    }


def normalize_result_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key in (
        "ndcg@10",
        "recall@10",
        "mrr@10",
        "hit@10",
        "precision@10",
        "itemcoverage@10",
        "run_time",
    ):
        value = coerce_float(normalized.get(key))
        normalized[key] = value if value is not None else normalized.get(key, "")
    normalized.setdefault("status", "")
    normalized.setdefault("model", "")
    normalized.setdefault("dataset", "")
    normalized.setdefault("valid_metric", "")
    normalized.setdefault("warnings", [])
    return normalized


def load_result_source(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    if path.exists():
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return normalize_result_record(json.load(handle))
        return normalize_result_record(parse_recbole_log(path))

    text = str(source).strip()
    if text.startswith("{") and text.endswith("}"):
        return normalize_result_record(json.loads(text))

    raise FileNotFoundError(f"result source not found or not supported: {source}")


def merge_notes(notes: str, warnings: list[str]) -> str:
    warning_text = "; ".join(warnings)
    if notes and warning_text:
        return f"{notes} | warnings: {warning_text}"
    if notes:
        return notes
    return f"warnings: {warning_text}" if warning_text else ""


def ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_row(csv_path: str | Path, row: dict[str, Any]) -> None:
    path = Path(csv_path)
    ensure_csv(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def build_csv_row(
    parsed: dict[str, Any],
    run_id: str,
    log_path: str,
    config_change: str,
    notes: str,
    status_override: str | None,
) -> dict[str, Any]:
    status = status_override or parsed.get("status", "")
    warnings = parsed.get("warnings", [])
    return {
        "run_id": run_id,
        "model": parsed.get("model", ""),
        "dataset": parsed.get("dataset", ""),
        "config_change": config_change,
        "ndcg@10": parsed.get("ndcg@10", ""),
        "recall@10": parsed.get("recall@10", ""),
        "mrr@10": parsed.get("mrr@10", ""),
        "hit@10": parsed.get("hit@10", ""),
        "precision@10": parsed.get("precision@10", ""),
        "itemcoverage@10": parsed.get("itemcoverage@10", ""),
        "valid_metric": parsed.get("valid_metric", ""),
        "run_time": parsed.get("run_time", ""),
        "status": status,
        "log_path": log_path,
        "notes": merge_notes(notes, warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a RecBole log file.")
    parser.add_argument("log_path", help="Path to a RecBole log file")
    parser.add_argument("--append-csv", help="Append the parsed result to a CSV file")
    parser.add_argument("--run-id", help="Run id to store in CSV")
    parser.add_argument(
        "--config-change",
        default="",
        help="Short description of the config change stored in CSV",
    )
    parser.add_argument("--notes", default="", help="Free-form note stored in CSV")
    parser.add_argument(
        "--log-path",
        dest="log_path_override",
        help="Explicit log path value to store in CSV",
    )
    parser.add_argument(
        "--status-override",
        choices=["success", "crash", "incomplete"],
        help="Force the stored status value",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    parsed = normalize_result_record(parse_recbole_log(log_path))
    if args.status_override:
        parsed["status"] = args.status_override

    output = {
        "run_id": args.run_id or log_path.stem,
        "model": parsed.get("model", ""),
        "dataset": parsed.get("dataset", ""),
        "ndcg@10": parsed.get("ndcg@10", ""),
        "recall@10": parsed.get("recall@10", ""),
        "mrr@10": parsed.get("mrr@10", ""),
        "hit@10": parsed.get("hit@10", ""),
        "precision@10": parsed.get("precision@10", ""),
        "itemcoverage@10": parsed.get("itemcoverage@10", ""),
        "valid_metric": parsed.get("valid_metric", ""),
        "run_time": parsed.get("run_time", ""),
        "status": parsed.get("status", ""),
        "metric_source": parsed.get("metric_source", ""),
        "log_path": str(args.log_path_override or log_path),
        "warnings": parsed.get("warnings", []),
    }

    json.dump(output, sys.stdout, indent=2, ensure_ascii=True)
    sys.stdout.write("\n")

    if args.append_csv:
        row = build_csv_row(
            parsed=parsed,
            run_id=output["run_id"],
            log_path=output["log_path"],
            config_change=args.config_change,
            notes=args.notes,
            status_override=args.status_override,
        )
        append_row(args.append_csv, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
