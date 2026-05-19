#!/usr/bin/env python3
"""Build offline RecClaw candidate search-tree artifacts.

This script is evidence-only. It does not steer the agent loop directly and it
must not read RecClaw_LabLog runtime inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
DEFAULT_PROPOSALS = PROJECT_ROOT / "results" / "candidate_proposals.jsonl"
DEFAULT_MEMORY = PROJECT_ROOT / "results" / "agent_memory.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "results.csv"
DEFAULT_OUT_JSON = PROJECT_ROOT / "results" / "candidate_search_tree.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "results" / "candidate_search_tree.md"
DEFAULT_OUT_MMD = PROJECT_ROOT / "results" / "candidate_search_tree.mmd"


def reject_lablog_path(path: str | Path, *, allow_lablog: bool = False) -> None:
    text = str(path).replace("\\", "/").lower()
    if not allow_lablog and "recclaw_lablog" in text:
        raise ValueError(f"candidate search tree refuses LabLog path: {path}")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def load_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    result = row.get("result")
    if isinstance(result, dict):
        value = to_float(result.get(metric))
        if value is not None:
            return value
    return to_float(row.get(metric))


def infer_candidate_id_from_run_id(run_id: str) -> str:
    text = str(run_id or "").strip()
    if not text.startswith("candidate_"):
        return ""
    stem = text[len("candidate_") :]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return parts[0]
    return stem


def infer_action_type(row: dict[str, Any]) -> str:
    explicit = str(row.get("action_type") or "").strip()
    if explicit:
        return explicit
    text = json.dumps(row, ensure_ascii=False).lower()
    if "rerank" in text or "coverage" in text:
        return "posthoc_rerank"
    if "hard_negative" in text or "debiased" in text or "popularity_aware" in text:
        return "sampling_wrapper"
    if "norm" in text or "regularized" in text:
        return "regularization"
    if "residual" in text or "layer" in text or "embedding" in text:
        return "aggregation"
    if "loss" in text or "margin" in text:
        return "local_loss"
    return "parameter_tuning"


def blank_node(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "parent_candidate_id": "",
        "base_model": "unknown",
        "action_type": "",
        "parameter_overrides": {},
        "status": "",
        "decision_counts": {},
        "best_ndcg@10": None,
        "mean_ndcg@10": None,
        "std_ndcg@10": 0.0,
        "run_count": 0,
        "crash_count": 0,
        "seed_validation": {},
        "children": [],
        "failure_reason": "",
        "sources": [],
        "_metrics": [],
        "_reasons": [],
    }


def ensure_node(nodes: dict[str, dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    candidate_id = candidate_id or "unknown"
    if candidate_id not in nodes:
        nodes[candidate_id] = blank_node(candidate_id)
    return nodes[candidate_id]


def merge_scalar(node: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        node[key] = value


def add_source(node: dict[str, Any], source: str) -> None:
    sources = node.setdefault("sources", [])
    if source not in sources:
        sources.append(source)


def add_registry(nodes: dict[str, dict[str, Any]], registry: dict[str, Any]) -> None:
    for item in registry.get("candidates", []):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id:
            continue
        node = ensure_node(nodes, candidate_id)
        add_source(node, "registry")
        merge_scalar(node, "base_model", item.get("base_model"))
        merge_scalar(node, "action_type", infer_action_type(item))
        merge_scalar(node, "status", item.get("status"))


def add_proposals(nodes: dict[str, dict[str, Any]], proposals: list[dict[str, Any]]) -> None:
    for item in proposals:
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id:
            continue
        node = ensure_node(nodes, candidate_id)
        add_source(node, "proposal")
        merge_scalar(node, "parent_candidate_id", item.get("parent_candidate_id"))
        merge_scalar(node, "base_model", item.get("base_model"))
        merge_scalar(node, "action_type", infer_action_type(item))
        merge_scalar(node, "status", item.get("runnable_level"))
        params = item.get("parameter_overrides")
        if isinstance(params, dict):
            node["parameter_overrides"] = params


def add_memory(nodes: dict[str, dict[str, Any]], memory_rows: list[dict[str, Any]], metric: str) -> None:
    for item in memory_rows:
        if item.get("event"):
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id:
            continue
        node = ensure_node(nodes, candidate_id)
        add_source(node, "agent_memory")
        merge_scalar(node, "parent_candidate_id", item.get("parent_candidate_id"))
        merge_scalar(node, "base_model", item.get("base_model"))
        merge_scalar(node, "action_type", infer_action_type(item))
        merge_scalar(node, "status", item.get("status"))
        params = item.get("params")
        if isinstance(params, dict) and params:
            node["parameter_overrides"] = params
        value = metric_value(item, metric)
        if value is not None:
            node["_metrics"].append(value)
        decision = str(item.get("decision") or "")
        if decision:
            counts = node.setdefault("decision_counts", {})
            counts[decision] = int(counts.get(decision) or 0) + 1
        status = str(item.get("status") or "").lower()
        if status in {"crash", "failed", "error"}:
            node["crash_count"] = int(node.get("crash_count") or 0) + 1
        seed_validation = item.get("seed_validation")
        if isinstance(seed_validation, dict) and seed_validation:
            node["seed_validation"] = seed_validation
        reason = str(item.get("reason") or item.get("next_action") or "")
        if reason and decision in {"discard", "revise", "crash"}:
            node.setdefault("_reasons", []).append(reason)


def add_results(nodes: dict[str, dict[str, Any]], results_rows: list[dict[str, Any]], metric: str) -> None:
    for item in results_rows:
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not candidate_id:
            candidate_id = infer_candidate_id_from_run_id(str(item.get("run_id") or ""))
        if not candidate_id:
            continue
        node = ensure_node(nodes, candidate_id)
        add_source(node, "results.csv")
        merge_scalar(node, "base_model", item.get("model"))
        merge_scalar(node, "status", item.get("status"))
        value = metric_value(item, metric)
        if value is not None:
            node["_metrics"].append(value)
        status = str(item.get("status") or "").lower()
        if status in {"crash", "failed", "error"}:
            node["crash_count"] = int(node.get("crash_count") or 0) + 1


def memory_run_ids(memory_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("run_id") or "").strip()
        for item in memory_rows
        if str(item.get("run_id") or "").strip()
    }


def results_not_in_memory(
    results_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen_run_ids = memory_run_ids(memory_rows)
    filtered = []
    for item in results_rows:
        run_id = str(item.get("run_id") or "").strip()
        if run_id and run_id in seen_run_ids:
            continue
        filtered.append(item)
    return filtered


def finalize_tree(nodes: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    metric_key = metric.replace("@", "_at_")
    missing_parent_ids = sorted(
        {
            str(node.get("parent_candidate_id") or "")
            for node in nodes.values()
            if str(node.get("parent_candidate_id") or "") and str(node.get("parent_candidate_id") or "") not in nodes
        }
    )
    for parent_id in missing_parent_ids:
        ensure_node(nodes, parent_id)
    for node in nodes.values():
        parent_id = str(node.get("parent_candidate_id") or "")
        if parent_id and parent_id in nodes:
            parent = nodes[parent_id]
            children = parent.setdefault("children", [])
            candidate_id = str(node.get("candidate_id") or "")
            if candidate_id and candidate_id not in children:
                children.append(candidate_id)
    public_nodes = []
    for candidate_id, node in sorted(nodes.items()):
        values = [float(value) for value in node.get("_metrics", [])]
        best = max(values) if values else None
        avg = mean(values)
        node[f"best_{metric_key}"] = round(best, 6) if best is not None else None
        node[f"mean_{metric_key}"] = round(avg, 6) if avg is not None else None
        node[f"std_{metric_key}"] = round(std(values), 6)
        if metric == "ndcg@10":
            node["best_ndcg@10"] = node.get(f"best_{metric_key}")
            node["mean_ndcg@10"] = node.get(f"mean_{metric_key}")
            node["std_ndcg@10"] = node.get(f"std_{metric_key}")
        node["run_count"] = len(values)
        reasons = node.get("_reasons") if isinstance(node.get("_reasons"), list) else []
        node["failure_reason"] = str(reasons[-1]) if reasons else ""
        node["children"] = sorted(node.get("children") or [])
        node["child_count"] = len(node["children"])
        public = {key: value for key, value in node.items() if not key.startswith("_")}
        public_nodes.append(public)
    roots = [
        node["candidate_id"]
        for node in public_nodes
        if not node.get("parent_candidate_id") or node.get("parent_candidate_id") not in nodes
    ]
    edges = [
        {"parent": node["candidate_id"], "child": child}
        for node in public_nodes
        for child in node.get("children", [])
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metric": metric,
        "metric_key": metric_key,
        "roots": sorted(roots),
        "edges": edges,
        "nodes": public_nodes,
        "summary": {
            "node_count": len(public_nodes),
            "edge_count": len(edges),
            "run_count": sum(int(node.get("run_count") or 0) for node in public_nodes),
            "crash_count": sum(int(node.get("crash_count") or 0) for node in public_nodes),
        },
    }


def label_for(node: dict[str, Any], metric: str) -> str:
    candidate_id = str(node.get("candidate_id") or "")
    metric_key = metric.replace("@", "_at_")
    best = node.get(f"best_{metric_key}")
    if best is None and metric == "ndcg@10":
        best = node.get("best_ndcg@10")
    decision_counts = node.get("decision_counts") if isinstance(node.get("decision_counts"), dict) else {}
    decision_text = ",".join(f"{key}:{value}" for key, value in sorted(decision_counts.items()))
    best_text = f" best={best}" if best is not None else ""
    decision_suffix = f" {decision_text}" if decision_text else ""
    return f"{candidate_id}{best_text}{decision_suffix}"


def build_markdown(tree: dict[str, Any]) -> str:
    nodes = {str(node.get("candidate_id")): node for node in tree.get("nodes", []) if isinstance(node, dict)}
    lines = [
        "# RecClaw Candidate Search Tree",
        "",
        f"Metric: {tree.get('metric')}",
        f"Nodes: {tree.get('summary', {}).get('node_count', 0)}",
        f"Edges: {tree.get('summary', {}).get('edge_count', 0)}",
        "",
    ]

    def walk(candidate_id: str, depth: int = 0, seen: set[str] | None = None) -> None:
        seen = seen or set()
        if candidate_id in seen:
            return
        seen.add(candidate_id)
        node = nodes.get(candidate_id, {"candidate_id": candidate_id})
        indent = "  " * depth
        lines.append(f"{indent}- {label_for(node, str(tree.get('metric') or 'ndcg@10'))}")
        for child in node.get("children", []) or []:
            walk(str(child), depth + 1, seen)

    for root in tree.get("roots", []):
        walk(str(root))
    lines.append("")
    return "\n".join(lines)


def mermaid_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text or "unknown")


def build_mermaid(tree: dict[str, Any]) -> str:
    nodes = {str(node.get("candidate_id")): node for node in tree.get("nodes", []) if isinstance(node, dict)}
    lines = ["graph TD"]
    for edge in tree.get("edges", []):
        parent = str(edge.get("parent") or "")
        child = str(edge.get("child") or "")
        if not parent or not child:
            continue
        parent_node = nodes.get(parent, {"candidate_id": parent})
        child_node = nodes.get(child, {"candidate_id": child})
        lines.append(
            f'  {mermaid_id(parent)}["{label_for(parent_node, str(tree.get("metric") or "ndcg@10"))}"]'
            f' --> {mermaid_id(child)}["{label_for(child_node, str(tree.get("metric") or "ndcg@10"))}"]'
        )
    if len(lines) == 1:
        for root in tree.get("roots", []):
            root_text = str(root)
            node = nodes.get(root_text, {"candidate_id": root_text})
            lines.append(f'  {mermaid_id(root_text)}["{label_for(node, str(tree.get("metric") or "ndcg@10"))}"]')
    return "\n".join(lines) + "\n"


def build_tree(
    *,
    registry: dict[str, Any],
    proposals: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    results_rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    add_registry(nodes, registry)
    add_proposals(nodes, proposals)
    add_memory(nodes, memory_rows, metric)
    add_results(nodes, results_not_in_memory(results_rows, memory_rows), metric)
    return finalize_tree(nodes, metric)


def write_outputs(tree: dict[str, Any], out_json: Path, out_md: Path, out_mmd: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_mmd.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(tree, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(tree), encoding="utf-8")
    out_mmd.write_text(build_mermaid(tree), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline RecClaw candidate search-tree artifacts.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--proposals", default=str(DEFAULT_PROPOSALS))
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--out-mmd", default=str(DEFAULT_OUT_MMD))
    parser.add_argument(
        "--allow-lablog-input",
        action="store_true",
        help="Report-only escape hatch. Do not use for agent runtime tree artifacts.",
    )
    args = parser.parse_args()

    for value in (
        args.registry,
        args.proposals,
        args.memory,
        args.results,
        args.out_json,
        args.out_md,
        args.out_mmd,
    ):
        reject_lablog_path(value, allow_lablog=bool(args.allow_lablog_input))

    tree = build_tree(
        registry=load_yaml(Path(args.registry)),
        proposals=load_jsonl(Path(args.proposals)),
        memory_rows=load_jsonl(Path(args.memory)),
        results_rows=load_results_csv(Path(args.results)),
        metric=str(args.metric),
    )
    write_outputs(tree, Path(args.out_json), Path(args.out_md), Path(args.out_mmd))
    print(json.dumps(tree.get("summary", {}), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
