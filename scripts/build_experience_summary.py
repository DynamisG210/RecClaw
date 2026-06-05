#!/usr/bin/env python3
"""Build a short runtime experience summary from current RecClaw results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from .action_space import load_action_space
except ImportError:
    from action_space import load_action_space

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "results" / "agent_memory.jsonl"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
DEFAULT_ACTION_SPACE = PROJECT_ROOT / "configs" / "action_space.yaml"
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "reflection_policy.yaml"
DEFAULT_SEARCH_POLICY = PROJECT_ROOT / "configs" / "search_policy.yaml"
DEFAULT_TREE_PATH = PROJECT_ROOT / "results" / "candidate_search_tree.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "results" / "experience_summary.md"
DEFAULT_OUT_JSON = PROJECT_ROOT / "results" / "experience_summary.json"
DEFAULT_OUT_JSONL = PROJECT_ROOT / "results" / "reflection_memory.jsonl"


def reject_lablog_path(path: str | Path, *, allow_lablog: bool = False) -> None:
    text = str(path).replace("\\", "/").lower()
    if not allow_lablog and "recclaw_lablog" in text:
        raise ValueError(f"runtime experience builder refuses LabLog path: {path}")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def load_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def infer_candidate_id_from_run_id(run_id: str) -> str:
    text = str(run_id or "").strip()
    if not text.startswith("candidate_"):
        return ""
    stem = text[len("candidate_") :]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[-2].isdigit() and parts[-1].isdigit():
        return parts[0]
    return stem


def result_rows_to_memory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            candidate_id = infer_candidate_id_from_run_id(str(row.get("run_id") or ""))
        payload: dict[str, Any] = {
            "source": "results.csv",
            "candidate_id": candidate_id or "unknown",
            "run_id": row.get("run_id", ""),
            "model": row.get("model", ""),
            "status": row.get("status", ""),
            "result": row,
            "params": {},
        }
        converted.append(payload)
    return converted


def combine_evidence_rows(memory_rows: list[dict[str, Any]], results_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memory_by_run_id = {
        str(row.get("run_id") or "").strip(): row
        for row in memory_rows
        if str(row.get("run_id") or "").strip()
    }
    combined = list(memory_rows)
    for row in result_rows_to_memory_rows(results_rows):
        run_id = str(row.get("run_id") or "").strip()
        if run_id and run_id in memory_by_run_id:
            target = memory_by_run_id[run_id]
            target_result = target.get("result")
            csv_result = row.get("result")
            if isinstance(csv_result, dict):
                if not isinstance(target_result, dict):
                    target["result"] = dict(csv_result)
                else:
                    for key, value in csv_result.items():
                        if target_result.get(key) in (None, "") and value not in (None, ""):
                            target_result[key] = value
            continue
        combined.append(row)
    return combined


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((item - mean) ** 2 for item in values) / (len(values) - 1))


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    return std(values) / mean


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator if denominator else 0.0


def convergence_trend(rows: list[dict[str, Any]], metric: str, policy: dict[str, Any]) -> dict[str, Any]:
    trend_policy = policy.get("convergence_trend") if isinstance(policy.get("convergence_trend"), dict) else {}
    if not bool(trend_policy.get("enable", False)):
        return {"trend": "disabled", "slope": 0.0, "cv": 0.0, "high_variance": False}
    min_runs = int(trend_policy.get("min_runs_for_trend", 3))
    improving = float(trend_policy.get("improving_slope_threshold", 0.001))
    degrading = float(trend_policy.get("degrading_slope_threshold", -0.001))
    high_variance = float(trend_policy.get("high_variance_cv_threshold", 0.10))
    sorted_rows = sorted(rows, key=lambda row: int(row.get("round_id") or 0))
    values = [metric_value(row, metric) for row in sorted_rows]
    values = [value for value in values if value is not None]
    if len(values) < min_runs:
        return {"trend": "insufficient_data", "slope": 0.0, "cv": 0.0, "high_variance": False}
    slope = linear_slope(values)
    cv = coefficient_of_variation(values)
    if slope >= improving:
        trend = "improving"
    elif slope <= degrading:
        trend = "degrading"
    else:
        trend = "stable"
    return {
        "trend": trend,
        "slope": round(slope, 6),
        "cv": round(cv, 6),
        "high_variance": cv >= high_variance,
    }


def domain_prior_warnings(
    *,
    base_model: str,
    params: dict[str, Any],
    policy: dict[str, Any],
) -> list[str]:
    priors = policy.get("domain_priors") if isinstance(policy.get("domain_priors"), dict) else {}
    warnings: list[str] = []
    model_priors = priors.get(base_model) if isinstance(priors, dict) else {}
    model_priors = model_priors if isinstance(model_priors, dict) else {}
    n_layers = to_float(params.get("n_layers"))
    if base_model == "LightGCN":
        optimal = model_priors.get("optimal_layers_range")
        if n_layers is not None and isinstance(optimal, list) and len(optimal) == 2:
            upper = to_float(optimal[1])
            if upper is not None and n_layers > upper:
                warnings.append(f"n_layers={n_layers:g} exceeds LightGCN typical range {optimal}; over-smoothing risk")
        threshold = to_float(model_priors.get("over_smoothing_threshold"))
        if n_layers is not None and threshold is not None and n_layers >= threshold:
            warnings.append(f"n_layers={n_layers:g} reaches LightGCN over-smoothing threshold {threshold:g}")
        edge_dropout = to_float(params.get("edge_dropout"))
        if edge_dropout is not None and edge_dropout > 0.2:
            warnings.append("edge_dropout exceeds the current action-space maximum used for stable graph regularization")
    if base_model == "BPR":
        if n_layers is not None:
            warnings.append("n_layers is a graph propagation parameter and should not be used with BPR")
        margin = to_float(params.get("margin"))
        if margin is not None and margin >= 0.5:
            warnings.append("margin is at the high end for BPR; monitor instability and collapse")
        hard_ratio = to_float(params.get("hard_negative_ratio"))
        if hard_ratio is not None and hard_ratio > 0.5:
            warnings.append("hard_negative_ratio above 0.5 can increase BPR training variance")
    general = priors.get("general") if isinstance(priors, dict) else {}
    if isinstance(general, dict):
        lambda_coverage = to_float(params.get("lambda_coverage"))
        if lambda_coverage is not None:
            warnings.append(str(general.get("coverage_note") or "Treat coverage changes as tie-breakers under NDCG guard."))
    return warnings


def infer_base_model(record: dict[str, Any]) -> str:
    for value in (
        record.get("base_model"),
        (record.get("result") or {}).get("model") if isinstance(record.get("result"), dict) else "",
        record.get("model"),
        record.get("candidate_id"),
        record.get("parent_candidate_id"),
    ):
        text = str(value or "")
        if "LightGCN" in text or "lightgcn" in text:
            return "LightGCN"
        if "BPR" in text or "bpr" in text:
            return "BPR"
    return "unknown"


def candidate_family(record: dict[str, Any]) -> str:
    return str(record.get("parent_candidate_id") or record.get("candidate_id") or "unknown")


def canonical_signature(raw: Any) -> str:
    return str(raw or "").strip()


def metric_value(record: dict[str, Any], metric: str) -> float | None:
    result = record.get("result")
    if isinstance(result, dict):
        value = to_float(result.get(metric))
        if value is not None:
            return value
    return to_float(record.get(metric))


def action_type_for_family(family: str, action_space: dict[str, Any]) -> str:
    lowered = family.lower()
    if "hard_negative" in lowered or "debiased" in lowered:
        return "sampling_wrapper"
    if "norm" in lowered or "regularized" in lowered:
        return "regularization"
    if "residual" in lowered or "layer" in lowered or "embedding" in lowered:
        return "aggregation"
    if "rerank" in lowered or "coverage" in lowered:
        return "posthoc_rerank"
    return "parameter_tuning"


def action_type_for_record(record: dict[str, Any], family: str, action_space: dict[str, Any]) -> str:
    explicit = str(record.get("action_type") or "").strip()
    if explicit:
        return explicit
    return action_type_for_family(family, action_space)


COMPOSITION_TOKENS = (
    "debias",
    "pareto",
    "contrastive",
    "coverage",
    "popularity",
    "hard_negative",
    "tail",
    "norm",
    "residual",
    "rank",
    "edge_dropout",
)


def composition_tokens(record: dict[str, Any]) -> list[str]:
    text = json.dumps(record, ensure_ascii=False).lower()
    return [token for token in COMPOSITION_TOKENS if token in text]


def discouraged_compositions(search_policy: dict[str, Any] | None) -> list[list[str]]:
    if not isinstance(search_policy, dict):
        return []
    rules = search_policy.get("composition_rules")
    if not isinstance(rules, dict):
        return []
    raw = rules.get("discourage_combinations")
    if not isinstance(raw, list):
        return []
    combos: list[list[str]] = []
    for item in raw:
        if isinstance(item, list):
            combo = [str(token).lower() for token in item if str(token).strip()]
            if combo:
                combos.append(combo)
    return combos


def failed_composition_names(
    record: dict[str, Any],
    *,
    search_policy: dict[str, Any] | None,
) -> list[str]:
    tokens = set(composition_tokens(record))
    matched = []
    for combo in discouraged_compositions(search_policy):
        if all(token in tokens for token in combo):
            matched.append(" + ".join(combo))
    if matched:
        return matched
    if len(tokens) >= 3:
        return [" + ".join(sorted(tokens))]
    return []


def is_failed_record(record: dict[str, Any], metric: str, collapse_ndcg: float) -> bool:
    status = str(record.get("status") or "").lower()
    decision = str(record.get("decision") or "").lower()
    value = metric_value(record, metric)
    if status in {"crash", "failed", "error"}:
        return True
    if decision in {"discard", "crash"}:
        return True
    return value is not None and value < collapse_ndcg


def family_matches(candidate_family: str, target_family: str) -> bool:
    candidate_family = str(candidate_family or "")
    target_family = str(target_family or "")
    return (
        candidate_family == target_family
        or candidate_family.startswith(f"{target_family}_")
        or target_family.startswith(f"{candidate_family}_")
    )


def coarse_family_key(family: str) -> str:
    family = str(family or "")
    known_prefixes = (
        "cand_lightgcn_shallow_alignment_rankaware_gate",
        "cand_lightgcn_shallow_rankaware_lastlayeralign",
        "cand_lightgcn_edge_dropout_residual_norm",
        "cand_lightgcn_residual_norm_constrained",
        "cand_bpr_hard_negative_margin",
        "cand_lightgcn_shallow_layers",
    )
    for prefix in known_prefixes:
        if family == prefix or family.startswith(f"{prefix}_"):
            return prefix
    parts = family.split("_")
    return "_".join(parts[:4]) if len(parts) >= 4 else family


def algorithm_policy_defaults(policy: dict[str, Any], search_policy: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    reflection_algorithm = policy.get("algorithm_first") if isinstance(policy.get("algorithm_first"), dict) else {}
    search_algorithm = (
        search_policy.get("algorithm_first")
        if isinstance(search_policy, dict) and isinstance(search_policy.get("algorithm_first"), dict)
        else {}
    )
    plateau_policy = reflection_algorithm.get("plateau_policy")
    plateau_policy = plateau_policy if isinstance(plateau_policy, dict) else {}
    anti_plateau = search_algorithm.get("anti_plateau")
    anti_plateau = anti_plateau if isinstance(anti_plateau, dict) else {}
    merged = {**plateau_policy, **anti_plateau}
    anchor_defs = reflection_algorithm.get("anchor_families")
    anchors = [str(item) for item in anchor_defs] if isinstance(anchor_defs, dict) else []
    search_anchors = search_algorithm.get("anchor_families")
    if isinstance(search_anchors, list):
        anchors.extend(str(item) for item in search_anchors if str(item))
    forced = merged.get("forced_exploration_families")
    if isinstance(forced, list):
        anchors = [str(item) for item in forced if str(item)] + anchors
    deduped: list[str] = []
    for item in anchors:
        if item and item not in deduped:
            deduped.append(item)
    return merged, deduped


def detect_plateau(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    policy: dict[str, Any],
    search_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    plateau_policy, forced_defaults = algorithm_policy_defaults(policy, search_policy)
    enabled = bool(plateau_policy.get("enabled", True))
    window = max(1, int(plateau_policy.get("window_metric_rows", 30)))
    overuse_window = max(1, int(plateau_policy.get("family_overuse_window", 20)))
    max_same_family = max(1, int(plateau_policy.get("max_same_family_trials", 8)))
    min_improvement = max(0.0, float(plateau_policy.get("min_global_improvement", 0.0005)))
    weak_family_ceiling = max(0.0, float(plateau_policy.get("weak_family_ceiling", 0.280)))
    trial_rows = [
        row
        for row in rows
        if not row.get("event") and "decision" in row
    ]
    source_rows = trial_rows if trial_rows else rows
    metric_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        if row.get("event"):
            continue
        status = str(row.get("status") or "").lower()
        if status and status not in {"success", "ok"}:
            continue
        value = metric_value(row, metric)
        if value is None:
            continue
        metric_rows.append(
            {
                "index": index,
                "round_id": int(row.get("round_id") or index + 1),
                "family": candidate_family(row),
                "candidate_id": str(row.get("candidate_id") or ""),
                "value": value,
            }
        )

    default_response = {
        "enabled": enabled,
        "plateau_detected": False,
        "metric_rows": len(metric_rows),
        "window_metric_rows": window,
        "local_repair_capped_families": [],
        "forced_exploration_families": forced_defaults,
        "recent_family_counts": {},
        "forced_algorithm_tasks": list(plateau_policy.get("forced_algorithm_tasks") or []),
    }
    if not enabled or len(metric_rows) <= window:
        return default_response

    values = [float(row["value"]) for row in metric_rows]
    best_value = max(values)
    best_index = values.index(best_value)
    best_row = metric_rows[best_index]
    prior_best = max(values[:-window])
    recent_rows = metric_rows[-window:]
    recent_best = max(float(row["value"]) for row in recent_rows)
    recent_improvement = recent_best - prior_best
    rows_since_best = len(metric_rows) - best_index - 1

    recent_family_counts: dict[str, int] = defaultdict(int)
    recent_cluster_counts: dict[str, int] = defaultdict(int)
    for row in metric_rows[-overuse_window:]:
        family = str(row["family"])
        recent_family_counts[family] += 1
        recent_cluster_counts[coarse_family_key(family)] += 1

    capped_families: list[str] = []
    for family, count in sorted(recent_cluster_counts.items(), key=lambda item: (-item[1], item[0])):
        overused = count >= max_same_family
        stale_global_best = rows_since_best >= window
        weak_recent_frontier = recent_best <= weak_family_ceiling
        is_forced_anchor = any(family_matches(family, anchor) for anchor in forced_defaults)
        if overused and stale_global_best and (weak_recent_frontier or not is_forced_anchor):
            capped_families.append(family)

    plateau_detected = (
        rows_since_best >= window
        and recent_improvement < min_improvement
    ) or bool(capped_families)
    if plateau_detected and rows_since_best >= window:
        stale_best_family = coarse_family_key(str(best_row.get("family") or ""))
        stale_best_is_anchor = any(family_matches(stale_best_family, anchor) for anchor in forced_defaults)
        if stale_best_family and not stale_best_is_anchor and stale_best_family not in capped_families:
            capped_families.append(stale_best_family)
    forced = [
        family
        for family in forced_defaults
        if not any(family_matches(family, capped) for capped in capped_families)
    ] or forced_defaults
    return {
        **default_response,
        "plateau_detected": bool(plateau_detected),
        "rows_since_best": rows_since_best,
        "best_metric": round(best_value, 6),
        "best_round_id": best_row.get("round_id"),
        "best_family": best_row.get("family"),
        "recent_best": round(recent_best, 6),
        "recent_improvement": round(recent_improvement, 6),
        "local_repair_capped_families": capped_families[:10],
        "forced_exploration_families": forced,
        "recent_family_counts": dict(sorted(recent_family_counts.items(), key=lambda item: (-item[1], item[0]))[:10]),
        "recent_family_clusters": dict(sorted(recent_cluster_counts.items(), key=lambda item: (-item[1], item[0]))[:10]),
    }


def summarize_memory(
    memory_rows: list[dict[str, Any]],
    *,
    action_space: dict[str, Any],
    policy: dict[str, Any],
    search_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric = str(policy.get("metric") or "ndcg@10")
    thresholds = policy.get("thresholds") if isinstance(policy.get("thresholds"), dict) else {}
    collapse_ndcg = float(thresholds.get("collapse_ndcg", 0.2171))
    strong_lift = float(thresholds.get("strong_lift", 0.01))
    weak_lift = float(thresholds.get("weak_lift", 0.003))
    high_collapse_rate = float(thresholds.get("high_collapse_rate", 0.25))
    minimum_family_runs = int(policy.get("minimum_family_runs_for_policy", 2))
    baseline_reference = policy.get("baseline_reference") if isinstance(policy.get("baseline_reference"), dict) else {}
    algorithm_policy = policy.get("algorithm_first") if isinstance(policy.get("algorithm_first"), dict) else {}
    anchor_defs = algorithm_policy.get("anchor_families") if isinstance(algorithm_policy.get("anchor_families"), dict) else {}
    search_algorithm_policy = (
        search_policy.get("algorithm_first")
        if isinstance(search_policy, dict) and isinstance(search_policy.get("algorithm_first"), dict)
        else {}
    )
    search_anchors = (
        search_algorithm_policy.get("anchor_families")
        if isinstance(search_algorithm_policy.get("anchor_families"), list)
        else []
    )
    anchor_families = {str(item) for item in anchor_defs}
    anchor_families.update(str(item) for item in search_anchors if str(item))
    historical_floor = float(algorithm_policy.get("historical_anchor_floor", 0.274))
    revisit_bucket = str(algorithm_policy.get("high_potential_policy_bucket") or "revisit")

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    do_not_repeat_signatures: list[str] = []
    failed_composition_counts: dict[str, int] = defaultdict(int)
    for row in memory_rows:
        if row.get("event"):
            continue
        signature = canonical_signature(row.get("parameter_signature"))
        if signature and signature not in do_not_repeat_signatures:
            do_not_repeat_signatures.append(signature)
        if is_failed_record(row, metric, collapse_ndcg):
            for name in failed_composition_names(row, search_policy=search_policy):
                failed_composition_counts[name] += 1
        family_rows[candidate_family(row)].append(row)

    family_summaries = []
    for family, rows in sorted(family_rows.items()):
        values = [metric_value(row, metric) for row in rows]
        values = [value for value in values if value is not None]
        crash_count = sum(
            1
            for row in rows
            if str(row.get("status") or "").lower() in {"crash", "failed", "error"}
        )
        if not values and crash_count == 0:
            continue
        base_model = infer_base_model(rows[0])
        baseline = to_float(baseline_reference.get(base_model))
        best = max(values) if values else None
        mean = sum(values) / len(values) if values else None
        collapse_count = sum(1 for value in values if value < collapse_ndcg)
        decisions = defaultdict(int)
        for row in rows:
            decision = str(row.get("decision") or "unknown")
            decisions[decision] += 1
        best_row = max(rows, key=lambda row: metric_value(row, metric) or -1)
        params = best_row.get("params") if isinstance(best_row.get("params"), dict) else {}
        lift = best - baseline if best is not None and baseline is not None else None
        collapse_rate = collapse_count / len(values) if values else 0.0
        trend = convergence_trend(rows, metric, policy)
        domain_warnings = domain_prior_warnings(base_model=base_model, params=params, policy=policy)
        anchor_info = anchor_defs.get(family) if isinstance(anchor_defs.get(family), dict) else {}
        historical_best = to_float(anchor_info.get("historical_best"))
        is_high_potential = (
            family in anchor_families
            or (historical_best is not None and historical_best >= historical_floor)
            or (best is not None and best >= historical_floor)
        )
        if not values:
            policy_bucket = "avoid"
        elif len(values) < minimum_family_runs:
            policy_bucket = "caution" if collapse_rate < high_collapse_rate else "avoid"
        elif lift is not None and lift >= strong_lift and collapse_rate < high_collapse_rate:
            policy_bucket = "encourage"
        elif collapse_rate >= high_collapse_rate or best < (baseline or collapse_ndcg):
            policy_bucket = "avoid"
        elif lift is not None and lift >= weak_lift:
            policy_bucket = "caution"
        else:
            policy_bucket = "avoid"
        if (
            policy_bucket == "avoid"
            and is_high_potential
            and crash_count < 2
            and collapse_rate < max(0.50, high_collapse_rate)
        ):
            policy_bucket = revisit_bucket
        family_summaries.append(
            {
                "family": family,
                "base_model": base_model,
                "action_type": action_type_for_record(best_row, family, action_space),
                "runs": len(values),
                "best": round(best, 6) if best is not None else None,
                "mean": round(mean, 6) if mean is not None else None,
                "std": round(std(values), 6),
                "baseline": baseline,
                "lift_best": round(lift, 6) if lift is not None else None,
                "collapse_count": collapse_count,
                "collapse_rate": round(collapse_rate, 6),
                "crash_count": crash_count,
                "decisions": dict(decisions),
                "effective_region": params,
                "trend": trend,
                "domain_warnings": domain_warnings,
                "historical_best": round(historical_best, 6) if historical_best is not None else None,
                "historical_mechanism": anchor_info.get("mechanism") if anchor_info else "",
                "high_potential": bool(is_high_potential),
                "policy_bucket": policy_bucket,
            }
        )

    family_summaries.sort(
        key=lambda item: (
            item["policy_bucket"] != "encourage",
            item["policy_bucket"] != "revisit",
            -float(item["best"] or 0.0),
            -int(item["crash_count"]),
        )
    )
    plateau_analysis = detect_plateau(
        memory_rows,
        metric=metric,
        policy=policy,
        search_policy=search_policy,
    )
    return {
        "metric": metric,
        "family_summaries": family_summaries,
        "successful_metric_rows": sum(item["runs"] for item in family_summaries),
        "do_not_repeat_signatures": do_not_repeat_signatures[-100:],
        "failed_compositions": [
            {"composition": name, "failures": count}
            for name, count in sorted(failed_composition_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "algorithm_tasks": list(algorithm_policy.get("algorithm_tasks") or []),
        "anchor_families": sorted(anchor_families),
        "plateau_analysis": plateau_analysis,
    }


def limited(items: list[dict[str, Any]], bucket: str, limit: int) -> list[dict[str, Any]]:
    selected = [item for item in items if item.get("policy_bucket") == bucket]
    selected.sort(key=lambda item: float(item.get("best") or 0.0), reverse=True)
    return selected[:limit]


def family_ids(items: list[dict[str, Any]], bucket: str, limit: int = 10) -> list[str]:
    return [str(item["family"]) for item in limited(items, bucket, limit)]


def promising_parameter_regions(items: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for item in items:
        if item.get("policy_bucket") not in {"encourage", "caution", "revisit"}:
            continue
        params = item.get("effective_region")
        if not isinstance(params, dict) or not params:
            continue
        regions.append(
            {
                "family": item["family"],
                "base_model": item["base_model"],
                "action_type": item["action_type"],
                "params": params,
                "best": item.get("best"),
                "mean": item.get("mean"),
            }
        )
    regions.sort(key=lambda item: float(item.get("best") or 0.0), reverse=True)
    return regions[:limit]


def tree_policy_hints(tree_summary: dict[str, Any]) -> dict[str, Any]:
    if not tree_summary:
        return {}
    if str(tree_summary.get("metric") or "ndcg@10") != "ndcg@10":
        return {}
    nodes = tree_summary.get("nodes")
    if not isinstance(nodes, list):
        nodes = []
    node_by_id = {
        str(node.get("candidate_id") or ""): node
        for node in nodes
        if isinstance(node, dict) and str(node.get("candidate_id") or "")
    }
    high_yield = []
    stalled = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        child_count = len(children) if isinstance(children, list) else int(node.get("child_count") or 0)
        child_values = [
            to_float(node_by_id.get(str(child), {}).get("best_ndcg@10"))
            for child in (children if isinstance(children, list) else [])
        ]
        values = [to_float(node.get("best_ndcg@10")), *child_values]
        values = [value for value in values if value is not None]
        best = max(values) if values else None
        mean = sum(values) / len(values) if values else None
        crashes = int(node.get("crash_count") or 0)
        node_id = str(node.get("candidate_id") or "")
        if child_count and best is not None and (mean is None or best >= mean):
            high_yield.append({"family": node_id, "children": child_count, "best": round(best, 6)})
        if child_count >= 3 and (best is None or crashes >= 2):
            stalled.append({"family": node_id, "children": child_count, "crashes": crashes})
    high_yield.sort(key=lambda item: float(item.get("best") or 0.0), reverse=True)
    stalled.sort(key=lambda item: (-int(item.get("children") or 0), str(item.get("family") or "")))
    return {
        "high_yield_parents": high_yield[:5],
        "stalled_or_fragile_parents": stalled[:5],
    }


def build_experience_policy(
    summary: dict[str, Any],
    *,
    search_policy: dict[str, Any],
    tree_summary: dict[str, Any],
) -> dict[str, Any]:
    families = summary.get("family_summaries")
    family_summaries = families if isinstance(families, list) else []
    family_budget = search_policy.get("family_budget") if isinstance(search_policy.get("family_budget"), dict) else {}
    validation_gates = (
        search_policy.get("validation_gates")
        if isinstance(search_policy.get("validation_gates"), dict)
        else {}
    )
    composition_rules = (
        search_policy.get("composition_rules")
        if isinstance(search_policy.get("composition_rules"), dict)
        else {}
    )
    algorithm_first = (
        search_policy.get("algorithm_first")
        if isinstance(search_policy.get("algorithm_first"), dict)
        else {}
    )
    anchor_families = {
        str(item)
        for item in (algorithm_first.get("anchor_families") or summary.get("anchor_families") or [])
        if str(item)
    }
    protect_anchors = bool(family_budget.get("protect_anchor_families_from_discard_freeze", True))
    freeze_crashes = int(family_budget.get("crashes_before_freeze", 2))
    freeze_collapse = float(family_budget.get("collapse_rate_before_freeze", 0.25))
    freeze_severe_collapse = float(family_budget.get("severe_collapse_rate_before_freeze", freeze_collapse))
    freeze_families = []
    for item in family_summaries:
        family = str(item.get("family") or "")
        crash_count = int(item.get("crash_count") or 0)
        collapse_rate = float(item.get("collapse_rate") or 0.0)
        high_potential = bool(item.get("high_potential"))
        decisions = item.get("decisions") if isinstance(item.get("decisions"), dict) else {}
        failed_decisions = int(decisions.get("discard") or 0) + int(decisions.get("crash") or 0)
        if protect_anchors and family in anchor_families:
            if crash_count >= freeze_crashes or collapse_rate >= freeze_severe_collapse:
                freeze_families.append(family)
            continue
        if high_potential and crash_count < freeze_crashes and collapse_rate < freeze_severe_collapse:
            continue
        if crash_count >= freeze_crashes or collapse_rate >= freeze_collapse or failed_decisions >= freeze_crashes:
            freeze_families.append(family)
    encourage = family_ids(family_summaries, "encourage")
    caution = family_ids(family_summaries, "caution")
    revisit = family_ids(family_summaries, "revisit")
    avoid = family_ids(family_summaries, "avoid")
    domain_prior_notes = []
    trend_notes = []
    for item in family_summaries:
        warnings = item.get("domain_warnings") if isinstance(item.get("domain_warnings"), list) else []
        for warning in warnings[:2]:
            domain_prior_notes.append({"family": item.get("family"), "note": warning})
        trend = item.get("trend") if isinstance(item.get("trend"), dict) else {}
        trend_name = str(trend.get("trend") or "")
        if trend_name in {"improving", "degrading"} or bool(trend.get("high_variance")):
            trend_notes.append(
                {
                    "family": item.get("family"),
                    "trend": trend_name,
                    "slope": trend.get("slope"),
                    "cv": trend.get("cv"),
                    "high_variance": bool(trend.get("high_variance")),
                }
            )
    search_stages = search_policy.get("search_stages") if isinstance(search_policy.get("search_stages"), dict) else {}
    exploit_defaults = []
    exploit_stage = search_stages.get("exploit") if isinstance(search_stages.get("exploit"), dict) else {}
    if isinstance(exploit_stage.get("priority_families"), list):
        exploit_defaults = [str(item) for item in exploit_stage.get("priority_families") if str(item)]
    algorithm_stage = search_stages.get("algorithm_discovery") if isinstance(search_stages.get("algorithm_discovery"), dict) else {}
    algorithm_defaults = []
    if isinstance(algorithm_stage.get("priority_families"), list):
        algorithm_defaults = [str(item) for item in algorithm_stage.get("priority_families") if str(item)]
    frozen_or_avoid = {family for family in [*avoid, *freeze_families] if family}
    frozen_or_avoid.difference_update(revisit)
    plateau = summary.get("plateau_analysis") if isinstance(summary.get("plateau_analysis"), dict) else {}
    plateau_detected = bool(plateau.get("plateau_detected"))
    capped_families = {str(item) for item in plateau.get("local_repair_capped_families", []) if str(item)}
    forced_exploration = [str(item) for item in plateau.get("forced_exploration_families", []) if str(item)]
    forced_tasks = [str(item) for item in plateau.get("forced_algorithm_tasks", []) if str(item)]
    if plateau_detected:
        prefer_candidates = [*forced_exploration, *algorithm_defaults]
    else:
        prefer_candidates = encourage or algorithm_defaults or exploit_defaults
    deduped_prefer: list[str] = []
    for family in prefer_candidates:
        if family not in deduped_prefer:
            deduped_prefer.append(family)
    prefer_candidates = deduped_prefer
    prefer_families = [family for family in prefer_candidates if family not in frozen_or_avoid]
    if plateau_detected:
        prefer_families = [
            family
            for family in prefer_families
            if not any(family_matches(family, capped) for capped in capped_families)
        ]
    repair_families = [family for family in [*revisit, *caution] if family not in frozen_or_avoid]
    if plateau_detected:
        repair_families = [
            family
            for family in repair_families
            if not any(family_matches(family, capped) for capped in capped_families)
            and any(family_matches(family, forced) for forced in forced_exploration)
        ]
    algorithm_tasks = [str(item) for item in summary.get("algorithm_tasks", []) if str(item)]
    if forced_tasks:
        algorithm_tasks = [*forced_tasks, *algorithm_tasks]
    hints = tree_policy_hints(tree_summary)
    instruction = (
        "Plateau detected: force cross-family algorithmic proposals around forced_exploration_families; "
        "do not spend the next proposal window on minor repairs of local_repair_capped_families. "
        "Use parameter-only tuning only after a new family crosses the tuned baseline."
    ) if plateau_detected else (
        "Prioritize algorithmic mechanism proposals and repairs around anchor/revisit families; "
        "use parameter-only tuning only after a credible algorithm signal; introduce at most the "
        "configured number of code_required proposals per window."
    )
    return {
        "encourage_families": encourage,
        "caution_families": caution,
        "revisit_families": revisit,
        "avoid_families": avoid,
        "freeze_families": sorted(set(freeze_families)),
        "plateau_analysis": plateau,
        "promising_parameter_regions": promising_parameter_regions(family_summaries),
        "failed_compositions": summary.get("failed_compositions", []),
        "do_not_repeat_signatures": summary.get("do_not_repeat_signatures", []),
        "algorithm_tasks": algorithm_tasks,
        "domain_prior_notes": domain_prior_notes[:10],
        "trend_notes": trend_notes[:10],
        "budget_hints": {
            "max_code_required_per_window": int(
                algorithm_first.get("algorithm_budget_per_window", family_budget.get("max_code_required_per_window", 1))
            ),
            "window_rounds": int(algorithm_first.get("window_rounds", family_budget.get("window_rounds", 5))),
            "max_active_families_per_stage": int(family_budget.get("max_active_families_per_stage", 4)),
            "keep_requires_multiseed": bool(validation_gates.get("keep_requires_multiseed", True)),
            "minimum_validation_seeds": int(validation_gates.get("minimum_validation_seeds", 3)),
            "parameter_tuning_cap_fraction": float(algorithm_first.get("parameter_tuning_cap_fraction", 0.10)),
        },
        "next_proposal_policy": {
            "stage_order": ["algorithm_discovery", "repair", "exploit", "explore"],
            "prefer_families": prefer_families[:8],
            "repair_families": repair_families[:8],
            "avoid_families": sorted(frozen_or_avoid)[:12],
            "anchor_families": sorted(anchor_families),
            "forced_exploration_families": forced_exploration[:8],
            "local_repair_capped_families": sorted(capped_families)[:8],
            "plateau_detected": plateau_detected,
            "algorithm_tasks": algorithm_tasks[:8],
            "composition_rules": composition_rules,
            "tree_hints": hints,
            "instruction": instruction,
        },
    }


def format_family_line(item: dict[str, Any]) -> str:
    params = item.get("effective_region") or {}
    if isinstance(params, dict) and params:
        param_text = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    else:
        param_text = "no stable parameter region yet"
    best_text = f"{item['best']:.4f}" if item.get("best") is not None else "n/a"
    mean_text = f"{item['mean']:.4f}" if item.get("mean") is not None else "n/a"
    return (
        f"- {item['family']} ({item['base_model']}, {item['action_type']}): "
        f"best={best_text}, mean={mean_text}, runs={item['runs']}, crashes={item['crash_count']}; {param_text}"
    )


def build_markdown(
    summary: dict[str, Any],
    policy: dict[str, Any],
    experience_policy: dict[str, Any] | None = None,
) -> str:
    experience_policy = experience_policy or summary.get("experience_policy") or {}
    minimum = int(policy.get("minimum_successful_runs", 3))
    if summary["successful_metric_rows"] < minimum:
        next_policy = experience_policy.get("next_proposal_policy") if isinstance(experience_policy, dict) else {}
        prefer = next_policy.get("prefer_families") if isinstance(next_policy, dict) else []
        tasks = next_policy.get("algorithm_tasks") if isinstance(next_policy, dict) else []
        return "\n".join(
            [
                "# RecClaw Experience Summary",
                "",
                "Status: insufficient evidence",
                "",
                f"Only {summary['successful_metric_rows']} successful metric rows were available.",
                "Do not infer family preferences yet. Continue with the declared action space only.",
                "",
                "## Domain Prior Notes",
                "- none",
                "",
                "## Trend Notes",
                "- none",
                "",
                "## Next Proposal Policy",
                f"- Prefer families: {', '.join(prefer) if prefer else 'use search_policy defaults'}",
                f"- Algorithm tasks: {'; '.join(str(item) for item in tasks[:4]) if tasks else 'use algorithm_first search_policy defaults'}",
                "- Keep all proposals inside configs/action_space.yaml and configs/search_policy.yaml.",
                "",
            ]
        )

    limits = policy.get("summary_limits") if isinstance(policy.get("summary_limits"), dict) else {}
    encourage = limited(summary["family_summaries"], "encourage", int(limits.get("encourage", 5)))
    caution = limited(summary["family_summaries"], "caution", int(limits.get("caution", 5)))
    revisit = limited(summary["family_summaries"], "revisit", int(limits.get("caution", 5)))
    avoid = limited(summary["family_summaries"], "avoid", int(limits.get("avoid", 5)))
    next_policy = experience_policy.get("next_proposal_policy") if isinstance(experience_policy, dict) else {}
    budget_hints = experience_policy.get("budget_hints") if isinstance(experience_policy, dict) else {}
    domain_notes = experience_policy.get("domain_prior_notes") if isinstance(experience_policy, dict) else []
    trend_notes = experience_policy.get("trend_notes") if isinstance(experience_policy, dict) else []
    prefer = next_policy.get("prefer_families") if isinstance(next_policy, dict) else []
    repair = next_policy.get("repair_families") if isinstance(next_policy, dict) else []
    avoid_policy = next_policy.get("avoid_families") if isinstance(next_policy, dict) else []
    tasks = next_policy.get("algorithm_tasks") if isinstance(next_policy, dict) else []
    forced = next_policy.get("forced_exploration_families") if isinstance(next_policy, dict) else []
    capped = next_policy.get("local_repair_capped_families") if isinstance(next_policy, dict) else []
    plateau = experience_policy.get("plateau_analysis") if isinstance(experience_policy, dict) else {}
    plateau = plateau if isinstance(plateau, dict) else {}
    domain_lines = [
        f"- {item.get('family')}: {item.get('note')}"
        for item in domain_notes
        if isinstance(item, dict) and item.get("note")
    ]
    trend_lines = [
        (
            f"- {item.get('family')}: trend={item.get('trend')}, "
            f"slope={item.get('slope')}, cv={item.get('cv')}"
            f"{', high_variance' if item.get('high_variance') else ''}"
        )
        for item in trend_notes
        if isinstance(item, dict) and item.get("trend")
    ]

    lines = [
        "# RecClaw Experience Summary",
        "",
        f"Metric: {summary['metric']}",
        "",
        "## Encourage",
        *(format_family_line(item) for item in encourage),
        "",
        "## Caution",
        *(format_family_line(item) for item in caution),
        "",
        "## Revisit",
        *(format_family_line(item) for item in revisit),
        "",
        "## Avoid",
        *(format_family_line(item) for item in avoid),
        "",
        "## Domain Prior Notes",
        *(domain_lines or ["- none"]),
        "",
        "## Trend Notes",
        *(trend_lines or ["- none"]),
        "",
        "## Plateau / Anti-Locality",
        f"- Detected: {bool(plateau.get('plateau_detected'))}",
        f"- Rows since best: {plateau.get('rows_since_best', 'n/a')}",
        f"- Capped local families: {', '.join(capped) if capped else 'none'}",
        f"- Forced exploration families: {', '.join(forced) if forced else 'none'}",
        "",
        "## Next Proposal Policy",
        f"- Prefer families: {', '.join(prefer) if prefer else 'none yet'}",
        f"- Repair families: {', '.join(repair) if repair else 'none yet'}",
        f"- Avoid/freeze families: {', '.join(avoid_policy) if avoid_policy else 'none yet'}",
        f"- Algorithm tasks: {'; '.join(str(item) for item in tasks[:4]) if tasks else 'none declared'}",
        (
            "- Code-required budget: "
            f"{budget_hints.get('max_code_required_per_window', 1)} per "
            f"{budget_hints.get('window_rounds', 5)} rounds"
        ),
        f"- Parameter-only tuning cap: {budget_hints.get('parameter_tuning_cap_fraction', 0.1)} of proposal budget unless an algorithm family has crossed the tuned baseline.",
        "- Avoid repeating signatures listed in experience_summary.json unless proposing a minimal repair.",
        "- Keep all proposals inside configs/action_space.yaml and the candidate proposal schema.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], markdown: str, out_md: Path, out_json: Path, out_jsonl: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **summary,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    with out_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact RecClaw runtime experience summary.")
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--action-space", default=str(DEFAULT_ACTION_SPACE))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--search-policy", default=str(DEFAULT_SEARCH_POLICY))
    parser.add_argument("--tree", default=str(DEFAULT_TREE_PATH))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-jsonl", default=str(DEFAULT_OUT_JSONL))
    parser.add_argument(
        "--allow-lablog-input",
        action="store_true",
        help="Explicit report-only escape hatch. Do not use for agent runtime summaries.",
    )
    args = parser.parse_args()

    for value in (
        args.memory,
        args.results,
        args.action_space,
        args.policy,
        args.search_policy,
        args.tree,
        args.out_md,
        args.out_json,
        args.out_jsonl,
    ):
        reject_lablog_path(value, allow_lablog=bool(args.allow_lablog_input))

    action_space = load_action_space(Path(args.action_space))
    policy = load_yaml(Path(args.policy))
    search_policy = load_yaml(Path(args.search_policy))
    tree_summary = load_json(Path(args.tree))
    memory_rows = load_jsonl(Path(args.memory))
    results_rows = load_results_csv(Path(args.results))
    combined_rows = combine_evidence_rows(memory_rows, results_rows)

    summary = summarize_memory(
        combined_rows,
        action_space=action_space,
        policy=policy,
        search_policy=search_policy,
    )
    experience_policy = build_experience_policy(
        summary,
        search_policy=search_policy,
        tree_summary=tree_summary,
    )
    summary["experience_policy"] = experience_policy
    summary["search_policy_version"] = search_policy.get("version")
    summary["tree_source"] = str(args.tree) if tree_summary else ""
    markdown = build_markdown(summary, policy, experience_policy)
    write_outputs(summary, markdown, Path(args.out_md), Path(args.out_json), Path(args.out_jsonl))
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
