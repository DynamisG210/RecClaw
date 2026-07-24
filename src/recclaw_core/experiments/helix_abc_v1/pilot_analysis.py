"""Frozen, treatment-blind M6 Pilot support and four-axis analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest


PILOT_MAX_FAILURE_RATE = 0.25
PILOT_MIN_SUCCESS_PER_INSTANCE = 2


def four_axis_frontiers(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["opaque_instance_id"])].append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for instance, values in sorted(grouped.items()):
        best: float | None = None
        executions = 0
        tokens = 0
        gpu_cost = 0
        points = []
        for row in sorted(values, key=lambda item: int(item["round_index"])):
            executions += int(row["ordinary_execution_count"])
            tokens += int(row["billed_tokens"])
            gpu_cost += int(row["gpu_cost_microunits"])
            metric = row.get("ndcg")
            if isinstance(metric, (int, float)):
                best = float(metric) if best is None else max(best, float(metric))
            points.append(
                {
                    "execution_axis": executions,
                    "frontier_ndcg": best,
                    "gpu_cost_axis_microunits": gpu_cost,
                    "round_axis": int(row["round_index"]),
                    "support": int(best is not None),
                    "token_axis": tokens,
                }
            )
        result[instance] = points
    return canonical_value(result)


def pilot_readiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_instances: int,
    expected_rounds_per_instance: int,
    guard_call_count: int,
    expected_guard_call_count: int,
    meta_versions: Mapping[str, int],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["opaque_instance_id"])].append(row)
    expected_rows = expected_instances * expected_rounds_per_instance
    completeness = (
        len(grouped) == expected_instances
        and len(rows) == expected_rows
        and all(
            len(values) == expected_rounds_per_instance
            for values in grouped.values()
        )
    )
    successes = sum(row.get("run_status") == "SUCCESS" for row in rows)
    failures = len(rows) - successes
    failure_rate = failures / len(rows) if rows else 1.0
    support_by_instance = {
        instance: sum(row.get("run_status") == "SUCCESS" for row in values)
        for instance, values in sorted(grouped.items())
    }
    support_ok = (
        completeness
        and all(
            value >= PILOT_MIN_SUCCESS_PER_INSTANCE
            for value in support_by_instance.values()
        )
    )
    metrics_complete = all(
        row.get("run_status") != "SUCCESS"
        or isinstance(row.get("ndcg"), (int, float))
        for row in rows
    )
    meta_ok = (
        set(meta_versions) == {"B", "C"}
        and all(int(value) == expected_rounds_per_instance + 1 for value in meta_versions.values())
    )
    checks = {
        "failure_rate_within_limit": failure_rate <= PILOT_MAX_FAILURE_RATE,
        "guard_call_count_exact": guard_call_count == expected_guard_call_count,
        "identity_completeness": completeness,
        "meta_versioned": meta_ok,
        "successful_metric_rows_complete": metrics_complete,
        "support_non_degenerate": support_ok,
    }
    if not rows or not completeness:
        verdict = "INSUFFICIENT_INFORMATION"
    elif all(checks.values()):
        verdict = "GO"
    else:
        verdict = "NOT_READY"
    return {
        "checks": checks,
        "failure_count": failures,
        "failure_rate": failure_rate,
        "frontier_digest": sha256_digest(four_axis_frontiers(rows)),
        "row_count": len(rows),
        "success_count": successes,
        "support_by_instance": support_by_instance,
        "verdict": verdict,
    }


__all__ = [
    "PILOT_MAX_FAILURE_RATE",
    "PILOT_MIN_SUCCESS_PER_INSTANCE",
    "four_axis_frontiers",
    "pilot_readiness",
]
