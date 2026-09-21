"""Pre-registered development-effect analysis for the 50-round Pilot."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest
from .pilot_analysis import analysis_observation_key


class EffectPilotAnalysisError(ValueError):
    """The frozen effect analysis inputs are incomplete or inconsistent."""


def _eligible(
    row: Mapping[str, Any],
    eligibility_by_observation: Mapping[str, str],
) -> bool:
    key = analysis_observation_key(row)
    try:
        value = str(eligibility_by_observation[key])
    except KeyError as exc:
        raise EffectPilotAnalysisError(
            f"missing SearchEligible disposition for {key}"
        ) from exc
    return value in {
        "SEARCH_ELIGIBLE",
        "SEARCH_ELIGIBLE_PRELIMINARY",
    }


def _cost_auc(
    points: Sequence[tuple[int, float]],
    *,
    budget_cap: int,
) -> float:
    if budget_cap <= 0:
        raise EffectPilotAnalysisError("budget cap must be positive")
    prior_axis = 0
    prior_frontier = 0.0
    area = 0.0
    for axis, frontier in points:
        if axis < prior_axis or axis > budget_cap:
            raise EffectPilotAnalysisError(
                "cost axis must be monotonic and within the frozen cap"
            )
        area += float(axis - prior_axis) * prior_frontier
        prior_axis = axis
        prior_frontier = frontier
    area += float(budget_cap - prior_axis) * prior_frontier
    return area / float(budget_cap)


def arm_trajectory_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    eligibility_by_observation: Mapping[str, str],
    rounds_per_arm: int,
    token_budget_cap: int,
    gpu_cost_budget_cap: int,
    useful_signal_delta: float,
    best_tolerance: float,
) -> dict[str, Any]:
    """Compute one Arm's exact pre-registered SearchEligible trajectory."""

    if len(rows) != rounds_per_arm:
        raise EffectPilotAnalysisError(
            "one exact row per scheduled Arm round is required"
        )
    ordered = sorted(rows, key=lambda item: int(item["round_index"]))
    if tuple(int(item["round_index"]) for item in ordered) != tuple(
        range(1, rounds_per_arm + 1)
    ):
        raise EffectPilotAnalysisError(
            "Arm round indexes must be the complete one-based schedule"
        )

    frontier: float | None = None
    running: list[float] = []
    token_axis = 0
    gpu_axis = 0
    token_points: list[tuple[int, float]] = []
    gpu_points: list[tuple[int, float]] = []
    useful_signals = 0
    executed = 0
    successful = 0
    eligible_successes = 0
    blocked_or_failed = 0
    semantic_digests: list[str] = []
    first_best_round: int | None = None
    first_best_tokens: int | None = None
    first_best_gpu_cost: int | None = None

    for row in ordered:
        tokens = int(row["billed_tokens"])
        gpu_cost = int(row["gpu_cost_microunits"])
        ordinary_execution = int(row["ordinary_execution_count"])
        if min(tokens, gpu_cost, ordinary_execution) < 0:
            raise EffectPilotAnalysisError(
                "resource and execution counts must be non-negative"
            )
        token_axis += tokens
        gpu_axis += gpu_cost
        executed += ordinary_execution
        success = str(row["run_status"]) == "SUCCESS"
        successful += int(success)
        metric = row.get("ndcg")
        enters = (
            success
            and isinstance(metric, (int, float))
            and _eligible(row, eligibility_by_observation)
        )
        if enters:
            eligible_successes += 1
            value = float(metric)
            if frontier is None or value >= frontier + useful_signal_delta:
                useful_signals += 1
            frontier = value if frontier is None else max(frontier, value)
        if ordinary_execution == 0 or not success:
            blocked_or_failed += 1
        semantic = row.get("mechanism_semantics_digest")
        if ordinary_execution and isinstance(semantic, str) and semantic:
            semantic_digests.append(semantic)
        current = 0.0 if frontier is None else frontier
        running.append(current)
        token_points.append((token_axis, current))
        gpu_points.append((gpu_axis, current))

    if token_axis > token_budget_cap or gpu_axis > gpu_cost_budget_cap:
        raise EffectPilotAnalysisError(
            "observed cost exceeds the frozen analysis budget cap"
        )
    best = max(running, default=0.0)
    if best > 0.0:
        target = best - best_tolerance
        cumulative_tokens = 0
        cumulative_gpu = 0
        for row, value in zip(ordered, running, strict=True):
            cumulative_tokens += int(row["billed_tokens"])
            cumulative_gpu += int(row["gpu_cost_microunits"])
            if value >= target:
                first_best_round = int(row["round_index"])
                first_best_tokens = cumulative_tokens
                first_best_gpu_cost = cumulative_gpu
                break
    distinct_semantics = len(set(semantic_digests))
    duplicate_count = max(0, len(semantic_digests) - distinct_semantics)
    return canonical_value(
        {
            "best_search_eligible_ndcg_at_10": best,
            "blocker_rate": blocked_or_failed / rounds_per_arm,
            "budget_to_best": {
                "gpu_cost_microunits": first_best_gpu_cost,
                "round": first_best_round,
                "tokens": first_best_tokens,
            },
            "distinct_executed_semantics": distinct_semantics,
            "duplicate_rate": (
                duplicate_count / len(semantic_digests)
                if semantic_digests
                else 1.0
            ),
            "eligible_success_count": eligible_successes,
            "executed_count": executed,
            "gpu_cost_auc": _cost_auc(
                gpu_points,
                budget_cap=gpu_cost_budget_cap,
            ),
            "gpu_cost_used": gpu_axis,
            "round_auc": sum(running) / rounds_per_arm,
            "successful_result_count": successful,
            "token_auc": _cost_auc(
                token_points,
                budget_cap=token_budget_cap,
            ),
            "tokens_used": token_axis,
            "useful_signal_count": useful_signals,
            "useful_signal_rate_per_execution": (
                useful_signals / executed if executed else 0.0
            ),
        }
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def research_capability_visibility(
    *,
    arm_a: Mapping[str, Any],
    arm_b: Mapping[str, Any],
    criteria: Mapping[str, Any],
    producer_role_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Evaluate the frozen B-A Research Capability visibility branches."""

    margins = criteria["research_capability"]
    deltas = {
        "best": float(arm_b["best_search_eligible_ndcg_at_10"])
        - float(arm_a["best_search_eligible_ndcg_at_10"]),
        "blocker_rate": float(arm_b["blocker_rate"])
        - float(arm_a["blocker_rate"]),
        "duplicate_rate": float(arm_b["duplicate_rate"])
        - float(arm_a["duplicate_rate"]),
        "round_auc": float(arm_b["round_auc"])
        - float(arm_a["round_auc"]),
        "useful_signal_rate": float(
            arm_b["useful_signal_rate_per_execution"]
        )
        - float(arm_a["useful_signal_rate_per_execution"]),
    }
    ratios = {
        "gpu_cost_auc": _ratio(
            float(arm_b["gpu_cost_auc"]),
            float(arm_a["gpu_cost_auc"]),
        ),
        "token_auc": _ratio(
            float(arm_b["token_auc"]),
            float(arm_a["token_auc"]),
        ),
    }
    efficacy = (
        deltas["round_auc"] >= float(margins["round_auc_superiority"])
        and deltas["best"] >= float(margins["best_noninferiority"])
    )
    breakthrough = (
        deltas["round_auc"] >= float(margins["round_auc_noninferiority"])
        and deltas["best"] >= float(margins["best_superiority"])
    )
    efficient_noninferiority = (
        deltas["round_auc"] >= float(margins["round_auc_noninferiority"])
        and ratios["gpu_cost_auc"]
        >= float(margins["minimum_efficiency_ratio"])
        and ratios["token_auc"]
        >= float(margins["minimum_efficiency_ratio"])
        and deltas["useful_signal_rate"]
        >= float(margins["useful_signal_rate_gain"])
    )
    role_floor = int(margins["minimum_selected_per_producer_role"])
    coverage = (
        set(producer_role_counts)
        == set(margins["required_producer_roles"])
        and all(int(value) >= role_floor for value in producer_role_counts.values())
        and int(arm_b["distinct_executed_semantics"])
        >= int(margins["minimum_distinct_executed_semantics"])
        and float(arm_b["duplicate_rate"])
        <= float(margins["maximum_duplicate_rate"])
        and float(arm_b["blocker_rate"])
        <= float(margins["maximum_blocker_rate"])
    )
    return canonical_value(
        {
            "branch_pass": {
                "breakthrough": breakthrough,
                "efficacy": efficacy,
                "efficient_noninferiority": efficient_noninferiority,
            },
            "coverage_gate": coverage,
            "deltas_b_minus_a": deltas,
            "efficiency_ratios_b_over_a": ratios,
            "producer_role_counts": dict(producer_role_counts),
            "verdict": (
                "PASS"
                if coverage
                and (efficacy or breakthrough or efficient_noninferiority)
                else "FAIL"
            ),
        }
    )


def evidence_guard_visibility(
    *,
    arm_b: Mapping[str, Any],
    arm_c: Mapping[str, Any],
    criteria: Mapping[str, Any],
    guard_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen C-B Guard safety, activity and non-suppression line."""

    margins = criteria["evidence_guard"]
    ndcg_non_suppression = (
        float(arm_c["round_auc"]) - float(arm_b["round_auc"])
        >= float(margins["round_auc_non_suppression"])
        and float(arm_c["best_search_eligible_ndcg_at_10"])
        - float(arm_b["best_search_eligible_ndcg_at_10"])
        >= float(margins["best_non_suppression"])
    )
    required_exact = {
        "cross_arm_contamination_count": 0,
        "false_allow_count": 0,
        "false_block_count": 0,
        "guard_private_input_leak_count": 0,
        "legal_candidate_permanent_suppression_count": 0,
        "preliminary_marked_confirmed_count": 0,
        "search_memory_pollution_count": 0,
        "seed_binding_mismatch_count": 0,
    }
    exact_safety = all(
        int(guard_metrics.get(key, -1)) == expected
        for key, expected in required_exact.items()
    )
    activity = (
        int(guard_metrics["nontrivial_intervention_count"])
        >= int(margins["minimum_nontrivial_interventions"])
        and int(guard_metrics["completed_validation_count"])
        >= int(margins["minimum_completed_validations"])
        and int(guard_metrics["successful_challenge_case_count"])
        >= int(margins["minimum_successful_challenge_cases"])
    )
    return canonical_value(
        {
            "activity_gate": activity,
            "c_minus_b": {
                "best": float(
                    arm_c["best_search_eligible_ndcg_at_10"]
                )
                - float(arm_b["best_search_eligible_ndcg_at_10"]),
                "round_auc": float(arm_c["round_auc"])
                - float(arm_b["round_auc"]),
            },
            "exact_safety_gate": exact_safety,
            "guard_metrics": dict(guard_metrics),
            "ndcg_non_suppression_gate": ndcg_non_suppression,
            "verdict": (
                "PASS"
                if activity and exact_safety and ndcg_non_suppression
                else "FAIL"
            ),
        }
    )


def effect_pilot_verdict(
    *,
    chain_checks: Mapping[str, bool],
    research_visibility: Mapping[str, Any],
    guard_visibility: Mapping[str, Any],
    criteria_digest: str,
) -> dict[str, Any]:
    """Combine frozen chain and co-primary development-effect lines."""

    chain_pass = bool(chain_checks) and all(chain_checks.values())
    effect_pass = (
        research_visibility["verdict"] == "PASS"
        and guard_visibility["verdict"] == "PASS"
    )
    result = canonical_value(
        {
            "authority": "NONE",
            "chain_checks": dict(chain_checks),
            "chain_line": "PASS" if chain_pass else "FAIL",
            "criteria_digest": criteria_digest,
            "effect_line": "PASS" if effect_pass else "FAIL",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "guard_visibility": dict(guard_visibility),
            "main_eligibility": False,
            "research_visibility": dict(research_visibility),
            "verdict": (
                "PASS" if chain_pass and effect_pass else "FAIL"
            ),
        }
    )
    return {**result, "analysis_digest": sha256_digest(result)}


__all__ = [
    "EffectPilotAnalysisError",
    "arm_trajectory_metrics",
    "effect_pilot_verdict",
    "evidence_guard_visibility",
    "research_capability_visibility",
]
