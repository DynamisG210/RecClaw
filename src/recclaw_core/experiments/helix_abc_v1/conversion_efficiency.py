"""Small, pre-outcome conversion helpers for the Q5-A follow-up.

The module deliberately contains no provider, metric, or training calls.  It
only describes the observed mechanical-repair boundary and the fixed
screen-to-promotion schedule consumed by the existing stage runner.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest


CONVERSION_SCHEMA = "recclaw.research-line.q5-conversion-efficiency.v1"
MECHANICAL_REPAIR_SCHEMA = (
    "recclaw.research-line.q5-mechanical-implementer-repair.v1"
)
SCREEN_EPOCHS = 20
FULL_EPOCHS = 100
MAX_REPAIR_TURNS = 2
FULL_DEVELOPMENT_SEEDS = (54304, 54305)
FULL_RESOURCE_BUDGET_SECONDS = 7200
FULL_WATCHDOG_SECONDS = 10800
RESOURCE_DEADLINE_MIN_SECONDS = 180
RESOURCE_DEADLINE_MARGIN = 1.10
CANDIDATE_LOCAL_ALLOWED_FILES = (
    "recclaw_ext/__init__.py",
    "recclaw_ext/candidate.py",
    "recclaw_ext/layers.py",
    "recclaw_ext/modules.py",
    "recclaw_ext/ops.py",
)
RECBole_INTERFACE_CONTRACT = {
    "recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
    "model_base": "recbole.model.abstract_recommender.GeneralRecommender",
    "constructor": "(config, dataset)",
    "required_methods": ("calculate_loss", "predict", "full_sort_predict"),
    "gpu_budget_gb": 10,
}
MECHANICAL_SHAPE_TEST_HINT = {
    "calculate_loss": {"user_id": [4], "item_id": [4], "neg_item_id": [4]},
    "predict": {"user_id": [4], "item_id": [4]},
    "full_sort_predict": {"user_id": [4], "candidate_count": 32},
    "complexity_hint": "avoid full_catalog_by_embedding_dim_by_embedding_dim intermediates",
}
MECHANICAL_REPAIR_STAGES = frozenset(
    {"SCHEMA", "STATIC_VALIDATION", "CONSTRUCTION", "API_CONTRACT", "UNIT"}
)
MECHANICAL_REPAIR_CLASSES = frozenset({"IMPLEMENTATION", "INTERFACE"})
_OUTCOME_WORDS = frozenset(
    {
        "effect",
        "metric",
        "ndcg",
        "outcome",
        "result",
        "winner",
        "qualification_result",
    }
)


def _contains_outcome(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _OUTCOME_WORDS or _contains_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_outcome(item) for item in value)
    return False


def is_mechanical_repair_failure(failure: Mapping[str, Any]) -> bool:
    """Return whether a qualifier/validator failure may enter one repair call."""

    if _contains_outcome(failure):
        return False
    stage = str(failure.get("stage", "")).upper()
    failure_class = str(failure.get("failure_class", "")).upper()
    return (
        stage in MECHANICAL_REPAIR_STAGES
        and failure_class in MECHANICAL_REPAIR_CLASSES
    )


def build_mechanical_repair_request(
    original_request: Mapping[str, Any],
    failure: Mapping[str, Any],
    *,
    current_source: Mapping[str, str] | None = None,
    failure_message: str | None = None,
    short_trace: str | None = None,
    repair_attempt: int = 1,
) -> dict[str, Any]:
    """Bind only a mechanical failure to the same blind implementer request."""

    if not is_mechanical_repair_failure(failure):
        raise ValueError("failure is outside the mechanical repair boundary")
    if int(repair_attempt) < 1 or int(repair_attempt) > MAX_REPAIR_TURNS:
        raise ValueError("repair attempt is outside the finite revision budget")
    source = {
        str(path): str(content)
        for path, content in (current_source or {}).items()
    }
    context = canonical_value(
        {
            "failure_class": str(failure["failure_class"]),
            "reason_code": str(failure.get("reason_code", "UNKNOWN")),
            "stage": str(failure["stage"]),
            "message": str(failure_message or failure.get("message", ""))[:2000],
            "short_trace": str(short_trace or failure.get("traceback", ""))[:4000],
            "current_source_files": source,
            "shape_test_hint": MECHANICAL_SHAPE_TEST_HINT,
        }
    )
    repaired = canonical_value(
        {
            **dict(original_request),
            "repair_context": context,
            "repair_attempt": int(repair_attempt),
            "schema": MECHANICAL_REPAIR_SCHEMA,
        }
    )
    if _contains_outcome(repaired):
        raise ValueError("mechanical repair request crossed the outcome boundary")
    return repaired


def build_conversion_execution_plan(
    candidate_ids: Sequence[str],
    *,
    screen_seed: int,
    full_seeds: Sequence[int] = FULL_DEVELOPMENT_SEEDS,
    promotion_limit: int = 2,
) -> dict[str, Any]:
    """Freeze the short-fidelity and fresh-seed full-run schedule."""

    ids = tuple(str(value) for value in candidate_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("conversion plan requires unique candidate identities")
    seeds = tuple(int(value) for value in full_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or int(screen_seed) in seeds:
        raise ValueError("full seeds must be distinct and fresh from the screen seed")
    if int(promotion_limit) < 1 or int(promotion_limit) > len(ids):
        raise ValueError("promotion limit is outside the candidate denominator")
    payload = canonical_value(
        {
            "schema": CONVERSION_SCHEMA,
            "candidate_ids": ids,
            "screen": {
                "epochs": SCREEN_EPOCHS,
                "seed": int(screen_seed),
                "one_pristine_parent_per_seed": True,
                "failure_is_missing": True,
            },
            "promotion": {
                "limit": int(promotion_limit),
                "requires_completed_stable_screen": True,
                "selection_input": "SCREEN_STATUS_AND_STABILITY_ONLY",
            },
            "full": {
                "epochs": FULL_EPOCHS,
                "fresh_development_seeds": seeds,
                "one_pristine_parent_per_seed": True,
                "screen_outcomes_reused_as_effect": False,
            },
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "plan_digest": sha256_digest(payload)}


def derive_resource_deadline_seconds(
    screen_wall_time_ms: int | float,
    *,
    screen_epochs: int = SCREEN_EPOCHS,
    full_epochs: int = FULL_EPOCHS,
) -> int:
    """Project a full-run deadline from pre-outcome screen wall time."""

    wall_seconds = float(screen_wall_time_ms) / 1000.0
    if wall_seconds <= 0 or int(screen_epochs) <= 0 or int(full_epochs) <= 0:
        raise ValueError("screen telemetry and epoch counts must be positive")
    projected = ceil(
        60.0
        + RESOURCE_DEADLINE_MARGIN
        * (wall_seconds * int(full_epochs) / int(screen_epochs))
    )
    deadline = max(RESOURCE_DEADLINE_MIN_SECONDS, projected)
    if deadline > FULL_WATCHDOG_SECONDS:
        raise ValueError("projected deadline exceeds the frozen watchdog")
    return int(deadline)


def choose_stable_promotions(
    screen_results: Sequence[Mapping[str, Any]],
    *,
    promotion_limit: int,
) -> tuple[str, ...]:
    """Choose a fixed small full-run set from completed screen telemetry only."""

    eligible = []
    for row in screen_results:
        if (
            row.get("status") == "COMPLETED_MATCHED_SCREEN"
            and bool(row.get("stable"))
            and isinstance(row.get("candidate_id"), str)
        ):
            eligible.append(
                (
                    -float(row.get("screen_signal", 0.0)),
                    str(row["candidate_id"]),
                )
            )
    eligible.sort()
    by_id = {
        str(row["candidate_id"]): row
        for row in screen_results
        if isinstance(row.get("candidate_id"), str)
        and str(row["candidate_id"]) in {candidate_id for _signal, candidate_id in eligible}
    }
    selected: list[str] = []
    policies = sorted(
        {
            str(policy)
            for row in by_id.values()
            for policy in row.get("policy_owners", ())
        }
    )
    for policy in policies:
        policy_candidates = [
            item
            for item in eligible
            if policy in tuple(str(value) for value in by_id[item[1]].get("policy_owners", ()))
        ]
        if policy_candidates:
            candidate_id = policy_candidates[0][1]
            if candidate_id not in selected:
                selected.append(candidate_id)
    exploration = [
        item
        for item in eligible
        if bool(by_id[item[1]].get("shared_exploration"))
    ]
    fallback = exploration or eligible
    for _signal, candidate_id in fallback:
        if candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) >= int(promotion_limit):
            break
    return tuple(selected[: int(promotion_limit)])


def choose_resource_bounded_promotions(
    screen_results: Sequence[Mapping[str, Any]],
    *,
    promotion_limit: int,
    full_seeds: Sequence[int] = FULL_DEVELOPMENT_SEEDS,
    total_budget_seconds: int = FULL_RESOURCE_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Apply the fixed screen/policy order to the full-run resource budget."""

    ordered = choose_stable_promotions(
        screen_results,
        promotion_limit=int(promotion_limit),
    )
    seeds = tuple(int(seed) for seed in full_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("full seeds must be unique")
    parent_wall_time_ms = next(
        (
            row.get("parent_screen_cost_ms")
            for row in screen_results
            if row.get("parent_screen_cost_ms") is not None
        ),
        None,
    )
    if parent_wall_time_ms is None:
        raise ValueError("stable screen results require shared-parent wall time")
    parent_deadline = derive_resource_deadline_seconds(parent_wall_time_ms)
    parent_reserve = parent_deadline * len(seeds)
    used = parent_reserve
    promoted: list[str] = []
    censored: list[dict[str, Any]] = []
    deadlines: dict[str, int] = {}
    candidate_costs: dict[str, int] = {}
    by_id = {str(row["candidate_id"]): row for row in screen_results}
    for candidate_id in ordered:
        wall_time_ms = by_id[candidate_id].get("screen_cost_ms")
        if wall_time_ms is None:
            censored.append(
                {
                    "candidate_id": candidate_id,
                    "status": "RESOURCE_CENSORED_NOT_PROMOTED",
                    "reason": "SCREEN_WALL_TIME_MISSING",
                }
            )
            continue
        try:
            deadline = derive_resource_deadline_seconds(wall_time_ms)
        except ValueError:
            censored.append(
                {
                    "candidate_id": candidate_id,
                    "status": "RESOURCE_CENSORED_NOT_PROMOTED",
                    "reason": "DEADLINE_EXCEEDS_WATCHDOG",
                }
            )
            continue
        cost = deadline * len(seeds)
        deadlines[candidate_id] = deadline
        candidate_costs[candidate_id] = cost
        if used + cost <= int(total_budget_seconds):
            promoted.append(candidate_id)
            used += cost
        else:
            censored.append(
                {
                    "candidate_id": candidate_id,
                    "status": "RESOURCE_CENSORED_NOT_PROMOTED",
                    "reason": "FULL_RESOURCE_BUDGET_EXCEEDED",
                    "required_seconds": cost,
                    "remaining_seconds": max(0, int(total_budget_seconds) - used),
                }
            )
    return canonical_value(
        {
            "promoted_candidate_ids": promoted,
            "resource_censored_not_promoted": censored,
            "full_deadline_seconds_by_candidate": deadlines,
            "shared_parent_deadline_seconds": parent_deadline,
            "shared_parent_reserve_seconds": parent_reserve,
            "candidate_full_cost_seconds": candidate_costs,
            "resource_budget_seconds": int(total_budget_seconds),
            "resource_budget_used_seconds": used,
            "screen_priority_order": list(ordered),
        }
    )


def finalize_conversion_execution_plan(
    plan: Mapping[str, Any],
    screen_results: Sequence[Mapping[str, Any]],
    promoted_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach the fixed promotions and expose parent/candidate run counts."""

    candidate_ids = {str(value) for value in plan["candidate_ids"]}
    promoted = tuple(str(value) for value in promoted_ids)
    if len(set(promoted)) != len(promoted) or not set(promoted).issubset(candidate_ids):
        raise ValueError("promotion set is not a subset of the frozen candidate denominator")
    screen_seed = int(plan["screen"]["seed"])
    full_seeds = tuple(int(value) for value in plan["full"]["fresh_development_seeds"])
    resource = choose_resource_bounded_promotions(
        screen_results,
        promotion_limit=int(plan["promotion"]["limit"]),
        full_seeds=full_seeds,
    )
    if tuple(promoted) != tuple(resource["promoted_candidate_ids"]):
        raise ValueError("promotion set does not match the frozen resource screen rule")
    payload = canonical_value(
        {
            **dict(plan),
            "promotion": {
                **dict(plan["promotion"]),
                "promoted_candidate_ids": promoted,
                "resource_censored_not_promoted": resource[
                    "resource_censored_not_promoted"
                ],
                "full_deadline_seconds_by_candidate": resource[
                    "full_deadline_seconds_by_candidate"
                ],
                "shared_parent_deadline_seconds": resource[
                    "shared_parent_deadline_seconds"
                ],
                "shared_parent_reserve_seconds": resource[
                    "shared_parent_reserve_seconds"
                ],
                "candidate_full_cost_seconds": resource[
                    "candidate_full_cost_seconds"
                ],
                "resource_budget_seconds": resource["resource_budget_seconds"],
                "resource_budget_used_seconds": resource[
                    "resource_budget_used_seconds"
                ],
            },
            "run_counts": {
                "screen_candidate_runs": len(candidate_ids),
                "screen_parent_runs": 1,
                "full_candidate_runs": len(promoted) * len(full_seeds),
                "full_parent_runs": len(full_seeds),
                "shared_parent_runs": 1 + len(full_seeds),
                "screen_seed": screen_seed,
                "full_seeds": full_seeds,
            },
        }
    )
    return {**payload, "plan_digest": sha256_digest(payload)}


def run_fail_soft_batch(
    candidate_ids: Sequence[str],
    run_one: Any,
) -> tuple[dict[str, Any], ...]:
    """Run each arm independently; one local failure becomes missing."""

    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        normalized = str(candidate_id)
        try:
            result = run_one(normalized)
        except Exception as error:  # noqa: BLE001 - failure isolation is the contract.
            rows.append(
                {
                    "candidate_id": normalized,
                    "status": "MISSING",
                    "failure_class": type(error).__name__,
                }
            )
        else:
            rows.append(
                {
                    "candidate_id": normalized,
                    "status": "COMPLETED",
                    "result": result,
                }
            )
    return tuple(rows)


def build_stage_feasibility_head(
    stage_labels: Sequence[Mapping[str, Any]],
    *,
    policy_order: Sequence[str],
) -> dict[str, Any]:
    """Update only feasibility from the 17-arm stage labels.

    Effect remains strongly shrunk because Q5-A supplied only two complete
    Episodes; this function never consumes effect values or selects ideas.
    """

    labels = tuple(stage_labels)
    denominator = len(labels)
    if denominator != 17:
        raise ValueError("Q5 conversion feasibility head requires the frozen 17-arm labels")
    stages = ("CONSTRUCT", "MATERIALIZE", "QUALIFY", "RESOURCE_ADMITTED", "FULL_EPISODE")
    stage_rates = {
        stage: sum(row.get(stage) == "PASS" for row in labels) / denominator
        for stage in stages
    }
    by_policy: dict[str, dict[str, float]] = {}
    for policy in policy_order:
        rows = [row for row in labels if policy in tuple(row.get("policies", ()))]
        count = len(rows)
        by_policy[str(policy)] = {
            stage: (sum(row.get(stage) == "PASS" for row in rows) / count if count else 0.0)
            for stage in stages
        }
    values = tuple(
        round(by_policy[str(policy)]["RESOURCE_ADMITTED"], 12)
        for policy in policy_order
    )
    return canonical_value(
        {
            "schema": CONVERSION_SCHEMA,
            "stage_denominator": denominator,
            "stage_completion_head": stage_rates,
            "policy_stage_completion_head": by_policy,
            "policy_output": "NONUNIFORM" if len(set(values)) > 1 else "TIE",
            "effect_head": {
                "observed_full_episode_count": sum(row.get("FULL_EPISODE") == "PASS" for row in labels),
                "shrinkage": "STRONG",
                "claim_allowed": False,
            },
            "held_out_reads": 0,
            "outcome_leakage": False,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )


__all__ = [
    "CANDIDATE_LOCAL_ALLOWED_FILES",
    "CONVERSION_SCHEMA",
    "FULL_DEVELOPMENT_SEEDS",
    "FULL_EPOCHS",
    "FULL_RESOURCE_BUDGET_SECONDS",
    "FULL_WATCHDOG_SECONDS",
    "MAX_REPAIR_TURNS",
    "MECHANICAL_REPAIR_SCHEMA",
    "MECHANICAL_SHAPE_TEST_HINT",
    "RECBole_INTERFACE_CONTRACT",
    "SCREEN_EPOCHS",
    "build_conversion_execution_plan",
    "build_mechanical_repair_request",
    "build_stage_feasibility_head",
    "choose_resource_bounded_promotions",
    "choose_stable_promotions",
    "derive_resource_deadline_seconds",
    "finalize_conversion_execution_plan",
    "is_mechanical_repair_failure",
    "run_fail_soft_batch",
]
