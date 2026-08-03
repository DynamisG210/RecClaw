"""Q5-A idea/feasibility pilot contracts over the accepted Q5 foundation.

This module owns only the Q5-A manifest boundary.  Provider, implementer,
qualification, resource, and RecBole execution remain the accepted stage
consumers; this layer freezes the three fresh pools, two policy selections,
the shared exploration attribution, and the one-realization union.
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest, validate_sha256


Q5A_SCHEMA = "recclaw.research-line.q5a-idea-feasibility-pilot.v1"
Q5A_PREFREEZE_SCHEMA = "recclaw.research-line.q5a-prefreeze-manifest.v1"
Q5A_POOL_SCHEMA = "recclaw.research-line.q5a-fresh-origin-blind-pool.v1"
Q5A_SELECTION_SCHEMA = "recclaw.research-line.q5a-policy-selection.v1"
Q5A_EXPLORATION_SCHEMA = "recclaw.research-line.q5a-shared-exploration.v1"
Q5A_UNION_SCHEMA = "recclaw.research-line.q5a-realization-union.v1"
Q5A_STAGE_SCHEMA = "recclaw.research-line.q5a-stage-denominator.v1"
Q5A_POOL_COUNT = 3
Q5A_POOL_SIZE = 8
Q5A_POLICY_SELECTION_BUDGET = 2
Q5A_SHARED_EXPLORATION_PER_POOL = 1
Q5A_EXPLORATION_PROBABILITY = 0.15
Q5A_POLICIES = ("STATIC", "CURRENT_F1", "OUTCOME_AWARE")
Q5A_STAGES = (
    "MATERIALIZE",
    "CONSTRUCT",
    "QUALIFY",
    "RESOURCE_ADMITTED",
    "FULL_EPISODE",
)


class Q5AIdeaFeasibilityError(RuntimeError):
    """A Q5-A frozen input, attribution, or denominator drifted."""


def _digest(value: Any, *, label: str) -> str:
    try:
        validate_sha256(value, field_name=label)
    except Exception as error:
        raise Q5AIdeaFeasibilityError(f"{label} is not a SHA-256 digest") from error
    return str(value)


def _commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise Q5AIdeaFeasibilityError(f"{label} is not a Git commit identity")
    try:
        int(value, 16)
    except ValueError as error:
        raise Q5AIdeaFeasibilityError(f"{label} is not a Git commit identity") from error
    return value


def _candidate_id(row: Mapping[str, Any]) -> str:
    candidate_id = row.get("preoutcome_score", {}).get("spec_digest")
    if not isinstance(candidate_id, str):
        raise Q5AIdeaFeasibilityError("pool row lacks a pre-outcome spec digest")
    return candidate_id


def _pool_rows(pool: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate_pools = pool.get("candidate_pools")
    if not isinstance(candidate_pools, Mapping):
        raise Q5AIdeaFeasibilityError("Q5-A pool lacks candidate_pools")
    rows = [
        dict(row)
        for group in sorted(candidate_pools)
        for row in candidate_pools[group]
    ]
    if len(rows) != Q5A_POOL_SIZE:
        raise Q5AIdeaFeasibilityError("Q5-A pool must contain exactly eight specs")
    ids = [_candidate_id(row) for row in rows]
    if len(set(ids)) != Q5A_POOL_SIZE:
        raise Q5AIdeaFeasibilityError("Q5-A pool has duplicate OpenSpec identities")
    for row in rows:
        if row.get("stage") != "OPENSPEC_FROZEN":
            raise Q5AIdeaFeasibilityError("Q5-A pool contains a non-frozen row")
        if row.get("manual_candidate_patches", 0) != 0:
            raise Q5AIdeaFeasibilityError("Q5-A pool contains a manual candidate patch")
        if row.get("outcome_fields_consumed", []) not in ([], None):
            raise Q5AIdeaFeasibilityError("Q5-A pool consumed an outcome field")
    return rows


def build_q5a_prefreeze_manifest(
    *,
    campaign_id: str,
    foundation_commit: str,
    foundation_package_digest: str,
    source_tree_digest: str,
    common_execution: Mapping[str, Any],
    pool_seeds: Sequence[int],
) -> dict[str, Any]:
    """Freeze scale, identity, budget, and information boundaries before calls."""

    _commit(foundation_commit, label="foundation_commit")
    _digest(foundation_package_digest, label="foundation_package_digest")
    _digest(source_tree_digest, label="source_tree_digest")
    if len(pool_seeds) != Q5A_POOL_COUNT or len(set(pool_seeds)) != Q5A_POOL_COUNT:
        raise Q5AIdeaFeasibilityError("Q5-A requires three distinct pool seeds")
    execution = canonical_value(dict(common_execution))
    if execution.get("retries") != 0 or execution.get("held_out_reads") != 0:
        raise Q5AIdeaFeasibilityError("Q5-A common execution boundary is relaxed")
    payload = canonical_value(
        {
            "schema": Q5A_PREFREEZE_SCHEMA,
            "campaign_id": campaign_id,
            "foundation_commit": foundation_commit,
            "foundation_package_digest": foundation_package_digest,
            "source_tree_digest": source_tree_digest,
            "pool_count": Q5A_POOL_COUNT,
            "pool_size": Q5A_POOL_SIZE,
            "pool_seeds": tuple(int(seed) for seed in pool_seeds),
            "policy_order": Q5A_POLICIES,
            "policy_selection_budget": Q5A_POLICY_SELECTION_BUDGET,
            "shared_random_exploration_per_pool": Q5A_SHARED_EXPLORATION_PER_POOL,
            "exploration_probability": Q5A_EXPLORATION_PROBABILITY,
            "common_execution": execution,
            "freeze_order": (
                "PREFREEZE_MANIFEST_AND_BUDGET",
                "COMPLETE_POOL",
                "POLICY_SELECTION_AND_TIE_RECORD",
                "SELECTION_UNION",
                "ONE_SHARED_REALIZATION_PER_UNIQUE_OPENSPEC",
                "ALL_STAGE_DENOMINATORS_MISSINGNESS_AND_COST",
            ),
            "forbidden_adaptations": (
                "RETRY",
                "SUPPLEMENTAL_SAMPLE",
                "SUCCESS_FILTERING",
                "CANDIDATE_SPECIFIC_REPAIR",
                "HELD_OUT_READ",
                "OUTCOME_LEAKAGE",
            ),
            "stage_order": Q5A_STAGES,
            "outcome_metric_primary": "NOT_NDCG_Q5A_FEASIBILITY_ONLY",
            "held_out_reads": 0,
            "outcome_leakage": False,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "prefreeze_digest": sha256_digest(payload)}


def build_q5a_pool_manifest(
    *,
    pool_index: int,
    pool_seed: int,
    pool: Mapping[str, Any],
    provider_usage: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one complete origin-blind eight-slot Provider pool."""

    if pool_index not in range(1, Q5A_POOL_COUNT + 1):
        raise Q5AIdeaFeasibilityError("Q5-A pool index is outside the frozen scale")
    rows = _pool_rows(pool)
    if int(provider_usage.get("physical_calls", -1)) != Q5A_POOL_SIZE:
        raise Q5AIdeaFeasibilityError("Q5-A Provider denominator is not eight")
    if int(provider_usage.get("retries", -1)) != 0:
        raise Q5AIdeaFeasibilityError("Q5-A Provider retries are forbidden")
    payload = canonical_value(
        {
            "schema": Q5A_POOL_SCHEMA,
            "pool_index": pool_index,
            "pool_seed": int(pool_seed),
            "candidate_pools": pool.get("candidate_pools"),
            "candidate_count": len(rows),
            "provider_denominator": Q5A_POOL_SIZE,
            "provider_usage": dict(provider_usage),
            "origin_blind": True,
            "implementation_or_qualification_outcomes_present_when_written": 0,
            "outcome_fields_consumed": [],
            "held_out_reads": 0,
            "retries": 0,
            "candidate_replacement": False,
            "successful_candidate_filtering": False,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "pool_digest": sha256_digest(payload)}


def build_q5a_selection_manifest(
    *,
    pool: Mapping[str, Any],
    pool_digest: str,
    policy_name: str,
    selected_candidate_ids: Sequence[str],
    candidate_probabilities: Mapping[str, float],
    tie_set: Sequence[str],
    selected_by: Mapping[str, str],
    selection_seed: int | None = None,
    exploration_draw: float | None = None,
    source_policy_digest: str | None = None,
    selection_semantics: str = "FROZEN_TOP_TWO",
    tie_tolerance: float | None = None,
    prior_usage: str = "NONE",
) -> dict[str, Any]:
    """Freeze two policy selections with explicit tie/probability accounting."""

    if policy_name not in Q5A_POLICIES:
        raise Q5AIdeaFeasibilityError("unknown Q5-A policy")
    _digest(pool_digest, label="pool_digest")
    rows = _pool_rows(pool)
    candidate_ids = {_candidate_id(row) for row in rows}
    selected = tuple(str(value) for value in selected_candidate_ids)
    if len(selected) != Q5A_POLICY_SELECTION_BUDGET or len(set(selected)) != len(selected):
        raise Q5AIdeaFeasibilityError("Q5-A policy must select exactly two unique specs")
    if not set(selected).issubset(candidate_ids):
        raise Q5AIdeaFeasibilityError("Q5-A policy selected a spec outside its pool")
    probabilities = {str(key): float(value) for key, value in candidate_probabilities.items()}
    if set(probabilities) != candidate_ids or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities.values()
    ):
        raise Q5AIdeaFeasibilityError("Q5-A selection probabilities are incomplete")
    if not math.isclose(sum(probabilities.values()), Q5A_POLICY_SELECTION_BUDGET):
        raise Q5AIdeaFeasibilityError("Q5-A selection probabilities do not sum to two")
    if not set(tie_set).issubset(candidate_ids):
        raise Q5AIdeaFeasibilityError("Q5-A tie set contains an unknown spec")
    if set(selected_by) != set(selected):
        raise Q5AIdeaFeasibilityError("Q5-A selected_by is incomplete")
    payload = canonical_value(
        {
            "schema": Q5A_SELECTION_SCHEMA,
            "policy_name": policy_name,
            "pool_digest": pool_digest,
            "pool_size": len(rows),
            "selection_budget": Q5A_POLICY_SELECTION_BUDGET,
            "selected_candidate_ids": selected,
            "candidate_probabilities": probabilities,
            "selection_probability_sum": round(sum(probabilities.values()), 12),
            "top_tie_set": tuple(sorted(str(value) for value in tie_set)),
            "selected_by": dict(selected_by),
            "selection_seed": selection_seed,
            "exploration_draw": exploration_draw,
            "source_policy_digest": source_policy_digest,
            "selection_semantics": selection_semantics,
            "tie_tolerance": tie_tolerance,
            "prior_usage": prior_usage,
            "outcome_fields_consumed": [],
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "selection_digest": sha256_digest(payload)}


def build_q5a_shared_exploration(
    *, pool: Mapping[str, Any], pool_digest: str, random_seed: int
) -> dict[str, Any]:
    """Select exactly one common random exploration spec for all policies."""

    _digest(pool_digest, label="pool_digest")
    rows = _pool_rows(pool)
    rng = random.Random(int(random_seed))
    random_draw = rng.random()
    selected_row = rows[rng.randrange(len(rows))]
    payload = canonical_value(
        {
            "schema": Q5A_EXPLORATION_SCHEMA,
            "pool_digest": pool_digest,
            "random_seed": int(random_seed),
            "exploration_probability": Q5A_EXPLORATION_PROBABILITY,
            "random_draw": round(random_draw, 12),
            "selected_candidate_id": _candidate_id(selected_row),
            "sampling": "UNIFORM_OVER_COMPLETE_POOL",
            "shared_by_policies": Q5A_POLICIES,
            "outcome_fields_consumed": [],
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "exploration_digest": sha256_digest(payload)}


def build_q5a_realization_union(
    *,
    prefreeze: Mapping[str, Any],
    pools: Sequence[Mapping[str, Any]],
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    explorations: Sequence[Mapping[str, Any]],
    realizations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze policy attribution before one physical realization per spec."""

    if prefreeze.get("schema") != Q5A_PREFREEZE_SCHEMA:
        raise Q5AIdeaFeasibilityError("Q5-A prefreeze schema drift")
    if len(pools) != Q5A_POOL_COUNT or len(explorations) != Q5A_POOL_COUNT:
        raise Q5AIdeaFeasibilityError("Q5-A union lacks all three pools")
    pool_by_digest = {}
    for pool in pools:
        digest = str(pool.get("pool_digest"))
        _digest(digest, label="pool_digest")
        pool_by_digest[digest] = pool
    attribution: dict[str, dict[str, tuple[str, ...]]] = {}
    selected_ids: set[str] = set()
    for policy in Q5A_POLICIES:
        rows: dict[str, tuple[str, ...]] = {}
        for selection in selections.get(policy, ()):
            pool_digest = str(selection["pool_digest"])
            if pool_digest not in pool_by_digest:
                raise Q5AIdeaFeasibilityError("selection names an unknown Q5-A pool")
            ids = tuple(str(value) for value in selection["selected_candidate_ids"])
            if len(ids) != Q5A_POLICY_SELECTION_BUDGET:
                raise Q5AIdeaFeasibilityError("selection budget drifted from two")
            rows[pool_digest] = ids
            selected_ids.update(ids)
        if len(rows) != Q5A_POOL_COUNT:
            raise Q5AIdeaFeasibilityError("policy selection is incomplete across pools")
        attribution[policy] = rows
    exploration_by_pool = {}
    for exploration in explorations:
        pool_digest = str(exploration["pool_digest"])
        if pool_digest not in pool_by_digest or pool_digest in exploration_by_pool:
            raise Q5AIdeaFeasibilityError("shared exploration attribution is invalid")
        candidate_id = str(exploration["selected_candidate_id"])
        if candidate_id not in {_candidate_id(row) for row in _pool_rows(pool_by_digest[pool_digest])}:
            raise Q5AIdeaFeasibilityError("shared exploration is outside its pool")
        exploration_by_pool[pool_digest] = candidate_id
        selected_ids.add(candidate_id)
    if len(exploration_by_pool) != Q5A_POOL_COUNT:
        raise Q5AIdeaFeasibilityError("Q5-A requires one exploration spec per pool")
    realization_rows = list(realizations or ())
    by_spec: dict[str, Mapping[str, Any]] = {}
    for realization in realization_rows:
        spec_digest = str(realization.get("research_spec_digest"))
        if spec_digest in by_spec and by_spec[spec_digest] != realization:
            raise Q5AIdeaFeasibilityError("one OpenSpec has multiple realizations")
        by_spec[spec_digest] = realization
    if realization_rows and set(by_spec) != selected_ids:
        raise Q5AIdeaFeasibilityError("realization union does not match selection union")
    payload = canonical_value(
        {
            "schema": Q5A_UNION_SCHEMA,
            "prefreeze_digest": prefreeze.get("prefreeze_digest"),
            "pool_count": len(pools),
            "selected_unique_spec_count": len(selected_ids),
            "selected_unique_spec_digests": tuple(sorted(selected_ids)),
            "policy_attribution": attribution,
            "shared_exploration_by_pool": exploration_by_pool,
            "realizations": tuple(by_spec[key] for key in sorted(by_spec)),
            "implementations_per_unique_openspec": 1,
            "outcome_key_rule": "ONE_OUTCOME_PER_CANDIDATE_PACKAGE_PER_SEED",
            "outcome_fields_consumed_before_union": [],
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "union_digest": sha256_digest(payload)}


def build_q5a_stage_denominator(
    *,
    realization_rows: Sequence[Mapping[str, Any]],
    stage_order: Sequence[str] = Q5A_STAGES,
    realization_denominator: int | None = None,
) -> dict[str, Any]:
    """Emit every frozen stage row, including explicit missingness and cost."""

    if tuple(stage_order) != Q5A_STAGES:
        raise Q5AIdeaFeasibilityError("Q5-A stage order drift")
    denominator = len(realization_rows) if realization_denominator is None else int(realization_denominator)
    if denominator < 0:
        raise Q5AIdeaFeasibilityError("Q5-A realization denominator cannot be negative")
    stats = []
    for stage in Q5A_STAGES:
        rows = [row for row in realization_rows if row.get("stage") == stage]
        successes = sum(row.get("status") in {"PASS", "SUCCESS", "COMPLETED"} for row in rows)
        costs = [int(row["cost_ms"]) for row in rows if row.get("cost_ms") is not None]
        stats.append(
            {
                "stage": stage,
                "denominator": denominator,
                "observed_count": len(rows),
                "success_count": successes,
                "empirical_completion_rate": (
                    successes / len(rows) if rows else None
                ),
                "missingness_count": denominator - len(rows),
                "cost_ms_total": sum(costs),
                "cost_observation_count": len(costs),
            }
        )
    payload = canonical_value(
        {
            "schema": Q5A_STAGE_SCHEMA,
            "stage_order": Q5A_STAGES,
            "realization_denominator": denominator,
            "stage_stats": stats,
            "missingness_is_not_zero": True,
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "stage_digest": sha256_digest(payload)}


__all__ = [
    "Q5A_EXPLORATION_PROBABILITY",
    "Q5A_POOL_COUNT",
    "Q5A_POOL_SIZE",
    "Q5A_POLICY_SELECTION_BUDGET",
    "Q5A_POLICIES",
    "Q5A_PREFREEZE_SCHEMA",
    "Q5A_SHARED_EXPLORATION_PER_POOL",
    "Q5A_STAGES",
    "Q5AIdeaFeasibilityError",
    "build_q5a_pool_manifest",
    "build_q5a_prefreeze_manifest",
    "build_q5a_realization_union",
    "build_q5a_selection_manifest",
    "build_q5a_shared_exploration",
    "build_q5a_stage_denominator",
]
