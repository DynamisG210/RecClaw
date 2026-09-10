"""Minimal shared inputs for the unified Research Line runtime.

These objects join existing production components; they do not replace the
OpenSpec, capability, Search, Episode, memory, or policy contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
import re
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV4,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    OpenResearchSpecV1,
)
from .bl_icf_realization import (
    MechanismImplementationCompanionV1,
    ProviderMechanismProgramProposalV1,
)
from .single_parent_search import (
    bound_parent_from_context,
    is_single_parent_context,
    project_single_parent_objective,
)


class ResearchLineInterfaceError(ValueError):
    """Raised when a shared production input is internally inconsistent."""


PRODUCER_VIEW_HISTORY_LIMIT = 1
_PROVIDER_NEGATIVE_EVIDENCE_LIMIT = 8
# Keep the projection bounded, but leave enough room for the four most recent
# executed mechanisms and their development outcomes.  The previous 2 KiB
# limit rejected round two as soon as result-faithful feedback was included.
PRODUCER_VIEW_JSON_BUDGET_BYTES = 8_192


def _provider_scalar_fields(
    value: Any,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        field: value[field]
        for field in fields
        if field in value
        and (
            isinstance(value[field], (str, int, float, bool))
            or value[field] is None
            or (
                isinstance(value[field], (tuple, list))
                and len(value[field]) <= 16
                and all(
                    isinstance(item, (str, int, float, bool))
                    or item is None
                    for item in value[field]
                )
            )
        )
    }


def _provider_latest(
    value: Any,
    fields: tuple[str, ...],
) -> dict[str, Any] | None:
    rows = list(value) if isinstance(value, (tuple, list)) else []
    for row in reversed(rows[-PRODUCER_VIEW_HISTORY_LIMIT:]):
        projected = _provider_scalar_fields(row, fields)
        if projected:
            return projected
    return None


def _provider_history_counts(
    memory: Mapping[str, Any],
    fields: tuple[str, ...],
) -> dict[str, int]:
    return {
        field: len(memory[field])
        for field in fields
        if isinstance(memory.get(field), (tuple, list))
    }


def research_producer_roles(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Actual research calls for the frozen controller mode; absent means legacy."""
    if metadata.get("research_mode") == "director_sequential":
        return ("frontier_architect",)
    return DISCOVERY_PRODUCERS


def _provider_alias_fields(
    value: Any,
    aliases: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        output: value[source]
        for output, source in aliases
        if source in value
        and (
            isinstance(value[source], (str, int, float, bool))
            or value[source] is None
            or (
                isinstance(value[source], (tuple, list))
                and len(value[source]) <= 16
                and all(
                    isinstance(item, (str, int, float, bool))
                    or item is None
                    for item in value[source]
                )
            )
        )
    }


def _provider_baseline_objective(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    if is_single_parent_context(value):
        return project_single_parent_objective(value)
    projected: dict[str, Any] = {}
    for field_name in ("basic_baseline", "strong_baseline"):
        baseline = _provider_scalar_fields(
            value.get(field_name),
            (
                "name",
                "role",
                "value",
                "reference_kind",
                "seed",
                "seeds",
                "aggregation",
            ),
        )
        if baseline:
            projected[field_name] = baseline
    pack = value.get("baseline_pack")
    if isinstance(pack, (tuple, list)):
        projected_pack = tuple(
            _provider_scalar_fields(
                item,
                ("rank", "name", "role", "mechanism"),
            )
            for item in pack[:8]
        )
        projected_pack = tuple(item for item in projected_pack if item)
        if projected_pack:
            projected["baseline_pack"] = projected_pack
    semantics = _provider_scalar_fields(
        value.get("decision_semantics"),
        ("basic_floor", "dynamic_frontier", "strong_target", "milestones"),
    )
    if semantics:
        projected["baseline_decision_semantics"] = semantics
    return projected


def _provider_numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _annotate_provider_result_against_baselines(
    latest_result: dict[str, Any],
    baseline_objective: Mapping[str, Any],
) -> None:
    candidate_value = _provider_numeric_value(latest_result.get("value"))
    if candidate_value is None:
        return
    parent = baseline_objective.get("parent_anchor")
    if isinstance(parent, Mapping):
        paired_metric = parent.get("paired_metric")
        parent_value = _provider_numeric_value(
            paired_metric.get("value")
            if isinstance(paired_metric, Mapping)
            else None
        )
        if parent_value is None:
            return
        delta = candidate_value - parent_value
        latest_result["delta_vs_root"] = delta
        latest_result["root_beaten"] = delta > 0.0
        latest_result["baseline_position"] = (
            "BEATS_FROZEN_ROOT"
            if delta > 0.0
            else "MATCHES_FROZEN_ROOT"
            if delta == 0.0
            else "BELOW_FROZEN_ROOT"
        )
        return
    basic = baseline_objective.get("basic_baseline")
    strong = baseline_objective.get("strong_baseline")
    basic_value = _provider_numeric_value(
        basic.get("value") if isinstance(basic, Mapping) else None
    )
    strong_value = _provider_numeric_value(
        strong.get("value") if isinstance(strong, Mapping) else None
    )
    if basic_value is not None:
        latest_result["delta_vs_basic"] = candidate_value - basic_value
    if strong_value is not None:
        latest_result["delta_vs_strong"] = candidate_value - strong_value
        latest_result["strong_baseline_reached"] = candidate_value >= strong_value
    if basic_value is not None and candidate_value < basic_value:
        latest_result["baseline_position"] = "BELOW_BASIC_BASELINE"
    elif strong_value is not None and candidate_value >= strong_value:
        latest_result["baseline_position"] = "AT_OR_ABOVE_STRONG_BASELINE"
    elif latest_result.get("frontier_updated") is True:
        latest_result["baseline_position"] = "FRONTIER_ADVANCE_BELOW_STRONG"
    elif basic_value is not None:
        latest_result["baseline_position"] = "VIABLE_NO_FRONTIER_ADVANCE"


def _provider_task_head(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    tasks = value.get("tasks", ())
    tasks = list(tasks) if isinstance(tasks, (tuple, list)) else []
    actionable = [
        task
        for task in tasks
        if isinstance(task, Mapping)
        and str(task.get("status", "")) in {"PENDING", "ACTIVE"}
    ]
    operation_order = {
        "MATCHED_CONTROL": 0,
        "REPAIR": 0,
        "MECHANISM_OFF": 1,
        "NEW_SEED": 2,
        "REPRODUCE": 3,
        "MOVE_ON": 4,
    }
    actionable.sort(
        key=lambda task: (
            operation_order.get(str(task.get("operation")), 99),
            -float(task.get("priority", 0.0)),
            int(task.get("created_round", 0)),
            str(task.get("task_id", "")),
        )
    )
    if not actionable:
        return None
    projected = _provider_scalar_fields(
        actionable[0],
        (
            "task_id",
            "operation",
            "candidate_id",
            "parent_candidate_id",
            "comparator_identity",
            "mechanism_program_digest",
            "required_seed_or_control",
            "status",
        ),
    )
    metadata = actionable[0].get("metadata")
    if isinstance(metadata, Mapping):
        projected.update(
            _provider_scalar_fields(
                metadata,
                (
                    "next_discriminative_test",
                    "mechanism_axis",
                    "mechanism_axis_footprint",
                    "evidence_class",
                    "failure_class",
                    "unresolved_confounding",
                    "core_mechanism_contrast",
                    "causal_credit_allowed",
                    "capability_family",
                    "expected_observable",
                    "falsifier",
                    "task_operation",
                    "execution_state",
                    "binding_requirement",
                    "frontier_candidate_id",
                    "frontier_candidate_semantic_digest",
                    "frontier_candidate_program_digest",
                    "frontier_parent_candidate_id",
                    "frontier_comparator_identity",
                    "effective_experiment_digest",
                    "effective_family_digest",
                ),
            )
        )
    return projected


def _provider_engineering_diagnostic(
    latest_feedback: Any,
) -> dict[str, Any] | None:
    """Project one actionable engineering fingerprint without scientific evidence."""

    if not isinstance(latest_feedback, Mapping):
        return None
    nested = latest_feedback.get("engineering_diagnostic")
    if not isinstance(nested, Mapping):
        nested = {}
    detail = latest_feedback.get("diagnostic_detail")
    if not isinstance(detail, Mapping):
        detail = {}

    summary: dict[str, Any] = {}
    for source in (latest_feedback, nested):
        summary.update(
            _provider_scalar_fields(
                source,
                (
                    "failure_class",
                    "failure_code",
                    "failure_fingerprint",
                    "failure_detail_digest",
                ),
            )
        )
    summary.update(
        _provider_scalar_fields(
            detail,
            ("reason", "stage", "failure_code", "producer_role"),
        )
    )
    if latest_feedback.get("schema") == (
        "recclaw.research-line.candidate-no-metric-feedback.v1"
    ):
        candidate_failures = latest_feedback.get("candidate_failures")
        if isinstance(candidate_failures, (tuple, list)):
            compact_failures: list[dict[str, Any]] = []
            for candidate_failure in candidate_failures[:8]:
                if not isinstance(candidate_failure, Mapping):
                    continue
                failure = candidate_failure.get("failure")
                failure = failure if isinstance(failure, Mapping) else {}
                compact_failure = _provider_scalar_fields(
                    candidate_failure,
                    ("candidate_id", "producer_role"),
                )
                primitive_ids = candidate_failure.get("primitive_ids")
                if isinstance(primitive_ids, (tuple, list)):
                    compact_failure["primitive_ids"] = tuple(
                        str(value) for value in primitive_ids[:16]
                    )
                failure_detail = _provider_scalar_fields(
                    failure,
                    (
                        "failure_class",
                        "failure_scope",
                        "stage",
                        "reason_code",
                    ),
                )
                message = failure.get("message")
                if isinstance(message, str) and message:
                    failure_detail["message"] = message[:1200]
                implicated_methods = failure.get("implicated_methods")
                if isinstance(implicated_methods, (tuple, list)):
                    failure_detail["implicated_methods"] = tuple(
                        str(value) for value in implicated_methods[:16]
                    )
                if failure_detail:
                    compact_failure["failure"] = failure_detail
                if compact_failure:
                    compact_failures.append(canonical_value(compact_failure))
            if compact_failures:
                summary["candidate_failures"] = tuple(compact_failures)
    if detail and "failure_detail_digest" not in summary:
        summary["failure_detail_digest"] = sha256_digest(detail)
    return summary or None


def _provider_guard_feedback(latest_feedback: Any) -> dict[str, Any] | None:
    """Project only decision-relevant Guard state into the next Provider call.

    The durable Guard response contains provenance, seed lists, closures, and
    control records.  Provider needs the scientific decision, not that private
    ledger.  Counts keep the prompt bounded and avoid exposing seed identities
    that could encourage outcome-conditioned proposal wording.
    """

    if not isinstance(latest_feedback, Mapping):
        return None
    summary = latest_feedback.get("evidence_summary")
    if not isinstance(summary, Mapping):
        return None
    directive = latest_feedback.get("validation_directive")
    if not isinstance(directive, Mapping):
        directive = {}
    projection = latest_feedback.get("control_projection")
    if not isinstance(projection, Mapping):
        projection = {}

    projected = _provider_alias_fields(
        summary,
        (
            ("axis", "mechanism_axis"),
            ("conclusion", "scientific_conclusion_strength"),
            ("attempt", "current_attempt_class"),
            ("mean_delta", "mean_comparator_delta"),
        ),
    )
    for output, source in (
        ("verified", "verified_seed_ids"),
        ("missing", "missing_seed_ids"),
    ):
        values = summary.get(source)
        if isinstance(values, (tuple, list)):
            projected[output] = len(values)
    action = directive.get("action")
    if isinstance(action, str) and action and action != "NO_ADDITIONAL_EVIDENCE_ALLOCATED":
        projected["next"] = action
    confidence = projection.get("confidence_weight")
    question = summary.get("discriminative_question")
    if isinstance(question, Mapping):
        projected["discriminative_question"] = canonical_value(dict(question))
    research_feedback = summary.get("research_feedback")
    if isinstance(research_feedback, Mapping):
        projected["research_feedback"] = canonical_value(dict(research_feedback))
    if latest_feedback.get("research_update_mode") == "PRESERVE_NATIVE_RESEARCH":
        projected["use"] = "INFORM_NEXT_CANDIDATE_NOT_A_VALIDATION_PREREQUISITE"
    if (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
    ):
        projected["confidence"] = float(confidence)
    return projected or None


def _provider_mechanism_negative_evidence(
    value: Any,
) -> tuple[dict[str, Any], ...]:
    """Expose a bounded recent window from the cumulative Guard-negative ledger."""

    if not isinstance(value, (tuple, list)):
        return ()
    projected: list[dict[str, Any]] = []
    for row in value[-_PROVIDER_NEGATIVE_EVIDENCE_LIMIT:]:
        summary = _provider_alias_fields(
            row,
            (
                ("round", "round_index"),
                ("candidate", "candidate_id"),
                ("axis", "mechanism_axis"),
                ("delta", "comparator_delta"),
                ("claim", "claim_state"),
                ("confidence", "confidence_weight"),
                ("evidence", "evidence_class"),
                ("weight", "negative_evidence_weight"),
                ("promotion_allowed", "promotion_allowed"),
            ),
        )
        if summary:
            projected.append(canonical_value(summary))
    return tuple(projected)


def _provider_guard_attribution(value: Any) -> dict[str, Any] | None:
    """Expose the latest descriptive control result without causal inflation."""

    projected = _provider_alias_fields(
        value,
        (
            ("axis", "source_mechanism_axis"),
            ("axis", "mechanism_axis"),
            ("state", "attribution_state"),
            ("delta", "incremental_delta"),
            ("pairing", "pairing_class"),
            ("evidence", "evidence_class"),
            ("contrast", "core_mechanism_contrast"),
            ("next", "next_discriminative_task"),
            ("causal_credit_allowed", "causal_credit_allowed"),
        ),
    )
    for output, source in (
        ("footprint", "mechanism_axis_footprint"),
        ("confounds", "unresolved_confounding"),
    ):
        values = value.get(source) if isinstance(value, Mapping) else None
        if isinstance(values, (tuple, list)):
            projected[output] = tuple(str(item) for item in values[:16])
    return projected or None


def _provider_recent_experiments(value: Any) -> tuple[dict[str, Any], ...]:
    """Expose a small recent semantic/effect window to the Provider.

    Opaque identity digests remain in machine memory for exact de-duplication;
    spending prompt tokens on them does not help the research agent choose a
    better mechanism.  Longer-horizon effect direction is carried separately
    by ``directional_search_utility``.
    """

    if not isinstance(value, (tuple, list)):
        return ()
    projected: list[dict[str, Any]] = []
    seen_observations: set[str] = set()
    for row in reversed(value):
        if not isinstance(row, Mapping):
            continue
        # A family groups different interventions. A later failed variant must
        # not erase an earlier variant's measured result. Only exact duplicate
        # observations are redundant in this view.
        observation = row.get("physical_observation_digest")
        observation_identity = str(observation) if isinstance(observation, str) else ""
        if observation_identity and observation_identity in seen_observations:
            continue
        if observation_identity:
            seen_observations.add(observation_identity)
        metrics = row.get("development_metrics")
        ndcg_at_10 = None
        if isinstance(metrics, Mapping):
            for key, metric_value in metrics.items():
                if str(key).lower().replace("_", "") in {
                    "ndcg@10",
                    "ndcg10",
                }:
                    ndcg_at_10 = metric_value
                    break
        primitives = row.get("primitive_ids")
        primitives = (
            tuple(str(item) for item in primitives[:10])
            if isinstance(primitives, (tuple, list))
            else ()
        )
        failure = row.get("failure")
        efficiency = row.get("implementation_efficiency_repair_context")
        efficiency = (
            {
                key: efficiency[key]
                for key in (
                    "reason_code",
                    "next_attempt_scope",
                    "preserve_mechanism_program",
                    "epochs_completed",
                    "best_observed_epoch",
                    "elapsed_seconds",
                    "worker_ceiling_seconds",
                    "mean_train_epoch_wall_time_ms",
                    "mean_eval_epoch_wall_time_ms",
                    "full_train_batches_per_epoch",
                    "full_validation_batches_per_eval",
                )
                if key in efficiency
            }
            if isinstance(efficiency, Mapping)
            else None
        )
        experiment_summary = {
            "round": row.get("round_index"),
            "primitives": primitives,
            "ndcg@10": ndcg_at_10,
            "delta": row.get("comparator_delta"),
            "frontier_updated": row.get("frontier_updated"),
            "frontier_delta": row.get("frontier_delta"),
            "parent": row.get("construction_parent"),
            "cost": row.get("measured_execution_cost") or None,
            "outcome": row.get("outcome"),
            "disposition": row.get("engineering_disposition"),
            "axis": row.get("mechanism_axis"),
            "mechanism": (
                str(row["core_mechanism_contrast"])[:600]
                if row.get("core_mechanism_contrast") else None
            ),
            "footprint": tuple(
                str(item)
                for item in row.get("mechanism_axis_footprint", ())[:16]
            )
            if isinstance(row.get("mechanism_axis_footprint"), (tuple, list))
            else (),
            "evidence": row.get("evidence_class"),
            "confounds": tuple(
                str(item)
                for item in row.get("unresolved_confounding", ())[:8]
            )
            if isinstance(row.get("unresolved_confounding"), (tuple, list))
            else (),
            "next_task": row.get("next_discriminative_task"),
            "failure": (
                failure.get("reason_code")
                if isinstance(failure, Mapping)
                else None
            ),
            "efficiency": efficiency,
        }
        projected.append(
            canonical_value(
                {
                    field_name: field_value
                    for field_name, field_value in experiment_summary.items()
                    if field_value is not None
                    and field_value != ()
                    and field_value != []
                }
            )
        )
        if len(projected) >= 12:
            break
    return tuple(reversed(projected))


def _provider_program_primitive_ids(value: Any) -> tuple[str, ...]:
    """Return the stable primitive vocabulary used by one mechanism program."""

    if not isinstance(value, Mapping):
        return ()
    payload = value.get("program_payload")
    if not isinstance(payload, Mapping):
        return ()
    components = payload.get("components")
    if not isinstance(components, (tuple, list)):
        return ()
    return tuple(
        dict.fromkeys(
            str(component["primitive_id"])
            for component in components
            if isinstance(component, Mapping)
            and isinstance(component.get("primitive_id"), str)
            and component["primitive_id"]
        )
    )


def _provider_pre_metric_failure_feedback(
    experiments: tuple[dict[str, Any], ...],
    *,
    parent_program: Any = None,
) -> dict[str, Any] | None:
    """Summarize recurring executable failures without rejecting the mechanism.

    Pre-metric failures are implementation/resource evidence, not measured
    recommendation evidence.  Grouping them by the primitive delta gives the
    research agent enough information to stop resubmitting the same executable
    realization while preserving its freedom to keep the hypothesis and change
    the implementation, or to explore another mechanism family.
    """

    parent_primitives = set(_provider_program_primitive_ids(parent_program))
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for experiment in experiments:
        primitives = experiment.get("primitives")
        if not isinstance(primitives, (tuple, list)):
            continue
        primitive_family = tuple(
            dict.fromkeys(
                str(primitive)
                for primitive in primitives
                if str(primitive)
            )
        )
        if not primitive_family:
            continue
        changed_primitives = tuple(
            primitive
            for primitive in primitive_family
            if primitive not in parent_primitives
        )
        family_key = changed_primitives or primitive_family
        group = groups.setdefault(
            family_key,
            {
                "attempts": 0,
                "failure_counts": {},
                "latest_round": None,
                "metric_bearing_attempts": 0,
            },
        )
        metric = _provider_numeric_value(experiment.get("ndcg@10"))
        if metric is not None:
            group["metric_bearing_attempts"] += 1
            continue
        failure = experiment.get("failure")
        if not isinstance(failure, str) or not failure:
            continue
        group["attempts"] += 1
        failure_counts = group["failure_counts"]
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
        round_index = experiment.get("round")
        if isinstance(round_index, int) and not isinstance(round_index, bool):
            latest_round = group["latest_round"]
            group["latest_round"] = (
                round_index
                if latest_round is None
                else max(latest_round, round_index)
            )

    repeated: list[dict[str, Any]] = []
    for primitive_family, group in groups.items():
        if group["attempts"] < 2:
            continue
        repeated.append(
            canonical_value(
                {
                    "changed_primitives": primitive_family,
                    "metricless_failures": group["attempts"],
                    "failure_counts": tuple(
                        {
                            "reason": reason,
                            "attempts": count,
                        }
                        for reason, count in sorted(
                            group["failure_counts"].items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    ),
                    "latest_round": group["latest_round"],
                    "recent_metric_bearing_attempts": group[
                        "metric_bearing_attempts"
                    ],
                }
            )
        )
    if not repeated:
        return None
    repeated.sort(
        key=lambda item: (
            -(item.get("latest_round") or -1),
            -item["metricless_failures"],
            tuple(item["changed_primitives"]),
        )
    )
    return canonical_value(
        {
            "semantics": "EXECUTION_EVIDENCE_NOT_MECHANISM_EFFECT_EVIDENCE",
            "repeated_families": tuple(repeated[:6]),
        }
    )




def _provider_related_mechanism_experiments(
    history: Any,
    recent: Any,
    relevant_axes: Sequence[str] = (),
    *,
    mechanism_queries: Sequence[str] = (),
    before_round: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Retrieve older related/repeated mechanisms; do not expose the full ledger."""

    if not isinstance(history, (tuple, list)):
        return ()
    recent_rounds = [
        row["round"] for row in _provider_recent_experiments(recent)
        if isinstance(row.get("round"), int)
    ]
    cutoff = min(recent_rounds) if recent_rounds else float("inf")
    query_terms = set(re.findall(r"[a-z][a-z0-9_]{3,}", " ".join(mechanism_queries).lower()))
    families: dict[str, list[Mapping[str, Any]]] = {}
    for row in history:
        if not isinstance(row, Mapping):
            continue
        family = row.get("effective_family_digest")
        round_index = row.get("round_index")
        if (
            isinstance(family, str)
            and isinstance(round_index, int)
            and (before_round is None or round_index < before_round)
        ):
            families.setdefault(family, []).append(row)
    # Task-related families first, then repeated hypotheses. This ranks context
    # relevance only; it neither selects candidates nor bans a research direction.
    ordered = sorted(
        families.values(),
        key=lambda rows: (
            len(query_terms.intersection(re.findall(
                r"[a-z][a-z0-9_]{3,}",
                " ".join(str(row.get("core_mechanism_contrast", "")) for row in rows).lower(),
            ))),
            any(row.get("mechanism_axis") in relevant_axes for row in rows),
            len(rows) > 1,
            max(row["round_index"] for row in rows),
        ),
        reverse=True,
    )
    selected: list[Mapping[str, Any]] = []
    for rows in ordered:
        older = [row for row in rows if row["round_index"] < cutoff]
        if not older:
            continue
        selected.extend(older[-min(8, 12 - len(selected)):])
        if len(selected) >= 12:
            break
    return _provider_recent_experiments(sorted(selected, key=lambda row: row["round_index"]))


def observation_updates_search_utility(
    observation: Mapping[str, Any],
    *,
    neutralizations: Sequence[Mapping[str, Any]] = (),
) -> bool:
    """One authority for whether retained evidence may steer search utility."""

    identity_fields = (
        "candidate_semantic_digest",
        "observation_seed",
        "round_index",
    )
    for neutralization in neutralizations:
        if not all(neutralization.get(field_name) is not None for field_name in identity_fields):
            continue
        if all(
            neutralization.get(field_name) == observation.get(field_name)
            for field_name in identity_fields
        ):
            return False
    explicit = observation.get("search_utility_update_allowed")
    if isinstance(explicit, bool):
        return explicit
    raw_confounds = observation.get("unresolved_confounding", ())
    confounds = (
        tuple(str(item).upper() for item in raw_confounds)
        if isinstance(raw_confounds, (tuple, list))
        else (str(raw_confounds).upper(),)
        if raw_confounds
        else ()
    )
    if any(
        "IMPLEMENTATION_FIDELITY" in item or "HIDDEN_MULTI_AXIS_DRIFT" in item
        for item in confounds
    ):
        return False
    return observation.get("common_outcome_class") == "SUCCESS"


def _provider_directional_search_utility(
    value: Any,
    *,
    root_baseline_value: float | None = None,
    construction_parents: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
    basic_baseline_value: float | None = None,
    strong_baseline_value: float | None = None,
    neutralizations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Project measured search direction without inflating publication claims.

    A single-seed development result can be inconclusive for a causal or
    publication claim and still be indispensable online search feedback.  The
    projection is deliberately descriptive: it reports observed metric
    direction and axis exposure, but does not authorize causal credit, ban an
    axis, or choose the next candidate for the research agent.
    """

    empty_projection = canonical_value(
        {
            "semantics": (
                "MEASURED_DEVELOPMENT_SEARCH_UTILITY_NOT_CAUSAL_OR_"
                "PUBLICATION_CLAIM"
            ),
            "axes": (),
            "recent": (),
        }
    )
    if not isinstance(value, (tuple, list)):
        return empty_projection
    normalized: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        if not observation_updates_search_utility(
            row, neutralizations=neutralizations
        ):
            continue
        delta = row.get("comparator_delta")
        if (
            not isinstance(delta, (int, float))
            or isinstance(delta, bool)
            or not math.isfinite(float(delta))
        ):
            continue
        raw_footprint = row.get("mechanism_axis_footprint", ())
        footprint = (
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_footprint
                    if str(item).strip()
                )
            )
            if isinstance(raw_footprint, (tuple, list))
            else ()
        )
        if not footprint:
            axis = row.get("mechanism_axis")
            if isinstance(axis, str) and axis.strip():
                footprint = (axis.strip(),)
        if not footprint:
            continue
        candidate_value = _provider_numeric_value(row.get("candidate_value"))
        frontier_delta = _provider_numeric_value(row.get("frontier_delta"))
        parent = (construction_parents or {}).get(
            (row.get("round_index"), row.get("candidate_id"))
        )
        observation = {
                "round": row.get("round_index"),
                "candidate": row.get("candidate_id"),
                "value": candidate_value,
                "delta": float(delta),
                "frontier_delta": frontier_delta,
                "delta_vs_root": (
                    candidate_value - root_baseline_value
                    if candidate_value is not None
                    and root_baseline_value is not None
                    else None
                ),
                "construction_parent": parent,
                "delta_vs_construction_parent": (
                    _provider_numeric_value(parent.get("delta"))
                    if isinstance(parent, Mapping) else None
                ),
                "delta_vs_basic": (
                    candidate_value - basic_baseline_value
                    if candidate_value is not None
                    and basic_baseline_value is not None
                    else None
                ),
                "delta_vs_strong": (
                    candidate_value - strong_baseline_value
                    if candidate_value is not None
                    and strong_baseline_value is not None
                    else None
                ),
                "footprint": footprint,
                "frontier_updated": bool(row.get("frontier_updated")),
                "claim_evidence": row.get("evidence_class"),
                "causal_credit_allowed": bool(row.get("causal_credit_allowed")),
        }
        normalized.append(
            canonical_value(
                {key: item for key, item in observation.items() if item is not None}
            )
        )
    if not normalized:
        return empty_projection

    by_axis: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        for axis in row["footprint"]:
            by_axis.setdefault(axis, []).append(row)

    def comparison_summary(
        rows: Sequence[Mapping[str, Any]], key: str, *, baseline: bool = False
    ) -> dict[str, Any]:
        values = [_provider_numeric_value(row.get(key)) for row in rows]
        known = [value for value in values if value is not None]
        consecutive_nonpositive = 0
        for value in reversed(values):
            if value is None or value > 0.0:
                break
            consecutive_nonpositive += 1
        summary: dict[str, Any] = {"observations": len(known)}
        if len(known) < len(rows):
            summary["unknown_observations"] = len(rows) - len(known)
        if known:
            if baseline:
                summary["best_delta"] = max(known)
            else:
                summary.update(
                    positive=sum(value > 0.0 for value in known),
                    negative=sum(value < 0.0 for value in known),
                    mean_delta=sum(known) / len(known),
                    consecutive_nonpositive=consecutive_nonpositive,
                )
        if values[-1] is not None:
            summary.update(latest_delta=values[-1], latest_round=rows[-1].get("round"))
        return summary

    axes: list[dict[str, Any]] = []
    for axis, rows in by_axis.items():
        axes.append(
            canonical_value(
                {
                    "axis": axis,
                    "observations": len(rows),
                    "baseline_performance": comparison_summary(
                        rows, "delta", baseline=True
                    ),
                    "construction_parent_increment": comparison_summary(
                        rows, "delta_vs_construction_parent"
                    ),
                    "frontier_increment": comparison_summary(rows, "frontier_delta"),
                    "frontier_updates": sum(
                        bool(row.get("frontier_updated")) for row in rows
                    ),
                }
            )
        )
    return canonical_value(
        {
            "semantics": (
                "MEASURED_DEVELOPMENT_SEARCH_UTILITY_NOT_CAUSAL_OR_"
                "PUBLICATION_CLAIM"
            ),
            "axes": tuple(axes[:32]),
            "axis_observation_semantics": (
                "Joint outcomes of the declared changed footprint, not independent "
                "causal credit per axis. Baseline performance includes inherited gains; "
                "parent and frontier increments describe this intervention's measured return. "
                "Use the existing recent-experiment measured costs when choosing the next hypothesis."
            ),
            "recent": tuple(normalized[-8:]),
        }
    )


_PROVIDER_RECENT_EXPERIMENT_FIELDS = (
    "round",
    "primitive_refs",
    "ndcg@10",
    "delta",
    "frontier_updated",
    "frontier_delta",
    "parent",
    "cost",
    "outcome",
    "disposition",
    "axis",
    "mechanism",
    "footprint",
    "evidence",
    "confounds",
    "next_task",
    "failure",
    "efficiency",
)


def _provider_pack_recent_experiments(
    experiments: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Losslessly pack the external attempted-family ledger only.

    Primitive strings are interned once, while repeated object field names are
    represented by ``attempt_fields`` and aligned rows.  A missing row value is
    encoded as ``None`` and decodes by omitting that field, preserving the
    sparse projected attempt exactly without changing durable memory.
    """

    primitive_catalog: list[str] = []
    primitive_indexes: dict[str, int] = {}
    attempts_with_refs: list[dict[str, Any]] = []
    for experiment in experiments:
        attempt = dict(experiment)
        primitives = attempt.pop("primitives", ())
        primitive_refs: list[int] = []
        if isinstance(primitives, (tuple, list)):
            for primitive in primitives:
                primitive_id = str(primitive)
                if primitive_id not in primitive_indexes:
                    primitive_indexes[primitive_id] = len(primitive_catalog)
                    primitive_catalog.append(primitive_id)
                primitive_refs.append(primitive_indexes[primitive_id])
        if primitive_refs:
            attempt["primitive_refs"] = tuple(primitive_refs)
        attempts_with_refs.append(canonical_value(attempt))
    attempt_fields = tuple(
        field_name
        for field_name in _PROVIDER_RECENT_EXPERIMENT_FIELDS
        if any(field_name in attempt for attempt in attempts_with_refs)
    )
    attempt_rows = tuple(
        tuple(attempt.get(field_name) for field_name in attempt_fields)
        for attempt in attempts_with_refs
    )
    return canonical_value(
        {
            "attempt_fields": attempt_fields,
            "attempts": attempt_rows,
            "primitive_catalog": tuple(primitive_catalog),
        }
    )


def project_provider_context_view(
    context_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact only the external Provider prompt view for V2 memory.

    ``ResearchContext.producer_view`` remains the compatibility contract for
    local and injected Producers.  The external Provider adapter applies this
    projection immediately before rendering its prompt, where the duplicated
    durable memory caused the observed round-7 token-ceiling failure.
    """

    scientific_memory = context_view.get("scientific_memory")
    if not isinstance(scientific_memory, Mapping):
        return canonical_value(dict(context_view))
    explicit_global = scientific_memory.get("global_memory")
    if not isinstance(explicit_global, Mapping):
        explicit_global = scientific_memory.get("global")
    if not isinstance(explicit_global, Mapping):
        return canonical_value(dict(context_view))
    role_memory = context_view.get("role_memory")
    if not isinstance(role_memory, Mapping):
        role_memory = {}

    knowledge_base = context_view.get("knowledge_base")
    if not isinstance(knowledge_base, Mapping):
        knowledge_base = {}
    source_identity = knowledge_base.get("source_identity")
    if not isinstance(source_identity, Mapping):
        source_identity = {}
    baseline_context = knowledge_base.get("baseline_context")
    baseline_objective = _provider_baseline_objective(baseline_context)
    single_parent_mode = is_single_parent_context(baseline_context)
    frozen_goal = context_view.get("frozen_goal")
    if not isinstance(frozen_goal, Mapping):
        frozen_goal = {}
    frontier = context_view.get("frontier")
    if not isinstance(frontier, Mapping):
        frontier = {}
    policy = context_view.get("policy")
    if not isinstance(policy, Mapping):
        policy = {}
    budget = context_view.get("budget")
    if not isinstance(budget, Mapping):
        budget = {}
    questions = context_view.get("unresolved_questions")
    questions = list(questions) if isinstance(questions, (tuple, list)) else []
    unresolved_axes = tuple(
        str(question["mechanism_axis"])
        for question in questions
        if isinstance(question, Mapping) and question.get("mechanism_axis")
    )

    latest_scientific = _provider_latest(
        explicit_global.get("scientific_observations"),
        (
            "round_index",
            "candidate_id",
            "candidate_value",
            "comparator_delta",
            "mechanism_axis",
            "mechanism_axis_footprint",
            "evidence_class",
            "failure_class",
            "unresolved_confounding",
            "core_mechanism_contrast",
            "causal_credit_allowed",
            "search_utility_update_allowed",
            "directional_evidence_class",
            "fidelity_classification",
            "frontier_updated",
        ),
    )
    latest_role_result = _provider_latest(
        role_memory.get("execution_history"),
        (
            "round_index",
            "candidate_id",
            "comparator_delta",
            "origin",
            "frontier_updated",
        ),
    )
    latest_role_failure = _provider_latest(
        role_memory.get("failure_history"),
        ("round_index", "failure_code"),
    )

    state_summary: dict[str, Any] = {
        "incumbent": _provider_alias_fields(
            frontier,
            (
                ("candidate", "incumbent_candidate_id"),
                ("round", "incumbent_round_index"),
                ("value", "incumbent_ndcg@10"),
            ),
        ),
        "latest_result": _provider_alias_fields(
            latest_scientific,
            (
                ("round", "round_index"),
                ("candidate", "candidate_id"),
                ("value", "candidate_value"),
                ("delta", "comparator_delta"),
                ("axis", "mechanism_axis"),
                ("footprint", "mechanism_axis_footprint"),
                ("evidence", "evidence_class"),
                ("confounds", "unresolved_confounding"),
                ("contrast", "core_mechanism_contrast"),
                ("causal_credit_allowed", "causal_credit_allowed"),
                ("search_utility_update_allowed", "search_utility_update_allowed"),
                ("directional_evidence_class", "directional_evidence_class"),
                ("fidelity_classification", "fidelity_classification"),
                ("frontier_updated", "frontier_updated"),
            ),
        ),
    }
    research_portfolio = context_view.get("research_portfolio")
    parent_options = context_view.get("construction_parent_options")
    if isinstance(parent_options, (tuple, list)):
        state_summary["construction_parent_selection"] = (
            "Choose the construction parent for this hypothesis explicitly from these existing parents. "
            "The best observed result is recorded independently and need not be the construction parent. "
            "Use the chosen identity in parent_refs or closest_parent; its exact program, configuration "
            "and implementation will be used for inheritance. A fixed research task retains its specified parent."
        )
        # Old checkpoints retain their frozen comparison inputs. Project the
        # current construction policy without repeating their best-only rule.
        if single_parent_mode:
            baseline_objective["search_objective"] = (
                "Maximize real fixed-protocol data/dev NDCG@10 over the frozen root. "
                "Choose the construction parent independently from the available options."
            )
            semantics = dict(baseline_objective.get("baseline_decision_semantics", {}))
            for key in ("active_lineage_parent", "ordinary_search", "innovation_lane"):
                semantics.pop(key, None)
            baseline_objective["baseline_decision_semantics"] = semantics
        state_summary["construction_parent_options"] = tuple({
            key: (str(option[key])[:120] if key in {"hypothesis", "mechanism_change"} else canonical_value(option[key]))
            for key in ("candidate_id", "program_digest", "kind", "observed_metric",
                        "mechanism_change" if option.get("mechanism_change") else "hypothesis")
            if key in option
        } for option in parent_options)
    if isinstance(research_portfolio, (tuple, list)):
        state_summary["research_portfolio"] = tuple(
            canonical_value(dict(item))
            for item in research_portfolio
            if isinstance(item, Mapping)
        )
    latest_result = state_summary.get("latest_result")
    if isinstance(latest_result, dict) and baseline_objective:
        _annotate_provider_result_against_baselines(
            latest_result,
            baseline_objective,
        )
    # Keep the fixed-root result, actual construction parent, and previous
    # frontier together. Root gains can remain positive after a failed refinement.
    round_attempts = explicit_global.get("round_attempts")
    construction_parents: dict[tuple[int, str], Mapping[str, Any]] = {}
    # Compact history retains parents after attempts leave the recent window.
    # Current attempt records take precedence over the older compact copy.
    for history in (explicit_global.get("mechanism_experiences"), round_attempts):
        if not isinstance(history, (tuple, list)):
            continue
        for attempt in history:
            if not isinstance(attempt, Mapping):
                continue
            parent = attempt.get("construction_parent")
            if (
                isinstance(parent, Mapping)
                and isinstance(attempt.get("round_index"), int)
                and isinstance(attempt.get("candidate_id"), str)
            ):
                key = (attempt["round_index"], attempt["candidate_id"])
                construction_parents[key] = parent
    if isinstance(latest_result, dict):
        parent = construction_parents.get(
            (latest_result.get("round"), latest_result.get("candidate"))
        )
        if parent is not None:
            latest_result["construction_parent"] = canonical_value(parent)
            parent_delta = _provider_numeric_value(parent.get("delta"))
            if parent_delta is not None:
                latest_result["delta_vs_construction_parent"] = parent_delta
    if isinstance(latest_result, dict) and isinstance(round_attempts, (tuple, list)):
        for attempt in reversed(round_attempts):
            if (
                isinstance(attempt, Mapping)
                and latest_result.get("candidate")
                and attempt.get("candidate_id") == latest_result["candidate"]
                and attempt.get("round_index") == latest_result.get("round")
            ):
                if "frontier_delta" in attempt:
                    latest_result["frontier_delta"] = canonical_value(attempt["frontier_delta"])
                break
    if single_parent_mode:
        frozen_parent = bound_parent_from_context(baseline_context)
        if frozen_parent is not None:
            state_summary["frozen_parent_binding"] = frozen_parent["binding"]
            state_summary["frozen_parent_mechanism_program"] = frozen_parent[
                "mechanism_program"
            ]
            parent_contract = frozen_parent.get("execution_contract")
            if not isinstance(parent_contract, Mapping):
                parent_contract = frozen_parent.get("source_bundle", {}).get("execution_contract")
            if isinstance(parent_contract, Mapping):
                state_summary["frozen_parent_execution_contract"] = canonical_value(dict(parent_contract))
    parent_source_bundle = (
        frozen_parent.get("source_bundle")
        if single_parent_mode and frozen_parent is not None else None
    )
    lineage_parent_program = context_view.get("lineage_parent_mechanism_program")
    if isinstance(lineage_parent_program, Mapping):
        candidate_id = lineage_parent_program.get("candidate_id")
        program_digest = lineage_parent_program.get("program_digest")
        program = lineage_parent_program.get("mechanism_program")
        if (
            isinstance(candidate_id, str)
            and isinstance(program_digest, str)
            and isinstance(program, Mapping)
        ):
            state_summary["lineage_parent_binding"] = {
                "candidate_id": candidate_id,
                "program_digest": program_digest,
            }
            state_summary["lineage_parent_mechanism_program"] = canonical_value(
                dict(program)
            )
            if isinstance(lineage_parent_program.get("execution_contract"), Mapping):
                state_summary["lineage_parent_execution_contract"] = canonical_value(
                    dict(lineage_parent_program["execution_contract"])
                )
            # A measured descendant replaces the frozen anchor as the actual
            # construction parent. Never label the frozen source as its code.
            parent_source_bundle = lineage_parent_program.get("source_bundle")
    elif not single_parent_mode:
        frontier_parent = frontier.get("lineage_parent_binding")
        if isinstance(frontier_parent, Mapping):
            candidate_id = frontier_parent.get("candidate_id")
            program_digest = frontier_parent.get("program_digest")
            if isinstance(candidate_id, str) and isinstance(program_digest, str):
                state_summary["lineage_parent_binding"] = {
                    "candidate_id": candidate_id,
                    "program_digest": program_digest,
                }
        activated = scientific_memory.get("activated_capability")
        if (
            "lineage_parent_binding" not in state_summary
            and isinstance(activated, Mapping)
        ):
            search_candidate = activated.get("search_candidate")
            if isinstance(search_candidate, Mapping):
                candidate_id = search_candidate.get("compiler_candidate_id")
                program_digest = search_candidate.get("mechanism_program_digest")
                if isinstance(candidate_id, str) and isinstance(program_digest, str):
                    state_summary["lineage_parent_binding"] = {
                        "candidate_id": candidate_id,
                        "program_digest": program_digest,
                    }
    if isinstance(parent_source_bundle, Mapping):
        state_summary["parent_implementation"] = {
            "candidate_id": parent_source_bundle["candidate_id"],
            "source_tree_digest": parent_source_bundle["source_tree_digest"],
            "files": [
                {"path": row["path"], "content": row["content"]}
                for row in parent_source_bundle["files"]
            ],
            "use": (
                f"Current executable candidate: {parent_source_bundle['candidate_id']}. "
                "These files show this candidate's implemented behavior. "
                "Its original mechanism proposal describes how it was generated: "
                "parent_refs and 'Starting from' in that proposal identify its "
                "ancestors, not the candidate implemented here. "
            ) + (
                "Reference implementation for the candidate_id shown here. "
                "Inspect its computation, configuration and measured results when "
                "comparing parents. Choose the construction parent explicitly from "
                "the available options; inheritance follows that chosen candidate. "
                "Keep useful behavior and training efficiency."
                if parent_options else
                "Use this actual construction parent, its configuration and "
                "measured results to design a more effective next candidate. "
                "Inspect the implemented computation when deciding what to "
                "retain, change or simplify, including differences from the "
                "declared mechanism. Keep useful behavior and training efficiency."
            ),
        }
    completed = context_view.get("latest_completed_execution")
    if isinstance(completed, Mapping):
        completed = dict(completed)
        implementation = completed.get("implementation")
        parent_implementation = state_summary.get("parent_implementation", {})
        if (
            isinstance(implementation, Mapping)
            and implementation.get("source_tree_digest")
            and implementation.get("source_tree_digest") == parent_implementation.get("source_tree_digest")
        ):
            completed["implementation"] = {
                "source_tree_digest": implementation["source_tree_digest"],
                "source_context_ref": "state.parent_implementation",
            }
        completed["semantics"] = (
            "This is the latest completed execution, not necessarily the construction parent. "
            "Compare its actual computation, configuration and native progress with the proposed "
            "mechanism and measured outcome when choosing what to retain, change or simplify. "
            "The proposal describes intent; a score alone does not establish that explanation."
        )
        state_summary["latest_completed_execution"] = canonical_value(completed)
    diagnostic_summary = _provider_engineering_diagnostic(
        explicit_global.get("latest_feedback")
    )
    if diagnostic_summary:
        state_summary["latest_diagnostic"] = diagnostic_summary
    guard_summary = _provider_guard_feedback(
        explicit_global.get("latest_guard_feedback")
    )
    if guard_summary:
        state_summary["evidence_guard"] = guard_summary
        # Guard interprets the completed result; it does not replace the raw
        # measured effect that the next research decision must still see.
    if "active_task_directive" in context_view:
        directive = context_view.get("active_task_directive")
        current_task = dict(directive) if isinstance(directive, Mapping) else None
    else:
        current_task = _provider_task_head(explicit_global.get("task_queue"))
    if current_task:
        task_summary = _provider_alias_fields(
            current_task,
            (
                ("id", "task_id"),
                ("operation", "operation"),
                ("candidate", "candidate_id"),
                ("parent", "parent_candidate_id"),
                ("comparator", "comparator_identity"),
                ("program", "mechanism_program_digest"),
                ("seed_or_control", "required_seed_or_control"),
                ("status", "status"),
                ("next_task", "next_discriminative_test"),
                ("axis", "mechanism_axis"),
                ("footprint", "mechanism_axis_footprint"),
                ("evidence", "evidence_class"),
                ("failure", "failure_class"),
                ("confounds", "unresolved_confounding"),
                ("contrast", "core_mechanism_contrast"),
                ("causal_credit_allowed", "causal_credit_allowed"),
                ("execution_state", "execution_state"),
                ("binding_requirement", "binding_requirement"),
                ("frontier_candidate", "frontier_candidate_id"),
                ("frontier_parent", "frontier_parent_candidate_id"),
                ("frontier_comparator", "frontier_comparator_identity"),
                ("task_digest", "task_digest"),
                ("observation_seed", "observation_seed"),
                ("protocol_digest", "protocol_digest"),
                (
                    "execution_eligible_this_round",
                    "execution_eligible_this_round",
                ),
            ),
        )
        task_record = current_task.get("task_record")
        task_record = (
            task_record if isinstance(task_record, Mapping) else current_task
        )
        task_metadata = task_record.get("metadata")
        task_metadata = task_metadata if isinstance(task_metadata, Mapping) else {}
        if "binding_requirement" not in task_summary:
            binding_requirement = task_metadata.get("binding_requirement")
            if isinstance(binding_requirement, str) and binding_requirement:
                task_summary["binding_requirement"] = binding_requirement
        exact_target = {
            "target_semantic_identity": task_record.get(
                "candidate_semantic_digest"
            ),
            "target_effective_experiment": task_metadata.get(
                "effective_experiment_digest"
            ),
            "target_effective_family": task_metadata.get(
                "effective_family_digest"
            ),
            "matched_control_requirement": task_metadata.get(
                "matched_control_requirement"
            ),
            "confirmation_target": task_metadata.get("confirmation_target"),
        }
        task_summary.update(
            {
                field_name: field_value
                for field_name, field_value in exact_target.items()
                if isinstance(field_value, str) and field_value
            }
        )
        if (
            task_summary.get("execution_eligible_this_round") is True
            and task_summary.get("binding_requirement")
            == "EXACT_EFFECTIVE_IDENTITY"
        ):
            target_program = task_record.get("mechanism_program")
            if isinstance(target_program, Mapping):
                task_summary["target_mechanism_program"] = canonical_value(
                    dict(target_program)
                )
        elif (
            task_summary.get("execution_eligible_this_round") is True
            and task_metadata.get("execution_state")
            == "AWAITING_CANDIDATE_BINDING"
        ):
            frontier_program = task_record.get("mechanism_program")
            if isinstance(frontier_program, Mapping):
                task_summary["frontier_mechanism_program"] = canonical_value(
                    dict(frontier_program)
                )
        latest_feedback = explicit_global.get("latest_feedback")
        task_transition = (
            latest_feedback.get("task_queue_transition")
            if isinstance(latest_feedback, Mapping)
            else None
        )
        deferred_requirements = (
            task_transition.get("deferred_requirements")
            if isinstance(task_transition, Mapping)
            else ()
        )
        if isinstance(deferred_requirements, (tuple, list)) and deferred_requirements:
            task_summary["deferred_requirements"] = tuple(
                str(item) for item in deferred_requirements
            )
        # The frontier fields are an explicit compatibility mirror of the
        # task's candidate, parent, and comparator.  Keep them when they add
        # information, but omit exact duplicates and empty optional values
        # from the bounded external Provider projection.
        for frontier_field, task_field in (
            ("frontier_candidate", "candidate"),
            ("frontier_parent", "parent"),
            ("frontier_comparator", "comparator"),
        ):
            if canonical_value(task_summary.get(frontier_field)) == canonical_value(
                task_summary.get(task_field)
            ):
                task_summary.pop(frontier_field, None)
        task_summary = canonical_value(
            {
                field_name: field_value
                for field_name, field_value in task_summary.items()
                if field_value is not None
                and field_value != ()
                and field_value != []
                and field_value != {}
            }
        )
        state_summary["task"] = task_summary

    global_counts = _provider_history_counts(
        explicit_global,
        ("scientific_observations",),
    )
    raw_mechanism_negatives = explicit_global.get("mechanism_negative_evidence")
    if isinstance(raw_mechanism_negatives, (tuple, list)):
        negative_count = len(raw_mechanism_negatives)
    else:
        legacy_negatives = explicit_global.get("negative_evidence")
        negative_count = (
            len(legacy_negatives)
            if isinstance(legacy_negatives, (tuple, list))
            else 0
        )
    role_counts = _provider_history_counts(
        role_memory,
        ("execution_history", "proposal_history", "failure_history"),
    )
    history_counts = {
        "scientific": global_counts.get("scientific_observations", 0),
        "negative": negative_count,
        "role_executions": role_counts.get("execution_history", 0),
        "role_proposals": role_counts.get("proposal_history", 0),
        "role_failures": role_counts.get("failure_history", 0),
    }
    role_credit = _provider_alias_fields(
        role_memory.get("credit"),
        (
            ("frontier_gains", "frontier_gain_count"),
            ("tasks_satisfied", "task_satisfied_count"),
        ),
    )
    memory_summary: dict[str, Any] = {
        "history_counts": history_counts,
        "credit": role_credit,
        "comparison_basis": "delta=fixed paired comparator; parent=actual construction parent; frontier_delta=pre-execution frontier; unknown is not zero",
    }
    mechanism_negatives = _provider_mechanism_negative_evidence(
        raw_mechanism_negatives
    )
    if mechanism_negatives:
        memory_summary["mechanism_negative_evidence"] = mechanism_negatives
    recent_experiments = _provider_recent_experiments(
        explicit_global.get("round_attempts")
    )
    if recent_experiments:
        memory_summary["recent_experiments"] = (
            _provider_pack_recent_experiments(recent_experiments)
        )
        parent_program = None
        lineage_parent = context_view.get("lineage_parent_mechanism_program")
        if isinstance(lineage_parent, Mapping) and isinstance(
            lineage_parent.get("mechanism_program"), Mapping
        ):
            parent_program = lineage_parent["mechanism_program"]
        elif isinstance(baseline_context, Mapping):
            parent_anchor = baseline_context.get("parent_anchor")
            if isinstance(parent_anchor, Mapping):
                parent_program = parent_anchor.get("mechanism_program")
        pre_metric_feedback = _provider_pre_metric_failure_feedback(
            recent_experiments,
            parent_program=parent_program,
        )
        if pre_metric_feedback:
            memory_summary["pre_metric_failure_feedback"] = pre_metric_feedback


    related_experiments = _provider_related_mechanism_experiments(
        explicit_global.get("mechanism_experiences"),
        explicit_global.get("round_attempts"),
        unresolved_axes,
        mechanism_queries=tuple(
            str(row.get("mechanism_change", ""))
            for row in context_view.get("research_portfolio", ())
            if isinstance(row, Mapping)
        ),
        before_round=context_view.get("round_index"),
    )
    if related_experiments:
        memory_summary["related_mechanism_experiments"] = (
            _provider_pack_recent_experiments(related_experiments)
        )
    basic_baseline = baseline_objective.get("basic_baseline")
    strong_baseline = baseline_objective.get("strong_baseline")
    parent_anchor = baseline_objective.get("parent_anchor")
    parent_metric = (
        parent_anchor.get("paired_metric")
        if isinstance(parent_anchor, Mapping)
        else None
    )
    directional_search_utility = _provider_directional_search_utility(
        explicit_global.get("scientific_observations"),
        construction_parents=construction_parents,
        root_baseline_value=_provider_numeric_value(
            parent_metric.get("value")
            if isinstance(parent_metric, Mapping)
            else None
        ),
        basic_baseline_value=_provider_numeric_value(
            basic_baseline.get("value")
            if isinstance(basic_baseline, Mapping)
            else None
        ),
        strong_baseline_value=_provider_numeric_value(
            strong_baseline.get("value")
            if isinstance(strong_baseline, Mapping)
            else None
        ),
        neutralizations=(
            tuple(
                row
                for row in explicit_global.get("utility_neutralizations", ())
                if isinstance(row, Mapping)
            )
            if isinstance(
                explicit_global.get("utility_neutralizations", ()),
                (tuple, list),
            )
            else ()
        ),
    )
    raw_neutralizations = explicit_global.get("utility_neutralizations", ())
    if isinstance(raw_neutralizations, (tuple, list)) and raw_neutralizations:
        memory_summary["fidelity_neutralizations"] = tuple(
            {
                "round": row.get("round_index"),
                "candidate": row.get("candidate_id"),
                "classification": row.get("classification"),
                "reason": row.get("reason"),
                "search_utility_update_allowed": False,
                "measurement_retained": True,
            }
            for row in raw_neutralizations[-16:]
            if isinstance(row, Mapping)
        )
    if directional_search_utility:
        memory_summary["directional_search_utility"] = (
            directional_search_utility
        )
    if latest_role_result:
        role_latest = _provider_alias_fields(
            latest_role_result,
            (
                ("round", "round_index"),
                ("candidate", "candidate_id"),
                ("delta", "comparator_delta"),
                ("origin", "origin"),
            ),
        )
        latest_result = state_summary.get("latest_result")
        duplicates_latest_result = (
            isinstance(latest_result, Mapping)
            and role_latest.get("candidate") == latest_result.get("candidate")
            and role_latest.get("round") == latest_result.get("round")
            and role_latest.get("delta") == latest_result.get("delta")
        )
        if not duplicates_latest_result:
            memory_summary["role_latest"] = role_latest
    if latest_role_failure:
        memory_summary["role_last_failure"] = _provider_alias_fields(
            latest_role_failure,
            (("round", "round_index"), ("code", "failure_code")),
        )
    guard_attribution = _provider_guard_attribution(
        explicit_global.get("latest_mechanism_attribution")
    )
    if guard_attribution:
        latest_result = state_summary.get("latest_result")
        task_summary = state_summary.get("task")
        duplicated_fields = {
            "axis",
            "footprint",
            "evidence",
            "confounds",
            "contrast",
            "causal_credit_allowed",
        }
        attribution_fields = set(guard_attribution)
        additional_attribution = attribution_fields - duplicated_fields - {"next"}
        duplicated_result = (
            isinstance(latest_result, Mapping)
            and not additional_attribution
            and bool(attribution_fields & duplicated_fields)
            and all(
                canonical_value(latest_result.get(field_name))
                == canonical_value(guard_attribution[field_name])
                for field_name in attribution_fields & duplicated_fields
            )
        )
        duplicated_next = (
            "next" not in guard_attribution
            or (
                isinstance(task_summary, Mapping)
                and task_summary.get("next_task") == guard_attribution["next"]
            )
        )
        if not (duplicated_result and duplicated_next):
            memory_summary["guard_attribution"] = guard_attribution
    engineering_failure = explicit_global.get("last_engineering_failure")
    if isinstance(engineering_failure, Mapping):
        memory_summary["engineering_failure"] = _provider_scalar_fields(
            engineering_failure,
            (
                "failure_class",
                "typed_blocker_class",
                "capability_family",
                "failure_fingerprint",
                "mechanism_effect_update_allowed",
                "round_index",
                "candidate_id",
                "implementation_digest",
            ),
        )
        memory_summary["engineering_failure"]["scope"] = "HISTORICAL_OBSERVATION_NOT_CURRENT_STATUS"

    provider_contract = {
        "proposal_ceiling": budget.get("producer_token_ceiling_each"),
        "experiments": budget.get("experiment_opportunities"),
        "producer_token_fraction": context_view.get(
            "producer_token_fraction"
        ),
    }
    for field_name in (
        "dataset",
        "evaluation_split",
        "candidate_universe",
        "heldout_access",
        "epochs_requested",
        "worker_ceiling_seconds",
        "research_window",
    ):
        if budget.get(field_name) is not None:
            provider_contract[field_name] = budget[field_name]

    objective_summary = {
        "space": knowledge_base.get("search_space"),
        "metric": frozen_goal.get("metric"),
        "direction": frozen_goal.get("direction"),
        "claim_ceiling": frozen_goal.get("single_seed_claim_ceiling"),
        "target_axes": policy.get("mechanism_axis_targeting", ()),
        "open_axes": unresolved_axes,
        "comparison_references": {
            "delta_and_directional_evidence_class": (
                "FROZEN_ROOT" if single_parent_mode else "FIXED_COMPARATOR"
            ),
            "directional_search_utility_axes": {
                "baseline_performance": "SAME_COMPARATOR_AS_DELTA",
                "construction_parent_increment": "ACTUAL_CONSTRUCTION_PARENT",
                "frontier_increment": "PREVIOUS_OBSERVED_FRONTIER",
            },
            "delta_vs_root": "FROZEN_ROOT",
            "construction_parent_and_delta_vs_construction_parent": "ACTUAL_CONSTRUCTION_PARENT",
            "frontier_delta": "PREVIOUS_OBSERVED_FRONTIER",
        },
    }
    if not single_parent_mode:
        objective_summary["baseline"] = source_identity.get(
            "frozen_ndcg_at_10"
        )
    objective_summary.update(baseline_objective)
    projected = canonical_value(
        {
            "schema": "recclaw.provider-context.v4",
            "context": {
                "digest": context_view.get("context_digest"),
                "round": context_view.get("round_index"),
                "role": context_view.get("producer_role"),
                **({"research_mode": budget["research_mode"]}
                   if budget.get("research_mode") == "director_sequential" else {}),
            },
            "objective": objective_summary,
            "state": state_summary,
            "memory": memory_summary,
            "contract": provider_contract,
        }
    )
    return projected


def lineage_parent_binding_from_context_view(
    context_view: Mapping[str, Any],
) -> dict[str, str] | None:
    """Return the authoritative activated lineage parent for this cycle."""

    projected = project_provider_context_view(context_view)
    binding = projected.get("state", {}).get("lineage_parent_binding")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise ResearchLineInterfaceError(
            "state.lineage_parent_binding must be an object"
        )
    candidate_id = binding.get("candidate_id")
    program_digest = binding.get("program_digest")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ResearchLineInterfaceError(
            "lineage_parent_binding.candidate_id must be non-empty"
        )
    if not isinstance(program_digest, str):
        raise ResearchLineInterfaceError(
            "lineage_parent_binding.program_digest must be a digest"
        )
    return {
        "candidate_id": candidate_id,
        "program_digest": validate_sha256(
            program_digest,
            field_name="lineage_parent_binding.program_digest",
        ),
    }


class ResearchTaskOperationV2(str, Enum):
    """Durable scientific work units understood by the Research Line."""

    NEW_SEED = "NEW_SEED"
    MATCHED_CONTROL = "MATCHED_CONTROL"
    MECHANISM_OFF = "MECHANISM_OFF"
    REPAIR = "REPAIR"
    REPRODUCE = "REPRODUCE"
    MOVE_ON = "MOVE_ON"


class ResearchTaskStatusV2(str, Enum):
    """Lifecycle state for a durable task, separate from legacy task slots."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SATISFIED = "SATISFIED"
    CLOSED = "CLOSED"


def _nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ResearchLineInterfaceError(f"{field_name} must be normalized and non-empty")
    return value


def _snapshot(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchLineInterfaceError(f"{field_name} must be a mapping")
    try:
        return canonical_value(dict(value))
    except (TypeError, ValueError) as error:
        raise ResearchLineInterfaceError(f"{field_name} is not canonicalizable") from error


@dataclass(frozen=True, slots=True)
class ResearchTaskRecordV2:
    """One durable, provenance-bearing task in the shared Research queue.

    The existing ``ResearchTaskV1`` remains the prompt/runtime compatibility
    projection.  This record carries the operation identity and closure state
    that the old single task slot cannot represent.
    """

    task_id: str
    operation: ResearchTaskOperationV2 | str
    candidate_id: str
    candidate_semantic_digest: str
    mechanism_program_digest: str
    parent_candidate_id: str | None
    comparator_identity: str
    protocol_digest: str
    required_seed_or_control: str
    priority: float
    created_round: int
    evidence_present: tuple[str, ...] = ()
    missing_seed_count: int = 1
    mechanism_program: Mapping[str, Any] = field(default_factory=dict)
    status: ResearchTaskStatusV2 | str = ResearchTaskStatusV2.PENDING
    producer_role: str | None = None
    provenance_digest: str | None = None
    deadline_round: int | None = None
    close_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema = "recclaw.research-line.task-record.v2"

    def __post_init__(self) -> None:
        _nonempty(self.task_id, field_name="task_id")
        _nonempty(self.candidate_id, field_name="candidate_id")
        _nonempty(self.comparator_identity, field_name="comparator_identity")
        _nonempty(
            self.required_seed_or_control,
            field_name="required_seed_or_control",
        )
        try:
            operation = (
                self.operation
                if isinstance(self.operation, ResearchTaskOperationV2)
                else ResearchTaskOperationV2(str(self.operation))
            )
        except ValueError as error:
            raise ResearchLineInterfaceError(
                "operation is not a ResearchTaskOperationV2"
            ) from error
        try:
            status = (
                self.status
                if isinstance(self.status, ResearchTaskStatusV2)
                else ResearchTaskStatusV2(str(self.status))
            )
        except ValueError as error:
            raise ResearchLineInterfaceError(
                "status is not a ResearchTaskStatusV2"
            ) from error
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        for field_name in (
            "candidate_semantic_digest",
            "mechanism_program_digest",
            "protocol_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name=field_name),
            )
        if self.parent_candidate_id is not None:
            _nonempty(self.parent_candidate_id, field_name="parent_candidate_id")
        if self.producer_role is not None and self.producer_role not in DISCOVERY_PRODUCERS:
            raise ResearchLineInterfaceError(
                "producer_role is outside the four-role portfolio"
            )
        if self.provenance_digest is not None:
            object.__setattr__(
                self,
                "provenance_digest",
                validate_sha256(self.provenance_digest, field_name="provenance_digest"),
            )
        if self.close_reason is not None:
            _nonempty(self.close_reason, field_name="close_reason")
        if self.created_round < 1 or self.missing_seed_count < 0:
            raise ResearchLineInterfaceError("task round/count is invalid")
        if self.deadline_round is not None and self.deadline_round < self.created_round:
            raise ResearchLineInterfaceError("deadline_round precedes created_round")
        if isinstance(self.priority, bool):
            raise ResearchLineInterfaceError("priority must be numeric")
        try:
            priority = float(self.priority)
        except (TypeError, ValueError) as error:
            raise ResearchLineInterfaceError("priority must be numeric") from error
        if not math.isfinite(priority) or not 0.0 <= priority <= 1.0:
            raise ResearchLineInterfaceError("priority must be finite and in [0,1]")
        object.__setattr__(self, "priority", priority)
        object.__setattr__(
            self,
            "evidence_present",
            tuple(dict.fromkeys(str(item) for item in self.evidence_present)),
        )
        object.__setattr__(
            self,
            "metadata",
            _snapshot(self.metadata, field_name="metadata"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchTaskRecordV2":
        if not isinstance(value, Mapping):
            raise ResearchLineInterfaceError("task record must be a mapping")
        return cls(
            task_id=value.get("task_id"),
            operation=value.get("operation"),
            candidate_id=value.get("candidate_id"),
            candidate_semantic_digest=value.get("candidate_semantic_digest"),
            mechanism_program_digest=value.get("mechanism_program_digest"),
            parent_candidate_id=value.get("parent_candidate_id"),
            comparator_identity=value.get("comparator_identity"),
            protocol_digest=value.get("protocol_digest"),
            required_seed_or_control=value.get("required_seed_or_control"),
            priority=value.get("priority", 0.5),
            created_round=value.get("created_round", 1),
            evidence_present=tuple(value.get("evidence_present", ())),
            missing_seed_count=value.get("missing_seed_count", 1),
            mechanism_program=value.get("mechanism_program", {}),
            status=value.get("status", ResearchTaskStatusV2.PENDING),
            producer_role=value.get("producer_role"),
            provenance_digest=value.get("provenance_digest"),
            deadline_round=value.get("deadline_round"),
            close_reason=value.get("close_reason"),
            metadata=value.get("metadata", {}),
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "task_id": self.task_id,
                "operation": self.operation.value,
                "candidate_id": self.candidate_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "mechanism_program_digest": self.mechanism_program_digest,
                "parent_candidate_id": self.parent_candidate_id,
                "comparator_identity": self.comparator_identity,
                "protocol_digest": self.protocol_digest,
                "required_seed_or_control": self.required_seed_or_control,
                "priority": self.priority,
                "created_round": self.created_round,
                "evidence_present": self.evidence_present,
                "missing_seed_count": self.missing_seed_count,
                "mechanism_program": self.mechanism_program,
                "status": self.status.value,
                "producer_role": self.producer_role,
                "provenance_digest": self.provenance_digest,
                "deadline_round": self.deadline_round,
                "close_reason": self.close_reason,
                "metadata": self.metadata,
            }
        )

    def legacy_task_type(self) -> str:
        return {
            ResearchTaskOperationV2.NEW_SEED: "VALIDATE_SAME_CANDIDATE",
            ResearchTaskOperationV2.MATCHED_CONTROL: "RUN_MATCHED_CONTROL",
            ResearchTaskOperationV2.MECHANISM_OFF: "RUN_ABLATION",
            ResearchTaskOperationV2.REPAIR: "REPAIR_IMPLEMENTATION",
            ResearchTaskOperationV2.REPRODUCE: "VALIDATE_SAME_CANDIDATE",
            ResearchTaskOperationV2.MOVE_ON: "PROTOCOL_BRANCH_DIAGNOSTIC",
        }[self.operation]

    def prompt_projection(self) -> dict[str, Any]:
        """Return the closed legacy task slot consumed by current runtime code."""

        status = {
            ResearchTaskStatusV2.PENDING: "PENDING",
            ResearchTaskStatusV2.ACTIVE: "ACTIVE",
            ResearchTaskStatusV2.SATISFIED: "COMPLETED",
            ResearchTaskStatusV2.CLOSED: "CANCELLED",
        }[self.status]
        return canonical_value(
            {
                "task_type": self.legacy_task_type(),
                "candidate_id": self.candidate_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "required_seed_or_control": self.required_seed_or_control,
                "task_status": status,
            }
        )

    def to_legacy_task(self):
        """Materialize the existing ResearchTaskV1 compatibility projection."""

        from recclaw_core.helix.scientific_attribution import (
            ResearchTaskStatusV1,
            ResearchTaskTypeV1,
            ResearchTaskV1,
        )

        legacy_status = {
            ResearchTaskStatusV2.PENDING: ResearchTaskStatusV1.PENDING,
            ResearchTaskStatusV2.ACTIVE: ResearchTaskStatusV1.ACTIVE,
            ResearchTaskStatusV2.SATISFIED: ResearchTaskStatusV1.COMPLETED,
            ResearchTaskStatusV2.CLOSED: ResearchTaskStatusV1.CANCELLED,
        }[self.status]
        return ResearchTaskV1(
            task_id=self.task_id,
            task_type=ResearchTaskTypeV1(self.legacy_task_type()),
            candidate_id=self.candidate_id,
            candidate_semantic_digest=self.candidate_semantic_digest,
            mechanism_program_digest=self.mechanism_program_digest,
            parent_candidate_id=self.parent_candidate_id,
            comparator_identity=self.comparator_identity,
            protocol_digest=self.protocol_digest,
            required_seed_or_control=self.required_seed_or_control,
            task_status=legacy_status,
            created_round=self.created_round,
            utility_priority=self.priority,
            missing_seed_count=self.missing_seed_count,
            mechanism_program=self.mechanism_program,
            owner_arm_instance_id=self.metadata.get("owner_arm_instance_id"),
        )


@dataclass(frozen=True, slots=True)
class ResearchTaskQueueV2:
    """Immutable durable queue with deterministic priority and lifecycle APIs."""

    tasks: tuple[ResearchTaskRecordV2, ...] = ()

    schema = "recclaw.research-line.task-queue.v2"

    _OPERATION_ORDER = {
        ResearchTaskOperationV2.MATCHED_CONTROL: 0,
        ResearchTaskOperationV2.MECHANISM_OFF: 1,
        ResearchTaskOperationV2.NEW_SEED: 2,
        ResearchTaskOperationV2.REPRODUCE: 3,
        ResearchTaskOperationV2.REPAIR: 0,
        ResearchTaskOperationV2.MOVE_ON: 4,
    }

    def __post_init__(self) -> None:
        normalized = tuple(
            item
            if isinstance(item, ResearchTaskRecordV2)
            else ResearchTaskRecordV2.from_dict(item)
            for item in self.tasks
        )
        if len({item.task_id for item in normalized}) != len(normalized):
            raise ResearchLineInterfaceError("task queue repeats a task_id")
        object.__setattr__(
            self,
            "tasks",
            tuple(sorted(normalized, key=lambda item: item.task_id)),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ResearchTaskQueueV2":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ResearchLineInterfaceError("task_queue must be a mapping")
        raw_tasks = value.get("tasks", ())
        if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, (tuple, list)):
            raise ResearchLineInterfaceError("task_queue.tasks must be a sequence")
        return cls(tuple(ResearchTaskRecordV2.from_dict(item) for item in raw_tasks))

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "tasks": tuple(item.to_dict() for item in self.tasks),
            }
        )

    def get(self, task_id: str) -> ResearchTaskRecordV2 | None:
        return next((item for item in self.tasks if item.task_id == task_id), None)

    def enqueue(self, task: ResearchTaskRecordV2) -> "ResearchTaskQueueV2":
        if not isinstance(task, ResearchTaskRecordV2):
            raise ResearchLineInterfaceError("task queue accepts ResearchTaskRecordV2")
        prior = self.get(task.task_id)
        if prior is None:
            return ResearchTaskQueueV2((*self.tasks, task))
        identity_fields = (
            "operation",
            "candidate_id",
            "candidate_semantic_digest",
            "mechanism_program_digest",
            "parent_candidate_id",
            "comparator_identity",
            "required_seed_or_control",
            "protocol_digest",
        )
        if any(getattr(prior, field_name) != getattr(task, field_name) for field_name in identity_fields):
            raise ResearchLineInterfaceError("task identity substitution")
        if prior.status not in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            return self
        merged = replace(
            prior,
            priority=max(prior.priority, task.priority),
            evidence_present=tuple(
                dict.fromkeys((*prior.evidence_present, *task.evidence_present))
            ),
            metadata={**prior.metadata, **task.metadata},
        )
        return ResearchTaskQueueV2(
            tuple(merged if item.task_id == task.task_id else item for item in self.tasks)
        )

    def select_next(self) -> ResearchTaskRecordV2 | None:
        pending = [
            item for item in self.tasks if item.status is ResearchTaskStatusV2.PENDING
        ]
        if not pending:
            return None
        return min(
            pending,
            key=lambda item: (
                self._OPERATION_ORDER[item.operation],
                -item.priority,
                item.created_round,
                item.deadline_round if item.deadline_round is not None else 2**31,
                item.task_id,
            ),
        )

    def activate(self, task_id: str) -> "ResearchTaskQueueV2":
        task = self.get(task_id)
        if task is None or task.status is not ResearchTaskStatusV2.PENDING:
            raise ResearchLineInterfaceError("only a pending task can be activated")
        return self._replace(
            replace(task, status=ResearchTaskStatusV2.ACTIVE)
        )

    def satisfy(
        self,
        task_id: str,
        *,
        evidence: tuple[str, ...] = (),
        reason: str = "REQUIRED_EVIDENCE_OBSERVED",
    ) -> "ResearchTaskQueueV2":
        task = self.get(task_id)
        if task is None or task.status not in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            raise ResearchLineInterfaceError("only a pending or active task can be satisfied")
        return self._replace(
            replace(
                task,
                status=ResearchTaskStatusV2.SATISFIED,
                evidence_present=tuple(dict.fromkeys((*task.evidence_present, *evidence))),
                close_reason=reason,
            )
        )

    def close(
        self,
        task_id: str,
        *,
        reason: str = "CLOSED_BY_RESEARCH_POLICY",
    ) -> "ResearchTaskQueueV2":
        task = self.get(task_id)
        if task is None:
            raise ResearchLineInterfaceError("cannot close an unknown task")
        if task.status is ResearchTaskStatusV2.CLOSED:
            return self
        if task.status is ResearchTaskStatusV2.SATISFIED:
            return self
        return self._replace(
            replace(task, status=ResearchTaskStatusV2.CLOSED, close_reason=reason)
        )

    def _replace(self, task: ResearchTaskRecordV2) -> "ResearchTaskQueueV2":
        return ResearchTaskQueueV2(
            tuple(task if item.task_id == task.task_id else item for item in self.tasks)
        )


@dataclass(frozen=True, slots=True)
class ResearchContext:
    """One arm-local context consumed throughout one Research Line round."""

    campaign_id: str
    round_index: int
    knowledge_base: Mapping[str, Any]
    frozen_goal: Mapping[str, Any]
    frontier: Mapping[str, Any]
    scientific_memory: Mapping[str, Any]
    unresolved_questions: tuple[Mapping[str, Any], ...]
    policy: Mapping[str, Any]
    budget: Mapping[str, Any]
    active_profile_ref: str
    active_profile_digest: str
    protocol_ref: str
    protocol_digest: str

    schema = "recclaw.research-line.context.v1"

    def __post_init__(self) -> None:
        _nonempty(self.campaign_id, field_name="campaign_id")
        if self.round_index < 1:
            raise ResearchLineInterfaceError("round_index must be positive")
        for field_name in ("active_profile_ref", "protocol_ref"):
            _nonempty(getattr(self, field_name), field_name=field_name)
        for field_name in ("active_profile_digest", "protocol_digest"):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "knowledge_base",
            "frozen_goal",
            "frontier",
            "scientific_memory",
            "policy",
            "budget",
        ):
            object.__setattr__(
                self,
                field_name,
                _snapshot(getattr(self, field_name), field_name=field_name),
            )
        questions = []
        for index, question in enumerate(self.unresolved_questions):
            questions.append(
                _snapshot(question, field_name=f"unresolved_questions[{index}]")
            )
        object.__setattr__(self, "unresolved_questions", tuple(questions))

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "round_index": self.round_index,
                "knowledge_base": self.knowledge_base,
                "frozen_goal": self.frozen_goal,
                "frontier": self.frontier,
                "scientific_memory": self.scientific_memory,
                "unresolved_questions": self.unresolved_questions,
                "policy": self.policy,
                "budget": self.budget,
                "active_profile_ref": self.active_profile_ref,
                "active_profile_digest": self.active_profile_digest,
                "protocol_ref": self.protocol_ref,
                "protocol_digest": self.protocol_digest,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    @property
    def context_ref(self) -> str:
        return f"research-context:{self.campaign_id}:round-{self.round_index}"

    def producer_view(self, producer_role: str) -> dict[str, Any]:
        """Return shared context plus only that role's provenance memory.

        Older Context payloads used ``by_role`` as a partial projection and
        omitted a role when it had no history.  Those payloads retain their
        historical fallback behavior.  Successor Contexts use the explicit
        ``global_memory``/``by_role`` split, so common feedback is no longer
        copied into all four role tails.
        """

        if producer_role not in DISCOVERY_PRODUCERS:
            raise ResearchLineInterfaceError("producer_role is outside the four-role portfolio")
        allocation = dict(self.policy.get("producer_token_allocation", ()))
        explicit_global = self.scientific_memory.get("global_memory")
        if not isinstance(explicit_global, Mapping):
            explicit_global = self.scientific_memory.get("global")
        has_explicit_global = isinstance(explicit_global, Mapping)
        shared_global = (
            dict(explicit_global)
            if has_explicit_global
            else {
                key: value
                for key, value in self.scientific_memory.items()
                if key != "by_role"
            }
        )
        role_table = self.scientific_memory.get("by_role")
        if isinstance(role_table, Mapping) and producer_role in role_table:
            candidate_role_memory = role_table.get(producer_role)
            role_memory = (
                candidate_role_memory
                if isinstance(candidate_role_memory, Mapping)
                else {}
            )
        elif has_explicit_global:
            role_memory = {
                "producer_role": producer_role,
                "provenance": {"history": "EMPTY"},
            }
        else:
            # Compatibility for pre-V2 contexts that did not materialize every
            # role's independent history yet.
            role_memory = self.scientific_memory

        return canonical_value(
            {
                "context_ref": self.context_ref,
                "context_digest": self.digest,
                "campaign_id": self.campaign_id,
                "round_index": self.round_index,
                "producer_role": producer_role,
                "knowledge_base": self.knowledge_base,
                "frozen_goal": self.frozen_goal,
                "frontier": self.frontier,
                "scientific_memory": self.scientific_memory,
                "memory": role_memory,
                "global_memory": shared_global,
                "role_memory": role_memory,
                "unresolved_questions": self.unresolved_questions,
                "policy": self.policy,
                "budget": self.budget,
                "active_profile_ref": self.active_profile_ref,
                "active_profile_digest": self.active_profile_digest,
                "protocol_ref": self.protocol_ref,
                "protocol_digest": self.protocol_digest,
                "producer_token_fraction": (
                    1.0 if self.budget.get("research_mode") == "director_sequential"
                    and producer_role == "frontier_architect"
                    else float(allocation.get(producer_role, 0.0))
                ),
                "mechanism_axis_targeting": self.policy.get(
                    "mechanism_axis_targeting", ()
                ),
                "memory_retrieval_policy": self.policy.get(
                    "memory_retrieval_policy", "UNSPECIFIED"
                ),
            }
        )

    @property
    def producer_inputs_digest(self) -> str:
        """Digest the actual scheduled Producer inputs used in this round."""

        return sha256_digest(
            tuple(self.producer_view(role) for role in research_producer_roles(self.budget))
        )


@dataclass(frozen=True, slots=True)
class ProducerOutcome:
    """One role's OpenSpec or typed failure, ready for Capability resolution."""

    producer_role: str
    context_ref: str
    context_digest: str
    spec: OpenResearchSpecV1 | None
    resolution_facts: Mapping[str, Any]
    source_proposal: CandidateProposalV4 | None = None
    source_mechanism_program: Mapping[str, Any] | None = None
    implementation_companion: MechanismImplementationCompanionV1 | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    schema = "recclaw.research-line.producer-outcome.v1"

    def __post_init__(self) -> None:
        if self.producer_role not in DISCOVERY_PRODUCERS:
            raise ResearchLineInterfaceError("producer_role is outside the four-role portfolio")
        _nonempty(self.context_ref, field_name="context_ref")
        object.__setattr__(
            self,
            "context_digest",
            validate_sha256(self.context_digest, field_name="context_digest"),
        )
        has_spec = self.spec is not None
        has_failure = self.failure_code is not None
        if has_spec == has_failure:
            raise ResearchLineInterfaceError("ProducerOutcome requires exactly one spec or failure")
        if has_spec:
            if not isinstance(self.spec, OpenResearchSpecV1):
                raise ResearchLineInterfaceError("spec must be OpenResearchSpecV1")
            if (
                self.spec.producer_role != self.producer_role
                or self.spec.context_ref != self.context_ref
                or self.spec.context_digest != self.context_digest
            ):
                raise ResearchLineInterfaceError("OpenSpec is not bound to its Producer context")
            if self.source_proposal is not None and not all(
                hasattr(self.source_proposal, field_name)
                for field_name in (
                    "candidate_id",
                    "producer_role",
                    "mechanism_program",
                    "digest",
                    "to_dict",
                )
            ):
                raise ResearchLineInterfaceError(
                    "source_proposal must expose the adapter proposal contract"
                )
            direct_source = self.source_mechanism_program is not None
            if direct_source != (self.implementation_companion is not None):
                raise ResearchLineInterfaceError(
                    "direct program source requires its implementation companion"
                )
            if direct_source:
                if self.source_proposal is not None:
                    raise ResearchLineInterfaceError(
                        "ProducerOutcome cannot carry fixed and direct program sources"
                    )
                if not isinstance(self.source_mechanism_program, Mapping):
                    raise ResearchLineInterfaceError(
                        "source_mechanism_program must be a mapping"
                    )
                object.__setattr__(
                    self,
                    "source_mechanism_program",
                    _snapshot(
                        self.source_mechanism_program,
                        field_name="source_mechanism_program",
                    ),
                )
                if not isinstance(
                    self.implementation_companion,
                    MechanismImplementationCompanionV1,
                ):
                    raise ResearchLineInterfaceError(
                        "implementation_companion has the wrong type"
                    )
        else:
            _nonempty(str(self.failure_code), field_name="failure_code")
            if self.source_proposal is not None:
                raise ResearchLineInterfaceError("failed Producer cannot carry a source proposal")
            if (
                self.source_mechanism_program is not None
                or self.implementation_companion is not None
            ):
                raise ResearchLineInterfaceError(
                    "failed Producer cannot carry a direct program source"
                )
        object.__setattr__(
            self,
            "resolution_facts",
            _snapshot(self.resolution_facts, field_name="resolution_facts"),
        )
        object.__setattr__(
            self,
            "provenance",
            _snapshot(self.provenance, field_name="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "producer_role": self.producer_role,
                "context_ref": self.context_ref,
                "context_digest": self.context_digest,
                "spec": self.spec.to_dict() if self.spec is not None else None,
                "resolution_facts": self.resolution_facts,
                "source_proposal": (
                    self.source_proposal.to_dict()
                    if self.source_proposal is not None
                    else None
                ),
                "source_mechanism_program": self.source_mechanism_program,
                "implementation_companion": (
                    self.implementation_companion.to_dict()
                    if self.implementation_companion is not None
                    else None
                ),
                "failure_code": self.failure_code,
                "failure_detail": self.failure_detail,
                "provenance": self.provenance,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class BehaviorProjection:
    """Decision inputs whose change proves that feedback affected a later round."""

    round_index: int
    context_ref: str
    context_digest: str
    profile_ref: str
    profile_digest: str
    policy_digest: str
    producer_inputs_digest: str
    producer_allocation: tuple[tuple[str, float], ...]
    axis_priorities: tuple[str, ...]
    memory_retrieval_policy: str
    acquisition_parameters: Mapping[str, Any]
    implementation_risk: Mapping[str, Any]

    schema = "recclaw.research-line.behavior-projection.v1"

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ResearchLineInterfaceError("round_index must be positive")
        for field_name in ("context_ref", "profile_ref", "memory_retrieval_policy"):
            _nonempty(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "context_digest",
            "profile_digest",
            "policy_digest",
            "producer_inputs_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name=field_name),
            )
        allocation = tuple(
            sorted((str(role), float(weight)) for role, weight in self.producer_allocation)
        )
        if set(role for role, _weight in allocation) != set(DISCOVERY_PRODUCERS):
            raise ResearchLineInterfaceError("producer_allocation must cover all four roles")
        object.__setattr__(self, "producer_allocation", allocation)
        object.__setattr__(
            self,
            "axis_priorities",
            tuple(dict.fromkeys(str(axis) for axis in self.axis_priorities)),
        )
        object.__setattr__(
            self,
            "acquisition_parameters",
            _snapshot(self.acquisition_parameters, field_name="acquisition_parameters"),
        )
        object.__setattr__(
            self,
            "implementation_risk",
            _snapshot(self.implementation_risk, field_name="implementation_risk"),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "round_index": self.round_index,
                "context_ref": self.context_ref,
                "context_digest": self.context_digest,
                "profile_ref": self.profile_ref,
                "profile_digest": self.profile_digest,
                "policy_digest": self.policy_digest,
                "producer_inputs_digest": self.producer_inputs_digest,
                "producer_allocation": self.producer_allocation,
                "axis_priorities": self.axis_priorities,
                "memory_retrieval_policy": self.memory_retrieval_policy,
                "acquisition_parameters": self.acquisition_parameters,
                "implementation_risk": self.implementation_risk,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def changed_fields(self, successor: "BehaviorProjection") -> tuple[str, ...]:
        if successor.round_index <= self.round_index:
            raise ResearchLineInterfaceError("successor must belong to a later round")
        ignored = {"schema", "round_index", "context_ref", "context_digest"}
        before = self.to_dict()
        after = successor.to_dict()
        return tuple(
            key
            for key in sorted(before)
            if key not in ignored and before[key] != after[key]
        )
