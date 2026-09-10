"""Outcome-blind, history-conditioned PortfolioCandidateV2 profiles.

This module is deliberately upstream of the Research Line runtime.  It turns
already-bound, pre-outcome identity and immutable history into one complete
``PortfolioCandidateV2``.  It does not run a candidate, inspect held-out data,
read a current result, or infer GPU cost from wall time.

The small injection contract is:

``build_portfolio_candidate_profile_v2(...)``
    returns a :class:`PortfolioCandidateProfileV2`; the complete candidate is
    ``result.candidate`` and the source/evidence digests are retained on the
    same immutable result.

``build_portfolio_candidate_v2(...)``
    is the convenience handoff returning only that candidate for existing
    consumers which require a ``PortfolioCandidateV2`` tuple.

Input records use explicit pre-outcome names.  Prior attempt summaries must
carry ``sealed`` and, when sealed, ``sealed_valid_seal``.  Stable prior
observations use ``stable`` plus ``stable_delta`` or
``stable_frontier_gain``.  Historical resource admission uses
``sealed_resource_admitted``.  A record at or after the temporal cutoff is an
error, rather than silently being used as if it were history.  A new candidate
without an exact pending/active task requires an explicit frozen
``exploration_mandate``; this module never fabricates a queue task.

Calibration is a bounded empirical hierarchy.  Let ``k`` be the documented
hierarchical shrinkage pseudo-count (default ``2``):

``p_level = (n * empirical_level + k * p_parent) / (n + k)``.

The levels are global -> compute pattern -> family -> exact candidate
identity.  When no sealed evidence exists, the caller must provide an
explicit frozen Beta calibration prior with positive ``alpha`` and ``beta``
and a source digest.  No calibration constant is fabricated in this module.
When global sealed evidence exists without that prior, the global empirical
posterior is the hierarchy root; an explicitly supplied prior is combined
with that global evidence before the subgroup shrinkage steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)

from .portfolio import (
    ParentValidationStateV2,
    PortfolioCandidateV2,
    ResourceAdmissionStateV2,
)


DEFAULT_SHRINKAGE_STRENGTH = 2.0
PROFILE_SCHEMA = "recclaw.research-line.portfolio-candidate-profile.v2"


class PortfolioProfileError(ValueError):
    """Raised when a profile cannot be built without an unsafe assumption."""


_OUTCOME_KEYS = frozenset(
    {
        "outcome",
        "result",
        "metric",
        "ndcg",
        "score",
        "test_score",
        "validation_score",
        "observed_outcome",
        "observed_effect",
        "observed_metric",
        "observed_result",
        "observed_delta",
        "current_outcome",
        "current_effect",
        "current_metric",
        "current_result",
        "current_delta",
        "current_score",
        "current_comparator_delta",
        "candidate_outcome",
        "candidate_effect",
        "candidate_metric",
        "candidate_result",
    }
)
_CURRENT_PREFIXES = (
    "current_outcome",
    "current_effect",
    "current_result",
    "current_metric",
    "current_delta",
    "current_score",
)
_CONTEXT_HISTORY_ROOTS = frozenset(
    {"frontier", "scientific_memory", "knowledge_base", "frozen_goal"}
)

_PRIOR_ATTEMPT_KEYS = frozenset(
    {
        "round",
        "round_index",
        "candidate_id",
        "semantic_digest",
        "candidate_semantic_digest",
        "family_id",
        "parent_id",
        "parent_candidate_id",
        "compute_pattern",
        "sealed",
        "sealed_valid_seal",
        "sealed_resource_admitted",
        "sealed_parent_validation",
        "attempt_ref",
        "attempt_digest",
        "source_digest",
    }
)
_FAMILY_HISTORY_KEYS = frozenset(
    {
        "round",
        "round_index",
        "family_id",
        "candidate_id",
        "semantic_digest",
        "candidate_semantic_digest",
        "stable",
        "stable_delta",
        "source_digest",
    }
)
_PARENT_HISTORY_KEYS = frozenset(
    {
        "round",
        "round_index",
        "parent_id",
        "parent_candidate_id",
        "stable",
        "stable_validation",
        "stable_lineage_risk",
        "stable_delta",
        "source_digest",
    }
)
_FRONTIER_HISTORY_KEYS = frozenset(
    {
        "round",
        "round_index",
        "family_id",
        "candidate_id",
        "semantic_digest",
        "candidate_semantic_digest",
        "stable",
        "stable_frontier_gain",
        "source_digest",
    }
)


def _fail(message: str) -> None:
    raise PortfolioProfileError(message)


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        try:
            return dict(canonical_value(dict(value)))
        except (TypeError, ValueError) as error:
            raise PortfolioProfileError(f"{name} is not canonicalizable") from error
    for method_name in ("to_dict", "canonical_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                projected = method()
            except Exception as error:  # pragma: no cover - foreign contract boundary.
                raise PortfolioProfileError(f"{name}.{method_name} failed") from error
            if isinstance(projected, Mapping):
                try:
                    return dict(canonical_value(dict(projected)))
                except (TypeError, ValueError) as error:
                    raise PortfolioProfileError(
                        f"{name} is not canonicalizable"
                    ) from error
    if is_dataclass(value):
        try:
            return dict(
                canonical_value(
                    {
                        item.name: getattr(value, item.name)
                        for item in fields(value)
                    }
                )
            )
        except (TypeError, ValueError) as error:
            raise PortfolioProfileError(f"{name} is not canonicalizable") from error
    raise PortfolioProfileError(f"{name} must be a mapping or canonical record")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _digest(value: Any, *, name: str) -> str:
    supplied = getattr(value, "digest", None)
    if isinstance(supplied, str) and _is_sha256(supplied):
        return supplied
    try:
        result = sha256_digest(_as_mapping(value, name=name))
    except (TypeError, ValueError) as error:
        raise PortfolioProfileError(f"{name} digest cannot be computed") from error
    if not _is_sha256(result):  # pragma: no cover - canonical helper invariant.
        raise PortfolioProfileError(f"{name} digest is not SHA-256")
    return result


def _require_digest(value: Any, *, name: str) -> str:
    try:
        return validate_sha256(value, field_name=name)
    except (TypeError, ValueError) as error:
        raise PortfolioProfileError(f"{name} must be a SHA-256 digest") from error


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PortfolioProfileError(f"{name} must be a normalized non-empty string")
    return value


def _finite(value: Any, *, name: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortfolioProfileError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PortfolioProfileError(f"{name} must be finite")
    if low is not None and result < low:
        raise PortfolioProfileError(f"{name} must be >= {low}")
    if high is not None and result > high:
        raise PortfolioProfileError(f"{name} must be <= {high}")
    return result


def _integer(value: Any, *, name: str, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise PortfolioProfileError(f"{name} must be an integer >= {low}")
    return int(value)


def _walk_no_leak(
    value: Any,
    *,
    name: str,
    context: bool = False,
    path: tuple[str, ...] = (),
) -> None:
    """Reject current/effect-bearing keys without rejecting goal metadata.

    A typed ResearchContext legitimately contains ``frozen_goal.metric`` and
    historical scientific-memory fields.  The builder never reads those
    payloads, so context scanning permits those established namespaces while
    still rejecting explicit current-attempt markers.  The other input
    domains are strict: a generic outcome/effect field is never an accepted
    profile input there.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            allow_goal_metric = (
                context
                and path
                and path[0] in _CONTEXT_HISTORY_ROOTS
                and key == "metric"
                and path[0] == "frozen_goal"
            )
            explicit_current = key in _OUTCOME_KEYS or key.startswith(_CURRENT_PREFIXES)
            if explicit_current and not allow_goal_metric:
                # ``predicted_outcome_signature`` is a hypothesis, not an
                # observed outcome, and is part of the existing binding.
                if key != "predicted_outcome_signature":
                    raise PortfolioProfileError(
                        f"{name} contains current outcome/effect field {raw_key!r}"
                    )
            if not context and key in {"metric", "ndcg", "score"}:
                raise PortfolioProfileError(
                    f"{name} contains outcome-bearing field {raw_key!r}"
                )
            _walk_no_leak(
                child,
                name=name,
                context=context,
                path=(*path, key),
            )
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _walk_no_leak(
                child,
                name=name,
                context=context,
                path=(*path, str(index)),
            )


def _present(mapping: Mapping[str, Any], *keys: str) -> list[tuple[str, Any]]:
    return [
        (key, mapping[key])
        for key in keys
        if key in mapping and mapping[key] is not None
    ]


def _consistent(
    values: Sequence[tuple[str, Any]],
    *,
    name: str,
    allow_none: bool = True,
) -> Any:
    if not values:
        return None
    normalized = [value for _source, value in values if value is not None]
    if not normalized and allow_none:
        return None
    if not normalized:
        _fail(f"{name} is missing")
    first = normalized[0]
    if any(value != first for value in normalized[1:]):
        details = ", ".join(f"{source}={value!r}" for source, value in values)
        _fail(f"{name} identity mismatch: {details}")
    return first


def _round_of(row: Mapping[str, Any], *, name: str) -> int:
    values = _present(row, "round", "round_index")
    if not values:
        _fail(f"{name} requires round")
    result = _integer(values[0][1], name=f"{name}.round", low=1)
    if len(values) > 1 and any(int(value) != result for _key, value in values[1:]):
        _fail(f"{name} has inconsistent round fields")
    return result


def _task_created_round(task: Mapping[str, Any], *, name: str) -> int:
    """Read the queue contract's ``created_round`` without weakening history rows."""

    values = _present(task, "created_round", "round", "round_index")
    if not values:
        _fail(f"{name} requires created_round")
    result = _integer(values[0][1], name=f"{name}.created_round", low=1)
    if len(values) > 1 and any(int(value) != result for _key, value in values[1:]):
        _fail(f"{name} has inconsistent created-round fields")
    return result


def _rows(
    value: Any,
    *,
    name: str,
    container_keys: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        candidates = [key for key in container_keys if key in value]
        if len(candidates) != 1:
            raise PortfolioProfileError(
                f"{name} must be a sequence or one of {tuple(container_keys)!r}"
            )
        value = value[candidates[0]]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PortfolioProfileError(f"{name} must be a sequence")
    result = []
    for index, item in enumerate(value):
        result.append(_as_mapping(item, name=f"{name}[{index}]"))
    return tuple(result)


def _check_history_row_keys(
    row: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    name: str,
) -> None:
    _walk_no_leak(row, name=name)
    unknown = sorted(str(key) for key in row if str(key) not in allowed)
    if unknown:
        raise PortfolioProfileError(
            f"{name} contains unsupported fields: {', '.join(unknown)}"
        )


def _validate_temporal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    cutoff: int,
) -> tuple[dict[str, Any], ...]:
    normalized = []
    for index, row in enumerate(rows):
        row_name = f"{name}[{index}]"
        round_index = _round_of(row, name=row_name)
        if round_index >= cutoff:
            raise PortfolioProfileError(
                f"{row_name} is at/after temporal cutoff {cutoff}"
            )
        normalized.append(dict(row))
    return tuple(normalized)


def _semantic_of(row: Mapping[str, Any], *, name: str) -> str:
    value = _consistent(
        _present(row, "semantic_digest", "candidate_semantic_digest"),
        name=f"{name}.semantic_digest",
    )
    if value is None:
        _fail(f"{name} requires semantic_digest")
    return _text(value, name=f"{name}.semantic_digest")


def _candidate_identity(
    candidate_identity: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    identity = dict(candidate_identity)
    _walk_no_leak(identity, name="candidate_identity")
    candidate_id = _consistent(
        _present(identity, "candidate_id"),
        name="candidate_id",
    )
    semantic_digest = _consistent(
        _present(identity, "semantic_digest", "semantic_identity_digest", "candidate_semantic_digest"),
        name="semantic_digest",
    )
    family_id = _consistent(_present(identity, "family_id"), name="family_id")
    compute_pattern = _consistent(
        _present(identity, "compute_pattern"),
        name="compute_pattern",
    )
    if candidate_id is None or semantic_digest is None or family_id is None or compute_pattern is None:
        _fail(
            "candidate_identity requires candidate_id, semantic_digest, family_id, and compute_pattern"
        )
    candidate_id = _text(candidate_id, name="candidate_id")
    semantic_digest = _text(semantic_digest, name="semantic_digest")
    family_id = _text(family_id, name="family_id")
    compute_pattern = _text(compute_pattern, name="compute_pattern")

    proposal = binding.get("proposal")
    proposal_map = proposal if isinstance(proposal, Mapping) else {}
    binding_candidate_id = _consistent(
        _present(binding, "candidate_id")
        + _present(proposal_map, "candidate_id"),
        name="binding.candidate_id",
    )
    if binding_candidate_id is not None and binding_candidate_id != candidate_id:
        _fail("binding candidate_id does not match candidate_identity")
    binding_semantic = _consistent(
        _present(binding, "mechanism_semantics_digest", "semantic_digest", "semantic_identity_digest")
        + _present(
            proposal_map,
            "mechanism_semantics_digest",
            "semantic_digest",
            "semantic_identity_digest",
        ),
        name="binding.semantic_digest",
    )
    if binding_semantic is not None and binding_semantic != semantic_digest:
        _fail("binding semantic identity does not match candidate_identity")
    parent_id = _consistent(
        _present(identity, "parent_id", "parent_candidate_id"),
        name="parent_id",
    )
    binding_parent = _consistent(
        _present(binding, "parent_id", "parent_candidate_id")
        + _present(proposal_map, "parent_id", "parent_candidate_id"),
        name="binding.parent_id",
    )
    if binding_parent is not None and binding_parent != parent_id:
        _fail("binding parent identity does not match candidate_identity")
    if parent_id is not None:
        parent_id = _text(parent_id, name="parent_id")

    package_digest = _consistent(
        _present(identity, "candidate_package_digest", "package_digest")
        + _present(proposal_map, "candidate_package_digest", "package_digest"),
        name="candidate_package_digest",
    )
    if package_digest is not None:
        package_digest = _require_digest(
            package_digest,
            name="candidate_package_digest",
        )
    source_digest = _consistent(
        _present(identity, "candidate_source_sha256", "source_sha256"),
        name="candidate_source_sha256",
    )
    if source_digest is not None:
        source_digest = _require_digest(
            source_digest,
            name="candidate_source_sha256",
        )
    resource_ref = _consistent(
        _present(identity, "resource_candidate_ref", "candidate_ref", "capability_ref")
        + _present(binding, "capability_ref"),
        name="resource_candidate_ref",
    )
    if resource_ref is not None:
        resource_ref = _text(resource_ref, name="resource_candidate_ref")
    else:
        resource_ref = candidate_id
    return {
        "candidate_id": candidate_id,
        "semantic_digest": semantic_digest,
        "family_id": family_id,
        "parent_id": parent_id,
        "compute_pattern": compute_pattern,
        "candidate_package_digest": package_digest,
        "candidate_source_sha256": source_digest,
        "resource_candidate_ref": resource_ref,
        "identity": canonical_value(identity),
    }


def _context_projection(value: Any) -> tuple[dict[str, Any], int]:
    mapping = _as_mapping(value, name="current_context")
    _walk_no_leak(mapping, name="current_context", context=True)
    round_index = _integer(
        mapping.get("round_index", mapping.get("round")),
        name="current_context.round_index",
        low=1,
    )
    campaign_id = _text(
        mapping.get("campaign_id", "research-line"),
        name="current_context.campaign_id",
    )
    projection = {
        "campaign_id": campaign_id,
        "round_index": round_index,
        "active_profile_ref": mapping.get("active_profile_ref"),
        "active_profile_digest": mapping.get("active_profile_digest"),
        "protocol_ref": mapping.get("protocol_ref"),
        "protocol_digest": mapping.get("protocol_digest"),
    }
    for field_name in ("active_profile_digest", "protocol_digest"):
        if projection[field_name] is not None:
            projection[field_name] = _require_digest(
                projection[field_name],
                name=f"current_context.{field_name}",
            )
    return canonical_value(projection), round_index


def _binding_and_resolution(
    *,
    binding: Any,
    resolution: Any,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    binding_map = _as_mapping(binding, name="binding")
    resolution_map = _as_mapping(resolution, name="resolution")
    _walk_no_leak(binding_map, name="binding")
    _walk_no_leak(resolution_map, name="resolution")
    if not binding_map:
        _fail("binding is empty")
    if not resolution_map:
        _fail("resolution is empty")
    resolution_value = resolution_map.get("resolution", resolution_map.get("status"))
    if resolution_value is None:
        _fail("resolution requires a resolution/status field")
    expected_binding_digest = identity.get("binding_digest")
    binding_digest = _digest(binding, name="binding")
    if expected_binding_digest is not None and _require_digest(
        expected_binding_digest,
        name="candidate_identity.binding_digest",
    ) != binding_digest:
        _fail("binding digest does not match candidate_identity")
    expected_resolution_digest = identity.get("resolution_digest")
    resolution_digest = _digest(resolution, name="resolution")
    if expected_resolution_digest is not None and _require_digest(
        expected_resolution_digest,
        name="candidate_identity.resolution_digest",
    ) != resolution_digest:
        _fail("resolution digest does not match candidate_identity")
    return binding_map, resolution_map, binding_digest, resolution_digest


def _task_from_queue(
    task_queue: Any,
    *,
    identity: Mapping[str, Any],
    binding: Mapping[str, Any],
    current_round: int,
    exploration_mandate: Any,
) -> tuple[
    dict[str, Any] | None,
    str,
    int,
    float,
    str,
    dict[str, Any],
]:
    queue_map = _as_mapping(task_queue, name="task_queue")
    _walk_no_leak(queue_map, name="task_queue")
    raw_tasks = queue_map.get("tasks")
    if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, Sequence):
        raise PortfolioProfileError("task_queue.tasks must be a sequence")
    candidate_id = str(identity["candidate_id"])
    semantic_digest = str(identity["semantic_digest"])
    matches: list[dict[str, Any]] = []
    other_pending_or_active = False
    for index, raw_task in enumerate(raw_tasks):
        task = _as_mapping(raw_task, name=f"task_queue.tasks[{index}]")
        task_id = _text(task.get("task_id"), name=f"task_queue.tasks[{index}].task_id")
        task_candidate = _text(
            task.get("candidate_id"),
            name=f"task_queue.tasks[{index}].candidate_id",
        )
        task_semantic = _semantic_of(task, name=f"task_queue.tasks[{index}]")
        if task_candidate == candidate_id and task_semantic != semantic_digest:
            _fail("task queue candidate semantic identity mismatch")
        status = str(task.get("status", "")).strip()
        exact_identity = (
            task_candidate == candidate_id and task_semantic == semantic_digest
        )
        if not exact_identity:
            if status in {"PENDING", "ACTIVE"}:
                other_pending_or_active = True
            continue
        if status not in {"PENDING", "ACTIVE"}:
            continue
        created_round = _task_created_round(task, name=f"task_queue.tasks[{index}]")
        if created_round > current_round:
            _fail("task queue task is created in the future")
        priority = _finite(
            task.get("priority"),
            name=f"task_queue.tasks[{index}].priority",
            low=0.0,
            high=1.0,
        )
        if priority <= 0.0:
            _fail("matching task priority must be positive")
        task_parent = task.get("parent_candidate_id", task.get("parent_id"))
        if task_parent != identity.get("parent_id"):
            _fail("task queue parent identity mismatch")
        matches.append(
            {
                **task,
                "task_id": task_id,
                "created_round": created_round,
                "priority": priority,
            }
        )
    if not matches:
        mandate, mandate_digest, age_rounds, priority = _exploration_mandate_view(
            exploration_mandate,
            binding=binding,
            current_round=current_round,
            queue_has_other_pending_or_active=other_pending_or_active,
        )
        return (
            None,
            _digest(queue_map, name="task_queue"),
            age_rounds,
            priority,
            "EXPLORATION_MANDATE",
            {
                "mandate": mandate,
                "mandate_digest": mandate_digest,
                "queue_has_other_pending_or_active": other_pending_or_active,
            },
        )
    selected = min(
        matches,
        key=lambda row: (-float(row["priority"]), int(row["created_round"]), row["task_id"]),
    )
    age_rounds = current_round - int(selected["created_round"])
    return (
        selected,
        _digest(queue_map, name="task_queue"),
        age_rounds,
        float(selected["priority"]),
        "TASK",
        {"task_record_digest": sha256_digest(selected)},
    )


def _exploration_mandate_view(
    value: Any,
    *,
    binding: Mapping[str, Any],
    current_round: int,
    queue_has_other_pending_or_active: bool,
) -> tuple[dict[str, Any], str, int, float]:
    if value is None:
        _fail(
            "exact candidate task is absent; an explicit exploration_mandate is required"
        )
    mandate = _as_mapping(value, name="exploration_mandate")
    _walk_no_leak(mandate, name="exploration_mandate")
    if _text(mandate.get("kind"), name="exploration_mandate.kind") != "EXPLORATION":
        _fail("exploration_mandate.kind must be EXPLORATION")
    if mandate.get("frozen") is not True:
        _fail("exploration_mandate must be explicitly frozen")
    source_digest = _require_digest(
        mandate.get("source_digest"),
        name="exploration_mandate.source_digest",
    )
    created_round = _integer(
        mandate.get("created_round"),
        name="exploration_mandate.created_round",
        low=1,
    )
    if created_round > current_round:
        _fail("exploration_mandate is created in the future")
    priority = _finite(
        mandate.get("priority"),
        name="exploration_mandate.priority",
        low=0.0,
        high=1.0,
    )
    if priority <= 0.0:
        _fail("exploration_mandate.priority must be positive")
    producer_role = _text(
        mandate.get("producer_role"),
        name="exploration_mandate.producer_role",
    )
    proposal = binding.get("proposal")
    if not isinstance(proposal, Mapping):
        _fail("exploration_mandate requires binding.proposal.producer_role")
    proposal_role = _text(
        proposal.get("producer_role"),
        name="binding.proposal.producer_role",
    )
    if producer_role != proposal_role:
        _fail("exploration_mandate producer_role does not match binding proposal")
    floor_authorized = mandate.get("exploration_floor_authorized")
    if floor_authorized is not None and not isinstance(floor_authorized, bool):
        _fail("exploration_mandate.exploration_floor_authorized must be boolean")
    if queue_has_other_pending_or_active and floor_authorized is not True:
        _fail(
            "queued confirmation/control task exists; exploration_mandate must "
            "explicitly set exploration_floor_authorized=true"
        )
    source_ref = mandate.get("source_ref")
    if source_ref is not None:
        source_ref = _text(source_ref, name="exploration_mandate.source_ref")
    normalized = dict(mandate)
    normalized.update(
        {
            "kind": "EXPLORATION",
            "frozen": True,
            "source_digest": source_digest,
            "created_round": created_round,
            "priority": priority,
            "producer_role": producer_role,
        }
    )
    if source_ref is not None:
        normalized["source_ref"] = source_ref
    return (
        canonical_value(normalized),
        _digest(mandate, name="exploration_mandate"),
        0,
        priority,
    )


def _normalize_histories(
    *,
    prior_attempts: Any,
    family_history: Any,
    parent_history: Any,
    frontier_history: Any,
    cutoff: int,
    identity: Mapping[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    attempts = _rows(
        prior_attempts,
        name="prior_attempts",
        container_keys=("attempts", "records", "history"),
    )
    families = _rows(
        family_history,
        name="family_history",
        container_keys=("observations", "records", "history"),
    )
    parents = _rows(
        parent_history,
        name="parent_history",
        container_keys=("observations", "records", "history"),
    )
    frontiers = _rows(
        frontier_history,
        name="frontier_history",
        container_keys=("observations", "records", "history"),
    )
    for index, row in enumerate(attempts):
        _check_history_row_keys(
            row,
            allowed=_PRIOR_ATTEMPT_KEYS,
            name=f"prior_attempts[{index}]",
        )
        if not isinstance(row.get("sealed"), bool):
            _fail(f"prior_attempts[{index}].sealed must be boolean")
        candidate_id = _text(
            row.get("candidate_id"),
            name=f"prior_attempts[{index}].candidate_id",
        )
        semantic = _semantic_of(row, name=f"prior_attempts[{index}]")
        family_id = _text(
            row.get("family_id"),
            name=f"prior_attempts[{index}].family_id",
        )
        compute_pattern = _text(
            row.get("compute_pattern"),
            name=f"prior_attempts[{index}].compute_pattern",
        )
        row["candidate_id"] = candidate_id
        row["semantic_digest"] = semantic
        row["family_id"] = family_id
        row["compute_pattern"] = compute_pattern
        if candidate_id == identity["candidate_id"] and semantic != identity["semantic_digest"]:
            _fail("prior attempt exact candidate has semantic identity mismatch")
        if row["sealed"] and not isinstance(row.get("sealed_valid_seal"), bool):
            _fail(f"prior_attempts[{index}] sealed row requires sealed_valid_seal")
        for key in ("sealed_valid_seal", "sealed_resource_admitted"):
            if key in row and row[key] is not None and not isinstance(row[key], bool):
                _fail(f"prior_attempts[{index}].{key} must be boolean")
        if "sealed_parent_validation" in row and row["sealed_parent_validation"] is not None:
            row["sealed_parent_validation"] = str(row["sealed_parent_validation"])
    attempts = _validate_temporal_rows(
        attempts,
        name="prior_attempts",
        cutoff=cutoff,
    )

    for index, row in enumerate(families):
        _check_history_row_keys(
            row,
            allowed=_FAMILY_HISTORY_KEYS,
            name=f"family_history[{index}]",
        )
        _integer(_round_of(row, name=f"family_history[{index}]"), name="family_history.round", low=1)
        family_id = _text(row.get("family_id"), name=f"family_history[{index}].family_id")
        if not isinstance(row.get("stable"), bool):
            _fail(f"family_history[{index}].stable must be boolean")
        row["family_id"] = family_id
        if row["stable"]:
            row["stable_delta"] = _finite(
                row.get("stable_delta"),
                name=f"family_history[{index}].stable_delta",
                low=-1.0,
                high=1.0,
            )
        if "candidate_id" in row and row["candidate_id"] is not None:
            row["candidate_id"] = _text(row["candidate_id"], name=f"family_history[{index}].candidate_id")
        if "semantic_digest" in row or "candidate_semantic_digest" in row:
            row["semantic_digest"] = _semantic_of(row, name=f"family_history[{index}]")
            if row.get("candidate_id") == identity["candidate_id"] and row["semantic_digest"] != identity["semantic_digest"]:
                _fail("family history exact candidate has semantic identity mismatch")
    families = _validate_temporal_rows(families, name="family_history", cutoff=cutoff)

    for index, row in enumerate(parents):
        _check_history_row_keys(
            row,
            allowed=_PARENT_HISTORY_KEYS,
            name=f"parent_history[{index}]",
        )
        _round_of(row, name=f"parent_history[{index}]")
        row["parent_id"] = _text(
            row.get("parent_id", row.get("parent_candidate_id")),
            name=f"parent_history[{index}].parent_id",
        )
        if not isinstance(row.get("stable"), bool):
            _fail(f"parent_history[{index}].stable must be boolean")
        if row["stable"]:
            if row.get("stable_validation") is None:
                _fail(f"parent_history[{index}] stable row requires stable_validation")
            row["stable_validation"] = str(row["stable_validation"])
            if row["stable_validation"] not in {
                "INDEPENDENT",
                "VALIDATED",
                "UNVERIFIED",
                "FAILED",
                "UNKNOWN",
            }:
                _fail(f"parent_history[{index}] has invalid stable_validation")
            if "stable_lineage_risk" in row and row["stable_lineage_risk"] is not None:
                row["stable_lineage_risk"] = _finite(
                    row["stable_lineage_risk"],
                    name=f"parent_history[{index}].stable_lineage_risk",
                    low=0.0,
                    high=1.0,
                )
            if "stable_delta" in row and row["stable_delta"] is not None:
                row["stable_delta"] = _finite(
                    row["stable_delta"],
                    name=f"parent_history[{index}].stable_delta",
                    low=-1.0,
                    high=1.0,
                )
    parents = _validate_temporal_rows(parents, name="parent_history", cutoff=cutoff)

    for index, row in enumerate(frontiers):
        _check_history_row_keys(
            row,
            allowed=_FRONTIER_HISTORY_KEYS,
            name=f"frontier_history[{index}]",
        )
        _round_of(row, name=f"frontier_history[{index}]")
        if not isinstance(row.get("stable"), bool):
            _fail(f"frontier_history[{index}].stable must be boolean")
        if row.get("family_id") is not None:
            row["family_id"] = _text(row["family_id"], name=f"frontier_history[{index}].family_id")
        if row.get("candidate_id") is not None:
            row["candidate_id"] = _text(row["candidate_id"], name=f"frontier_history[{index}].candidate_id")
        if "semantic_digest" in row or "candidate_semantic_digest" in row:
            row["semantic_digest"] = _semantic_of(row, name=f"frontier_history[{index}]")
            if row.get("candidate_id") == identity["candidate_id"] and row["semantic_digest"] != identity["semantic_digest"]:
                _fail("frontier history exact candidate has semantic identity mismatch")
        if row["stable"]:
            row["stable_frontier_gain"] = _finite(
                row.get("stable_frontier_gain"),
                name=f"frontier_history[{index}].stable_frontier_gain",
                low=-1.0,
                high=1.0,
            )
    frontiers = _validate_temporal_rows(
        frontiers,
        name="frontier_history",
        cutoff=cutoff,
    )
    return attempts, families, parents, frontiers


def _raw_rate(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[int, int, float | None]:
    if not rows:
        return 0, 0, None
    successes = sum(bool(row[key]) for row in rows)
    return len(rows), successes, successes / len(rows)


def _calibration_prior_view(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    prior = _as_mapping(value, name="calibration_prior")
    _walk_no_leak(prior, name="calibration_prior")
    if prior.get("frozen") is not True:
        _fail("calibration_prior must be explicitly frozen")
    alpha = _finite(
        prior.get("alpha"),
        name="calibration_prior.alpha",
        low=0.0,
    )
    beta = _finite(
        prior.get("beta"),
        name="calibration_prior.beta",
        low=0.0,
    )
    if alpha <= 0.0 or beta <= 0.0:
        _fail("calibration_prior alpha and beta must be positive")
    if not math.isfinite(alpha + beta):
        _fail("calibration_prior alpha plus beta must be finite")
    source_digest = _require_digest(
        prior.get("source_digest"),
        name="calibration_prior.source_digest",
    )
    source_ref = prior.get("source_ref")
    if source_ref is not None:
        source_ref = _text(source_ref, name="calibration_prior.source_ref")
    return canonical_value(
        {
            "alpha": alpha,
            "beta": beta,
            "frozen": True,
            "source_digest": source_digest,
            "source_ref": source_ref,
        }
    )


def _calibrate(
    attempts: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    shrinkage_strength: float,
    calibration_prior: Mapping[str, Any] | None,
) -> tuple[float, float, dict[str, Any]]:
    sealed = [row for row in attempts if row.get("sealed") is True]
    compute = [
        row for row in sealed if row.get("compute_pattern") == identity["compute_pattern"]
    ]
    family = [row for row in sealed if row.get("family_id") == identity["family_id"]]
    exact = [
        row
        for row in sealed
        if row.get("candidate_id") == identity["candidate_id"]
        and row.get("semantic_digest") == identity["semantic_digest"]
    ]
    levels: dict[str, Any] = {}
    prior_mean = None
    if calibration_prior is not None:
        prior_mean = float(calibration_prior["alpha"]) / (
            float(calibration_prior["alpha"])
            + float(calibration_prior["beta"])
        )
    global_count, global_valid_count, global_raw = _raw_rate(
        sealed,
        "sealed_valid_seal",
    )
    if global_raw is None:
        if prior_mean is None:
            _fail(
                "no sealed prior attempts are available; an explicit frozen "
                "calibration_prior is required"
            )
        global_probability = prior_mean
        global_source = "EXPLICIT_BETA_CALIBRATION_PRIOR"
    elif calibration_prior is None:
        global_probability = global_raw
        global_source = "SEALED_GLOBAL_EMPIRICAL_POSTERIOR"
    else:
        alpha = float(calibration_prior["alpha"])
        beta = float(calibration_prior["beta"])
        global_probability = (alpha + global_valid_count) / (
            alpha + beta + global_count
        )
        global_source = "BETA_PRIOR_PLUS_SEALED_GLOBAL_POSTERIOR"
    global_probability = min(1.0, max(0.0, float(global_probability)))
    levels["global"] = {
        "attempt_count": global_count,
        "valid_seal_count": global_valid_count,
        "raw_probability": global_raw,
        "probability": global_probability,
        "source": global_source,
    }
    parent_probability = global_probability
    for level_name, rows in (
        ("compute", compute),
        ("family", family),
        ("exact", exact),
    ):
        count, valid_count, raw = _raw_rate(rows, "sealed_valid_seal")
        if raw is None:
            probability = parent_probability
        else:
            probability = (
                count * raw + shrinkage_strength * parent_probability
            ) / (count + shrinkage_strength)
        probability = min(1.0, max(0.0, float(probability)))
        levels[level_name] = {
            "attempt_count": count,
            "valid_seal_count": valid_count,
            "raw_probability": raw,
            "probability": probability,
        }
        parent_probability = probability
    exact_count = int(levels["exact"]["attempt_count"])
    uncertainty = 1.0 - exact_count / (exact_count + shrinkage_strength)
    evidence = {
        "method": "BOUNDED_BETA_SEALED_GLOBAL_COMPUTE_FAMILY_EXACT_V2",
        "shrinkage_strength": shrinkage_strength,
        "calibration_prior": calibration_prior,
        "levels": levels,
        "selected_level": "exact",
        "valid_seal_probability": parent_probability,
        "exploration_uncertainty": uncertainty,
    }
    return parent_probability, uncertainty, canonical_value(evidence)


def _stable_mean(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[int, float | None]:
    values = [row[key] for row in rows if row.get("stable") is True and key in row]
    if not values:
        return 0, None
    return len(values), sum(float(value) for value in values) / len(values)


def _shrink_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    target: Any,
    target_key: str,
    low: float,
    high: float,
    shrinkage_strength: float,
) -> tuple[float, dict[str, Any]]:
    stable_rows = [row for row in rows if row.get("stable") is True and key in row]
    global_count, global_mean = _stable_mean(rows, key)
    target_rows = [row for row in stable_rows if row.get(target_key) == target]
    target_count, target_mean = _stable_mean(target_rows, key)
    parent = 0.0 if global_mean is None else float(global_mean)
    if target_mean is None:
        result = parent
    else:
        result = (
            target_count * float(target_mean) + shrinkage_strength * parent
        ) / (target_count + shrinkage_strength)
    result = min(high, max(low, float(result)))
    return result, {
        "observation_count": target_count,
        "raw_value": target_mean,
        "global_observation_count": global_count,
        "global_raw_value": global_mean,
        "value": result,
        "target": target,
        "stable_only": True,
    }


def _frontier_prior(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    shrinkage_strength: float,
) -> tuple[float, dict[str, Any]]:
    stable_rows = [
        row
        for row in rows
        if row.get("stable") is True and "stable_frontier_gain" in row
    ]
    global_count, global_mean = _stable_mean(stable_rows, "stable_frontier_gain")
    family_rows = [
        row for row in stable_rows if row.get("family_id") == identity["family_id"]
    ]
    family_count, family_mean = _stable_mean(family_rows, "stable_frontier_gain")
    exact_rows = [
        row
        for row in family_rows
        if row.get("candidate_id") == identity["candidate_id"]
        and (
            row.get("semantic_digest") is None
            or row.get("semantic_digest") == identity["semantic_digest"]
        )
    ]
    exact_count, exact_mean = _stable_mean(exact_rows, "stable_frontier_gain")
    p_global = 0.0 if global_mean is None else float(global_mean)
    p_family = (
        p_global
        if family_mean is None
        else (family_count * float(family_mean) + shrinkage_strength * p_global)
        / (family_count + shrinkage_strength)
    )
    p_exact = (
        p_family
        if exact_mean is None
        else (exact_count * float(exact_mean) + shrinkage_strength * p_family)
        / (exact_count + shrinkage_strength)
    )
    value = min(1.0, max(-1.0, float(p_exact)))
    return value, {
        "global_observation_count": global_count,
        "global_raw_value": global_mean,
        "family_observation_count": family_count,
        "family_raw_value": family_mean,
        "exact_observation_count": exact_count,
        "exact_raw_value": exact_mean,
        "value": value,
        "stable_only": True,
    }


def _parent_view(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    shrinkage_strength: float,
) -> tuple[ParentValidationStateV2, float, float, dict[str, Any]]:
    parent_id = identity.get("parent_id")
    if parent_id is None:
        return (
            ParentValidationStateV2.INDEPENDENT,
            0.0,
            0.0,
            {
                "parent_id": None,
                "stable_observation_count": 0,
                "state": ParentValidationStateV2.INDEPENDENT.value,
                "lineage_risk": 0.0,
            },
        )
    target_rows = [
        row for row in rows if row.get("stable") is True and row.get("parent_id") == parent_id
    ]
    ordered = sorted(target_rows, key=lambda row: _round_of(row, name="parent_history"))
    state = ParentValidationStateV2.UNKNOWN
    if ordered:
        state_value = str(ordered[-1].get("stable_validation", "UNKNOWN"))
        state = ParentValidationStateV2(state_value)
    risk_rows = [row for row in target_rows if "stable_lineage_risk" in row]
    if risk_rows:
        global_rows = [row for row in rows if row.get("stable") is True and "stable_lineage_risk" in row]
        global_mean = sum(float(row["stable_lineage_risk"]) for row in global_rows) / len(global_rows)
        target_mean = sum(float(row["stable_lineage_risk"]) for row in risk_rows) / len(risk_rows)
        lineage_risk = (
            len(risk_rows) * target_mean + shrinkage_strength * global_mean
        ) / (len(risk_rows) + shrinkage_strength)
        lineage_evidence = len(risk_rows)
    else:
        failed = sum(
            row.get("stable_validation") == ParentValidationStateV2.FAILED.value
            for row in target_rows
        )
        validated = sum(
            row.get("stable_validation") == ParentValidationStateV2.VALIDATED.value
            for row in target_rows
        )
        total = failed + validated
        lineage_risk = failed / total if total else 0.0
        if total:
            lineage_risk = total * lineage_risk / (total + shrinkage_strength)
        lineage_evidence = total
    parent_delta, delta_evidence = _shrink_value(
        rows,
        key="stable_delta",
        target=parent_id,
        target_key="parent_id",
        low=-1.0,
        high=1.0,
        shrinkage_strength=shrinkage_strength,
    )
    detail = {
        "parent_id": parent_id,
        "stable_observation_count": len(target_rows),
        "state": state.value,
        "lineage_risk": min(1.0, max(0.0, float(lineage_risk))),
        "lineage_evidence_count": lineage_evidence,
        "parent_delta": delta_evidence,
    }
    return state, float(detail["lineage_risk"]), parent_delta, canonical_value(detail)


def _compute_risk(
    attempts: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    shrinkage_strength: float,
) -> tuple[float, dict[str, Any]]:
    resource_rows = [
        row
        for row in attempts
        if row.get("sealed") is True and isinstance(row.get("sealed_resource_admitted"), bool)
    ]
    compute_rows = [
        row
        for row in resource_rows
        if row.get("compute_pattern") == identity["compute_pattern"]
    ]
    global_failures = sum(not bool(row["sealed_resource_admitted"]) for row in resource_rows)
    global_raw = global_failures / len(resource_rows) if resource_rows else None
    compute_failures = sum(not bool(row["sealed_resource_admitted"]) for row in compute_rows)
    compute_raw = compute_failures / len(compute_rows) if compute_rows else None
    parent = 0.0 if global_raw is None else global_raw
    risk = (
        parent
        if compute_raw is None
        else (len(compute_rows) * compute_raw + shrinkage_strength * parent)
        / (len(compute_rows) + shrinkage_strength)
    )
    detail = {
        "global_attempt_count": len(resource_rows),
        "global_failure_count": global_failures,
        "global_raw_risk": global_raw,
        "compute_attempt_count": len(compute_rows),
        "compute_failure_count": compute_failures,
        "compute_raw_risk": compute_raw,
        "risk": min(1.0, max(0.0, float(risk))),
        "stable_sealed_resource_evidence_only": True,
    }
    return float(detail["risk"]), canonical_value(detail)


def _resource_view(
    resource_profile: Any,
    *,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], ResourceAdmissionStateV2, float]:
    profile = _as_mapping(resource_profile, name="resource_profile")
    _walk_no_leak(profile, name="resource_profile")
    profile_digest_value = profile.get("profile_digest")
    profile_digest = _require_digest(profile_digest_value, name="resource_profile.profile_digest")
    candidate_ref = _text(profile.get("candidate_ref"), name="resource_profile.candidate_ref")
    if candidate_ref != identity["resource_candidate_ref"]:
        _fail("resource profile candidate_ref does not match identity-bound resource ref")
    profile_package = profile.get("candidate_package_digest")
    expected_package = identity.get("candidate_package_digest")
    if profile_package is not None:
        profile_package = _require_digest(
            profile_package,
            name="resource_profile.candidate_package_digest",
        )
    if profile_package != expected_package:
        _fail("resource profile candidate_package_digest does not match identity")
    profile_source = _require_digest(
        profile.get("candidate_source_sha256"),
        name="resource_profile.candidate_source_sha256",
    )
    expected_source = identity.get("candidate_source_sha256")
    if expected_source is not None and profile_source != expected_source:
        _fail("resource profile candidate_source_sha256 does not match identity")
    prediction = profile.get("prediction")
    if not isinstance(prediction, Mapping):
        _fail("resource profile requires a prediction mapping")
    prediction_identity = prediction.get("identity")
    if isinstance(prediction_identity, Mapping):
        for key, expected in (
            ("candidate_id", identity["candidate_id"]),
            ("candidate_ref", identity["resource_candidate_ref"]),
            ("candidate_package_digest", expected_package),
            ("candidate_source_sha256", expected_source),
            ("semantic_digest", identity["semantic_digest"]),
        ):
            if key in prediction_identity and prediction_identity[key] != expected:
                _fail(f"resource prediction identity mismatch at {key}")

    gpu_values = []
    gpu_paths = []
    for field_name in ("predicted_gpu_seconds", "predicted_gpu_worker_seconds"):
        if field_name in profile:
            gpu_values.append(profile[field_name])
            gpu_paths.append(field_name)
        if field_name in prediction:
            gpu_values.append(prediction[field_name])
            gpu_paths.append(f"prediction.{field_name}")
    if not gpu_values:
        _fail(
            "resource profile lacks an explicit identity-bound "
            "predicted_gpu_seconds or predicted_gpu_worker_seconds; "
            "estimated_total_wall_time_seconds is not substituted"
        )
    gpu_seconds = _finite(
        gpu_values[0],
        name="resource_profile.predicted_gpu_seconds_or_worker_seconds",
        low=0.0,
    )
    if gpu_seconds <= 0.0:
        _fail("resource_profile GPU-worker prediction must be positive")
    if len(gpu_values) > 1:
        other_gpu = _finite(
            gpu_values[1],
            name="resource_profile.predicted_gpu_seconds_or_worker_seconds",
            low=0.0,
        )
        if not math.isclose(gpu_seconds, other_gpu, rel_tol=0.0, abs_tol=1e-12):
            _fail("resource profile has inconsistent GPU-worker prediction fields")

    nested_probability = _finite(
        prediction.get("completion_probability"),
        name="resource_profile.prediction.completion_probability",
        low=0.0,
        high=1.0,
    )
    top_probability = _finite(
        profile.get("completion_probability"),
        name="resource_profile.completion_probability",
        low=0.0,
        high=1.0,
    )
    if not math.isclose(nested_probability, top_probability, rel_tol=0.0, abs_tol=1e-9):
        _fail("resource profile completion_probability fields disagree")
    interval = prediction.get("prediction_interval_seconds")
    profile_interval = profile.get("prediction_interval_seconds")
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        _fail("resource profile prediction interval is missing or malformed")
    lower = _finite(interval[0], name="resource_profile.prediction_interval_seconds[0]", low=0.0)
    upper = _finite(interval[1], name="resource_profile.prediction_interval_seconds[1]", low=0.0)
    if lower > upper:
        _fail("resource profile prediction interval is inverted")
    if profile_interval is not None:
        if not isinstance(profile_interval, (tuple, list)) or len(profile_interval) != 2:
            _fail("resource profile top-level prediction interval is malformed")
        profile_lower = _finite(profile_interval[0], name="resource_profile.prediction_interval_seconds[0]", low=0.0)
        profile_upper = _finite(profile_interval[1], name="resource_profile.prediction_interval_seconds[1]", low=0.0)
        if (profile_lower, profile_upper) != (lower, upper):
            _fail("resource profile prediction interval fields disagree")
    point_seconds = _finite(
        prediction.get("estimated_total_wall_time_seconds"),
        name="resource_profile.prediction.estimated_total_wall_time_seconds",
        low=0.0,
    )
    if point_seconds <= 0.0 or not lower <= point_seconds <= upper:
        _fail("resource profile estimated wall time is outside its interval")
    budget_seconds = _finite(
        profile.get("full_run_budget_after_probes_seconds"),
        name="resource_profile.full_run_budget_after_probes_seconds",
        low=0.0,
    )
    if budget_seconds <= 0.0:
        _fail("resource profile prediction interval exceeds its finite budget")
    if upper > budget_seconds:
        # The raw min/max interval is retained as uncertainty evidence.  A
        # worker-ceiling overrun is admissible only when the prediction itself
        # carries the same explicit hard-ceiling semantics used by the
        # resource scheduler; wall-time interval bounds are not silently
        # clamped to the budget.
        if prediction.get("prediction_interval_exceeds_worker_ceiling") is not True:
            _fail("resource profile prediction interval exceeds its finite budget")
        worker_ceiling_seconds = _finite(
            prediction.get("worker_ceiling_seconds"),
            name="resource_profile.prediction.worker_ceiling_seconds",
            low=0.0,
        )
        native_budget_required_seconds = _finite(
            prediction.get("execution_budget_required_seconds",
                           prediction.get("native_early_stop_budget_required_seconds")),
            name="resource_profile.prediction.execution_budget_required_seconds",
            low=0.0,
        )
        effective_execution_limit = min(worker_ceiling_seconds, budget_seconds)
        if (
            native_budget_required_seconds < point_seconds
            or native_budget_required_seconds > effective_execution_limit
        ):
            _fail("resource profile prediction interval exceeds its finite budget")
    peak_memory = _finite(
        prediction.get("peak_memory_prediction_mib"),
        name="resource_profile.prediction.peak_memory_prediction_mib",
        low=0.0,
    )
    memory_limit = profile.get("gpu_memory_total_mib", prediction.get("gpu_memory_total_mib"))
    if memory_limit is not None:
        memory_limit = _finite(memory_limit, name="resource_profile.gpu_memory_total_mib", low=0.0)
        if memory_limit <= 0.0 or peak_memory >= 0.95 * memory_limit:
            _fail("resource profile peak memory exceeds its declared finite limit")
    status = _text(profile.get("status"), name="resource_profile.status")
    if status not in {"RESOURCE_ADMITTED", "RESOURCE_DEFERRED", "RESOURCE_INFEASIBLE", "RESOURCE_PROBE_FAILED"}:
        _fail(f"resource profile status is outside the admission contract: {status}")
    for field_name, expected in (
        ("outcome_fields_consumed", []),
        ("effect_fields_consumed", []),
        ("held_out_reads", 0),
    ):
        if field_name in profile and profile[field_name] != expected:
            _fail(f"resource profile violates {field_name}={expected!r}")
    if status == "RESOURCE_ADMITTED":
        process = profile.get("probe_process")
        if not isinstance(process, Mapping) or not (
            process.get("process_isolated") is True
            and process.get("status") == "RESULT"
            and process.get("exit_code") == 0
        ):
            _fail("RESOURCE_ADMITTED profile lacks successful disposable probe evidence")
        schedule = profile.get("schedule")
        if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
            _fail("RESOURCE_ADMITTED profile lacks a schedule")
        if not any(isinstance(row, Mapping) and row.get("arm") == candidate_ref for row in schedule):
            _fail("RESOURCE_ADMITTED profile schedule is not bound to candidate_ref")
        if nested_probability <= 0.0:
            _fail("RESOURCE_ADMITTED profile has zero completion probability")
        admission_state = ResourceAdmissionStateV2.RESOURCE_ADMITTED
    else:
        admission_state = ResourceAdmissionStateV2.QUARANTINED
    return (
        {
            "candidate_ref": candidate_ref,
            "candidate_package_digest": profile_package,
            "candidate_source_sha256": profile_source,
            "profile_digest": profile_digest,
            "status": status,
            "completion_probability": nested_probability,
            "predicted_gpu_seconds": gpu_seconds,
            "predicted_gpu_seconds_paths": tuple(gpu_paths),
            "prediction_interval_seconds": (lower, upper),
            "estimated_total_wall_time_seconds": point_seconds,
            "full_run_budget_after_probes_seconds": budget_seconds,
            "peak_memory_prediction_mib": peak_memory,
            "admission_evidence": "SCHEDULE_AND_DISPOSABLE_PROBE"
            if status == "RESOURCE_ADMITTED"
            else "EXPLICIT_DEFERRED_OR_INFEASIBLE_STATUS",
        },
        admission_state,
        gpu_seconds,
    )


@dataclass(frozen=True, slots=True)
class PortfolioCandidateProfileV2:
    """A candidate plus auditable source/evidence digests for the handoff."""

    candidate: PortfolioCandidateV2
    source_digests: Mapping[str, str]
    evidence_digest: str
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, PortfolioCandidateV2):
            raise PortfolioProfileError("candidate must be PortfolioCandidateV2")
        normalized_sources = {}
        for key, value in dict(self.source_digests).items():
            normalized_sources[_text(str(key), name="source_digest.key")] = _require_digest(
                value,
                name=f"source_digests.{key}",
            )
        object.__setattr__(self, "source_digests", canonical_value(normalized_sources))
        object.__setattr__(
            self,
            "evidence_digest",
            _require_digest(self.evidence_digest, name="evidence_digest"),
        )
        if not isinstance(self.evidence, Mapping):
            raise PortfolioProfileError("evidence must be a mapping")
        object.__setattr__(self, "evidence", canonical_value(dict(self.evidence)))

    @property
    def profile_digest(self) -> str:
        return self.evidence_digest

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": PROFILE_SCHEMA,
                "candidate": self.candidate.to_dict(),
                "source_digests": self.source_digests,
                "evidence_digest": self.evidence_digest,
                "evidence": self.evidence,
            }
        )


def build_portfolio_candidate_profile_v2(
    *,
    current_context: Any,
    candidate_identity: Mapping[str, Any],
    binding: Any,
    resolution: Any,
    task_queue: Any,
    exploration_mandate: Any = None,
    prior_attempts: Any = (),
    calibration_prior: Any = None,
    family_history: Any = (),
    parent_history: Any = (),
    frontier_history: Any = (),
    resource_profile: Any,
    temporal_cutoff_round: int | None = None,
    shrinkage_strength: float = DEFAULT_SHRINKAGE_STRENGTH,
) -> PortfolioCandidateProfileV2:
    """Build one outcome-blind PortfolioCandidateV2 and evidence ledger.

    ``calibration_prior`` is required when no sealed prior attempt is
    available.  It must be a frozen mapping with positive ``alpha`` and
    ``beta`` plus a SHA-256 ``source_digest``.  A non-empty sealed global
    history can be used directly when the prior is absent.

    ``resource_profile`` must expose a positive, finite
    ``predicted_gpu_seconds`` or the explicitly equivalent
    ``predicted_gpu_worker_seconds`` at the profile or nested prediction level.
    The existing fixed-batch profile's arbitrary wall-time field is
    intentionally not an accepted alias.

    If no exact pending/active task exists, ``exploration_mandate`` must carry
    ``kind=EXPLORATION``, ``frozen=True``, a SHA-256 ``source_digest``, a
    non-future ``created_round``, positive finite ``priority``, and a
    ``producer_role`` exactly matching ``binding.proposal.producer_role``.
    """

    if not isinstance(candidate_identity, Mapping):
        raise PortfolioProfileError("candidate_identity must be a mapping")
    shrinkage_strength = _finite(
        shrinkage_strength,
        name="shrinkage_strength",
        low=0.0,
    )
    if shrinkage_strength <= 0.0:
        _fail("shrinkage_strength must be positive")
    context_projection, current_round = _context_projection(current_context)
    cutoff = current_round if temporal_cutoff_round is None else _integer(
        temporal_cutoff_round,
        name="temporal_cutoff_round",
        low=1,
    )
    if cutoff > current_round:
        _fail("temporal_cutoff_round cannot be after current_context.round_index")
    binding_map = _as_mapping(binding, name="binding")
    identity = _candidate_identity(candidate_identity, binding=binding_map)
    binding_map, resolution_map, binding_digest, resolution_digest = _binding_and_resolution(
        binding=binding,
        resolution=resolution,
        identity=identity | {
            key: candidate_identity[key]
            for key in ("binding_digest", "resolution_digest")
            if key in candidate_identity
        },
    )
    (
        task,
        queue_digest,
        age_rounds,
        task_priority,
        selection_kind,
        selection_evidence,
    ) = _task_from_queue(
        task_queue,
        identity=identity,
        binding=binding_map,
        current_round=current_round,
        exploration_mandate=exploration_mandate,
    )
    attempts, families, parents, frontiers = _normalize_histories(
        prior_attempts=prior_attempts,
        family_history=family_history,
        parent_history=parent_history,
        frontier_history=frontier_history,
        cutoff=cutoff,
        identity=identity,
    )
    normalized_calibration_prior = _calibration_prior_view(calibration_prior)
    resource, resource_state, predicted_gpu_seconds = _resource_view(
        resource_profile,
        identity=identity,
    )
    resource_digest = resource["profile_digest"]

    probability, exploration_uncertainty, calibration = _calibrate(
        attempts,
        identity=identity,
        shrinkage_strength=shrinkage_strength,
        calibration_prior=normalized_calibration_prior,
    )
    family_delta, family_evidence = _shrink_value(
        families,
        key="stable_delta",
        target=identity["family_id"],
        target_key="family_id",
        low=-1.0,
        high=1.0,
        shrinkage_strength=shrinkage_strength,
    )
    parent_state, lineage_risk, parent_delta, parent_evidence = _parent_view(
        parents,
        identity=identity,
        shrinkage_strength=shrinkage_strength,
    )
    frontier_gain, frontier_evidence = _frontier_prior(
        frontiers,
        identity=identity,
        shrinkage_strength=shrinkage_strength,
    )
    correlated_risk, correlated_evidence = _compute_risk(
        attempts,
        identity=identity,
        shrinkage_strength=shrinkage_strength,
    )
    repeat_count = sum(
        row.get("candidate_id") == identity["candidate_id"]
        and row.get("semantic_digest") == identity["semantic_digest"]
        for row in attempts
    )
    information_value = min(1.0, max(0.0, task_priority * exploration_uncertainty))
    candidate = PortfolioCandidateV2(
        candidate_id=identity["candidate_id"],
        semantic_digest=identity["semantic_digest"],
        family_id=identity["family_id"],
        parent_id=identity["parent_id"],
        valid_seal_probability=probability,
        family_delta=family_delta,
        parent_delta=parent_delta,
        information_value=information_value,
        predicted_gpu_seconds=predicted_gpu_seconds,
        age_rounds=age_rounds,
        repeat_count=repeat_count,
        lineage_risk=lineage_risk,
        compute_pattern=identity["compute_pattern"],
        resource_admission_state=resource_state,
        parent_state=parent_state,
        task_priority=task_priority,
        frontier_gain=frontier_gain,
        correlated_compute_risk=correlated_risk,
        parent_rebound=False,
    )
    normalized_identity = canonical_value(
        {
            "candidate_id": identity["candidate_id"],
            "semantic_digest": identity["semantic_digest"],
            "family_id": identity["family_id"],
            "parent_id": identity["parent_id"],
            "compute_pattern": identity["compute_pattern"],
            "resource_candidate_ref": identity["resource_candidate_ref"],
            "candidate_package_digest": identity["candidate_package_digest"],
            "candidate_source_sha256": identity["candidate_source_sha256"],
        }
    )
    source_digests = {
        "context": sha256_digest(context_projection),
        "candidate_identity": sha256_digest(normalized_identity),
        "binding": binding_digest,
        "resolution": resolution_digest,
        "task_queue": queue_digest,
        "prior_attempts": sha256_digest(attempts),
        "family_history": sha256_digest(families),
        "parent_history": sha256_digest(parents),
        "frontier_history": sha256_digest(frontiers),
        "resource_profile": resource_digest,
    }
    if selection_kind == "TASK":
        source_digests["task_record"] = selection_evidence["task_record_digest"]
    else:
        source_digests["exploration_mandate"] = selection_evidence[
            "mandate_digest"
        ]
    if normalized_calibration_prior is not None:
        source_digests["calibration_prior"] = normalized_calibration_prior[
            "source_digest"
        ]
    task_evidence = None
    mandate_evidence = None
    if selection_kind == "TASK":
        task_evidence = {
            "task_id": task["task_id"],
            "created_round": task["created_round"],
            "age_rounds": age_rounds,
            "priority": task_priority,
            "repeat_count": repeat_count,
        }
    else:
        mandate_evidence = {
            **selection_evidence["mandate"],
            "mandate_digest": selection_evidence["mandate_digest"],
            "queue_has_other_pending_or_active": selection_evidence[
                "queue_has_other_pending_or_active"
            ],
            "age_rounds": age_rounds,
        }
    evidence = canonical_value(
        {
            "schema": PROFILE_SCHEMA,
            "temporal_cutoff_round": cutoff,
            "candidate_identity_digest": source_digests["candidate_identity"],
            "source_digests": source_digests,
            "selection_kind": selection_kind,
            "calibration": calibration,
            "priors": {
                "family": family_evidence,
                "parent": parent_evidence,
                "frontier": frontier_evidence,
            },
            "task": task_evidence,
            "exploration_mandate": mandate_evidence,
            "lineage": parent_evidence,
            "correlated_compute": correlated_evidence,
            "resource_admission": resource,
            "outcome_fields_consumed": [],
            "effect_fields_consumed": [],
            "held_out_reads": 0,
        }
    )
    evidence_digest = sha256_digest(evidence)
    return PortfolioCandidateProfileV2(
        candidate=candidate,
        source_digests=source_digests,
        evidence_digest=evidence_digest,
        evidence=evidence,
    )


def build_portfolio_candidate_v2(**kwargs: Any) -> PortfolioCandidateV2:
    """Return exactly one complete PortfolioCandidateV2 for D/main."""

    return build_portfolio_candidate_profile_v2(**kwargs).candidate


__all__ = [
    "DEFAULT_SHRINKAGE_STRENGTH",
    "PROFILE_SCHEMA",
    "PortfolioCandidateProfileV2",
    "PortfolioProfileError",
    "build_portfolio_candidate_profile_v2",
    "build_portfolio_candidate_v2",
]
