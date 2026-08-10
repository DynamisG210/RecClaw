"""Outcome-blind Research Line candidate-portfolio controls.

The module is independent of campaign/runtime state.  It contains only
pre-outcome resource admission, lineage controls, portfolio scoring, and
deterministic in-round failover.  The existing StrongStaticRouterV1 remains
the compatibility boundary for legacy compilation and hard gates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)


class PortfolioError(ValueError):
    """Raised when outcome-blind portfolio state is malformed."""


class ResourceAdmissionStateV2(str, Enum):
    SPECULATIVE = "SPECULATIVE"
    QUALIFIED = "QUALIFIED"
    RESOURCE_ADMITTED = "RESOURCE_ADMITTED"
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    DOMINATED = "DOMINATED"
    QUARANTINED = "QUARANTINED"
    RETIRED = "RETIRED"
    CORRELATED_RISK_BLOCKED = "CORRELATED_RISK_BLOCKED"


class ParentValidationStateV2(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    VALIDATED = "VALIDATED"
    UNVERIFIED = "UNVERIFIED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class AttemptFailureScopeV2(str, Enum):
    CANDIDATE_LOCAL = "CANDIDATE_LOCAL"
    LINEAGE_COMPUTE_PATTERN = "LINEAGE_COMPUTE_PATTERN"
    WORKER_TRANSIENT = "WORKER_TRANSIENT"
    SHARED_INFRASTRUCTURE = "SHARED_INFRASTRUCTURE"


class PortfolioEligibilityV2(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    FAILED_ATTEMPT = "FAILED_ATTEMPT"
    NOT_RESOURCE_ADMITTED = "NOT_RESOURCE_ADMITTED"
    UNVERIFIED_PARENT = "UNVERIFIED_PARENT"
    FAILED_PARENT = "FAILED_PARENT"
    STALE = "STALE"
    DOMINATED = "DOMINATED"
    CORRELATED_COMPUTE_PATTERN = "CORRELATED_COMPUTE_PATTERN"
    LINEAGE_RISK = "LINEAGE_RISK"


_ADMITTED_STATES = frozenset(
    {ResourceAdmissionStateV2.RESOURCE_ADMITTED, ResourceAdmissionStateV2.ACTIVE}
)
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
        "observed_delta",
        "current_outcome",
        "current_metric",
        "current_result",
        "current_comparator_delta",
    }
)


def _text(value: Any, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise PortfolioError(f"{name} must be non-empty")
    return result


def _number(value: Any, *, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortfolioError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or not low <= result <= high:
        raise PortfolioError(f"{name} must be finite and in [{low}, {high}]")
    return result


def _positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortfolioError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise PortfolioError(f"{name} must be finite and positive")
    return result


def _count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PortfolioError(f"{name} must be a non-negative integer")
    return int(value)


def _enum(value: Any, enum_type: type[Enum], *, name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise PortfolioError(f"{name} is outside its closed domain") from exc


def _reject_outcome_fields(value: Mapping[str, Any], *, name: str) -> None:
    for key in value:
        normalized = str(key).strip().lower()
        if normalized in _OUTCOME_KEYS or normalized.startswith("current_outcome"):
            raise PortfolioError(f"{name} contains outcome-bearing field {key!r}")


@dataclass(frozen=True, slots=True)
class PortfolioCandidateV2:
    """Separable candidate heads and pre-outcome portfolio controls.

    family_delta and parent_delta are stable priors from earlier sealed
    evidence; they are never the current attempt's result.  GPU cost is a
    predicted formal GPU duration, not Provider token cost.
    """

    candidate_id: str
    semantic_digest: str
    family_id: str
    parent_id: str | None
    valid_seal_probability: float
    family_delta: float
    parent_delta: float
    information_value: float
    predicted_gpu_seconds: float
    age_rounds: int
    repeat_count: int
    lineage_risk: float
    compute_pattern: str
    resource_admission_state: ResourceAdmissionStateV2 = (
        ResourceAdmissionStateV2.RESOURCE_ADMITTED
    )
    parent_state: ParentValidationStateV2 = ParentValidationStateV2.INDEPENDENT
    task_priority: float = 1.0
    frontier_gain: float = 0.0
    correlated_compute_risk: float = 0.0
    dominated_by: str | None = None
    parent_rebound: bool = False

    def __post_init__(self) -> None:
        for name in ("candidate_id", "semantic_digest", "family_id", "compute_pattern"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.parent_id is not None:
            object.__setattr__(self, "parent_id", _text(self.parent_id, name="parent_id"))
        for name in ("valid_seal_probability", "information_value", "lineage_risk",
                     "correlated_compute_risk"):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name=name, low=0.0, high=1.0),
            )
        for name in ("family_delta", "parent_delta", "frontier_gain"):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name=name, low=-1.0, high=1.0),
            )
        object.__setattr__(
            self,
            "predicted_gpu_seconds",
            _positive(self.predicted_gpu_seconds, name="predicted_gpu_seconds"),
        )
        for name in ("age_rounds", "repeat_count"):
            object.__setattr__(self, name, _count(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "task_priority",
            _positive(self.task_priority, name="task_priority"),
        )
        object.__setattr__(
            self,
            "resource_admission_state",
            _enum(
                self.resource_admission_state,
                ResourceAdmissionStateV2,
                name="resource_admission_state",
            ),
        )
        object.__setattr__(
            self,
            "parent_state",
            _enum(self.parent_state, ParentValidationStateV2, name="parent_state"),
        )
        if self.dominated_by is not None:
            object.__setattr__(
                self, "dominated_by", _text(self.dominated_by, name="dominated_by")
            )

    @property
    def resource_admission(self) -> ResourceAdmissionStateV2:
        """Short alias used by the module-level report terminology."""

        return self.resource_admission_state

    @property
    def effective_lineage_risk(self) -> float:
        if self.parent_rebound or self.parent_state in {
            ParentValidationStateV2.INDEPENDENT,
            ParentValidationStateV2.VALIDATED,
        }:
            return round(self.lineage_risk, 12)
        increments = {
            ParentValidationStateV2.UNVERIFIED: 0.25,
            ParentValidationStateV2.UNKNOWN: 0.35,
            ParentValidationStateV2.FAILED: 0.55,
        }
        return round(min(1.0, self.lineage_risk + increments[self.parent_state]), 12)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PortfolioCandidateV2":
        if not isinstance(value, Mapping):
            raise PortfolioError("portfolio candidate must be a mapping")
        _reject_outcome_fields(value, name="portfolio candidate")
        gpu_seconds = value.get("predicted_gpu_seconds", value.get("gpu_seconds"))
        if gpu_seconds is None:
            raise PortfolioError("portfolio candidate requires predicted_gpu_seconds")
        return cls(
            candidate_id=value.get("candidate_id"),
            semantic_digest=value.get("semantic_digest"),
            family_id=value.get("family_id"),
            parent_id=value.get("parent_id"),
            valid_seal_probability=value.get("valid_seal_probability"),
            family_delta=value.get("family_delta", 0.0),
            parent_delta=value.get("parent_delta", 0.0),
            information_value=value.get("information_value", value.get("info_value", 0.0)),
            predicted_gpu_seconds=gpu_seconds,
            age_rounds=value.get("age_rounds", value.get("age", 0)),
            repeat_count=value.get("repeat_count", value.get("repeats", 0)),
            lineage_risk=value.get("lineage_risk", 0.0),
            compute_pattern=value.get("compute_pattern", "unknown"),
            resource_admission_state=value.get(
                "resource_admission_state",
                value.get("resource_state", ResourceAdmissionStateV2.RESOURCE_ADMITTED),
            ),
            parent_state=value.get("parent_state", ParentValidationStateV2.INDEPENDENT),
            task_priority=value.get("task_priority", 1.0),
            frontier_gain=value.get("frontier_gain", 0.0),
            correlated_compute_risk=value.get("correlated_compute_risk", 0.0),
            dominated_by=value.get("dominated_by"),
            parent_rebound=bool(value.get("parent_rebound", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class PortfolioAttemptFailureV2:
    """Failure marker for a candidate-local in-round retry decision."""

    candidate_id: str
    scope: AttemptFailureScopeV2 = AttemptFailureScopeV2.CANDIDATE_LOCAL
    reason: str = "engineering_failure"
    compute_pattern: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, name="candidate_id"))
        object.__setattr__(
            self, "scope", _enum(self.scope, AttemptFailureScopeV2, name="scope")
        )
        object.__setattr__(self, "reason", _text(self.reason, name="reason"))
        if self.compute_pattern is not None:
            object.__setattr__(
                self,
                "compute_pattern",
                _text(self.compute_pattern, name="compute_pattern"),
            )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def _count_map(
    value: Mapping[str, Any] | Sequence[tuple[str, int]],
    *,
    name: str,
) -> tuple[tuple[str, int], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    result: dict[str, int] = {}
    for key, count in items:
        result[_text(key, name=f"{name}.key")] = _count(count, name=f"{name}[{key}]")
    return tuple(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class PortfolioControlStateV2:
    """Temporal portfolio controls and the first routing-window size."""

    attempt_failures: tuple[PortfolioAttemptFailureV2, ...] = ()
    failed_candidate_ids: tuple[str, ...] = ()
    failed_parent_ids: tuple[str, ...] = ()
    unverified_parent_ids: tuple[str, ...] = ()
    dominated_candidate_ids: tuple[str, ...] = ()
    repeat_counts: tuple[tuple[str, int], ...] = ()
    compute_pattern_failures: tuple[tuple[str, int], ...] = ()
    compute_pattern_attempts: tuple[tuple[str, int], ...] = ()
    stale_after_rounds: int = 7
    correlated_failure_minimum: int = 2
    correlated_failure_rate: float = 0.75
    lineage_risk_ceiling: float = 0.85
    first_window_size: int = 4
    lambda_information: float = 1.0
    repeat_penalty: float = 0.25
    shared_infrastructure_hold: bool = False

    def __post_init__(self) -> None:
        failures = tuple(self.attempt_failures)
        if not all(isinstance(item, PortfolioAttemptFailureV2) for item in failures):
            raise PortfolioError("attempt_failures must use PortfolioAttemptFailureV2")
        object.__setattr__(self, "attempt_failures", failures)
        for name in (
            "failed_candidate_ids",
            "failed_parent_ids",
            "unverified_parent_ids",
            "dominated_candidate_ids",
        ):
            object.__setattr__(
                self,
                name,
                tuple(sorted({_text(item, name=name) for item in getattr(self, name)})),
            )
        for name in (
            "repeat_counts",
            "compute_pattern_failures",
            "compute_pattern_attempts",
        ):
            object.__setattr__(
                self, name, _count_map(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "stale_after_rounds", _count(self.stale_after_rounds, name="stale_after_rounds")
        )
        object.__setattr__(
            self,
            "correlated_failure_minimum",
            _count(self.correlated_failure_minimum, name="correlated_failure_minimum"),
        )
        if self.stale_after_rounds <= 0 or self.correlated_failure_minimum <= 0:
            raise PortfolioError("stale and correlated failure thresholds must be positive")
        for name in ("correlated_failure_rate", "lineage_risk_ceiling", "repeat_penalty"):
            object.__setattr__(
                self, name, _number(getattr(self, name), name=name, low=0.0, high=1.0)
            )
        if (
            isinstance(self.first_window_size, bool)
            or not isinstance(self.first_window_size, int)
            or self.first_window_size <= 0
        ):
            raise PortfolioError("first_window_size must be a positive integer")
        object.__setattr__(
            self,
            "lambda_information",
            _number(self.lambda_information, name="lambda_information", low=0.0, high=10.0),
        )

    @property
    def failed_attempt_candidate_ids(self) -> frozenset[str]:
        return frozenset(
            {
                *self.failed_candidate_ids,
                *(
                    item.candidate_id
                    for item in self.attempt_failures
                    if item.scope
                    in {
                        AttemptFailureScopeV2.CANDIDATE_LOCAL,
                        AttemptFailureScopeV2.LINEAGE_COMPUTE_PATTERN,
                        AttemptFailureScopeV2.WORKER_TRANSIENT,
                    }
                ),
            }
        )

    def repeat_count_for(self, candidate: PortfolioCandidateV2) -> int:
        counts = dict(self.repeat_counts)
        return max(
            candidate.repeat_count,
            counts.get(candidate.candidate_id, 0),
            counts.get(candidate.semantic_digest, 0),
        )

    def pattern_is_blocked(self, pattern: str) -> bool:
        failures = dict(self.compute_pattern_failures).get(pattern, 0)
        attempts = dict(self.compute_pattern_attempts).get(pattern, failures)
        return (
            failures >= self.correlated_failure_minimum
            and (attempts <= 0 or failures / attempts >= self.correlated_failure_rate)
        )

    def with_attempt_failure(
        self, failure: PortfolioAttemptFailureV2 | str
    ) -> "PortfolioControlStateV2":
        if isinstance(failure, str):
            failure = PortfolioAttemptFailureV2(candidate_id=failure)
        attempts = dict(self.compute_pattern_attempts)
        failures = dict(self.compute_pattern_failures)
        if failure.compute_pattern is not None:
            pattern = failure.compute_pattern
            attempts[pattern] = attempts.get(pattern, 0) + 1
            if failure.scope is not AttemptFailureScopeV2.SHARED_INFRASTRUCTURE:
                failures[pattern] = failures.get(pattern, 0) + 1
        return replace(
            self,
            attempt_failures=(*self.attempt_failures, failure),
            compute_pattern_failures=tuple(failures.items()),
            compute_pattern_attempts=tuple(attempts.items()),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class PortfolioRankedCandidateV2:
    candidate: PortfolioCandidateV2
    score: float
    eligible: bool
    eligibility: PortfolioEligibilityV2
    effective_valid_seal_probability: float
    repeat_count: int
    repeat_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class PortfolioRankingV2:
    """Full-pool order plus the bounded first routing window."""

    records: tuple[PortfolioRankedCandidateV2, ...]
    ranked_candidate_ids: tuple[str, ...]
    first_window_candidate_ids: tuple[str, ...]
    selected_candidate_id: str | None
    policy_digest: str

    @property
    def eligible_records(self) -> tuple[PortfolioRankedCandidateV2, ...]:
        by_id = {item.candidate.candidate_id: item for item in self.records}
        return tuple(by_id[item] for item in self.ranked_candidate_ids)

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def _eligibility(
    candidate: PortfolioCandidateV2,
    *,
    control: PortfolioControlStateV2,
) -> PortfolioEligibilityV2:
    if candidate.candidate_id in control.failed_attempt_candidate_ids:
        return PortfolioEligibilityV2.FAILED_ATTEMPT
    if candidate.candidate_id in control.dominated_candidate_ids or candidate.dominated_by:
        return PortfolioEligibilityV2.DOMINATED
    if candidate.resource_admission_state is ResourceAdmissionStateV2.DOMINATED:
        return PortfolioEligibilityV2.DOMINATED
    if candidate.resource_admission_state is ResourceAdmissionStateV2.STALE:
        return PortfolioEligibilityV2.STALE
    if (
        candidate.resource_admission_state
        is ResourceAdmissionStateV2.CORRELATED_RISK_BLOCKED
    ):
        return PortfolioEligibilityV2.CORRELATED_COMPUTE_PATTERN
    if candidate.age_rounds >= control.stale_after_rounds:
        return PortfolioEligibilityV2.STALE
    if candidate.resource_admission_state not in _ADMITTED_STATES:
        return PortfolioEligibilityV2.NOT_RESOURCE_ADMITTED
    if (
        candidate.parent_state is ParentValidationStateV2.FAILED
        or candidate.parent_id in control.failed_parent_ids
    ) and not candidate.parent_rebound:
        return PortfolioEligibilityV2.FAILED_PARENT
    if (
        candidate.parent_state
        in {ParentValidationStateV2.UNVERIFIED, ParentValidationStateV2.UNKNOWN}
        or candidate.parent_id in control.unverified_parent_ids
    ) and not candidate.parent_rebound:
        return PortfolioEligibilityV2.UNVERIFIED_PARENT
    if control.pattern_is_blocked(candidate.compute_pattern):
        return PortfolioEligibilityV2.CORRELATED_COMPUTE_PATTERN
    if candidate.effective_lineage_risk > control.lineage_risk_ceiling:
        return PortfolioEligibilityV2.LINEAGE_RISK
    return PortfolioEligibilityV2.ELIGIBLE


def _score(
    candidate: PortfolioCandidateV2,
    *,
    control: PortfolioControlStateV2,
    repeat_count: int,
) -> tuple[float, float, float]:
    effective_probability = round(
        candidate.valid_seal_probability
        * (1.0 - candidate.effective_lineage_risk)
        * (1.0 - candidate.correlated_compute_risk),
        12,
    )
    research_value = candidate.task_priority * (
        candidate.frontier_gain
        + control.lambda_information * candidate.information_value
        + 0.25 * candidate.family_delta
        + 0.25 * candidate.parent_delta
    )
    age_bonus = 0.02 * min(
        1.0, candidate.age_rounds / max(1, control.stale_after_rounds)
    )
    repeat_penalty = control.repeat_penalty * min(4, repeat_count)
    score = (
        effective_probability * research_value / candidate.predicted_gpu_seconds
        + age_bonus
        - repeat_penalty
    )
    return round(score, 12), effective_probability, round(repeat_penalty, 12)


def rank_candidate_portfolio_v2(
    candidates: Sequence[PortfolioCandidateV2],
    *,
    control: PortfolioControlStateV2 | None = None,
    first_window_size: int | None = None,
) -> PortfolioRankingV2:
    """Rank the full pool; first_window_candidate_ids is only the first slate."""

    state = control or PortfolioControlStateV2()
    normalized = tuple(candidates)
    if not normalized:
        raise PortfolioError("candidate portfolio cannot be empty")
    if not all(isinstance(item, PortfolioCandidateV2) for item in normalized):
        raise PortfolioError("candidates must use PortfolioCandidateV2")
    ids = [item.candidate_id for item in normalized]
    if len(set(ids)) != len(ids):
        raise PortfolioError("candidate portfolio IDs must be unique")
    window = state.first_window_size if first_window_size is None else first_window_size
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise PortfolioError("first_window_size must be a positive integer")
    records: list[PortfolioRankedCandidateV2] = []
    for candidate in normalized:
        status = _eligibility(candidate, control=state)
        repeat_count = state.repeat_count_for(candidate)
        score, probability, repeat_penalty = _score(
            candidate, control=state, repeat_count=repeat_count
        )
        records.append(
            PortfolioRankedCandidateV2(
                candidate=candidate,
                score=score,
                eligible=status is PortfolioEligibilityV2.ELIGIBLE,
                eligibility=status,
                effective_valid_seal_probability=probability,
                repeat_count=repeat_count,
                repeat_penalty=repeat_penalty,
            )
        )
    eligible = sorted(
        (item for item in records if item.eligible),
        key=lambda item: (
            -item.score,
            -item.effective_valid_seal_probability,
            item.candidate.predicted_gpu_seconds,
            item.candidate.candidate_id,
        ),
    )
    ranked_ids = tuple(item.candidate.candidate_id for item in eligible)
    first_window = ranked_ids[:window]
    policy_digest = sha256_digest(
        {
            "schema": "recclaw.research-line.portfolio.v2",
            "control": state.to_dict(),
            "candidates": tuple(item.to_dict() for item in normalized),
            "ranked_candidate_ids": ranked_ids,
            "first_window_size": window,
        }
    )
    return PortfolioRankingV2(
        records=tuple(records),
        ranked_candidate_ids=ranked_ids,
        first_window_candidate_ids=first_window,
        selected_candidate_id=first_window[0] if first_window else None,
        policy_digest=policy_digest,
    )


def rerank_after_attempt_failure_v2(
    candidates: Sequence[PortfolioCandidateV2],
    failure: PortfolioAttemptFailureV2 | str,
    *,
    control: PortfolioControlStateV2 | None = None,
    first_window_size: int | None = None,
) -> PortfolioRankingV2:
    """Exclude the failed exact candidate and re-rank the remaining full pool."""

    state = control or PortfolioControlStateV2()
    if isinstance(failure, str):
        failure = PortfolioAttemptFailureV2(candidate_id=failure)
    return rank_candidate_portfolio_v2(
        candidates,
        control=state.with_attempt_failure(failure),
        first_window_size=first_window_size,
    )


@dataclass(frozen=True, slots=True)
class PortfolioReserveAssessmentV2:
    eligible_candidate_ids: tuple[str, ...]
    correlated_compute_groups: tuple[tuple[str, tuple[str, ...], float], ...]
    probability_at_least_one_valid_seal: float
    readiness_target: float
    ready: bool
    held_by_shared_infrastructure: bool
    policy_digest: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def assess_resource_reserve_v2(
    candidates: Sequence[PortfolioCandidateV2],
    *,
    control: PortfolioControlStateV2 | None = None,
    readiness_target: float = 0.80,
) -> PortfolioReserveAssessmentV2:
    """Use max-per-pattern probabilities instead of false independent products."""

    target = _number(readiness_target, name="readiness_target", low=0.0, high=1.0)
    ranking = rank_candidate_portfolio_v2(candidates, control=control)
    groups: dict[str, list[PortfolioRankedCandidateV2]] = {}
    for record in ranking.eligible_records:
        groups.setdefault(record.candidate.compute_pattern, []).append(record)
    rows: list[tuple[str, tuple[str, ...], float]] = []
    complement = 1.0
    for pattern in sorted(groups):
        members = groups[pattern]
        probability = max(item.effective_valid_seal_probability for item in members)
        ids = tuple(item.candidate.candidate_id for item in members)
        rows.append((pattern, ids, round(probability, 12)))
        complement *= 1.0 - probability
    probability = round(1.0 - complement, 12)
    held = bool(control is not None and control.shared_infrastructure_hold)
    return PortfolioReserveAssessmentV2(
        eligible_candidate_ids=ranking.ranked_candidate_ids,
        correlated_compute_groups=tuple(rows),
        probability_at_least_one_valid_seal=probability,
        readiness_target=target,
        ready=probability >= target and not held,
        held_by_shared_infrastructure=held,
        policy_digest=sha256_digest(
            {
                "ranking_policy_digest": ranking.policy_digest,
                "readiness_target": target,
                "correlated_compute_groups": tuple(rows),
            }
        ),
    )


__all__ = [
    "AttemptFailureScopeV2",
    "ParentValidationStateV2",
    "PortfolioAttemptFailureV2",
    "PortfolioCandidateV2",
    "PortfolioControlStateV2",
    "PortfolioEligibilityV2",
    "PortfolioError",
    "PortfolioRankedCandidateV2",
    "PortfolioRankingV2",
    "PortfolioReserveAssessmentV2",
    "ResourceAdmissionStateV2",
    "assess_resource_reserve_v2",
    "rank_candidate_portfolio_v2",
    "rerank_after_attempt_failure_v2",
]
