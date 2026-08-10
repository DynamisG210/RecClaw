"""Small checkpointed multi-round boundary for the Research Line runtime.

This module owns only arm-local state and round persistence.  Provider,
qualification, and experiment composition remain injected by the caller; one
round is still executed by :func:`run_research_round`.
"""

from __future__ import annotations

import json
import os
import copyreg
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import pickle
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import COMMON_EVALUATOR
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemorySnapshotV1,
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import CandidateProposalV4
from recclaw_core.helix.scientific_attribution import SearchUtilityEventV2
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    OpenSpecSearchCandidateV1,
    SearchExecutableProfileV1,
    SearchProfileActivationV1,
)

from .interfaces import (
    ResearchContext,
    ResearchTaskQueueV2,
    ResearchTaskStatusV2,
)
from .interpreter import interpret_missing_search_opportunity
from .portfolio import PortfolioCandidateV2
from .profile_source import ResearchProfileSourceV1
from .producers import ResearchProducer, produce_research_specs
from .runtime import (
    CandidateHandoffFactory,
    ExperimentRunner,
    ImplementerGateway,
    InnovationRuntimeInputs,
    MetaResearchInputs,
    PreparedResearchRoundV1,
    RoundAttemptV1,
    ResearchRoundResult,
    activate_promoted_meta_strategy,
    activate_staged_innovation,
    run_research_round,
)


def _reduce_mapping_proxy(value: MappingProxyType) -> tuple[type[dict[Any, Any]], tuple[dict[Any, Any]]]:
    return dict, (dict(value),)


copyreg.pickle(MappingProxyType, _reduce_mapping_proxy)


class CampaignError(RuntimeError):
    """Raised when a persisted campaign cannot be resumed safely."""


PHYSICAL_CONTEXT_SCHEMA = "recclaw.research-line.physical-execution-context.v1"


def _immutable_canonical_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    canonical = canonical_value(dict(value))

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(canonical)


def _physical_identity_fields(
    candidate_run: Mapping[str, Any],
    *,
    context_digest: str,
    context_applied: bool,
) -> dict[str, Any]:
    physical = candidate_run.get("physical_identity")
    physical = physical if isinstance(physical, Mapping) else {}
    experiment_binding = candidate_run.get("experiment_binding")
    experiment_binding = (
        experiment_binding if isinstance(experiment_binding, Mapping) else {}
    )
    evidence = candidate_run.get("gpu_reservation_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}

    def first(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    actual_context_digest = first(physical.get("context_digest"), context_digest)
    if actual_context_digest != context_digest:
        raise CampaignError("physical runner context identity drift")
    return canonical_value(
        {
            "physical_run_id": first(
                physical.get("run_id"),
                candidate_run.get("physical_run_id"),
                experiment_binding.get("run_id"),
                candidate_run.get("run_id"),
            ),
            "physical_seed": first(
                physical.get("seed"),
                candidate_run.get("physical_seed"),
                candidate_run.get("seed"),
                experiment_binding.get("seed"),
            ),
            "physical_context_digest": actual_context_digest,
            "research_context_digest": candidate_run.get(
                "research_context_digest"
            ),
            "cuda_visible_devices": first(
                physical.get("cuda_visible_devices"),
                candidate_run.get("cuda_visible_devices"),
            ),
            "reservation_digest": first(
                physical.get("reservation_digest"),
                candidate_run.get("reservation_digest"),
                evidence.get("reservation_digest"),
            ),
            "reservation_status": first(
                physical.get("reservation_status"),
                candidate_run.get("gpu_reservation_status"),
                candidate_run.get("reservation_status"),
            ),
            "final_worker_ceiling_seconds": first(
                physical.get("final_worker_ceiling_seconds"),
                candidate_run.get("final_worker_ceiling_seconds"),
            ),
            "physical_context_applied": context_applied,
        }
    )


@dataclass(frozen=True, slots=True)
class CampaignState:
    """The minimum arm-local state needed to build the next Research round."""

    campaign_id: str
    next_round_index: int
    context: ResearchContext
    active_profile: SearchExecutableProfileV1
    policy: VersionedResearchPolicyV1
    search_memory_head: SearchMemorySnapshotV1 | None
    carryover_proposals: tuple[CandidateProposalV4, ...]
    carryover_open_candidates: tuple[OpenSpecSearchCandidateV1, ...]
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]]
    candidate_root_by_capability: Mapping[str, str]
    incumbent_observation: Mapping[str, Any]
    frontier: Mapping[str, Any]
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None = None
    last_round_result_digest: str | None = None

    schema = "recclaw.research-line.campaign-state.v1"

    @classmethod
    def initial(
        cls,
        *,
        context: ResearchContext,
        active_profile: SearchExecutableProfileV1,
        policy: VersionedResearchPolicyV1,
        incumbent_observation: Mapping[str, Any],
        carryover_proposals: Sequence[CandidateProposalV4] = (),
        carryover_open_candidates: Sequence[OpenSpecSearchCandidateV1] = (),
        search_memory_head: SearchMemorySnapshotV1 | None = None,
        qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] = (),
        candidate_root_by_capability: Mapping[str, str] = (),
        resource_profile_by_capability: Mapping[str, Mapping[str, Any]] = (),
        frontier: Mapping[str, Any] | None = None,
    ) -> "CampaignState":
        return cls(
            campaign_id=context.campaign_id,
            next_round_index=context.round_index,
            context=context,
            active_profile=active_profile,
            policy=policy,
            search_memory_head=search_memory_head,
            carryover_proposals=tuple(carryover_proposals),
            carryover_open_candidates=tuple(carryover_open_candidates),
            qualified_execution_by_capability=qualified_execution_by_capability,
            candidate_root_by_capability=candidate_root_by_capability,
            resource_profile_by_capability=resource_profile_by_capability,
            incumbent_observation=incumbent_observation,
            frontier=context.frontier if frontier is None else frontier,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise CampaignError("campaign_id must be non-empty")
        if isinstance(self.next_round_index, bool) or self.next_round_index < 1:
            raise CampaignError("next_round_index must be positive")
        if not isinstance(self.context, ResearchContext):
            raise CampaignError("context must be ResearchContext")
        if not isinstance(self.active_profile, SearchExecutableProfileV1):
            raise CampaignError("active_profile must be SearchExecutableProfileV1")
        if not isinstance(self.policy, VersionedResearchPolicyV1):
            raise CampaignError("policy must be VersionedResearchPolicyV1")
        if self.context.round_index != self.next_round_index:
            raise CampaignError("context and next_round_index differ")
        if (
            self.context.campaign_id,
            self.context.active_profile_ref,
            self.context.active_profile_digest,
        ) != (
            self.campaign_id,
            self.active_profile.profile_ref,
            self.active_profile.profile_digest,
        ):
            raise CampaignError("campaign state profile/context identity drift")
        if canonical_value(self.context.policy) != canonical_value(self.policy.to_dict()):
            raise CampaignError("campaign state policy is not bound to Context")
        if (
            self.context.protocol_ref != self.active_profile.protocol_ref
            or self.context.protocol_digest != self.active_profile.protocol_digest
        ):
            raise CampaignError("campaign state protocol identity drift")
        if self.search_memory_head is not None and not isinstance(
            self.search_memory_head, SearchMemorySnapshotV1
        ):
            raise CampaignError("search_memory_head must be SearchMemorySnapshotV1")
        for item in self.carryover_proposals:
            if not isinstance(item, CandidateProposalV4):
                raise CampaignError("carryover_proposals contain an invalid proposal")
        for item in self.carryover_open_candidates:
            if not isinstance(item, OpenSpecSearchCandidateV1):
                raise CampaignError("carryover_open_candidates contain an invalid candidate")
        object.__setattr__(self, "carryover_proposals", tuple(self.carryover_proposals))
        object.__setattr__(
            self,
            "carryover_open_candidates",
            tuple(self.carryover_open_candidates),
        )
        object.__setattr__(
            self,
            "qualified_execution_by_capability",
            canonical_value(dict(self.qualified_execution_by_capability)),
        )
        object.__setattr__(
            self,
            "candidate_root_by_capability",
            canonical_value(dict(self.candidate_root_by_capability)),
        )
        object.__setattr__(
            self,
            "resource_profile_by_capability",
            canonical_value(dict(self.resource_profile_by_capability or {})),
        )
        object.__setattr__(
            self,
            "incumbent_observation",
            canonical_value(dict(self.incumbent_observation)),
        )
        object.__setattr__(self, "frontier", canonical_value(dict(self.frontier)))

    @property
    def active_executable_profile(self) -> SearchExecutableProfileV1:
        return self.active_profile

    @property
    def round_index(self) -> int:
        return self.next_round_index

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "next_round_index": self.next_round_index,
                "context": self.context.to_dict(),
                "active_profile": self.active_profile.canonical_dict(),
                "policy": self.policy.to_dict(),
                "search_memory_head": (
                    self.search_memory_head.to_dict()
                    if self.search_memory_head is not None
                    else None
                ),
                "carryover_proposals": tuple(
                    item.to_dict() for item in self.carryover_proposals
                ),
                "carryover_open_candidates": tuple(
                    item.to_dict() for item in self.carryover_open_candidates
                ),
                "qualified_execution_by_capability": self.qualified_execution_by_capability,
                "candidate_root_by_capability": self.candidate_root_by_capability,
                "resource_profile_by_capability": getattr(
                    self, "resource_profile_by_capability", {}
                ),
                "incumbent_observation": self.incumbent_observation,
                "frontier": self.frontier,
                "last_round_result_digest": self.last_round_result_digest,
            }
        )


ResearchCampaignState = CampaignState


@dataclass(frozen=True, slots=True)
class CampaignRoundInputs:
    """Round-local inputs supplied by the production composition."""

    producer_bindings: Mapping[str, Any]
    resolver_environment: Mapping[str, Any]
    budget_snapshot: Mapping[str, Any]
    router: StrongStaticRouterV1
    metric_contract_digest: str
    observation_seed: str
    next_discriminative_test: str
    confirmation_seed: str | None = None
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] = ()
    innovation_inputs: InnovationRuntimeInputs | None = None
    meta_research_inputs: MetaResearchInputs | None = None
    attempt_scheduler: bool = False
    max_attempts_per_round: int | None = None
    portfolio_candidates: tuple[PortfolioCandidateV2, ...] = ()
    research_profile_source: ResearchProfileSourceV1 | None = None
    candidate_handoff_factory: CandidateHandoffFactory | None = None
    candidate_root_by_capability: Mapping[str, str] = ()
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] = ()


def _configured_attempt_budget(inputs: CampaignRoundInputs) -> int:
    """Return the frozen configuration cap for this round's attempts."""

    value: Any = inputs.max_attempts_per_round
    if value is None:
        for field_name in (
            "max_attempts_per_round",
            "round_attempt_budget",
            "remaining_attempt_budget",
            "attempt_budget",
        ):
            candidate = inputs.budget_snapshot.get(field_name)
            if candidate is not None:
                value = candidate
                break
    if value is None:
        raise CampaignError(
            "attempt_scheduler=True requires an explicit frozen "
            "max_attempts_per_round or round attempt budget"
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignError("round attempt budget must be a non-negative integer")
    return value


class CampaignRoundInputFactory(Protocol):
    def __call__(self, state: CampaignState) -> CampaignRoundInputs: ...


@dataclass(frozen=True, slots=True)
class CampaignRoundRecord:
    """Durable in-process result and before/after state for one round."""

    round_index: int
    opportunity_ref: str
    status: str
    state_before: CampaignState
    result: ResearchRoundResult
    state_after: CampaignState

    schema = "recclaw.research-line.campaign-round.v1"

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    @property
    def outcome_kind(self) -> str:
        return self.status

    def to_dict(self) -> dict[str, Any]:
        interpretation = self.result.interpretation
        return canonical_value(
            {
                "schema": self.schema,
                "round_index": self.round_index,
                "opportunity_ref": self.opportunity_ref,
                "status": self.status,
                "failure_taxonomy": (
                    interpretation.failure_taxonomy if interpretation is not None else None
                ),
                "experiment_executed": self.result.candidate_run is not None,
                "metric_bearing_experiment": self.result.has_metric_bearing_attempt,
                "attempt_count": len(self.result.attempts),
                "metric_bearing_attempt_index": self.result.metric_bearing_attempt_index,
                "state_before": self.state_before.to_dict(),
                "result": self.result.to_dict(),
                "state_after": self.state_after.to_dict(),
            }
        )


def _write_once(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return False
    return True


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CampaignError(f"campaign JSON root is not an object: {path}")
    return value


def _read_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, ImportError, AttributeError) as error:
        raise CampaignError(f"cannot read campaign checkpoint {path}: {error}") from error


def _unique_by_id(
    items: Sequence[CandidateProposalV4 | OpenSpecSearchCandidateV1],
) -> tuple[CandidateProposalV4 | OpenSpecSearchCandidateV1, ...]:
    by_id: dict[str, CandidateProposalV4 | OpenSpecSearchCandidateV1] = {}
    for item in items:
        by_id[item.candidate_id] = item
    return tuple(by_id.values())


def _compact_round_attempt_summary(
    attempt: RoundAttemptV1,
    *,
    round_index: int,
) -> Mapping[str, Any]:
    """Keep only bounded attempt identity in successor scientific memory.

    The lossless attempt, including diagnostic successor Context and Memory
    snapshots, remains in the current ``CampaignRoundRecord`` checkpoint and
    trace.  Copying those snapshots into the successor Context would make the
    next round embed the complete prior ``round_attempts`` history again.
    """

    candidate_run = attempt.candidate_run
    acquisition = attempt.acquisition

    def digest_or_none(value: Any) -> str | None:
        return sha256_digest(value) if value is not None else None

    return canonical_value(
        {
            "schema": "recclaw.research-line.round-attempt-summary.v1",
            "round_index": round_index,
            "attempt_index": attempt.attempt_index,
            "candidate_id": attempt.candidate_id,
            "candidate_semantic_digest": attempt.binding.mechanism_semantics_digest,
            "observation_seed": candidate_run.get("seed"),
            "outcome": candidate_run.get(
                "exit_status", candidate_run.get("status")
            ),
            "outcome_digest": sha256_digest(candidate_run),
            "metric_digest": digest_or_none(candidate_run.get("metrics")),
            "failure_scope": attempt.failure_scope,
            "engineering_disposition": attempt.engineering_disposition,
            "metric_bearing": attempt.metric_bearing,
            "physical_observation_ref": attempt.observation_ref,
            "physical_observation_digest": attempt.observation_digest,
            "execution_recipe_digest": sha256_digest(attempt.execution_recipe),
            "binding_digest": attempt.binding.digest,
            "route_trace_digest": (
                acquisition.route_trace.digest if acquisition is not None else None
            ),
            "failure_detail_digest": digest_or_none(attempt.failure_detail),
            "resource_prediction_digest": digest_or_none(attempt.resource_prediction),
            "live_health_decisions_digest": digest_or_none(
                attempt.live_health_decisions
            ),
            "diagnostic_feedback_digest": digest_or_none(attempt.diagnostic_feedback),
        }
    )


_PORTFOLIO_HISTORY_LIMIT = 64
_PORTFOLIO_HISTORY_CONTAINER_KEYS = ("attempts", "records", "history", "observations")


def _history_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        for key in _PORTFOLIO_HISTORY_CONTAINER_KEYS:
            nested = value.get(key)
            if isinstance(nested, (tuple, list)):
                return tuple(item for item in nested if isinstance(item, Mapping))
        return (value,) if "candidate_id" in value else ()
    if isinstance(value, (tuple, list)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _explicit_resource_compute_pattern(
    resource_prediction: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(resource_prediction, Mapping):
        return None
    prediction = resource_prediction.get("prediction")
    nested = prediction.get("compute_pattern") if isinstance(prediction, Mapping) else None
    top_level = resource_prediction.get("compute_pattern")
    values = [value for value in (nested, top_level) if value is not None]
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        return None
    if len(set(values)) != 1:
        return None
    return values[0].strip()


def _prepared_portfolio_candidate(
    result: ResearchRoundResult,
    attempt: RoundAttemptV1,
) -> tuple[PortfolioCandidateV2 | None, Any | None]:
    prepared = result.prepared
    if prepared is None:
        return None, None
    for handoff in tuple(getattr(prepared, "candidate_handoffs", ()) or ()):
        if getattr(handoff, "candidate_id", None) != attempt.candidate_id:
            continue
        candidate = getattr(handoff, "portfolio_candidate", None)
        if (
            isinstance(candidate, PortfolioCandidateV2)
            and candidate.candidate_id == attempt.candidate_id
            and candidate.semantic_digest == attempt.binding.mechanism_semantics_digest
        ):
            return candidate, handoff
        return None, None
    for candidate in tuple(getattr(prepared, "portfolio_candidates", ()) or ()):
        if (
            isinstance(candidate, PortfolioCandidateV2)
            and candidate.candidate_id == attempt.candidate_id
            and candidate.semantic_digest == attempt.binding.mechanism_semantics_digest
        ):
            return candidate, None
    return None, None


def _typed_resource_admission(attempt: RoundAttemptV1) -> bool | None:
    if attempt.metric_bearing:
        return True
    typed_resource_values = {
        "RESOURCE",
        "RESOURCE_CENSORED",
        "HEALTH_RESOURCE",
        "HEALTH_RESOURCE_CENSORED",
        "RESOURCE_HEALTH",
    }
    for evidence in (
        attempt.failure_detail,
        attempt.diagnostic_feedback,
        *attempt.live_health_decisions,
    ):
        if not isinstance(evidence, Mapping):
            continue
        for field_name in (
            "failure_class",
            "typed_blocker_class",
            "resource_failure_class",
            "resource_disposition",
            "health_resource_class",
            "censoring_trigger",
        ):
            value = evidence.get(field_name)
            normalized = str(value).strip().upper() if value is not None else ""
            if normalized in typed_resource_values:
                return False
    return None


def _portfolio_attempt_row(
    result: ResearchRoundResult,
    attempt: RoundAttemptV1,
    *,
    round_index: int,
) -> tuple[Mapping[str, Any], PortfolioCandidateV2 | None] | None:
    candidate, handoff = _prepared_portfolio_candidate(result, attempt)
    proposal = attempt.binding.proposal
    compute_pattern = (
        candidate.compute_pattern
        if candidate is not None
        else _explicit_resource_compute_pattern(attempt.resource_prediction)
    )
    family_id = (
        candidate.family_id
        if candidate is not None
        else getattr(proposal, "mechanism_axis", None)
    )
    parent_id = (
        candidate.parent_id
        if candidate is not None
        else getattr(proposal, "parent_candidate_id", None)
    )
    if (
        not isinstance(compute_pattern, str)
        or not compute_pattern.strip()
        or not isinstance(family_id, str)
        or not family_id.strip()
    ):
        return None
    semantic_digest = (
        candidate.semantic_digest
        if candidate is not None
        else attempt.binding.mechanism_semantics_digest
    )
    source_digest = sha256_digest(
        {
            "candidate_handoff_digest": (
                handoff.digest
                if handoff is not None
                else None
            ),
            "portfolio_candidate_digest": (
                sha256_digest(candidate.to_dict()) if candidate is not None else None
            ),
            "binding_digest": attempt.binding.digest,
            "resource_prediction_digest": (
                sha256_digest(attempt.resource_prediction)
                if attempt.resource_prediction is not None
                else None
            ),
        }
    )
    row: dict[str, Any] = {
        "round_index": round_index,
        "candidate_id": attempt.candidate_id,
        "candidate_semantic_digest": semantic_digest,
        "family_id": family_id.strip(),
        "compute_pattern": compute_pattern.strip(),
        "sealed": True,
        "sealed_valid_seal": bool(attempt.metric_bearing),
        "attempt_digest": attempt.digest,
        "source_digest": source_digest,
    }
    if parent_id is not None:
        if not isinstance(parent_id, str) or not parent_id.strip():
            return None
        row["parent_id"] = parent_id.strip()
    resource_admitted = _typed_resource_admission(attempt)
    if resource_admitted is not None:
        row["sealed_resource_admitted"] = resource_admitted
    return canonical_value(row), candidate


def _finite_comparator_delta(event: SearchUtilityEventV2) -> float | None:
    delta = event.comparator_delta
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        return None
    delta = float(delta)
    return delta if math.isfinite(delta) else None


def _frontier_explicitly_updated(
    frontier: Mapping[str, Any],
    *,
    event: SearchUtilityEventV2,
    round_index: int,
) -> bool:
    candidates: list[Any] = []
    trajectory = frontier.get("effect_trajectory")
    if isinstance(trajectory, (tuple, list)):
        candidates.extend(reversed(trajectory))
    global_bank = frontier.get("global")
    if isinstance(global_bank, Mapping):
        observations = global_bank.get("observations")
        if isinstance(observations, (tuple, list)):
            candidates.extend(reversed(observations))
        candidates.append(global_bank.get("last_observation"))
    for record in candidates:
        if not isinstance(record, Mapping):
            continue
        if (
            record.get("round_index") == round_index
            and record.get("candidate_id") == event.candidate_id
            and record.get("candidate_semantic_digest")
            == event.candidate_semantic_digest
            and isinstance(record.get("frontier_updated"), bool)
        ):
            return record["frontier_updated"]
    return False


def _typed_parent_validation(interpretation: Any) -> str:
    allowed = {"INDEPENDENT", "VALIDATED", "UNVERIFIED", "FAILED", "UNKNOWN"}
    for owner in (
        interpretation,
        getattr(interpretation, "closure", None),
        getattr(interpretation, "episode", None),
    ):
        for field_name in (
            "stable_validation",
            "parent_validation",
            "validation_status",
            "development_validation_status",
        ):
            value = getattr(owner, field_name, None)
            value = getattr(value, "value", value)
            normalized = str(value).strip().upper() if value is not None else ""
            if normalized in allowed:
                return normalized
    return "UNKNOWN"


def _portfolio_history_projection(
    before: CampaignState,
    successor: ResearchContext,
    result: ResearchRoundResult,
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    prior_rows: list[Mapping[str, Any]] = []
    row_candidates: dict[str, PortfolioCandidateV2 | None] = {}
    for attempt in result.attempts:
        projected = _portfolio_attempt_row(
            result,
            attempt,
            round_index=before.next_round_index,
        )
        if projected is None:
            continue
        row, candidate = projected
        prior_rows.append(row)
        row_candidates[attempt.candidate_id] = candidate

    family_rows: list[Mapping[str, Any]] = []
    parent_rows: list[Mapping[str, Any]] = []
    frontier_rows: list[Mapping[str, Any]] = []
    interpretation = result.interpretation
    metric_index = result.metric_bearing_attempt_index
    metric_attempt = (
        result.attempts[metric_index]
        if isinstance(metric_index, int)
        and not isinstance(metric_index, bool)
        and 0 <= metric_index < len(result.attempts)
        else None
    )
    event = getattr(interpretation, "search_utility_event", None)
    episode = getattr(interpretation, "episode", None)
    if (
        metric_attempt is not None
        and episode is not None
        and isinstance(event, SearchUtilityEventV2)
        and event.candidate_id == metric_attempt.candidate_id
        and event.candidate_semantic_digest
        == metric_attempt.binding.mechanism_semantics_digest
    ):
        delta = _finite_comparator_delta(event)
        candidate = row_candidates.get(metric_attempt.candidate_id)
        if delta is not None and candidate is not None:
            source_digest = event.digest
            family_rows.append(
                canonical_value(
                    {
                        "round_index": before.next_round_index,
                        "family_id": candidate.family_id,
                        "candidate_id": candidate.candidate_id,
                        "candidate_semantic_digest": candidate.semantic_digest,
                        "stable": True,
                        "stable_delta": delta,
                        "source_digest": source_digest,
                    }
                )
            )
            if candidate.parent_id is not None:
                parent_rows.append(
                    canonical_value(
                        {
                            "round_index": before.next_round_index,
                            "parent_id": candidate.parent_id,
                            "stable": True,
                            "stable_validation": _typed_parent_validation(interpretation),
                            "stable_delta": delta,
                            "source_digest": source_digest,
                        }
                    )
                )
            frontier_gain = (
                delta
                if delta > 0.0
                and _frontier_explicitly_updated(
                    successor.frontier,
                    event=event,
                    round_index=before.next_round_index,
                )
                else 0.0
            )
            frontier_rows.append(
                canonical_value(
                    {
                        "round_index": before.next_round_index,
                        "family_id": candidate.family_id,
                        "candidate_id": candidate.candidate_id,
                        "candidate_semantic_digest": candidate.semantic_digest,
                        "stable": True,
                        "stable_frontier_gain": frontier_gain,
                        "source_digest": source_digest,
                    }
                )
            )
    return {
        "portfolio_prior_attempts": tuple(prior_rows),
        "portfolio_family_history": tuple(family_rows),
        "portfolio_parent_history": tuple(parent_rows),
        "portfolio_frontier_history": tuple(frontier_rows),
    }


def _opportunity_ref(state: CampaignState) -> str:
    return "research-opportunity:" + sha256_digest(
        {
            "campaign_id": state.campaign_id,
            "round_index": state.next_round_index,
            "context_digest": state.context.digest,
            "profile_digest": state.active_profile.profile_digest,
            "incumbent_digest": sha256_digest(state.incumbent_observation),
        }
    )


def _fresh_profile_with_same_entries(
    predecessor: SearchExecutableProfileV1,
    *,
    fresh_campaign_id: str,
) -> SearchExecutableProfileV1:
    """Give a promoted policy a real fresh boundary when no capability changed."""

    payload = canonical_value(
        {
            "predecessor_profile_ref": predecessor.profile_ref,
            "predecessor_profile_digest": predecessor.profile_digest,
            "campaign_id": fresh_campaign_id,
            "protocol_ref": predecessor.protocol_ref,
            "protocol_digest": predecessor.protocol_digest,
            "entries": tuple(entry.canonical_dict() for entry in predecessor.entries),
        }
    )
    return SearchExecutableProfileV1(
        campaign_id=fresh_campaign_id,
        profile_ref=content_id(
            "recclaw-research-line-same-entry-next-round-profile-v1", payload
        ),
        profile_digest=sha256_digest(payload),
        protocol_ref=predecessor.protocol_ref,
        protocol_digest=predecessor.protocol_digest,
        activation=SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN,
        predecessor_campaign_id=predecessor.campaign_id,
        predecessor_profile_ref=predecessor.profile_ref,
        predecessor_profile_digest=predecessor.profile_digest,
        entries=predecessor.entries,
    )


def _promoted_incumbent(
    incumbent: Mapping[str, Any], frontier: Mapping[str, Any]
) -> Mapping[str, Any]:
    metric = frontier.get("incumbent_ndcg@10")
    prior = incumbent.get("frozen_ndcg@10")
    if (
        not isinstance(metric, (int, float))
        or isinstance(metric, bool)
        or not math.isfinite(float(metric))
        or not isinstance(prior, (int, float))
        or isinstance(prior, bool)
        or not math.isfinite(float(prior))
        or float(metric) <= float(prior)
        or not isinstance(frontier.get("incumbent_ref"), str)
        or not frontier["incumbent_ref"].strip()
    ):
        return incumbent
    try:
        digest = validate_sha256(
            frontier.get("incumbent_digest"), field_name="incumbent_digest"
        )
    except (TypeError, ValueError):
        return incumbent
    return canonical_value(
        {
            **dict(incumbent),
            "comparator_ref": frontier["incumbent_ref"],
            "comparator_digest": digest,
            "frozen_ndcg@10": float(metric),
        }
    )


class ResearchCampaign:
    """Run successive rounds while preserving the arm's executable state."""

    state_filename = "CAMPAIGN_STATE.pkl"
    state_projection_filename = "CAMPAIGN_STATE.json"

    def __init__(
        self,
        *,
        root: Path,
        state: CampaignState,
        producer: ResearchProducer,
        runner: ExperimentRunner,
        round_inputs: CampaignRoundInputs | CampaignRoundInputFactory,
        implementer: ImplementerGateway | None = None,
        memory_writer: SearchMemoryWriterV1 | None = None,
        _persist_initial: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self._state = state
        self.producer = producer
        self.runner = runner
        self.round_inputs = round_inputs
        self.implementer = implementer
        self.memory_writer = memory_writer or SearchMemoryWriterV1(
            "DEVELOPMENT_ONLY/SEARCH_MEMORY"
        )
        self._restore_memory_head()
        if _persist_initial:
            self._initialize_checkpoint()

    @classmethod
    def start(cls, **kwargs: Any) -> "ResearchCampaign":
        return cls(**kwargs)

    @classmethod
    def resume(
        cls,
        *,
        root: Path,
        producer: ResearchProducer,
        runner: ExperimentRunner,
        round_inputs: CampaignRoundInputs | CampaignRoundInputFactory,
        implementer: ImplementerGateway | None = None,
        memory_writer: SearchMemoryWriterV1 | None = None,
    ) -> "ResearchCampaign":
        root = Path(root).resolve()
        state_path = root / cls.state_filename
        state = _read_pickle(state_path)
        if not isinstance(state, CampaignState):
            raise CampaignError("campaign checkpoint does not contain CampaignState")
        projection_path = root / cls.state_projection_filename
        if projection_path.is_file():
            projection = _read_json(projection_path)
            if projection.get("state_digest") != state.digest:
                raise CampaignError("campaign state projection digest drift")
        return cls(
            root=root,
            state=state,
            producer=producer,
            runner=runner,
            round_inputs=round_inputs,
            implementer=implementer,
            memory_writer=memory_writer,
            _persist_initial=False,
        )

    load = resume

    @property
    def state(self) -> CampaignState:
        return self._state

    @property
    def current_state(self) -> CampaignState:
        return self._state

    def round_checkpoint_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_CHECKPOINT.pkl"

    def round_trace_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_TRACE.json"

    def _started_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_STARTED.json"

    def _physical_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_PHYSICAL_OBSERVATION.json"

    def _attempt_physical_path(self, round_index: int, attempt_index: int) -> Path:
        return self.root / (
            f"ROUND_{round_index:02d}_ATTEMPT_{attempt_index:02d}_"
            "PHYSICAL_OBSERVATION.json"
        )

    def _attempt_manifest_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_ATTEMPT_MANIFEST.json"

    def _prepared_round_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_PREPARED_CHECKPOINT.pkl"

    @staticmethod
    def _physical_manifest_row(
        payload: Mapping[str, Any],
        path: Path,
    ) -> dict[str, Any]:
        legacy = payload.get("schema") == "recclaw.research-line.physical-observation.v1"
        required = (
            "opportunity_ref",
            "execution_recipe_digest",
            "candidate_run",
        )
        if any(field_name not in payload for field_name in required):
            raise CampaignError(f"physical observation is incomplete: {path}")
        if not isinstance(payload["candidate_run"], Mapping):
            raise CampaignError(f"physical observation candidate_run is invalid: {path}")
        candidate_id = payload.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            candidate_id = "__LEGACY_SINGLE_ATTEMPT__"
        attempt_index = payload.get("attempt_index", 0)
        observation_core = canonical_value(
            {
                key: value
                for key, value in payload.items()
                if key not in {"observation_ref", "observation_digest"}
            }
        )
        return canonical_value(
            {
                "attempt_index": int(attempt_index),
                "candidate_id": candidate_id,
                "binding_digest": payload.get("binding_digest"),
                "execution_recipe_digest": str(payload["execution_recipe_digest"]),
                "candidate_run": dict(payload["candidate_run"]),
                "physical_run_id": payload.get("physical_run_id"),
                "physical_seed": payload.get("physical_seed"),
                "physical_context_digest": payload.get("physical_context_digest"),
                "research_context_digest": payload.get("research_context_digest"),
                "cuda_visible_devices": payload.get("cuda_visible_devices"),
                "reservation_digest": payload.get("reservation_digest"),
                "reservation_status": payload.get("reservation_status"),
                "final_worker_ceiling_seconds": payload.get(
                    "final_worker_ceiling_seconds"
                ),
                "physical_context_applied": payload.get(
                    "physical_context_applied", False
                ),
                "physical_observation_path": str(path),
                "observation_ref": str(
                    payload.get("observation_ref")
                    or content_id("recclaw-research-line-physical-observation-v1", observation_core)
                ),
                "observation_digest": str(
                    payload.get("observation_digest") or sha256_digest(observation_core)
                ),
                "legacy_single_attempt": legacy,
            }
        )

    def _write_attempt_manifest(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_digest: str,
        context_digest: str,
        profile_digest: str,
        attempt_budget: int | None,
        attempts: Sequence[Mapping[str, Any]],
        status: str = "IN_PROGRESS",
        metric_bearing_attempt_index: int | None = None,
        incomplete_reason: str | None = None,
    ) -> Mapping[str, Any]:
        payload = canonical_value(
            {
                "schema": "recclaw.research-line.round-attempt-manifest.v1",
                "round_index": round_index,
                "opportunity_ref": opportunity_ref,
                "state_digest": state_digest,
                "context_digest": context_digest,
                "profile_digest": profile_digest,
                "attempt_budget": attempt_budget,
                "status": status,
                "metric_bearing_attempt_index": metric_bearing_attempt_index,
                "incomplete_reason": incomplete_reason,
                "attempts": tuple(attempts),
            }
        )
        _atomic_write(
            self._attempt_manifest_path(round_index),
            canonical_json_bytes(payload) + b"\n",
        )
        return payload

    def _load_attempt_manifest(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_digest: str,
        context_digest: str,
        profile_digest: str,
        attempt_budget: int | None,
    ) -> dict[str, Any]:
        manifest_path = self._attempt_manifest_path(round_index)
        if manifest_path.is_file():
            raw = dict(_read_json(manifest_path))
            if raw.get("opportunity_ref") != opportunity_ref:
                raise CampaignError("round attempt manifest opportunity identity drift")
            for field_name, expected in (
                ("state_digest", state_digest),
                ("context_digest", context_digest),
                ("profile_digest", profile_digest),
            ):
                if raw.get(field_name) != expected:
                    raise CampaignError(f"round attempt manifest {field_name} drift")
            if (
                attempt_budget is not None
                and raw.get("attempt_budget", attempt_budget) != attempt_budget
            ):
                raise CampaignError("round attempt manifest attempt_budget drift")
            raw_attempts = raw.get("attempts", ())
            if not isinstance(raw_attempts, (tuple, list)):
                raise CampaignError("round attempt manifest attempts are invalid")
            attempts = [
                canonical_value(dict(item))
                for item in raw_attempts
                if isinstance(item, Mapping)
            ]
        else:
            attempts = []
            raw = {}

        known_by_index: dict[int, dict[str, Any]] = {}
        known_candidates: set[str] = set()
        for item in attempts:
            try:
                index = int(item["attempt_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise CampaignError("round attempt manifest index is invalid") from error
            if index < 0 or index in known_by_index:
                raise CampaignError("round attempt manifest index is duplicated")
            candidate_id = item.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise CampaignError("round attempt manifest candidate identity is invalid")
            if candidate_id != "__LEGACY_SINGLE_ATTEMPT__":
                if candidate_id in known_candidates:
                    raise CampaignError("round attempt manifest candidate is duplicated")
                known_candidates.add(candidate_id)
            known_by_index[index] = item
        physical_paths = []
        legacy_path = self._physical_path(round_index)
        if legacy_path.is_file():
            physical_paths.append(legacy_path)
        physical_paths.extend(
            sorted(self.root.glob(f"ROUND_{round_index:02d}_ATTEMPT_*_PHYSICAL_OBSERVATION.json"))
        )
        changed = not manifest_path.is_file() or "attempt_budget" not in raw
        for physical_path in physical_paths:
            physical = _read_json(physical_path)
            if physical.get("opportunity_ref") != opportunity_ref:
                raise CampaignError("physical observation opportunity identity drift")
            if physical.get("round_index") is not None and int(
                physical.get("round_index", -1)
            ) != round_index:
                raise CampaignError("physical observation round index drift")
            row = self._physical_manifest_row(physical, physical_path)
            index = int(row["attempt_index"])
            prior = known_by_index.get(index)
            if prior is None:
                if (
                    row["candidate_id"] != "__LEGACY_SINGLE_ATTEMPT__"
                    and row["candidate_id"] in known_candidates
                ):
                    raise CampaignError("round attempt manifest candidate is duplicated")
                known_by_index[index] = row
                changed = True
            elif (
                prior.get("observation_digest") != row["observation_digest"]
                or (
                    prior.get("candidate_id") != row["candidate_id"]
                    and not (
                        row.get("legacy_single_attempt") is True
                        and prior.get("candidate_id")
                        != "__LEGACY_SINGLE_ATTEMPT__"
                    )
                )
            ):
                raise CampaignError("round attempt manifest conflicts with physical observation")
            if row["candidate_id"] != "__LEGACY_SINGLE_ATTEMPT__":
                known_candidates.add(row["candidate_id"])
        attempts = [known_by_index[index] for index in sorted(known_by_index)]
        if [int(item["attempt_index"]) for item in attempts] != list(range(len(attempts))):
            raise CampaignError("round attempt manifest indices are not contiguous")
        result = {
            "schema": "recclaw.research-line.round-attempt-manifest.v1",
            "round_index": round_index,
            "opportunity_ref": opportunity_ref,
            "state_digest": state_digest,
            "context_digest": context_digest,
            "profile_digest": profile_digest,
            "attempt_budget": raw.get("attempt_budget", attempt_budget),
            "status": raw.get("status", "IN_PROGRESS"),
            "metric_bearing_attempt_index": raw.get("metric_bearing_attempt_index"),
            "incomplete_reason": raw.get("incomplete_reason"),
            "attempts": attempts,
        }
        if changed:
            self._write_attempt_manifest(
                round_index=round_index,
                opportunity_ref=opportunity_ref,
                state_digest=state_digest,
                context_digest=context_digest,
                profile_digest=profile_digest,
                attempt_budget=attempt_budget,
                attempts=attempts,
                status=str(result["status"]),
                metric_bearing_attempt_index=result["metric_bearing_attempt_index"],
                incomplete_reason=result["incomplete_reason"],
            )
        return result

    def _validate_prepared_round(
        self,
        *,
        prepared: Any,
        round_index: int,
        opportunity_ref: str,
        state_before: CampaignState,
        attempt_budget: int,
        budget_snapshot: Mapping[str, Any],
    ) -> PreparedResearchRoundV1:
        if not isinstance(prepared, PreparedResearchRoundV1):
            raise CampaignError("prepared round checkpoint has an invalid prepared round")
        if prepared.context_digest != state_before.context.digest:
            raise CampaignError("prepared round context identity drift")
        if (
            prepared.profile_ref != state_before.active_profile.profile_ref
            or prepared.profile_digest != state_before.active_profile.profile_digest
        ):
            raise CampaignError("prepared round profile identity drift")
        slate = prepared.search_slate
        if slate is not None and (
            slate.campaign_id != state_before.active_profile.campaign_id
            or slate.profile_ref != state_before.active_profile.profile_ref
            or slate.profile_digest != state_before.active_profile.profile_digest
        ):
            raise CampaignError("prepared round search slate identity drift")
        if slate is not None and sha256_digest(slate.budget_snapshot) != sha256_digest(
            budget_snapshot
        ):
            raise CampaignError("prepared round budget snapshot drift")
        if isinstance(round_index, bool) or round_index < 1:
            raise CampaignError("prepared round index is invalid")
        if not isinstance(opportunity_ref, str) or not opportunity_ref:
            raise CampaignError("prepared round opportunity identity is invalid")
        if round_index != state_before.next_round_index:
            raise CampaignError("prepared round index is not the current opportunity")
        if opportunity_ref != _opportunity_ref(state_before):
            raise CampaignError("prepared round opportunity identity drift")
        if (
            isinstance(attempt_budget, bool)
            or not isinstance(attempt_budget, int)
            or attempt_budget < 0
        ):
            raise CampaignError("prepared round attempt budget is invalid")
        return prepared

    def _persist_prepared_round(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_before: CampaignState,
        attempt_budget: int,
        budget_snapshot: Mapping[str, Any],
        prepared: PreparedResearchRoundV1,
    ) -> PreparedResearchRoundV1:
        """Atomically persist the scheduler's non-physical round boundary."""

        self._validate_prepared_round(
            prepared=prepared,
            round_index=round_index,
            opportunity_ref=opportunity_ref,
            state_before=state_before,
            attempt_budget=attempt_budget,
            budget_snapshot=budget_snapshot,
        )
        payload = {
            "schema": "recclaw.research-line.prepared-round-checkpoint.v1",
            "round_index": round_index,
            "opportunity_ref": opportunity_ref,
            "state_digest": state_before.digest,
            "context_digest": state_before.context.digest,
            "profile_ref": state_before.active_profile.profile_ref,
            "profile_digest": state_before.active_profile.profile_digest,
            "attempt_budget": attempt_budget,
            "budget_snapshot_digest": sha256_digest(budget_snapshot),
            "prepared_digest": prepared.digest,
            "prepared": prepared,
        }
        path = self._prepared_round_path(round_index)
        if path.is_file():
            existing = self._load_prepared_round(
                round_index=round_index,
                opportunity_ref=opportunity_ref,
                state_digest=state_before.digest,
                context_digest=state_before.context.digest,
                profile_ref=state_before.active_profile.profile_ref,
                profile_digest=state_before.active_profile.profile_digest,
                attempt_budget=attempt_budget,
                budget_snapshot=budget_snapshot,
            )
            if existing is None or existing.digest != prepared.digest:
                raise CampaignError("prepared round checkpoint identity drift")
            return existing
        _atomic_write(
            path,
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        return prepared

    def _load_prepared_round(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_digest: str,
        context_digest: str,
        profile_ref: str,
        profile_digest: str,
        attempt_budget: int,
        budget_snapshot: Mapping[str, Any],
    ) -> PreparedResearchRoundV1 | None:
        path = self._prepared_round_path(round_index)
        if not path.is_file():
            return None
        payload = _read_pickle(path)
        if not isinstance(payload, Mapping):
            raise CampaignError("prepared round checkpoint root is invalid")
        if payload.get("schema") != "recclaw.research-line.prepared-round-checkpoint.v1":
            raise CampaignError("prepared round checkpoint schema drift")
        for field_name, expected in (
            ("round_index", round_index),
            ("opportunity_ref", opportunity_ref),
            ("state_digest", state_digest),
            ("context_digest", context_digest),
            ("profile_ref", profile_ref),
            ("profile_digest", profile_digest),
            ("attempt_budget", attempt_budget),
            ("budget_snapshot_digest", sha256_digest(budget_snapshot)),
        ):
            if payload.get(field_name) != expected:
                raise CampaignError(f"prepared round checkpoint {field_name} drift")
        prepared = self._validate_prepared_round(
            prepared=payload.get("prepared"),
            round_index=round_index,
            opportunity_ref=opportunity_ref,
            state_before=self._state,
            attempt_budget=attempt_budget,
            budget_snapshot=budget_snapshot,
        )
        if payload.get("prepared_digest") != prepared.digest:
            raise CampaignError("prepared round checkpoint digest drift")
        return prepared

    def _restore_memory_head(self) -> None:
        expected = self._state.search_memory_head
        actual = self.memory_writer.head
        if expected is None:
            if actual is not None:
                self._state = replace(self._state, search_memory_head=actual)
            return
        if actual is None:
            # SearchMemoryWriterV1 is intentionally a small in-process writer;
            # restore its head from the trusted local checkpoint on resume.
            self.memory_writer._head = expected
        elif actual.digest != expected.digest:
            raise CampaignError("injected Search Memory writer is ahead of checkpoint")

    def _round_memory_writer(self) -> SearchMemoryWriterV1:
        """Keep an incomplete round from mutating the campaign Memory head."""

        writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
        writer._head = self._state.search_memory_head
        return writer

    def _initialize_checkpoint(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        state_path = self.root / self.state_filename
        if state_path.exists():
            existing = _read_pickle(state_path)
            if not isinstance(existing, CampaignState) or existing.digest != self._state.digest:
                raise CampaignError("campaign root already contains different state")
            return
        self._persist_state()

    def _persist_state(self) -> None:
        state_payload = pickle.dumps(self._state, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write(self.root / self.state_filename, state_payload)
        projection = canonical_value(
            {
                "schema": "recclaw.research-line.campaign-state-projection.v1",
                "state_digest": self._state.digest,
                "state": self._state.to_dict(),
            }
        )
        _atomic_write(
            self.root / self.state_projection_filename,
            canonical_json_bytes(projection) + b"\n",
        )

    def _round_inputs(self) -> CampaignRoundInputs:
        supplied = (
            self.round_inputs(self._state)
            if callable(self.round_inputs)
            else self.round_inputs
        )
        if not isinstance(supplied, CampaignRoundInputs):
            raise CampaignError("round_inputs must return CampaignRoundInputs")
        qualified = dict(self._state.qualified_execution_by_capability)
        qualified.update(dict(supplied.qualified_execution_by_capability))
        roots = dict(self._state.candidate_root_by_capability)
        roots.update(dict(supplied.candidate_root_by_capability))
        resource_profiles = dict(
            getattr(self._state, "resource_profile_by_capability", {}) or {}
        )
        resource_profiles.update(dict(supplied.resource_profile_by_capability))
        innovation_inputs = supplied.innovation_inputs
        if innovation_inputs is not None and self.implementer is not None:
            innovation_inputs = replace(innovation_inputs, implementer=self.implementer)
        return replace(
            supplied,
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resource_profiles,
            innovation_inputs=innovation_inputs,
        )

    def _runner_for_round(
        self,
        round_index: int,
        opportunity_ref: str,
        manifest: dict[str, Any] | None = None,
        observation_seed: str | int | None = None,
    ) -> ExperimentRunner:
        progress = manifest if manifest is not None else {"attempts": []}

        def guarded_runner(
            recipe: Mapping[str, Any], binding: Any
        ) -> Mapping[str, Any]:
            candidate_id = getattr(getattr(binding, "proposal", None), "candidate_id", None)
            if not isinstance(candidate_id, str) or not candidate_id:
                raise CampaignError("runner binding lacks a candidate identity")
            existing = next(
                (
                    item
                    for item in progress.get("attempts", ())
                    if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
                ),
                None,
            )
            if existing is None:
                existing = next(
                    (
                        item
                        for item in progress.get("attempts", ())
                        if isinstance(item, Mapping)
                        and item.get("legacy_single_attempt") is True
                    ),
                    None,
                )
            if existing is not None:
                observation_path = Path(str(existing["physical_observation_path"]))
                payload = _read_json(observation_path)
                if payload.get("opportunity_ref") != opportunity_ref:
                    raise CampaignError("physical observation opportunity identity drift")
                payload_candidate_id = payload.get("candidate_id")
                if payload_candidate_id not in {
                    None,
                    candidate_id,
                    "__LEGACY_SINGLE_ATTEMPT__",
                }:
                    raise CampaignError("physical observation candidate identity drift")
                if payload.get("attempt_index") is not None and int(
                    payload["attempt_index"]
                ) != int(existing["attempt_index"]):
                    raise CampaignError("physical observation attempt identity drift")
                if payload.get("execution_recipe_digest") != sha256_digest(recipe):
                    raise CampaignError("physical observation recipe identity drift")
                value = payload.get("candidate_run")
                if not isinstance(value, Mapping):
                    raise CampaignError("physical observation lacks candidate_run")
                return value

            attempt_index = max(
                (
                    int(item["attempt_index"])
                    for item in progress.get("attempts", ())
                    if isinstance(item, Mapping) and "attempt_index" in item
                ),
                default=-1,
            ) + 1
            context_value = canonical_value(
                {
                    "schema": PHYSICAL_CONTEXT_SCHEMA,
                    "campaign_id": self._state.campaign_id,
                    "round_index": round_index,
                    "opportunity_ref": opportunity_ref,
                    "attempt_index": attempt_index,
                    "candidate_id": candidate_id,
                    "binding_digest": getattr(binding, "digest", None),
                    "candidate_semantic_digest": getattr(
                        binding, "mechanism_semantics_digest", None
                    ),
                    "research_context_digest": self._state.context.digest,
                    "profile_digest": self._state.active_profile.profile_digest,
                    "seed": observation_seed,
                }
            )
            context_digest = sha256_digest(context_value)
            physical_context = _immutable_canonical_mapping(context_value)
            context_runner = getattr(
                self.runner, "run_with_physical_context", None
            )
            if context_runner is not None and not callable(context_runner):
                raise CampaignError(
                    "runner run_with_physical_context is not callable"
                )
            if context_runner is None:
                value = self.runner(recipe, binding)
            else:
                value = context_runner(
                    recipe,
                    binding,
                    physical_context,
                )
            if not isinstance(value, Mapping):
                raise CampaignError("injected runner must return a mapping")
            candidate_run = canonical_value(dict(value))
            physical_identity = _physical_identity_fields(
                candidate_run,
                context_digest=context_digest,
                context_applied=context_runner is not None,
            )
            observation_core = canonical_value(
                {
                    "schema": "recclaw.research-line.physical-observation.v2",
                    "round_index": round_index,
                    "attempt_index": attempt_index,
                    "opportunity_ref": opportunity_ref,
                    "candidate_id": candidate_id,
                    "execution_recipe_digest": sha256_digest(recipe),
                    "binding_digest": getattr(binding, "digest", None),
                    "candidate_run": candidate_run,
                    **physical_identity,
                }
            )
            observation_digest = sha256_digest(observation_core)
            observation_ref = content_id(
                "recclaw-research-line-physical-observation-v2",
                observation_core,
            )
            payload = canonical_value(
                {
                    **dict(observation_core),
                    "observation_ref": observation_ref,
                    "observation_digest": observation_digest,
                }
            )
            observation_path = self._attempt_physical_path(round_index, attempt_index)
            _write_once(observation_path, canonical_json_bytes(payload) + b"\n")
            # The old single-attempt reader remains a compatibility projection
            # of the first physical attempt.
            if attempt_index == 0:
                _write_once(
                    self._physical_path(round_index),
                    canonical_json_bytes(payload) + b"\n",
                )
            row = self._physical_manifest_row(payload, observation_path)
            progress.setdefault("attempts", []).append(row)
            self._write_attempt_manifest(
                round_index=round_index,
                opportunity_ref=opportunity_ref,
                state_digest=self._state.digest,
                context_digest=self._state.context.digest,
                profile_digest=self._state.active_profile.profile_digest,
                attempt_budget=(
                    int(progress["attempt_budget"])
                    if progress.get("attempt_budget") is not None
                    else None
                ),
                attempts=progress["attempts"],
                status="IN_PROGRESS",
            )
            return candidate_run

        return guarded_runner

    def _advance_state(
        self,
        before: CampaignState,
        result: ResearchRoundResult,
    ) -> CampaignState:
        interpretation = result.interpretation
        if interpretation is None:
            raise CampaignError("Research round did not produce typed feedback")
        successor = interpretation.successor_context
        policy = interpretation.policy_successor
        active_profile = before.active_profile
        qualified = dict(before.qualified_execution_by_capability)
        roots = dict(before.candidate_root_by_capability)
        resource_profiles = dict(
            getattr(before, "resource_profile_by_capability", {}) or {}
        )

        proposal_items: list[CandidateProposalV4] = list(before.carryover_proposals)
        for outcome in (*result.producer_outcomes, *result.carryover_outcomes):
            if outcome.source_proposal is not None:
                proposal_items.append(outcome.source_proposal)
        open_items: list[OpenSpecSearchCandidateV1] = list(
            before.carryover_open_candidates
        )
        attempted_candidate_ids = {
            attempt.candidate_id for attempt in result.attempts
        }
        legacy_selected_binding = (
            result.search_acquisition.selected_binding
            if not result.attempts
            and result.candidate_run is not None
            and result.search_acquisition is not None
            else None
        )
        legacy_candidate_id = (
            legacy_selected_binding.proposal.candidate_id
            if legacy_selected_binding is not None
            else None
        )
        if legacy_candidate_id is not None:
            attempted_candidate_ids.add(legacy_candidate_id)
        queue_memory = successor.scientific_memory.get("global_memory", {})
        queue = ResearchTaskQueueV2.from_dict(
            queue_memory.get("task_queue")
            if isinstance(queue_memory, Mapping)
            else None
        )
        pending_confirmation_ids = {
            task.candidate_id
            for task in queue.tasks
            if task.status in {
                ResearchTaskStatusV2.PENDING,
                ResearchTaskStatusV2.ACTIVE,
            }
        }
        metric_candidate_id = (
            result.attempts[result.metric_bearing_attempt_index].candidate_id
            if result.metric_bearing_attempt_index is not None
            else legacy_candidate_id
            if result.has_metric_bearing_attempt
            else None
        )
        retained_attempt_ids = (
            {metric_candidate_id}
            if metric_candidate_id in pending_confirmation_ids
            else set()
        )
        retired_attempt_ids = attempted_candidate_ids - retained_attempt_ids
        if attempted_candidate_ids:
            proposal_items = [
                proposal
                for proposal in proposal_items
                if proposal.candidate_id not in retired_attempt_ids
            ]
            open_items = [
                candidate
                for candidate in open_items
                if candidate.candidate_id not in retired_attempt_ids
            ]

        innovation = result.innovation
        if innovation is not None and innovation.activation_ready:
            active_profile, successor, candidate, execution = activate_staged_innovation(
                result
            )
            qualified[candidate.capability_ref] = canonical_value(dict(execution))
            root = innovation.candidate_root or str(
                execution.get("candidate_root_ref", "")
            )
            if root:
                roots[candidate.capability_ref] = root
            if innovation.resource_profile is not None:
                resource_profiles[candidate.capability_ref] = canonical_value(
                    dict(innovation.resource_profile)
                )
            if isinstance(candidate, CandidateProposalV4):
                proposal_items.append(candidate)
            else:
                open_items.append(candidate)

        meta = result.meta_research
        if meta is not None and meta.activated_policy is not None:
            if meta.activation_receipt is None:
                raise CampaignError("promoted Meta policy lacks an activation receipt")
            if (
                active_profile.campaign_id == before.active_profile.campaign_id
                and active_profile.profile_digest == before.active_profile.profile_digest
            ):
                active_profile = _fresh_profile_with_same_entries(
                    before.active_profile,
                    fresh_campaign_id=meta.activation_receipt.campaign_id,
                )
            if active_profile.campaign_id != meta.activation_receipt.campaign_id:
                raise CampaignError("Meta activation is not bound to the next-round profile")
            successor, policy = activate_promoted_meta_strategy(
                result,
                next_profile=active_profile,
            )

        if result.attempts:
            prior_attempts = successor.scientific_memory.get("round_attempts", ())
            prior_attempts = (
                tuple(prior_attempts)
                if isinstance(prior_attempts, (tuple, list))
                else ()
            )
            compact_attempts = tuple(
                _compact_round_attempt_summary(
                    attempt,
                    round_index=before.next_round_index,
                )
                for attempt in result.attempts
            )
            combined_attempts = (*prior_attempts, *compact_attempts)
            successor = replace(
                successor,
                scientific_memory={
                    **successor.scientific_memory,
                    "round_attempts": (
                        *combined_attempts[-64:],
                    ),
                },
            )

        portfolio_history = _portfolio_history_projection(
            before,
            successor,
            result,
        )
        if any(portfolio_history.values()) or any(
            alias in successor.scientific_memory.get("global_memory", {})
            if isinstance(successor.scientific_memory.get("global_memory"), Mapping)
            else False
            for alias in portfolio_history
        ):
            existing_global_memory = successor.scientific_memory.get("global_memory", {})
            global_memory = (
                dict(existing_global_memory)
                if isinstance(existing_global_memory, Mapping)
                else {}
            )
            for alias, rows in portfolio_history.items():
                existing_rows = _history_rows(global_memory.get(alias))
                if rows or existing_rows:
                    global_memory[alias] = canonical_value(
                        (*existing_rows, *rows)[-_PORTFOLIO_HISTORY_LIMIT:]
                    )
            successor = replace(
                successor,
                scientific_memory={
                    **successor.scientific_memory,
                    "global_memory": global_memory,
                },
            )

        successor = replace(
            successor,
            campaign_id=active_profile.campaign_id,
            active_profile_ref=active_profile.profile_ref,
            active_profile_digest=active_profile.profile_digest,
            policy=policy.to_dict(),
        )
        if successor.round_index != before.next_round_index + 1:
            raise CampaignError("Research successor did not advance exactly one round")
        return CampaignState(
            campaign_id=active_profile.campaign_id,
            next_round_index=successor.round_index,
            context=successor,
            active_profile=active_profile,
            policy=policy,
            search_memory_head=interpretation.search_memory_snapshot,
            carryover_proposals=tuple(
                item
                for item in _unique_by_id(proposal_items)
                if isinstance(item, CandidateProposalV4)
            ),
            carryover_open_candidates=tuple(
                item
                for item in _unique_by_id(open_items)
                if isinstance(item, OpenSpecSearchCandidateV1)
            ),
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resource_profiles,
            incumbent_observation=_promoted_incumbent(
                before.incumbent_observation, successor.frontier
            ),
            frontier=successor.frontier,
            last_round_result_digest=sha256_digest(result.to_dict()),
        )

    @staticmethod
    def _attach_observation_identities(
        result: ResearchRoundResult,
        manifest: Mapping[str, Any],
    ) -> ResearchRoundResult:
        rows = {
            str(item.get("candidate_id")): item
            for item in manifest.get("attempts", ())
            if isinstance(item, Mapping)
        }
        legacy_rows = tuple(
            item
            for item in manifest.get("attempts", ())
            if isinstance(item, Mapping)
            and item.get("legacy_single_attempt") is True
        )

        def row_for(attempt: RoundAttemptV1) -> Mapping[str, Any]:
            row = rows.get(attempt.candidate_id)
            if row is not None:
                return row
            return next(
                (
                    item
                    for item in legacy_rows
                    if int(item.get("attempt_index", -1)) == attempt.attempt_index
                ),
                {},
            )

        attempts = tuple(
            replace(
                attempt,
                observation_ref=(
                    str(row_for(attempt)["observation_ref"])
                    if row_for(attempt).get("observation_ref") is not None
                    else attempt.observation_ref
                ),
                observation_digest=(
                    str(row_for(attempt)["observation_digest"])
                    if row_for(attempt).get("observation_digest") is not None
                    else attempt.observation_digest
                ),
            )
            for attempt in result.attempts
        )
        return replace(result, attempts=attempts)

    def _persist_attempt_result_manifest(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_before: CampaignState,
        manifest: Mapping[str, Any],
        result: ResearchRoundResult,
        status: str,
    ) -> Mapping[str, Any]:
        rows: list[dict[str, Any]] = [
            dict(item)
            for item in manifest.get("attempts", ())
            if isinstance(item, Mapping)
        ]
        by_candidate = {
            str(item.get("candidate_id")): item
            for item in rows
            if item.get("candidate_id") is not None
        }
        for attempt in result.attempts:
            row = by_candidate.get(attempt.candidate_id)
            if row is None:
                row = next(
                    (
                        item
                        for item in rows
                        if item.get("legacy_single_attempt") is True
                        and int(item.get("attempt_index", -1)) == attempt.attempt_index
                    ),
                    None,
                )
                if row is not None:
                    old_candidate_id = row.get("candidate_id")
                    row["candidate_id"] = attempt.candidate_id
                    row["legacy_single_attempt"] = False
                    if old_candidate_id is not None:
                        by_candidate.pop(str(old_candidate_id), None)
                    by_candidate[attempt.candidate_id] = row
            if row is None:
                row = {
                    "attempt_index": attempt.attempt_index,
                    "candidate_id": attempt.candidate_id,
                    "candidate_run": attempt.candidate_run,
                    "execution_recipe_digest": sha256_digest(attempt.execution_recipe),
                }
                rows.append(row)
                by_candidate[attempt.candidate_id] = row
            row.update(
                {
                    "attempt_digest": attempt.digest,
                    "engineering_disposition": attempt.engineering_disposition,
                    "failure_scope": attempt.failure_scope,
                    "failure_detail": attempt.failure_detail,
                    "diagnostic_feedback": attempt.diagnostic_feedback,
                    "route_trace_digest": (
                        attempt.acquisition.route_trace.digest
                        if attempt.acquisition is not None
                        else None
                    ),
                }
            )
        return self._write_attempt_manifest(
            round_index=round_index,
            opportunity_ref=opportunity_ref,
            state_digest=state_before.digest,
            context_digest=state_before.context.digest,
            profile_digest=state_before.active_profile.profile_digest,
            attempt_budget=(
                int(manifest["attempt_budget"])
                if manifest.get("attempt_budget") is not None
                else None
            ),
            attempts=tuple(sorted(rows, key=lambda item: int(item["attempt_index"]))),
            status=status,
            metric_bearing_attempt_index=result.metric_bearing_attempt_index,
            incomplete_reason=result.incomplete_reason,
        )

    @staticmethod
    def _hold_state_after_incomplete_attempts(
        before: CampaignState,
        result: ResearchRoundResult,
    ) -> CampaignState:
        """Retire only candidate-scoped failures while holding the round index."""

        retryable_ids = {
            attempt.candidate_id
            for attempt in result.attempts
            if attempt.failure_scope in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}
        }
        if not retryable_ids:
            return before
        return CampaignState(
            campaign_id=before.campaign_id,
            next_round_index=before.next_round_index,
            context=before.context,
            active_profile=before.active_profile,
            policy=before.policy,
            search_memory_head=before.search_memory_head,
            carryover_proposals=tuple(
                item
                for item in before.carryover_proposals
                if item.candidate_id not in retryable_ids
            ),
            carryover_open_candidates=tuple(
                item
                for item in before.carryover_open_candidates
                if item.candidate_id not in retryable_ids
            ),
            qualified_execution_by_capability=before.qualified_execution_by_capability,
            candidate_root_by_capability=before.candidate_root_by_capability,
            resource_profile_by_capability=getattr(
                before, "resource_profile_by_capability", {}
            ),
            incumbent_observation=before.incumbent_observation,
            frontier=before.frontier,
            last_round_result_digest=sha256_digest(result.to_dict()),
        )

    @staticmethod
    def _status(result: ResearchRoundResult) -> str:
        if result.attempt_scheduler_enabled and not result.has_metric_bearing_attempt:
            return "INCOMPLETE"
        if result.selected_outcome is None or result.candidate_run is None:
            return "OUTCOME_MISSING"
        if result.interpretation is None or getattr(result.interpretation, "episode", None) is None:
            return "TYPED_FAILURE"
        return "TYPED_EPISODE"

    def _load_round(self, round_index: int) -> CampaignRoundRecord:
        record = _read_pickle(self.round_checkpoint_path(round_index))
        if not isinstance(record, CampaignRoundRecord):
            raise CampaignError("round checkpoint does not contain CampaignRoundRecord")
        if record.round_index != round_index:
            raise CampaignError("round checkpoint index drift")
        trace_path = self.round_trace_path(round_index)
        if trace_path.is_file():
            trace = _read_json(trace_path)
            if trace.get("record_digest") != record.digest:
                raise CampaignError("round trace digest drift")
        return record

    def _seal_record(self, record: CampaignRoundRecord) -> CampaignRoundRecord:
        checkpoint_payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_path = self.round_checkpoint_path(record.round_index)
        replace_incomplete = False
        if checkpoint_path.is_file():
            prior = self._load_round(record.round_index)
            if not (
                prior.status == "INCOMPLETE"
                and prior.result.attempt_scheduler_enabled
            ):
                raise CampaignError("cannot overwrite a sealed completed round")
            if prior.opportunity_ref != record.opportunity_ref:
                raise CampaignError("incomplete round opportunity identity drift")
            if prior.state_before.digest != record.state_before.digest:
                raise CampaignError("incomplete round predecessor state drift")
            replace_incomplete = True
        if replace_incomplete:
            _atomic_write(checkpoint_path, checkpoint_payload)
        elif not _write_once(checkpoint_path, checkpoint_payload):
            raise CampaignError("round checkpoint was sealed concurrently")
        trace = canonical_value(
            {
                "schema": "recclaw.research-line.campaign-round-trace.v1",
                "record_digest": record.digest,
                "checkpoint_sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
                "record": record.to_dict(),
            }
        )
        trace_payload = canonical_json_bytes(trace) + b"\n"
        trace_path = self.round_trace_path(record.round_index)
        if replace_incomplete:
            _atomic_write(trace_path, trace_payload)
        elif not _write_once(trace_path, trace_payload):
            raise CampaignError("round trace was sealed concurrently")
        self._state = record.state_after
        self.memory_writer._head = record.state_after.search_memory_head
        self._persist_state()
        return record

    def record_missing_round(
        self,
        round_index: int | None = None,
        *,
        reason: str = "interrupted before a durable physical observation",
        failure_code: str = "INTERRUPTED_BEFORE_DURABLE_PHYSICAL_OBSERVATION",
    ) -> CampaignRoundRecord:
        """Consume one interrupted opportunity without replaying external work."""

        expected = self._state.next_round_index
        index = expected if round_index is None else int(round_index)
        checkpoint_path = self.round_checkpoint_path(index)
        if checkpoint_path.is_file():
            record = self._load_round(index)
            if record.status == "INCOMPLETE" and record.result.attempt_scheduler_enabled:
                return self.run_round(index)
            return record
        if index != expected:
            raise CampaignError(
                f"campaign is at round {expected}; cannot record round {index}"
            )
        inputs = self._round_inputs()
        if inputs.attempt_scheduler and (
            self._started_path(index).is_file()
            or self._prepared_round_path(index).is_file()
        ):
            # A scheduler round with a start marker is recoverable only from
            # its prepared boundary; never turn it into a synthetic missing
            # round or re-run preparation here.
            return self.run_round(index)
        if self._physical_path(index).is_file():
            raise CampaignError(
                "durable physical observation exists; recover the ordinary round"
            )
        opportunity_ref = _opportunity_ref(self._state)
        _write_once(
            self._started_path(index),
            canonical_json_bytes(
                {
                    "schema": "recclaw.research-line.campaign-round-started.v1",
                    "round_index": index,
                    "opportunity_ref": opportunity_ref,
                    "state_digest": self._state.digest,
                    "context_digest": self._state.context.digest,
                    "profile_digest": self._state.active_profile.profile_digest,
                }
            )
            + b"\n",
        )
        round_memory = self._round_memory_writer()

        def interrupted_producer(_role: str, _view: Mapping[str, Any]) -> Any:
            raise RuntimeError(reason)

        producer_outcomes = produce_research_specs(
            self._state.context,
            interrupted_producer,
            inputs.producer_bindings,
        )
        interpretation = interpret_missing_search_opportunity(
            context=self._state.context,
            producer_outcomes=producer_outcomes,
            diagnostic_detail={
                "reason": reason,
                "failure_code": failure_code,
                "opportunity_ref": opportunity_ref,
            },
            next_discriminative_test=inputs.next_discriminative_test,
            policy=self._state.policy,
            memory_writer=round_memory,
        )
        result = ResearchRoundResult(
            context=self._state.context,
            active_profile=self._state.active_profile,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=(),
            resolutions=(),
            deferred_innovation_outcomes=(),
            deferred_search_outcomes=(),
            search_acquisition=None,
            innovation=None,
            selected_outcome=None,
            execution_recipe=None,
            candidate_run=None,
            interpretation=interpretation,
            provider_traces=(),
            meta_research=None,
            attempt_scheduler_enabled=inputs.attempt_scheduler,
            incomplete_reason=(
                "ROUND_INTERRUPTED_BEFORE_PHYSICAL_OBSERVATION"
                if inputs.attempt_scheduler
                else None
            ),
        )
        before = self._state
        after = before if inputs.attempt_scheduler else self._advance_state(before, result)
        return self._seal_record(
            CampaignRoundRecord(
                round_index=index,
                opportunity_ref=opportunity_ref,
                status=("INCOMPLETE" if inputs.attempt_scheduler else "OUTCOME_MISSING"),
                state_before=before,
                result=result,
                state_after=after,
            )
        )

    def run_round(self, round_index: int | None = None) -> CampaignRoundRecord:
        expected = self._state.next_round_index
        index = expected if round_index is None else round_index
        if isinstance(index, bool) or index < 1:
            raise CampaignError("round_index must be positive")

        checkpoint_path = self.round_checkpoint_path(index)
        resuming_incomplete = False
        if checkpoint_path.is_file():
            record = self._load_round(index)
            if record.status == "INCOMPLETE" and record.result.attempt_scheduler_enabled:
                if index != self._state.next_round_index:
                    raise CampaignError(
                        "incomplete round is not the campaign's current opportunity"
                    )
                if self._state.digest not in {
                    record.state_before.digest,
                    record.state_after.digest,
                }:
                    raise CampaignError("incomplete round checkpoint state drift")
                # The state_after projection of an incomplete round may have
                # retired candidate-local failures.  Rebuild the frozen round
                # from state_before so the manifest can replay those attempts
                # and route only the still-unattempted candidates.
                self._state = record.state_before
                self.memory_writer._head = record.state_before.search_memory_head
                resuming_incomplete = True
            else:
                if index == self._state.next_round_index:
                    if record.state_before.digest != self._state.digest:
                        raise CampaignError("round checkpoint predecessor state drift")
                    self._state = record.state_after
                    self.memory_writer._head = record.state_after.search_memory_head
                    self._persist_state()
                return record
        if index != expected:
            raise CampaignError(
                f"campaign is at round {expected}; cannot run round {index}"
            )
        if self.round_trace_path(index).is_file() and not resuming_incomplete:
            raise CampaignError("round trace exists without its checkpoint")

        inputs = self._round_inputs()
        configured_attempt_budget = (
            _configured_attempt_budget(inputs) if inputs.attempt_scheduler else None
        )
        if resuming_incomplete and not inputs.attempt_scheduler:
            raise CampaignError(
                "an incomplete scheduler round must resume with attempt_scheduler=True"
            )
        opportunity_ref = _opportunity_ref(self._state)
        state_before = self._state
        started = canonical_value(
            {
                "schema": "recclaw.research-line.campaign-round-started.v1",
                "round_index": index,
                "opportunity_ref": opportunity_ref,
                "state_digest": self._state.digest,
                "context_digest": self._state.context.digest,
                "profile_digest": self._state.active_profile.profile_digest,
            }
        )
        started_path = self._started_path(index)
        started_exists = started_path.exists()
        if started_path.exists():
            started_value = _read_json(started_path)
            if started_value.get("opportunity_ref") != opportunity_ref:
                raise CampaignError(
                    "round start marker has a different opportunity identity"
                )
            manifest = self._load_attempt_manifest(
                round_index=index,
                opportunity_ref=opportunity_ref,
                state_digest=self._state.digest,
                context_digest=self._state.context.digest,
                profile_digest=self._state.active_profile.profile_digest,
                attempt_budget=configured_attempt_budget,
            )
            if (
                not manifest["attempts"]
                and not resuming_incomplete
                and not inputs.attempt_scheduler
            ):
                return self.record_missing_round(
                    index,
                    reason="round was interrupted before a durable physical observation",
                    failure_code="INTERRUPTED_BEFORE_DURABLE_PHYSICAL_OBSERVATION",
                )
        else:
            _write_once(started_path, canonical_json_bytes(started) + b"\n")
            manifest = self._load_attempt_manifest(
                round_index=index,
                opportunity_ref=opportunity_ref,
                state_digest=self._state.digest,
                context_digest=self._state.context.digest,
                profile_digest=self._state.active_profile.profile_digest,
                attempt_budget=configured_attempt_budget,
            )

        prepared_round: PreparedResearchRoundV1 | None = None
        if inputs.attempt_scheduler:
            prepared_round = self._load_prepared_round(
                round_index=index,
                opportunity_ref=opportunity_ref,
                state_digest=state_before.digest,
                context_digest=state_before.context.digest,
                profile_ref=state_before.active_profile.profile_ref,
                profile_digest=state_before.active_profile.profile_digest,
                attempt_budget=int(manifest["attempt_budget"]),
                budget_snapshot=inputs.budget_snapshot,
            )
            if resuming_incomplete:
                checkpoint_prepared = record.result.prepared
                if prepared_round is None:
                    prepared_round = self._validate_prepared_round(
                        prepared=checkpoint_prepared,
                        round_index=index,
                        opportunity_ref=opportunity_ref,
                        state_before=state_before,
                        attempt_budget=int(manifest["attempt_budget"]),
                        budget_snapshot=inputs.budget_snapshot,
                    )
                elif isinstance(checkpoint_prepared, PreparedResearchRoundV1):
                    if checkpoint_prepared.digest != prepared_round.digest:
                        raise CampaignError(
                            "incomplete round prepared checkpoint digest drift"
                        )
            if prepared_round is None and (started_exists or manifest["attempts"]):
                raise CampaignError(
                    "scheduler round has no prepared checkpoint for safe recovery"
                )
        if inputs.attempt_scheduler and prepared_round is None and (
            not started_exists and not manifest["attempts"]
        ):
            # The callback below is the only path that may create this
            # scheduler-only checkpoint, and it runs before the first runner.
            prepare_callback: Callable[[PreparedResearchRoundV1], None] | None = (
                lambda prepared: self._persist_prepared_round(
                    round_index=index,
                    opportunity_ref=opportunity_ref,
                    state_before=state_before,
                    attempt_budget=int(manifest["attempt_budget"]),
                    budget_snapshot=inputs.budget_snapshot,
                    prepared=prepared,
                )
            )
        else:
            prepare_callback = None
        round_memory = self._round_memory_writer()
        result = run_research_round(
            context=self._state.context,
            active_profile=self._state.active_profile,
            producer=self.producer,
            producer_bindings=inputs.producer_bindings,
            resolver_environment=inputs.resolver_environment,
            carryover_proposals=self._state.carryover_proposals,
            carryover_open_candidates=self._state.carryover_open_candidates,
            budget_snapshot=inputs.budget_snapshot,
            router=inputs.router,
            policy=self._state.policy,
            memory_writer=round_memory,
            runner=self._runner_for_round(
                index,
                opportunity_ref,
                manifest,
                observation_seed=inputs.observation_seed,
            ),
            incumbent_observation=self._state.incumbent_observation,
            metric_contract_digest=inputs.metric_contract_digest,
            observation_seed=inputs.observation_seed,
            next_discriminative_test=inputs.next_discriminative_test,
            confirmation_seed=inputs.confirmation_seed,
            qualified_execution_by_capability=inputs.qualified_execution_by_capability,
            research_profile_source=inputs.research_profile_source,
            candidate_handoff_factory=inputs.candidate_handoff_factory,
            candidate_root_by_capability=inputs.candidate_root_by_capability,
            resource_profile_by_capability=inputs.resource_profile_by_capability,
            innovation_inputs=inputs.innovation_inputs,
            meta_research_inputs=inputs.meta_research_inputs,
            attempt_scheduler=inputs.attempt_scheduler,
            max_attempts_per_round=(
                int(manifest["attempt_budget"])
                if inputs.attempt_scheduler
                else inputs.max_attempts_per_round
            ),
            recovered_attempts=(
                tuple(manifest["attempts"]) if inputs.attempt_scheduler else ()
            ),
            prepared_round=prepared_round,
            on_prepared=prepare_callback,
            portfolio_candidates=(
                inputs.portfolio_candidates if inputs.portfolio_candidates else None
            ),
        )
        if inputs.attempt_scheduler:
            result = self._attach_observation_identities(result, manifest)
        status = self._status(result)
        self._persist_attempt_result_manifest(
            round_index=index,
            opportunity_ref=opportunity_ref,
            state_before=state_before,
            manifest=manifest,
            result=result,
            status=status,
        )
        state_after = (
            self._hold_state_after_incomplete_attempts(state_before, result)
            if inputs.attempt_scheduler and not result.has_metric_bearing_attempt
            else self._advance_state(state_before, result)
        )
        record = CampaignRoundRecord(
            round_index=index,
            opportunity_ref=opportunity_ref,
            status=status,
            state_before=state_before,
            result=result,
            state_after=state_after,
        )
        return self._seal_record(record)


__all__ = [
    "CampaignError",
    "CampaignRoundInputFactory",
    "CampaignRoundInputs",
    "CampaignRoundRecord",
    "CampaignState",
    "ResearchCampaign",
    "ResearchCampaignState",
]
