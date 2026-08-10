"""Small production composition for one complete Research Line round."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from recclaw_core.experiments.helix_abc_v1.capability_admission import (
    VersionedCapabilityRegistry,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (
    MAX_REPAIR_TURNS,
    build_mechanical_repair_request,
    is_mechanical_repair_failure,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    ExperimentBindingError,
    ExperimentBindingV1,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.idea_quality import (
    admit_research_innovation_candidate,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    MechanicalRecBoleAdapterV1,
    MechanicalQualificationRun,
    RecBoleQualificationFixture,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    InnovationSpineError,
    MaterializedCandidate,
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
)
from recclaw_core.experiments.helix_abc_v1.next_fresh_profile import (
    NextFreshProfileBuildManifest,
)
from recclaw_core.experiments.helix_abc_v1.meta_control import (
    MetaControlActivationReceiptV1,
    MetaControlPromotionDecisionV1,
    MetaControlShadowEvaluationV1,
    MetaControlUpdateProposalV1,
    ProposalOnlyShadowMetricsV1,
    activate_promoted_control_policy,
    build_meta_update_proposal,
    decide_meta_control_promotion,
    evaluate_proposal_only_shadow,
    materialize_proposed_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    project_candidate_proposal_v4,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    ExperimentAcquisitionResultV1,
    FrozenExperimentSlateV1,
    OpenSpecSearchCandidateV1,
    SearchCandidateBindingV1,
    SearchExecutableProfileV1,
    SearchProfileEntryOriginV1,
    activate_next_fresh_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
    open_spec_realization_identity,
    predecessor_executable_entries,
    route_frozen_experiment_slate,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityKindV1,
    CapabilityResolutionResultV1,
    CapabilityResolutionV1,
    CurrentProfileExpressibilityV1,
    ExecutableProfileVNext,
    ProfileBuildReceiptV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_orchestration import (
    admit_local_qualification,
    build_local_next_fresh_profile,
    qualify_local_innovation_candidate,
    resolve_producer_outcomes,
)

from .execution import (
    execution_recipe_for_search_binding,
    project_common_execution_feedback,
)
from .interfaces import ProducerOutcome, ResearchContext, ResearchTaskQueueV2
from .interpreter import (
    EpisodeInterpretation,
    MissingSearchInterpretation,
    interpret_missing_search_opportunity,
    interpret_scientific_diagnostic,
    interpret_typed_research_episode,
)
from .portfolio import PortfolioAttemptFailureV2, PortfolioCandidateV2
from .portfolio_profile_builder import PortfolioCandidateProfileV2
from .producers import ResearchProducer, produce_research_specs
from .profile_source import (
    ResearchProfileRecordV1,
    ResearchProfileSourceV1,
    lineage_identity_digest,
)


_NEXT_DEVELOPMENT_SEED = "NEXT_DEVELOPMENT_SEED"


def _trace_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_trace_value(item) for item in value)
    for method_name in ("to_dict", "canonical_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _trace_value(method())
    if is_dataclass(value):
        return {
            field.name: _trace_value(getattr(value, field.name, None))
            for field in fields(value)
        }
    return canonical_value(value)


def _trace_dataclass(value: Any) -> dict[str, Any]:
    return canonical_value(
        {
            field.name: _trace_value(getattr(value, field.name, None))
            for field in fields(value)
        }
    )


class ImplementerGateway(Protocol):
    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ExperimentRunner(Protocol):
    def __call__(
        self,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
    ) -> Mapping[str, Any]: ...


class CandidateHandoffFactory(Protocol):
    """Build the complete, outcome-blind handoff for a frozen round."""

    def __call__(
        self,
        *,
        context: ResearchContext,
        active_profile: SearchExecutableProfileV1,
        resolutions: Sequence[
            tuple[ProducerOutcome, CapabilityResolutionV1 | None]
        ],
        search_bindings: Sequence[SearchCandidateBindingV1],
        qualified_execution_by_capability: Mapping[str, Mapping[str, Any]],
        candidate_root_by_capability: Mapping[str, str | Path],
        resource_profile_by_capability: Mapping[str, Mapping[str, Any]],
    ) -> Sequence["RoundCandidateHandoffV1"]: ...


class InnovationResourceProbe(Protocol):
    def __call__(
        self,
        *,
        candidate_root: Path,
        source_path: Path,
        entrypoint: str,
        source_sha256: str,
        execution_recipe: Mapping[str, Any],
        probe_root: Path,
        candidate_ref: str,
        candidate_package_digest: str,
    ) -> Mapping[str, Any]: ...


class MetaShadowReplay(Protocol):
    def __call__(
        self,
        *,
        context: ResearchContext,
        champion_policy: VersionedResearchPolicyV1,
        challenger_policy: VersionedResearchPolicyV1,
        search_memory: Any,
    ) -> Mapping[str, Any]: ...


SearchCandidate = CandidateProposalV4 | OpenSpecSearchCandidateV1


@dataclass(frozen=True, slots=True)
class InnovationRuntimeInputs:
    """Inputs with immediate consumers in implementation, admission, or activation."""

    implementer: ImplementerGateway
    policy: SharedImplementerPolicy
    candidate_parent: Path
    fixture_factory: Callable[
        [SharedImplementerPolicy, int, Path], RecBoleQualificationFixture
    ]
    unit_check_factory: Callable[
        [SharedImplementerPolicy], Callable[[Any, Any, Any], None]
    ]
    capability_kind: CapabilityKindV1
    capability_version: str
    registry_version: str
    predecessor_registry_ref: str
    predecessor_registry_digest: str
    profile_version: str
    fresh_campaign_id: str
    resource_admission_required: bool = False
    resource_probe: InnovationResourceProbe | None = None
    resource_probe_parent: Path | None = None
    resource_structural_context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        contract = self.policy.execution_contract
        required = {"capability_family", "model", "base_model_config", "config"}
        if contract is not None and (
            not isinstance(contract, Mapping) or set(contract) != required
        ):
            raise ValueError(
                "Innovation execution_contract, when static, must contain exactly the four qualified execution fields"
            )
        if not isinstance(self.resource_admission_required, bool):
            raise ValueError("resource_admission_required must be a boolean")
        configured = self.resource_probe is not None or self.resource_probe_parent is not None
        if self.resource_admission_required and not configured:
            raise ValueError(
                "resource-admitted Innovation requires a disposable resource probe"
            )
        if configured and (
            not callable(self.resource_probe)
            or not isinstance(self.resource_probe_parent, Path)
        ):
            raise ValueError(
                "resource_probe and resource_probe_parent must be supplied together"
            )
        if self.resource_probe_parent is not None:
            object.__setattr__(
                self,
                "resource_probe_parent",
                self.resource_probe_parent.resolve(),
            )
        if self.resource_structural_context is not None and not isinstance(
            self.resource_structural_context, Mapping
        ):
            raise ValueError("resource_structural_context must be a mapping")
        object.__setattr__(
            self,
            "resource_structural_context",
            canonical_value(dict(self.resource_structural_context or {})),
        )


@dataclass(frozen=True, slots=True)
class IdeaAcquisitionResult:
    """Minimal policy-consuming acquisition trace for OpenSpec Innovation."""

    pool_digest: str
    ranked_spec_digests: tuple[str, ...]
    selected_spec_digest: str
    score_by_spec_digest: Mapping[str, float]
    policy_digest: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MetaResearchInputs:
    """One offline replay consumer and the fresh campaign it may promote into."""

    offline_replay: MetaShadowReplay
    next_campaign_id: str

    def __post_init__(self) -> None:
        if not self.next_campaign_id.strip():
            raise ValueError("next_campaign_id must be non-empty")


@dataclass(frozen=True, slots=True)
class MetaResearchResult:
    update_proposal: MetaControlUpdateProposalV1
    challenger_policy: VersionedResearchPolicyV1
    offline_replay: Mapping[str, Any]
    shadow_evaluation: MetaControlShadowEvaluationV1
    promotion_decision: MetaControlPromotionDecisionV1
    activated_policy: VersionedResearchPolicyV1 | None
    activation_receipt: MetaControlActivationReceiptV1 | None

    def to_dict(self) -> dict[str, Any]:
        return _trace_dataclass(self)


@dataclass(frozen=True, slots=True)
class InnovationLaneResult:
    selected_outcome: ProducerOutcome
    resolution: CapabilityResolutionV1
    idea_acquisition: IdeaAcquisitionResult
    attempts: tuple[Mapping[str, Any], ...]
    materialized: MaterializedCandidate | None
    qualification: MechanicalQualificationRun | None
    capability: QualifiedCapabilityV1 | None
    registry: VersionedCapabilityRegistry | None
    profile_manifest: NextFreshProfileBuildManifest | None
    next_profile: ExecutableProfileVNext | None
    profile_receipt: ProfileBuildReceiptV1 | None
    qualified_execution: Mapping[str, Any] | None
    search_candidate: SearchCandidate | None
    candidate_root: str | None
    fresh_campaign_id: str
    resource_profile: Mapping[str, Any] | None = None
    quality_admission: Mapping[str, Any] | None = None

    @property
    def admitted(self) -> bool:
        if self.quality_admission is not None:
            return bool(self.quality_admission.get("admitted") is True)
        return self.capability is not None

    @property
    def mechanically_qualified(self) -> bool:
        return self.capability is not None

    @property
    def resource_admitted(self) -> bool:
        return bool(
            self.quality_admission is not None
            and self.quality_admission.get("admitted") is True
            and self.quality_admission.get("status") == "RESOURCE_ADMITTED"
        )

    @property
    def activation_ready(self) -> bool:
        return (
            self.admitted
            and self.next_profile is not None
            and self.search_candidate is not None
        )

    def to_dict(self) -> dict[str, Any]:
        payload = _trace_dataclass(self)
        payload["selected_outcome"]["digest"] = self.selected_outcome.digest
        payload["idea_acquisition"]["digest"] = self.idea_acquisition.digest
        return canonical_value(payload)


@dataclass(frozen=True, slots=True)
class RoundCandidateHandoffV1:
    """Identity-bound, outcome-blind inputs for one current-round candidate.

    ``portfolio_candidate`` remains a Router feature record.  The qualified
    execution, candidate-local root, and resource profile are separate
    physical-admission inputs and are carried together only at this explicit
    runtime boundary.
    """

    candidate_id: str
    binding_digest: str
    portfolio_candidate: PortfolioCandidateV2
    qualified_execution: Mapping[str, Any] | None = None
    candidate_root_path: str | None = None
    resource_profile: Mapping[str, Any] | None = None
    portfolio_profile: PortfolioCandidateProfileV2 | None = None
    profile_digest: str | None = None
    source_schema_version: str | None = None
    source_ref: str | None = None
    source_digest: str | None = None
    lineage_identity_digest: str | None = None
    durable_evidence_digests: Mapping[str, str] | None = None

    schema = "recclaw.research-line.round-candidate-handoff.v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or self.candidate_id != self.candidate_id.strip()
        ):
            raise ValueError("candidate_id must be normalized and non-empty")
        try:
            binding_digest = validate_sha256(
                self.binding_digest,
                field_name="binding_digest",
            )
        except (TypeError, ValueError) as error:
            raise ValueError("binding_digest must be a sha256 digest") from error
        object.__setattr__(self, "binding_digest", binding_digest)
        if not isinstance(self.portfolio_candidate, PortfolioCandidateV2):
            raise ValueError("portfolio_candidate must be PortfolioCandidateV2")
        if self.portfolio_candidate.candidate_id != self.candidate_id:
            raise ValueError("handoff candidate_id differs from portfolio candidate")
        if self.qualified_execution is not None:
            if not isinstance(self.qualified_execution, Mapping):
                raise ValueError("qualified_execution must be a mapping")
            object.__setattr__(
                self,
                "qualified_execution",
                canonical_value(dict(self.qualified_execution)),
            )
        root = self.candidate_root_path
        if root is not None:
            if isinstance(root, Path):
                root = str(root.resolve())
            elif isinstance(root, str) and root.strip():
                root = str(Path(root).resolve())
            else:
                raise ValueError("candidate_root_path must be a non-empty path")
            object.__setattr__(self, "candidate_root_path", root)
        if self.resource_profile is not None:
            if not isinstance(self.resource_profile, Mapping):
                raise ValueError("resource_profile must be a mapping")
            object.__setattr__(
                self,
                "resource_profile",
                canonical_value(dict(self.resource_profile)),
            )
        profile = self.portfolio_profile
        source_fields = (
            self.profile_digest,
            self.source_schema_version,
            self.source_ref,
            self.source_digest,
            self.lineage_identity_digest,
        )
        if profile is not None and not isinstance(
            profile,
            PortfolioCandidateProfileV2,
        ):
            raise ValueError(
                "portfolio_profile must be PortfolioCandidateProfileV2"
            )
        if any(value is not None for value in source_fields):
            if profile is None or any(value is None for value in source_fields):
                raise ValueError(
                    "profile/source identity must be complete when supplied"
                )
            try:
                profile_digest = validate_sha256(
                    self.profile_digest,
                    field_name="profile_digest",
                )
                source_digest = validate_sha256(
                    self.source_digest,
                    field_name="source_digest",
                )
                lineage_digest = validate_sha256(
                    self.lineage_identity_digest,
                    field_name="lineage_identity_digest",
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "profile/source identity fields must be sha256 digests"
                ) from error
            if profile_digest != profile.profile_digest:
                raise ValueError("profile_digest differs from portfolio_profile")
            if lineage_digest != lineage_identity_digest(profile.candidate):
                raise ValueError("lineage_identity_digest differs from profile")
            object.__setattr__(self, "profile_digest", profile_digest)
            object.__setattr__(self, "source_digest", source_digest)
            object.__setattr__(self, "lineage_identity_digest", lineage_digest)
            if (
                not isinstance(self.source_schema_version, str)
                or not self.source_schema_version.strip()
                or not isinstance(self.source_ref, str)
                or not self.source_ref.strip()
            ):
                raise ValueError(
                    "source_schema_version and source_ref must be non-empty"
                )
            object.__setattr__(
                self,
                "source_schema_version",
                self.source_schema_version.strip(),
            )
            object.__setattr__(self, "source_ref", self.source_ref.strip())
        elif profile is not None:
            raise ValueError(
                "portfolio_profile requires profile/source identity fields"
            )
        if self.durable_evidence_digests is not None:
            if profile is None or any(value is None for value in source_fields):
                raise ValueError(
                    "durable evidence digests require a complete profile/source identity"
                )
            if not isinstance(self.durable_evidence_digests, Mapping):
                raise ValueError("durable_evidence_digests must be a mapping")
            normalized_evidence_digests: dict[str, str] = {}
            for key, value in self.durable_evidence_digests.items():
                try:
                    normalized_evidence_digests[str(key)] = validate_sha256(
                        value,
                        field_name=f"durable_evidence_digests.{key}",
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "durable_evidence_digests must contain SHA-256 values"
                    ) from error
            object.__setattr__(
                self,
                "durable_evidence_digests",
                canonical_value(normalized_evidence_digests),
            )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "candidate_id": self.candidate_id,
                "binding_digest": self.binding_digest,
                "portfolio_candidate": self.portfolio_candidate.to_dict(),
                "qualified_execution": self.qualified_execution,
                "candidate_root_path": self.candidate_root_path,
                "resource_profile": self.resource_profile,
                **(
                    {
                        "portfolio_profile": self.portfolio_profile.to_dict(),
                        "profile_digest": self.profile_digest,
                        "source_schema_version": self.source_schema_version,
                        "source_ref": self.source_ref,
                        "source_digest": self.source_digest,
                        "lineage_identity_digest": self.lineage_identity_digest,
                        **(
                            {
                                "durable_evidence_digests": self.durable_evidence_digests
                            }
                            if self.durable_evidence_digests is not None
                            else {}
                        ),
                    }
                    if self.portfolio_profile is not None
                    else {}
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedResearchRoundV1:
    """All non-physical preparation that must be reused on round resume."""

    context_digest: str
    profile_ref: str
    profile_digest: str
    producer_outcomes: tuple[ProducerOutcome, ...]
    carryover_outcomes: tuple[ProducerOutcome, ...]
    resolutions: tuple[tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...]
    deferred_innovation_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    deferred_search_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    search_acquisition: ExperimentAcquisitionResultV1 | None
    search_slate: FrozenExperimentSlateV1 | None
    search_bindings: tuple[SearchCandidateBindingV1, ...]
    search_pairs: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...]
    open_search_pairs: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1, OpenSpecSearchCandidateV1],
        ...,
    ]
    innovation: InnovationLaneResult | None
    metric_contract_digest: str
    observation_seed: str
    next_discriminative_test: str
    provider_traces: tuple[Mapping[str, Any], ...]
    confirmation_seed: str | None = None
    portfolio_candidates: tuple[PortfolioCandidateV2, ...] = ()
    candidate_handoffs: tuple[RoundCandidateHandoffV1, ...] = ()

    schema = "recclaw.research-line.prepared-round.v1"

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "context_digest": self.context_digest,
            "profile_ref": self.profile_ref,
            "profile_digest": self.profile_digest,
            "producer_outcomes": tuple(
                _trace_value(item) for item in self.producer_outcomes
            ),
            "carryover_outcomes": tuple(
                _trace_value(item) for item in self.carryover_outcomes
            ),
            "resolutions": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.resolutions
            ),
            "deferred_innovation_outcomes": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.deferred_innovation_outcomes
            ),
            "deferred_search_outcomes": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.deferred_search_outcomes
            ),
            "search_acquisition": _trace_value(self.search_acquisition),
            "search_slate": _trace_value(self.search_slate),
            "search_bindings": tuple(
                _trace_value(item) for item in self.search_bindings
            ),
            "search_pairs": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.search_pairs
            ),
            "open_search_pairs": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                    "candidate": _trace_value(candidate),
                }
                for outcome, resolution, candidate in self.open_search_pairs
            ),
            "innovation": _trace_value(self.innovation),
            "metric_contract_digest": self.metric_contract_digest,
            "observation_seed": self.observation_seed,
            "next_discriminative_test": self.next_discriminative_test,
            "confirmation_seed": getattr(self, "confirmation_seed", None),
            "provider_traces": _trace_value(self.provider_traces),
            "portfolio_candidates": tuple(
                item.to_dict() for item in self.portfolio_candidates
            ),
        }
        handoffs = tuple(getattr(self, "candidate_handoffs", ()) or ())
        if handoffs:
            payload["candidate_handoffs"] = tuple(
                item.to_dict() for item in handoffs
            )
        return canonical_value(payload)


@dataclass(frozen=True, slots=True)
class RoundAttemptV1:
    """One immutable engineering/scientific attempt inside a Research round.

    The existing ``search_acquisition``/``candidate_run`` fields on
    :class:`ResearchRoundResult` remain the compatibility projection for the
    metric-bearing attempt.  This record is the lossless in-round view used by
    the scheduler and campaign persistence layer.
    """

    attempt_index: int
    acquisition: ExperimentAcquisitionResultV1 | None
    binding: SearchCandidateBindingV1
    execution_recipe: Mapping[str, Any]
    candidate_run: Mapping[str, Any]
    engineering_disposition: str
    failure_scope: str | None = None
    failure_detail: Mapping[str, Any] | None = None
    observation_ref: str | None = None
    observation_digest: str | None = None
    resource_prediction: Mapping[str, Any] | None = None
    assigned_deadline_seconds: float | None = None
    live_health_decisions: tuple[Mapping[str, Any], ...] = ()
    diagnostic_feedback: Mapping[str, Any] | None = None
    diagnostic_successor_context: Mapping[str, Any] | None = None
    diagnostic_policy_successor: Mapping[str, Any] | None = None
    diagnostic_search_memory_snapshot: Mapping[str, Any] | None = None

    schema = "recclaw.research-line.round-attempt.v1"

    def __post_init__(self) -> None:
        if isinstance(self.attempt_index, bool) or self.attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        if not isinstance(self.binding, SearchCandidateBindingV1):
            raise ValueError("binding must be SearchCandidateBindingV1")
        if not isinstance(self.execution_recipe, Mapping):
            raise ValueError("execution_recipe must be a mapping")
        if not isinstance(self.candidate_run, Mapping):
            raise ValueError("candidate_run must be a mapping")
        if not isinstance(self.engineering_disposition, str) or not self.engineering_disposition:
            raise ValueError("engineering_disposition must be non-empty")
        object.__setattr__(self, "execution_recipe", canonical_value(dict(self.execution_recipe)))
        object.__setattr__(self, "candidate_run", canonical_value(dict(self.candidate_run)))
        if self.failure_detail is not None:
            object.__setattr__(
                self,
                "failure_detail",
                canonical_value(dict(self.failure_detail)),
            )
        if self.resource_prediction is not None:
            object.__setattr__(
                self,
                "resource_prediction",
                canonical_value(dict(self.resource_prediction)),
            )
        object.__setattr__(
            self,
            "live_health_decisions",
            tuple(canonical_value(dict(item)) for item in self.live_health_decisions),
        )
        if self.diagnostic_feedback is not None:
            object.__setattr__(
                self,
                "diagnostic_feedback",
                canonical_value(dict(self.diagnostic_feedback)),
            )
        for field_name in (
            "diagnostic_successor_context",
            "diagnostic_policy_successor",
            "diagnostic_search_memory_snapshot",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_value(dict(value)),
                )

    @property
    def candidate_id(self) -> str:
        return self.binding.proposal.candidate_id

    @property
    def metric_bearing(self) -> bool:
        return self.engineering_disposition == "METRIC_BEARING"

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "attempt_index": self.attempt_index,
                "acquisition": _trace_value(self.acquisition),
                "binding": self.binding.canonical_dict(),
                "candidate_id": self.candidate_id,
                "execution_recipe": self.execution_recipe,
                "candidate_run": self.candidate_run,
                "engineering_disposition": self.engineering_disposition,
                "failure_scope": self.failure_scope,
                "failure_detail": self.failure_detail,
                "observation_ref": self.observation_ref,
                "observation_digest": self.observation_digest,
                "resource_prediction": self.resource_prediction,
                "assigned_deadline_seconds": self.assigned_deadline_seconds,
                "live_health_decisions": self.live_health_decisions,
                "diagnostic_feedback": self.diagnostic_feedback,
                "diagnostic_successor_context": self.diagnostic_successor_context,
                "diagnostic_policy_successor": self.diagnostic_policy_successor,
                "diagnostic_search_memory_snapshot": (
                    self.diagnostic_search_memory_snapshot
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ResearchRoundResult:
    """Queryable Context-to-feedback trace for one experiment opportunity."""

    context: ResearchContext
    active_profile: SearchExecutableProfileV1
    producer_outcomes: tuple[ProducerOutcome, ...]
    carryover_outcomes: tuple[ProducerOutcome, ...]
    resolutions: tuple[tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...]
    deferred_innovation_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    deferred_search_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    search_acquisition: ExperimentAcquisitionResultV1 | None
    innovation: InnovationLaneResult | None
    selected_outcome: ProducerOutcome | None
    execution_recipe: Mapping[str, Any] | None
    candidate_run: Mapping[str, Any] | None
    interpretation: EpisodeInterpretation | MissingSearchInterpretation | None
    provider_traces: tuple[Mapping[str, Any], ...]
    meta_research: MetaResearchResult | None = None
    attempts: tuple[RoundAttemptV1, ...] = ()
    metric_bearing_attempt_index: int | None = None
    attempt_scheduler_enabled: bool = False
    incomplete_reason: str | None = None
    prepared: PreparedResearchRoundV1 | None = None

    @property
    def successor_context(self) -> ResearchContext:
        if self.interpretation is None:
            raise ValueError("round without closed execution feedback has no successor Context")
        return self.interpretation.successor_context

    def to_dict(self) -> dict[str, Any]:
        payload = _trace_dataclass(self)
        payload["context"].update(
            context_ref=self.context.context_ref,
            context_digest=self.context.digest,
        )
        for projected, outcome in zip(
            payload["producer_outcomes"], self.producer_outcomes, strict=True
        ):
            projected["digest"] = outcome.digest
        for projected, outcome in zip(
            payload["carryover_outcomes"], self.carryover_outcomes, strict=True
        ):
            projected["digest"] = outcome.digest
        if self.search_acquisition is not None:
            payload["search_acquisition"]["route_trace"]["digest"] = (
                self.search_acquisition.route_trace.digest
            )
            binding = self.search_acquisition.selected_binding
            if binding is not None:
                payload["search_acquisition"]["selected_binding"]["digest"] = (
                    binding.digest
                )
        payload["selected_outcome_digest"] = (
            self.selected_outcome.digest if self.selected_outcome is not None else None
        )
        payload["attempt_digests"] = tuple(item.digest for item in self.attempts)
        payload["metric_bearing_attempt_index"] = self.metric_bearing_attempt_index
        payload["attempt_scheduler_enabled"] = self.attempt_scheduler_enabled
        payload["incomplete_reason"] = self.incomplete_reason
        payload["prepared_digest"] = self.prepared.digest if self.prepared else None
        return canonical_value(payload)

    @property
    def has_metric_bearing_attempt(self) -> bool:
        if self.metric_bearing_attempt_index is not None:
            return True
        # Compatibility projection for callers that still use the old
        # single-attempt shape and do not populate ``attempts``.
        return (
            not self.attempt_scheduler_enabled
            and self.candidate_run is not None
            and self.interpretation is not None
            and getattr(self.interpretation, "episode", None) is not None
        )


def _trace_starts(*owners: Any) -> tuple[tuple[Any, int], ...]:
    starts: list[tuple[Any, int]] = []
    seen: set[int] = set()
    for owner in owners:
        if owner is None or id(owner) in seen:
            continue
        traces = getattr(owner, "call_traces", None)
        if isinstance(traces, (tuple, list)):
            starts.append((owner, len(traces)))
            seen.add(id(owner))
    return tuple(starts)


def _new_provider_traces(
    starts: Sequence[tuple[Any, int]],
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for owner, start in starts:
        traces = getattr(owner, "call_traces", ())
        if not isinstance(traces, (tuple, list)):
            continue
        for trace in traces[start:]:
            if isinstance(trace, Mapping):
                result.append(canonical_value(dict(trace)))
    return tuple(result)


def bindings_for_context(
    context: ResearchContext,
    *,
    active_profile: SearchExecutableProfileV1,
    implementation_requirements: Sequence[str],
    compatibility_requirements: Sequence[str],
) -> dict[str, Any]:
    """Build Producer bindings from the actually active profile, fixed or fresh."""

    return canonical_value(
        {
            "context_ref": context.context_ref,
            "context_digest": context.digest,
            "protocol_ref": context.protocol_ref,
            "protocol_digest": context.protocol_digest,
            "current_profile_ref": context.active_profile_ref,
            "current_profile_digest": context.active_profile_digest,
            "current_capability_semantics": tuple(
                entry.semantic_identity_digest for entry in active_profile.entries
            ),
            "implementation_requirements": tuple(implementation_requirements),
            "compatibility_requirements": tuple(compatibility_requirements),
        }
    )


def resolver_environment_for_profile(
    profile: SearchExecutableProfileV1,
    *,
    available_dependencies: Sequence[str],
    budget_limits: Mapping[str, int],
    protocol_requirements: Sequence[str],
) -> dict[str, Any]:
    """Expose every active entry, including admitted capabilities, to Resolver."""

    return canonical_value(
        {
            "current_profile_ref": profile.profile_ref,
            "current_profile_digest": profile.profile_digest,
            "current_capabilities": tuple(
                {
                    "capability_ref": entry.capability_ref,
                    "capability_digest": entry.capability_digest,
                    "semantics_digest": entry.semantic_identity_digest,
                }
                for entry in profile.entries
            ),
            "protocol_ref": profile.protocol_ref,
            "protocol_digest": profile.protocol_digest,
            "protocol_requirements": tuple(protocol_requirements),
            "available_dependencies": tuple(available_dependencies),
            "budget_limits": dict(budget_limits),
        }
    )


def _carryover_outcomes(
    context: ResearchContext,
    proposals: Sequence[CandidateProposalV4],
    bindings: Mapping[str, Any],
) -> tuple[ProducerOutcome, ...]:
    outcomes = []
    for proposal in proposals:
        spec, facts = project_candidate_proposal_v4(proposal, bindings=bindings)
        outcomes.append(
            ProducerOutcome(
                producer_role=proposal.producer_role,
                context_ref=context.context_ref,
                context_digest=context.digest,
                spec=spec,
                resolution_facts=facts,
                source_proposal=proposal,
            )
        )
    return tuple(outcomes)


def _carryover_open_outcomes(
    context: ResearchContext,
    candidates: Sequence[OpenSpecSearchCandidateV1],
) -> tuple[tuple[ProducerOutcome, OpenSpecSearchCandidateV1], ...]:
    """Rebind admitted OpenSpec lineage to the fresh Context for Search."""

    result = []
    for candidate in candidates:
        source = candidate.spec
        spec = replace(
            source,
            context_ref=context.context_ref,
            context_digest=context.digest,
            current_profile_ref=context.active_profile_ref,
            current_profile_digest=context.active_profile_digest,
            current_profile_expressibility_claim=(
                CurrentProfileExpressibilityV1.EXPRESSIBLE
            ),
        )
        outcome = ProducerOutcome(
            producer_role=spec.producer_role,
            context_ref=context.context_ref,
            context_digest=context.digest,
            spec=spec,
            resolution_facts={
                "requested_current_semantics_digest": (
                    candidate.semantic_identity_digest
                ),
                "capability_diff": (),
                "high_change_dimensions": (),
                "required_dependencies": (),
                "required_budget": {},
            },
            source_proposal=None,
        )
        result.append((outcome, candidate))
    return tuple(result)


def _source_files(response: Mapping[str, Any]) -> dict[str, str]:
    files = response.get("files", ())
    if not isinstance(files, (tuple, list)):
        return {}
    return {
        str(item["path"]): str(item["content"])
        for item in files
        if isinstance(item, Mapping) and "path" in item and "content" in item
    }


def _materialization_failure(error: InnovationSpineError) -> dict[str, Any]:
    return canonical_value(
        {
            "stage": "SCHEMA",
            "failure_class": error.failure_class,
            "reason_code": error.reason_code,
            "message": str(error),
        }
    )


def _qualified_execution(
    policy: SharedImplementerPolicy,
    materialized: MaterializedCandidate,
    capability: QualifiedCapabilityV1,
) -> dict[str, Any]:
    contract = policy.execution_contract
    if not isinstance(contract, Mapping):
        raise ValueError("admitted capability requires an explicit execution_contract")
    required = {"capability_family", "model", "base_model_config", "config"}
    if set(contract) != required:
        raise ValueError(
            "execution_contract must contain capability_family, model, base_model_config, and config"
        )
    source_path = (
        materialized.package.executable_entrypoint.split(":", 1)[0].replace(".", "/")
        + ".py"
    )
    manifest = materialized.implementation_receipt["written_files"]
    source = next((row for row in manifest if row["path"] == source_path), None)
    if source is None:
        raise ValueError("candidate entrypoint source is absent from its package manifest")
    return canonical_value(
        {
            **dict(contract),
            "entrypoint_source_sha256": source["sha256"],
            "candidate_package_ref": capability.candidate_package_ref,
            "candidate_package_digest": capability.candidate_package_digest,
            "candidate_root_ref": materialized.package.candidate_root_ref,
            "candidate_root_digest": materialized.package.candidate_root_digest,
            "candidate_source_tree_digest": capability.source_tree_digest,
        }
    )


def _resource_probe_recipe(
    *,
    current_profile: SearchExecutableProfileV1,
    capability: QualifiedCapabilityV1,
    qualified_execution: Mapping[str, Any],
    entrypoint: str,
    mechanism_id: str,
    mechanism_semantics_digest: str,
) -> dict[str, Any]:
    """Bind the disposable probe to the exact realized package and protocol.

    The candidate is not in the active profile until resource admission, so
    the probe records the active profile as its predecessor boundary while
    binding the new capability/package/source identities explicitly.
    """

    recipe = canonical_value(
        {
            **dict(qualified_execution),
            "capability_ref": capability.capability_id,
            "capability_digest": capability.digest,
            "profile_ref": current_profile.profile_ref,
            "profile_digest": current_profile.profile_digest,
            "entrypoint": entrypoint,
            "mechanism_id": mechanism_id,
            "mechanism_semantics_digest": mechanism_semantics_digest,
            "dataset": COMMON_DATASET,
            "split": COMMON_SPLIT,
            "evaluator": COMMON_EVALUATOR,
            "execution_role": "CANDIDATE",
        }
    )
    validate_execution_recipe(recipe)
    return recipe


def _acquire_innovation_spec(
    candidates: Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]],
    *,
    research_policy: VersionedResearchPolicyV1,
) -> tuple[ProducerOutcome, CapabilityResolutionV1, IdeaAcquisitionResult] | None:
    """Rank feasible OpenSpecs without inventing executable program semantics."""

    feasible = tuple(
        (outcome, resolution)
        for outcome, resolution in candidates
        if outcome.spec is not None
    )
    if not feasible:
        return None
    allocations = dict(research_policy.producer_token_allocation)
    axis_targets = tuple(research_policy.mechanism_axis_targeting)
    rows: list[tuple[float, str, ProducerOutcome, CapabilityResolutionV1]] = []
    for outcome, resolution in feasible:
        assert outcome.spec is not None
        dimensions = tuple(outcome.resolution_facts.get("high_change_dimensions", ()))
        projected_axes = tuple(
            {
                "COMPOSITE_MECHANISM": "architecture",
                "CORE_OBJECTIVE": "objective",
                "CORE_RELATION": "relation",
                "CORE_REPRESENTATION": "geometry",
                "CUSTOM_EXECUTABLE_CAPABILITY": "architecture",
                "INTERACTION_STRUCTURE": "message_transform",
                "MODEL_STRUCTURE": "architecture",
                "PROPAGATION_MECHANISM": "propagation",
                "TRAINING_PROCEDURE": "training",
            }.get(str(dimension), str(dimension).lower())
            for dimension in dimensions
        )
        axis_rank = next(
            (index for index, axis in enumerate(axis_targets) if axis in projected_axes),
            len(axis_targets),
        )
        axis_priority = (
            1.0 - axis_rank / max(1, len(axis_targets))
            if axis_rank < len(axis_targets)
            else 0.0
        )
        budget = dict(outcome.resolution_facts.get("required_budget", {}))
        token_cost = float(budget.get("implementation_token_ceiling", 0))
        cost_priority = 1.0 - min(1.0, token_cost / 25_000.0)
        score = round(
            0.5 * float(allocations.get(outcome.producer_role, 0.0))
            + 0.3 * axis_priority
            + 0.2 * cost_priority,
            12,
        )
        rows.append((score, outcome.spec.digest, outcome, resolution))
    rows.sort(key=lambda row: (-row[0], row[1]))
    score_by_spec = {digest: score for score, digest, _outcome, _resolution in rows}
    acquisition = IdeaAcquisitionResult(
        pool_digest=sha256_digest(
            tuple(
                {
                    "producer_outcome_digest": outcome.digest,
                    "resolution_digest": resolution.digest,
                }
                for _score, _digest, outcome, resolution in rows
            )
        ),
        ranked_spec_digests=tuple(digest for _score, digest, _outcome, _resolution in rows),
        selected_spec_digest=rows[0][1],
        score_by_spec_digest=score_by_spec,
        policy_digest=research_policy.digest,
    )
    return rows[0][2], rows[0][3], acquisition


def _implementation_policy(
    inputs: InnovationRuntimeInputs,
    outcome: ProducerOutcome,
) -> SharedImplementerPolicy:
    assert outcome.spec is not None
    contract = outcome.spec.execution_contract
    if contract is None:
        if inputs.policy.execution_contract is None:
            raise ValueError("selected Innovation OpenSpec lacks an execution contract")
        return inputs.policy
    return replace(inputs.policy, execution_contract=contract)


def _open_mechanism_axis(outcome: ProducerOutcome) -> str:
    dimensions = tuple(outcome.resolution_facts.get("high_change_dimensions", ()))
    mapping = {
        "COMPOSITE_MECHANISM": "architecture",
        "CORE_OBJECTIVE": "objective",
        "CORE_RELATION": "relation",
        "CORE_REPRESENTATION": "geometry",
        "CUSTOM_EXECUTABLE_CAPABILITY": "architecture",
        "INTERACTION_STRUCTURE": "message_transform",
        "MODEL_STRUCTURE": "architecture",
        "PROPAGATION_MECHANISM": "propagation",
        "TRAINING_PROCEDURE": "training",
    }
    return next(
        (mapping[str(dimension)] for dimension in dimensions if str(dimension) in mapping),
        "architecture",
    )


def _qualified_open_features(
    outcome: ProducerOutcome,
) -> tuple[SearchUtilityFeaturesV1, RouterFeatureEvidenceV1]:
    budget = dict(outcome.resolution_facts.get("required_budget", {}))
    token_cost = float(budget.get("implementation_token_ceiling", 0))
    estimated_cost = min(0.8, max(0.0, token_cost / 25_000.0))
    utility = SearchUtilityFeaturesV1(
        runnable_probability=1.0,
        useful_signal=0.5,
        # Qualification proves executability, not effect.  Keep the effect
        # prior neutral while giving an as-yet-unobserved high-change
        # realization its actual first-experiment information value.
        frontier_potential=1.0,
        information_gain=1.0,
        cost=estimated_cost,
        blocker_risk=0.0,
    )
    evidence = RouterFeatureEvidenceV1(
        compile_valid=True,
        handler_available=True,
        materializer_available=True,
        blocker_rate=0.0,
        semantic_duplicate=False,
        parent_available=bool(outcome.spec and outcome.spec.closest_parent),
        mechanism_depth=len(outcome.spec.causal_chain) if outcome.spec else 0,
        estimated_cost=estimated_cost,
        llm_diagnostic=utility,
    )
    return utility, evidence


def _run_innovation_lane(
    candidates: Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]],
    *,
    current_profile: SearchExecutableProfileV1,
    current_slate_ref: str,
    current_slate_digest: str,
    research_policy: VersionedResearchPolicyV1,
    inputs: InnovationRuntimeInputs,
) -> InnovationLaneResult | None:
    acquired = _acquire_innovation_spec(
        candidates,
        research_policy=research_policy,
    )
    if acquired is None:
        return None
    selected_outcome, resolution, idea_acquisition = acquired
    assert selected_outcome.spec is not None
    implementer_policy = _implementation_policy(inputs, selected_outcome)
    request = build_shared_implementer_request(
        selected_outcome.spec,
        policy=implementer_policy,
    )
    blind_candidate_id = str(request["blind_candidate_id"])
    attempts: list[Mapping[str, Any]] = []
    response: Mapping[str, Any] | None = None
    materialized: MaterializedCandidate | None = None
    qualification: MechanicalQualificationRun | None = None
    candidate_root: Path | None = None

    for attempt_index in range(MAX_REPAIR_TURNS + 1):
        gateway_request = (
            request
            if attempt_index == 0
            else build_mechanical_repair_request(
                request,
                attempts[-1]["failure"],
                current_source=_source_files(response or {}),
                repair_attempt=attempt_index,
            )
        )
        try:
            response = inputs.implementer(gateway_request)
        except Exception as error:  # external implementer boundary
            attempts.append(
                canonical_value(
                    {
                        "attempt": attempt_index,
                        "failure": {
                            "stage": "PROVIDER",
                            "failure_class": "PROVIDER",
                            "reason_code": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
            )
            break
        attempt_parent = Path(inputs.candidate_parent) / f"attempt-{attempt_index:02d}"
        attempt_parent.mkdir(parents=True, exist_ok=False)
        candidate_root = attempt_parent / blind_candidate_id
        root_ref = content_id(
            "recclaw-innovation-candidate-root-v1",
            {
                "campaign_id": current_profile.campaign_id,
                "round": attempt_index,
                "blind_candidate_id": blind_candidate_id,
            },
        )
        try:
            fixture = inputs.fixture_factory(
                implementer_policy,
                attempt_index,
                candidate_root,
            )
            unit_check = inputs.unit_check_factory(implementer_policy)
            if inputs.resource_admission_required:
                materialized = materialize_candidate_package(
                    selected_outcome.spec,
                    policy=implementer_policy,
                    implementation_response=response,
                    candidate_root=candidate_root,
                    candidate_root_ref=root_ref,
                )
                qualification = MechanicalRecBoleAdapterV1().qualify_disposable(
                    materialized.package,
                    research_spec=selected_outcome.spec,
                    candidate_root=candidate_root,
                    fixture=fixture,
                    # The disposable adapter already runs the shared model/API
                    # unit stage. Arbitrary caller closures are intentionally
                    # not executed in the spawned process.
                    unit_check=None,
                )
            else:
                materialized, qualification = qualify_local_innovation_candidate(
                    selected_outcome.spec,
                    policy=implementer_policy,
                    implementation_response=response,
                    candidate_root=candidate_root,
                    candidate_root_ref=root_ref,
                    fixture=fixture,
                    unit_check=unit_check,
                )
            failure = qualification.failure_detail
        except InnovationSpineError as error:
            materialized = None
            qualification = None
            failure = _materialization_failure(error)
        attempts.append(
            canonical_value(
                {
                    "attempt": attempt_index,
                    "candidate_root": str(candidate_root),
                    "qualification": (
                        qualification.to_dict() if qualification is not None else None
                    ),
                    "failure": failure,
                }
            )
        )
        if (
            qualification is not None
            and qualification.receipt.status is QualificationStatusV1.PASS
        ):
            break
        if failure is None or not is_mechanical_repair_failure(failure):
            break

    if (
        materialized is None
        or qualification is None
        or qualification.receipt.status is not QualificationStatusV1.PASS
        or candidate_root is None
    ):
        return InnovationLaneResult(
            selected_outcome=selected_outcome,
            resolution=resolution,
            idea_acquisition=idea_acquisition,
            attempts=tuple(attempts),
            materialized=materialized,
            qualification=qualification,
            capability=None,
            registry=None,
            profile_manifest=None,
            next_profile=None,
            profile_receipt=None,
            qualified_execution=None,
            search_candidate=None,
            candidate_root=(str(candidate_root) if candidate_root is not None else None),
            fresh_campaign_id=inputs.fresh_campaign_id,
        )

    source_proposal = selected_outcome.source_proposal
    if source_proposal is not None:
        from recclaw_core.mechanism_space import compile_program
        from recclaw_core.mechanism_space.canonical import deep_thaw

        compile_report = compile_program(deep_thaw(source_proposal.mechanism_program))
        if (
            not compile_report.is_valid
            or compile_report.mechanism_semantics_digest is None
        ):
            raise ValueError("qualified legacy Innovation proposal lacks compiled semantics")
        semantic_identity_ref = f"bl-icf-mechanism:{source_proposal.mechanism_id}"
        semantic_identity_digest = compile_report.mechanism_semantics_digest
    else:
        if implementer_policy.execution_contract is None:
            raise ValueError("qualified OpenSpec lacks its resolved execution contract")
        semantic_identity_ref, semantic_identity_digest = (
            open_spec_realization_identity(
                selected_outcome.spec,
                candidate_package_ref=materialized.package.package_id,
                candidate_package_digest=materialized.package.digest,
                candidate_root_ref=materialized.package.candidate_root_ref,
                candidate_root_digest=materialized.package.candidate_root_digest,
                source_tree_digest=materialized.package.source_tree_digest,
                executable_entrypoint=materialized.package.executable_entrypoint,
                execution_contract=implementer_policy.execution_contract,
            )
        )
    capability, registry = admit_local_qualification(
        selected_outcome.spec,
        materialized,
        qualification,
        capability_kind=inputs.capability_kind,
        capability_version=inputs.capability_version,
        semantic_identity_ref=semantic_identity_ref,
        semantic_identity_digest=semantic_identity_digest,
        registry_version=inputs.registry_version,
        predecessor_registry_ref=inputs.predecessor_registry_ref,
        predecessor_registry_digest=inputs.predecessor_registry_digest,
    )
    qualified_execution = _qualified_execution(
        implementer_policy,
        materialized,
        capability,
    )
    resource_profile: Mapping[str, Any] | None = None
    quality_admission: Mapping[str, Any] | None = None
    if inputs.resource_probe is not None:
        assert inputs.resource_probe_parent is not None
        source_relative = (
            materialized.package.executable_entrypoint.split(":", 1)[0]
            .replace(".", "/")
            + ".py"
        )
        source_path = candidate_root / source_relative
        mechanism_id = (
            source_proposal.mechanism_id
            if source_proposal is not None
            else f"OPEN_{semantic_identity_digest[:16].upper()}"
        )
        probe_recipe = _resource_probe_recipe(
            current_profile=current_profile,
            capability=capability,
            qualified_execution=qualified_execution,
            entrypoint=materialized.package.executable_entrypoint,
            mechanism_id=mechanism_id,
            mechanism_semantics_digest=semantic_identity_digest,
        )
        resource_profile = canonical_value(
            dict(
                inputs.resource_probe(
                    candidate_root=candidate_root,
                    source_path=source_path,
                    entrypoint=materialized.package.executable_entrypoint,
                    source_sha256=str(
                        qualified_execution["entrypoint_source_sha256"]
                    ),
                    execution_recipe=probe_recipe,
                    probe_root=(
                        inputs.resource_probe_parent
                        / f"probe-{semantic_identity_digest[:16]}"
                    ),
                    candidate_ref=capability.capability_id,
                    candidate_package_digest=capability.candidate_package_digest,
                )
            )
        )
        quality_admission = admit_research_innovation_candidate(
            spec=selected_outcome.spec,
            resolution=resolution,
            qualification=qualification,
            resource_profile=resource_profile,
            structural_context=inputs.resource_structural_context,
        )
        if (
            inputs.resource_admission_required
            and quality_admission.get("admitted") is not True
        ):
            return InnovationLaneResult(
                selected_outcome=selected_outcome,
                resolution=resolution,
                idea_acquisition=idea_acquisition,
                attempts=tuple(attempts),
                materialized=materialized,
                qualification=qualification,
                capability=capability,
                registry=registry,
                profile_manifest=None,
                next_profile=None,
                profile_receipt=None,
                qualified_execution=qualified_execution,
                search_candidate=None,
                candidate_root=str(candidate_root),
                fresh_campaign_id=inputs.fresh_campaign_id,
                resource_profile=resource_profile,
                quality_admission=quality_admission,
            )
    if source_proposal is not None:
        if quality_admission is not None:
            utility = SearchUtilityFeaturesV1(
                **dict(quality_admission["utility_features"])
            )
            feature_evidence = RouterFeatureEvidenceV1(
                **dict(quality_admission["feature_evidence"])
            )
            search_candidate: SearchCandidate = replace(
                source_proposal,
                mechanism_program=deep_thaw(source_proposal.mechanism_program),
                utility_features=utility,
                feature_evidence=feature_evidence,
            )
        else:
            search_candidate = source_proposal
    else:
        assert implementer_policy.execution_contract is not None
        if quality_admission is not None:
            utility = SearchUtilityFeaturesV1(
                **dict(quality_admission["utility_features"])
            )
            feature_evidence = RouterFeatureEvidenceV1(
                **dict(quality_admission["feature_evidence"])
            )
        else:
            utility, feature_evidence = _qualified_open_features(selected_outcome)
        search_candidate = OpenSpecSearchCandidateV1(
            spec=selected_outcome.spec,
            capability_ref=capability.capability_id,
            capability_digest=capability.digest,
            candidate_package_ref=materialized.package.package_id,
            candidate_package_digest=materialized.package.digest,
            candidate_root_ref=materialized.package.candidate_root_ref,
            candidate_root_digest=materialized.package.candidate_root_digest,
            source_tree_digest=materialized.package.source_tree_digest,
            executable_entrypoint=materialized.package.executable_entrypoint,
            execution_contract=implementer_policy.execution_contract,
            semantic_identity_ref=semantic_identity_ref,
            semantic_identity_digest=semantic_identity_digest,
            qualification_receipt_ref=qualification.receipt.receipt_id,
            qualification_receipt_digest=qualification.receipt.digest,
            mechanism_axis=_open_mechanism_axis(selected_outcome),
            utility_features=utility,
            feature_evidence=feature_evidence,
        )
    manifest, next_profile, receipt = build_local_next_fresh_profile(
        registry,
        profile_version=inputs.profile_version,
        predecessor_profile_ref=current_profile.profile_ref,
        predecessor_profile_digest=current_profile.profile_digest,
        current_campaign_slate_ref=current_slate_ref,
        current_campaign_slate_digest=current_slate_digest,
        predecessor_executable_entries=predecessor_executable_entries(current_profile),
        compatibility_requirements=selected_outcome.spec.compatibility_requirements,
    )
    return InnovationLaneResult(
        selected_outcome=selected_outcome,
        resolution=resolution,
        idea_acquisition=idea_acquisition,
        attempts=tuple(attempts),
        materialized=materialized,
        qualification=qualification,
        capability=capability,
        registry=registry,
        profile_manifest=manifest,
        next_profile=next_profile,
        profile_receipt=receipt,
        qualified_execution=qualified_execution,
        search_candidate=search_candidate,
        candidate_root=str(candidate_root),
        fresh_campaign_id=inputs.fresh_campaign_id,
        resource_profile=resource_profile,
        quality_admission=quality_admission,
    )


def _run_meta_research(
    *,
    context: ResearchContext,
    interpretation: EpisodeInterpretation | MissingSearchInterpretation,
    inputs: MetaResearchInputs,
) -> MetaResearchResult:
    champion = interpretation.policy_successor
    search_memory = interpretation.search_memory_snapshot
    proposal = build_meta_update_proposal(
        policy=champion,
        search_memory=search_memory,
    )
    challenger = materialize_proposed_control_policy(
        parent=champion,
        proposal=proposal,
    )
    replay = inputs.offline_replay(
        context=context,
        champion_policy=champion,
        challenger_policy=challenger,
        search_memory=search_memory,
    )
    if not isinstance(replay, Mapping):
        raise ValueError("offline Meta replay must return a mapping")
    expected_replay_identity = {
        "source_context_digest": context.digest,
        "source_search_memory_digest": search_memory.digest,
        "champion_policy_digest": champion.digest,
        "challenger_policy_digest": challenger.digest,
    }
    if any(replay.get(key) != value for key, value in expected_replay_identity.items()):
        raise ValueError("offline Meta replay is not bound to its Context, Memory, and policies")
    champion_metrics = replay.get("champion")
    challenger_metrics = replay.get("challenger")
    if not isinstance(champion_metrics, ProposalOnlyShadowMetricsV1) or not isinstance(
        challenger_metrics, ProposalOnlyShadowMetricsV1
    ):
        raise ValueError("offline Meta replay must return the existing shadow metric type")
    paired = replay.get("same_model_prompt_schema_and_contexts") is True
    deterministic = replay.get("deterministic_directive_replay") is True
    evaluation = evaluate_proposal_only_shadow(
        proposal=proposal,
        champion_policy_digest=champion.digest,
        challenger_policy_digest=challenger.digest,
        champion=champion_metrics,
        challenger=challenger_metrics,
        same_model_prompt_schema_and_contexts=paired,
        deterministic_directive_replay=deterministic,
    )
    decision = decide_meta_control_promotion(
        proposal=proposal,
        evaluation=evaluation,
    )
    activated_policy = None
    activation_receipt = None
    if decision.verdict == "PROMOTE":
        activated_policy, activation_receipt = activate_promoted_control_policy(
            parent=champion,
            proposal=proposal,
            decision=decision,
            campaign_id=inputs.next_campaign_id,
        )
    return MetaResearchResult(
        update_proposal=proposal,
        challenger_policy=challenger,
        offline_replay={
            **expected_replay_identity,
            "champion": champion_metrics.to_dict(),
            "challenger": challenger_metrics.to_dict(),
            "same_model_prompt_schema_and_contexts": paired,
            "deterministic_directive_replay": deterministic,
        },
        shadow_evaluation=evaluation,
        promotion_decision=decision,
        activated_policy=activated_policy,
        activation_receipt=activation_receipt,
    )


def _validate_experiment_binding(
    binding: ExperimentBindingV1,
    *,
    recipe: Mapping[str, Any],
    observation_seed: str,
) -> None:
    expected_recipe_digest = sha256_digest(recipe)
    if binding.execution_recipe_digest != expected_recipe_digest:
        raise ValueError("runner Experiment Binding is not bound to the selected recipe")
    if str(binding.seed) != observation_seed:
        raise ValueError("runner Experiment Binding seed differs from the frozen seed")
    recipe_fields = (
        "capability_family",
        "capability_ref",
        "capability_digest",
        "profile_ref",
        "profile_digest",
        "mechanism_id",
        "candidate_package_ref",
        "candidate_package_digest",
        "candidate_root_ref",
        "candidate_root_digest",
        "candidate_source_tree_digest",
        "entrypoint",
        "entrypoint_source_sha256",
        "model",
        "base_model_config",
        "config",
        "dataset",
        "split",
        "evaluator",
        "execution_role",
        "comparator_ref",
        "comparator_digest",
    )
    mismatched = [
        field_name
        for field_name in recipe_fields
        if canonical_value(getattr(binding, field_name))
        != canonical_value(recipe.get(field_name))
    ]
    for field_name in ("execution_purpose", "run_id"):
        if field_name in recipe and getattr(binding, field_name) != recipe[field_name]:
            mismatched.append(field_name)
    if "seed" in recipe and int(recipe["seed"]) != binding.seed:
        mismatched.append("seed")
    if mismatched:
        raise ValueError(
            "runner Experiment Binding differs from the selected recipe for: "
            + ", ".join(mismatched)
        )


def _search_ranking_inputs(
    context: ResearchContext,
) -> tuple[
    tuple[tuple[str, str], ...],
    Mapping[str, Any] | None,
    Mapping[str, float],
]:
    memory = context.scientific_memory
    global_memory = memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    raw_observations = global_memory.get(
        "executed_observations",
        memory.get("executed_observations", ()),
    )
    pairs: list[tuple[str, str]] = []
    effects_by_axis: dict[str, list[float]] = {}
    if isinstance(raw_observations, (tuple, list)):
        for observation in raw_observations:
            if not isinstance(observation, Mapping):
                continue
            semantic_digest = observation.get("candidate_semantic_digest")
            observation_seed = observation.get("observation_seed")
            if not isinstance(semantic_digest, str) or not isinstance(
                observation_seed, str
            ):
                continue
            pair = (semantic_digest, observation_seed)
            if pair not in pairs:
                pairs.append(pair)
            axis = observation.get("mechanism_axis")
            effect = observation.get("comparator_delta")
            if isinstance(axis, str) and isinstance(effect, (int, float)):
                effects_by_axis.setdefault(axis, []).append(float(effect))
    raw_queue = global_memory.get("task_queue", memory.get("task_queue"))
    queued = ResearchTaskQueueV2.from_dict(raw_queue).select_next()
    if queued is not None:
        pending_task = {
            **queued.prompt_projection(),
            "task_id": queued.task_id,
            "candidate_id": queued.candidate_id,
            "priority": queued.priority,
            "operation": queued.operation.value,
            "evidence_present": queued.evidence_present,
        }
    else:
        latest_feedback = global_memory.get(
            "latest_feedback",
            memory.get("latest_feedback"),
        )
        pending_task = (
            latest_feedback.get("research_task_slot")
            if isinstance(latest_feedback, Mapping)
            else None
        )
    return (
        tuple(pairs),
        pending_task if isinstance(pending_task, Mapping) else None,
        {
            axis: sum(effects) / len(effects)
            for axis, effects in effects_by_axis.items()
            if effects
        },
    )


def _explicit_attempt_budget(
    budget_snapshot: Mapping[str, Any],
    explicit_limit: int | None,
    *,
    required: bool = True,
) -> int | None:
    if explicit_limit is not None:
        value = explicit_limit
    else:
        value = None
        for field_name in (
            "max_attempts_per_round",
            "round_attempt_budget",
            "remaining_attempt_budget",
            "attempt_budget",
        ):
            candidate = budget_snapshot.get(field_name)
            if candidate is not None:
                value = candidate
                break
        if value is None:
            if required:
                raise ValueError(
                    "attempt_scheduler=True requires an explicit frozen "
                    "max_attempts_per_round or round attempt budget"
                )
            return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("round attempt budget must be a non-negative integer")

    return value


def _attempt_budget_limit(
    budget_snapshot: Mapping[str, Any],
    explicit_limit: int | None,
    pool_size: int,
) -> int:
    value = _explicit_attempt_budget(
        budget_snapshot,
        explicit_limit,
        required=True,
    )
    assert value is not None
    return min(value, pool_size)


def _attempt_failure_scope(
    candidate_run: Mapping[str, Any],
    failure_detail: Mapping[str, Any] | None,
) -> str:
    """Read an explicit infrastructure scope without guessing from a status.

    Existing workers do not emit a scope, so their typed resource/runtime
    failures retain the historical candidate-local default.  A worker or
    adapter that knows the failure is shared must say so explicitly; this is
    what prevents a shared outage from quarantining the rest of the pool.
    """

    raw: Any = None
    for field_name in (
        "failure_scope",
        "engineering_scope",
        "failure_scope_class",
    ):
        if candidate_run.get(field_name) is not None:
            raw = candidate_run[field_name]
            break
    if raw is None and isinstance(candidate_run.get("resource_telemetry"), Mapping):
        telemetry = candidate_run["resource_telemetry"]
        for field_name in ("failure_scope", "engineering_scope", "scope"):
            if telemetry.get(field_name) is not None:
                raw = telemetry[field_name]
                break
    if raw is None and isinstance(failure_detail, Mapping):
        raw = failure_detail.get("failure_scope") or failure_detail.get(
            "engineering_scope"
        )
    normalized = str(raw).strip().upper() if raw is not None else ""
    if candidate_run.get("shared_infrastructure_failure") is True:
        normalized = "SHARED_INFRASTRUCTURE"
    if normalized in {
        "CANDIDATE_LOCAL",
        "LINEAGE_COMPUTE_PATTERN",
        "WORKER_TRANSIENT",
        "SHARED_INFRASTRUCTURE",
    }:
        return normalized
    return "CANDIDATE_LOCAL"


def _candidate_run_binding(
    candidate_run: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    observation_seed: str,
) -> ExperimentBindingV1:
    try:
        run_binding = ExperimentBindingV1.from_canonical_dict(
            candidate_run.get("experiment_binding")
        )
    except ExperimentBindingError as error:
        raise ValueError(str(error)) from error
    _validate_experiment_binding(
        run_binding,
        recipe=recipe,
        observation_seed=observation_seed,
    )
    if candidate_run.get("execution_recipe_digest") != sha256_digest(recipe):
        raise ValueError("runner result is not bound to the selected execution recipe")
    if str(candidate_run.get("seed")) != observation_seed:
        raise ValueError("runner result seed differs from the frozen observation seed")
    run_binding_digest = candidate_run.get("experiment_binding_digest")
    if (
        run_binding_digest != run_binding.digest
        or candidate_run.get("experiment_binding_ref") != run_binding.ref
        or candidate_run.get("binding_digest", run_binding_digest) != run_binding_digest
    ):
        raise ValueError("runner result carries an inconsistent Experiment Binding identity")
    return run_binding


def _selected_outcome_for_binding(
    binding: SearchCandidateBindingV1,
    *,
    search_pairs: Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]],
    open_search_pairs: Sequence[
        tuple[ProducerOutcome, CapabilityResolutionV1, OpenSpecSearchCandidateV1]
    ],
) -> ProducerOutcome:
    candidate_id = binding.proposal.candidate_id
    if isinstance(binding.proposal, CandidateProposalV4):
        for outcome, _resolution in search_pairs:
            if (
                outcome.source_proposal is not None
                and outcome.source_proposal.candidate_id == candidate_id
            ):
                return outcome
    else:
        for outcome, _resolution, candidate in open_search_pairs:
            if candidate.candidate_id == candidate_id:
                return outcome
    raise ValueError("routed binding has no matching Producer outcome")


def _validate_handoff_resource_profile(
    profile: Mapping[str, Any] | None,
    *,
    binding: SearchCandidateBindingV1,
    qualified_execution: Mapping[str, Any] | None,
    require_complete: bool,
    require_explicit_prediction: bool = False,
) -> None:
    if profile is None:
        if require_complete:
            raise ValueError(
                "portfolio handoff lacks the admitted resource profile for "
                + binding.proposal.candidate_id
            )
        return
    if not isinstance(profile, Mapping) or not profile:
        raise ValueError("resource profile must be a non-empty mapping")
    candidate_ref = profile.get("candidate_ref")
    if candidate_ref is not None and candidate_ref != binding.capability_ref:
        raise ValueError("resource profile candidate identity drift")
    if require_complete and candidate_ref != binding.capability_ref:
        raise ValueError("resource profile lacks the binding capability identity")
    if (
        qualified_execution is not None
        and profile.get("candidate_package_digest") is not None
        and profile.get("candidate_package_digest")
        != qualified_execution.get("candidate_package_digest")
    ):
        raise ValueError("resource profile package identity drift")
    if profile.get("effect_fields_consumed") not in (None, (), []):
        raise ValueError("resource profile contains mechanism-effect evidence")
    if profile.get("outcome_fields_consumed") not in (None, (), []):
        raise ValueError("resource profile contains outcome evidence")
    if profile.get("held_out_reads") not in (None, 0):
        raise ValueError("resource profile contains heldout evidence")
    if (
        "mechanism_effect_update_allowed" in profile
        and profile.get("mechanism_effect_update_allowed") is not False
    ):
        raise ValueError("resource profile permits mechanism-effect updates")
    if require_explicit_prediction:
        prediction = profile.get("prediction")
        prediction = prediction if isinstance(prediction, Mapping) else {}
        explicit = tuple(
            value
            for value in (
                profile.get("predicted_gpu_seconds"),
                profile.get("predicted_gpu_worker_seconds"),
                prediction.get("predicted_gpu_seconds"),
                prediction.get("predicted_gpu_worker_seconds"),
            )
            if value is not None
        )
        if not explicit:
            raise ValueError(
                "formal profile handoff requires explicit identity-bound "
                "predicted_gpu_seconds or predicted_gpu_worker_seconds"
            )
        try:
            numeric = tuple(float(value) for value in explicit)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "formal profile handoff GPU-worker prediction is not numeric"
            ) from error
        if any(value <= 0.0 for value in numeric):
            raise ValueError(
                "formal profile handoff GPU-worker prediction must be positive"
            )
        if any(value != numeric[0] for value in numeric[1:]):
            raise ValueError(
                "formal profile handoff GPU-worker prediction fields disagree"
            )


def _durable_evidence_digests(
    *,
    qualified_execution: Mapping[str, Any] | None,
    candidate_root_path: str | None,
    resource_profile: Mapping[str, Any] | None,
) -> dict[str, str]:
    if resource_profile is None:
        raise ValueError("durable evidence digest requires a resource profile")
    digests = {"resource_profile": sha256_digest(resource_profile)}
    if qualified_execution is not None:
        digests["qualified_execution"] = sha256_digest(qualified_execution)
    if candidate_root_path is not None:
        digests["candidate_root_path"] = sha256_digest(
            {"candidate_root_path": str(Path(candidate_root_path).resolve())}
        )
    return canonical_value(digests)


def _binding_parent_id(binding: SearchCandidateBindingV1) -> str | None:
    proposal = binding.proposal
    parent = getattr(proposal, "parent_candidate_id", None)
    if parent is None:
        parent = getattr(getattr(proposal, "spec", None), "parent_candidate_id", None)
    return parent


def _validate_candidate_handoff(
    handoff: RoundCandidateHandoffV1,
    *,
    binding: SearchCandidateBindingV1,
    active_profile: SearchExecutableProfileV1,
    require_complete: bool,
) -> None:
    if not isinstance(handoff, RoundCandidateHandoffV1):
        raise ValueError("candidate handoff must be RoundCandidateHandoffV1")
    if handoff.candidate_id != binding.proposal.candidate_id:
        raise ValueError("candidate handoff candidate identity drift")
    if handoff.binding_digest != binding.digest:
        raise ValueError("candidate handoff binding digest drift")
    if handoff.portfolio_candidate.semantic_digest != binding.mechanism_semantics_digest:
        raise ValueError("candidate handoff portfolio semantic identity drift")
    formal_profile = handoff.portfolio_profile is not None
    if formal_profile:
        if handoff.resource_profile is None:
            raise ValueError("formal profile handoff lacks resource evidence")
        if handoff.portfolio_profile.candidate != handoff.portfolio_candidate:
            raise ValueError("candidate handoff profile/candidate identity drift")
        expected_family = getattr(binding.proposal, "mechanism_axis", None)
        if (
            expected_family is not None
            and handoff.portfolio_candidate.family_id != expected_family
        ):
            raise ValueError("candidate handoff family identity drift")
        expected_parent = _binding_parent_id(binding)
        if handoff.portfolio_candidate.parent_id != expected_parent:
            raise ValueError("candidate handoff parent identity drift")
        if handoff.lineage_identity_digest != lineage_identity_digest(
            handoff.portfolio_candidate
        ):
            raise ValueError("candidate handoff lineage identity drift")
        if handoff.durable_evidence_digests is not None:
            if handoff.durable_evidence_digests != _durable_evidence_digests(
                qualified_execution=handoff.qualified_execution,
                candidate_root_path=handoff.candidate_root_path,
                resource_profile=handoff.resource_profile,
            ):
                raise ValueError("candidate handoff durable evidence digest drift")
    if binding.entry_origin is SearchProfileEntryOriginV1.FIXED_66:
        if handoff.qualified_execution is not None:
            raise ValueError("FIXED_66 handoff cannot carry qualified execution")
        if handoff.candidate_root_path is not None:
            raise ValueError("FIXED_66 handoff cannot carry a candidate root")
        qualified_execution = None
    elif binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
        if handoff.qualified_execution is None:
            raise ValueError("qualified handoff lacks explicit execution inputs")
        try:
            execution_recipe_for_search_binding(
                binding,
                profile=active_profile,
                qualified_execution=handoff.qualified_execution,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "qualified handoff execution identity is invalid"
            ) from error
        if handoff.candidate_root_path is None:
            raise ValueError("qualified handoff lacks a candidate root")
        root = Path(handoff.candidate_root_path)
        if not root.is_dir() or not (root / "recclaw_ext").is_dir():
            raise ValueError(
                "qualified handoff candidate root is unavailable: " + str(root)
            )
        qualified_execution = handoff.qualified_execution
    else:
        raise ValueError("candidate handoff has an unknown Search profile origin")
    _validate_handoff_resource_profile(
        handoff.resource_profile,
        binding=binding,
        qualified_execution=qualified_execution,
        require_complete=require_complete,
        require_explicit_prediction=formal_profile,
    )


def _validate_candidate_handoffs(
    handoffs: Sequence[RoundCandidateHandoffV1],
    *,
    bindings: Sequence[SearchCandidateBindingV1],
    active_profile: SearchExecutableProfileV1,
    require_complete: bool,
) -> tuple[RoundCandidateHandoffV1, ...]:
    normalized = tuple(handoffs)
    by_candidate: dict[str, RoundCandidateHandoffV1] = {}
    for handoff in normalized:
        if not isinstance(handoff, RoundCandidateHandoffV1):
            raise ValueError("candidate handoff factory returned an invalid record")
        if handoff.candidate_id in by_candidate:
            raise ValueError("candidate handoff candidate identity is duplicated")
        by_candidate[handoff.candidate_id] = handoff
    expected_ids = tuple(binding.proposal.candidate_id for binding in bindings)
    expected_set = set(expected_ids)
    actual_set = set(by_candidate)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(
            "portfolio handoff coverage is not exact (" + "; ".join(detail) + ")"
        )
    ordered: list[RoundCandidateHandoffV1] = []
    for binding in bindings:
        handoff = by_candidate[binding.proposal.candidate_id]
        _validate_candidate_handoff(
            handoff,
            binding=binding,
            active_profile=active_profile,
            require_complete=require_complete,
        )
        ordered.append(handoff)
    return tuple(ordered)


def _static_candidate_handoffs(
    portfolio_candidates: Sequence[PortfolioCandidateV2],
    *,
    bindings: Sequence[SearchCandidateBindingV1],
    active_profile: SearchExecutableProfileV1,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]],
    candidate_root_by_capability: Mapping[str, str | Path],
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]],
) -> tuple[RoundCandidateHandoffV1, ...]:
    by_candidate: dict[str, PortfolioCandidateV2] = {}
    for candidate in portfolio_candidates:
        if candidate.candidate_id in by_candidate:
            raise ValueError("portfolio candidate identity is duplicated")
        by_candidate[candidate.candidate_id] = candidate
    expected_ids = {binding.proposal.candidate_id for binding in bindings}
    if set(by_candidate) != expected_ids:
        missing = sorted(expected_ids - set(by_candidate))
        extra = sorted(set(by_candidate) - expected_ids)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(
            "portfolio candidate coverage is not exact ("
            + "; ".join(detail)
            + ")"
        )
    handoffs = tuple(
        RoundCandidateHandoffV1(
            candidate_id=binding.proposal.candidate_id,
            binding_digest=binding.digest,
            portfolio_candidate=by_candidate[binding.proposal.candidate_id],
            qualified_execution=(
                qualified_execution_by_capability.get(binding.capability_ref)
                if binding.entry_origin
                is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
                else None
            ),
            candidate_root_path=(
                candidate_root_by_capability.get(binding.capability_ref)
                if binding.entry_origin
                is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
                else None
            ),
            resource_profile=resource_profile_by_capability.get(
                binding.capability_ref
            ),
        )
        for binding in bindings
        if binding.proposal.candidate_id in by_candidate
    )
    return _validate_candidate_handoffs(
        handoffs,
        bindings=bindings,
        active_profile=active_profile,
        require_complete=False,
    )


def _candidate_handoffs_for_round(
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    resolutions: Sequence[
        tuple[ProducerOutcome, CapabilityResolutionV1 | None]
    ],
    search_bindings: tuple[SearchCandidateBindingV1, ...],
    portfolio_candidates: tuple[PortfolioCandidateV2, ...],
    research_profile_source: ResearchProfileSourceV1 | None,
    candidate_handoff_factory: CandidateHandoffFactory | None,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None,
    candidate_root_by_capability: Mapping[str, str | Path] | None,
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[RoundCandidateHandoffV1, ...]:
    qualified = dict(qualified_execution_by_capability or {})
    roots = dict(candidate_root_by_capability or {})
    resources = dict(resource_profile_by_capability or {})
    if research_profile_source is not None:
        if not isinstance(research_profile_source, ResearchProfileSourceV1):
            raise ValueError(
                "research_profile_source must be ResearchProfileSourceV1"
            )
        if candidate_handoff_factory is not None:
            raise ValueError(
                "research_profile_source and candidate_handoff_factory are mutually exclusive"
            )
        records = research_profile_source.build_profiles(
            context=context,
            active_profile=active_profile,
            resolutions=tuple(resolutions),
            search_bindings=search_bindings,
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resources,
        )
        if len(records) != len(search_bindings):
            raise ValueError("research profile source did not cover every binding")
        handoffs = tuple(
            RoundCandidateHandoffV1(
                candidate_id=binding.proposal.candidate_id,
                binding_digest=binding.digest,
                portfolio_candidate=record.portfolio_profile.candidate,
                qualified_execution=record.qualified_execution,
                candidate_root_path=record.candidate_root_path,
                resource_profile=record.resource_profile,
                portfolio_profile=record.portfolio_profile,
                profile_digest=record.profile_digest,
                source_schema_version=research_profile_source.schema_version,
                source_ref=research_profile_source.source_ref,
                source_digest=research_profile_source.source_digest,
                lineage_identity_digest=record.lineage_identity_digest,
                durable_evidence_digests=record.durable_evidence_digests,
            )
            for binding, record in zip(search_bindings, records, strict=True)
        )
        handoffs = _validate_candidate_handoffs(
            handoffs,
            bindings=search_bindings,
            active_profile=active_profile,
            require_complete=True,
        )
        if portfolio_candidates:
            expected = {
                item.candidate_id: item.to_dict() for item in portfolio_candidates
            }
            actual = {
                item.candidate_id: item.portfolio_candidate.to_dict()
                for item in handoffs
            }
            if actual != expected:
                raise ValueError(
                    "static portfolio candidates disagree with profile source"
                )
        return handoffs
    if candidate_handoff_factory is not None:
        if not callable(candidate_handoff_factory):
            raise ValueError("candidate_handoff_factory must be callable")
        produced = candidate_handoff_factory(
            context=context,
            active_profile=active_profile,
            resolutions=tuple(resolutions),
            search_bindings=search_bindings,
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resources,
        )
        if isinstance(produced, Mapping):
            produced = tuple(produced.values())
        if isinstance(produced, (str, bytes)) or not isinstance(produced, Sequence):
            raise ValueError(
                "candidate_handoff_factory must return a sequence of handoffs"
            )
        handoffs = _validate_candidate_handoffs(
            tuple(produced),
            bindings=search_bindings,
            active_profile=active_profile,
            require_complete=True,
        )
        for binding, handoff in zip(search_bindings, handoffs, strict=True):
            if binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
                expected_execution = qualified.get(binding.capability_ref)
                if (
                    expected_execution is not None
                    and handoff.qualified_execution != expected_execution
                ):
                    raise ValueError("candidate handoff execution state drift")
                expected_root = roots.get(binding.capability_ref)
                if expected_root is not None and (
                    handoff.candidate_root_path is None
                    or Path(handoff.candidate_root_path).resolve()
                    != Path(expected_root).resolve()
                ):
                    raise ValueError("candidate handoff root state drift")
            expected_resource = resources.get(binding.capability_ref)
            if expected_resource is not None and (
                handoff.resource_profile is None
                or canonical_value(dict(handoff.resource_profile))
                != canonical_value(dict(expected_resource))
            ):
                raise ValueError("candidate handoff resource state drift")
        if portfolio_candidates:
            expected = {
                item.candidate_id: item.to_dict() for item in portfolio_candidates
            }
            actual = {
                item.candidate_id: item.portfolio_candidate.to_dict()
                for item in handoffs
            }
            if actual != expected:
                raise ValueError(
                    "static portfolio candidates disagree with dynamic handoff"
                )
        return handoffs
    if not portfolio_candidates:
        return ()
    return _static_candidate_handoffs(
        portfolio_candidates,
        bindings=search_bindings,
        active_profile=active_profile,
        qualified_execution_by_capability=qualified,
        candidate_root_by_capability=roots,
        resource_profile_by_capability=resources,
    )


def _execution_recipe_for_handoff(
    binding: SearchCandidateBindingV1,
    *,
    profile: SearchExecutableProfileV1,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None,
    candidate_handoffs: Sequence[RoundCandidateHandoffV1],
) -> dict[str, Any]:
    by_candidate = {item.candidate_id: item for item in candidate_handoffs}
    handoff = by_candidate.get(binding.proposal.candidate_id)
    if candidate_handoffs and handoff is None:
        raise ValueError("selected binding has no candidate handoff")
    qualified_execution = (
        handoff.qualified_execution
        if handoff is not None
        and binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
        else (
            (qualified_execution_by_capability or {}).get(binding.capability_ref)
            if binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
            else None
        )
    )
    recipe = execution_recipe_for_search_binding(
        binding,
        profile=profile,
        qualified_execution=qualified_execution,
    )
    if handoff is not None:
        if handoff.resource_profile is not None:
            recipe["resource_prediction"] = handoff.resource_profile
        if handoff.candidate_root_path is not None:
            recipe["candidate_root_path"] = handoff.candidate_root_path
        recipe = canonical_value(recipe)
    return recipe


def _mechanism_projection_for_binding(
    binding: SearchCandidateBindingV1,
    selected_outcome: ProducerOutcome,
) -> Mapping[str, Any]:
    if isinstance(binding.proposal, CandidateProposalV4):
        return binding.proposal.mechanism_program
    return {
        "schema": "recclaw-open-spec-realization-semantics-e0.v1",
        "semantic_identity_ref": binding.proposal.semantic_identity_ref,
        "semantic_identity_digest": binding.proposal.semantic_identity_digest,
        "research_spec_ref": binding.proposal.spec.spec_id,
        "research_spec_digest": binding.proposal.spec.digest,
        "candidate_package_ref": binding.proposal.candidate_package_ref,
        "candidate_package_digest": binding.proposal.candidate_package_digest,
        "execution_contract": binding.proposal.execution_contract,
        "parent_candidate_id": (
            selected_outcome.source_proposal.parent_candidate_id
            if selected_outcome.source_proposal is not None
            else None
        ),
    }


def _route_metadata_for_attempt(
    *,
    acquisition: ExperimentAcquisitionResultV1,
    selected_outcome: ProducerOutcome,
    binding: SearchCandidateBindingV1,
    candidate_handoff: RoundCandidateHandoffV1 | None = None,
    next_discriminative_test: str,
    observation_seed: str,
    failure_scope: str | None = None,
    attempt_index: int | None = None,
    pending_task: Mapping[str, Any] | None = None,
    confirmation_seed: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "route_trace_digest": acquisition.route_trace.digest,
        "selected_producer_role": selected_outcome.producer_role,
        "selected_candidate_id": binding.proposal.candidate_id,
        "selected_candidate_semantic_digest": binding.mechanism_semantics_digest,
        "selected_mechanism_axis": binding.proposal.mechanism_axis,
        "required_selected_runnable_probability": (
            binding.proposal.utility_features.runnable_probability
        ),
        "mechanism_program": _mechanism_projection_for_binding(
            binding, selected_outcome
        ),
        "comparator_identity": None,
        "required_seed_or_control": observation_seed,
        "next_discriminative_test": next_discriminative_test,
    }
    if candidate_handoff is not None:
        portfolio_candidate = candidate_handoff.portfolio_candidate
        metadata.update(
            {
                "selected_candidate_family_id": portfolio_candidate.family_id,
                "selected_candidate_parent_id": portfolio_candidate.parent_id,
                "selected_compute_pattern": portfolio_candidate.compute_pattern,
                "selected_resource_admission_state": (
                    portfolio_candidate.resource_admission_state.value
                ),
            }
        )
        resource_profile = candidate_handoff.resource_profile
        profile = candidate_handoff.portfolio_profile
        durable_digests = candidate_handoff.durable_evidence_digests
        resource_digest = (
            durable_digests.get("resource_profile")
            if isinstance(durable_digests, Mapping)
            else None
        )
        if resource_digest is None and resource_profile is not None:
            resource_digest = sha256_digest(resource_profile)
        if profile is not None:
            metadata["selected_portfolio_profile_digest"] = profile.profile_digest
        if resource_digest is not None:
            metadata["selected_resource_evidence_digest"] = resource_digest
    if failure_scope is not None:
        metadata["failure_scope"] = failure_scope
    if attempt_index is not None:
        metadata["round_attempt_index"] = attempt_index
    if confirmation_seed is not None:
        metadata["confirmation_seed"] = confirmation_seed
    if isinstance(pending_task, Mapping):
        metadata["active_task_id"] = pending_task.get("task_id")
        task_type = pending_task.get("task_type")
        task_type = getattr(task_type, "value", task_type)
        same_candidate = (
            pending_task.get("candidate_semantic_digest")
            == binding.mechanism_semantics_digest
            and (
                pending_task.get("candidate_id") is None
                or pending_task.get("candidate_id")
                == binding.proposal.candidate_id
            )
        )
        required = pending_task.get("required_seed_or_control")
        evidence_present = pending_task.get("evidence_present", ())
        if isinstance(evidence_present, str):
            evidence_present = (evidence_present,)
        generic_new_seed = (
            required == _NEXT_DEVELOPMENT_SEED
            and isinstance(evidence_present, (tuple, list))
            and observation_seed not in evidence_present
        )
        if same_candidate and (
            (
                task_type == "VALIDATE_SAME_CANDIDATE"
                and (required == observation_seed or generic_new_seed)
            )
            or task_type == "REPAIR_IMPLEMENTATION"
            or (
                task_type in {"RUN_MATCHED_CONTROL", "RUN_ABLATION"}
                and required == observation_seed
            )
        ):
            metadata["satisfies_task_id"] = pending_task.get("task_id")
    return metadata


def _run_scheduled_attempts(
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    producer_outcomes: tuple[ProducerOutcome, ...],
    carryover_outcomes: tuple[ProducerOutcome, ...],
    resolutions: tuple[tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...],
    deferred_innovation: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...],
    deferred_search: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...],
    search_acquisition: ExperimentAcquisitionResultV1,
    search_bindings: tuple[SearchCandidateBindingV1, ...],
    search_pairs: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...],
    open_search_pairs: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1, OpenSpecSearchCandidateV1]
    ],
    innovation: InnovationLaneResult | None,
    budget_snapshot: Mapping[str, Any],
    router: StrongStaticRouterV1,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    runner: ExperimentRunner,
    incumbent_observation: Mapping[str, Any],
    metric_contract_digest: str,
    observation_seed: str,
    next_discriminative_test: str,
    confirmation_seed: str | None,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None,
    meta_research_inputs: MetaResearchInputs | None,
    provider_traces: tuple[Mapping[str, Any], ...],
    prepared_round: PreparedResearchRoundV1,
    max_attempts_per_round: int | None,
    recovered_attempts: Sequence[Mapping[str, Any]],
    executed_semantic_seed_pairs: tuple[tuple[str, str], ...],
    pending_task: Mapping[str, Any] | None,
    mechanism_axis_effects: Mapping[str, float],
    portfolio_candidates: tuple[PortfolioCandidateV2, ...],
    candidate_handoffs: tuple[RoundCandidateHandoffV1, ...],
) -> ResearchRoundResult:
    """Run a frozen candidate pool until one metric seals or the pool stops.

    ``recovered_attempts`` are replayed from immutable physical observations
    before new routing begins.  Their candidate identities consume the same
    round budget and are removed from the remaining pool, so resuming an
    incomplete round cannot invoke the physical runner twice for one attempt.
    """

    outcome_by_candidate: dict[str, ProducerOutcome] = {}
    for outcome, _resolution in search_pairs:
        if outcome.source_proposal is not None:
            outcome_by_candidate[outcome.source_proposal.candidate_id] = outcome
    for outcome, _resolution, candidate in open_search_pairs:
        outcome_by_candidate[candidate.candidate_id] = outcome
    binding_by_candidate = {
        binding.proposal.candidate_id: binding for binding in search_bindings
    }
    handoff_by_candidate = {
        item.candidate_id: item for item in candidate_handoffs
    }
    prepared_slate = prepared_round.search_slate
    if prepared_slate is not None:
        if tuple(prepared_slate.bindings) != tuple(search_bindings):
            raise ValueError("prepared search slate candidate identity drift")
        if (
            search_acquisition is None
            or search_acquisition.slate_digest != prepared_slate.digest
        ):
            raise ValueError("prepared search slate acquisition identity drift")

    limit = _attempt_budget_limit(
        budget_snapshot,
        max_attempts_per_round,
        len(search_bindings),
    )
    recovered: tuple[Mapping[str, Any], ...] = tuple(recovered_attempts)
    if len(recovered) > limit:
        raise ValueError("recovered attempt manifest exceeds the frozen round budget")
    seen_recovered_candidates: set[str] = set()
    for index, row in enumerate(recovered):
        if not isinstance(row, Mapping):
            raise ValueError("recovered attempt manifest row is invalid")
        raw_index = row.get("attempt_index")
        if isinstance(raw_index, bool) or raw_index != index:
            raise ValueError("recovered attempt manifest indices are not contiguous")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("recovered attempt lacks candidate identity")
        if candidate_id != "__LEGACY_SINGLE_ATTEMPT__":
            if candidate_id not in binding_by_candidate:
                raise ValueError("recovered attempt candidate is outside the frozen pool")
            if candidate_id in seen_recovered_candidates:
                raise ValueError("recovered attempt candidate identity is duplicated")
            seen_recovered_candidates.add(candidate_id)
        if not isinstance(row.get("candidate_run"), Mapping):
            raise ValueError("recovered attempt lacks its immutable candidate_run")

    attempts: list[RoundAttemptV1] = []
    attempted_candidate_ids: set[str] = set()
    working_context = context
    working_policy = policy
    acquisition = search_acquisition
    incomplete_reason = (
        "ROUND_ATTEMPT_BUDGET_EXHAUSTED"
        if limit == 0
        else "ROUND_ATTEMPT_POOL_EXHAUSTED"
    )

    def route_remaining(
        remaining_bindings: tuple[SearchCandidateBindingV1, ...],
    ) -> ExperimentAcquisitionResultV1:
        if not attempts:
            return search_acquisition
        (
            current_executed_pairs,
            current_pending_task,
            current_axis_effects,
        ) = _search_ranking_inputs(working_context)
        slate = freeze_experiment_slate(
            profile=active_profile,
            bindings=remaining_bindings,
            budget_snapshot=budget_snapshot,
        )
        remaining_candidate_ids = {
            binding.proposal.candidate_id for binding in remaining_bindings
        }
        routed_portfolio = (
            tuple(item.portfolio_candidate for item in candidate_handoffs)
            if candidate_handoffs
            else portfolio_candidates
        )
        remaining_portfolio = tuple(
            item
            for item in routed_portfolio
            if item.candidate_id in remaining_candidate_ids
        )
        return route_frozen_experiment_slate(
            profile=active_profile,
            slate=slate,
            router=router,
            policy_projection=working_policy.to_dict(),
            executed_semantic_seed_pairs=tuple(
                (
                    *current_executed_pairs,
                    *(
                        (item.binding.mechanism_semantics_digest, observation_seed)
                        for item in attempts
                    ),
                )
            ),
            current_observation_seed=observation_seed,
            pending_task=current_pending_task,
            mechanism_axis_effects=current_axis_effects,
            portfolio_candidates=(
                remaining_portfolio if routed_portfolio else None
            ),
            attempt_failures=(
                tuple(
                    PortfolioAttemptFailureV2(
                        candidate_id=item.candidate_id,
                        scope=item.failure_scope or "CANDIDATE_LOCAL",
                        reason=(
                            str(item.failure_detail.get("reason_code"))
                            if isinstance(item.failure_detail, Mapping)
                            and item.failure_detail.get("reason_code") is not None
                            else item.engineering_disposition
                        ),
                        compute_pattern=(
                            handoff_by_candidate[item.candidate_id]
                            .portfolio_candidate.compute_pattern
                            if item.candidate_id in handoff_by_candidate
                            else None
                        ),
                    )
                    for item in attempts
                    if not item.metric_bearing
                )
                if routed_portfolio
                else ()
            ),
        )

    def materialize_attempt(
        *,
        attempt_acquisition: ExperimentAcquisitionResultV1,
        binding: SearchCandidateBindingV1,
        recipe: Mapping[str, Any],
        candidate_run: Mapping[str, Any],
        manifest_row: Mapping[str, Any] | None = None,
    ) -> tuple[
        RoundAttemptV1,
        ProducerOutcome,
        Any,
        Any,
        Any,
        Any,
        dict[str, Any],
    ]:
        selected_outcome = _selected_outcome_for_binding(
            binding,
            search_pairs=search_pairs,
            open_search_pairs=open_search_pairs,
        )
        event, identity, episode, closure, failure_detail = (
            project_common_execution_feedback(
                context=context,
                selected_outcome=selected_outcome,
                binding=binding,
                candidate_run=candidate_run,
                incumbent_observation=incumbent_observation,
                metric_contract_digest=metric_contract_digest,
                observation_seed=observation_seed,
                next_discriminative_test=next_discriminative_test,
            )
        )
        if episode is not None:
            disposition = "METRIC_BEARING"
            failure_scope = None
        else:
            failure_scope = _attempt_failure_scope(candidate_run, failure_detail)
            disposition = "ENGINEERING_FAILURE"
        observation_ref = candidate_run.get("physical_observation_ref")
        observation_digest = candidate_run.get("physical_observation_digest")
        if manifest_row is not None:
            observation_ref = observation_ref or manifest_row.get("observation_ref")
            observation_digest = observation_digest or manifest_row.get(
                "observation_digest"
            )
            stored_recipe_digest = manifest_row.get("execution_recipe_digest")
            if (
                stored_recipe_digest is not None
                and stored_recipe_digest != sha256_digest(recipe)
            ):
                raise ValueError("recovered attempt recipe identity drift")
            stored_binding_digest = manifest_row.get("binding_digest")
            if (
                stored_binding_digest is not None
                and stored_binding_digest != binding.digest
            ):
                raise ValueError("recovered attempt binding identity drift")
            stored_disposition = manifest_row.get("engineering_disposition")
            if stored_disposition is not None and stored_disposition != disposition:
                raise ValueError("recovered attempt disposition drift")
            stored_scope = manifest_row.get("failure_scope")
            if stored_scope is not None and stored_scope != failure_scope:
                raise ValueError("recovered attempt failure scope drift")
        attempt = RoundAttemptV1(
            attempt_index=len(attempts),
            acquisition=attempt_acquisition,
            binding=binding,
            execution_recipe=recipe,
            candidate_run=candidate_run,
            engineering_disposition=disposition,
            failure_scope=failure_scope,
            failure_detail=failure_detail,
            observation_ref=(str(observation_ref) if observation_ref is not None else None),
            observation_digest=(
                str(observation_digest) if observation_digest is not None else None
            ),
            resource_prediction=(
                candidate_run.get("resource_prediction")
                if isinstance(candidate_run.get("resource_prediction"), Mapping)
                else None
            ),
            assigned_deadline_seconds=(
                float(candidate_run["assigned_deadline_seconds"])
                if isinstance(candidate_run.get("assigned_deadline_seconds"), (int, float))
                and not isinstance(candidate_run.get("assigned_deadline_seconds"), bool)
                else None
            ),
            live_health_decisions=tuple(
                item
                for item in candidate_run.get("live_health_decisions", ())
                if isinstance(item, Mapping)
            ),
        )
        route_metadata = _route_metadata_for_attempt(
            acquisition=attempt_acquisition,
            selected_outcome=selected_outcome,
            binding=binding,
            candidate_handoff=handoff_by_candidate.get(binding.proposal.candidate_id),
            next_discriminative_test=next_discriminative_test,
            observation_seed=observation_seed,
            failure_scope=failure_scope,
            attempt_index=attempt.attempt_index,
            pending_task=_search_ranking_inputs(working_context)[1],
            confirmation_seed=confirmation_seed,
        )
        route_metadata["comparator_identity"] = identity.comparator_ref
        return (
            attempt,
            selected_outcome,
            event,
            identity,
            episode,
            closure,
            route_metadata,
        )

    def record_diagnostic(
        *,
        attempt: RoundAttemptV1,
        selected_outcome: ProducerOutcome,
        event: Any,
        identity: Any,
        closure: Any,
        route_metadata: Mapping[str, Any],
    ) -> None:
        nonlocal working_context, working_policy
        diagnostic = interpret_scientific_diagnostic(
            closure=closure,
            comparison_identity=identity,
            context=working_context,
            producer_outcomes=producer_outcomes,
            route_metadata=route_metadata,
            evaluator_projection=event,
            policy=working_policy,
            memory_writer=memory_writer,
            selected_outcome=selected_outcome,
            frozen_context=context,
            advance_round=False,
        )
        working_context = diagnostic.successor_context
        working_policy = diagnostic.policy_successor
        attempts[-1] = replace(
            attempt,
            diagnostic_feedback=diagnostic.feedback_projection.to_dict(),
            diagnostic_successor_context=diagnostic.successor_context.to_dict(),
            diagnostic_policy_successor=diagnostic.policy_successor.to_dict(),
            diagnostic_search_memory_snapshot=(
                diagnostic.search_memory_snapshot.to_dict()
            ),
        )

    def metric_result(
        *,
        attempt: RoundAttemptV1,
        selected_outcome: ProducerOutcome,
        event: Any,
        identity: Any,
        episode: Any,
        closure: Any,
        route_metadata: Mapping[str, Any],
    ) -> ResearchRoundResult:
        interpretation = interpret_typed_research_episode(
            episode=episode,
            comparison_identity=identity,
            closure=closure,
            context=working_context,
            producer_outcomes=producer_outcomes,
            route_metadata=route_metadata,
            evaluator_projection=event,
            policy=working_policy,
            memory_writer=memory_writer,
            selected_outcome=selected_outcome,
            frozen_context=context,
        )
        meta_research = (
            _run_meta_research(
                context=working_context,
                interpretation=interpretation,
                inputs=meta_research_inputs,
            )
            if meta_research_inputs is not None
            else None
        )
        return ResearchRoundResult(
            context=context,
            active_profile=active_profile,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation_outcomes=deferred_innovation,
            deferred_search_outcomes=deferred_search,
            search_acquisition=attempt.acquisition,
            innovation=innovation,
            selected_outcome=selected_outcome,
            execution_recipe=attempt.execution_recipe,
            candidate_run=attempt.candidate_run,
            interpretation=interpretation,
            provider_traces=provider_traces,
            meta_research=meta_research,
            attempts=tuple(attempts),
            metric_bearing_attempt_index=attempt.attempt_index,
            attempt_scheduler_enabled=True,
            prepared=prepared_round,
        )

    for manifest_row in recovered:
        remaining_bindings = tuple(
            binding
            for binding in search_bindings
            if binding.proposal.candidate_id not in attempted_candidate_ids
        )
        if not remaining_bindings:
            raise ValueError("recovered attempt has no remaining frozen binding")
        attempt_acquisition = route_remaining(remaining_bindings)
        recovered_candidate_id = str(manifest_row["candidate_id"])
        if recovered_candidate_id == "__LEGACY_SINGLE_ATTEMPT__":
            if attempts or attempt_acquisition.selected_binding is None:
                raise ValueError("legacy physical observation is not the initial attempt")
            binding = attempt_acquisition.selected_binding
        else:
            binding = binding_by_candidate[recovered_candidate_id]
            selected_binding = attempt_acquisition.selected_binding
            if (
                selected_binding is None
                or selected_binding.proposal.candidate_id != recovered_candidate_id
            ):
                raise ValueError("recovered attempt route identity drift")
        recipe = _execution_recipe_for_handoff(
            binding,
            profile=active_profile,
            qualified_execution_by_capability=qualified_execution_by_capability,
            candidate_handoffs=candidate_handoffs,
        )
        candidate_run = canonical_value(dict(manifest_row["candidate_run"]))
        _candidate_run_binding(
            candidate_run,
            recipe=recipe,
            observation_seed=observation_seed,
        )
        (
            attempt,
            selected_outcome,
            event,
            identity,
            episode,
            closure,
            route_metadata,
        ) = materialize_attempt(
            attempt_acquisition=attempt_acquisition,
            binding=binding,
            recipe=recipe,
            candidate_run=candidate_run,
            manifest_row=manifest_row,
        )
        attempts.append(attempt)
        attempted_candidate_ids.add(binding.proposal.candidate_id)
        if episode is not None:
            return metric_result(
                attempt=attempt,
                selected_outcome=selected_outcome,
                event=event,
                identity=identity,
                episode=episode,
                closure=closure,
                route_metadata=route_metadata,
            )
        record_diagnostic(
            attempt=attempt,
            selected_outcome=selected_outcome,
            event=event,
            identity=identity,
            closure=closure,
            route_metadata=route_metadata,
        )

    while len(attempts) < limit:
        remaining_bindings = tuple(
            binding
            for binding in search_bindings
            if binding.proposal.candidate_id not in attempted_candidate_ids
        )
        if not remaining_bindings:
            incomplete_reason = (
                "ROUND_ATTEMPT_POOL_EXHAUSTED_AFTER_ENGINEERING_FAILURE"
            )
            break
        acquisition = route_remaining(remaining_bindings)
        binding = acquisition.selected_binding
        if binding is None:
            incomplete_reason = "ROUND_ATTEMPT_NO_ELIGIBLE_REMAINING_BINDING"
            break
        recipe = _execution_recipe_for_handoff(
            binding,
            profile=active_profile,
            qualified_execution_by_capability=qualified_execution_by_capability,
            candidate_handoffs=candidate_handoffs,
        )
        candidate_run = canonical_value(dict(runner(recipe, binding)))
        _candidate_run_binding(
            candidate_run,
            recipe=recipe,
            observation_seed=observation_seed,
        )
        (
            attempt,
            selected_outcome,
            event,
            identity,
            episode,
            closure,
            route_metadata,
        ) = materialize_attempt(
            attempt_acquisition=acquisition,
            binding=binding,
            recipe=recipe,
            candidate_run=candidate_run,
        )
        attempts.append(attempt)
        attempted_candidate_ids.add(binding.proposal.candidate_id)
        if episode is not None:
            return metric_result(
                attempt=attempt,
                selected_outcome=selected_outcome,
                event=event,
                identity=identity,
                episode=episode,
                closure=closure,
                route_metadata=route_metadata,
            )
        record_diagnostic(
            attempt=attempt,
            selected_outcome=selected_outcome,
            event=event,
            identity=identity,
            closure=closure,
            route_metadata=route_metadata,
        )
        if attempt.failure_scope in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}:
            incomplete_reason = "ROUND_ATTEMPT_RETRYABLE_ENGINEERING_FAILURE"
            continue
        incomplete_reason = (
            "ROUND_ATTEMPT_SHARED_INFRASTRUCTURE_STOP"
            if attempt.failure_scope == "SHARED_INFRASTRUCTURE"
            else "ROUND_ATTEMPT_WORKER_TRANSIENT_STOP"
        )
        break

    if attempts and len(attempts) >= limit and not attempts[-1].metric_bearing:
        incomplete_reason = "ROUND_ATTEMPT_BUDGET_EXHAUSTED"
    last_attempt = attempts[-1] if attempts else None
    return ResearchRoundResult(
        context=context,
        active_profile=active_profile,
        producer_outcomes=producer_outcomes,
        carryover_outcomes=carryover_outcomes,
        resolutions=resolutions,
        deferred_innovation_outcomes=deferred_innovation,
        deferred_search_outcomes=deferred_search,
        search_acquisition=(last_attempt.acquisition if last_attempt is not None else acquisition),
        innovation=innovation,
        selected_outcome=(
            outcome_by_candidate.get(last_attempt.candidate_id)
            if last_attempt is not None
            else None
        ),
        execution_recipe=(last_attempt.execution_recipe if last_attempt is not None else None),
        candidate_run=(last_attempt.candidate_run if last_attempt is not None else None),
        interpretation=None,
        provider_traces=provider_traces,
        meta_research=None,
        attempts=tuple(attempts),
        metric_bearing_attempt_index=None,
        attempt_scheduler_enabled=True,
        incomplete_reason=incomplete_reason,
        prepared=prepared_round,
    )


def run_research_round(
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    producer: ResearchProducer,
    producer_bindings: Mapping[str, Any],
    resolver_environment: Mapping[str, Any],
    carryover_proposals: Sequence[CandidateProposalV4],
    carryover_open_candidates: Sequence[OpenSpecSearchCandidateV1] = (),
    budget_snapshot: Mapping[str, Any],
    router: StrongStaticRouterV1,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    runner: ExperimentRunner,
    incumbent_observation: Mapping[str, Any],
    metric_contract_digest: str,
    observation_seed: str,
    next_discriminative_test: str,
    confirmation_seed: str | None = None,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None = None,
    innovation_inputs: InnovationRuntimeInputs | None = None,
    meta_research_inputs: MetaResearchInputs | None = None,
    attempt_scheduler: bool = False,
    max_attempts_per_round: int | None = None,
    recovered_attempts: Sequence[Mapping[str, Any]] = (),
    prepared_round: PreparedResearchRoundV1 | None = None,
    on_prepared: Callable[[PreparedResearchRoundV1], None] | None = None,
    portfolio_candidates: Sequence[PortfolioCandidateV2] | None = None,
    research_profile_source: ResearchProfileSourceV1 | None = None,
    candidate_handoff_factory: CandidateHandoffFactory | None = None,
    candidate_root_by_capability: Mapping[str, str | Path] | None = None,
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None = None,
) -> ResearchRoundResult:
    """Run one four-Producer, dual-lane, one-experiment Research Line round."""

    if (
        active_profile.campaign_id != context.campaign_id
        or active_profile.profile_ref != context.active_profile_ref
        or active_profile.profile_digest != context.active_profile_digest
    ):
        raise ValueError("active Search profile is not the Research Context profile")
    if attempt_scheduler:
        _explicit_attempt_budget(
            budget_snapshot,
            max_attempts_per_round,
            required=True,
        )
    normalized_portfolio = tuple(portfolio_candidates or ())
    if any(
        not isinstance(item, PortfolioCandidateV2)
        for item in normalized_portfolio
    ):
        raise ValueError(
            "portfolio_candidates must contain PortfolioCandidateV2 records"
        )
    if prepared_round is not None:
        if not attempt_scheduler:
            raise ValueError("prepared round resume requires attempt_scheduler=True")
        if on_prepared is not None:
            raise ValueError("prepared round resume cannot prepare the round again")
        if (
            prepared_round.context_digest != context.digest
            or prepared_round.profile_ref != active_profile.profile_ref
            or prepared_round.profile_digest != active_profile.profile_digest
        ):
            raise ValueError("prepared round context/profile identity drift")
        if (
            prepared_round.search_slate is not None
            and sha256_digest(prepared_round.search_slate.budget_snapshot)
            != sha256_digest(budget_snapshot)
        ):
            raise ValueError("prepared round budget snapshot drift")
        if confirmation_seed != getattr(prepared_round, "confirmation_seed", None):
            raise ValueError("prepared round confirmation seed identity drift")
        prepared_handoffs = tuple(
            getattr(prepared_round, "candidate_handoffs", ()) or ()
        )
        if research_profile_source is not None and not isinstance(
            research_profile_source,
            ResearchProfileSourceV1,
        ):
            raise ValueError(
                "research_profile_source must be ResearchProfileSourceV1"
            )
        if research_profile_source is not None and prepared_handoffs:
            for handoff in prepared_handoffs:
                if (
                    handoff.source_schema_version
                    != research_profile_source.schema_version
                    or handoff.source_ref != research_profile_source.source_ref
                    or handoff.source_digest != research_profile_source.source_digest
                ):
                    raise ValueError(
                        "prepared round profile source identity drift"
                    )
        if (
            research_profile_source is not None
            and not prepared_handoffs
            and prepared_round.portfolio_candidates
        ):
            raise ValueError(
                "prepared round lacks the complete profile-source handoff"
            )
        prepared_portfolio = tuple(prepared_round.portfolio_candidates)
        if not prepared_portfolio and prepared_handoffs:
            prepared_portfolio = tuple(
                item.portfolio_candidate for item in prepared_handoffs
            )
        if normalized_portfolio and normalized_portfolio != prepared_portfolio:
            raise ValueError("prepared round portfolio candidate identity drift")
        if prepared_handoffs:
            candidate_handoffs = _validate_candidate_handoffs(
                prepared_handoffs,
                bindings=prepared_round.search_bindings,
                active_profile=active_profile,
                require_complete=False,
            )
            handoff_portfolio = tuple(
                item.portfolio_candidate for item in candidate_handoffs
            )
            if prepared_portfolio and handoff_portfolio != prepared_portfolio:
                raise ValueError("prepared round handoff portfolio identity drift")
            normalized_portfolio = handoff_portfolio
        elif prepared_round.portfolio_candidates:
            candidate_handoffs = _static_candidate_handoffs(
                tuple(prepared_round.portfolio_candidates),
                bindings=prepared_round.search_bindings,
                active_profile=active_profile,
                qualified_execution_by_capability=(
                    qualified_execution_by_capability or {}
                ),
                candidate_root_by_capability=(candidate_root_by_capability or {}),
                resource_profile_by_capability=(
                    resource_profile_by_capability or {}
                ),
            )
            normalized_portfolio = tuple(
                item.portfolio_candidate for item in candidate_handoffs
            )
        else:
            candidate_handoffs = ()
            normalized_portfolio = ()
        if prepared_round.search_acquisition is None or not prepared_round.search_bindings:
            return ResearchRoundResult(
                context=context,
                active_profile=active_profile,
                producer_outcomes=prepared_round.producer_outcomes,
                carryover_outcomes=prepared_round.carryover_outcomes,
                resolutions=prepared_round.resolutions,
                deferred_innovation_outcomes=prepared_round.deferred_innovation_outcomes,
                deferred_search_outcomes=prepared_round.deferred_search_outcomes,
                search_acquisition=prepared_round.search_acquisition,
                innovation=prepared_round.innovation,
                selected_outcome=None,
                execution_recipe=None,
                candidate_run=None,
                interpretation=None,
                provider_traces=prepared_round.provider_traces,
                meta_research=None,
                attempts=(),
                metric_bearing_attempt_index=None,
                attempt_scheduler_enabled=True,
                incomplete_reason="ROUND_ATTEMPT_NO_LEGAL_INITIAL_BINDING",
                prepared=prepared_round,
            )
        (
            executed_semantic_seed_pairs,
            pending_task,
            mechanism_axis_effects,
        ) = _search_ranking_inputs(context)
        frozen_budget_snapshot = (
            prepared_round.search_slate.budget_snapshot
            if prepared_round.search_slate is not None
            else budget_snapshot
        )
        return _run_scheduled_attempts(
            context=context,
            active_profile=active_profile,
            producer_outcomes=prepared_round.producer_outcomes,
            carryover_outcomes=prepared_round.carryover_outcomes,
            resolutions=prepared_round.resolutions,
            deferred_innovation=prepared_round.deferred_innovation_outcomes,
            deferred_search=prepared_round.deferred_search_outcomes,
            search_acquisition=prepared_round.search_acquisition,
            search_bindings=prepared_round.search_bindings,
            search_pairs=prepared_round.search_pairs,
            open_search_pairs=prepared_round.open_search_pairs,
            innovation=prepared_round.innovation,
            budget_snapshot=frozen_budget_snapshot,
            router=router,
            policy=policy,
            memory_writer=memory_writer,
            runner=runner,
            incumbent_observation=incumbent_observation,
            metric_contract_digest=prepared_round.metric_contract_digest,
            observation_seed=prepared_round.observation_seed,
            next_discriminative_test=prepared_round.next_discriminative_test,
            confirmation_seed=getattr(prepared_round, "confirmation_seed", None),
            qualified_execution_by_capability=qualified_execution_by_capability,
            meta_research_inputs=meta_research_inputs,
            provider_traces=prepared_round.provider_traces,
            prepared_round=prepared_round,
            max_attempts_per_round=max_attempts_per_round,
            recovered_attempts=recovered_attempts,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            pending_task=pending_task,
            mechanism_axis_effects=mechanism_axis_effects,
            portfolio_candidates=normalized_portfolio,
            candidate_handoffs=candidate_handoffs,
        )
    (
        executed_semantic_seed_pairs,
        pending_task,
        mechanism_axis_effects,
    ) = _search_ranking_inputs(context)
    replay_producer = (
        getattr(meta_research_inputs.offline_replay, "producer", None)
        if meta_research_inputs is not None
        else None
    )
    trace_starts = _trace_starts(
        producer,
        innovation_inputs.implementer if innovation_inputs is not None else None,
        replay_producer,
    )
    producer_outcomes = produce_research_specs(context, producer, producer_bindings)
    legacy_carryover_outcomes = _carryover_outcomes(
        context, carryover_proposals, producer_bindings
    )
    open_carryover_pairs = _carryover_open_outcomes(
        context,
        carryover_open_candidates,
    )
    open_candidate_by_outcome_digest = {
        outcome.digest: candidate for outcome, candidate in open_carryover_pairs
    }
    carryover_outcomes = (
        *legacy_carryover_outcomes,
        *(outcome for outcome, _candidate in open_carryover_pairs),
    )
    resolutions = resolve_producer_outcomes(
        (*producer_outcomes, *carryover_outcomes),
        environment=resolver_environment,
    )
    search_pairs = tuple(
        (outcome, resolution)
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
        and outcome.source_proposal is not None
    )
    open_search_pairs = tuple(
        (outcome, resolution, open_candidate_by_outcome_digest[outcome.digest])
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
        and outcome.digest in open_candidate_by_outcome_digest
    )
    deferred_search = tuple(
        (outcome, resolution)
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
        and outcome.source_proposal is None
        and outcome.digest not in open_candidate_by_outcome_digest
    )
    search_bindings = tuple(
        [
            bind_search_candidate(
                profile=active_profile,
                proposal=outcome.source_proposal,
                capability_ref=resolution.resolved_current_capability_ref,
            )
            for outcome, resolution in search_pairs
            if outcome.source_proposal is not None
            and resolution.resolved_current_capability_ref is not None
        ]
        + [
            bind_search_candidate(
                profile=active_profile,
                proposal=candidate,
                capability_ref=candidate.capability_ref,
            )
            for _outcome, _resolution, candidate in open_search_pairs
        ]
    )
    candidate_handoffs = _candidate_handoffs_for_round(
        context=context,
        active_profile=active_profile,
        resolutions=resolutions,
        search_bindings=search_bindings,
        portfolio_candidates=normalized_portfolio,
        research_profile_source=research_profile_source,
        candidate_handoff_factory=candidate_handoff_factory,
        qualified_execution_by_capability=qualified_execution_by_capability,
        candidate_root_by_capability=candidate_root_by_capability,
        resource_profile_by_capability=resource_profile_by_capability,
    )
    if candidate_handoffs:
        normalized_portfolio = tuple(
            item.portfolio_candidate for item in candidate_handoffs
        )
    slate: FrozenExperimentSlateV1 | None = None
    if search_bindings:
        slate = freeze_experiment_slate(
            profile=active_profile,
            bindings=search_bindings,
            budget_snapshot=budget_snapshot,
        )
        acquisition = route_frozen_experiment_slate(
            profile=active_profile,
            slate=slate,
            router=router,
            policy_projection=policy.to_dict(),
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            current_observation_seed=observation_seed,
            pending_task=pending_task,
            mechanism_axis_effects=mechanism_axis_effects,
            portfolio_candidates=(
                normalized_portfolio if normalized_portfolio else None
            ),
        )
    else:
        acquisition = None
    innovation_pairs = tuple(
        (outcome, resolution)
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.INNOVATION_REQUIRED
    )
    deferred_innovation = tuple(
        (outcome, resolution)
        for outcome, resolution in innovation_pairs
        if innovation_inputs is None
    )
    if acquisition is not None:
        current_slate_ref = acquisition.slate_ref
        current_slate_digest = acquisition.slate_digest
    else:
        missing_slate = {
            "campaign_id": context.campaign_id,
            "profile_ref": active_profile.profile_ref,
            "profile_digest": active_profile.profile_digest,
            "budget_snapshot": canonical_value(budget_snapshot),
            "search_candidate_count": 0,
        }
        current_slate_ref = content_id(
            "recclaw-missing-search-opportunity-v1",
            missing_slate,
        )
        current_slate_digest = sha256_digest(missing_slate)
    innovation = (
        _run_innovation_lane(
            innovation_pairs,
            current_profile=active_profile,
            current_slate_ref=current_slate_ref,
            current_slate_digest=current_slate_digest,
            research_policy=policy,
            inputs=innovation_inputs,
        )
        if innovation_inputs is not None and innovation_pairs
        else None
    )
    if innovation is not None:
        deferred_innovation = tuple(
            pair
            for pair in innovation_pairs
            if pair[0].digest != innovation.selected_outcome.digest
        )
    provider_traces = _new_provider_traces(trace_starts)
    prepared = (
        PreparedResearchRoundV1(
            context_digest=context.digest,
            profile_ref=active_profile.profile_ref,
            profile_digest=active_profile.profile_digest,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation_outcomes=deferred_innovation,
            deferred_search_outcomes=deferred_search,
            search_acquisition=acquisition,
            search_slate=slate,
            search_bindings=search_bindings,
            search_pairs=search_pairs,
            open_search_pairs=open_search_pairs,
            innovation=innovation,
            metric_contract_digest=metric_contract_digest,
            observation_seed=observation_seed,
            next_discriminative_test=next_discriminative_test,
            confirmation_seed=confirmation_seed,
            provider_traces=provider_traces,
            portfolio_candidates=normalized_portfolio,
            candidate_handoffs=candidate_handoffs,
        )
        if attempt_scheduler
        else None
    )
    if attempt_scheduler and prepared is not None and on_prepared is not None:
        # This is the scheduler's durable boundary: the callback must return
        # before the first physical runner invocation below.
        on_prepared(prepared)
    binding = acquisition.selected_binding if acquisition is not None else None
    if binding is None:
        if attempt_scheduler:
            return ResearchRoundResult(
                context=context,
                active_profile=active_profile,
                producer_outcomes=producer_outcomes,
                carryover_outcomes=carryover_outcomes,
                resolutions=resolutions,
                deferred_innovation_outcomes=deferred_innovation,
                deferred_search_outcomes=deferred_search,
                search_acquisition=acquisition,
                innovation=innovation,
                selected_outcome=None,
                execution_recipe=None,
                candidate_run=None,
                interpretation=None,
                provider_traces=provider_traces,
                meta_research=None,
                attempts=(),
                metric_bearing_attempt_index=None,
                attempt_scheduler_enabled=True,
                incomplete_reason="ROUND_ATTEMPT_NO_LEGAL_INITIAL_BINDING",
                prepared=prepared,
            )
        resolution_classes: dict[str, int] = {}
        producer_failures: dict[str, str] = {}
        for outcome, resolution in resolutions:
            label = (
                resolution.resolution.value
                if resolution is not None
                else "PRODUCER_FAILURE"
            )
            resolution_classes[label] = resolution_classes.get(label, 0) + 1
            if outcome.failure_code is not None:
                producer_failures[outcome.producer_role] = outcome.failure_code
        diagnostic_detail = {
            "reason": "NO_LEGAL_SEARCH_BINDING",
            "resolution_classes": resolution_classes,
            "producer_failures": producer_failures,
            "search_ready_count": len(search_pairs) + len(open_search_pairs),
            "deferred_search_count": len(deferred_search),
            "innovation_required_count": len(innovation_pairs),
        }
        missing_interpretation = interpret_missing_search_opportunity(
            context=context,
            producer_outcomes=producer_outcomes,
            diagnostic_detail=diagnostic_detail,
            next_discriminative_test=next_discriminative_test,
            policy=policy,
            memory_writer=memory_writer,
            route_trace_digest=(
                acquisition.route_trace.digest if acquisition is not None else None
            ),
        )
        meta_research = (
            _run_meta_research(
                context=context,
                interpretation=missing_interpretation,
                inputs=meta_research_inputs,
            )
            if meta_research_inputs is not None
            else None
        )
        return ResearchRoundResult(
            context=context,
            active_profile=active_profile,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation_outcomes=deferred_innovation,
            deferred_search_outcomes=deferred_search,
            search_acquisition=acquisition,
            innovation=innovation,
            selected_outcome=None,
            execution_recipe=None,
            candidate_run=None,
            interpretation=missing_interpretation,
            provider_traces=provider_traces,
            meta_research=meta_research,
        )
    if attempt_scheduler:
        return _run_scheduled_attempts(
            context=context,
            active_profile=active_profile,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation=deferred_innovation,
            deferred_search=deferred_search,
            search_acquisition=acquisition,
            search_bindings=search_bindings,
            search_pairs=search_pairs,
            open_search_pairs=open_search_pairs,
            innovation=innovation,
            budget_snapshot=budget_snapshot,
            router=router,
            policy=policy,
            memory_writer=memory_writer,
            runner=runner,
            incumbent_observation=incumbent_observation,
            metric_contract_digest=metric_contract_digest,
            observation_seed=observation_seed,
            next_discriminative_test=next_discriminative_test,
            confirmation_seed=confirmation_seed,
            qualified_execution_by_capability=qualified_execution_by_capability,
            meta_research_inputs=meta_research_inputs,
            provider_traces=provider_traces,
            prepared_round=prepared,
            max_attempts_per_round=max_attempts_per_round,
            recovered_attempts=recovered_attempts,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            pending_task=pending_task,
            mechanism_axis_effects=mechanism_axis_effects,
            portfolio_candidates=normalized_portfolio,
            candidate_handoffs=candidate_handoffs,
        )
    if isinstance(binding.proposal, CandidateProposalV4):
        selected_outcome = next(
            outcome
            for outcome, _resolution in search_pairs
            if outcome.source_proposal is not None
            and outcome.source_proposal.candidate_id
            == binding.proposal.candidate_id
        )
        mechanism_projection: Mapping[str, Any] = (
            binding.proposal.mechanism_program
        )
    else:
        selected_outcome = next(
            outcome
            for outcome, _resolution, candidate in open_search_pairs
            if candidate.candidate_id == binding.proposal.candidate_id
        )
        mechanism_projection = {
            "schema": "recclaw-open-spec-realization-semantics-e0.v1",
            "semantic_identity_ref": binding.proposal.semantic_identity_ref,
            "semantic_identity_digest": binding.proposal.semantic_identity_digest,
            "research_spec_ref": binding.proposal.spec.spec_id,
            "research_spec_digest": binding.proposal.spec.digest,
            "candidate_package_ref": binding.proposal.candidate_package_ref,
            "candidate_package_digest": binding.proposal.candidate_package_digest,
            "execution_contract": binding.proposal.execution_contract,
        }
    recipe = _execution_recipe_for_handoff(
        binding,
        profile=active_profile,
        qualified_execution_by_capability=qualified_execution_by_capability,
        candidate_handoffs=candidate_handoffs,
    )
    candidate_run = canonical_value(dict(runner(recipe, binding)))
    try:
        run_experiment_binding = ExperimentBindingV1.from_canonical_dict(
            candidate_run.get("experiment_binding")
        )
    except ExperimentBindingError as error:
        raise ValueError(str(error)) from error
    _validate_experiment_binding(
        run_experiment_binding,
        recipe=recipe,
        observation_seed=observation_seed,
    )
    if candidate_run.get("execution_recipe_digest") != sha256_digest(recipe):
        raise ValueError("runner result is not bound to the selected execution recipe")
    if str(candidate_run.get("seed")) != observation_seed:
        raise ValueError("runner result seed differs from the frozen observation seed")
    run_binding_digest = candidate_run.get("experiment_binding_digest")
    if (
        run_binding_digest != run_experiment_binding.digest
        or candidate_run.get("experiment_binding_ref")
        != run_experiment_binding.ref
        or candidate_run.get("binding_digest", run_binding_digest)
        != run_binding_digest
    ):
        raise ValueError("runner result carries an inconsistent Experiment Binding identity")
    event, identity, episode, closure, _failure_detail = (
        project_common_execution_feedback(
            context=context,
            selected_outcome=selected_outcome,
            binding=binding,
            candidate_run=candidate_run,
            incumbent_observation=incumbent_observation,
            metric_contract_digest=metric_contract_digest,
            observation_seed=observation_seed,
            next_discriminative_test=next_discriminative_test,
        )
    )
    route_metadata = {
        "route_trace_digest": acquisition.route_trace.digest,
        "selected_producer_role": selected_outcome.producer_role,
        "selected_candidate_id": binding.proposal.candidate_id,
        "selected_candidate_semantic_digest": binding.mechanism_semantics_digest,
        "selected_mechanism_axis": binding.proposal.mechanism_axis,
        "required_selected_runnable_probability": (
            binding.proposal.utility_features.runnable_probability
        ),
        "mechanism_program": mechanism_projection,
        "comparator_identity": identity.comparator_ref,
        "required_seed_or_control": observation_seed,
        "next_discriminative_test": next_discriminative_test,
    }
    if confirmation_seed is not None:
        route_metadata["confirmation_seed"] = confirmation_seed
    if episode is None:
        interpretation = interpret_scientific_diagnostic(
            closure=closure,
            comparison_identity=identity,
            context=context,
            producer_outcomes=producer_outcomes,
            route_metadata=route_metadata,
            evaluator_projection=event,
            policy=policy,
            memory_writer=memory_writer,
            selected_outcome=selected_outcome,
        )
    else:
        interpretation = interpret_typed_research_episode(
            episode=episode,
            comparison_identity=identity,
            closure=closure,
            context=context,
            producer_outcomes=producer_outcomes,
            route_metadata=route_metadata,
            evaluator_projection=event,
            policy=policy,
            memory_writer=memory_writer,
            selected_outcome=selected_outcome,
        )
    meta_research = (
        _run_meta_research(
            context=context,
            interpretation=interpretation,
            inputs=meta_research_inputs,
        )
        if meta_research_inputs is not None
        else None
    )
    return ResearchRoundResult(
        context=context,
        active_profile=active_profile,
        producer_outcomes=producer_outcomes,
        carryover_outcomes=carryover_outcomes,
        resolutions=resolutions,
        deferred_innovation_outcomes=deferred_innovation,
        deferred_search_outcomes=deferred_search,
        search_acquisition=acquisition,
        innovation=innovation,
        selected_outcome=selected_outcome,
        execution_recipe=recipe,
        candidate_run=candidate_run,
        interpretation=interpretation,
            provider_traces=provider_traces,
        meta_research=meta_research,
    )


def activate_promoted_meta_strategy(
    result: ResearchRoundResult,
    *,
    next_profile: SearchExecutableProfileV1,
) -> tuple[ResearchContext, VersionedResearchPolicyV1]:
    """Bind a promoted strategy to the supplied next-campaign executable profile."""

    meta = result.meta_research
    if (
        result.interpretation is None
        or meta is None
        or meta.activated_policy is None
        or meta.activation_receipt is None
    ):
        raise ValueError("round has no promoted Meta strategy to activate")
    if next_profile.campaign_id != meta.activation_receipt.campaign_id:
        raise ValueError("next Search profile is not the Meta activation campaign")
    successor = result.interpretation.successor_context
    if (
        next_profile.protocol_ref != successor.protocol_ref
        or next_profile.protocol_digest != successor.protocol_digest
    ):
        raise ValueError("Meta activation cannot change the frozen protocol")
    meta_memory = {
        "proposal_digest": meta.update_proposal.digest,
        "shadow_evaluation_digest": meta.shadow_evaluation.digest,
        "promotion_decision_digest": meta.promotion_decision.digest,
        "activated_policy_digest": meta.activated_policy.digest,
    }
    context = replace(
        successor,
        campaign_id=next_profile.campaign_id,
        active_profile_ref=next_profile.profile_ref,
        active_profile_digest=next_profile.profile_digest,
        policy=meta.activated_policy.to_dict(),
        scientific_memory={
            **successor.scientific_memory,
            "meta_strategy": meta_memory,
        },
    )
    return context, meta.activated_policy


def activate_staged_innovation(
    result: ResearchRoundResult,
) -> tuple[
    SearchExecutableProfileV1,
    ResearchContext,
    SearchCandidate,
    Mapping[str, Any],
]:
    """Activate an admitted primitive only across the next-campaign boundary."""

    innovation = result.innovation
    if (
        innovation is None
        or not innovation.admitted
        or innovation.next_profile is None
        or innovation.registry is None
        or innovation.qualified_execution is None
        or innovation.search_candidate is None
    ):
        raise ValueError("round has no admitted Innovation capability to activate")
    active = activate_next_fresh_search_profile(
        predecessor=result.active_profile,
        next_profile=innovation.next_profile,
        registry=innovation.registry,
        fresh_campaign_id=innovation.fresh_campaign_id,
    )
    successor = result.successor_context
    memory = {
        **successor.scientific_memory,
        "activated_capability": {
            "capability": innovation.capability.canonical_dict(),
            "search_candidate": innovation.search_candidate.to_dict(),
            "qualified_execution": innovation.qualified_execution,
        },
    }
    successor = replace(
        successor,
        campaign_id=active.campaign_id,
        active_profile_ref=active.profile_ref,
        active_profile_digest=active.profile_digest,
        scientific_memory=memory,
    )
    return (
        active,
        successor,
        innovation.search_candidate,
        innovation.qualified_execution,
    )


__all__ = [
    "CandidateHandoffFactory",
    "IdeaAcquisitionResult",
    "InnovationLaneResult",
    "InnovationRuntimeInputs",
    "MetaResearchInputs",
    "MetaResearchResult",
    "PreparedResearchRoundV1",
    "ResearchProfileSourceV1",
    "RoundCandidateHandoffV1",
    "RoundAttemptV1",
    "ResearchRoundResult",
    "SearchCandidate",
    "activate_promoted_meta_strategy",
    "activate_staged_innovation",
    "bindings_for_context",
    "resolver_environment_for_profile",
    "run_research_round",
]
