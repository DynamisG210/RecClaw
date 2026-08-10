"""Standalone, checkpointed Research/BLICF campaign composition.

This module is the Research-side production entry for more than one round.
It composes the existing :class:`ResearchCampaign` and the existing Provider,
runner, evaluator, resource, portfolio, role-memory, and task-queue contracts.
It intentionally has no paired-arm controller or paired-runtime dependency.

The composition layer does not read a dataset partition or a baseline receipt.
The caller must provide explicit Research-only source and baseline identities;
those identities are sealed into the campaign context and checked on resume.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (
    MAX_REPAIR_TURNS,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_EVALUATOR,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    meta_v20_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
    run_disposable_fixed_batch_resource_probe,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchExecutableProfileV1,
    adapt_current_search_profile,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import CapabilityKindV1

from .bootstrap import bootstrap_search_pool
from .campaign import (
    CampaignRoundInputs,
    CampaignState,
    ResearchCampaign,
)
from .fresh_runner import (
    FreshRunnerError,
    GpuReservationEvidenceProvider,
    Launcher,
    FreshExperimentRunner,
    get_gpu_reservation_provider_identity,
    make_fresh_runner,
)
from .interfaces import ResearchContext, ResearchTaskQueueV2
from .portfolio import PortfolioCandidateV2
from .profile_source import ResearchProfileSourceV1
from .provider import (
    ConfigSource,
    ProviderCall,
    ProviderImplementerGateway,
    ProviderResearchProducer,
)
from .replay import OfflineProducerReplayV1
from .runtime import (
    CandidateHandoffFactory,
    InnovationRuntimeInputs,
    MetaResearchInputs,
    bindings_for_context,
    resolver_environment_for_profile,
)
from .single_round import ResearchBaselineSourceV1, _bytes_sha256


class StandaloneCampaignError(ValueError):
    """Raised when a standalone campaign cannot be composed or resumed safely."""


MAX_STANDALONE_ROUNDS = 50
STANDALONE_SCHEMA = "recclaw.research-line.standalone-campaign.v1"
META_REPLAY_INTERVAL = 10
RESEARCH_CONSUMER_COMPOSITION_SCHEMA = (
    "recclaw.research-line.standalone-research-consumers.v1"
)
RESOURCE_COMPUTE_PATTERN_SCHEMA = (
    "recclaw.research-line.standalone-compute-pattern.v1"
)

# These are the fields carried by runtime._resource_probe_recipe that describe
# the realized compute shape.  Capability/package/source/mechanism identities
# are deliberately excluded so correlated resource observations remain useful.
_RESOURCE_COMPUTE_PATTERN_FIELDS = (
    "model",
    "base_model_config",
    "config",
    "entrypoint",
    "dataset",
    "split",
    "evaluator",
    "execution_role",
)

_IMPLEMENTATION_REQUIREMENTS = (
    "RecBole GeneralRecommender interface",
    "candidate-local recclaw_ext package",
    "finite pairwise loss and full-sort scores",
)
_COMPATIBILITY_REQUIREMENTS = (
    "general collaborative filtering",
    "offline top-n evaluation",
    "pairwise input",
    "train-only fitting",
)


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StandaloneCampaignError(
            f"{field_name} must be normalized and non-empty"
        )
    return value


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StandaloneCampaignError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StandaloneCampaignError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _gpu_selector(config: "StandaloneResearchConfig") -> str | None:
    """Return the physical reservation selector without selecting an env mode."""

    if config.gpu_id is not None:
        return str(config.gpu_id)
    return config.cuda_visible_devices


def _bounded_round_count(value: Any, *, field_name: str = "round_count") -> int:
    result = _positive_int(value, field_name=field_name)
    if result > MAX_STANDALONE_ROUNDS:
        raise StandaloneCampaignError(
            f"{field_name} must be <= {MAX_STANDALONE_ROUNDS}"
        )
    return result


def _canonical_mapping(
    value: Mapping[str, Any], *, field_name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StandaloneCampaignError(f"{field_name} must be a mapping")
    try:
        return canonical_value(dict(value))
    except (TypeError, ValueError) as error:
        raise StandaloneCampaignError(f"{field_name} is not canonicalizable") from error


def _source_baseline_identity(config: "StandaloneResearchConfig") -> dict[str, Any]:
    return canonical_value(config.baseline_source.canonical_dict())


def _seed_schedule_identity(config: "StandaloneResearchConfig") -> dict[str, Any]:
    schedule = config.observation_seed_schedule
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.observation-seed-schedule.v1",
            "enabled": schedule is not None,
            "seeds": tuple(schedule) if schedule is not None else (),
        }
    )
    return canonical_value({**dict(payload), "digest": sha256_digest(payload)})


def _observation_seed_for_round(
    config: "StandaloneResearchConfig", round_index: int
) -> str:
    if config.observation_seed_schedule is None:
        return str(config.seed)
    if round_index < 1 or round_index > len(config.observation_seed_schedule):
        raise StandaloneCampaignError(
            "observation_seed_schedule does not cover the requested round"
        )
    return str(config.observation_seed_schedule[round_index - 1])


def _confirmation_seed_for_round(
    config: "StandaloneResearchConfig", round_index: int
) -> str | None:
    schedule = config.observation_seed_schedule
    if schedule is None or round_index >= len(schedule):
        return None
    return str(schedule[round_index])


def _meta_replay_due(round_index: int) -> bool:
    return round_index == 1 or round_index % META_REPLAY_INTERVAL == 0


def _gpu_provider_identity(
    provider: GpuReservationEvidenceProvider | None,
) -> Mapping[str, Any] | None:
    try:
        return get_gpu_reservation_provider_identity(provider)
    except FreshRunnerError as error:
        raise StandaloneCampaignError(
            f"GPU reservation provider identity is invalid: {error}"
        ) from error


def _resource_compute_pattern(execution_recipe: Mapping[str, Any]) -> str:
    """Derive a reusable compute identity from frozen recipe structure only."""

    if not isinstance(execution_recipe, Mapping):
        raise StandaloneCampaignError("resource probe requires an execution recipe")
    missing = tuple(
        field_name
        for field_name in _RESOURCE_COMPUTE_PATTERN_FIELDS
        if field_name not in execution_recipe
    )
    if missing:
        raise StandaloneCampaignError(
            "resource probe recipe lacks compute-pattern fields: "
            + ", ".join(missing)
        )
    structure = canonical_value(
        {
            field_name: execution_recipe[field_name]
            for field_name in _RESOURCE_COMPUTE_PATTERN_FIELDS
        }
    )
    identity = canonical_value(
        {
            "schema": RESOURCE_COMPUTE_PATTERN_SCHEMA,
            "recipe_structure": structure,
        }
    )
    return f"{RESOURCE_COMPUTE_PATTERN_SCHEMA}:{sha256_digest(identity)}"


def _resource_probe_run_id(candidate_ref: str) -> str:
    """Mirror the existing disposable scheduler arm-to-run-id derivation.

    ``run_disposable_fixed_batch_resource_probe`` derives its worker ``run_id``
    from the candidate arm before launching the spawn worker.  The evidence
    provider must receive that exact identity before the worker starts.
    """

    candidate_ref = _text(candidate_ref, field_name="resource probe candidate_ref")
    arm_id = "candidate_" + sha256_digest({"candidate_ref": candidate_ref})[:16]
    return arm_id.replace("_", "-")


def _research_consumer_composition_identity(
    config: "StandaloneResearchConfig",
) -> dict[str, Any]:
    payload = canonical_value(
        {
            "schema": RESEARCH_CONSUMER_COMPOSITION_SCHEMA,
            "innovation": {
                "enabled": True,
                "capability_kind": CapabilityKindV1.COMPLETE_MODEL.value,
                "resource_admission_required": True,
                "gpu_reservation_evidence_required": (
                    config.require_gpu_reservation_evidence
                ),
                "resource_policy": "DISPOSABLE_FIXED_BATCH_RESOURCE_PROFILE_V1",
                "probe_physical_identity_derivation": (
                    "candidate_ref_sha256_prefix16_arm_v1"
                ),
                "final_worker_ceiling_seconds": (
                    config.final_worker_ceiling_seconds
                ),
                "prediction_and_gpu_seconds_are_separate": True,
            },
            "meta": {
                "first_round": 1,
                "interval_rounds": META_REPLAY_INTERVAL,
                "equal_replay_token_charge": fresh_r1.PROPOSAL_TOKEN_CEILING,
                "deterministic_directive_replay": True,
            },
            "token_ceilings": {
                "proposal": fresh_r1.PROPOSAL_TOKEN_CEILING,
                "implementation": fresh_r1.IMPLEMENTATION_TOKEN_CEILING,
            },
            "compute_pattern_derivation_schema": RESOURCE_COMPUTE_PATTERN_SCHEMA,
        }
    )
    return canonical_value({**dict(payload), "digest": sha256_digest(payload)})


def _probe_gpu_reservation_evidence(
    config: "StandaloneResearchConfig",
    *,
    candidate_ref: str,
    candidate_package_digest: str | None,
    execution_recipe: Mapping[str, Any],
    compute_pattern: str,
) -> Mapping[str, Any] | None:
    provider = config.gpu_reservation_evidence_provider
    if provider is None:
        if config.require_gpu_reservation_evidence:
            raise StandaloneCampaignError(
                "formal resource probe requires GPU reservation evidence"
            )
        return None
    reservation_selector = _gpu_selector(config)
    physical_run_id = _resource_probe_run_id(candidate_ref)
    request = canonical_value(
        {
            "schema": "recclaw.research-line.resource-probe-reservation-request.v1",
            "physical_run_id": physical_run_id,
            "purpose": "RESOURCE_PROBE_ONLY",
            "run_identity": "research-innovation-resource-probe-v1",
            "candidate_ref": candidate_ref,
            "candidate_package_digest": candidate_package_digest,
            "compute_pattern": compute_pattern,
            "cuda_visible_devices": reservation_selector,
            "final_worker_ceiling_seconds": config.final_worker_ceiling_seconds,
            "execution_recipe": execution_recipe,
        }
    )
    try:
        evidence = provider(request)
        validated = fresh_r1.validate_gpu_reservation_evidence(
            evidence,
            cuda_visible_devices=config.cuda_visible_devices,
            physical_gpu_selector=(
                str(config.gpu_id) if config.gpu_id is not None else None
            ),
            run_id=physical_run_id,
        )
    except (fresh_r1.FreshR1Error, TypeError, ValueError) as error:
        raise StandaloneCampaignError(
            f"resource probe GPU reservation evidence is invalid: {error}"
        ) from error
    if validated is None:
        if config.require_gpu_reservation_evidence:
            raise StandaloneCampaignError(
                "formal resource probe received no GPU reservation evidence"
            )
        return None
    if not isinstance(validated, Mapping):
        raise StandaloneCampaignError(
            "resource probe GPU reservation evidence is not a mapping"
        )
    return canonical_value(dict(validated))


def _production_innovation_inputs(
    config: "StandaloneResearchConfig",
    implementer: ProviderImplementerGateway,
    *,
    campaign_id: str,
    seed: int,
    round_index: int,
) -> InnovationRuntimeInputs:
    prompt = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources/research_line_implementer_prompt_v1.txt"
    )
    policy = fresh_r1._shared_policy(
        _bytes_sha256(prompt),
        sha256_digest(
            {
                "tools": (),
                "network": False,
                "allowed_files": (
                    "recclaw_ext/__init__.py",
                    "recclaw_ext/candidate.py",
                ),
            }
        ),
        execution_contract=None,
    )

    def contract(value: Any) -> Mapping[str, Any]:
        execution_contract = value.execution_contract
        if not isinstance(execution_contract, Mapping):
            raise ValueError("qualified OpenSpec lacks execution_contract")
        return execution_contract

    def fixture_factory(
        value: Any,
        attempt: int,
        _candidate_root: Path,
    ) -> Any:
        execution_contract = contract(value)
        fixture = fresh_r1._qualification_fixture(
            config.repo_root,
            seed=seed + attempt,
            root=(
                config.run_root
                / "innovation_qualification"
                / f"round-{round_index:04d}"
                / f"attempt-{attempt:02d}"
            ),
            base_model_config=str(execution_contract["base_model_config"]),
        )
        return replace(
            fixture,
            runtime_identity_ref=value.runtime_identity_ref,
            runtime_identity_digest=value.runtime_identity_digest,
        )

    def unit_check_factory(value: Any) -> Any:
        return fresh_r1._shared_behavioral_unit_check(
            {},
            base_model_config=str(contract(value)["base_model_config"]),
        )

    def resource_probe(**kwargs: Any) -> Mapping[str, Any]:
        execution_recipe = kwargs.get("execution_recipe")
        if not isinstance(execution_recipe, Mapping):
            raise StandaloneCampaignError(
                "Research Innovation resource probe requires execution_recipe"
            )
        candidate_ref = kwargs.get("candidate_ref")
        candidate_package_digest = kwargs.get("candidate_package_digest")
        compute_pattern = _resource_compute_pattern(execution_recipe)
        reservation_evidence = _probe_gpu_reservation_evidence(
            config,
            candidate_ref=_text(
                candidate_ref,
                field_name="resource probe candidate_ref",
            ),
            candidate_package_digest=(
                str(candidate_package_digest)
                if candidate_package_digest is not None
                else None
            ),
            execution_recipe=execution_recipe,
            compute_pattern=compute_pattern,
        )
        probe_kwargs = dict(kwargs)
        for field_name in (
            "compute_pattern",
            "cuda_visible_devices",
            "gpu_reservation_evidence",
            "total_budget_seconds",
            "probe_seed",
        ):
            probe_kwargs.pop(field_name, None)
        return run_disposable_fixed_batch_resource_probe(
            config.repo_root,
            total_budget_seconds=config.final_worker_ceiling_seconds,
            probe_seed=seed,
            compute_pattern=compute_pattern,
            cuda_visible_devices=config.cuda_visible_devices,
            gpu_id=config.gpu_id,
            gpu_reservation_evidence=reservation_evidence,
            **probe_kwargs,
        )

    return InnovationRuntimeInputs(
        implementer=implementer,
        policy=policy,
        candidate_parent=(
            config.run_root
            / "innovation_candidates"
            / f"round-{round_index:04d}"
        ),
        fixture_factory=fixture_factory,
        unit_check_factory=unit_check_factory,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version=f"{campaign_id}:round-{round_index:04d}:qualified-v1",
        registry_version=f"{campaign_id}:round-{round_index:04d}:registry-v1",
        predecessor_registry_ref="registry:fixed-66",
        predecessor_registry_digest=sha256_digest({"registry": "fixed-66"}),
        profile_version=(
            f"{campaign_id}:round-{round_index:04d}:next-fresh-profile-v1"
        ),
        fresh_campaign_id=f"{campaign_id}:round-{round_index:04d}:next-fresh",
        resource_admission_required=True,
        resource_probe=resource_probe,
        resource_probe_parent=(
            config.run_root
            / "innovation_resource_probes"
            / f"round-{round_index:04d}"
        ),
    )


@dataclass(frozen=True, slots=True)
class StandaloneResearchConfig:
    """Frozen, caller-supplied inputs for one standalone campaign root."""

    repo_root: Path
    run_root: Path
    api_config_source: ConfigSource
    campaign_id: str
    baseline_source: ResearchBaselineSourceV1
    seed: int = 54303
    epochs: int = fresh_r1.EXPERIMENT_EPOCHS
    timeout_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS
    watchdog_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS
    final_worker_ceiling_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS
    cuda_visible_devices: str | None = None
    observation_seed_schedule: tuple[int, ...] | None = None
    gpu_reservation_evidence_provider: GpuReservationEvidenceProvider | None = None
    require_gpu_reservation_evidence: bool = False
    round_count: int = 1
    attempt_scheduler: bool = False
    max_attempts_per_round: int | None = None
    portfolio_candidates: tuple[PortfolioCandidateV2, ...] = ()
    research_profile_source: ResearchProfileSourceV1 | None = None
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    candidate_root_by_capability: Mapping[str, Path | str] = field(
        default_factory=dict
    )
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    candidate_handoff_factory: CandidateHandoffFactory | None = None
    gpu_id: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("repo_root", "run_root"):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise StandaloneCampaignError(f"{field_name} must be a Path")
            object.__setattr__(self, field_name, value.resolve())
        if not self.repo_root.is_dir():
            raise StandaloneCampaignError(
                f"repo_root is not a directory: {self.repo_root}"
            )
        _text(self.campaign_id, field_name="campaign_id")
        if not isinstance(self.baseline_source, ResearchBaselineSourceV1):
            raise StandaloneCampaignError(
                "baseline_source must be ResearchBaselineSourceV1"
            )
        if self.baseline_source.receipt_path is not None:
            raise StandaloneCampaignError(
                "standalone Research requires an identity-backed baseline source; "
                "receipt-backed baselines remain single-round compatible"
            )
        if self.baseline_source.protocol_digest is None:
            raise StandaloneCampaignError(
                "standalone baseline source must carry protocol_digest"
            )
        if self.baseline_source.seed is None:
            raise StandaloneCampaignError(
                "standalone baseline source must carry seed"
            )
        object.__setattr__(
            self,
            "seed",
            _nonnegative_int(self.seed, field_name="seed"),
        )
        for field_name in (
            "epochs",
            "timeout_seconds",
            "watchdog_seconds",
            "final_worker_ceiling_seconds",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "round_count",
            _bounded_round_count(self.round_count),
        )
        if self.cuda_visible_devices is not None:
            _text(self.cuda_visible_devices, field_name="cuda_visible_devices")
        if self.gpu_id is not None:
            object.__setattr__(
                self,
                "gpu_id",
                _nonnegative_int(self.gpu_id, field_name="gpu_id"),
            )
        if self.gpu_id is not None and self.cuda_visible_devices is not None:
            raise StandaloneCampaignError(
                "gpu_id and cuda_visible_devices are mutually exclusive"
            )
        schedule = self.observation_seed_schedule
        if schedule is not None:
            if isinstance(schedule, (str, bytes)):
                raise StandaloneCampaignError(
                    "observation_seed_schedule must be a sequence of seeds"
                )
            try:
                normalized_schedule = tuple(schedule)
            except TypeError as error:
                raise StandaloneCampaignError(
                    "observation_seed_schedule must be a sequence of seeds"
                ) from error
            if not normalized_schedule:
                raise StandaloneCampaignError(
                    "observation_seed_schedule must not be empty"
                )
            normalized_schedule = tuple(
                _positive_int(seed, field_name="observation_seed_schedule item")
                for seed in normalized_schedule
            )
            if normalized_schedule[0] != self.seed:
                raise StandaloneCampaignError(
                    "observation_seed_schedule[0] must equal the configured seed"
                )
            if len(normalized_schedule) < self.round_count:
                raise StandaloneCampaignError(
                    "observation_seed_schedule does not cover round_count"
                )
            object.__setattr__(self, "observation_seed_schedule", normalized_schedule)
        if self.gpu_reservation_evidence_provider is not None and not callable(
            self.gpu_reservation_evidence_provider
        ):
            raise StandaloneCampaignError(
                "gpu_reservation_evidence_provider must be callable"
            )
        if not isinstance(self.require_gpu_reservation_evidence, bool):
            raise StandaloneCampaignError(
                "require_gpu_reservation_evidence must be a boolean"
            )
        provider_identity = _gpu_provider_identity(
            self.gpu_reservation_evidence_provider
        )
        if self.require_gpu_reservation_evidence:
            if _gpu_selector(self) is None:
                raise StandaloneCampaignError(
                    "required GPU reservation evidence needs explicit "
                    "cuda_visible_devices or gpu_id"
                )
            if self.gpu_reservation_evidence_provider is None:
                raise StandaloneCampaignError(
                    "required GPU reservation evidence needs a provider"
                )
            if provider_identity is None:
                raise StandaloneCampaignError(
                    "required GPU reservation evidence needs a stable provider "
                    "identity"
                )
        if not isinstance(self.attempt_scheduler, bool):
            raise StandaloneCampaignError("attempt_scheduler must be a boolean")
        if self.max_attempts_per_round is not None:
            if (
                isinstance(self.max_attempts_per_round, bool)
                or not isinstance(self.max_attempts_per_round, int)
                or self.max_attempts_per_round < 0
            ):
                raise StandaloneCampaignError(
                    "max_attempts_per_round must be a non-negative integer"
                )
        if self.attempt_scheduler and self.max_attempts_per_round is None:
            raise StandaloneCampaignError(
                "attempt_scheduler requires max_attempts_per_round"
            )
        if self.candidate_handoff_factory is not None and not callable(
            self.candidate_handoff_factory
        ):
            raise StandaloneCampaignError(
                "candidate_handoff_factory must be callable"
            )
        candidates = tuple(self.portfolio_candidates)
        if any(not isinstance(item, PortfolioCandidateV2) for item in candidates):
            raise StandaloneCampaignError(
                "portfolio_candidates must contain complete PortfolioCandidateV2 values"
            )
        candidate_ids = [item.candidate_id for item in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise StandaloneCampaignError("portfolio_candidates repeat candidate_id")
        object.__setattr__(self, "portfolio_candidates", candidates)
        if self.research_profile_source is not None and not isinstance(
            self.research_profile_source,
            ResearchProfileSourceV1,
        ):
            raise StandaloneCampaignError(
                "research_profile_source must be ResearchProfileSourceV1"
            )
        object.__setattr__(
            self,
            "qualified_execution_by_capability",
            _canonical_mapping(
                self.qualified_execution_by_capability,
                field_name="qualified_execution_by_capability",
            ),
        )
        roots: dict[str, Path] = {}
        for capability_ref, root in self.candidate_root_by_capability.items():
            roots[_text(capability_ref, field_name="candidate_root capability_ref")] = (
                Path(root).resolve()
            )
        object.__setattr__(self, "candidate_root_by_capability", roots)
        object.__setattr__(
            self,
            "resource_profile_by_capability",
            _canonical_mapping(
                self.resource_profile_by_capability,
                field_name="resource_profile_by_capability",
            ),
        )
        if not isinstance(self.api_config_source, Mapping):
            config_path = Path(self.api_config_source).resolve()
            object.__setattr__(self, "api_config_source", config_path)


@dataclass(frozen=True, slots=True)
class StandaloneResearchComposition:
    """Bound production objects for a resumable standalone campaign."""

    config: StandaloneResearchConfig
    campaign: ResearchCampaign
    provider: ProviderResearchProducer
    implementer: ProviderImplementerGateway
    runner: FreshExperimentRunner
    profile: SearchExecutableProfileV1
    policy: VersionedResearchPolicyV1
    manifest: Mapping[str, Any]

    def run(self, round_count: int | None = None) -> tuple[Any, ...]:
        """Execute at most the explicitly bounded number of Research rounds."""

        count = self.config.round_count if round_count is None else round_count
        count = _bounded_round_count(count)
        results: list[Any] = []
        for _ in range(count):
            before = self.campaign.state.next_round_index
            if before > MAX_STANDALONE_ROUNDS:
                raise StandaloneCampaignError(
                    f"campaign is beyond the {MAX_STANDALONE_ROUNDS}-round bound"
                )
            result = self.campaign.run_round()
            results.append(result)
            after = self.campaign.state.next_round_index
            if after == before:
                # Attempt-scheduler checkpoints intentionally retain the current
                # round until a later clean resume can consume the observations.
                break
            if after != before + 1:
                raise StandaloneCampaignError(
                    "ResearchCampaign did not advance exactly one round"
                )
        return tuple(results)


def _initial_incumbent(config: StandaloneResearchConfig) -> dict[str, Any]:
    source = config.baseline_source
    return canonical_value(
        {
            "comparator_ref": source.comparator_ref,
            "comparator_digest": source.comparator_digest,
            "frozen_ndcg@10": source.frozen_ndcg_at_10,
            "source_ref": source.source_ref,
            "source_receipt": None,
            "source_receipt_sha256": source.source_sha256,
        }
    )


def _initial_context(
    *,
    config: StandaloneResearchConfig,
    profile: SearchExecutableProfileV1,
    policy: VersionedResearchPolicyV1,
    incumbent: Mapping[str, Any],
) -> ResearchContext:
    identity = _source_baseline_identity(config)
    return ResearchContext(
        campaign_id=config.campaign_id,
        round_index=1,
        knowledge_base={
            "kind": "RESEARCH_ONLY_BLICF_STANDALONE",
            "search_space": "BL-ICF",
            "active_executable_capability_count": len(profile.entries),
            "source_identity": identity,
            "protocol": "Research-side fixed-protocol offline top-n evaluation",
        },
        frozen_goal={
            "metric": "NDCG@10",
            "direction": "maximize",
            "research_objective": (
                "improve the Research-side BLICF frontier while identifying the "
                "mechanism responsible for an observed signal"
            ),
            "ordinary_experiment_opportunities": 1,
            "single_seed_claim_ceiling": "INCONCLUSIVE",
        },
        frontier={
            "incumbent_ndcg@10": incumbent["frozen_ndcg@10"],
            "incumbent_ref": incumbent["comparator_ref"],
            "incumbent_digest": incumbent["comparator_digest"],
        },
        scientific_memory={
            "by_role": {role: {} for role in DISCOVERY_PRODUCERS},
            "global_memory": {
                "standalone_identity": identity,
                "task_queue": ResearchTaskQueueV2().to_dict(),
                "executed_observations": (),
                "negative_evidence": (),
            },
            "prior_round_count": 0,
        },
        unresolved_questions=(
            {
                "question": (
                    "Which executable BLICF mechanism can improve NDCG@10 beyond "
                    "the frozen Research baseline under the same protocol?"
                ),
                "mechanism_axis": "architecture",
            },
            {
                "question": (
                    "What competing explanation would reproduce the selected "
                    "mechanism's predicted signature?"
                ),
                "mechanism_axis": "objective",
            },
        ),
        policy=policy.to_dict(),
        budget={
            "producer_logical_calls": len(DISCOVERY_PRODUCERS),
            "producer_token_ceiling_each": fresh_r1.PROPOSAL_TOKEN_CEILING,
            "implementer_logical_calls_max": MAX_REPAIR_TURNS + 1,
            "implementer_token_ceiling_each": fresh_r1.IMPLEMENTATION_TOKEN_CEILING,
            "physical_attempts_per_logical_call_max": fresh_r1.MAX_PHYSICAL_ATTEMPTS,
            "experiment_opportunities": 1,
            "campaign_round_bound": MAX_STANDALONE_ROUNDS,
        },
        active_profile_ref=profile.profile_ref,
        active_profile_digest=profile.profile_digest,
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
    )


def _router() -> StrongStaticRouterV1:
    return StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=4,
    )


def _round_input_factory(
    config: StandaloneResearchConfig,
    *,
    provider: ProviderResearchProducer,
    implementer: ProviderImplementerGateway,
    portfolio_candidates: tuple[PortfolioCandidateV2, ...],
):
    def inputs(state: CampaignState) -> CampaignRoundInputs:
        bindings = bindings_for_context(
            state.context,
            active_profile=state.active_profile,
            implementation_requirements=_IMPLEMENTATION_REQUIREMENTS,
            compatibility_requirements=_COMPATIBILITY_REQUIREMENTS,
        )
        seed = int(_observation_seed_for_round(config, state.round_index))
        meta_inputs = None
        if _meta_replay_due(state.round_index):
            meta_inputs = MetaResearchInputs(
                offline_replay=OfflineProducerReplayV1(
                    producer=provider,
                    producer_bindings=bindings,
                    equal_replay_token_charge=fresh_r1.PROPOSAL_TOKEN_CEILING,
                    deterministic_directive_replay=True,
                ),
                next_campaign_id=(
                    f"{state.campaign_id}:round-{state.round_index:04d}:next-fresh"
                ),
            )
        return CampaignRoundInputs(
            producer_bindings=bindings,
            resolver_environment=resolver_environment_for_profile(
                state.active_profile,
                available_dependencies=fresh_r1.AVAILABLE_DEPENDENCIES,
                budget_limits=fresh_r1.BUDGET_LIMITS,
                protocol_requirements=fresh_r1.PROTOCOL_REQUIREMENTS,
            ),
            budget_snapshot={
                "experiment_opportunities": 1,
                "producer_logical_calls": len(DISCOVERY_PRODUCERS),
                "physical_attempts_per_logical_call_max": (
                    fresh_r1.MAX_PHYSICAL_ATTEMPTS
                ),
                **(
                    {"max_attempts_per_round": config.max_attempts_per_round}
                    if config.max_attempts_per_round is not None
                    else {}
                ),
            },
            router=_router(),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed=str(seed),
            confirmation_seed=_confirmation_seed_for_round(
                config, state.round_index
            ),
            next_discriminative_test=(
                "Use the typed Research feedback and task queue to select the "
                "next independent BLICF opportunity."
            ),
            qualified_execution_by_capability=(
                state.qualified_execution_by_capability
            ),
            attempt_scheduler=config.attempt_scheduler,
            max_attempts_per_round=config.max_attempts_per_round,
            candidate_handoff_factory=config.candidate_handoff_factory,
            research_profile_source=config.research_profile_source,
            candidate_root_by_capability=state.candidate_root_by_capability,
            resource_profile_by_capability=(
                state.resource_profile_by_capability or {}
            ),
            innovation_inputs=_production_innovation_inputs(
                config,
                implementer,
                campaign_id=state.campaign_id,
                seed=seed,
                round_index=state.round_index,
            ),
            meta_research_inputs=meta_inputs,
            # Passing this through is deliberately opt-in.  An empty tuple keeps
            # the legacy route; a non-empty tuple must already be complete and
            # identity-bound, so runtime/search_adapter performs the final match.
            portfolio_candidates=portfolio_candidates,
        )

    return inputs


def _build_provider_objects(
    config: StandaloneResearchConfig,
    *,
    provider_call: ProviderCall | None,
    launch: Launcher | None,
) -> tuple[
    ProviderResearchProducer,
    ProviderImplementerGateway,
    FreshExperimentRunner,
]:
    if not isinstance(config.api_config_source, Mapping) and not Path(
        config.api_config_source
    ).is_file():
        raise StandaloneCampaignError(
            f"api_config_source is not a file: {config.api_config_source}"
        )
    call_root = config.run_root / "provider_calls"
    provider = ProviderResearchProducer(
        config_source=config.api_config_source,
        call_root=call_root,
        session_id=config.campaign_id,
        provider_call=provider_call,
        proposal_seed=config.seed,
    )
    implementer = ProviderImplementerGateway(
        config_source=config.api_config_source,
        call_root=call_root,
        session_id=config.campaign_id,
        provider_call=provider_call,
    )
    runner = make_fresh_runner(
        repo_root=config.repo_root,
        side_root=config.run_root / "execution",
        run_id=f"research-line-standalone:{config.campaign_id}",
        seed=config.seed,
        epochs=config.epochs,
        timeout_seconds=config.timeout_seconds,
        execution_purpose="RESEARCH_LINE_STANDALONE_BLICF",
        candidate_root_by_capability=config.candidate_root_by_capability,
        recbole_commit_identity=str(
            fresh_r1.campaign_training_runtime_release().backend_identity[
                "recbole_commit"
            ]
        ),
        resource_telemetry=True,
        watchdog_seconds=config.watchdog_seconds,
        cuda_visible_devices=config.cuda_visible_devices,
        gpu_id=config.gpu_id,
        final_worker_ceiling_seconds=config.final_worker_ceiling_seconds,
        resource_prediction_by_capability=config.resource_profile_by_capability,
        gpu_reservation_evidence_provider=(
            config.gpu_reservation_evidence_provider
        ),
        require_gpu_reservation_evidence=(
            config.require_gpu_reservation_evidence
        ),
        launch=launch,
    )
    return provider, implementer, runner


def _manifest(
    *,
    config: StandaloneResearchConfig,
    profile: SearchExecutableProfileV1,
    policy: VersionedResearchPolicyV1,
    context: ResearchContext,
    provider: ProviderResearchProducer,
    runner: FreshExperimentRunner,
) -> dict[str, Any]:
    runner_identity = _runner_identity(runner)
    return canonical_value(
        {
            "schema": STANDALONE_SCHEMA,
            "controller": "ResearchCampaign",
            "research_only": True,
            "arm_semantics": "standalone",
            "campaign_id": config.campaign_id,
            "run_root": str(config.run_root),
            "source_baseline_identity": _source_baseline_identity(config),
            "observation_seed_schedule": _seed_schedule_identity(config),
            "profile": {
                "ref": profile.profile_ref,
                "digest": profile.profile_digest,
                "entry_count": len(profile.entries),
            },
            "policy": {
                "digest": policy.digest,
                "value": policy.to_dict(),
            },
            "context": {
                "ref": context.context_ref,
                "digest": context.digest,
                "value": context.to_dict(),
            },
            "provider": {
                "config_identity": provider.config_identity,
                "model": fresh_r1.MODEL,
                "proposal_token_ceiling": fresh_r1.PROPOSAL_TOKEN_CEILING,
                "implementation_token_ceiling": (
                    fresh_r1.IMPLEMENTATION_TOKEN_CEILING
                ),
            },
            "runner": runner_identity,
            "execution": _execution_identity(config),
            "portfolio": _portfolio_identity(config),
            "round_bound": MAX_STANDALONE_ROUNDS,
            "metric_contract_digest": sha256_digest(COMMON_EVALUATOR),
        }
    )


def _runner_identity(runner: FreshExperimentRunner) -> dict[str, Any]:
    config = runner.config
    identity = {
        "run_id": config.run_id,
        "execution_purpose": config.execution_purpose,
        "seed": config.seed,
        "epochs": config.epochs,
        "timeout_seconds": config.timeout_seconds,
        "watchdog_seconds": config.watchdog_seconds,
        "cuda_visible_devices": config.cuda_visible_devices,
        "final_worker_ceiling_seconds": config.final_worker_ceiling_seconds,
        "resource_telemetry": config.resource_telemetry,
        "resource_profile_by_capability": (
            config.resource_prediction_by_capability
        ),
        "require_gpu_reservation_evidence": (
            config.require_gpu_reservation_evidence
        ),
        "gpu_reservation_provider_identity": (
            runner.gpu_reservation_provider_identity()
        ),
        "candidate_root_by_capability": {
            key: str(value)
            for key, value in config.candidate_root_by_capability.items()
        },
    }
    if config.gpu_id is not None:
        identity["gpu_id"] = config.gpu_id
    return canonical_value(identity)


def _profile_source_identity(
    source: ResearchProfileSourceV1 | None,
) -> Mapping[str, str] | None:
    if source is None:
        return None
    return canonical_value(source.identity)


def _execution_identity(config: StandaloneResearchConfig) -> dict[str, Any]:
    identity = {
        "attempt_scheduler": config.attempt_scheduler,
        "max_attempts_per_round": config.max_attempts_per_round,
        "candidate_handoff_factory_enabled": (
            config.candidate_handoff_factory is not None
        ),
        "profile_source_identity": _profile_source_identity(
            config.research_profile_source
        ),
        "qualified_execution_by_capability": (
            config.qualified_execution_by_capability
        ),
        "observation_seed_schedule": _seed_schedule_identity(config),
        "cuda_visible_devices": config.cuda_visible_devices,
        "final_worker_ceiling_seconds": config.final_worker_ceiling_seconds,
        "gpu_reservation_evidence_provider_enabled": (
            config.gpu_reservation_evidence_provider is not None
        ),
        "require_gpu_reservation_evidence": (
            config.require_gpu_reservation_evidence
        ),
        "gpu_reservation_provider_identity": _gpu_provider_identity(
            config.gpu_reservation_evidence_provider
        ),
        "research_consumer_composition": _research_consumer_composition_identity(
            config
        ),
    }
    if config.gpu_id is not None:
        identity["gpu_id"] = config.gpu_id
    return canonical_value(identity)


def _portfolio_identity(
    config: StandaloneResearchConfig,
) -> dict[str, Any]:
    return canonical_value(
        {
            "enabled": bool(
                config.portfolio_candidates
                or config.research_profile_source is not None
                or config.candidate_handoff_factory is not None
            ),
            "mode": (
                "PROFILE_SOURCE"
                if config.research_profile_source is not None
                else (
                "DYNAMIC_HANDOFF_FACTORY"
                if config.candidate_handoff_factory is not None
                else "STATIC_COMPATIBILITY"
                )
            ),
            "candidate_ids": tuple(
                item.candidate_id for item in config.portfolio_candidates
            ),
            "profiles": tuple(
                item.to_dict() for item in config.portfolio_candidates
            ),
            "profile_source_identity": _profile_source_identity(
                config.research_profile_source
            ),
        }
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise StandaloneCampaignError(f"file already exists: {path}") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StandaloneCampaignError(f"cannot read JSON manifest: {path}") from error
    if not isinstance(value, Mapping):
        raise StandaloneCampaignError(f"manifest root must be an object: {path}")
    return value


def _validate_manifest_for_resume(
    *,
    config: StandaloneResearchConfig,
    manifest: Mapping[str, Any],
    state: CampaignState,
    provider: ProviderResearchProducer,
    runner: FreshExperimentRunner,
) -> None:
    if manifest.get("schema") != STANDALONE_SCHEMA:
        raise StandaloneCampaignError("standalone campaign manifest schema drift")
    if manifest.get("controller") != "ResearchCampaign":
        raise StandaloneCampaignError("campaign is not owned by ResearchCampaign")
    expected_identity = _source_baseline_identity(config)
    if manifest.get("source_baseline_identity") != expected_identity:
        raise StandaloneCampaignError(
            "Research baseline/source identity differs from the sealed manifest"
        )
    if manifest.get("observation_seed_schedule") != _seed_schedule_identity(
        config
    ):
        raise StandaloneCampaignError(
            "observation seed schedule differs from the sealed manifest"
        )
    if manifest.get("campaign_id") != config.campaign_id:
        raise StandaloneCampaignError("campaign_id differs from the sealed manifest")
    if config.baseline_source.seed != config.seed:
        raise StandaloneCampaignError(
            "baseline source seed differs from the requested Research seed"
        )
    if state.context.protocol_digest != config.baseline_source.protocol_digest:
        raise StandaloneCampaignError(
            "baseline source protocol differs from the resumed CampaignState"
        )
    provider_manifest = manifest.get("provider")
    if not isinstance(provider_manifest, Mapping):
        raise StandaloneCampaignError("standalone manifest lacks provider identity")
    if provider_manifest.get("config_identity") != provider.config_identity:
        raise StandaloneCampaignError("Provider config identity differs on resume")
    runner_manifest = manifest.get("runner")
    if not isinstance(runner_manifest, Mapping):
        raise StandaloneCampaignError("standalone manifest lacks runner identity")
    if runner_manifest != _runner_identity(runner):
        raise StandaloneCampaignError("runner identity differs on resume")
    execution_manifest = manifest.get("execution")
    if execution_manifest != _execution_identity(config):
        raise StandaloneCampaignError("campaign execution inputs differ on resume")
    if manifest.get("portfolio") != _portfolio_identity(config):
        raise StandaloneCampaignError("portfolio profile set differs on resume")
    state_identity = state.incumbent_observation
    if (
        state_identity.get("source_ref")
        != config.baseline_source.source_ref
        or state_identity.get("source_receipt_sha256")
        != config.baseline_source.source_sha256
    ):
        raise StandaloneCampaignError(
            "CampaignState baseline/source identity differs from the requested resume"
        )
    global_memory = state.context.scientific_memory.get("global_memory")
    if (
        not isinstance(global_memory, Mapping)
        or global_memory.get("standalone_identity") != expected_identity
    ):
        raise StandaloneCampaignError(
            "CampaignState standalone identity differs from the sealed manifest"
        )


def compose_standalone_campaign(
    config: StandaloneResearchConfig,
    *,
    resume: bool = False,
    provider_call: ProviderCall | None = None,
    launch: Launcher | None = None,
    candidate_handoff_factory: CandidateHandoffFactory | None = None,
    research_profile_source: ResearchProfileSourceV1 | None = None,
) -> StandaloneResearchComposition:
    """Compose a new or resumed standalone ResearchCampaign.

    ``research_profile_source`` is the stable, identity-bound portfolio
    boundary.  ``candidate_handoff_factory`` and ``portfolio_candidates`` remain
    compatibility inputs for callers that have not adopted the source bundle.
    ``provider_call`` and ``launch`` are injectable only at the existing
    Provider/runner boundaries.
    """

    if not isinstance(config, StandaloneResearchConfig):
        raise TypeError("config must be StandaloneResearchConfig")
    if candidate_handoff_factory is not None:
        if (
            config.candidate_handoff_factory is not None
            and config.candidate_handoff_factory is not candidate_handoff_factory
        ):
            raise StandaloneCampaignError(
                "candidate_handoff_factory is specified twice with different callables"
            )
        config = replace(
            config,
            candidate_handoff_factory=candidate_handoff_factory,
        )
    if research_profile_source is not None:
        if (
            config.research_profile_source is not None
            and config.research_profile_source.identity
            != research_profile_source.identity
        ):
            raise StandaloneCampaignError(
                "research_profile_source is specified twice with different identities"
            )
        config = replace(
            config,
            research_profile_source=research_profile_source,
        )
    if resume:
        if not config.run_root.is_dir():
            raise StandaloneCampaignError(
                f"cannot resume missing run root: {config.run_root}"
            )
        state_path = config.run_root / ResearchCampaign.state_filename
        manifest_path = config.run_root / "STANDALONE_CAMPAIGN_MANIFEST.json"
        if not state_path.is_file() or not manifest_path.is_file():
            raise StandaloneCampaignError(
                "resume requires both CAMPAIGN_STATE.pkl and "
                "STANDALONE_CAMPAIGN_MANIFEST.json"
            )
        manifest = _read_json(manifest_path)
        if manifest.get("execution") != _execution_identity(config):
            raise StandaloneCampaignError(
                "campaign execution inputs differ on resume"
            )
        provider, implementer, runner = _build_provider_objects(
            config,
            provider_call=provider_call,
            launch=launch,
        )
        campaign = ResearchCampaign.resume(
            root=config.run_root,
            producer=provider,
            runner=runner,
            round_inputs=_round_input_factory(
                config,
                provider=provider,
                implementer=implementer,
                portfolio_candidates=config.portfolio_candidates,
            ),
            implementer=implementer,
            memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        )
        _validate_manifest_for_resume(
            config=config,
            manifest=manifest,
            state=campaign.state,
            provider=provider,
            runner=runner,
        )
        return StandaloneResearchComposition(
            config=config,
            campaign=campaign,
            provider=provider,
            implementer=implementer,
            runner=runner,
            profile=campaign.state.active_profile,
            policy=campaign.state.policy,
            manifest=manifest,
        )

    if config.run_root.exists():
        raise StandaloneCampaignError(
            "start requires a new run_root; use --resume for an existing campaign"
        )
    provider, implementer, runner = _build_provider_objects(
        config,
        provider_call=provider_call,
        launch=launch,
    )
    profile = adapt_current_search_profile(campaign_id=config.campaign_id)
    policy = meta_v20_research_control_policy()
    if config.baseline_source.seed != config.seed:
        raise StandaloneCampaignError(
            "baseline source seed differs from the requested Research seed"
        )
    if config.baseline_source.protocol_digest != profile.protocol_digest:
        raise StandaloneCampaignError(
            "baseline source protocol differs from the active Research profile"
        )
    incumbent = _initial_incumbent(config)
    context = _initial_context(
        config=config,
        profile=profile,
        policy=policy,
        incumbent=incumbent,
    )
    state = CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=incumbent,
        carryover_proposals=bootstrap_search_pool(context, profile, policy),
        qualified_execution_by_capability=config.qualified_execution_by_capability,
        candidate_root_by_capability={
            key: str(value)
            for key, value in config.candidate_root_by_capability.items()
        },
        resource_profile_by_capability=config.resource_profile_by_capability,
        frontier=context.frontier,
    )
    manifest = _manifest(
        config=config,
        profile=profile,
        policy=policy,
        context=context,
        provider=provider,
        runner=runner,
    )
    config.run_root.mkdir(parents=True, exist_ok=False)
    _write_new_json(config.run_root / "STANDALONE_CAMPAIGN_MANIFEST.json", manifest)
    campaign = ResearchCampaign.start(
        root=config.run_root,
        state=state,
        producer=provider,
        runner=runner,
        round_inputs=_round_input_factory(
            config,
            provider=provider,
            implementer=implementer,
            portfolio_candidates=config.portfolio_candidates,
        ),
        implementer=implementer,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )
    return StandaloneResearchComposition(
        config=config,
        campaign=campaign,
        provider=provider,
        implementer=implementer,
        runner=runner,
        profile=profile,
        policy=policy,
        manifest=manifest,
    )


def load_portfolio_candidates(path: Path) -> tuple[PortfolioCandidateV2, ...]:
    """Load a frozen, complete portfolio profile set without filling fields.

    This loader is intentionally strict because a profile activates the
    portfolio route.  Every dataclass field must be present in the source,
    including resource, parent, age, repeat, and correlated-compute state.
    """

    path = Path(path).resolve()
    payload = _read_json(path)
    raw_candidates = payload.get("candidates")
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, (tuple, list)
    ):
        raise StandaloneCampaignError(
            "portfolio profile JSON must contain a candidates array"
        )
    required = {
        "candidate_id",
        "semantic_digest",
        "family_id",
        "parent_id",
        "valid_seal_probability",
        "family_delta",
        "parent_delta",
        "information_value",
        "predicted_gpu_seconds",
        "age_rounds",
        "repeat_count",
        "lineage_risk",
        "compute_pattern",
        "resource_admission_state",
        "parent_state",
        "task_priority",
        "frontier_gain",
        "correlated_compute_risk",
        "dominated_by",
        "parent_rebound",
    }
    result: list[PortfolioCandidateV2] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise StandaloneCampaignError(
                f"portfolio candidate {index} must be an object"
            )
        missing = sorted(required - set(raw))
        if missing:
            raise StandaloneCampaignError(
                f"portfolio candidate {index} is incomplete: {', '.join(missing)}"
            )
        try:
            result.append(PortfolioCandidateV2.from_mapping(raw))
        except (TypeError, ValueError) as error:
            raise StandaloneCampaignError(
                f"portfolio candidate {index} is invalid"
            ) from error
    return tuple(result)


def load_research_profile_source(path: Path) -> ResearchProfileSourceV1:
    """Load the explicit stable-identity Research profile source bundle."""

    payload = _read_json(Path(path).resolve())
    try:
        return ResearchProfileSourceV1.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise StandaloneCampaignError(
            "profile source JSON is invalid or has identity drift"
        ) from error


__all__ = [
    "MAX_STANDALONE_ROUNDS",
    "STANDALONE_SCHEMA",
    "StandaloneCampaignError",
    "StandaloneResearchComposition",
    "StandaloneResearchConfig",
    "compose_standalone_campaign",
    "load_portfolio_candidates",
    "load_research_profile_source",
]
