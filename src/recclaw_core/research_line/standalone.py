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

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.provider_model_routing import (
    model_routing_manifest,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_scientific_profile_ref,
)
from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (
    MAX_REPAIR_TURNS,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    MechanicalRecBoleAdapterV1,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
    P4_SPARSE_SPECTRAL_EVALUATOR,
    P4_SPARSE_SPECTRAL_SPLIT,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    CAMPAIGN_DEVELOPMENT_TRAINING_RELEASE_RESOURCE,
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
    NATIVE_EVAL_STEP,
    NATIVE_STOPPING_STEP,
    run_disposable_fixed_batch_resource_probe,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchProfileActivationV1,
    SearchExecutableProfileV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import CapabilityKindV1
from recclaw_core.experiments.helix_abc_v1.realization_identity import (
    bl_icf_search_space_conformance,
)
from recclaw_core.search_spaces.bl_icf_v1 import PROVIDER as BL_ICF_PROVIDER

from .campaign import (
    CampaignRoundInputs,
    CampaignState,
    MAX_DISCOVERY_GENERATIONS_PER_ROUND,
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
from .interfaces import ResearchContext, ResearchTaskQueueV2, research_producer_roles
from .portfolio import PortfolioCandidateV2
from .profile_source import ResearchProfileSourceV1
from .provider import (
    ConfigSource,
    PROPOSAL_LANE_OPEN_SPEC,
    ProviderCall,
    ProviderImplementerGateway,
    ProviderResearchProducer,
    RESEARCH_PROPOSAL_OUTPUT_TOKEN_CEILING,
    RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING,
    load_provider_proposal_replay,
    provider_proposal_replay_identity,
)
from .replay import OfflineProducerReplayV1
from .search_space_adapter import SearchSpaceAdapter
from .runtime import (
    CandidateHandoffFactory,
    EvidenceGuardPort,
    InnovationRuntimeInputs,
    InitialSearchPoolResult,
    MetaResearchInputs,
    PreparedResearchRoundV1,
    _default_search_space_adapter,
    _prepared_has_transient_producer_failure,
    _prepared_has_unfinished_producer_failure,
    bindings_for_context,
    resolver_environment_for_profile,
)
from .single_round import (
    ResearchBaselineSourceV1,
    _bytes_sha256,
    validated_search_partition_identity,
)
from .single_parent_search import (
    is_bl_icf_single_parent_context,
    is_single_parent_context,
    single_parent_research_axes,
    single_parent_research_axis_questions,
    validate_single_parent_runtime_context,
)


class StandaloneCampaignError(ValueError):
    """Raised when a standalone campaign cannot be composed or resumed safely."""


MAX_STANDALONE_ROUNDS = 200
STANDALONE_SCHEMA = "recclaw.research-line.standalone-campaign.v1"
STANDALONE_EFFECTIVE_MANIFEST_FILENAME = (
    "STANDALONE_CAMPAIGN_EFFECTIVE_MANIFEST.json"
)
STANDALONE_EFFECTIVE_MANIFEST_MIGRATION_PREFIX = (
    "STANDALONE_CAMPAIGN_EFFECTIVE_MANIFEST_MIGRATION_"
)
_IMPLEMENTER_CEILING_MIGRATION_SCHEMA = (
    "recclaw.research-line.implementer-ceiling-resume-migration.v1"
)
_REPAIR39_IMPLEMENTATION_TOTAL_TOKEN_CEILING = 32_000
_REPAIR39_IMPLEMENTATION_OUTPUT_TOKEN_CEILING = 20_000
_REPAIR40_IMPLEMENTATION_TOKEN_CEILING = 64_000


@dataclass(frozen=True, slots=True)
class StandaloneControllerInterface:
    """Controller-owned proposal/selection hooks over the shared substrate."""

    identity: Mapping[str, Any]
    provider: ProviderResearchProducer
    router: Any
    enable_meta_research: bool = True
    initial_state_transition: (
        Callable[[CampaignState], CampaignState] | None
    ) = None
    post_round_state_transition: (
        Callable[[CampaignState, Any, int, str], CampaignState] | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Mapping):
            raise StandaloneCampaignError("controller identity must be a mapping")
        object.__setattr__(self, "identity", canonical_value(dict(self.identity)))
        if not isinstance(self.provider, ProviderResearchProducer):
            raise StandaloneCampaignError(
                "controller provider must reuse ProviderResearchProducer"
            )
        if not callable(getattr(self.router, "route", None)):
            raise StandaloneCampaignError("controller router must expose route()")
        if not isinstance(self.enable_meta_research, bool):
            raise StandaloneCampaignError(
                "enable_meta_research must be a boolean"
            )
META_REPLAY_INTERVAL = 10
RESEARCH_CONSUMER_COMPOSITION_SCHEMA = (
    "recclaw.research-line.standalone-research-consumers.v1"
)
RESOURCE_COMPUTE_PATTERN_SCHEMA = (
    "recclaw.research-line.standalone-compute-pattern.v1"
)
STRICT_BL_ICF_PROTOCOL_REF = "recclaw.campaign.ml1m-full-sort.v1"
STRICT_BL_ICF_PROTOCOL_DIGEST = (
    "7f623fd953001f999e8b5d2657749f6a3ca86c7be5410a48bb8281241a258bbe"
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


def _remaining_campaign_checkpoint_slots(*, round_count: int, round_index: int) -> int:
    """Return the current-and-future worker slots still able to retain state."""

    return max(1, round_count - round_index + 1)


def _meta_replay_due(round_index: int) -> bool:
    # Meta strategy promotion is a campaign-boundary operation.  Round one has
    # no outcome history to replay and must never mutate discovery policy.
    return round_index >= META_REPLAY_INTERVAL and round_index % META_REPLAY_INTERVAL == 0


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


def _bind_profile_cadence(
    execution_recipe: Mapping[str, Any],
    *,
    epochs: int,
    eval_step: int,
    stopping_step: int,
) -> Mapping[str, Any]:
    """Put the frozen Profile cadence in the sole downstream recipe contract."""

    recipe_config = execution_recipe.get("config")
    if not isinstance(recipe_config, Mapping):
        raise StandaloneCampaignError(
            "Research Innovation execution recipe requires config"
        )
    cadence = {
        "epochs": epochs,
        "eval_step": eval_step,
        "stopping_step": stopping_step,
    }
    mismatched = [
        field_name
        for field_name, expected in cadence.items()
        if field_name in recipe_config and recipe_config[field_name] != expected
    ]
    if mismatched:
        raise StandaloneCampaignError(
            "execution recipe cadence differs from the frozen Profile: "
            + ", ".join(mismatched)
        )
    return canonical_value(
        {
            **dict(execution_recipe),
            "config": {**dict(recipe_config), **cadence},
        }
    )


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
                "resource_admission_required": not config.native_resource_observation,
                "gpu_reservation_evidence_required": (
                    config.require_gpu_reservation_evidence
                ),
                "resource_policy": (
                    "NATIVE_TRAINING_RESOURCE_OBSERVATION_V1"
                    if config.native_resource_observation
                    else "DISPOSABLE_FIXED_BATCH_RESOURCE_PROFILE_V1"
                ),
                "probe_physical_identity_derivation": (
                    None if config.native_resource_observation
                    else "candidate_ref_sha256_prefix16_arm_v1"
                ),
                "final_worker_ceiling_seconds": (
                    config.final_worker_ceiling_seconds
                ),
                "prediction_and_gpu_seconds_are_separate": True,
            },
            "meta": {
                "first_round": 1,
                "interval_rounds": META_REPLAY_INTERVAL,
                "equal_replay_token_charge": (
                    RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING
                ),
                "deterministic_directive_replay": True,
            },
            "token_ceilings": {
                "proposal_total": RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING,
                "proposal_output": RESEARCH_PROPOSAL_OUTPUT_TOKEN_CEILING,
                "implementation": (
                    config.implementation_total_token_ceiling_per_call
                ),
                "implementation_output": (
                    config.implementation_total_token_ceiling_per_call
                ),
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
    search_space_adapter: SearchSpaceAdapter | None = None,
    campaign_id: str,
    seed: int,
    round_index: int,
    discovery_generation: int = 0,
    predecessor_registry_ref: str | None = None,
    predecessor_registry_digest: str | None = None,
) -> InnovationRuntimeInputs:
    if predecessor_registry_ref is None or predecessor_registry_digest is None:
        if (predecessor_registry_ref is None) != (
            predecessor_registry_digest is None
        ):
            raise StandaloneCampaignError(
                "predecessor registry ref and digest must be supplied together"
            )
        initial_profile = _strict_bl_icf_provider_profile(config)
        predecessor_registry_ref = initial_profile.profile_ref
        predecessor_registry_digest = initial_profile.profile_digest
    generation_suffix = (
        "" if discovery_generation == 0 else f":generation-{discovery_generation:04d}"
    )
    generation_path = (
        Path(f"round-{round_index:04d}")
        if discovery_generation == 0
        else Path(f"round-{round_index:04d}")
        / f"generation-{discovery_generation:04d}"
    )
    prompt = implementer.implementation_template_for_search_space(None)
    allowed_files = tuple(
        getattr(
            search_space_adapter,
            "implementation_allowed_files",
            (
                "recclaw_ext/__init__.py",
                "recclaw_ext/candidate.py",
                "recclaw_ext/trainer.py",
            ),
        )
    )
    policy = fresh_r1._shared_policy(
        sha256_digest(prompt),
        sha256_digest(
            {
                "tools": (),
                "network": False,
                "allowed_files": allowed_files,
            }
        ),
        allowed_files=allowed_files,
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
        candidate_root: Path,
    ) -> Any:
        execution_contract = contract(value)
        fixture = fresh_r1._qualification_fixture(
            config.repo_root,
            seed=seed,
            root=(
                config.run_root
                / "innovation_qualification"
                / generation_path
                / candidate_root.parent.name
            ),
            base_model_config=str(execution_contract["base_model_config"]),
        )
        return replace(
            fixture,
            runtime_identity_ref=value.runtime_identity_ref,
            runtime_identity_digest=value.runtime_identity_digest,
        )

    def unit_check_factory(value: Any) -> Any:
        adapter_check = getattr(
            search_space_adapter,
            "qualification_unit_check",
            None,
        )
        if callable(adapter_check):
            execution_contract = contract(value)
            return adapter_check(
                base_model_config=str(execution_contract["base_model_config"]),
                execution_contract=execution_contract,
            )
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
        execution_recipe = _bind_profile_cadence(
            execution_recipe,
            epochs=config.epochs,
            eval_step=config.native_eval_step,
            stopping_step=config.native_stopping_step,
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
        # The native offline-credit path requires an actual reservation.
        # Shared-GPU E1 work is charged by its supervisor and uses the native
        # unreserved wall-time profile, never fabricated exclusive evidence.
        exclude_reserved_probe = reservation_evidence is not None
        probe_kwargs = dict(kwargs)
        probe_kwargs["execution_recipe"] = execution_recipe
        for field_name in (
            "campaign_checkpoint_horizon_slots",
            "compute_pattern",
            "cuda_visible_devices",
            "gpu_reservation_evidence",
            "total_budget_seconds",
            "probe_seed",
        ):
            probe_kwargs.pop(field_name, None)
        from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
            _completed_probe_request_identity,
            PROBE_TIMEOUT_SECONDS,
        )
        if "probe_timeout_seconds" not in probe_kwargs:
            from .p4_runtime import p4_fit_mode

            probe_kwargs["probe_timeout_seconds"] = (
                min(1800, config.final_worker_ceiling_seconds)
                if execution_recipe.get("split") == P4_SPARSE_SPECTRAL_SPLIT
                and p4_fit_mode(execution_recipe.get("config", {})) == "TRAIN_ONLY_PRECOMPUTE"
                else PROBE_TIMEOUT_SECONDS
            )

        probe_identity = _completed_probe_request_identity(
            arm_id="", candidate_ref=str(candidate_ref),
            candidate_package_digest=candidate_package_digest,
            candidate_binding_digest=probe_kwargs.get("candidate_binding_digest"),
            compute_pattern=compute_pattern,
            entrypoint=probe_kwargs["entrypoint"], source_sha256=probe_kwargs["source_sha256"],
            execution_recipe=execution_recipe,
            probe_seed=seed, probe_timeout_seconds=probe_kwargs["probe_timeout_seconds"],
            total_budget_seconds=config.final_worker_ceiling_seconds,
            cuda_visible_devices=config.cuda_visible_devices, gpu_id=config.gpu_id,
            offline_calibration_probe_excluded_from_future_budget=exclude_reserved_probe,
            expected_recbole_source_tree_digest=fresh_r1.recbole_source_identity(fresh_r1.RECBole_ROOT)["source_tree_digest"],
        )
        probe_kwargs["probe_root"] = config.run_root / "innovation_resource_probes" / ("probe-" + sha256_digest(probe_identity))
        probe_runner = (getattr(search_space_adapter, 'resource_probe_runner', None)
                        or run_disposable_fixed_batch_resource_probe)
        return probe_runner(
            config.repo_root,
            total_budget_seconds=config.final_worker_ceiling_seconds,
            probe_seed=seed,
            compute_pattern=compute_pattern,
            cuda_visible_devices=config.cuda_visible_devices,
            gpu_id=config.gpu_id,
            gpu_reservation_evidence=reservation_evidence,
            search_data_identity=_active_search_data_identity(config),
            campaign_checkpoint_horizon_slots=_remaining_campaign_checkpoint_slots(
                round_count=config.round_count,
                round_index=round_index,
            ),
            offline_calibration_probe_excluded_from_future_budget=(
                reservation_evidence is not None
            ),
            **({"process_launcher": search_space_adapter.process_launcher}
               if getattr(search_space_adapter, "process_launcher", None) is not None else {}),
            **probe_kwargs,
        )

    return InnovationRuntimeInputs(
        implementer=implementer,
        policy=policy,
        candidate_parent=(
            config.run_root
            / "innovation_candidates"
            / generation_path
        ),
        fixture_factory=fixture_factory,
        unit_check_factory=unit_check_factory,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version=(
            f"{campaign_id}:round-{round_index:04d}{generation_suffix}:qualified-v1"
        ),
        registry_version=(
            f"{campaign_id}:round-{round_index:04d}{generation_suffix}:registry-v1"
        ),
        predecessor_registry_ref=predecessor_registry_ref,
        predecessor_registry_digest=predecessor_registry_digest,
        profile_version=(
            f"{campaign_id}:round-{round_index:04d}{generation_suffix}:"
            "next-fresh-profile-v1"
        ),
        fresh_campaign_id=(
            f"{campaign_id}:round-{round_index:04d}{generation_suffix}:next-fresh"
        ),
        resource_admission_required=not config.native_resource_observation,
        resource_probe=None if config.native_resource_observation else resource_probe,
        resource_probe_parent=(
            None if config.native_resource_observation else config.run_root
            / "innovation_resource_probes"
            / generation_path
        ),
        qualification_executor=(
            getattr(search_space_adapter, "qualification_executor", None)
            or (MechanicalRecBoleAdapterV1().qualify_disposable
                if config.native_resource_observation else None)
        ),
        native_training_cadence=(
            {"epochs": config.epochs, "eval_step": config.native_eval_step,
             "stopping_step": config.native_stopping_step}
            if config.native_resource_observation else None
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
    search_data_identity: Mapping[str, Any] | None = None
    dataset: str = COMMON_DATASET
    native_eval_step: int = NATIVE_EVAL_STEP
    native_stopping_step: int = NATIVE_STOPPING_STEP
    implementation_requirements: tuple[str, ...] = _IMPLEMENTATION_REQUIREMENTS
    compatibility_requirements: tuple[str, ...] = _COMPATIBILITY_REQUIREMENTS
    protocol_requirements: tuple[str, ...] = fresh_r1.PROTOCOL_REQUIREMENTS
    available_dependencies: tuple[str, ...] = fresh_r1.AVAILABLE_DEPENDENCIES
    allow_resume_source_sha256_drift: bool = False
    allow_resume_endpoint1_to_endpoint2_and_reasoning_drift: bool = False
    allow_resume_endpoint2_to_endpoint1: bool = False
    allow_resume_exhausted_engineering_generation: bool = False
    seed: int = 54201
    search_seed: int | None = None
    epochs: int = fresh_r1.EXPERIMENT_EPOCHS
    timeout_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS
    watchdog_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS
    final_worker_ceiling_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS
    cuda_visible_devices: str | None = None
    observation_seed_schedule: tuple[int, ...] | None = None
    gpu_reservation_evidence_provider: GpuReservationEvidenceProvider | None = None
    require_gpu_reservation_evidence: bool = False
    research_mode: str = "portfolio"
    search_policy_mode: str = "adaptive"
    round_count: int = 1
    attempt_scheduler: bool = False
    max_attempts_per_round: int | None = None
    prebinding_token_ceiling_retry: bool = True
    close_exhausted_no_metric_slot: bool = False
    provider_maximum_physical_attempts: int | None = fresh_r1.MAX_PHYSICAL_ATTEMPTS
    implementation_total_token_ceiling_per_call: int = (
        fresh_r1.IMPLEMENTATION_TOKEN_CEILING
    )
    proposal_output_token_ceiling_total_per_slot: int | None = None
    implementation_output_token_ceiling_total_per_candidate: int | None = None
    max_implementation_calls_per_candidate: int = MAX_REPAIR_TURNS + 1
    evidence_port: EvidenceGuardPort | None = None
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
    evaluator: Mapping[str, Any] = field(
        default_factory=lambda: COMMON_EVALUATOR
    )
    split: str = COMMON_SPLIT
    frozen_profile_ref: Mapping[str, Any] = field(
        default_factory=campaign_scientific_profile_ref
    )
    protocol_ref: str | None = None
    protocol_digest: str | None = None
    execution_purpose: str = "RESEARCH_LINE_STANDALONE_BLICF"
    bootstrap_fixed_candidates: bool = False
    bootstrap_max_proposals: int = 4
    proposal_replay_run_root: Path | None = None
    baseline_context: Mapping[str, Any] = field(default_factory=dict)
    arm_code: str | None = None
    shared_implementation_root: Path | None = None
    # Observe the formal worker's native phases without a disposable prefix run.
    native_resource_observation: bool = False

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
        if not isinstance(self.native_resource_observation, bool):
            raise StandaloneCampaignError("native_resource_observation must be a boolean")
        if self.native_resource_observation and (
            self.research_profile_source is not None
            or self.candidate_handoff_factory is not None
        ):
            raise StandaloneCampaignError(
                "native resource observation uses direct candidate execution; "
                "predictive portfolio handoffs still require pre-training resource profiles"
            )
        if self.research_mode not in {"portfolio", "director_sequential"}:
            raise StandaloneCampaignError("research_mode must be portfolio or director_sequential")
        if self.search_policy_mode not in {"adaptive", "fixed"}:
            raise StandaloneCampaignError("search_policy_mode must be adaptive or fixed")
        object.__setattr__(
            self,
            "dataset",
            _text(self.dataset, field_name="dataset"),
        )
        if self.search_data_identity is not None:
            if not isinstance(self.search_data_identity, Mapping):
                raise StandaloneCampaignError(
                    "search_data_identity must be a mapping when supplied"
                )
            object.__setattr__(
                self,
                "search_data_identity",
                canonical_value(dict(self.search_data_identity)),
            )
        for field_name in ("native_eval_step", "native_stopping_step"):
            object.__setattr__(
                self,
                field_name,
                _positive_int(getattr(self, field_name), field_name=field_name),
            )
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
        if not isinstance(self.baseline_context, Mapping):
            raise StandaloneCampaignError("baseline_context must be a mapping")
        normalized_baseline_context = canonical_value(dict(self.baseline_context))
        if is_bl_icf_single_parent_context(normalized_baseline_context):
            try:
                normalized_baseline_context = validate_single_parent_runtime_context(
                    normalized_baseline_context,
                    baseline_seed=self.baseline_source.seed,
                    baseline_value=self.baseline_source.frozen_ndcg_at_10,
                    active_profile_ref=self.frozen_profile_ref,
                )
            except ValueError as error:
                raise StandaloneCampaignError(str(error)) from error
        if (
            is_single_parent_context(normalized_baseline_context)
            and self.bootstrap_fixed_candidates
        ):
            raise StandaloneCampaignError(
                "single-parent search cannot bootstrap unrelated fixed candidates"
            )
        object.__setattr__(
            self,
            "baseline_context",
            normalized_baseline_context,
        )
        if (self.arm_code is None) != (self.shared_implementation_root is None):
            raise StandaloneCampaignError(
                "arm_code and shared_implementation_root must be supplied together"
            )
        if self.arm_code is not None:
            if self.arm_code not in {"A", "B", "C"}:
                raise StandaloneCampaignError("arm_code must be A, B, or C")
            if not isinstance(self.shared_implementation_root, Path):
                raise StandaloneCampaignError(
                    "shared_implementation_root must be a Path"
                )
            object.__setattr__(
                self,
                "shared_implementation_root",
                self.shared_implementation_root.resolve(),
            )
        object.__setattr__(
            self,
            "seed",
            _nonnegative_int(self.seed, field_name="seed"),
        )
        object.__setattr__(
            self,
            "search_seed",
            _nonnegative_int(
                self.seed if self.search_seed is None else self.search_seed,
                field_name="search_seed",
            ),
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
        if not isinstance(self.evaluator, Mapping):
            raise StandaloneCampaignError("evaluator must be a mapping")
        normalized_evaluator = canonical_value(dict(self.evaluator))
        if not (
            (self.split == COMMON_SPLIT and normalized_evaluator == COMMON_EVALUATOR)
            or (self.split == P4_SPARSE_SPECTRAL_SPLIT and normalized_evaluator == P4_SPARSE_SPECTRAL_EVALUATOR)
            or (
                self.split == DEVELOPMENT_SPLIT
                and normalized_evaluator == DEVELOPMENT_EVALUATOR
            )
        ):
            raise StandaloneCampaignError(
                "split/evaluator must be one exact supported contract"
            )
        object.__setattr__(self, "evaluator", normalized_evaluator)
        if not isinstance(self.frozen_profile_ref, Mapping):
            raise StandaloneCampaignError("frozen_profile_ref must be a mapping")
        object.__setattr__(
            self,
            "frozen_profile_ref",
            canonical_value(dict(self.frozen_profile_ref)),
        )
        if (self.protocol_ref is None) != (self.protocol_digest is None):
            raise StandaloneCampaignError(
                "protocol_ref and protocol_digest must be supplied together"
            )
        if self.protocol_ref is not None:
            _text(self.protocol_ref, field_name="protocol_ref")
            _text(self.protocol_digest, field_name="protocol_digest")
        _text(self.execution_purpose, field_name="execution_purpose")
        if not isinstance(self.bootstrap_fixed_candidates, bool):
            raise StandaloneCampaignError(
                "bootstrap_fixed_candidates must be a boolean"
            )
        if self.bootstrap_fixed_candidates:
            raise StandaloneCampaignError(
                "standalone scientific execution forbids the legacy FIXED_66 "
                "bootstrap; strict BL-ICF Provider/compiler search is mandatory"
            )
        object.__setattr__(
            self,
            "bootstrap_max_proposals",
            _positive_int(
                self.bootstrap_max_proposals,
                field_name="bootstrap_max_proposals",
            ),
        )
        if self.bootstrap_max_proposals > 4:
            raise StandaloneCampaignError(
                "bootstrap_max_proposals must be <= 4"
            )
        if self.proposal_replay_run_root is not None:
            replay_root = Path(self.proposal_replay_run_root).resolve()
            if not replay_root.is_dir():
                raise StandaloneCampaignError(
                    f"proposal_replay_run_root is not a directory: {replay_root}"
                )
            try:
                load_provider_proposal_replay(replay_root)
            except (OSError, TypeError, ValueError) as error:
                raise StandaloneCampaignError(
                    f"proposal replay evidence is invalid: {error}"
                ) from error
            object.__setattr__(self, "proposal_replay_run_root", replay_root)
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
        if not isinstance(self.allow_resume_endpoint2_to_endpoint1, bool):
            raise StandaloneCampaignError(
                "allow_resume_endpoint2_to_endpoint1 must be a boolean"
            )
        if not isinstance(self.allow_resume_source_sha256_drift, bool):
            raise StandaloneCampaignError(
                "allow_resume_source_sha256_drift must be a boolean"
            )
        if not isinstance(
            self.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift,
            bool,
        ):
            raise StandaloneCampaignError(
                "allow_resume_endpoint1_to_endpoint2_and_reasoning_drift "
                "must be a boolean"
            )
        if not isinstance(
            self.allow_resume_exhausted_engineering_generation,
            bool,
        ):
            raise StandaloneCampaignError(
                "allow_resume_exhausted_engineering_generation must be a boolean"
            )
        if not isinstance(self.attempt_scheduler, bool):
            raise StandaloneCampaignError("attempt_scheduler must be a boolean")
        if not isinstance(self.prebinding_token_ceiling_retry, bool):
            raise StandaloneCampaignError(
                "prebinding_token_ceiling_retry must be a boolean"
            )
        if not isinstance(self.close_exhausted_no_metric_slot, bool):
            raise StandaloneCampaignError(
                "close_exhausted_no_metric_slot must be a boolean"
            )
        if self.close_exhausted_no_metric_slot and not self.attempt_scheduler:
            raise StandaloneCampaignError(
                "close_exhausted_no_metric_slot requires attempt_scheduler"
            )
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
        if self.provider_maximum_physical_attempts is not None and (
            isinstance(self.provider_maximum_physical_attempts, bool)
            or not isinstance(self.provider_maximum_physical_attempts, int)
            or self.provider_maximum_physical_attempts < 1
        ):
            raise StandaloneCampaignError(
                "provider_maximum_physical_attempts must be positive or None"
            )
        if (
            isinstance(self.implementation_total_token_ceiling_per_call, bool)
            or not isinstance(
                self.implementation_total_token_ceiling_per_call,
                int,
            )
            or self.implementation_total_token_ceiling_per_call
            < fresh_r1.IMPLEMENTATION_TOKEN_CEILING
        ):
            raise StandaloneCampaignError(
                "implementation_total_token_ceiling_per_call must be an integer "
                "at least as large as the implementation output ceiling"
            )
        for field_name in (
            "proposal_output_token_ceiling_total_per_slot",
            "implementation_output_token_ceiling_total_per_candidate",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise StandaloneCampaignError(
                    f"{field_name} must be a positive integer when supplied"
                )
        if (
            isinstance(self.max_implementation_calls_per_candidate, bool)
            or not isinstance(self.max_implementation_calls_per_candidate, int)
            or not 1 <= self.max_implementation_calls_per_candidate <= MAX_REPAIR_TURNS + 1
        ):
            raise StandaloneCampaignError(
                "max_implementation_calls_per_candidate is outside the bounded policy"
            )
        if self.evidence_port is not None:
            if not self.attempt_scheduler:
                raise StandaloneCampaignError(
                    "Evidence Guard requires the bounded Research attempt scheduler"
                )
            if not callable(getattr(self.evidence_port, "pre_run", None)) or not callable(
                getattr(self.evidence_port, "post_run", None)
            ):
                raise StandaloneCampaignError(
                    "evidence_port must expose callable pre_run and post_run"
                )
            guard_schedule = getattr(
                self.evidence_port, "validation_seed_schedule", None
            )
            if schedule is None or guard_schedule is None:
                raise StandaloneCampaignError(
                    "Evidence Guard requires frozen discovery and verification seed schedules"
                )
            discovery_seeds = {str(item) for item in schedule}
            verification_seeds = {str(item) for item in guard_schedule}
            if not discovery_seeds.issubset(verification_seeds):
                raise StandaloneCampaignError(
                    "Evidence Guard verification schedule must include every discovery seed"
                )
            for identity_field in (
                "expected_dataset_manifest_digest",
                "minimum_effect_delta",
            ):
                if getattr(self.evidence_port, identity_field, None) is None:
                    raise StandaloneCampaignError(
                        f"Evidence Guard identity lacks {identity_field}"
                    )
            guard_comparator = getattr(self.evidence_port, "comparator", None)
            if (
                guard_comparator is not None
                and str(guard_comparator) != self.baseline_source.comparator_ref
            ):
                raise StandaloneCampaignError(
                    "Evidence Guard comparator must equal the baseline comparator"
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

        if self.config.search_policy_mode == "fixed":
            from .fixed_search_policy import fixed_search_policy

            with fixed_search_policy(self):
                return self._run(round_count)
        return self._run(round_count)

    def _run(self, round_count: int | None) -> tuple[Any, ...]:

        count = self.config.round_count if round_count is None else round_count
        count = _bounded_round_count(count)
        results: list[Any] = []
        for _ in range(count):
            transient_provider_failures = 0
            last_transient_prepared_digest: str | None = None
            terminal_worker_recovery_boundary: tuple[int, Any] | None = None
            while True:
                before_state = self.campaign.state
                before = before_state.next_round_index
                before_generation = before_state.context.scientific_memory.get(
                    "discovery_generation", 0
                )
                if before > self.config.round_count:
                    return tuple(results)
                result = self.campaign.run_round()
                after_state = self.campaign.state
                after = after_state.next_round_index
                after_generation = after_state.context.scientific_memory.get(
                    "discovery_generation", 0
                )
                if (
                    result.result.incomplete_reason
                    == "ROUND_EXISTING_OBSERVATION_REBOUND"
                ):
                    # The exact task was closed without a runner or a new
                    # metric. Continue immediately into the one generation
                    # allowance recorded by the campaign state transition.
                    continue
                if after == before and after_generation == before_generation + 1:
                    # Campaign owns the slot, its bounded discovery-generation
                    # budget, and the typed negative fed to the next Provider
                    # call. Continue the same logical round in this process so
                    # a recoverable proposal failure does not terminate an
                    # otherwise autonomous multi-round campaign.
                    continue
                recovery_boundary = (before, before_generation)
                if (
                    self.config.close_exhausted_no_metric_slot
                    and after == before
                    and after_generation == before_generation
                    and result.result.incomplete_reason
                    in {
                        "ROUND_ATTEMPT_WORKER_TRANSIENT_STOP",
                        "ROUND_ATTEMPT_SHARED_INFRASTRUCTURE_STOP",
                    }
                    and terminal_worker_recovery_boundary != recovery_boundary
                ):
                    # The first call made the terminal physical observation
                    # durable. Re-enter once at the exact same boundary so the
                    # campaign can absorb it without replaying Provider or
                    # worker work and advance its discovery generation.
                    terminal_worker_recovery_boundary = recovery_boundary
                    continue
                prepared = getattr(result.result, "prepared", None)
                if (
                    after == before
                    and isinstance(prepared, PreparedResearchRoundV1)
                    and not _prepared_has_transient_producer_failure(prepared)
                    and _prepared_has_unfinished_producer_failure(prepared)
                ):
                    failures = "; ".join(
                        f"{outcome.producer_role}: {outcome.failure_detail}"
                        for outcome in prepared.producer_outcomes
                        if outcome.spec is None and outcome.failure_detail
                    )
                    raise StandaloneCampaignError(
                        "Provider request stopped; completed research stages "
                        f"are retained for resume: {failures}"
                    )
                if (
                    after == before
                    and isinstance(prepared, PreparedResearchRoundV1)
                    and _prepared_has_transient_producer_failure(prepared)
                ):
                    # A Provider transport outage is neither a mechanism result
                    # nor a discovery generation. Keep the process alive and
                    # refresh the same prepared boundary with bounded backoff.
                    prepared_digest = getattr(prepared, "digest", None)
                    if (
                        isinstance(prepared_digest, str)
                        and prepared_digest == last_transient_prepared_digest
                    ):
                        raise StandaloneCampaignError(
                            "Provider transport retry made no durable progress"
                        )
                    last_transient_prepared_digest = (
                        prepared_digest
                        if isinstance(prepared_digest, str)
                        else None
                    )
                    delay_seconds = min(2**transient_provider_failures, 60)
                    transient_provider_failures += 1
                    time.sleep(delay_seconds)
                    continue
                results.append(result)
                if after == before:
                    return tuple(results)
                if after != before + 1:
                    raise StandaloneCampaignError(
                        "ResearchCampaign did not advance exactly one round"
                    )
                # Discovery stays discovery; explicit evidence work is caller-
                # scheduled. There is no automatic post-round auxiliary worker.
                break
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


def _strict_bl_icf_provider_profile(
    config: StandaloneResearchConfig,
) -> SearchExecutableProfileV1:
    """Return the strict Provider search identity with no executable fallback.

    The 264 registry primitives are a mechanism language, not 264 independently
    executable models.  The initial executable registry is therefore empty;
    candidates enter it only after implementation, mechanical qualification,
    and compiler-bound activation.
    """

    identity = BL_ICF_PROVIDER.identity().to_dict()
    protocol_ref = config.protocol_ref or STRICT_BL_ICF_PROTOCOL_REF
    protocol_digest = (
        config.protocol_digest or config.baseline_source.protocol_digest
    )
    conformance = bl_icf_search_space_conformance(
        space_identity=identity,
        profile_ref=config.frozen_profile_ref,
        fixed_fallback=False,
    )
    if conformance["ordered_primitive_ids_count"] != 264:
        raise StandaloneCampaignError(
            "strict BL-ICF registry no longer contains exactly 264 primitives"
        )
    return SearchExecutableProfileV1(
        campaign_id=config.campaign_id,
        profile_ref=str(conformance["search_space_id"]),
        profile_digest=str(conformance["search_space_digest"]),
        protocol_ref=protocol_ref,
        protocol_digest=protocol_digest,
        activation=SearchProfileActivationV1.CURRENT_FROZEN_CAMPAIGN,
        predecessor_campaign_id=None,
        predecessor_profile_ref=None,
        predecessor_profile_digest=None,
        entries=(),
    )


def _provider_profile(
    config: StandaloneResearchConfig,
    search_space_adapter: SearchSpaceAdapter | None,
) -> SearchExecutableProfileV1:
    if search_space_adapter is None:
        return _strict_bl_icf_provider_profile(config)
    protocol_digest = config.protocol_digest or config.baseline_source.protocol_digest
    protocol_ref = config.protocol_ref or f"protocol-digest:{protocol_digest}"
    return SearchExecutableProfileV1(
        campaign_id=config.campaign_id,
        profile_ref=search_space_adapter.adapter_id,
        profile_digest=sha256_digest(
            {
                "adapter_id": search_space_adapter.adapter_id,
                "protocol_ref": protocol_ref,
                "protocol_digest": protocol_digest,
            }
        ),
        protocol_ref=protocol_ref,
        protocol_digest=protocol_digest,
        activation=SearchProfileActivationV1.CURRENT_FROZEN_CAMPAIGN,
        predecessor_campaign_id=None,
        predecessor_profile_ref=None,
        predecessor_profile_digest=None,
        entries=(),
    )


def _search_space_conformance(
    config: StandaloneResearchConfig,
    *,
    profile: SearchExecutableProfileV1,
    search_space_adapter: SearchSpaceAdapter | None,
) -> dict[str, Any]:
    if search_space_adapter is not None:
        return canonical_value(
            {
                "adapter_id": search_space_adapter.adapter_id,
                "search_space_id": profile.profile_ref,
                "search_space_digest": profile.profile_digest,
                "active_executable_capability_count": len(profile.entries),
                "fixed_fallback": False,
            }
        )
    return bl_icf_search_space_conformance(
        space_identity=BL_ICF_PROVIDER.identity().to_dict(),
        profile_ref=config.frozen_profile_ref,
        fixed_fallback=False,
    )


def _single_parent_research_policy(
    policy: VersionedResearchPolicyV1,
    baseline_context: Mapping[str, Any],
) -> VersionedResearchPolicyV1:
    profile_axes = single_parent_research_axes(baseline_context)
    return (
        replace(policy, mechanism_axis_targeting=profile_axes)
        if profile_axes
        else policy
    )


def _initial_context(
    *,
    config: StandaloneResearchConfig,
    profile: SearchExecutableProfileV1,
    policy: VersionedResearchPolicyV1,
    incumbent: Mapping[str, Any],
    search_space_adapter: SearchSpaceAdapter | None = None,
) -> ResearchContext:
    profile_questions = single_parent_research_axis_questions(
        config.baseline_context
    )
    identity = _source_baseline_identity(config)
    generic_search_space = search_space_adapter is not None
    search_space_id = (
        search_space_adapter.adapter_id
        if search_space_adapter is not None
        else "BL_ICF_MECHANISM_SPACE_V1"
    )
    knowledge_base = {
        "kind": (
            "RESEARCH_ONLY_STANDALONE"
            if generic_search_space
            else "RESEARCH_ONLY_BLICF_STANDALONE"
        ),
        "search_space": search_space_id,
        "search_space_conformance": _search_space_conformance(
            config,
            profile=profile,
            search_space_adapter=search_space_adapter,
        ),
        "active_executable_capability_count": len(profile.entries),
        "source_identity": identity,
        "protocol": "Research-side fixed-protocol offline top-n evaluation",
    }
    if config.baseline_context:
        knowledge_base["baseline_context"] = config.baseline_context
    return ResearchContext(
        campaign_id=profile.campaign_id,
        round_index=1,
        knowledge_base=knowledge_base,
        frozen_goal={
            "metric": "NDCG@10",
            "direction": "maximize",
            "research_objective": (
                "beat the frozen root with a faithful mechanism; a successful "
                "measured descendant may become the next construction parent "
                "while the root remains the paired comparator"
                if is_single_parent_context(config.baseline_context)
                else "improve the Research-side frontier while identifying the "
                "mechanism responsible for an observed signal"
                if generic_search_space
                else "improve the Research-side BLICF frontier while identifying "
                "the mechanism responsible for an observed signal"
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
        unresolved_questions=profile_questions
        or (
            {
                "question": (
                    "Which executable mechanism can improve NDCG@10 beyond the "
                    "frozen Research baseline under the same protocol?"
                    if generic_search_space
                    else "Which executable BLICF mechanism can improve NDCG@10 "
                    "beyond the frozen Research baseline under the same protocol?"
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
            **({"research_mode": config.research_mode}
               if config.research_mode != "portfolio" else {}),
            "producer_logical_calls": len(research_producer_roles(
                {"research_mode": config.research_mode},
            )),
            "producer_token_ceiling_each": (
                RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING
            ),
            "implementer_logical_calls_max": (
                config.max_implementation_calls_per_candidate
            ),
            "implementer_token_ceiling_each": (
                config.implementation_total_token_ceiling_per_call
            ),
            "implementer_output_token_ceiling_each": (
                config.implementation_total_token_ceiling_per_call
            ),
            "physical_attempts_per_logical_call_max": (
                config.provider_maximum_physical_attempts
            ),
            "experiment_opportunities": 1,
            "campaign_round_bound": MAX_STANDALONE_ROUNDS,
            "dataset": config.dataset,
            "evaluation_split": config.split,
            "candidate_universe": config.evaluator.get("candidate_universe"),
            "heldout_access": config.evaluator.get("heldout_access"),
            "epochs_requested": config.epochs,
            "eval_step": config.native_eval_step,
            "stopping_step": config.native_stopping_step,
            "worker_ceiling_seconds": config.final_worker_ceiling_seconds,
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


def _resolver_protocol_requirements(
    profile_requirements: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((*fresh_r1.PROTOCOL_REQUIREMENTS, *profile_requirements))
    )


def _resource_repair_context(state: CampaignState) -> Mapping[str, Any]:
    """Carry one closed round's resource repair into its immediate successor."""

    rows = state.context.scientific_memory.get("round_attempts", ())
    if not isinstance(rows, (tuple, list)):
        return {}
    source_round = int(state.next_round_index) - 1
    if source_round < 1:
        return {}
    for row in reversed(rows):
        if not isinstance(row, Mapping) or row.get("round_index") != source_round:
            continue
        repair = row.get("implementation_efficiency_repair_context")
        candidate_id = row.get("candidate_id")
        semantic_digest = row.get("candidate_semantic_digest")
        if (
            isinstance(repair, Mapping)
            and isinstance(candidate_id, str)
            and candidate_id
            and isinstance(semantic_digest, str)
            and semantic_digest
        ):
            source_files = {}
            candidate_root = row.get("candidate_root_path")
            if isinstance(candidate_root, str):
                source_root = Path(candidate_root)
                model_path = source_root / "recclaw_ext/candidate.py"
                if model_path.is_file() and hashlib.sha256(model_path.read_bytes()).hexdigest() == row.get("entrypoint_source_sha256"):
                    source_files = {
                        str(path.relative_to(source_root)): path.read_text(encoding="utf-8")
                        for path in sorted((source_root / "recclaw_ext").glob("*.py"))
                    }
            return canonical_value(
                {
                    "implementation_efficiency_repair_context": dict(repair),
                    "implementation_efficiency_repair_round_index": source_round,
                    "implementation_efficiency_repair_candidate_id": candidate_id,
                    "implementation_efficiency_repair_semantic_digest": (
                        semantic_digest
                    ),
                    "implementation_efficiency_repair_source_files": source_files,
                    "implementation_efficiency_repair_execution_contract": row.get("execution_contract"),
                }
            )
    return {}


def _round_input_factory(
    config: StandaloneResearchConfig,
    *,
    provider: ProviderResearchProducer,
    implementer: ProviderImplementerGateway,
    portfolio_candidates: tuple[PortfolioCandidateV2, ...],
    search_space_adapter: SearchSpaceAdapter | None = None,
    controller_interface: StandaloneControllerInterface | None = None,
    engineering_source_identity: Mapping[str, Any] | None = None,
):
    def inputs(state: CampaignState) -> CampaignRoundInputs:
        window_publisher = getattr(search_space_adapter, "research_window_budget_snapshot", None)
        research_window = window_publisher() if window_publisher is not None else None
        discovery_generation = int(
            state.context.scientific_memory.get("discovery_generation", 0)
        )
        generation_suffix = (
            ""
            if discovery_generation == 0
            else f":generation-{discovery_generation:04d}"
        )
        bindings = bindings_for_context(
            state.context,
            active_profile=state.active_profile,
            implementation_requirements=tuple(getattr(
                search_space_adapter, "implementation_requirements", config.implementation_requirements,
            )),
            compatibility_requirements=getattr(
                search_space_adapter, "compatibility_requirements", config.compatibility_requirements,
            ),
        )
        seed = int(_observation_seed_for_round(config, state.round_index))
        meta_inputs = None
        if config.research_mode == "portfolio" and _meta_replay_due(state.round_index) and (
            controller_interface is None
            or controller_interface.enable_meta_research
        ):
            meta_inputs = MetaResearchInputs(
                offline_replay=OfflineProducerReplayV1(
                    producer=provider,
                    producer_bindings=bindings,
                    equal_replay_token_charge=(
                        RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING
                    ),
                    deterministic_directive_replay=True,
                ),
                next_campaign_id=(
                    f"{config.campaign_id}:round-{state.round_index:04d}"
                    f"{generation_suffix}:next-fresh"
                ),
            )
        return CampaignRoundInputs(
            producer_bindings=bindings,
            resolver_environment=resolver_environment_for_profile(
                state.active_profile,
                available_dependencies=tuple(getattr(
                    search_space_adapter, "available_dependencies", config.available_dependencies,
                )),
                budget_limits=fresh_r1.BUDGET_LIMITS,
                protocol_requirements=getattr(
                    search_space_adapter, "protocol_requirements",
                    getattr(search_space_adapter, "compatibility_requirements",
                        _resolver_protocol_requirements((
                            *config.protocol_requirements, *config.compatibility_requirements,
                        )),
                    ),
                ),
            ),
            budget_snapshot={
                **({"research_window": canonical_value(research_window)}
                   if research_window is not None else {}),
                "experiment_opportunities": 1,
                **({"research_mode": config.research_mode}
                   if config.research_mode != "portfolio" else {}),
                "producer_logical_calls": len(research_producer_roles(
                    {"research_mode": config.research_mode},
                )),
                "epochs_requested": config.epochs,
                "eval_step": config.native_eval_step,
                "stopping_step": config.native_stopping_step,
                "worker_ceiling_seconds": (
                    config.final_worker_ceiling_seconds
                ),
                "physical_attempts_per_logical_call_max": (
                    config.provider_maximum_physical_attempts
                ),
                **(
                    {"max_attempts_per_round": config.max_attempts_per_round}
                    if config.max_attempts_per_round is not None
                    else {}
                ),
            },
            router=(
                _router()
                if controller_interface is None
                else controller_interface.router
            ),
            metric_contract_digest=sha256_digest(config.evaluator),
            observation_seed=str(seed),
            confirmation_seed=_confirmation_seed_for_round(
                config, state.round_index
            ),
            next_discriminative_test=(
                "Use the typed Research feedback and task queue to select the "
                + (
                    "next independent search-space opportunity."
                    if search_space_adapter is not None
                    else "next independent BLICF opportunity."
                )
            ),
            qualified_execution_by_capability=(
                state.qualified_execution_by_capability
            ),
            attempt_scheduler=config.attempt_scheduler,
            max_attempts_per_round=config.max_attempts_per_round,
            prebinding_token_ceiling_retry=(
                config.prebinding_token_ceiling_retry
            ),
            close_exhausted_no_metric_slot=(
                config.close_exhausted_no_metric_slot
            ),
            bootstrap_fixed_candidates=config.bootstrap_fixed_candidates,
            evidence_port=config.evidence_port,
            observation_seed_schedule=(
                tuple(str(item) for item in config.observation_seed_schedule)
                if config.observation_seed_schedule is not None
                else None
            ),
            verification_seed_schedule=(
                tuple(
                    str(item)
                    for item in getattr(
                        config.evidence_port,
                        "validation_seed_schedule",
                        (),
                    )
                )
                if config.evidence_port is not None
                else None
            ),
            candidate_handoff_factory=config.candidate_handoff_factory,
            research_profile_source=config.research_profile_source,
            candidate_root_by_capability=state.candidate_root_by_capability,
            resource_profile_by_capability=(
                state.resource_profile_by_capability or {}
            ),
            innovation_inputs=_production_innovation_inputs(
                config,
                implementer,
                search_space_adapter=search_space_adapter,
                campaign_id=config.campaign_id,
                seed=seed,
                round_index=state.round_index,
                discovery_generation=discovery_generation,
                predecessor_registry_ref=state.active_profile.profile_ref,
                predecessor_registry_digest=state.active_profile.profile_digest,
            ),
            meta_research_inputs=meta_inputs,
            # Passing this through is deliberately opt-in.  An empty tuple keeps
            # the legacy route; a non-empty tuple must already be complete and
            # identity-bound, so runtime/search_adapter performs the final match.
            portfolio_candidates=portfolio_candidates,
            evaluator=config.evaluator,
            split=config.split,
            frozen_profile_ref=config.frozen_profile_ref,
            round_role="DISCOVERY",
            engineering_source_identity=engineering_source_identity,
            # Generation is diagnostic only. Campaign attempt/metric state is
            # the sole scientific slot authority.
            feedback_proposal_generation_exhausted=(
                discovery_generation + 1
                >= MAX_DISCOVERY_GENERATIONS_PER_ROUND
                and (
                    discovery_generation < MAX_DISCOVERY_GENERATIONS_PER_ROUND
                    or config.allow_resume_exhausted_engineering_generation
                )
            ),
        )

    return inputs


def _active_search_data_identity(
    config: StandaloneResearchConfig,
) -> Mapping[str, Any] | None:
    if config.search_data_identity is None:
        return None
    try:
        return validated_search_partition_identity(config.search_data_identity)
    except ValueError as error:
        raise StandaloneCampaignError(str(error)) from error


def _build_provider_objects(
    config: StandaloneResearchConfig,
    *,
    provider_call: ProviderCall | None,
    launch: Launcher | None,
    search_space_adapter: SearchSpaceAdapter | None,
    controller_interface: StandaloneControllerInterface | None = None,
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
    provider_kwargs = {
        "config_source": config.api_config_source,
        "call_root": call_root,
        "session_id": config.campaign_id,
        "provider_call": provider_call,
        "proposal_seed": config.search_seed,
        "frozen_profile_ref": config.frozen_profile_ref,
        "maximum_physical_attempts": config.provider_maximum_physical_attempts,
        "proposal_output_token_ceiling_total_per_context": (
            config.proposal_output_token_ceiling_total_per_slot
        ),
        "proposal_lane_by_role": (
            {role: PROPOSAL_LANE_OPEN_SPEC for role in DISCOVERY_PRODUCERS}
            if search_space_adapter is not None
            else None
        ),
        "proposal_replay_by_role": (
            load_provider_proposal_replay(config.proposal_replay_run_root)
            if config.proposal_replay_run_root is not None
            else None
        ),
    }
    provider_factory = (
        getattr(search_space_adapter, "build_provider_research_producer", None)
        if search_space_adapter is not None
        else None
    )
    if controller_interface is not None:
        provider = controller_interface.provider
    elif callable(provider_factory):
        provider = provider_factory(**provider_kwargs)
    else:
        provider = ProviderResearchProducer(
            allowed_frozen_profile_kinds=(
                search_space_adapter.supported_frozen_profile_kinds
                if search_space_adapter is not None
                else ("OFFLINE_TOPN",)
            ),
            proposal_context_appendix=getattr(search_space_adapter, "proposal_context_appendix", ""),
            **({"proposal_template_source": search_space_adapter.proposal_prompt_source}
               if getattr(search_space_adapter, "proposal_prompt_source", None) is not None else {}),
            **({"proposal_schema_source": search_space_adapter.proposal_schema_source,
                "proposal_schema_path": Path(search_space_adapter.proposal_schema_source)}
               if getattr(search_space_adapter, "proposal_schema_source", None) is not None else {}),
            **provider_kwargs,
        )
    implementer_kwargs = {
        "config_source": config.api_config_source,
        "call_root": call_root,
        "session_id": config.campaign_id,
        "provider_call": provider_call,
        "total_token_ceiling": (
            config.implementation_total_token_ceiling_per_call
        ),
        "maximum_physical_attempts": config.provider_maximum_physical_attempts,
        "implementation_output_token_ceiling_total_per_candidate": (
            config.implementation_output_token_ceiling_total_per_candidate
        ),
        "shared_implementation_root": config.shared_implementation_root,
        "paired_search_seed": (
            config.search_seed
            if config.shared_implementation_root is not None
            else None
        ),
        "arm": config.arm_code,
    }
    implementer_factory = (
        getattr(search_space_adapter, "build_provider_implementer", None)
        if search_space_adapter is not None
        else None
    )
    if callable(implementer_factory):
        implementer = implementer_factory(**implementer_kwargs)
    else:
        implementer = ProviderImplementerGateway(
            **({"implementation_template_source": search_space_adapter.implementation_prompt_source}
               if getattr(search_space_adapter, "implementation_prompt_source", None) is not None else {}),
            **({"implementation_schema_source": search_space_adapter.implementation_schema_source,
                "implementation_schema_path": Path(search_space_adapter.implementation_schema_source)}
               if getattr(search_space_adapter, "implementation_schema_source", None) is not None else {}),
            **implementer_kwargs,
        )
    runner = make_fresh_runner(
        repo_root=config.repo_root,
        side_root=config.run_root / "execution",
        run_id=f"research-line-standalone:{config.campaign_id}",
        seed=config.seed,
        epochs=config.epochs,
        timeout_seconds=config.timeout_seconds,
        execution_purpose=config.execution_purpose,
        candidate_root_by_capability=config.candidate_root_by_capability,
        recbole_commit_identity=str(
            fresh_r1.campaign_training_runtime_release(
                CAMPAIGN_DEVELOPMENT_TRAINING_RELEASE_RESOURCE
                if config.evaluator == DEVELOPMENT_EVALUATOR
                else None
            ).backend_identity[
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
        search_data_identity=_active_search_data_identity(config),
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
    search_space_adapter: SearchSpaceAdapter | None,
    controller_interface: StandaloneControllerInterface | None = None,
) -> dict[str, Any]:
    runner_identity = _runner_identity(runner)
    payload: dict[str, Any] = {
            "schema": STANDALONE_SCHEMA,
            "controller": "ResearchCampaign",
            "research_only": True,
            "arm_semantics": "standalone",
            "proposal_selection_controller": (
                {
                    "kind": "RESEARCH_LINE",
                    "meta_research_enabled": config.research_mode == "portfolio",
                }
                if controller_interface is None
                else controller_interface.identity
            ),
            "campaign_id": config.campaign_id,
            "run_root": str(config.run_root),
            "source_baseline_identity": _source_baseline_identity(config),
            "baseline_context": config.baseline_context,
            "observation_seed_schedule": _seed_schedule_identity(config),
            "profile": {
                "ref": profile.profile_ref,
                "digest": profile.profile_digest,
                "entry_count": len(profile.entries),
                "search_space_conformance": _search_space_conformance(
                    config,
                    profile=profile,
                    search_space_adapter=search_space_adapter,
                ),
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
            "seed_identity": {
                "training_seed": config.seed,
                "search_seed": config.search_seed,
            },
            "provider": {
                "config_identity": provider.config_identity,
                "proposal_seed": config.search_seed,
                "model": fresh_r1.MODEL,
                "model_routing": model_routing_manifest(),
                "proposal_total_token_ceiling": (
                    RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING
                ),
                "proposal_output_token_ceiling": (
                    RESEARCH_PROPOSAL_OUTPUT_TOKEN_CEILING
                ),
                "implementation_token_ceiling": (
                    config.implementation_total_token_ceiling_per_call
                ),
                "implementation_output_token_ceiling_per_call": (
                    config.implementation_total_token_ceiling_per_call
                ),
                "proposal_output_token_ceiling_total_per_slot": (
                    config.proposal_output_token_ceiling_total_per_slot
                ),
                "implementation_output_token_ceiling_total_per_candidate": (
                    config.implementation_output_token_ceiling_total_per_candidate
                ),
                "maximum_physical_attempts": (
                    config.provider_maximum_physical_attempts
                ),
                "prebinding_token_ceiling_retry": (
                    config.prebinding_token_ceiling_retry
                ),
                "proposal_replay": provider_proposal_replay_identity(
                    config.proposal_replay_run_root
                ),
            },
            "runner": runner_identity,
            "execution": _execution_identity(config),
            "portfolio": _portfolio_identity(config),
            "round_bound": MAX_STANDALONE_ROUNDS,
            "metric_contract_digest": sha256_digest(config.evaluator),
            "metric_contract": config.evaluator,
            "split": config.split,
            "frozen_profile_ref": config.frozen_profile_ref,
            "evidence_guard": {
                "mode": (
                    "SEALED_GUARD" if config.evidence_port is not None else "NULL_PORT"
                ),
                "identity_digest": (
                    getattr(config.evidence_port, "identity_digest", None)
                    if config.evidence_port is not None
                    else "NONE"
                ),
            },
        }
    if callable(payload["evidence_guard"].get("identity_digest")):
        payload["evidence_guard"]["identity_digest"] = payload["evidence_guard"][
            "identity_digest"
        ]()
    if config.evidence_port is not None:
        search_data_identity = _active_search_data_identity(config)
        from .p4_runtime import is_p4_recipe, dataset_roots_for_recipe
        if is_p4_recipe({"split": config.split, "evaluator": config.evaluator}):
            data_root, _, actual_dataset_digest, _ = dataset_roots_for_recipe(
                {"split": config.split, "evaluator": config.evaluator},
                default_root=fresh_r1.SEARCH_DATA_ROOT, default_partition={},
            )
            dataset_manifest_path = data_root / "ml-1m/ml-1m.inter"
        elif search_data_identity is None:
            dataset_manifest_path = (
                fresh_r1.SEARCH_DATA_ROOT / "search_partition_manifest.json"
            )
            if not dataset_manifest_path.is_file():
                raise StandaloneCampaignError(
                    "Evidence Guard requires the active dataset manifest before run"
                )
            actual_dataset_digest = _bytes_sha256(dataset_manifest_path)
        else:
            dataset_manifest_path = Path(
                str(search_data_identity["manifest_path"])
            )
            actual_dataset_digest = str(
                search_data_identity["manifest_sha256"]
            )
        expected_dataset_digest = getattr(
            config.evidence_port,
            "expected_dataset_manifest_digest",
            None,
        )
        if expected_dataset_digest is not None and expected_dataset_digest != actual_dataset_digest:
            raise StandaloneCampaignError(
                "Evidence Guard dataset-manifest identity differs from the active "
                "dataset manifest"
            )
        payload["experiment"] = {
            "dataset_manifest_sha256": actual_dataset_digest,
            "dataset_manifest_path": str(dataset_manifest_path),
        }
        guard_identity = getattr(config.evidence_port, "identity_digest", None)
        if callable(guard_identity):
            guard_identity = guard_identity()
        allocation_policy = getattr(
            config.evidence_port, "allocation_policy", None
        )
        payload["evidence_guard"] = {
            "mode": "SEALED_GUARD",
            "identity_digest": guard_identity,
            "expected_dataset_manifest_digest": expected_dataset_digest,
            "required_seed_count": getattr(
                config.evidence_port, "required_seed_count", None
            ),
            "validation_seed_schedule": getattr(
                config.evidence_port, "validation_seed_schedule", ()
            ),
            "minimum_effect_delta": getattr(
                config.evidence_port, "minimum_effect_delta", None
            ),
            "replication_trigger_delta": getattr(
                config.evidence_port, "replication_trigger_delta", None
            ),
            "allocation_policy": (
                allocation_policy.to_dict()
                if allocation_policy is not None
                and callable(getattr(allocation_policy, "to_dict", None))
                else None
            ),
            "guard_provider_calls": 0,
            "budget_semantics": (
                "AUXILIARY_VERIFICATION_DOES_NOT_CONSUME_DISCOVERY_ROUNDS"
            ),
        }
    return canonical_value(payload)


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
        "search_data_identity": config.search_data_identity,
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


_RUNTIME_PLACEMENT_IDENTITY_KEYS = frozenset(
    {"cuda_visible_devices", "gpu_id"}
)


def _without_runtime_placement(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in identity.items()
        if key not in _RUNTIME_PLACEMENT_IDENTITY_KEYS
    }


def _resume_runner_identity_compatible(
    sealed: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Compare runner behavior independently of physical GPU placement."""

    return _without_runtime_placement(sealed) == _without_runtime_placement(
        current
    )


def _profile_source_identity(
    source: ResearchProfileSourceV1 | None,
) -> Mapping[str, str] | None:
    if source is None:
        return None
    return canonical_value(source.identity)


def _execution_identity(config: StandaloneResearchConfig) -> dict[str, Any]:
    identity = {
        **({"search_policy_mode": config.search_policy_mode}
           if config.search_policy_mode != "adaptive" else {}),
        **({"research_mode": config.research_mode}
           if config.research_mode != "portfolio" else {}),
        "model_routing": model_routing_manifest(),
        "training_seed": config.seed,
        "search_seed": config.search_seed,
        "attempt_scheduler": config.attempt_scheduler,
        "max_attempts_per_round": config.max_attempts_per_round,
        "prebinding_token_ceiling_retry": (
            config.prebinding_token_ceiling_retry
        ),
        "close_exhausted_no_metric_slot": (
            config.close_exhausted_no_metric_slot
        ),
        "provider_maximum_physical_attempts": (
            config.provider_maximum_physical_attempts
        ),
        "implementation_total_token_ceiling_per_call": (
            config.implementation_total_token_ceiling_per_call
        ),
        "proposal_output_token_ceiling_total_per_slot": (
            config.proposal_output_token_ceiling_total_per_slot
        ),
        "implementation_output_token_ceiling_total_per_candidate": (
            config.implementation_output_token_ceiling_total_per_candidate
        ),
        "max_implementation_calls_per_candidate": (
            config.max_implementation_calls_per_candidate
        ),
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
        "dataset": config.dataset,
        "search_data_identity": config.search_data_identity,
        "native_eval_step": config.native_eval_step,
        "native_stopping_step": config.native_stopping_step,
        **({"native_resource_observation": True}
           if config.native_resource_observation else {}),
        "implementation_requirements": config.implementation_requirements,
        "compatibility_requirements": config.compatibility_requirements,
        "protocol_requirements": config.protocol_requirements,
        "evaluator": config.evaluator,
        "split": config.split,
        "frozen_profile_ref": config.frozen_profile_ref,
        "protocol_ref": config.protocol_ref,
        "protocol_digest": config.protocol_digest,
        "execution_purpose": config.execution_purpose,
        "bootstrap_fixed_candidates": config.bootstrap_fixed_candidates,
        "bootstrap_max_proposals": config.bootstrap_max_proposals,
        "proposal_replay": provider_proposal_replay_identity(
            config.proposal_replay_run_root
        ),
        "arm_code": config.arm_code,
        "shared_implementation_root": (
            str(config.shared_implementation_root)
            if config.shared_implementation_root is not None
            else None
        ),
    }
    if config.gpu_id is not None:
        identity["gpu_id"] = config.gpu_id
    identity["evidence_guard_mode"] = (
        "SEALED_GUARD" if config.evidence_port is not None else "NULL_PORT"
    )
    if config.evidence_port is not None:
        guard_identity = getattr(config.evidence_port, "identity_digest", None)
        if callable(guard_identity):
            guard_identity = guard_identity()
        identity["evidence_guard_port_identity"] = (
            str(guard_identity)
            if guard_identity is not None
            else f"{type(config.evidence_port).__module__}.{type(config.evidence_port).__qualname__}"
        )
    else:
        identity["evidence_guard_port_identity"] = "NONE"
    return canonical_value(identity)


def _valid_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _resume_source_artifact_identity_compatible(
    sealed: Any,
    current: Any,
    *,
    allow_drift: bool,
    source_ref_key: str = "source_ref",
    source_sha256_key: str = "source_sha256",
) -> bool:
    """Compare a sealed identity with one explicit source-artifact migration."""

    if sealed == current:
        return True
    if (
        not allow_drift
        or not isinstance(sealed, Mapping)
        or not isinstance(current, Mapping)
    ):
        return False
    differing_keys = {
        key
        for key in set(sealed) | set(current)
        if sealed.get(key) != current.get(key)
    }
    sealed_sha256 = sealed.get(source_sha256_key)
    current_sha256 = current.get(source_sha256_key)
    if not (
        _valid_sha256_digest(sealed_sha256)
        and _valid_sha256_digest(current_sha256)
    ):
        return False
    if differing_keys == {source_sha256_key}:
        # Preserve the existing explicit same-ref digest recovery contract.
        return True
    if differing_keys != {source_ref_key, source_sha256_key}:
        return False
    sealed_ref = sealed.get(source_ref_key)
    current_ref = current.get(source_ref_key)
    return (
        isinstance(sealed_ref, str)
        and bool(sealed_ref.strip())
        and isinstance(current_ref, str)
        and bool(current_ref.strip())
    )


def _resume_source_identity_compatible(
    sealed: Any,
    config: StandaloneResearchConfig,
) -> bool:
    return _resume_source_artifact_identity_compatible(
        sealed,
        _source_baseline_identity(config),
        allow_drift=config.allow_resume_source_sha256_drift,
    )


def _resume_execution_identity_compatible(
    manifest: Mapping[str, Any],
    config: StandaloneResearchConfig,
) -> bool:
    """Allow only explicitly bounded resume identity migrations."""

    sealed = manifest.get("execution")
    current = _execution_identity(config)
    if sealed == current:
        return True
    if not isinstance(sealed, Mapping):
        return False

    # GPU placement is operational provenance, not a scientific execution
    # input.  Keep it in manifests, but never make a valid checkpoint depend
    # on the same physical device remaining available at resume time.
    sealed_comparable = _without_runtime_placement(sealed)
    current_comparable = _without_runtime_placement(current)
    if sealed_comparable == current_comparable:
        return True
    differing_keys = {
        key
        for key in set(sealed_comparable) | set(current_comparable)
        if sealed_comparable.get(key) != current_comparable.get(key)
    }
    if (
        config.allow_resume_source_sha256_drift
        and _resume_implementer_ceiling_migration_compatible(
            sealed_comparable,
            current_comparable,
            sealed_provider=manifest.get("provider"),
            allow_model_routing_migration=(
                config.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
            ),
        )
    ):
        return True
    if (
        differing_keys == {"model_routing"}
        and config.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
        and _resume_model_routing_reasoning_compatible(
            sealed_comparable.get("model_routing"),
            current_comparable.get("model_routing"),
        )
    ):
        return True
    identity_key = "evidence_guard_port_identity"
    if differing_keys != {identity_key}:
        return False
    sealed_identity = sealed_comparable.get(identity_key)
    current_identity = current_comparable.get(identity_key)
    if not (
        _valid_sha256_digest(sealed_identity)
        and _valid_sha256_digest(current_identity)
    ):
        return False
    if (
        sealed_comparable.get("evidence_guard_mode") != "SEALED_GUARD"
        or current_comparable.get("evidence_guard_mode") != "SEALED_GUARD"
        or config.evidence_port is None
    ):
        return False
    sealed_guard = manifest.get("evidence_guard")
    if (
        not isinstance(sealed_guard, Mapping)
        or sealed_guard.get("mode") != "SEALED_GUARD"
        or sealed_guard.get("identity_digest") != sealed_identity
    ):
        return False
    port_identity = getattr(config.evidence_port, "identity_digest", None)
    if callable(port_identity):
        port_identity = port_identity()
    return port_identity == current_identity


def _resume_implementer_ceiling_migration_compatible(
    sealed: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    sealed_provider: Any,
    allow_model_routing_migration: bool = False,
) -> bool:
    """Recognize a sealed pre-64k-to-current Implementer migration."""

    differing_keys = {
        key
        for key in set(sealed) | set(current)
        if sealed.get(key) != current.get(key)
    }
    ceiling_keys = {
        "implementation_total_token_ceiling_per_call",
        "research_consumer_composition",
    }
    if differing_keys not in (ceiling_keys, ceiling_keys | {"model_routing"}):
        return False
    if "model_routing" in differing_keys and not (
        allow_model_routing_migration
        and _resume_model_routing_reasoning_compatible(
            sealed.get("model_routing"),
            current.get("model_routing"),
        )
    ):
        return False
    sealed_total = sealed.get("implementation_total_token_ceiling_per_call")
    current_total = current.get("implementation_total_token_ceiling_per_call")
    if (
        sealed_total
        not in {
            _REPAIR39_IMPLEMENTATION_OUTPUT_TOKEN_CEILING,
            _REPAIR39_IMPLEMENTATION_TOTAL_TOKEN_CEILING,
        }
        or current_total != _REPAIR40_IMPLEMENTATION_TOKEN_CEILING
        or sealed.get("implementation_output_token_ceiling_total_per_candidate")
        is not None
        or current.get("implementation_output_token_ceiling_total_per_candidate")
        is not None
    ):
        return False
    shared_root = sealed.get("shared_implementation_root")
    if shared_root is not None and (
        not isinstance(shared_root, str) or not shared_root.strip()
    ):
        return False
    if current.get("shared_implementation_root") != shared_root:
        return False
    if not isinstance(sealed_provider, Mapping):
        return False
    sealed_output = sealed_provider.get(
        "implementation_output_token_ceiling_per_call"
    )
    if (
        sealed_output != _REPAIR39_IMPLEMENTATION_OUTPUT_TOKEN_CEILING
        or {
            "implementation_token_ceiling": sealed_provider.get(
                "implementation_token_ceiling"
            ),
            "implementation_output_token_ceiling_per_call": sealed_provider.get(
                "implementation_output_token_ceiling_per_call"
            ),
            "implementation_output_token_ceiling_total_per_candidate": (
                sealed_provider.get(
                    "implementation_output_token_ceiling_total_per_candidate"
                )
            ),
        }
        != {
            "implementation_token_ceiling": sealed_total,
            "implementation_output_token_ceiling_per_call": sealed_output,
            "implementation_output_token_ceiling_total_per_candidate": None,
        }
    ):
        return False

    sealed_composition = sealed.get("research_consumer_composition")
    current_composition = current.get("research_consumer_composition")
    if not isinstance(sealed_composition, Mapping) or not isinstance(
        current_composition,
        Mapping,
    ):
        return False
    if set(sealed_composition) != set(current_composition):
        return False
    sealed_payload = {
        key: value
        for key, value in sealed_composition.items()
        if key != "digest"
    }
    current_payload = {
        key: value
        for key, value in current_composition.items()
        if key != "digest"
    }
    if (
        sealed_composition.get("digest") != sha256_digest(sealed_payload)
        or current_composition.get("digest") != sha256_digest(current_payload)
    ):
        return False
    sealed_tokens = sealed_payload.get("token_ceilings")
    current_tokens = current_payload.get("token_ceilings")
    if not isinstance(sealed_tokens, Mapping) or not isinstance(
        current_tokens,
        Mapping,
    ):
        return False
    if set(sealed_tokens) != set(current_tokens):
        return False
    if {
        key
        for key in set(sealed_tokens) | set(current_tokens)
        if sealed_tokens.get(key) != current_tokens.get(key)
    } != {"implementation", "implementation_output"}:
        return False
    if (
        sealed_tokens.get("implementation") != sealed_total
        or sealed_tokens.get("implementation_output") != sealed_output
        or current_tokens.get("implementation")
        != _REPAIR40_IMPLEMENTATION_TOKEN_CEILING
        or current_tokens.get("implementation_output")
        != _REPAIR40_IMPLEMENTATION_TOKEN_CEILING
    ):
        return False
    sealed_without_tokens = {
        key: value for key, value in sealed_payload.items() if key != "token_ceilings"
    }
    current_without_tokens = {
        key: value for key, value in current_payload.items() if key != "token_ceilings"
    }
    return sealed_without_tokens == current_without_tokens


def _resume_model_routing_reasoning_compatible(
    sealed: Any,
    current: Any,
) -> bool:
    """Permit only the two explicitly authorized Provider routing migrations."""

    if not isinstance(sealed, Mapping) or not isinstance(current, Mapping):
        return False
    p1_routing = model_routing_manifest()
    if (
        current == p1_routing
        and p1_routing.get("role_routing_digest")
        == "abc8a37f3c319ebcf1126dd54d22b2d055b5d5e665d5bb25e036faceb7495767"
    ):
        p1_roles = p1_routing.get("role_models")
        sealed_roles = sealed.get("role_models")
        expected_sealed = dict(p1_routing)
        if isinstance(p1_roles, Mapping):
            expected_sealed["efficient_model"] = p1_routing["strong_model"]
            expected_sealed["role_models"] = {
                role: p1_routing["strong_model"] for role in p1_roles
            }
            expected_sealed["role_routing_digest"] = (
                "83c76c97fae7b6032005273016c865bb36624d59cbf371855bde6507846f911e"
            )
            if sealed == expected_sealed:
                return True
    stable_keys = ("efficient_model", "strong_model", "role_models")
    if any(sealed.get(key) != current.get(key) for key in stable_keys):
        return False
    allowed_differences = {
        "efficient_reasoning_effort",
        "strong_reasoning_effort",
        "role_reasoning_effort",
        "role_routing_digest",
    }
    differing_keys = {
        key
        for key in set(sealed) | set(current)
        if sealed.get(key) != current.get(key)
    }
    if not differing_keys or not differing_keys <= allowed_differences:
        return False
    sealed_roles = sealed.get("role_reasoning_effort")
    current_roles = current.get("role_reasoning_effort")
    if not isinstance(sealed_roles, Mapping) or not isinstance(
        current_roles,
        Mapping,
    ):
        return False
    if set(sealed_roles) != set(current_roles):
        return False
    top_level_reasoning_is_bounded = all(
        sealed.get(key) is None and current.get(key) in {None, "low"}
        for key in ("efficient_reasoning_effort", "strong_reasoning_effort")
    )
    role_reasoning_is_bounded = all(
        sealed_roles[role] is None and current_roles[role] in {None, "low"}
        for role in sealed_roles
    )
    return top_level_reasoning_is_bounded and role_reasoning_is_bounded


def _resume_provider_config_identity_compatible(
    sealed: Any,
    current: Any,
    *,
    allow_drift: bool,
    allow_endpoint2_to_endpoint1: bool = False,
) -> bool:
    """Permit only an explicitly authorized endpoint direction; no model drift."""

    if sealed == current:
        return True
    if (
        not (allow_drift or allow_endpoint2_to_endpoint1)
        or not isinstance(sealed, Mapping)
        or not isinstance(current, Mapping)
    ):
        return False
    if sealed.get("source_kind") != "path" or current.get("source_kind") != "path":
        return False
    sealed_ref = sealed.get("source_ref")
    current_ref = current.get("source_ref")
    if not isinstance(sealed_ref, str) or not isinstance(current_ref, str):
        return False
    if allow_endpoint2_to_endpoint1 and sealed_ref.endswith("/llm_api_endpoint2_only.md"):
        return current_ref.endswith("/llm_api_endpoint1_only.md")
    if not allow_drift:
        return False
    if sealed_ref.endswith("/llm_api_endpoint1_only.md"):
        return current_ref.endswith("/llm_api_endpoint2_only.md")
    endpoint2_ref = (
        "/ssd/tingrangan/recclaw_research_line/canary/"
        "validation-only-strict-blicf-20260814-01/canary-secrets/"
        "llm_api_endpoint2_only.md"
    )
    return (sealed_ref, current_ref) in {
        (
            "/ssd/tingrangan/recclaw_p5_endpoint1/"
            "llm_api_endpoint1_only.pre_newapi_20260828T0310Z.md",
            endpoint2_ref,
        ),
        (
            "/ssd/tingrangan/recclaw_research_line/p7_stage_20260805/secrets/"
            "llm_api_endpoint1_only.pre_newapi_20260828T0310Z.md",
            endpoint2_ref,
        ),
    }


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


def _effective_resume_manifest(
    *,
    sealed_manifest: Mapping[str, Any],
    sealed_manifest_sha256: str,
    config: StandaloneResearchConfig,
    provider_config_identity: Mapping[str, Any],
    implementer_ceiling_migration: bool = True,
    provider_routing_migration: bool = False,
) -> dict[str, Any]:
    """Project the validated migration onto a current, separately persisted view."""

    effective = dict(canonical_value(sealed_manifest))
    sealed_execution = sealed_manifest.get("execution")
    provider = effective.get("provider")
    if not isinstance(sealed_execution, Mapping) or not isinstance(
        provider, Mapping
    ):
        raise StandaloneCampaignError("standalone manifest lacks provider identity")
    sealed_total = sealed_execution.get(
        "implementation_total_token_ceiling_per_call"
    )
    sealed_output = provider.get("implementation_output_token_ceiling_per_call")
    effective_provider = dict(provider)
    effective_provider.update(
        {
            "config_identity": provider_config_identity,
            "model_routing": model_routing_manifest(),
            "implementation_token_ceiling": (
                config.implementation_total_token_ceiling_per_call
            ),
            "implementation_output_token_ceiling_per_call": (
                config.implementation_total_token_ceiling_per_call
            ),
            "implementation_output_token_ceiling_total_per_candidate": (
                config.implementation_output_token_ceiling_total_per_candidate
            ),
        }
    )
    effective["provider"] = effective_provider
    effective["execution"] = _execution_identity(config)
    effective["source_baseline_identity"] = _source_baseline_identity(config)
    context = effective.get("context")
    if not isinstance(context, Mapping) or not isinstance(
        context.get("value"),
        Mapping,
    ):
        raise StandaloneCampaignError("standalone manifest lacks context identity")
    effective_context = dict(context)
    effective_context_value = dict(context["value"])
    budget = effective_context_value.get("budget")
    if not isinstance(budget, Mapping):
        raise StandaloneCampaignError("standalone manifest context lacks budget")
    effective_budget = dict(budget)
    effective_budget.update(
        {
            "implementer_token_ceiling_each": (
                config.implementation_total_token_ceiling_per_call
            ),
            "implementer_output_token_ceiling_each": (
                config.implementation_total_token_ceiling_per_call
            ),
        }
    )
    effective_context_value["budget"] = effective_budget
    effective_context["value"] = effective_context_value
    effective_context["digest"] = sha256_digest(effective_context_value)
    effective["context"] = effective_context
    resume_migration: dict[str, Any] = {
        "schema": (
            _IMPLEMENTER_CEILING_MIGRATION_SCHEMA
            if implementer_ceiling_migration
            else "recclaw.research-line.provider-routing-resume-migration.v1"
        ),
        "authorization": (
            "allow_resume_source_sha256_drift"
            if implementer_ceiling_migration
            else "allow_resume_endpoint1_to_endpoint2_and_reasoning_drift"
        ),
        "sealed_manifest_filename": "STANDALONE_CAMPAIGN_MANIFEST.json",
        "sealed_manifest_sha256": sealed_manifest_sha256,
    }
    if implementer_ceiling_migration:
        resume_migration["implementation_per_physical_call"] = {
                "sealed_total": sealed_total,
                "sealed_output": sealed_output,
                "current_total": (
                    config.implementation_total_token_ceiling_per_call
                ),
                "current_output": (
                    config.implementation_total_token_ceiling_per_call
                ),
                "aggregate_output": (
                    config.implementation_output_token_ceiling_total_per_candidate
                ),
        }
    if provider_routing_migration:
        resume_migration["provider_routing"] = {
            "schema": "recclaw.research-line.provider-routing-resume-migration.v1",
            "authorization": (
                "allow_resume_endpoint1_to_endpoint2_and_reasoning_drift"
            ),
            "sealed_config_identity": provider.get("config_identity"),
            "current_config_identity": provider_config_identity,
            "sealed_model_routing": sealed_execution.get("model_routing"),
            "current_model_routing": model_routing_manifest(),
        }
    effective["resume_migration"] = canonical_value(resume_migration)
    return canonical_value(effective)


def _persist_effective_resume_manifest(
    path: Path,
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path]:
    """Append an immutable, hash-linked engineering-source migration hop."""

    if not path.is_file():
        _write_new_json(path, manifest)
        return canonical_value(dict(manifest)), path

    previous_path = path
    previous = _read_json(previous_path)
    hop = 2
    while True:
        comparable = dict(previous)
        comparable.pop("resume_migration_chain", None)
        if canonical_value(comparable) == canonical_value(dict(manifest)):
            return previous, previous_path
        next_path = path.with_name(
            f"{STANDALONE_EFFECTIVE_MANIFEST_MIGRATION_PREFIX}{hop:04d}.json"
        )
        if not next_path.is_file():
            chained = canonical_value(
                {
                    **dict(manifest),
                    "resume_migration_chain": {
                        "schema": (
                            "recclaw.research-line.engineering-source-"
                            "migration-link.v1"
                        ),
                        "hop": hop,
                        "previous_filename": previous_path.name,
                        "previous_sha256": _bytes_sha256(previous_path),
                    },
                }
            )
            _write_new_json(next_path, chained)
            return chained, next_path
        chained = _read_json(next_path)
        link = chained.get("resume_migration_chain")
        if not isinstance(link, Mapping) or link != {
            "schema": "recclaw.research-line.engineering-source-migration-link.v1",
            "hop": hop,
            "previous_filename": previous_path.name,
            "previous_sha256": _bytes_sha256(previous_path),
        }:
            raise StandaloneCampaignError(
                "effective resume migration history is not a valid hash chain"
            )
        previous_path = next_path
        previous = chained
        hop += 1


def _validate_manifest_for_resume(
    *,
    config: StandaloneResearchConfig,
    manifest: Mapping[str, Any],
    state: CampaignState,
    provider: ProviderResearchProducer,
    runner: FreshExperimentRunner,
    controller_interface: StandaloneControllerInterface | None = None,
) -> None:
    if manifest.get("schema") != STANDALONE_SCHEMA:
        raise StandaloneCampaignError("standalone campaign manifest schema drift")
    if manifest.get("controller") != "ResearchCampaign":
        raise StandaloneCampaignError("campaign is not owned by ResearchCampaign")
    expected_controller = (
        {"kind": "RESEARCH_LINE", "meta_research_enabled": config.research_mode == "portfolio"}
        if controller_interface is None
        else controller_interface.identity
    )
    sealed_controller = manifest.get("proposal_selection_controller")
    if not (
        controller_interface is None and sealed_controller is None
    ) and sealed_controller != expected_controller:
        raise StandaloneCampaignError(
            "proposal/selection controller identity differs on resume"
        )
    if not _resume_source_identity_compatible(
        manifest.get("source_baseline_identity"),
        config,
    ):
        raise StandaloneCampaignError(
            "Research baseline/source identity differs from the sealed manifest"
        )
    if manifest.get("baseline_context", {}) != config.baseline_context:
        raise StandaloneCampaignError(
            "baseline context differs from the sealed manifest"
        )
    if manifest.get("observation_seed_schedule") != _seed_schedule_identity(
        config
    ):
        raise StandaloneCampaignError(
            "observation seed schedule differs from the sealed manifest"
        )
    if manifest.get("campaign_id") != config.campaign_id:
        raise StandaloneCampaignError("campaign_id differs from the sealed manifest")
    if state.context.protocol_digest != config.baseline_source.protocol_digest:
        raise StandaloneCampaignError(
            "baseline source protocol differs from the resumed CampaignState"
        )
    provider_manifest = manifest.get("provider")
    if not isinstance(provider_manifest, Mapping):
        raise StandaloneCampaignError("standalone manifest lacks provider identity")
    if not _resume_provider_config_identity_compatible(
        provider_manifest.get("config_identity"),
        provider.config_identity,
        allow_endpoint2_to_endpoint1=config.allow_resume_endpoint2_to_endpoint1,
        allow_drift=(
            config.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
        ),
    ):
        raise StandaloneCampaignError("Provider config identity differs on resume")
    runner_manifest = manifest.get("runner")
    if not isinstance(runner_manifest, Mapping):
        raise StandaloneCampaignError("standalone manifest lacks runner identity")
    if not _resume_runner_identity_compatible(
        runner_manifest, _runner_identity(runner)
    ):
        raise StandaloneCampaignError("runner identity differs on resume")
    if not _resume_execution_identity_compatible(manifest, config):
        raise StandaloneCampaignError("campaign execution inputs differ on resume")
    if config.evidence_port is not None:
        experiment_manifest = manifest.get("experiment")
        expected_dataset_digest = getattr(
            config.evidence_port,
            "expected_dataset_manifest_digest",
            None,
        )
        if (
            not isinstance(experiment_manifest, Mapping)
            or experiment_manifest.get("dataset_manifest_sha256")
            != expected_dataset_digest
        ):
            raise StandaloneCampaignError(
                "sealed dataset-manifest identity differs on guarded resume"
            )
    if manifest.get("portfolio") != _portfolio_identity(config):
        raise StandaloneCampaignError("portfolio profile set differs on resume")
    state_identity = state.incumbent_observation
    if not _resume_source_artifact_identity_compatible(
        {
            "source_ref": state_identity.get("source_ref"),
            "source_receipt_sha256": state_identity.get(
                "source_receipt_sha256"
            ),
        },
        {
            "source_ref": config.baseline_source.source_ref,
            "source_receipt_sha256": config.baseline_source.source_sha256,
        },
        allow_drift=config.allow_resume_source_sha256_drift,
        source_sha256_key="source_receipt_sha256",
    ):
        raise StandaloneCampaignError(
            "CampaignState baseline/source identity differs from the requested resume"
        )
    global_memory = state.context.scientific_memory.get("global_memory")
    if (
        not isinstance(global_memory, Mapping)
        or not _resume_source_identity_compatible(
            global_memory.get("standalone_identity"),
            config,
        )
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
    search_space_adapter: SearchSpaceAdapter | None = None,
    controller_interface: StandaloneControllerInterface | None = None,
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
    if config.allow_resume_exhausted_engineering_generation and not resume:
        raise StandaloneCampaignError(
            "allow_resume_exhausted_engineering_generation is only valid with resume"
        )
    if config.allow_resume_endpoint2_to_endpoint1 and not resume:
        raise StandaloneCampaignError(
            "allow_resume_endpoint2_to_endpoint1 is only valid with resume"
        )
    if (
        config.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
        and not resume
    ):
        raise StandaloneCampaignError(
            "allow_resume_endpoint1_to_endpoint2_and_reasoning_drift is only "
            "valid with resume"
        )
    if (
        config.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
        and not config.allow_resume_source_sha256_drift
    ):
        raise StandaloneCampaignError(
            "endpoint/reasoning resume migration requires source drift authorization"
        )
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
        sealed_execution = manifest.get("execution")
        current_execution = _execution_identity(config)
        implementer_ceiling_migration = (
            isinstance(sealed_execution, Mapping)
            and config.allow_resume_source_sha256_drift
            and _resume_implementer_ceiling_migration_compatible(
                sealed_execution,
                current_execution,
                sealed_provider=manifest.get("provider"),
                allow_model_routing_migration=(
                    config.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
                ),
            )
        )
        if not _resume_execution_identity_compatible(manifest, config):
            raise StandaloneCampaignError(
                "campaign execution inputs differ on resume"
            )
        provider, implementer, runner = _build_provider_objects(
            config,
            provider_call=provider_call,
            launch=launch,
            search_space_adapter=search_space_adapter,
            controller_interface=controller_interface,
        )
        sealed_provider = manifest.get("provider")
        endpoint_model_migration = (
            config.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
            and isinstance(sealed_execution, Mapping)
            and isinstance(sealed_provider, Mapping)
            and _resume_model_routing_reasoning_compatible(
                sealed_execution.get("model_routing"),
                current_execution.get("model_routing"),
            )
            and _resume_provider_config_identity_compatible(
                sealed_provider.get("config_identity"),
                provider.config_identity,
                allow_drift=True,
            )
        )
        effective_manifest: Mapping[str, Any] = manifest
        engineering_source_identity: Mapping[str, Any] | None = None
        projected_manifest = (
            _effective_resume_manifest(
                sealed_manifest=manifest,
                sealed_manifest_sha256=_bytes_sha256(manifest_path),
                config=config,
                provider_config_identity=provider.config_identity,
                implementer_ceiling_migration=implementer_ceiling_migration,
                provider_routing_migration=endpoint_model_migration,
            )
            if implementer_ceiling_migration or endpoint_model_migration
            else None
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
                search_space_adapter=search_space_adapter,
                controller_interface=controller_interface,
            ),
            implementer=implementer,
            memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
            search_space_adapter=search_space_adapter,
            post_round_state_transition=(
                controller_interface.post_round_state_transition
                if controller_interface is not None
                else None
            ),
        )
        _validate_manifest_for_resume(
            config=config,
            manifest=manifest,
            state=campaign.state,
            provider=provider,
            runner=runner,
            controller_interface=controller_interface,
        )
        if projected_manifest is not None:
            effective_manifest, effective_path = _persist_effective_resume_manifest(
                config.run_root / STANDALONE_EFFECTIVE_MANIFEST_FILENAME,
                projected_manifest,
            )
            engineering_source_identity = canonical_value(
                {
                    "manifest_filename": effective_path.name,
                    "manifest_sha256": _bytes_sha256(effective_path),
                    "source_baseline_identity": effective_manifest[
                        "source_baseline_identity"
                    ],
                }
            )
            campaign.round_inputs = _round_input_factory(
                config,
                provider=provider,
                implementer=implementer,
                portfolio_candidates=config.portfolio_candidates,
                search_space_adapter=search_space_adapter,
                controller_interface=controller_interface,
                engineering_source_identity=engineering_source_identity,
            )
        return StandaloneResearchComposition(
            config=config,
            campaign=campaign,
            provider=provider,
            implementer=implementer,
            runner=runner,
            profile=campaign.state.active_profile,
            policy=campaign.state.policy,
            manifest=effective_manifest,
        )

    if config.run_root.exists():
        raise StandaloneCampaignError(
            "start requires a new run_root; use --resume for an existing campaign"
        )
    provider, implementer, runner = _build_provider_objects(
        config,
        provider_call=provider_call,
        launch=launch,
        search_space_adapter=search_space_adapter,
        controller_interface=controller_interface,
    )
    policy = _single_parent_research_policy(
        meta_v20_research_control_policy(),
        config.baseline_context,
    )
    bootstrap_profile = replace(
        _provider_profile(config, search_space_adapter),
        campaign_id=f"{config.campaign_id}:bootstrap-source",
    )
    if config.baseline_source.protocol_digest != bootstrap_profile.protocol_digest:
        raise StandaloneCampaignError(
            "baseline source protocol differs from the active Research profile"
        )
    incumbent = _initial_incumbent(config)
    bootstrap_context = _initial_context(
        config=config,
        profile=bootstrap_profile,
        policy=policy,
        incumbent=incumbent,
        search_space_adapter=search_space_adapter,
    )
    config.run_root.mkdir(parents=True, exist_ok=False)
    initial_pool = InitialSearchPoolResult(
        active_profile=bootstrap_profile,
        proposals=(),
        open_candidates=(),
        qualified_execution_by_capability={},
        candidate_root_by_capability={},
        resource_profile_by_capability={},
        innovations=(),
    )

    profile = replace(
        initial_pool.active_profile,
        campaign_id=config.campaign_id,
    )
    context = _initial_context(
        config=config,
        profile=profile,
        policy=policy,
        incumbent=incumbent,
        search_space_adapter=search_space_adapter,
    )
    state = CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=incumbent,
        carryover_proposals=initial_pool.proposals,
        carryover_open_candidates=initial_pool.open_candidates,
        qualified_execution_by_capability={
            **dict(config.qualified_execution_by_capability),
            **dict(initial_pool.qualified_execution_by_capability),
        },
        candidate_root_by_capability={
            **{
                key: str(value)
                for key, value in config.candidate_root_by_capability.items()
            },
            **dict(initial_pool.candidate_root_by_capability),
        },
        resource_profile_by_capability={
            **dict(config.resource_profile_by_capability),
            **dict(initial_pool.resource_profile_by_capability),
        },
        frontier=context.frontier,
    )
    if (
        controller_interface is not None
        and controller_interface.initial_state_transition is not None
    ):
        state = controller_interface.initial_state_transition(state)
        if not isinstance(state, CampaignState):
            raise StandaloneCampaignError(
                "initial controller state transition must return CampaignState"
            )
    manifest = _manifest(
        config=config,
        profile=profile,
        policy=policy,
        context=context,
        provider=provider,
        runner=runner,
        search_space_adapter=search_space_adapter,
        controller_interface=controller_interface,
    )
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
            search_space_adapter=search_space_adapter,
            controller_interface=controller_interface,
        ),
        implementer=implementer,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        search_space_adapter=search_space_adapter,
        post_round_state_transition=(
            controller_interface.post_round_state_transition
            if controller_interface is not None
            else None
        ),
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
    "MAX_DISCOVERY_GENERATIONS_PER_ROUND",
    "MAX_STANDALONE_ROUNDS",
    "STANDALONE_SCHEMA",
    "StandaloneCampaignError",
    "StandaloneResearchComposition",
    "StandaloneResearchConfig",
    "compose_standalone_campaign",
    "load_portfolio_candidates",
    "load_research_profile_source",
]
