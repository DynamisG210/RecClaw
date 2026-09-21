"""Thin P9 composition of two arm-local Research Line runtimes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    bl_icf_executable_profile_v2,
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import COMMON_EVALUATOR
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
    run_disposable_fixed_batch_resource_probe,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import CapabilityKindV1

from .bootstrap import bootstrap_search_pool
from .campaign import CampaignRoundInputs, CampaignState, ResearchCampaign
from .fresh_runner import Launcher, make_fresh_runner
from .original_arm import OriginalOutcomeClass, PinnedOriginalArmV1
from .paired import (
    ORIGINAL_RECCLAW_ARM,
    RESEARCH_LINE_ARM,
    STATUS_CENSORED,
    ArmRequest,
    ArmResult,
    FrozenExogenousConditions,
    PairedCampaignResult,
    PairedScheduler,
)
from .provider import ProviderCall, ProviderImplementerGateway, ProviderResearchProducer
from .replay import OfflineProducerReplayV1
from .runtime import (
    InnovationRuntimeInputs,
    MetaResearchInputs,
    bindings_for_context,
    resolver_environment_for_profile,
)
from .single_round import (
    _COMPATIBILITY_REQUIREMENTS,
    _IMPLEMENTATION_REQUIREMENTS,
    _bytes_sha256,
    compose_single_round,
)


class PairedRuntimeError(ValueError):
    """The paired boundary received an invalid value."""


META_REPLAY_INTERVAL = 10

_COMPOSABLE_V2_MODULE = "recclaw_ext.models.composable_v2"
_COMPOSABLE_V2_ENTRYPOINTS = frozenset(
    {
        f"{_COMPOSABLE_V2_MODULE}:BPRComposableV2",
        f"{_COMPOSABLE_V2_MODULE}:LightGCNComposableV2",
    }
)
_COMPOSABLE_V2_SOURCE_SHA256 = (
    "7d5644055c93189bf07be890733aef59cca11f3de6bd93daaa214e2eccead464"
)


def _meta_replay_due(round_index: int) -> bool:
    return round_index >= META_REPLAY_INTERVAL and round_index % META_REPLAY_INTERVAL == 0


class RunnerFactory(Protocol):
    def __call__(self, request: ArmRequest) -> Any: ...


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def project_original_executable_space() -> dict[str, Any]:
    """Copy the exact current BL-ICF executable identities into arm-local JSON."""

    profile = bl_icf_executable_profile_v2()
    mechanisms = executable_mechanisms()
    candidate_id_by_mechanism_id = {
        mechanism.mechanism_id: mechanism.candidate_id for mechanism in mechanisms
    }
    candidates: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        config = _json_clone(dict(mechanism.config))
        is_algorithmic_variant = mechanism.parent_mechanism_id is not None
        parent_candidate_id = (
            candidate_id_by_mechanism_id[mechanism.base_mechanism_id]
            if is_algorithmic_variant
            else None
        )
        candidates.append(
            {
                "candidate_id": mechanism.candidate_id,
                "mechanism_id": mechanism.mechanism_id,
                "mechanism_axis": mechanism.mechanism_axis,
                "mechanism_semantics_digest": mechanism.mechanism_semantics_digest,
                "base_mechanism_id": mechanism.base_mechanism_id,
                "parent_candidate_id": parent_candidate_id,
                **(
                    {"proposal_type": "algorithmic_variant"}
                    if is_algorithmic_variant
                    else {}
                ),
                "runnable_level": "config_only",
                "mechanism": mechanism.mechanism_id,
                "mechanism_composition": [
                    mechanism.base_mechanism_id,
                    *mechanism.operator_ids,
                ],
                "base_model": mechanism.base_model_config,
                "model": mechanism.model,
                "base_model_config": mechanism.base_model_config,
                "entrypoint": mechanism.entrypoint,
                "entrypoint_source_sha256": mechanism.entrypoint_source_sha256,
                "priority": "high",
                "status": "implemented",
                "wired": True,
                "runner_type": "config_only",
                "config": config,
                "parameter_overrides": _json_clone(config),
                "consumes": sorted(str(key) for key in config),
                "execution_recipe": _json_clone(mechanism.execution_recipe()),
                "execution_recipe_digest": mechanism.execution_recipe_digest,
            }
        )
    if len(candidates) != 66 or int(profile.get("executable_mechanism_count", -1)) != 66:
        raise PairedRuntimeError("current BL-ICF executable profile is not exactly 66 entries")
    return {
        "profile_ref": str(profile["profile_id"]),
        "profile_digest": str(profile["profile_digest"]),
        "candidates": candidates,
    }


def _materialize_original_candidate_configs(arm: PinnedOriginalArmV1) -> None:
    """Provide the fixed-66 health records and shared model source."""

    profile = project_original_executable_space()
    candidates = list(profile["candidates"])
    composable_candidates = []
    for candidate in candidates:
        entrypoint = str(candidate.get("entrypoint") or "")
        module = entrypoint.split(":", 1)[0]
        if module != _COMPOSABLE_V2_MODULE:
            continue
        if entrypoint not in _COMPOSABLE_V2_ENTRYPOINTS:
            raise PairedRuntimeError(
                "fixed BLICF composable entrypoint drift: " + entrypoint
            )
        if candidate.get("entrypoint_source_sha256") != _COMPOSABLE_V2_SOURCE_SHA256:
            raise PairedRuntimeError(
                "fixed BLICF composable source digest drift: "
                + str(candidate.get("candidate_id"))
            )
        composable_candidates.append(candidate)
    if len(composable_candidates) != 64:
        raise PairedRuntimeError(
            "fixed BLICF composable source projection must cover exactly 64 entries"
        )

    source = arm.repository_root / "recclaw_ext" / "models" / "composable_v2.py"
    if not source.is_file() or _bytes_sha256(source) != _COMPOSABLE_V2_SOURCE_SHA256:
        raise PairedRuntimeError(
            "fixed BLICF composable source is missing or has the wrong SHA256: "
            + str(source)
        )
    destination = arm.source_root / "recclaw_ext" / "models" / "composable_v2.py"
    if destination.exists():
        if not destination.is_file() or _bytes_sha256(destination) != _COMPOSABLE_V2_SOURCE_SHA256:
            raise PairedRuntimeError(
                "arm-local fixed BLICF composable source drift: " + str(destination)
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    root = arm.source_root / "configs" / "candidates"
    for candidate in candidates:
        path = root / f"{candidate['candidate_id']}.yaml"
        payload = json.dumps(candidate, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise PairedRuntimeError(f"Original candidate config drift: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(payload, encoding="utf-8")


def make_pinned_original_arm(
    *,
    repository_root: Path,
    campaign_root: Path,
    protocol: Mapping[str, Any],
    search_seed: int = 42,
    proposal_every: int = 3,
    proposal_count: int = 6,
    llm_responder: Any = None,
    experiment_runner: Any = None,
    llm_runtime: Mapping[str, str] | None = None,
    use_native_registry: bool = False,
) -> PinnedOriginalArmV1:
    arm = PinnedOriginalArmV1(
        repository_root=repository_root,
        campaign_root=campaign_root,
        initial_executable_space=project_original_executable_space(),
        protocol=protocol,
        search_seed=search_seed,
        proposal_every=proposal_every,
        proposal_count=proposal_count,
        llm_responder=llm_responder,
        experiment_runner=experiment_runner,
        llm_runtime={} if llm_runtime is None else llm_runtime,
        use_native_registry=use_native_registry,
    )
    if not use_native_registry:
        _materialize_original_candidate_configs(arm)
    return arm


def _number(value: Any) -> float | None:
    return None if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)


def _ndcg(value: Any) -> float | None:
    return _number(value.get("ndcg@10", value.get("ndcg"))) if isinstance(value, Mapping) else None


def _protocol_frontier(protocol: Mapping[str, Any]) -> float | None:
    values = [
        value
        for key in ("incumbent_ndcg@10", "frontier_ndcg@10")
        if (value := _number(protocol.get(key))) is not None
    ]
    baselines = protocol.get("baseline_results", {})
    rows = (
        (baselines,)
        if isinstance(baselines, Mapping) and "model" in baselines
        else baselines.values() if isinstance(baselines, Mapping) else ()
    )
    values.extend(value for row in rows if (value := _ndcg(row)) is not None)
    return max(values, default=None)


def _completed_original_frontier(arm: PinnedOriginalArmV1) -> float | None:
    values = [value for value in (_protocol_frontier(arm.protocol),) if value is not None]
    rounds = arm.campaign_root / "original_state" / "rounds"
    for path in sorted(rounds.glob("round-*.json")) if rounds.is_dir() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        result = payload.get("result") if isinstance(payload, Mapping) else None
        candidate = result.get("candidate_result") if isinstance(result, Mapping) else None
        value = _ndcg(candidate)
        if value is not None:
            values.append(value)
    return max(values, default=None)


def _public_metrics(
    *,
    ndcg: Any,
    frontier: Any,
    incumbent: Any,
    cost: Any,
    elapsed_hours: Any,
    episodes: int,
    mechanism_ids: tuple[str, ...],
) -> dict[str, Any]:
    candidate_ndcg, current_frontier, prior = _number(ndcg), _number(frontier), _number(incumbent)
    metrics: dict[str, Any] = {"episodes": int(episodes)}
    if candidate_ndcg is not None:
        metrics["ndcg"] = candidate_ndcg
    if current_frontier is not None:
        metrics["frontier"] = current_frontier
    if candidate_ndcg is not None and prior not in (None, 0.0):
        metrics["incumbent_relative_improvement"] = (candidate_ndcg - prior) / abs(prior)
    if (numeric_cost := _number(cost)) is not None:
        metrics["cost"] = numeric_cost
    if (hours := _number(elapsed_hours)) is not None:
        metrics["elapsed_hours"] = hours
    if mechanism_ids:
        metrics["mechanism_ids"] = sorted(set(mechanism_ids))
    return metrics


def _research_metrics(record: Any) -> dict[str, Any]:
    result = record.result
    run = result.candidate_run if isinstance(result.candidate_run, Mapping) else {}
    raw = run.get("metrics") if isinstance(run.get("metrics"), Mapping) else run
    ndcg = _ndcg(raw)
    before = _number(record.state_before.frontier.get("incumbent_ndcg@10"))
    after = _number(record.state_after.frontier.get("incumbent_ndcg@10"))
    wall_ms = _number(run.get("wall_time_ms"))
    cost = run.get("cost") if run.get("cost") is not None else wall_ms / 1000 if wall_ms is not None else None
    hours = run.get("elapsed_hours") if run.get("elapsed_hours") is not None else wall_ms / 3_600_000 if wall_ms is not None else None
    mechanism_ids: list[str] = []
    selected = result.selected_outcome
    mechanism_id = getattr(getattr(selected, "spec", None), "mechanism_id", None)
    if isinstance(mechanism_id, str) and mechanism_id:
        mechanism_ids.append(mechanism_id)
    if not mechanism_ids and isinstance(result.execution_recipe, Mapping):
        value = result.execution_recipe.get("mechanism_id")
        if isinstance(value, str) and value:
            mechanism_ids.append(value)
    return _public_metrics(
        ndcg=ndcg,
        frontier=after if after is not None else ndcg,
        incumbent=before,
        cost=cost,
        elapsed_hours=hours,
        episodes=int(getattr(result.interpretation, "episode", None) is not None),
        mechanism_ids=tuple(mechanism_ids),
    )


def _original_metrics(result: Any, *, incumbent: float | None) -> dict[str, Any]:
    candidate_result = result.candidate_result
    raw = candidate_result.get("metrics") if isinstance(candidate_result.get("metrics"), Mapping) else candidate_result
    ndcg = _ndcg(raw)
    run_time, wall_ms = _number(candidate_result.get("run_time")), _number(candidate_result.get("wall_time_ms"))
    elapsed = candidate_result.get("elapsed_hours")
    if elapsed is None:
        elapsed = wall_ms / 3_600_000 if wall_ms is not None else run_time / 3600 if run_time is not None else None
    cost = candidate_result.get("cost")
    if cost is None:
        cost = wall_ms / 1000 if wall_ms is not None else run_time
    mechanism_identity = result.selected_candidate.get("mechanism_id")
    if not isinstance(mechanism_identity, str) or not mechanism_identity:
        mechanism_identity = result.selected_candidate.get("candidate_id")
    mechanisms = (
        (mechanism_identity,)
        if isinstance(mechanism_identity, str) and mechanism_identity
        else ()
    )
    return _public_metrics(
        ndcg=ndcg,
        frontier=ndcg,
        incumbent=incumbent,
        cost=cost,
        elapsed_hours=elapsed,
        episodes=int(result.execution_calls > 0),
        mechanism_ids=mechanisms,
    )


def _prior_censored(root: Path, arm: str, round_index: int) -> bool:
    if round_index <= 1:
        return False
    path = root / "rounds" / f"round-{round_index - 1:04d}" / f"{arm}.receipt.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairedRuntimeError(f"cannot read prior paired receipt: {path}") from error
    return isinstance(value, Mapping) and str(value.get("status", "")).upper() == STATUS_CENSORED


@dataclass(slots=True)
class ResearchLineArmAdapter:
    campaign: ResearchCampaign
    paired_root: Path
    runner_factory: RunnerFactory | None = None

    def __post_init__(self) -> None:
        self.paired_root = Path(self.paired_root).resolve()

    def _sync_prior_censor(self, request: ArmRequest) -> None:
        if _prior_censored(self.paired_root, RESEARCH_LINE_ARM, request.round_index) and self.campaign.state.next_round_index <= request.round_index - 1:
            self.campaign.record_missing_round(
                request.round_index - 1,
                reason="paired scheduler censored the prior Research opportunity",
                failure_code="PAIRED_PRIOR_OPPORTUNITY_CENSORED",
            )

    def __call__(self, request: ArmRequest) -> ArmResult:
        if not isinstance(request, ArmRequest) or request.arm != RESEARCH_LINE_ARM:
            raise PairedRuntimeError("Research adapter received the wrong ArmRequest")
        self._sync_prior_censor(request)
        prior_runner = self.campaign.runner
        if self.runner_factory is not None:
            self.campaign.runner = self.runner_factory(request)
            if not callable(self.campaign.runner):
                raise PairedRuntimeError("Research runner factory must return a callable")
        try:
            record = self.campaign.run_round(request.round_index)
        finally:
            self.campaign.runner = prior_runner
        if record.status in {"OUTCOME_MISSING", "TYPED_FAILURE"}:
            return ArmResult.missing(f"RESEARCH_{record.status}")
        return ArmResult.completed(_research_metrics(record))


@dataclass(slots=True)
class OriginalArmAdapter:
    arm: PinnedOriginalArmV1
    paired_root: Path
    runner_factory: RunnerFactory | None = None
    frontier: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.paired_root = Path(self.paired_root).resolve()
        self.frontier = _completed_original_frontier(self.arm)

    def _sync_prior_censor(self, request: ArmRequest) -> None:
        if _prior_censored(self.paired_root, ORIGINAL_RECCLAW_ARM, request.round_index):
            self.arm.record_missing_round(
                request.round_index - 1,
                reason="paired scheduler censored the prior Original opportunity",
                censored=True,
            )

    def __call__(self, request: ArmRequest) -> ArmResult:
        if not isinstance(request, ArmRequest) or request.arm != ORIGINAL_RECCLAW_ARM:
            raise PairedRuntimeError("Original adapter received the wrong ArmRequest")
        self._sync_prior_censor(request)
        prior_runner = self.arm.experiment_runner
        bind_request = getattr(self.arm.llm_responder, "bind_request", None)
        if callable(bind_request):
            bind_request(request)
        if self.runner_factory is not None:
            self.arm.experiment_runner = self.runner_factory(request)
            if not callable(self.arm.experiment_runner):
                raise PairedRuntimeError("Original runner factory must return a callable")
        try:
            result = self.arm.run_round(request.round_index)
        finally:
            self.arm.experiment_runner = prior_runner
        if result.outcome_class == OriginalOutcomeClass.CENSORED.value:
            return ArmResult.censored("ORIGINAL_OPPORTUNITY_CENSORED")
        if result.outcome_class != OriginalOutcomeClass.SUCCESS.value:
            return ArmResult.missing(f"ORIGINAL_{result.outcome_class}")
        metrics = _original_metrics(result, incumbent=self.frontier)
        ndcg = _number(metrics.get("ndcg"))
        if ndcg is not None:
            self.frontier = ndcg if self.frontier is None else max(self.frontier, ndcg)
            metrics["frontier"] = self.frontier
        return ArmResult.completed(metrics)


@dataclass(slots=True)
class PairedRuntime:
    scheduler: PairedScheduler
    research_line: ResearchLineArmAdapter
    original_recclaw: OriginalArmAdapter

    def run(self, rounds: int | None = None, *, parallel_arms: bool | None = None) -> PairedCampaignResult:
        return self.scheduler.run(rounds, parallel_arms=parallel_arms)


def compose_paired_runtime(
    *,
    root: Path,
    campaign_id: str,
    conditions: FrozenExogenousConditions,
    research_campaign: ResearchCampaign,
    original_arm: PinnedOriginalArmV1,
    research_runner_factory: RunnerFactory | None = None,
    original_runner_factory: RunnerFactory | None = None,
    parallel_arms: bool = False,
    independent_arm_queues: bool = False,
) -> PairedRuntime:
    paired_root = Path(root).resolve()
    research = ResearchLineArmAdapter(research_campaign, paired_root, research_runner_factory)
    original = OriginalArmAdapter(original_arm, paired_root, original_runner_factory)
    scheduler = PairedScheduler(
        root=paired_root,
        campaign_id=campaign_id,
        conditions=conditions,
        research_line=research,
        original_recclaw=original,
        parallel_arms=parallel_arms,
        independent_arm_queues=independent_arm_queues,
    )
    return PairedRuntime(scheduler, research, original)


def _production_innovation_inputs(
    composition: Any,
    implementer: ProviderImplementerGateway,
    *,
    campaign_id: str,
    seed: int,
    round_index: int,
    predecessor_registry_ref: str,
    predecessor_registry_digest: str,
) -> InnovationRuntimeInputs:
    prompt = Path(fresh_r1.__file__).resolve().parent / "resources/research_line_implementer_prompt_v1.txt"
    policy = fresh_r1._shared_policy(
        _bytes_sha256(prompt),
        sha256_digest({"tools": (), "network": False, "allowed_files": ("recclaw_ext/__init__.py", "recclaw_ext/candidate.py", "recclaw_ext/trainer.py")}),
        allowed_files=(
            "recclaw_ext/__init__.py",
            "recclaw_ext/candidate.py",
            "recclaw_ext/trainer.py",
        ),
        execution_contract=None,
    )

    def contract(policy: Any) -> Mapping[str, Any]:
        value = policy.execution_contract
        if not isinstance(value, Mapping):
            raise ValueError("qualified OpenSpec lacks execution_contract")
        return value

    def fixture_factory(policy: Any, attempt: int, candidate_root: Path) -> Any:
        value = contract(policy)
        fixture = fresh_r1._qualification_fixture(
            composition.repo_root,
            seed=seed,
            root=(
                composition.run_root
                / "innovation_qualification"
                / f"round-{round_index:04d}"
                / candidate_root.parent.name
            ),
            base_model_config=str(value["base_model_config"]),
        )
        return replace(fixture, runtime_identity_ref=policy.runtime_identity_ref, runtime_identity_digest=policy.runtime_identity_digest)

    def unit_check_factory(policy: Any) -> Any:
        return fresh_r1._shared_behavioral_unit_check({}, base_model_config=str(contract(policy)["base_model_config"]))

    return InnovationRuntimeInputs(
        implementer=implementer,
        policy=policy,
        candidate_parent=composition.run_root / "innovation_candidates" / f"round-{round_index:04d}",
        fixture_factory=fixture_factory,
        unit_check_factory=unit_check_factory,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version=f"{campaign_id}:round-{round_index:04d}:qualified-v1",
        registry_version=f"{campaign_id}:round-{round_index:04d}:registry-v1",
        predecessor_registry_ref=predecessor_registry_ref,
        predecessor_registry_digest=predecessor_registry_digest,
        profile_version=f"{campaign_id}:round-{round_index:04d}:next-fresh-profile-v1",
        fresh_campaign_id=f"{campaign_id}:round-{round_index:04d}:next-fresh",
        resource_admission_required=True,
        resource_probe=lambda **kwargs: run_disposable_fixed_batch_resource_probe(
            composition.repo_root,
            total_budget_seconds=fresh_r1.MAX_WORKER_CEILING_SECONDS,
            **kwargs,
        ),
        resource_probe_parent=(
            composition.run_root
            / "innovation_resource_probes"
            / f"round-{round_index:04d}"
        ),
    )


def _production_research_campaign(
    *,
    composition: Any,
    root: Path,
    seed_schedule: tuple[int, ...],
    total_token_ceiling: int,
    provider_call: ProviderCall | None,
) -> ResearchCampaign:
    research_root = root / "research_line"
    provider = ProviderResearchProducer(
        config_source=composition.api_config_path,
        call_root=research_root / "provider_calls",
        session_id=composition.context.campaign_id,
        total_token_ceiling=total_token_ceiling,
        provider_call=provider_call,
    )
    implementer = ProviderImplementerGateway(
        config_source=composition.api_config_path,
        call_root=research_root / "provider_calls",
        session_id=composition.context.campaign_id,
        total_token_ceiling=total_token_ceiling,
        provider_call=provider_call,
    )
    initial = CampaignState.initial(
        context=composition.context,
        active_profile=composition.profile,
        policy=composition.policy,
        incumbent_observation=composition.incumbent,
        carryover_proposals=bootstrap_search_pool(composition.context, composition.profile, composition.policy),
    )

    def inputs(state: CampaignState) -> CampaignRoundInputs:
        seed = int(seed_schedule[state.round_index - 1])
        bindings = bindings_for_context(
            state.context,
            active_profile=state.active_profile,
            implementation_requirements=_IMPLEMENTATION_REQUIREMENTS,
            compatibility_requirements=_COMPATIBILITY_REQUIREMENTS,
        )
        meta_inputs = None
        if _meta_replay_due(state.round_index):
            replay = OfflineProducerReplayV1(
                producer=provider,
                producer_bindings=bindings,
                equal_replay_token_charge=fresh_r1.PROPOSAL_TOKEN_CEILING,
                deterministic_directive_replay=True,
            )
            meta_inputs = MetaResearchInputs(
                offline_replay=replay,
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
            budget_snapshot={"experiment_opportunities": 1},
            router=StrongStaticRouterV1(runnable_floor=0.0, utility_floor=0.0, blocker_ceiling=1.0, cost_ceiling=1.0, slate_ceiling=4),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed=str(seed),
            next_discriminative_test="Use typed arm-local feedback to choose the next discriminative test.",
            confirmation_seed=(
                str(seed_schedule[state.round_index])
                if state.round_index < len(seed_schedule)
                else None
            ),
            qualified_execution_by_capability=state.qualified_execution_by_capability,
            innovation_inputs=_production_innovation_inputs(
                composition,
                implementer,
                campaign_id=state.campaign_id,
                seed=seed,
                round_index=state.round_index,
                predecessor_registry_ref=state.active_profile.profile_ref,
                predecessor_registry_digest=state.active_profile.profile_digest,
            ),
            meta_research_inputs=meta_inputs,
        )

    runner = lambda _recipe, _binding: {"exit_status": "MISSING", "metrics": {}, "experiment_binding": {}}
    common = dict(
        root=research_root,
        producer=provider,
        runner=runner,
        round_inputs=inputs,
        implementer=implementer,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )
    if (research_root / ResearchCampaign.state_filename).is_file():
        return ResearchCampaign.resume(**common)
    return ResearchCampaign(state=initial, **common)


def make_production_paired_runtime(
    *,
    repo_root: Path,
    root: Path,
    campaign_id: str,
    api_config_path: Path,
    conditions: FrozenExogenousConditions,
    incumbent_receipt_path: Path | None = None,
    provider_call: ProviderCall | None = None,
    launch: Launcher | None = None,
    original_llm_responder: Any = None,
    original_runner_factory: RunnerFactory | None = None,
    original_llm_runtime: Mapping[str, str] | None = None,
    original_protocol: Mapping[str, Any] | None = None,
    original_search_seed: int = 42,
    parallel_arms: bool = False,
    independent_arm_queues: bool = False,
) -> PairedRuntime:
    """Build the future real path without invoking Provider or GPU seams."""

    if not conditions.seed_schedule:
        raise PairedRuntimeError("production paired runtime needs a seed schedule")
    root = Path(root).resolve()
    composition = compose_single_round(
        repo_root=repo_root,
        run_root=root / "research_line",
        api_config_path=api_config_path,
        campaign_id=campaign_id,
        seed=conditions.seed_schedule[0],
        incumbent_receipt_path=incumbent_receipt_path,
    )
    if len(composition.profile.entries) != 66 or conditions.profile_ref != composition.profile.profile_ref or conditions.profile_digest != composition.profile.profile_digest:
        raise PairedRuntimeError("paired conditions do not match the current 66-entry profile")
    research_campaign = _production_research_campaign(
        composition=composition,
        root=root,
        seed_schedule=tuple(conditions.seed_schedule),
        total_token_ceiling=conditions.token_budget,
        provider_call=provider_call,
    )
    if original_llm_responder is None or original_llm_runtime is None:
        from .paired_execution import OriginalProviderResponder, original_llm_runtime as make_llm_runtime

        original_llm_responder = original_llm_responder or OriginalProviderResponder(
            api_config_path=api_config_path,
            call_root=root / "original_recclaw" / "provider_calls",
            campaign_id=campaign_id,
            total_token_ceiling=conditions.token_budget,
        )
        original_llm_runtime = original_llm_runtime or make_llm_runtime(api_config_path)
    original = make_pinned_original_arm(
        repository_root=repo_root,
        campaign_root=root / "original_recclaw",
        protocol={} if original_protocol is None else original_protocol,
        search_seed=original_search_seed,
        llm_responder=original_llm_responder,
        llm_runtime=original_llm_runtime,
    )

    if original_runner_factory is None:
        from .paired_execution import make_original_runner_factory

        original_runner_factory = make_original_runner_factory(
            arm=original,
            repo_root=composition.repo_root,
            root=root,
            epochs=composition.epochs,
            timeout_seconds=composition.timeout_seconds,
            recbole_commit_identity=str(composition.manifest["runtime"]["recbole_commit"]),
            expected_recbole_source_tree_digest=str(composition.recbole_identity["source_tree_digest"]),
            watchdog_seconds=composition.watchdog_seconds,
            launch=launch,
        )

    def research_runner_factory(request: ArmRequest) -> Any:
        roots = {capability: Path(path) for capability, path in research_campaign.state.candidate_root_by_capability.items()}
        predictions = {
            capability: dict(profile["prediction"])
            for capability, profile in (
                research_campaign.state.resource_profile_by_capability or {}
            ).items()
            if isinstance(profile, Mapping)
            and isinstance(profile.get("prediction"), Mapping)
            and profile.get("status") == "RESOURCE_ADMITTED"
        }
        return make_fresh_runner(
            repo_root=composition.repo_root,
            side_root=root / "research_line" / "execution" / f"round-{request.round_index:04d}",
            run_id=f"{campaign_id}-round-{request.round_index:04d}",
            seed=request.seed,
            epochs=composition.epochs,
            timeout_seconds=composition.timeout_seconds,
            execution_purpose="RESEARCH_LINE_PAIRED_OFFLINE_TOPN",
            candidate_root_by_capability=roots,
            recbole_commit_identity=str(composition.manifest["runtime"]["recbole_commit"]),
            expected_recbole_source_tree_digest=str(composition.recbole_identity["source_tree_digest"]),
            resource_telemetry=True,
            watchdog_seconds=composition.watchdog_seconds,
            cuda_visible_devices=request.device,
            final_worker_ceiling_seconds=fresh_r1.MAX_WORKER_CEILING_SECONDS,
            resource_prediction_by_capability=predictions,
            launch=launch,
        )

    return compose_paired_runtime(
        root=root,
        campaign_id=campaign_id,
        conditions=conditions,
        research_campaign=research_campaign,
        original_arm=original,
        research_runner_factory=research_runner_factory,
        original_runner_factory=original_runner_factory,
        parallel_arms=parallel_arms,
        independent_arm_queues=independent_arm_queues,
    )


__all__ = [
    "OriginalArmAdapter",
    "PairedRuntime",
    "PairedRuntimeError",
    "ResearchLineArmAdapter",
    "RunnerFactory",
    "compose_paired_runtime",
    "make_pinned_original_arm",
    "make_production_paired_runtime",
    "project_original_executable_space",
]
