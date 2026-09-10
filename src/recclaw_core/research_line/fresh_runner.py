"""Thin adapter from the Research Line runner protocol to fresh R1 launch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    ExperimentBindingError,
    ExperimentBindingV1,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    DIRECT_GPU_SELECTION_MODE,
    FreshR1Error,
    MAX_WORKER_CEILING_SECONDS,
    GPU_RESERVATION_STATUS_UNMEASURED,
    _validated_gpu_id,
    development_run_root,
    resolve_candidate_deadline_seconds,
    run_development_training,
    validate_gpu_reservation_evidence,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchCandidateBindingV1,
    SearchProfileEntryOriginV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)


class FreshRunnerError(ValueError):
    """The explicit Research Line to fresh-run boundary is invalid."""


Launcher: TypeAlias = Callable[..., Mapping[str, Any]]
GpuReservationEvidenceProvider: TypeAlias = Callable[
    [Mapping[str, Any]], Mapping[str, Any] | None
]
GpuReservationProviderIdentityAccessor: TypeAlias = Callable[[], Mapping[str, Any]]

PHYSICAL_CONTEXT_SCHEMA = "recclaw.research-line.physical-execution-context.v1"
PHYSICAL_IDENTITY_SCHEMA = "recclaw.research-line.physical-execution-identity.v1"
_NEXT_DEVELOPMENT_SEED = "NEXT_DEVELOPMENT_SEED"


def get_gpu_reservation_provider_identity(
    provider: GpuReservationEvidenceProvider | None,
) -> Mapping[str, Any] | None:
    """Read stable provider identity without canonicalizing callable state."""

    if provider is None:
        return None
    accessor = getattr(provider, "provider_identity", None)
    if accessor is None:
        return None
    if not callable(accessor):
        raise FreshRunnerError("provider_identity accessor must be callable")
    identity = accessor()
    if identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise FreshRunnerError("provider_identity accessor must return a mapping")
    return canonical_value(dict(identity))


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FreshRunnerError(f"{field_name} must be normalized and non-empty")
    return value


def _physical_seed(value: object) -> int:
    if isinstance(value, bool):
        raise FreshRunnerError("physical context seed must be a positive integer")
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str):
        if value.strip().upper() == _NEXT_DEVELOPMENT_SEED:
            raise FreshRunnerError(
                "NEXT_DEVELOPMENT_SEED is a task sentinel, not a worker seed"
            )
        try:
            seed = int(value)
        except ValueError as error:
            raise FreshRunnerError(
                "physical context seed must be a positive integer"
            ) from error
    else:
        raise FreshRunnerError("physical context seed must be a positive integer")
    if seed < 1:
        raise FreshRunnerError("physical context seed must be a positive integer")
    return seed


def _immutable_canonical_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    canonical = canonical_value(dict(value))

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(canonical)


@dataclass(frozen=True, slots=True)
class FreshRunnerConfig:
    """All run-scoped values required by one fresh runner instance."""

    repo_root: Path
    side_root: Path
    run_id: str
    seed: int
    epochs: int
    timeout_seconds: int
    execution_purpose: str
    candidate_root_by_capability: Mapping[str, Path]
    run_identity: str = "recclaw-research-line-v1"
    authority: str = "user-delegated-research-line-physical-execution"
    recbole_commit_identity: str | None = None
    expected_recbole_source_tree_digest: str | None = None
    resource_telemetry: bool = False
    watchdog_seconds: int | None = None
    prefix_contract_path: Path | None = None
    cuda_visible_devices: str | None = None
    gpu_id: int | None = None
    final_worker_ceiling_seconds: int = MAX_WORKER_CEILING_SECONDS
    resource_prediction_by_capability: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    search_data_identity: Mapping[str, Any] | None = None
    gpu_reservation_evidence: Mapping[str, Any] | None = None
    require_gpu_reservation_evidence: bool = False

    def __post_init__(self) -> None:
        for field_name in ("repo_root", "side_root"):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise FreshRunnerError(f"{field_name} must be a Path")
            object.__setattr__(self, field_name, value.resolve())
        _require_text(self.run_id, field_name="run_id")
        _require_text(self.execution_purpose, field_name="execution_purpose")
        _require_text(self.run_identity, field_name="run_identity")
        _require_text(self.authority, field_name="authority")
        if self.recbole_commit_identity is not None:
            _require_text(
                self.recbole_commit_identity,
                field_name="recbole_commit_identity",
            )
        if self.expected_recbole_source_tree_digest is not None:
            object.__setattr__(
                self,
                "expected_recbole_source_tree_digest",
                validate_sha256(
                    self.expected_recbole_source_tree_digest,
                    field_name="expected_recbole_source_tree_digest",
                ),
            )
        if not isinstance(self.resource_telemetry, bool):
            raise FreshRunnerError("resource_telemetry must be a boolean")
        if not isinstance(self.require_gpu_reservation_evidence, bool):
            raise FreshRunnerError(
                "require_gpu_reservation_evidence must be a boolean"
            )
        if self.watchdog_seconds is not None and (
            isinstance(self.watchdog_seconds, bool)
            or not isinstance(self.watchdog_seconds, int)
            or self.watchdog_seconds < 1
        ):
            raise FreshRunnerError("watchdog_seconds must be a positive integer")
        if (
            isinstance(self.final_worker_ceiling_seconds, bool)
            or not isinstance(self.final_worker_ceiling_seconds, int)
            or self.final_worker_ceiling_seconds < 1
        ):
            raise FreshRunnerError(
                "final_worker_ceiling_seconds must be a positive integer"
            )
        if self.prefix_contract_path is not None:
            if not isinstance(self.prefix_contract_path, Path):
                raise FreshRunnerError("prefix_contract_path must be a Path")
            prefix_path = self.prefix_contract_path.resolve()
            if not prefix_path.is_file():
                raise FreshRunnerError("prefix_contract_path must exist")
            if not self.resource_telemetry:
                raise FreshRunnerError("fixed-batch prefix requires resource telemetry")
            object.__setattr__(self, "prefix_contract_path", prefix_path)
        if self.cuda_visible_devices is not None:
            _require_text(
                self.cuda_visible_devices,
                field_name="cuda_visible_devices",
            )
        try:
            validated_gpu_id = _validated_gpu_id(self.gpu_id)
        except FreshR1Error as error:
            raise FreshRunnerError(str(error)) from error
        if validated_gpu_id is not None and self.cuda_visible_devices is not None:
            raise FreshRunnerError(
                "gpu_id and cuda_visible_devices are mutually exclusive"
            )
        object.__setattr__(self, "gpu_id", validated_gpu_id)
        if self.gpu_reservation_evidence is not None and not isinstance(
            self.gpu_reservation_evidence, Mapping
        ):
            raise FreshRunnerError("gpu_reservation_evidence must be a mapping")
        try:
            validated_reservation_evidence = validate_gpu_reservation_evidence(
                self.gpu_reservation_evidence,
                cuda_visible_devices=self.cuda_visible_devices,
                physical_gpu_selector=(
                    str(validated_gpu_id) if validated_gpu_id is not None else None
                ),
                run_id=self.run_id,
            )
        except (FreshR1Error, TypeError, ValueError) as error:
            raise FreshRunnerError(str(error)) from error
        object.__setattr__(
            self,
            "gpu_reservation_evidence",
            validated_reservation_evidence,
        )
        for field_name in ("seed", "epochs", "timeout_seconds"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise FreshRunnerError(f"{field_name} must be a positive integer")
        if not isinstance(self.candidate_root_by_capability, Mapping):
            raise FreshRunnerError("candidate_root_by_capability must be a mapping")
        roots: dict[str, Path] = {}
        for capability_ref, root in self.candidate_root_by_capability.items():
            _require_text(capability_ref, field_name="candidate capability ref")
            if not isinstance(root, Path):
                raise FreshRunnerError(
                    "candidate_root_by_capability values must be Paths"
                )
            roots[capability_ref] = root.resolve()
        object.__setattr__(
            self,
            "candidate_root_by_capability",
            MappingProxyType(roots),
        )
        if not isinstance(self.resource_prediction_by_capability, Mapping):
            raise FreshRunnerError(
                "resource_prediction_by_capability must be a mapping"
            )
        predictions: dict[str, Mapping[str, Any]] = {}
        for capability_ref, prediction in self.resource_prediction_by_capability.items():
            _require_text(capability_ref, field_name="resource prediction capability ref")
            if not isinstance(prediction, Mapping):
                raise FreshRunnerError(
                    "resource prediction values must be mappings"
                )
            predictions[capability_ref] = canonical_value(dict(prediction))
        object.__setattr__(
            self,
            "resource_prediction_by_capability",
            MappingProxyType(predictions),
        )
        if self.search_data_identity is not None:
            if not isinstance(self.search_data_identity, Mapping):
                raise FreshRunnerError("search_data_identity must be a mapping")
            object.__setattr__(
                self,
                "search_data_identity",
                _immutable_canonical_mapping(self.search_data_identity),
            )


class FreshExperimentRunner:
    """Callable adapter implementing the runtime ``ExperimentRunner`` shape."""

    __slots__ = ("config", "_launch", "_gpu_reservation_evidence_provider")

    def __init__(
        self,
        config: FreshRunnerConfig,
        *,
        launch: Launcher | None = None,
        gpu_reservation_evidence_provider: GpuReservationEvidenceProvider | None = None,
    ) -> None:
        if not isinstance(config, FreshRunnerConfig):
            raise FreshRunnerError("config must be FreshRunnerConfig")
        if gpu_reservation_evidence_provider is not None and not callable(
            gpu_reservation_evidence_provider
        ):
            raise FreshRunnerError(
                "gpu_reservation_evidence_provider must be callable"
            )
        self.config = config
        self._launch = run_development_training if launch is None else launch
        self._gpu_reservation_evidence_provider = (
            gpu_reservation_evidence_provider
        )

    def gpu_reservation_provider_identity(self) -> Mapping[str, Any] | None:
        """Return provider config identity for a later manifest consumer."""

        return get_gpu_reservation_provider_identity(
            self._gpu_reservation_evidence_provider
        )

    def _reservation_selector(self) -> str | None:
        return (
            str(self.config.gpu_id)
            if self.config.gpu_id is not None
            else self.config.cuda_visible_devices
        )

    def __call__(
        self,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
    ) -> Mapping[str, Any]:
        return self._invoke(
            recipe,
            binding,
            run_id=self.config.run_id,
            seed=self.config.seed,
            gpu_reservation_evidence=self.config.gpu_reservation_evidence,
        )

    def run_with_physical_context(
        self,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        physical_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute one attempt with a durable, identity-bound physical context.

        The context is deliberately separate from ``recipe``.  It can therefore
        select a unique physical run without changing the scientific recipe or
        its execution-recipe digest.
        """

        if not isinstance(physical_context, Mapping):
            raise FreshRunnerError("physical_context must be a mapping")
        context = _immutable_canonical_mapping(physical_context)
        self._validate_physical_context(context, binding)
        context_digest = sha256_digest(dict(context))
        physical_run_id = (
            f"{self.config.run_id}:physical:{context_digest}"
        )
        seed = _physical_seed(context["seed"])
        if self.config.require_gpu_reservation_evidence:
            if self._reservation_selector() is None:
                raise FreshRunnerError(
                    "required GPU reservation evidence needs explicit "
                    "physical GPU selector"
                )
            if self._gpu_reservation_evidence_provider is None:
                raise FreshRunnerError(
                    "required GPU reservation evidence needs a provider"
                )
        if self.config.gpu_reservation_evidence is not None:
            raise FreshRunnerError(
                "static gpu_reservation_evidence cannot be rebound to a derived "
                "physical run_id; provide gpu_reservation_evidence_provider"
            )
        evidence: Mapping[str, Any] | None = None
        provider = self._gpu_reservation_evidence_provider
        if provider is not None:
            request = _immutable_canonical_mapping(
                {
                    "schema": "recclaw.research-line.physical-run-request.v1",
                    "physical_run_id": physical_run_id,
                    "seed": seed,
                    "context_digest": context_digest,
                    "cuda_visible_devices": self._reservation_selector(),
                    "require_gpu_reservation_evidence": (
                        self.config.require_gpu_reservation_evidence
                    ),
                    "final_worker_ceiling_seconds": (
                        self.config.final_worker_ceiling_seconds
                    ),
                    "physical_context": dict(context),
                }
            )
            if self.config.gpu_id is not None:
                request = _immutable_canonical_mapping(
                    {
                        **dict(request),
                        "gpu_id": self.config.gpu_id,
                        "selection_mode": DIRECT_GPU_SELECTION_MODE,
                    }
                )
            try:
                supplied = provider(request)
            except Exception as error:  # pragma: no cover - provider boundary
                raise FreshRunnerError(
                    "gpu_reservation_evidence_provider failed"
                ) from error
            if supplied is not None and not isinstance(supplied, Mapping):
                raise FreshRunnerError(
                    "gpu_reservation_evidence_provider must return a mapping or None"
                )
            if self.config.require_gpu_reservation_evidence and supplied is None:
                raise FreshRunnerError(
                    "required GPU reservation evidence provider returned no evidence"
                )
            try:
                evidence = validate_gpu_reservation_evidence(
                    supplied,
                    cuda_visible_devices=self.config.cuda_visible_devices,
                    physical_gpu_selector=(
                        str(self.config.gpu_id)
                        if self.config.gpu_id is not None
                        else None
                    ),
                    run_id=physical_run_id,
                )
            except (FreshR1Error, TypeError, ValueError) as error:
                raise FreshRunnerError(str(error)) from error
        result = self._invoke(
            recipe,
            binding,
            run_id=physical_run_id,
            seed=seed,
            gpu_reservation_evidence=evidence,
        )
        if (
            result.get("cuda_visible_devices") is not None
            and result.get("cuda_visible_devices")
            != self.config.cuda_visible_devices
        ):
            raise FreshRunnerError(
                "launcher CUDA identity differs from the context-aware runner"
            )
        physical_identity = canonical_value(
            {
                "schema": PHYSICAL_IDENTITY_SCHEMA,
                "run_id": physical_run_id,
                "seed": seed,
                "context_digest": context_digest,
                "cuda_visible_devices": (
                    result.get("cuda_visible_devices")
                    if result.get("cuda_visible_devices") is not None
                    else self.config.cuda_visible_devices
                ),
                "reservation_digest": (
                    evidence.get("reservation_digest")
                    if evidence is not None
                    else (
                        result.get("gpu_reservation_evidence", {}).get(
                            "reservation_digest"
                        )
                        if isinstance(
                            result.get("gpu_reservation_evidence"), Mapping
                        )
                        else None
                    )
                ),
                "reservation_status": (
                    result.get("gpu_reservation_status")
                    if result.get("gpu_reservation_status") is not None
                    else (
                        GPU_RESERVATION_STATUS_UNMEASURED
                        if evidence is None
                        else None
                    )
                ),
                "final_worker_ceiling_seconds": result.get(
                    "final_worker_ceiling_seconds",
                    self.config.final_worker_ceiling_seconds,
                ),
            }
        )
        if self.config.gpu_id is not None:
            physical_identity = canonical_value(
                {
                    **dict(physical_identity),
                    "gpu_id": self.config.gpu_id,
                    "physical_gpu_id": str(self.config.gpu_id),
                    "selection_mode": DIRECT_GPU_SELECTION_MODE,
                }
            )
        return canonical_value(
            {
                **dict(result),
                "physical_identity": physical_identity,
            }
        )

    def has_durable_worker_result(self, physical_context_digest: str) -> bool:
        """Report only whether the exact physical run has a durable worker row."""

        try:
            validate_sha256(
                physical_context_digest,
                field_name="physical_context_digest",
            )
        except Exception as error:
            raise FreshRunnerError("physical context digest is invalid") from error
        physical_run_id = (
            f"{self.config.run_id}:physical:{physical_context_digest}"
        )
        return (
            development_run_root(self.config.side_root, physical_run_id)
            / "worker"
            / "worker_result.json"
        ).is_file()

    def _invoke(
        self,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        *,
        run_id: str,
        seed: int,
        gpu_reservation_evidence: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if not isinstance(binding, SearchCandidateBindingV1):
            raise FreshRunnerError("binding must be SearchCandidateBindingV1")
        if not isinstance(recipe, Mapping):
            raise FreshRunnerError("recipe must be a mapping")
        _require_text(run_id, field_name="run_id")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
            raise FreshRunnerError("seed must be a positive integer")
        if self.config.require_gpu_reservation_evidence:
            if self._reservation_selector() is None:
                raise FreshRunnerError(
                    "required GPU reservation evidence needs explicit "
                    "physical GPU selector"
                )
            if gpu_reservation_evidence is None:
                raise FreshRunnerError(
                    "required GPU reservation evidence is missing"
                )
        try:
            validate_execution_recipe(recipe)
        except (ExperimentBindingError, TypeError, ValueError) as error:
            raise FreshRunnerError(str(error)) from error
        self._validate_recipe_binding(recipe, binding)
        candidate_root = self._candidate_root(binding, recipe)
        resource_prediction = recipe.get("resource_prediction")
        if resource_prediction is None:
            resource_prediction = self.config.resource_prediction_by_capability.get(
                binding.capability_ref
            )
        if resource_prediction is not None and not isinstance(
            resource_prediction, Mapping
        ):
            raise FreshRunnerError("resource_prediction must be a mapping")
        try:
            candidate_deadline_seconds = resolve_candidate_deadline_seconds(
                default_seconds=self.config.timeout_seconds,
                prediction=resource_prediction,
                final_worker_ceiling_seconds=self.config.final_worker_ceiling_seconds,
            )
        except (FreshR1Error, TypeError, ValueError) as error:
            raise FreshRunnerError(str(error)) from error
        launch_kwargs: dict[str, Any] = dict(
            repo_root=self.config.repo_root,
            side_root=self.config.side_root,
            run_id=run_id,
            seed=seed,
            candidate_root=candidate_root,
            entrypoint=str(recipe["entrypoint"]),
            source_sha256=str(recipe["entrypoint_source_sha256"]),
            timeout_seconds=candidate_deadline_seconds,
            epochs=self.config.epochs,
            execution_purpose=self.config.execution_purpose,
            run_identity=self.config.run_identity,
            authority=self.config.authority,
            recbole_commit_identity=self.config.recbole_commit_identity,
            expected_recbole_source_tree_digest=(
                self.config.expected_recbole_source_tree_digest
            ),
            resource_telemetry=self.config.resource_telemetry,
            watchdog_seconds=self.config.watchdog_seconds,
            prefix_contract_path=self.config.prefix_contract_path,
            execution_recipe=recipe,
            cuda_visible_devices=self.config.cuda_visible_devices,
            final_worker_ceiling_seconds=self.config.final_worker_ceiling_seconds,
            resource_prediction=resource_prediction,
            search_data_identity=self.config.search_data_identity,
        )
        if self.config.gpu_id is not None:
            launch_kwargs["gpu_id"] = self.config.gpu_id
        if gpu_reservation_evidence is not None:
            launch_kwargs["gpu_reservation_evidence"] = gpu_reservation_evidence
        allocated_timeout = None
        timeout_for_run = getattr(self._launch, "timeout_for_run", None)
        if callable(timeout_for_run):
            # An explicit experiment supervisor freezes its remaining-resource
            # deadline before launch; validation uses that same trusted value.
            allocated_timeout = timeout_for_run(
                run_id, launch_kwargs["source_sha256"], self.config.timeout_seconds
            )
            launch_kwargs["timeout_seconds"] = allocated_timeout
            launch_kwargs["final_worker_ceiling_seconds"] = allocated_timeout
        result = self._launch(**launch_kwargs)
        return self._validate_launch_result(
            result,
            recipe=recipe,
            binding=binding,
            candidate_root=candidate_root,
            expected_run_id=run_id,
            expected_seed=seed,
            allocated_timeout_seconds=allocated_timeout,
        )

    @staticmethod
    def _validate_physical_context(
        context: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
    ) -> None:
        if context.get("schema") != PHYSICAL_CONTEXT_SCHEMA:
            raise FreshRunnerError("physical context schema is invalid")
        for field_name in ("campaign_id", "opportunity_ref", "candidate_id"):
            _require_text(context.get(field_name), field_name=field_name)
        round_index = context.get("round_index")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index < 1
        ):
            raise FreshRunnerError("physical context round_index is invalid")
        attempt_index = context.get("attempt_index")
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise FreshRunnerError("physical context attempt_index is invalid")
        if context.get("candidate_id") != binding.proposal.candidate_id:
            raise FreshRunnerError("physical context candidate identity drift")
        if context.get("binding_digest") != binding.digest:
            raise FreshRunnerError("physical context binding identity drift")
        if context.get("candidate_semantic_digest") != (
            binding.mechanism_semantics_digest
        ):
            raise FreshRunnerError(
                "physical context candidate semantic identity drift"
            )
        _physical_seed(context.get("seed"))

    @staticmethod
    def _validate_recipe_binding(
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
    ) -> None:
        if recipe.get("execution_role") != "CANDIDATE":
            raise FreshRunnerError("SearchCandidateBinding requires a CANDIDATE recipe")
        expected = {
            "capability_ref": binding.capability_ref,
            "capability_digest": binding.capability_digest,
            "entrypoint": binding.executable_entrypoint,
            "mechanism_id": binding.proposal.mechanism_id,
            "mechanism_semantics_digest": binding.mechanism_semantics_digest,
        }
        mismatched = [
            field_name
            for field_name, expected_value in expected.items()
            if recipe.get(field_name) != expected_value
        ]
        if mismatched:
            raise FreshRunnerError(
                "recipe is not bound to SearchCandidateBinding: "
                + ", ".join(mismatched)
            )

    def _candidate_root(
        self,
        binding: SearchCandidateBindingV1,
        recipe: Mapping[str, Any],
    ) -> Path | None:
        if binding.entry_origin is SearchProfileEntryOriginV1.FIXED_66:
            if recipe.get("candidate_root_path") is not None:
                raise FreshRunnerError(
                    "FIXED_66 execution cannot carry a candidate-local root"
                )
            return None
        if binding.entry_origin is not SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
            raise FreshRunnerError("unknown Search profile entry origin")
        supplied_root = recipe.get("candidate_root_path")
        if supplied_root is not None:
            if not isinstance(supplied_root, (str, Path)) or not str(supplied_root):
                raise FreshRunnerError(
                    "qualified execution candidate_root_path is invalid"
                )
            root = Path(supplied_root).resolve()
        else:
            root = self.config.candidate_root_by_capability.get(binding.capability_ref)
        if root is None:
            raise FreshRunnerError(
                "qualified capability has no candidate-local root: "
                + binding.capability_ref
            )
        if not root.is_dir() or not (root / "recclaw_ext").is_dir():
            raise FreshRunnerError(
                "qualified capability candidate-local root is unavailable: "
                + str(root)
            )
        return root

    def _validate_launch_result(
        self,
        result: Mapping[str, Any],
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        candidate_root: Path | None,
        expected_run_id: str | None = None,
        expected_seed: int | None = None,
        allocated_timeout_seconds: int | None = None,
    ) -> Mapping[str, Any]:
        if not isinstance(result, Mapping):
            raise FreshRunnerError("fresh launcher must return a mapping")
        if self.config.gpu_id is not None:
            direct_identity = {
                "gpu_id": self.config.gpu_id,
                "physical_gpu_id": str(self.config.gpu_id),
                "selection_mode": DIRECT_GPU_SELECTION_MODE,
            }
            mismatched_direct = [
                field_name
                for field_name, expected_value in direct_identity.items()
                if result.get(field_name) != expected_value
            ]
            if result.get("cuda_visible_devices") is not None:
                mismatched_direct.append("cuda_visible_devices")
            if mismatched_direct:
                raise FreshRunnerError(
                    "direct gpu_id launcher identity mismatch: "
                    + ", ".join(mismatched_direct)
                )
        payload = result.get("experiment_binding")
        if not isinstance(payload, Mapping):
            raise FreshRunnerError(
                "fresh launcher result lacks complete ExperimentBinding mapping"
            )
        try:
            experiment_binding = ExperimentBindingV1.from_canonical_dict(payload)
        except (ExperimentBindingError, TypeError, ValueError) as error:
            raise FreshRunnerError(str(error)) from error
        expected_root = (
            str(candidate_root.resolve()) if candidate_root is not None else None
        )
        try:
            expected_timeout_seconds = resolve_candidate_deadline_seconds(
                default_seconds=(allocated_timeout_seconds or self.config.timeout_seconds),
                prediction=(
                    recipe.get("resource_prediction")
                    if isinstance(recipe.get("resource_prediction"), Mapping)
                    else self.config.resource_prediction_by_capability.get(
                        binding.capability_ref
                    )
                ),
                final_worker_ceiling_seconds=(
                    allocated_timeout_seconds or self.config.final_worker_ceiling_seconds
                ),
            )
        except (FreshR1Error, TypeError, ValueError) as error:
            raise FreshRunnerError(str(error)) from error
        if expected_run_id is None:
            expected_run_id = self.config.run_id
        if expected_seed is None:
            expected_seed = self.config.seed
        expected = {
            "capability_family": recipe["capability_family"],
            "capability_ref": binding.capability_ref,
            "capability_digest": binding.capability_digest,
            "entrypoint": binding.executable_entrypoint,
            "model": recipe["model"],
            "run_id": expected_run_id,
            "seed": expected_seed,
            "epochs": self.config.epochs,
            "timeout_seconds": expected_timeout_seconds,
            "execution_purpose": self.config.execution_purpose,
            "execution_role": "CANDIDATE",
            "candidate_root_path": expected_root,
            "resource_telemetry": self.config.resource_telemetry,
            "watchdog_seconds": self.config.watchdog_seconds,
        }
        mismatched = [
            field_name
            for field_name, expected_value in expected.items()
            if getattr(experiment_binding, field_name) != expected_value
        ]
        if mismatched:
            raise FreshRunnerError(
                "launcher ExperimentBinding identity mismatch: "
                + ", ".join(mismatched)
            )
        if experiment_binding.execution_recipe_digest != sha256_digest(recipe):
            raise FreshRunnerError("launcher ExperimentBinding recipe digest mismatch")
        expected_source_digest = self.config.expected_recbole_source_tree_digest
        if expected_source_digest is not None:
            source_identity = result.get("recbole_source_identity")
            if (
                not isinstance(source_identity, Mapping)
                or source_identity.get("source_tree_digest")
                != expected_source_digest
            ):
                raise FreshRunnerError(
                    "launcher RecBole source tree identity mismatch"
                )
        digest_fields = {
            "binding_digest": experiment_binding.digest,
            "experiment_binding_digest": experiment_binding.digest,
            "experiment_binding_ref": experiment_binding.ref,
            "execution_recipe_digest": experiment_binding.execution_recipe_digest,
            "seed": experiment_binding.seed,
        }
        mismatched = [
            field_name
            for field_name, expected_value in digest_fields.items()
            if result.get(field_name) != expected_value
        ]
        if mismatched:
            raise FreshRunnerError(
                "fresh launcher result identity mismatch: "
                + ", ".join(mismatched)
            )
        return canonical_value(dict(result))


def make_fresh_runner(
    *,
    repo_root: Path,
    side_root: Path,
    run_id: str,
    seed: int,
    epochs: int,
    timeout_seconds: int,
    execution_purpose: str,
    candidate_root_by_capability: Mapping[str, Path],
    run_identity: str = "recclaw-research-line-v1",
    authority: str = "user-delegated-research-line-physical-execution",
    recbole_commit_identity: str | None = None,
    expected_recbole_source_tree_digest: str | None = None,
    resource_telemetry: bool = False,
    watchdog_seconds: int | None = None,
    prefix_contract_path: Path | None = None,
    cuda_visible_devices: str | None = None,
    gpu_id: int | None = None,
    final_worker_ceiling_seconds: int = MAX_WORKER_CEILING_SECONDS,
    resource_prediction_by_capability: Mapping[str, Mapping[str, Any]] | None = None,
    search_data_identity: Mapping[str, Any] | None = None,
    gpu_reservation_evidence: Mapping[str, Any] | None = None,
    require_gpu_reservation_evidence: bool = False,
    gpu_reservation_evidence_provider: GpuReservationEvidenceProvider | None = None,
    launch: Launcher | None = None,
) -> FreshExperimentRunner:
    """Build one explicit runner callable for injection into the runtime."""

    return FreshExperimentRunner(
        FreshRunnerConfig(
            repo_root=repo_root,
            side_root=side_root,
            run_id=run_id,
            seed=seed,
            epochs=epochs,
            timeout_seconds=timeout_seconds,
            execution_purpose=execution_purpose,
            candidate_root_by_capability=candidate_root_by_capability,
            run_identity=run_identity,
            authority=authority,
            recbole_commit_identity=recbole_commit_identity,
            expected_recbole_source_tree_digest=(
                expected_recbole_source_tree_digest
            ),
            resource_telemetry=resource_telemetry,
            watchdog_seconds=watchdog_seconds,
            prefix_contract_path=prefix_contract_path,
            cuda_visible_devices=cuda_visible_devices,
            gpu_id=gpu_id,
            final_worker_ceiling_seconds=final_worker_ceiling_seconds,
            resource_prediction_by_capability=(
                resource_prediction_by_capability or {}
            ),
            search_data_identity=search_data_identity,
            gpu_reservation_evidence=gpu_reservation_evidence,
            require_gpu_reservation_evidence=require_gpu_reservation_evidence,
        ),
        launch=launch,
        gpu_reservation_evidence_provider=gpu_reservation_evidence_provider,
    )


__all__ = [
    "FreshExperimentRunner",
    "FreshRunnerConfig",
    "FreshRunnerError",
    "GpuReservationEvidenceProvider",
    "GpuReservationProviderIdentityAccessor",
    "Launcher",
    "PHYSICAL_CONTEXT_SCHEMA",
    "PHYSICAL_IDENTITY_SCHEMA",
    "get_gpu_reservation_provider_identity",
    "make_fresh_runner",
]
