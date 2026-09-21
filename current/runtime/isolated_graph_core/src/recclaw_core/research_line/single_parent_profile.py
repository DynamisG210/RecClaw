"""Load one exact, non-BL single-strong-parent Research launch descriptor.

The descriptor is deliberately data-only.  It binds the package-owned search
space, frozen protocol, exact executable parent, and seed-54201 development
metric before Provider construction.  Templates may retain unresolved ``null``
values for unavailable evidence, but a runnable bundle cannot.
"""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
)
from recclaw_core.mechanism_space.catalog import resolve_provider
from recclaw_core.mechanism_space.declarative_provider import (
    DeclarativeMechanismSpaceProvider,
)

from .declarative_search_space_adapter import DeclarativeSearchSpaceAdapterV1
from .frozen_family_profile import (
    FrozenFamilyProfileError,
    build_frozen_family_profile,
    verify_frozen_family_assets,
)
from .single_parent_search import (
    BL_ICF_SINGLE_PARENT_SPEC,
    SingleParentSearchSpaceSpec,
    focused_mechanism_language,
    exact_parent_bundle_from_context,
    single_parent_spec_for_profile_key,
    validate_single_parent_runtime_context,
)
from .single_round import ResearchBaselineSourceV1
from .standalone import StandaloneResearchConfig


LAUNCH_SCHEMA = "recclaw.single-parent-search-space-launch.v1"
ACTUAL_TRAINING_SEED = 54201
IMPLEMENTATION_ALLOWED_FILES = (
    "recclaw_ext/__init__.py",
    "recclaw_ext/candidate.py",
    "recclaw_ext/trainer.py",
)

_TOP_LEVEL_FIELDS = {
    "schema",
    "profile_key",
    "protocol_ref",
    "seed",
    "frozen_profile",
    "execution_contract",
    "parent_source",
    "baseline_context",
    "research_runtime",
    "evidence",
}
_EXECUTION_FIELDS = {
    "capability_family",
    "model",
    "base_model_config",
    "required_dependencies",
    "config_overrides",
    "search_data",
}
_PARENT_SOURCE_FIELDS = {
    "source_ref",
    "source_sha256",
    "comparator_ref",
    "comparator_digest",
    "frozen_ndcg_at_10",
}
_RUNTIME_FIELDS = {
    "epochs",
    "eval_step",
    "stopping_step",
    "implementation_requirements",
    "compatibility_requirements",
    "protocol_requirements",
}
_FROZEN_PROFILE_FIELDS = {"profile_id", "frozen_fields"}
_FROZEN_CONFIG_KEYS = {
    "profile_kind",
    "dataset",
    "protocol_digest",
    "frozen_profile",
    "recclaw_verified_assets",
    "data_path",
    "benchmark_filename",
    "recclaw_search_data_manifest",
    "recclaw_search_data_manifest_sha256",
    "recclaw_search_partition_roles",
}
_PARENT_OBSERVATION_FIELDS = {
    "schema",
    "profile_key",
    "parent_binding",
    "protocol_digest",
    "asset_manifest_digest",
    "seed",
    "split",
    "metric",
    "comparator_ref",
    "comparator_digest",
    "worker_result_ref",
    "worker_result_digest",
}
_PARENT_BINDING_FIELDS = {
    "candidate_id",
    "program_digest",
    "source_tree_digest",
}
_PARENT_METRIC_FIELDS = {
    "name",
    "source",
    "partition_role",
    "value",
}


class SingleParentProfileError(ValueError):
    """Raised when a launch descriptor is incomplete or identity-inconsistent."""


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SingleParentProfileError(f"{field_name} must be an object")
    return value


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SingleParentProfileError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SingleParentProfileError(f"{field_name} must be a positive integer")
    return value


def _ndcg_at_10(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise SingleParentProfileError(
            "frozen_ndcg_at_10 must be a finite JSON number in [0, 1]"
        )
    return float(value)


def _frozen_runtime_cadence(
    frozen_fields: Mapping[str, Any],
) -> tuple[int, int, int]:
    budget = next(
        (
            frozen_fields[key]
            for key in (
                "solver_update_budget",
                "token_update_budget",
                "recommender_update_budget",
                "training_update_budget",
            )
            if key in frozen_fields
        ),
        None,
    )
    if not isinstance(budget, Mapping):
        raise SingleParentProfileError(
            "frozen profile lacks a training/update budget"
        )
    epochs = _positive_int(budget.get("epochs_ceiling"), field_name="epochs_ceiling")
    eval_step = _positive_int(
        budget.get("validation_every"), field_name="validation_every"
    )
    if "patience" in budget:
        stopping_step = _positive_int(
            budget.get("patience"), field_name="patience"
        )
    else:
        patience_epochs = _positive_int(
            budget.get("patience_epochs"), field_name="patience_epochs"
        )
        if patience_epochs % eval_step:
            raise SingleParentProfileError(
                "frozen patience_epochs must be divisible by validation_every"
            )
        stopping_step = patience_epochs // eval_step
    return epochs, eval_step, stopping_step


def _text_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SingleParentProfileError(f"{field_name} must be an array")
    result = tuple(
        _text(item, field_name=f"{field_name} item") for item in value
    )
    if not result or len(set(result)) != len(result):
        raise SingleParentProfileError(
            f"{field_name} must contain unique non-empty strings"
        )
    return result


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field_name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = ",".join(sorted(expected - actual))
        extra = ",".join(sorted(actual - expected))
        raise SingleParentProfileError(
            f"{field_name} fields differ; missing={missing}; extra={extra}"
        )


def _null_paths(value: Any, *, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    if value is None:
        return (prefix or "$",)
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_null_paths(value[key], prefix=child))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            paths.extend(_null_paths(item, prefix=f"{prefix}[{index}]"))
    return tuple(paths)


_MATERIALIZATION_INPUT_PATHS = (
    ("execution_contract", "search_data"),
    ("parent_source",),
    ("baseline_context", "parent_anchor", "binding"),
    ("baseline_context", "parent_anchor", "mechanism_program"),
    ("baseline_context", "parent_anchor", "source_bundle"),
    ("baseline_context", "parent_anchor", "paired_metric"),
)


def _unresolved_launch_inputs(value: Mapping[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for parts in _MATERIALIZATION_INPUT_PATHS:
        current: Any = value
        for part in parts:
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        prefix = ".".join(parts)
        paths.extend(_null_paths(current, prefix=prefix))
    return tuple(paths)


def inspect_single_parent_launch_payload(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a no-run readiness projection without accepting null evidence."""

    payload = _mapping(value, field_name="launch descriptor")
    profile_key = payload.get("profile_key")
    spec: SingleParentSearchSpaceSpec | None = None
    if isinstance(profile_key, str):
        try:
            spec = single_parent_spec_for_profile_key(profile_key)
        except KeyError:
            spec = None
    nulls = _unresolved_launch_inputs(payload)
    return canonical_value(
        {
            "schema": LAUNCH_SCHEMA,
            "status": "READY_NO_RUN" if not nulls else "BLOCKED_MISSING_INPUTS",
            "profile_key": profile_key,
            "search_space_id": (
                spec.mechanism_space_id if spec is not None else None
            ),
            "parent_name": spec.parent_name if spec is not None else None,
            "required_training_seed": ACTUAL_TRAINING_SEED,
            "unresolved_inputs": nulls,
        }
    )


def inspect_single_parent_launch(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SingleParentProfileError(
            f"cannot read single-parent launch JSON: {path}"
        ) from error
    return inspect_single_parent_launch_payload(
        _mapping(payload, field_name="launch descriptor")
    )


@dataclass(frozen=True, slots=True)
class SingleParentProfileBundleV1:
    profile_key: str
    spec: SingleParentSearchSpaceSpec
    protocol_ref: str
    protocol_digest: str
    seed: int
    dataset: str
    frozen_profile_ref: Mapping[str, Any]
    execution_contract: Mapping[str, Any]
    baseline_source: ResearchBaselineSourceV1
    baseline_context: Mapping[str, Any]
    adapter: DeclarativeSearchSpaceAdapterV1
    epochs: int
    eval_step: int
    stopping_step: int
    implementation_requirements: tuple[str, ...]
    compatibility_requirements: tuple[str, ...]
    protocol_requirements: tuple[str, ...]
    evidence: Mapping[str, Any]

    def standalone_config(
        self,
        *,
        repo_root: Path,
        run_root: Path,
        api_config_source: Any,
        campaign_id: str,
        round_count: int,
        timeout_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS,
        watchdog_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS,
        final_worker_ceiling_seconds: int = fresh_r1.MAX_WORKER_CEILING_SECONDS,
        cuda_visible_devices: str | None = None,
        gpu_id: int | None = None,
    ) -> StandaloneResearchConfig:
        """Build the shared standalone config; this performs no Provider call."""

        return StandaloneResearchConfig(
            repo_root=repo_root,
            run_root=run_root,
            api_config_source=api_config_source,
            campaign_id=campaign_id,
            baseline_source=self.baseline_source,
            search_data_identity=(
                self.execution_contract["config"].get("recclaw_verified_assets")
                if isinstance(self.execution_contract.get("config"), Mapping)
                else None
            ),
            seed=self.seed,
            search_seed=self.seed,
            epochs=self.epochs,
            timeout_seconds=timeout_seconds,
            watchdog_seconds=watchdog_seconds,
            final_worker_ceiling_seconds=final_worker_ceiling_seconds,
            cuda_visible_devices=cuda_visible_devices,
            gpu_id=gpu_id,
            observation_seed_schedule=(self.seed,) * round_count,
            round_count=round_count,
            attempt_scheduler=True,
            max_attempts_per_round=4,
            close_exhausted_no_metric_slot=True,
            implementation_total_token_ceiling_per_call=64_000,
            dataset=self.dataset,
            evaluator=DEVELOPMENT_EVALUATOR,
            split=DEVELOPMENT_SPLIT,
            frozen_profile_ref=self.frozen_profile_ref,
            protocol_ref=self.protocol_ref,
            protocol_digest=self.protocol_digest,
            execution_purpose=(
                f"DEVELOPMENT_SINGLE_PARENT_{self.profile_key.upper()}"
            ),
            bootstrap_fixed_candidates=False,
            baseline_context=self.baseline_context,
            native_eval_step=self.eval_step,
            native_stopping_step=self.stopping_step,
            implementation_requirements=self.implementation_requirements,
            compatibility_requirements=self.compatibility_requirements,
            protocol_requirements=self.protocol_requirements,
            available_dependencies=tuple(
                dict.fromkeys(
                    (
                        *fresh_r1.AVAILABLE_DEPENDENCIES,
                        *self.execution_contract["required_dependencies"],
                    )
                )
            ),
        )


def load_single_parent_launch_payload(
    value: Mapping[str, Any],
) -> SingleParentProfileBundleV1:
    payload = _mapping(value, field_name="launch descriptor")
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, field_name="launch descriptor")
    if payload.get("schema") != LAUNCH_SCHEMA:
        raise SingleParentProfileError("launch descriptor schema is unsupported")
    unresolved = _unresolved_launch_inputs(payload)
    if unresolved:
        raise SingleParentProfileError(
            "launch descriptor has unresolved inputs: " + ", ".join(unresolved)
        )

    profile_key = _text(payload["profile_key"], field_name="profile_key")
    try:
        spec = single_parent_spec_for_profile_key(profile_key)
    except KeyError as error:
        raise SingleParentProfileError(str(error)) from error
    if spec == BL_ICF_SINGLE_PARENT_SPEC:
        raise SingleParentProfileError(
            "BL-ICF keeps its existing specialized single-parent loader"
        )
    provider = resolve_provider(spec.mechanism_space_id)
    if not isinstance(provider, DeclarativeMechanismSpaceProvider):
        raise SingleParentProfileError(
            "single-parent profile is not backed by a declarative family provider"
        )

    seed = _positive_int(payload["seed"], field_name="seed")
    if seed != ACTUAL_TRAINING_SEED:
        raise SingleParentProfileError(
            "actual training seed must be 54201; any other seed needs explicit authorization"
        )
    protocol_ref = _text(payload["protocol_ref"], field_name="protocol_ref")

    frozen = _mapping(payload["frozen_profile"], field_name="frozen_profile")
    _require_exact_fields(
        frozen, _FROZEN_PROFILE_FIELDS, field_name="frozen_profile"
    )
    frozen_profile_ref, frozen_binding = build_frozen_family_profile(
        provider,
        profile_id=_text(frozen["profile_id"], field_name="profile_id"),
        frozen_fields=_mapping(
            frozen["frozen_fields"], field_name="frozen_fields"
        ),
    )
    protocol_digest = str(frozen_profile_ref["profile_digest"])
    dataset = _text(frozen_binding["dataset"], field_name="dataset")

    execution_source = _mapping(
        payload["execution_contract"], field_name="execution_contract"
    )
    _require_exact_fields(
        execution_source, _EXECUTION_FIELDS, field_name="execution_contract"
    )
    base_model_config = _text(
        execution_source["base_model_config"], field_name="base_model_config"
    )
    if base_model_config != spec.parent_base_model_config:
        raise SingleParentProfileError(
            "execution base_model_config differs from the frozen parent ABI"
        )
    overrides = _mapping(
        execution_source["config_overrides"], field_name="config_overrides"
    )
    overlap = set(overrides) & _FROZEN_CONFIG_KEYS
    if overlap:
        raise SingleParentProfileError(
            "config_overrides replace frozen fields: " + ",".join(sorted(overlap))
        )
    dependencies = _text_tuple(
        execution_source["required_dependencies"],
        field_name="required_dependencies",
    )
    try:
        verified_assets = verify_frozen_family_assets(
            frozen_fields=_mapping(
                frozen["frozen_fields"], field_name="frozen_fields"
            ),
            config_overrides=overrides,
            search_data=_mapping(
                execution_source["search_data"], field_name="search_data"
            ),
        )
    except FrozenFamilyProfileError as error:
        raise SingleParentProfileError(str(error)) from error
    execution_contract = canonical_value(
        {
            "capability_family": _text(
                execution_source["capability_family"],
                field_name="capability_family",
            ),
            "model": _text(execution_source["model"], field_name="model"),
            "base_model_config": base_model_config,
            "config": {
                **dict(overrides),
                **dict(frozen_binding),
                "load_col": None,
                "data_path": verified_assets["search_data"]["data_path"],
                "benchmark_filename": verified_assets["search_data"][
                    "benchmark_filename"
                ],
                "recclaw_search_data_manifest": verified_assets["search_data"][
                    "manifest_ref"
                ],
                "recclaw_search_data_manifest_sha256": verified_assets[
                    "search_data"
                ]["manifest_sha256"],
                "recclaw_search_partition_roles": verified_assets["search_data"][
                    "partition_roles"
                ],
                "recclaw_verified_assets": verified_assets,
            },
            "required_dependencies": dependencies,
        }
    )

    parent_source = _mapping(payload["parent_source"], field_name="parent_source")
    _require_exact_fields(
        parent_source, _PARENT_SOURCE_FIELDS, field_name="parent_source"
    )
    baseline_context_payload = _mapping(
        payload["baseline_context"], field_name="baseline_context"
    )
    # This descriptor is the adapter-owned execution record for the exact
    # runnable parent. Carry its verified contract into the core's parent
    # context; no framework defaults or inferred parameter ownership are used.
    baseline_context_payload = canonical_value(dict(baseline_context_payload))
    baseline_context_payload["parent_anchor"] = {
        **dict(_mapping(
            baseline_context_payload["parent_anchor"],
            field_name="baseline_context.parent_anchor",
        )),
        "execution_contract": {
            key: execution_contract[key]
            for key in ("base_model_config", "capability_family", "model", "config")
        },
    }
    parent_anchor = _mapping(
        baseline_context_payload["parent_anchor"],
        field_name="baseline_context.parent_anchor",
    )
    paired_metric = _mapping(
        parent_anchor["paired_metric"],
        field_name="baseline_context.parent_anchor.paired_metric",
    )
    parent_seed = _positive_int(
        paired_metric["seed"],
        field_name="baseline_context.parent_anchor.paired_metric.seed",
    )
    baseline_source = ResearchBaselineSourceV1.from_identity(
        source_ref=_text(parent_source["source_ref"], field_name="source_ref"),
        source_sha256=_text(
            parent_source["source_sha256"], field_name="source_sha256"
        ),
        comparator_ref=_text(
            parent_source["comparator_ref"], field_name="comparator_ref"
        ),
        comparator_digest=_text(
            parent_source["comparator_digest"], field_name="comparator_digest"
        ),
        frozen_ndcg_at_10=_ndcg_at_10(parent_source["frozen_ndcg_at_10"]),
        protocol_digest=protocol_digest,
        seed=parent_seed,
    )
    try:
        baseline_context = validate_single_parent_runtime_context(
            baseline_context_payload,
            baseline_seed=parent_seed,
            baseline_value=baseline_source.frozen_ndcg_at_10,
            allowed_files=IMPLEMENTATION_ALLOWED_FILES,
            active_profile_ref=frozen_profile_ref,
            require_executable_parent_abi=True,
        )
    except ValueError as error:
        raise SingleParentProfileError(str(error)) from error
    context_spec = single_parent_spec_for_profile_key(profile_key)
    if (
        baseline_context.get("research_profile_id")
        != context_spec.research_profile_id
    ):
        raise SingleParentProfileError(
            "baseline context differs from the selected single-parent profile"
        )
    exact_parent = exact_parent_bundle_from_context(
        baseline_context,
        allowed_files=IMPLEMENTATION_ALLOWED_FILES,
        require_executable_abi=True,
    )
    if exact_parent is None:
        raise SingleParentProfileError("baseline context lacks exact parent source")
    receipt_path = Path(baseline_source.source_ref).expanduser()
    if not receipt_path.is_absolute() or not receipt_path.is_file():
        raise SingleParentProfileError(
            "parent source_ref must name an existing absolute observation receipt"
        )
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != baseline_source.source_sha256:
        raise SingleParentProfileError(
            "parent source_sha256 differs from the observation receipt bytes"
        )
    try:
        receipt = _mapping(
            json.loads(receipt_bytes.decode("utf-8")),
            field_name="parent observation receipt",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SingleParentProfileError(
            "parent observation receipt is not valid UTF-8 JSON"
        ) from error
    _require_exact_fields(
        receipt,
        _PARENT_OBSERVATION_FIELDS,
        field_name="parent observation receipt",
    )
    if receipt["schema"] != "recclaw.single-parent-parent-observation.v1":
        raise SingleParentProfileError("parent observation receipt schema is unsupported")
    parent_binding = _mapping(
        receipt["parent_binding"], field_name="receipt parent_binding"
    )
    _require_exact_fields(
        parent_binding,
        _PARENT_BINDING_FIELDS,
        field_name="receipt parent_binding",
    )
    expected_parent_binding = {
        "candidate_id": exact_parent["candidate_id"],
        "program_digest": exact_parent["program_digest"],
        "source_tree_digest": exact_parent["source_tree_digest"],
    }
    if canonical_value(dict(parent_binding)) != canonical_value(
        expected_parent_binding
    ):
        raise SingleParentProfileError(
            "parent observation receipt differs from the exact parent source"
        )
    metric = _mapping(receipt["metric"], field_name="receipt metric")
    _require_exact_fields(
        metric, _PARENT_METRIC_FIELDS, field_name="receipt metric"
    )
    if (
        receipt["profile_key"] != profile_key
        or receipt["protocol_digest"] != protocol_digest
        or receipt["asset_manifest_digest"] != verified_assets["digest"]
        or receipt["seed"] != parent_seed
        or receipt["split"] != "data/dev"
        or metric["name"] != "NDCG@10"
        or metric["source"] != "BEST_VALID_RESULT"
        or metric["partition_role"] != "DEVELOPMENT_VALIDATION"
        or _ndcg_at_10(metric["value"])
        != baseline_source.frozen_ndcg_at_10
        or receipt["comparator_ref"] != baseline_source.comparator_ref
        or receipt["comparator_ref"] != exact_parent["capability_ref"]
    ):
        raise SingleParentProfileError(
            "parent observation receipt differs from launch protocol, assets, "
            "seed, comparator, or development metric"
        )
    expected_comparator_digest = sha256_digest(
        canonical_value(
            {
                "schema": "recclaw.single-parent-comparator-executable.v1",
                "parent_binding": expected_parent_binding,
                "protocol_digest": protocol_digest,
                "asset_manifest_digest": verified_assets["digest"],
            }
        )
    )
    worker_result_digest = receipt["worker_result_digest"]
    if (
        baseline_source.comparator_digest != expected_comparator_digest
        or receipt["comparator_digest"] != expected_comparator_digest
        or not isinstance(worker_result_digest, str)
        or len(worker_result_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in worker_result_digest
        )
    ):
        raise SingleParentProfileError(
            "parent receipt comparator or worker result identity is invalid"
        )
    worker_result_path = Path(
        _text(receipt["worker_result_ref"], field_name="worker_result_ref")
    ).expanduser()
    if not worker_result_path.is_absolute() or not worker_result_path.is_file():
        raise SingleParentProfileError(
            "worker_result_ref must name an existing absolute worker result"
        )
    worker_result_bytes = worker_result_path.read_bytes()
    if hashlib.sha256(worker_result_bytes).hexdigest() != worker_result_digest:
        raise SingleParentProfileError(
            "worker result sha256 differs from worker_result_digest"
        )
    try:
        worker_result = _mapping(
            json.loads(worker_result_bytes.decode("utf-8")),
            field_name="parent worker result",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SingleParentProfileError(
            "parent worker result is not valid UTF-8 JSON"
        ) from error
    best_valid_result = _mapping(
        worker_result.get("best_valid_result"),
        field_name="parent worker best_valid_result",
    )
    ndcg_key = next(
        (
            str(key)
            for key in best_valid_result
            if str(key).casefold() == "ndcg@10"
        ),
        None,
    )
    if ndcg_key is None:
        raise SingleParentProfileError(
            "parent worker result lacks best_valid_result NDCG@10"
        )
    worker_ndcg = _ndcg_at_10(best_valid_result[ndcg_key])
    if (
        worker_result.get("exit_status") != "SUCCESS"
        or worker_result.get("seed") != seed
        or worker_result.get("split") != "data/dev"
        or worker_result.get("metric_source") != "BEST_VALID_RESULT"
        or worker_result.get("online_partition_role")
        != "DEVELOPMENT_VALIDATION"
        or worker_result.get("test_result") is not None
        or _ndcg_at_10(worker_result.get("best_valid_score")) != worker_ndcg
        or worker_ndcg != _ndcg_at_10(metric["value"])
        or worker_ndcg != baseline_source.frozen_ndcg_at_10
    ):
        raise SingleParentProfileError(
            "parent worker result differs from the launch seed, data/dev metric, "
            "or development-only result contract"
        )

    runtime = _mapping(payload["research_runtime"], field_name="research_runtime")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, field_name="research_runtime")
    frozen_epochs, frozen_eval_step, frozen_stopping_step = (
        _frozen_runtime_cadence(
            _mapping(frozen["frozen_fields"], field_name="frozen_fields")
        )
    )
    declared_cadence = (
        _positive_int(runtime["epochs"], field_name="epochs"),
        _positive_int(runtime["eval_step"], field_name="eval_step"),
        _positive_int(runtime["stopping_step"], field_name="stopping_step"),
    )
    config_cadence = tuple(
        _positive_int(execution_contract["config"].get(field), field_name=field)
        for field in ("epochs", "eval_step", "stopping_step")
    )
    frozen_cadence = (frozen_epochs, frozen_eval_step, frozen_stopping_step)
    if declared_cadence != frozen_cadence or config_cadence != frozen_cadence:
        raise SingleParentProfileError(
            "runtime cadence differs from the frozen profile budget"
        )
    implementation_prompt = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources/research_line_declarative_implementer_prompt_v1.txt"
    )
    adapter = DeclarativeSearchSpaceAdapterV1(
        provider,
        adapter_id=spec.adapter_id,
        execution_contract=execution_contract,
        implementation_template_source=implementation_prompt,
        focused_language=focused_mechanism_language(spec),
    )
    return SingleParentProfileBundleV1(
        profile_key=profile_key,
        spec=spec,
        protocol_ref=protocol_ref,
        protocol_digest=protocol_digest,
        seed=seed,
        dataset=dataset,
        frozen_profile_ref=frozen_profile_ref,
        execution_contract=execution_contract,
        baseline_source=baseline_source,
        baseline_context=baseline_context,
        adapter=adapter,
        epochs=frozen_epochs,
        eval_step=frozen_eval_step,
        stopping_step=frozen_stopping_step,
        implementation_requirements=_text_tuple(
            runtime["implementation_requirements"],
            field_name="implementation_requirements",
        ),
        compatibility_requirements=_text_tuple(
            runtime["compatibility_requirements"],
            field_name="compatibility_requirements",
        ),
        protocol_requirements=_text_tuple(
            runtime["protocol_requirements"], field_name="protocol_requirements"
        ),
        evidence=canonical_value(
            dict(_mapping(payload["evidence"], field_name="evidence"))
        ),
    )


def load_single_parent_launch(path: Path) -> SingleParentProfileBundleV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SingleParentProfileError(
            f"cannot read single-parent launch JSON: {path}"
        ) from error
    return load_single_parent_launch_payload(
        _mapping(payload, field_name="launch descriptor")
    )


__all__ = [
    "ACTUAL_TRAINING_SEED",
    "IMPLEMENTATION_ALLOWED_FILES",
    "LAUNCH_SCHEMA",
    "SingleParentProfileBundleV1",
    "SingleParentProfileError",
    "inspect_single_parent_launch",
    "inspect_single_parent_launch_payload",
    "load_single_parent_launch",
    "load_single_parent_launch_payload",
]
