"""Mechanical RecBole qualification for an RC0 candidate package.

The adapter deliberately evaluates only implementation feasibility.  It does
not compare recommendation metrics, update mechanism beliefs, or emit a
scientific episode.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import itertools
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .canonical import canonical_value, sha256_digest, validate_sha256
from .conversion_efficiency import normalize_qualification_failure
from .epoch_sampler_scaffold import (
    CandidateCardinalityContractError,
    POST_DEVICE_MECHANISM_INIT_FLAG,
    TrainSpectralBasisContractError,
    prepare_candidate_train_data,
)
from .vnext_contracts import (
    CandidatePackageV1,
    OpenResearchSpecV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationReceiptV1,
    QualificationStageV1,
    QualificationStatusV1,
)


class MechanicalRecBoleAdapterError(RuntimeError):
    """Raised when the qualifier itself receives an invalid invocation."""


DISPOSABLE_QUALIFICATION_TIMEOUT_SECONDS = 900


class _QualificationStageFailure(RuntimeError):
    def __init__(
        self,
        *,
        stage: QualificationStageV1,
        failure_class: QualificationFailureClassV1,
        reason_code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.failure_class = failure_class
        self.reason_code = reason_code
        self.details = canonical_value(details) if details is not None else None


@dataclass(frozen=True, slots=True)
class RecBoleQualificationFixture:
    """Frozen local inputs used by the mechanical qualification stages."""

    project_root: Path
    recbole_root: Path
    data_path: Path
    dataset: str
    base_model_config: str
    seed: int
    checkpoint_dir: Path
    runtime_identity_ref: str
    runtime_identity_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "project_root",
            "recbole_root",
            "data_path",
            "checkpoint_dir",
        ):
            path = Path(getattr(self, field_name)).resolve()
            object.__setattr__(self, field_name, path)
        if not self.project_root.is_dir():
            raise MechanicalRecBoleAdapterError("project_root does not exist")
        if not self.recbole_root.is_dir():
            raise MechanicalRecBoleAdapterError("recbole_root does not exist")
        if not self.data_path.is_dir():
            raise MechanicalRecBoleAdapterError("data_path does not exist")
        if (
            not self.dataset
            or not self.dataset.replace("-", "_").isalnum()
            or not self.base_model_config.isidentifier()
        ):
            raise MechanicalRecBoleAdapterError(
                "dataset and base_model_config must be path-safe identifiers"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise MechanicalRecBoleAdapterError("seed must be an integer")
        if (
            not isinstance(self.runtime_identity_ref, str)
            or not self.runtime_identity_ref
            or self.runtime_identity_ref != self.runtime_identity_ref.strip()
        ):
            raise MechanicalRecBoleAdapterError(
                "runtime_identity_ref must be normalized and non-empty"
            )
        validate_sha256(
            self.runtime_identity_digest,
            field_name="runtime_identity_digest",
        )


@dataclass(frozen=True, slots=True)
class MechanicalQualificationRun:
    """In-memory execution details accompanying the authoritative RC0 receipt."""

    receipt: QualificationReceiptV1
    stage_observations: Mapping[str, Any]
    failure_detail: Mapping[str, Any] | None
    smoke_executions: int

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "failure_detail": self.failure_detail,
                "receipt": self.receipt.canonical_dict(),
                "smoke_executions": self.smoke_executions,
                "stage_observations": self.stage_observations,
            }
        )


def snapshot_candidate_tree(root: Path) -> tuple[dict[str, Any], ...]:
    """Return a deterministic byte manifest without importing candidate code."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise MechanicalRecBoleAdapterError("candidate root does not exist")
    rows: list[dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise MechanicalRecBoleAdapterError(
                "candidate tree may not contain symlinks"
            )
        if not path.is_file():
            continue
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return tuple(rows)


def candidate_tree_identity(
    root: Path,
    *,
    candidate_root_ref: str,
) -> tuple[str, str]:
    """Derive the source-tree and candidate-root digests used by the adapter."""

    manifest = snapshot_candidate_tree(root)
    source_tree_digest = sha256_digest({"files": manifest})
    candidate_root_digest = sha256_digest(
        {
            "candidate_root_ref": candidate_root_ref,
            "source_tree_digest": source_tree_digest,
        }
    )
    return source_tree_digest, candidate_root_digest


def _stage_failure(
    stage: QualificationStageV1,
    failure_class: QualificationFailureClassV1,
    reason_code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> _QualificationStageFailure:
    return _QualificationStageFailure(
        stage=stage,
        failure_class=failure_class,
        reason_code=reason_code,
        message=message,
        details=details,
    )


def _trainer_repair_details(trainer_entrypoint: str) -> dict[str, str]:
    module_name, _class_name = trainer_entrypoint.split(":", 1)
    return {
        "trainer_entrypoint": trainer_entrypoint,
        "implicated_file": module_name.replace(".", "/") + ".py",
    }


def _candidate_method_repair_details(
    *method_names: str,
) -> dict[str, tuple[str, ...]]:
    """Bind a typed qualifier failure to exact candidate-owned methods.

    Mechanical revisions are scoped from these machine-owned symbols.  A
    validator that omits them leaves the revision service unable to distinguish
    a real repair from an unrelated rewrite, so keep the ownership at the
    failure-producing boundary instead of inferring it later from prose.
    """

    return {
        "implicated_methods": tuple(dict.fromkeys(method_names)),
        "implicated_files": ("recclaw_ext/candidate.py",),
    }


def _validate_direct_train_epoch_abi(
    trainer_class: type,
    *,
    repair_details: Mapping[str, Any] | None = None,
) -> None:
    train_epoch = trainer_class.__dict__.get("_train_epoch")
    if train_epoch is None:
        return
    try:
        inspect.signature(train_epoch).bind(
            None,
            None,
            0,
            loss_func=None,
            show_progress=False,
        )
    except TypeError as error:
        raise _stage_failure(
            QualificationStageV1.ONE_EPOCH_SMOKE,
            QualificationFailureClassV1.INTERFACE,
            "TRAINER_TRAIN_EPOCH_ABI_INVALID",
            (
                "a directly overridden _train_epoch must accept the frozen "
                "RecBole call ABI _train_epoch(train_data, epoch_idx, "
                "loss_func=None, show_progress=False)"
            ),
            details=repair_details,
        ) from error


def _reject_nonfinite_masking(source: str, relative: str) -> None:
    tree = ast.parse(source, filename=relative, mode="exec")
    masking_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else None
        )
        in {"nan_to_num", "nan_to_num_"}
    )
    if not masking_calls:
        return
    implicated_methods = tuple(
        method.name
        for class_node in tree.body
        if isinstance(class_node, ast.ClassDef)
        for method in class_node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(call in tuple(ast.walk(method)) for call in masking_calls)
    )
    function = masking_calls[0].func
    function_name = (
        function.id
        if isinstance(function, ast.Name)
        else function.attr
        if isinstance(function, ast.Attribute)
        else "nan_to_num"
    )
    raise _stage_failure(
        QualificationStageV1.STATIC_VALIDATION,
        QualificationFailureClassV1.IMPLEMENTATION,
        "NONFINITE_MASKING_FORBIDDEN",
        (
            f"{relative} masks NaN or infinity with {function_name}; "
            "the execution contract requires nonfinite values to fail "
            "qualification or training"
        ),
        details=(
            _candidate_method_repair_details(*implicated_methods)
            if relative == "recclaw_ext/candidate.py" and implicated_methods
            else {"implicated_files": (relative,)}
        ),
    )


def _reject_unseeded_random_generators(source: str, relative: str) -> None:
    """Keep candidate-local randomness under the campaign's frozen seed."""

    tree = ast.parse(source, filename=relative, mode="exec")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "default_rng"
        ):
            continue
        positional_seed_missing = not node.args
        positional_seed_is_none = (
            len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value is None
        )
        keyword_seed = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "seed"),
            None,
        )
        keyword_seed_missing = not any(
            keyword.arg == "seed" for keyword in node.keywords
        )
        keyword_seed_is_none = (
            isinstance(keyword_seed, ast.Constant)
            and keyword_seed.value is None
        )
        if not (
            (positional_seed_missing and keyword_seed_missing)
            or positional_seed_is_none
            or keyword_seed_is_none
        ):
            continue
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.IMPLEMENTATION,
            "UNSEEDED_RANDOM_GENERATOR_FORBIDDEN",
            (
                f"{relative} constructs default_rng without the frozen seed; "
                "candidate-local randomness must derive from the campaign seed"
            ),
        )


def _candidate_source_path(
    package: CandidatePackageV1,
    candidate_root: Path,
) -> tuple[Path, str]:
    module_name, class_name = package.executable_entrypoint.split(":", 1)
    relative = PurePosixPath(module_name.replace(".", "/") + ".py").as_posix()
    if relative not in package.allowed_files:
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.IMPLEMENTATION,
            "ENTRYPOINT_OUTSIDE_PACKAGE",
            "entrypoint source is not an allowed candidate-package file",
        )
    source_path = candidate_root.joinpath(*PurePosixPath(relative).parts)
    if not source_path.is_file():
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.IMPLEMENTATION,
            "ENTRYPOINT_SOURCE_MISSING",
            "entrypoint source is missing",
        )
    return source_path, class_name


def _validate_package_bindings(
    package: CandidatePackageV1,
    research_spec: OpenResearchSpecV1,
    fixture: RecBoleQualificationFixture,
    candidate_root: Path,
) -> dict[str, Any]:
    if (
        package.research_spec_ref != research_spec.spec_id
        or package.research_spec_digest != research_spec.digest
    ):
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.IMPLEMENTATION,
            "PACKAGE_SPEC_BINDING_MISMATCH",
            "CandidatePackageV1 does not bind the supplied OpenResearchSpecV1",
        )
    if (
        package.protocol_ref != research_spec.protocol_ref
        or package.protocol_digest != research_spec.protocol_digest
    ):
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.PROTOCOL,
            "PACKAGE_PROTOCOL_BINDING_MISMATCH",
            "candidate package protocol differs from the research specification",
        )
    if (
        package.runtime_identity_ref != fixture.runtime_identity_ref
        or package.runtime_identity_digest != fixture.runtime_identity_digest
    ):
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.RUNTIME,
            "PACKAGE_RUNTIME_BINDING_MISMATCH",
            "candidate package runtime differs from the qualifier runtime",
        )
    source_path, class_name = _candidate_source_path(package, candidate_root)
    execution_contract = research_spec.execution_contract
    if (
        execution_contract is not None
        and package.executable_entrypoint
        not in {
            "recclaw_ext.candidate:FreshCandidateModel",
            "recclaw_ext.models.e1_multvae:FreshCandidateModel",
        }
        and class_name != str(execution_contract["model"])
    ):
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.INTERFACE,
            "ENTRYPOINT_MODEL_MISMATCH",
            "entrypoint class is not the model selected by execution_contract",
        )
    return {
        "candidate_package_digest": package.digest,
        "entrypoint_class": class_name,
        "entrypoint_source": source_path.relative_to(candidate_root).as_posix(),
        "protocol_digest": package.protocol_digest,
        "research_spec_digest": package.research_spec_digest,
        "runtime_identity_digest": package.runtime_identity_digest,
    }


def _validate_static_package(
    package: CandidatePackageV1,
    candidate_root: Path,
) -> dict[str, Any]:
    manifest = snapshot_candidate_tree(candidate_root)
    observed_files = tuple(row["path"] for row in manifest)
    if observed_files != package.allowed_files:
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.IMPLEMENTATION,
            "CANDIDATE_TREE_FILE_SET_MISMATCH",
            "candidate tree files differ from the package allowlist",
        )
    source_digest, root_digest = candidate_tree_identity(
        candidate_root,
        candidate_root_ref=package.candidate_root_ref,
    )
    if (
        source_digest != package.source_tree_digest
        or root_digest != package.candidate_root_digest
    ):
        raise _stage_failure(
            QualificationStageV1.STATIC_VALIDATION,
            QualificationFailureClassV1.IMPLEMENTATION,
            "CANDIDATE_TREE_DIGEST_MISMATCH",
            "candidate tree identity differs from CandidatePackageV1",
        )
    compiled_files = 0
    for row in manifest:
        relative = str(row["path"])
        if not relative.endswith(".py"):
            continue
        source = candidate_root.joinpath(
            *PurePosixPath(relative).parts
        ).read_text(encoding="utf-8")
        _reject_nonfinite_masking(source, relative)
        _reject_unseeded_random_generators(source, relative)
        compile(source, relative, "exec", dont_inherit=True)
        compiled_files += 1
    _candidate_source_path(package, candidate_root)
    return {
        "candidate_file_count": len(manifest),
        "compiled_python_files": compiled_files,
        "source_tree_digest": source_digest,
    }


def _assert_candidate_tree_unchanged(
    candidate_root: Path,
    *,
    expected_manifest: tuple[dict[str, Any], ...],
    stage: QualificationStageV1,
) -> None:
    try:
        observed = snapshot_candidate_tree(candidate_root)
    except Exception as error:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.IMPLEMENTATION,
            "CANDIDATE_TREE_UNREADABLE_AFTER_STAGE",
            "candidate tree could not be re-read after qualification stage",
        ) from error
    if observed != expected_manifest:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.IMPLEMENTATION,
            "CANDIDATE_TREE_MUTATED_DURING_QUALIFICATION",
            "qualification stage changed candidate-package bytes",
        )


@contextmanager
def _candidate_package_import_root(candidate_root: Path):
    """Import one candidate as a real, isolated ``recclaw_ext`` package."""

    previous_path = list(sys.path)
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "recclaw_ext" or name.startswith("recclaw_ext.")
    }
    for name in tuple(previous_modules):
        sys.modules.pop(name, None)
    resolved_root = candidate_root.resolve()
    filtered_path = []
    for entry in previous_path:
        search_root = Path.cwd() if not entry else Path(entry)
        try:
            shadows_candidate = (
                search_root.resolve() != resolved_root
                and (search_root.resolve() / "recclaw_ext").is_dir()
            )
        except OSError:
            shadows_candidate = False
        if not shadows_candidate:
            filtered_path.append(entry)
    sys.path[:] = [str(resolved_root), *filtered_path]
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "recclaw_ext" or name.startswith("recclaw_ext."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path
        importlib.invalidate_caches()


def _load_candidate_class(
    package: CandidatePackageV1,
    candidate_root: Path,
) -> type[Any]:
    source_path, class_name = _candidate_source_path(package, candidate_root)
    module_name, _ = package.executable_entrypoint.split(":", 1)
    previous_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with _candidate_package_import_root(candidate_root):
            try:
                module = importlib.import_module(module_name)
            except Exception as error:
                raise _stage_failure(
                    QualificationStageV1.CONSTRUCTION,
                    QualificationFailureClassV1.INTERFACE,
                    "ENTRYPOINT_IMPORT_FAILED",
                    str(error) or "candidate package import failed",
                ) from error
    finally:
        sys.dont_write_bytecode = previous_dont_write
    candidate_class = getattr(module, class_name, None)
    if not isinstance(candidate_class, type):
        raise _stage_failure(
            QualificationStageV1.CONSTRUCTION,
            QualificationFailureClassV1.INTERFACE,
            "ENTRYPOINT_CLASS_MISSING",
            "entrypoint attribute is not a model class",
        )
    primitive_ids = getattr(module, "RECCLAW_IMPLEMENTED_PRIMITIVE_IDS", ())
    if isinstance(primitive_ids, (tuple, list)):
        setattr(
            candidate_class,
            "__recclaw_implemented_primitive_ids__",
            tuple(str(item) for item in primitive_ids),
        )
    component_specs = getattr(module, "RECCLAW_IMPLEMENTED_COMPONENT_SPECS", {})
    if isinstance(component_specs, Mapping):
        setattr(
            candidate_class,
            "__recclaw_implemented_component_specs__",
            canonical_value(component_specs),
        )
    setattr(
        candidate_class,
        "__recclaw_precompute_required_closed_form_solver__",
        bool(
            getattr(
                module,
                "RECCLAW_PRECOMPUTE_REQUIRED_CLOSED_FORM_SOLVER",
                False,
            )
        ),
    )
    return candidate_class


def _validate_precomputed_solver_runtime(
    runtime: Mapping[str, Any],
    interaction: Any,
    *,
    stage: QualificationStageV1,
) -> str:
    """Prove solver work occurs during construction and not execution paths."""

    import torch
    from recbole.data.interaction import Interaction

    candidate_class = runtime["candidate_class"]
    if not getattr(
        candidate_class,
        "__recclaw_precompute_required_closed_form_solver__",
        False,
    ):
        return "NOT_APPLICABLE"

    solver_calls = 0
    original_methods: dict[str, Any] = {}
    original_linalg: dict[str, Any] = {}

    def counted(callable_value: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal solver_calls
            solver_calls += 1
            return callable_value(*args, **kwargs)

        return wrapper

    for name, value in vars(candidate_class).items():
        if callable(value) and any(
            token in name.lower()
            for token in ("solve", "solver", "inverse", "pinv", "cholesky")
        ):
            original_methods[name] = value
            setattr(candidate_class, name, counted(value))
    for name in ("solve", "inv", "pinv", "cholesky"):
        value = getattr(torch.linalg, name, None)
        if callable(value):
            original_linalg[name] = value
            setattr(torch.linalg, name, counted(value))
    try:
        shadow = candidate_class(
            runtime["config"],
            runtime["train_data"]._dataset,
        ).to(runtime["config"]["device"])
        construction_calls = solver_calls
        if construction_calls < 1:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "precompute-required solver did not execute during construction",
                _candidate_method_repair_details("__init__"),
            )
        shadow.calculate_loss(interaction)
        shadow.predict(interaction)
        shadow.full_sort_predict(
            Interaction({shadow.USER_ID: interaction[shadow.USER_ID]})
        )
        if solver_calls != construction_calls:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "precompute-required solver recomputed during loss or scoring",
                _candidate_method_repair_details(
                    "__init__",
                    "calculate_loss",
                    "predict",
                    "full_sort_predict",
                ),
            )
    finally:
        for name, value in original_methods.items():
            setattr(candidate_class, name, value)
        for name, value in original_linalg.items():
            setattr(torch.linalg, name, value)
    return "BUILD_PHASE_ONLY_REUSED_BY_ALL_EXECUTION_PATHS"


def _negative_sampler_component_specs(candidate_class: type[Any]) -> tuple[Mapping[str, Any], ...]:
    component_specs = getattr(
        candidate_class,
        "__recclaw_implemented_component_specs__",
        {},
    )
    if not isinstance(component_specs, Mapping):
        return ()
    return tuple(
        spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping) and spec.get("slot_id") == "NEGATIVE_SAMPLER"
    )


def _requires_epoch_sampler_refresh(candidate_class: type[Any]) -> bool:
    return any(
        isinstance(spec.get("parameters"), Mapping)
        and spec["parameters"].get("refresh_frequency") == "EPOCH"
        for spec in _negative_sampler_component_specs(candidate_class)
    )


def _declares_model_curriculum_sampler(candidate_class: type[Any]) -> bool:
    """Return whether sampling hardness must follow the evolving model state."""

    return any(
        isinstance(spec.get("parameters"), Mapping)
        and spec["parameters"].get("hardness") in {"DYNAMIC", "CURRICULUM"}
        for spec in _negative_sampler_component_specs(candidate_class)
    )


def _declares_epoch_curriculum_sampler(candidate_class: type[Any]) -> bool:
    """Return whether hardness must progress with the declared epoch."""

    return any(
        isinstance(spec.get("parameters"), Mapping)
        and spec["parameters"].get("hardness") == "CURRICULUM"
        for spec in _negative_sampler_component_specs(candidate_class)
    )


def _intervene_sampler_model_parameters(candidate: Any) -> None:
    """Deterministically replace trainable model state for sampler causality."""

    import torch

    candidate_parameters = tuple(
        parameter
        for parameter in candidate.parameters()
        if parameter.requires_grad
        and (parameter.is_floating_point() or parameter.is_complex())
    )
    with torch.no_grad():
        for index, parameter in enumerate(candidate_parameters):
            if parameter.is_complex():
                real = torch.linspace(
                    -3.0,
                    3.0,
                    parameter.numel(),
                    device=parameter.device,
                    dtype=parameter.real.dtype,
                ).reshape(parameter.shape)
                replacement = torch.complex(
                    real,
                    torch.flip(real.reshape(-1), dims=(0,)).reshape(real.shape),
                )
            else:
                replacement = torch.linspace(
                    -3.0,
                    3.0,
                    parameter.numel(),
                    device=parameter.device,
                    dtype=parameter.dtype,
                ).reshape(parameter.shape)
            if index % 2:
                replacement = torch.flip(
                    replacement.reshape(-1), dims=(0,)
                ).reshape(parameter.shape)
            parameter.copy_(replacement)


def _validate_curriculum_sampler_causality(
    model: Any,
    interaction: Any,
    *,
    refresh_before_step: bool,
    stage: QualificationStageV1,
) -> str:
    """Prove that declared model-dependent hardness changes with model state."""

    import random

    import numpy as np
    import torch

    parameter_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    unsupported = object()
    framework_attributes = {
        "_parameters",
        "_buffers",
        "_modules",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }

    def clone_sampler_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, Mapping):
            cloned = copy(value)
            cloned.clear()
            for key, item in value.items():
                cloned_item = clone_sampler_value(item)
                if cloned_item is unsupported:
                    return unsupported
                cloned[key] = cloned_item
            return cloned
        if isinstance(value, list):
            cloned_items = [clone_sampler_value(item) for item in value]
            return (
                unsupported
                if any(item is unsupported for item in cloned_items)
                else cloned_items
            )
        if isinstance(value, tuple):
            cloned_items = tuple(clone_sampler_value(item) for item in value)
            return (
                unsupported
                if any(item is unsupported for item in cloned_items)
                else cloned_items
            )
        if isinstance(value, (set, frozenset)):
            return type(value)(value)
        module_name = type(value).__module__.split(".", 1)[0]
        copier = getattr(value, "copy", None)
        if module_name in {"numpy", "scipy"} and callable(copier):
            return copier()
        return unsupported

    sampler_attribute_names = {
        name
        for name, value in vars(model).items()
        if name not in framework_attributes
        and clone_sampler_value(value) is not unsupported
    }
    sampler_state = {
        name: clone_sampler_value(getattr(model, name))
        for name in sampler_attribute_names
    }

    def restore_sampler_state() -> None:
        for name in tuple(vars(model)):
            if name not in framework_attributes and name not in sampler_attribute_names:
                candidate = clone_sampler_value(getattr(model, name))
                if candidate is not unsupported:
                    delattr(model, name)
        for name, value in sampler_state.items():
            setattr(model, name, clone_sampler_value(value))

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    def restore_rng() -> None:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    def sampler_output(candidate: Any, epoch_idx: int = 0) -> Any:
        if refresh_before_step:
            candidate.recclaw_sampler_refresh(epoch_idx)
        output = candidate.recclaw_sampler_step(interaction)
        return output.detach().clone() if isinstance(output, torch.Tensor) else output

    try:
        restore_rng()
        control = sampler_output(model)
        model.load_state_dict(parameter_state)
        restore_sampler_state()
        restore_rng()
        _intervene_sampler_model_parameters(model)
        intervention = sampler_output(model)
    finally:
        model.load_state_dict(parameter_state)
        restore_sampler_state()
        restore_rng()
    hardness_changed = (
        isinstance(control, torch.Tensor)
        and isinstance(intervention, torch.Tensor)
        and control.shape == intervention.shape
        and not torch.equal(control.detach(), intervention.detach())
    )
    if not hardness_changed:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.IMPLEMENTATION,
            "SAMPLER_HARDNESS_NOT_CAUSAL",
            "DYNAMIC/CURRICULUM sampler output is invariant to the model "
            "state and therefore does not execute declared hardness",
            details={
                "sampler_hardness_probe": "MODEL_STATE_INTERVENTION_NO_EFFECT",
                **_candidate_method_repair_details(
                    "recclaw_sampler_refresh",
                    "recclaw_sampler_step",
                ),
            },
        )
    if _declares_epoch_curriculum_sampler(model.__class__):
        if not refresh_before_step:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "SAMPLER_CURRICULUM_EPOCH_REFRESH_MISSING",
                "CURRICULUM hardness requires an epoch-owned refresh lifecycle",
                _candidate_method_repair_details(
                    "recclaw_sampler_refresh",
                    "recclaw_sampler_step",
                ),
            )
        epoch_outputs: list[Any] = []
        try:
            for epoch_idx in (0, 1, 10, 50, 99):
                model.load_state_dict(parameter_state)
                restore_sampler_state()
                restore_rng()
                epoch_outputs.append(sampler_output(model, epoch_idx))
        finally:
            model.load_state_dict(parameter_state)
            restore_sampler_state()
            restore_rng()
        baseline = epoch_outputs[0]
        epoch_changed = any(
            isinstance(baseline, torch.Tensor)
            and isinstance(observed, torch.Tensor)
            and baseline.shape == observed.shape
            and not torch.equal(baseline.detach(), observed.detach())
            for observed in epoch_outputs[1:]
        )
        if not epoch_changed:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "SAMPLER_CURRICULUM_EPOCH_NOT_CAUSAL",
                "CURRICULUM sampler output is invariant across the training epoch progression",
                details={
                    "sampler_hardness_probe": "EPOCH_INTERVENTION_NO_EFFECT",
                    **_candidate_method_repair_details(
                        "recclaw_sampler_refresh",
                        "recclaw_sampler_step",
                    ),
                },
            )
        return "MODEL_AND_EPOCH_INTERVENTION_PASS"
    return "MODEL_STATE_INTERVENTION_PASS"


def _validate_epoch_sampler_runtime(
    model: Any,
    interaction: Any,
    *,
    stage: QualificationStageV1,
) -> tuple[Any, int]:
    """Prove that an EPOCH sampler is refreshed once then frozen in-epoch."""

    import torch

    refresh = getattr(model, "recclaw_sampler_refresh", None)
    sampler_step = getattr(model, "recclaw_sampler_step", None)
    if not callable(refresh) or not callable(sampler_step):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.IMPLEMENTATION,
            "SAMPLER_EPOCH_REFRESH_MISSING",
            "an EPOCH sampler requires recclaw_sampler_refresh and recclaw_sampler_step",
            _candidate_method_repair_details(
                "recclaw_sampler_refresh",
                "recclaw_sampler_step",
            ),
        )
    refresh(0)
    baseline = sampler_step(interaction)
    parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and (parameter.is_floating_point() or parameter.is_complex())
    )
    snapshots = tuple(parameter.detach().clone() for parameter in parameters)
    try:
        _intervene_sampler_model_parameters(model)
        observed = sampler_step(interaction)
    finally:
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)
    same = (
        isinstance(baseline, torch.Tensor)
        and isinstance(observed, torch.Tensor)
        and baseline.shape == observed.shape
        and torch.equal(baseline.detach(), observed.detach())
    )
    if not same:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.IMPLEMENTATION,
            "SAMPLER_EPOCH_CACHE_INVALID",
            "EPOCH sampler output changed before the next epoch refresh",
            _candidate_method_repair_details(
                "recclaw_sampler_refresh",
                "recclaw_sampler_step",
            ),
        )
    return baseline, 1


def _validate_entity_keyed_contrast_duplicate_invariance(
    single_total: Any,
    duplicated_total: Any,
    *,
    stage: QualificationStageV1,
) -> str:
    """Reject row-index contrast when repeated rows denote the same entity."""

    import torch

    if not (
        isinstance(single_total, torch.Tensor)
        and isinstance(duplicated_total, torch.Tensor)
        and single_total.numel() == duplicated_total.numel() == 1
        and torch.isclose(
            single_total.detach(),
            duplicated_total.detach(),
            rtol=1.0e-5,
            atol=1.0e-6,
        ).item()
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.IMPLEMENTATION,
            "ENTITY_KEYED_CONTRAST_DUPLICATE_ID_INVALID",
            "cross-layer entity InfoNCE changes when identical entity rows are "
            "duplicated; row positions are being treated as distinct negatives",
            details={
                "entity_keyed_contrast_probe": "DUPLICATE_ENTITY_CHANGED_LOSS"
            },
        )
    return "DUPLICATE_ENTITY_INVARIANCE_PASS"


def _numpy_compatibility_aliases() -> None:
    import numpy as np

    aliases = {
        "complex_": np.complex128,
        "float_": np.float64,
        "int_": np.int64,
        "string_": np.bytes_,
        "unicode_": np.str_,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def validate_p4_sparse_spectral_model_contract(
    model: Any,
    config: Any,
    dataset: Any,
    research_spec: OpenResearchSpecV1 | None = None,
) -> dict[str, Any]:
    """Fail closed when a P4 package smuggles in a BL model substrate."""

    import torch

    from recbole.utils import InputType, ModelType

    from .p4_fagsp_parent import FrozenFaGSPResidualModelBase

    candidate_class = model.__class__
    if candidate_class.__bases__ != (FrozenFaGSPResidualModelBase,):
        raise AssertionError(
            "P4 candidate must directly subclass FrozenFaGSPResidualModelBase"
        )
    mro_names = {base.__name__.lower() for base in candidate_class.__mro__}
    if {"bpr", "lightgcn"} & mro_names:
        raise AssertionError("P4 candidate inherits a forbidden BL substrate")
    if config["MODEL_TYPE"] is not ModelType.GENERAL:
        raise AssertionError("P4 candidate is not a general recommender")
    if model.input_type is not InputType.PAIRWISE:
        raise AssertionError("P4 candidate must use the fixed pairwise loader ABI")
    frozen_methods = (
        "calculate_loss",
        "predict",
        "full_sort_predict",
        "set_mechanism_enabled",
        "_score_matrix",
    )
    overridden_methods = tuple(
        name for name in frozen_methods if name in candidate_class.__dict__
    )
    if overridden_methods:
        raise AssertionError(
            "P4 candidate may not override the frozen parent path: "
            + ", ".join(overridden_methods)
        )
    if "_mechanism_residual_scores" not in candidate_class.__dict__:
        raise AssertionError("P4 candidate must directly implement its residual operator")
    embedding_modules = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Embedding)
    )
    if embedding_modules:
        raise AssertionError("P4 candidate contains trainable embedding modules")
    forbidden_parameter_names = tuple(
        name
        for name, _parameter in model.named_parameters()
        if any(
            token in name.lower().replace("_", "")
            for token in ("userembedding", "itemembedding")
        )
    )
    if forbidden_parameter_names:
        raise AssertionError("P4 candidate contains user/item embedding parameters")
    trainable_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_parameter_count = sum(
        int(parameter.numel()) for parameter in trainable_parameters
    )
    from recclaw_core.research_line.p4_runtime import p4_fit_mode
    fit_mode = p4_fit_mode(getattr(config, "final_config_dict", config))
    if fit_mode == "TRAIN_ONLY_PRECOMPUTE" and trainable_parameter_count:
        raise AssertionError("P4 train-only precompute has undeclared optimizer parameters")
    if fit_mode == "BPR_COEFFICIENTS" and trainable_parameter_count < 1:
        raise AssertionError(
            "P4 candidate requires at least one trainable operator coefficient"
        )
    if trainable_parameter_count > 4096:
        raise AssertionError(
            "P4 candidate exceeds the small operator-parameter allowance"
        )
    observation = {
        "canonical_fagsp_residual_subclass": True,
        "embedding_module_count": 0,
        "p4_model_contract": "PASS",
        "trainable_operator_parameter_count": trainable_parameter_count,
    }
    return observation


def _construct_runtime(
    package: CandidatePackageV1,
    research_spec: OpenResearchSpecV1,
    candidate_root: Path,
    fixture: RecBoleQualificationFixture,
) -> dict[str, Any]:
    _numpy_compatibility_aliases()
    candidate_class = _load_candidate_class(package, candidate_root)

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.abstract_recommender import GeneralRecommender
    from recbole.utils import init_seed

    execution_contract = research_spec.execution_contract
    base_model_config = fixture.base_model_config
    contract_config: Mapping[str, Any] = {}
    dataset_name = fixture.dataset
    data_path = str(fixture.data_path)
    benchmark_filename = ["train", "valid", "test"]
    if execution_contract is not None:
        required_contract_fields = {
            "capability_family",
            "model",
            "base_model_config",
            "config",
        }
        if set(execution_contract) != required_contract_fields:
            raise _stage_failure(
                QualificationStageV1.CONSTRUCTION,
                QualificationFailureClassV1.INTERFACE,
                "EXECUTION_CONTRACT_INVALID",
                "execution_contract does not contain the required execution fields",
            )
        base_model_config = str(execution_contract["base_model_config"])
        contract_config = dict(execution_contract["config"])
        # P4 qualification uses the bounded fixture; its real raw-data
        # protocol is applied only by the physical worker.
        if base_model_config != "P4SparseSpectral":
            # Legacy/core specs use the qualification fixture. Profiles with
            # explicit data bindings retain their actual dataset and partition.
            dataset_name = str(contract_config.get("dataset", dataset_name))
            data_path = str(contract_config.get("data_path", data_path))
            bound_benchmark = contract_config.get("benchmark_filename", benchmark_filename)
            benchmark_filename = list(bound_benchmark) if bound_benchmark is not None else None
        if fixture.base_model_config != base_model_config:
            fixture = replace(fixture, base_model_config=base_model_config)

    p4_sparse_spectral = base_model_config == "P4SparseSpectral"
    config_files = (
        fixture.project_root / "configs" / "p4_sparse_spectral.yaml"
        if p4_sparse_spectral else fixture.recbole_root
        / "recbole"
        / "properties"
        / "model"
        / f"{base_model_config}.yaml",
        fixture.project_root / "configs" / "task_ml1m.yaml",
        fixture.project_root / "configs" / "lightgcn_metrics.yaml",
    )
    if any(not path.is_file() for path in config_files):
        raise _stage_failure(
            QualificationStageV1.CONSTRUCTION,
            QualificationFailureClassV1.RUNTIME,
            "FROZEN_CONFIG_SOURCE_MISSING",
            "a frozen RecBole configuration source is missing",
        )
    launcher_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    config = Config(
        model=candidate_class if p4_sparse_spectral else base_model_config,
        dataset=dataset_name,
        config_file_list=[str(path) for path in config_files],
        config_dict={
            **contract_config,
            "benchmark_filename": benchmark_filename,
            "checkpoint_dir": str(fixture.checkpoint_dir),
            "data_path": data_path,
            "epochs": 1,
            # Qualification exercises optimizer/evaluator wiring. Full data
            # measurement belongs to the fixed-protocol physical worker.
            "recclaw_qualification_train_batch_limit": contract_config.get(
                "recclaw_qualification_train_batch_limit", 2
            ),
            "recclaw_qualification_eval_batch_limit": contract_config.get(
                "recclaw_qualification_eval_batch_limit", 2
            ),
            "eval_batch_size": 32,
            "eval_step": 1,
            "reproducibility": True,
            "seed": fixture.seed,
            "show_progress": False,
            "state": "ERROR",
            "stopping_step": 1,
            "topk": [3],
            "train_batch_size": 8,
            **(
                {"gpu_id": launcher_cuda_visible_devices}
                if execution_contract is not None and launcher_cuda_visible_devices
                else {}
            ),
            "use_gpu": execution_contract is not None,
            "valid_metric": "NDCG@3",
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = candidate_class(config, train_data._dataset).to(config["device"])
    prepare_candidate_train_data(model, train_data)
    if execution_contract is None and not isinstance(model, GeneralRecommender):
        raise _stage_failure(
            QualificationStageV1.CONSTRUCTION,
            QualificationFailureClassV1.INTERFACE,
            "NOT_GENERAL_RECOMMENDER",
            "entrypoint is not a RecBole GeneralRecommender",
        )
    if model.type is not config["MODEL_TYPE"]:
        raise _stage_failure(
            QualificationStageV1.CONSTRUCTION,
            QualificationFailureClassV1.INTERFACE,
            "MODEL_TYPE_MISMATCH",
            "candidate model type does not match the frozen RecBole Config",
        )
    return {
        "candidate_class": candidate_class,
        "candidate_root": candidate_root,
        "config": config,
        "dataset": dataset,
        "execution_contract": execution_contract,
        "model": model,
        "train_dataset": train_data._dataset,
        "research_spec": research_spec,
        "p4_sparse_spectral": p4_sparse_spectral,
        "test_data": test_data,
        "train_data": train_data,
        "valid_data": valid_data,
    }


def _declared_relation_score_paths(candidate_class: type[Any]) -> tuple[str, ...]:
    """Return relation components declared upstream of a score head."""

    specs = getattr(
        candidate_class,
        "__recclaw_implemented_component_specs__",
        {},
    )
    if not isinstance(specs, Mapping):
        return ()
    relation_ids = {
        str(component_id)
        for component_id, spec in specs.items()
        if isinstance(spec, Mapping)
        and (
            spec.get("slot_id") == "RELATION_VIEW"
            or any(
                isinstance(input_spec, Mapping)
                and isinstance(input_spec.get("source"), Mapping)
                and input_spec["source"].get("kind") == "DATA"
                and (
                    str(input_spec["source"].get("data_role", "")).endswith("_GRAPH")
                    or "RELATION" in str(input_spec["source"].get("data_role", ""))
                    or "ADJACENCY" in str(input_spec["source"].get("data_role", ""))
                )
                for input_spec in spec.get("inputs", ())
            )
        )
    }
    score_ids = {
        str(component_id)
        for component_id, spec in specs.items()
        if isinstance(spec, Mapping) and spec.get("slot_id") == "SCORE_HEAD"
    }
    outgoing: dict[str, set[str]] = {}
    for component_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            continue
        for input_spec in spec.get("inputs", ()):
            if not isinstance(input_spec, Mapping):
                continue
            source = input_spec.get("source")
            if not isinstance(source, Mapping) or source.get("kind") != "COMPONENT":
                continue
            source_id = source.get("component_id")
            if isinstance(source_id, str) and source_id:
                outgoing.setdefault(source_id, set()).add(str(component_id))

    structural_ids = {
        str(component_id)
        for component_id, spec in specs.items()
        if isinstance(spec, Mapping)
        and (
            spec.get("slot_id") in {"MESSAGE", "PROPAGATION_AGGREGATION"}
            or (
                spec.get("slot_id") == "ENCODER"
                and spec.get("primitive_id") != "encoder.none_mf"
                and any(
                    isinstance(input_spec, Mapping)
                    and isinstance(input_spec.get("source"), Mapping)
                    and input_spec["source"].get("kind") == "COMPONENT"
                    and input_spec["source"].get("component_id") in relation_ids
                    for input_spec in spec.get("inputs", ())
                )
            )
        )
    }
    causal_relations: list[str] = []
    for relation_id in sorted(relation_ids):
        frontier = [(relation_id, relation_id in structural_ids)]
        visited = set(frontier)
        while frontier:
            current, seen_structural = frontier.pop()
            if current in score_ids and seen_structural:
                causal_relations.append(relation_id)
                break
            for successor in outgoing.get(current, ()):
                state = (
                    successor,
                    seen_structural or successor in structural_ids,
                )
                if state not in visited:
                    visited.add(state)
                    frontier.append(state)
    return tuple(causal_relations)


def _declared_training_only_relation_dependencies(
    candidate_class: type[Any],
) -> tuple[str, ...]:
    """Return declared relation consumers whose authority is training-only."""

    specs = getattr(candidate_class, "__recclaw_implemented_component_specs__", {})
    if not isinstance(specs, Mapping):
        return ()
    relation_ids = {
        str(component_id)
        for component_id, spec in specs.items()
        if isinstance(spec, Mapping)
        and (
            spec.get("slot_id") == "RELATION_VIEW"
            or any(
                isinstance(input_spec, Mapping)
                and isinstance(input_spec.get("source"), Mapping)
                and input_spec["source"].get("kind") == "DATA"
                and (
                    str(input_spec["source"].get("data_role", "")).endswith("_GRAPH")
                    or "RELATION" in str(input_spec["source"].get("data_role", ""))
                    or "ADJACENCY" in str(input_spec["source"].get("data_role", ""))
                )
                for input_spec in spec.get("inputs", ())
            )
        )
    }
    outgoing: dict[str, set[str]] = {}
    for component_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            continue
        for input_spec in spec.get("inputs", ()):
            if not isinstance(input_spec, Mapping):
                continue
            source = input_spec.get("source")
            if isinstance(source, Mapping) and source.get("kind") == "COMPONENT":
                source_id = source.get("component_id")
                if isinstance(source_id, str):
                    outgoing.setdefault(source_id, set()).add(str(component_id))

    training_sinks = {
        str(component_id)
        for component_id, spec in specs.items()
        if isinstance(spec, Mapping) and spec.get("slot_id") == "TRAINING_PROCEDURE"
    }

    def reaches_training_without_score(component_id: str) -> bool:
        pending = [component_id]
        visited = {component_id}
        while pending:
            current = pending.pop()
            if current in training_sinks:
                return True
            for successor in outgoing.get(current, ()):
                successor_spec = specs.get(successor)
                if (
                    isinstance(successor_spec, Mapping)
                    and successor_spec.get("slot_id") == "SCORE_HEAD"
                ):
                    continue
                if successor not in visited:
                    visited.add(successor)
                    pending.append(successor)
        return False

    score_relation_ids = set(_declared_relation_score_paths(candidate_class))
    dependencies: set[str] = set()
    for relation_id in relation_ids - score_relation_ids:
        pending = list(outgoing.get(relation_id, ()))
        visited = set(pending)
        while pending:
            current = pending.pop()
            spec = specs.get(current)
            if not isinstance(spec, Mapping):
                continue
            slot_id = spec.get("slot_id")
            if slot_id == "SCORE_HEAD" or not reaches_training_without_score(current):
                continue
            if slot_id == "RELATION_VIEW":
                for successor in outgoing.get(current, ()):
                    if successor not in visited:
                        visited.add(successor)
                        pending.append(successor)
                continue
            dependencies.add(current)

    for component_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            continue
        slot_id = spec.get("slot_id")
        direct_roles = {
            str(input_spec["source"].get("data_role", ""))
            for input_spec in spec.get("inputs", ())
            if isinstance(input_spec, Mapping)
            and isinstance(input_spec.get("source"), Mapping)
            and input_spec["source"].get("kind") == "DATA"
        }
        sampler_direct = (
            slot_id == "NEGATIVE_SAMPLER"
            and bool(direct_roles.intersection({"TRAIN_INTERACTIONS", "TRAIN_USER_ITEM_GRAPH"}))
        )
        structural_direct = any(
            role.endswith("_GRAPH") or "RELATION" in role or "ADJACENCY" in role
            for role in direct_roles
        )
        if sampler_direct or (
            structural_direct
            and str(component_id) not in relation_ids
            and slot_id not in {"SCORE_HEAD", "TRAINING_PROCEDURE"}
            and reaches_training_without_score(str(component_id))
        ):
            dependencies.add(str(component_id))
    return tuple(sorted(dependencies))


def _degree_preserving_relation_probe_dataset(
    dataset: Any,
    *,
    user_field: str,
    item_field: str,
) -> tuple[Any, tuple[int, ...], tuple[tuple[int, int], ...]] | None:
    """Clone training data with deterministic bipartite double-edge swaps."""

    import torch
    from recbole.data.interaction import Interaction

    inter_feat = getattr(dataset, "inter_feat", None)
    if inter_feat is None:
        return None
    users = inter_feat[user_field].detach().cpu().long().tolist()
    items = inter_feat[item_field].detach().cpu().long().tolist()
    original_items = list(items)
    if len(users) < 4 or len(users) != len(items):
        return None
    edges = set(zip(users, items))
    changed_users: set[int] = set()
    swap_count = 0
    swap_limit = min(64, max(2, len(users) // 8))
    for left in range(len(users)):
        if swap_count >= swap_limit:
            break
        for right in range(left + 1, len(users)):
            user_left, item_left = users[left], items[left]
            user_right, item_right = users[right], items[right]
            if user_left == user_right or item_left == item_right:
                continue
            swapped_left = (user_left, item_right)
            swapped_right = (user_right, item_left)
            if swapped_left in edges or swapped_right in edges:
                continue
            edges.remove((user_left, item_left))
            edges.remove((user_right, item_right))
            edges.add(swapped_left)
            edges.add(swapped_right)
            items[left], items[right] = item_right, item_left
            changed_users.update((user_left, user_right))
            swap_count += 1
            break
    if swap_count < 2:
        return None

    values = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inter_feat.interaction.items()
    }
    values[item_field] = torch.tensor(
        items,
        dtype=inter_feat[item_field].dtype,
        device=inter_feat[item_field].device,
    )
    probe = copy(dataset)
    probe.inter_feat = Interaction(values)
    rewired_edges = set(zip(users, items))
    affected_edges = tuple(
        (int(user_id), int(original_item))
        for user_id, original_item, rewired_item in zip(
            users, original_items, items
        )
        if original_item != rewired_item
        and (int(user_id), int(original_item)) not in rewired_edges
    )
    actually_changed_users = tuple(sorted({user_id for user_id, _ in affected_edges}))
    if not affected_edges:
        return None
    return probe, actually_changed_users, affected_edges


def _validate_declared_relation_causality(
    runtime: Mapping[str, Any],
    interaction: Any,
    *,
    stage: QualificationStageV1,
) -> dict[str, Any]:
    """Prove a declared relation-to-score path affects train and eval behavior."""

    import torch
    from recbole.data.interaction import Interaction
    from recbole.utils import init_seed

    model = runtime["model"]
    candidate_class = runtime["candidate_class"]
    relation_paths = _declared_relation_score_paths(candidate_class)
    training_only_dependencies = _declared_training_only_relation_dependencies(
        candidate_class
    )
    component_specs = getattr(
        candidate_class, "__recclaw_implemented_component_specs__", {}
    )
    sampler_only_dependencies = tuple(
        dependency
        for dependency in training_only_dependencies
        if isinstance(component_specs, Mapping)
        and isinstance(component_specs.get(dependency), Mapping)
        and component_specs[dependency].get("slot_id") == "NEGATIVE_SAMPLER"
    )
    loss_only_dependencies = tuple(
        dependency
        for dependency in training_only_dependencies
        if dependency not in sampler_only_dependencies
    )
    probe_result = _degree_preserving_relation_probe_dataset(
        runtime["train_dataset"],
        user_field=model.USER_ID,
        item_field=model.ITEM_ID,
    )
    if probe_result is None:
        return {
            "declared_relation_score_paths": relation_paths,
            "declared_training_only_relation_dependencies": training_only_dependencies,
            "declared_sampler_only_relation_dependencies": sampler_only_dependencies,
            "declared_loss_only_relation_dependencies": loss_only_dependencies,
            "relation_causality_probe": "STATIC_CAUSAL_PATH_PROOF_ONLY",
            "relation_intervention_unavailable": True,
        }
    probe_dataset, changed_users, affected_edges = probe_result
    config = runtime["config"]
    init_seed(config["seed"], config["reproducibility"])
    shadow = candidate_class(config, probe_dataset).to(config["device"])

    # Hold learned state exactly fixed.  Only dataset-derived relation state may
    # differ between the original and intervention models; independent random
    # initialization would otherwise manufacture a false causal effect.
    original_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, parameter in shadow.named_parameters():
            original = original_parameters.get(name)
            if original is not None and original.shape == parameter.shape:
                parameter.copy_(original.detach())

    for candidate in (model, shadow):
        refresh = getattr(candidate, "recclaw_sampler_refresh", None)
        if callable(refresh):
            refresh(0)
        candidate.eval()

    selected_edges = []
    seen_users = set()
    for user_id, item_id in affected_edges:
        if user_id not in seen_users:
            selected_edges.append((user_id, item_id))
            seen_users.add(user_id)
        if len(selected_edges) >= 8:
            break
    probe_users = torch.tensor(
        [user_id for user_id, _ in selected_edges],
        dtype=interaction[model.USER_ID].dtype,
        device=config["device"],
    )
    probe_positive_items = torch.tensor(
        [item_id for _, item_id in selected_edges],
        dtype=interaction[model.ITEM_ID].dtype,
        device=config["device"],
    )

    original_feat = runtime["train_dataset"].inter_feat
    rewired_feat = probe_dataset.inter_feat
    forbidden: dict[int, set[int]] = {}
    for feat in (original_feat, rewired_feat):
        feat_users = feat[model.USER_ID].detach().cpu().long().tolist()
        feat_items = feat[model.ITEM_ID].detach().cpu().long().tolist()
        for user_id, item_id in zip(feat_users, feat_items):
            if user_id in seen_users:
                forbidden.setdefault(int(user_id), set()).add(int(item_id))
    probe_negative_items = []
    for user_id, positive_id in selected_edges:
        negative_id = next(
            item_id
            for item_id in range(1, int(model.n_items))
            if item_id != positive_id
            and item_id not in forbidden.get(user_id, set())
        )
        probe_negative_items.append(negative_id)
    loss_interaction = Interaction(
        {
            model.USER_ID: probe_users,
            model.ITEM_ID: probe_positive_items,
            model.NEG_ITEM_ID: torch.tensor(
                probe_negative_items,
                dtype=interaction[model.NEG_ITEM_ID].dtype,
                device=config["device"],
            ),
        }
    )

    # A relation intervention must not be confounded by a different negative
    # sample.  Both models receive the identical sampler output; sampler
    # fidelity is proven separately by the sampler hook checks.
    model_sampler = getattr(model, "recclaw_sampler_step", None)
    if callable(model_sampler):
        sampled_negative = model_sampler(loss_interaction)
        fixed_negative = (
            sampled_negative.detach().clone()
            if isinstance(sampled_negative, torch.Tensor)
            else loss_interaction[model.NEG_ITEM_ID].detach().clone()
        )
    else:
        fixed_negative = loss_interaction[model.NEG_ITEM_ID].detach().clone()
    original_sampler_hooks = {}
    for candidate in (model, shadow):
        hook = getattr(candidate, "recclaw_sampler_step", None)
        if callable(hook):
            original_sampler_hooks[candidate] = hook

            def fixed_sampler(*_args: Any, _value: Any = fixed_negative, **_kwargs: Any) -> Any:
                return _value.clone()

            setattr(candidate, "recclaw_sampler_step", fixed_sampler)

    full_sort_interaction = Interaction({model.USER_ID: probe_users})
    pair_interaction = Interaction(
        {
            model.USER_ID: probe_users,
            model.ITEM_ID: probe_positive_items,
        }
    )

    def loss_total(candidate: Any, probe_interaction: Any) -> Any:
        value = candidate.calculate_loss(probe_interaction)
        parts = value if isinstance(value, tuple) else (value,)
        if not parts or any(
            not isinstance(part, torch.Tensor)
            or part.numel() != 1
            or not torch.isfinite(part).all().item()
            for part in parts
        ):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.INTERFACE,
                "LOSS_CONTRACT_FAILED",
                "calculate_loss must return finite scalar tensor values",
                _candidate_method_repair_details("calculate_loss"),
            )
        return torch.stack([part.reshape(()) for part in parts]).sum()

    def install_fixed_sampler(value: Any) -> None:
        for candidate in original_sampler_hooks:

            def fixed_sampler(*_args: Any, _value: Any = value, **_kwargs: Any) -> Any:
                return _value.clone()

            setattr(candidate, "recclaw_sampler_step", fixed_sampler)

    def clear_standard_restore_cache(candidate: Any) -> None:
        # RecBole full-sort inference may retain the previous representation.
        # Clear only the stable RecBole restore ABI, never candidate-specific
        # mechanism state, so both sides observe their own intervened relation.
        for attribute in ("restore_user_e", "restore_item_e"):
            if hasattr(candidate, attribute):
                setattr(candidate, attribute, None)

    try:
        install_fixed_sampler(fixed_negative)
        with torch.no_grad():
            original_loss = loss_total(model, loss_interaction)
            shadow_loss = loss_total(shadow, loss_interaction)

            original_single_losses = []
            shadow_single_losses = []
            for index in range(len(selected_edges)):
                singleton = Interaction(
                    {
                        model.USER_ID: probe_users[index : index + 1],
                        model.ITEM_ID: probe_positive_items[index : index + 1],
                        model.NEG_ITEM_ID: loss_interaction[model.NEG_ITEM_ID][
                            index : index + 1
                        ],
                    }
                )
                singleton_negative = (
                    fixed_negative[index : index + 1]
                    if isinstance(fixed_negative, torch.Tensor)
                    and fixed_negative.ndim > 0
                    and fixed_negative.shape[0] == len(selected_edges)
                    else singleton[model.NEG_ITEM_ID].detach().clone()
                )
                install_fixed_sampler(singleton_negative)
                original_single_losses.append(loss_total(model, singleton))
                shadow_single_losses.append(loss_total(shadow, singleton))
            install_fixed_sampler(fixed_negative)
            original_single_loss_vector = torch.stack(original_single_losses)
            shadow_single_loss_vector = torch.stack(shadow_single_losses)

            clear_standard_restore_cache(model)
            clear_standard_restore_cache(shadow)
            original_predict = model.predict(pair_interaction)
            shadow_predict = shadow.predict(pair_interaction)
            clear_standard_restore_cache(model)
            clear_standard_restore_cache(shadow)
            original_full_sort = model.full_sort_predict(full_sort_interaction)
            shadow_full_sort = shadow.full_sort_predict(full_sort_interaction)
    finally:
        for candidate, hook in original_sampler_hooks.items():
            setattr(candidate, "recclaw_sampler_step", hook)

    for method_name, values in (
        ("predict", (original_predict, shadow_predict)),
        ("full_sort_predict", (original_full_sort, shadow_full_sort)),
    ):
        reason_code = {
            "predict": "PREDICT_CONTRACT_FAILED",
            "full_sort_predict": "FULL_SORT_CONTRACT_FAILED",
        }[method_name]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.INTERFACE,
                reason_code,
                f"{method_name} must return tensor values",
                _candidate_method_repair_details(method_name),
            )
        expected_cardinality = {
            "predict": len(selected_edges),
            "full_sort_predict": len(selected_edges) * int(model.n_items),
        }[method_name]
        if any(value.numel() != expected_cardinality for value in values):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.INTERFACE,
                reason_code,
                f"{method_name} must return exactly {expected_cardinality} score values",
                _candidate_method_repair_details(method_name),
            )
        if any(not torch.isfinite(value).all().item() for value in values):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.INTERFACE,
                reason_code,
                f"{method_name} must return finite tensor values",
                _candidate_method_repair_details(method_name),
            )

    def changed(left: Any, right: Any) -> bool:
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and torch.isfinite(left).all().item()
            and torch.isfinite(right).all().item()
            and not torch.equal(left.detach(), right.detach())
        )

    def maximum_absolute_delta(left: Any, right: Any) -> float | None:
        if (
            not isinstance(left, torch.Tensor)
            or not isinstance(right, torch.Tensor)
            or left.shape != right.shape
        ):
            return None
        return float((left.detach() - right.detach()).abs().max().cpu().item())

    loss_per_sample_abs_delta = (
        original_single_loss_vector.detach() - shadow_single_loss_vector.detach()
    ).abs()
    predict_per_sample_abs_delta = (
        original_predict.detach() - shadow_predict.detach()
    ).abs().reshape(len(selected_edges), -1).max(dim=1).values
    full_sort_per_sample_abs_delta = (
        original_full_sort.detach() - shadow_full_sort.detach()
    ).abs().reshape(len(selected_edges), -1).max(dim=1).values
    effects = {
        "calculate_loss": changed(original_loss, shadow_loss)
        or changed(original_single_loss_vector, shadow_single_loss_vector),
        "predict": changed(original_predict, shadow_predict),
        "full_sort_predict": changed(original_full_sort, shadow_full_sort),
    }
    effect_deltas = {
        "calculate_loss": maximum_absolute_delta(original_loss, shadow_loss),
        "predict": maximum_absolute_delta(original_predict, shadow_predict),
        "full_sort_predict": maximum_absolute_delta(
            original_full_sort, shadow_full_sort
        ),
    }
    effect_details = {
        "relation_intervention_effects": effects,
        "relation_intervention_max_abs_delta": effect_deltas,
        "relation_intervention_loss_per_sample_abs_delta": [
            float(value) for value in loss_per_sample_abs_delta.cpu().tolist()
        ],
        "relation_intervention_predict_per_sample_max_abs_delta": [
            float(value) for value in predict_per_sample_abs_delta.cpu().tolist()
        ],
        "relation_intervention_full_sort_per_sample_max_abs_delta": [
            float(value) for value in full_sort_per_sample_abs_delta.cpu().tolist()
        ],
        "relation_intervention_probe_users": [
            int(user_id) for user_id, _ in selected_edges
        ],
        "relation_intervention_affected_edges": [
            [int(user_id), int(item_id)] for user_id, item_id in selected_edges
        ],
        "static_call_responsibility": {
            "declared_relation_score_paths": relation_paths,
            "declared_sampler_only_relation_dependencies": sampler_only_dependencies,
            "declared_loss_only_relation_dependencies": loss_only_dependencies,
        },
    }
    if relation_paths:
        if not effects["predict"] or not effects["full_sort_predict"]:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "DECLARED_RELATION_NOT_SCORE_CAUSAL",
                "degree-preserving training-edge intervention did not affect both "
                "declared evaluation score paths",
                details={
                    **effect_details,
                    **_candidate_method_repair_details(
                        "predict",
                        "full_sort_predict",
                    ),
                },
            )
        if not effects["calculate_loss"]:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "TRAIN_EVAL_MECHANISM_CAUSALITY_MISMATCH",
                "declared relation affects evaluation but not the training objective",
                details={
                    **effect_details,
                    **_candidate_method_repair_details("calculate_loss"),
                },
            )
        probe_status = "DECLARED_SCORE_RELATION_CAUSAL"
    elif loss_only_dependencies:
        if effects["predict"] or effects["full_sort_predict"]:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "TRAINING_ONLY_RELATION_LEAKED_INTO_SCORE",
                "a sampler/regularizer-only relation dependency changed an evaluation score path",
                details=effect_details,
            )
        if not effects["calculate_loss"]:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "DECLARED_TRAINING_ONLY_RELATION_NOT_LOSS_CAUSAL",
                "declared relation-dependent regularizer or auxiliary objective did not affect calculate_loss",
                details={
                    **effect_details,
                    **_candidate_method_repair_details("calculate_loss"),
                },
            )
        probe_status = "DECLARED_TRAINING_ONLY_RELATION_CONFINED"
    else:
        if any(effects.values()):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "UNDECLARED_RELATION_CAUSALITY",
                "training-edge intervention changed an execution path without a compiler-owned relation responsibility",
                details=effect_details,
            )
        probe_status = "NO_UNDECLARED_RELATION_EFFECT"
    return {
        "declared_relation_score_paths": relation_paths,
        "declared_training_only_relation_dependencies": training_only_dependencies,
        "declared_sampler_only_relation_dependencies": sampler_only_dependencies,
        "declared_loss_only_relation_dependencies": loss_only_dependencies,
        "relation_causality_probe": probe_status,
        "relation_intervention_effects": effects,
        "relation_intervention_max_abs_delta": effect_deltas,
        "relation_intervention_loss_per_sample_abs_delta": effect_details[
            "relation_intervention_loss_per_sample_abs_delta"
        ],
        "relation_intervention_predict_per_sample_max_abs_delta": effect_details[
            "relation_intervention_predict_per_sample_max_abs_delta"
        ],
        "relation_intervention_full_sort_per_sample_max_abs_delta": effect_details[
            "relation_intervention_full_sort_per_sample_max_abs_delta"
        ],
        "relation_intervention_probe_user_ids": effect_details[
            "relation_intervention_probe_users"
        ],
        "relation_intervention_affected_edges": effect_details[
            "relation_intervention_affected_edges"
        ],
        "static_call_responsibility": effect_details["static_call_responsibility"],
        "relation_intervention_edge_count": len(affected_edges),
        "relation_intervention_probe_users": len(selected_edges),
        "rewired_user_count": len(changed_users),
    }


def _validate_profile_native_api_contract(
    runtime: Mapping[str, Any],
    *,
    stage: QualificationStageV1,
    require_backward: bool,
) -> dict[str, Any]:
    """Exercise the native RecBole API for non-General-pairwise profiles."""

    import torch

    from recbole.utils import InputType

    model = runtime["model"]
    config = runtime["config"]
    train_data = runtime["train_data"]
    input_type = config["MODEL_INPUT_TYPE"]
    if model.type is not config["MODEL_TYPE"]:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "MODEL_TYPE_MISMATCH",
            "candidate model type does not match the frozen RecBole Config",
        )
    for method_name in ("calculate_loss", "full_sort_predict"):
        if not callable(getattr(model, method_name, None)):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.INTERFACE,
                "REQUIRED_METHOD_MISSING",
                f"candidate lacks required method {method_name}",
            )

    interaction = next(iter(train_data)).to(config["device"])
    available_fields = frozenset(interaction.interaction)
    batch_size = len(interaction)
    model.train()
    loss = model.calculate_loss(interaction)
    losses = loss if isinstance(loss, tuple) else (loss,)
    if not losses or any(
        not isinstance(item, torch.Tensor)
        or item.numel() != 1
        or not torch.isfinite(item).all().item()
        for item in losses
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "LOSS_CONTRACT_FAILED",
            "calculate_loss must return finite scalar tensor values",
            _candidate_method_repair_details("calculate_loss"),
        )

    backward_parameter_count = 0
    if require_backward:
        model.zero_grad(set_to_none=True)
        total_loss = torch.stack([item.reshape(()) for item in losses]).sum()
        if not total_loss.requires_grad:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "calculate_loss did not produce a differentiable scalar",
                _candidate_method_repair_details("calculate_loss"),
            )
        try:
            total_loss.backward()
        except Exception as error:  # noqa: BLE001 - typed below.
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "calculate_loss could not backpropagate",
                _candidate_method_repair_details("calculate_loss"),
            ) from error
        gradients = tuple(
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if not gradients or any(
            not torch.isfinite(gradient).all().item() for gradient in gradients
        ):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "calculate_loss produced no finite parameter gradients",
                _candidate_method_repair_details("calculate_loss"),
            )
        backward_parameter_count = len(gradients)

    native_batch = interaction[: min(2, batch_size)]
    model.eval()
    with torch.no_grad():
        full_sort = model.full_sort_predict(native_batch)
        prediction = None
        if input_type is not InputType.LISTWISE and callable(
            getattr(model, "predict", None)
        ):
            prediction = model.predict(interaction)
    expected_full_sort = len(native_batch) * int(model.n_items)
    if (
        not isinstance(full_sort, torch.Tensor)
        or full_sort.numel() != expected_full_sort
        or not torch.isfinite(full_sort).all().item()
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "FULL_SORT_CONTRACT_FAILED",
            "full_sort_predict output does not cover every item",
            _candidate_method_repair_details("full_sort_predict"),
        )
    if prediction is not None and (
        not isinstance(prediction, torch.Tensor)
        or prediction.numel() != batch_size
        or not torch.isfinite(prediction).all().item()
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "PREDICT_CONTRACT_FAILED",
            "predict output does not match the interaction batch",
            _candidate_method_repair_details("predict"),
        )
    return {
        "backward_parameter_count": backward_parameter_count,
        "calculate_loss_scalars": len(losses),
        "full_sort_values": full_sort.numel(),
        "input_fields": tuple(sorted(available_fields)),
        "input_type": input_type.name.lower(),
        "model_type": model.type.name.lower(),
        "predict_values": 0 if prediction is None else prediction.numel(),
        "profile_native_contract": True,
    }


def _validate_userwise_autoencoder_contract(
    runtime: Mapping[str, Any],
    *,
    stage: QualificationStageV1,
    require_backward: bool,
) -> dict[str, Any]:
    """Validate RecBole's registered user-wise autoencoder training ABI.

    RecBole intentionally labels MultiVAE ``PAIRWISE`` while routing it through
    ``UserDataLoader``.  Such batches contain a user id only; positive and
    negative item ids are scoring inputs, not loss inputs.  Keeping this branch
    separate prevents weakening the pairwise checks used by every existing
    BPR/LightGCN candidate.
    """

    import torch

    from recbole.data.interaction import Interaction

    model = runtime["model"]
    config = runtime["config"]
    train_data = runtime["train_data"]
    interaction = next(iter(train_data)).to(config["device"])
    available_fields = frozenset(interaction.interaction)
    if model.USER_ID not in available_fields:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "INTERACTION_FIELDS_MISSING",
            "user-wise interaction is missing the user id field",
        )
    for method_name in ("calculate_loss", "predict", "full_sort_predict"):
        if not callable(getattr(model, method_name, None)):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.INTERFACE,
                "REQUIRED_METHOD_MISSING",
                f"candidate lacks required method {method_name}",
            )

    model.train()
    loss = model.calculate_loss(interaction)
    losses = loss if isinstance(loss, tuple) else (loss,)
    if not losses or any(
        not isinstance(item, torch.Tensor)
        or item.numel() != 1
        or not torch.isfinite(item).all().item()
        for item in losses
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "LOSS_CONTRACT_FAILED",
            "user-wise calculate_loss must return finite scalar tensors",
            _candidate_method_repair_details("calculate_loss"),
        )

    backward_parameter_count = 0
    if require_backward:
        model.zero_grad(set_to_none=True)
        total_loss = torch.stack([item.reshape(()) for item in losses]).sum()
        if not total_loss.requires_grad:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "user-wise loss is not differentiable",
                _candidate_method_repair_details("calculate_loss"),
            )
        try:
            total_loss.backward()
        except Exception as error:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "user-wise loss could not backpropagate",
                _candidate_method_repair_details("calculate_loss"),
            ) from error
        gradients = tuple(
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if not gradients or any(
            not torch.isfinite(gradient).all().item() for gradient in gradients
        ):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "user-wise loss produced no finite parameter gradients",
                _candidate_method_repair_details("calculate_loss"),
            )
        backward_parameter_count = len(gradients)

    users = interaction[model.USER_ID]
    batch_size = int(users.shape[0])
    probe_size = min(2, batch_size)
    probe_users = users[:probe_size]
    probe_items = torch.arange(
        1,
        probe_size + 1,
        dtype=torch.long,
        device=config["device"],
    )
    probe_items = 1 + ((probe_items - 1) % max(1, int(model.n_items) - 1))
    scoring_interaction = Interaction(
        {
            model.USER_ID: probe_users,
            model.ITEM_ID: probe_items,
        }
    )
    model.eval()
    with torch.no_grad():
        prediction = model.predict(scoring_interaction)
        full_sort = model.full_sort_predict(
            Interaction({model.USER_ID: probe_users})
        )
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.numel() != probe_size
        or not torch.isfinite(prediction).all().item()
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "PREDICT_CONTRACT_FAILED",
            "user-wise predict output does not match its scoring batch",
            _candidate_method_repair_details("predict"),
        )
    expected_full_sort = probe_size * int(model.n_items)
    if (
        not isinstance(full_sort, torch.Tensor)
        or full_sort.numel() != expected_full_sort
        or not torch.isfinite(full_sort).all().item()
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "FULL_SORT_CONTRACT_FAILED",
            "user-wise full_sort_predict does not cover every item",
            _candidate_method_repair_details("full_sort_predict"),
        )
    return {
        "calculate_loss_scalars": len(losses),
        "solver_precompute_probe": "NOT_APPLICABLE",
        "backward_parameter_count": backward_parameter_count,
        "input_fields": (model.USER_ID,),
        "input_type": model.input_type.name.lower(),
        "model_type": model.type.name.lower(),
        "training_loader_contract": "USERWISE_AUTOENCODER",
        "predict_values": prediction.numel(),
        "full_sort_values": full_sort.numel(),
        "loss_batch_size": batch_size,
    }


def _validate_api_contract(
    runtime: Mapping[str, Any],
    *,
    stage: QualificationStageV1 = QualificationStageV1.API_CONTRACT,
    require_backward: bool = False,
) -> dict[str, Any]:
    import torch

    from recbole.data.interaction import Interaction
    from recbole.utils import InputType, ModelType

    model = runtime["model"]
    config = runtime["config"]
    train_data = runtime["train_data"]
    if runtime.get("execution_contract") is not None and not (
        model.type is ModelType.GENERAL and model.input_type is InputType.PAIRWISE
    ):
        return _validate_profile_native_api_contract(
            runtime,
            stage=stage,
            require_backward=require_backward,
        )
    if model.type is not ModelType.GENERAL:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "MODEL_TYPE_MISMATCH",
            "candidate model type is not ModelType.GENERAL",
        )
    if model.input_type is not InputType.PAIRWISE:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "INPUT_TYPE_MISMATCH",
            "candidate input_type is not InputType.PAIRWISE",
        )
    if (
        str(config["model"]) == "MultiVAE"
        and type(train_data).__name__ == "UserDataLoader"
    ):
        return _validate_userwise_autoencoder_contract(
            runtime,
            stage=stage,
            require_backward=require_backward,
        )
    interaction = next(iter(train_data)).to(config["device"])
    required_fields = (model.USER_ID, model.ITEM_ID, model.NEG_ITEM_ID)
    available_fields = frozenset(interaction.interaction)
    missing = tuple(field for field in required_fields if field not in available_fields)
    if missing:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "INTERACTION_FIELDS_MISSING",
            "pairwise interaction is missing required model input fields",
        )
    for method_name in ("calculate_loss", "predict", "full_sort_predict"):
        if not callable(getattr(model, method_name, None)):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.INTERFACE,
                "REQUIRED_METHOD_MISSING",
                f"candidate lacks required method {method_name}",
            )
    solver_precompute_probe = _validate_precomputed_solver_runtime(
        runtime,
        interaction,
        stage=stage,
    )

    if runtime.get("p4_sparse_spectral"):
        # FaGSP intentionally depends on train interactions; it has no BL
        # compiler relation components. Its frozen-parent/residual contract
        # owns the causal check while shared API, gradient and smoke checks run.
        from recclaw_core.research_line.p4_implementation import p4_unit_check
        validate_p4_sparse_spectral_model_contract(
            model, config, runtime["train_dataset"], runtime["research_spec"]
        )
        p4_unit_check(model, config, runtime["train_dataset"])
        from recclaw_core.research_line.p4_runtime import p4_fit_mode
        if p4_fit_mode(config.final_config_dict) == "TRAIN_ONLY_PRECOMPUTE":
            require_backward = False
        relation_causality = {"relation_causality_contract": "P4_FROZEN_PARENT_AND_LIVE_RESIDUAL"}
    else:
        relation_causality = _validate_declared_relation_causality(
            runtime,
            interaction,
            stage=stage,
        )

    primitive_ids = tuple(
        getattr(model.__class__, "__recclaw_implemented_primitive_ids__", ())
    )
    sampler_primitives = tuple(
        item for item in primitive_ids if item.startswith("sampler.")
    )
    nonuniform_samplers = tuple(
        item for item in sampler_primitives if item != "sampler.uniform"
    )
    sampler_calls = 0
    sampler_epoch_refresh_calls = 0
    sampler_hardness_probe = "NOT_APPLICABLE"
    original_sampler_step = getattr(model, "recclaw_sampler_step", None)
    if nonuniform_samplers:
        if not callable(original_sampler_step):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "SAMPLER_PRIMITIVE_NOT_EXECUTABLE",
                "a non-uniform sampler requires recclaw_sampler_step",
                _candidate_method_repair_details("recclaw_sampler_step"),
            )

    requires_epoch_cache = _requires_epoch_sampler_refresh(model.__class__)
    declares_curriculum_epoch_signal = _declares_epoch_curriculum_sampler(
        model.__class__
    )
    sampler_refresh = getattr(model, "recclaw_sampler_refresh", None)
    if _declares_model_curriculum_sampler(model.__class__):
        if not callable(original_sampler_step):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "SAMPLER_PRIMITIVE_NOT_EXECUTABLE",
                "a DYNAMIC/CURRICULUM sampler requires recclaw_sampler_step",
                _candidate_method_repair_details("recclaw_sampler_step"),
            )
        if requires_epoch_cache and not callable(sampler_refresh):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "SAMPLER_EPOCH_REFRESH_MISSING",
                "an EPOCH sampler requires recclaw_sampler_refresh and recclaw_sampler_step",
                _candidate_method_repair_details(
                    "recclaw_sampler_refresh",
                    "recclaw_sampler_step",
                ),
            )
        sampler_hardness_probe = _validate_curriculum_sampler_causality(
            model,
            interaction,
            refresh_before_step=(
                requires_epoch_cache
                or (
                    declares_curriculum_epoch_signal
                    and callable(sampler_refresh)
                )
            ),
            stage=stage,
        )

    supplied_negative = interaction[model.NEG_ITEM_ID]
    sampler_outputs: tuple[Any, ...] = ()
    if sampler_primitives and callable(original_sampler_step):
        if requires_epoch_cache:
            (
                first_sampler_output,
                sampler_epoch_refresh_calls,
            ) = (
                _validate_epoch_sampler_runtime(
                    model,
                    interaction,
                    stage=stage,
                )
            )
            sampler_outputs = (
                first_sampler_output,
                *(original_sampler_step(interaction) for _ in range(7)),
            )
        else:
            sampler_outputs = tuple(original_sampler_step(interaction) for _ in range(8))
        known_positive_items: dict[int, set[int]] = {}
        # Validate against training positives only. Using the unsplit dataset here
        # would treat future dev/test interactions as known at training time.
        dataset = getattr(train_data, "dataset", None)
        inter_feat = getattr(dataset, "inter_feat", None)
        try:
            if inter_feat is not None:
                history_users = inter_feat[model.USER_ID]
                history_items = inter_feat[model.ITEM_ID]
                if isinstance(history_users, torch.Tensor):
                    history_users = history_users.detach().cpu().tolist()
                if isinstance(history_items, torch.Tensor):
                    history_items = history_items.detach().cpu().tolist()
                for history_user, history_item in zip(history_users, history_items):
                    known_positive_items.setdefault(int(history_user), set()).add(
                        int(history_item)
                    )
        except (KeyError, TypeError, AttributeError):
            known_positive_items = {}

        users = interaction[model.USER_ID].detach().cpu().reshape(-1).tolist()
        positives = interaction[model.ITEM_ID].detach().cpu().reshape(-1).tolist()
        floating_dtypes = {
            torch.bool,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
        }
        for sampler_output in sampler_outputs:
            if not isinstance(sampler_output, torch.Tensor):
                if "sampler.uniform" in sampler_primitives:
                    raise _stage_failure(
                        stage,
                        QualificationFailureClassV1.IMPLEMENTATION,
                        "SAMPLER_OUTPUT_INVALID",
                        "uniform sampler must return item-id tensors",
                        _candidate_method_repair_details("recclaw_sampler_step"),
                    )
                continue
            if not torch.isfinite(sampler_output).all().item():
                raise _stage_failure(
                    stage,
                    QualificationFailureClassV1.IMPLEMENTATION,
                    "SAMPLER_OUTPUT_INVALID",
                    "sampler output contains non-finite values",
                    _candidate_method_repair_details("recclaw_sampler_step"),
                )
            if sampler_output.dtype in floating_dtypes:
                if "sampler.uniform" in sampler_primitives:
                    raise _stage_failure(
                        stage,
                        QualificationFailureClassV1.IMPLEMENTATION,
                        "SAMPLER_OUTPUT_INVALID",
                        "uniform sampler must return integer item ids",
                        _candidate_method_repair_details("recclaw_sampler_step"),
                    )
                continue
            if sampler_output.ndim < 1 or sampler_output.shape[0] != len(users):
                raise _stage_failure(
                    stage,
                    QualificationFailureClassV1.IMPLEMENTATION,
                    "SAMPLER_OUTPUT_INVALID",
                    "sampler item-id output does not preserve the interaction batch",
                    _candidate_method_repair_details("recclaw_sampler_step"),
                )
            sampled_ids = sampler_output.detach().cpu().reshape(len(users), -1)
            if (
                (sampled_ids < 1).any().item()
                or (sampled_ids >= int(model.n_items)).any().item()
            ):
                raise _stage_failure(
                    stage,
                    QualificationFailureClassV1.IMPLEMENTATION,
                    "SAMPLER_OUTPUT_INVALID",
                    "sampler emitted padding or an out-of-range item id",
                    _candidate_method_repair_details("recclaw_sampler_step"),
                )
            for row, (user_id, positive_id) in enumerate(zip(users, positives)):
                row_ids = tuple(int(item) for item in sampled_ids[row].tolist())
                if int(positive_id) in row_ids or any(
                    item in known_positive_items.get(int(user_id), set())
                    for item in row_ids
                ):
                    raise _stage_failure(
                        stage,
                        QualificationFailureClassV1.IMPLEMENTATION,
                        "SAMPLER_OUTPUT_INVALID",
                        "sampler emitted a current or previously observed positive item",
                        _candidate_method_repair_details("recclaw_sampler_step"),
                    )

    if nonuniform_samplers and callable(original_sampler_step):
        first_sampler_output = sampler_outputs[0]
        rotated_negative = supplied_negative.remainder(int(model.n_items) - 1) + 1
        probe_interaction = Interaction(
            {
                key: (
                    rotated_negative
                    if key == model.NEG_ITEM_ID
                    else value.clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in interaction.interaction.items()
            }
        )
        second_sampler_output = original_sampler_step(probe_interaction)
        if (
            isinstance(first_sampler_output, torch.Tensor)
            and isinstance(second_sampler_output, torch.Tensor)
            and first_sampler_output.shape == supplied_negative.shape
            and second_sampler_output.shape == rotated_negative.shape
            and torch.equal(first_sampler_output.detach(), supplied_negative)
            and torch.equal(second_sampler_output.detach(), rotated_negative)
        ):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "SAMPLER_PASSTHROUGH_INVALID",
                "non-uniform sampler only returns RecBole's supplied NEG_ITEM_ID",
                _candidate_method_repair_details("recclaw_sampler_step"),
            )

    if sampler_primitives and callable(original_sampler_step):
        def counted_sampler_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal sampler_calls
            sampler_calls += 1
            return original_sampler_step(*args, **kwargs)

        setattr(model, "recclaw_sampler_step", counted_sampler_step)

    model.train()
    try:
        loss = model.calculate_loss(interaction)
    finally:
        if original_sampler_step is not None:
            setattr(model, "recclaw_sampler_step", original_sampler_step)
    if nonuniform_samplers and callable(original_sampler_step) and sampler_calls < 1:
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.IMPLEMENTATION,
            "SAMPLER_PRIMITIVE_NOT_EXECUTED",
            "calculate_loss did not execute its declared sampler hook",
            _candidate_method_repair_details(
                "calculate_loss",
                "recclaw_sampler_step",
            ),
        )
    losses = loss if isinstance(loss, tuple) else (loss,)
    if not losses or any(
        not isinstance(item, torch.Tensor)
        or item.numel() != 1
        or not torch.isfinite(item).all().item()
        for item in losses
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "LOSS_CONTRACT_FAILED",
            "calculate_loss must return finite scalar tensor values",
            _candidate_method_repair_details("calculate_loss"),
        )

    batch_size = int(interaction[model.USER_ID].shape[0])
    loss_batch_scaling_ratio: float | None = None
    entity_keyed_contrast_probe = "NOT_APPLICABLE"
    if batch_size > 1:
        duplicated = Interaction(
            {
                key: (
                    torch.cat((value, value), dim=0)
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and int(value.shape[0]) == batch_size
                    else value.clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in interaction.interaction.items()
            }
        )
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        def probe_total(value: Any) -> torch.Tensor:
            parts = value if isinstance(value, tuple) else (value,)
            if not parts or any(
                not isinstance(item, torch.Tensor)
                or item.numel() != 1
                or not torch.isfinite(item).all().item()
                for item in parts
            ):
                raise _stage_failure(
                    stage,
                    QualificationFailureClassV1.IMPLEMENTATION,
                    "LOSS_BATCH_REDUCTION_INVALID",
                    "loss reduction probe did not return finite scalars",
                    _candidate_method_repair_details("calculate_loss"),
                )
            return torch.stack([item.reshape(()) for item in parts]).sum()

        fixed_sampler = None
        if sampler_primitives and callable(original_sampler_step):
            def fixed_sampler(
                probe_interaction: Any,
                *_args: Any,
                **_kwargs: Any,
            ) -> Any:
                observed_size = int(probe_interaction[model.USER_ID].shape[0])
                if observed_size == batch_size:
                    return supplied_negative.clone()
                if observed_size == 2 * batch_size:
                    return torch.cat((supplied_negative, supplied_negative), dim=0)
                return original_sampler_step(probe_interaction)

            setattr(model, "recclaw_sampler_step", fixed_sampler)
        was_training = bool(model.training)
        model.eval()
        try:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            single_total = probe_total(model.calculate_loss(interaction))
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            duplicated_total = probe_total(model.calculate_loss(duplicated))
        finally:
            if fixed_sampler is not None:
                setattr(model, "recclaw_sampler_step", original_sampler_step)
            model.train(was_training)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        single_value = abs(float(single_total.detach().cpu().item()))
        duplicated_value = abs(float(duplicated_total.detach().cpu().item()))
        if single_value > 1.0e-8:
            loss_batch_scaling_ratio = duplicated_value / single_value
            if 1.8 <= loss_batch_scaling_ratio <= 2.2:
                raise _stage_failure(
                    stage,
                    QualificationFailureClassV1.IMPLEMENTATION,
                    "LOSS_BATCH_REDUCTION_INVALID",
                    "loss scales linearly with duplicated batch size; use a "
                    "batch-normalized training objective",
                    _candidate_method_repair_details("calculate_loss"),
                )
        if "ssl.objective.cross_layer_info_nce" in primitive_ids:
            entity_keyed_contrast_probe = (
                _validate_entity_keyed_contrast_duplicate_invariance(
                    single_total,
                    duplicated_total,
                    stage=stage,
                )
            )

    backward_parameter_count = 0
    if require_backward:
        model.zero_grad(set_to_none=True)
        total_loss = torch.stack([item.reshape(()) for item in losses]).sum()
        if not total_loss.requires_grad:
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "calculate_loss did not produce a differentiable scalar",
                _candidate_method_repair_details("calculate_loss"),
            )
        try:
            total_loss.backward()
        except Exception as error:  # noqa: BLE001 - typed below.
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "calculate_loss could not backpropagate",
                _candidate_method_repair_details("calculate_loss"),
            ) from error
        gradients = tuple(
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if not gradients or any(
            not torch.isfinite(gradient).all().item() for gradient in gradients
        ):
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "calculate_loss produced no finite parameter gradients",
                _candidate_method_repair_details("calculate_loss"),
            )
        backward_parameter_count = len(gradients)

    model.eval()
    with torch.no_grad():
        prediction = model.predict(interaction)
        user_batch = interaction[model.USER_ID][: min(2, batch_size)]
        full_sort = model.full_sort_predict(
            Interaction({model.USER_ID: user_batch})
        )
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.numel() != batch_size
        or not torch.isfinite(prediction).all().item()
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "PREDICT_CONTRACT_FAILED",
            "predict output does not match the interaction batch",
            _candidate_method_repair_details("predict"),
        )
    expected_full_sort = int(user_batch.shape[0]) * int(model.n_items)
    if (
        not isinstance(full_sort, torch.Tensor)
        or full_sort.numel() != expected_full_sort
        or not torch.isfinite(full_sort).all().item()
    ):
        raise _stage_failure(
            stage,
            QualificationFailureClassV1.INTERFACE,
            "FULL_SORT_CONTRACT_FAILED",
            "full_sort_predict output does not cover every item",
            _candidate_method_repair_details("full_sort_predict"),
        )
    return {
        "calculate_loss_scalars": len(losses),
        "solver_precompute_probe": solver_precompute_probe,
        "backward_parameter_count": backward_parameter_count,
        "input_fields": tuple(sorted(required_fields)),
        "input_type": model.input_type.name.lower(),
        "model_type": model.type.name.lower(),
        "predict_values": prediction.numel(),
        "full_sort_values": full_sort.numel(),
        "sampler_hook_calls": sampler_calls,
        "sampler_epoch_refresh_calls": sampler_epoch_refresh_calls,
        "sampler_hardness_probe": sampler_hardness_probe,
        "sampler_primitives": sampler_primitives,
        "loss_batch_size": batch_size,
        "loss_batch_scaling_ratio": loss_batch_scaling_ratio,
        "entity_keyed_contrast_probe": entity_keyed_contrast_probe,
        **relation_causality,
    }


def _validate_general_recommender_unit(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the family-neutral GeneralRecommender behavioral contract."""

    observation = _validate_api_contract(
        runtime,
        stage=QualificationStageV1.UNIT,
        require_backward=True,
    )
    model = runtime["model"]
    condition_spec = getattr(model, "__recclaw_diffusion_condition_component__", None)
    if isinstance(condition_spec, Mapping) and condition_spec.get("primitive_id") == (
        "condition.masked_spectral_history_classifier_free"
    ):
        from recclaw_core.search_spaces.diffusion_flow_cf_v1.mechanism_scaffold import (
            DiffusionFlowMechanismBindingError,
            validate_masked_spectral_condition,
        )

        interaction = next(iter(runtime["train_data"])).to(runtime["config"]["device"])
        history = model.get_rating_matrix(interaction[model.USER_ID][:8])
        try:
            validate_masked_spectral_condition(model, history)
        except DiffusionFlowMechanismBindingError as error:
            raise _stage_failure(
                QualificationStageV1.UNIT,
                QualificationFailureClassV1.IMPLEMENTATION,
                "DIFFUSION_CONDITION_SEMANTICS_MISSING",
                str(error),
                _candidate_method_repair_details("recclaw_diffusion_condition"),
            ) from error
        observation = {**observation, "masked_spectral_condition_behavior": "PASS"}
    return observation


class _QualificationBatchPrefix:
    """Present a small deterministic prefix of a real RecBole data loader."""

    def __init__(self, source: Any, limit: int):
        self._source = source
        self._limit = min(limit, len(source))
        self.batches_yielded = 0
        self.iterations_started = 0

    def __len__(self) -> int:
        return self._limit

    @property
    def __class__(self) -> type:
        # RecBole selects full-sort versus sampled evaluation with isinstance.
        # A bounded view must retain the source loader's evaluation semantics.
        return self._source.__class__

    def __iter__(self) -> Any:
        self.iterations_started += 1
        for batch in itertools.islice(self._source, self._limit):
            self.batches_yielded += 1
            yield batch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


def _one_epoch_smoke(
    runtime: Mapping[str, Any],
    fixture: RecBoleQualificationFixture,
) -> dict[str, Any]:
    from recbole.utils import get_trainer, init_seed

    candidate_class = runtime["candidate_class"]
    config = runtime["config"]
    train_data = runtime["train_data"]
    valid_data = runtime["valid_data"]
    final_config = getattr(config, "final_config_dict", {})
    qualification_batch_limit = final_config.get(
        "recclaw_qualification_train_batch_limit"
    )
    if qualification_batch_limit is not None:
        if (
            not isinstance(qualification_batch_limit, int)
            or isinstance(qualification_batch_limit, bool)
            or qualification_batch_limit <= 0
        ):
            raise _stage_failure(
                QualificationStageV1.ONE_EPOCH_SMOKE,
                QualificationFailureClassV1.INTERFACE,
                "QUALIFICATION_BATCH_LIMIT_INVALID",
                "recclaw_qualification_train_batch_limit must be a positive integer",
            )
        train_data = _QualificationBatchPrefix(
            train_data,
            qualification_batch_limit,
        )
    qualification_eval_batch_limit = final_config.get(
        "recclaw_qualification_eval_batch_limit"
    )
    if qualification_eval_batch_limit is not None:
        if (
            not isinstance(qualification_eval_batch_limit, int)
            or isinstance(qualification_eval_batch_limit, bool)
            or qualification_eval_batch_limit <= 0
        ):
            raise _stage_failure(
                QualificationStageV1.ONE_EPOCH_SMOKE,
                QualificationFailureClassV1.INTERFACE,
                "QUALIFICATION_EVAL_BATCH_LIMIT_INVALID",
                "recclaw_qualification_eval_batch_limit must be a positive integer",
            )
        valid_data = _QualificationBatchPrefix(
            valid_data,
            qualification_eval_batch_limit,
        )
    prepared_model = runtime.get("model")
    if (
        prepared_model is not None
        and getattr(prepared_model, POST_DEVICE_MECHANISM_INIT_FLAG, False) is True
    ):
        model = prepared_model
        model.zero_grad(set_to_none=True)
    else:
        init_seed(config["seed"], config["reproducibility"])
        model = candidate_class(config, train_data._dataset).to(config["device"])
        prepare_candidate_train_data(model, train_data)
    train_data_fit_roles = tuple(
        getattr(candidate_class, "_recclaw_machine_owned_train_data_fit_roles_v1", ())
    )
    train_data_fit_hook_calls = int(
        getattr(model, "_recclaw_machine_owned_train_data_fit_hook_calls", 0)
    )
    if train_data_fit_roles and train_data_fit_hook_calls != 1:
        raise _stage_failure(
            QualificationStageV1.ONE_EPOCH_SMOKE,
            QualificationFailureClassV1.IMPLEMENTATION,
            "TRAIN_DATA_FIT_CALL_COUNT_INVALID",
            "one train-derived fit must complete before trainer construction",
            _candidate_method_repair_details("recclaw_fit_operator"),
        )
    primitive_ids = tuple(
        getattr(candidate_class, "__recclaw_implemented_primitive_ids__", ())
    )
    training_primitives = tuple(
        item for item in primitive_ids if item.startswith("training.")
    )
    plain_optimizers = {
        "training.adam": "Adam",
        "training.sgd": "SGD",
        "training.adamw": "AdamW",
    }
    procedural_training = tuple(
        item for item in training_primitives if item not in plain_optimizers
    )
    trainer_entrypoint = config["recclaw_trainer_entrypoint"]
    trainer_repair_details: Mapping[str, Any] = {}
    if trainer_entrypoint:
        trainer_entrypoint = str(trainer_entrypoint)
        trainer_repair_details = _trainer_repair_details(trainer_entrypoint)
        module_name, class_name = trainer_entrypoint.split(":", 1)
        previous_dont_write = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            try:
                with _candidate_package_import_root(runtime["candidate_root"]):
                    trainer_module = importlib.import_module(module_name)
                    trainer_class = getattr(trainer_module, class_name, None)
            except Exception as error:
                raise _stage_failure(
                    QualificationStageV1.ONE_EPOCH_SMOKE,
                    QualificationFailureClassV1.INTERFACE,
                    "TRAINER_ENTRYPOINT_IMPORT_FAILED",
                    "candidate-local trainer entrypoint could not be imported",
                    details={
                        **trainer_repair_details,
                        "exception_type": type(error).__name__,
                    },
                ) from error
        finally:
            sys.dont_write_bytecode = previous_dont_write
        if not isinstance(trainer_class, type):
            raise _stage_failure(
                QualificationStageV1.ONE_EPOCH_SMOKE,
                QualificationFailureClassV1.INTERFACE,
                "TRAINER_ENTRYPOINT_INVALID",
                "candidate-local trainer entrypoint is not a class",
                details=trainer_repair_details,
            )
        if training_primitives:
            if "_build_optimizer" not in trainer_class.__dict__:
                raise _stage_failure(
                    QualificationStageV1.ONE_EPOCH_SMOKE,
                    QualificationFailureClassV1.IMPLEMENTATION,
                    "TRAINING_PRIMITIVE_NOT_EXPLICIT",
                    "FreshCandidateTrainer must directly implement _build_optimizer",
                    details=trainer_repair_details,
                )
            optimizer_signature = inspect.signature(
                trainer_class.__dict__["_build_optimizer"]
            )
            optimizer_parameters = tuple(optimizer_signature.parameters.values())
            if (
                not optimizer_parameters
                or optimizer_parameters[0].name != "self"
                or any(
                    parameter.kind
                    is inspect.Parameter.POSITIONAL_ONLY
                    or (
                        parameter.kind
                        is inspect.Parameter.POSITIONAL_OR_KEYWORD
                        and parameter.default is inspect.Parameter.empty
                    )
                    for parameter in optimizer_parameters[1:]
                )
                or not any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in optimizer_parameters[1:]
                )
            ):
                raise _stage_failure(
                    QualificationStageV1.ONE_EPOCH_SMOKE,
                    QualificationFailureClassV1.INTERFACE,
                    "TRAINER_OPTIMIZER_ABI_INVALID",
                    "_build_optimizer must be callable as _build_optimizer() and accept **kwargs",
                    details=trainer_repair_details,
                )
        _validate_direct_train_epoch_abi(
            trainer_class,
            repair_details=trainer_repair_details,
        )
        if procedural_training and (
            "recclaw_training_step" not in trainer_class.__dict__
            or not ({"_train_epoch", "fit"} & set(trainer_class.__dict__))
        ):
            raise _stage_failure(
                QualificationStageV1.ONE_EPOCH_SMOKE,
                QualificationFailureClassV1.IMPLEMENTATION,
                "TRAINING_PROCEDURE_NOT_EXECUTABLE",
                "non-default training requires recclaw_training_step and a direct training-loop override",
                details=trainer_repair_details,
            )
    else:
        trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    private_working_directory = fixture.checkpoint_dir.parent
    private_working_directory.mkdir(parents=True, exist_ok=True)
    previous_working_directory = Path.cwd()
    trainer = None
    training_hook_calls = 0
    sampler_epoch_refresh_calls = 0
    original_training_step = None
    original_sampler_refresh = None
    started_ns = time.monotonic_ns()
    try:
        os.chdir(private_working_directory)
        trainer = trainer_class(config, model)
        if _requires_epoch_sampler_refresh(candidate_class):
            original_sampler_refresh = getattr(model, "recclaw_sampler_refresh", None)
            if not callable(original_sampler_refresh):
                raise _stage_failure(
                    QualificationStageV1.ONE_EPOCH_SMOKE,
                    QualificationFailureClassV1.IMPLEMENTATION,
                    "SAMPLER_EPOCH_REFRESH_MISSING",
                    "an EPOCH sampler requires recclaw_sampler_refresh",
                    _candidate_method_repair_details("recclaw_sampler_refresh"),
                )

            def counted_sampler_refresh(*args: Any, **kwargs: Any) -> Any:
                nonlocal sampler_epoch_refresh_calls
                sampler_epoch_refresh_calls += 1
                return original_sampler_refresh(*args, **kwargs)

            setattr(model, "recclaw_sampler_refresh", counted_sampler_refresh)
        expected_optimizer = next(
            (
                expected
                for primitive, expected in plain_optimizers.items()
                if primitive in training_primitives
            ),
            None,
        )
        optimizer_name = type(trainer.optimizer).__name__
        if expected_optimizer and optimizer_name != expected_optimizer:
            raise _stage_failure(
                QualificationStageV1.ONE_EPOCH_SMOKE,
                QualificationFailureClassV1.IMPLEMENTATION,
                "TRAINING_OPTIMIZER_MISMATCH",
                f"declared optimizer requires {expected_optimizer}, observed {optimizer_name}",
                details=trainer_repair_details or None,
            )
        if procedural_training:
            original_training_step = getattr(trainer, "recclaw_training_step")

            def counted_training_step(*args: Any, **kwargs: Any) -> Any:
                nonlocal training_hook_calls
                training_hook_calls += 1
                return original_training_step(*args, **kwargs)

            setattr(trainer, "recclaw_training_step", counted_training_step)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=False,
            show_progress=False,
        )
    finally:
        if trainer is not None:
            if original_training_step is not None:
                setattr(trainer, "recclaw_training_step", original_training_step)
            if original_sampler_refresh is not None:
                setattr(model, "recclaw_sampler_refresh", original_sampler_refresh)
            trainer.tensorboard.close()
        os.chdir(previous_working_directory)
    if not isinstance(best_valid_score, (int, float)) or not math.isfinite(
        float(best_valid_score)
    ):
        raise _stage_failure(
            QualificationStageV1.ONE_EPOCH_SMOKE,
            QualificationFailureClassV1.RUNTIME,
            "SMOKE_RESULT_INVALID",
            "one-epoch smoke did not return a finite validation score",
        )
    if not isinstance(best_valid_result, Mapping):
        raise _stage_failure(
            QualificationStageV1.ONE_EPOCH_SMOKE,
            QualificationFailureClassV1.RUNTIME,
            "SMOKE_RESULT_INVALID",
            "one-epoch smoke did not return a RecBole result mapping",
        )
    if (
        qualification_eval_batch_limit is not None
        and (
            valid_data.iterations_started != 1
            or valid_data.batches_yielded != len(valid_data)
        )
    ):
        raise _stage_failure(
            QualificationStageV1.ONE_EPOCH_SMOKE,
            QualificationFailureClassV1.RUNTIME,
            "SMOKE_EVAL_BATCH_COUNT_INVALID",
            "bounded one-epoch smoke did not score every presented dev batch",
        )
    if procedural_training and training_hook_calls < 1:
        raise _stage_failure(
            QualificationStageV1.ONE_EPOCH_SMOKE,
            QualificationFailureClassV1.IMPLEMENTATION,
            "TRAINING_PROCEDURE_NOT_EXECUTED",
            "one-epoch smoke did not execute recclaw_training_step",
            details=trainer_repair_details or None,
        )
    if _requires_epoch_sampler_refresh(candidate_class) and sampler_epoch_refresh_calls != 1:
        raise _stage_failure(
            QualificationStageV1.ONE_EPOCH_SMOKE,
            QualificationFailureClassV1.IMPLEMENTATION,
            "SAMPLER_EPOCH_REFRESH_COUNT_INVALID",
            "one training epoch must execute recclaw_sampler_refresh exactly once",
            details=trainer_repair_details or None,
        )
    p4_precompute = (runtime.get("p4_sparse_spectral")
                     and getattr(trainer, "p4_fit_mode", None) == "TRAIN_ONLY_PRECOMPUTE")
    observation = {
        "completed_epochs": 0 if p4_precompute else 1,
        "train_batches_executed": 0 if p4_precompute else len(train_data),
        **({"p4_fit_mode": "TRAIN_ONLY_PRECOMPUTE", "operator_fit_completed": True}
           if p4_precompute else {}),
        "train_batch_limit": qualification_batch_limit,
        "metric_values_excluded_from_qualification": True,
        "trainer_class": trainer_class.__name__,
        "optimizer_class": optimizer_name,
        "training_hook_calls": training_hook_calls,
        "sampler_epoch_refresh_calls": sampler_epoch_refresh_calls,
        "train_data_fit_hook_calls": train_data_fit_hook_calls,
        "train_data_fit_roles": train_data_fit_roles,
        "training_primitives": training_primitives,
        "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
    }
    if qualification_eval_batch_limit is not None:
        observation.update(
            {
                "eval_batches_executed": valid_data.batches_yielded,
                "eval_batch_limit": qualification_eval_batch_limit,
                "eval_passes_executed": valid_data.iterations_started,
            }
        )
    return observation


def _generic_failure(
    stage: QualificationStageV1,
    error: Exception,
) -> _QualificationStageFailure:
    default_failure_class = {
        QualificationStageV1.STATIC_VALIDATION: (
            QualificationFailureClassV1.IMPLEMENTATION
        ),
        QualificationStageV1.CONSTRUCTION: QualificationFailureClassV1.INTERFACE,
        QualificationStageV1.API_CONTRACT: QualificationFailureClassV1.INTERFACE,
        QualificationStageV1.UNIT: QualificationFailureClassV1.IMPLEMENTATION,
        QualificationStageV1.ONE_EPOCH_SMOKE: QualificationFailureClassV1.RUNTIME,
    }[stage]
    if isinstance(error, TrainSpectralBasisContractError):
        default_failure_class = QualificationFailureClassV1.IMPLEMENTATION
    message = str(error) or type(error).__name__
    normalized = normalize_qualification_failure(
        {
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "failure_class": default_failure_class.value,
            "message": message,
            "reason_code": getattr(
                error,
                "reason_code",
                type(error).__name__.upper(),
            ),
            "stage": stage.value,
        }
    )
    extracted_frames = traceback.extract_tb(error.__traceback__)
    candidate_frames = tuple(
        frame.name
        for frame in extracted_frames
        if frame.filename.replace("\\", "/").endswith(
            "/recclaw_ext/candidate.py"
        )
        and frame.name.isidentifier()
    )
    explicit_methods = (
        tuple(error.implicated_methods)
        if isinstance(
            error,
            (CandidateCardinalityContractError, TrainSpectralBasisContractError),
        )
        else ()
    )
    implicated_methods = tuple(
        dict.fromkeys((*explicit_methods, *candidate_frames[-1:]))
    )
    traceback_files = tuple(
        dict.fromkeys(
            "recclaw_ext/" + frame.filename.replace("\\", "/").rsplit(
                "/recclaw_ext/", 1
            )[1]
            for frame in extracted_frames
            if "/recclaw_ext/" in frame.filename.replace("\\", "/")
        )
    )
    explicit_files = (
        tuple(error.implicated_files)
        if isinstance(
            error,
            (CandidateCardinalityContractError, TrainSpectralBasisContractError),
        )
        else ()
    )
    implicated_files = tuple(dict.fromkeys((*explicit_files, *traceback_files)))
    return _stage_failure(
        stage,
        QualificationFailureClassV1(str(normalized["failure_class"])),
        str(normalized["reason_code"]),
        message,
        details=(
            {
                **(
                    {"implicated_methods": implicated_methods}
                    if implicated_methods
                    else {}
                ),
                **(
                    {"implicated_files": implicated_files}
                    if implicated_files
                    else {}
                ),
            }
            if implicated_methods or implicated_files
            else None
        ),
    )


def _failure_detail(
    failure: _QualificationStageFailure,
) -> dict[str, Any]:
    message = str(failure)
    return canonical_value(
        {
            "error_type": type(failure).__name__,
            "failure_class": failure.failure_class.value,
            "message": message[:2000],
            "reason_code": failure.reason_code,
            "stage": failure.stage.value,
            "traceback": (
                f"{failure.failure_class.value}/"
                f"{failure.stage.value}/{failure.reason_code}: {message[:1000]}"
            ),
            **(dict(failure.details) if failure.details is not None else {}),
        }
    )


def _receipt(
    package: CandidatePackageV1,
    *,
    terminal_stage: QualificationStageV1,
    failure: _QualificationStageFailure | None,
) -> tuple[QualificationReceiptV1, Mapping[str, Any] | None]:
    ordered_stages = tuple(QualificationStageV1)
    terminal_index = ordered_stages.index(terminal_stage)
    statuses = {
        stage: (
            QualificationCheckStatusV1.PASS
            if index < terminal_index or failure is None
            else (
                QualificationCheckStatusV1.FAIL
                if index == terminal_index
                else QualificationCheckStatusV1.NOT_RUN
            )
        )
        for index, stage in enumerate(ordered_stages)
    }
    detail = _failure_detail(failure) if failure is not None else None
    return (
        QualificationReceiptV1(
            candidate_package_ref=package.package_id,
            candidate_package_digest=package.digest,
            research_spec_ref=package.research_spec_ref,
            research_spec_digest=package.research_spec_digest,
            candidate_root_ref=package.candidate_root_ref,
            candidate_root_digest=package.candidate_root_digest,
            source_tree_digest=package.source_tree_digest,
            runtime_identity_ref=package.runtime_identity_ref,
            runtime_identity_digest=package.runtime_identity_digest,
            protocol_ref=package.protocol_ref,
            protocol_digest=package.protocol_digest,
            stage=terminal_stage,
            status=(
                QualificationStatusV1.FAIL
                if failure is not None
                else QualificationStatusV1.PASS
            ),
            failure_class=(
                failure.failure_class
                if failure is not None
                else QualificationFailureClassV1.NONE
            ),
            static_result=statuses[QualificationStageV1.STATIC_VALIDATION],
            construction_result=statuses[QualificationStageV1.CONSTRUCTION],
            api_contract_result=statuses[QualificationStageV1.API_CONTRACT],
            unit_result=statuses[QualificationStageV1.UNIT],
            smoke_result=statuses[QualificationStageV1.ONE_EPOCH_SMOKE],
            failure_detail_ref=(
                "qualification-failure:"
                + failure.stage.value.lower()
                + ":"
                + failure.reason_code.lower()
                if failure is not None
                else None
            ),
            failure_detail_digest=(
                sha256_digest(detail) if detail is not None else None
            ),
        ),
        detail,
    )


def _disposable_process_failure(
    package: CandidatePackageV1,
    *,
    stage: QualificationStageV1,
    reason_code: str,
    message: str,
    process_observation: Mapping[str, Any],
) -> MechanicalQualificationRun:
    failure = _stage_failure(
        stage,
        QualificationFailureClassV1.RESOURCE,
        reason_code,
        message,
    )
    receipt, detail = _receipt(
        package,
        terminal_stage=stage,
        failure=failure,
    )
    return MechanicalQualificationRun(
        receipt=receipt,
        failure_detail=detail,
        smoke_executions=0,
        stage_observations=canonical_value(
            {"DISPOSABLE_PROCESS": dict(process_observation)}
        ),
    )


def _qualification_process_worker(
    connection: Any,
    package: CandidatePackageV1,
    research_spec: OpenResearchSpecV1,
    candidate_root: Path,
    fixture: RecBoleQualificationFixture,
    unit_check: Callable[[Any, Any, Any], None] | None,
) -> None:
    """Run the existing qualifier behind a process boundary.

    The worker deliberately calls the legacy ``qualify`` implementation.  That
    keeps Q0R/Q1 callers byte- and behavior-compatible while ensuring any
    native/CUDA failure is contained to this short-lived process.
    """

    try:
        result = MechanicalRecBoleAdapterV1().qualify(
            package,
            research_spec=research_spec,
            candidate_root=candidate_root,
            fixture=fixture,
            unit_check=unit_check,
        )
    except BaseException as error:  # pragma: no cover - crash path is defensive.
        payload = {
            "kind": "WORKER_EXCEPTION",
            "error_type": type(error).__name__,
            "message": str(error)[:2000],
            "traceback": traceback.format_exc()[-4000:],
        }
        try:
            connection.send(payload)
        except Exception:
            pass
    else:
        try:
            connection.send({"kind": "RESULT", "result": result})
        except Exception:
            pass
    finally:
        connection.close()


class MechanicalRecBoleAdapterV1:
    """Run the five RC0 qualification stages on one candidate package."""

    def qualify(
        self,
        package: CandidatePackageV1,
        *,
        research_spec: OpenResearchSpecV1,
        candidate_root: Path,
        fixture: RecBoleQualificationFixture,
        unit_check: Callable[[Any, Any, Any], None] | None = None,
        allow_optional_unit_check: bool = False,
    ) -> MechanicalQualificationRun:
        if not isinstance(package, CandidatePackageV1):
            raise MechanicalRecBoleAdapterError(
                "package must be CandidatePackageV1"
            )
        if not isinstance(research_spec, OpenResearchSpecV1):
            raise MechanicalRecBoleAdapterError(
                "research_spec must be OpenResearchSpecV1"
            )
        if unit_check is not None and not callable(unit_check):
            raise MechanicalRecBoleAdapterError("unit_check must be callable")
        if not isinstance(allow_optional_unit_check, bool):
            raise MechanicalRecBoleAdapterError(
                "allow_optional_unit_check must be a boolean"
            )
        root = candidate_root.resolve()
        observations: dict[str, Any] = {}
        failure: _QualificationStageFailure | None = None
        stage = QualificationStageV1.STATIC_VALIDATION
        smoke_executions = 0
        initial_manifest: tuple[dict[str, Any], ...] | None = None
        try:
            binding_observation = _validate_package_bindings(
                package,
                research_spec,
                fixture,
                root,
            )
            initial_manifest = snapshot_candidate_tree(root)
            observations[stage.value] = {
                **binding_observation,
                **_validate_static_package(package, root),
            }
            _assert_candidate_tree_unchanged(
                root,
                expected_manifest=initial_manifest,
                stage=stage,
            )

            stage = QualificationStageV1.CONSTRUCTION
            runtime = _construct_runtime(package, research_spec, root, fixture)
            observations[stage.value] = {
                "config_model": str(runtime["config"]["model"]),
                "dataset_class": type(runtime["dataset"]).__name__,
                "model_class": type(runtime["model"]).__name__,
                "model_type": runtime["config"]["MODEL_TYPE"].name.lower(),
            }
            _assert_candidate_tree_unchanged(
                root,
                expected_manifest=initial_manifest,
                stage=stage,
            )

            stage = QualificationStageV1.API_CONTRACT
            observations[stage.value] = _validate_api_contract(runtime)
            _assert_candidate_tree_unchanged(
                root,
                expected_manifest=initial_manifest,
                stage=stage,
            )

            stage = QualificationStageV1.UNIT
            unit_observation = _validate_general_recommender_unit(runtime)
            optional_unit_check: dict[str, Any] = {"status": "NOT_PROVIDED"}
            if unit_check is not None:
                if allow_optional_unit_check:
                    try:
                        unit_check(
                            runtime["model"],
                            runtime["config"],
                            runtime["dataset"],
                        )
                    except Exception as error:  # noqa: BLE001 - optional evidence only.
                        optional_unit_check = {
                            "error_type": type(error).__name__,
                            "message": str(error)[:2000],
                            "status": "FAIL_OPTIONAL",
                        }
                    else:
                        optional_unit_check = {"status": "PASS"}
                else:
                    unit_check(
                        runtime["model"],
                        runtime["config"],
                        runtime["dataset"],
                    )
            observations[stage.value] = {
                **unit_observation,
                "optional_unit_check": optional_unit_check,
                "shared_unit_check": (
                    "PASS"
                    if optional_unit_check["status"] == "PASS"
                    else "NOT_PROVIDED"
                    if optional_unit_check["status"] == "NOT_PROVIDED"
                    else "FAIL_OPTIONAL"
                ),
            }
            _assert_candidate_tree_unchanged(
                root,
                expected_manifest=initial_manifest,
                stage=stage,
            )

            stage = QualificationStageV1.ONE_EPOCH_SMOKE
            smoke_executions = 1
            observations[stage.value] = _one_epoch_smoke(runtime, fixture)
            _assert_candidate_tree_unchanged(
                root,
                expected_manifest=initial_manifest,
                stage=stage,
            )
        except _QualificationStageFailure as error:
            failure = error
            stage = error.stage
        except Exception as error:  # noqa: BLE001 - converted to typed receipt.
            failure = _generic_failure(stage, error)
        if initial_manifest is not None:
            try:
                _assert_candidate_tree_unchanged(
                    root,
                    expected_manifest=initial_manifest,
                    stage=stage,
                )
            except _QualificationStageFailure as tree_failure:
                failure = tree_failure
                stage = tree_failure.stage
        receipt, detail = _receipt(
            package,
            terminal_stage=stage,
            failure=failure,
        )
        return MechanicalQualificationRun(
            receipt=receipt,
            stage_observations=canonical_value(observations),
            failure_detail=detail,
            smoke_executions=smoke_executions,
        )

    def qualify_disposable(
        self,
        package: CandidatePackageV1,
        *,
        research_spec: OpenResearchSpecV1,
        candidate_root: Path,
        fixture: RecBoleQualificationFixture,
        unit_check: Callable[[Any, Any, Any], None] | None = None,
        timeout_seconds: int = DISPOSABLE_QUALIFICATION_TIMEOUT_SECONDS,
    ) -> MechanicalQualificationRun:
        """Qualify one package in a disposable child process.

        ``qualify`` remains the compatibility path used by the existing Q0R/Q1
        consumers.  Research Innovation should call this method so a native
        failure cannot poison the parent campaign process.  The returned
        receipt has the same RC0 contract and is enriched only with a local
        process-isolation observation.
        """

        if not isinstance(package, CandidatePackageV1):
            raise MechanicalRecBoleAdapterError("package must be CandidatePackageV1")
        if not isinstance(research_spec, OpenResearchSpecV1):
            raise MechanicalRecBoleAdapterError(
                "research_spec must be OpenResearchSpecV1"
            )
        if unit_check is not None and not callable(unit_check):
            raise MechanicalRecBoleAdapterError("unit_check must be callable")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
            raise MechanicalRecBoleAdapterError("timeout_seconds must be an integer")
        if timeout_seconds <= 0:
            raise MechanicalRecBoleAdapterError("timeout_seconds must be positive")

        root = candidate_root.resolve()
        start_method = "spawn"
        try:
            context = mp.get_context(start_method)
        except ValueError as error:
            return _disposable_process_failure(
                package,
                stage=QualificationStageV1.CONSTRUCTION,
                reason_code="DISPOSABLE_SPAWN_UNAVAILABLE",
                message=str(error)[:2000],
                process_observation={
                    "process_isolated": False,
                    "start_method": start_method,
                    "status": "SPAWN_UNAVAILABLE",
                },
            )
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_qualification_process_worker,
            args=(
                child_connection,
                package,
                research_spec,
                root,
                fixture,
                unit_check,
            ),
        )
        started_ns = time.monotonic_ns()
        try:
            process.start()
        except Exception as error:
            child_connection.close()
            parent_connection.close()
            return _disposable_process_failure(
                package,
                stage=QualificationStageV1.CONSTRUCTION,
                reason_code="DISPOSABLE_PROCESS_START_FAILED",
                message=str(error)[:2000],
                process_observation={
                    "process_isolated": False,
                    "start_method": start_method,
                    "status": "START_FAILED",
                },
            )
        child_connection.close()

        payload: Mapping[str, Any] | None = None
        deadline = time.monotonic() + float(timeout_seconds)
        try:
            while time.monotonic() < deadline:
                remaining = max(0.01, min(0.25, deadline - time.monotonic()))
                if parent_connection.poll(remaining):
                    received = parent_connection.recv()
                    if isinstance(received, Mapping):
                        payload = received
                    break
                if not process.is_alive():
                    break
        except (EOFError, OSError):
            payload = None
        finally:
            parent_connection.close()

        timed_out = payload is None and process.is_alive()
        if timed_out:
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        elapsed_ms = max(1, (time.monotonic_ns() - started_ns) // 1_000_000)
        process_observation = {
            "elapsed_wall_time_ms": elapsed_ms,
            "exit_code": process.exitcode,
            "pid": process.pid,
            "process_isolated": True,
            "start_method": start_method,
            "status": (
                "TIMEOUT"
                if timed_out
                else "RESULT"
                if payload is not None and payload.get("kind") == "RESULT"
                else "WORKER_FAILURE"
            ),
        }
        if payload is not None and payload.get("kind") == "WORKER_EXCEPTION":
            process_observation.update(
                {
                    "worker_error_message": str(payload.get("message", ""))[:2000],
                    "worker_error_type": str(payload.get("error_type", "WorkerError")),
                }
            )
        if (
            payload is not None
            and payload.get("kind") == "RESULT"
            and isinstance(payload.get("result"), MechanicalQualificationRun)
        ):
            result = payload["result"]
            observations = dict(result.stage_observations)
            observations["DISPOSABLE_PROCESS"] = process_observation
            return replace(
                result,
                stage_observations=canonical_value(observations),
            )

        worker_error = (
            payload.get("message", "")
            if payload is not None
            else "disposable qualification process exited without a receipt"
        )
        return _disposable_process_failure(
            package,
            stage=QualificationStageV1.CONSTRUCTION,
            reason_code=(
                "DISPOSABLE_PROCESS_TIMEOUT"
                if timed_out
                else "DISPOSABLE_PROCESS_FAILED"
            ),
            message=str(worker_error)[:2000],
            process_observation=process_observation,
        )


__all__ = [
    "DISPOSABLE_QUALIFICATION_TIMEOUT_SECONDS",
    "MechanicalQualificationRun",
    "MechanicalRecBoleAdapterError",
    "MechanicalRecBoleAdapterV1",
    "RecBoleQualificationFixture",
    "candidate_tree_identity",
    "snapshot_candidate_tree",
]
