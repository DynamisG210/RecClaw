"""Mechanical RecBole qualification for an RC0 candidate package.

The adapter deliberately evaluates only implementation feasibility.  It does
not compare recommendation metrics, update mechanism beliefs, or emit a
scientific episode.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .canonical import canonical_value, sha256_digest, validate_sha256
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
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.failure_class = failure_class
        self.reason_code = reason_code


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
) -> _QualificationStageFailure:
    return _QualificationStageFailure(
        stage=stage,
        failure_class=failure_class,
        reason_code=reason_code,
        message=message,
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
    if execution_contract is not None and class_name != str(
        execution_contract["model"]
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
            except (ImportError, ModuleNotFoundError) as error:
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
    return candidate_class


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
    from recbole.utils import ModelType, init_seed

    execution_contract = research_spec.execution_contract
    base_model_config = fixture.base_model_config
    contract_config: Mapping[str, Any] = {}
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
        if fixture.base_model_config != base_model_config:
            fixture = replace(fixture, base_model_config=base_model_config)

    config_files = (
        fixture.recbole_root
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
    config = Config(
        model=base_model_config,
        dataset=fixture.dataset,
        config_file_list=[str(path) for path in config_files],
        config_dict={
            **contract_config,
            "benchmark_filename": ["train", "valid", "test"],
            "checkpoint_dir": str(fixture.checkpoint_dir),
            "data_path": str(fixture.data_path),
            "epochs": 1,
            "eval_batch_size": 32,
            "eval_step": 1,
            "reproducibility": True,
            "seed": fixture.seed,
            "show_progress": False,
            "state": "ERROR",
            "stopping_step": 1,
            "topk": [3],
            "train_batch_size": 8,
            "use_gpu": False,
            "valid_metric": "NDCG@3",
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = candidate_class(config, train_data._dataset).to(config["device"])
    if not isinstance(model, GeneralRecommender):
        raise _stage_failure(
            QualificationStageV1.CONSTRUCTION,
            QualificationFailureClassV1.INTERFACE,
            "NOT_GENERAL_RECOMMENDER",
            "entrypoint is not a RecBole GeneralRecommender",
        )
    if config["MODEL_TYPE"] is not ModelType.GENERAL:
        raise _stage_failure(
            QualificationStageV1.CONSTRUCTION,
            QualificationFailureClassV1.INTERFACE,
            "MODEL_TYPE_MISMATCH",
            "frozen RecBole Config did not resolve ModelType.GENERAL",
        )
    return {
        "candidate_class": candidate_class,
        "config": config,
        "dataset": dataset,
        "model": model,
        "test_data": test_data,
        "train_data": train_data,
        "valid_data": valid_data,
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
            )
        try:
            total_loss.backward()
        except Exception as error:  # noqa: BLE001 - typed below.
            raise _stage_failure(
                stage,
                QualificationFailureClassV1.IMPLEMENTATION,
                "LOSS_BACKPROP_FAILED",
                "calculate_loss could not backpropagate",
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
            )
        backward_parameter_count = len(gradients)

    model.eval()
    batch_size = int(interaction[model.USER_ID].shape[0])
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
        )
    return {
        "calculate_loss_scalars": len(losses),
        "backward_parameter_count": backward_parameter_count,
        "input_fields": tuple(sorted(required_fields)),
        "input_type": model.input_type.name.lower(),
        "model_type": model.type.name.lower(),
        "predict_values": prediction.numel(),
        "full_sort_values": full_sort.numel(),
    }


def _validate_general_recommender_unit(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the family-neutral GeneralRecommender behavioral contract."""

    return _validate_api_contract(
        runtime,
        stage=QualificationStageV1.UNIT,
        require_backward=True,
    )


def _one_epoch_smoke(
    runtime: Mapping[str, Any],
    fixture: RecBoleQualificationFixture,
) -> dict[str, Any]:
    from recbole.utils import get_trainer, init_seed

    candidate_class = runtime["candidate_class"]
    config = runtime["config"]
    train_data = runtime["train_data"]
    valid_data = runtime["valid_data"]
    init_seed(config["seed"], config["reproducibility"])
    model = candidate_class(config, train_data._dataset).to(config["device"])
    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    private_working_directory = fixture.checkpoint_dir.parent
    private_working_directory.mkdir(parents=True, exist_ok=True)
    previous_working_directory = Path.cwd()
    trainer = None
    started_ns = time.monotonic_ns()
    try:
        os.chdir(private_working_directory)
        trainer = trainer_class(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=False,
            show_progress=False,
        )
    finally:
        if trainer is not None:
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
    return {
        "completed_epochs": 1,
        "metric_values_excluded_from_qualification": True,
        "trainer_class": trainer_class.__name__,
        "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
    }


def _generic_failure(
    stage: QualificationStageV1,
    error: Exception,
) -> _QualificationStageFailure:
    failure_class = {
        QualificationStageV1.STATIC_VALIDATION: (
            QualificationFailureClassV1.IMPLEMENTATION
        ),
        QualificationStageV1.CONSTRUCTION: QualificationFailureClassV1.INTERFACE,
        QualificationStageV1.API_CONTRACT: QualificationFailureClassV1.INTERFACE,
        QualificationStageV1.UNIT: QualificationFailureClassV1.IMPLEMENTATION,
        QualificationStageV1.ONE_EPOCH_SMOKE: QualificationFailureClassV1.RUNTIME,
    }[stage]
    return _stage_failure(
        stage,
        failure_class,
        type(error).__name__.upper(),
        str(error) or type(error).__name__,
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
