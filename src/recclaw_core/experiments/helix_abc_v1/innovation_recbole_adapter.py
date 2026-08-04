"""Mechanical RecBole qualification for an RC0 candidate package.

The adapter deliberately evaluates only implementation feasibility.  It does
not compare recommendation metrics, update mechanism beliefs, or emit a
scientific episode.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
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
    sys.path.insert(0, str(candidate_root.resolve()))
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "recclaw_ext" or name.startswith("recclaw_ext."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path


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
    candidate_root: Path,
    fixture: RecBoleQualificationFixture,
) -> dict[str, Any]:
    _numpy_compatibility_aliases()
    candidate_class = _load_candidate_class(package, candidate_root)

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.abstract_recommender import GeneralRecommender
    from recbole.utils import ModelType, init_seed

    config_files = (
        fixture.recbole_root
        / "recbole"
        / "properties"
        / "model"
        / f"{fixture.base_model_config}.yaml",
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
        model=fixture.base_model_config,
        dataset=fixture.dataset,
        config_file_list=[str(path) for path in config_files],
        config_dict={
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


def _validate_api_contract(runtime: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    from recbole.data.interaction import Interaction
    from recbole.utils import InputType, ModelType

    model = runtime["model"]
    config = runtime["config"]
    train_data = runtime["train_data"]
    if model.type is not ModelType.GENERAL:
        raise _stage_failure(
            QualificationStageV1.API_CONTRACT,
            QualificationFailureClassV1.INTERFACE,
            "MODEL_TYPE_MISMATCH",
            "candidate model type is not ModelType.GENERAL",
        )
    if model.input_type is not InputType.PAIRWISE:
        raise _stage_failure(
            QualificationStageV1.API_CONTRACT,
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
            QualificationStageV1.API_CONTRACT,
            QualificationFailureClassV1.INTERFACE,
            "INTERACTION_FIELDS_MISSING",
            "pairwise interaction is missing required model input fields",
        )
    for method_name in ("calculate_loss", "predict", "full_sort_predict"):
        if not callable(getattr(model, method_name, None)):
            raise _stage_failure(
                QualificationStageV1.API_CONTRACT,
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
            QualificationStageV1.API_CONTRACT,
            QualificationFailureClassV1.INTERFACE,
            "LOSS_CONTRACT_FAILED",
            "calculate_loss must return finite scalar tensor values",
        )

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
            QualificationStageV1.API_CONTRACT,
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
            QualificationStageV1.API_CONTRACT,
            QualificationFailureClassV1.INTERFACE,
            "FULL_SORT_CONTRACT_FAILED",
            "full_sort_predict output does not cover every item",
        )
    return {
        "calculate_loss_scalars": len(losses),
        "input_fields": tuple(sorted(required_fields)),
        "input_type": model.input_type.name.lower(),
        "model_type": model.type.name.lower(),
        "predict_values": prediction.numel(),
        "full_sort_values": full_sort.numel(),
    }


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


class MechanicalRecBoleAdapterV1:
    """Run the five RC0 qualification stages on one candidate package."""

    def qualify(
        self,
        package: CandidatePackageV1,
        *,
        research_spec: OpenResearchSpecV1,
        candidate_root: Path,
        fixture: RecBoleQualificationFixture,
        unit_check: Callable[[Any, Any, Any], None],
    ) -> MechanicalQualificationRun:
        if not isinstance(package, CandidatePackageV1):
            raise MechanicalRecBoleAdapterError(
                "package must be CandidatePackageV1"
            )
        if not isinstance(research_spec, OpenResearchSpecV1):
            raise MechanicalRecBoleAdapterError(
                "research_spec must be OpenResearchSpecV1"
            )
        if not callable(unit_check):
            raise MechanicalRecBoleAdapterError("unit_check must be callable")
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
            runtime = _construct_runtime(package, root, fixture)
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
            unit_check(runtime["model"], runtime["config"], runtime["dataset"])
            observations[stage.value] = {"shared_unit_check": "PASS"}
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


__all__ = [
    "MechanicalQualificationRun",
    "MechanicalRecBoleAdapterError",
    "MechanicalRecBoleAdapterV1",
    "RecBoleQualificationFixture",
    "candidate_tree_identity",
    "snapshot_candidate_tree",
]
