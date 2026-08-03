#!/usr/bin/env python3
"""Run one package-owned RecBole Pilot training execution."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _numeric_loss(value: object) -> float | list[float] | None:
    """Convert RecBole's epoch loss return into a JSON-safe observation."""

    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "detach") and hasattr(value, "numel"):
        detached = value.detach()
        if int(detached.numel()) == 1:
            return float(detached.item())
    if isinstance(value, tuple):
        converted = [_numeric_loss(item) for item in value]
        if all(isinstance(item, float) for item in converted):
            return [float(item) for item in converted]
    return None


def _install_fixed_batch_iteration(
    source: object,
    batches: tuple[object, ...],
    source_indices: tuple[int, ...],
    *,
    on_batch_started: object,
    on_batch_completed: object,
) -> object:
    """Limit one loader in place without changing its concrete RecBole type."""

    loader_type = type(source)
    original_iter = loader_type.__iter__

    def fixed_iter(self: object) -> object:
        if self is not source:
            yield from original_iter(self)
            return
        for position, (source_index, batch) in enumerate(
            zip(source_indices, batches, strict=True)
        ):
            on_batch_started(position, source_index)
            started_ns = time.monotonic_ns()
            yield batch
            on_batch_completed(position, source_index, started_ns)

    loader_type.__iter__ = fixed_iter

    def restore() -> None:
        loader_type.__iter__ = original_iter

    return restore


def _preallocate_batches(
    source: object,
    indices: tuple[int, ...],
) -> tuple[object, ...]:
    selected: list[object] = []
    wanted = set(indices)
    maximum = max(indices)
    for source_index, batch in enumerate(source):
        if source_index in wanted:
            selected.append(batch)
        if source_index >= maximum:
            break
    if len(selected) != len(indices):
        raise RuntimeError("fixed-batch prefix index exceeds data loader")
    return tuple(selected)


def _install_resource_telemetry(
    trainer: object,
    *,
    torch: object,
    train_data: object,
    valid_data: object,
    telemetry_path: Path,
    prefix_contract: dict[str, object] | None,
    preallocated_train_batches: tuple[object, ...] | None,
    preallocated_valid_batches: tuple[object, ...] | None,
    worker_started_ns: int,
) -> tuple[dict[str, object], object]:
    """Measure RecBole phases and durably preserve fixed-batch progress."""

    phases: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    original_train_epoch = trainer._train_epoch
    original_valid_epoch = trainer._valid_epoch
    active: dict[str, object] = {
        "epoch": None,
        "phase": None,
        "status": "INITIALIZED",
    }
    telemetry: dict[str, object] = {
        "active_progress": active,
        "batch_records": batches,
        "full_train_batches_per_epoch": len(train_data),
        "full_validation_batches_per_eval": len(valid_data),
        "initialization_wall_time_ms": max(
            1, (time.monotonic_ns() - worker_started_ns) // 1_000_000
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in trainer.model.parameters()
        ),
        "phase_records": phases,
        "prefix_contract": prefix_contract,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        ),
    }

    def flush() -> None:
        _write_durable_json(telemetry_path, _finalize_resource_telemetry(telemetry))

    def memory_begin() -> int | None:
        if not torch.cuda.is_available():
            return None
        device = int(torch.cuda.current_device())
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        return device

    def memory_finish(device: int | None) -> dict[str, object]:
        if device is None:
            return {
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
            }
        torch.cuda.synchronize(device)
        divisor = 1024 * 1024
        return {
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / divisor,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / divisor,
        }

    active_batch: dict[str, object] = {"device": None, "loss": None}

    def batch_started(position: int, source_index: int) -> None:
        active_batch["device"] = memory_begin()
        active_batch["loss"] = None
        active.update(
            {
                "batch_position": position,
                "source_batch_index": source_index,
                "status": "BATCH_STARTED",
            }
        )
        flush()

    def batch_completed(position: int, source_index: int, started_ns: int) -> None:
        record = {
            "batch_position": position,
            "epoch": active["epoch"],
            "loss": active_batch["loss"],
            "phase": active["phase"],
            "source_batch_index": source_index,
            "status": "BATCH_COMPLETED",
            "wall_time_ms": max(
                1, (time.monotonic_ns() - started_ns) // 1_000_000
            ),
            **memory_finish(active_batch["device"]),
        }
        batches.append(record)
        active.update(
            {
                "batch_position": position,
                "completed_batch_records": len(batches),
                "source_batch_index": source_index,
                "status": "BATCH_COMPLETED",
            }
        )
        flush()

    restore_iterations: list[object] = []
    original_loss_for_restore: object | None = None
    if prefix_contract is not None:
        if preallocated_train_batches is None or preallocated_valid_batches is None:
            raise RuntimeError("fixed-batch prefix was not preallocated")
        train_indices = tuple(int(value) for value in prefix_contract["train_batch_indices"])
        valid_indices = tuple(int(value) for value in prefix_contract["eval_batch_indices"])
        if type(train_data) is type(valid_data):
            raise RuntimeError("fixed-batch train and eval loader types overlap")
        restore_iterations.append(_install_fixed_batch_iteration(
            train_data,
            preallocated_train_batches,
            train_indices,
            on_batch_started=batch_started,
            on_batch_completed=batch_completed,
        ))
        restore_iterations.append(_install_fixed_batch_iteration(
            valid_data,
            preallocated_valid_batches,
            valid_indices,
            on_batch_started=batch_started,
            on_batch_completed=batch_completed,
        ))
        original_loss = trainer.model.calculate_loss
        original_loss_for_restore = original_loss

        def measured_loss(interaction: object) -> object:
            result = original_loss(interaction)
            active_batch["loss"] = _numeric_loss(result)
            active["last_loss_observation"] = active_batch["loss"]
            flush()
            return result

        trainer.model.calculate_loss = measured_loss

    def begin_phase(phase: str, epoch: int) -> tuple[int | None, int]:
        active.clear()
        active.update({"epoch": epoch, "phase": phase, "status": "PHASE_STARTED"})
        flush()
        return memory_begin(), time.monotonic_ns()

    def finish_phase(
        *,
        phase: str,
        epoch: int,
        device: int | None,
        started_ns: int,
        status: str,
        loss: object = None,
        valid_score: float | None = None,
    ) -> None:
        phase_batch_rows = [
            row
            for row in batches
            if row["phase"] == phase and int(row["epoch"]) == epoch
        ]
        allocated_peaks = [
            float(row["peak_allocated_mib"])
            for row in phase_batch_rows
            if isinstance(row.get("peak_allocated_mib"), (int, float))
        ]
        reserved_peaks = [
            float(row["peak_reserved_mib"])
            for row in phase_batch_rows
            if isinstance(row.get("peak_reserved_mib"), (int, float))
        ]
        memory = memory_finish(device) if not phase_batch_rows else {
            "peak_allocated_mib": (
                max(allocated_peaks) if allocated_peaks else None
            ),
            "peak_reserved_mib": (
                max(reserved_peaks) if reserved_peaks else None
            ),
        }
        phases.append(
            {
                "batch_count": (
                    len(phase_batch_rows)
                    if prefix_contract is not None
                    else (len(train_data) if phase == "TRAIN" else len(valid_data))
                ),
                "epoch": epoch,
                "loss": _numeric_loss(loss),
                "phase": phase,
                "status": status,
                "valid_score": valid_score,
                "wall_time_ms": max(
                    1, (time.monotonic_ns() - started_ns) // 1_000_000
                ),
                **memory,
            }
        )
        active.clear()
        active.update({"epoch": epoch, "phase": phase, "status": "PHASE_COMPLETED"})
        flush()

    def measured_train_epoch(
        epoch_train_data: object,
        epoch_idx: int,
        loss_func: object = None,
        show_progress: bool = False,
    ) -> object:
        device, started_ns = begin_phase("TRAIN", int(epoch_idx))
        loss: object = None
        status = "SUCCESS"
        try:
            loss = original_train_epoch(
                epoch_train_data,
                epoch_idx,
                loss_func=loss_func,
                show_progress=show_progress,
            )
            return loss
        except Exception:
            status = "RUNTIME_FAILURE"
            raise
        finally:
            finish_phase(
                phase="TRAIN",
                epoch=int(epoch_idx),
                device=device,
                started_ns=started_ns,
                status=status,
                loss=loss,
            )

    def measured_valid_epoch(
        epoch_valid_data: object,
        show_progress: bool = False,
    ) -> object:
        epoch = max(
            (int(row["epoch"]) for row in phases if row["phase"] == "TRAIN"),
            default=-1,
        )
        device, started_ns = begin_phase("EVAL", epoch)
        result: object = None
        status = "SUCCESS"
        try:
            result = original_valid_epoch(
                epoch_valid_data,
                show_progress=show_progress,
            )
            return result
        except Exception:
            status = "RUNTIME_FAILURE"
            raise
        finally:
            valid_score = None
            if (
                isinstance(result, tuple)
                and result
                and isinstance(result[0], (int, float))
            ):
                valid_score = float(result[0])
            finish_phase(
                phase="EVAL",
                epoch=epoch,
                device=device,
                started_ns=started_ns,
                status=status,
                valid_score=valid_score,
            )

    trainer._train_epoch = measured_train_epoch
    trainer._valid_epoch = measured_valid_epoch
    flush()

    def restore() -> None:
        for restore_iteration in reversed(restore_iterations):
            restore_iteration()
        if original_loss_for_restore is not None:
            trainer.model.calculate_loss = original_loss_for_restore

    return telemetry, restore


def _finalize_resource_telemetry(value: dict[str, object]) -> dict[str, object]:
    phases = list(value["phase_records"])
    batches = list(value.get("batch_records", ()))
    train_rows = [row for row in phases if row["phase"] == "TRAIN"]
    eval_rows = [row for row in phases if row["phase"] == "EVAL"]
    valid_rows = [
        row for row in eval_rows if isinstance(row.get("valid_score"), (int, float))
    ]
    best_epoch = None
    if valid_rows:
        best_epoch = max(valid_rows, key=lambda row: float(row["valid_score"]))["epoch"]
    peaks = [
        float(row[key])
        for row in (*phases, *batches)
        for key in ("peak_allocated_mib", "peak_reserved_mib")
        if isinstance(row.get(key), (int, float))
    ]
    return {
        **value,
        "best_observed_epoch": best_epoch,
        "completed_batch_records": len(batches),
        "epochs_completed": len(train_rows),
        "loss_trend": [row.get("loss") for row in train_rows],
        "peak_gpu_memory_mib": max(peaks) if peaks else None,
        "schema": "recclaw.worker-resource-telemetry.v2",
    }


def _write_durable_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load_prefix_contract(
    path: Path,
    *,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "ml-1m",
        "epochs": epochs,
        "execution_purpose": "RESOURCE_PROBE_ONLY",
        "seed": seed,
    }
    drift = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if payload.get("schema") != "recclaw.q0r-fixed-batch-prefix-contract.v1":
        drift["schema"] = payload.get("schema")
    for field in ("train_batch_indices", "eval_batch_indices"):
        values = payload.get(field)
        if (
            not isinstance(values, list)
            or not values
            or values != sorted(set(values))
            or any(not isinstance(value, int) or value < 0 for value in values)
        ):
            drift[field] = values
    if drift:
        raise RuntimeError(
            "fixed-batch prefix contract mismatch: "
            + json.dumps(drift, sort_keys=True)
        )
    payload["contract_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def _await_start_gate(
    path: Path, expected: dict[str, object], *, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            observed = json.loads(path.read_text(encoding="utf-8"))
            if observed != {**expected, "gate_status": "TRAINING_AUTHORIZED"}:
                raise RuntimeError("training start gate identity mismatch")
            return
        time.sleep(0.05)
    raise TimeoutError("training start gate was not accepted")


def _mount_bind(source: Path, target: Path) -> None:
    subprocess.run(
        ["/usr/bin/mount", "--bind", str(source), str(target)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _activate_filesystem_capability(
    capability: object,
) -> dict[str, object]:
    from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
        filesystem_mount_audit,
        make_mount_tree_read_only,
        make_mount_writable,
    )

    working = Path(capability.run_working_directory)
    if Path.cwd().resolve() != working.resolve():
        raise RuntimeError("training worker cwd is not instance-private")
    result_root = Path(capability.result_root)
    subprocess.run(
        ["/usr/bin/mount", "--make-rprivate", "/"],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["/usr/bin/mount", "--bind", str(result_root), str(result_root)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    runtime_access_mounts = tuple(
        Path(path).resolve()
        for path in (
            *capability.device_access_mounts,
            *capability.runtime_control_mounts,
        )
    )
    for path in runtime_access_mounts:
        _mount_bind(path, path)
    make_mount_tree_read_only(Path("/"))
    make_mount_writable(result_root)
    for path in runtime_access_mounts:
        make_mount_writable(path)
    os.chdir(working)
    temporary = Path(capability.temp_root)
    writable_mounts = (
        result_root,
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/dev/shm"),
        *runtime_access_mounts,
    )
    _mount_bind(temporary, writable_mounts[1])
    _mount_bind(temporary / "var_tmp", writable_mounts[2])
    _mount_bind(temporary / "dev_shm", writable_mounts[3])
    for path in writable_mounts[1:4]:
        make_mount_writable(path)
    expected_environment = capability.environment
    if any(os.environ.get(key) != value for key, value in expected_environment.items()):
        raise RuntimeError("training writable environment projection mismatch")
    audit = filesystem_mount_audit(writable_mounts)
    if audit["status"] != "PASS":
        raise RuntimeError(
            "training mount confinement audit failed: "
            f"{audit['unexpected_writable_mount_targets']}"
        )
    return audit


def _activate_hash_audited_filesystem(
    capability: object,
) -> dict[str, object]:
    from recclaw_core.experiments.helix_abc_v1.canonical import (
        sha256_digest,
    )

    working = Path(capability.run_working_directory)
    result_root = Path(capability.result_root)
    if (
        Path.cwd().resolve() != working.resolve()
        or not working.resolve().is_relative_to(result_root.resolve())
    ):
        raise RuntimeError("training worker cwd is not run-private")
    expected_environment = capability.environment
    if any(
        os.environ.get(key) != value
        for key, value in expected_environment.items()
    ):
        raise RuntimeError("training writable environment projection mismatch")
    payload = {
        "allowed_writable_mount_targets": [result_root.resolve().as_posix()],
        "enforcement": [
            "PACKAGE_OWNED_WORKER_ONLY",
            "NO_CANDIDATE_EXECUTABLE_CODE",
            "RUN_PRIVATE_CWD_AND_ENVIRONMENT",
            "PROTECTED_AND_SIBLING_ROOT_HASH_AUDIT",
        ],
        "isolation_mode": "HASH_AUDITED_PRIVATE_ROOT_V1",
        "missing_writable_mount_targets": [],
        "mount_count": 0,
        "status": "PASS",
        "unexpected_writable_mount_targets": [],
        "writable_mount_targets": [result_root.resolve().as_posix()],
    }
    return {**payload, "audit_digest": sha256_digest(payload)}


def main() -> int:
    worker_started_ns = time.monotonic_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-digest", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--execution-purpose", required=True)
    parser.add_argument("--execution-recipe-path", required=True)
    parser.add_argument("--filesystem-capability-path", required=True)
    parser.add_argument(
        "--filesystem-mode",
        choices=(
            "HASH_AUDITED_PRIVATE_ROOT_V1",
            "READ_ONLY_MOUNT_NAMESPACE_V2",
        ),
        required=True,
    )
    parser.add_argument("--force-failure", action="store_true")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--permit-digest", required=True)
    parser.add_argument("--prefix-contract-path")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recbole-root", required=True)
    parser.add_argument("--resource-telemetry", action="store_true")
    parser.add_argument("--resource-telemetry-path")
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runner-abi", required=True)
    parser.add_argument("--runtime-binding-digest", required=True)
    parser.add_argument("--runtime-release-digest", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--start-confirmation-path", required=True)
    parser.add_argument("--start-gate-path", required=True)
    args = parser.parse_args()
    if args.prefix_contract_path and not args.resource_telemetry:
        raise RuntimeError("fixed-batch prefix requires resource telemetry")
    if args.resource_telemetry and not args.resource_telemetry_path:
        raise RuntimeError("resource telemetry path is required")

    project_root = Path(args.project_root).resolve()
    recbole_root = Path(args.recbole_root).resolve()
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root))
    from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
        TrainingFilesystemCapabilityV2,
    )

    capability_payload = json.loads(
        Path(args.filesystem_capability_path).read_text(encoding="utf-8")
    )
    expected_capability_digest = capability_payload.pop("capability_digest")
    capability = TrainingFilesystemCapabilityV2.create(capability_payload)
    if capability.capability_digest != expected_capability_digest:
        raise RuntimeError("training filesystem capability digest mismatch")
    mount_audit = (
        _activate_filesystem_capability(capability)
        if args.filesystem_mode == "READ_ONLY_MOUNT_NAMESPACE_V2"
        else _activate_hash_audited_filesystem(capability)
    )
    start_identity = {
        "binding_digest": args.binding_digest,
        "claim_id": args.claim_id,
        "execution_purpose": args.execution_purpose,
        "ordinary_launch_attempt_ordinal": 1,
        "permit_digest": args.permit_digest,
        "round_id": args.round_id,
        "run_id": args.run_id,
        "runner_abi": args.runner_abi,
        "runtime_binding_digest": args.runtime_binding_digest,
        "runtime_release_digest": args.runtime_release_digest,
    }
    _write_durable_json(
        Path(args.start_confirmation_path),
        {
            **start_identity,
            "pid": os.getpid(),
            "start_status": "START_CONFIRMED",
        },
    )
    _await_start_gate(
        Path(args.start_gate_path),
        start_identity,
        timeout_seconds=30.0,
    )

    sys.path.insert(0, str(project_root / "scripts"))
    sys.path.insert(0, str(recbole_root))

    import numpy as np

    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_

    import run_candidate

    run_candidate.install_optional_dependency_stubs()
    run_candidate.patch_recbole_runtime_compat()
    recipe_document = json.loads(
        Path(args.execution_recipe_path).read_text(encoding="utf-8")
    )
    recipe = dict(
        recipe_document.get("execution_recipe", recipe_document)
    )
    if (
        recipe.get("model") != args.model
        or recipe.get("entrypoint") is None
        or recipe.get("mechanism_id") is None
    ):
        raise RuntimeError("training execution recipe identity mismatch")
    entrypoint = str(recipe["entrypoint"])
    entrypoint_object = run_candidate.import_object(entrypoint)
    entrypoint_module = inspect.getmodule(entrypoint_object)
    if entrypoint_module is None or not getattr(entrypoint_module, "__file__", None):
        raise RuntimeError("training entrypoint source is unavailable")
    source_sha256 = hashlib.sha256(
        Path(str(entrypoint_module.__file__)).read_bytes()
    ).hexdigest()
    if source_sha256 != recipe.get("entrypoint_source_sha256"):
        raise RuntimeError("training entrypoint source digest mismatch")
    if entrypoint.startswith("recclaw_ext."):
        run_candidate.patch_recbole_model_lookup(
            {args.model: entrypoint_object}
        )

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed

    output_path = Path(args.output_path)
    log_path = Path(args.log_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object]
    exit_code = 0
    argv_before = sys.argv[:]
    device_evidence: dict[str, object] = {}
    resource_telemetry: dict[str, object] | None = None
    telemetry_path = (
        Path(args.resource_telemetry_path)
        if args.resource_telemetry_path is not None
        else None
    )
    prefix_contract = (
        _load_prefix_contract(
            Path(args.prefix_contract_path),
            epochs=args.epochs,
            seed=args.seed,
        )
        if args.prefix_contract_path is not None
        else None
    )
    try:
        import torch

        device_evidence = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "torch_cuda_version": torch.version.cuda,
        }
        if (
            device_evidence["cuda_available"] is not True
            or int(device_evidence["cuda_device_count"]) < 1
        ):
            raise RuntimeError("M6E_CUDA_DEVICE_CAPABILITY_UNAVAILABLE")
        config_files = [
            recbole_root
            / "recbole"
            / "properties"
            / "model"
            / f"{recipe['base_model_config']}.yaml",
            project_root / "configs" / "task_ml1m.yaml",
            project_root / "configs" / "lightgcn_metrics.yaml",
        ]
        config_dict = {
            **dict(recipe.get("config", {})),
            "benchmark_filename": ["train", "dev", "dev"],
            "checkpoint_dir": str(checkpoint_dir),
            "data_path": str(Path(args.data_path).resolve()),
            "epochs": args.epochs,
            "eval_step": 1,
            "reproducibility": True,
            "seed": args.seed,
            "show_progress": False,
            "state": "ERROR",
            "stopping_step": min(10, args.epochs),
            "use_gpu": True,
        }
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                if args.force_failure:
                    raise RuntimeError("M6E_CONTROLLED_FORCED_RUNTIME_FAILURE")
                config = Config(
                    model=args.model,
                    dataset=args.dataset,
                    config_file_list=[str(path) for path in config_files],
                    config_dict=config_dict,
                )
                init_seed(config["seed"], config["reproducibility"])
                dataset = create_dataset(config)
                train_data, valid_data, _unused_test_data = data_preparation(
                    config, dataset
                )
                preallocated_train_batches = None
                preallocated_valid_batches = None
                if prefix_contract is not None:
                    init_seed(config["seed"], config["reproducibility"])
                    preallocated_train_batches = _preallocate_batches(
                        train_data,
                        tuple(
                            int(value)
                            for value in prefix_contract["train_batch_indices"]
                        ),
                    )
                    init_seed(config["seed"], config["reproducibility"])
                    preallocated_valid_batches = _preallocate_batches(
                        valid_data,
                        tuple(
                            int(value)
                            for value in prefix_contract["eval_batch_indices"]
                        ),
                    )
                init_seed(config["seed"], config["reproducibility"])
                model_class = get_model(config["model"])
                model = model_class(config, train_data._dataset).to(
                    config["device"]
                )
                trainer = get_trainer(
                    config["MODEL_TYPE"], config["model"]
                )(config, model)
                restore_fixed_batch_iteration = lambda: None
                if args.resource_telemetry:
                    if telemetry_path is None:
                        raise RuntimeError("resource telemetry path is unavailable")
                    (
                        resource_telemetry,
                        restore_fixed_batch_iteration,
                    ) = _install_resource_telemetry(
                        trainer,
                        torch=torch,
                        train_data=train_data,
                        valid_data=valid_data,
                        telemetry_path=telemetry_path,
                        prefix_contract=prefix_contract,
                        preallocated_train_batches=preallocated_train_batches,
                        preallocated_valid_batches=preallocated_valid_batches,
                        worker_started_ns=worker_started_ns,
                    )
                try:
                    best_valid_score, best_valid_result = trainer.fit(
                        train_data,
                        valid_data,
                        saved=False,
                        show_progress=False,
                    )
                finally:
                    restore_fixed_batch_iteration()
        payload = {
            "best_valid_result": best_valid_result,
            "best_valid_score": best_valid_score,
            "exit_status": "SUCCESS",
            "execution_recipe_digest": recipe.get("execution_recipe_digest"),
            "filesystem_mount_audit": mount_audit,
            "metric_source": "BEST_VALID_RESULT",
            "model": args.model,
            "online_partition_role": "DEVELOPMENT_VALIDATION",
            "training_device_evidence": device_evidence,
        }
    except Exception as error:  # noqa: BLE001 - failure is a Pilot outcome.
        exit_code = 1
        payload = {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "exit_status": "RUNTIME_FAILURE",
            "filesystem_mount_audit": mount_audit,
            "model": args.model,
            "traceback": traceback.format_exc(),
            "training_device_evidence": device_evidence,
        }
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(payload["traceback"])
    finally:
        sys.argv = argv_before
    if args.resource_telemetry:
        payload["resource_telemetry"] = (
            _finalize_resource_telemetry(resource_telemetry)
            if resource_telemetry is not None
            else None
        )
        if telemetry_path is not None and resource_telemetry is not None:
            _write_durable_json(
                telemetry_path,
                _finalize_resource_telemetry(resource_telemetry),
            )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
