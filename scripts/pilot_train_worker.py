#!/usr/bin/env python3
"""Run one package-owned RecBole Pilot training execution."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


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
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-digest", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--execution-purpose", required=True)
    parser.add_argument("--filesystem-capability-path", required=True)
    parser.add_argument("--force-failure", action="store_true")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--permit-digest", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recbole-root", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runner-abi", required=True)
    parser.add_argument("--runtime-binding-digest", required=True)
    parser.add_argument("--runtime-release-digest", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--start-confirmation-path", required=True)
    parser.add_argument("--start-gate-path", required=True)
    args = parser.parse_args()

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
    mount_audit = _activate_filesystem_capability(capability)
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
    from recbole.quick_start import run

    run_candidate.patch_recbole_runtime_compat()
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
        sys.argv = [
            "pilot_train_worker.py",
            f"--model={args.model}",
            f"--dataset={args.dataset}",
        ]
        config_files = [
            project_root / "configs" / "task_ml1m.yaml",
            project_root / "configs" / "lightgcn_metrics.yaml",
        ]
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                if args.force_failure:
                    raise RuntimeError("M6E_CONTROLLED_FORCED_RUNTIME_FAILURE")
                result = run(
                    args.model,
                    args.dataset,
                    config_file_list=[str(path) for path in config_files],
                    config_dict={
                        "checkpoint_dir": str(checkpoint_dir),
                        "data_path": str(Path(args.data_path).resolve()),
                        "epochs": args.epochs,
                        "eval_step": 1,
                        "reproducibility": True,
                        "seed": args.seed,
                        "show_progress": False,
                        "state": "ERROR",
                        "stopping_step": args.epochs,
                        "use_gpu": True,
                    },
                    saved=False,
                )
        payload = {
            "best_valid_result": result.get("best_valid_result", {}),
            "best_valid_score": result.get("best_valid_score"),
            "exit_status": "SUCCESS",
            "filesystem_mount_audit": mount_audit,
            "model": args.model,
            "test_result": result.get("test_result", {}),
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
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
