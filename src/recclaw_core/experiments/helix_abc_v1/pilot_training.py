"""Package-owned training launcher with M6R release/start/result closure."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json_bytes, sha256_digest
from .state_store import RegisterArtifactCommand
from .training_execution_guard import CommonTrainingExecutionGuardV1
from .training_runtime_contracts import (
    CandidateExecutionBindingV3,
    CommonExecutionPermitV2,
    ExecutionStartConfirmationV1,
    ExecutionStartReceiptV2,
    RawResultEnvelopeV2,
    TrainingRawRunOutputV1,
    TrainingResourceAccountingV1,
    TrainingRuntimeBindingV1,
)
from .training_runtime_release import resolve_bound_training_release
from .training_state_store import (
    MarkTrainingExecutionFinishedCommandV1,
    MarkTrainingExecutionStartedCommandV1,
    PrepareTrainingAttemptCommandV1,
    TrainingSingleWriterExperimentStoreV1,
)


_RESOURCE = (
    Path(__file__).resolve().parent
    / "resources"
    / "pilot_training_profile_v1.json"
)


def pilot_training_profile() -> dict[str, Any]:
    return json.loads(_RESOURCE.read_text(encoding="utf-8"))


def pilot_training_profile_digest() -> str:
    return sha256_digest(pilot_training_profile())


def training_model_for_primitives(primitives: Sequence[str]) -> str:
    values = set(str(item) for item in primitives)
    for mapping in pilot_training_profile()["supported_mappings"]:
        required = set(mapping["required_primitives"])
        forbidden = set(mapping.get("forbidden_primitives", []))
        if required.issubset(values) and not forbidden.intersection(values):
            return str(mapping["model"])
    raise ValueError("mechanism program is outside the Pilot training profile")


def training_model_for_program(program: Mapping[str, Any]) -> str:
    primitives = [
        str(component["primitive_id"])
        for component in program["program_payload"]["components"]
        if "primitive_id" in component
    ]
    return training_model_for_primitives(primitives)


def classify_training_termination(
    *, return_code: int, timed_out: bool, worker_status: str | None
) -> tuple[str, str]:
    if timed_out:
        return "RUNTIME_FAILURE", "TIMEOUT"
    if return_code == 0 and worker_status == "SUCCESS":
        return "SUCCESS", "SUCCESS"
    return "RUNTIME_FAILURE", "CRASH_OR_RUNTIME_FAILURE"


def _register(
    store: TrainingSingleWriterExperimentStoreV1,
    *,
    round_id: str,
    artifact_type: str,
    relative_path: str,
    producer: str,
    idempotency_key: str,
    payload: bytes,
) -> dict[str, Any]:
    return store.register_artifact(
        RegisterArtifactCommand(
            round_id=round_id,
            artifact_type=artifact_type,
            relative_path=relative_path,
            producer=producer,
            idempotency_key=idempotency_key,
        ),
        payload,
    )


def _write_gate(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes(payload) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PilotTrainingLauncherV1:
    def __init__(
        self,
        store: TrainingSingleWriterExperimentStoreV1,
        *,
        project_root: Path,
        recbole_root: Path,
        data_path: Path,
        python_executable: Path,
    ) -> None:
        self._store = store
        self._project_root = project_root.resolve()
        self._recbole_root = recbole_root.resolve()
        self._data_path = data_path.resolve()
        self._python = python_executable.resolve()

    def launch(
        self,
        *,
        permit: CommonExecutionPermitV2,
        binding: CandidateExecutionBindingV3,
        runtime_binding: TrainingRuntimeBindingV1,
        materialization_artifacts: tuple[dict[str, Any], ...],
    ) -> tuple[TrainingRawRunOutputV1, RawResultEnvelopeV2]:
        profile = pilot_training_profile()
        claim = self._store.get_execution_claim(str(binding.round_id))
        release = resolve_bound_training_release(
            runner_abi=str(claim["runner_abi"]),
            runtime_release_digest=str(claim["runtime_release_digest"]),
            execution_purpose=str(claim["execution_purpose"]),
        )
        if (
            claim["claim_state"] != "CLAIMED"
            or claim["permit_digest"] != permit.digest
            or claim["binding_digest"] != binding.digest
            or claim["budget_digest"] != binding.budget_digest
            or claim["candidate_id"] != binding.candidate_id
            or claim["experiment_id"] != runtime_binding.experiment_id
            or claim["run_id"] != binding.run_id
            or claim["runtime_release_digest"] != binding.runtime_release_digest
            or claim["runtime_binding_digest"] != runtime_binding.digest
            or claim["runner_abi"] != binding.runner_abi
            or claim["execution_purpose"] != binding.execution_purpose
            or runtime_binding.release_digest != release.digest
        ):
            raise ValueError("training launcher requires the exact committed release")

        root = Path(str(binding.arm_private_root))
        config_path = (
            root
            / "recclaw_ext"
            / "generated"
            / str(binding.candidate_id)
            / "handler_config.json"
        )
        handler = json.loads(config_path.read_text(encoding="utf-8"))
        model = training_model_for_primitives(handler["primitives"])
        self._store.prepare_training_attempt(
            PrepareTrainingAttemptCommandV1(
                round_id=str(binding.round_id),
                claim_id=str(claim["claim_id"]),
                permit_digest=permit.digest,
                binding_digest=binding.digest,
                runtime_release_digest=str(binding.runtime_release_digest),
                runtime_binding_digest=runtime_binding.digest,
                runner_abi=str(binding.runner_abi),
                execution_purpose=str(binding.execution_purpose),
            )
        )

        run_root = root / "pilot_runs" / str(binding.run_id)
        checkpoint_dir = run_root / "checkpoints"
        if (
            runtime_binding.result_root != run_root.resolve().as_posix()
            or runtime_binding.checkpoint_root
            != checkpoint_dir.resolve().as_posix()
        ):
            raise ValueError("training launcher roots differ from purpose binding")
        run_root.mkdir(parents=True, exist_ok=False)
        log_path = run_root / "training.log"
        worker_path = run_root / "worker_result.json"
        confirmation_path = run_root / "start_confirmation.json"
        gate_path = run_root / "start_gate.json"
        command = [
            str(self._python),
            str(self._project_root / "scripts" / "pilot_train_worker.py"),
            "--binding-digest",
            binding.digest,
            "--claim-id",
            str(claim["claim_id"]),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--data-path",
            str(self._data_path),
            "--dataset",
            str(profile["dataset"]),
            "--epochs",
            str(profile["max_epochs"]),
            "--execution-purpose",
            str(binding.execution_purpose),
            "--log-path",
            str(log_path),
            "--model",
            model,
            "--output-path",
            str(worker_path),
            "--permit-digest",
            permit.digest,
            "--project-root",
            str(self._project_root),
            "--recbole-root",
            str(self._recbole_root),
            "--round-id",
            str(binding.round_id),
            "--run-id",
            str(binding.run_id),
            "--runner-abi",
            str(binding.runner_abi),
            "--runtime-binding-digest",
            runtime_binding.digest,
            "--runtime-release-digest",
            str(binding.runtime_release_digest),
            "--seed",
            str(profile["ordinary_execution_seed"]),
            "--start-confirmation-path",
            str(confirmation_path),
            "--start-gate-path",
            str(gate_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=self._project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        confirmation_deadline = time.monotonic() + 30.0
        while not confirmation_path.is_file():
            if process.poll() is not None:
                raise RuntimeError("training wrapper exited before START_CONFIRMED")
            if time.monotonic() >= confirmation_deadline:
                process.kill()
                process.wait()
                raise TimeoutError("training wrapper did not confirm its start")
            time.sleep(0.05)

        confirmation = ExecutionStartConfirmationV1(
            json.loads(confirmation_path.read_text(encoding="utf-8"))
        )
        if confirmation.pid != process.pid:
            process.kill()
            process.wait()
            raise RuntimeError("START_CONFIRMED does not identify the spawned process")
        confirmation_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="EXECUTION_START_CONFIRMATION_V1",
            relative_path=(
                f"artifacts/{binding.run_id}/execution_start_confirmation.v1.json"
            ),
            producer="PilotTrainingLauncherV1",
            idempotency_key=f"m6r-confirmation:{claim['claim_id']}",
            payload=canonical_json_bytes(confirmation.to_dict()),
        )
        receipt = ExecutionStartReceiptV2(
            {
                "binding_digest": binding.digest,
                "claim_id": claim["claim_id"],
                "execution_purpose": binding.execution_purpose,
                "ordinary_launch_attempt_ordinal": 1,
                "permit_digest": permit.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": binding.runner_abi,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": binding.runtime_release_digest,
                "start_confirmation_digest": confirmation.digest,
                "start_status": "STARTED",
            }
        )
        receipt_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="EXECUTION_START_RECEIPT_V2",
            relative_path=f"artifacts/{binding.run_id}/execution_start_receipt.v2.json",
            producer="PilotTrainingLauncherV1",
            idempotency_key=f"m6r-receipt:{claim['claim_id']}",
            payload=canonical_json_bytes(receipt.to_dict()),
        )
        self._store.mark_training_execution_started(
            MarkTrainingExecutionStartedCommandV1(
                round_id=str(binding.round_id),
                claim_id=str(claim["claim_id"]),
                receipt_artifact_id=str(receipt_artifact["artifact_id"]),
                confirmation_artifact_id=str(
                    confirmation_artifact["artifact_id"]
                ),
                idempotency_key=f"m6r-start:{claim['claim_id']}",
            )
        )
        _write_gate(
            gate_path,
            {
                "binding_digest": binding.digest,
                "claim_id": claim["claim_id"],
                "execution_purpose": binding.execution_purpose,
                "gate_status": "TRAINING_AUTHORIZED",
                "ordinary_launch_attempt_ordinal": 1,
                "permit_digest": permit.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": binding.runner_abi,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": binding.runtime_release_digest,
            },
        )

        started = time.monotonic_ns()
        timed_out = False
        try:
            _stdout, launcher_stderr = process.communicate(
                timeout=int(profile["wall_time_ceiling_ms_per_execution"]) / 1000,
            )
            return_code = int(process.returncode)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            process.kill()
            process.communicate()
            return_code = 124
            launcher_stderr = str(error)
        wall_time_ms = max(1, (time.monotonic_ns() - started) // 1_000_000)
        rate = int(
            profile["gpu_meter"]["normalized_rate_microunits_per_device_hour"]
        )
        gpu_cost = round(wall_time_ms * rate / 3_600_000)
        if worker_path.exists():
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
        else:
            worker = {
                "error_message": launcher_stderr,
                "error_type": "TIMEOUT" if timed_out else "WORKER_NO_RESULT",
                "exit_status": "RUNTIME_FAILURE",
                "model": model,
            }
            worker_path.write_bytes(canonical_json_bytes(worker) + b"\n")
        metrics = {
            str(key).lower(): float(value)
            for key, value in dict(worker.get("test_result", {})).items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if "ndcg@10" in metrics:
            metrics["ndcg"] = metrics["ndcg@10"]
        exit_status, termination_class = classify_training_termination(
            return_code=return_code,
            timed_out=timed_out,
            worker_status=worker.get("exit_status"),
        )
        raw_output = TrainingRawRunOutputV1(
            {
                "binding_digest": binding.digest,
                "budget_digest": binding.budget_digest,
                "candidate_id": binding.candidate_id,
                "epochs_requested": int(profile["max_epochs"]),
                "environment_lock_digest": (
                    runtime_binding.environment_lock_digest
                ),
                "execution_purpose": binding.execution_purpose,
                "experiment_id": runtime_binding.experiment_id,
                "exit_status": exit_status,
                "gpu_cost_microunits": gpu_cost,
                "gpu_device_time_ms": wall_time_ms,
                "launcher_return_code": return_code,
                "lineage_digest": runtime_binding.lineage_digest,
                "metric_contract_digest": runtime_binding.metric_contract_digest,
                "model": model,
                "normalized_metrics": metrics,
                "permit_digest": permit.digest,
                "partition_purpose": runtime_binding.partition_purpose,
                "protocol_digest": runtime_binding.protocol_digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": binding.runner_abi,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": binding.runtime_release_digest,
                "seed": int(profile["ordinary_execution_seed"]),
                "training_backend_started": True,
                "termination_class": termination_class,
                "training_config_budget_digest": (
                    runtime_binding.training_config_budget_digest
                ),
                "wall_time_ms": wall_time_ms,
                "worker_result_digest": sha256_digest(worker),
            }
        )
        log_bytes = log_path.read_bytes() if log_path.exists() else b""
        worker_bytes = worker_path.read_bytes()
        log_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="PILOT_TRAINING_LOG",
            relative_path=f"artifacts/{binding.run_id}/training.log",
            producer="PilotTrainingRunnerV1",
            idempotency_key=f"m6r-log:{claim['claim_id']}",
            payload=log_bytes,
        )
        worker_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="PILOT_WORKER_RESULT_V1",
            relative_path=f"artifacts/{binding.run_id}/worker_result.v1.json",
            producer="PilotTrainingRunnerV1",
            idempotency_key=f"m6r-worker:{claim['claim_id']}",
            payload=worker_bytes,
        )
        raw_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="TRAINING_RAW_RUN_OUTPUT_V1",
            relative_path=(
                f"artifacts/{binding.run_id}/training_raw_run_output.v1.json"
            ),
            producer="PilotTrainingRunnerV1",
            idempotency_key=f"m6r-raw:{claim['claim_id']}",
            payload=canonical_json_bytes(raw_output.to_dict()),
        )
        self._store.mark_training_execution_finished(
            MarkTrainingExecutionFinishedCommandV1(
                round_id=str(binding.round_id),
                claim_id=str(claim["claim_id"]),
                raw_output_artifact_id=str(raw_artifact["artifact_id"]),
            )
        )
        resource_accounting = TrainingResourceAccountingV1(
            {
                "binding_digest": binding.digest,
                "claim_id": claim["claim_id"],
                "debits": [
                    {
                        "dimension": "GPU_COST_MICROUNITS",
                        "quantity": gpu_cost,
                    },
                    {
                        "dimension": "GPU_DEVICE_TIME_MS",
                        "quantity": wall_time_ms,
                    },
                    {"dimension": "ORDINARY_EXECUTION", "quantity": 1},
                    {"dimension": "WALL_TIME_MS", "quantity": wall_time_ms},
                ],
                "permit_digest": permit.digest,
                "raw_output_digest": raw_output.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": binding.runtime_release_digest,
            }
        )
        accounting_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="TRAINING_RESOURCE_ACCOUNTING_V1",
            relative_path=(
                f"artifacts/{binding.run_id}/training_resource_accounting.v1.json"
            ),
            producer="PilotTrainingLauncherV1",
            idempotency_key=f"m6r-accounting:{claim['claim_id']}",
            payload=canonical_json_bytes(resource_accounting.to_dict()),
        )
        artifact_closure = list(
            materialization_artifacts
            + (
                confirmation_artifact,
                receipt_artifact,
                log_artifact,
                worker_artifact,
                raw_artifact,
                accounting_artifact,
            )
        )
        closure, envelope = CommonTrainingExecutionGuardV1().close_result(
            permit=permit,
            binding=binding,
            runtime_binding=runtime_binding,
            claim=self._store.get_execution_claim(str(binding.round_id)),
            confirmation=confirmation,
            receipt=receipt,
            raw_output=raw_output,
            resource_accounting=resource_accounting,
            artifact_closure=artifact_closure,
            seed=int(profile["ordinary_execution_seed"]),
        )
        if envelope is None:
            raise RuntimeError("CommonExecutionGuard rejected the training result")
        _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="COMMON_RESULT_CLOSURE_V2",
            relative_path=f"artifacts/{binding.run_id}/common_result_closure.v2.json",
            producer="CommonTrainingExecutionGuardV1",
            idempotency_key=f"m6r-closure:{claim['claim_id']}",
            payload=canonical_json_bytes(closure.to_dict()),
        )
        _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="RAW_RESULT_ENVELOPE_V2",
            relative_path=f"artifacts/{binding.run_id}/raw_result_envelope.v2.json",
            producer="RawResultEnvelopeWriterV2",
            idempotency_key=f"m6r-envelope:{claim['claim_id']}",
            payload=canonical_json_bytes(envelope.to_dict()),
        )
        return raw_output, envelope


__all__ = [
    "classify_training_termination",
    "PilotTrainingLauncherV1",
    "pilot_training_profile",
    "pilot_training_profile_digest",
    "training_model_for_primitives",
    "training_model_for_program",
]
