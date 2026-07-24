"""Package-owned M6 Pilot training adapter and mechanical result closure."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json_bytes, sha256_digest
from .runtime_contracts import (
    ClosedRuntimeRecord,
    CommonResultClosureV1,
    ExecutionStartReceiptV1,
    RawResultEnvelopeV1,
    StartStatus,
)
from .state_store import (
    MarkExecutionFinishedCommand,
    MarkExecutionStartedCommand,
    RegisterArtifactCommand,
    SingleWriterExperimentStoreV1,
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


class PilotRawTrainingOutputV1(ClosedRuntimeRecord):
    record_type = "PilotRawTrainingOutputV1"
    required_fields = frozenset(
        {
            "binding_digest",
            "candidate_id",
            "epochs_requested",
            "exit_status",
            "gpu_cost_microunits",
            "gpu_device_time_ms",
            "model",
            "normalized_metrics",
            "permit_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "training_backend_started",
            "wall_time_ms",
            "worker_result_digest",
        }
    )


def _register(
    store: SingleWriterExperimentStoreV1,
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


class PilotTrainingLauncherV1:
    def __init__(
        self,
        store: SingleWriterExperimentStoreV1,
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
        permit: Any,
        binding: Any,
        materialization_artifacts: tuple[dict[str, Any], ...],
    ) -> tuple[PilotRawTrainingOutputV1, RawResultEnvelopeV1]:
        profile = pilot_training_profile()
        claim = self._store.get_execution_claim(str(binding.round_id))
        if (
            claim["claim_state"] != "CLAIMED"
            or claim["permit_digest"] != permit.digest
            or claim["binding_digest"] != binding.digest
        ):
            raise ValueError("Pilot launcher requires the exact committed claim")
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
        runner_abi = str(profile["runner_abi"])
        receipt = ExecutionStartReceiptV1(
            {
                "binding_digest": binding.digest,
                "claim_id": claim["claim_id"],
                "ordinary_launch_attempt_ordinal": 1,
                "permit_digest": permit.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": runner_abi,
                "start_status": StartStatus.STARTED.value,
            }
        )
        receipt_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="EXECUTION_START_RECEIPT_V1",
            relative_path=f"artifacts/{binding.run_id}/execution_start_receipt.v1.json",
            producer="PilotTrainingLauncherV1",
            idempotency_key=f"m6-receipt:{claim['claim_id']}",
            payload=canonical_json_bytes(receipt.to_dict()),
        )
        self._store.mark_execution_started(
            MarkExecutionStartedCommand(
                round_id=str(binding.round_id),
                claim_id=str(claim["claim_id"]),
                receipt_artifact_id=str(receipt_artifact["artifact_id"]),
                idempotency_key=f"m6-start:{claim['claim_id']}",
            )
        )

        run_root = root / "pilot_runs" / str(binding.run_id)
        run_root.mkdir(parents=True, exist_ok=False)
        log_path = run_root / "training.log"
        worker_path = run_root / "worker_result.json"
        checkpoint_dir = run_root / "checkpoints"
        command = [
            str(self._python),
            str(self._project_root / "scripts" / "pilot_train_worker.py"),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--data-path",
            str(self._data_path),
            "--dataset",
            str(profile["dataset"]),
            "--epochs",
            str(profile["max_epochs"]),
            "--log-path",
            str(log_path),
            "--model",
            model,
            "--output-path",
            str(worker_path),
            "--project-root",
            str(self._project_root),
            "--recbole-root",
            str(self._recbole_root),
            "--seed",
            str(profile["ordinary_execution_seed"]),
        ]
        started = time.monotonic_ns()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=self._project_root,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(profile["wall_time_ceiling_ms_per_execution"]) / 1000,
            )
            return_code = int(completed.returncode)
            launcher_stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            return_code = 124
            launcher_stderr = str(error)
        wall_time_ms = max(1, (time.monotonic_ns() - started) // 1_000_000)
        rate = int(
            profile["gpu_meter"][
                "normalized_rate_microunits_per_device_hour"
            ]
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
        exit_status = (
            "SUCCESS"
            if return_code == 0 and worker.get("exit_status") == "SUCCESS"
            else "RUNTIME_FAILURE"
        )
        raw_output = PilotRawTrainingOutputV1(
            {
                "binding_digest": binding.digest,
                "candidate_id": binding.candidate_id,
                "epochs_requested": int(profile["max_epochs"]),
                "exit_status": exit_status,
                "gpu_cost_microunits": gpu_cost,
                "gpu_device_time_ms": wall_time_ms,
                "model": model,
                "normalized_metrics": metrics,
                "permit_digest": permit.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": runner_abi,
                "training_backend_started": True,
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
            idempotency_key=f"m6-log:{claim['claim_id']}",
            payload=log_bytes,
        )
        worker_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="PILOT_WORKER_RESULT_V1",
            relative_path=f"artifacts/{binding.run_id}/worker_result.v1.json",
            producer="PilotTrainingRunnerV1",
            idempotency_key=f"m6-worker:{claim['claim_id']}",
            payload=worker_bytes,
        )
        raw_artifact = _register(
            self._store,
            round_id=str(binding.round_id),
            artifact_type="RAW_RUN_OUTPUT_V1",
            relative_path=f"artifacts/{binding.run_id}/raw_run_output.v1.json",
            producer="PilotTrainingRunnerV1",
            idempotency_key=f"m6-raw:{claim['claim_id']}",
            payload=canonical_json_bytes(raw_output.to_dict()),
        )
        self._store.mark_execution_finished(
            MarkExecutionFinishedCommand(
                round_id=str(binding.round_id),
                claim_id=str(claim["claim_id"]),
                raw_output_artifact_id=str(raw_artifact["artifact_id"]),
                idempotency_key=f"m6-finish:{claim['claim_id']}",
            )
        )
        artifact_closure = list(
            materialization_artifacts
            + (receipt_artifact, log_artifact, worker_artifact, raw_artifact)
        )
        closure = CommonResultClosureV1(
            {
                "claim_id": str(claim["claim_id"]),
                "decision": "PASS",
                "permit_digest": permit.digest,
                "raw_output_digest": raw_output.digest,
                "reason_codes": [],
                "release_projection_digest": pilot_training_profile_digest(),
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "start_receipt_digest": receipt.digest,
                "subchecks": [
                    {"check": "EXACT_PROFILE", "passed": True},
                    {"check": "SINGLE_START", "passed": True},
                    {"check": "ARTIFACT_CLOSURE", "passed": True},
                ],
            }
        )
        envelope = RawResultEnvelopeV1(
            {
                "artifact_closure": artifact_closure,
                "binding_digest": binding.digest,
                "candidate_id": binding.candidate_id,
                "common_result_closure_digest": closure.digest,
                "evaluation_purpose": "DEVELOPMENT_PILOT_OFFLINE_TOPN",
                "exit_status": exit_status,
                "metric_source": "RECBOLE_FULL_SORT_TEST_RESULT",
                "normalized_metrics": metrics,
                "ordinary_execution_start_index": 1,
                "partition_role": "PILOT_EXCLUDED_FROM_MAIN",
                "raw_output_digest": raw_output.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "seed": int(profile["ordinary_execution_seed"]),
            }
        )
        return raw_output, envelope


__all__ = [
    "PilotRawTrainingOutputV1",
    "PilotTrainingLauncherV1",
    "pilot_training_profile",
    "pilot_training_profile_digest",
    "training_model_for_primitives",
    "training_model_for_program",
]
