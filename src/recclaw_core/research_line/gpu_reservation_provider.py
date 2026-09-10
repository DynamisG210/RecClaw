"""Bounded pre-launch NVIDIA reservation evidence collection.

This module observes one explicitly selected physical device immediately before
a worker launch.  It does not allocate a lease, maintain a ledger, or measure
GPU-active time.  A formal caller must provide an explicit authority reference
for its reservation claim.  Strict mode requires an empty process snapshot;
an explicit user-authorized shared-execution mode records every pre-existing
process instead of treating the device as empty.
"""

from __future__ import annotations

import csv
import io
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, TypeAlias

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    FreshR1Error,
    GPU_RESERVATION_EVIDENCE_SCHEMA,
    validate_gpu_reservation_evidence,
)


GPU_RESERVATION_PROVIDER_IDENTITY_SCHEMA = (
    "recclaw.research-line.gpu-reservation-provider-identity.v1"
)
GPU_RESERVATION_PROVIDER_CONFIG_SCHEMA = (
    "recclaw.research-line.gpu-reservation-provider-config.v1"
)
GPU_RESERVATION_PROVIDER_VERSION = "nvidia-smi-prelaunch-snapshot-v1"
GPU_RESERVATION_PROVIDER_SCOPE = "ONE_CANDIDATE_ONE_WORKER"
GPU_RESERVATION_EVIDENCE_SEMANTICS = (
    "CALLER_SUPPLIED_EXCLUSIVE_AUTHORITY_REF_PLUS_PRELAUNCH_AVAILABILITY; "
    "NOT_GPU_ACTIVE_TIME"
)
GPU_RESERVATION_SHARED_EXECUTION_POLICY = (
    "USER_AUTHORIZED_SHARED_EXECUTION_WITH_FULL_PROCESS_SNAPSHOT"
)
GPU_RESERVATION_SHARED_EXECUTION_SEMANTICS = (
    "CALLER_SUPPLIED_AUTHORITY_REF_PLUS_PRELAUNCH_AVAILABILITY; "
    "PREEXISTING_COMPUTE_PROCESSES_RECORDED; "
    "SHARED_EXECUTION_USER_AUTHORIZED; NOT_GPU_ACTIVE_TIME"
)
GPU_RESERVATION_EMPTY_PROCESS_POLICY = "EMPTY_PRELAUNCH_PROCESS_SNAPSHOT_REQUIRED"
DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS = 5
_NVIDIA_SMI_EXECUTABLE = "nvidia-smi"
_INVENTORY_QUERY = "index,uuid,name,pci.bus_id,memory.total,compute_mode"
_PROCESS_QUERY = "pid,process_name,used_memory"
_INVENTORY_FIELDS = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "memory.total",
    "compute_mode",
)
_PROCESS_FIELDS = ("pid", "process_name", "used_memory")


class GpuReservationProviderError(ValueError):
    """The target device cannot produce valid reservation evidence."""


CommandRunner: TypeAlias = Callable[..., Any]
Clock: TypeAlias = Callable[[], datetime]


def _single_device_token(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GpuReservationProviderError(
            "cuda_visible_devices must be one normalized physical device token"
        )
    if "," in value or any(character.isspace() for character in value):
        raise GpuReservationProviderError(
            "cuda_visible_devices must bind exactly one concrete device"
        )
    if value in {"-1", "NoDevFiles"}:
        raise GpuReservationProviderError(
            "cuda_visible_devices must bind exactly one concrete device"
        )
    if value.isdecimal() and str(int(value)) != value:
        raise GpuReservationProviderError(
            "numeric cuda_visible_devices must be canonically formatted"
        )
    return value


def _normalized_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GpuReservationProviderError(
            f"{field_name} must be normalized and non-empty"
        )
    return value


def _normalized_reference(value: object, *, field_name: str) -> str:
    reference = _normalized_text(value, field_name=field_name)
    if any(character.isspace() for character in reference):
        raise GpuReservationProviderError(
            f"{field_name} must not contain whitespace"
        )
    return reference


def _validated_timeout(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GpuReservationProviderError(
            "nvidia-smi timeout_seconds must be a positive integer"
        )
    return value


def _validated_allow_existing_compute_processes(value: object) -> bool:
    if not isinstance(value, bool):
        raise GpuReservationProviderError(
            "allow_existing_compute_processes must be a boolean"
        )
    return value


def _utc_timestamp(clock: Clock) -> str:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise GpuReservationProviderError(
            "clock must return a timezone-aware datetime"
        )
    if observed.utcoffset() is None:
        raise GpuReservationProviderError(
            "clock must return a timezone-aware datetime"
        )
    return (
        observed.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _provider_config(
    *,
    cuda_visible_devices: str,
    hostname: str,
    reservation_authority_ref: str,
    timeout_seconds: int,
    allow_existing_compute_processes: bool,
) -> Mapping[str, Any]:
    return canonical_value(
        {
            "schema": GPU_RESERVATION_PROVIDER_CONFIG_SCHEMA,
            "provider_version": GPU_RESERVATION_PROVIDER_VERSION,
            "executable": _NVIDIA_SMI_EXECUTABLE,
            "timeout_seconds": timeout_seconds,
            "cuda_visible_devices": cuda_visible_devices,
            "hostname": hostname,
            "reservation_authority_ref": reservation_authority_ref,
            "allow_existing_compute_processes": allow_existing_compute_processes,
            "existing_compute_process_policy": (
                GPU_RESERVATION_SHARED_EXECUTION_POLICY
                if allow_existing_compute_processes
                else GPU_RESERVATION_EMPTY_PROCESS_POLICY
            ),
            "inventory_query": _INVENTORY_QUERY,
            "process_query": _PROCESS_QUERY,
            "scope": GPU_RESERVATION_PROVIDER_SCOPE,
            "evidence_semantics": GPU_RESERVATION_EVIDENCE_SEMANTICS,
        }
    )


def gpu_reservation_provider_identity(
    *,
    cuda_visible_devices: str,
    hostname: str,
    reservation_authority_ref: str | None = None,
    timeout_seconds: int = DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
    allow_existing_compute_processes: bool = False,
) -> Mapping[str, Any]:
    """Return canonical provider identity without including callable state."""

    token = _single_device_token(cuda_visible_devices)
    target_hostname = _normalized_reference(hostname, field_name="hostname")
    authority_ref = _normalized_reference(
        reservation_authority_ref,
        field_name="reservation_authority_ref",
    )
    timeout = _validated_timeout(timeout_seconds)
    allow_existing = _validated_allow_existing_compute_processes(
        allow_existing_compute_processes
    )
    config = _provider_config(
        cuda_visible_devices=token,
        hostname=target_hostname,
        reservation_authority_ref=authority_ref,
        timeout_seconds=timeout,
        allow_existing_compute_processes=allow_existing,
    )
    return canonical_value(
        {
            "schema": GPU_RESERVATION_PROVIDER_IDENTITY_SCHEMA,
            "provider_version": GPU_RESERVATION_PROVIDER_VERSION,
            "allow_existing_compute_processes": allow_existing,
            "config": config,
            "config_digest": sha256_digest(config),
        }
    )


def _default_command_runner(argv: Sequence[str], **kwargs: Any) -> Any:
    return subprocess.run(argv, **kwargs)


def _parse_csv_rows(
    stdout: object,
    *,
    field_names: Sequence[str],
    output_name: str,
) -> list[dict[str, str]]:
    if not isinstance(stdout, str):
        raise GpuReservationProviderError(
            f"nvidia-smi {output_name} output must be text"
        )
    try:
        rows = csv.reader(io.StringIO(stdout))
        parsed: list[dict[str, str]] = []
        for row in rows:
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) != len(field_names):
                raise GpuReservationProviderError(
                    f"nvidia-smi {output_name} output has malformed columns"
                )
            normalized = [field.strip() for field in row]
            if any(not field for field in normalized):
                raise GpuReservationProviderError(
                    f"nvidia-smi {output_name} output has empty fields"
                )
            parsed.append(dict(zip(field_names, normalized, strict=True)))
    except csv.Error as error:
        raise GpuReservationProviderError(
            f"nvidia-smi {output_name} output is malformed CSV"
        ) from error
    return parsed


def _run_query(
    command_runner: CommandRunner,
    args: Sequence[str],
    *,
    timeout_seconds: int,
    output_name: str,
) -> str:
    argv = (_NVIDIA_SMI_EXECUTABLE, *args)
    try:
        result = command_runner(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError) as error:
        raise GpuReservationProviderError(
            f"nvidia-smi {output_name} command failed"
        ) from error
    returncode = getattr(result, "returncode", None)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise GpuReservationProviderError(
            f"nvidia-smi {output_name} command returned no status"
        )
    if returncode != 0:
        raise GpuReservationProviderError(
            f"nvidia-smi {output_name} command failed"
        )
    return getattr(result, "stdout", None)


def _inventory_argv(token: str) -> tuple[str, ...]:
    return (
        "--id",
        token,
        "--query-gpu",
        _INVENTORY_QUERY,
        "--format=csv,noheader,nounits",
    )


def _process_argv(token: str) -> tuple[str, ...]:
    return (
        "--id",
        token,
        "--query-compute-apps",
        _PROCESS_QUERY,
        "--format=csv,noheader,nounits",
    )


class NvidiaSmiGpuReservationEvidenceProvider:
    """Collect one owner-bound, pre-launch target-device snapshot."""

    __slots__ = (
        "_command_runner",
        "_cuda_visible_devices",
        "_reservation_authority_ref",
        "_hostname",
        "_clock",
        "_timeout_seconds",
        "_allow_existing_compute_processes",
        "_identity",
    )

    def __init__(
        self,
        *,
        cuda_visible_devices: str,
        reservation_authority_ref: str | None = None,
        command_runner: CommandRunner | None = None,
        hostname: str | None = None,
        clock: Clock | None = None,
        timeout_seconds: int = DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
        allow_existing_compute_processes: bool = False,
    ) -> None:
        self._cuda_visible_devices = _single_device_token(cuda_visible_devices)
        self._reservation_authority_ref = _normalized_reference(
            reservation_authority_ref,
            field_name="reservation_authority_ref",
        )
        self._timeout_seconds = _validated_timeout(timeout_seconds)
        self._allow_existing_compute_processes = (
            _validated_allow_existing_compute_processes(
                allow_existing_compute_processes
            )
        )
        self._command_runner = (
            _default_command_runner if command_runner is None else command_runner
        )
        if not callable(self._command_runner):
            raise GpuReservationProviderError("command_runner must be callable")
        self._hostname = _normalized_reference(
            socket.gethostname() if hostname is None else hostname,
            field_name="hostname",
        )
        self._clock = _utc_now if clock is None else clock
        if not callable(self._clock):
            raise GpuReservationProviderError("clock must be callable")
        self._identity = gpu_reservation_provider_identity(
            cuda_visible_devices=self._cuda_visible_devices,
            hostname=self._hostname,
            reservation_authority_ref=self._reservation_authority_ref,
            timeout_seconds=self._timeout_seconds,
            allow_existing_compute_processes=self._allow_existing_compute_processes,
        )

    def provider_identity(self) -> Mapping[str, Any]:
        """Return stable config identity; callable and clock state are excluded."""

        return canonical_value(dict(self._identity))

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            raise GpuReservationProviderError("reservation request must be a mapping")
        owner = _normalized_text(
            request.get("physical_run_id"),
            field_name="physical_run_id",
        )
        request_token = _single_device_token(request.get("cuda_visible_devices"))
        if request_token != self._cuda_visible_devices:
            raise GpuReservationProviderError(
                "reservation request device differs from provider configuration"
            )
        inventory_stdout = _run_query(
            self._command_runner,
            _inventory_argv(request_token),
            timeout_seconds=self._timeout_seconds,
            output_name="inventory",
        )
        inventory_rows = _parse_csv_rows(
            inventory_stdout,
            field_names=_INVENTORY_FIELDS,
            output_name="inventory",
        )
        if len(inventory_rows) != 1:
            raise GpuReservationProviderError(
                "nvidia-smi inventory must resolve exactly one target device"
            )
        device = inventory_rows[0]
        if request_token.isdecimal():
            if device["index"] != request_token:
                raise GpuReservationProviderError(
                    "nvidia-smi inventory device index does not match target"
                )
        elif device["uuid"] != request_token:
            raise GpuReservationProviderError(
                "nvidia-smi inventory device UUID does not match target"
            )
        if not device["uuid"].startswith("GPU-"):
            raise GpuReservationProviderError(
                "nvidia-smi inventory lacks a physical GPU UUID"
            )
        try:
            memory_total = int(device["memory.total"])
        except ValueError as error:
            raise GpuReservationProviderError(
                "nvidia-smi inventory memory.total is malformed"
            ) from error
        if memory_total < 1:
            raise GpuReservationProviderError(
                "nvidia-smi inventory memory.total is invalid"
            )
        process_stdout = _run_query(
            self._command_runner,
            _process_argv(request_token),
            timeout_seconds=self._timeout_seconds,
            output_name="compute-process",
        )
        process_rows = _parse_csv_rows(
            process_stdout,
            field_names=_PROCESS_FIELDS,
            output_name="compute-process",
        )
        if process_rows and not self._allow_existing_compute_processes:
            raise GpuReservationProviderError(
                "target GPU has an existing compute process"
            )
        observed_at_utc = _utc_timestamp(self._clock)
        inventory_projection = {
            "schema": "recclaw.research-line.gpu-inventory-snapshot.v1",
            "cuda_visible_devices": request_token,
            "device": device,
        }
        process_projection = {
            "schema": "recclaw.research-line.gpu-process-snapshot.v1",
            "cuda_visible_devices": request_token,
            "preexisting_compute_process_count": len(process_rows),
            "processes": canonical_value(process_rows),
        }
        shared_execution_user_authorized = bool(
            process_rows and self._allow_existing_compute_processes
        )
        reservation_semantics = (
            GPU_RESERVATION_SHARED_EXECUTION_SEMANTICS
            if shared_execution_user_authorized
            else GPU_RESERVATION_EVIDENCE_SEMANTICS
        )
        identity = canonical_value(
            {
                "host": self._hostname,
                "physical_gpu_id": request_token,
                "cuda_visible_devices": request_token,
                "reservation_owner_ref": owner,
                "observed_at_utc": observed_at_utc,
                "device_inventory_sha256": sha256_digest(inventory_projection),
                "process_snapshot_sha256": sha256_digest(process_projection),
                "exclusive": True,
                "scope": GPU_RESERVATION_PROVIDER_SCOPE,
                "reservation_authority_ref": self._reservation_authority_ref,
                "device_uuid": device["uuid"],
                "device_ref": f"{self._hostname}:gpu:{device['uuid']}",
                "device_name": device["name"],
                "device_pci_bus_id": device["pci.bus_id"],
                "reservation_semantics": reservation_semantics,
                "preexisting_compute_process_count": len(process_rows),
                "shared_execution_user_authorized": (
                    shared_execution_user_authorized
                ),
                "shared_execution_policy": (
                    GPU_RESERVATION_SHARED_EXECUTION_POLICY
                    if shared_execution_user_authorized
                    else GPU_RESERVATION_EMPTY_PROCESS_POLICY
                ),
                "process_snapshot": process_projection,
            }
        )
        reservation_ref = (
            f"{self._reservation_authority_ref}:nvidia-smi:{self._hostname}:"
            f"gpu:{device['uuid']}:owner:{owner}"
        )
        identity_digest = sha256_digest(identity)
        sealed = {
            "identity": canonical_value(dict(identity)),
            "identity_digest": identity_digest,
            "reservation_ref": reservation_ref,
            "schema": GPU_RESERVATION_EVIDENCE_SCHEMA,
        }
        evidence = canonical_value(
            {
                "schema": GPU_RESERVATION_EVIDENCE_SCHEMA,
                "reservation_ref": reservation_ref,
                "identity": identity,
                "identity_digest": identity_digest,
                "reservation_digest": sha256_digest(sealed),
                "preexisting_compute_process_count": len(process_rows),
                "process_snapshot": process_projection,
                "shared_execution_user_authorized": (
                    shared_execution_user_authorized
                ),
            }
        )
        try:
            validated = validate_gpu_reservation_evidence(
                evidence,
                cuda_visible_devices=request_token,
                run_id=owner,
            )
        except (FreshR1Error, TypeError, ValueError) as error:
            raise GpuReservationProviderError(
                "constructed GPU reservation evidence failed validation"
            ) from error
        if validated is None:  # pragma: no cover - evidence is constructed above
            raise GpuReservationProviderError(
                "constructed GPU reservation evidence was empty"
            )
        return validated


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_nvidia_smi_gpu_reservation_provider(
    *,
    cuda_visible_devices: str,
    reservation_authority_ref: str | None = None,
    command_runner: CommandRunner | None = None,
    hostname: str | None = None,
    clock: Clock | None = None,
    timeout_seconds: int = DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
    allow_existing_compute_processes: bool = False,
) -> NvidiaSmiGpuReservationEvidenceProvider:
    """Build the narrow provider used by a Fresh runner or disposable probe."""

    return NvidiaSmiGpuReservationEvidenceProvider(
        cuda_visible_devices=cuda_visible_devices,
        reservation_authority_ref=reservation_authority_ref,
        command_runner=command_runner,
        hostname=hostname,
        clock=clock,
        timeout_seconds=timeout_seconds,
        allow_existing_compute_processes=allow_existing_compute_processes,
    )


__all__ = [
    "Clock",
    "CommandRunner",
    "DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS",
    "GPU_RESERVATION_EMPTY_PROCESS_POLICY",
    "GPU_RESERVATION_EVIDENCE_SEMANTICS",
    "GPU_RESERVATION_PROVIDER_CONFIG_SCHEMA",
    "GPU_RESERVATION_PROVIDER_IDENTITY_SCHEMA",
    "GPU_RESERVATION_PROVIDER_SCOPE",
    "GPU_RESERVATION_PROVIDER_VERSION",
    "GPU_RESERVATION_SHARED_EXECUTION_POLICY",
    "GPU_RESERVATION_SHARED_EXECUTION_SEMANTICS",
    "GpuReservationProviderError",
    "NvidiaSmiGpuReservationEvidenceProvider",
    "gpu_reservation_provider_identity",
    "make_nvidia_smi_gpu_reservation_provider",
]
