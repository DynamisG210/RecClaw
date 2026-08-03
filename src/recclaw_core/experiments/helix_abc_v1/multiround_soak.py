"""Minimal immutable manifest and sealed-stage resume contract for Q4 soak.

This module is deliberately only a filesystem consumer.  It does not add a
service, database, transport, retry policy, or scientific fallback.  Physical
Provider and training stages remain owned by the existing RecClaw execution
path; this contract freezes their inputs and prevents a resumed orchestrator
from invoking a stage whose evidence has already been sealed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from .open_meta_q3 import consume_q3_active_policy


ROUND_STAGE_ORDER = (
    "MANIFEST_FROZEN",
    "PROVIDER_RESOLVER",
    "IMPLEMENTER",
    "MATERIALIZE_QUALIFIER",
    "Q0R2_RESOURCE_ADMISSION",
    "MATCHED_DEVELOPMENT_EXECUTION",
    "TYPED_RESEARCH_EPISODE",
    "AUTHORITY_CORRECT_UPDATE",
    "POLICY_ACTIVATION",
)

ACQUISITION_TASKS = ("IDEA", "EXPERIMENT", "REPLICATION")

FROZEN_EXECUTION_DIGEST_FIELDS = (
    "provider_endpoint_digest",
    "provider_model_digest",
    "data_digest",
    "config_digest",
    "denominator_rule_digest",
    "outcome_interpretation_digest",
)


class MultiRoundSoakError(RuntimeError):
    """The immutable manifest or sealed stage chain is invalid."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultiRoundSoakError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MultiRoundSoakError(f"JSON root is not an object: {path}")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(value) + b"\n")
    except FileExistsError as exc:
        raise MultiRoundSoakError(f"refusing to overwrite sealed artifact: {path}") from exc


def _write_ledger(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.next")
    if temporary.exists():
        raise MultiRoundSoakError(f"stale ledger transaction exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _digest_payload(value: Mapping[str, Any], digest_field: str) -> str:
    expected = value.get(digest_field)
    if not isinstance(expected, str):
        raise MultiRoundSoakError(f"missing {digest_field}")
    payload = {key: item for key, item in value.items() if key != digest_field}
    observed = sha256_digest(payload)
    if observed != expected:
        raise MultiRoundSoakError(
            f"{digest_field} mismatch: expected {expected}, observed {observed}"
        )
    return expected


def _validate_frozen_execution(value: Mapping[str, Any]) -> dict[str, Any]:
    frozen = canonical_value(value)
    if not isinstance(frozen, dict):
        raise MultiRoundSoakError("frozen_execution must be an object")
    missing = sorted(
        {
            *FROZEN_EXECUTION_DIGEST_FIELDS,
            "budget_seconds",
            "deadline_seconds",
            "held_out_reads",
        }
        - set(frozen)
    )
    if missing:
        raise MultiRoundSoakError(
            f"frozen_execution is missing required fields: {', '.join(missing)}"
        )
    for field in FROZEN_EXECUTION_DIGEST_FIELDS:
        try:
            validate_sha256(str(frozen[field]), field_name=field)
        except ValueError as exc:
            raise MultiRoundSoakError(str(exc)) from exc
    if frozen["held_out_reads"] != 0:
        raise MultiRoundSoakError("held-out access is forbidden")
    for field in ("budget_seconds", "deadline_seconds"):
        value = frozen[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MultiRoundSoakError(f"{field} must be a positive integer")
    return frozen


def freeze_round_manifest(
    *,
    output_path: Path,
    campaign_id: str,
    round_index: int,
    policy_path: Path,
    activation_path: Path,
    full_pool_path: Path,
    acquisition_seeds: Mapping[str, int],
    execution_task_type: str,
    frozen_execution: Mapping[str, Any],
    previous_round_activation_digest: str | None,
) -> dict[str, Any]:
    """Freeze one round from real on-disk policy, activation, and full pool."""

    if not campaign_id or any(char.isspace() for char in campaign_id):
        raise MultiRoundSoakError("campaign_id must be non-empty and whitespace-free")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 1:
        raise MultiRoundSoakError("round_index must be a positive integer")
    if set(acquisition_seeds) != set(ACQUISITION_TASKS):
        raise MultiRoundSoakError("acquisition seeds must cover IDEA/EXPERIMENT/REPLICATION")
    if execution_task_type not in ACQUISITION_TASKS:
        raise MultiRoundSoakError("unsupported execution_task_type")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in acquisition_seeds.values()
    ):
        raise MultiRoundSoakError("acquisition seeds must be integers")

    policy_path = policy_path.resolve()
    activation_path = activation_path.resolve()
    full_pool_path = full_pool_path.resolve()
    for path in (policy_path, activation_path, full_pool_path):
        if not path.is_file():
            raise MultiRoundSoakError(f"frozen input is missing: {path}")

    acquisitions = {
        task: consume_q3_active_policy(
            policy_path=policy_path,
            activation_path=activation_path,
            frozen_pool_path=full_pool_path,
            task_type=task,
            random_seed=int(acquisition_seeds[task]),
        )
        for task in ACQUISITION_TASKS
    }
    pool_digests = {value["pool_digest"] for value in acquisitions.values()}
    if len(pool_digests) != 1:
        raise MultiRoundSoakError("task acquisitions did not consume one full pool")
    if any(value["held_out_reads"] != 0 for value in acquisitions.values()):
        raise MultiRoundSoakError("acquisition attempted held-out use")

    activation = _read_object(activation_path)
    active_activation_digest = activation.get("activation_digest")
    if not isinstance(active_activation_digest, str):
        raise MultiRoundSoakError("active policy is missing activation_digest")
    if previous_round_activation_digest is not None:
        try:
            validate_sha256(
                previous_round_activation_digest,
                field_name="previous_round_activation_digest",
            )
        except ValueError as exc:
            raise MultiRoundSoakError(str(exc)) from exc
        if previous_round_activation_digest != active_activation_digest:
            raise MultiRoundSoakError(
                "round did not consume the expected previous activation"
            )

    execution = _validate_frozen_execution(frozen_execution)
    selected = acquisitions[execution_task_type]
    frozen_inputs = canonical_value(
        {
            "policy_file": str(policy_path),
            "policy_file_sha256": bytes_sha256(policy_path.read_bytes()),
            "policy_digest": selected["policy_digest"],
            "activation_file": str(activation_path),
            "activation_file_sha256": bytes_sha256(activation_path.read_bytes()),
            "activation_digest": active_activation_digest,
            "full_pool_file": str(full_pool_path),
            "full_pool_file_sha256": bytes_sha256(full_pool_path.read_bytes()),
            "full_pool_digest": next(iter(pool_digests)),
        }
    )
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-round-manifest.v1",
            "campaign_id": campaign_id,
            "round_index": round_index,
            "stage": "MANIFEST_FROZEN",
            "stage_order": ROUND_STAGE_ORDER,
            "execution_task_type": execution_task_type,
            "frozen_inputs": frozen_inputs,
            "frozen_execution": execution,
            "task_acquisitions": acquisitions,
            "selected_candidate": {
                "task_type": execution_task_type,
                "candidate_id": selected["selected_candidate_id"],
                "selection_probability": next(
                    row["selection_probability"]
                    for row in selected["candidates"]
                    if row["selected"]
                ),
                "exploration_probability": selected["exploration_probability"],
                "exploration_draw": selected["random_draw"],
                "exploration_selected": selected["exploration_selected"],
            },
            "input_activation_digest": active_activation_digest,
            "expected_previous_round_activation_digest": (
                previous_round_activation_digest
            ),
            "physical_provider_calls": 0,
            "physical_training_runs": 0,
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    manifest = {**payload, "manifest_digest": sha256_digest(payload)}
    _write_new(output_path.resolve(), manifest)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_object(path)
    _digest_payload(manifest, "manifest_digest")
    if tuple(manifest.get("stage_order", ())) != ROUND_STAGE_ORDER:
        raise MultiRoundSoakError("round stage order drift")
    if manifest.get("held_out_reads") != 0:
        raise MultiRoundSoakError("manifest held-out boundary drift")
    return manifest


def _artifact_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted((value.resolve() for value in paths), key=str):
        if not path.is_file():
            raise MultiRoundSoakError(f"stage artifact is not a file: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": bytes_sha256(path.read_bytes()),
            }
        )
    if not records:
        raise MultiRoundSoakError("a sealed stage requires at least one artifact")
    return records


def _stage_record(
    *,
    stage: str,
    status: str,
    artifacts: Sequence[Mapping[str, Any]],
    process: Mapping[str, Any],
) -> dict[str, Any]:
    if not status.startswith("SEALED_"):
        raise MultiRoundSoakError("stage status must be sealed")
    process_value = canonical_value(process)
    if not isinstance(process_value, dict) or not process_value.get("host"):
        raise MultiRoundSoakError("stage process must record its host")
    physical_call_ids = process_value.get("physical_call_ids")
    if not isinstance(physical_call_ids, list) or any(
        not isinstance(value, str) or not value for value in physical_call_ids
    ):
        raise MultiRoundSoakError("physical_call_ids must be a list of non-empty strings")
    if len(physical_call_ids) != len(set(physical_call_ids)):
        raise MultiRoundSoakError("duplicate physical_call_ids are forbidden")
    payload = canonical_value(
        {
            "stage": stage,
            "status": status,
            "artifacts": list(artifacts),
            "process": process_value,
        }
    )
    return {**payload, "stage_digest": sha256_digest(payload)}


def create_stage_ledger(*, manifest_path: Path, ledger_path: Path) -> dict[str, Any]:
    """Create the mutable ledger with only the immutable manifest stage sealed."""

    manifest_path = manifest_path.resolve()
    ledger_path = ledger_path.resolve()
    manifest = _load_manifest(manifest_path)
    manifest_stage = _stage_record(
        stage="MANIFEST_FROZEN",
        status="SEALED_SUCCESS",
        artifacts=(
            {
                "path": str(manifest_path),
                "sha256": bytes_sha256(manifest_path.read_bytes()),
            },
        ),
        process={
            "host": "local-wsl",
            "logical_call_id": None,
            "physical_call_ids": [],
            "return_code": 0,
        },
    )
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-stage-ledger.v1",
            "campaign_id": manifest["campaign_id"],
            "round_index": manifest["round_index"],
            "manifest_path": str(manifest_path),
            "manifest_digest": manifest["manifest_digest"],
            "stage_order": ROUND_STAGE_ORDER,
            "stages": [manifest_stage],
            "retries": 0,
            "held_out_reads": 0,
        }
    )
    ledger = {**payload, "ledger_digest": sha256_digest(payload)}
    _write_new(ledger_path, ledger)
    return ledger


def _load_ledger(manifest_path: Path, ledger_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    ledger_path = ledger_path.resolve()
    manifest = _load_manifest(manifest_path)
    ledger = _read_object(ledger_path)
    _digest_payload(ledger, "ledger_digest")
    if (
        ledger.get("manifest_path") != str(manifest_path)
        or ledger.get("manifest_digest") != manifest["manifest_digest"]
        or ledger.get("campaign_id") != manifest["campaign_id"]
        or ledger.get("round_index") != manifest["round_index"]
        or tuple(ledger.get("stage_order", ())) != ROUND_STAGE_ORDER
        or ledger.get("retries") != 0
        or ledger.get("held_out_reads") != 0
    ):
        raise MultiRoundSoakError("stage ledger binding drift")

    stages = ledger.get("stages")
    if not isinstance(stages, list) or not stages:
        raise MultiRoundSoakError("stage ledger is empty")
    observed_names = []
    seen_physical_calls: set[str] = set()
    for record in stages:
        if not isinstance(record, dict):
            raise MultiRoundSoakError("stage record is not an object")
        _digest_payload(record, "stage_digest")
        stage = record.get("stage")
        observed_names.append(stage)
        for artifact in record.get("artifacts", []):
            path = Path(str(artifact.get("path")))
            if not path.is_file() or bytes_sha256(path.read_bytes()) != artifact.get(
                "sha256"
            ):
                raise MultiRoundSoakError(
                    f"sealed artifact hash drift at stage {stage}: {path}"
                )
        for call_id in record.get("process", {}).get("physical_call_ids", []):
            if call_id in seen_physical_calls:
                raise MultiRoundSoakError("physical call identity was reused")
            seen_physical_calls.add(call_id)
    if observed_names != list(ROUND_STAGE_ORDER[: len(observed_names)]):
        raise MultiRoundSoakError("sealed stages are not a contiguous prefix")
    return manifest, ledger


def resume_round(*, manifest_path: Path, ledger_path: Path) -> dict[str, Any]:
    """Verify sealed evidence and identify the sole next stage, if any."""

    _manifest, ledger = _load_ledger(manifest_path, ledger_path)
    stages = ledger["stages"]
    physical_ids = [
        call_id
        for record in stages
        for call_id in record["process"]["physical_call_ids"]
    ]
    return canonical_value(
        {
            "next_stage": (
                ROUND_STAGE_ORDER[len(stages)]
                if len(stages) < len(ROUND_STAGE_ORDER)
                else None
            ),
            "sealed_stages": [record["stage"] for record in stages],
            "reuse_forbidden_physical_call_ids": physical_ids,
            "complete": len(stages) == len(ROUND_STAGE_ORDER),
            "retries": 0,
            "held_out_reads": 0,
        }
    )


def seal_stage(
    *,
    manifest_path: Path,
    ledger_path: Path,
    stage: str,
    status: str,
    artifact_paths: Sequence[Path],
    process: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal exactly the next stage; an existing sealed stage is never replaced."""

    _manifest, ledger = _load_ledger(manifest_path, ledger_path)
    sealed_names = [record["stage"] for record in ledger["stages"]]
    if stage in sealed_names:
        raise MultiRoundSoakError(f"stage already sealed: {stage}")
    if len(sealed_names) == len(ROUND_STAGE_ORDER):
        raise MultiRoundSoakError("round is already complete")
    expected = ROUND_STAGE_ORDER[len(sealed_names)]
    if stage != expected:
        raise MultiRoundSoakError(f"next stage is {expected}, not {stage}")

    existing_physical_ids = {
        call_id
        for record in ledger["stages"]
        for call_id in record["process"]["physical_call_ids"]
    }
    requested_physical_ids = process.get("physical_call_ids", [])
    if any(call_id in existing_physical_ids for call_id in requested_physical_ids):
        raise MultiRoundSoakError("physical call identity was reused")

    record = _stage_record(
        stage=stage,
        status=status,
        artifacts=_artifact_records(artifact_paths),
        process=process,
    )
    payload = canonical_value(
        {
            key: value
            for key, value in ledger.items()
            if key != "ledger_digest"
        }
    )
    payload["stages"] = [*payload["stages"], record]
    updated = {**payload, "ledger_digest": sha256_digest(payload)}
    _write_ledger(ledger_path.resolve(), updated)
    return record


__all__ = [
    "ACQUISITION_TASKS",
    "MultiRoundSoakError",
    "ROUND_STAGE_ORDER",
    "create_stage_ledger",
    "freeze_round_manifest",
    "resume_round",
    "seal_stage",
]
