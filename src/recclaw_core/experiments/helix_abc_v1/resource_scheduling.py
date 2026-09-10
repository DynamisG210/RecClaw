"""Q0R DEVELOPMENT_ONLY resource probes, model, and outcome-blind scheduling."""

from __future__ import annotations

import ast
import json
from importlib.metadata import PackageNotFoundError, version
import math
import multiprocessing as mp
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import traceback
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .canonical import bytes_sha256, canonical_value, sha256_digest, validate_sha256
from .experiment_binding import (
    COMMON_DATASET,
    ExperimentBindingError,
    ExperimentBindingV1,
    validate_execution_recipe,
)
from .fresh_r1 import (
    EXPECTED_SEARCH_FILES,
    FreshR1Error,
    GPU_RESERVATION_STATUS_MEASURED,
    GPU_WORKER_SECONDS_SEMANTICS,
    DIRECT_GPU_SELECTION_MODE,
    MAX_WORKER_CEILING_SECONDS,
    PYTHON_EXECUTABLE,
    RECBole_ROOT,
    SEARCH_DATA_ROOT,
    _initialization_resume_suffix,
    _worker_environment,
    _write_new_json,
    recbole_source_identity,
    resolve_candidate_deadline_seconds,
    run_development_training,
    validate_gpu_reservation_evidence,
)
from .innovation_recbole_adapter import snapshot_candidate_tree


ACCEPTED_Q0_COMMIT = "ec8c419aa678bca1ab7468c1ae96b822256b53f0"
ACCEPTED_Q0_PARENT = "0041d1cd4dafb3e1a1e2aced97c6a4988db72fc8"
ACCEPTED_Q0_TREE = "b63410ff78b877bde0ce3619feb1d5039500d244"
Q0_REPO_RECEIPT_SHA256 = (
    "4bd58902c18c1d55da78c5855d23d6c20dedcc84ecf465af56e96fac748f88a5"
)
Q0_EXTERNAL_RECEIPT_SHA256 = (
    "48913312fd7fce0a75115933f24b38a398e968487689b4ca5d97369b8875d49e"
)
ACCEPTED_Q0R_V1_COMMIT = "5d87d755fb8468adccf38a41f416501aca4dd1fd"
Q0R_V1_REPO_RECEIPT_SHA256 = (
    "4b6ec74891aeceac44bcd0482bcdc0f05121e26b1d7d8db98d2933b43260e133"
)
Q0R_V1_PHYSICAL_RECEIPT_SHA256 = (
    "bfaa4124c22b55fb9c990f643f2c270c72db962e038e4bceca0b32516277c3b6"
)
ACCEPTED_Q0R_V2_COMMIT = "ac423acdb10dbe210d9e1f1f0478bac88f6f66ad"
Q0R_V2_REPO_RECEIPT_SHA256 = (
    "2eac12ba13f3ecaab8d6f1f4e7debca1ea2281749728fa7a58845784918dc7b4"
)
Q0R_V2_PHYSICAL_RECEIPT_SHA256 = (
    "1b36b50066b338ea49e7e70ba0fe11ed9dac2f920a9db6956e45d69ebb89fa7b"
)
Q0R_V2_PREFIX_CONTRACT_SHA256 = (
    "c87190d7a1a0b997f2f513c8bc7605a5e9c7d2cf6ec3ad6cf3c6f2c7351e446c"
)
ACCEPTED_Q0R_V3_COMMIT = "5372da07829c92d73b530855fded95de4f57059b"
Q0R_V3_REPO_RECEIPT_SHA256 = (
    "91341e1ed6f4b324503d7e79a8b24f3c0b04482028e5e646fe00920781b791e6"
)
Q0R_RUN_IDENTITY = "q0r-resource-scheduling-v3-type-preserving-fixed-batch"
Q0R_BRANCH = "feat/research-line-resource-scheduling"
Q0R2_RUN_IDENTITY = "q0r2-first-principles-resource-admission"
Q0R_ROOT = Path(
    os.environ.get(
        "RECCLAW_Q0R_ROOT",
        "/root/projects/RecClaw_resource_scheduling_runs/q0r_v3_type_preserving",
    )
)
ARM_ORDER = (
    "matched_bpr_control",
    "parent_equivalent_null",
    "known_good_reference",
    "frontier_candidate",
)
PROBE_EPOCHS = 3
FULL_EPOCHS = 100
NATIVE_EVAL_STEP = 1
NATIVE_STOPPING_STEP = 5
# RecBole stops only when cur_step > stopping_step.  With eval_step=1, the
# earliest legal stop is the initial best epoch plus six non-improving
# epochs: seven completed epochs in total.
NATIVE_EARLY_STOP_MIN_EPOCHS = NATIVE_EVAL_STEP * (NATIVE_STOPPING_STEP + 2)
PROBE_TIMEOUT_SECONDS = 300
CAMPAIGN_TOTAL_BUDGET_SECONDS = 7200
ENGINEERING_WATCHDOG_SECONDS = MAX_WORKER_CEILING_SECONDS
GPU_MEMORY_TOTAL_MIB = 10240
CAMPAIGN_CHECKPOINT_HORIZON_SLOTS = 100
TRAINING_SEED = 54102
FIXED_TRAIN_BATCH_INDICES = tuple(range(32))
FIXED_EVAL_BATCH_INDICES = tuple(range(64))
FIXED_LOCAL_EVALUATOR_BATCH_INDICES = tuple(range(8))
PREDICTED_GPU_WORKER_SECONDS_SEMANTICS = (
    "PREDICTED_EXCLUSIVE_GPU_WORKER_RESERVATION_SECONDS"
)
SHARED_CAMPAIGN_PROBE_BUDGET_MODE = "SHARED_CAMPAIGN_PROBE_DEBIT"
OFFLINE_CALIBRATION_PROBE_BUDGET_MODE = (
    "OFFLINE_CALIBRATION_PROBE_EXCLUDED_FROM_FUTURE_FULL_RUN_BUDGET"
)
COMPLETED_PROBE_OUTER_ENVELOPE_FILENAME = "COMPLETED_PROBE_OUTER_ENVELOPE.json"


class ResourceSchedulingError(RuntimeError):
    """Q0R identity, telemetry, or scheduling failure."""


_OOM_ERROR_TYPES = frozenset(
    {"OutOfMemoryError", "torch.OutOfMemoryError"}
)
_OOM_PROCESS_MEMORY_RE = re.compile(
    r"\bProcess\s+(\d+)\s+has\s+[0-9]+(?:\.[0-9]+)?\s+"
    r"(?:MiB|GiB)\s+memory in use\b",
    flags=re.IGNORECASE,
)


def _probe_failure_scope(
    probe_run: Mapping[str, Any],
) -> str:
    """Keep a single-process OOM candidate-local unless co-residency is explicit."""

    error_type = str(
        probe_run.get("worker_error_type")
        or probe_run.get("error_type")
        or ""
    )
    message = str(
        probe_run.get("worker_error_message")
        or probe_run.get("error_message")
        or ""
    )
    is_oom = (
        error_type in _OOM_ERROR_TYPES
        or "out of memory" in message.lower()
    )
    reported_scope = probe_run.get("failure_scope")
    scope = (
        str(reported_scope)
        if reported_scope
        in {
            "CANDIDATE_LOCAL",
            "LINEAGE_COMPUTE_PATTERN",
            "WORKER_TRANSIENT",
            "SHARED_INFRASTRUCTURE",
        }
        else "CANDIDATE_LOCAL"
        if is_oom
        else "WORKER_TRANSIENT"
    )
    if not is_oom or scope == "LINEAGE_COMPUTE_PATTERN":
        return scope
    resident_process_ids = {
        match.group(1) for match in _OOM_PROCESS_MEMORY_RE.finditer(message)
    }
    if len(resident_process_ids) >= 2:
        return "SHARED_INFRASTRUCTURE"
    return "CANDIDATE_LOCAL"


def _validated_gpu_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceSchedulingError("gpu_id must be a non-negative integer")
    return int(value)


def _validate_gpu_selection_arguments(
    *,
    cuda_visible_devices: str | None,
    gpu_id: int | None,
) -> int | None:
    validated_gpu_id = _validated_gpu_id(gpu_id)
    if validated_gpu_id is not None and cuda_visible_devices is not None:
        raise ResourceSchedulingError(
            "gpu_id and cuda_visible_devices are mutually exclusive"
        )
    return validated_gpu_id


def build_fixed_batch_prefix_contract(
    *,
    seed: int = TRAINING_SEED,
    dataset: str = COMMON_DATASET,
    execution_recipe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = (
        execution_recipe.get("config")
        if isinstance(execution_recipe, Mapping)
        else None
    )
    eval_batch_indices = (
        FIXED_LOCAL_EVALUATOR_BATCH_INDICES
        if isinstance(config, Mapping)
        and config.get("recclaw_candidate_local_evaluator") is True
        else FIXED_EVAL_BATCH_INDICES
    )
    return canonical_value(
        {
            "dataset": dataset,
            "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
            "deadline_rule": {
                "engineering_watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
                "legacy_1500_seconds_controls_probe": False,
                "resource_deadline_seconds": PROBE_TIMEOUT_SECONDS,
            },
            "epochs": PROBE_EPOCHS,
            "eval_batch_indices": list(eval_batch_indices),
            "execution_purpose": "RESOURCE_PROBE_ONLY",
            "held_out_reads": 0,
            "selection_rule": (
                "preallocate the listed source-loader positions before model "
                "construction under the shared seed; reuse those exact batches "
                "for every prefix epoch and every arm; a shorter loader uses "
                "all available listed positions without repeating batches"
            ),
            "short_loader_policy": "USE_AVAILABLE_POSITIONS",
            "seed": seed,
            "schema": "recclaw.q0r-fixed-batch-prefix-contract.v1",
            "train_batch_indices": list(FIXED_TRAIN_BATCH_INDICES),
            "uniform_across_arms": True,
        }
    )


def _validate_prior_q0r_seals(repo_root: Path) -> dict[str, str]:
    paths = {
        "q0r_v1_repository": (
            repo_root
            / "docs/research_line/vnext/"
            "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
            Q0R_V1_REPO_RECEIPT_SHA256,
        ),
        "q0r_v1_physical": (
            repo_root
            / "results/research_line/q0r_resource_scheduling_20260802_01/"
            "Q0R_PHYSICAL_HARD_BLOCK_RECEIPT.json",
            Q0R_V1_PHYSICAL_RECEIPT_SHA256,
        ),
        "q0r_v2_repository": (
            repo_root
            / "docs/research_line/vnext/"
            "Q0R_FIXED_BATCH_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
            Q0R_V2_REPO_RECEIPT_SHA256,
        ),
        "q0r_v2_physical": (
            repo_root
            / "results/research_line/"
            "q0r_fixed_batch_resource_scheduling_20260802_01/"
            "Q0R_FIXED_BATCH_PHYSICAL_HARD_BLOCK_RECEIPT.json",
            Q0R_V2_PHYSICAL_RECEIPT_SHA256,
        ),
        "q0r_v2_prefix_contract": (
            repo_root
            / "results/research_line/"
            "q0r_fixed_batch_resource_scheduling_20260802_01/"
            "FIXED_BATCH_PREFIX_CONTRACT.json",
            Q0R_V2_PREFIX_CONTRACT_SHA256,
        ),
    }
    observed = {}
    for name, (path, expected) in paths.items():
        digest = bytes_sha256(path.read_bytes())
        if digest != expected:
            raise ResourceSchedulingError(f"sealed {name} byte drift")
        observed[name] = digest
    return observed


def _source_tree_digest(root: Path) -> str:
    return sha256_digest({"files": snapshot_candidate_tree(root)})


def _validate_q0_receipts(
    repo_root: Path,
    q0_external_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_path = (
        repo_root
        / "docs/research_line/vnext/Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json"
    )
    if bytes_sha256(repo_path.read_bytes()) != Q0_REPO_RECEIPT_SHA256:
        raise ResourceSchedulingError("sealed Q0 repository receipt byte drift")
    if (
        bytes_sha256(q0_external_receipt_path.read_bytes())
        != Q0_EXTERNAL_RECEIPT_SHA256
    ):
        raise ResourceSchedulingError("sealed Q0 external receipt byte drift")
    repo_receipt = json.loads(repo_path.read_text(encoding="utf-8"))
    external_receipt = json.loads(
        q0_external_receipt_path.read_text(encoding="utf-8")
    )
    if repo_receipt.get("external_receipt_sha256") != Q0_EXTERNAL_RECEIPT_SHA256:
        raise ResourceSchedulingError("Q0 receipt cross-binding drift")
    for document in (repo_receipt, external_receipt):
        if (
            document.get("held_out_reads") != 0
            or document.get("attempt_identity", {}).get("held_out_reads") != 0
            or document.get("runtime_environment_identity", {}).get("held_out_reads")
            != 0
        ):
            raise ResourceSchedulingError("Q0 held-out boundary drift")
    return repo_receipt, external_receipt


def _runtime_environment(repo_root: Path) -> dict[str, Any]:
    import torch

    release = json.loads(
        (
            repo_root
            / "src/recclaw_core/experiments/helix_abc_v1/resources/"
            "training_runtime_release_v17.json"
        ).read_text(encoding="utf-8")
    )
    backend = release["backend_identity"]
    verified_files = 0
    for row in backend["recbole_source_manifest"]:
        if row["kind"] != "FILE":
            continue
        path = RECBole_ROOT / row["path"]
        if bytes_sha256(path.read_bytes()) != row["sha256"]:
            raise ResourceSchedulingError(
                f"RecBole sealed source snapshot drift: {row['path']}"
            )
        verified_files += 1
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().split(",")
    return canonical_value(
        {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "driver_version": query[2].strip(),
            "gpu_memory_total_mib": int(query[1].strip()),
            "gpu_name": query[0].strip(),
            "held_out_reads": 0,
            "hostname": socket.gethostname(),
            "linux": platform.platform(),
            "python_executable": str(PYTHON_EXECUTABLE),
            "python_executable_sha256": bytes_sha256(PYTHON_EXECUTABLE.read_bytes()),
            "recbole_commit": backend["recbole_commit"],
            "recbole_identity_source": "BYTE_VERIFIED_TRAINING_RUNTIME_RELEASE_V17",
            "recbole_source_files_verified": verified_files,
            "recbole_tree": backend["recbole_tree"],
            "search_partition_files": {
                name: bytes_sha256((SEARCH_DATA_ROOT / "ml-1m" / name).read_bytes())
                for name in EXPECTED_SEARCH_FILES
            },
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
    )


def _validate_runtime_environment(identity: Mapping[str, Any]) -> None:
    expected = {
        "cuda_available": True,
        "cuda_device_count": 1,
        "driver_version": "550.76",
        "gpu_memory_total_mib": GPU_MEMORY_TOTAL_MIB,
        "gpu_name": "NVIDIA GeForce RTX 3080",
        "hostname": "gpu35-tingrangan",
        "recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
        "recbole_tree": "ca6386c4121ce2aae478ced7e136894ac1d7c218",
        "search_partition_files": EXPECTED_SEARCH_FILES,
        "held_out_reads": 0,
    }
    drift = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if drift:
        raise ResourceSchedulingError(
            "Q0R runtime environment drift: " + json.dumps(drift, sort_keys=True)
        )


def structural_features(source_path: Path) -> dict[str, Any]:
    """Extract small, visible source features used only for resource explanation."""

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    lowered = source.lower()
    evidence = {
        "dense_compute": (
            any(token in lowered for token in ("matmul", "einsum"))
            or any(
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)
                for node in ast.walk(tree)
            )
        ),
        "full_sort_path": "full_sort_predict" in method_names,
        "graph_propagation": any(
            token in lowered
            for token in ("propagat", "norm_adj", "get_ego_embeddings")
        ),
        "routing_path": any(
            token in lowered for token in ("router", "prototype", "route_weight")
        ),
        "sparse_compute": any(
            token in lowered for token in ("torch.sparse", "sparse.mm", "sparse_coo")
        ),
    }
    return canonical_value(
        {
            **evidence,
            "ast_call_count": sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.Call)
            ),
            "bottleneck_feature_count": sum(bool(value) for value in evidence.values()),
            "source_bytes": len(source.encode("utf-8")),
            "source_sha256": bytes_sha256(source_path.read_bytes()),
        }
    )


def _arm_inputs(
    q0_receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    baseline_source = RECBole_ROOT / "recbole/model/general_recommender/bpr.py"
    inputs: dict[str, dict[str, Any]] = {
        "matched_bpr_control": {
            "candidate_root": None,
            "entrypoint": "recbole.model.general_recommender.bpr:BPR",
            "source_path": baseline_source,
            "source_sha256": bytes_sha256(baseline_source.read_bytes()),
        }
    }
    for arm in ARM_ORDER[1:]:
        record = q0_receipt["arm_records"][arm]
        candidate_root = Path(record["candidate_root"])
        observed_tree = _source_tree_digest(candidate_root)
        if observed_tree != record["candidate_source_tree_digest"]:
            raise ResourceSchedulingError(f"sealed candidate source drift: {arm}")
        source_path = candidate_root / "recclaw_ext/candidate.py"
        inputs[arm] = {
            "candidate_package_digest": record["candidate_package_digest"],
            "candidate_root": candidate_root,
            "candidate_source_tree_digest": observed_tree,
            "entrypoint": record["entrypoint"],
            "qualification_receipt_digest": record["qualification_receipt_digest"],
            "source_path": source_path,
            "source_sha256": bytes_sha256(source_path.read_bytes()),
        }
    return inputs


def project_resource_only_evidence(
    *,
    q0_receipt: Mapping[str, Any],
    q0r_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project prior observations onto the fields allowed to inform resources.

    Q0 metrics and comparisons are intentionally unreachable from the returned
    value.  The exact package/runtime bindings come from the accepted Q0R input
    identity, while completion and censoring facts come from the accepted Q0
    physical executions.
    """

    common_runtime = q0_receipt["prefrozen_manifest"]["common_runtime"]
    early_stopping = {
        "epochs_ceiling": int(common_runtime["epochs"]),
        "evaluation_interval_epochs": 1,
        "patience_evaluations": int(common_runtime["early_stopping_patience"]),
        "semantics": "STOP_AFTER_PATIENCE_WITHOUT_VALIDATION_IMPROVEMENT",
    }
    if early_stopping != {
        "epochs_ceiling": 100,
        "evaluation_interval_epochs": 1,
        "patience_evaluations": 10,
        "semantics": "STOP_AFTER_PATIENCE_WITHOUT_VALIDATION_IMPROVEMENT",
    }:
        raise ResourceSchedulingError("Q0 early-stopping contract drift")
    q0_runtime = q0_receipt["runtime_environment_identity"]
    q0r_runtime = q0r_receipt["runtime_environment_identity"]
    runtime_identity = {
        "gpu_name": q0_runtime["gpu_name"],
        "gpu_memory_total_mib": q0_runtime["gpu_memory_total_mib"],
        "python_executable_sha256": q0_runtime["python_executable_sha256"],
        "recbole_commit": q0_runtime["recbole_commit"],
        "recbole_tree": q0_runtime["recbole_tree"],
        "search_partition_files": q0_runtime["search_partition_files"],
    }
    if any(q0r_runtime.get(key) != value for key, value in runtime_identity.items()):
        raise ResourceSchedulingError("Q0/Q0R runtime identity mismatch")

    q0r_inputs = q0r_receipt["input_identity"]["candidate_inputs"]
    q0_timeout_ms = int(common_runtime["timeout_seconds_per_run"]) * 1000
    projected: dict[str, Any] = {}
    for arm in ARM_ORDER:
        if arm == "matched_bpr_control":
            training = q0_receipt["evaluation"]["matched_bpr_control"]
            expected_source = q0_runtime["recbole_bpr_source_sha256"]
            if q0r_inputs[arm]["source_sha256"] != expected_source:
                raise ResourceSchedulingError("matched BPR package identity mismatch")
        else:
            q0_arm = q0_receipt["arm_records"][arm]
            training = q0_arm["training_run"]
            for key in ("candidate_package_digest", "candidate_source_tree_digest"):
                if q0r_inputs[arm][key] != q0_arm[key]:
                    raise ResourceSchedulingError(f"Q0/Q0R package mismatch: {arm}")
        if training.get("exit_status") == "SUCCESS":
            observation_status = "SUCCESS"
            censor_bound_ms = None
            completion_semantics = "OBSERVED_EARLY_STOPPING_AWARE_COMPLETION"
        elif (
            training.get("launcher_return_code") == 124
            and int(training.get("wall_time_ms", 0)) >= q0_timeout_ms
        ):
            observation_status = "RIGHT_CENSORED"
            censor_bound_ms = q0_timeout_ms
            completion_semantics = "NOT_OBSERVED_BEYOND_CENSOR_BOUND"
        else:
            raise ResourceSchedulingError(f"Q0 resource status unsupported: {arm}")

        probe = q0r_receipt["probe_runs"][arm]
        telemetry = probe.get("resource_telemetry") or {}
        batches = [
            row
            for row in telemetry.get("batch_records", ())
            if row.get("status") == "BATCH_COMPLETED"
            and row.get("phase") in {"TRAIN", "EVAL"}
            and isinstance(row.get("wall_time_ms"), (int, float))
        ]
        phase_cost_ms = {}
        for phase in ("TRAIN", "EVAL"):
            values = [float(row["wall_time_ms"]) for row in batches if row["phase"] == phase]
            phase_cost_ms[phase.lower()] = (
                sum(values) / len(values) if values else None
            )
        projected[arm] = {
            "arm": arm,
            "early_stopping_contract": early_stopping,
            "mechanism_effect_update_allowed": False,
            "package_identity": {
                key: value
                for key, value in q0r_inputs[arm].items()
                if key in {
                    "candidate_package_digest",
                    "candidate_source_tree_digest",
                    "entrypoint",
                    "source_sha256",
                }
            },
            "prefix_resource_observation": {
                "completed_eval_batches": sum(row["phase"] == "EVAL" for row in batches),
                "completed_train_batches": sum(row["phase"] == "TRAIN" for row in batches),
                "exit_status": probe["exit_status"],
                "mean_batch_wall_time_ms_by_phase": phase_cost_ms,
                "peak_gpu_memory_mib": telemetry.get("peak_gpu_memory_mib"),
                "wall_time_ms": probe["wall_time_ms"],
            },
            "prior_completion_observation": {
                "censor_bound_ms": censor_bound_ms,
                "completion_semantics": completion_semantics,
                "status": observation_status,
                "wall_time_ms": int(training["wall_time_ms"]),
            },
            "recipe_identity": {
                "dataset_partition": common_runtime["dataset_partition"],
                "epochs_ceiling": int(training["epochs_requested"]),
                "execution_recipe_digest": training["execution_recipe_digest"],
                "runtime_binding_digest": training["runtime_binding_digest"],
                "runtime_release_digest": training["runtime_release_digest"],
                "seed": int(training["seed"]),
            },
            "runtime_identity": runtime_identity,
        }
    return {
        "arms": projected,
        "effect_fields_consumed": [],
        "held_out_reads": 0,
        "projection_rule": "EXPLICIT_RESOURCE_ONLY_ALLOWLIST",
        "schema": "recclaw.q0r2-resource-only-evidence.v1",
    }


def admit_resource_only_evidence(
    evidence: Mapping[str, Any],
    *,
    total_budget_seconds: int = CAMPAIGN_TOTAL_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Admit the empirically completable subset without arm-specific rules."""

    eligible: list[tuple[float, str, str, int]] = []
    deferred: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    for arm, row in evidence["arms"].items():
        if row.get("mechanism_effect_update_allowed") is not False:
            raise ResourceSchedulingError("resource evidence may not update effect")
        observation = row["prior_completion_observation"]
        identity_digest = sha256_digest(
            {
                "package": row["package_identity"],
                "recipe": row["recipe_identity"],
                "runtime": row["runtime_identity"],
            }
        )
        prediction = {
            "completion_basis": None,
            "completion_probability": 0.0,
            "empirical_completion_seconds": None,
            "identity_digest": identity_digest,
            "mechanism_effect_update_allowed": False,
            "right_censored_lower_bound_seconds": (
                observation["censor_bound_ms"] / 1000
                if observation["censor_bound_ms"] is not None
                else None
            ),
        }
        if observation["status"] == "SUCCESS":
            empirical_seconds = observation["wall_time_ms"] / 1000
            deadline = max(300, math.ceil(60 + 2.0 * empirical_seconds))
            prediction.update(
                {
                    "completion_basis": "EXACT_IDENTITY_EMPIRICAL_SUCCESS",
                    "completion_probability": 0.90,
                    "empirical_completion_seconds": empirical_seconds,
                }
            )
            eligible.append((empirical_seconds, identity_digest, arm, deadline))
        else:
            prediction["completion_basis"] = "RIGHT_CENSORED_NO_COMPLETION_TIME"
            deferred.append(
                {
                    "arm": arm,
                    "future_eligible": True,
                    "mechanism_effect_update_allowed": False,
                    "reason": "RIGHT_CENSORED_WITHOUT_EXACT_IDENTITY_COMPLETION",
                    "resource_disposition": "RESOURCE_DEFERRED",
                }
            )
        predictions[arm] = prediction

    schedule = []
    allocated = 0
    for _seconds, _identity, arm, deadline in sorted(eligible):
        if allocated + deadline > total_budget_seconds:
            predictions[arm]["completion_probability"] = 0.0
            deferred.append(
                {
                    "arm": arm,
                    "future_eligible": True,
                    "mechanism_effect_update_allowed": False,
                    "reason": "EMPIRICAL_DEADLINE_EXCEEDS_REMAINING_CAMPAIGN_BUDGET",
                    "resource_disposition": "RESOURCE_DEFERRED",
                }
            )
            continue
        allocated += deadline
        schedule.append(
            {
                "arm": arm,
                "deadline_seconds": deadline,
                "ordinal": len(schedule) + 1,
            }
        )
    return canonical_value(
        {
            "campaign_total_budget_seconds": total_budget_seconds,
            "deadline_formula": (
                "for an exact package/recipe/runtime SUCCESS under the same "
                "early-stopping contract, max(300, ceil(60 + 2 * empirical "
                "completion)); right-censored observations remain future-eligible "
                "but are not admitted without an observed completion"
            ),
            "deferred_arms": deferred,
            "effect_fields_consumed": [],
            "engineering_watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
            "predictions": predictions,
            "schedule": schedule,
            "schedule_rule": (
                "ascending empirical completion time with exact identity digest tie-break"
            ),
            "schema": "recclaw.q0r2-first-principles-admission.v1",
        }
    )


def predict_resources(
    *,
    arm_features: Mapping[str, Mapping[str, Any]],
    probe_runs: Mapping[str, Mapping[str, Any]],
    total_budget_seconds: int = CAMPAIGN_TOTAL_BUDGET_SECONDS,
    execution_recipe: Mapping[str, Any] | None = None,
    arm_order: Sequence[str] = ARM_ORDER,
    probe_seed: int = TRAINING_SEED,
    offline_calibration_probe_accounting: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Fit the fixed-batch auditable model and allocate one campaign budget.

    By default, probe wall time is debited from the shared campaign budget.
    The fixed-66 offline calibration path may pass bounded accounting evidence
    produced after sealed reservation validation; that path re-credits only the
    measured probe wall interval to the caller's future full-run budget.
    """

    order = tuple(arm_order)
    if not order or len(set(order)) != len(order):
        raise ResourceSchedulingError("arm order must be non-empty and unique")
    if set(arm_features) != set(order) or set(probe_runs) != set(order):
        raise ResourceSchedulingError("resource inputs do not match frozen arm order")
    resource_parameters = _execution_resource_parameters(
        execution_recipe,
        total_budget_seconds=total_budget_seconds,
    )
    execution_config = (
        execution_recipe.get("config")
        if isinstance(execution_recipe, Mapping)
        else None
    )
    from recclaw_core.research_line.p4_runtime import is_p4_recipe, p4_fit_mode
    p4_precompute = (
        isinstance(execution_recipe, Mapping) and is_p4_recipe(execution_recipe)
        and p4_fit_mode(execution_config) == "TRAIN_ONLY_PRECOMPUTE"
    )
    uses_candidate_local_evaluator = (
        isinstance(execution_config, Mapping)
        and execution_config.get("recclaw_candidate_local_evaluator") is True
    )
    requested_epochs = resource_parameters["requested_epochs"]
    native_eval_step = resource_parameters["native_eval_step"]
    native_stopping_step = resource_parameters["native_stopping_step"]
    worker_ceiling_seconds = resource_parameters["worker_ceiling_seconds"]
    efficiency_envelope = resource_parameters["efficiency_envelope"]
    profile_owned = resource_parameters["profile_owned"]
    native_early_stop_min_epochs = min(
        requested_epochs,
        native_eval_step * (native_stopping_step + 2),
    )
    full_eval_events = math.ceil(requested_epochs / native_eval_step)
    native_eval_events = math.ceil(
        native_early_stop_min_epochs / native_eval_step
    )
    if (
        offline_calibration_probe_accounting is not None
        and not isinstance(offline_calibration_probe_accounting, Mapping)
    ):
        raise ResourceSchedulingError(
            "offline calibration accounting must be capability keyed"
        )
    predictions: dict[str, dict[str, Any]] = {}
    requested_deadlines: dict[str, int] = {}
    deferred: dict[str, dict[str, Any]] = {}
    offline_accounting_evidence: dict[str, dict[str, Any]] = {}
    if offline_calibration_probe_accounting is not None:
        if set(offline_calibration_probe_accounting) != set(order):
            raise ResourceSchedulingError(
                "offline calibration accounting does not match frozen arm order"
            )
        for arm in order:
            accounting = offline_calibration_probe_accounting[arm]
            if not isinstance(accounting, Mapping):
                raise ResourceSchedulingError(
                    f"offline calibration accounting is unavailable: {arm}"
                )
            if accounting.get("mode") != OFFLINE_CALIBRATION_PROBE_BUDGET_MODE:
                raise ResourceSchedulingError(
                    f"offline calibration accounting mode is invalid: {arm}"
                )
            probe_wall_time_ms = _finite_resource_number(
                probe_runs[arm].get("wall_time_ms"),
                field_name=f"{arm}.probe_run.wall_time_ms",
                positive=True,
            )
            accounted_wall_time_ms = _finite_resource_number(
                accounting.get("probe_wall_time_ms"),
                field_name=f"{arm}.offline_accounting.probe_wall_time_ms",
                positive=True,
            )
            parent_interval_wall_time_ms = _finite_resource_number(
                accounting.get("parent_process_interval_wall_time_ms"),
                field_name=(
                    f"{arm}.offline_accounting."
                    "parent_process_interval_wall_time_ms"
                ),
                positive=True,
            )
            if not math.isclose(
                accounted_wall_time_ms,
                probe_wall_time_ms,
                rel_tol=0.0,
                abs_tol=1.0,
            ) or not math.isclose(
                parent_interval_wall_time_ms,
                probe_wall_time_ms,
                rel_tol=0.0,
                abs_tol=1.0,
            ):
                raise ResourceSchedulingError(
                    f"offline calibration probe wall evidence does not match: {arm}"
                )
            try:
                reservation_digest = validate_sha256(
                    accounting.get("reservation_digest"),
                    field_name=f"{arm}.offline_accounting.reservation_digest",
                )
                reservation_identity_digest = validate_sha256(
                    accounting.get("reservation_identity_digest"),
                    field_name=(
                        f"{arm}.offline_accounting.reservation_identity_digest"
                    ),
                )
            except (TypeError, ValueError) as error:
                raise ResourceSchedulingError(
                    f"offline calibration reservation provenance is invalid: {arm}"
                ) from error
            offline_accounting_evidence[arm] = {
                "parent_process_interval_wall_time_ms": (
                    parent_interval_wall_time_ms
                ),
                "probe_wall_time_ms": probe_wall_time_ms,
                "reservation_digest": reservation_digest,
                "reservation_identity_digest": reservation_identity_digest,
            }
    probe_cost_seconds = math.ceil(
        sum(int(probe_runs[arm]["wall_time_ms"]) for arm in order) / 1000
    )
    probe_cost_recredited_seconds = (
        probe_cost_seconds
        if offline_calibration_probe_accounting is not None and not profile_owned
        else 0
    )
    probe_cost_outside_worker_ceiling_seconds = (
        probe_cost_seconds if profile_owned else 0
    )
    effective_total_budget_seconds = (
        total_budget_seconds
        + probe_cost_recredited_seconds
        + probe_cost_outside_worker_ceiling_seconds
    )
    full_run_budget_seconds = effective_total_budget_seconds - probe_cost_seconds
    if full_run_budget_seconds <= 0:
        raise ResourceSchedulingError("prefix probes exhausted campaign budget")
    budget_accounting = canonical_value(
        {
            "basis": (
                "validated sealed parent-process interval wall time"
                if offline_calibration_probe_accounting is not None
                else "probe_run.wall_time_ms"
            ),
            "caller_total_budget_seconds": total_budget_seconds,
            "effective_total_budget_seconds": effective_total_budget_seconds,
            "full_run_budget_after_probes_seconds": full_run_budget_seconds,
            "mode": (
                OFFLINE_CALIBRATION_PROBE_BUDGET_MODE
                if offline_calibration_probe_accounting is not None
                else SHARED_CAMPAIGN_PROBE_BUDGET_MODE
            ),
            "probe_cost_recredited_seconds": probe_cost_recredited_seconds,
            "probe_cost_outside_worker_ceiling_seconds": (
                probe_cost_outside_worker_ceiling_seconds
            ),
            "probe_cost_seconds": probe_cost_seconds,
            "probe_wall_time_ms": {
                arm: probe_runs[arm]["wall_time_ms"] for arm in order
            },
            "sealed_probe_evidence": offline_accounting_evidence,
        }
    )
    contract_digests: set[str] = set()
    contract_shapes: set[
        tuple[int, tuple[int, ...], tuple[int, ...]]
    ] = set()
    for arm in order:
        run = probe_runs[arm]
        device_memory_mib = _selected_device_capacity_mib(run)
        telemetry = run.get("resource_telemetry")
        if run.get("exit_status") not in {"SUCCESS", "RESOURCE_CENSORED"} or not isinstance(
            telemetry, Mapping
        ):
            raise ResourceSchedulingError(f"real prefix telemetry unavailable: {arm}")
        contract = telemetry.get("prefix_contract")
        if not isinstance(contract, Mapping):
            raise ResourceSchedulingError(f"fixed-batch contract binding missing: {arm}")
        contract_digest = contract.get("contract_file_sha256")
        if not isinstance(contract_digest, str):
            raise ResourceSchedulingError(f"fixed-batch contract digest missing: {arm}")
        contract_digests.add(contract_digest)
        contract_epochs = contract.get("epochs", PROBE_EPOCHS)
        contract_train_batch_indices = contract.get(
            "train_batch_indices", FIXED_TRAIN_BATCH_INDICES
        )
        contract_eval_batch_indices = contract.get(
            "eval_batch_indices", FIXED_EVAL_BATCH_INDICES
        )
        if (
            not isinstance(contract_epochs, int)
            or contract_epochs <= 0
            or not isinstance(contract_train_batch_indices, (list, tuple))
            or not contract_train_batch_indices
            or not isinstance(contract_eval_batch_indices, (list, tuple))
            or not contract_eval_batch_indices
        ):
            raise ResourceSchedulingError(
                f"fixed-batch contract shape is unavailable: {arm}"
            )
        contract_train_batch_indices = tuple(contract_train_batch_indices)
        contract_eval_batch_indices = tuple(contract_eval_batch_indices)
        contract_shapes.add(
            (
                contract_epochs,
                contract_train_batch_indices,
                contract_eval_batch_indices,
            )
        )
        batches = [
            row
            for row in telemetry.get("batch_records", ())
            if row.get("status") == "BATCH_COMPLETED"
            and isinstance(row.get("wall_time_ms"), (int, float))
        ]
        train_rows = [row for row in batches if row.get("phase") == "TRAIN"]
        eval_rows = [row for row in batches if row.get("phase") == "EVAL"]
        if p4_precompute:
            prediction, failure = _predict_p4_precomputed_operator(
                run, telemetry, eval_rows, train_rows,
                features=arm_features[arm], device_memory_mib=device_memory_mib,
                requested_epochs=requested_epochs,
                worker_ceiling_seconds=worker_ceiling_seconds,
                full_run_budget_seconds=full_run_budget_seconds,
            )
            # Preserve the requested recipe cadence for downstream identity
            # checks; static fitting still performs zero optimization epochs.
            predictions[arm] = {
                **prediction,
                "native_early_stop_eval_step": native_eval_step,
                "native_early_stop_stopping_step": native_stopping_step,
            }
            if failure is None:
                requested_deadlines[arm] = worker_ceiling_seconds
            else:
                deferred[arm] = {"reason": failure, "resource_disposition": "RESOURCE_DEFERRED"}
            continue
        if not train_rows:
            raise ResourceSchedulingError(
                f"fixed-batch prefix has no completed train consumer signal: {arm}"
            )
        if not any(row.get("loss") is not None for row in train_rows):
            raise ResourceSchedulingError(f"fixed-batch loss trend unavailable: {arm}")
        full_train_batches = telemetry.get("full_train_batches_per_epoch")
        full_eval_batches = telemetry.get("full_validation_batches_per_eval")
        setup_ms = telemetry.get("initialization_wall_time_ms")
        if not all(
            isinstance(value, int) and value > 0
            for value in (full_train_batches, full_eval_batches, setup_ms)
        ):
            raise ResourceSchedulingError(f"fixed-batch scale inputs unavailable: {arm}")
        train_ms = [float(row["wall_time_ms"]) for row in train_rows]
        eval_ms = [float(row["wall_time_ms"]) for row in eval_rows]
        # Some backends compile kernels on the first training batch of a fresh
        # process.  That process-local cost happens once; multiplying it by
        # every batch in every epoch can reject an otherwise parent-speed
        # implementation.  Compare the first batch only with the same fixed
        # batch in later probe epochs, retain its normal repeated cost, and
        # charge only the observed first-epoch excess once.
        train_warmup_excess_ms = 0.0
        adjusted_train_ms = list(train_ms)
        if len(train_rows) > 1:
            first_train_row = train_rows[0]
            first_position = first_train_row.get("batch_position")
            first_source_index = first_train_row.get("source_batch_index")
            first_epoch = first_train_row.get("epoch")
            later_same_batch_ms = (
                [
                    float(row["wall_time_ms"])
                    for row in train_rows[1:]
                    if row.get("batch_position") == first_position
                    and row.get("source_batch_index") == first_source_index
                    and row.get("epoch") != first_epoch
                ]
                if all(
                    isinstance(value, int)
                    for value in (first_position, first_source_index, first_epoch)
                )
                else []
            )
            if later_same_batch_ms:
                repeated_first_batch_ms = sum(later_same_batch_ms) / len(
                    later_same_batch_ms
                )
                train_warmup_excess_ms = max(
                    0.0,
                    train_ms[0] - repeated_first_batch_ms,
                )
                adjusted_train_ms[0] -= train_warmup_excess_ms
        eval_warmup_excess_ms = 0.0
        adjusted_eval_ms = list(eval_ms)
        # Only candidate-local evaluators use the bounded repeated eval sample
        # whose first observation includes one-time generator warmup.  Legacy
        # 64-batch probes retain their established all-observation projection.
        if uses_candidate_local_evaluator and len(eval_rows) > 1:
            first_eval_row = eval_rows[0]
            first_eval_position = first_eval_row.get("batch_position")
            first_eval_source_index = first_eval_row.get("source_batch_index")
            first_eval_epoch = first_eval_row.get("epoch")
            later_same_eval_batch_ms = (
                [
                    float(row["wall_time_ms"])
                    for row in eval_rows[1:]
                    if row.get("batch_position") == first_eval_position
                    and row.get("source_batch_index")
                    == first_eval_source_index
                    and row.get("epoch") != first_eval_epoch
                ]
                if all(
                    isinstance(value, int)
                    for value in (
                        first_eval_position,
                        first_eval_source_index,
                        first_eval_epoch,
                    )
                )
                else []
            )
            if later_same_eval_batch_ms:
                repeated_first_eval_batch_ms = sum(
                    later_same_eval_batch_ms
                ) / len(later_same_eval_batch_ms)
                eval_warmup_excess_ms = max(
                    0.0,
                    eval_ms[0] - repeated_first_eval_batch_ms,
                )
                adjusted_eval_ms[0] -= eval_warmup_excess_ms
        train_point_epoch_ms = (
            sum(adjusted_train_ms)
            / len(adjusted_train_ms)
            * int(full_train_batches)
        )
        train_lower_epoch_ms = min(adjusted_train_ms) * int(full_train_batches)
        train_upper_epoch_ms = max(adjusted_train_ms) * int(full_train_batches)
        checkpoint_footprint = run.get("checkpoint_footprint")
        storage_observation = run.get("storage_observation")
        checkpoint_bytes = (
            checkpoint_footprint.get("bytes")
            if isinstance(checkpoint_footprint, Mapping)
            else None
        )
        storage_available_bytes = (
            storage_observation.get("available_after_probe_cleanup_bytes")
            if isinstance(storage_observation, Mapping)
            else None
        )
        campaign_checkpoint_horizon_slots = (
            storage_observation.get(
                "campaign_checkpoint_horizon_slots",
                CAMPAIGN_CHECKPOINT_HORIZON_SLOTS,
            )
            if isinstance(storage_observation, Mapping)
            else CAMPAIGN_CHECKPOINT_HORIZON_SLOTS
        )
        if (
            isinstance(campaign_checkpoint_horizon_slots, bool)
            or not isinstance(campaign_checkpoint_horizon_slots, int)
            or campaign_checkpoint_horizon_slots <= 0
        ):
            raise ResourceSchedulingError(
                f"campaign checkpoint horizon is invalid: {arm}"
            )
        projected_campaign_checkpoint_bytes = (
            checkpoint_bytes * campaign_checkpoint_horizon_slots
            if isinstance(checkpoint_bytes, int) and checkpoint_bytes > 0
            else None
        )
        storage_safe = (
            projected_campaign_checkpoint_bytes is None
            or (
                isinstance(storage_available_bytes, int)
                and storage_available_bytes > 0
                and projected_campaign_checkpoint_bytes <= storage_available_bytes
            )
        )
        combined_features = canonical_value(
            {
                **arm_features[arm],
                "campaign_checkpoint_horizon_slots": (
                    campaign_checkpoint_horizon_slots
                ),
                "checkpoint_storage_bytes": checkpoint_bytes,
                "parameter_count": telemetry.get("parameter_count"),
                "projected_campaign_checkpoint_bytes": (
                    projected_campaign_checkpoint_bytes
                ),
                "storage_available_after_probe_cleanup_bytes": (
                    storage_available_bytes
                ),
                "trainable_parameter_count": telemetry.get(
                    "trainable_parameter_count"
                ),
            }
        )
        complexity_count = int(combined_features["bottleneck_feature_count"])
        peak_observed = max(
            float(row[key])
            for row in batches
            for key in ("peak_allocated_mib", "peak_reserved_mib")
            if isinstance(row.get(key), (int, float))
        )
        peak_prediction = peak_observed * (1.0 + 0.03 * complexity_count)
        full_training_lower_bound_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + train_lower_epoch_ms * requested_epochs
        )
        native_training_lower_bound_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + train_lower_epoch_ms * native_early_stop_min_epochs
        )
        memory_safe = peak_prediction < device_memory_mib
        if not memory_safe:
            deferred[arm] = {
                "reason": "PREDICTED_GPU_MEMORY_RESERVE_ENVELOPE",
                "resource_disposition": "RESOURCE_INFEASIBLE",
            }
        elif not storage_safe:
            deferred[arm] = {
                "reason": "PROJECTED_CAMPAIGN_CHECKPOINT_STORAGE",
                "resource_disposition": "RESOURCE_INFEASIBLE",
            }
        elif not eval_rows:
            raise ResourceSchedulingError(
                f"fixed-batch prefix eval signal unavailable for feasible arm: {arm}"
            )

        if eval_rows:
            eval_point_epoch_ms = (
                sum(adjusted_eval_ms)
                / len(adjusted_eval_ms)
                * int(full_eval_batches)
            )
            eval_lower_epoch_ms = min(adjusted_eval_ms) * int(full_eval_batches)
            eval_upper_epoch_ms = max(adjusted_eval_ms) * int(full_eval_batches)
            estimate_scope = "TRAIN_AND_FULL_SORT_EVAL"
        else:
            eval_point_epoch_ms = 0.0
            eval_lower_epoch_ms = 0.0
            eval_upper_epoch_ms = 0.0
            estimate_scope = "TRAINING_ONLY_RESOURCE_LOWER_BOUND"
        profiled_probe_ms = (
            int(setup_ms) + sum(train_ms) + sum(eval_ms)
        )
        unprofiled_probe_ms = max(
            0.0,
            float(run["wall_time_ms"]) - float(profiled_probe_ms),
        )
        unprofiled_epoch_ms = unprofiled_probe_ms / contract_epochs
        # Fixed-prefix batch timings omit epoch hooks and other whole-epoch
        # work (for example sampler refresh or graph construction).  Attribute
        # the measured residual to the same probe epochs before extrapolating;
        # otherwise admission compares a different execution path from the
        # full worker it is meant to predict.
        train_point_with_hooks_ms = train_point_epoch_ms + unprofiled_epoch_ms
        train_lower_with_hooks_ms = train_lower_epoch_ms + unprofiled_epoch_ms
        train_upper_with_hooks_ms = train_upper_epoch_ms + unprofiled_epoch_ms
        full_point_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + eval_warmup_excess_ms
            + train_point_with_hooks_ms * requested_epochs
            + eval_point_epoch_ms * full_eval_events
        )
        full_lower_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + eval_warmup_excess_ms
            + train_lower_with_hooks_ms * requested_epochs
            + eval_lower_epoch_ms * full_eval_events
        )
        full_upper_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + eval_warmup_excess_ms
            + (
                train_upper_with_hooks_ms * requested_epochs
                + eval_upper_epoch_ms * full_eval_events
            )
            * (1.0 + 0.05 * complexity_count)
        )
        native_point_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + eval_warmup_excess_ms
            + train_point_with_hooks_ms * native_early_stop_min_epochs
            + eval_point_epoch_ms * native_eval_events
        )
        native_lower_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + eval_warmup_excess_ms
            + train_lower_with_hooks_ms * native_early_stop_min_epochs
            + eval_lower_epoch_ms * native_eval_events
        )
        native_upper_ms = (
            int(setup_ms)
            + train_warmup_excess_ms
            + eval_warmup_excess_ms
            + (
                train_upper_with_hooks_ms * native_early_stop_min_epochs
                + eval_upper_epoch_ms * native_eval_events
            )
            * (1.0 + 0.05 * complexity_count)
        )
        total_point_without_setup_ms = max(0.0, full_point_ms - int(setup_ms))
        eval_share = (
            eval_point_epoch_ms * full_eval_events / total_point_without_setup_ms
            if total_point_without_setup_ms > 0.0
            else 0.0
        )
        native_budget_required_seconds = max(
            180,
            math.ceil(1.10 * native_point_ms / 1000),
        )
        native_budget_limit_seconds = min(
            worker_ceiling_seconds,
            full_run_budget_seconds,
        )
        native_budget_exceeded = (
            native_budget_required_seconds > native_budget_limit_seconds
        )
        full_requested_epoch_estimate_exceeds_ceiling = (
            full_point_ms / 1000 > worker_ceiling_seconds
        )
        if arm not in deferred and native_budget_exceeded:
            deferred[arm] = {
                "reason": "NATIVE_EARLY_STOP_WINDOW_EXCEEDS_EXECUTION_BUDGET",
                "resource_disposition": "RESOURCE_DEFERRED",
            }
        if peak_prediction >= device_memory_mib * 0.90:
            bottleneck = "GPU_MEMORY_CAPACITY"
        elif eval_share >= 0.60:
            bottleneck = "FULL_SORT_EVALUATION"
        elif arm_features[arm]["routing_path"]:
            bottleneck = "ROUTING_DENSE_COMPUTE"
        elif arm_features[arm]["graph_propagation"]:
            bottleneck = "GRAPH_PROPAGATION"
        else:
            bottleneck = "TRAINING_THROUGHPUT"
        observed_batches = len(train_rows) + len(eval_rows)
        expected_batches = contract_epochs * (
            len(contract.get("effective_train_batch_indices", contract_train_batch_indices))
            + len(contract.get("effective_eval_batch_indices", contract_eval_batch_indices))
        )
        confidence = (
            "MEDIUM"
            if observed_batches == expected_batches and run.get("exit_status") == "SUCCESS"
            else "LOW"
        )
        estimated_epoch_wall_time_ms = (
            total_point_without_setup_ms / requested_epochs
        )
        efficiency_envelope_exceeded = bool(
            isinstance(efficiency_envelope, Mapping)
            and estimated_epoch_wall_time_ms
            > float(efficiency_envelope["maximum_epoch_wall_time_ms"])
        )
        if arm not in deferred and efficiency_envelope_exceeded:
            deferred[arm] = {
                "reason": "PROFILE_IMPLEMENTATION_EFFICIENCY_ENVELOPE",
                "resource_disposition": "RESOURCE_DEFERRED",
            }
        requested_deadline_seconds = None
        if arm not in deferred:
            requested_deadline_seconds = (
                worker_ceiling_seconds
                if profile_owned
                else min(
                    worker_ceiling_seconds,
                    max(180, math.ceil(1.10 * full_point_ms / 1000)),
                )
            )
            requested_deadlines[arm] = requested_deadline_seconds
        predictions[arm] = canonical_value(
            {
                "bottleneck_category": bottleneck,
                "confidence": confidence,
                "completed_eval_batches": len(eval_rows),
                "completed_train_batches": len(train_rows),
                "estimated_total_wall_time_seconds": native_point_ms / 1000,
                "estimated_epoch_wall_time_ms": estimated_epoch_wall_time_ms,
                "estimate_scope": "NATIVE_EARLY_STOP_MINIMUM_WINDOW",
                "first_progress_deadline_seconds": min(
                    worker_ceiling_seconds,
                    max(
                        30,
                        math.ceil(
                            2.0
                            * native_point_ms
                            / native_early_stop_min_epochs
                            / 1000
                        ),
                    ),
                ),
                "features": combined_features,
                "fixed_batch_eval_mean_wall_time_ms": (
                    sum(eval_ms) / len(eval_ms) if eval_ms else None
                ),
                "fixed_batch_eval_steady_mean_wall_time_ms": (
                    sum(adjusted_eval_ms) / len(adjusted_eval_ms)
                    if adjusted_eval_ms
                    else None
                ),
                "fixed_batch_eval_first_epoch_warmup_excess_ms": (
                    eval_warmup_excess_ms
                ),
                "fixed_batch_eval_sample_batches_per_event": len(
                    contract_eval_batch_indices
                ),
                "fixed_batch_train_mean_wall_time_ms": sum(train_ms) / len(train_ms),
                "fixed_batch_train_steady_mean_wall_time_ms": (
                    sum(adjusted_train_ms) / len(adjusted_train_ms)
                ),
                "fixed_batch_train_first_epoch_warmup_excess_ms": (
                    train_warmup_excess_ms
                ),
                "full_eval_batches_per_epoch": full_eval_batches,
                "full_eval_event_estimated_wall_time_ms": eval_point_epoch_ms,
                "full_train_batches_per_epoch": full_train_batches,
                "full_requested_epoch_estimate_scope": estimate_scope,
                "full_requested_epoch_estimated_wall_time_seconds": (
                    full_point_ms / 1000
                ),
                "full_requested_epoch_estimate_exceeds_ceiling": (
                    full_requested_epoch_estimate_exceeds_ceiling
                ),
                "full_requested_epoch_estimate_semantics": (
                    "NON_ADMISSION_COST_RISK_ONLY; REQUESTED_EPOCHS_IS_AN_UPPER_"
                    "BOUND_AND_NATIVE_STOP_REMAINS_AUTHORITATIVE"
                ),
                "full_requested_epoch_prediction_interval_seconds": [
                    full_lower_ms / 1000,
                    full_upper_ms / 1000,
                ],
                "full_requested_epoch_training_only_lower_bound_seconds": (
                    full_training_lower_bound_ms / 1000
                ),
                "model": "FIXED_BATCH_THROUGHPUT_NATIVE_EARLY_STOP_EXTRAPOLATION_V5",
                "native_early_stop_budget_exceeded": native_budget_exceeded,
                "native_early_stop_budget_required_seconds": (
                    native_budget_required_seconds
                ),
                "native_early_stop_eval_step": native_eval_step,
                "native_early_stop_min_epochs": native_early_stop_min_epochs,
                "native_early_stop_stopping_step": native_stopping_step,
                "unprofiled_epoch_wall_time_ms": unprofiled_epoch_ms,
                "unprofiled_probe_wall_time_ms": unprofiled_probe_ms,
                "unprofiled_probe_wall_time_semantics": (
                    "MEASURED_PROBE_WALL_TIME_MINUS_SETUP_AND_PROFILED_BATCHES; "
                    "ATTRIBUTED_EQUALLY_ACROSS_PROBE_EPOCHS"
                ),
                "peak_memory_candidate_ceiling_mib": device_memory_mib,
                "peak_memory_prediction_mib": peak_prediction,
                "peak_memory_observed_mib": peak_observed,
                "checkpoint_storage_bytes": checkpoint_bytes,
                "projected_campaign_checkpoint_bytes": (
                    projected_campaign_checkpoint_bytes
                ),
                "storage_available_after_probe_cleanup_bytes": (
                    storage_available_bytes
                ),
                "prediction_interval_seconds": [
                    native_lower_ms / 1000,
                    native_upper_ms / 1000,
                ],
                "prediction_interval_exceeds_worker_ceiling": (
                    native_upper_ms / 1000 > worker_ceiling_seconds
                ),
                "requested_deadline_seconds": requested_deadline_seconds,
                "requested_epochs": requested_epochs,
                "probe_wall_time_ms": run["wall_time_ms"],
                "setup_wall_time_ms": setup_ms,
                "worker_ceiling_seconds": worker_ceiling_seconds,
                "profile_resource_efficiency_envelope": efficiency_envelope,
                "profile_resource_efficiency_envelope_exceeded": (
                    efficiency_envelope_exceeded
                ),
                "training_only_lower_bound_seconds": (
                    native_training_lower_bound_ms / 1000
                ),
                "uncertainty_basis": (
                    "min/max completed fixed train and eval batch throughput; "
                    "first-epoch excess over the same fixed train or eval batch "
                    "in later probe epochs is charged once as process warmup; "
                    "upper bound widened five percent per visible bottleneck feature; "
                    "admission uses the earliest legal native-stop window with ten "
                    "percent headroom; the requested-epoch estimate is non-blocking "
                    "cost-risk metadata because requested epochs is an upper bound"
                ),
            }
        )

    if len(contract_digests) != 1 or len(contract_shapes) != 1:
        raise ResourceSchedulingError("fixed-batch prefix contract differs across arms")
    probe_epochs, probe_train_batch_indices, probe_eval_batch_indices = next(
        iter(contract_shapes)
    )

    schedule_order = sorted(
        requested_deadlines,
        key=lambda arm: (
            float(predictions[arm]["estimated_total_wall_time_seconds"]),
            order.index(arm),
        ),
    )
    schedule: list[dict[str, Any]] = []
    allocated_seconds = 0
    for arm in schedule_order:
        prediction = predictions[arm]
        lower, upper = prediction["prediction_interval_seconds"]
        point = prediction["estimated_total_wall_time_seconds"]
        deadline = requested_deadlines[arm]
        if allocated_seconds + deadline > full_run_budget_seconds:
            deferred[arm] = {
                "reason": "REQUESTED_DEADLINE_NOT_ADMITTED_BY_CAMPAIGN_BUDGET",
                "resource_disposition": "RESOURCE_DEFERRED",
            }
            prediction["completion_probability"] = 0.0
            continue
        allocated_seconds += deadline
        completion_probability = 0.90
        prediction["completion_probability"] = completion_probability
        schedule.append(
            {
                "arm": arm,
                "deadline_seconds": deadline,
                "decision_inputs": {
                    "completion_probability": completion_probability,
                    "estimated_total_wall_time_seconds": point,
                    "peak_memory_prediction_mib": prediction[
                        "peak_memory_prediction_mib"
                    ],
                    "checkpoint_storage_bytes": prediction.get(
                        "checkpoint_storage_bytes"
                    ),
                    "projected_campaign_checkpoint_bytes": prediction.get(
                        "projected_campaign_checkpoint_bytes"
                    ),
                    "storage_available_after_probe_cleanup_bytes": prediction.get(
                        "storage_available_after_probe_cleanup_bytes"
                    ),
                    "prediction_interval_seconds": [lower, upper],
                },
                "ordinal": len(schedule) + 1,
            }
        )
    for arm, disposition in deferred.items():
        predictions[arm]["completion_probability"] = 0.0
        disposition.update(
            {
                "arm": arm,
                "estimated_total_wall_time_seconds": predictions[arm][
                    "estimated_total_wall_time_seconds"
                ],
                "mechanism_effect_update_allowed": False,
                "requested_deadline_seconds": requested_deadlines.get(arm),
            }
        )
    return canonical_value(
        {
            "campaign_total_budget_seconds": total_budget_seconds,
            "budget_accounting": budget_accounting,
            "deadline_allocation_scale": 1.0,
            "deadline_formula": (
                "treat the requested-epoch point estimate as "
                "NON_ADMISSION_COST_RISK_ONLY because requested epochs is an upper "
                "bound; defer candidate admission only when the actual Profile cadence's "
                "earliest legal native early-stop window plus ten percent headroom exceeds the "
                "caller-owned worker ceiling or remaining full-run campaign budget; "
                "for a Profile-owned recipe the resource probe is charged separately "
                "from that worker ceiling, when predicted "
                "peak memory violates the machine-owned reserve envelope, or when "
                "serialized state cannot fit the remaining 100-slot checkpoint horizon; "
                "allocate a deadline using "
                "min(worker_ceiling, max(180, ceil(1.10 * full_requested_epoch_point))) "
                "and schedule in ascending predicted-time order while the unified "
                "campaign budget can fund that capped deadline; the min/max interval "
                "is retained as uncertainty and is not used as the admission deadline"
            ),
            "deferred_arms": list(deferred.values()),
            "full_epochs": requested_epochs,
            "full_run_budget_after_probes_seconds": full_run_budget_seconds,
            "outcome_fields_consumed": [],
            "predictions": predictions,
            "probe_contract": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "epochs": probe_epochs,
                "eval_batch_indices": list(probe_eval_batch_indices),
                "execution_purpose": "RESOURCE_PROBE_ONLY",
                "prefix_contract_sha256": next(iter(contract_digests)),
                "seed": probe_seed,
                "timeout_seconds": PROBE_TIMEOUT_SECONDS,
                "train_batch_indices": list(probe_train_batch_indices),
                "uniform_across_arms": True,
            },
            "probe_cost_seconds": probe_cost_seconds,
            "requested_deadlines_seconds": requested_deadlines,
            "schedule": schedule,
            "schedule_rule": (
                "ascending predicted total wall time; stable original order tie-break"
            ),
            "schema": "recclaw.q0r-resource-prediction-and-schedule.v4",
        }
    )


def _predict_p4_precomputed_operator(
    run, telemetry, eval_rows, train_rows, *, features, device_memory_mib,
    requested_epochs, worker_ceiling_seconds, full_run_budget_seconds,
):
    """One measured train-only fit plus one full-sort evaluation, with no epochs."""
    if (telemetry.get("p4_fit_mode") != "TRAIN_ONLY_PRECOMPUTE"
            or telemetry.get("operator_fit_completed") is not True
            or telemetry.get("trainable_parameter_count") != 0
            or train_rows or not eval_rows or run.get("exit_status") != "SUCCESS"):
        raise ResourceSchedulingError("P4 fit-once probe lacks a completed declared operator and evaluation")
    setup_ms = telemetry["initialization_wall_time_ms"]
    full_eval_batches = telemetry["full_validation_batches_per_eval"]
    eval_ms = [float(row["wall_time_ms"]) for row in eval_rows]
    # Keep all measured setup and checkpoint/process overhead once; extrapolate
    # only the sampled full-sort evaluation batches, never a training trajectory.
    fixed_ms = max(float(setup_ms), float(run["wall_time_ms"]) - sum(eval_ms))
    point = (fixed_ms + sum(eval_ms) / len(eval_ms) * full_eval_batches) / 1000
    interval = [(fixed_ms + min(eval_ms) * full_eval_batches) / 1000,
                (fixed_ms + max(eval_ms) * full_eval_batches) / 1000]
    peaks = [float(row[key]) for row in eval_rows
             for key in ("peak_allocated_mib", "peak_reserved_mib")
             if isinstance(row.get(key), (int, float))]
    peaks.extend(float(telemetry[key]) for key in (
        "initialization_peak_allocated_mib", "initialization_peak_reserved_mib")
        if isinstance(telemetry.get(key), (int, float)))
    if not peaks:
        raise ResourceSchedulingError("P4 GPU resource prediction lacks observed memory evidence")
    peak = max(peaks)
    peak_prediction = peak * (1.0 + 0.03 * int(features.get("bottleneck_feature_count", 0)))
    checkpoint_bytes = run["checkpoint_footprint"]["bytes"]
    available_bytes = run["storage_observation"]["available_after_probe_cleanup_bytes"]
    projected_bytes = checkpoint_bytes * int(features.get("campaign_checkpoint_horizon_slots", 1))
    failure = ("PREDICTED_GPU_MEMORY_RESERVE_ENVELOPE" if peak_prediction >= device_memory_mib
               else "PROJECTED_CAMPAIGN_CHECKPOINT_STORAGE" if projected_bytes > available_bytes
               else "FIT_ONCE_COST_EXCEEDS_EXECUTION_BUDGET"
               if 1.10 * point > min(worker_ceiling_seconds, full_run_budget_seconds)
               else None)
    return canonical_value({
        "model": "P4_TRAIN_ONLY_FIT_ONCE_FULL_SORT_COST_V1",
        "p4_fit_mode": "TRAIN_ONLY_PRECOMPUTE", "operator_fit_completed": True,
        "estimate_scope": "ONE_TRAIN_ONLY_FIT_PLUS_ONE_FULL_SORT_EVALUATION",
        "completed_train_batches": 0, "completed_eval_batches": len(eval_rows),
        "optimization_epochs_completed": 0, "requested_epochs": requested_epochs,
        "execution_budget_required_seconds": 1.10 * point,
        "prediction_interval_exceeds_worker_ceiling": interval[1] > worker_ceiling_seconds,
        "estimated_total_wall_time_seconds": point,
        "prediction_interval_seconds": interval, "setup_wall_time_ms": setup_ms,
        "probe_wall_time_ms": run["wall_time_ms"],
        "first_progress_deadline_seconds": worker_ceiling_seconds,
        "requested_deadline_seconds": worker_ceiling_seconds if failure is None else None,
        "worker_ceiling_seconds": worker_ceiling_seconds,
        "peak_memory_candidate_ceiling_mib": device_memory_mib,
        "peak_memory_prediction_mib": peak_prediction, "peak_memory_observed_mib": peak,
        "checkpoint_storage_bytes": checkpoint_bytes,
        "projected_campaign_checkpoint_bytes": projected_bytes,
        "storage_available_after_probe_cleanup_bytes": available_bytes,
        "confidence": "LOW", "bottleneck_category": "OPERATOR_FIT_AND_FULL_SORT_EVALUATION",
        "uncertainty_basis": "measured once-only setup plus min/max sampled evaluation batch cost",
        "features": features,
    }), failure


_RESOURCE_PROFILE_FORBIDDEN_KEYS = frozenset(
    {
        "candidate_minus_bpr",
        "candidate_ndcg_at_10",
        "metrics",
        "ndcg@10",
        "test_feedback",
        "test_metric",
    }
)


def _assert_resource_profile_outcome_blind(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RESOURCE_PROFILE_FORBIDDEN_KEYS:
                raise ResourceSchedulingError(
                    f"resource profile contains forbidden outcome field: {key}"
                )
            _assert_resource_profile_outcome_blind(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_resource_profile_outcome_blind(child)


def _finite_resource_number(
    value: Any,
    *,
    field_name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceSchedulingError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResourceSchedulingError(f"{field_name} must be finite")
    if positive and result <= 0.0:
        raise ResourceSchedulingError(f"{field_name} must be positive")
    return result


def _resource_identity_sources(
    *,
    source_features: Mapping[str, Any],
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
    reservation_evidence: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    sources: list[tuple[str, Mapping[str, Any]]] = [
        ("source_features", source_features),
        ("probe_run", probe_run),
        ("process_observation", process_observation),
    ]
    for name, value in (
        ("probe_run.resource_probe_identity", probe_run.get("resource_probe_identity")),
        ("probe_run.resource_prediction", probe_run.get("resource_prediction")),
        (
            "process_observation.resource_probe_identity",
            process_observation.get("resource_probe_identity"),
        ),
    ):
        if isinstance(value, Mapping):
            identity = value.get("identity") if name.endswith("resource_prediction") else value
            if isinstance(identity, Mapping):
                sources.append((name, identity))
    if isinstance(reservation_evidence, Mapping):
        identity = reservation_evidence.get("identity")
        if isinstance(identity, Mapping):
            sources.append(("gpu_reservation_evidence.identity", identity))
    return tuple(sources)


def _assert_resource_identity_bound(
    *,
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    candidate_ref: str,
    candidate_package_digest: str | None,
    candidate_binding_digest: str | None,
    candidate_source_sha256: str,
    compute_pattern: str | None,
) -> None:
    aliases = {
        "candidate_ref": ("candidate_ref", "resource_candidate_ref"),
        "candidate_package_digest": (
            "candidate_package_digest",
            "package_digest",
        ),
        "candidate_binding_digest": (
            "candidate_binding_digest",
            "binding_digest",
        ),
        "candidate_source_sha256": ("candidate_source_sha256", "source_sha256"),
        "compute_pattern": ("compute_pattern",),
    }
    expected = {
        "candidate_ref": candidate_ref,
        "candidate_package_digest": candidate_package_digest,
        "candidate_binding_digest": candidate_binding_digest,
        "candidate_source_sha256": candidate_source_sha256,
        "compute_pattern": compute_pattern,
    }
    for source_name, source in sources:
        for identity_name, source_keys in aliases.items():
            if identity_name == "candidate_binding_digest":
                explicit_candidate_binding = source.get(
                    "candidate_binding_digest"
                )
                if explicit_candidate_binding is not None:
                    source_keys = ("candidate_binding_digest",)
                else:
                    source_keys = ("binding_digest",)
            for source_key in source_keys:
                if source_key not in source or source[source_key] is None:
                    continue
                if (
                    identity_name == "candidate_binding_digest"
                    and expected[identity_name] is None
                ):
                    # Legacy callers may expose a runtime binding_digest while
                    # not supplying a candidate-specific expected digest.
                    continue
                if source[source_key] != expected[identity_name]:
                    raise ResourceSchedulingError(
                        f"{source_name}.{source_key} does not match "
                        f"{identity_name}"
                    )


def _resolve_resource_compute_pattern(
    *,
    explicit: str | None,
    source_features: Mapping[str, Any],
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
) -> str | None:
    values: list[tuple[str, Any]] = []
    if explicit is not None:
        values.append(("compute_pattern", explicit))
    for source_name, source in (
        ("source_features", source_features),
        ("probe_run", probe_run),
        ("process_observation", process_observation),
    ):
        if "compute_pattern" in source and source["compute_pattern"] is not None:
            values.append((f"{source_name}.compute_pattern", source["compute_pattern"]))
    if not values:
        return None
    normalized: list[tuple[str, str]] = []
    for source_name, value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ResourceSchedulingError(
                f"{source_name} must be a normalized non-empty compute_pattern"
            )
        normalized.append((source_name, value))
    first = normalized[0][1]
    if any(value != first for _source_name, value in normalized[1:]):
        raise ResourceSchedulingError("compute_pattern identity mismatch")
    return first


def _consistent_resource_field(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    field_name: str,
) -> Any:
    values = [
        (source_name, source[field_name])
        for source_name, source in sources
        if field_name in source and source[field_name] is not None
    ]
    if not values:
        return None
    first = values[0][1]
    if any(value != first for _source_name, value in values[1:]):
        details = ", ".join(f"{name}={value!r}" for name, value in values)
        raise ResourceSchedulingError(
            f"{field_name} is inconsistent across probe evidence: {details}"
        )
    return first


def _probe_run_identity(
    *,
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
) -> str | None:
    values: list[tuple[str, Any]] = []
    for source_name, source in (
        ("probe_run", probe_run),
        ("process_observation", process_observation),
    ):
        if source.get("run_id") is not None:
            values.append((f"{source_name}.run_id", source["run_id"]))
        binding = source.get("experiment_binding")
        if isinstance(binding, Mapping) and binding.get("run_id") is not None:
            values.append(
                (f"{source_name}.experiment_binding.run_id", binding["run_id"])
            )
    if not values:
        return None
    normalized: list[tuple[str, str]] = []
    for source_name, value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ResourceSchedulingError(
                f"{source_name} must be a normalized non-empty run id"
            )
        normalized.append((source_name, value))
    first = normalized[0][1]
    if any(value != first for _source_name, value in normalized[1:]):
        raise ResourceSchedulingError("GPU reservation run identity is inconsistent")
    return first


def _resolve_probe_gpu_selection(
    *,
    requested_gpu_id: int | None,
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
    device_evidence: Mapping[str, Any] | None,
) -> tuple[int | None, str | None]:
    """Resolve explicit direct-gpu metadata without treating it as CVD."""

    requested = _validated_gpu_id(requested_gpu_id)
    sources = (
        ("probe_run", probe_run),
        ("process_observation", process_observation),
        ("training_device_evidence", device_evidence or {}),
    )
    gpu_ids: list[tuple[str, int]] = []
    selection_modes: list[tuple[str, Any]] = []
    physical_ids: list[tuple[str, str]] = []
    for source_name, source in sources:
        if source.get("gpu_id") is not None:
            gpu_id = _validated_gpu_id(source.get("gpu_id"))
            if gpu_id is not None:
                gpu_ids.append((f"{source_name}.gpu_id", gpu_id))
        if source.get("selection_mode") is not None:
            selection_modes.append(
                (f"{source_name}.selection_mode", source["selection_mode"])
            )
        if source.get("physical_gpu_id") is not None:
            value = source["physical_gpu_id"]
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ResourceSchedulingError(
                    f"{source_name}.physical_gpu_id is invalid"
                )
            normalized = str(value)
            if not normalized or normalized != normalized.strip():
                raise ResourceSchedulingError(
                    f"{source_name}.physical_gpu_id is invalid"
                )
            physical_ids.append(
                (f"{source_name}.physical_gpu_id", normalized)
            )
    if gpu_ids:
        first_gpu_id = gpu_ids[0][1]
        if any(value != first_gpu_id for _name, value in gpu_ids[1:]):
            raise ResourceSchedulingError(
                "direct gpu_id is inconsistent across probe evidence"
            )
        if requested is not None and requested != first_gpu_id:
            raise ResourceSchedulingError(
                "probe gpu_id does not match the direct launch selector"
            )
        requested = first_gpu_id
    if requested is None:
        if any(mode == DIRECT_GPU_SELECTION_MODE for _name, mode in selection_modes):
            raise ResourceSchedulingError(
                "direct GPU probe evidence lacks an explicit gpu_id"
            )
        return None, None
    if selection_modes and any(
        mode != DIRECT_GPU_SELECTION_MODE for _name, mode in selection_modes
    ):
        raise ResourceSchedulingError(
            "direct gpu_id probe evidence has an incompatible selection_mode"
        )
    expected_physical_id = str(requested)
    if physical_ids and any(
        value != expected_physical_id for _name, value in physical_ids
    ):
        raise ResourceSchedulingError(
            "direct gpu_id does not match the probe physical_gpu_id"
        )
    return requested, expected_physical_id


def _visible_device_from_probe(
    *,
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
    device_evidence: Mapping[str, Any] | None,
    reservation_evidence: Mapping[str, Any],
    gpu_id: int | None = None,
) -> str:
    if gpu_id is not None:
        selector = str(_validated_gpu_id(gpu_id))
        values: list[tuple[str, Any]] = []
        for source_name, source in (
            ("probe_run", probe_run),
            ("process_observation", process_observation),
            ("training_device_evidence", device_evidence or {}),
        ):
            for field_name in ("cuda_visible_devices", "visible_device"):
                if source.get(field_name) is not None:
                    raise ResourceSchedulingError(
                        "direct gpu_id probe must not report CUDA_VISIBLE_DEVICES"
                    )
            if source.get("physical_gpu_id") is not None:
                values.append(
                    (f"{source_name}.physical_gpu_id", source["physical_gpu_id"])
                )
        identity = reservation_evidence.get("identity")
        if isinstance(identity, Mapping):
            if identity.get("cuda_visible_devices") is not None:
                values.append(
                    (
                        "gpu_reservation_evidence.identity.cuda_visible_devices",
                        identity["cuda_visible_devices"],
                    )
                )
            if identity.get("physical_gpu_id") is not None:
                values.append(
                    (
                        "gpu_reservation_evidence.identity.physical_gpu_id",
                        identity["physical_gpu_id"],
                    )
                )
        if not values:
            raise ResourceSchedulingError(
                "direct gpu_id reservation requires physical GPU identity evidence"
            )
        if any(str(value) != selector for _name, value in values):
            details = ", ".join(f"{name}={value!r}" for name, value in values)
            raise ResourceSchedulingError(
                f"direct GPU binding is inconsistent across probe evidence: {details}"
            )
        return selector
    values: list[tuple[str, Any]] = []
    for source_name, source in (
        ("probe_run", probe_run),
        ("process_observation", process_observation),
        ("training_device_evidence", device_evidence or {}),
    ):
        for field_name in ("cuda_visible_devices", "visible_device"):
            if source.get(field_name) is not None:
                values.append((f"{source_name}.{field_name}", source[field_name]))
    identity = reservation_evidence.get("identity")
    if isinstance(identity, Mapping) and identity.get("cuda_visible_devices") is not None:
        values.append(("gpu_reservation_evidence.identity.cuda_visible_devices", identity["cuda_visible_devices"]))
    if not values:
        raise ResourceSchedulingError(
            "sealed GPU reservation requires explicit cuda_visible_devices"
        )
    first = values[0][1]
    if not isinstance(first, str) or not first or first != first.strip():
        raise ResourceSchedulingError("cuda_visible_devices must be normalized")
    if any(value != first for _source_name, value in values[1:]):
        details = ", ".join(f"{name}={value!r}" for name, value in values)
        raise ResourceSchedulingError(
            f"single-GPU binding is inconsistent across probe evidence: {details}"
        )
    if "," in first or first in {"-1", "NoDevFiles"}:
        raise ResourceSchedulingError(
            "sealed GPU reservation must bind exactly one concrete device"
        )
    return first


def _training_device_from_probe(
    probe_run: Mapping[str, Any],
) -> Mapping[str, Any]:
    values = [
        (field_name, probe_run[field_name])
        for field_name in ("training_device_evidence", "device_evidence")
        if isinstance(probe_run.get(field_name), Mapping)
    ]
    if not values:
        raise ResourceSchedulingError(
            "sealed GPU reservation requires training device evidence"
        )
    first = values[0][1]
    if any(dict(value) != dict(first) for _field_name, value in values[1:]):
        raise ResourceSchedulingError(
            "training device evidence is internally inconsistent"
        )
    return first


def _selected_device_capacity_mib(probe_run: Mapping[str, Any]) -> float:
    device_evidence = _training_device_from_probe(probe_run)
    capacity = device_evidence.get("total_memory_mib")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, (int, float))
        or not math.isfinite(float(capacity))
        or float(capacity) <= 0.0
    ):
        raise ResourceSchedulingError(
            "training device evidence lacks positive total_memory_mib"
        )
    return float(capacity)


def _execution_resource_parameters(
    execution_recipe: Mapping[str, Any] | None,
    *,
    total_budget_seconds: int,
) -> dict[str, Any]:
    if isinstance(total_budget_seconds, bool) or not isinstance(
        total_budget_seconds, int
    ) or total_budget_seconds <= 0:
        raise ResourceSchedulingError(
            "total_budget_seconds must be a positive integer"
        )
    if execution_recipe is None:
        return {
            "requested_epochs": FULL_EPOCHS,
            "native_eval_step": NATIVE_EVAL_STEP,
            "native_stopping_step": NATIVE_STOPPING_STEP,
            "worker_ceiling_seconds": min(
                MAX_WORKER_CEILING_SECONDS, total_budget_seconds
            ),
            "efficiency_envelope": None,
            "profile_owned": False,
        }
    if not isinstance(execution_recipe, Mapping):
        raise ResourceSchedulingError("execution_recipe must be a mapping")
    config = execution_recipe.get("config")
    if not isinstance(config, Mapping):
        raise ResourceSchedulingError("execution_recipe.config must be a mapping")
    cadence: dict[str, int] = {}
    for field_name in ("epochs", "eval_step", "stopping_step"):
        value = config.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResourceSchedulingError(
                f"execution_recipe.config.{field_name} must be a positive integer"
            )
        cadence[field_name] = value
    efficiency_envelope = config.get("recclaw_resource_efficiency_envelope")
    normalized_envelope = None
    if efficiency_envelope is not None:
        if (
            not isinstance(efficiency_envelope, Mapping)
            or set(efficiency_envelope)
            != {
                "schema",
                "parent_reference_epoch_wall_time_ms",
                "maximum_candidate_to_parent_ratio",
            }
            or efficiency_envelope.get("schema")
            != "recclaw.profile-resource-efficiency-envelope.v1"
        ):
            raise ResourceSchedulingError(
                "execution recipe resource efficiency envelope is invalid"
            )
        parent_ms = efficiency_envelope.get(
            "parent_reference_epoch_wall_time_ms"
        )
        maximum_ratio = efficiency_envelope.get(
            "maximum_candidate_to_parent_ratio"
        )
        for field_name, value in (
            ("parent_reference_epoch_wall_time_ms", parent_ms),
            ("maximum_candidate_to_parent_ratio", maximum_ratio),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ResourceSchedulingError(
                    f"resource efficiency envelope {field_name} must be positive"
                )
        normalized_envelope = canonical_value(
            {
                **dict(efficiency_envelope),
                "maximum_epoch_wall_time_ms": (
                    float(parent_ms) * float(maximum_ratio)
                ),
            }
        )
    return {
        "requested_epochs": cadence["epochs"],
        "native_eval_step": cadence["eval_step"],
        "native_stopping_step": cadence["stopping_step"],
        "worker_ceiling_seconds": total_budget_seconds,
        "efficiency_envelope": normalized_envelope,
        "profile_owned": True,
    }


def _validate_training_device_binding(
    device_evidence: Mapping[str, Any],
    *,
    visible_device: str,
    reservation_identity: Mapping[str, Any],
    probe_run: Mapping[str, Any],
    gpu_id: int | None = None,
) -> None:
    if device_evidence.get("cuda_available") is not True:
        raise ResourceSchedulingError(
            "training device evidence does not prove CUDA availability"
        )
    device_count = device_evidence.get("cuda_device_count")
    if (
        isinstance(device_count, bool)
        or not isinstance(device_count, int)
        or device_count < 1
    ):
        raise ResourceSchedulingError(
            "training device evidence does not prove CUDA device availability"
        )
    if gpu_id is not None:
        for field_name in ("logical_device", "cuda_device_index", "current_device"):
            if (
                field_name in device_evidence
                and device_evidence[field_name] != 0
            ):
                raise ResourceSchedulingError(
                    "training device evidence contradicts masked logical device 0: "
                    f"{field_name}"
                )
        if device_evidence.get("selection_mode") != DIRECT_GPU_SELECTION_MODE:
            raise ResourceSchedulingError(
                "training device evidence lacks direct gpu_id selection mode"
            )
        if device_evidence.get("gpu_id") != gpu_id:
            raise ResourceSchedulingError(
                "training device evidence contradicts gpu_id"
            )
        if str(device_evidence.get("physical_gpu_id")) != str(gpu_id):
            raise ResourceSchedulingError(
                "training device evidence contradicts physical_gpu_id"
            )
        if any(
            device_evidence.get(field_name) is not None
            for field_name in ("cuda_visible_devices", "visible_device")
        ):
            raise ResourceSchedulingError(
                "direct gpu_id training evidence must not report CUDA_VISIBLE_DEVICES"
            )
    else:
        if device_count != 1:
            raise ResourceSchedulingError(
                "CUDA_VISIBLE_DEVICES evidence does not prove exactly one CUDA device"
            )
        for field_name in ("logical_device", "cuda_device_index", "current_device"):
            if field_name in device_evidence and device_evidence[field_name] != 0:
                raise ResourceSchedulingError(
                    f"training device evidence contradicts logical device 0: {field_name}"
                )
        if device_evidence.get("selection_mode") == DIRECT_GPU_SELECTION_MODE:
            raise ResourceSchedulingError(
                "direct gpu_id training evidence lacks a direct selector"
            )
        for field_name in ("cuda_visible_devices", "visible_device"):
            if (
                field_name in device_evidence
                and device_evidence[field_name] != visible_device
            ):
                raise ResourceSchedulingError(
                    f"training device evidence contradicts {field_name}"
                )
    for field_name in ("physical_gpu_id", "device_uuid", "device_ref"):
        worker_value = device_evidence.get(field_name)
        reserved_value = reservation_identity.get(field_name)
        if worker_value is not None and reserved_value is not None:
            if str(worker_value) != str(reserved_value):
                raise ResourceSchedulingError(
                    f"training device evidence contradicts reservation {field_name}"
                )
    for worker_field in ("cuda_device_name", "gpu_name"):
        worker_value = device_evidence.get(worker_field)
        reserved_value = reservation_identity.get("device_name")
        if worker_value is not None and reserved_value is not None:
            if str(worker_value) != str(reserved_value):
                raise ResourceSchedulingError(
                    f"training device evidence contradicts reservation {worker_field}"
                )
    validation = probe_run.get("device_evidence_validation")
    if validation is not None and validation != "AVAILABLE_AND_CONSISTENT":
        raise ResourceSchedulingError(
            "physical probe reports contradictory training device evidence"
        )
    if not any(
        field_name in device_evidence
        for field_name in (
            "cuda_device_name",
            "gpu_name",
            "physical_gpu_id",
            "device_uuid",
            "device_ref",
        )
    ):
        raise ResourceSchedulingError(
            "training device evidence lacks a physical device identity"
        )


def _validated_gpu_worker_probe(
    *,
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
    source_features: Mapping[str, Any],
    candidate_ref: str,
    candidate_package_digest: str | None,
    candidate_source_sha256: str,
    compute_pattern: str | None,
    candidate_binding_digest: str | None = None,
    gpu_id: int | None = None,
) -> dict[str, Any] | None:
    evidence_sources = (
        ("probe_run", probe_run),
        ("process_observation", process_observation),
    )
    evidence_values = [
        (source_name, source.get("gpu_reservation_evidence"))
        for source_name, source in evidence_sources
        if source.get("gpu_reservation_evidence") is not None
    ]
    if not evidence_values:
        return None
    evidence = evidence_values[0][1]
    if not isinstance(evidence, Mapping):
        raise ResourceSchedulingError("gpu_reservation_evidence must be a mapping")
    if any(dict(value) != dict(evidence) for _source_name, value in evidence_values[1:]):
        raise ResourceSchedulingError(
            "GPU reservation evidence is inconsistent across probe result and process observation"
        )
    device_evidence = _training_device_from_probe(probe_run)
    resolved_gpu_id, physical_selector = _resolve_probe_gpu_selection(
        requested_gpu_id=gpu_id,
        probe_run=probe_run,
        process_observation=process_observation,
        device_evidence=device_evidence,
    )
    visible_device = _visible_device_from_probe(
        probe_run=probe_run,
        process_observation=process_observation,
        device_evidence=device_evidence,
        reservation_evidence=evidence,
        gpu_id=resolved_gpu_id,
    )
    run_id = _probe_run_identity(
        probe_run=probe_run,
        process_observation=process_observation,
    )
    if run_id is None:
        raise ResourceSchedulingError(
            "sealed GPU reservation requires an identity-bound training run id"
        )
    try:
        validated_evidence = validate_gpu_reservation_evidence(
            evidence,
            cuda_visible_devices=(
                None if resolved_gpu_id is not None else visible_device
            ),
            physical_gpu_selector=physical_selector,
            run_id=run_id,
        )
    except (FreshR1Error, TypeError, ValueError) as error:
        raise ResourceSchedulingError(
            f"GPU reservation evidence validation failed: {error}"
        ) from error
    if not isinstance(validated_evidence, Mapping):
        raise ResourceSchedulingError("sealed GPU reservation evidence is unavailable")
    reservation_identity = validated_evidence.get("identity")
    if not isinstance(reservation_identity, Mapping):
        raise ResourceSchedulingError("sealed GPU reservation identity is unavailable")
    _assert_resource_identity_bound(
        sources=_resource_identity_sources(
            source_features=source_features,
            probe_run=probe_run,
            process_observation=process_observation,
            reservation_evidence=validated_evidence,
        ),
        candidate_ref=candidate_ref,
        candidate_package_digest=candidate_package_digest,
        candidate_binding_digest=candidate_binding_digest,
        candidate_source_sha256=candidate_source_sha256,
        compute_pattern=compute_pattern,
    )
    if probe_run.get("gpu_reservation_status") != GPU_RESERVATION_STATUS_MEASURED:
        raise ResourceSchedulingError(
            "GPU reservation status is not a measured exclusive process interval"
        )
    if (
        probe_run.get("reserved_gpu_worker_seconds_semantics")
        != GPU_WORKER_SECONDS_SEMANTICS
    ):
        raise ResourceSchedulingError(
            "reserved GPU worker seconds semantics are not reservation-scoped"
        )
    if process_observation.get("start_method") != "spawn":
        raise ResourceSchedulingError(
            "sealed GPU resource evidence requires the spawn disposable process path"
        )
    measured_seconds = _finite_resource_number(
        probe_run.get("reserved_gpu_worker_seconds"),
        field_name="reserved_gpu_worker_seconds",
        positive=True,
    )
    parent_interval = probe_run.get("parent_process_interval")
    if not isinstance(parent_interval, Mapping):
        raise ResourceSchedulingError(
            "sealed GPU reservation requires parent_process_interval evidence"
        )
    interval_seconds = _finite_resource_number(
        parent_interval.get("parent_process_interval_seconds"),
        field_name="parent_process_interval.parent_process_interval_seconds",
        positive=True,
    )
    interval_wall_time_ms = _finite_resource_number(
        parent_interval.get("parent_process_interval_wall_time_ms"),
        field_name="parent_process_interval.parent_process_interval_wall_time_ms",
        positive=True,
    )
    if not math.isclose(measured_seconds, interval_seconds, rel_tol=1e-9, abs_tol=1e-9):
        raise ResourceSchedulingError(
            "reserved GPU worker seconds do not match the parent process interval"
        )
    probe_wall_time_ms = _finite_resource_number(
        probe_run.get("wall_time_ms"),
        field_name="probe_run.wall_time_ms",
        positive=True,
    )
    if not math.isclose(probe_wall_time_ms, interval_wall_time_ms, rel_tol=0.0, abs_tol=1.0):
        raise ResourceSchedulingError(
            "probe wall time does not match the parent-observed process interval"
        )
    _validate_training_device_binding(
        device_evidence,
        visible_device=visible_device,
        reservation_identity=reservation_identity,
        probe_run=probe_run,
        gpu_id=resolved_gpu_id,
    )
    return {
        "device_evidence": canonical_value(dict(device_evidence)),
        "measured_reserved_probe_seconds": measured_seconds,
        "parent_process_interval_seconds": interval_seconds,
        "parent_process_interval_wall_time_ms": interval_wall_time_ms,
        "probe_wall_time_ms": probe_wall_time_ms,
        "reservation_evidence": canonical_value(dict(validated_evidence)),
        "reservation_identity": canonical_value(dict(reservation_identity)),
        "run_id": run_id,
        "gpu_id": resolved_gpu_id,
        "physical_gpu_id": physical_selector,
        "selection_mode": (
            DIRECT_GPU_SELECTION_MODE if resolved_gpu_id is not None else None
        ),
        "visible_device": (
            None if resolved_gpu_id is not None else visible_device
        ),
    }


def _attach_gpu_worker_prediction(
    prediction: Mapping[str, Any],
    *,
    gpu_probe: Mapping[str, Any] | None,
    probe_run: Mapping[str, Any],
    prediction_identity: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(prediction)
    result["identity"] = canonical_value(dict(prediction_identity))
    if prediction_identity.get("compute_pattern") is not None:
        result["compute_pattern"] = prediction_identity["compute_pattern"]
    if gpu_probe is None:
        return result
    if (
        prediction_identity.get("candidate_package_digest") is None
        and prediction_identity.get("candidate_binding_digest") is None
    ):
        raise ResourceSchedulingError(
            "GPU worker prediction requires candidate_package_digest or "
            "candidate_binding_digest identity"
        )
    if not prediction_identity.get("compute_pattern"):
        raise ResourceSchedulingError(
            "GPU worker prediction requires compute_pattern identity"
        )
    model_probe_seconds = _finite_resource_number(
        result.get("probe_wall_time_ms"),
        field_name="prediction.probe_wall_time_ms",
        positive=True,
    ) / 1000.0
    native_execution_seconds = _finite_resource_number(
        result.get("estimated_total_wall_time_seconds"),
        field_name="prediction.estimated_total_wall_time_seconds",
        positive=True,
    )
    interval = result.get("prediction_interval_seconds")
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise ResourceSchedulingError(
            "prediction interval is unavailable for GPU worker scaling"
        )
    lower_seconds = _finite_resource_number(
        interval[0],
        field_name="prediction.prediction_interval_seconds[0]",
    )
    upper_seconds = _finite_resource_number(
        interval[1],
        field_name="prediction.prediction_interval_seconds[1]",
    )
    if lower_seconds <= 0.0 or upper_seconds < lower_seconds:
        raise ResourceSchedulingError(
            "prediction interval is invalid for GPU worker scaling"
        )
    measured_seconds = float(gpu_probe["measured_reserved_probe_seconds"])
    scale_factor = native_execution_seconds / model_probe_seconds
    _finite_resource_number(
        scale_factor,
        field_name="gpu_worker_prediction.scale_factor",
        positive=True,
    )
    predicted_seconds = measured_seconds * scale_factor
    predicted_interval = [
        measured_seconds * lower_seconds / model_probe_seconds,
        measured_seconds * upper_seconds / model_probe_seconds,
    ]
    _finite_resource_number(
        predicted_seconds,
        field_name="predicted_gpu_worker_seconds",
        positive=True,
    )
    if any(
        not math.isfinite(value) or value <= 0.0 for value in predicted_interval
    ):
        raise ResourceSchedulingError(
            "predicted GPU worker interval is not finite and positive"
        )
    reservation_evidence = gpu_probe["reservation_evidence"]
    reservation_identity = gpu_probe["reservation_identity"]
    source_digests = {
        "reservation_digest": reservation_evidence["reservation_digest"],
        "reservation_identity_digest": reservation_evidence["identity_digest"],
        "device_inventory_sha256": reservation_identity["device_inventory_sha256"],
        "process_snapshot_sha256": reservation_identity["process_snapshot_sha256"],
    }
    for field_name in ("result_sha256", "resource_telemetry_sha256"):
        value = probe_run.get(field_name)
        if value is not None:
            try:
                source_digests[field_name] = validate_sha256(
                    value,
                    field_name=f"probe_run.{field_name}",
                )
            except (TypeError, ValueError) as error:
                raise ResourceSchedulingError(str(error)) from error
    gpu_worker_prediction = {
        "formula": (
            "measured_reserved_probe_interval_seconds * "
            "(estimated_total_wall_time_seconds / "
            "model_probe_wall_time_seconds)"
        ),
        "formula_inputs": {
            "estimated_native_early_stop_wall_time_seconds": (
                native_execution_seconds
            ),
            "native_early_stop_prediction_interval_seconds": [
                lower_seconds,
                upper_seconds,
            ],
            "measured_reserved_probe_interval_seconds": measured_seconds,
            "model_probe_wall_time_seconds": model_probe_seconds,
            "model": result.get("model"),
            "probe_epochs": PROBE_EPOCHS,
            "native_early_stop_min_epochs": result.get(
                "native_early_stop_min_epochs"
            ),
            "requested_epochs": result.get("requested_epochs"),
        },
        "predicted_gpu_worker_seconds_interval": predicted_interval,
        "prediction_uncertainty_basis": result.get("uncertainty_basis"),
        "reservation_ref": reservation_evidence["reservation_ref"],
        "reservation_owner_ref": reservation_identity[
            "reservation_owner_ref"
        ],
        "reservation_scope": reservation_identity["scope"],
        "scale_factor": scale_factor,
        "source_digests": source_digests,
        "source_process_interval_seconds": gpu_probe[
            "parent_process_interval_seconds"
        ],
    }
    if gpu_probe.get("selection_mode") == DIRECT_GPU_SELECTION_MODE:
        gpu_worker_prediction.update(
            {
                "source_physical_gpu_id": gpu_probe["physical_gpu_id"],
                "source_selection_mode": DIRECT_GPU_SELECTION_MODE,
            }
        )
    else:
        gpu_worker_prediction["source_visible_device"] = gpu_probe[
            "visible_device"
        ]
    result.update(
        {
            "gpu_worker_prediction": gpu_worker_prediction,
            "predicted_gpu_worker_seconds": predicted_seconds,
            "predicted_gpu_worker_seconds_interval": predicted_interval,
            "predicted_gpu_worker_seconds_semantics": (
                PREDICTED_GPU_WORKER_SECONDS_SEMANTICS
            ),
        }
    )
    return canonical_value(result)


def _resource_probe_evidence_projection(run: Mapping[str, Any]) -> dict[str, Any]:
    """Carry sealed GPU facts without copying raw training telemetry."""

    projection: dict[str, Any] = {}
    for field_name in (
        "candidate_ref",
        "candidate_package_digest",
        "candidate_binding_digest",
        "candidate_source_sha256",
        "compute_pattern",
        "cuda_visible_devices",
        "gpu_id",
        "physical_gpu_id",
        "selection_mode",
        "device_evidence_validation",
        "gpu_reservation_status",
        "reserved_gpu_worker_seconds",
        "reserved_gpu_worker_seconds_semantics",
        "resource_telemetry_sha256",
        "result_sha256",
        "run_id",
    ):
        if field_name in run:
            projection[field_name] = run[field_name]
    for field_name in (
        "checkpoint_footprint",
        "device_evidence",
        "training_device_evidence",
        "gpu_reservation_evidence",
        "parent_process_interval",
        "storage_observation",
    ):
        value = run.get(field_name)
        if isinstance(value, Mapping):
            projection[field_name] = dict(value)
    binding = run.get("experiment_binding")
    if isinstance(binding, Mapping) and binding.get("run_id") is not None:
        projection["experiment_binding"] = {"run_id": binding["run_id"]}
    return projection


def _gpu_probe_telemetry_metadata(run: Mapping[str, Any]) -> dict[str, Any]:
    evidence = run.get("gpu_reservation_evidence")
    if not isinstance(evidence, Mapping):
        return {}
    identity = evidence.get("identity")
    if not isinstance(identity, Mapping):
        return {}
    metadata = {
        "gpu_reservation_status": run.get("gpu_reservation_status"),
        "gpu_id": run.get("gpu_id"),
        "physical_gpu_id": run.get("physical_gpu_id"),
        "selection_mode": run.get("selection_mode"),
        "reserved_gpu_worker_seconds": run.get("reserved_gpu_worker_seconds"),
        "reserved_gpu_worker_seconds_semantics": run.get(
            "reserved_gpu_worker_seconds_semantics"
        ),
        "gpu_reservation_ref": evidence.get("reservation_ref"),
        "gpu_reservation_identity_digest": evidence.get("identity_digest"),
        "gpu_reservation_digest": evidence.get("reservation_digest"),
        "gpu_device_inventory_sha256": identity.get("device_inventory_sha256"),
        "gpu_process_snapshot_sha256": identity.get("process_snapshot_sha256"),
        "device_evidence_validation": run.get("device_evidence_validation"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _resource_probe_input(run: Mapping[str, Any]) -> dict[str, Any]:
    """Project a training result to fields consumed by the resource model."""

    telemetry = run.get("resource_telemetry")
    if not isinstance(telemetry, Mapping):
        return canonical_value(
            {
                "exit_status": run.get("exit_status"),
                "resource_telemetry": None,
                "wall_time_ms": int(run.get("wall_time_ms", 0) or 0),
                **_resource_probe_evidence_projection(run),
            }
        )
    rows = []
    for row in telemetry.get("batch_records") or ():
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                key: row.get(key)
                for key in (
                    "batch_position",
                    "epoch",
                    "loss",
                    "phase",
                    "source_batch_index",
                    "status",
                    "wall_time_ms",
                    "peak_allocated_mib",
                    "peak_reserved_mib",
                )
                if key in row
            }
        )
    prefix_contract = telemetry.get("prefix_contract")
    safe_contract = (
        {
            key: prefix_contract.get(key)
            for key in (
                "contract_file_sha256",
                "epochs",
                "eval_batch_indices",
                "train_batch_indices",
            )
            if key in prefix_contract
        }
        if isinstance(prefix_contract, Mapping)
        else None
    )
    safe_telemetry = {
        "batch_records": rows,
        "full_train_batches_per_epoch": telemetry.get("full_train_batches_per_epoch"),
        "full_validation_batches_per_eval": telemetry.get(
            "full_validation_batches_per_eval"
        ),
        "initialization_wall_time_ms": telemetry.get("initialization_wall_time_ms"),
        "parameter_count": telemetry.get("parameter_count"),
        "prefix_contract": safe_contract,
        "trainable_parameter_count": telemetry.get("trainable_parameter_count"),
    }
    if telemetry.get("p4_fit_mode") == "TRAIN_ONLY_PRECOMPUTE":
        safe_telemetry.update({key: telemetry.get(key) for key in (
            "p4_fit_mode", "operator_fit_completed", "optimization_epochs_completed",
            "initialization_peak_allocated_mib", "initialization_peak_reserved_mib",
        )})
    return canonical_value(
        {
            "exit_status": run.get("exit_status"),
            "resource_telemetry": safe_telemetry,
            "wall_time_ms": int(run.get("wall_time_ms", 0) or 0),
            **_resource_probe_evidence_projection(run),
        }
    )


def _probe_telemetry_projection(run: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only resource facts from a physical probe result.

    ``run_development_training`` also returns development metrics.  They are
    intentionally not copied into the Research Innovation admission profile;
    only fixed-batch throughput, memory, and process status are admissible
    inputs here.
    """

    telemetry = run.get("resource_telemetry")
    if not isinstance(telemetry, Mapping):
        return {
            "completed_eval_batches": 0,
            "completed_train_batches": 0,
            "peak_gpu_memory_mib": None,
            "telemetry_present": False,
            "wall_time_ms": int(run.get("wall_time_ms", 0) or 0),
        }
    batches = telemetry.get("batch_records") or ()
    completed = [
        row
        for row in batches
        if isinstance(row, Mapping) and row.get("status") == "BATCH_COMPLETED"
    ]
    completed_train_batches = sum(
        row.get("phase") == "TRAIN" for row in completed
    )
    completed_eval_batches = sum(
        row.get("phase") == "EVAL" for row in completed
    )
    phase_records = telemetry.get("phase_records") or ()

    def completed_phase_batches(phase: str, batch_count_field: str) -> int:
        batches_per_phase = telemetry.get(batch_count_field)
        if (
            isinstance(batches_per_phase, bool)
            or not isinstance(batches_per_phase, int)
            or batches_per_phase < 1
        ):
            return 0
        completed_epochs = {
            row.get("epoch"): row.get("batch_count", batches_per_phase)
            for row in phase_records
            if isinstance(row, Mapping)
            and str(row.get("phase") or "").upper() == phase
            and str(row.get("status") or "").upper()
            in {"SUCCESS", "PHASE_COMPLETED"}
            and isinstance(row.get("epoch"), int)
            and not isinstance(row.get("epoch"), bool)
        }
        # An explicit zero is measured zero, not a missing full-epoch count.
        # Only older phase-only telemetry without batch_count needs the legacy
        # full-loader projection; a fixed prefix must retain its actual count.
        return sum(
            count for count in completed_epochs.values()
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0
        )

    if completed_train_batches == 0:
        completed_train_batches = completed_phase_batches(
            "TRAIN", "full_train_batches_per_epoch"
        )
    if completed_eval_batches == 0:
        completed_eval_batches = completed_phase_batches(
            "EVAL", "full_validation_batches_per_eval"
        )
    observed_peaks = [
        float(row[key])
        for row in completed
        for key in ("peak_allocated_mib", "peak_reserved_mib")
        if isinstance(row.get(key), (int, float))
    ]
    peak = telemetry.get("peak_gpu_memory_mib")
    if not isinstance(peak, (int, float)) and observed_peaks:
        peak = max(observed_peaks)
    return canonical_value(
        {
            "completed_eval_batches": completed_eval_batches,
            "completed_train_batches": completed_train_batches,
            "peak_gpu_memory_mib": peak,
            "telemetry_present": True,
            "wall_time_ms": int(run.get("wall_time_ms", 0) or 0),
            **_gpu_probe_telemetry_metadata(run),
        }
    )


def build_innovation_resource_profile(
    *,
    candidate_ref: str,
    candidate_package_digest: str | None,
    candidate_source_sha256: str,
    candidate_binding_digest: str | None = None,
    source_features: Mapping[str, Any],
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
    total_budget_seconds: int = CAMPAIGN_TOTAL_BUDGET_SECONDS,
    execution_recipe: Mapping[str, Any] | None = None,
    probe_seed: int = TRAINING_SEED,
    compute_pattern: str | None = None,
    gpu_id: int | None = None,
    offline_calibration_probe_excluded_from_future_budget: bool = False,
) -> dict[str, Any]:
    """Build an outcome-blind Research Innovation resource admission profile.

    This is the pure consumer seam for the existing fixed-batch model.  It
    retains the prediction interval, memory prediction, and budget schedule,
    while refusing to expose the probe's scientific metrics to callers.  An
    explicit ``compute_pattern`` plus a fresh-R1 sealed reservation/device
    result can additionally produce an identity-bound
    ``predicted_gpu_worker_seconds`` field.  Missing or incomplete reservation
    evidence leaves the legacy wall-time profile unchanged; wall time is never
    treated as GPU-worker time by itself.
    """

    if not isinstance(candidate_ref, str) or not candidate_ref:
        raise ResourceSchedulingError("candidate_ref must be non-empty")
    if not isinstance(source_features, Mapping):
        raise ResourceSchedulingError("source_features must be a mapping")
    if not isinstance(probe_run, Mapping):
        raise ResourceSchedulingError("probe_run must be a mapping")
    if not isinstance(process_observation, Mapping):
        raise ResourceSchedulingError("process_observation must be a mapping")
    if not isinstance(
        offline_calibration_probe_excluded_from_future_budget,
        bool,
    ):
        raise ResourceSchedulingError(
            "offline calibration budget exclusion must be a boolean"
        )
    validated_gpu_id = _validate_gpu_selection_arguments(
        cuda_visible_devices=(
            probe_run.get("cuda_visible_devices")
            if probe_run.get("cuda_visible_devices") is not None
            else None
        ),
        gpu_id=gpu_id,
    )
    try:
        validate_sha256(candidate_source_sha256, field_name="candidate_source_sha256")
        if candidate_package_digest is not None:
            validate_sha256(
                candidate_package_digest,
                field_name="candidate_package_digest",
            )
        if candidate_binding_digest is not None:
            validate_sha256(
                candidate_binding_digest,
                field_name="candidate_binding_digest",
            )
    except ValueError as error:
        raise ResourceSchedulingError(str(error)) from error
    if source_features.get("source_sha256") != candidate_source_sha256:
        raise ResourceSchedulingError(
            "source feature digest does not match candidate_source_sha256"
        )
    _assert_resource_profile_outcome_blind(source_features)
    resolved_compute_pattern = _resolve_resource_compute_pattern(
        explicit=compute_pattern,
        source_features=source_features,
        probe_run=probe_run,
        process_observation=process_observation,
    )
    identity_sources = _resource_identity_sources(
        source_features=source_features,
        probe_run=probe_run,
        process_observation=process_observation,
    )
    _assert_resource_identity_bound(
        sources=identity_sources,
        candidate_ref=candidate_ref,
        candidate_package_digest=candidate_package_digest,
        candidate_binding_digest=candidate_binding_digest,
        candidate_source_sha256=candidate_source_sha256,
        compute_pattern=resolved_compute_pattern,
    )
    if not (
        process_observation.get("process_isolated") is True
        and process_observation.get("status") == "RESULT"
        and process_observation.get("exit_code") == 0
    ):
        raise ResourceSchedulingError(
            "Research Innovation resource profiles require a successful disposable probe process"
        )
    gpu_probe: dict[str, Any] | None = None
    offline_accounting: dict[str, Mapping[str, Any]] | None = None
    if offline_calibration_probe_excluded_from_future_budget:
        gpu_probe = _validated_gpu_worker_probe(
            probe_run=probe_run,
            process_observation=process_observation,
            source_features=source_features,
            candidate_ref=candidate_ref,
            candidate_package_digest=candidate_package_digest,
            candidate_binding_digest=candidate_binding_digest,
            candidate_source_sha256=candidate_source_sha256,
            compute_pattern=resolved_compute_pattern,
            gpu_id=validated_gpu_id,
        )
        if gpu_probe is None:
            raise ResourceSchedulingError(
                "offline calibration budget exclusion requires validated sealed GPU reservation evidence"
            )
        reservation_evidence = gpu_probe["reservation_evidence"]
        reservation_identity = gpu_probe["reservation_identity"]
        offline_accounting = {
            candidate_ref: {
                "mode": OFFLINE_CALIBRATION_PROBE_BUDGET_MODE,
                "parent_process_interval_wall_time_ms": gpu_probe[
                    "parent_process_interval_wall_time_ms"
                ],
                "probe_wall_time_ms": gpu_probe["probe_wall_time_ms"],
                "reservation_digest": reservation_evidence["reservation_digest"],
                "reservation_identity_digest": reservation_evidence[
                    "identity_digest"
                ],
            }
        }
    resource_probe = _resource_probe_input(probe_run)
    checkpoint_footprint = resource_probe.get("checkpoint_footprint")
    storage_observation = resource_probe.get("storage_observation")
    if not (
        isinstance(checkpoint_footprint, Mapping)
        and isinstance(checkpoint_footprint.get("bytes"), int)
        and checkpoint_footprint.get("bytes") > 0
        and isinstance(storage_observation, Mapping)
        and isinstance(storage_observation.get("available_after_probe_cleanup_bytes"), int)
        and storage_observation.get("available_after_probe_cleanup_bytes") > 0
    ):
        raise ResourceSchedulingError(
            "resource probe lacks serialized-state and storage-capacity evidence"
        )
    decision = predict_resources(
        arm_features={candidate_ref: source_features},
        probe_runs={candidate_ref: resource_probe},
        total_budget_seconds=total_budget_seconds,
        execution_recipe=execution_recipe,
        arm_order=(candidate_ref,),
        probe_seed=probe_seed,
        offline_calibration_probe_accounting=offline_accounting,
    )
    prediction_identity = {
        "candidate_binding_digest": candidate_binding_digest,
        "candidate_package_digest": candidate_package_digest,
        "candidate_ref": candidate_ref,
        "candidate_source_sha256": candidate_source_sha256,
        "compute_pattern": resolved_compute_pattern,
    }
    if not offline_calibration_probe_excluded_from_future_budget:
        gpu_probe = _validated_gpu_worker_probe(
            probe_run=probe_run,
            process_observation=process_observation,
            source_features=source_features,
            candidate_ref=candidate_ref,
            candidate_package_digest=candidate_package_digest,
            candidate_binding_digest=candidate_binding_digest,
            candidate_source_sha256=candidate_source_sha256,
            compute_pattern=resolved_compute_pattern,
            gpu_id=validated_gpu_id,
        )
    prediction = _attach_gpu_worker_prediction(
        decision["predictions"][candidate_ref],
        gpu_probe=gpu_probe,
        probe_run=probe_run,
        prediction_identity=prediction_identity,
    )
    schedule = [
        row for row in decision["schedule"] if row.get("arm") == candidate_ref
    ]
    deferred = [
        row for row in decision["deferred_arms"] if row.get("arm") == candidate_ref
    ]
    admitted = bool(schedule)
    profile_identity = {
        "candidate_binding_digest": candidate_binding_digest,
        "candidate_package_digest": candidate_package_digest,
        "candidate_ref": candidate_ref,
        "candidate_source_sha256": candidate_source_sha256,
        "compute_pattern": resolved_compute_pattern,
        "budget_accounting": decision["budget_accounting"],
        "probe_contract": decision["probe_contract"],
        "prediction": prediction,
    }
    profile = {
        "candidate_binding_digest": candidate_binding_digest,
        "candidate_package_digest": candidate_package_digest,
        "candidate_ref": candidate_ref,
        "candidate_source_sha256": candidate_source_sha256,
        "completion_probability": float(prediction["completion_probability"]),
        "budget_accounting": decision["budget_accounting"],
        "deferred": deferred,
        "effect_fields_consumed": [],
        "full_run_budget_after_probes_seconds": decision[
            "full_run_budget_after_probes_seconds"
        ],
        "held_out_reads": 0,
        "mechanism_effect_update_allowed": False,
        "prediction": prediction,
        "prediction_interval_seconds": prediction["prediction_interval_seconds"],
        "probe": _probe_telemetry_projection(resource_probe),
        "probe_contract": decision["probe_contract"],
        "probe_process": canonical_value(dict(process_observation)),
        "profile_digest": sha256_digest(profile_identity),
        "schedule": schedule,
        "status": "RESOURCE_ADMITTED" if admitted else "RESOURCE_DEFERRED",
        "outcome_fields_consumed": [],
        "schema": "recclaw.research-line.innovation-resource-profile.v1",
    }
    if "predicted_gpu_worker_seconds" in prediction:
        profile.update(
            {
                "compute_pattern": resolved_compute_pattern,
                "predicted_gpu_worker_seconds": prediction[
                    "predicted_gpu_worker_seconds"
                ],
                "predicted_gpu_worker_seconds_semantics": prediction[
                    "predicted_gpu_worker_seconds_semantics"
                ],
            }
        )
    return canonical_value(profile)


def _failed_innovation_resource_profile(
    *,
    candidate_ref: str,
    candidate_package_digest: str | None,
    candidate_source_sha256: str,
    candidate_binding_digest: str | None = None,
    compute_pattern: str | None = None,
    process_observation: Mapping[str, Any],
    reason_code: str,
    failure_phase: str = "DISPOSABLE_PROCESS",
    error_type: str | None = None,
    error_message: str | None = None,
    failure_scope: str | None = None,
    probe_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_scope = (
        str(failure_scope).upper()
        if failure_scope
        in {
            "CANDIDATE_LOCAL",
            "LINEAGE_COMPUTE_PATTERN",
            "WORKER_TRANSIENT",
            "SHARED_INFRASTRUCTURE",
            "RECOVERY",
        }
        else None
    )
    recoverable_probe_failure = normalized_scope in {
        "WORKER_TRANSIENT",
        "SHARED_INFRASTRUCTURE",
        "RECOVERY",
    }
    diagnostic_type = error_type or reason_code
    diagnostic_message = str(error_message or reason_code)[:2000]
    return canonical_value(
        {
            "candidate_binding_digest": candidate_binding_digest,
            "candidate_package_digest": candidate_package_digest,
            "candidate_ref": candidate_ref,
            "candidate_source_sha256": candidate_source_sha256,
            "compute_pattern": compute_pattern,
            "completion_probability": 0.0,
            "deferred": [
                {
                    "arm": candidate_ref,
                    "mechanism_effect_update_allowed": False,
                    "reason": reason_code,
                    "resource_disposition": (
                        "RESOURCE_PROBE_RECOVERY_REQUIRED"
                        if recoverable_probe_failure
                        else "RESOURCE_INFEASIBLE"
                    ),
                }
            ],
            "effect_fields_consumed": [],
            "held_out_reads": 0,
            "mechanism_effect_update_allowed": False,
            "prediction": {
                "completion_probability": 0.0,
                "identity": {
                    "candidate_binding_digest": candidate_binding_digest,
                    "candidate_package_digest": candidate_package_digest,
                    "candidate_ref": candidate_ref,
                    "candidate_source_sha256": candidate_source_sha256,
                    "compute_pattern": compute_pattern,
                },
                "model": "FIXED_BATCH_THROUGHPUT_NATIVE_EARLY_STOP_EXTRAPOLATION_V5",
            },
            "probe": (
                _probe_telemetry_projection(probe_run)
                if isinstance(probe_run, Mapping)
                else {
                    "completed_eval_batches": 0,
                    "completed_train_batches": 0,
                    "peak_gpu_memory_mib": None,
                    "telemetry_present": False,
                    "wall_time_ms": 0,
                }
            ),
            "probe_process": canonical_value(dict(process_observation)),
            "resource_probe_diagnostic": {
                "error_type": diagnostic_type,
                "message": diagnostic_message,
                "outcome_fields_consumed": [],
                "phase": failure_phase,
                **(
                    {"failure_scope": normalized_scope}
                    if normalized_scope is not None
                    else {}
                ),
            },
            "schedule": [],
            "status": "RESOURCE_PROBE_FAILED",
            **(
                {"failure_scope": normalized_scope}
                if normalized_scope is not None
                else {}
            ),
            "outcome_fields_consumed": [],
            "schema": "recclaw.research-line.innovation-resource-profile.v1",
        }
    )


def _partial_probe_run(
    probe_root: Path,
    arm_id: str,
    process_observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load batch telemetry left by a probe stopped at its time boundary."""

    telemetry_path = _completed_probe_artifact_paths(probe_root, arm_id)[
        "resource_telemetry"
    ]
    try:
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    prefix_contract = (
        telemetry.get("prefix_contract")
        if isinstance(telemetry, Mapping)
        else None
    )
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("schema") != "recclaw.worker-resource-telemetry.v2"
        or not isinstance(prefix_contract, Mapping)
        or prefix_contract.get("execution_purpose") != "RESOURCE_PROBE_ONLY"
        or not isinstance(telemetry.get("batch_records"), (list, tuple))
    ):
        return None
    elapsed = process_observation.get("elapsed_wall_time_ms")
    return canonical_value(
        {
            "resource_telemetry": dict(telemetry),
            "wall_time_ms": (
                int(elapsed)
                if isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                and elapsed >= 0
                else 0
            ),
        }
    )


def _relocated_partial_probe_run(
    probe_root: Path,
    *,
    current_arm_id: str,
    candidate_source_sha256: str,
    execution_recipe: Mapping[str, Any],
    probe_seed: int,
    probe_timeout_seconds: int,
    process_observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recover progress for the same code rematerialized under a new wrapper.

    Candidate package, capability, and root identities include their physical
    materialization boundary and therefore change on checkpoint resume.  The
    behavior-source tree and content identities do not.  A prior probe under
    this semantic probe root is reusable as progress evidence only when those
    stable code identities and the fixed probe protocol still match.
    """

    source_tree_digest = execution_recipe.get("candidate_source_tree_digest")
    source_content_digest = execution_recipe.get(
        "candidate_source_content_digest"
    )
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in (source_tree_digest, source_content_digest)
    ):
        return None
    stable_recipe_fields = (
        "base_model_config",
        "capability_family",
        "config",
        "dataset",
        "entrypoint",
        "evaluator",
        "execution_role",
        "mechanism_id",
        "model",
        "split",
    )
    prefix_path = probe_root / "FIXED_BATCH_PREFIX_CONTRACT.json"
    try:
        prefix_digest = bytes_sha256(prefix_path.read_bytes())
    except OSError:
        return None
    best: dict[str, Any] | None = None
    best_completed = -1
    experiments_root = probe_root / "runs" / "experiments"
    run_roots = (
        tuple(experiments_root.iterdir()) if experiments_root.is_dir() else ()
    )
    for run_root in sorted(run_roots):
        try:
            binding = json.loads(
                (run_root / "experiment_binding.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if (
            not isinstance(binding, Mapping)
            or binding.get("candidate_source_tree_digest") != source_tree_digest
            or binding.get("candidate_source_content_digest")
            != source_content_digest
            or binding.get("entrypoint_source_sha256")
            != candidate_source_sha256
            or binding.get("execution_purpose") != "RESOURCE_PROBE_ONLY"
            or binding.get("seed") != probe_seed
            or binding.get("timeout_seconds") != probe_timeout_seconds
            or binding.get("prefix_contract_digest") != prefix_digest
            or any(
                binding.get(field) != execution_recipe.get(field)
                for field in stable_recipe_fields
            )
        ):
            continue
        arm_id = (
            current_arm_id
            if run_root.name == current_arm_id.replace("_", "-")
            else run_root.name.replace("-", "_", 1)
        )
        partial = _partial_probe_run(probe_root, arm_id, process_observation)
        if partial is None:
            continue
        projection = _probe_telemetry_projection(partial)
        completed = int(projection.get("completed_train_batches", 0))
        if completed > best_completed:
            best = partial
            best_completed = completed
    return best


_DETERMINISTIC_ZERO_BATCH_STALL_MIN_OBSERVATIONS = 3


def _zero_batch_training_stall_signature(
    probe_root: Path,
    arm_id: str,
    *,
    candidate_package_digest: str,
    candidate_source_sha256: str,
) -> dict[str, Any] | None:
    """Return the stable identity of one initialized zero-batch train stall."""

    paths = _completed_probe_artifact_paths(probe_root, arm_id)
    try:
        telemetry = json.loads(
            paths["resource_telemetry"].read_text(encoding="utf-8")
        )
        binding = json.loads(
            paths["experiment_binding"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    active_progress = (
        telemetry.get("active_progress")
        if isinstance(telemetry, Mapping)
        else None
    )
    prefix_contract = (
        telemetry.get("prefix_contract")
        if isinstance(telemetry, Mapping)
        else None
    )
    initialization_ms = (
        telemetry.get("initialization_wall_time_ms")
        if isinstance(telemetry, Mapping)
        else None
    )
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("schema") != "recclaw.worker-resource-telemetry.v2"
        or not isinstance(binding, Mapping)
        or binding.get("schema")
        != "recclaw.research-line.experiment-binding.v1"
        or binding.get("execution_purpose") != "RESOURCE_PROBE_ONLY"
        or binding.get("candidate_package_digest") != candidate_package_digest
        or binding.get("entrypoint_source_sha256") != candidate_source_sha256
        or not isinstance(prefix_contract, Mapping)
        or prefix_contract.get("execution_purpose") != "RESOURCE_PROBE_ONLY"
        or binding.get("prefix_contract_digest")
        != prefix_contract.get("contract_file_sha256")
        or not isinstance(active_progress, Mapping)
        or active_progress.get("epoch") != 0
        or str(active_progress.get("phase") or "").upper() != "TRAIN"
        or str(active_progress.get("status") or "").upper()
        not in {"PHASE_STARTED", "RUNTIME_FAILURE"}
        or telemetry.get("batch_records") != []
        or not isinstance(telemetry.get("phase_records"), list)
        or any(
            not isinstance(row, Mapping)
            or row.get("epoch") != 0
            or row.get("phase") != "TRAIN"
            or row.get("status") != "RUNTIME_FAILURE"
            or row.get("batch_count") != 0
            for row in telemetry.get("phase_records", ())
        )
        or telemetry.get("completed_batch_records") != 0
        or telemetry.get("epochs_completed") != 0
        or isinstance(initialization_ms, bool)
        or not isinstance(initialization_ms, int)
        or initialization_ms <= 0
    ):
        return None
    identity_fields = (
        "candidate_package_digest",
        "entrypoint_source_sha256",
        "execution_recipe_digest",
        "prefix_contract_digest",
        "runtime_binding_digest",
        "runtime_release_digest",
        "seed",
        "timeout_seconds",
    )
    telemetry_fields = (
        "full_train_batches_per_epoch",
        "full_validation_batches_per_eval",
        "parameter_count",
        "trainable_parameter_count",
    )
    return canonical_value(
        {
            **{field: binding.get(field) for field in identity_fields},
            **{field: telemetry.get(field) for field in telemetry_fields},
        }
    )


def reclassify_progressing_timeout_profile(
    profile: Mapping[str, Any],
    *,
    probe_roots: Sequence[Path] = (),
) -> dict[str, Any] | None:
    """Turn a measured resource stop into candidate-local resource evidence.

    A timeout before any batch telemetry remains recoverable infrastructure.
    Once the candidate has completed a real training phase, however, the fixed
    probe has measured its implementation throughput.  The same rule applies
    when a parent interruption is discovered only after rematerialization.  A
    normally returned ``RESOURCE_CENSORED`` probe is likewise a completed
    resource observation, not a transport failure.  Retrying any of these
    cannot answer a new question.
    """

    diagnostic = profile.get("resource_probe_diagnostic")
    if not isinstance(diagnostic, Mapping):
        return None
    scope = str(
        profile.get("failure_scope") or diagnostic.get("failure_scope") or ""
    ).upper()
    error_type = str(diagnostic.get("error_type") or "").upper()
    process_observation = profile.get("probe_process")
    process_observation = (
        process_observation if isinstance(process_observation, Mapping) else {}
    )
    completed_resource_censor = (
        error_type == "RESOURCE_CENSORED"
        and str(process_observation.get("status") or "").upper() == "RESULT"
        and process_observation.get("exit_code") == 0
    )
    progressing_interrupted_recovery = (
        error_type
        in {
            "RESOURCE_PROBE_INTERRUPTED_RECOVERY",
            "RESOURCE_PROBE_LEGACY_INTERVAL_UNAVAILABLE",
        }
        and scope == "RECOVERY"
    )
    if (
        str(profile.get("status") or "") != "RESOURCE_PROBE_FAILED"
        or not (
            progressing_interrupted_recovery
            or (
                error_type in {"PROBE_TIMEOUT", "RESOURCE_CENSORED"}
                and scope == "WORKER_TRANSIENT"
            )
        )
        or (error_type == "RESOURCE_CENSORED" and not completed_resource_censor)
    ):
        return None

    projection = profile.get("probe")
    best_projection = (
        canonical_value(dict(projection))
        if isinstance(projection, Mapping)
        else None
    )
    candidate_ref = profile.get("candidate_ref")
    candidate_package_digest = profile.get("candidate_package_digest")
    candidate_source_sha256 = profile.get("candidate_source_sha256")
    stall_signature_counts: dict[str, int] = {}
    if (
        isinstance(candidate_ref, str)
        and candidate_ref
        and isinstance(candidate_package_digest, str)
        and isinstance(candidate_source_sha256, str)
    ):
        arm_id = "candidate_" + sha256_digest(
            {"candidate_ref": candidate_ref}
        )[:16]
        process_observation = profile.get("probe_process")
        process_observation = (
            process_observation
            if isinstance(process_observation, Mapping)
            else {}
        )
        for root in probe_roots:
            root = Path(root)
            partial_run = _partial_probe_run(
                root, arm_id, process_observation
            )
            if partial_run is None:
                continue
            candidate_projection = _probe_telemetry_projection(partial_run)
            if (
                best_projection is None
                or (
                    best_projection.get("telemetry_present") is not True
                    and candidate_projection.get("telemetry_present") is True
                )
                or int(candidate_projection.get("completed_train_batches", 0))
                > int(best_projection.get("completed_train_batches", 0))
            ):
                best_projection = candidate_projection
            stall_signature = _zero_batch_training_stall_signature(
                root,
                arm_id,
                candidate_package_digest=candidate_package_digest,
                candidate_source_sha256=candidate_source_sha256,
            )
            if stall_signature is not None:
                signature_digest = sha256_digest(stall_signature)
                stall_signature_counts[signature_digest] = (
                    stall_signature_counts.get(signature_digest, 0) + 1
                )
    repeated_zero_batch_stall_observations = max(
        stall_signature_counts.values(), default=0
    )
    repeated_zero_batch_stall = (
        error_type == "PROBE_TIMEOUT"
        and repeated_zero_batch_stall_observations
        >= _DETERMINISTIC_ZERO_BATCH_STALL_MIN_OBSERVATIONS
    )
    if (
        not completed_resource_censor
        and not repeated_zero_batch_stall
        and (
            not isinstance(best_projection, Mapping)
            or best_projection.get("telemetry_present") is not True
            or int(best_projection.get("completed_train_batches", 0)) <= 0
        )
    ):
        return None

    completed = (
        int(best_projection.get("completed_train_batches", 0))
        if isinstance(best_projection, Mapping)
        else 0
    )
    deferred = tuple(
        canonical_value(
            {
                **dict(item),
                "resource_disposition": "RESOURCE_INFEASIBLE",
            }
        )
        for item in profile.get("deferred", ())
        if isinstance(item, Mapping)
    )
    if repeated_zero_batch_stall and isinstance(best_projection, Mapping):
        best_projection = canonical_value(
            {
                **dict(best_projection),
                "deterministic_stall_observations": (
                    repeated_zero_batch_stall_observations
                ),
            }
        )
    return canonical_value(
        {
            **dict(profile),
            "deferred": deferred,
            "failure_scope": "CANDIDATE_LOCAL",
            "probe": best_projection,
            **(
                {"status": "RESOURCE_INFEASIBLE"}
                if repeated_zero_batch_stall
                else {}
            ),
            "resource_probe_diagnostic": {
                **dict(diagnostic),
                "failure_scope": "CANDIDATE_LOCAL",
                **(
                    {"phase": "RESOURCE_PROBE_MEASURED_PROGRESS"}
                    if progressing_interrupted_recovery
                    else {}
                ),
                "message": (
                    "the same initialized candidate package stalled at epoch 0 "
                    "TRAIN phase start with zero completed batches in "
                    f"{repeated_zero_batch_stall_observations} independent fixed "
                    "resource probes; the candidate is infeasible under the "
                    "fixed probe deadline"
                    if repeated_zero_batch_stall
                    else "resource probe returned RESOURCE_CENSORED at the fixed "
                    "resource boundary; the candidate is infeasible under "
                    "the fixed probe budget"
                    if completed_resource_censor
                    else "an interrupted resource probe had already completed "
                    f"{completed} training batches before rematerialization; "
                    "repeating its physical worker cannot answer a new "
                    "resource question"
                    if progressing_interrupted_recovery
                    else "bounded resource probe timed out after completing "
                    f"{completed} training batches; measured candidate "
                    "throughput is infeasible for the fixed probe deadline"
                ),
            },
        }
    )


def _resource_probe_process_worker(
    connection: Any,
    request: Mapping[str, Any],
) -> None:
    try:
        result = _run_one(**dict(request))
    except BaseException as error:  # pragma: no cover - defensive crash boundary.
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


def _linux_descendant_pids(root_pid: int) -> tuple[int, ...]:
    """Return descendants of one known probe wrapper without scanning processes."""

    if platform.system() != "Linux" or root_pid <= 0:
        return ()
    pending = [root_pid]
    seen = {root_pid}
    descendants: list[int] = []
    while pending:
        parent_pid = pending.pop(0)
        children_path = Path(
            f"/proc/{parent_pid}/task/{parent_pid}/children"
        )
        try:
            child_tokens = children_path.read_text(encoding="utf-8").split()
        except OSError:
            continue
        for token in child_tokens:
            try:
                child_pid = int(token)
            except ValueError:
                continue
            if child_pid <= 0 or child_pid in seen:
                continue
            seen.add(child_pid)
            descendants.append(child_pid)
            pending.append(child_pid)
    return tuple(descendants)


def _terminate_process_tree(process: Any, *, grace_seconds: float = 5.0) -> None:
    """Stop a disposable probe and every worker it launched."""

    descendants = _linux_descendant_pids(int(process.pid or -1))
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # An unprivileged worker may sit below a root-owned sudo monitor.
            # Stop its user-owned leaf; the monitor reaps that process itself.
            pass
    # Let the trusted child reap its worker and release private mounts/SHM.
    # Killing it immediately after the leaf skips its normal cleanup.
    process.join(timeout=max(0.0, grace_seconds))
    if process.is_alive():
        process.terminate()
    process.join(timeout=max(0.0, grace_seconds))
    for pid in reversed(descendants):
        if not Path(f"/proc/{pid}").exists():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if process.is_alive():
        process.kill()
    process.join(timeout=max(0.0, grace_seconds))


def _run_disposable_resource_probe(
    request: Mapping[str, Any],
    *,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    start_method = "spawn"
    try:
        context = mp.get_context(start_method)
    except ValueError as error:
        return None, {
            "error_message": str(error)[:2000],
            "error_type": type(error).__name__,
            "process_isolated": False,
            "start_method": start_method,
            "status": "SPAWN_UNAVAILABLE",
        }, "PROBE_SPAWN_UNAVAILABLE"
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_resource_probe_process_worker,
        args=(child_connection, request),
    )
    started_ns = time.monotonic_ns()
    try:
        process.start()
    except Exception as error:
        child_connection.close()
        parent_connection.close()
        return None, {
            "error_message": str(error)[:2000],
            "error_type": type(error).__name__,
            "process_isolated": False,
            "start_method": start_method,
            "status": "START_FAILED",
        }, type(error).__name__
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
        _terminate_process_tree(process)
    else:
        process.join(timeout=5)
        if process.is_alive():
            _terminate_process_tree(process)
    observation = {
        "elapsed_wall_time_ms": max(
            1, (time.monotonic_ns() - started_ns) // 1_000_000
        ),
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
    # The reservation is caller-supplied sealed evidence, not child telemetry.
    # Keep that exact request-bound value on the parent observation so a child
    # result that omits it cannot silently turn an explicitly reserved probe
    # into an unreserved one. _validated_gpu_worker_probe still validates it
    # and compares it with any child-returned copy; invalid or inconsistent
    # evidence remains fail-closed.
    arm_input = request.get("arm_input")
    if isinstance(arm_input, Mapping):
        supplied_evidence = arm_input.get("gpu_reservation_evidence")
        if supplied_evidence is not None:
            observation["gpu_reservation_evidence"] = (
                canonical_value(dict(supplied_evidence))
                if isinstance(supplied_evidence, Mapping)
                else supplied_evidence
            )
        supplied_gpu_id = arm_input.get("gpu_id")
        if supplied_gpu_id is not None:
            direct_gpu_id = _validated_gpu_id(supplied_gpu_id)
            observation.update(
                {
                    "gpu_id": direct_gpu_id,
                    "physical_gpu_id": str(direct_gpu_id),
                    "selection_mode": DIRECT_GPU_SELECTION_MODE,
                }
            )
    if payload is not None and payload.get("kind") == "WORKER_EXCEPTION":
        observation.update(
            {
                "worker_error_message": str(payload.get("message", ""))[:2000],
                "worker_error_type": str(payload.get("error_type", "WorkerError")),
            }
        )
    if (
        payload is not None
        and payload.get("kind") == "RESULT"
        and isinstance(payload.get("result"), Mapping)
    ):
        return dict(payload["result"]), observation, None
    return None, observation, (
        "PROBE_TIMEOUT" if timed_out else "PROBE_PROCESS_FAILED"
    )


def _dispose_probe_checkpoints(
    probe_root: Path,
    expected_footprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Discard only probe-derived model state after sealing its resource facts."""

    runs_root = (probe_root / "runs").resolve()
    files = [
        path
        for checkpoint_root in runs_root.rglob("checkpoints")
        if checkpoint_root.is_dir()
        for path in checkpoint_root.rglob("*")
        if path.is_file()
    ]
    for path in files:
        try:
            path.resolve().relative_to(runs_root)
        except ValueError as error:
            raise ResourceSchedulingError(
                "probe checkpoint escaped the disposable runs root"
            ) from error
    observed_bytes = sum(path.stat().st_size for path in files)
    if (
        observed_bytes != expected_footprint.get("bytes")
        or len(files) != expected_footprint.get("file_count")
    ):
        raise ResourceSchedulingError(
            "probe checkpoint footprint changed before disposition"
        )
    for path in files:
        path.unlink()
    receipt = canonical_value(
        {
            "disposed_bytes": observed_bytes,
            "disposed_file_count": len(files),
            "retained_checkpoint_bytes": 0,
            "retention_reason": "RESOURCE_FACTS_SEALED_NO_SCIENTIFIC_METRIC",
            "schema": "recclaw.resource-probe-checkpoint-disposition.v1",
        }
    )
    receipt_path = probe_root / "PROBE_CHECKPOINT_DISPOSITION.json"
    _write_new_json(receipt_path, receipt)
    return canonical_value(
        {**receipt, "receipt_sha256": bytes_sha256(receipt_path.read_bytes())}
    )


def _dispose_observed_probe_checkpoints(probe_root: Path) -> dict[str, Any] | None:
    """Dispose partial probe state when no trusted worker footprint returned."""

    runs_root = (probe_root / "runs").resolve()
    files = [
        path
        for checkpoint_root in runs_root.rglob("checkpoints")
        if checkpoint_root.is_dir()
        for path in checkpoint_root.rglob("*")
        if path.is_file()
    ]
    if not files:
        return None
    return _dispose_probe_checkpoints(
        probe_root,
        {
            "bytes": sum(path.stat().st_size for path in files),
            "file_count": len(files),
        },
    )


def _seal_interrupted_probe_checkpoint_disposition(
    probe_root: Path,
) -> dict[str, Any]:
    """Idempotently dispose probe-only model state after an interrupted parent."""

    receipt_path = probe_root / "PROBE_CHECKPOINT_DISPOSITION.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema")
            != "recclaw.resource-probe-checkpoint-disposition.v1"
            or receipt.get("retained_checkpoint_bytes") != 0
        ):
            raise ResourceSchedulingError(
                "existing probe checkpoint disposition receipt is invalid"
            )
        return canonical_value(
            {**dict(receipt), "receipt_sha256": bytes_sha256(receipt_path.read_bytes())}
        )
    observed = _dispose_observed_probe_checkpoints(probe_root)
    if observed is not None:
        return observed
    return _dispose_probe_checkpoints(
        probe_root,
        {"bytes": 0, "file_count": 0},
    )


def _probe_initialization_can_resume(probe_root: Path) -> bool:
    """An initialized directory is not evidence that training was started."""

    if not (probe_root / "FIXED_BATCH_PREFIX_CONTRACT.json").is_file():
        return False
    receipt_path = probe_root / "PROBE_CHECKPOINT_DISPOSITION.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("disposed_bytes") != 0 or receipt.get("disposed_file_count") != 0:
            return False
    runs_root = probe_root / "runs"
    for path in runs_root.rglob("*"):
        if (
            path.match("start_gate*.json")
            or path.name in {"worker_result.json", "resource_telemetry.json"}
            or (path.name == "checkpoints" and any(path.iterdir()))
        ):
            return False
    for worker_root in runs_root.glob("experiments/*/worker"):
        try:
            _initialization_resume_suffix(worker_root)
        except FreshR1Error:
            return False
    return True


def _probe_behavior_recipe(execution_recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Execution inputs, excluding materialization receipts and directory names."""

    return canonical_value({key: execution_recipe.get(key) for key in (
        "base_model_config", "capability_family", "config", "dataset", "entrypoint",
        "entrypoint_source_sha256", "candidate_source_content_digest", "candidate_source_tree_digest",
        "evaluator", "execution_role", "model", "split",
    )})


def _probe_environment_identity() -> Mapping[str, Any]:
    dependencies = {}
    for package in ("torch", "numpy", "scipy", "recbole"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    source_root = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "worker_python": str(PYTHON_EXECUTABLE),
        "dependencies": dependencies,
        "runtime_sources": {
            name: bytes_sha256((source_root / name).read_bytes())
            for name in ("resource_scheduling.py", "fresh_r1.py", "epoch_sampler_scaffold.py", "compiled_efficiency_kernels.py")
        },
    }


def _completed_probe_request_identity(
    *,
    arm_id: str,
    candidate_ref: str,
    candidate_package_digest: str | None,
    candidate_binding_digest: str | None,
    compute_pattern: str | None,
    entrypoint: str,
    source_sha256: str,
    execution_recipe: Mapping[str, Any],
    probe_seed: int,
    probe_timeout_seconds: int,
    total_budget_seconds: int,
    cuda_visible_devices: str | None,
    gpu_id: int | None,
    offline_calibration_probe_excluded_from_future_budget: bool,
    expected_recbole_source_tree_digest: str,
) -> dict[str, Any]:
    """Bind a completed physical probe to the request that may replay it."""

    return canonical_value(
        {
            "compute_pattern": compute_pattern,
            "cuda_visible_devices": cuda_visible_devices,
            "entrypoint": entrypoint,
            "execution_recipe_digest": sha256_digest(_probe_behavior_recipe(execution_recipe)),
            "runtime_environment": _probe_environment_identity(),
            "gpu_id": gpu_id,
            "offline_calibration_probe_excluded_from_future_budget": (
                offline_calibration_probe_excluded_from_future_budget
            ),
            "probe_seed": probe_seed,
            "probe_timeout_seconds": probe_timeout_seconds,
            "recbole_source_tree_digest": expected_recbole_source_tree_digest,
            "source_sha256": source_sha256,
            "total_budget_seconds": total_budget_seconds,
        }
    )


def _completed_probe_artifact_paths(
    probe_root: Path,
    arm_id: str,
) -> dict[str, Path]:
    run_root = probe_root / "runs" / "experiments" / arm_id.replace("_", "-")
    worker_root = run_root / "worker"
    return {
        "fixed_batch_prefix_contract": (
            probe_root / "FIXED_BATCH_PREFIX_CONTRACT.json"
        ),
        "experiment_binding": run_root / "experiment_binding.json",
        "recbole_source_identity": run_root / "recbole_source_identity.json",
        "worker_result": worker_root / "worker_result.json",
        "resource_telemetry": worker_root / "resource_telemetry.json",
    }


def _completed_probe_artifact_digests(
    probe_root: Path,
    arm_id: str,
) -> dict[str, str] | None:
    try:
        return canonical_value(
            {
                name: bytes_sha256(path.read_bytes())
                for name, path in _completed_probe_artifact_paths(
                    probe_root, arm_id
                ).items()
            }
        )
    except OSError:
        return None


def _completed_probe_replay_candidate_root(
    probe_root: Path,
    arm_id: str,
    *,
    entrypoint: str,
    source_sha256: str,
    execution_recipe: Mapping[str, Any],
    probe_seed: int,
    probe_timeout_seconds: int,
    expected_prefix_contract: Mapping[str, Any],
    expected_recbole_source_tree_digest: str,
) -> Path | None:
    """Return the historical root only for one exactly rehydratable probe."""

    run_root = probe_root / "runs" / "experiments" / arm_id.replace("_", "-")
    prefix_path = probe_root / "FIXED_BATCH_PREFIX_CONTRACT.json"
    binding_path = run_root / "experiment_binding.json"
    recbole_identity_path = run_root / "recbole_source_identity.json"
    worker_path = run_root / "worker" / "worker_result.json"
    telemetry_path = run_root / "worker" / "resource_telemetry.json"
    try:
        prefix_contract = json.loads(prefix_path.read_text(encoding="utf-8"))
        binding = ExperimentBindingV1.from_canonical_dict(
            json.loads(binding_path.read_text(encoding="utf-8"))
        )
        recbole_identity = json.loads(
            recbole_identity_path.read_text(encoding="utf-8")
        )
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        if not isinstance(worker, Mapping):
            return None
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ExperimentBindingError, AttributeError):
        return None
    if (
        not isinstance(prefix_contract, Mapping)
        or not isinstance(recbole_identity, Mapping)
        or not isinstance(telemetry, Mapping)
        or canonical_value(prefix_contract) != canonical_value(expected_prefix_contract)
    ):
        return None
    prefix_digest = bytes_sha256(prefix_path.read_bytes())
    telemetry_prefix = telemetry.get("prefix_contract")
    if (
        binding.prefix_contract_digest != prefix_digest
        or not isinstance(telemetry_prefix, Mapping)
        or telemetry_prefix.get("contract_file_sha256") != prefix_digest
    ):
        return None
    resource_prediction = execution_recipe.get("resource_prediction")
    if resource_prediction is not None and not isinstance(resource_prediction, Mapping):
        return None
    try:
        expected_deadline = resolve_candidate_deadline_seconds(
            default_seconds=probe_timeout_seconds,
            prediction=resource_prediction,
            final_worker_ceiling_seconds=MAX_WORKER_CEILING_SECONDS,
        )
    except FreshR1Error:
        return None
    expected_binding = {
        "run_id": arm_id.replace("_", "-"),
        "seed": probe_seed,
        "epochs": PROBE_EPOCHS,
        "timeout_seconds": expected_deadline,
        "execution_purpose": "RESOURCE_PROBE_ONLY",
        "resource_telemetry": True,
        "watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
        "entrypoint": entrypoint,
        "entrypoint_source_sha256": source_sha256,
    }
    if any(
        getattr(binding, field_name) != expected_value
        for field_name, expected_value in expected_binding.items()
    ):
        return None
    if _probe_behavior_recipe(binding.canonical_dict()) != _probe_behavior_recipe(execution_recipe):
        return None
    candidate_root_path = binding.candidate_root_path
    source_tree_digest = recbole_identity.get("source_tree_digest")
    try:
        validate_sha256(source_tree_digest, field_name="recbole_source_tree_digest")
        validate_sha256(
            expected_recbole_source_tree_digest,
            field_name="expected_recbole_source_tree_digest",
        )
    except ValueError:
        return None
    if (
        source_tree_digest != expected_recbole_source_tree_digest
        or
        worker.get("exit_status") != "SUCCESS"
        or not isinstance(candidate_root_path, str)
        or not Path(candidate_root_path).is_absolute()
    ):
        return None
    return Path(candidate_root_path)


def _seal_completed_probe_outer_envelope(
    probe_root: Path,
    arm_id: str,
    *,
    request_identity: Mapping[str, Any],
    probe_run: Mapping[str, Any],
    process_observation: Mapping[str, Any],
) -> str:
    """Persist the original completed GPU/process facts before cleanup."""

    artifacts = _completed_probe_artifact_digests(probe_root, arm_id)
    projected_run = _resource_probe_input(probe_run)
    if artifacts is None:
        raise ResourceSchedulingError(
            "completed probe artifacts are unavailable for outer-envelope sealing"
        )
    if projected_run.get("exit_status") != "SUCCESS":
        raise ResourceSchedulingError(
            "only a successful physical probe may seal a completed outer envelope"
        )
    if (
        projected_run.get("result_sha256") != artifacts["worker_result"]
        or projected_run.get("resource_telemetry_sha256")
        != artifacts["resource_telemetry"]
    ):
        raise ResourceSchedulingError(
            "completed probe result or telemetry digest changed before envelope sealing"
        )
    if not (
        process_observation.get("process_isolated") is True
        and process_observation.get("start_method") == "spawn"
        and process_observation.get("status") == "RESULT"
        and process_observation.get("exit_code") == 0
    ):
        raise ResourceSchedulingError(
            "completed probe lacks the successful original disposable process evidence"
        )
    envelope = canonical_value(
        {
            "arm_id": arm_id,
            "artifact_sha256": artifacts,
            "probe_process": dict(process_observation),
            "probe_run": projected_run,
            "request_identity": dict(request_identity),
            "schema": "recclaw.completed-resource-probe-outer-envelope.v1",
        }
    )
    return _write_new_json(
        probe_root / COMPLETED_PROBE_OUTER_ENVELOPE_FILENAME,
        envelope,
    )


def resource_probe_execution_key(execution_recipe: Mapping[str, Any]) -> str:
    """Identify the executable probe, independently of materialization receipts.

    A mechanism can survive an implementation or environment repair.  Its old
    failed physical probe must not prevent the repaired execution from running.
    """

    repo_root = Path(__file__).resolve().parents[4]
    fields = (
        "model", "base_model_config", "config", "entrypoint",
        "entrypoint_source_sha256", "candidate_source_content_digest",
        "dataset", "split", "evaluator",
    )
    return sha256_digest({
        "execution": {key: execution_recipe.get(key) for key in fields},
        "python_executable": str(PYTHON_EXECUTABLE),
        "worker_environment": _worker_environment(SimpleNamespace(environment={}), None),
        "worker_source": bytes_sha256((repo_root / "scripts/campaign_train_worker.py").read_bytes()),
        "environment_builder_source": bytes_sha256(Path(__file__).with_name("fresh_r1.py").read_bytes()),
    })


def _completed_probe_replay_evidence(
    probe_root: Path,
    arm_id: str,
    *,
    request_identity: Mapping[str, Any],
    entrypoint: str,
    source_sha256: str,
    execution_recipe: Mapping[str, Any],
    probe_seed: int,
    probe_timeout_seconds: int,
    expected_prefix_contract: Mapping[str, Any],
    expected_recbole_source_tree_digest: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Load only a byte-bound original outer envelope; never re-time replay."""

    candidate_root = _completed_probe_replay_candidate_root(
        probe_root,
        arm_id,
        entrypoint=entrypoint,
        source_sha256=source_sha256,
        execution_recipe=execution_recipe,
        probe_seed=probe_seed,
        probe_timeout_seconds=probe_timeout_seconds,
        expected_prefix_contract=expected_prefix_contract,
        expected_recbole_source_tree_digest=(
            expected_recbole_source_tree_digest
        ),
    )
    if candidate_root is None:
        return None
    envelope_path = probe_root / COMPLETED_PROBE_OUTER_ENVELOPE_FILENAME
    try:
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    artifacts = _completed_probe_artifact_digests(probe_root, arm_id)
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("schema")
        != "recclaw.completed-resource-probe-outer-envelope.v1"
        or envelope.get("arm_id") != arm_id
        or canonical_value(envelope.get("request_identity"))
        != canonical_value(request_identity)
        or artifacts is None
        or canonical_value(envelope.get("artifact_sha256")) != artifacts
    ):
        return None
    probe_run = envelope.get("probe_run")
    process_observation = envelope.get("probe_process")
    if not isinstance(probe_run, Mapping) or not isinstance(
        process_observation, Mapping
    ):
        return None
    binding = probe_run.get("experiment_binding")
    if (
        probe_run.get("exit_status") != "SUCCESS"
        or probe_run.get("result_sha256") != artifacts["worker_result"]
        or probe_run.get("resource_telemetry_sha256")
        != artifacts["resource_telemetry"]
        or not isinstance(binding, Mapping)
        or binding.get("run_id") != arm_id.replace("_", "-")
        or process_observation.get("process_isolated") is not True
        or process_observation.get("start_method") != "spawn"
        or process_observation.get("status") != "RESULT"
        or process_observation.get("exit_code") != 0
    ):
        return None
    return (
        candidate_root,
        canonical_value(dict(probe_run)),
        canonical_value(dict(process_observation)),
    )


def run_disposable_fixed_batch_resource_probe(
    repo_root: Path,
    *,
    candidate_root: Path | None,
    source_path: Path,
    entrypoint: str,
    source_sha256: str,
    execution_recipe: Mapping[str, Any],
    probe_root: Path,
    candidate_ref: str = "candidate",
    candidate_package_digest: str | None = None,
    candidate_binding_digest: str | None = None,
    total_budget_seconds: int = CAMPAIGN_TOTAL_BUDGET_SECONDS,
    probe_seed: int = TRAINING_SEED,
    probe_timeout_seconds: int = PROBE_TIMEOUT_SECONDS,
    compute_pattern: str | None = None,
    cuda_visible_devices: str | None = None,
    gpu_id: int | None = None,
    gpu_reservation_evidence: Mapping[str, Any] | None = None,
    search_data_identity: Mapping[str, Any] | None = None,
    offline_calibration_probe_excluded_from_future_budget: bool = False,
    campaign_checkpoint_horizon_slots: int = CAMPAIGN_CHECKPOINT_HORIZON_SLOTS,
    process_launcher: Any = None,
    probe_executor: Any = None,
) -> dict[str, Any]:
    """Run the existing fixed-batch probe in a disposable process.

    The caller supplies the explicit execution recipe already bound by the
    Research Innovation package.  No generic BPR fallback is synthesized.  The
    optional ``cuda_visible_devices`` and sealed
    ``gpu_reservation_evidence`` values are passed through to the disposable
    physical worker; they do not allocate a lease.  The returned profile
    contains only resource prediction/admission facts.
    """

    validated_gpu_id = _validate_gpu_selection_arguments(
        cuda_visible_devices=cuda_visible_devices,
        gpu_id=gpu_id,
    )
    repo_root = repo_root.resolve()
    if candidate_root is not None:
        candidate_root = candidate_root.resolve()
    source_path = source_path.resolve()
    probe_root = probe_root.resolve()
    if (candidate_root is not None and not candidate_root.is_dir()) or not source_path.is_file():
        raise ResourceSchedulingError("candidate package source is unavailable")
    if candidate_root is not None:
        try:
            source_path.relative_to(candidate_root)
        except ValueError as error:
            raise ResourceSchedulingError(
                "candidate source must remain inside candidate_root"
            ) from error
    if not isinstance(execution_recipe, Mapping):
        raise ResourceSchedulingError("an explicit execution_recipe is required")
    try:
        validate_sha256(source_sha256, field_name="source_sha256")
    except ValueError as error:
        raise ResourceSchedulingError(str(error)) from error
    try:
        validate_execution_recipe(execution_recipe)
    except Exception as error:
        raise ResourceSchedulingError(
            f"execution_recipe is invalid: {error}"
        ) from error
    if gpu_reservation_evidence is not None and not isinstance(
        gpu_reservation_evidence, Mapping
    ):
        raise ResourceSchedulingError(
            "gpu_reservation_evidence must be a mapping"
        )
    if (
        isinstance(campaign_checkpoint_horizon_slots, bool)
        or not isinstance(campaign_checkpoint_horizon_slots, int)
        or campaign_checkpoint_horizon_slots <= 0
    ):
        raise ResourceSchedulingError(
            "campaign_checkpoint_horizon_slots must be a positive integer"
        )
    try:
        expected_recbole_source_tree_digest = recbole_source_identity(
            RECBole_ROOT
        )["source_tree_digest"]
    except (FreshR1Error, KeyError, OSError) as error:
        raise ResourceSchedulingError(
            f"current RecBole source identity is unavailable: {error}"
        ) from error
    arm_id = "candidate_" + sha256_digest({"candidate_ref": candidate_ref})[:16]
    prefix_contract_path = probe_root / "FIXED_BATCH_PREFIX_CONTRACT.json"
    expected_prefix_contract = build_fixed_batch_prefix_contract(
        seed=probe_seed,
        dataset=execution_recipe["dataset"],
        execution_recipe=execution_recipe,
    )
    completed_request_identity = _completed_probe_request_identity(
        arm_id=arm_id,
        candidate_ref=candidate_ref,
        candidate_package_digest=candidate_package_digest,
        candidate_binding_digest=candidate_binding_digest,
        compute_pattern=compute_pattern,
        entrypoint=entrypoint,
        source_sha256=source_sha256,
        execution_recipe=execution_recipe,
        probe_seed=probe_seed,
        probe_timeout_seconds=probe_timeout_seconds,
        total_budget_seconds=total_budget_seconds,
        cuda_visible_devices=cuda_visible_devices,
        gpu_id=validated_gpu_id,
        offline_calibration_probe_excluded_from_future_budget=(
            offline_calibration_probe_excluded_from_future_budget
        ),
        expected_recbole_source_tree_digest=(
            expected_recbole_source_tree_digest
        ),
    )
    envelope_path = probe_root / COMPLETED_PROBE_OUTER_ENVELOPE_FILENAME
    if envelope_path.is_file():
        try:
            envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            envelope = None
        if isinstance(envelope, Mapping) and envelope.get("request_identity") == completed_request_identity:
            # Reuse the actual historical physical attempt, while the returned
            # profile below is rebound to this request's current capability.
            arm_id = str(envelope["arm_id"])
    completed_artifact_root = (
        _completed_probe_replay_candidate_root(
            probe_root,
            arm_id,
            entrypoint=entrypoint,
            source_sha256=source_sha256,
            execution_recipe=execution_recipe,
            probe_seed=probe_seed,
            probe_timeout_seconds=probe_timeout_seconds,
            expected_prefix_contract=expected_prefix_contract,
            expected_recbole_source_tree_digest=(
                expected_recbole_source_tree_digest
            ),
        )
        if probe_root.is_dir()
        else None
    )
    completed_replay_evidence = (
        _completed_probe_replay_evidence(
            probe_root,
            arm_id,
            request_identity=completed_request_identity,
            entrypoint=entrypoint,
            source_sha256=source_sha256,
            execution_recipe=execution_recipe,
            probe_seed=probe_seed,
            probe_timeout_seconds=probe_timeout_seconds,
            expected_prefix_contract=expected_prefix_contract,
            expected_recbole_source_tree_digest=(
                expected_recbole_source_tree_digest
            ),
        )
        if probe_root.is_dir()
        else None
    )
    completed_probe_replay = completed_replay_evidence is not None
    legacy_interval_unavailable = (
        completed_artifact_root is not None
        and completed_replay_evidence is None
    )
    resume_initialization = (
        probe_root.is_dir()
        and not completed_probe_replay
        and _probe_initialization_can_resume(probe_root)
    )
    if probe_root.exists() and not completed_probe_replay and not resume_initialization:
        disposition = _seal_interrupted_probe_checkpoint_disposition(probe_root)
        reason_code = (
            "RESOURCE_PROBE_LEGACY_INTERVAL_UNAVAILABLE"
            if legacy_interval_unavailable
            else "RESOURCE_PROBE_INTERRUPTED_RECOVERY"
        )
        process_observation = {
            "exit_code": None,
            "process_isolated": True,
            "start_method": "checkpoint_resume",
            "status": "INTERRUPTED_RECOVERY",
        }
        partial_probe_run = _relocated_partial_probe_run(
            probe_root,
            current_arm_id=arm_id,
            candidate_source_sha256=source_sha256,
            execution_recipe=execution_recipe,
            probe_seed=probe_seed,
            probe_timeout_seconds=probe_timeout_seconds,
            process_observation=process_observation,
        )
        profile = _failed_innovation_resource_profile(
            candidate_ref=candidate_ref,
            candidate_package_digest=candidate_package_digest,
            candidate_binding_digest=candidate_binding_digest,
            candidate_source_sha256=source_sha256,
            compute_pattern=compute_pattern,
            process_observation=process_observation,
            reason_code=reason_code,
            failure_phase="RESOURCE_PROBE_RECOVERY",
            error_type=reason_code,
            failure_scope="RECOVERY",
            error_message=(
                "completed legacy worker lacks its original durable GPU/process "
                "interval; no physical worker or synthetic timing was used"
                if legacy_interval_unavailable
                else "sealed an interrupted probe root without repeating its "
                "physical worker"
            ),
            probe_run=partial_probe_run,
        )
        profile = reclassify_progressing_timeout_profile(profile) or profile
        return canonical_value(
            {**profile, "probe_checkpoint_disposition": disposition}
        )
    if completed_probe_replay or resume_initialization:
        try:
            existing_prefix_contract = json.loads(
                prefix_contract_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ResourceSchedulingError(
                "existing probe prefix contract is unreadable"
            ) from error
        if canonical_value(existing_prefix_contract) != canonical_value(
            expected_prefix_contract
        ):
            raise ResourceSchedulingError(
                "existing probe prefix contract does not match the recovery request"
            )
    else:
        probe_root.mkdir(parents=True)
        _write_new_json(prefix_contract_path, expected_prefix_contract)
    if resume_initialization:
        # Retain the old zero-work disposition; the resumed probe will seal its
        # own actual checkpoint footprint at the normal disposition path.
        receipt_path = probe_root / "PROBE_CHECKPOINT_DISPOSITION.json"
        if receipt_path.is_file():
            index = 1
            while (probe_root / f"PROBE_INITIALIZATION_DISPOSITION_{index:02d}.json").exists():
                index += 1
            receipt_path.rename(probe_root / f"PROBE_INITIALIZATION_DISPOSITION_{index:02d}.json")
    arm_input = {
        "candidate_root": candidate_root,
        "candidate_binding_digest": candidate_binding_digest,
        "entrypoint": entrypoint,
        "execution_recipe": canonical_value(dict(execution_recipe)),
        "expected_recbole_source_tree_digest": (
            expected_recbole_source_tree_digest
        ),
        "compute_pattern": compute_pattern,
        "cuda_visible_devices": cuda_visible_devices,
        "gpu_id": validated_gpu_id,
        "gpu_reservation_evidence": (
            canonical_value(dict(gpu_reservation_evidence))
            if isinstance(gpu_reservation_evidence, Mapping)
            else gpu_reservation_evidence
        ),
        "search_data_identity": (
            canonical_value(dict(search_data_identity))
            if isinstance(search_data_identity, Mapping)
            else search_data_identity
        ),
        "seed": probe_seed,
        "source_sha256": source_sha256,
    }
    request = {
        "arm": arm_id,
        "arm_input": arm_input,
        "authority": "user-delegated-research-innovation-resource-probe",
        "epochs": PROBE_EPOCHS,
        "prefix_contract_path": prefix_contract_path,
        "purpose": "RESOURCE_PROBE_ONLY",
        "repo_root": repo_root,
        "run_identity": "research-innovation-resource-probe-v1",
        "side_root": probe_root / "runs",
        "timeout_seconds": probe_timeout_seconds,
        **({"process_launcher": process_launcher} if process_launcher is not None else {}),
    }
    if completed_replay_evidence is not None:
        probe_run = completed_replay_evidence[1]
        process_observation = completed_replay_evidence[2]
        failure = None
    else:
        try:
            probe_run, process_observation, failure = (
                (probe_executor or _run_disposable_resource_probe)(
                    request,
                    timeout_seconds=max(1, int(probe_timeout_seconds) + 60),
                )
            )
        except BaseException:
            _seal_interrupted_probe_checkpoint_disposition(probe_root)
            raise
    if probe_run is None:
        process_status = str(process_observation.get("status") or "")
        partial_probe_run = _partial_probe_run(
            probe_root,
            arm_id,
            process_observation,
        )
        profile = _failed_innovation_resource_profile(
            candidate_ref=candidate_ref,
            candidate_package_digest=candidate_package_digest,
            candidate_binding_digest=candidate_binding_digest,
            candidate_source_sha256=source_sha256,
            compute_pattern=compute_pattern,
            process_observation=process_observation,
            reason_code=failure or "PROBE_PROCESS_FAILED",
            error_type=(
                str(process_observation.get("worker_error_type"))
                if process_observation.get("worker_error_type") is not None
                else failure
            ),
            error_message=(
                str(process_observation.get("worker_error_message"))
                if process_observation.get("worker_error_message") is not None
                else process_observation.get("error_message")
            ),
            failure_scope=(
                "SHARED_INFRASTRUCTURE"
                if process_status in {"SPAWN_UNAVAILABLE", "START_FAILED"}
                else "WORKER_TRANSIENT"
            ),
            probe_run=partial_probe_run,
        )
        profile = (
            reclassify_progressing_timeout_profile(profile)
            or profile
        )
        disposition = _dispose_observed_probe_checkpoints(probe_root)
        return canonical_value(
            {**profile, "probe_checkpoint_disposition": disposition}
            if disposition is not None
            else profile
        )
    probe_run = dict(probe_run)
    for identity_name, identity_value in (
        ("candidate_binding_digest", candidate_binding_digest),
        ("candidate_package_digest", candidate_package_digest),
        ("candidate_ref", candidate_ref),
        ("candidate_source_sha256", source_sha256),
        ("compute_pattern", compute_pattern),
    ):
        if completed_probe_replay:
            # The immutable envelope retains the original physical identity;
            # this derived admission profile belongs to the current wrapper.
            probe_run[identity_name] = identity_value
        else:
            probe_run.setdefault(identity_name, identity_value)
    checkpoint_footprint = probe_run.get("checkpoint_footprint")
    recovered_disposition: dict[str, Any] | None = None
    if completed_probe_replay:
        recovered_disposition = _seal_interrupted_probe_checkpoint_disposition(
            probe_root
        )
        checkpoint_bytes = (
            checkpoint_footprint.get("bytes")
            if isinstance(checkpoint_footprint, Mapping)
            else None
        )
        if not isinstance(checkpoint_bytes, int) or checkpoint_bytes <= 0:
            disposed_bytes = recovered_disposition.get("disposed_bytes")
            disposed_file_count = recovered_disposition.get("disposed_file_count")
            if (
                isinstance(disposed_bytes, bool)
                or not isinstance(disposed_bytes, int)
                or disposed_bytes <= 0
                or isinstance(disposed_file_count, bool)
                or not isinstance(disposed_file_count, int)
                or disposed_file_count <= 0
            ):
                raise ResourceSchedulingError(
                    "completed probe disposition lacks a positive historical "
                    "checkpoint footprint"
                )
            checkpoint_footprint = canonical_value(
                {
                    "bytes": disposed_bytes,
                    "file_count": disposed_file_count,
                    "source": "PROBE_CHECKPOINT_DISPOSITION",
                }
            )
            probe_run["checkpoint_footprint"] = checkpoint_footprint
    if isinstance(checkpoint_footprint, Mapping):
        checkpoint_bytes = checkpoint_footprint.get("bytes")
        if isinstance(checkpoint_bytes, int) and checkpoint_bytes > 0:
            prior_storage_observation = probe_run.get("storage_observation")
            replay_available_bytes = (
                prior_storage_observation.get(
                    "available_after_probe_cleanup_bytes"
                )
                if completed_probe_replay
                and isinstance(prior_storage_observation, Mapping)
                else None
            )
            current_available_bytes = shutil.disk_usage(probe_root).free
            storage_available_after_cleanup_bytes = (
                replay_available_bytes
                if isinstance(replay_available_bytes, int)
                and replay_available_bytes > 0
                else current_available_bytes
                if completed_probe_replay
                else current_available_bytes + checkpoint_bytes
            )
            probe_run["storage_observation"] = canonical_value(
                {
                    "available_after_probe_cleanup_bytes": (
                        storage_available_after_cleanup_bytes
                    ),
                    "campaign_checkpoint_horizon_slots": (
                        campaign_checkpoint_horizon_slots
                    ),
                    "checkpoint_storage_bytes": checkpoint_bytes,
                }
            )
    if probe_run.get("exit_status") != "SUCCESS":
        error_type = str(
            probe_run.get("worker_error_type")
            or probe_run.get("error_type")
            or probe_run.get("exit_status")
            or "RESOURCE_PROBE_RUN_FAILED"
        )
        error_message = str(
            probe_run.get("worker_error_message")
            or probe_run.get("error_message")
            or error_type
        )
        profile = _failed_innovation_resource_profile(
            candidate_ref=candidate_ref,
            candidate_package_digest=candidate_package_digest,
            candidate_binding_digest=candidate_binding_digest,
            candidate_source_sha256=source_sha256,
            compute_pattern=compute_pattern,
            process_observation=process_observation,
            reason_code=error_type.upper(),
            failure_phase="RESOURCE_PROBE_EXECUTION",
            error_type=error_type,
            error_message=error_message,
            failure_scope=_probe_failure_scope(probe_run),
            probe_run=probe_run,
        )
        profile = reclassify_progressing_timeout_profile(profile) or profile
        disposition = _dispose_observed_probe_checkpoints(probe_root)
        return canonical_value(
            {**profile, "probe_checkpoint_disposition": disposition}
            if disposition is not None
            else profile
        )
    if not completed_probe_replay:
        _seal_completed_probe_outer_envelope(
            probe_root,
            arm_id,
            request_identity=completed_request_identity,
            probe_run=probe_run,
            process_observation=process_observation,
        )
    try:
        profile = build_innovation_resource_profile(
            candidate_ref=candidate_ref,
            candidate_package_digest=candidate_package_digest,
            candidate_source_sha256=source_sha256,
            candidate_binding_digest=candidate_binding_digest,
            gpu_id=validated_gpu_id,
            source_features=structural_features(source_path),
            probe_run=probe_run,
            process_observation=process_observation,
            total_budget_seconds=total_budget_seconds,
            execution_recipe=execution_recipe,
            probe_seed=probe_seed,
            compute_pattern=compute_pattern,
            offline_calibration_probe_excluded_from_future_budget=(
                offline_calibration_probe_excluded_from_future_budget
            ),
        )
        disposition = recovered_disposition or _dispose_probe_checkpoints(
            probe_root, checkpoint_footprint
        )
        return canonical_value(
            {**dict(profile), "probe_checkpoint_disposition": disposition}
        )
    except ResourceSchedulingError as error:
        profile = _failed_innovation_resource_profile(
            candidate_ref=candidate_ref,
            candidate_package_digest=candidate_package_digest,
            candidate_source_sha256=source_sha256,
            candidate_binding_digest=candidate_binding_digest,
            compute_pattern=compute_pattern,
            process_observation=process_observation,
            reason_code=type(error).__name__.upper(),
            failure_phase="POST_PROBE_RESOURCE_VALIDATION",
            error_type=type(error).__name__,
            error_message=str(error),
            failure_scope="CANDIDATE_LOCAL",
        )
        disposition = _dispose_observed_probe_checkpoints(probe_root)
        return canonical_value(
            {**profile, "probe_checkpoint_disposition": disposition}
            if disposition is not None
            else profile
        )


def _run_one(
    *,
    repo_root: Path,
    side_root: Path,
    arm: str,
    arm_input: Mapping[str, Any],
    epochs: int,
    purpose: str,
    timeout_seconds: int,
    prefix_contract_path: Path | None = None,
    run_identity: str = Q0R_RUN_IDENTITY,
    authority: str = "user-delegated-q0r-resource-scheduling",
    process_launcher: Any = None,
) -> dict[str, Any]:
    try:
        expected_recbole_source_tree_digest = arm_input.get(
            "expected_recbole_source_tree_digest"
        )
        if expected_recbole_source_tree_digest is not None:
            validate_sha256(
                expected_recbole_source_tree_digest,
                field_name="expected_recbole_source_tree_digest",
            )
        return run_development_training(
            repo_root=repo_root,
            side_root=side_root,
            run_id=arm.replace("_", "-"),
            seed=int(arm_input.get("seed", TRAINING_SEED)),
            candidate_root=arm_input["candidate_root"],
            entrypoint=str(arm_input["entrypoint"]),
            source_sha256=str(arm_input["source_sha256"]),
            expected_recbole_source_tree_digest=(
                expected_recbole_source_tree_digest
            ),
            run_identity=run_identity,
            authority=authority,
            timeout_seconds=timeout_seconds,
            recbole_commit_identity="7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
            epochs=epochs,
            execution_purpose=purpose,
            resource_telemetry=True,
            watchdog_seconds=ENGINEERING_WATCHDOG_SECONDS,
            prefix_contract_path=prefix_contract_path,
            execution_recipe=arm_input.get("execution_recipe"),
            cuda_visible_devices=arm_input.get("cuda_visible_devices"),
            gpu_id=arm_input.get("gpu_id"),
            gpu_reservation_evidence=arm_input.get("gpu_reservation_evidence"),
            search_data_identity=arm_input.get("search_data_identity"),
            **({"process_launcher": process_launcher} if process_launcher is not None else {}),
        )
    except Exception as error:  # one physical arm must not abort the campaign
        return canonical_value(
            {
                "error_message": str(error),
                "error_type": type(error).__name__,
                "exit_status": "INTERFACE_OR_RESOURCE_FAILURE",
                "epochs_requested": epochs,
                "metrics": {},
                "resource_telemetry": None,
                "mechanism_effect_update_allowed": False,
                "resource_deadline_seconds": timeout_seconds,
                "wall_time_ms": 0,
                "watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
            }
        )


def _missingness(run: Mapping[str, Any]) -> dict[str, Any]:
    if run.get("exit_status") == "SUCCESS":
        return {"missing": False, "reason": None, "resource_disposition": None}
    if run.get("exit_status") == "RESOURCE_CENSORED":
        reason = run.get("censoring_trigger") or "RESOURCE_CENSORED"
        disposition = "RESOURCE_DEFERRED"
    elif run.get("worker_error_type") in {
        "OutOfMemoryError",
        "AcceleratorError",
        "torch.OutOfMemoryError",
    }:
        reason = run.get("worker_error_type")
        disposition = "RESOURCE_DEFERRED"
    else:
        reason = run.get("worker_error_type") or run.get("error_type") or "RUNTIME_FAILURE"
        disposition = "RESOURCE_INFEASIBLE"
    return {
        "missing": True,
        "reason": reason,
        "resource_disposition": disposition,
    }


def plan_resource_admission(
    repo_root: Path,
    *,
    campaign_root: Path,
    q0_external_receipt_path: Path,
) -> dict[str, Any]:
    """Validate accepted evidence and freeze Q0R2 admission before outcomes."""

    repo_root = repo_root.resolve()
    campaign_root = campaign_root.resolve()
    if campaign_root.exists():
        raise ResourceSchedulingError(f"Q0R2 root already exists: {campaign_root}")
    q0_repo, q0_external = _validate_q0_receipts(
        repo_root, q0_external_receipt_path.resolve()
    )
    q0r_path = (
        repo_root
        / "docs/research_line/vnext/"
        "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
    )
    if bytes_sha256(q0r_path.read_bytes()) != Q0R_V3_REPO_RECEIPT_SHA256:
        raise ResourceSchedulingError("sealed Q0R v3 repository receipt byte drift")
    q0r_receipt = json.loads(q0r_path.read_text(encoding="utf-8"))
    if (
        q0r_receipt.get("status") != "HARD_BLOCK"
        or q0r_receipt.get("held_out_reads") != 0
        or q0r_receipt.get("full_outcomes_present") != 0
    ):
        raise ResourceSchedulingError("Q0R v3 hard-block identity drift")
    runtime = _runtime_environment(repo_root)
    _validate_runtime_environment(runtime)
    arm_inputs = _arm_inputs(q0_external)
    evidence = project_resource_only_evidence(
        q0_receipt=q0_external,
        q0r_receipt=q0r_receipt,
    )
    decision = admit_resource_only_evidence(evidence)
    if not decision["schedule"]:
        raise ResourceSchedulingError("Q0R2 first-principles schedule is empty")

    campaign_root.mkdir(parents=True)
    evidence_path = campaign_root / "RESOURCE_ONLY_EVIDENCE.json"
    runtime_path = campaign_root / "RUNTIME_ENVIRONMENT.json"
    decision_path = campaign_root / "PREDICTION_AND_SCHEDULE_BEFORE_OUTCOME.json"
    evidence_digest = _write_new_json(evidence_path, evidence)
    runtime_digest = _write_new_json(runtime_path, runtime)
    decision_digest = _write_new_json(decision_path, decision)
    binding = canonical_value(
        {
            "accepted_q0_commit": ACCEPTED_Q0_COMMIT,
            "accepted_q0_external_receipt_sha256": Q0_EXTERNAL_RECEIPT_SHA256,
            "accepted_q0_repo_receipt_sha256": Q0_REPO_RECEIPT_SHA256,
            "accepted_q0r_v3_commit": ACCEPTED_Q0R_V3_COMMIT,
            "accepted_q0r_v3_receipt_sha256": Q0R_V3_REPO_RECEIPT_SHA256,
            "branch": "feat/research-line-resource-admission",
            "candidate_source_sha256": {
                arm: row["source_sha256"] for arm, row in arm_inputs.items()
            },
            "decision_artifact_sha256": decision_digest,
            "full_outcomes_present_when_written": 0,
            "held_out_reads": 0,
            "q0_status": q0_repo["status"],
            "resource_only_evidence_sha256": evidence_digest,
            "runtime_environment_sha256": runtime_digest,
            "schema": "recclaw.q0r2-plan-binding.v1",
        }
    )
    binding_path = campaign_root / "PLAN_BINDING.json"
    binding_digest = _write_new_json(binding_path, binding)
    for path in (evidence_path, runtime_path, decision_path, binding_path):
        path.chmod(0o444)
    return canonical_value(
        {
            "decision": decision,
            "decision_artifact_sha256": decision_digest,
            "full_outcomes_present": 0,
            "plan_binding_sha256": binding_digest,
            "resource_only_evidence_sha256": evidence_digest,
            "status": "FROZEN_NON_EMPTY_SCHEDULE",
        }
    )


def execute_resource_admission(
    repo_root: Path,
    *,
    campaign_root: Path,
    q0_external_receipt_path: Path,
) -> dict[str, Any]:
    """Execute each arm in the already-frozen Q0R2 schedule exactly once."""

    repo_root = repo_root.resolve()
    campaign_root = campaign_root.resolve()
    paths = {
        "evidence": campaign_root / "RESOURCE_ONLY_EVIDENCE.json",
        "runtime": campaign_root / "RUNTIME_ENVIRONMENT.json",
        "decision": campaign_root / "PREDICTION_AND_SCHEDULE_BEFORE_OUTCOME.json",
        "binding": campaign_root / "PLAN_BINDING.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ResourceSchedulingError("Q0R2 frozen plan is incomplete")
    if any(path.stat().st_mode & 0o222 for path in paths.values()):
        raise ResourceSchedulingError("Q0R2 decision artifacts are not read-only")
    binding = json.loads(paths["binding"].read_text(encoding="utf-8"))
    observed = {
        "decision_artifact_sha256": bytes_sha256(paths["decision"].read_bytes()),
        "resource_only_evidence_sha256": bytes_sha256(paths["evidence"].read_bytes()),
        "runtime_environment_sha256": bytes_sha256(paths["runtime"].read_bytes()),
    }
    if any(binding[key] != value for key, value in observed.items()):
        raise ResourceSchedulingError("Q0R2 frozen plan byte drift")
    full_root = campaign_root / "fresh_full_runs"
    if full_root.exists():
        raise ResourceSchedulingError("Q0R2 full outcome root already exists")

    _q0_repo, q0_external = _validate_q0_receipts(
        repo_root, q0_external_receipt_path.resolve()
    )
    q0r_path = (
        repo_root
        / "docs/research_line/vnext/"
        "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
    )
    if bytes_sha256(q0r_path.read_bytes()) != Q0R_V3_REPO_RECEIPT_SHA256:
        raise ResourceSchedulingError("sealed Q0R v3 repository receipt byte drift")
    runtime = _runtime_environment(repo_root)
    if canonical_value(runtime) != canonical_value(
        json.loads(paths["runtime"].read_text(encoding="utf-8"))
    ):
        raise ResourceSchedulingError("runtime changed after Q0R2 plan freeze")
    arm_inputs = _arm_inputs(q0_external)
    evidence = json.loads(paths["evidence"].read_text(encoding="utf-8"))
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    initial_sources = {
        arm: bytes_sha256(Path(row["source_path"]).read_bytes())
        for arm, row in arm_inputs.items()
    }

    full_runs: dict[str, Any] = {}
    result_digests: dict[str, str] = {}
    for scheduled in decision["schedule"]:
        arm = scheduled["arm"]
        run = _run_one(
            repo_root=repo_root,
            side_root=full_root,
            arm=arm,
            arm_input=arm_inputs[arm],
            epochs=FULL_EPOCHS,
            purpose="Q0R2_FRESH_SCHEDULED_DEVELOPMENT_VALIDATION",
            timeout_seconds=int(scheduled["deadline_seconds"]),
            run_identity=Q0R2_RUN_IDENTITY,
            authority="user-delegated-q0r2-resource-admission",
        )
        full_runs[arm] = run
        result_digests[arm] = _write_new_json(full_root / f"{arm}.json", run)

    source_unchanged = initial_sources == {
        arm: bytes_sha256(Path(row["source_path"]).read_bytes())
        for arm, row in arm_inputs.items()
    }
    seals_unchanged = (
        source_unchanged
        and bytes_sha256(q0_external_receipt_path.read_bytes())
        == Q0_EXTERNAL_RECEIPT_SHA256
        and bytes_sha256(q0r_path.read_bytes()) == Q0R_V3_REPO_RECEIPT_SHA256
        and bytes_sha256(paths["decision"].read_bytes())
        == binding["decision_artifact_sha256"]
    )
    scheduled_by_arm = {row["arm"]: row for row in decision["schedule"]}
    deferred_by_arm = {row["arm"]: row for row in decision["deferred_arms"]}
    fresh_results = {}
    exact_binding = True
    for arm in ARM_ORDER:
        run = full_runs.get(arm)
        if run is None:
            disposition = deferred_by_arm[arm]
            fresh_results[arm] = {
                "exit_status": "NOT_RUN_RESOURCE_DEFERRED",
                "mechanism_effect_update_allowed": False,
                "missingness": {
                    "missing": True,
                    "reason": disposition["reason"],
                    "resource_disposition": disposition["resource_disposition"],
                },
                "physical_run_executed": False,
                "result_artifact_sha256": None,
                "wall_time_ms": 0,
            }
            continue
        recipe = evidence["arms"][arm]["recipe_identity"]
        binding_ok = (
            run.get("execution_recipe_digest") == recipe["execution_recipe_digest"]
            and run.get("runtime_binding_digest") == recipe["runtime_binding_digest"]
            and run.get("runtime_release_digest") == recipe["runtime_release_digest"]
            and run.get("seed") == recipe["seed"]
        )
        exact_binding = exact_binding and binding_ok
        fresh_results[arm] = {
            "deadline_seconds": scheduled_by_arm[arm]["deadline_seconds"],
            "exact_package_recipe_runtime_binding": binding_ok,
            "exit_status": run.get("exit_status"),
            "mechanism_effect_update_allowed": False,
            "missingness": _missingness(run),
            "physical_run_executed": True,
            "result_artifact_ref": str(full_root / f"{arm}.json"),
            "result_artifact_sha256": result_digests[arm],
            "wall_time_ms": int(run.get("wall_time_ms", 0)),
        }

    scheduled_success = bool(scheduled_by_arm) and all(
        run.get("exit_status") == "SUCCESS" for run in full_runs.values()
    )
    all_accounted = set(scheduled_by_arm) | set(deferred_by_arm) == set(ARM_ORDER)
    gates = {
        "function_real_and_runnable": scheduled_success,
        "end_to_end_result_chain_real_and_valid": (
            len(full_runs) == len(scheduled_by_arm)
            and all_accounted
            and exact_binding
            and seals_unchanged
        ),
        "serves_open_algorithm_research_target": (
            bool(scheduled_by_arm)
            and all(
                row["mechanism_effect_update_allowed"] is False
                for row in fresh_results.values()
            )
        ),
        "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": (
            decision["effect_fields_consumed"] == []
            and evidence["effect_fields_consumed"] == []
            and len(full_runs) == len(scheduled_by_arm)
        ),
    }
    accepted_pass = all(gates.values())
    physical = canonical_value(
        {
            "cost": {
                "campaign_budget_seconds": CAMPAIGN_TOTAL_BUDGET_SECONDS,
                "fresh_full_run_wall_time_ms": sum(
                    int(run.get("wall_time_ms", 0)) for run in full_runs.values()
                ),
                "physical_full_runs": len(full_runs),
                "retries": 0,
            },
            "decision_before_outcome": {
                "artifact_ref": str(paths["decision"]),
                "artifact_sha256": binding["decision_artifact_sha256"],
                "full_outcomes_present_when_written": 0,
                "schedule": decision["schedule"],
            },
            "development_only": True,
            "effect_untouched": {
                "effect_fields_consumed": [],
                "mechanism_effect_updates": 0,
                "resource_censor_updates_effect": False,
            },
            "engineering_safety": {
                "engineering_watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
                "legacy_1500_seconds_controls_full_runs": False,
                "watchdog_is_research_budget": False,
            },
            "evaluation": {
                "formal_acceptance_self_approved": False,
                "formal_scientific_experiment": False,
                "gates": gates,
                "h2_resource_modeling_and_admission": (
                    "CLOSED_DEVELOPMENT_ONLY" if accepted_pass else "NOT_CLOSED"
                ),
                "q1_allowed": accepted_pass,
                "scientific_effect_claim": False,
            },
            "fresh_results": fresh_results,
            "held_out_reads": 0,
            "input_identity": binding,
            "prediction_and_schedule": decision,
            "resource_only_evidence": {
                "artifact_ref": str(paths["evidence"]),
                "artifact_sha256": binding["resource_only_evidence_sha256"],
            },
            "schema": "recclaw.research-line.q0r2-resource-admission.v1",
            "scientific_effect_claim": False,
            "sealed_inputs_unchanged": seals_unchanged,
            "status": "PASS" if accepted_pass else "HARD_BLOCK",
        }
    )
    physical_path = campaign_root / "Q0R2_RESOURCE_ADMISSION_PHYSICAL_RECEIPT.json"
    physical_digest = _write_new_json(physical_path, physical)
    physical_path.chmod(0o444)
    return canonical_value(
        {
            **physical,
            "physical_receipt_sha256": physical_digest,
        }
    )


def finalize_resource_admission_receipt(
    physical_receipt_path: Path,
    *,
    canonical_receipt_path: Path,
    external_receipt_ref: str,
) -> dict[str, Any]:
    """Bind an immutable Q0R2 physical receipt into the repository."""

    physical_digest = bytes_sha256(physical_receipt_path.read_bytes())
    physical = json.loads(physical_receipt_path.read_text(encoding="utf-8"))
    if (
        physical.get("schema")
        != "recclaw.research-line.q0r2-resource-admission.v1"
        or physical.get("held_out_reads") != 0
        or physical.get("scientific_effect_claim") is not False
        or physical.get("status") not in {"PASS", "HARD_BLOCK"}
    ):
        raise ResourceSchedulingError("invalid Q0R2 physical receipt")
    canonical = canonical_value(
        {
            **physical,
            "external_receipt_ref": external_receipt_ref,
            "external_receipt_sha256": physical_digest,
        }
    )
    _write_new_json(canonical_receipt_path, canonical)
    return canonical


def _engineering_budget_separation() -> dict[str, Any]:
    return {
        "censoring_updates_mechanism_effect": False,
        "engineering_watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
        "legacy_1500_seconds_controls_q0r_full_runs": False,
        "normal_deadlines_derived_from": (
            "FRESH_PREFIX_TELEMETRY_UNIFIED_FORMULA_AND_CAMPAIGN_BUDGET"
        ),
        "watchdog_purpose": "HANG_OR_UNBOUNDED_EXECUTION_ONLY",
        "watchdog_trigger_semantics": "RESOURCE_CENSORED",
    }


def finalize_hard_block_receipt(
    external_receipt_path: Path,
    *,
    canonical_receipt_path: Path,
    external_receipt_ref: str | None = None,
) -> dict[str, Any]:
    """Add acceptance semantics to an immutable physical HARD_BLOCK receipt."""

    external_receipt_path = external_receipt_path.resolve()
    external_digest = bytes_sha256(external_receipt_path.read_bytes())
    receipt = json.loads(external_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "HARD_BLOCK"
        or receipt.get("held_out_reads") != 0
        or receipt.get("q1_allowed") is not False
    ):
        raise ResourceSchedulingError("not a valid Q0R physical HARD_BLOCK receipt")
    probe_runs = receipt.get("probe_runs", {})
    blockers = {
        arm: {
            "censoring_semantics": run.get("censoring_semantics"),
            "censoring_trigger": run.get("censoring_trigger"),
            "exit_status": run.get("exit_status"),
            "mechanism_effect_update_allowed": False,
            "resource_deadline_seconds": run.get("resource_deadline_seconds"),
            "resource_disposition": "RESOURCE_DEFERRED",
            "wall_time_ms": run.get("wall_time_ms"),
            "watchdog_seconds": run.get("watchdog_seconds"),
        }
        for arm, run in probe_runs.items()
        if run.get("exit_status") != "SUCCESS"
    }
    if not blockers:
        raise ResourceSchedulingError("HARD_BLOCK receipt has no physical blocker")
    repository_receipt = canonical_value(
        {
            **receipt,
            "engineering_safety_and_research_budget_separation": (
                _engineering_budget_separation()
            ),
            "evaluation": {
                "all_core_gates_pass": False,
                "formal_acceptance_self_approved": False,
                "gates": {
                    "end_to_end_result_chain_real_and_valid": False,
                    "function_real_and_runnable": False,
                    "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": True,
                    "serves_open_algorithm_research_target": True,
                },
                "h2_resource_modeling_and_scheduling": "NOT_CLOSED",
                "mechanism_effect_interpretation": "NOT_ADJUDICATED_RESOURCE_CENSORED",
                "q1_allowed": False,
                "scientific_effect_claim": False,
            },
            "external_receipt_ref": (
                external_receipt_ref
                if external_receipt_ref is not None
                else str(external_receipt_path)
            ),
            "external_receipt_sha256": external_digest,
            "resource_blockers": blockers,
            "unique_next_recommendation": (
                "Before any new physical campaign, replace epoch-completion-only "
                "probe output with the same worker's durable per-phase telemetry "
                "artifact and prefreeze one uniform fixed-batch prefix contract; "
                "then run all four matched arms once in a new user-authorized Q0R "
                "campaign. This is falsified if frontier still yields no phase "
                "throughput before the prefrozen probe budget."
            ),
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


def finalize_fixed_batch_hard_block_receipt(
    external_receipt_path: Path,
    *,
    canonical_receipt_path: Path,
    external_receipt_ref: str,
) -> dict[str, Any]:
    """Bind the immutable v2 physical block to explicit Q0R gate semantics."""

    external_digest = bytes_sha256(external_receipt_path.read_bytes())
    receipt = json.loads(external_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "recclaw.research-line.q0r-canonical-receipt.v2"
        or receipt.get("status") != "HARD_BLOCK"
        or receipt.get("held_out_reads") != 0
        or receipt.get("q1_allowed") is not False
    ):
        raise ResourceSchedulingError("not a valid fixed-batch Q0R HARD_BLOCK receipt")
    blockers = {}
    for arm, run in receipt.get("probe_runs", {}).items():
        telemetry = run.get("resource_telemetry") or {}
        completed = telemetry.get("batch_records") or []
        blockers[arm] = {
            "active_progress": telemetry.get("active_progress"),
            "completed_eval_batches": sum(
                row.get("phase") == "EVAL" for row in completed
            ),
            "completed_train_batches": sum(
                row.get("phase") == "TRAIN" for row in completed
            ),
            "exit_status": run.get("exit_status"),
            "mechanism_effect_update_allowed": False,
            "resource_disposition": _missingness(run)["resource_disposition"],
            "wall_time_ms": run.get("wall_time_ms"),
            "worker_error_message": run.get("worker_error_message"),
            "worker_error_type": run.get("worker_error_type"),
        }
    repository_receipt = canonical_value(
        {
            **receipt,
            "evaluation": {
                "all_core_gates_pass": False,
                "formal_acceptance_self_approved": False,
                "formal_scientific_experiment": False,
                "gates": {
                    "end_to_end_result_chain_real_and_valid": False,
                    "function_real_and_runnable": False,
                    "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": True,
                    "serves_open_algorithm_research_target": True,
                },
                "h2_resource_modeling_and_scheduling": "NOT_CLOSED",
                "mechanism_effect_interpretation": (
                    "NOT_ADJUDICATED_RESOURCE_OR_INTERFACE_BLOCKED"
                ),
                "q1_allowed": False,
                "scientific_effect_claim": False,
            },
            "external_receipt_ref": external_receipt_ref,
            "external_receipt_sha256": external_digest,
            "full_outcomes_present": 0,
            "resource_blockers": blockers,
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


def finalize_type_preserving_hard_block_receipt(
    campaign_root: Path,
    *,
    physical_receipt_path: Path,
    canonical_receipt_path: Path,
    external_receipt_ref: str,
) -> dict[str, Any]:
    """Seal the exited v3 chain without rerunning or revising its decision."""

    run_root = campaign_root / "run"
    decision_path = run_root / "PREDICTION_AND_SCHEDULE_BEFORE_OUTCOME.json"
    input_path = run_root / "INPUT_IDENTITY.json"
    runtime_path = run_root / "RUNTIME_ENVIRONMENT.json"
    prefix_path = run_root / "FIXED_BATCH_PREFIX_CONTRACT.json"
    stderr_path = campaign_root / "launcher.stderr"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    input_identity = json.loads(input_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    prefix_contract = json.loads(prefix_path.read_text(encoding="utf-8"))
    stderr = stderr_path.read_text(encoding="utf-8")
    if decision.get("schedule") != []:
        raise ResourceSchedulingError("v3 HARD_BLOCK decision schedule is not empty")
    full_root = run_root / "fresh_full_runs"
    if full_root.exists() and any(full_root.rglob("*.json")):
        raise ResourceSchedulingError("v3 HARD_BLOCK unexpectedly has full outcomes")
    if (
        "TypeError: object supporting the buffer API required" not in stderr
        or "resource_scheduling.py\", line 988" not in stderr
    ):
        raise ResourceSchedulingError("v3 launcher failure evidence mismatch")
    if (
        input_identity.get("held_out_reads") != 0
        or runtime.get("held_out_reads") != 0
        or prefix_contract.get("held_out_reads") != 0
    ):
        raise ResourceSchedulingError("v3 held-out boundary drift")

    probe_runs = {
        arm: json.loads(
            (run_root / "resource_probes" / f"{arm}.json").read_text(
                encoding="utf-8"
            )
        )
        for arm in ARM_ORDER
    }
    probe_summary = {}
    for arm, run in probe_runs.items():
        telemetry = run.get("resource_telemetry") or {}
        batches = telemetry.get("batch_records") or []
        probe_summary[arm] = {
            "active_progress": telemetry.get("active_progress"),
            "completed_eval_batches": sum(
                row.get("phase") == "EVAL" for row in batches
            ),
            "completed_train_batches": sum(
                row.get("phase") == "TRAIN" for row in batches
            ),
            "exit_status": run.get("exit_status"),
            "mechanism_effect_update_allowed": False,
            "peak_gpu_memory_mib": telemetry.get("peak_gpu_memory_mib"),
            "resource_telemetry_sha256": run.get("resource_telemetry_sha256"),
            "wall_time_ms": run.get("wall_time_ms"),
        }
    expected_eval_batches = PROBE_EPOCHS * len(FIXED_EVAL_BATCH_INDICES)
    type_semantics_pass = all(
        probe_summary[arm]["exit_status"] == "SUCCESS"
        and probe_summary[arm]["completed_eval_batches"] == expected_eval_batches
        for arm in ARM_ORDER[:-1]
    )
    frontier = probe_summary["frontier_candidate"]
    durable_telemetry_pass = (
        frontier["exit_status"] == "RESOURCE_CENSORED"
        and int(frontier["completed_train_batches"]) > 0
        and frontier["active_progress"].get("status") == "BATCH_STARTED"
    )
    deferred = {row["arm"]: row for row in decision.get("deferred_arms", ())}
    frontier_deferred_correctly = (
        deferred.get("frontier_candidate", {}).get("reason")
        == "TRAINING_LOWER_BOUND_EXCEEDS_CAMPAIGN_BUDGET"
    )
    if not (
        type_semantics_pass
        and durable_telemetry_pass
        and frontier_deferred_correctly
    ):
        raise ResourceSchedulingError("v3 partial gate evidence mismatch")

    physical = canonical_value(
        {
            "cost": {
                "campaign_budget_seconds": CAMPAIGN_TOTAL_BUDGET_SECONDS,
                "full_physical_runs": 0,
                "physical_probe_runs": len(probe_runs),
                "probe_wall_time_ms": sum(
                    int(run.get("wall_time_ms", 0)) for run in probe_runs.values()
                ),
                "retries": 0,
            },
            "decision_before_outcome": {
                "artifact_ref": str(decision_path),
                "artifact_sha256": bytes_sha256(decision_path.read_bytes()),
                "full_outcomes_present_when_written": 0,
                "schedule": [],
            },
            "development_only": True,
            "engineering_safety_and_research_budget_separation": (
                _engineering_budget_separation()
            ),
            "evaluation": {
                "all_core_gates_pass": False,
                "durable_atomic_telemetry": "PASS",
                "formal_acceptance_self_approved": False,
                "formal_scientific_experiment": False,
                "frontier_resource_disposition": "RESOURCE_DEFERRED",
                "full_sort_loader_type_semantics": "PASS",
                "gates": {
                    "end_to_end_result_chain_real_and_valid": False,
                    "function_real_and_runnable": False,
                    "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": True,
                    "serves_open_algorithm_research_target": True,
                },
                "h2_resource_modeling_and_scheduling": "NOT_CLOSED",
                "mechanism_effect_interpretation": (
                    "NOT_ADJUDICATED_RESOURCE_OR_SCHEDULER_BLOCKED"
                ),
                "q1_allowed": False,
                "scientific_effect_claim": False,
            },
            "full_outcomes_present": 0,
            "held_out_reads": 0,
            "input_identity": input_identity,
            "launcher_failure": {
                "error_message": "object supporting the buffer API required",
                "error_type": "TypeError",
                "stage": "POST_RUN_SEAL_CHECK",
                "stderr_sha256": bytes_sha256(stderr_path.read_bytes()),
            },
            "prediction_and_schedule": decision,
            "prefix_contract": {
                "artifact_ref": str(prefix_path),
                "artifact_sha256": bytes_sha256(prefix_path.read_bytes()),
                "scientific_outcome": False,
            },
            "probe_runs": probe_runs,
            "probe_summary": probe_summary,
            "resource_deadline_estimator_blocker": {
                "observed_behavior": (
                    "MAX_BATCH_UPPER_EXTRAPOLATION_DEFERRED_EVERY_ARM_AND_LEFT_"
                    "SCHEDULE_EMPTY"
                ),
                "point_estimates_seconds": {
                    arm: decision["predictions"][arm][
                        "estimated_total_wall_time_seconds"
                    ]
                    for arm in ARM_ORDER
                },
                "requested_deadlines_seconds": decision[
                    "requested_deadlines_seconds"
                ],
            },
            "runtime_environment_identity": runtime,
            "schema": "recclaw.research-line.q0r-canonical-receipt.v3",
            "scientific_effect_claim": False,
            "status": "HARD_BLOCK",
            "unique_next_recommendation": (
                "Re-derive the prefix estimator and stopping semantics from first "
                "principles, including an explicit policy for using prior "
                "resource-only observations without consuming mechanism effects; "
                "do not stack a fourth deadline patch or rerun this v3 chain."
            ),
        }
    )
    physical_digest = _write_new_json(physical_receipt_path, physical)
    repository_receipt = canonical_value(
        {
            **physical,
            "external_receipt_ref": external_receipt_ref,
            "external_receipt_sha256": physical_digest,
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


def run_resource_scheduling(
    repo_root: Path,
    *,
    q0_external_receipt_path: Path,
    canonical_receipt_path: Path,
) -> dict[str, Any]:
    """Run fresh probes, freeze decisions, then run each admitted full arm once."""

    started_ns = time.monotonic_ns()
    repo_root = repo_root.resolve()
    if Q0R_ROOT.exists():
        raise ResourceSchedulingError(f"Q0R root already exists: {Q0R_ROOT}")
    if canonical_receipt_path.exists():
        raise ResourceSchedulingError(
            f"Q0R canonical receipt already exists: {canonical_receipt_path}"
        )
    sealed_prior_q0r = _validate_prior_q0r_seals(repo_root)
    q0_repo_receipt, q0_external_receipt = _validate_q0_receipts(
        repo_root, q0_external_receipt_path.resolve()
    )
    runtime = _runtime_environment(repo_root)
    _validate_runtime_environment(runtime)
    arm_inputs = _arm_inputs(q0_external_receipt)
    initial_source_digests = {
        arm: row["source_sha256"] for arm, row in arm_inputs.items()
    }
    features = {
        arm: structural_features(Path(row["source_path"]))
        for arm, row in arm_inputs.items()
    }

    Q0R_ROOT.mkdir(parents=True)
    input_identity = canonical_value(
        {
            "accepted_q0_commit": ACCEPTED_Q0_COMMIT,
            "accepted_q0_parent": ACCEPTED_Q0_PARENT,
            "accepted_q0_tree": ACCEPTED_Q0_TREE,
            "accepted_q0r_v1_commit": ACCEPTED_Q0R_V1_COMMIT,
            "accepted_q0r_v2_commit": ACCEPTED_Q0R_V2_COMMIT,
            "branch": Q0R_BRANCH,
            "candidate_inputs": {
                arm: {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in row.items()
                    if key not in {"source_path"}
                }
                for arm, row in arm_inputs.items()
            },
            "held_out_reads": 0,
            "q0_external_receipt_ref": str(q0_external_receipt_path.resolve()),
            "q0_external_receipt_sha256": Q0_EXTERNAL_RECEIPT_SHA256,
            "q0_repo_receipt_sha256": Q0_REPO_RECEIPT_SHA256,
            "q0_status": q0_repo_receipt["status"],
            "prior_q0r_outcome_fields_consumed": [],
            "prior_q0r_sealed_artifacts": sealed_prior_q0r,
            "sealed_prior_q0r_modified": False,
            "sealed_q0_modified": False,
        }
    )
    _write_new_json(Q0R_ROOT / "INPUT_IDENTITY.json", input_identity)
    _write_new_json(Q0R_ROOT / "RUNTIME_ENVIRONMENT.json", runtime)
    prefix_contract_path = Q0R_ROOT / "FIXED_BATCH_PREFIX_CONTRACT.json"
    prefix_contract_digest = _write_new_json(
        prefix_contract_path, build_fixed_batch_prefix_contract()
    )

    probe_runs: dict[str, Any] = {}
    for arm in ARM_ORDER:
        probe_runs[arm] = _run_one(
            repo_root=repo_root,
            side_root=Q0R_ROOT / "resource_probes",
            arm=arm,
            arm_input=arm_inputs[arm],
            epochs=PROBE_EPOCHS,
            purpose="RESOURCE_PROBE_ONLY",
            timeout_seconds=PROBE_TIMEOUT_SECONDS,
            prefix_contract_path=prefix_contract_path,
        )
        _write_new_json(
            Q0R_ROOT / "resource_probes" / f"{arm}.json", probe_runs[arm]
        )

    try:
        decision = predict_resources(arm_features=features, probe_runs=probe_runs)
    except ResourceSchedulingError as error:
        hard_block = canonical_value(
            {
                "development_only": True,
                "engineering_safety_and_research_budget_separation": (
                    _engineering_budget_separation()
                ),
                "error": str(error),
                "formal_scientific_experiment": False,
                "held_out_reads": 0,
                "input_identity": input_identity,
                "prefix_contract": {
                    "artifact_ref": str(prefix_contract_path),
                    "artifact_sha256": prefix_contract_digest,
                },
                "probe_runs": probe_runs,
                "q1_allowed": False,
                "schema": "recclaw.research-line.q0r-canonical-receipt.v3",
                "scientific_effect_claim": False,
                "status": "HARD_BLOCK",
                "unique_next_recommendation": "NONE_WITHIN_AUTHORIZED_Q0R_REPAIR",
            }
        )
        external_digest = _write_new_json(
            Q0R_ROOT
            / "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
            hard_block,
        )
        repository_receipt = canonical_value(
            {
                **hard_block,
                "external_receipt_ref": str(
                    Q0R_ROOT
                    / "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
                ),
                "external_receipt_sha256": external_digest,
            }
        )
        _write_new_json(canonical_receipt_path, repository_receipt)
        return repository_receipt

    decision_digest = _write_new_json(
        Q0R_ROOT / "PREDICTION_AND_SCHEDULE_BEFORE_OUTCOME.json", decision
    )
    full_runs: dict[str, Any] = {}
    for scheduled in decision["schedule"]:
        arm = scheduled["arm"]
        full_runs[arm] = _run_one(
            repo_root=repo_root,
            side_root=Q0R_ROOT / "fresh_full_runs",
            arm=arm,
            arm_input=arm_inputs[arm],
            epochs=FULL_EPOCHS,
            purpose="Q0R_FRESH_SCHEDULED_DEVELOPMENT_VALIDATION",
            timeout_seconds=int(scheduled["deadline_seconds"]),
            prefix_contract_path=None,
        )
        _write_new_json(Q0R_ROOT / "fresh_full_runs" / f"{arm}.json", full_runs[arm])

    post_source_digests = {
        arm: bytes_sha256(Path(row["source_path"]).read_bytes())
        for arm, row in arm_inputs.items()
    }
    prior_q0r_unchanged = (
        _validate_prior_q0r_seals(repo_root) == sealed_prior_q0r
    )
    sealed_unchanged = (
        initial_source_digests == post_source_digests
        and bytes_sha256(q0_external_receipt_path.read_bytes())
        == Q0_EXTERNAL_RECEIPT_SHA256
        and bytes_sha256(
            (
                repo_root
                / "docs/research_line/vnext/"
                "Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json"
            ).read_bytes()
        )
        == Q0_REPO_RECEIPT_SHA256
        and prior_q0r_unchanged
    )
    scheduled_by_arm = {row["arm"]: row for row in decision["schedule"]}
    deferred_by_arm = {row["arm"]: row for row in decision["deferred_arms"]}
    completion = {}
    for arm in ARM_ORDER:
        run = full_runs.get(arm)
        if run is not None:
            completion[arm] = {
                "actual_completed": run.get("exit_status") == "SUCCESS",
                "deadline_seconds": scheduled_by_arm[arm]["deadline_seconds"],
                "mechanism_effect_update_allowed": False,
                "metrics": run.get("metrics", {}),
                "missingness": _missingness(run),
                "physical_run_executed": True,
                "predicted_completion_probability": decision["predictions"][arm][
                    "completion_probability"
                ],
                "wall_time_ms": run.get("wall_time_ms", 0),
            }
        else:
            disposition = deferred_by_arm[arm]
            completion[arm] = {
                "actual_completed": False,
                "deadline_seconds": None,
                "mechanism_effect_update_allowed": False,
                "metrics": {},
                "missingness": {
                    "missing": True,
                    "reason": disposition["reason"],
                    "resource_disposition": disposition["resource_disposition"],
                },
                "physical_run_executed": False,
                "predicted_completion_probability": 0.0,
                "wall_time_ms": 0,
            }
    all_probes_real = all(
        run.get("exit_status") in {"SUCCESS", "RESOURCE_CENSORED"}
        and isinstance(run.get("resource_telemetry"), Mapping)
        for run in probe_runs.values()
    )
    all_full_complete = bool(scheduled_by_arm) and all(
        full_runs.get(arm, {}).get("exit_status") == "SUCCESS"
        for arm in scheduled_by_arm
    )
    all_arms_accounted = set(scheduled_by_arm) | set(deferred_by_arm) == set(ARM_ORDER)
    gates = {
        "function_real_and_runnable": (
            all_probes_real and all_full_complete and all_arms_accounted
        ),
        "end_to_end_result_chain_real_and_valid": (
            all_probes_real
            and len(full_runs) == len(scheduled_by_arm)
            and all_arms_accounted
            and sealed_unchanged
        ),
        "serves_open_algorithm_research_target": (
            all(arm in decision["predictions"] for arm in ARM_ORDER)
            and all(
                row["mechanism_effect_update_allowed"] is False
                for row in completion.values()
            )
        ),
        "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": (
            decision["outcome_fields_consumed"] == []
            and sealed_unchanged
            and len(full_runs) == len(scheduled_by_arm)
        ),
    }
    accepted_pass = all(gates.values())
    receipt = canonical_value(
        {
            "completion": completion,
            "cost": {
                "campaign_budget_seconds": CAMPAIGN_TOTAL_BUDGET_SECONDS,
                "engineering_watchdog_seconds_per_arm": ENGINEERING_WATCHDOG_SECONDS,
                "full_physical_run_wall_time_ms": sum(
                    int(run.get("wall_time_ms", 0)) for run in full_runs.values()
                ),
                "physical_full_runs": len(full_runs),
                "physical_probe_runs": len(probe_runs),
                "probe_wall_time_ms": sum(
                    int(run.get("wall_time_ms", 0)) for run in probe_runs.values()
                ),
                "retries": 0,
            },
            "decision_before_outcome": {
                "artifact_ref": str(
                    Q0R_ROOT / "PREDICTION_AND_SCHEDULE_BEFORE_OUTCOME.json"
                ),
                "artifact_sha256": decision_digest,
                "full_outcomes_present_when_written": 0,
            },
            "development_only": True,
            "engineering_safety_and_research_budget_separation": (
                _engineering_budget_separation()
            ),
            "evaluation": {
                "formal_acceptance_self_approved": False,
                "formal_scientific_experiment": False,
                "gates": gates,
                "h2_resource_modeling_and_scheduling": (
                    "CLOSED_FOR_Q1_DEVELOPMENT" if accepted_pass else "NOT_CLOSED"
                ),
                "mechanism_effect_interpretation": "OUT_OF_SCOPE_Q0R",
                "q1_allowed": accepted_pass,
                "scientific_effect_claim": False,
            },
            "full_runs": full_runs,
            "held_out_reads": 0,
            "input_identity": input_identity,
            "prefix_contract": {
                "artifact_ref": str(prefix_contract_path),
                "artifact_sha256": prefix_contract_digest,
                "scientific_outcome": False,
            },
            "prediction_and_schedule": decision,
            "probe_runs": probe_runs,
            "runtime_environment_identity": runtime,
            "schema": "recclaw.research-line.q0r-canonical-receipt.v3",
            "sealed_q0_unchanged": sealed_unchanged,
            "sealed_prior_q0r_unchanged": prior_q0r_unchanged,
            "status": "PASS" if accepted_pass else "RESOURCE_INFEASIBLE",
            "wall_time_ms": max(
                1, (time.monotonic_ns() - started_ns) // 1_000_000
            ),
        }
    )
    external_digest = _write_new_json(
        Q0R_ROOT
        / "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
        receipt,
    )
    repository_receipt = canonical_value(
        {
            **receipt,
            "external_receipt_ref": str(
                Q0R_ROOT
                / "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
            ),
            "external_receipt_sha256": external_digest,
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


__all__ = [
    "PREDICTED_GPU_WORKER_SECONDS_SEMANTICS",
    "ResourceSchedulingError",
    "build_innovation_resource_profile",
    "build_fixed_batch_prefix_contract",
    "finalize_fixed_batch_hard_block_receipt",
    "finalize_hard_block_receipt",
    "finalize_type_preserving_hard_block_receipt",
    "predict_resources",
    "run_disposable_fixed_batch_resource_probe",
    "run_resource_scheduling",
    "structural_features",
]
