"""Q0R DEVELOPMENT_ONLY resource probes, model, and outcome-blind scheduling."""

from __future__ import annotations

import ast
import json
import math
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import bytes_sha256, canonical_value, sha256_digest
from .fresh_r1 import (
    EXPECTED_SEARCH_FILES,
    PYTHON_EXECUTABLE,
    RECBole_ROOT,
    SEARCH_DATA_ROOT,
    _write_new_json,
    run_development_training,
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
PROBE_TIMEOUT_SECONDS = 300
CAMPAIGN_TOTAL_BUDGET_SECONDS = 7200
ENGINEERING_WATCHDOG_SECONDS = 10800
GPU_MEMORY_TOTAL_MIB = 10240
TRAINING_SEED = 54102
FIXED_TRAIN_BATCH_INDICES = tuple(range(32))
FIXED_EVAL_BATCH_INDICES = tuple(range(64))


class ResourceSchedulingError(RuntimeError):
    """Q0R identity, telemetry, or scheduling failure."""


def build_fixed_batch_prefix_contract(
    *,
    seed: int = TRAINING_SEED,
) -> dict[str, Any]:
    return canonical_value(
        {
            "dataset": "ml-1m",
            "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
            "deadline_rule": {
                "engineering_watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
                "legacy_1500_seconds_controls_probe": False,
                "resource_deadline_seconds": PROBE_TIMEOUT_SECONDS,
            },
            "epochs": PROBE_EPOCHS,
            "eval_batch_indices": list(FIXED_EVAL_BATCH_INDICES),
            "execution_purpose": "RESOURCE_PROBE_ONLY",
            "held_out_reads": 0,
            "selection_rule": (
                "preallocate the listed source-loader positions before model "
                "construction under the shared seed; reuse those exact batches "
                "for every prefix epoch and every arm"
            ),
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
    arm_order: Sequence[str] = ARM_ORDER,
    probe_seed: int = TRAINING_SEED,
) -> dict[str, Any]:
    """Fit the fixed-batch auditable model and allocate one campaign budget."""

    order = tuple(arm_order)
    if not order or len(set(order)) != len(order):
        raise ResourceSchedulingError("arm order must be non-empty and unique")
    if set(arm_features) != set(order) or set(probe_runs) != set(order):
        raise ResourceSchedulingError("resource inputs do not match frozen arm order")
    predictions: dict[str, dict[str, Any]] = {}
    requested_deadlines: dict[str, int] = {}
    deferred: dict[str, dict[str, Any]] = {}
    probe_cost_seconds = math.ceil(
        sum(int(probe_runs[arm]["wall_time_ms"]) for arm in order) / 1000
    )
    full_run_budget_seconds = total_budget_seconds - probe_cost_seconds
    if full_run_budget_seconds <= 0:
        raise ResourceSchedulingError("prefix probes exhausted campaign budget")
    contract_digests: set[str] = set()
    for arm in order:
        run = probe_runs[arm]
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
        batches = [
            row
            for row in telemetry.get("batch_records", ())
            if row.get("status") == "BATCH_COMPLETED"
            and isinstance(row.get("wall_time_ms"), (int, float))
        ]
        train_rows = [row for row in batches if row.get("phase") == "TRAIN"]
        eval_rows = [row for row in batches if row.get("phase") == "EVAL"]
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
        train_point_epoch_ms = (
            sum(train_ms) / len(train_ms) * int(full_train_batches)
        )
        train_lower_epoch_ms = min(train_ms) * int(full_train_batches)
        train_upper_epoch_ms = max(train_ms) * int(full_train_batches)
        combined_features = canonical_value(
            {
                **arm_features[arm],
                "parameter_count": telemetry.get("parameter_count"),
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
        training_lower_bound_ms = (
            int(setup_ms) + train_lower_epoch_ms * FULL_EPOCHS
        )
        memory_safe = peak_prediction < GPU_MEMORY_TOTAL_MIB * 0.95
        if not memory_safe:
            deferred[arm] = {
                "reason": "PREDICTED_GPU_MEMORY_CAPACITY",
                "resource_disposition": "RESOURCE_INFEASIBLE",
            }
        elif training_lower_bound_ms / 1000 > full_run_budget_seconds:
            deferred[arm] = {
                "reason": "TRAINING_LOWER_BOUND_EXCEEDS_CAMPAIGN_BUDGET",
                "resource_disposition": "RESOURCE_DEFERRED",
            }
        elif not eval_rows:
            raise ResourceSchedulingError(
                f"fixed-batch prefix eval signal unavailable for feasible arm: {arm}"
            )

        if eval_rows:
            eval_point_epoch_ms = (
                sum(eval_ms) / len(eval_ms) * int(full_eval_batches)
            )
            eval_lower_epoch_ms = min(eval_ms) * int(full_eval_batches)
            eval_upper_epoch_ms = max(eval_ms) * int(full_eval_batches)
            point_epoch_ms = train_point_epoch_ms + eval_point_epoch_ms
            lower_epoch_ms = train_lower_epoch_ms + eval_lower_epoch_ms
            upper_epoch_ms = train_upper_epoch_ms + eval_upper_epoch_ms
            estimate_scope = "TRAIN_AND_FULL_SORT_EVAL"
            eval_share = eval_point_epoch_ms / point_epoch_ms
        else:
            point_epoch_ms = train_point_epoch_ms
            lower_epoch_ms = train_lower_epoch_ms
            upper_epoch_ms = train_upper_epoch_ms
            estimate_scope = "TRAINING_ONLY_RESOURCE_LOWER_BOUND"
            eval_share = 0.0
        point_ms = int(setup_ms) + point_epoch_ms * FULL_EPOCHS
        lower_ms = int(setup_ms) + lower_epoch_ms * FULL_EPOCHS
        upper_ms = int(setup_ms) + upper_epoch_ms * FULL_EPOCHS * (
            1.0 + 0.05 * complexity_count
        )
        if peak_prediction >= GPU_MEMORY_TOTAL_MIB * 0.90:
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
        expected_batches = PROBE_EPOCHS * (
            len(FIXED_TRAIN_BATCH_INDICES) + len(FIXED_EVAL_BATCH_INDICES)
        )
        confidence = (
            "MEDIUM"
            if observed_batches == expected_batches and run.get("exit_status") == "SUCCESS"
            else "LOW"
        )
        if arm not in deferred:
            requested_deadlines[arm] = max(
                180, math.ceil(60 + 1.10 * upper_ms / 1000)
            )
        predictions[arm] = canonical_value(
            {
                "bottleneck_category": bottleneck,
                "confidence": confidence,
                "completed_eval_batches": len(eval_rows),
                "completed_train_batches": len(train_rows),
                "estimated_total_wall_time_seconds": point_ms / 1000,
                "estimate_scope": estimate_scope,
                "features": combined_features,
                "fixed_batch_eval_mean_wall_time_ms": (
                    sum(eval_ms) / len(eval_ms) if eval_ms else None
                ),
                "fixed_batch_train_mean_wall_time_ms": sum(train_ms) / len(train_ms),
                "full_eval_batches_per_epoch": full_eval_batches,
                "full_train_batches_per_epoch": full_train_batches,
                "model": "FIXED_BATCH_THROUGHPUT_LINEAR_EXTRAPOLATION_V3",
                "peak_memory_prediction_mib": peak_prediction,
                "peak_memory_observed_mib": peak_observed,
                "prediction_interval_seconds": [lower_ms / 1000, upper_ms / 1000],
                "probe_wall_time_ms": run["wall_time_ms"],
                "setup_wall_time_ms": setup_ms,
                "training_only_lower_bound_seconds": (
                    training_lower_bound_ms / 1000
                ),
                "uncertainty_basis": (
                    "min/max completed fixed train and eval batch throughput; "
                    "upper bound widened five percent per visible bottleneck feature"
                ),
            }
        )

    if len(contract_digests) != 1:
        raise ResourceSchedulingError("fixed-batch prefix contract differs across arms")

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
            "deadline_allocation_scale": 1.0,
            "deadline_formula": (
                "first defer any arm whose training-only lower bound exceeds the "
                "remaining full-run campaign budget or whose predicted peak exceeds "
                "95 percent of GPU memory; otherwise request max(180, ceil(60 + "
                "1.10 * upper_fixed_batch_extrapolation)) and admit in ascending "
                "predicted-time order while the unified campaign budget remains"
            ),
            "deferred_arms": list(deferred.values()),
            "full_epochs": FULL_EPOCHS,
            "full_run_budget_after_probes_seconds": full_run_budget_seconds,
            "outcome_fields_consumed": [],
            "predictions": predictions,
            "probe_contract": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "epochs": PROBE_EPOCHS,
                "eval_batch_indices": list(FIXED_EVAL_BATCH_INDICES),
                "execution_purpose": "RESOURCE_PROBE_ONLY",
                "prefix_contract_sha256": next(iter(contract_digests)),
                "seed": probe_seed,
                "timeout_seconds": PROBE_TIMEOUT_SECONDS,
                "train_batch_indices": list(FIXED_TRAIN_BATCH_INDICES),
                "uniform_across_arms": True,
            },
            "probe_cost_seconds": probe_cost_seconds,
            "requested_deadlines_seconds": requested_deadlines,
            "schedule": schedule,
            "schedule_rule": (
                "ascending predicted total wall time; stable original order tie-break"
            ),
            "schema": "recclaw.q0r-resource-prediction-and-schedule.v3",
        }
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
) -> dict[str, Any]:
    try:
        return run_development_training(
            repo_root=repo_root,
            side_root=side_root,
            run_id=arm.replace("_", "-"),
            seed=TRAINING_SEED,
            candidate_root=arm_input["candidate_root"],
            entrypoint=str(arm_input["entrypoint"]),
            source_sha256=str(arm_input["source_sha256"]),
            run_identity=run_identity,
            authority=authority,
            timeout_seconds=timeout_seconds,
            recbole_commit_identity="7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
            epochs=epochs,
            execution_purpose=purpose,
            resource_telemetry=True,
            watchdog_seconds=ENGINEERING_WATCHDOG_SECONDS,
            prefix_contract_path=prefix_contract_path,
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
    "ResourceSchedulingError",
    "build_fixed_batch_prefix_contract",
    "finalize_fixed_batch_hard_block_receipt",
    "finalize_hard_block_receipt",
    "finalize_type_preserving_hard_block_receipt",
    "predict_resources",
    "run_resource_scheduling",
    "structural_features",
]
