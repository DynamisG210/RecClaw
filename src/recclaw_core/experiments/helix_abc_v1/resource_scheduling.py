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
from typing import Any, Mapping

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
Q0R_RUN_IDENTITY = "q0r-resource-scheduling-v1"
Q0R_BRANCH = "feat/research-line-resource-scheduling"
Q0R_ROOT = Path(
    os.environ.get(
        "RECCLAW_Q0R_ROOT",
        "/root/projects/RecClaw_resource_scheduling_runs/q0r_v1",
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


class ResourceSchedulingError(RuntimeError):
    """Q0R identity, telemetry, or scheduling failure."""


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


def predict_resources(
    *,
    arm_features: Mapping[str, Mapping[str, Any]],
    probe_runs: Mapping[str, Mapping[str, Any]],
    total_budget_seconds: int = CAMPAIGN_TOTAL_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Fit the fixed auditable prefix-extrapolation model and allocate one budget."""

    predictions: dict[str, dict[str, Any]] = {}
    requested_deadlines: dict[str, int] = {}
    probe_cost_seconds = math.ceil(
        sum(int(probe_runs[arm]["wall_time_ms"]) for arm in ARM_ORDER) / 1000
    )
    full_run_budget_seconds = total_budget_seconds - probe_cost_seconds
    if full_run_budget_seconds <= 0:
        raise ResourceSchedulingError("prefix probes exhausted campaign budget")
    for arm in ARM_ORDER:
        run = probe_runs[arm]
        telemetry = run.get("resource_telemetry")
        if run.get("exit_status") != "SUCCESS" or not isinstance(telemetry, Mapping):
            raise ResourceSchedulingError(f"real prefix telemetry unavailable: {arm}")
        phases = list(telemetry.get("phase_records", ()))
        train_rows = [row for row in phases if row.get("phase") == "TRAIN"]
        eval_rows = [row for row in phases if row.get("phase") == "EVAL"]
        if len(train_rows) != PROBE_EPOCHS or len(eval_rows) != PROBE_EPOCHS:
            raise ResourceSchedulingError(f"prefix phase count mismatch: {arm}")
        epoch_ms = []
        for epoch in range(PROBE_EPOCHS):
            epoch_ms.append(
                sum(
                    int(row["wall_time_ms"])
                    for row in phases
                    if int(row.get("epoch", -1)) == epoch
                )
            )
        phase_total_ms = sum(int(row["wall_time_ms"]) for row in phases)
        setup_ms = max(0, int(run["wall_time_ms"]) - phase_total_ms)
        mean_epoch_ms = sum(epoch_ms) / len(epoch_ms)
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
        point_ms = setup_ms + mean_epoch_ms * FULL_EPOCHS
        lower_ms = setup_ms + min(epoch_ms) * FULL_EPOCHS
        upper_ms = setup_ms + max(epoch_ms) * FULL_EPOCHS * (
            1.0 + 0.05 * complexity_count
        )
        peak_observed = max(
            float(row[key])
            for row in phases
            for key in ("peak_allocated_mib", "peak_reserved_mib")
            if isinstance(row.get(key), (int, float))
        )
        peak_prediction = peak_observed * (1.0 + 0.03 * complexity_count)
        eval_share = sum(int(row["wall_time_ms"]) for row in eval_rows) / phase_total_ms
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
        coefficient_of_range = (max(epoch_ms) - min(epoch_ms)) / max(mean_epoch_ms, 1)
        confidence = "MEDIUM" if coefficient_of_range <= 0.25 else "LOW"
        raw_deadline = max(180, math.ceil(60 + 1.10 * upper_ms / 1000))
        requested_deadlines[arm] = raw_deadline
        predictions[arm] = canonical_value(
            {
                "bottleneck_category": bottleneck,
                "confidence": confidence,
                "epoch_wall_time_ms": epoch_ms,
                "estimated_total_wall_time_seconds": point_ms / 1000,
                "features": combined_features,
                "model": "PREFIX_PHASE_LINEAR_EXTRAPOLATION_V1",
                "peak_memory_prediction_mib": peak_prediction,
                "peak_memory_observed_mib": peak_observed,
                "prediction_interval_seconds": [lower_ms / 1000, upper_ms / 1000],
                "probe_wall_time_ms": run["wall_time_ms"],
                "setup_wall_time_ms": setup_ms,
                "uncertainty_basis": (
                    "three fresh same-recipe epochs; range widened five percent "
                    "per visible bottleneck feature"
                ),
            }
        )

    minimum = 180
    requested_total = sum(requested_deadlines.values())
    if requested_total <= full_run_budget_seconds:
        deadlines = dict(requested_deadlines)
        scale = 1.0
    else:
        scalable_budget = full_run_budget_seconds - minimum * len(ARM_ORDER)
        scalable_request = sum(
            requested_deadlines[arm] - minimum for arm in ARM_ORDER
        )
        if scalable_budget <= 0 or scalable_request <= 0:
            raise ResourceSchedulingError("campaign budget cannot fund generic minimums")
        scale = scalable_budget / scalable_request
        deadlines = {
            arm: minimum
            + math.floor((requested_deadlines[arm] - minimum) * scale)
            for arm in ARM_ORDER
        }
    schedule_order = sorted(
        ARM_ORDER,
        key=lambda arm: (
            float(predictions[arm]["estimated_total_wall_time_seconds"]),
            ARM_ORDER.index(arm),
        ),
    )
    schedule: list[dict[str, Any]] = []
    for ordinal, arm in enumerate(schedule_order, start=1):
        prediction = predictions[arm]
        lower, upper = prediction["prediction_interval_seconds"]
        point = prediction["estimated_total_wall_time_seconds"]
        deadline = deadlines[arm]
        memory_safe = prediction["peak_memory_prediction_mib"] < (
            GPU_MEMORY_TOTAL_MIB * 0.95
        )
        if not memory_safe:
            completion_probability = 0.10
        elif upper <= deadline:
            completion_probability = 0.90
        elif point <= deadline:
            completion_probability = 0.65
        elif lower <= deadline:
            completion_probability = 0.35
        else:
            completion_probability = 0.10
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
                "ordinal": ordinal,
            }
        )
    return canonical_value(
        {
            "campaign_total_budget_seconds": total_budget_seconds,
            "deadline_allocation_scale": scale,
            "deadline_formula": (
                "max(180, ceil(60 + 1.10 * upper_prefix_extrapolation)); if sum "
                "exceeds campaign budget, scale every arm excess above 180 by "
                "one common factor"
            ),
            "full_epochs": FULL_EPOCHS,
            "full_run_budget_after_probes_seconds": full_run_budget_seconds,
            "outcome_fields_consumed": [],
            "predictions": predictions,
            "probe_contract": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "epochs": PROBE_EPOCHS,
                "execution_purpose": "RESOURCE_PROBE_ONLY",
                "seed": TRAINING_SEED,
                "timeout_seconds": PROBE_TIMEOUT_SECONDS,
                "uniform_across_arms": True,
            },
            "probe_cost_seconds": probe_cost_seconds,
            "requested_deadlines_seconds": requested_deadlines,
            "schedule": schedule,
            "schedule_rule": (
                "ascending predicted total wall time; stable original order tie-break"
            ),
            "schema": "recclaw.q0r-resource-prediction-and-schedule.v1",
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
            run_identity=Q0R_RUN_IDENTITY,
            authority="user-delegated-q0r-resource-scheduling",
            timeout_seconds=timeout_seconds,
            recbole_commit_identity="7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
            epochs=epochs,
            execution_purpose=purpose,
            resource_telemetry=True,
            watchdog_seconds=ENGINEERING_WATCHDOG_SECONDS,
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


def run_resource_scheduling(
    repo_root: Path,
    *,
    q0_external_receipt_path: Path,
    canonical_receipt_path: Path,
) -> dict[str, Any]:
    """Run fresh probes, freeze decisions, then run every matched full arm once."""

    started_ns = time.monotonic_ns()
    repo_root = repo_root.resolve()
    if Q0R_ROOT.exists():
        raise ResourceSchedulingError(f"Q0R root already exists: {Q0R_ROOT}")
    if canonical_receipt_path.exists():
        raise ResourceSchedulingError(
            f"Q0R canonical receipt already exists: {canonical_receipt_path}"
        )
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
            "sealed_q0_modified": False,
        }
    )
    _write_new_json(Q0R_ROOT / "INPUT_IDENTITY.json", input_identity)
    _write_new_json(Q0R_ROOT / "RUNTIME_ENVIRONMENT.json", runtime)

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
                "probe_runs": probe_runs,
                "q1_allowed": False,
                "schema": "recclaw.research-line.q0r-canonical-receipt.v1",
                "scientific_effect_claim": False,
                "status": "HARD_BLOCK",
                "unique_next_recommendation": (
                    "Persist per-phase telemetry durably inside the existing worker "
                    "and prefreeze one uniform fixed-batch prefix before a new "
                    "user-authorized matched Q0R campaign."
                ),
            }
        )
        external_digest = _write_new_json(
            Q0R_ROOT / "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json", hard_block
        )
        repository_receipt = canonical_value(
            {
                **hard_block,
                "external_receipt_ref": str(
                    Q0R_ROOT / "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
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
        )
        _write_new_json(Q0R_ROOT / "fresh_full_runs" / f"{arm}.json", full_runs[arm])

    post_source_digests = {
        arm: bytes_sha256(Path(row["source_path"]).read_bytes())
        for arm, row in arm_inputs.items()
    }
    sealed_unchanged = (
        initial_source_digests == post_source_digests
        and bytes_sha256(q0_external_receipt_path.read_bytes())
        == Q0_EXTERNAL_RECEIPT_SHA256
        and bytes_sha256(
            repo_root
            / "docs/research_line/vnext/Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json"
        )
        == Q0_REPO_RECEIPT_SHA256
    )
    completion = {
        arm: {
            "actual_completed": run.get("exit_status") == "SUCCESS",
            "deadline_seconds": next(
                row["deadline_seconds"]
                for row in decision["schedule"]
                if row["arm"] == arm
            ),
            "mechanism_effect_update_allowed": False,
            "metrics": run.get("metrics", {}),
            "missingness": _missingness(run),
            "predicted_completion_probability": decision["predictions"][arm][
                "completion_probability"
            ],
            "wall_time_ms": run.get("wall_time_ms", 0),
        }
        for arm, run in full_runs.items()
    }
    all_probes_real = all(run.get("exit_status") == "SUCCESS" for run in probe_runs.values())
    all_full_complete = all(
        full_runs.get(arm, {}).get("exit_status") == "SUCCESS" for arm in ARM_ORDER
    )
    gates = {
        "function_real_and_runnable": all_probes_real and all_full_complete,
        "end_to_end_result_chain_real_and_valid": (
            all_probes_real and len(full_runs) == len(ARM_ORDER) and sealed_unchanged
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
            and len(full_runs) == len(ARM_ORDER)
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
            "prediction_and_schedule": decision,
            "probe_runs": probe_runs,
            "runtime_environment_identity": runtime,
            "schema": "recclaw.research-line.q0r-canonical-receipt.v1",
            "sealed_q0_unchanged": sealed_unchanged,
            "status": "PASS" if accepted_pass else "RESOURCE_INFEASIBLE",
            "wall_time_ms": max(
                1, (time.monotonic_ns() - started_ns) // 1_000_000
            ),
        }
    )
    external_digest = _write_new_json(
        Q0R_ROOT / "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json", receipt
    )
    repository_receipt = canonical_value(
        {
            **receipt,
            "external_receipt_ref": str(
                Q0R_ROOT / "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"
            ),
            "external_receipt_sha256": external_digest,
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


__all__ = [
    "ResourceSchedulingError",
    "finalize_hard_block_receipt",
    "predict_resources",
    "run_resource_scheduling",
    "structural_features",
]
