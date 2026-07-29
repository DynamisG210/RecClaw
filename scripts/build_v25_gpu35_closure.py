#!/usr/bin/env python3
"""Build and verify the fresh V25 gpu35 50-round Effect Pilot closure."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v16 import (  # noqa: E402
    v16_arm_policies,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v25 import (  # noqa: E402
    V25_EXECUTABLE_PROFILE_DIGEST,
    V25_PILOT_ASSIGNMENT_NONCE,
    V25_PILOT_CHECKPOINTS,
    V25_PILOT_ROUNDS_PER_ARM,
    V25_PILOT_SEARCH_SEED,
    V25PilotStoreContractV1,
    v25_arm_policies,
    v25_resource_ceilings,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    MetaV20CampaignRuntimeV1,
    meta_v20_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    PrivateTreatmentAssignmentV1,
)


DOCS = ROOT / "docs/research_line/continuous_program"
RESOURCES = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
QUALIFICATION_ROOT = ROOT.parent
BACKEND_ROOT = QUALIFICATION_ROOT.parent
V24_CONTRACT = DOCS / "V24_FROZEN_CHAIN_PILOT_CONTRACT.json"
V24_DIAGNOSIS = DOCS / "V24_FIVE_ROUND_METHOD_DIAGNOSIS.json"
M6I_EXACT_REPORT = DOCS / "M6I_V20_EXACT_100X50_REPORT.json"
M6I_SYNTHETIC_REPORT = DOCS / "M6I_V20_SYNTHETIC_100X50_REPORT.json"
MARGIN_POLICY = DOCS / "V25_GPU35_RESOURCE_MARGIN_POLICY_V1.json"
RESOURCE_ENVELOPE = (
    RESOURCES / "pilot_v25_gpu35_resource_envelope.json"
)
EFFECT_CRITERIA = DOCS / "PILOT_EFFECT_VISIBILITY_CRITERIA_V1.json"
SOURCE_PROJECTION = (
    DOCS / "V25_SCIENTIFIC_SOURCE_PROJECTION.json"
)
RUNTIME_AUDIT = DOCS / "V25_GPU35_RUNTIME_AUDIT.json"
BACKEND_AUDIT = DOCS / "V25_GPU35_BACKEND_CONFORMANCE_AUDIT.json"
SCIENTIFIC_GATE = DOCS / "V25_SCIENTIFIC_ATTRIBUTION_GATE.json"
DEFAULT_OUTPUT = DOCS / "V25_FROZEN_EFFECT_PILOT_CONTRACT.json"
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "pilot_9227_v25"
TRAINING_RELEASE = RESOURCES / "training_runtime_release_v17.json"
TRAINING_LOCK = RESOURCES / "training_runtime_v17_lock.json"
BROKER_RELEASE = (
    RESOURCES / "lab_api_broker_release_v1_v24_schema_v6.json"
)
PROPOSAL_SCHEMA = RESOURCES / "campaign_proposal_response_v2.schema.json"
META_CHECKPOINT = RESOURCES / "meta_vnext_policy_checkpoint_v20.json"
PROFILE_TEST_LOG = BACKEND_ROOT / "V25_PROFILE_AND_ISOLATION_TEST.log"
TRAINING_PYTHON = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/runtime_exact_v2/bin/python"
)
RECBOLE_ROOT = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/recbole"
)
DATA_PATH = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/search_dataset"
)
GIT_EXECUTABLE = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/tools/git-focal/usr/bin/git"
)
GIT_EXEC_PATH = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/tools/git-focal/usr/lib/git-core"
)
RECORD_SCHEMA = "recclaw.v25-effect-pilot-contract.v1"
RUNTIME_RELEASE_ID = "TRAINING_RUNTIME_RELEASE_V17"
CANARY_EXECUTION_SOURCE_HEAD = (
    "28cfe21dbe9391307e3aa7c92b5de577a4852b83"
)
CANARY_EXECUTION_SOURCE_PATHS = (
    "configs",
    "recclaw_ext",
    "src",
    "scripts/campaign_train_worker.py",
    "scripts/launch_v24_qualification_canary.py",
    "scripts/run_candidate.py",
    "scripts/run_v24_full_recipe_canary.py",
)
CANARY_SPECS = (
    ("bpr", "BPR_MF", 9383),
    ("lightgcn", "LIGHTGCN", 9384),
    (
        "compositional",
        "LIGHTGCN__LGCN_AUX_ALIGNMENT__LGCN_DUAL_PATH",
        9385,
    ),
    ("tail_reweight", "BPR_MF__BPR_TAIL_REWEIGHT", 9386),
)
SELECTED_SCRIPTS = {
    "scripts/build_v25_gpu35_closure.py",
    "scripts/campaign_train_worker.py",
    "scripts/freeze_campaign_training_runtime_release_v17.py",
    "scripts/launch_v24_qualification_canary.py",
    "scripts/run_v13_pilot.py",
    "scripts/run_v24_full_recipe_canary.py",
    "scripts/run_v25_effect_pilot.py",
    "scripts/run_candidate.py",
    "scripts/run_m6i_exact_scheduler_stress.py",
    "scripts/run_m6i_provider_isolation_probe.py",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _with_digest(
    value: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    return {**value, field: sha256_digest(value)}


def _verify_record_digest(
    value: dict[str, Any],
    field: str,
    label: str,
) -> None:
    preimage = dict(value)
    expected = preimage.pop(field, None)
    if not expected or sha256_digest(preimage) != expected:
        raise RuntimeError(f"{label} digest is invalid")


def _ceil_to(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def activate_original_git_tool() -> None:
    os.environ["PATH"] = (
        GIT_EXECUTABLE.parent.as_posix()
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    os.environ["GIT_EXEC_PATH"] = GIT_EXEC_PATH.as_posix()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        [
            str(GIT_EXECUTABLE),
            f"--git-dir={ROOT / '.git'}",
            f"--work-tree={ROOT}",
            *arguments,
        ],
        text=True,
    ).strip()


def _canary_execution_source_subset() -> dict[str, str]:
    unchanged = subprocess.run(
        [
            str(GIT_EXECUTABLE),
            f"--git-dir={ROOT / '.git'}",
            f"--work-tree={ROOT}",
            "diff",
            "--quiet",
            CANARY_EXECUTION_SOURCE_HEAD,
            "HEAD",
            "--",
            *CANARY_EXECUTION_SOURCE_PATHS,
        ],
        check=False,
    )
    if unchanged.returncode != 0:
        raise RuntimeError(
            "V25 canary execution source changed after canary launch"
        )
    tracked = _git("ls-files", "--", *CANARY_EXECUTION_SOURCE_PATHS)
    return {
        relative: file_sha256(ROOT / relative)
        for relative in tracked.splitlines()
        if (ROOT / relative).is_file()
    }


def _runtime_release_digest() -> str:
    from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (
        TrainingRuntimeReleaseV3,
    )

    return TrainingRuntimeReleaseV3(_read(TRAINING_RELEASE)).digest


def _broker_release() -> dict[str, Any]:
    release = _read(BROKER_RELEASE)
    preimage = dict(release)
    expected = preimage.pop("release_digest")
    if (
        sha256_digest(preimage) != expected
        or release.get("model") != "gpt-5.4"
        or release.get("request_mode") != "SINGLE_JSON_SCHEMA_NO_TOOLS"
        or release.get("temperature") != 0.0
        or release.get("retry_count") != 0
        or release.get("timeout_ms") != 900_000
        or release.get("max_total_tokens_per_call") != 20_000
        or release.get("response_schema_digest")
        != file_sha256(PROPOSAL_SCHEMA)
    ):
        raise RuntimeError("V25 Provider transport release is invalid")
    return release


def _monitor_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    clean = [row for row in rows if not row.get("monitor_error")]
    if not clean:
        raise RuntimeError(f"GPU monitor has no valid rows: {path}")

    def numbers(key: str) -> list[float]:
        return [float(row[key]) for row in clean]

    return {
        "gpu_util_peak_percent": max(numbers("gpu_util_percent")),
        "memory_free_min_mib": min(numbers("memory_free_mib")),
        "memory_used_peak_mib": max(numbers("memory_used_mib")),
        "monitor_error_count": len(rows) - len(clean),
        "power_peak_w": max(numbers("power_w")),
        "sample_count": len(rows),
        "temperature_peak_c": max(numbers("temperature_c")),
    }


def _canary_records() -> list[dict[str, Any]]:
    parent = QUALIFICATION_ROOT / "backend_canaries_v25"
    records = []
    for name, mechanism_id, seed in CANARY_SPECS:
        root = parent / f"canary_{name}_{seed}_full_recipe_v1"
        result_path = (
            root / f"GPU35_V17_{name.upper()}_{seed}_RESULT.json"
        )
        status_path = parent / f"canary_{name}_{seed}_status.json"
        monitor_path = parent / f"canary_{name}_{seed}_gpu_monitor.csv"
        result = _read(result_path)
        status = _read(status_path)
        if (
            result["verdict"] != "PASS"
            or not all(result["gates"].values())
            or int(status["exit_code"]) != 0
            or result["campaign_mechanism_id"] != mechanism_id
            or int(result["search_seed"]) != seed
            or result["runtime_release_digest"]
            != _runtime_release_digest()
        ):
            raise RuntimeError(f"qualification canary failed: {name}")
        records.append(
            {
                "all_closure_gates_pass": True,
                "campaign_mechanism_id": mechanism_id,
                "gpu_cost_microunits": int(
                    result["ledger"]["GPU_COST_MICROUNITS"]
                ),
                "gpu_device_time_ms": int(
                    result["ledger"]["GPU_DEVICE_TIME_MS"]
                ),
                "monitor": _monitor_summary(monitor_path),
                "name": name,
                "ndcg_at_10": float(
                    result["normalized_metrics"]["ndcg@10"]
                ),
                "result_path": result_path.as_posix(),
                "result_sha256": file_sha256(result_path),
                "runtime_release_digest": result[
                    "runtime_release_digest"
                ],
                "search_seed": seed,
                "status_path": status_path.as_posix(),
                "status_sha256": file_sha256(status_path),
                "verdict": "PASS",
                "wall_time_ms": int(result["ledger"]["WALL_TIME_MS"]),
                "wrapper_elapsed_ms": int(status["elapsed_ms"]),
            }
        )
    return records


def _build_resource_envelope(
    policy: dict[str, Any],
    canaries: list[dict[str, Any]],
) -> dict[str, Any]:
    max_gpu = max(int(row["gpu_device_time_ms"]) for row in canaries)
    max_wall = max(
        max(int(row["wall_time_ms"]), int(row["wrapper_elapsed_ms"]))
        for row in canaries
    )
    gpu_ceiling = _ceil_to(max_gpu * 1.25, 60_000)
    wall_ceiling = _ceil_to(max_wall * 1.25, 60_000)
    cost_ceiling = _ceil_to(
        gpu_ceiling * 1_000_000 / 3_600_000,
        10_000,
    )
    envelope = {
        "authority": "NONE",
        "canary_measurement_digest": sha256_digest(canaries),
        "derivation": {
            "gpu_device_time_base_ms": max_gpu,
            "gpu_device_time_margin_multiplier": 1.25,
            "gpu_device_time_rounding_ms": 60_000,
            "wall_time_base_ms": max_wall,
            "wall_time_margin_multiplier": 1.25,
            "wall_time_rounding_ms": 60_000,
        },
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "formal_acceptance": False,
        "margin_policy_digest": sha256_digest(policy),
        "pilot_outcomes_used": False,
        "record_schema": "recclaw.v25-resource-envelope.v1",
        "resource_ceilings": {
            "common_validation_count": 4,
            "gpu_cost_microunits": cost_ceiling,
            "gpu_device_time_ms": gpu_ceiling,
            "ordinary_executions": 1,
            "proposal_attempt_debit": 4,
            "retry_debit": 0,
            "total_billed_token_debit": 80_000,
            "total_input_tokens": 60_000,
            "total_output_tokens": 20_000,
            "total_proposal_count": 4,
            "wall_time_ms": wall_ceiling,
        },
        "status": "FROZEN_PRE_OUTCOME",
    }
    return _with_digest(envelope, "resource_envelope_digest")


def _tracked_source_projection() -> tuple[str, dict[str, str]]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError(
            "tracked source must be clean before V25 freeze"
        )
    tracked = _git("ls-files").splitlines()
    selected = sorted(
        path
        for path in tracked
        if path.startswith(("src/", "recclaw_ext/", "configs/"))
        or path in SELECTED_SCRIPTS
    )
    files = {
        path: file_sha256(ROOT / path)
        for path in selected
        if (ROOT / path).is_file()
    }
    authorization = (
        "RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md"
    )
    if (ROOT / authorization).is_file():
        files[authorization] = file_sha256(ROOT / authorization)
    return _git("rev-parse", "HEAD"), dict(sorted(files.items()))


def _dataset_identity(v24: dict[str, Any]) -> dict[str, Any]:
    dataset = copy.deepcopy(v24["dataset"])
    search = DATA_PATH / "ml-1m"
    mismatches = [
        name
        for name, digest in dataset["search_files"].items()
        if file_sha256(search / name) != digest
    ]
    if mismatches:
        raise RuntimeError(f"dataset identity mismatch: {mismatches}")
    dataset["search_dataset"] = search.as_posix()
    dataset["search_parent"] = DATA_PATH.as_posix()
    dataset["online_backend_heldout_mount"] = "ABSENT"
    dataset["online_heldout_access"] = False
    return dataset


def _runtime_audit(
    *,
    source_manifest_digest: str,
    dataset_manifest_digest: str,
) -> dict[str, Any]:
    from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
        campaign_training_runtime_release,
        validate_campaign_training_runtime_release,
    )

    release = campaign_training_runtime_release()
    failures = validate_campaign_training_runtime_release(
        data_path=DATA_PATH,
        python_executable=TRAINING_PYTHON,
        recbole_root=RECBOLE_ROOT,
    )
    if (
        release.release_id != RUNTIME_RELEASE_ID
        or release.digest != _runtime_release_digest()
        or failures
    ):
        raise RuntimeError(f"Runtime V17 validation failed: {failures}")
    audit = {
        "authority": "NONE",
        "dataset_manifest_digest": dataset_manifest_digest,
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "formal_acceptance": False,
        "record_schema": "recclaw.v25-gpu35-runtime-audit.v1",
        "release_digest": release.digest,
        "release_file_sha256": file_sha256(TRAINING_RELEASE),
        "release_id": release.release_id,
        "release_manifest_path": TRAINING_RELEASE.as_posix(),
        "runtime_lock_path": TRAINING_LOCK.as_posix(),
        "runtime_lock_sha256": file_sha256(TRAINING_LOCK),
        "source_manifest_digest": source_manifest_digest,
        "training_python": TRAINING_PYTHON.as_posix(),
        "training_python_sha256": file_sha256(TRAINING_PYTHON),
        "validation_failure_codes": list(failures),
        "verdict": "PASS",
    }
    return _with_digest(audit, "runtime_audit_digest")


def _effect_criteria(
    envelope: dict[str, Any],
    source_files: dict[str, str],
) -> dict[str, Any]:
    criteria = {
        "analysis_source": {
            "path": (
                "src/recclaw_core/experiments/helix_abc_v1/"
                "effect_pilot_analysis.py"
            ),
            "sha256": source_files[
                "src/recclaw_core/experiments/helix_abc_v1/"
                "effect_pilot_analysis.py"
            ],
        },
        "authority": "NONE",
        "effect_estimands": {
            "primary": ["B_MINUS_A", "C_MINUS_B"],
            "secondary_only": ["C_MINUS_A"],
        },
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "evidence_guard": {
            "best_non_suppression": -0.003,
            "minimum_completed_validations": 2,
            "minimum_nontrivial_interventions": 2,
            "minimum_successful_challenge_cases": 10,
            "round_auc_non_suppression": -0.003,
            "zero_required": [
                "cross_arm_contamination_count",
                "false_allow_count",
                "false_block_count",
                "guard_private_input_leak_count",
                "legal_candidate_permanent_suppression_count",
                "preliminary_marked_confirmed_count",
                "search_memory_pollution_count",
                "seed_binding_mismatch_count"
            ],
        },
        "formal_acceptance": False,
        "frozen_analysis_budgets_per_arm": {
            "gpu_cost_microunits": (
                int(
                    envelope["resource_ceilings"][
                        "gpu_cost_microunits"
                    ]
                )
                * V25_PILOT_ROUNDS_PER_ARM
            ),
            "rounds": V25_PILOT_ROUNDS_PER_ARM,
            "tokens": (
                int(
                    envelope["resource_ceilings"][
                        "total_billed_token_debit"
                    ]
                )
                * V25_PILOT_ROUNDS_PER_ARM
            ),
        },
        "main_eligibility": False,
        "outcome_use_boundary": {
            "pilot_effect_values_used": False,
            "v24_use": (
                "CHAIN_FEASIBILITY_AND_DEFECT_DIAGNOSIS_ONLY"
            ),
        },
        "record_schema": (
            "recclaw.pilot-effect-visibility-criteria.v1"
        ),
        "research_capability": {
            "best_noninferiority": -0.0005,
            "best_superiority": 0.002,
            "best_tolerance": 0.0005,
            "maximum_blocker_rate": 0.25,
            "maximum_duplicate_rate": 0.50,
            "minimum_distinct_executed_semantics": 8,
            "minimum_efficiency_ratio": 1.05,
            "minimum_selected_per_producer_role": 4,
            "required_producer_roles": [
                "mechanism_composer",
                "lineage_refiner",
                "falsification_designer",
                "frontier_architect"
            ],
            "round_auc_noninferiority": -0.0005,
            "round_auc_superiority": 0.0015,
            "useful_signal_delta": 0.0005,
            "useful_signal_rate_gain": 0.05,
            "visibility_branches": [
                "EFFICACY",
                "BREAKTHROUGH_WITH_AUC_NONINFERIORITY",
                "EFFICIENCY_AND_USEFUL_SIGNAL_WITH_AUC_NONINFERIORITY"
            ],
        },
        "status": "FROZEN_BEFORE_V25_PROVIDER_CALLS",
    }
    return _with_digest(criteria, "criteria_digest")


def _backend_audit(
    *,
    canaries: list[dict[str, Any]],
    envelope: dict[str, Any],
    runtime: dict[str, Any],
    source_head: str,
) -> dict[str, Any]:
    profile_log = PROFILE_TEST_LOG.read_text(encoding="utf-8")
    canary_source_files = _canary_execution_source_subset()
    if (
        "44 passed, 82 subtests passed" not in profile_log
        or any(row["verdict"] != "PASS" for row in canaries)
    ):
        raise RuntimeError("V25 profile/canary qualification is not PASS")
    release = _read(TRAINING_RELEASE)
    backend = release["backend_identity"]
    torch_cuda = backend["torch_cuda_environment"]
    audit = {
        "authority": "NONE",
        "backend_release": {
            "backend_class": backend["backend_class"],
            "device": torch_cuda["primary_device_name"],
            "driver_version": torch_cuda["nvidia_driver_version"],
            "runtime_audit_digest": runtime["runtime_audit_digest"],
            "runtime_release_digest": runtime["release_digest"],
            "runtime_release_id": runtime["release_id"],
        },
        "compile_and_materialize": {
            "current_byte_effect_analysis_tests": True,
            "executable_mechanism_count": 66,
            "full_profile_compiled_and_materialized": True,
            "m6i_isolation_tests": True,
            "producer_v20_tests": True,
            "result": "44 passed, 82 subtests passed",
            "test_log_path": PROFILE_TEST_LOG.as_posix(),
            "test_log_sha256": file_sha256(PROFILE_TEST_LOG),
        },
        "canary_execution_source": {
            "files_digest": sha256_digest(canary_source_files),
            "launch_head": CANARY_EXECUTION_SOURCE_HEAD,
            "qualification_head": source_head,
            "subset_unchanged_between_heads": True,
        },
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "fixed_full_recipe_canaries": canaries,
        "formal_acceptance": False,
        "p0": 0,
        "p1": 0,
        "p2": [
            "REMOTE_LEGACY_LOCAL_PATH_TESTS_ARE_NOT_THE_ACTIVE_"
            "CAMPAIGN_RUNTIME_GATE; LOCAL_FULL_M6R_AND_REMOTE_"
            "RUNTIME_V17_VALIDATION_PASS"
        ],
        "pilot_outcomes_used": False,
        "qualification_source_head": source_head,
        "qualification_training_executions": len(canaries),
        "record_schema": (
            "recclaw.v25-gpu35-backend-conformance-audit.v1"
        ),
        "resource_envelope_digest": envelope[
            "resource_envelope_digest"
        ],
        "verdict": "PASS",
    }
    return _with_digest(audit, "backend_audit_digest")


def _scientific_gate(
    *,
    backend: dict[str, Any],
    runtime: dict[str, Any],
    source_projection: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    exact = _read(M6I_EXACT_REPORT)
    synthetic = _read(M6I_SYNTHETIC_REPORT)
    if any(
        report.get("status") != "PASS"
        or report.get("p0") != 0
        or report.get("p1") != 0
        for report in (exact, synthetic)
    ):
        raise RuntimeError("current-byte M6I qualification is not PASS")
    if (
        exact.get("schema")
        != "recclaw.m6i.exact-scheduler-stress.v1"
        or exact.get("rounds_per_arm") != 50
        or exact.get("randomized_schedule_seeds") != 100
        or exact.get("real_provider_calls") != 0
        or exact.get("real_training_executions") != 0
        or exact.get("source_projection_unchanged") is not True
        or sha256_digest(exact["source_projection"])
        != exact.get("source_projection_digest")
        or synthetic.get("schema")
        != "recclaw.m6i.synthetic-50r-report.v1"
        or synthetic.get("rounds_per_arm") != 50
        or synthetic.get("randomized_schedule_seeds") != 100
        or synthetic.get("real_provider_calls") != 0
        or synthetic.get("real_training_executions") != 0
    ):
        raise RuntimeError("M6I 100x50 qualification identity is invalid")
    for relative, digest in exact["source_projection"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(
                f"M6I current-byte source drifted: {relative}"
            )
    gate = {
        "authority": "NONE",
        "backend_audit_digest": backend["backend_audit_digest"],
        "checked_invariants": [
            "arm_a_exact_pinned_original_path",
            "b_c_same_research_line_except_evidence_port_and_downstream_state",
            "meta_v20_inherits_exact_v19_coefficients",
            "meta_v20_only_adds_deterministic_arm_private_producer_opportunity",
            "same_repaired_66_semantics_profile_all_arms",
            "execution_seed_binding_persisted_and_checked",
            "tail_reweight_finite_positive_and_live_canary_pass",
            "matched_control_same_arm_exact_program_protocol_seed_only",
            "incremental_arm_private_and_neutral_checkpoints",
            "effect_analysis_frozen_before_provider_calls",
            "heldout_online_mount_absent",
            "authority_none_and_no_confirmed_search_frontier",
        ],
        "criteria_digest": criteria["criteria_digest"],
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "formal_acceptance": False,
        "m6i_exact_report_sha256": file_sha256(M6I_EXACT_REPORT),
        "m6i_synthetic_report_sha256": file_sha256(
            M6I_SYNTHETIC_REPORT
        ),
        "p0": 0,
        "p1": 0,
        "p2": 1,
        "pilot_outcomes_used_for_thresholds": False,
        "provider_calls": 0,
        "record_schema": "recclaw.v25-scientific-attribution-gate.v1",
        "runtime_audit_digest": runtime["runtime_audit_digest"],
        "scientific_source_projection_digest": source_projection[
            "projection_digest"
        ],
        "training_executions": len(CANARY_SPECS),
        "verdict": "PASS",
    }
    return _with_digest(gate, "gate_result_digest")


def _schedule() -> list[dict[str, Any]]:
    producer_roles = [
        "mechanism_composer",
        "lineage_refiner",
        "falsification_designer",
        "frontier_architect",
    ]
    return [
        {
            "A": (
                "ORIGINAL_REFRESH_4"
                if (round_index - 1) % 3 == 0
                else "ORIGINAL_CACHED_SLATE"
            ),
            "B": producer_roles,
            "C": producer_roles,
            "ordinary_execution_opportunity_per_arm": 1,
            "round_index": round_index,
        }
        for round_index in range(1, V25_PILOT_ROUNDS_PER_ARM + 1)
    ]


def _build_contract(
    *,
    v24: dict[str, Any],
    dataset: dict[str, Any],
    envelope: dict[str, Any],
    source_head: str,
    source_files: dict[str, str],
    source_projection: dict[str, Any],
    runtime: dict[str, Any],
    backend: dict[str, Any],
    gate: dict[str, Any],
    criteria: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    store = V25PilotStoreContractV1.create()
    if (
        tuple(item.to_dict() for item in v25_arm_policies())
        != tuple(item.to_dict() for item in v16_arm_policies())
        or v25_resource_ceilings().to_dict()
        != envelope["resource_ceilings"]
    ):
        raise RuntimeError("V25 treatment/resource binding mismatch")
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id,
        nonce=V25_PILOT_ASSIGNMENT_NONCE,
    )
    payload = copy.deepcopy(v24)
    payload.pop("content_digest")
    payload.update(
        {
            "output_root": output_root.as_posix(),
            "pilot_started": False,
            "record_schema": RECORD_SCHEMA,
            "status": "FROZEN_PRE_OUTCOME",
        }
    )
    payload["pilot"] = {
        "checkpoints": list(V25_PILOT_CHECKPOINTS),
        "experiment_id": store.experiment_id,
        "ordinary_execution_seed": store.ordinary_execution_seed,
        "rounds_per_arm": store.scheduled_slots_per_arm_seed,
        "search_seeds": list(store.search_seeds),
        "store_contract_identity_digest": store.identity_digest,
    }
    payload["assignment"] = {
        "commitment": assignment.commitment,
        "opaque": True,
    }
    payload["arm_policies"] = [
        item.to_dict() for item in store.arm_policies
    ]
    schedule = _schedule()
    payload["candidate_schedule"] = {
        "guard_extra_proposals": 0,
        "guard_pre_fallback": "NEXT_FROM_ROUTER_FROZEN_SAME_SLATE",
        "schedule": schedule,
        "schedule_digest": sha256_digest(schedule),
    }
    payload["budget_per_arm_round"] = envelope["resource_ceilings"]
    payload["dataset"] = dataset
    payload["training"] = {
        **v24["training"],
        "release_digest": runtime["release_digest"],
        "release_id": runtime["release_id"],
        "release_manifest_path": TRAINING_RELEASE.as_posix(),
        "release_manifest_sha256": file_sha256(TRAINING_RELEASE),
        "python": TRAINING_PYTHON.as_posix(),
        "recbole_root": RECBOLE_ROOT.as_posix(),
    }
    broker = _broker_release()
    payload["broker"] = {
        **v24["broker"],
        "endpoint_digest": broker["endpoint_digest"],
        "max_total_tokens_per_call": broker[
            "max_total_tokens_per_call"
        ],
        "model": broker["model"],
        "release_digest": broker["release_digest"],
        "release_manifest_path": BROKER_RELEASE.as_posix(),
        "release_manifest_sha256": file_sha256(BROKER_RELEASE),
        "request_mode": broker["request_mode"],
        "response_schema_digest": broker["response_schema_digest"],
        "response_schema_path": PROPOSAL_SCHEMA.as_posix(),
        "response_schema_sha256": file_sha256(PROPOSAL_SCHEMA),
        "retry_count": broker["retry_count"],
        "temperature": broker["temperature"],
        "timeout_ms": broker["timeout_ms"],
        "transport": broker["transport"],
    }
    checkpoint = _read(META_CHECKPOINT)
    policy = meta_v20_research_control_policy()
    payload["meta"] = {
        "activation_boundary": "NEXT_FRESH_CAMPAIGN",
        "checkpoint_path": META_CHECKPOINT.as_posix(),
        "checkpoint_sha256": file_sha256(META_CHECKPOINT),
        "control_policy_digest": policy.digest,
        "development_activation_decision_digest": checkpoint[
            "development_activation_decision_digest"
        ],
        "fast_residual_use": (
            "ONLY_WHEN_ACTUAL_TASK_CONTEXT_SUPPORTED"
        ),
        "parent_policy_bundle_digest": checkpoint[
            "parent_policy_bundle_digest"
        ],
        "policy_bundle_digest": checkpoint["policy_bundle_digest"],
        "producer_opportunity_policy": checkpoint[
            "producer_opportunity_policy"
        ],
        "promotion": False,
        "task_context": v24["meta"]["task_context"],
        "task_support_projection": v24["meta"][
            "task_support_projection"
        ],
    }
    payload["guard_and_fusion"] = {
        **v24["guard_and_fusion"],
        "gate_path": SCIENTIFIC_GATE.relative_to(ROOT).as_posix(),
        "gate_result_digest": gate["gate_result_digest"],
        "gate_sha256": file_sha256(SCIENTIFIC_GATE),
    }
    payload["analysis"] = {
        "confirmed_requires_frozen_post_selection_evaluator": True,
        "criteria_digest": criteria["criteria_digest"],
        "criteria_path": EFFECT_CRITERIA.as_posix(),
        "criteria_sha256": file_sha256(EFFECT_CRITERIA),
        "frontier_projections": [
            "OBSERVED",
            "SEARCH_ELIGIBLE",
            "CONFIRMED",
        ],
        "intent_to_treat_round_rows": True,
        "no_execution_rows_preserved": True,
        "pilot_computes_treatment_effect": (
            "DEVELOPMENT_ONLY_PRE_REGISTERED"
        ),
    }
    original = dict(v24["original"])
    original.pop("git_config_path", None)
    original.pop("git_config_sha256", None)
    payload["original"] = {
        **original,
        "git_exec_path": GIT_EXEC_PATH.as_posix(),
        "git_executable": GIT_EXECUTABLE.as_posix(),
        "git_executable_sha256": file_sha256(GIT_EXECUTABLE),
        "repository_access_mode": "EXPLICIT_GIT_DIR_NO_WORKTREE_DISCOVERY",
    }
    payload["backend_qualification"] = {
        "audit_path": BACKEND_AUDIT.as_posix(),
        "audit_sha256": file_sha256(BACKEND_AUDIT),
        "audit_verdict": backend["verdict"],
        "backend_audit_digest": backend["backend_audit_digest"],
        "p0": backend["p0"],
        "p1": backend["p1"],
        "release_id": runtime["release_id"],
        "runtime_release_digest": runtime["release_digest"],
    }
    payload["backend_migration"] = {
        "from_backend": "gpu35 RTX 3080 Runtime V16 sealed V24",
        "old_v24_contract_modified": False,
        "old_v24_root_reused": False,
        "resource_envelope_digest": envelope[
            "resource_envelope_digest"
        ],
        "resource_envelope_path": RESOURCE_ENVELOPE.as_posix(),
        "resource_envelope_sha256": file_sha256(RESOURCE_ENVELOPE),
        "runtime_audit_digest": runtime["runtime_audit_digest"],
        "runtime_audit_path": RUNTIME_AUDIT.as_posix(),
        "runtime_audit_sha256": file_sha256(RUNTIME_AUDIT),
        "scientific_source_projection_digest": source_projection[
            "projection_digest"
        ],
        "scientific_source_projection_path": (
            SOURCE_PROJECTION.as_posix()
        ),
        "scientific_source_projection_sha256": file_sha256(
            SOURCE_PROJECTION
        ),
        "to_backend": "gpu35 RTX 3080 Runtime V17 fresh V25",
    }
    payload["common_substrate"] = {
        **v24["common_substrate"],
        "executable_profile_digest": V25_EXECUTABLE_PROFILE_DIGEST,
        "executable_mechanism_count": 66,
    }
    payload["source"] = {
        "files": source_files,
        "git_head_at_freeze": source_head,
        "manifest_digest": sha256_digest(source_files),
        "scientific_source_projection_digest": source_projection[
            "projection_digest"
        ],
        "selected_source_manifest_is_authoritative": True,
        "v24_parent_manifest_digest": v24["source"]["manifest_digest"],
    }
    return {**payload, "content_digest": sha256_digest(payload)}


def verify_v25_pilot_contract(path: Path) -> dict[str, Any]:
    activate_original_git_tool()
    contract = _read(path)
    preimage = dict(contract)
    expected = preimage.pop("content_digest")
    envelope = _read(RESOURCE_ENVELOPE)
    backend = _read(BACKEND_AUDIT)
    runtime = _read(RUNTIME_AUDIT)
    gate = _read(SCIENTIFIC_GATE)
    criteria = _read(EFFECT_CRITERIA)
    source_projection = _read(SOURCE_PROJECTION)
    checkpoint = _read(META_CHECKPOINT)
    for value, field, label in (
        (
            envelope,
            "resource_envelope_digest",
            "V25 resource envelope",
        ),
        (runtime, "runtime_audit_digest", "V25 runtime audit"),
        (backend, "backend_audit_digest", "V25 backend audit"),
        (gate, "gate_result_digest", "V25 scientific gate"),
        (criteria, "criteria_digest", "V25 effect criteria"),
        (
            source_projection,
            "projection_digest",
            "V25 source projection",
        ),
    ):
        _verify_record_digest(value, field, label)
    if (
        sha256_digest(preimage) != expected
        or contract["record_schema"] != RECORD_SCHEMA
        or contract["status"] != "FROZEN_PRE_OUTCOME"
        or contract["pilot_started"] is not False
        or contract["pilot"]["search_seeds"]
        != [V25_PILOT_SEARCH_SEED]
        or contract["pilot"]["rounds_per_arm"]
        != V25_PILOT_ROUNDS_PER_ARM
        or contract["pilot"]["checkpoints"]
        != list(V25_PILOT_CHECKPOINTS)
        or contract["budget_per_arm_round"]
        != envelope["resource_ceilings"]
        or contract["backend_qualification"]["audit_verdict"] != "PASS"
        or contract["backend_qualification"]["p0"] != 0
        or contract["backend_qualification"]["p1"] != 0
        or gate["verdict"] != "PASS"
        or gate["p0"] != 0
        or gate["p1"] != 0
        or criteria["status"] != "FROZEN_BEFORE_V25_PROVIDER_CALLS"
        or contract["analysis"]["criteria_digest"]
        != criteria["criteria_digest"]
        or contract["source"]["manifest_digest"]
        != sha256_digest(contract["source"]["files"])
        or contract["source"]["scientific_source_projection_digest"]
        != source_projection["projection_digest"]
        or contract["common_substrate"]["executable_profile_digest"]
        != V25_EXECUTABLE_PROFILE_DIGEST
        or contract["common_substrate"]["executable_mechanism_count"]
        != 66
        or contract["meta"]["policy_bundle_digest"]
        != checkpoint["policy_bundle_digest"]
        or contract["meta"]["control_policy_digest"]
        != meta_v20_research_control_policy().digest
        or runtime["release_digest"] != _runtime_release_digest()
        or backend["resource_envelope_digest"]
        != envelope["resource_envelope_digest"]
        or contract["output_root"] != DEFAULT_OUTPUT_ROOT.as_posix()
        or Path(contract["output_root"]).exists()
    ):
        raise RuntimeError("V25 contract closure invariant mismatch")
    for relative, digest in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"V25 source identity mismatch: {relative}")
    exact = {
        Path(contract["backend_qualification"]["audit_path"]): contract[
            "backend_qualification"
        ]["audit_sha256"],
        Path(contract["broker"]["release_manifest_path"]): contract[
            "broker"
        ]["release_manifest_sha256"],
        Path(contract["broker"]["response_schema_path"]): contract[
            "broker"
        ]["response_schema_sha256"],
        Path(contract["meta"]["checkpoint_path"]): contract["meta"][
            "checkpoint_sha256"
        ],
        Path(contract["training"]["release_manifest_path"]): contract[
            "training"
        ]["release_manifest_sha256"],
        Path(contract["original"]["git_executable"]): contract[
            "original"
        ]["git_executable_sha256"],
        Path(contract["original"]["source_bundle_path"]): contract[
            "original"
        ]["source_bundle_sha256"],
        Path(contract["analysis"]["criteria_path"]): contract[
            "analysis"
        ]["criteria_sha256"],
        Path(
            contract["backend_migration"]["resource_envelope_path"]
        ): contract["backend_migration"]["resource_envelope_sha256"],
        Path(
            contract["backend_migration"]["runtime_audit_path"]
        ): contract["backend_migration"]["runtime_audit_sha256"],
        Path(
            contract["backend_migration"][
                "scientific_source_projection_path"
            ]
        ): contract["backend_migration"][
            "scientific_source_projection_sha256"
        ],
    }
    gate_path = Path(contract["guard_and_fusion"]["gate_path"])
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    exact[gate_path] = contract["guard_and_fusion"]["gate_sha256"]
    for artifact, digest in exact.items():
        if file_sha256(artifact) != digest:
            raise RuntimeError(f"V25 release identity mismatch: {artifact}")
    if _git("rev-parse", "HEAD") != contract["source"][
        "git_head_at_freeze"
    ]:
        raise RuntimeError("V25 source repository HEAD drifted")
    return contract


def build(output: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(
            f"fresh V25 output root already exists: {output_root}"
        )
    if output.exists():
        raise RuntimeError(f"V25 contract already exists: {output}")
    activate_original_git_tool()
    v24 = _read(V24_CONTRACT)
    canaries = _canary_records()
    envelope = _build_resource_envelope(_read(MARGIN_POLICY), canaries)
    _write(RESOURCE_ENVELOPE, envelope)
    source_head, source_files = _tracked_source_projection()
    source_projection = _with_digest(
        {
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
            "formal_acceptance": False,
            "manifest_digest": sha256_digest(source_files),
            "record_schema": (
                "recclaw.v25-scientific-source-projection.v1"
            ),
            "source_file_count": len(source_files),
            "source_head": source_head,
            "treatment_change": (
                "META_V20_ARM_PRIVATE_PRODUCER_OPPORTUNITY_ONLY"
            ),
            "v24_diagnosis_sha256": file_sha256(V24_DIAGNOSIS),
            "v24_outcomes_imported_into_state": False,
            "v24_root_reused": False,
        },
        "projection_digest",
    )
    _write(SOURCE_PROJECTION, source_projection)
    dataset = _dataset_identity(v24)
    runtime = _runtime_audit(
        source_manifest_digest=source_projection["manifest_digest"],
        dataset_manifest_digest=dataset["manifest_digest"],
    )
    _write(RUNTIME_AUDIT, runtime)
    criteria = _effect_criteria(envelope, source_files)
    _write(EFFECT_CRITERIA, criteria)
    backend = _backend_audit(
        canaries=canaries,
        envelope=envelope,
        runtime=runtime,
        source_head=source_head,
    )
    _write(BACKEND_AUDIT, backend)
    gate = _scientific_gate(
        backend=backend,
        runtime=runtime,
        source_projection=source_projection,
        criteria=criteria,
    )
    _write(SCIENTIFIC_GATE, gate)
    contract = _build_contract(
        v24=v24,
        dataset=dataset,
        envelope=envelope,
        source_head=source_head,
        source_files=source_files,
        source_projection=source_projection,
        runtime=runtime,
        backend=backend,
        gate=gate,
        criteria=criteria,
        output_root=output_root,
    )
    _write(output, contract)
    verify_v25_pilot_contract(output)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    contract = (
        verify_v25_pilot_contract(args.output.resolve())
        if args.verify
        else build(args.output.resolve(), args.output_root.resolve())
    )
    print(
        json.dumps(
            {
                "content_digest": contract["content_digest"],
                "output_root": contract["output_root"],
                "pilot_started": contract["pilot_started"],
                "status": contract["status"],
                "verdict": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
