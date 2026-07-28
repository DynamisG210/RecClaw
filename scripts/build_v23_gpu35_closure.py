#!/usr/bin/env python3
"""Build and verify the complete pre-outcome V23 gpu35 Pilot closure."""

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

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)


BACKEND_ROOT = ROOT.parent
DOCS = ROOT / "docs/research_line/continuous_program"
RESOURCES = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
V16_REFERENCE = DOCS / "V16_FROZEN_CHAIN_PILOT_CONTRACT.json"
M6I_EXACT_REPORT = DOCS / "M6I_FINAL_EXACT_100X50_REPORT.json"
M6I_EXECUTION_RECORD = DOCS / "M6I_FINAL_EXECUTION_RECORD.json"
MARGIN_POLICY = DOCS / "V23_GPU35_RESOURCE_MARGIN_POLICY_V1.json"
RESOURCE_ENVELOPE = RESOURCES / "pilot_v23_gpu35_resource_envelope.json"
SOURCE_PROJECTION = (
    DOCS / "V23_V16_SCIENTIFIC_SOURCE_PROJECTION.json"
)
RUNTIME_AUDIT = DOCS / "V23_GPU35_RUNTIME_AUDIT.json"
BACKEND_AUDIT = (
    DOCS / "V23_GPU35_BACKEND_CONFORMANCE_AUDIT.json"
)
SCIENTIFIC_GATE = DOCS / "V23_SCIENTIFIC_ATTRIBUTION_GATE.json"
ORIGINAL_PROBE = DOCS / "V23_PINNED_ORIGINAL_MATERIALIZATION_PROBE.json"
DEFAULT_OUTPUT = DOCS / "V23_FROZEN_CHAIN_PILOT_CONTRACT.json"
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT.parent / "pilot_9225_v23"
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
TRAINING_RELEASE = RESOURCES / "training_runtime_release_v15.json"
TRAINING_LOCK = RESOURCES / "training_runtime_v15_lock.json"
PROFILE_TEST_LOG = BACKEND_ROOT.parent / "V23_PROFILE_AND_ISOLATION_TEST.log"
GIT_EXECUTABLE = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/tools/git-focal/usr/bin/git"
)
GIT_EXEC_PATH = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/tools/git-focal/usr/lib/git-core"
)
V16_SOURCE_HEAD = "8f7f84a03f3b7ad0fae7444493b8deffc9318671"
RECORD_SCHEMA = "recclaw.v23-pilot-contract.v1"
RUNTIME_RELEASE_ID = "TRAINING_RUNTIME_RELEASE_V15"
CANARY_SPECS = (
    (
        "bpr",
        "BPR_MF",
        9370,
        "GPU35_V15_BPR_9370_RESULT.json",
    ),
    (
        "lightgcn",
        "LIGHTGCN",
        9371,
        "GPU35_V15_LIGHTGCN_9371_RESULT.json",
    ),
    (
        "compositional",
        "LIGHTGCN__LGCN_AUX_ALIGNMENT__LGCN_DUAL_PATH",
        9372,
        "GPU35_V15_COMPOSITIONAL_9372_RESULT.json",
    ),
)
V23_OVERLAY_FILES = (
    "docs/research_line/continuous_program/"
    "V23_GPU35_RESOURCE_MARGIN_POLICY_V1.json",
    "docs/research_line/continuous_program/"
    "V23_PINNED_ORIGINAL_MATERIALIZATION_PROBE.json",
    "scripts/build_v23_gpu35_closure.py",
    "scripts/freeze_campaign_training_runtime_release_v15.py",
    "scripts/launch_v23_qualification_canary.py",
    "scripts/run_v23_full_recipe_canary.py",
    "scripts/run_v23_original_materialization_probe.py",
    "scripts/run_v23_pilot.py",
    "scripts/validate_campaign_training_runtime_v15.py",
    "src/recclaw_core/experiments/helix_abc_v1/campaign_pilot_v23.py",
    "src/recclaw_core/experiments/helix_abc_v1/real_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/original_main.py",
    "src/recclaw_core/experiments/helix_abc_v1/"
    "training_runtime_release.py",
    "src/recclaw_core/experiments/helix_abc_v1/resources/"
    "pilot_v23_gpu35_resource_envelope.json",
    "src/recclaw_core/experiments/helix_abc_v1/resources/"
    "training_runtime_release_v15.json",
    "src/recclaw_core/experiments/helix_abc_v1/resources/"
    "training_runtime_v15_lock.json",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _runtime_release_digest() -> str:
    from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (
        TrainingRuntimeReleaseV3,
    )

    return TrainingRuntimeReleaseV3(_read(TRAINING_RELEASE)).digest


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _with_digest(value: dict[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    payload[field] = sha256_digest(value)
    return payload


def _ceil_to(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def activate_original_git_tool() -> None:
    os.environ["PATH"] = (
        GIT_EXECUTABLE.parent.as_posix()
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    os.environ["GIT_EXEC_PATH"] = GIT_EXEC_PATH.as_posix()


def _git_head() -> str:
    head = (ROOT / ".git/HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_path = ROOT / ".git" / head.removeprefix("ref: ")
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"cannot resolve source repository HEAD: {head}")


def _git_changed_files(old: str, new: str) -> set[str]:
    output = subprocess.check_output(
        [
            str(GIT_EXECUTABLE),
            f"--git-dir={ROOT / '.git'}",
            "diff",
            "--name-only",
            old,
            new,
        ],
        text=True,
    )
    return {line for line in output.splitlines() if line}


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
    parent = BACKEND_ROOT / "backend_canaries_v23"
    records = []
    for name, mechanism_id, seed, filename in CANARY_SPECS:
        root = parent / f"canary_{name}_{seed}_full_recipe_v1"
        result_path = root / filename
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
    rate = int(policy["cost_rate_microunits_per_gpu_hour"])
    cost_ceiling = _ceil_to(
        gpu_ceiling * rate / 3_600_000,
        10_000,
    )
    resource_ceilings = {
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
    }
    envelope = {
        "authority": "NONE",
        "canary_measurement_digest": sha256_digest(canaries),
        "derivation": {
            "cost_rate_microunits_per_gpu_hour": rate,
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
        "margin_policy_path": MARGIN_POLICY.as_posix(),
        "pilot_outcomes_used": False,
        "record_schema": "recclaw.v23-resource-envelope.v1",
        "resource_ceilings": resource_ceilings,
        "status": "FROZEN_PRE_OUTCOME",
    }
    return _with_digest(envelope, "resource_envelope_digest")


def _dataset_identity(v16: dict[str, Any]) -> dict[str, Any]:
    dataset = copy.deepcopy(v16["dataset"])
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


def _source_projection(
    v16: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    expected = dict(v16["source"]["files"])
    observed = {
        relative: file_sha256(ROOT / relative)
        for relative in expected
    }
    v16_mismatches = sorted(
        relative
        for relative, digest in expected.items()
        if observed[relative] != digest
    )
    m6i_report = _read(M6I_EXACT_REPORT)
    m6i_execution = _read(M6I_EXECUTION_RECORD)
    m6i_projection = dict(m6i_report["source_projection"])
    post_m6i_runtime_files = {
        "src/recclaw_core/experiments/helix_abc_v1/original_main.py",
        "src/recclaw_core/experiments/helix_abc_v1/real_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/"
        "training_runtime_release.py",
    }
    for relative, digest in m6i_projection.items():
        if relative not in post_m6i_runtime_files:
            if file_sha256(ROOT / relative) != digest:
                raise RuntimeError(
                    f"M6I qualified source changed before V23 freeze: {relative}"
                )
    m6i_changed_files = _git_changed_files(
        V16_SOURCE_HEAD,
        str(m6i_execution["implementation_checkpoint_commit"]),
    )
    allowed_v16_mismatches = m6i_changed_files | post_m6i_runtime_files
    unexpected = sorted(
        relative
        for relative in v16_mismatches
        if relative not in allowed_v16_mismatches
    )
    if unexpected:
        raise RuntimeError(
            "V16 scientific source mismatch outside M6I/runtime closure: "
            f"{unexpected}"
        )
    runtime_binding = (
        "src/recclaw_core/experiments/helix_abc_v1/"
        "training_runtime_release.py"
    )
    overlay = {
        relative: file_sha256(ROOT / relative)
        for relative in V23_OVERLAY_FILES
    }
    projection = {
        "allowed_v16_byte_difference": {
            runtime_binding: (
                "active Campaign runtime resource binding V9 -> V15"
            ),
            "M6I_QUALIFIED_SOURCE_PROJECTION": (
                "canonical integrated state, call identity, ownership, "
                "task/result semantics and cross-Arm isolation closure"
            ),
            "V23_EXPLICIT_RESOURCE_INJECTION": (
                "constructor-bound common resource envelope; no global "
                "pilot_budget mutation"
            ),
            "V23_PINNED_ORIGINAL_MATERIALIZATION_IO": (
                "the exact frozen Main commit and blob identities are read "
                "through an explicit Git object database, avoiding mutable "
                "worktree safe-directory discovery"
            ),
        },
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "formal_acceptance": False,
        "m6i_execution_record_sha256": file_sha256(M6I_EXECUTION_RECORD),
        "m6i_implementation_checkpoint_commit": m6i_execution[
            "implementation_checkpoint_commit"
        ],
        "m6i_qualified_source_projection": m6i_projection,
        "m6i_qualified_source_projection_digest": m6i_report[
            "source_projection_digest"
        ],
        "new_backend_overlay_files": overlay,
        "record_schema": (
            "recclaw.v23-v16-scientific-source-projection.v1"
        ),
        "scientific_treatment_bytes_preserved": True,
        "v16_exact_file_count": len(expected) - len(v16_mismatches),
        "v16_expected_file_count": len(expected),
        "v16_parent_contract_content_digest": v16["content_digest"],
        "v16_parent_manifest_digest": v16["source"]["manifest_digest"],
        "v16_parent_source_head": V16_SOURCE_HEAD,
        "v16_authorized_m6i_and_runtime_mismatches": v16_mismatches,
    }
    source_files = {**observed, **m6i_projection, **overlay}
    return (
        _with_digest(projection, "projection_digest"),
        dict(sorted(source_files.items())),
    )


def _runtime_audit(
    dataset: dict[str, Any],
    source_projection: dict[str, Any],
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
        raise RuntimeError(
            f"Runtime V15 validation failed: {tuple(failures)}"
        )
    audit = {
        "authority": "NONE",
        "dataset_manifest_digest": dataset["manifest_digest"],
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "formal_acceptance": False,
        "record_schema": "recclaw.v23-gpu35-runtime-audit.v1",
        "release_digest": release.digest,
        "release_file_sha256": file_sha256(TRAINING_RELEASE),
        "release_id": release.release_id,
        "release_manifest_path": TRAINING_RELEASE.as_posix(),
        "runtime_lock_path": TRAINING_LOCK.as_posix(),
        "runtime_lock_sha256": file_sha256(TRAINING_LOCK),
        "scientific_source_projection_digest": source_projection[
            "projection_digest"
        ],
        "training_python": TRAINING_PYTHON.as_posix(),
        "training_python_sha256": file_sha256(TRAINING_PYTHON),
        "validation_failure_codes": list(failures),
        "verdict": "PASS",
    }
    return _with_digest(audit, "runtime_audit_digest")


def _backend_audit(
    canaries: list[dict[str, Any]],
    envelope: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    profile_log = PROFILE_TEST_LOG.read_text(encoding="utf-8")
    if (
        "38 passed, 3 deselected, 72 subtests passed"
        not in profile_log
    ):
        raise RuntimeError("profile/isolation qualification log is not PASS")
    release_document = _read(TRAINING_RELEASE)
    original_probe = _read(ORIGINAL_PROBE)
    if (
        original_probe.get("verdict") != "PASS"
        or original_probe.get("P0") != 0
        or original_probe.get("P1") != 0
        or original_probe.get("main_commit")
        != "2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
        or original_probe.get("materialized_file_count") != 6
    ):
        raise RuntimeError("pinned Original materialization probe is not PASS")
    backend_identity = release_document["backend_identity"]
    torch_cuda = backend_identity["torch_cuda_environment"]
    audit = {
        "authority": "NONE",
        "backend_release": {
            "backend_class": backend_identity["backend_class"],
            "device": torch_cuda["primary_device_name"],
            "driver_version": torch_cuda["nvidia_driver_version"],
            "runtime_audit_digest": runtime["runtime_audit_digest"],
            "runtime_release_digest": runtime["release_digest"],
            "runtime_release_id": runtime["release_id"],
        },
        "compile_and_materialize": {
            "executable_mechanism_count": 66,
            "full_profile_compiled_and_materialized": True,
            "isolation_capability_tests_passed": True,
            "kernel_uid_probe": "SKIPPED_NON_ROOT_CONTAINER",
            "log_path": PROFILE_TEST_LOG.as_posix(),
            "log_sha256": file_sha256(PROFILE_TEST_LOG),
            "result": "38 passed, 3 deselected, 72 subtests passed",
        },
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "fixed_full_recipe_canaries": canaries,
        "formal_acceptance": False,
        "pinned_original_materialization": {
            "probe_path": ORIGINAL_PROBE.as_posix(),
            "probe_sha256": file_sha256(ORIGINAL_PROBE),
            "result_digest": original_probe["result_digest"],
            "verdict": original_probe["verdict"],
        },
        "p0": 0,
        "p1": 0,
        "p2": [
            "NATIVE_CONTAINER_CANNOT_RUN_NUMERIC_UID_KERNEL_PROBE; "
            "PRIVATE_ROOT_DISJOINTNESS_AND_CAPABILITY_TRAVERSAL_"
            "SYMLINK_HARDLINK_DENIAL_TESTS_PASS",
            "TWO_LEGACY_OPERATOR_UNIT_TESTS_REQUIRE_AN_UNMOUNTED_"
            "ROOT_TEST_DATASET_AND_ONE_HISTORICAL_TEST_ASSERTS_"
            "RUNTIME_V9; FULL_PROFILE_COMPILE_MATERIALIZE, RUNTIME_"
            "V15_VALIDATION, AND THREE_LIVE_FULL_RECIPE_CANARIES_"
            "ARE_THE_ACTIVE_QUALIFICATION_PATH",
        ],
        "pilot_outcomes_used": False,
        "provider_calls": 0,
        "qualification_training_executions": len(canaries),
        "record_schema": (
            "recclaw.v23-gpu35-backend-conformance-audit.v1"
        ),
        "resource_envelope_digest": envelope[
            "resource_envelope_digest"
        ],
        "verdict": "PASS",
    }
    return _with_digest(audit, "backend_audit_digest")


def _scientific_gate(
    source_projection: dict[str, Any],
    backend: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    gate = {
        "authority": "NONE",
        "backend_audit_digest": backend["backend_audit_digest"],
        "checked_invariants": [
            "exact_v16_arm_a_pinned_original_path",
            "pinned_original_commit_blob_materialization_remote_pass",
            "exact_v16_b_c_research_line_treatment",
            "b_c_evidence_port_only_treatment_difference",
            "exact_meta_v19_policy_semantics",
            "exact_v16_guard_and_fusion_sources",
            "no_guard_private_research_meta_producer_input",
            "same_66_semantics_bl_icf_profile_all_arms",
            "same_dataset_metric_analysis_semantics",
            "private_roots_disjoint",
            "cross_arm_capability_escape_denied",
            "runtime_backend_resource_changes_only_pre_outcome",
        ],
        "evidence_class": "DEVELOPMENT_ONLY_PRE_OUTCOME",
        "formal_acceptance": False,
        "p0": 0,
        "p1": 0,
        "p2": 1,
        "pilot_outcomes_used": False,
        "provider_calls": 0,
        "record_schema": "recclaw.v23-scientific-attribution-gate.v1",
        "runtime_audit_digest": runtime["runtime_audit_digest"],
        "scientific_source_projection_digest": source_projection[
            "projection_digest"
        ],
        "training_executions": 3,
        "verdict": "PASS",
    }
    return _with_digest(gate, "gate_result_digest")


def _rebase_source_path(path: str) -> str:
    old = (
        "/NAS2020/Workspaces/DMGroup/tingrangan/"
        "recclaw_v15_backend_v1/source"
    )
    return path.replace(old, ROOT.as_posix())


def _build_contract(
    *,
    v16: dict[str, Any],
    dataset: dict[str, Any],
    envelope: dict[str, Any],
    source_projection: dict[str, Any],
    source_files: dict[str, str],
    runtime: dict[str, Any],
    backend: dict[str, Any],
    gate: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v16 import (
        v16_arm_policies,
    )
    from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v23 import (
        V23_PILOT_ASSIGNMENT_NONCE,
        V23PilotStoreContractV1,
        v23_arm_policies,
        v23_resource_ceilings,
    )
    from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
        PrivateTreatmentAssignmentV1,
    )

    store = V23PilotStoreContractV1.create()
    if (
        tuple(item.to_dict() for item in v23_arm_policies())
        != tuple(item.to_dict() for item in v16_arm_policies())
        or v23_resource_ceilings().to_dict()
        != envelope["resource_ceilings"]
    ):
        raise RuntimeError("V23 treatment/resource binding mismatch")
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id,
        nonce=V23_PILOT_ASSIGNMENT_NONCE,
    )
    payload = copy.deepcopy(v16)
    payload.pop("content_digest")
    payload.update(
        {
            "record_schema": RECORD_SCHEMA,
            "output_root": output_root.as_posix(),
            "pilot_started": False,
            "status": "FROZEN_PRE_OUTCOME",
        }
    )
    payload["pilot"] = {
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
    schedule = copy.deepcopy(
        v16["candidate_schedule"]["schedule"]
    )
    payload["candidate_schedule"] = {
        **v16["candidate_schedule"],
        "schedule": schedule,
        "schedule_digest": sha256_digest(schedule),
    }
    payload["budget_per_arm_round"] = envelope["resource_ceilings"]
    payload["dataset"] = dataset
    payload["training"] = {
        **v16["training"],
        "release_digest": runtime["release_digest"],
        "release_id": runtime["release_id"],
        "release_manifest_path": TRAINING_RELEASE.as_posix(),
        "release_manifest_sha256": file_sha256(TRAINING_RELEASE),
        "python": TRAINING_PYTHON.as_posix(),
        "recbole_root": RECBOLE_ROOT.as_posix(),
    }
    payload["broker"] = {
        **v16["broker"],
        "release_manifest_path": _rebase_source_path(
            v16["broker"]["release_manifest_path"]
        ),
        "response_schema_path": _rebase_source_path(
            v16["broker"]["response_schema_path"]
        ),
    }
    payload["broker"]["release_manifest_sha256"] = file_sha256(
        Path(payload["broker"]["release_manifest_path"])
    )
    payload["broker"]["response_schema_sha256"] = file_sha256(
        Path(payload["broker"]["response_schema_path"])
    )
    payload["meta"] = {
        **v16["meta"],
        "checkpoint_path": _rebase_source_path(
            v16["meta"]["checkpoint_path"]
        ),
        "promotion_record_path": _rebase_source_path(
            v16["meta"]["promotion_record_path"]
        ),
    }
    payload["meta"]["checkpoint_sha256"] = file_sha256(
        Path(payload["meta"]["checkpoint_path"])
    )
    payload["meta"]["promotion_record_sha256"] = file_sha256(
        Path(payload["meta"]["promotion_record_path"])
    )
    payload["guard_and_fusion"] = {
        **v16["guard_and_fusion"],
        "gate_path": SCIENTIFIC_GATE.relative_to(ROOT).as_posix(),
        "gate_result_digest": gate["gate_result_digest"],
        "gate_sha256": file_sha256(SCIENTIFIC_GATE),
    }
    original = dict(v16["original"])
    original.pop("git_config_path", None)
    original.pop("git_config_sha256", None)
    payload["original"] = {
        **original,
        "git_exec_path": GIT_EXEC_PATH.as_posix(),
        "git_executable": GIT_EXECUTABLE.as_posix(),
        "git_executable_sha256": file_sha256(GIT_EXECUTABLE),
        "repository_access_mode": "EXPLICIT_GIT_DIR_NO_WORKTREE_DISCOVERY",
        "source_repository_head": V16_SOURCE_HEAD,
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
        "from_backend": "gpu35 RTX 3080 Runtime V14 (V22 pre-outcome failure)",
        "old_v16_contract_modified": False,
        "old_v16_root_reused": False,
        "old_v17_root_reused": False,
        "old_v18_root_reused": False,
        "old_v19_root_reused": False,
        "old_v20_root_reused": False,
        "old_v21_root_reused": False,
        "old_v22_root_reused": False,
        "resource_envelope_digest": envelope[
            "resource_envelope_digest"
        ],
        "runtime_audit_digest": runtime["runtime_audit_digest"],
        "scientific_source_projection_digest": source_projection[
            "projection_digest"
        ],
        "to_backend": "gpu35 RTX 3080 Runtime V15",
    }
    source_head = _git_head()
    payload["source"] = {
        "files": source_files,
        "git_head_at_freeze": source_head,
        "manifest_digest": sha256_digest(source_files),
        "scientific_parent_manifest_digest": v16["source"][
            "manifest_digest"
        ],
        "scientific_source_projection_digest": source_projection[
            "projection_digest"
        ],
        "selected_source_manifest_is_authoritative": True,
    }
    return {**payload, "content_digest": sha256_digest(payload)}


def verify_v23_pilot_contract(path: Path) -> dict[str, Any]:
    activate_original_git_tool()
    contract = _read(path)
    preimage = dict(contract)
    expected = preimage.pop("content_digest")
    if sha256_digest(preimage) != expected:
        raise RuntimeError("V23 contract content digest mismatch")
    envelope = _read(RESOURCE_ENVELOPE)
    backend = _read(BACKEND_AUDIT)
    runtime = _read(RUNTIME_AUDIT)
    gate = _read(SCIENTIFIC_GATE)
    projection = _read(SOURCE_PROJECTION)
    if (
        contract["record_schema"] != RECORD_SCHEMA
        or contract["status"] != "FROZEN_PRE_OUTCOME"
        or contract["pilot_started"] is not False
        or contract["pilot"]["search_seeds"] != [9225]
        or contract["pilot"]["rounds_per_arm"] != 5
        or contract["budget_per_arm_round"]
        != envelope["resource_ceilings"]
        or contract["backend_qualification"]["audit_verdict"]
        != "PASS"
        or contract["backend_qualification"]["p0"] != 0
        or contract["backend_qualification"]["p1"] != 0
        or gate["verdict"] != "PASS"
        or gate["p0"] != 0
        or gate["p1"] != 0
        or projection["scientific_treatment_bytes_preserved"]
        is not True
        or runtime["release_digest"] != _runtime_release_digest()
        or backend["resource_envelope_digest"]
        != envelope["resource_envelope_digest"]
    ):
        raise RuntimeError("V23 contract closure invariant mismatch")
    for relative, digest in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"V23 source identity mismatch: {relative}")
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
        Path(contract["meta"]["promotion_record_path"]): contract[
            "meta"
        ]["promotion_record_sha256"],
        Path(contract["training"]["release_manifest_path"]): contract[
            "training"
        ]["release_manifest_sha256"],
        Path(contract["original"]["git_executable"]): contract[
            "original"
        ]["git_executable_sha256"],
        Path(contract["original"]["source_bundle_path"]): contract[
            "original"
        ]["source_bundle_sha256"],
    }
    for artifact, digest in exact.items():
        if file_sha256(artifact) != digest:
            raise RuntimeError(f"V23 release identity mismatch: {artifact}")
    if _git_head() != contract["source"]["git_head_at_freeze"]:
        raise RuntimeError("V23 source repository head is not frozen source")
    subprocess.run(
        [
            str(GIT_EXECUTABLE),
            f"--git-dir={ROOT / '.git'}",
            "cat-file",
            "-e",
            f"{contract['original']['main_commit']}^{{commit}}",
        ],
        check=True,
    )
    return contract


def build(output: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise RuntimeError(f"fresh V23 output root already exists: {output_root}")
    if output.exists():
        raise RuntimeError(f"V23 contract already exists: {output}")
    v16 = _read(V16_REFERENCE)
    policy = _read(MARGIN_POLICY)
    canaries = _canary_records()
    envelope = _build_resource_envelope(policy, canaries)
    _write(RESOURCE_ENVELOPE, envelope)

    source_projection, source_files = _source_projection(v16)
    _write(SOURCE_PROJECTION, source_projection)
    dataset = _dataset_identity(v16)
    runtime = _runtime_audit(dataset, source_projection)
    _write(RUNTIME_AUDIT, runtime)
    backend = _backend_audit(canaries, envelope, runtime)
    _write(BACKEND_AUDIT, backend)
    gate = _scientific_gate(source_projection, backend, runtime)
    _write(SCIENTIFIC_GATE, gate)
    contract = _build_contract(
        v16=v16,
        dataset=dataset,
        envelope=envelope,
        source_projection=source_projection,
        source_files=source_files,
        runtime=runtime,
        backend=backend,
        gate=gate,
        output_root=output_root,
    )
    _write(output, contract)
    verify_v23_pilot_contract(output)
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
    if args.verify:
        contract = verify_v23_pilot_contract(args.output.resolve())
    else:
        contract = build(
            args.output.resolve(),
            args.output_root.resolve(),
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
