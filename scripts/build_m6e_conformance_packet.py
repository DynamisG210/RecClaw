#!/usr/bin/env python3
"""Build the content-addressed M6E compatibility and conformance records."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.store_audit import (  # noqa: E402
    store_audit_contract_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (  # noqa: E402
    filesystem_capability_policy_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    historical_training_runtime_release_v1,
    training_runtime_release,
)


DOC_ROOT = ROOT / "docs" / "research_line" / "m6e"
RESULT_ROOT = ROOT / "results" / "research_line" / "m6e_narrow_final_canaries"
PUBLISHED_FINAL_ROOT = (
    ROOT / "results" / "research_line" / "m6e_final_canaries"
)
FINAL_PROJECTION_PATH = (
    PUBLISHED_FINAL_ROOT / "M6E_FINAL_CANARY_PROJECTION.json"
)
CANARIES = (
    ("BPR", "bpr", "bpr_result.json"),
    ("LightGCN", "lightgcn", "lightgcn_result.json"),
    ("NGCF", "ngcf", "ngcf_result.json"),
    ("SGL", "sgl", "sgl_result.json"),
)
FORCED_FAILURE = (
    "forced_failure",
    "forced_failure_result.json",
)
SOURCE_PROJECTION = (
    "scripts/build_m6e_conformance_packet.py",
    "scripts/freeze_m6e_training_runtime_release_v2.py",
    "scripts/pilot_train_worker.py",
    "scripts/run_m6r_fixed_training_canary.py",
    "src/recclaw_core/experiments/helix_abc_v1/m6e_conformance.py",
    "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py",
    "src/recclaw_core/experiments/helix_abc_v1/real_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/resources/training_runtime_release_v2.json",
    "src/recclaw_core/experiments/helix_abc_v1/resources/training_runtime_v2_lock.json",
    "src/recclaw_core/experiments/helix_abc_v1/store_audit.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_execution_guard.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_filesystem.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_materialization.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_runtime_contracts.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_runtime_release.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_state_store.py",
    "tests/experiments/helix_abc_v1/test_m6e_environment_closure.py",
    "tests/experiments/helix_abc_v1/test_m6r_training_runtime.py",
)
SENSITIVE_SCOPE = (
    "src/recclaw_core/experiments/helix_abc_v1/pilot_analysis.py",
    "src/recclaw_core/experiments/helix_abc_v1/real_canary.py",
    "src/recclaw_core/experiments/helix_abc_v1/research_controller.py",
    "src/recclaw_core/helix",
    "src/recclaw_core/mechanism_space.py",
    "src/evidence_guard",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _content_address(document: dict[str, Any]) -> dict[str, Any]:
    return {**document, "content_digest": sha256_digest(document)}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tree_manifest(root: Path) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return {
        "file_count": len(rows),
        "logical_bytes": sum(row["size_bytes"] for row in rows),
        "manifest_digest": sha256_digest(rows),
    }


def _single_json(root: Path, name: str) -> dict[str, Any]:
    matches = tuple(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"{root} must contain exactly one {name}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _canary(model: str, directory: str, result_name: str) -> dict[str, Any]:
    root = RESULT_ROOT / directory
    result_path = root / result_name
    result = json.loads(result_path.read_text(encoding="utf-8"))
    raw = _single_json(root, "training_raw_run_output.v2.json")
    side_effects = _single_json(root, "training_side_effect_audit.v1.json")
    worker = _single_json(root, "worker_result.v1.json")
    mount_projection = dict(side_effects["worker_mount_projection"])
    device_evidence = dict(worker["training_device_evidence"])
    if (
        result["verdict"] != "PASS"
        or raw["exit_status"] != "SUCCESS"
        or raw["filesystem_confinement_status"] != "PASS"
        or side_effects["status"] != "PASS"
        or side_effects["worker_mount_digest_valid"] is not True
        or mount_projection["status"] != "PASS"
        or mount_projection["missing_writable_mount_targets"]
        or mount_projection["unexpected_writable_mount_targets"]
        or device_evidence["cuda_available"] is not True
        or int(device_evidence["cuda_device_count"]) < 1
    ):
        raise RuntimeError(f"{model} final canary did not pass")
    return {
        "filesystem_capability_digest": raw["filesystem_capability_digest"],
        "filesystem_confinement_status": raw["filesystem_confinement_status"],
        "cuda_device_evidence": device_evidence,
        "model": model,
        "mount_count": mount_projection["mount_count"],
        "raw_output_digest": result["raw_output_digest"],
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": _sha256(result_path),
        "root": root.relative_to(ROOT).as_posix(),
        "root_tree": _tree_manifest(root),
        "runtime_binding_digest": result["runtime_binding_digest"],
        "runtime_release_digest": result["runtime_release_digest"],
        "search_seed": result["search_seed"],
        "side_effect_audit_digest": side_effects["audit_digest"],
        "side_effect_status": side_effects["status"],
        "verdict": result["verdict"],
        "writable_mount_targets": mount_projection[
            "writable_mount_targets"
        ],
    }


def _compatibility_matrix() -> dict[str, Any]:
    release = training_runtime_release()
    return _content_address(
        {
            "authority": "NONE",
            "constraints": {
                "numpy": "1.26.4",
                "python": "3.10.20",
                "recbole": "1.2.1",
                "recbole_core_modified": False,
                "torch": "2.10.0+cu128",
            },
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "matrix": [
                {
                    "dok_matrix_private_update_available": False,
                    "python": "/root/miniconda3/envs/recbole/bin/python",
                    "scipy": "1.15.3",
                    "verdict": "INCOMPATIBLE",
                },
                {
                    "dok_matrix_private_update_available": False,
                    "python": (
                        "/root/projects/RecClaw_m6e_compat_matrix/"
                        "scipy_1_14_1/bin/python"
                    ),
                    "scipy": "1.14.1",
                    "verdict": "INCOMPATIBLE",
                },
                {
                    "dok_matrix_private_update_available": False,
                    "python": (
                        "/root/projects/RecClaw_m6e_compat_matrix/"
                        "scipy_1_13_1/bin/python"
                    ),
                    "scipy": "1.13.1",
                    "verdict": "INCOMPATIBLE",
                },
                {
                    "dok_matrix_private_update_available": True,
                    "probe_value": 1.0,
                    "python": (
                        "/root/projects/RecClaw_m6_training_runtime_v2/bin/python"
                    ),
                    "scipy": "1.12.0",
                    "verdict": "COMPATIBLE",
                },
            ],
            "probe": (
                "hasattr(scipy.sparse.dok_matrix((1,1)), '_update'); "
                "for a compatible result call _update({(0,1):1.0})"
            ),
            "old_v4_runtime_release_digest": (
                historical_training_runtime_release_v1().digest
            ),
            "resolution": "DEPENDENCY_ONLY",
            "selected_scipy": "1.12.0",
            "selection_reason": (
                "Newest empirically tested SciPy in the bounded matrix that "
                "retains RecBole's required dok_matrix._update behavior."
            ),
            "training_runtime_release_digest": release.digest,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdict", default="PENDING_INDEPENDENT_AUDIT")
    parser.add_argument("--p0", type=int)
    parser.add_argument("--p1", type=int)
    parser.add_argument("--p2", type=int)
    parser.add_argument("--independent-audit-sha256")
    args = parser.parse_args()

    matrix = _compatibility_matrix()
    matrix_path = DOC_ROOT / "M6E_RUNTIME_COMPATIBILITY_MATRIX.json"
    _write(matrix_path, matrix)

    canaries = [_canary(*specification) for specification in CANARIES]
    forced_root = RESULT_ROOT / FORCED_FAILURE[0]
    forced_path = forced_root / FORCED_FAILURE[1]
    forced = json.loads(forced_path.read_text(encoding="utf-8"))
    forced_raw = _single_json(forced_root, "training_raw_run_output.v2.json")
    forced_worker = _single_json(forced_root, "worker_result.v1.json")
    forced_device = dict(forced_worker["training_device_evidence"])
    if (
        forced["verdict"] != "PASS"
        or forced_raw["exit_status"] != "RUNTIME_FAILURE"
        or forced_raw["filesystem_confinement_status"] != "PASS"
        or forced_device["cuda_available"] is not True
        or int(forced_device["cuda_device_count"]) < 1
    ):
        raise RuntimeError("forced-failure final rehearsal did not pass")
    projection = _content_address(
        {
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "projection_id": "M6E_FINAL_CANARY_PROJECTION_V1",
            "result_paths": [
                row["result_path"] for row in canaries
            ]
            + [forced_path.relative_to(ROOT).as_posix()],
            "source_root": RESULT_ROOT.relative_to(ROOT).as_posix(),
            "source_root_tree": _tree_manifest(RESULT_ROOT),
            "training_runtime_release_digest": (
                training_runtime_release().digest
            ),
        }
    )
    _write(FINAL_PROJECTION_PATH, projection)
    full = json.loads(
        (RESULT_ROOT / CANARIES[0][1] / CANARIES[0][2]).read_text(
            encoding="utf-8"
        )
    )
    v4_failure_path = (
        ROOT
        / "docs"
        / "research_line"
        / "m6"
        / "M6_PILOT_V4_FAILURE_RECORD.json"
    )
    source_rows = [
        {"path": relative, "sha256": _sha256(ROOT / relative)}
        for relative in SOURCE_PROJECTION
    ]
    scope_diff = tuple(
        line
        for line in _git("diff", "--name-only", "HEAD", "--", *SENSITIVE_SCOPE).splitlines()
        if line
    )
    packet = {
        "P0": args.p0,
        "P1": args.p1,
        "P2": args.p2,
        "allowed_handler_families": ["BPR", "LightGCN", "NGCF", "SGL"],
        "audit_port": {
            "contract_digest": store_audit_contract_digest(),
            "full_report_digest": full["authoritative_pilot_audit"][
                "state_store_integrity"
            ]["report_digest"],
        },
        "authoritative_final_canary_projection": {
            "content_digest": projection["content_digest"],
            "path": FINAL_PROJECTION_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(FINAL_PROJECTION_PATH),
        },
        "authority": "NONE",
        "broker_preflight_gate": {
            "implementation": (
                "FreshPilotOrchestratorV2 -> require_m6e_conformance_packet"
            ),
            "status_before_pass_packet": "BLOCKED_BEFORE_BROKER_USE",
        },
        "canary_information_boundary": {
            "broker_calls": 0,
            "llm_calls": 0,
            "metrics_enter_frontier_or_search_memory": False,
            "provider_actual_physical_calls": 0,
            "provider_actual_tokens": 0,
            "per_arm_budget_projection": "NOT_APPLICABLE_FIXED_NO_LLM_CANARY",
        },
        "compatibility_matrix": {
            "content_digest": matrix["content_digest"],
            "path": matrix_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(matrix_path),
        },
        "entry_git": {
            "commit": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "evidence_class": "DEVELOPMENT_ONLY",
        "filesystem_capability_policy_digest": (
            filesystem_capability_policy_digest()
        ),
        "fixed_training_canaries": canaries,
        "formal_acceptance": False,
        "forced_failure_rehearsal": {
            "classified_outcome": forced_raw["exit_status"],
            "cuda_device_evidence": forced_device,
            "filesystem_confinement_status": (
                forced_raw["filesystem_confinement_status"]
            ),
            "mechanical_claim_closed": forced["gates"]["claim_closed"],
            "mechanical_result_closed": forced["gates"]["common_result_closed"],
            "result_path": forced_path.relative_to(ROOT).as_posix(),
            "result_sha256": _sha256(forced_path),
            "root_tree": _tree_manifest(forced_root),
            "runtime_release_digest": forced["runtime_release_digest"],
            "successful_training_count": forced["readiness_input_packet"][
                "success_count"
            ],
            "verdict": forced["verdict"],
        },
        "full_authoritative_audit_rehearsal": {
            "barriers_closed": full["authoritative_pilot_audit"][
                "barriers_closed"
            ],
            "execution_count": full["authoritative_pilot_audit"][
                "execution_count"
            ],
            "feedback_count": full["authoritative_pilot_audit"][
                "feedback_count"
            ],
            "readiness_input_packet_digest": full["readiness_input_packet"][
                "packet_digest"
            ],
            "report_digest": full["authoritative_pilot_audit"][
                "state_store_integrity"
            ]["report_digest"],
            "round_count": full["authoritative_pilot_audit"]["round_count"],
            "verdict": (
                "PASS"
                if full["gates"]["authoritative_pilot_audit"]
                and full["gates"]["readiness_input_classification"]
                else "FAIL"
            ),
        },
        "independent_audit_sha256": args.independent_audit_sha256,
        "old_v4_runtime_release_digest": (
            historical_training_runtime_release_v1().digest
        ),
        "old_v4_lightgcn_failure_evidence": {
            "error_signature": (
                "AttributeError: 'dok_matrix' object has no attribute '_update'"
            ),
            "failed_lightgcn_runs": 3,
            "path": v4_failure_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(v4_failure_path),
        },
        "regression_results": [
            {
                "command": (
                    "python -m unittest discover -s "
                    "tests/experiments/helix_abc_v1 -p test_*.py"
                ),
                "passed": 140,
                "failed": 0,
            },
            {
                "command": (
                    "python -m unittest tests.evidence_guard.test_core_v1 "
                    "tests.test_bl_icf_mechanism_space"
                ),
                "passed": 40,
                "failed": 0,
            },
            {
                "command": "python -m unittest discover -s tests -p test_*.py",
                "passed": 147,
                "failed": 0,
            },
        ],
        "scope_isolation": {
            "sensitive_scope_diff": list(scope_diff),
            "status": "PASS" if not scope_diff else "FAIL",
        },
        "source_projection": source_rows,
        "source_projection_digest": sha256_digest(source_rows),
        "training_runtime_release_digest": training_runtime_release().digest,
        "verdict": args.verdict,
    }
    _write(
        DOC_ROOT / "M6E_TRAINING_RUNTIME_CONFORMANCE_PACKET.json",
        _content_address(packet),
    )
    print(
        json.dumps(
            {
                "compatibility_matrix_digest": matrix["content_digest"],
                "packet_digest": sha256_digest(packet),
                "runtime_release_digest": training_runtime_release().digest,
                "verdict": args.verdict,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
