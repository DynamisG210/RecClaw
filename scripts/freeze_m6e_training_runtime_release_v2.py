#!/usr/bin/env python3
"""Freeze the exact package-owned M6E training runtime release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.store_audit import (  # noqa: E402
    store_audit_contract_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (  # noqa: E402
    filesystem_capability_policy_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (  # noqa: E402
    CandidateExecutionBindingV3,
    CommonExecutionPermitV2,
    CommonResultClosureV2,
    ExecutionStartConfirmationV1,
    ExecutionStartReceiptV2,
    RawResultEnvelopeV2,
    RuntimeProfileIdV1,
    TrainingRawRunOutputV2,
    TrainingResourceAccountingV1,
    TrainingRuntimeBindingV2,
    TrainingRuntimePlanDecisionV1,
    TrainingRuntimeReleaseV2,
)
from recclaw_core.experiments.helix_abc_v1.training_state_store import (  # noqa: E402
    TRAINING_CLAIM_IDENTITY_FIELDS,
    TRAINING_SCHEMA_VERSION,
)


RESOURCE_ROOT = (
    ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
LOCK_PATH = RESOURCE_ROOT / "training_runtime_v2_lock.json"
RELEASE_PATH = RESOURCE_ROOT / "training_runtime_release_v2.json"
TRAINING_RUNNER_ABI = "recclaw.package-owned-search-training-runner.v1"


def _schema_digest(record_type: type[object]) -> str:
    return sha256_digest(
        {
            "record_type": record_type.record_type,
            "required_fields": sorted(record_type.required_fields),
            "optional_fields": sorted(record_type.optional_fields),
        }
    )


def _source_row(path: str) -> dict[str, str]:
    return {"path": path, "sha256": bytes_sha256((ROOT / path).read_bytes())}


def _torch_cuda_identity(python: Path) -> dict[str, object]:
    script = (
        "import json,torch;"
        "v={'cuda_available':torch.cuda.is_available(),"
        "'cudnn_version':torch.backends.cudnn.version(),"
        "'device_count':torch.cuda.device_count(),"
        "'torch_cuda_build':torch.version.cuda,"
        "'torch_version':torch.__version__};"
        "v.update({'primary_device_capability':list(torch.cuda.get_device_capability(0)),"
        "'primary_device_name':torch.cuda.get_device_name(0)}"
        " if torch.cuda.is_available() else {});"
        "print(json.dumps(v,sort_keys=True,separators=(',',':')))"
    )
    result = json.loads(
        subprocess.check_output([str(python), "-c", script], text=True)
    )
    nvidia_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    result["nvidia_driver_version"] = (
        subprocess.check_output(
            [
                str(nvidia_smi),
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .splitlines()[0]
        .strip()
        if nvidia_smi.is_file()
        else "UNAVAILABLE"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--recbole-root", required=True, type=Path)
    args = parser.parse_args()
    python = args.python.absolute()
    recbole_root = args.recbole_root.resolve()
    package_names = ("numpy", "recbole", "rfc8785", "scipy", "torch")
    observed_versions = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata as m,json,sys;"
                    "names=json.loads(sys.argv[1]);"
                    "print(json.dumps({n:m.version(n) for n in names},"
                    "sort_keys=True,separators=(',',':')))"
                ),
                json.dumps(package_names),
            ],
            text=True,
        )
    )
    versions = observed_versions
    if observed_versions["scipy"] != "1.12.0":
        raise RuntimeError("runtime package identities do not match M6E selection")
    sparse_probe = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                (
                    "import json,scipy.sparse as s;"
                    "x=s.dok_matrix((2,2));"
                    "x._update({(0,1):1.0});"
                    "print(json.dumps({'dok_update':x.toarray().tolist()},"
                    "sort_keys=True,separators=(',',':')))"
                ),
            ],
            text=True,
        )
    )
    pip_freeze = subprocess.check_output(
        [str(python), "-m", "pip", "freeze", "--all"]
    )
    recbole_commit = subprocess.check_output(
        ["git", "-C", str(recbole_root), "rev-parse", "HEAD"], text=True
    ).strip()
    recbole_tree = subprocess.check_output(
        ["git", "-C", str(recbole_root), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(recbole_root), "status", "--porcelain"],
        text=True,
    ):
        raise RuntimeError("RecBole runtime source is not clean")
    torch_cuda = _torch_cuda_identity(python)
    lock = {
        "lock_schema": "recclaw.training-runtime-v2-lock.v1",
        "package_versions": versions,
        "pip_freeze_digest": bytes_sha256(pip_freeze),
        "python_executable_sha256": bytes_sha256(python.read_bytes()),
        "python_version": subprocess.check_output(
            [str(python), "-c", "import platform;print(platform.python_version())"],
            text=True,
        ).strip(),
        "recbole_commit": recbole_commit,
        "recbole_tree": recbole_tree,
        "scipy_sparse_probe": sparse_probe,
        "torch_cuda_environment": torch_cuda,
    }
    LOCK_PATH.write_bytes(canonical_json_bytes(lock) + b"\n")

    source_paths = [
        "scripts/pilot_train_worker.py",
        "src/recclaw_core/experiments/helix_abc_v1/migrations/002_training_runtime_release.sql",
        "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py",
        "src/recclaw_core/experiments/helix_abc_v1/state_store.py",
        "src/recclaw_core/experiments/helix_abc_v1/store_audit.py",
        "src/recclaw_core/experiments/helix_abc_v1/training_execution_guard.py",
        "src/recclaw_core/experiments/helix_abc_v1/training_filesystem.py",
        "src/recclaw_core/experiments/helix_abc_v1/training_materialization.py",
        "src/recclaw_core/experiments/helix_abc_v1/training_runtime_contracts.py",
        "src/recclaw_core/experiments/helix_abc_v1/training_runtime_release.py",
        "src/recclaw_core/experiments/helix_abc_v1/training_state_store.py",
    ]
    source_manifest = [_source_row(path) for path in source_paths]
    source_map = {row["path"]: row["sha256"] for row in source_manifest}
    profile_path = (
        "src/recclaw_core/experiments/helix_abc_v1/resources/"
        "pilot_training_profile_v1.json"
    )
    training_config_paths = [
        "configs/lightgcn_metrics.yaml",
        "configs/task_ml1m.yaml",
        "scripts/run_candidate.py",
        profile_path,
    ]
    training_config = {
        "files": [_source_row(path) for path in training_config_paths],
        "optimizer": "ADAM",
        "reproducibility": True,
    }
    read_contract = {
        "dataset": "ML-1M",
        "dataset_dir": "ml-1m",
        "dataset_files": {
            "ml-1m.inter": "e943abb91013a54c385828fdf5ab4ce49e957ca3a772adb30cde2a7d5539b389",
            "ml-1m.item": "6eff097333db47adf003039eb90cb6ab13978fa9577d915d2366e4d073844dc3",
            "ml-1m.user": "bbc7b4f8d3204d00e0465cf1451b8e8bae2c9e94378292223603c229cf2df09f",
        },
        "generated_candidate_input": "HANDLER_CONFIG_CANONICAL_JSON",
        "recbole_source": "READ_ONLY_MOUNT_NAMESPACE",
    }
    write_contract = {
        "allowed_outputs": [
            "CHECKPOINT_DIRECTORY",
            "PRIVATE_DEFAULT_RECBOLE_LOG",
            "TRAINING_LOG",
            "WORKER_RESULT",
            "START_CONFIRMATION",
            "PRIVATE_CACHE_AND_TEMP",
        ],
        "root": "EXACT_RUN_PRIVATE_ROOT",
    }
    confinement_contract = {
        "candidate_executable_code": "FORBIDDEN",
        "default_relative_log": "PRIVATE_WORKING_DIRECTORY/log",
        "environment_writable_roots": "PRIVATE",
        "mount_namespace": "PRIVATE",
        "one_spawn_per_claim": True,
        "protected_shared_roots": "READ_ONLY_ROOT_BIND_MOUNT",
        "sibling_arm_roots": "NOT_IN_RUNNER_INPUT",
    }
    close_contract = {
        "common_guard": "CommonTrainingExecutionGuardV1",
        "raw_result_contract": "RawResultEnvelopeV2",
        "receipt_contract": "ExecutionStartReceiptV2",
        "resource_accounting_contract": "TrainingResourceAccountingV1",
        "start_confirmation_contract": "ExecutionStartConfirmationV1",
    }
    metric_contract = {
        "candidate_universe": "FULL_SORT",
        "metric": "NDCG@10",
        "nonfinite_policy": "REJECT",
        "source": "RECBOLE_TEST_RESULT",
    }
    resource_contract = {
        "gpu_cost_meter": "NORMALIZED_MICROUNITS",
        "max_ordinary_runner_starts": 1,
        "required_axes": [
            "ROUND",
            "EXECUTION_COUNT",
            "BILLED_TOKEN",
            "GPU_NORMALIZED_COST",
        ],
        "wall_time_meter": "MONOTONIC_MILLISECONDS",
    }
    backend_identity = {
        "pip_freeze_digest": bytes_sha256(pip_freeze),
        "python_executable_sha256": bytes_sha256(python.read_bytes()),
        "python_package_versions": versions,
        "python_version": lock["python_version"],
        "recbole_commit": recbole_commit,
        "recbole_tree": recbole_tree,
        "recbole_version": versions["recbole"],
        "scipy_sparse_compatibility": sparse_probe,
        "torch_cuda_environment": torch_cuda,
        "torch_cuda_environment_digest": sha256_digest(torch_cuda),
    }
    migration_path = (
        ROOT
        / "src/recclaw_core/experiments/helix_abc_v1/migrations/"
        "002_training_runtime_release.sql"
    )
    environment_lock = {
        "files": [
            {
                "path": LOCK_PATH.relative_to(ROOT).as_posix(),
                "sha256": bytes_sha256(LOCK_PATH.read_bytes()),
            }
        ],
        "kind": "PIP_FREEZE_VENV_V1",
    }
    handler_registry = {
        "closed_abis": [
            "recclaw.fake-non-training-runner.v1",
            TRAINING_RUNNER_ABI,
        ],
        "fake_release_resource": "runtime_release_v1.json",
        "training_release_resource": "training_runtime_release_v2.json",
        "training_release_id": "TRAINING_RUNTIME_RELEASE_V2",
    }
    release = {
        "allowed_read_roots_policy_digest": sha256_digest(read_contract),
        "allowed_write_roots_policy_digest": sha256_digest(write_contract),
        "backend_identity": backend_identity,
        "candidate_binding_schema_digest": _schema_digest(
            CandidateExecutionBindingV3
        ),
        "close_contract": close_contract,
        "close_result_policy_digest": sha256_digest(
            {
                "close_contract": close_contract,
                "guard_source": source_map[
                    "src/recclaw_core/experiments/helix_abc_v1/"
                    "training_execution_guard.py"
                ],
            }
        ),
        "close_result_schema_digest": _schema_digest(CommonResultClosureV2),
        "common_guard_policy_digest": sha256_digest(
            {
                "guard": close_contract["common_guard"],
                "source": source_map[
                    "src/recclaw_core/experiments/helix_abc_v1/"
                    "training_execution_guard.py"
                ],
            }
        ),
        "config_schema_digest": sha256_digest(training_config),
        "confinement_contract": confinement_contract,
        "confinement_policy_digest": sha256_digest(confinement_contract),
        "environment_lock": environment_lock,
        "filesystem_capability_policy_digest": (
            filesystem_capability_policy_digest()
        ),
        "launch_protocol_id": "CLAIM_PREPARE_CONFIRM_START_V1",
        "launcher_abi": "recclaw.package-owned-training-launcher.v1",
        "launcher_source_digest": source_map[
            "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py"
        ],
        "metric_contract": metric_contract,
        "metric_parser_digest": sha256_digest(
            {
                "metric_contract": metric_contract,
                "worker_source": source_map["scripts/pilot_train_worker.py"],
            }
        ),
        "package_owned_handler_registry_digest": sha256_digest(
            handler_registry
        ),
        "permit_schema_digest": _schema_digest(CommonExecutionPermitV2),
        "profile_id": RuntimeProfileIdV1.PACKAGE_TRAINING_V2.value,
        "python_environment_lock_digest": sha256_digest(environment_lock),
        "raw_output_schema_digest": _schema_digest(TrainingRawRunOutputV2),
        "raw_result_schema_digest": _schema_digest(RawResultEnvelopeV2),
        "read_contract": read_contract,
        "receipt_schema_digest": _schema_digest(ExecutionStartReceiptV2),
        "recbole_identity_digest": sha256_digest(
            {
                "recbole_commit": recbole_commit,
                "recbole_tree": recbole_tree,
                "recbole_version": versions["recbole"],
            }
        ),
        "release_id": "TRAINING_RUNTIME_RELEASE_V2",
        "resource_accounting_schema_digest": _schema_digest(
            TrainingResourceAccountingV1
        ),
        "resource_contract": resource_contract,
        "resource_meter_policy_digest": sha256_digest(resource_contract),
        "runner_abi": TRAINING_RUNNER_ABI,
        "runner_entrypoint_digest": source_map["scripts/pilot_train_worker.py"],
        "runner_source_digest": source_map["scripts/pilot_train_worker.py"],
        "runtime_binding_schema_digest": _schema_digest(
            TrainingRuntimeBindingV2
        ),
        "source_manifest": source_manifest,
        "start_confirmation_schema_digest": _schema_digest(
            ExecutionStartConfirmationV1
        ),
        "state_store_claim_schema_digest": sha256_digest(
            {
                "identity_fields": list(TRAINING_CLAIM_IDENTITY_FIELDS),
                "migration_sha256": bytes_sha256(migration_path.read_bytes()),
                "schema_version": TRAINING_SCHEMA_VERSION,
            }
        ),
        "store_audit_contract_digest": store_audit_contract_digest(),
        "supported_execution_purposes": [
            "DEVELOPMENT_FIXED_TRAINING_CANARY",
            "DEVELOPMENT_PILOT_OFFLINE_TOPN",
        ],
        "training_config": training_config,
        "training_plan_schema_digest": _schema_digest(
            TrainingRuntimePlanDecisionV1
        ),
        "training_profile_digest": sha256_digest(
            json.loads((ROOT / profile_path).read_bytes())
        ),
        "write_contract": write_contract,
    }
    RELEASE_PATH.write_bytes(canonical_json_bytes(release) + b"\n")
    print(
        json.dumps(
            {
                "lock_sha256": bytes_sha256(LOCK_PATH.read_bytes()),
                "release_digest": TrainingRuntimeReleaseV2(release).digest,
                "release_sha256": bytes_sha256(RELEASE_PATH.read_bytes()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
