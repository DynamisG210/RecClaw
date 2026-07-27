#!/usr/bin/env python3
"""Freeze the native-Linux Campaign training runtime release V8."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_dataset import (
    campaign_partition_profile,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_runtime_profile,
    campaign_training_profile,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
    filesystem_capability_policy_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    _directory_content_manifest,
    _live_torch_cuda_environment,
)


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
)


def _refresh_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    refreshed = []
    for row in rows:
        path = ROOT / str(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        refreshed.append(
            {
                "path": str(row["path"]),
                "sha256": bytes_sha256(path.read_bytes()),
            }
        )
    return refreshed


def _package_versions(
    python_executable: Path, names: list[str]
) -> dict[str, str]:
    return json.loads(
        subprocess.check_output(
            [
                str(python_executable),
                "-c",
                (
                    "import importlib.metadata as m,json,sys;"
                    "names=json.loads(sys.argv[1]);"
                    "print(json.dumps({name:m.version(name) for name in names},"
                    "sort_keys=True,separators=(',',':')))"
                ),
                json.dumps(sorted(names)),
            ],
            text=True,
        )
    )


def _scipy_compatibility(
    python_executable: Path,
    recbole_root: Path,
) -> dict[str, object]:
    return json.loads(
        subprocess.check_output(
            [
                str(python_executable),
                "-c",
                (
                    "import json,sys;"
                    "sys.path.insert(0,sys.argv[1]);"
                    "sys.path.insert(0,sys.argv[2]);"
                    "from scipy.sparse import dok_matrix;"
                    "before=hasattr(dok_matrix((2,2),dtype=float),'_update');"
                    "import run_candidate;"
                    "run_candidate.patch_recbole_runtime_compat();"
                    "m=dok_matrix((2,2),dtype=float);"
                    "m._update({(0,1):1.0});"
                    "print(json.dumps({'dok_update':m.toarray().tolist(),"
                    "'native_dok_update':before,"
                    "'package_patch_applied':not before},"
                    "sort_keys=True,separators=(',',':')))"
                ),
                (ROOT / "scripts").as_posix(),
                recbole_root.as_posix(),
            ],
            text=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--recbole-root", type=Path, required=True)
    parser.add_argument(
        "--recbole-commit",
        default="7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
    )
    parser.add_argument(
        "--recbole-tree",
        default="ca6386c4121ce2aae478ced7e136894ac1d7c218",
    )
    args = parser.parse_args()

    python_executable = args.python.absolute()
    data_path = args.data_path.resolve()
    recbole_root = args.recbole_root.resolve()
    base = json.loads(
        (RESOURCE_ROOT / "training_runtime_release_v7.json").read_text(
            encoding="utf-8"
        )
    )
    source_manifest = _refresh_rows(base["source_manifest"])
    training_config = {
        **base["training_config"],
        "files": _refresh_rows(base["training_config"]["files"]),
    }
    sources = {row["path"]: row["sha256"] for row in source_manifest}
    pip_freeze = subprocess.check_output(
        [str(python_executable), "-m", "pip", "freeze", "--all"]
    )
    package_names = list(
        base["backend_identity"]["python_package_versions"]
    )
    package_versions = _package_versions(
        python_executable, package_names
    )
    torch_cuda = _live_torch_cuda_environment(python_executable)
    if (
        not torch_cuda.get("cuda_available")
        or torch_cuda.get("primary_device_name")
        != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError("V8 must be frozen on the qualified RTX 4090 backend")
    recbole_manifest = _directory_content_manifest(recbole_root)
    if not recbole_manifest:
        raise RuntimeError("V8 RecBole source manifest is empty")
    backend_identity = {
        **base["backend_identity"],
        "backend_class": "LAB_GPU5_RTX4090_NATIVE_LINUX_V1",
        "filesystem_mode": "HASH_AUDITED_PRIVATE_ROOT_V1",
        "pip_freeze_digest": hashlib.sha256(pip_freeze).hexdigest(),
        "python_executable_sha256": bytes_sha256(
            python_executable.read_bytes()
        ),
        "python_package_versions": package_versions,
        "python_version": subprocess.check_output(
            [
                str(python_executable),
                "-c",
                "import platform;print(platform.python_version())",
            ],
            text=True,
        ).strip(),
        "recbole_commit": args.recbole_commit,
        "recbole_source_manifest": recbole_manifest,
        "recbole_source_manifest_digest": sha256_digest(
            recbole_manifest
        ),
        "recbole_tree": args.recbole_tree,
        "recbole_version": package_versions["recbole"],
        "scipy_sparse_compatibility": _scipy_compatibility(
            python_executable,
            recbole_root,
        ),
        "system": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "torch_cuda_environment": torch_cuda,
        "torch_cuda_environment_digest": sha256_digest(torch_cuda),
    }
    lock = {
        "backend_class": backend_identity["backend_class"],
        "pip_freeze_digest": backend_identity["pip_freeze_digest"],
        "python_executable_sha256": backend_identity[
            "python_executable_sha256"
        ],
        "python_package_versions": package_versions,
        "python_version": backend_identity["python_version"],
        "torch_cuda_environment_digest": backend_identity[
            "torch_cuda_environment_digest"
        ],
    }
    lock_path = RESOURCE_ROOT / "training_runtime_v8_lock.json"
    lock_path.write_bytes(canonical_json_bytes(lock) + b"\n")
    environment_lock = {
        "files": [
            {
                "path": lock_path.relative_to(ROOT).as_posix(),
                "sha256": bytes_sha256(lock_path.read_bytes()),
            }
        ],
        "kind": "PIP_FREEZE_NATIVE_LINUX_V3",
    }
    read_contract = {
        **base["read_contract"],
        "dataset_files": {
            name: bytes_sha256(
                (data_path / str(base["read_contract"]["dataset_dir"]) / name)
                .read_bytes()
            )
            for name in base["read_contract"]["dataset_files"]
        },
    }
    runtime_profile = campaign_runtime_profile()
    base.update(
        {
            "allowed_read_roots_policy_digest": sha256_digest(
                read_contract
            ),
            "backend_identity": backend_identity,
            "campaign_runtime_profile_digest": runtime_profile[
                "profile_digest"
            ],
            "confinement_contract": {
                "candidate_executable_code": "FORBIDDEN",
                "default_relative_log": "PRIVATE_WORKING_DIRECTORY/log",
                "environment_writable_roots": "RUN_PRIVATE",
                "mount_namespace": "UNAVAILABLE_ON_BACKEND",
                "one_spawn_per_claim": True,
                "protected_shared_roots": "HASH_BEFORE_AFTER",
                "sibling_arm_roots": "HASH_BEFORE_AFTER_AND_NOT_IN_RUNNER_INPUT",
            },
            "environment_lock": environment_lock,
            "filesystem_capability_policy_digest": (
                filesystem_capability_policy_digest()
            ),
            "partition_profile_digest": sha256_digest(
                campaign_partition_profile()
            ),
            "python_environment_lock_digest": sha256_digest(
                environment_lock
            ),
            "read_contract": read_contract,
            "recbole_identity_digest": sha256_digest(
                {
                    "recbole_commit": args.recbole_commit,
                    "recbole_tree": args.recbole_tree,
                    "recbole_version": package_versions["recbole"],
                }
            ),
            "release_id": "TRAINING_RUNTIME_RELEASE_V8",
            "source_manifest": source_manifest,
            "training_config": training_config,
            "training_profile_digest": sha256_digest(
                campaign_training_profile()
            ),
        }
    )
    base["launcher_source_digest"] = sources[
        "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py"
    ]
    base["runner_entrypoint_digest"] = sources[
        "scripts/campaign_train_worker.py"
    ]
    base["runner_source_digest"] = sources[
        "scripts/campaign_train_worker.py"
    ]
    base["close_result_policy_digest"] = sha256_digest(
        {
            "close_contract": base["close_contract"],
            "guard_source": sources[
                "src/recclaw_core/experiments/helix_abc_v1/"
                "training_execution_guard.py"
            ],
        }
    )
    base["common_guard_policy_digest"] = sha256_digest(
        {
            "guard": base["close_contract"]["common_guard"],
            "source": sources[
                "src/recclaw_core/experiments/helix_abc_v1/"
                "training_execution_guard.py"
            ],
        }
    )
    base["metric_parser_digest"] = sha256_digest(
        {
            "metric_contract": base["metric_contract"],
            "worker_source": sources["scripts/campaign_train_worker.py"],
        }
    )
    base["config_schema_digest"] = sha256_digest(training_config)
    base["confinement_policy_digest"] = sha256_digest(
        base["confinement_contract"]
    )
    base["package_owned_handler_registry_digest"] = sha256_digest(
        {
            "campaign_runtime_profile_digest": runtime_profile[
                "profile_digest"
            ],
            "closed_abis": [
                "recclaw.fake-non-training-runner.v1",
                "recclaw.package-owned-search-training-runner.v1",
                "recclaw.package-owned-search-training-runner.v2",
            ],
            "training_release_id": "TRAINING_RUNTIME_RELEASE_V8",
            "training_release_resource": "training_runtime_release_v8.json",
        }
    )
    output = RESOURCE_ROOT / "training_runtime_release_v8.json"
    output.write_bytes(canonical_json_bytes(base) + b"\n")
    print(
        json.dumps(
            {
                "release_digest": sha256_digest(base),
                "release_path": output.as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
