"""Content-bound registry, purpose binding and preflight for M6R training."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .campaign_dataset import campaign_partition_profile
from .campaign_runtime import (
    campaign_readiness_failures,
    campaign_runtime_profile,
)
from .canonical import bytes_sha256, sha256_digest
from .runtime_release import runtime_release_contract, runtime_release_digest
from .store_audit import store_audit_contract_digest
from .training_filesystem import (
    build_training_filesystem_capability,
    filesystem_capability_policy_digest,
)
from .training_runtime_contracts import (
    CandidateExecutionBindingV3,
    CommonExecutionPermitV2,
    CommonResultClosureV2,
    ExecutionStartConfirmationV1,
    ExecutionStartReceiptV2,
    RawResultEnvelopeV2,
    RuntimeProfileIdV1,
    TrainingCompatibilityStatusV1,
    TrainingExecutionPurposeV1,
    TrainingRawRunOutputV2,
    TrainingResourceAccountingV1,
    TrainingRuntimeBindingV2,
    TrainingRuntimeCompatibilityFixtureV1,
    TrainingRuntimeCompatibilityPreflightV1,
    TrainingRuntimePlanDecisionV1,
    TrainingRuntimeReleaseV1,
    TrainingRuntimeReleaseV2,
    TrainingRuntimeReleaseV3,
)


RESOURCE_PACKAGE = "recclaw_core.experiments.helix_abc_v1.resources"
HISTORICAL_TRAINING_RELEASE_V1_RESOURCE = "training_runtime_release_v1.json"
TRAINING_RELEASE_RESOURCE = "training_runtime_release_v2.json"
TRAINING_RUNNER_ABI = "recclaw.package-owned-search-training-runner.v1"
TRAINING_LAUNCHER_ABI = "recclaw.package-owned-training-launcher.v1"
TRAINING_LAUNCH_PROTOCOL_ID = "CLAIM_PREPARE_CONFIRM_START_V1"
CAMPAIGN_TRAINING_RELEASE_ID = "TRAINING_RUNTIME_RELEASE_V17"
CAMPAIGN_TRAINING_RELEASE_RESOURCE = "training_runtime_release_v17.json"
CAMPAIGN_TRAINING_RUNNER_ABI = (
    "recclaw.package-owned-search-training-runner.v2"
)
CAMPAIGN_TRAINING_LAUNCHER_ABI = (
    "recclaw.package-owned-training-launcher.v2"
)
CAMPAIGN_TRAINING_LAUNCH_PROTOCOL_ID = (
    "CLAIM_PREPARE_CONFIRM_START_V2"
)

_COMPONENT_NAMES = (
    "campaign_contract",
    "runtime_release",
    "purpose_binding",
    "handler_registry",
    "candidate_binding",
    "common_guard",
    "permit_template",
    "state_store_claim",
    "launcher",
    "receipt_schema",
    "raw_result_schema",
    "close_result",
    "metric_policy",
    "resource_policy",
    "filesystem_capability",
    "store_audit",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resource_json(name: str) -> dict[str, Any]:
    value = json.loads(resources.files(RESOURCE_PACKAGE).joinpath(name).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"package resource {name} must contain an object")
    return value


def _schema_digest(record_type: type[Any]) -> str:
    return sha256_digest(
        {
            "record_type": record_type.record_type,
            "required_fields": sorted(record_type.required_fields),
            "optional_fields": sorted(record_type.optional_fields),
        }
    )


def _source_manifest_map(
    release: TrainingRuntimeReleaseV1 | TrainingRuntimeReleaseV2,
) -> dict[str, str]:
    return {str(row["path"]): str(row["sha256"]) for row in release.source_manifest}


def _handler_registry_projection(
    runner_abi: str = TRAINING_RUNNER_ABI,
) -> dict[str, Any]:
    if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI:
        return {
            "campaign_runtime_profile_digest": (
                campaign_runtime_profile()["profile_digest"]
            ),
            "closed_abis": [
                runtime_release_contract()["runner_abi"],
                TRAINING_RUNNER_ABI,
                CAMPAIGN_TRAINING_RUNNER_ABI,
            ],
            "training_release_id": CAMPAIGN_TRAINING_RELEASE_ID,
            "training_release_resource": (
                CAMPAIGN_TRAINING_RELEASE_RESOURCE
            ),
        }
    return {
        "closed_abis": [
            "recclaw.fake-non-training-runner.v1",
            TRAINING_RUNNER_ABI,
        ],
        "fake_release_resource": "runtime_release_v1.json",
        "training_release_resource": TRAINING_RELEASE_RESOURCE,
        "training_release_id": "TRAINING_RUNTIME_RELEASE_V2",
    }


def _live_torch_cuda_environment(python_executable: Path) -> dict[str, Any]:
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
    payload = json.loads(
        subprocess.check_output(
            [str(python_executable.absolute()), "-c", script],
            text=True,
            timeout=20,
        )
    )
    if not isinstance(payload, dict):
        raise ValueError("torch/CUDA identity probe did not return an object")
    nvidia_smi = next(
        (
            Path(path)
            for path in (
                shutil.which("nvidia-smi"),
                Path("/usr/lib/wsl/lib/nvidia-smi"),
                Path("/usr/bin/nvidia-smi"),
            )
            if path is not None and Path(path).is_file()
        ),
        None,
    )
    if nvidia_smi is not None:
        payload["nvidia_driver_version"] = subprocess.check_output(
            [
                str(nvidia_smi),
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).splitlines()[0].strip()
    else:
        payload["nvidia_driver_version"] = "UNAVAILABLE"
    return payload


def historical_training_runtime_release_v1() -> TrainingRuntimeReleaseV1:
    return TrainingRuntimeReleaseV1(
        _resource_json(HISTORICAL_TRAINING_RELEASE_V1_RESOURCE)
    )


def training_runtime_release() -> TrainingRuntimeReleaseV2:
    return TrainingRuntimeReleaseV2(_resource_json(TRAINING_RELEASE_RESOURCE))


def campaign_training_runtime_release() -> TrainingRuntimeReleaseV3:
    return TrainingRuntimeReleaseV3(
        _resource_json(CAMPAIGN_TRAINING_RELEASE_RESOURCE)
    )


def training_runtime_release_digest() -> str:
    return training_runtime_release().digest


def resolve_runtime_release(runner_abi: str) -> Mapping[str, Any]:
    """Resolve only the two package-owned runtime profiles."""

    if runner_abi == runtime_release_contract()["runner_abi"]:
        return {
            "profile_id": RuntimeProfileIdV1.FAKE_NON_TRAINING.value,
            "release_digest": runtime_release_digest(),
            "runner_abi": runner_abi,
        }
    if runner_abi == TRAINING_RUNNER_ABI:
        release = training_runtime_release()
        return {
            "profile_id": release.profile_id,
            "release_digest": release.digest,
            "runner_abi": release.runner_abi,
        }
    if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI:
        release = campaign_training_runtime_release()
        return {
            "profile_id": release.profile_id,
            "release_digest": release.digest,
            "runner_abi": release.runner_abi,
        }
    raise ValueError(f"unknown package runtime ABI: {runner_abi}")


def training_release_for_abi(
    runner_abi: str,
) -> TrainingRuntimeReleaseV2 | TrainingRuntimeReleaseV3:
    if runner_abi == TRAINING_RUNNER_ABI:
        return training_runtime_release()
    if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI:
        return campaign_training_runtime_release()
    raise ValueError(f"unknown package training ABI: {runner_abi}")


def resolve_bound_training_release(
    *,
    runner_abi: str,
    runtime_release_digest: str,
    execution_purpose: str,
) -> TrainingRuntimeReleaseV2 | TrainingRuntimeReleaseV3:
    """Resolve claim-bound identity without trusting later receipt bytes."""

    resolved = resolve_runtime_release(runner_abi)
    release = training_release_for_abi(runner_abi)
    if (
        runner_abi
        not in {TRAINING_RUNNER_ABI, CAMPAIGN_TRAINING_RUNNER_ABI}
        or resolved["release_digest"] != release.digest
        or runtime_release_digest != release.digest
        or execution_purpose not in set(release.supported_execution_purposes)
    ):
        raise ValueError("claim does not bind the package-owned training release")
    return release


def _git_identity(path: Path) -> tuple[str, str, str]:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        timeout=5,
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"],
        text=True,
        timeout=5,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"],
        text=True,
        timeout=5,
    )
    return commit, tree, status


def _directory_content_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = path.resolve()
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [
            name
            for name in sorted(names)
            if name not in {".git", "__pycache__"}
        ]
        directory_path = Path(directory)
        for name in sorted(files):
            file_path = directory_path / name
            if file_path.suffix == ".pyc":
                continue
            relative = file_path.relative_to(root).as_posix()
            if file_path.is_symlink():
                payload = os.readlink(file_path).encode("utf-8")
                kind = "SYMLINK"
            else:
                payload = file_path.read_bytes()
                kind = "FILE"
            rows.append(
                {
                    "kind": kind,
                    "path": relative,
                    "sha256": bytes_sha256(payload),
                    "size_bytes": len(payload),
                }
            )
    return rows


def validate_training_runtime_release(
    *,
    data_path: Path,
    python_executable: Path,
    recbole_root: Path,
) -> tuple[str, ...]:
    release = training_runtime_release()
    root = _project_root()
    failures: list[str] = []

    if (
        release.profile_id != RuntimeProfileIdV1.PACKAGE_TRAINING_V2.value
        or release.runner_abi != TRAINING_RUNNER_ABI
        or release.launcher_abi != TRAINING_LAUNCHER_ABI
        or release.launch_protocol_id != TRAINING_LAUNCH_PROTOCOL_ID
    ):
        failures.append("TRAINING_RELEASE_PROFILE_MISMATCH")

    for row in release.source_manifest:
        path = root / str(row["path"])
        if not path.is_file() or bytes_sha256(path.read_bytes()) != row["sha256"]:
            failures.append(f"TRAINING_SOURCE_MISMATCH:{row['path']}")

    for contract_name in ("environment_lock", "training_config"):
        contract = getattr(release, contract_name)
        for row in contract["files"]:
            path = root / str(row["path"])
            if not path.is_file() or bytes_sha256(path.read_bytes()) != row["sha256"]:
                failures.append(f"TRAINING_INPUT_MISMATCH:{row['path']}")

    dataset_root = data_path.resolve() / str(release.read_contract["dataset_dir"])
    for name, expected in release.read_contract["dataset_files"].items():
        path = dataset_root / str(name)
        if not path.is_file() or bytes_sha256(path.read_bytes()) != expected:
            failures.append(f"TRAINING_DATASET_MISMATCH:{name}")

    python_path = python_executable.absolute()
    if (
        not python_path.is_file()
        or bytes_sha256(python_path.read_bytes())
        != release.backend_identity["python_executable_sha256"]
    ):
        failures.append("TRAINING_PYTHON_IDENTITY_MISMATCH")
    else:
        try:
            environment_lock = subprocess.check_output(
                [
                    str(python_path),
                    "-m",
                    "pip",
                    "freeze",
                    "--all",
                ],
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            failures.append("TRAINING_ENVIRONMENT_LOCK_UNAVAILABLE")
        else:
            if (
                bytes_sha256(environment_lock)
                != release.backend_identity["pip_freeze_digest"]
            ):
                failures.append("TRAINING_ENVIRONMENT_LOCK_MISMATCH")
        try:
            package_versions = json.loads(
                subprocess.check_output(
                    [
                        str(python_path),
                        "-c",
                        (
                            "import importlib.metadata as m,json,sys;"
                            "names=json.loads(sys.argv[1]);"
                            "print(json.dumps({name:m.version(name) for name in names},"
                            "sort_keys=True,separators=(',',':')))"
                        ),
                        json.dumps(
                            sorted(
                                release.backend_identity[
                                    "python_package_versions"
                                ]
                            )
                        ),
                    ],
                    text=True,
                    timeout=20,
                )
            )
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            failures.append("TRAINING_PYTHON_PACKAGE_LOCK_UNAVAILABLE")
        else:
            if package_versions != dict(
                release.backend_identity["python_package_versions"]
            ):
                failures.append("TRAINING_PYTHON_PACKAGE_LOCK_MISMATCH")
        try:
            torch_cuda = _live_torch_cuda_environment(python_path)
        except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
            failures.append("TRAINING_TORCH_CUDA_IDENTITY_UNAVAILABLE")
        else:
            observed_digest = sha256_digest(torch_cuda)
            if (
                observed_digest
                != release.backend_identity["torch_cuda_environment_digest"]
                or sha256_digest(
                    release.backend_identity["torch_cuda_environment"]
                )
                != observed_digest
            ):
                failures.append("TRAINING_TORCH_CUDA_IDENTITY_MISMATCH")

    try:
        commit, tree, status = _git_identity(recbole_root.resolve())
    except (OSError, subprocess.SubprocessError):
        failures.append("TRAINING_RECBOLE_IDENTITY_UNAVAILABLE")
    else:
        if (
            commit != release.backend_identity["recbole_commit"]
            or tree != release.backend_identity["recbole_tree"]
            or status
        ):
            failures.append("TRAINING_RECBOLE_IDENTITY_MISMATCH")
    return tuple(sorted(failures))


def validate_campaign_training_runtime_release(
    *,
    data_path: Path,
    python_executable: Path,
    recbole_root: Path,
) -> tuple[str, ...]:
    """Validate the Main-equivalent Campaign release without mutating V2."""

    release = campaign_training_runtime_release()
    root = _project_root()
    failures: list[str] = []
    if (
        release.profile_id
        != RuntimeProfileIdV1.PACKAGE_TRAINING_V3.value
        or release.runner_abi != CAMPAIGN_TRAINING_RUNNER_ABI
        or release.launcher_abi != CAMPAIGN_TRAINING_LAUNCHER_ABI
        or release.launch_protocol_id
        != CAMPAIGN_TRAINING_LAUNCH_PROTOCOL_ID
        or release.online_metric_source != "BEST_CHECKPOINT_TEST_RESULT"
        or release.campaign_runtime_profile_digest
        != campaign_runtime_profile()["profile_digest"]
        or release.partition_profile_digest
        != sha256_digest(campaign_partition_profile())
    ):
        failures.append("CAMPAIGN_TRAINING_RELEASE_PROFILE_MISMATCH")
    for row in release.source_manifest:
        path = root / str(row["path"])
        if (
            not path.is_file()
            or bytes_sha256(path.read_bytes()) != row["sha256"]
        ):
            failures.append(
                f"CAMPAIGN_TRAINING_SOURCE_MISMATCH:{row['path']}"
            )
    for contract_name in ("environment_lock", "training_config"):
        for row in getattr(release, contract_name)["files"]:
            path = root / str(row["path"])
            if (
                not path.is_file()
                or bytes_sha256(path.read_bytes()) != row["sha256"]
            ):
                failures.append(
                    f"CAMPAIGN_TRAINING_INPUT_MISMATCH:{row['path']}"
                )
    dataset_root = (
        data_path.resolve() / str(release.read_contract["dataset_dir"])
    )
    for name, expected in release.read_contract["dataset_files"].items():
        path = dataset_root / str(name)
        if (
            not path.is_file()
            or bytes_sha256(path.read_bytes()) != expected
        ):
            failures.append(f"CAMPAIGN_TRAINING_DATASET_MISMATCH:{name}")
    python_path = python_executable.absolute()
    if (
        not python_path.is_file()
        or bytes_sha256(python_path.read_bytes())
        != release.backend_identity["python_executable_sha256"]
    ):
        failures.append("CAMPAIGN_TRAINING_PYTHON_IDENTITY_MISMATCH")
    else:
        try:
            frozen = subprocess.check_output(
                [str(python_path), "-m", "pip", "freeze", "--all"],
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            failures.append(
                "CAMPAIGN_TRAINING_ENVIRONMENT_LOCK_UNAVAILABLE"
            )
        else:
            if (
                bytes_sha256(frozen)
                != release.backend_identity["pip_freeze_digest"]
            ):
                failures.append(
                    "CAMPAIGN_TRAINING_ENVIRONMENT_LOCK_MISMATCH"
                )
        try:
            package_versions = json.loads(
                subprocess.check_output(
                    [
                        str(python_path),
                        "-c",
                        (
                            "import importlib.metadata as m,json,sys;"
                            "names=json.loads(sys.argv[1]);"
                            "print(json.dumps({name:m.version(name) for name in names},"
                            "sort_keys=True,separators=(',',':')))"
                        ),
                        json.dumps(
                            sorted(
                                release.backend_identity[
                                    "python_package_versions"
                                ]
                            )
                        ),
                    ],
                    text=True,
                    timeout=20,
                )
            )
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            failures.append(
                "CAMPAIGN_TRAINING_PYTHON_PACKAGE_LOCK_UNAVAILABLE"
            )
        else:
            if package_versions != dict(
                release.backend_identity["python_package_versions"]
            ):
                failures.append(
                    "CAMPAIGN_TRAINING_PYTHON_PACKAGE_LOCK_MISMATCH"
                )
        try:
            torch_cuda = _live_torch_cuda_environment(python_path)
        except (
            OSError,
            ValueError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
        ):
            failures.append(
                "CAMPAIGN_TRAINING_TORCH_CUDA_IDENTITY_UNAVAILABLE"
            )
        else:
            observed_digest = sha256_digest(torch_cuda)
            if (
                observed_digest
                != release.backend_identity[
                    "torch_cuda_environment_digest"
                ]
                or sha256_digest(
                    release.backend_identity["torch_cuda_environment"]
                )
                != observed_digest
            ):
                failures.append(
                    "CAMPAIGN_TRAINING_TORCH_CUDA_IDENTITY_MISMATCH"
                )
    frozen_recbole_manifest = release.backend_identity.get(
        "recbole_source_manifest"
    )
    if frozen_recbole_manifest is not None:
        observed_recbole_manifest = _directory_content_manifest(
            recbole_root
        )
        if (
            observed_recbole_manifest != list(frozen_recbole_manifest)
            or sha256_digest(observed_recbole_manifest)
            != release.backend_identity[
                "recbole_source_manifest_digest"
            ]
        ):
            failures.append("CAMPAIGN_RECBOLE_IDENTITY_MISMATCH")
    else:
        try:
            commit, tree, status = _git_identity(recbole_root.resolve())
        except (OSError, subprocess.SubprocessError):
            failures.append("CAMPAIGN_RECBOLE_IDENTITY_UNAVAILABLE")
        else:
            if (
                commit != release.backend_identity["recbole_commit"]
                or tree != release.backend_identity["recbole_tree"]
                or status
            ):
                failures.append("CAMPAIGN_RECBOLE_IDENTITY_MISMATCH")
    failures.extend(
        f"CAMPAIGN_RUNTIME:{item}"
        for item in campaign_readiness_failures(
            import_entrypoints=True
        )
    )
    return tuple(sorted(set(failures)))


def _resolved_root(value: str) -> str:
    return Path(value).resolve().as_posix()


def build_training_runtime_binding(
    *,
    accepted_evidence_eligibility: str,
    arm_common_projection_digest: str,
    budget_digest: str,
    candidate_id: str,
    checkpoint_root: str,
    evaluation_purpose: str,
    execution_purpose: str,
    experiment_id: str,
    frontier_eligibility: str,
    gpu_cost_ceiling_microunits: int,
    gpu_device_time_ceiling_ms: int,
    implementation_digest: str,
    instance_private_root: str,
    lineage_digest: str,
    opaque_arm_instance_id: str,
    partition_purpose: str,
    protocol_digest: str,
    protocol_profile_ref: str,
    result_root: str,
    round_id: str,
    run_id: str,
    search_memory_eligibility: str,
    seed_policy_digest: str,
    training_config_budget_digest: str,
    filesystem_capability_digest: str,
    runner_abi: str = TRAINING_RUNNER_ABI,
) -> TrainingRuntimeBindingV2:
    release = training_release_for_abi(runner_abi)
    if execution_purpose not in set(release.supported_execution_purposes):
        raise ValueError(
            f"training release does not support purpose: {execution_purpose}"
        )
    if evaluation_purpose != execution_purpose:
        raise ValueError("evaluation and execution purpose must be identical")
    if execution_purpose in {
        TrainingExecutionPurposeV1.FIXED_CANARY.value,
        TrainingExecutionPurposeV1.PILOT.value,
    } and (
        frontier_eligibility != "NOT_ELIGIBLE_FOR_MAIN_FRONTIER"
        or search_memory_eligibility != "NOT_ELIGIBLE_FOR_MAIN_SEARCH_MEMORY"
        or accepted_evidence_eligibility != "NOT_ELIGIBLE_FOR_ACCEPTED_EVIDENCE"
    ):
        raise ValueError("development training cannot write Main/evidence state")
    private_root = Path(instance_private_root).resolve()
    resolved_result = Path(result_root).resolve()
    resolved_checkpoint = Path(checkpoint_root).resolve()
    if (
        not resolved_result.is_relative_to(private_root)
        or not resolved_checkpoint.is_relative_to(resolved_result)
    ):
        raise ValueError("training result/checkpoint roots escape instance-private root")
    return TrainingRuntimeBindingV2(
        {
            "accepted_evidence_eligibility": accepted_evidence_eligibility,
            "arm_common_projection_digest": arm_common_projection_digest,
            "backend_identity_digest": sha256_digest(release.backend_identity),
            "budget_digest": budget_digest,
            "candidate_id": candidate_id,
            "checkpoint_root": resolved_checkpoint.as_posix(),
            "confinement_contract_digest": sha256_digest(
                release.confinement_contract
            ),
            "environment_lock_digest": sha256_digest(release.environment_lock),
            "evaluation_purpose": evaluation_purpose,
            "execution_purpose": execution_purpose,
            "experiment_id": experiment_id,
            "frontier_eligibility": frontier_eligibility,
            "filesystem_capability_digest": filesystem_capability_digest,
            "gpu_cost_ceiling_microunits": gpu_cost_ceiling_microunits,
            "gpu_device_time_ceiling_ms": gpu_device_time_ceiling_ms,
            "implementation_digest": implementation_digest,
            "instance_private_root_digest": sha256_digest(
                {"resolved_instance_private_root": private_root.as_posix()}
            ),
            "lineage_digest": lineage_digest,
            "metric_contract_digest": sha256_digest(release.metric_contract),
            "opaque_arm_instance_id": opaque_arm_instance_id,
            "partition_purpose": partition_purpose,
            "profile_id": release.profile_id,
            "protocol_digest": protocol_digest,
            "protocol_profile_ref": protocol_profile_ref,
            "read_contract_digest": sha256_digest(release.read_contract),
            "release_digest": release.digest,
            "release_id": release.release_id,
            "resource_contract_digest": sha256_digest(release.resource_contract),
            "result_root": resolved_result.as_posix(),
            "round_id": round_id,
            "run_id": run_id,
            "runner_abi": release.runner_abi,
            "search_memory_eligibility": search_memory_eligibility,
            "seed_policy_digest": seed_policy_digest,
            "source_manifest_digest": sha256_digest(release.source_manifest),
            "training_config_budget_digest": training_config_budget_digest,
            "training_config_digest": sha256_digest(release.training_config),
            "write_contract_digest": sha256_digest(release.write_contract),
        }
    )


def training_runtime_component_abis(
    overrides: Mapping[str, str] | None = None,
    *,
    runner_abi: str = TRAINING_RUNNER_ABI,
) -> dict[str, str]:
    values = {name: runner_abi for name in _COMPONENT_NAMES}
    values.update(dict(overrides or {}))
    return values


def training_runtime_compatibility_preflight(
    *,
    data_path: Path,
    fixture: TrainingRuntimeCompatibilityFixtureV1,
    python_executable: Path,
    recbole_root: Path,
    runner_abi: str = TRAINING_RUNNER_ABI,
) -> TrainingRuntimeCompatibilityPreflightV1:
    """Traverse the concrete campaign-to-close release chain before broker use."""

    release = training_release_for_abi(runner_abi)
    failures = list(
        (
            validate_campaign_training_runtime_release(
                data_path=data_path,
                python_executable=python_executable,
                recbole_root=recbole_root,
            )
            if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI
            else validate_training_runtime_release(
                data_path=data_path,
                python_executable=python_executable,
                recbole_root=recbole_root,
            )
        )
    )
    checks: list[dict[str, Any]] = []

    def check(name: str, verified: bool, failure: str) -> None:
        checks.append(
            {
                "name": name,
                "status": "VERIFIED" if verified else "REJECTED",
            }
        )
        if not verified:
            failures.append(failure)

    try:
        capability = build_training_filesystem_capability(
            instance_private_root=Path(fixture.instance_private_root),
            result_root=Path(fixture.result_root),
            checkpoint_root=Path(fixture.checkpoint_root),
            project_root=_project_root(),
            recbole_root=recbole_root,
            dataset_root=(
                data_path.resolve() / str(release.read_contract["dataset_dir"])
            ),
        )
        binding = build_training_runtime_binding(
            accepted_evidence_eligibility=fixture.accepted_evidence_eligibility,
            arm_common_projection_digest=fixture.arm_common_projection_digest,
            budget_digest=fixture.budget_digest,
            candidate_id=fixture.candidate_id,
            checkpoint_root=fixture.checkpoint_root,
            evaluation_purpose=fixture.evaluation_purpose,
            execution_purpose=fixture.execution_purpose,
            experiment_id=fixture.experiment_id,
            frontier_eligibility=fixture.frontier_eligibility,
            gpu_cost_ceiling_microunits=fixture.gpu_cost_ceiling_microunits,
            gpu_device_time_ceiling_ms=fixture.gpu_device_time_ceiling_ms,
            implementation_digest=fixture.implementation_digest,
            instance_private_root=fixture.instance_private_root,
            lineage_digest=fixture.lineage_digest,
            opaque_arm_instance_id=fixture.opaque_arm_instance_id,
            partition_purpose=fixture.partition_purpose,
            protocol_digest=fixture.protocol_digest,
            protocol_profile_ref=fixture.protocol_profile_ref,
            result_root=fixture.result_root,
            round_id=fixture.round_id,
            run_id=fixture.run_id,
            search_memory_eligibility=fixture.search_memory_eligibility,
            seed_policy_digest=fixture.seed_policy_digest,
            training_config_budget_digest=fixture.training_config_budget_digest,
            filesystem_capability_digest=capability.capability_digest,
            runner_abi=runner_abi,
        )
    except ValueError:
        capability = None
        binding = None
        failures.append("TRAINING_PURPOSE_BINDING_MISMATCH")

    component_abis = dict(fixture.component_runner_abis)
    check(
        "CAMPAIGN_CONTRACT_TO_PURPOSE_BINDING",
        binding is not None
        and binding.experiment_id == fixture.experiment_id
        and binding.evaluation_purpose == fixture.evaluation_purpose
        and binding.protocol_digest == fixture.protocol_digest
        and binding.budget_digest == fixture.budget_digest,
        "TRAINING_CAMPAIGN_BINDING_MISMATCH",
    )
    check(
        "RUNTIME_RELEASE_TO_PURPOSE_BINDING",
        binding is not None
        and binding.release_digest == release.digest
        and binding.runner_abi == release.runner_abi,
        "TRAINING_RELEASE_BINDING_MISMATCH",
    )
    check(
        "PACKAGE_HANDLER_REGISTRY",
        release.package_owned_handler_registry_digest
        == sha256_digest(_handler_registry_projection(runner_abi)),
        "TRAINING_HANDLER_REGISTRY_MISMATCH",
    )
    sources = _source_manifest_map(release)
    worker_source_path = (
        "scripts/campaign_train_worker.py"
        if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI
        else "scripts/pilot_train_worker.py"
    )
    check(
        "RELEASE_DECLARED_IDENTITIES",
        release.runner_entrypoint_digest
        == sources.get(worker_source_path)
        and release.runner_source_digest
        == sources.get(worker_source_path)
        and release.launcher_source_digest
        == sources.get(
            "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py"
        )
        and release.python_environment_lock_digest
        == sha256_digest(release.environment_lock)
        and release.recbole_identity_digest
        == sha256_digest(
            {
                "recbole_commit": release.backend_identity["recbole_commit"],
                "recbole_tree": release.backend_identity["recbole_tree"],
                "recbole_version": release.backend_identity["recbole_version"],
            }
        )
        and release.config_schema_digest == sha256_digest(release.training_config)
        and release.allowed_read_roots_policy_digest
        == sha256_digest(release.read_contract)
        and release.allowed_write_roots_policy_digest
        == sha256_digest(release.write_contract)
        and release.confinement_policy_digest
        == sha256_digest(release.confinement_contract)
        and release.close_result_policy_digest
        == sha256_digest(
            {
                "close_contract": release.close_contract,
                "guard_source": sources.get(
                    "src/recclaw_core/experiments/helix_abc_v1/"
                    "training_execution_guard.py"
                ),
            }
        )
        and release.filesystem_capability_policy_digest
        == filesystem_capability_policy_digest()
        and release.store_audit_contract_digest
        == store_audit_contract_digest()
        and release.training_profile_digest
        == sha256_digest(
            _resource_json(
                "campaign_training_profile_v1.json"
                if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI
                else "pilot_training_profile_v1.json"
            )
        ),
        "TRAINING_DECLARED_RELEASE_IDENTITY_MISMATCH",
    )
    check(
        "CANDIDATE_BINDING_SCHEMA",
        {
            "execution_purpose",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
        }.issubset(CandidateExecutionBindingV3.required_fields)
        and release.candidate_binding_schema_digest
        == _schema_digest(CandidateExecutionBindingV3),
        "TRAINING_CANDIDATE_BINDING_SCHEMA_MISMATCH",
    )
    check(
        "COMMON_GUARD_POLICY",
        release.common_guard_policy_digest
        == sha256_digest(
            {
                "guard": release.close_contract["common_guard"],
                "source": _source_manifest_map(release).get(
                    "src/recclaw_core/experiments/helix_abc_v1/"
                    "training_execution_guard.py"
                ),
            }
        ),
        "TRAINING_COMMON_GUARD_POLICY_MISMATCH",
    )
    check(
        "PERMIT_TEMPLATE_SCHEMA",
        release.permit_schema_digest == _schema_digest(CommonExecutionPermitV2),
        "TRAINING_PERMIT_SCHEMA_MISMATCH",
    )

    from .training_state_store import (
        TRAINING_CLAIM_IDENTITY_FIELDS,
        TRAINING_SCHEMA_VERSION,
    )

    migration = (
        Path(__file__).with_name("migrations") / "002_training_runtime_release.sql"
    ).read_text(encoding="utf-8")
    check(
        "STATE_STORE_CLAIM_SCHEMA",
        TRAINING_SCHEMA_VERSION == 2
        and all(field in migration for field in TRAINING_CLAIM_IDENTITY_FIELDS)
        and release.state_store_claim_schema_digest
        == sha256_digest(
            {
                "identity_fields": list(TRAINING_CLAIM_IDENTITY_FIELDS),
                "migration_sha256": bytes_sha256(migration.encode("utf-8")),
                "schema_version": TRAINING_SCHEMA_VERSION,
            }
        ),
        "TRAINING_STATE_STORE_SCHEMA_MISMATCH",
    )
    check(
        "LAUNCHER_PROTOCOL",
        release.launcher_abi
        == (
            CAMPAIGN_TRAINING_LAUNCHER_ABI
            if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI
            else TRAINING_LAUNCHER_ABI
        )
        and release.launch_protocol_id
        == (
            CAMPAIGN_TRAINING_LAUNCH_PROTOCOL_ID
            if runner_abi == CAMPAIGN_TRAINING_RUNNER_ABI
            else TRAINING_LAUNCH_PROTOCOL_ID
        )
        and bool(
            _source_manifest_map(release).get(
                "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py"
            )
        )
        and bool(_source_manifest_map(release).get(worker_source_path)),
        "TRAINING_LAUNCHER_PROTOCOL_MISMATCH",
    )
    expected_schemas = {
        "receipt_schema_digest": _schema_digest(ExecutionStartReceiptV2),
        "raw_output_schema_digest": _schema_digest(TrainingRawRunOutputV2),
        "raw_result_schema_digest": _schema_digest(RawResultEnvelopeV2),
        "close_result_schema_digest": _schema_digest(CommonResultClosureV2),
        "resource_accounting_schema_digest": _schema_digest(
            TrainingResourceAccountingV1
        ),
        "start_confirmation_schema_digest": _schema_digest(
            ExecutionStartConfirmationV1
        ),
        "runtime_binding_schema_digest": _schema_digest(
            TrainingRuntimeBindingV2
        ),
        "training_plan_schema_digest": _schema_digest(
            TrainingRuntimePlanDecisionV1
        ),
    }
    check(
        "RECEIPT_RAW_CLOSE_SCHEMAS",
        all(getattr(release, key) == value for key, value in expected_schemas.items()),
        "TRAINING_RESULT_SCHEMA_MISMATCH",
    )
    check(
        "METRIC_RESOURCE_POLICIES",
        binding is not None
        and binding.metric_contract_digest == sha256_digest(release.metric_contract)
        and binding.resource_contract_digest
        == sha256_digest(release.resource_contract)
        and release.metric_parser_digest
        == sha256_digest(
            {
                "metric_contract": release.metric_contract,
                "worker_source": _source_manifest_map(release).get(
                    worker_source_path
                ),
            }
        )
        and release.resource_meter_policy_digest
        == sha256_digest(release.resource_contract),
        "TRAINING_METRIC_RESOURCE_POLICY_MISMATCH",
    )
    for component in _COMPONENT_NAMES:
        check(
            f"ABI:{component}",
            component_abis.get(component) == runner_abi,
            f"TRAINING_COMPONENT_ABI_MISMATCH:{component}",
        )

    closure_projection = {
        "binding_digest": binding.digest if binding is not None else None,
        "checks": checks,
        "component_runner_abis": component_abis,
        "fixture_digest": fixture.digest,
        "release_digest": release.digest,
    }
    return TrainingRuntimeCompatibilityPreflightV1(
        {
            "checked_release_digest": release.digest,
            "closure_projection_digest": sha256_digest(closure_projection),
            "component_checks": checks,
            "expected_runner_abi": runner_abi,
            "failure_codes": sorted(set(failures)),
            "fixture_digest": fixture.digest,
            "profile_id": release.profile_id,
            "runtime_binding_digest": binding.digest if binding is not None else None,
            "status": (
                TrainingCompatibilityStatusV1.COMPATIBLE.value
                if not failures
                else TrainingCompatibilityStatusV1.INCOMPATIBLE.value
            ),
        }
    )


__all__ = [
    "CAMPAIGN_TRAINING_LAUNCHER_ABI",
    "CAMPAIGN_TRAINING_LAUNCH_PROTOCOL_ID",
    "CAMPAIGN_TRAINING_RUNNER_ABI",
    "TRAINING_LAUNCHER_ABI",
    "TRAINING_LAUNCH_PROTOCOL_ID",
    "TRAINING_RUNNER_ABI",
    "build_training_runtime_binding",
    "campaign_training_runtime_release",
    "historical_training_runtime_release_v1",
    "resolve_bound_training_release",
    "resolve_runtime_release",
    "training_runtime_compatibility_preflight",
    "training_runtime_component_abis",
    "training_runtime_release",
    "training_runtime_release_digest",
    "validate_campaign_training_runtime_release",
    "validate_training_runtime_release",
]
