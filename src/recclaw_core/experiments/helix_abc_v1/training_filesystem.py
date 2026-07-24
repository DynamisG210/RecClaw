"""Closed per-run filesystem capability for package-owned training workers."""

from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import sha256_digest


FILESYSTEM_CAPABILITY_POLICY_V2 = {
    "capability_id": "TrainingFilesystemCapabilityV2",
    "environment_keys": [
        "HOME",
        "MPLCONFIGDIR",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
    ],
    "mount_namespace": "PRIVATE",
    "process_cwd": "RUN_PRIVATE_WORKING_DIRECTORY",
    "root_filesystem": "READ_ONLY_BIND_REMOUNT",
    "writable_mounts": [
        "EXACT_RUN_ROOT",
        "PRIVATE_TMP",
        "PRIVATE_VAR_TMP",
        "PRIVATE_DEV_SHM",
        "WSL_GPU_DEVICE_NODE",
        "NON_PERSISTENT_PROC_SELF_RUNTIME_CONTROL",
    ],
}

_AT_FDCWD = -100
_AT_RECURSIVE = 0x8000
_MOUNT_ATTR_RDONLY = 0x00000001
_SYS_MOUNT_SETATTR = 442


class _MountAttr(ctypes.Structure):
    _fields_ = [
        ("attr_set", ctypes.c_uint64),
        ("attr_clr", ctypes.c_uint64),
        ("propagation", ctypes.c_uint64),
        ("userns_fd", ctypes.c_uint64),
    ]


def filesystem_capability_policy_digest() -> str:
    return sha256_digest(FILESYSTEM_CAPABILITY_POLICY_V2)


def _mount_set_read_only(path: Path, *, read_only: bool, recursive: bool) -> None:
    """Set per-mount read-only state through Linux mount_setattr(2)."""

    attributes = _MountAttr(
        _MOUNT_ATTR_RDONLY if read_only else 0,
        0 if read_only else _MOUNT_ATTR_RDONLY,
        0,
        0,
    )
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        _SYS_MOUNT_SETATTR,
        _AT_FDCWD,
        os.fsencode(path),
        _AT_RECURSIVE if recursive else 0,
        ctypes.byref(attributes),
        ctypes.sizeof(attributes),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            path.as_posix(),
        )


def make_mount_tree_read_only(path: Path) -> None:
    _mount_set_read_only(path, read_only=True, recursive=True)


def make_mount_writable(path: Path) -> None:
    _mount_set_read_only(path, read_only=False, recursive=False)


def _decode_mountinfo_path(value: str) -> str:
    return (
        value.replace(r"\040", " ")
        .replace(r"\011", "\t")
        .replace(r"\012", "\n")
        .replace(r"\134", "\\")
    )


def filesystem_mount_audit(
    allowed_writable_mounts: Iterable[Path],
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> dict[str, Any]:
    allowed = sorted(
        {
            _checked_directory(
                path, field_name="allowed_writable_mount"
            ).as_posix()
            for path in allowed_writable_mounts
        }
    )
    rows = [
        line.split()
        for line in mountinfo_path.read_text(encoding="utf-8").splitlines()
    ]
    writable = sorted(
        {
            _decode_mountinfo_path(row[4])
            for row in rows
            if "rw" in row[5].split(",")
        }
    )
    payload = {
        "allowed_writable_mount_targets": allowed,
        "missing_writable_mount_targets": sorted(set(allowed) - set(writable)),
        "mount_count": len(rows),
        "unexpected_writable_mount_targets": sorted(
            set(writable) - set(allowed)
        ),
        "writable_mount_targets": writable,
    }
    payload["status"] = (
        "PASS"
        if not payload["missing_writable_mount_targets"]
        and not payload["unexpected_writable_mount_targets"]
        else "FAIL"
    )
    return {**payload, "audit_digest": sha256_digest(payload)}


def _checked_directory(path: Path, *, field_name: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink():
        raise ValueError(f"{field_name} must not be a symlink")
    for parent in (path, *path.parents):
        if parent == resolved.anchor:
            break
        if parent.exists() and parent.is_symlink():
            raise ValueError(f"{field_name} traverses a symlink")
    return resolved


@dataclass(frozen=True, slots=True)
class TrainingFilesystemCapabilityV2:
    instance_private_root: str
    run_working_directory: str
    result_root: str
    log_root: str
    checkpoint_root: str
    temp_root: str
    cache_root: str
    home_root: str
    device_access_mounts: tuple[str, ...]
    runtime_control_mounts: tuple[str, ...]
    allowed_read_only_roots: tuple[str, ...]
    forbidden_shared_roots: tuple[str, ...]
    environment_projection: tuple[tuple[str, str], ...]
    policy_digest: str
    capability_digest: str

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "TrainingFilesystemCapabilityV2":
        normalized = {
            "allowed_read_only_roots": tuple(payload["allowed_read_only_roots"]),
            "cache_root": str(payload["cache_root"]),
            "checkpoint_root": str(payload["checkpoint_root"]),
            "device_access_mounts": tuple(payload["device_access_mounts"]),
            "environment_projection": tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in dict(
                        payload["environment_projection"]
                    ).items()
                )
            ),
            "forbidden_shared_roots": tuple(payload["forbidden_shared_roots"]),
            "home_root": str(payload["home_root"]),
            "instance_private_root": str(payload["instance_private_root"]),
            "log_root": str(payload["log_root"]),
            "policy_digest": str(payload["policy_digest"]),
            "result_root": str(payload["result_root"]),
            "runtime_control_mounts": tuple(payload["runtime_control_mounts"]),
            "run_working_directory": str(payload["run_working_directory"]),
            "temp_root": str(payload["temp_root"]),
        }
        digest_payload = {
            **normalized,
            "allowed_read_only_roots": list(
                normalized["allowed_read_only_roots"]
            ),
            "environment_projection": dict(
                normalized["environment_projection"]
            ),
            "forbidden_shared_roots": list(
                normalized["forbidden_shared_roots"]
            ),
        }
        return cls(
            **normalized,
            capability_digest=sha256_digest(digest_payload),
        )

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.environment_projection)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_read_only_roots": list(self.allowed_read_only_roots),
            "cache_root": self.cache_root,
            "capability_digest": self.capability_digest,
            "checkpoint_root": self.checkpoint_root,
            "device_access_mounts": list(self.device_access_mounts),
            "environment_projection": self.environment,
            "forbidden_shared_roots": list(self.forbidden_shared_roots),
            "home_root": self.home_root,
            "instance_private_root": self.instance_private_root,
            "log_root": self.log_root,
            "policy_digest": self.policy_digest,
            "result_root": self.result_root,
            "runtime_control_mounts": list(self.runtime_control_mounts),
            "run_working_directory": self.run_working_directory,
            "temp_root": self.temp_root,
        }


def build_training_filesystem_capability(
    *,
    instance_private_root: Path,
    result_root: Path,
    checkpoint_root: Path,
    project_root: Path,
    recbole_root: Path,
    dataset_root: Path,
) -> TrainingFilesystemCapabilityV2:
    private = _checked_directory(
        instance_private_root, field_name="instance_private_root"
    )
    result = _checked_directory(result_root, field_name="result_root")
    checkpoint = _checked_directory(
        checkpoint_root, field_name="checkpoint_root"
    )
    project = _checked_directory(project_root, field_name="project_root")
    recbole = _checked_directory(recbole_root, field_name="recbole_root")
    dataset = _checked_directory(dataset_root, field_name="dataset_root")
    if (
        not result.is_relative_to(private)
        or not checkpoint.is_relative_to(result)
    ):
        raise ValueError("training filesystem roots escape the instance capability")

    working = result / "work"
    log_root = working / "log"
    temp_root = result / "tmp"
    cache_root = result / "cache"
    home_root = result / "home"
    environment = {
        "HOME": home_root.as_posix(),
        "MPLCONFIGDIR": (cache_root / "matplotlib").as_posix(),
        "PYTHONPYCACHEPREFIX": (cache_root / "pycache").as_posix(),
        "TMPDIR": temp_root.as_posix(),
        "TORCH_HOME": (cache_root / "torch").as_posix(),
        "XDG_CACHE_HOME": cache_root.as_posix(),
    }
    return TrainingFilesystemCapabilityV2.create(
        {
            "allowed_read_only_roots": [
                project.as_posix(),
                recbole.as_posix(),
                dataset.as_posix(),
            ],
            "cache_root": cache_root.as_posix(),
            "checkpoint_root": checkpoint.as_posix(),
            "device_access_mounts": ["/dev/dxg"],
            "environment_projection": environment,
            "forbidden_shared_roots": [
                (project / "log").as_posix(),
                (project / "configs").as_posix(),
                (project / "results").as_posix(),
                project.as_posix(),
                recbole.as_posix(),
                dataset.as_posix(),
            ],
            "home_root": home_root.as_posix(),
            "instance_private_root": private.as_posix(),
            "log_root": log_root.as_posix(),
            "policy_digest": filesystem_capability_policy_digest(),
            "result_root": result.as_posix(),
            "runtime_control_mounts": ["/proc/self"],
            "run_working_directory": working.as_posix(),
            "temp_root": temp_root.as_posix(),
        }
    )


def materialize_training_filesystem_capability(
    capability: TrainingFilesystemCapabilityV2,
) -> None:
    for path in (
        capability.result_root,
        capability.run_working_directory,
        capability.log_root,
        capability.checkpoint_root,
        capability.temp_root,
        capability.cache_root,
        capability.home_root,
        Path(capability.temp_root) / "var_tmp",
        Path(capability.temp_root) / "dev_shm",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def _is_excluded(path: Path, excluded: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in excluded)


def protected_side_effect_manifest(
    roots: Mapping[str, Path],
    *,
    excluded_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Hash all protected shared files without following symlink directories."""

    excluded = tuple(path.resolve() for path in excluded_roots)
    projection: list[dict[str, Any]] = []
    for label, root_value in sorted(roots.items()):
        root = root_value.resolve()
        if _is_excluded(root, excluded):
            continue
        for directory, names, files in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            names[:] = [
                name
                for name in sorted(names)
                if name != ".git"
                and not _is_excluded(directory_path / name, excluded)
            ]
            for name in sorted(files):
                path = directory_path / name
                if _is_excluded(path, excluded):
                    continue
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    data = os.readlink(path).encode("utf-8")
                    kind = "SYMLINK"
                else:
                    data = path.read_bytes()
                    kind = "FILE"
                projection.append(
                    {
                        "kind": kind,
                        "path": f"{label}/{relative}",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size_bytes": len(data),
                    }
                )
    return {
        "file_count": len(projection),
        "manifest_digest": sha256_digest(projection),
        "projection": "SORTED_LABEL_PATH_KIND_SHA256_SIZE_JSON_V1",
        "total_bytes": sum(int(row["size_bytes"]) for row in projection),
    }


def side_effect_audit(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "after_manifest_digest": str(after["manifest_digest"]),
        "before_manifest_digest": str(before["manifest_digest"]),
        "status": (
            "PASS"
            if before["manifest_digest"] == after["manifest_digest"]
            else "FAIL"
        ),
    }
    return {**payload, "audit_digest": sha256_digest(payload)}


def filesystem_confinement_audit(
    shared_side_effect_audit: Mapping[str, Any],
    worker_mount_audit: Mapping[str, Any],
) -> dict[str, Any]:
    mount_payload = {
        key: value
        for key, value in worker_mount_audit.items()
        if key != "audit_digest"
    }
    mount_digest_valid = (
        worker_mount_audit.get("audit_digest") == sha256_digest(mount_payload)
    )
    payload = {
        "shared_root_after_manifest_digest": str(
            shared_side_effect_audit["after_manifest_digest"]
        ),
        "shared_root_before_manifest_digest": str(
            shared_side_effect_audit["before_manifest_digest"]
        ),
        "shared_root_status": str(shared_side_effect_audit["status"]),
        "worker_mount_audit_digest": str(
            worker_mount_audit.get("audit_digest", "")
        ),
        "worker_mount_digest_valid": mount_digest_valid,
        "worker_mount_status": str(worker_mount_audit.get("status", "FAIL")),
    }
    payload["status"] = (
        "PASS"
        if payload["shared_root_status"] == "PASS"
        and payload["worker_mount_status"] == "PASS"
        and mount_digest_valid
        else "FAIL"
    )
    return {
        **payload,
        "audit_digest": sha256_digest(payload),
        "worker_mount_projection": dict(worker_mount_audit),
    }


__all__ = [
    "TrainingFilesystemCapabilityV2",
    "build_training_filesystem_capability",
    "filesystem_confinement_audit",
    "filesystem_capability_policy_digest",
    "filesystem_mount_audit",
    "make_mount_tree_read_only",
    "make_mount_writable",
    "materialize_training_filesystem_capability",
    "protected_side_effect_manifest",
    "side_effect_audit",
]
