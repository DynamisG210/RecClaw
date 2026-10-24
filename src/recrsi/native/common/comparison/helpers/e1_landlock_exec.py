#!/usr/bin/env python3
"""Landlock filesystem policy for unprivileged x86-64 workers.

The caller supplies isolated network/IPC namespaces and read/write roots.
PID namespace isolation is recorded separately because CUDA requires a
consistent procfs view.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import resource
import re
import sys
from typing import Sequence


SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38

ACCESS_EXECUTE = 1 << 0
ACCESS_WRITE_FILE = 1 << 1
ACCESS_READ_FILE = 1 << 2
ACCESS_READ_DIR = 1 << 3
ACCESS_REMOVE_DIR = 1 << 4
ACCESS_REMOVE_FILE = 1 << 5
ACCESS_MAKE_CHAR = 1 << 6
ACCESS_MAKE_DIR = 1 << 7
ACCESS_MAKE_REG = 1 << 8
ACCESS_MAKE_SOCK = 1 << 9
ACCESS_MAKE_FIFO = 1 << 10
ACCESS_MAKE_BLOCK = 1 << 11
ACCESS_MAKE_SYM = 1 << 12
ACCESS_REFER = 1 << 13
ACCESS_TRUNCATE = 1 << 14

ABI1_HANDLED = (1 << 13) - 1
READ_FILE_ACCESS = ACCESS_EXECUTE | ACCESS_READ_FILE
READ_DIR_ACCESS = READ_FILE_ACCESS | ACCESS_READ_DIR
WRITE_FILE_ACCESS = READ_FILE_ACCESS | ACCESS_WRITE_FILE
WRITE_DIR_ACCESS_ABI1 = ABI1_HANDLED

SAFE_ENV_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "E1_HOST_IPC_NS",
    "E1_HOST_NET_NS",
    "E1_HOST_PID_NS",
    "E1_SANDBOX_ROLE",
    "HOME",
    "LANG",
    "LC_ALL",
    "MKL_NUM_THREADS",
    "MPLCONFIGDIR",
    "LD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PATH",
    "PWD",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "SHLVL",
    "TMPDIR",
    "TORCH_HOME",
    "TZ",
    "XDG_CACHE_HOME",
    "_",
}
FORBIDDEN_PATH_MARKERS = (
    "heldout",
    "p4_sparse",
    "recclaw_v2r4",
    "recclaw_three_arm",
    "three_arm",
)
FORBIDDEN_BROAD_ROOTS = {
    Path("/"),
    Path("/ssd"),
    Path("/NAS2020"),
}


class IsolationError(RuntimeError):
    """Raised when the requested boundary cannot be proven fail-closed."""


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class AllowedPath:
    path: Path
    writable: bool
    access: int | None = None


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = int(libc.syscall(number, *args))
    if result < 0:
        error_number = ctypes.get_errno()
        raise IsolationError(
            f"syscall {number} failed: errno={error_number} "
            f"({os.strerror(error_number)})"
        )
    return result


def landlock_abi(libc: ctypes.CDLL) -> int:
    return _syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
    )


def _handled_rights(abi: int) -> int:
    rights = ABI1_HANDLED
    if abi >= 2:
        rights |= ACCESS_REFER
    if abi >= 3:
        rights |= ACCESS_TRUNCATE
    return rights


def _allowed_rights(path: Path, *, writable: bool, abi: int) -> int:
    is_directory = path.is_dir()
    if not writable:
        return READ_DIR_ACCESS if is_directory else READ_FILE_ACCESS
    if not is_directory:
        rights = WRITE_FILE_ACCESS
        if abi >= 3:
            rights |= ACCESS_TRUNCATE
        return rights
    rights = WRITE_DIR_ACCESS_ABI1
    if abi >= 2:
        rights |= ACCESS_REFER
    if abi >= 3:
        rights |= ACCESS_TRUNCATE
    return rights


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    allowed: AllowedPath,
    *,
    abi: int,
) -> None:
    path_fd = os.open(allowed.path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = PathBeneathAttr(
            allowed_access=allowed.access & _handled_rights(abi) if allowed.access is not None else _allowed_rights(
                allowed.path,
                writable=allowed.writable,
                abi=abi,
            ),
            parent_fd=path_fd,
        )
        _syscall(
            libc,
            SYS_LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint32(0),
        )
    finally:
        os.close(path_fd)


def apply_landlock(paths: Sequence[AllowedPath]) -> int:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise IsolationError("unsupported syscall architecture; refusing to guess")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = landlock_abi(libc)
    if abi < 1:
        raise IsolationError(f"Landlock ABI {abi} cannot enforce filesystem rules")
    ruleset_attr = RulesetAttr(handled_access_fs=_handled_rights(abi))
    ruleset_fd = _syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.c_size_t(ctypes.sizeof(ruleset_attr)),
        ctypes.c_uint32(0),
    )
    try:
        for allowed in paths:
            _add_path_rule(libc, ruleset_fd, allowed, abi=abi)
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            raise IsolationError(
                f"PR_SET_NO_NEW_PRIVS failed: errno={error_number} "
                f"({os.strerror(error_number)})"
            )
        _syscall(
            libc,
            SYS_LANDLOCK_RESTRICT_SELF,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint32(0),
        )
    finally:
        os.close(ruleset_fd)
    return abi


def _resolve_allowed(raw_paths: Sequence[str], *, writable: bool) -> list[AllowedPath]:
    resolved: list[AllowedPath] = []
    for raw in raw_paths:
        path = Path(raw)
        # /proc/self is the one intentional dynamic symlink: it resolves to this
        # launcher process and remains the same process across exec.  All other
        # symlink roots are refused to keep the allowlist identity unambiguous.
        if path.is_symlink() and raw != "/proc/self":
            raise IsolationError(f"allowlist path must not be a symlink: {path}")
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise IsolationError(f"allowlist path does not exist: {raw}") from exc
        if path in FORBIDDEN_BROAD_ROOTS:
            raise IsolationError(f"allowlist path is an unsafe broad root: {path}")
        lowered = str(path).casefold()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            raise IsolationError(f"forbidden E1 path marker in allowlist: {path}")
        resolved.append(AllowedPath(path=path, writable=writable))
    return resolved


def _status_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields


def gpu_device_for_uuid(gpu_uuid: str, proc_root: Path = Path('/proc/driver/nvidia/gpus')) -> Path:
    """Resolve the driver node, never confuse NVML/CUDA ordinal with its minor."""
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', gpu_uuid):
        raise IsolationError('CUDA_VISIBLE_DEVICES must be one full GPU UUID')
    matches = []
    for info in proc_root.glob('*/information'):
        fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
        if fields.get('GPU UUID', '').strip() == gpu_uuid:
            minor = fields.get('Device Minor', '').strip()
            if not minor.isdigit():
                raise IsolationError('driver GPU information has no numeric device minor')
            matches.append(Path('/dev') / f'nvidia{minor}')
    if len(matches) != 1:
        raise IsolationError('GPU UUID does not resolve to exactly one driver device')
    return matches[0]


def _verify_preconditions(required_uid: int) -> dict[str, object]:
    if os.getuid() != required_uid or os.geteuid() != required_uid:
        raise IsolationError(
            f"UID drop missing: expected {required_uid}, got uid/euid "
            f"{os.getuid()}/{os.geteuid()}"
        )
    status = _status_fields()
    for field in ("CapInh", "CapPrm", "CapEff", "CapAmb"):
        if int(status.get(field, "-1"), 16) != 0:
            raise IsolationError(f"capability drop missing: {field}={status.get(field)}")
    namespace_inodes: dict[str, int] = {}
    for name in ("net", "ipc"):
        own_inode = os.stat(f"/proc/self/ns/{name}").st_ino
        host_variable = f"E1_HOST_{name.upper()}_NS"
        try:
            host_inode = int(os.environ[host_variable])
        except (KeyError, ValueError) as exc:
            raise IsolationError(f"missing valid trusted {host_variable}") from exc
        if own_inode == host_inode:
            raise IsolationError(f"external {name} namespace was not created")
        namespace_inodes[name] = own_inode
    own_pid_inode = os.stat("/proc/self/ns/pid").st_ino
    try:
        host_pid_inode = int(os.environ["E1_HOST_PID_NS"])
    except (KeyError, ValueError) as exc:
        raise IsolationError("missing valid trusted E1_HOST_PID_NS") from exc
    namespace_inodes["pid"] = own_pid_inode
    # An inherited sysfs superblock can still expose host interface names in a
    # new network namespace.  procfs' per-process network view is namespace
    # aware without requiring an LXC-forbidden sysfs remount.
    interfaces = sorted(
        line.split(":", 1)[0].strip()
        for line in Path("/proc/self/net/dev").read_text(encoding="utf-8").splitlines()[2:]
        if ":" in line
    )
    if interfaces != ["lo"]:
        raise IsolationError(
            f"network namespace is not isolated to loopback: {interfaces}"
        )
    unknown_env = sorted(set(os.environ).difference(SAFE_ENV_NAMES))
    if unknown_env:
        raise IsolationError(
            "environment is not allowlisted; unexpected names: " + ", ".join(unknown_env)
        )
    gpu_uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpu_device = gpu_device_for_uuid(gpu_uuid)
    return {
        'gpu_uuid': gpu_uuid,
        'gpu_device': str(gpu_device),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "groups": os.getgroups(),
        "namespace_inodes": namespace_inodes,
        "pid_namespace_isolated": own_pid_inode != host_pid_inode,
        "interfaces": interfaces,
        "environment_names": sorted(os.environ),
    }


def _within(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ro", action="append", default=[])
    parser.add_argument("--rw", action="append", default=[])
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--require-uid", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise IsolationError("a command after -- is required")

    readonly = _resolve_allowed(args.ro, writable=False)
    writable = _resolve_allowed(args.rw, writable=True)
    if not readonly or not writable:
        raise IsolationError("at least one --ro and one --rw path are required")
    writable_roots = [entry.path for entry in writable if entry.path.is_dir()]
    for read_entry in readonly:
        if _within(read_entry.path, writable_roots):
            raise IsolationError(
                "read-only allowlist path must not be enclosed by a writable root: "
                f"{read_entry.path}"
            )
    workdir = Path(args.workdir).resolve(strict=True)
    if not _within(workdir, writable_roots):
        raise IsolationError("workdir must resolve below an allowed --rw directory")
    receipt = Path(args.receipt).resolve(strict=False) if args.receipt else None
    if receipt is not None and not _within(receipt, writable_roots):
        raise IsolationError("receipt must resolve below an allowed --rw directory")

    preconditions = _verify_preconditions(args.require_uid)
    writable_gpu_nodes = [str(entry.path) for entry in writable
                          if re.fullmatch(r'/dev/nvidia[0-9]+', str(entry.path))]
    if writable_gpu_nodes != [preconditions['gpu_device']]:
        raise IsolationError('writable GPU node differs from selected GPU UUID')
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    close_limit = 1_048_576 if soft_limit == resource.RLIM_INFINITY else int(soft_limit)
    os.closerange(3, close_limit)
    os.chdir(workdir)
    # CUDA names newly-created helper threads through task/<tid>/comm. These
    # inodes do not exist when the ruleset is installed. Permit write/truncate
    # under this process's task tree, not arbitrary procfs or another process.
    # Proc fd/root aliases still undergo the destination's Landlock checks.
    thread_metadata = AllowedPath(Path('/proc/self/task'), True, ACCESS_WRITE_FILE | ACCESS_TRUNCATE)
    abi = apply_landlock([*readonly, *writable, thread_metadata])
    if _status_fields().get("NoNewPrivs") != "1":
        raise IsolationError("no_new_privs was not retained after Landlock enforcement")

    receipt_payload = {
        "schema_version": "e1-landlock-receipt-v1",
        "landlock_abi": abi,
        "readonly": [str(entry.path) for entry in readonly],
        "writable": [str(entry.path) for entry in writable],
        "workdir": str(workdir),
        'self_task_write_right': 'WRITE_FILE_AND_SUPPORTED_TRUNCATE',
        **preconditions,
    }
    if receipt is not None:
        receipt.write_text(
            json.dumps(receipt_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "E1_LANDLOCK_READY=" + json.dumps(receipt_payload, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    os.execvpe(command[0], command, dict(os.environ))
    raise AssertionError("unreachable")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        print(f"E1_LANDLOCK_ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
        raise SystemExit(125) from exc
