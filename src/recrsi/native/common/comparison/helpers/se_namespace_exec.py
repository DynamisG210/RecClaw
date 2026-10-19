"""Trusted SE launcher: filesystem isolation, private IPC, physical memory limit.

SE evaluates within its controller process, so provider networking is retained.
This deliberately does not claim the E1 network-denied worker contract. Only the
generic existing Landlock and read-only-mount primitives are reused.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_landlock_exec import (
    ACCESS_TRUNCATE, ACCESS_WRITE_FILE, AllowedPath, _resolve_allowed,
    _status_fields, apply_landlock, gpu_device_for_uuid,
)
from e1_namespace_exec import CLONE_NEWIPC, CLONE_NEWNS, CLONE_NEWNET, MS_BIND, mount, readonly_bind


def child(job, memory_group):
    libc = ctypes.CDLL(None, use_errno=True)
    (memory_group / "cgroup.procs").write_text(str(os.getpid()))
    network_isolated = job.get("network_isolated", False)
    namespaces = CLONE_NEWNS | CLONE_NEWIPC | (CLONE_NEWNET if network_isolated else 0)
    if libc.unshare(namespaces):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # This LXC host has slave/private mounts and denies remounting /. Verify
    # there is no outward shared propagation before changing our namespace.
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        if any(field.startswith("shared:") for field in line.split(" - ", 1)[0].split()[6:]):
            raise RuntimeError("shared mount propagation must be isolated before SE launch")
    for path in job["readonly_mounts"]:
        readonly_bind(Path(path))
    mount(job["private_shm"], "/dev/shm", MS_BIND)
    gpu_node = gpu_device_for_uuid(job["environment"]["CUDA_VISIBLE_DEVICES"])
    readonly = _resolve_allowed(job["readonly"], writable=False)
    writable = _resolve_allowed(job["writable"], writable=True)
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    if job.get("worker_pid_path"):
        Path(job["worker_pid_path"]).write_text(str(os.getpid()))
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
        raise RuntimeError("parent-death binding failed")
    os.sched_setaffinity(0, job["cpus"])
    os.environ.clear()
    os.environ.update(job["environment"])
    os.chdir(job["cwd"])
    os.closerange(3, 1_048_576)
    readonly.extend(AllowedPath(Path(path), False) for path in (
        "/proc/self", "/proc/cpuinfo", "/proc/devices", "/proc/driver/nvidia",
        "/proc/sys/vm/mmap_min_addr", "/sys/class/net", "/sys/bus/pci",
        "/sys/devices", "/sys/module/nvidia", "/dev/null", "/dev/zero", "/dev/urandom"))
    writable.extend(AllowedPath(Path(path), True) for path in (
        str(gpu_node), "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/shm"))
    writable.append(AllowedPath(Path("/proc/self/task"), True, ACCESS_WRITE_FILE | ACCESS_TRUNCATE))
    abi = apply_landlock([*readonly, *writable])
    status = _status_fields()
    if any(int(status[name], 16) for name in ("CapEff", "CapPrm", "CapInh", "CapAmb")):
        raise RuntimeError("capabilities remain after uid drop")
    receipt = {"schema": job.get("receipt_schema", "se-e2-os-boundary-v1"), "landlock_abi": abi,
               "uid": os.getuid(), "gid": os.getgid(), "no_new_privs": status["NoNewPrivs"],
               "memory_limit_bytes": job["memory_limit_bytes"], "memory_cgroup": str(memory_group),
               "cpu_affinity": sorted(os.sched_getaffinity(0)), "gpu_node": str(gpu_node),
               "readonly_mounts": job["readonly_mounts"],
               "network": "isolated" if network_isolated else "retained for native provider calls",
               "candidate_controller_same_process": not network_isolated}
    if network_isolated:
        interfaces = [line.split(":", 1)[0].strip() for line in
                      Path("/proc/self/net/dev").read_text().splitlines()[2:] if ":" in line]
        if interfaces != ["lo"]:
            raise RuntimeError("isolated worker has a non-loopback network interface")
        receipt["network_interfaces"] = interfaces
    Path(job["receipt"]).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.execve(job["command"][0], job["command"], job["environment"])


def main():
    if os.geteuid() != 0:
        raise PermissionError("use authorized sudo namespace setup")
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # NFS uses root squash: the owning user prepares writable modes, while VFS
    # readonly mounts protect input bytes. Do not chown shared NAS paths as root.
    run_root = Path(job["run_root"]).resolve(strict=True)
    if run_root.parent != Path(job["runs_parent"]).resolve(strict=True):
        raise ValueError("run root must be one direct child of the explicit runs directory")
    for path in [run_root, *run_root.rglob("*")]:
        if path.is_symlink():
            raise ValueError("symlink in prepared run tree")
    memory_parent = Path("/sys/fs/cgroup/memory").resolve(strict=True)
    memory_group = memory_parent / f"recclaw-se-e2-{os.getpid()}"
    memory_group.mkdir()
    try:
        (memory_group / "memory.limit_in_bytes").write_text(str(job["memory_limit_bytes"]))
        if int((memory_group / "memory.limit_in_bytes").read_text()) != job["memory_limit_bytes"]:
            raise RuntimeError("physical memory limit not installed")
        process = os.fork()
        if process == 0:
            try:
                child(job, memory_group)
            except BaseException as error:
                os.write(2, f"SE_BOUNDARY_ERROR={type(error).__name__}: {error}\n".encode())
                trace = error.__traceback__
                while trace:
                    os.write(2, f"  {trace.tb_frame.f_code.co_filename}:{trace.tb_lineno}\n".encode())
                    trace = trace.tb_next
                os._exit(125)
        _, status = os.waitpid(process, 0)
        code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
        Path(job["exit_record"]).write_text(json.dumps({
            "exit_code": code,
            "memory_peak_bytes": int((memory_group / "memory.max_usage_in_bytes").read_text()),
            "memory_limit_hits": int((memory_group / "memory.failcnt").read_text()),
        }, indent=2), encoding="utf-8")
        return code if code >= 0 else 128 - code
    finally:
        # This exact newly-created empty cgroup only; never a recursive cleanup.
        memory_group.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
