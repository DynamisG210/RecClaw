"""RH E2 worker process using the already exercised E2 physical-resource launcher.

The launch helper is supplied explicitly by the prepared run. Native SE keeps
its frozen running copy; RH requests network isolation and a supervisor PID.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

from gagc.e1_worker import WorkerProcess


class E2Process(WorkerProcess):
    def send_signal(self, sig):
        if self.poll() is not None:
            return
        # E2 uses nobody (NAS root-squash), unlike E1's launching user. Target
        # only the trusted published worker PID, never a broad process pattern.
        target = self.worker_pid if self.worker_pid is not None else self.pid
        subprocess.run(["sudo", "-n", "/bin/kill", "-s", str(int(sig)), str(target)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


class E2Boundary:
    def __init__(self, *, worker_python: Path, driver_root: Path, control_root: Path,
                 launcher: Path, gpu_uuid: str, cpus: list[int]):
        self.worker_python = Path(worker_python).absolute()  # preserve the venv entry
        self.driver = Path(driver_root).resolve(strict=True)
        self.control_root = Path(control_root).resolve()
        self.launcher = Path(launcher).resolve(strict=True)
        self.resource_entry = self.launcher.with_name("se_confined_entry.py").resolve(strict=True)
        self.gpu_uuid, self.cpus = gpu_uuid, list(cpus)
        self.prefixes = json.loads(subprocess.check_output([
            str(self.worker_python), "-c", "import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))"],
            text=True))

    def popen(self, command, *, cwd, env, readonly, writable, **kwargs):
        output = Path(cwd).resolve(strict=True)
        if env.get("CUDA_VISIBLE_DEVICES") != self.gpu_uuid:
            raise ValueError("E2 worker GPU must match its bound UUID")
        roots = [Path(path).resolve(strict=True) for path in
                 [*self.prefixes, self.driver, self.launcher.parent, *readonly,
                  "/usr/bin", "/usr/lib", "/usr/share/zoneinfo", "/etc/ld.so.cache",
                  "/etc/passwd", "/etc/nsswitch.conf"]]
        ro = sorted({str(path) for path in roots})
        rw = sorted({str(Path(path).resolve(strict=True)) for path in writable})
        if any(Path(path).is_relative_to(Path(root)) for path in ro for root in rw):
            raise ValueError("E2 immutable input is inside a writable root")
        output.chmod(0o777)  # this fresh worker output only; inputs are mount-readonly
        self.control_root.mkdir(parents=True, exist_ok=True)
        control = Path(tempfile.mkdtemp(prefix="e2-", dir=self.control_root))
        control.chmod(0o777)
        shm = tempfile.TemporaryDirectory(prefix="shm-", dir=output)
        Path(shm.name).chmod(0o777)
        environment = dict(env, LD_LIBRARY_PATH=str(self.driver), PYTHONNOUSERSITE="1",
                           PYTHONDONTWRITEBYTECODE="1", SE_E2_OS_RECEIPT=str(output / "BOUNDARY.json"))
        if list(map(str, command[:3])) != [str(self.worker_python), "-B", "-u"]:
            raise ValueError("Expected the trusted E2 worker entry command")
        confined_command = [*command[:3], str(self.resource_entry), *command[3:]]
        job = dict(run_root=str(output), runs_parent=str(output.parent), cwd=str(output),
                   command=list(map(str, confined_command)), environment=environment,
                   readonly=ro, readonly_mounts=ro, writable=rw, private_shm=shm.name,
                   cpus=self.cpus, memory_limit_bytes=32 * 1024 ** 3,
                   network_isolated=True, receipt_schema="rh-e2-os-boundary-v1",
                   receipt=str(output / "BOUNDARY.json"), exit_record=str(output / "EXIT.json"),
                   worker_pid_path=str(control / "worker.pid"))
        job_path = control / "job.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        return E2Process(["sudo", "-n", "/usr/bin/python3", "-I", str(self.launcher), str(job_path)],
                         shared_memory=shm, worker_pid_path=control / "worker.pid",
                         env={"PATH": "/usr/bin:/bin"}, cwd=cwd, **kwargs)
