"""One explicit CUDA-worker boundary shared by E1's two controllers.

The controllers never import candidate code. Existing native qualification and
training retain their semantics; only interpreter/process ownership changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def resolve_gpu_uuid(driver_root: Path, index: int) -> str:
    result = subprocess.run([str(driver_root / 'nvidia-smi'), '-i', str(index),
        '--query-gpu=uuid', '--format=csv,noheader'], check=True, capture_output=True,
        text=True, env={'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': str(driver_root)})
    value = result.stdout.strip()
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', value):
        raise ValueError('selected GPU did not resolve to one full UUID')
    return value


class WorkerProcess(subprocess.Popen):
    """Keep the private shared-memory bind alive until the actual process exits."""

    def __init__(self, command, *, shared_memory, worker_pid_path, **kwargs):
        self.shared_memory = shared_memory
        self.worker_pid = None
        try:
            super().__init__(command, **kwargs)
            deadline = time.monotonic() + 30
            while not worker_pid_path.is_file():
                if self.poll() is not None:
                    stdout, stderr = self.communicate()
                    raise RuntimeError(f'worker namespace setup failed: {stderr}')
                if time.monotonic() >= deadline:
                    self.kill()
                    self.wait()
                    raise TimeoutError('worker namespace setup did not publish its PID')
                time.sleep(0.01)
            self.worker_pid = int(worker_pid_path.read_text())
        except BaseException:
            if getattr(self, 'pid', None) is not None and self.poll() is None:
                self.kill()
                self.wait()
            shared_memory.cleanup()
            raise

    def send_signal(self, sig):
        if self.poll() is not None:
            return
        if self.worker_pid is not None:
            try:
                os.kill(self.worker_pid, sig)
            except ProcessLookupError:
                pass
        else:
            # Only the still-privileged bootstrap needs sudo cancellation.
            subprocess.run(['sudo', '-n', '/bin/kill', '-s', str(int(sig)), str(self.pid)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def wait(self, *args, **kwargs):
        result = super().wait(*args, **kwargs)
        self.shared_memory.cleanup()
        return result


@dataclass(frozen=True)
class WorkerBoundary:
    worker_python: Path
    repo_root: Path
    recbole_root: Path
    driver_root: Path
    control_root: Path
    gpu_uuid: str

    def popen(self, command, *, cwd, env, readonly, writable, **kwargs):
        worker = self.worker_python.resolve(strict=True)
        ro = [worker.parent.parent, self.driver_root, SCRIPTS,
              Path('/usr/lib/x86_64-linux-gnu'), Path('/usr/share/zoneinfo'),
              Path('/etc/ld.so.cache'), Path('/etc/passwd'), Path('/etc/nsswitch.conf'),
              *readonly]
        ro = sorted({str(Path(p).resolve(strict=True)) for p in ro})
        rw = sorted({str(Path(p).resolve(strict=True)) for p in writable})
        # A writable ancestor would defeat read-only intent through Landlock's
        # additive rules. Each job has separate immutable inputs and outputs.
        if any(Path(p).is_relative_to(Path(root)) for p in ro for root in rw):
            raise ValueError('worker read-only input lies inside a writable root')
        environment = dict(env)
        if environment.get('CUDA_VISIBLE_DEVICES') != self.gpu_uuid:
            raise ValueError('worker GPU selection must be the bound full UUID')
        environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                           LD_LIBRARY_PATH=str(self.driver_root))
        self.control_root.mkdir(parents=True, exist_ok=True)
        shared_memory = tempfile.TemporaryDirectory(prefix='recharness-e1-worker.', dir='/dev/shm')
        job = dict(command=[str(p) for p in command], cwd=str(cwd),
                   environment=environment, readonly=ro, writable=rw,
                   uid=os.getuid(), gid=os.getgid(), private_shm=shared_memory.name)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='launch-',
                                             dir=self.control_root, delete=False) as stream:
                worker_pid_path = Path(stream.name).with_suffix('.pid')
                job['worker_pid_path'] = str(worker_pid_path)
                json.dump(job, stream)
                job_path = stream.name
            return WorkerProcess(['sudo', '-n', '/usr/bin/python3', '-I',
                str(SCRIPTS / 'e1_namespace_exec.py'), job_path],
                shared_memory=shared_memory, worker_pid_path=worker_pid_path,
                cwd=cwd, env={'PATH': '/usr/bin:/bin'}, **kwargs)
        except BaseException:
            shared_memory.cleanup()
            raise

    def qualify(self, package, *, research_spec, candidate_root, fixture,
                unit_check=None, timeout_seconds=900):
        from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
            _disposable_process_failure, MechanicalQualificationRun,
        )
        from recclaw_core.experiments.helix_abc_v1.vnext_contracts import QualificationStageV1
        if unit_check is not None:
            raise ValueError('E1 qualification uses the native shared unit stage, not caller closures')
        work = fixture.checkpoint_dir.parent.resolve()
        work.mkdir(parents=True, exist_ok=True)
        for name in ('home', 'tmp', 'cache', 'checkpoints'):
            (work / name).mkdir(exist_ok=True)
        self.control_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', prefix='qualify-', suffix='.json',
                                         dir=self.control_root, delete=False) as stream:
            json.dump(dict(package=package.to_dict(), research_spec=research_spec.to_dict(),
                           candidate_root=str(candidate_root), fixture=asdict(fixture)),
                      stream, default=str)
            request = Path(stream.name)
        output = work / f'{request.stem}.result.json'
        environment = dict(PATH=f'{self.worker_python.parent}:/usr/bin:/bin',
            HOME=str(work / 'home'), TMPDIR=str(work / 'tmp'), XDG_CACHE_HOME=str(work / 'cache'),
            LANG='C.UTF-8', LC_ALL='C.UTF-8', CUDA_VISIBLE_DEVICES=self.gpu_uuid,
            PYTHONPATH=os.pathsep.join(map(str, (self.repo_root / 'src', self.repo_root,
                                                self.recbole_root, SCRIPTS))))
        started = time.monotonic()
        process = self.popen([self.worker_python, SCRIPTS / 'e1_qualification_worker.py', request, output],
            cwd=work, env=environment,
            readonly=[self.repo_root, self.recbole_root, candidate_root, fixture.data_path, request],
            writable=[work], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
        (work / 'qualification.stdout').write_text(stdout)
        (work / 'qualification.stderr').write_text(stderr)
        observation = dict(elapsed_wall_time_ms=int(1000 * (time.monotonic() - started)),
            exit_code=process.returncode, pid=process.worker_pid, process_isolated=True,
            start_method='explicit_worker', status='TIMEOUT' if timed_out else 'RESULT',
            worker_python=str(self.worker_python))
        if process.returncode == 0 and output.is_file():
            from scripts.e1_qualification_worker import decode_receipt
            payload = json.loads(output.read_text())
            receipt = decode_receipt(payload['receipt'])
            if receipt.candidate_package_digest != package.digest:
                raise ValueError('qualification result differs from the requested candidate')
            return MechanicalQualificationRun(receipt=receipt,
                stage_observations={**payload['stage_observations'], 'DISPOSABLE_PROCESS': observation},
                failure_detail=payload['failure_detail'], smoke_executions=payload['smoke_executions'])
        observation['status'] = 'TIMEOUT' if timed_out else 'WORKER_FAILURE'
        return _disposable_process_failure(package, stage=QualificationStageV1.CONSTRUCTION,
            reason_code='DISPOSABLE_PROCESS_TIMEOUT' if timed_out else 'DISPOSABLE_PROCESS_FAILED',
            message=stderr[-2000:], process_observation=observation)

    def training_popen(self, command, *, cwd, env, capability, candidate_root, **kwargs):
        # Only this run's outputs are writable, never the campaign/ledger root.
        read_paths = [self.repo_root, self.recbole_root, candidate_root,
                      *capability.allowed_read_only_roots]
        # Recipes/capabilities and prefix contracts are native command inputs.
        for flag in ('--execution-recipe-path', '--filesystem-capability-path', '--prefix-contract-path'):
            if flag in command:
                read_paths.append(Path(command[command.index(flag) + 1]))
        return self.popen(command, cwd=cwd, env=env, readonly=read_paths,
                          writable=[capability.result_root], **kwargs)
