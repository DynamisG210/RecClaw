"""Measure device-bound native worker execution, outside parent orchestration.

GPU job time is selected-card count times worker duration, not utilization or
kernel time. Native qualification/training paths and OS isolation are unchanged.
Open intervals remain unknown after interruption, never silently zero charged.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from gagc.e1_worker import WorkerBoundary
from worker_watch import append_event, open_pidfd


def summarize(path):
    starts, ends = {}, {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        (starts if row['event'] == 'START' else ends)[row['job_id']] = row
    seconds = 0.0
    for job_id, end in ends.items():
        start = starts[job_id]
        elapsed = (end['finished_ns'] - start['started_ns']) / 1e9
        if elapsed < 0:
            raise ValueError('Host clock moved backwards during GPU work; time unresolved')
        seconds += elapsed * start['cards']
    return dict(gpu_job_seconds=seconds, completed_jobs=len(ends),
                open_jobs=[row for job_id, row in starts.items() if job_id not in ends])


@dataclass(frozen=True)
class TimedWorkerBoundary(WorkerBoundary):
    timing_path: Path
    resource_budget: object = None
    active_clock: object = None
    qualification_timeout_seconds: int = 900
    training_timeout_seconds: int = 1800

    def popen(self, command, **kwargs):
        token = None
        if self.resource_budget is not None:
            self.active_clock.resume()
            self.active_clock.require_capacity()
            kind = 'qualification' if any(Path(str(arg)).name in ('e1_qualification_worker.py', 'bert_qualification_worker.py')
                                           for arg in command) else 'training'
            requested = (self.qualification_timeout_seconds if kind == 'qualification'
                         else self.training_timeout_seconds)
            token, allowance = self.resource_budget.reserve(requested, kind=kind)
        # If bootstrap interrupts without a trustworthy exit, capacity stays
        # pending for recovery; never assume a missing return means zero work.
        process = super().popen(command, **kwargs)
        # This trusted namespace launcher publishes the exec-stable worker PID
        # immediately before applying its final allowlist and exec, not when
        # the parent starts compilation/qualification/LLM work.
        job_path = Path(process.args[-1])
        pid_path = Path(json.loads(job_path.read_text())['worker_pid_path'])
        started_ns = pid_path.stat().st_mtime_ns
        if token is not None:
            self.resource_budget.started(token, started_ns=started_ns,
                worker_pid=process.worker_pid, control_job=job_path, gpu_uuid=self.gpu_uuid)
        job_id = uuid.uuid4().hex
        append_event(self.timing_path, dict(event='START', job_id=job_id,
            started_ns=started_ns, pid=process.worker_pid,
            gpu_uuid=self.gpu_uuid, cards=1, control_job=str(job_path),
            basis='DEVICE_BOUND_NATIVE_WORKER_EXECUTION', clock='HOST_REALTIME_NS'))
        if token is not None:
            return self._durable_watch(process, token, allowance, requested, started_ns, job_id)
        completed = threading.Event()
        errors = []

        def observe():
            try:
                while process.poll() is None:
                    completed.wait(.05)
                finished_ns = time.time_ns()
                append_event(self.timing_path, dict(event='END', job_id=job_id,
                    finished_ns=finished_ns, exit_code=process.returncode,
                    observation_resolution_seconds=.05))
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        original_wait = process.wait

        def wait(*args, **kwargs):
            result = original_wait(*args, **kwargs)
            completed.wait()
            if errors:
                raise errors[0]
            return result

        process.wait = wait
        threading.Thread(target=observe, name='native-worker-time', daemon=True).start()
        return process

    def _durable_watch(self, process, token, allowance, requested, started_ns, job_id):
        """Keep physical settlement/deadlines alive when the consumer exits."""
        self.control_root.mkdir(parents=True, exist_ok=True)
        config_path = self.control_root / f'watch-{token}.json'
        result_path = self.control_root / f'watch-{token}.result.json'
        config = dict(budget_path=str(self.resource_budget.path.resolve()),
            budget_contract=self.resource_budget.contract,
            clock_path=str(self.active_clock.path.resolve()), clock_identity=self.active_clock.identity,
            token=token, allowance=allowance, requested=requested, started_ns=started_ns,
            result_path=str(result_path.resolve()), timing_path=str(self.timing_path.resolve()), job_id=job_id)
        config_path.write_text(json.dumps(config, sort_keys=True))
        try:
            pidfd = open_pidfd(process.worker_pid)
        except ProcessLookupError:
            pidfd = -1  # Already exited; observe now, never invent an earlier end.
        try:
            with config_path.with_suffix('.log').open('a') as log:
                watcher = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('worker_watch.py').resolve()),
                    str(config_path.resolve()), str(pidfd)], stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=() if pidfd < 0 else (pidfd,), start_new_session=True)
        finally:
            if pidfd >= 0:
                os.close(pidfd)
        original_wait = process.wait

        def wait(*args, **kwargs):
            result = original_wait(*args, **kwargs)
            if watcher.wait(timeout=30) != 0:
                raise RuntimeError(f'Worker timing supervisor failed; retain pending cost: {config_path}')
            observed = json.loads(result_path.read_text())
            self.resource_budget.record_wait_status(token, result)
            # Durable settlement and END already exist if the controller died.
            if observed['stopped_by_common_limit']:
                from resource_stop import SearchResourceStop
                raise SearchResourceStop('MAIN physical GPU/active-wall deadline')
            return result

        process.wait = wait
        return process
