"""Reserve physical GPU jobs, settle measured worker time, retain wrapper time.

The existing runtime's training callbacks remain audit events only. Qualification
and training are charged at the same actual protected worker boundary, never by
the parent wrapper. Pending jobs retain capacity across reconstruction.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time
import uuid


class GPUJobBudget:
    def __init__(self, path, identity, *, max_gpu_seconds=None):
        self.path = Path(path)
        self.contract = dict(identity=identity, max_gpu_seconds=max_gpu_seconds)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                state = json.loads(self.path.read_text())
                if state['contract'] != self.contract:
                    raise ValueError('Cannot change a run GPU budget or identity')
            else:
                state = dict(contract=self.contract, jobs={}, training_wrappers={})
            yield state
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, sort_keys=True, indent=2))
            temporary.replace(self.path)

    def _totals(self, state):
        jobs = state['jobs'].values()
        used = state.get('root_reference_cost', {}).get('gpu_seconds', 0.) + sum(row['gpu_seconds'] for row in jobs if row.get('finished_ns') is not None)
        reserved = sum(row['reserved_seconds'] for row in state['jobs'].values()
                       if row.get('finished_ns') is None)
        return used, reserved

    def remaining_gpu_secs(self):
        with self.transaction() as state:
            used, reserved = self._totals(state)
            return float('inf') if self.contract['max_gpu_seconds'] is None else max(0., self.contract['max_gpu_seconds'] - used - reserved)

    def reserve(self, requested_seconds, *, kind):
        from resource_stop import SearchResourceStop
        with self.transaction() as state:
            if kind == 'graph_training' and sum(row['kind'] == 'graph_training' for row in state['jobs'].values()) >= 36:
                raise SearchResourceStop('Original 36 training admissions exhausted')
            used, reserved = self._totals(state)
            seconds = requested_seconds if self.contract['max_gpu_seconds'] is None else min(requested_seconds if requested_seconds is not None else float('inf'), self.contract['max_gpu_seconds'] - used - reserved)
            if seconds is not None and seconds < 1:
                raise SearchResourceStop('MAIN GPU capacity exhausted or retained by unfinished jobs')
            token = uuid.uuid4().hex
            row = dict(kind=kind, reserved_seconds=0. if seconds is None else seconds, requested_seconds=requested_seconds, unlimited_stage=(seconds is None),
                       reserved_ns=time.time_ns(), started_ns=None, finished_ns=None)
            state['jobs'][token] = row
            return token, seconds

    def started(self, token, *, started_ns, worker_pid, control_job, gpu_uuid):
        with self.transaction() as state:
            row = state['jobs'][token]
            if row['started_ns'] is not None:
                raise ValueError('GPU job already has a physical worker')
            row.update(started_ns=started_ns, worker_pid=worker_pid,
                       control_job=str(control_job), gpu_uuid=gpu_uuid, cards=1)

    def finish(self, token, *, finished_ns, exit_code, stopped_by_common_limit):
        with self.transaction() as state:
            row = state['jobs'][token]
            if row['started_ns'] is None or finished_ns < row['started_ns']:
                raise ValueError('Physical GPU duration is unresolved')
            values = dict(finished_ns=finished_ns, exit_code=exit_code,
                          stopped_by_common_limit=stopped_by_common_limit,
                          gpu_seconds=(finished_ns-row['started_ns'])/1e9*row['cards'])
            if row['finished_ns'] is not None and any(row[k] != v for k,v in values.items()):
                raise ValueError('Physical job completion replay differs')
            row.update(values)

    def admit_output(self):
        with self.transaction() as state:
            used, _ = self._totals(state)
            now = time.time_ns()
            used += sum(max(0., (now-row['started_ns'])/1e9) * row['cards']
                        for row in state['jobs'].values()
                        if row['started_ns'] is not None and row['finished_ns'] is None)
            return (self.contract['max_gpu_seconds'] is None or used <= self.contract['max_gpu_seconds']) and not any(
                row.get('stopped_by_common_limit') and not row.get('resource_limit_cancelled_by_user') for row in state['jobs'].values())

    def record_wait_status(self, token, exit_code):
        """Attach a real parent wait status to an independently observed exit."""
        with self.transaction() as state:
            row = state['jobs'][token]
            if row['finished_ns'] is None:
                raise ValueError('Worker exit has not yet been observed')
            if row['exit_code'] not in (None, exit_code):
                raise ValueError('Worker wait status changed')
            row['exit_code'] = exit_code

    def start_training(self, source_sha256, timeout_secs):
        # Existing E1Runtime callback. Do not reserve/charge this parent wrapper.
        token = uuid.uuid4().hex
        with self.transaction() as state:
            state['training_wrappers'][token] = dict(source_sha256=source_sha256,
                timeout_seconds=timeout_secs, status='started')
        return token

    def finish_training(self, token, seconds, status):
        with self.transaction() as state:
            state['training_wrappers'][token].update(wrapper_seconds=seconds, status=status,
                                                    charged_as_gpu=False)

    def snapshot(self):
        with self.transaction() as state:
            used, reserved = self._totals(state)
            return dict(gpu_seconds=used, reserved_seconds=reserved, **state)
