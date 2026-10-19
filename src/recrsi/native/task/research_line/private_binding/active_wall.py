"""One persisted run clock; parallel activities do not multiply elapsed time.

Only the trusted run supervisor records proven run-wide exclusive interruptions.
A single failed API request is not proof that the entire run was inactive.
The controller creates this clock before initialization/AutoSpace; consumers open
that same clock, never start a fresh allowance after resuming or in a child.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time


class ActiveWallClock:
    def __init__(self, path, identity, *, now_ns=time.time_ns):
        self.path = Path(path)
        self.identity, self.now_ns = identity, now_ns

    @classmethod
    def create(cls, path, identity, *, limit_seconds=None, now_ns=time.time_ns):
        clock = cls(path, identity, now_ns=now_ns)
        clock.path.parent.mkdir(parents=True, exist_ok=True)
        with clock._locked():
            if clock.path.exists():
                state = clock._read()
                if state['limit_seconds'] != limit_seconds:
                    raise ValueError('Cannot reset or change the active-wall allowance')
            else:
                clock._write(dict(identity=identity, started_ns=now_ns(),
                    limit_seconds=limit_seconds, exclusions=[], stopped=False))
        return clock

    @contextmanager
    def _locked(self):
        with self.path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _read(self):
        state = json.loads(self.path.read_text())
        if state['identity'] != self.identity:
            raise ValueError('Active-wall clock belongs to a different run')
        return state

    def _write(self, state):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2))
        temporary.replace(self.path)

    @staticmethod
    def _elapsed(state, now):
        if now < state['started_ns']:
            raise ValueError('Host clock predates the run; elapsed time is unresolved')
        # Union, never sum overlapping platform/external/queue interruptions.
        excluded = 0
        end = state['started_ns']
        intervals = list(state['exclusions'])
        if state.get('pause'):
            intervals.append(dict(start_ns=state['pause']['start_ns'], end_ns=now))
        for row in sorted(intervals, key=lambda r: r['start_ns']):
            start, stop = max(end, row['start_ns']), min(now, row['end_ns'])
            if stop > start:
                excluded += stop - start
                end = stop
        return ((now - state['started_ns'] - excluded) / 1e9
                + state.get('root_reference_cost', {}).get('active_seconds', 0.)
                + state.get('autospace_reference_cost', {}).get('active_seconds', 0.))

    def record_exclusive_interruption(self, *, start_ns, end_ns, reason, evidence):
        """Supervisor-only: evidence must establish exclusion for the whole run.

        Do not call for one relay failure while another worker/agent is active.
        Once output admission stopped the run, later evidence cannot revive it.
        """
        if reason not in {'EXTERNAL_SERVICE', 'PLATFORM_INTERRUPTION', 'RESOURCE_QUEUE'} or not evidence:
            raise ValueError('An exclusion requires a defined external cause and evidence')
        with self._locked():
            state = self._read()
            if not state['started_ns'] <= start_ns <= end_ns <= self.now_ns():
                raise ValueError('Exclusion lies outside the actual run interval')
            row = dict(start_ns=start_ns, end_ns=end_ns, reason=reason, evidence=evidence)
            if row not in state['exclusions']:
                state['exclusions'].append(row)
                self._write(state)

    def pause(self, *, owner, start_ns, reason, evidence):
        """Trusted supervisor established no other active work in this interval."""
        if reason not in {'EXTERNAL_SERVICE', 'PLATFORM_INTERRUPTION', 'RESOURCE_QUEUE'} or not evidence:
            raise ValueError('A whole-run pause requires its cause and evidence')
        with self._locked():
            state = self._read()
            if not state['started_ns'] <= start_ns <= self.now_ns():
                raise ValueError('Pause starts outside the actual run')
            if state.get('pause'):
                raise ValueError('A whole-run pause is already open')
            state['pause'] = dict(owner=owner, start_ns=start_ns, reason=reason, evidence=evidence)
            self._write(state)

    def resume(self, *, owner=None):
        """Settle the interruption BEFORE any new activity checks the deadline."""
        with self._locked():
            state = self._read()
            pause = state.get('pause')
            if pause and (owner is None or pause['owner'] == owner):
                end_ns = self.now_ns()
                if end_ns < pause['start_ns']:
                    raise ValueError('Pause duration is unresolved')
                state['exclusions'].append(dict(start_ns=pause['start_ns'], end_ns=end_ns,
                    reason=pause['reason'], evidence=pause['evidence']))
                state['pause'] = None
                self._write(state)

    def snapshot(self):
        with self._locked():
            state = self._read()
            elapsed = self._elapsed(state, self.now_ns())
            return dict(active_seconds=elapsed, limit_seconds=state['limit_seconds'],
                remaining_seconds=(float('inf') if state['limit_seconds'] is None else max(0., state['limit_seconds'] - elapsed)),
                stopped=state['stopped'])

    def admit_output(self):
        with self._locked():
            state = self._read()
            admitted = not state['stopped'] and (state['limit_seconds'] is None or self._elapsed(state, self.now_ns()) <= state['limit_seconds'])
            if not admitted and not state['stopped']:
                state['stopped'] = True
                self._write(state)
            return admitted

    def require_capacity(self):
        from resource_stop import SearchResourceStop
        with self._locked():
            state = self._read()
            exhausted = state['stopped'] or (state['limit_seconds'] is not None and self._elapsed(state, self.now_ns()) >= state['limit_seconds'])
            if exhausted and not state['stopped']:
                state['stopped'] = True
                self._write(state)
        if exhausted:
            raise SearchResourceStop('MAIN active-wall allowance exhausted')
