"""Shared MAIN admission consumed by provider and native worker/evaluator."""
import json
from pathlib import Path

from active_wall import ActiveWallClock
from gpu_job_budget import GPUJobBudget
from resource_stop import SearchResourceStop


class MainResources:
    def __init__(self, root, identity, active_wall_path, api_ledger):
        self.root = Path(root)
        self.clock = ActiveWallClock(active_wall_path, identity)
        self.gpu = GPUJobBudget(self.root/'GPU_JOB_BUDGET.json', identity)
        self.api = api_ledger

    def _require_physical_capacity(self):
        self.clock.resume()
        self.clock.require_capacity()
        gpu = self.gpu.snapshot()
        # Reservations hold worker capacity; they are not consumed GPU time.
        # GPUJobBudget.reserve still atomically enforces new-worker admission.
        if (gpu['gpu_seconds'] >= self.gpu.contract['max_gpu_seconds']
                or not self.gpu.admit_output()):
            raise SearchResourceStop('MAIN GPU capacity exhausted')
        api = self.api.summary()
        if api['unheld_unknown_usage_failures'] or api['fatal_provider_statuses']:
            raise SearchResourceStop('MAIN unresolved provider state pauses new physical work')
        if not self.api.admit_output() or api['normalized_cost_usd'] >= 7:
            raise SearchResourceStop('MAIN reference API allowance exhausted or unresolved')

    @staticmethod
    def _evaluation_occupancy(state):
        reserved = {row['source_sha256'] for row in state['pending'].values() if row['new']}
        return state['genuine_count'] + len(reserved)

    def require_capacity(self):
        """Admit work that has not reserved a MAIN evaluation slot yet."""
        self._require_physical_capacity()
        path = self.root/'EVALUATION_LEDGER.json'
        if path.exists():
            # Atomic snapshots produced by EvaluationLedger; never candidate state.
            state = json.loads(path.read_text())
            if self._evaluation_occupancy(state) >= state['contract']['maximum']:
                raise SearchResourceStop('MAIN genuine evaluation capacity exhausted or in flight')

    def require_reserved_evaluation_capacity(self, source_sha256):
        """Recheck resources for a physical run whose MAIN slot is already pending."""
        self._require_physical_capacity()
        path = self.root/'EVALUATION_LEDGER.json'
        if not path.exists():
            raise ValueError('Reserved evaluation has no MAIN evaluation ledger')
        state = json.loads(path.read_text())
        matches = [row for row in state['pending'].values()
                   if row['source_sha256'] == source_sha256]
        if len(matches) != 1:
            raise ValueError('Physical evaluation needs exactly one matching MAIN reservation')
        # The matching reservation is already part of occupancy. Equality is the
        # legal final slot; only an actual over-reservation must stop this worker.
        if self._evaluation_occupancy(state) > state['contract']['maximum']:
            raise SearchResourceStop('MAIN genuine evaluation capacity exceeded')

    def admit_output(self):
        # Finishing the last admitted evaluation is allowed; count is settled by
        # its existing reservation. Further provider/worker admission is stopped.
        path = self.root/'EVALUATION_LEDGER.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['genuine_count'] >= state['contract']['maximum']:
                return False
        return self.clock.admit_output() and self.gpu.admit_output() and self.api.admit_output()

    def pause_empty_relay(self, call_id, started_ns, evidence):
        """Prepared serial OE loop: verify absence of overlapping work as well.

        Parallel native consumers still need their activity-aware pause binding;
        an overlapping worker/request is never treated as an exclusive outage.
        """
        now = self.clock.now_ns()
        jobs = self.gpu.snapshot()['jobs'].values()
        if any(row['reserved_ns'] <= now and
               (row['finished_ns'] is None or row['finished_ns'] > started_ns) for row in jobs):
            return False
        calls = {}
        for event in self.api.events():
            if event.get('call_id') != call_id:
                calls.setdefault(event.get('call_id'), {})[event['event']] = event
        for rows in calls.values():
            reserved = rows.get('RESERVED')
            terminal = rows.get('SUCCESS') or rows.get('FAILED') or rows.get('EXTERNAL_EMPTY_STOP')
            if reserved and reserved['timestamp_ns'] <= now and (
                    terminal is None or terminal['timestamp_ns'] > started_ns):
                return False
        self.clock.pause(owner=call_id, start_ns=started_ns, reason='EXTERNAL_SERVICE', evidence=evidence)
        return True
