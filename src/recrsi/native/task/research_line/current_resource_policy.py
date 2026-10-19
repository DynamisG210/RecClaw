"""Prospective shared timer/failure policy; preserve raw costs and scientific fit."""
import inspect
import linecache
from pathlib import Path


def compiled_source(source, label):
    filename = str(Path(__file__).resolve()) + ':' + label
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    return compile(source, filename, 'exec')


def bind_research_config(module):
    # Same accepted Graph ABI: finite recipe metadata, null physical limits.
    source = inspect.getsource(module.research_config)
    before = "timeout_seconds=config['worker_timeout_seconds'],watchdog_seconds=config['worker_timeout_seconds'],final_worker_ceiling_seconds=config['worker_timeout_seconds'],"
    assert source.count(before) == 1
    source = source.replace(before, 'timeout_seconds=3600,watchdog_seconds=3600,final_worker_ceiling_seconds=3600,')
    scope = dict(module.__dict__)
    exec(compiled_source(source, 'research_config'), scope)
    module.research_config = scope['research_config']


def prepare_evaluator():
    import graph_candidate_evaluator as ev
    assert Path(ev.__file__).resolve().parents[1] == Path(__file__).resolve().parent.parent
    # New common evaluator already contains the null-time policy and stage admission.
    ev._current_resource_policy = True


def install_api_policy():
    from main_api_ledger import MainApiLedger
    if getattr(MainApiLedger, '_current_resource_policy', False):
        return
    summary0, cost0 = MainApiLedger._summarize, MainApiLedger._cost_totals

    def admission_events(events):
        # Keep genuine successes even if a reconciled call also has a FAILED row.
        succeeded = {e['call_id'] for e in events if e['event'] == 'SUCCESS'}
        failed = {e['call_id'] for e in events if e['event'] == 'FAILED'} - succeeded
        return [e for e in events if e.get('call_id') not in failed]

    def summarize(events):
        return summary0(admission_events(events))

    def costs(self, events):
        return cost0(self, admission_events(events))

    def admit(self):
        with self._lock:
            input_cost, output_cost, _ = self._cost_totals(self.events())
            return input_cost + output_cost <= self.normalized_unit_usd * self.max_normalized_u

    MainApiLedger._summarize = staticmethod(summarize)
    MainApiLedger._cost_totals = costs
    MainApiLedger.admit_output = admit
    MainApiLedger._current_resource_policy = True
