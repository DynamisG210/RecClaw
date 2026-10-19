"""MAIN monetary reservations on the existing common usage/error ledger.

Keeps native token accounting and error evidence. New reservations use the actual
request model, not the most expensive model or uncharged in-flight requests.
"""
import fcntl
import threading
import time
import uuid
from se_shared_api_ledger import ApiLedger, BudgetExhausted, ProviderExecutionPaused, SCHEMA


class _ProcessLock:
    def __init__(self, path):
        self.path, self.local = path, threading.RLock()
        self.depth, self.stream = 0, None

    def __enter__(self):
        self.local.acquire()
        try:
            if not self.depth:
                self.stream = self.path.open('a+b')
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
            self.depth += 1
        except BaseException:
            if self.stream:self.stream.close()
            self.local.release()
            raise

    def __exit__(self, *args):
        self.depth -= 1
        if not self.depth:self.stream.close()
        self.local.release()


class MainApiLedger(ApiLedger):
    def __init__(self, *args, auxiliary_model_prices=None, **kwargs):
        super().__init__(*args, **kwargs)
        if auxiliary_model_prices is not None:
            expected = {'text-embedding-3-small': dict(input=.02, output=0.,
                cache_read=.02, cache_creation=.02)}
            if auxiliary_model_prices != expected:
                raise ValueError('Auxiliary pricing differs from the approved embedding-only exception')
            self.model_prices.update(auxiliary_model_prices)
        self._lock = _ProcessLock(self.path.with_suffix('.lock'))

    @staticmethod
    def _reserved_usd(events):
        calls={}
        for event in events:
            calls.setdefault(event.get('call_id'),{})[event['event']]=event
        return sum(row['RESERVED']['reserved_reference_usd'] for row in calls.values()
                   if 'RESERVED' in row and not ('SUCCESS' in row or 'FAILED' in row or 'EXTERNAL_EMPTY_STOP' in row))

    @staticmethod
    def _summarize(events):
        exempt = {row['call_id'] for row in events if row['event'] == 'EXTERNAL_EMPTY_STOP'}
        result = ApiLedger._summarize([row for row in events if row.get('call_id') not in exempt])
        result['logical_calls'] += len(exempt)
        result['external_unbilled_calls'] = len(exempt)
        held = {row['call_id']: row for row in events if row['event'] == 'UNRESOLVED_USAGE_HOLD'}
        unknown = {row['call_id'] for row in events if row['event'] == 'FAILED'
                   and not row.get('usage_reported')} - exempt
        result['unheld_unknown_usage_failures'] = len(unknown - held.keys())
        result['unresolved_reference_hold_usd'] = sum(
            row['reference_hold_usd'] for key, row in held.items() if key in unknown)
        return result

    def retain_unknown_hold(self, *, call_id, evidence):
        """Explicit recovery only: retain unknown usage, never manufacture zero.

        Hold the configured maximum request cost separately from measured cost.
        This permits a preserved request to continue without claiming exact cost
        matching. Uninspected failures still stop by default.
        """
        with self._lock:
            rows = [row for row in self.events() if row.get('call_id') == call_id]
            if any(row['event'] == 'UNRESOLVED_USAGE_HOLD' for row in rows):
                return
            failed = next(row for row in reversed(rows) if row['event'] == 'FAILED')
            if failed.get('usage_reported') or failed.get('http_status') != 502 or not evidence:
                raise ValueError('This recovery requires an inspected unresolved HTTP 502')
            reserved = next(row for row in rows if row['event'] == 'RESERVED')
            prices = self.model_prices[reserved['model']]
            bound = (self.max_input_tokens*prices['input'] + self.max_output_tokens*prices['output'])/1e6
            self._append_unlocked(dict(schema=SCHEMA, event='UNRESOLVED_USAGE_HOLD',
                call_id=call_id, timestamp_ns=time.time_ns(), reference_hold_usd=bound,
                usage_reported=False, evidence=evidence))

    def record_external_empty_stop(self, *, call_id, model, evidence):
        # A trusted common stop occurred between an already-classified empty
        # relay failure and its retry. There was no later physical dispatch.
        # Keep provider usage unreported, not a fabricated usage={0,0} response.
        with self._lock:
            self._append_unlocked(dict(schema=SCHEMA, event='EXTERNAL_EMPTY_STOP',
                call_id=call_id, model=model, timestamp_ns=time.time_ns(),
                scientific_requests=0, scientific_input_tokens=0, scientific_output_tokens=0,
                usage_reported=False, accounting_basis='CONFIRMED_EMPTY_RELAY_BEFORE_RETRY',
                evidence=evidence))

    def reserve(self, *, channel, label, model, requests=1, estimated_input_tokens=None):
        if requests != 1 or type(estimated_input_tokens) is not int or estimated_input_tokens < 0:
            raise ValueError('MAIN reserves one concrete model request with measured input estimate')
        prices=self.model_prices[model]
        input_bound=estimated_input_tokens+self.input_estimate_margin
        if input_bound>self.max_input_tokens:
            raise BudgetExhausted('MAIN request exceeds the unchanged context allowance')
        reserved_usd=(input_bound*prices['input']+self.max_output_tokens*prices['output'])/1_000_000
        with self._lock:
            events=self.events();summary=self._summarize(events)
            if summary['fatal_provider_statuses'] or summary['unheld_unknown_usage_failures']:
                raise ProviderExecutionPaused('MAIN retains unresolved/fatal provider state; no silent refund')
            if summary['requests']+1>self.max_requests:
                raise BudgetExhausted('MAIN request safety ceiling reached')
            input_cost,output_cost,_=self._cost_totals(events)
            cap=self.normalized_unit_usd*self.max_normalized_u
            if (input_cost+output_cost+self._reserved_usd(events)+reserved_usd
                    + summary['unresolved_reference_hold_usd']>cap):
                raise BudgetExhausted('MAIN completed plus in-flight reference API capacity exhausted')
            call_id=uuid.uuid4().hex
            self._append_unlocked(dict(schema=SCHEMA,event='RESERVED',call_id=call_id,
                channel=channel,label=label,model=model,timestamp_ns=time.time_ns(),requests=1,
                input_tokens=0,output_tokens=0,estimated_input_tokens=estimated_input_tokens,
                reserved_reference_usd=reserved_usd,usage_reported=False,elapsed_seconds=None,
                error_type=None,error_digest=None))
            return call_id

    def summary(self):
        with self._lock:
            result=super().summary()
            result['reserved_reference_usd']=self._reserved_usd(self.events())
            return result

    def admit_output(self):
        """Admit completed output without mistaking another failed call for lateness.

        Unresolved calls still pause new reservations. For an already completed
        response, retain their configured maximum request cost, never zero usage.
        """
        with self._lock:
            events = self.events()
            summary = self._summarize(events)
            covered = {e['call_id'] for e in events
                       if e['event'] in {'EXTERNAL_EMPTY_STOP', 'UNRESOLVED_USAGE_HOLD'}}
            unknown = {e['call_id']: e for e in events if e['event'] == 'FAILED'
                       and not e.get('usage_reported') and e['call_id'] not in covered}
            bound = 0.0
            for event in unknown.values():
                prices = self.model_prices.get(event.get('model'))
                if prices is None:
                    return False  # No justified upper bound for an unknown model.
                bound += (self.max_input_tokens * prices['input']
                          + self.max_output_tokens * prices['output']) / 1e6
            input_cost, output_cost, _ = self._cost_totals(events)
            return (input_cost + output_cost + summary['unresolved_reference_hold_usd'] + bound
                    <= self.normalized_unit_usd * self.max_normalized_u)
