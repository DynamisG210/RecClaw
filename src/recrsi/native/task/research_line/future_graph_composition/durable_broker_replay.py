"""Prospective graph-process scope: consume a bound paid response without HTTP.

Requires v4 transport durability, an unchanged native broker, and MAIN's external
ledger (no broker request_budget). Does not change any stable-task defaults.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from weakref import WeakKeyDictionary


VERSION = 'graph-bound-durable-broker-replay-v1'
_binding = ContextVar('graph_paid_broker_binding', default=None)
_replay_bytes = ContextVar('graph_paid_broker_replay_bytes', default=None)
_native_replays = ContextVar('graph_native_paid_replays', default=None)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def immutable_json(path, value):
    data = json.dumps(value, sort_keys=True).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Conflicting durable broker binding')
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def durable_broker_scope(broker_module, durability_module, api_ledger):
    """Enter inside native_ro_provider, instead of standalone v4 durability.

    The native provider still enforces capacity before reaching the broker.
    Unsettled/late/mismatched evidence is retained and stops replay, never redrawn.
    An old v4 response without a broker binding is deliberately not reassigned.
    """
    ledger = Path(api_ledger).resolve()
    root = ledger.parent / 'durable_provider_responses'
    cls = broker_module.LabApiCanaryBrokerV1
    from recclaw_core.experiments.helix_abc_v1 import fresh_r1
    original_bounded = fresh_r1._bounded_provider_call_direct
    original_seal = cls.sealed_request_identity
    original_call = cls.call_with_session
    original_replay = cls.replay_stored
    original_persist = durability_module.persist_paid_response
    original_send_builder = durability_module.corrected_send
    sealed = WeakKeyDictionary()
    upstream = broker_module.urlrequest

    def seal(self, **kwargs):
        identity = original_seal(self, **kwargs)
        sealed[self] = (dict(kwargs), identity)
        return identity

    def make_binding(self, logical_id, identity):
        return dict(
            version=VERSION, logical_call_id=logical_id,
            proposal_generation_session_id=identity['proposal_generation_session_id'],
            request_digest=broker_module.sha256_digest(identity),
            broker_release_digest=self.release.release_digest,
            request_body_sha256=sha(broker_module.canonical_json_bytes(identity['request_payload'])),
        )

    def binding_path(binding):
        key = broker_module.canonical_json_bytes([
            binding['logical_call_id'], binding['proposal_generation_session_id']])
        return root / (sha(key) + '.binding.json')

    def persist(settings, call_id, label, payload, raw, request_data):
        binding = _binding.get()
        if binding is None or Path(settings['api_ledger']).resolve() != ledger:
            raise ValueError('Durable response lacks this graph broker binding')
        if binding['request_body_sha256'] != label:
            raise ValueError('Broker identity differs from wire request')
        original_persist(settings, call_id, label, payload, raw, request_data)
        immutable_json(binding_path(binding), dict(binding, call_id=call_id,
                       response_sha256=sha(raw)))

    def send_builder(module):
        send = original_send_builder(module)

        async def admitted_send(self, request, timeout):
            raw = await send(self, request, timeout)
            # send has completed the original accounting and late-output gate.
            path = binding_path(_binding.get())
            immutable_json(path.with_suffix('.admitted.json'), json.loads(path.read_bytes()))
            return raw

        return admitted_send

    def load_response(self, binding, identity):
        path = binding_path(binding)
        if not path.exists():
            return None
        record = json.loads(path.read_bytes())
        if any(record.get(k) != v for k, v in binding.items()):
            raise ValueError('Sealed durable broker identity mismatch')
        call_id = record['call_id']
        if not isinstance(call_id, str) or not call_id.isalnum():
            raise ValueError('Invalid durable call identity')
        request = (root / (call_id + '.request')).read_bytes()
        raw = (root / (call_id + '.response')).read_bytes()
        receipt = json.loads((root / (call_id + '.json')).read_bytes())
        payload = identity['request_payload']
        if (request != broker_module.canonical_json_bytes(payload)
                or sha(raw) != record['response_sha256']
                or receipt.get('call_id') != call_id
                or receipt.get('request_body_sha256') != sha(request)
                or receipt.get('response_sha256') != sha(raw)
                or receipt.get('model') != payload['model']):
            raise ValueError('Durable request/response receipt mismatch')
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        events = [row for row in rows if row.get('call_id') == call_id]
        successes = [row for row in events if row.get('event') == 'SUCCESS']
        usage = json.loads(raw).get('usage', {})
        if (len(successes) != 1 or any(row.get('event') not in {'RESERVED', 'SUCCESS'} for row in events)
                or successes[0].get('label') != sha(request)
                or successes[0].get('model') != payload['model']
                or successes[0].get('requests') != 1
                or not successes[0].get('usage_reported')
                or any(successes[0].get(k) != usage.get(k) for k in ('input_tokens', 'output_tokens'))):
            raise ValueError('Original paid accounting unresolved or mismatched; no redraw')
        if (ledger.parent / 'late_provider_outputs' / (call_id + '.json')).exists():
            raise ValueError('Late provider output cannot enter native replay')
        admission = path.with_suffix('.admitted.json')
        if not admission.exists() or json.loads(admission.read_bytes()) != record:
            raise ValueError('Original output admission unresolved; retain without redraw')
        return raw, record

    def urlopen(request, *args, **kwargs):
        saved = _replay_bytes.get()
        if saved is None:
            return upstream.urlopen(request, *args, **kwargs)
        expected, raw = saved
        if request.data != expected:
            raise ValueError('Native replay wire bytes changed')
        response = io.BytesIO(raw)
        response.status = 200
        return response

    def recover(self, kwargs, identity, binding, saved):
        if self.request_budget is not None:
            raise ValueError('Replay scope is limited to externally accounted MAIN broker')
        raw, record = saved
        token = _replay_bytes.set((broker_module.canonical_json_bytes(identity['request_payload']), raw))
        try:
            try:
                original_call(self, **kwargs)
            except broker_module.CanaryBrokerError:
                # Original parser has persisted a terminal failure where applicable.
                # replay_stored reports it with zero new usage, preserving rejection.
                if self._connection.execute('SELECT 1 FROM calls WHERE logical_call_id=?',
                                            (kwargs['logical_call_id'],)).fetchone() is None:
                    raise
        finally:
            _replay_bytes.reset(token)
        immutable_json(binding_path(binding).with_suffix('.recovered.json'), dict(
            version=VERSION, original=record, real_http=0, new_accounting_events=0,
            receipt_kind='RECONSTRUCTED_NATIVE_CACHE_NOT_ORIGINAL_BROKER_RECEIPT'))
        mark_replay(binding['request_digest'])
        return original_replay(self, logical_call_id=kwargs['logical_call_id'],
            proposal_generation_session_id=identity['proposal_generation_session_id'],
            request_digest=binding['request_digest'])

    def call(self, **kwargs):
        args = {k: v for k, v in kwargs.items() if k != 'logical_call_id'}
        identity = seal(self, **args)
        binding = make_binding(self, kwargs['logical_call_id'], identity)
        # Preserve native success/failure behavior when its cache already exists.
        if self._connection.execute('SELECT 1 FROM calls WHERE logical_call_id=?',
                                    (kwargs['logical_call_id'],)).fetchone():
            return original_call(self, **kwargs)
        saved = load_response(self, binding, identity)
        if saved is not None:
            return recover(self, kwargs, identity, binding, saved)
        token = _binding.set(binding)
        try:
            return original_call(self, **kwargs)
        finally:
            _binding.reset(token)

    def replay(self, **kwargs):
        if self._connection.execute('SELECT 1 FROM calls WHERE logical_call_id=?',
                                    (kwargs['logical_call_id'],)).fetchone():
            row = self._connection.execute('SELECT request_digest, proposal_generation_session_id '
                'FROM calls WHERE logical_call_id=?', (kwargs['logical_call_id'],)).fetchone()
            if row == (kwargs['request_digest'], kwargs['proposal_generation_session_id']):
                mark_replay(kwargs['request_digest'])
            return original_replay(self, **kwargs)
        args, identity = sealed.get(self, (None, None))
        if args is None:
            return original_replay(self, **kwargs)
        binding = make_binding(self, kwargs['logical_call_id'], identity)
        if (kwargs['request_digest'] != binding['request_digest']
                or kwargs['proposal_generation_session_id'] != binding['proposal_generation_session_id']):
            raise ValueError('Replay differs from native sealed request')
        saved = load_response(self, binding, identity)
        if saved is None:
            return original_replay(self, **kwargs)
        return recover(self, dict(args, logical_call_id=kwargs['logical_call_id']),
                       identity, binding, saved)

    def mark_replay(digest):
        replays = _native_replays.get()
        if replays is not None:
            replays.add(digest)

    def bounded(**kwargs):
        # fresh_r1 otherwise reads historical failure usage from SQLite and
        # reports it as a new bill, even though replay_stored charged zero.
        replays = set()
        token = _native_replays.set(replays)
        try:
            result = original_bounded(**kwargs)
        finally:
            _native_replays.reset(token)
        if not replays:
            return result

        def zero_new_usage(row):
            if row is None or row.get('request_digest') not in replays:
                return row
            row = dict(row)
            row['native_usage_before_replay_normalization'] = {k: row[k] for k in
                ('input_tokens', 'cached_input_tokens', 'output_tokens', 'billed_tokens') if k in row}
            for k in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'billed_tokens',
                      'physical_call_count', 'latency_ms'):
                row[k] = 0
            row['usage_charge_basis'] = 'STORED_PAID_REPLAY_NO_NEW_CHARGE'
            row['durable_response_replay'] = True
            return row

        return replace(result, attempts=tuple(zero_new_usage(x) for x in result.attempts),
                       failure=zero_new_usage(result.failure))

    proxy = SimpleNamespace(**vars(upstream))
    proxy.urlopen = urlopen
    with patch.object(cls, 'sealed_request_identity', seal), \
         patch.object(cls, 'call_with_session', call), \
         patch.object(cls, 'replay_stored', replay), \
         patch.object(broker_module, 'urlrequest', proxy), \
         patch.object(fresh_r1, '_bounded_provider_call_direct', bounded), \
         patch.object(durability_module, 'persist_paid_response', persist), \
         patch.object(durability_module, 'corrected_send', send_builder), \
         durability_module.durable_transport_scope():
        yield
