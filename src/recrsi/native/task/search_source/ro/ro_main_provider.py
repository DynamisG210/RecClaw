"""Common MAIN resources at the native RO broker's actual HTTP boundary.

Keep native schema validation, role routing, sealed requests, replay and receipts.
Only transport/accounting is adapted. Use inside one dedicated RO process;
parallel research roles share the existing process-safe MAIN ledger.
"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

from resource_stop import SearchResourceStop

expected_retry_label = ContextVar('expected_retry_label', default=None)


class MainROTransport:
    channel = 'recclaw:native_broker'

    def __init__(self, settings, ledger):
        from main_resources import MainResources
        self.settings, self.ledger = settings, ledger
        self.resources = MainResources(settings['resource_root'], settings['run_identity'],
                                       settings['active_wall_path'], ledger)

    def urlopen(self, request, *, timeout=None, context=None):
        # Synchronous native broker; its existing hard deadline remains outside
        # this call. A fresh async client runs in the current research-role thread.
        raw = asyncio.run(self.send(request, timeout))
        result = io.BytesIO(raw)
        result.status = 200
        return result

    async def send(self, request, timeout):
        import httpx
        from main_relay_retry import async_client
        from se_shared_api_ledger import BudgetExhausted, InputBudgetExceeded, ProviderExecutionPaused
        payload = json.loads(request.data)
        model = payload['model']
        if (model not in self.ledger.model_prices
                or payload.get('reasoning', {}).get('effort') != 'medium'
                or not 0 < payload.get('max_output_tokens', 0) <= self.ledger.max_output_tokens):
            raise ValueError('Native RO request differs from MAIN model/effort/token limits')
        channel = self.channel
        label = hashlib.sha256(request.data).hexdigest()
        if expected_retry_label.get() not in (None, label):
            raise ValueError('Recovered request differs from the original HTTP payload')
        self.resources.require_capacity()
        try:
            estimate = self.ledger.verify_input_budget(payload, channel=channel)
            call_id = self.ledger.reserve(channel=channel, label=label, model=model,
                                          estimated_input_tokens=self.ledger.max_input_tokens - self.ledger.input_estimate_margin)
        except (BudgetExhausted, InputBudgetExceeded, ProviderExecutionPaused) as error:
            raise SearchResourceStop(str(error)) from error
        started, usage = time.monotonic(), None
        try:
            url, headers = request.full_url, dict(request.header_items())
            route_config = self.settings.get('model_api_configs', {}).get(model)
            if route_config:
                from se_shared_api_ledger import _load_route_credentials
                route = _load_route_credentials(route_config)
                url = route['base_url'].rstrip('/') + '/responses'
                headers = {k: v for k, v in headers.items() if k.lower() != 'authorization'}
                headers['Authorization'] = 'Bearer ' + route['api_key']
            route_event = dict(call_id=call_id, model=model, request_body_sha256=label,
                url=url, credential_config=route_config or self.settings['api_config'],
                timestamp_ns=time.time_ns())
            route_log = Path(self.settings['api_ledger']).with_name('PROVIDER_ROUTE_EVENTS.jsonl')
            with route_log.open('a') as stream:
                stream.write(json.dumps(route_event, sort_keys=True) + '\n')
            limits = httpx.Timeout(timeout or 900, connect=30, write=60, pool=30)
            async with async_client(Path(self.settings['api_ledger']).with_name('EXTERNAL_RELAY_EVENTS.jsonl'),
                    limits, resources=self.resources, call_id=lambda: call_id) as client:
                response = await client.post(url, content=request.data, headers=headers)
                raw = response.content
                if response.is_error:
                    failed = Path(self.settings['api_ledger']).parent/'failed_provider_responses'
                    failed.mkdir(exist_ok=True)
                    (failed/(call_id+'.body')).write_bytes(raw)
                    (failed/(call_id+'.json')).write_text(json.dumps(dict(
                        call_id=call_id, request_body_sha256=label, http_status=response.status_code,
                        error_body_sha256=hashlib.sha256(raw).hexdigest())))
                try:
                    envelope = response.json()
                except ValueError:
                    envelope = None
                measured = envelope.get('usage') if isinstance(envelope, dict) else None
                if isinstance(measured, dict) and all(type(measured.get(k)) is int and measured[k] >= 0
                        for k in ('input_tokens', 'output_tokens')):
                    usage = dict(measured, requests=1)
                response.raise_for_status()
                if usage is None:
                    raise ValueError('Native RO response usage is unresolved')
        except BaseException as error:
            empty = getattr(error, 'confirmed_external_empty_evidence', None)
            if isinstance(error, SearchResourceStop) and empty is not None:
                self.ledger.record_external_empty_stop(call_id=call_id, model=model, evidence=empty)
            else:
                self.ledger.record('FAILED', call_id=call_id, channel=channel, label=label,
                    model=model, usage=usage, error=error, elapsed_seconds=time.monotonic()-started)
            stop = SearchResourceStop('Native transport interrupted; original accounting retained')
            stop.transport_interruption = dict(call_id=call_id, request_body_sha256=label)
            raise stop from error
        self.ledger.record('SUCCESS', call_id=call_id, channel=channel, label=label,
            model=model, usage=usage, elapsed_seconds=time.monotonic()-started)
        if not self.resources.admit_output():
            late = Path(self.settings['api_ledger']).parent/'late_provider_outputs'
            late.mkdir(exist_ok=True)
            (late/(call_id+'.json')).write_bytes(raw)
            raise SearchResourceStop('MAIN late RO output retained but not admitted')
        # Paid scientific/schema errors are deliberately passed to the unchanged
        # native parser; a malformed candidate is not a free external failure.
        return raw


@contextmanager
def native_ro_provider(config_path):
    from recclaw_core.experiments.helix_abc_v1 import fresh_r1, lab_api_broker
    from openevolve_provider import _settings_and_ledger
    settings, ledger = _settings_and_ledger(str(config_path))
    transport = MainROTransport(settings, ledger)
    native_urlrequest = lab_api_broker.urlrequest
    # Replace only this broker module's reference, not global urllib.urlopen.
    module = SimpleNamespace(**vars(native_urlrequest))
    module.urlopen = transport.urlopen

    def call(**kwargs):
        if os.environ.get('RECCLAW_PAIRED_PROVIDER_CALL_ROOT'):
            raise ValueError('Independent MAIN repeats cannot share paired-arm Provider calls')
        transport.resources.require_capacity()
        route_config = settings.get('model_api_configs', {}).get(kwargs['requested_model'], settings['api_config'])
        kwargs.update(wire_api='responses', token_ceiling=148000,
            credential_config_path=Path(route_config), credential_index=0,
            credential_schedule=(0,), request_budget=None)
        return fresh_r1.bounded_provider_call(**kwargs)

    with patch.object(lab_api_broker, 'urlrequest', module):
        yield call
