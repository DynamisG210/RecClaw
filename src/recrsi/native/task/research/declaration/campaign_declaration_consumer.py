"""Explicit cold native consumer hooks. Original broker receipts stay failures.

Install only inside a prospectively frozen, single-owner campaign process.
No existing search entrypoint imports this preparation module.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from declaration_provider_v3 import DeclarationProvider, CompletionPending, ScientificCompletionFailed
from declaration_provider_v2 import digest
from native_failure_bridge import NativeFailureBridge


@dataclass(frozen=True)
class CompositeConsumption:
    native: object
    response: dict
    artifact: dict
    completion_attempts: tuple
    physical_attempts: tuple

    @property
    def call(self): return None  # never forge a successful CanaryBrokerCall
    @property
    def failure(self): return self.native.failure
    @property
    def sealed_request(self): return self.native.sealed_request
    @property
    def attempts(self): return (*self.native.attempts, *self.completion_attempts)


class CampaignDeclarationConsumer:
    def __init__(self, bridge):
        self.bridge = bridge
        self.consumed = {}  # original object retained, so id cannot be reused
        self.finalized = set()

    @classmethod
    def from_manifest(cls, path, *, expected_sha256):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Frozen consumer manifest changed")
        manifest = json.loads(raw)
        if manifest['schema'] != 'PROSPECTIVE_COLD_DECLARATION_V1':
            raise ValueError("Unknown prospective consumer contract")
        config = Path(manifest['provider_config'])
        if hashlib.sha256(config.read_bytes()).hexdigest() != manifest['provider_config_sha256']:
            raise ValueError("Frozen provider configuration changed")
        completion = DeclarationProvider.from_config(config, model=manifest['completion_model'])
        if completion.settings['run_identity'] != manifest['run_identity']:
            raise ValueError("Run identity differs from frozen consumer")
        return cls(NativeFailureBridge(completion, frozen_session_id=manifest['native_session_id']))

    def _evidence(self, result):
        root = self.bridge.root
        logical = result.sealed_request['logical_call_id']
        call_root = root/'broker_calls'/hashlib.sha256(logical.encode()).hexdigest()
        # MAIN durable output capture stores actual wire bytes, not a regenerated request.
        saved = root/'durable_provider_responses'
        matches = []
        for request in saved.glob('*.request'):
            if json.loads(request.read_bytes()) == result.sealed_request['request_payload']:
                matches.append(request)
        if len(matches) != 1:
            raise ValueError("One saved actual wire is required; never redraw SUCCESS")
        request = matches[0]
        return dict(call_root=call_root, wire_bytes=request.read_bytes(),
                    response_bytes=request.with_suffix('.response').read_bytes())

    def _consume(self, owner, budget_key, result, native_outputs):
        if owner.session_id != self.bridge.frozen_session_id:
            raise ValueError("Campaign owner session differs from frozen manifest")
        total = owner.proposal_output_token_ceiling_total_per_context
        spent = owner._proposal_output_tokens_by_context.get(budget_key, 0)
        remaining = self.bridge.completion.ledger.max_output_tokens
        if total is not None: remaining = min(remaining, total-spent-native_outputs(result))
        if remaining <= 0:
            raise ScientificCompletionFailed("Original role output budget has no completion allowance")
        old = self.bridge.completion
        # Same ledger/resources/route; only the actual wire ceiling is bounded by this role.
        bounded = DeclarationProvider(old.settings, old.ledger, old.resources, model=old.model,
                                      http_transport=old.http_transport, output_token_ceiling=remaining)
        bridge = NativeFailureBridge(bounded, frozen_session_id=self.bridge.frozen_session_id)
        evidence = self._evidence(result)
        args = bridge.resolve(result, **evidence)
        identity = dict(run_identity=old.settings['run_identity'], logical_call_id=args['logical_call_id'],
                        session_id=args['session_id'])
        attempt = self.bridge.root/'declaration_completions'/digest(identity)
        request = attempt/'REQUEST.json'
        if request.exists():
            original_ceiling = json.loads(request.read_bytes())['max_output_tokens']
            if original_ceiling > remaining:
                raise CompletionPending("Saved wire exceeds current role remainder; no wire rewrite")
            bounded.output_token_ceiling = original_ceiling
        response = bounded.complete(**args)
        artifact = attempt/'COMPOSITE_VALIDATED.json'
        calls = {json.loads(p.read_bytes())['call_id'] for p in attempt.glob('physical/*/CALL.json')}
        events = old.ledger.events()
        infra = {e['call_id'] for e in events if e.get('event')=='COMPLETION_INFRASTRUCTURE_FAILURE'}
        physical = []
        for call in sorted(calls):
            terminal = [e for e in events if e.get('call_id')==call and e['event'] in {'SUCCESS','FAILED'}]
            if len(terminal)!=1: raise CompletionPending("Completion physical accounting unresolved")
            event = terminal[0]
            if not event.get('usage_reported') and call not in infra:
                raise CompletionPending("Non-infrastructure completion usage is unresolved")
            physical.append(dict(call_id=call, status=event['event'],
                output_tokens=event.get('output_tokens'), input_tokens=event.get('input_tokens'),
                usage_reported=event.get('usage_reported'), infrastructure=call in infra,
                authority='COMMON_LEDGER_COMPLETION_PHYSICAL_NOT_NATIVE_ATTEMPT'))
        return CompositeConsumption(result, response,
            dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
            tuple(e for e in physical if not e['infrastructure']), tuple(physical))

    @contextmanager
    def install(self, module):
        original_finalize=module._finalize_provider_result
        original_require=module._require_success
        original_trace=module._call_trace
        native_outputs=module._physical_attempt_output_tokens
        def finalize(owner, **kwargs):
            result=kwargs['result']
            eligible=(kwargs['kind']=='research_producer' and result.call is None and
                      (result.failure or {}).get('failure_class')=='SCHEMA_VALIDATION_FAILURE')
            if not eligible: return original_finalize(owner, **kwargs)
            key=(id(owner),kwargs['budget_key'],result.sealed_request['logical_call_id'])
            wrapped=self.consumed.get(id(result))
            if wrapped is None:
                wrapped=self._consume(owner,kwargs['budget_key'],result,native_outputs)
                self.consumed[id(result)]=wrapped
            if key in self.finalized: return
            original_finalize(owner,**dict(kwargs,result=wrapped))
            self.finalized.add(key)
        def require(result, *, kind):
            wrapped=self.consumed.get(id(result))
            if wrapped is not None: return wrapped.response
            return original_require(result,kind=kind)
        def trace(**kwargs):
            wrapped=kwargs['result']
            if not isinstance(wrapped,CompositeConsumption):return original_trace(**kwargs)
            value=original_trace(**dict(kwargs,result=wrapped.native))
            value['consumer_status']='DECLARATION_COMPOSITE_VALIDATED'
            value['composite']=wrapped.artifact
            value['completion_physical_attempts']=list(wrapped.physical_attempts)
            value['scientific_output_tokens']=native_outputs(wrapped)
            value['native_receipt_unchanged']=True
            return value
        module._finalize_provider_result=finalize
        module._require_success=require
        module._call_trace=trace
        try: yield self
        finally:
            module._finalize_provider_result=original_finalize
            module._require_success=original_require
            module._call_trace=original_trace
