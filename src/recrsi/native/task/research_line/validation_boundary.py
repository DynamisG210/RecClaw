"""Stop the affected fresh run at its next request after an observed contract failure.

A failure is retained by the native caller; already admitted work is not cancelled.
No numerical-quality rule, response rewriting, retry or new request is introduced.
"""
from contextlib import contextmanager, ExitStack
import fcntl
import functools
import json
from pathlib import Path
from unittest.mock import patch

class ContractBoundary:
    def __init__(self, output):
        self.path = Path(output) / 'CONTRACT_STOP.json'

    def require_new_request(self):
        from resource_stop import SearchResourceStop
        if self.path.exists():
            raise SearchResourceStop('REGION_VALIDATION_CONTRACT_STOP: prior failure retained; no new request')

    def record(self, stage, detail):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                value = dict(status='STOP_BEFORE_NEXT_REQUEST', stage=stage, detail=detail,
                             prior_work_retained=True, automatic_retry=False)
                temporary = self.path.with_suffix('.tmp')
                temporary.write_text(json.dumps(value, indent=2) + '\n')
                temporary.replace(self.path)

    def provider(self, original):
        @functools.wraps(original)
        def call(**kwargs):
            self.require_new_request()
            result = original(**kwargs)
            failure = result.failure or {}
            if failure.get('failure_class') == 'SCHEMA_VALIDATION_FAILURE':
                self.record('provider_schema', dict(failure_class='SCHEMA_VALIDATION_FAILURE',
                            logical_call_id=kwargs.get('logical_call_id')))
            return result
        return call

    def contract(self, original, stage):
        @functools.wraps(original)
        def call(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except ValueError as error:
                self.record(stage, dict(error_type=type(error).__name__, reason=str(error)))
                raise
        return call

@contextmanager
def stop_on_contract_failure(output):
    from recclaw_core.experiments.helix_abc_v1 import fresh_r1
    import research_actions
    import runtime_fix
    boundary = ContractBoundary(output)
    with ExitStack() as stack:
        stack.enter_context(patch.object(fresh_r1, 'bounded_provider_call', boundary.provider(fresh_r1.bounded_provider_call)))
        cls = research_actions.ResearchActionAdapter
        stack.enter_context(patch.object(cls, 'canonicalize_implementation_response', boundary.contract(cls.canonicalize_implementation_response, 'implementation_contract')))
        stack.enter_context(patch.object(runtime_fix, 'qualified_config', boundary.contract(runtime_fix.qualified_config, 'qualification_metadata')))
        yield boundary
