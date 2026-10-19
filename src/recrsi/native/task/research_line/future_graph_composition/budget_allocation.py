"""Executor-owned resource allocation, preserving authored estimates and real caps."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from unittest.mock import patch

VERSION = 'graph-executor-resource-allocation-v1'
GRANT = dict(implementation_token_ceiling=20000,
             qualification_gpu_minutes=10, qualification_wall_minutes=30)
NOTICE = '''\nRESOURCE CONTRACT (graph-executor-resource-allocation-v1):
The legacy resolution_facts.required_budget fields are advisory estimates, not
measured requirements or authority to allocate resources. Do not claim measured
runtime before execution. The executor allocates at most 20000 implementation
tokens; legacy qualification allocation is 10 GPU minutes / 30 wall minutes.
Actual worker limits are stricter: qualification 300 seconds and training 3600
seconds; cumulative 21600 GPU seconds, 86400 active seconds, 36 proposal/physical
safety limit and the existing API cap remain enforced. Estimates alone do not
reject an otherwise valid candidate; measured resource exhaustion remains real.
The schema is unchanged. Keep all dependency, protocol, source and parent facts
accurate. These resource rules apply equally to DISCOVER and REFINE.
'''


def allocated_resolution(native, spec, facts, environment):
    original = native(spec, resolution_facts=facts, environment=environment)
    # Retain malformed contracts, unknown dimensions, dependency and protocol
    # failures. Only the three observed estimate-only budget reasons change.
    allowed = {'BUDGET_EXCEEDED:' + key.upper() for key in GRANT}
    reasons = set(original.reason_codes)
    if not reasons or not reasons.issubset(allowed):
        return original, original, None
    if dict(environment['budget_limits']) != GRANT:
        raise ValueError('Executor allocation differs from pinned original limits')
    allocated = copy.deepcopy(facts)
    allocated['required_budget'] = {
        key: GRANT[key] for key in facts['required_budget']}
    corrected = native(spec, resolution_facts=allocated, environment=environment)
    return original, corrected, allocated['required_budget']


@contextmanager
def allocation_scope(output, evidence, amendment):
    from recclaw_core.experiments.helix_abc_v1 import vnext_orchestration as orchestration
    import independent_discovery
    if amendment['version'] != VERSION or amendment['first_new_prompt_round'] != 15:
        raise ValueError('Unexpected resource amendment')
    native = orchestration.resolve_capability
    install = independent_discovery.install_producer
    def current_round():
        return json.loads((Path(output)/'campaign/CAMPAIGN_STATE.json').read_bytes())['state']['next_round_index']
    def resolve(spec, *, resolution_facts, environment):
        before, after, grant = allocated_resolution(native, spec, resolution_facts, environment)
        if grant is not None:
            receipt = dict(version=VERSION, round=current_round(), spec_digest=spec.digest,
                authored_estimate=resolution_facts['required_budget'], executor_allocation=grant,
                original_reasons=list(before.reason_codes), result=after.canonical_dict(),
                original_facts_unchanged=True, actual_limits_changed=False)
            directory=Path(evidence)/'allocations'; directory.mkdir(exist_ok=True)
            target=directory/(spec.digest+'.json')
            if not target.exists():
                target.write_text(json.dumps(receipt,sort_keys=True,indent=2),encoding='utf8')
        return after
    def install_with_contract(producer, authority_notice):
        install(producer, authority_notice)
        call, provider = producer._call, producer.provider_call
        def request(role, view, **options):
            if current_round() < amendment['first_new_prompt_round']:
                return call(role, view, **options)
            with patch.object(producer, 'call_root', producer.call_root / VERSION):
                return call(role, view, **options)
        def deliver(**options):
            if current_round() >= amendment['first_new_prompt_round']:
                options['prompt'] += NOTICE
            return provider(**options)
        producer._call, producer.provider_call = request, deliver
        return producer
    with patch.object(orchestration,'resolve_capability',resolve), patch.object(
            independent_discovery,'install_producer',install_with_contract):
        yield

