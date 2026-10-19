"""Prospective graph composition. No release, launcher, or historical recovery."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from unittest.mock import patch

VERSION = 'graph-action-aware-composition-v1'


def namespace(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'CONFIG.json').read_bytes())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Composition component changed: ' + name)
    return VERSION + '-' + hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


@contextmanager
def fresh_allocation_scope(evidence):
    """Apply the verified resolver from the first call, with no old round gate."""
    from recclaw_core.experiments.helix_abc_v1 import vnext_orchestration as orchestration
    import independent_discovery
    from budget_allocation import allocated_resolution, NOTICE, VERSION as allocation_version
    native, install = orchestration.resolve_capability, independent_discovery.install_producer

    def resolve(spec, *, resolution_facts, environment):
        before, after, grant = allocated_resolution(native, spec, resolution_facts, environment)
        if grant is not None:
            receipt = dict(version=allocation_version, spec_digest=spec.digest,
                authored_estimate=resolution_facts['required_budget'], executor_allocation=grant,
                original_reasons=list(before.reason_codes), result=after.canonical_dict(),
                original_facts_unchanged=True, actual_limits_changed=False)
            directory = Path(evidence) / 'allocations'
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / (spec.digest + '.json')
            if target.exists():
                if json.loads(target.read_bytes()) != receipt:
                    raise ValueError('Allocation receipt changed on reentry')
            else:
                target.write_text(json.dumps(receipt, sort_keys=True), encoding='utf8')
        return after

    def install_with_contract(producer, authority_notice):
        install(producer, authority_notice)
        provider = producer.provider_call
        def deliver(**options):
            return provider(**{**options, 'prompt': options['prompt'] + NOTICE})
        producer.provider_call = deliver
        return producer

    with patch.object(orchestration, 'resolve_capability', resolve), patch.object(
            independent_discovery, 'install_producer', install_with_contract):
        yield


@contextmanager
def outer_scope(directory, evidence, binding=None):
    import branch_search
    from independent_discovery import independent_scope
    from action_aware_branch import action_aware_scope
    from continuation_binding import route_new_evaluators
    from pilot_limits import pilot_scope
    from abi_guidance import abi_scope
    # Factory must be changed before branch_scope builds the concrete adapter.
    namespace(directory)
    with action_aware_scope(branch_search), fresh_allocation_scope(evidence), abi_scope(binding), \
            independent_scope(), pilot_scope(), \
            route_new_evaluators(Path(directory) / 'evaluator_entry.py'):
        yield


@contextmanager
def inner_scope(directory, api_ledger):
    import program_adapter
    from combined_boundary import combined_scope, install_producer
    # The full identity includes branch policy and allocation as well as prompts.
    full_namespace = namespace(directory)
    with combined_scope(api_ledger, directory):
        base = program_adapter.ProgramAdapter.build_provider_research_producer
        def build(adapter, **kwargs):
            wrapped = base(adapter, **kwargs)
            producer = wrapped._producer
            install_producer(producer, producer.proposal_template, full_namespace)
            return wrapped
        with patch.object(program_adapter.ProgramAdapter, 'build_provider_research_producer', build):
            yield


def bind_execute(execute, directory, api_ledger, evidence, *, outer=True):
    """Compose the original execute body, retaining native provider/JSON nesting.

    Caller still owns fresh inputs, full source pins, resources, root preparation,
    qualification metadata and physical locks. This function does not supply them.
    Historical corrected-parent views are intentionally absent: fresh normal
    parents already share the native archived view; no recovered rows are imported.
    """
    from combined_boundary import INNER_BOUNDARY
    from pilot_limits import bounded
    directory = Path(directory)
    namespace(directory)
    source = textwrap.dedent(inspect.getsource(execute))
    replacements = {
        INNER_BOUNDARY: INNER_BOUNDARY[:-1] + ',_full_inner_scope():',
        'total_api_budget_usd=7': 'total_api_budget_usd=7',
        'max(0,7-api': 'max(0,7-api',
        'if path.exists(): budget=json.loads(path.read_bytes())':
            "if path.exists(): budget={key:json.loads(path.read_bytes())[key] for key in ('total_api_budget_usd','total_recorded_spent_usd','remaining_api_budget_usd','scope')}",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('Review changed original execute boundary: ' + old)
        source = source.replace(old, new)
    scope = dict(execute.__globals__, _full_inner_scope=lambda: inner_scope(directory, api_ledger))
    original_config = scope['research_config']
    scope['research_config'] = lambda inputs: bounded(original_config(inputs))
    exec(compile(source, str(directory / 'composition.py'), 'exec'), scope)
    native = scope[execute.__name__]
    if not outer:
        return native
    def run(inputs, resources):
        with outer_scope(directory, evidence, inputs["binding"]):
            return native(inputs, resources)
    return run
