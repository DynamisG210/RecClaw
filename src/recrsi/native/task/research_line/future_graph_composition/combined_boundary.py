"""Prospective graph-only composition boundary; never a campaign launcher."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from unittest.mock import patch

VERSION = 'graph-native-boundary-score-choice-replay-v1'
DISCOVERY_SCORE_GUIDANCE = (
    ' For LEARNED, score_parameter_targets must be nonempty; for FIXED it must '
    'be empty. Do not label a declaration FIXED while listing learned parameter '
    'targets. A parameter-free transformation does not make listed trainable '
    'scoring parameters fixed. UNCHANGED cannot establish equality to hidden '
    'root code and the adapter emits no scoring-contract entries for UNCHANGED; '
    'listing targets under it does not perform a declared-leaf connectivity check.'
)
INNER_BOUNDARY = "with graph_materializer_abi(),patch.object(LabApiCanaryBrokerV1,'sealed_request_identity',identity):"


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'BUNDLE.json').read_bytes())
    for relative, expected in manifest['files'].items():
        actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Prospective graph bundle changed: ' + relative)
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    return VERSION + '-' + hashlib.sha256(encoded).hexdigest()[:20]


def install_producer(producer, template, namespace):
    """Called after the original independent producer wrapper has been built."""
    producer.proposal_template = template
    native_call = producer._call

    def call(*args, **kwargs):
        with patch.object(producer, 'call_root', producer.call_root / namespace):
            return native_call(*args, **kwargs)

    producer._call = call
    return producer


@contextmanager
def combined_scope(api_ledger, directory):
    """Inside transport + JSON identity, with independent_scope already active."""
    import independent_discovery
    import program_adapter
    import transport_durability
    from recclaw_core.experiments.helix_abc_v1 import lab_api_broker
    from durable_broker_replay import durable_broker_scope
    from choice_guidance import graph_program_choice_scope

    namespace = verify_bundle(directory)
    template = (Path(directory) / 'program_producer.txt').read_text(encoding='utf8')
    base = program_adapter.ProgramAdapter.build_provider_research_producer
    if base.__module__ != independent_discovery.__name__:
        raise ValueError('Enter original independent_scope before combined graph scope')

    def build(adapter, **kwargs):
        wrapped = base(adapter, **kwargs)
        install_producer(wrapped._producer, template, namespace)
        return wrapped

    # DISCOVER replaces the rendered prefix, so changing only the template
    # silently drops its score correction. Preserve its existing input policy.
    discovery = independent_discovery.DISCOVERY_INSTRUCTION + DISCOVERY_SCORE_GUIDANCE
    with patch.object(program_adapter.ProgramAdapter, 'build_provider_research_producer', build), \
         patch.object(independent_discovery, 'DISCOVERY_INSTRUCTION', discovery), \
         graph_program_choice_scope(), \
         durable_broker_scope(lab_api_broker, transport_durability, api_ledger):
        yield namespace


def bind_native_execute(execute, scope_factory):
    """Add scope at the actual native inner boundary, leaving the body intact.

    The future launcher must supply its own version-pinned baseline execute and
    original limits. This helper neither loads nor resumes any old campaign.
    """
    source = textwrap.dedent(inspect.getsource(execute))
    if source.count(INNER_BOUNDARY) != 1:
        raise ValueError('Review changed native graph composition boundary')
    changed = source.replace(INNER_BOUNDARY, INNER_BOUNDARY[:-1] + ',_future_graph_scope():')
    namespace = dict(execute.__globals__, _future_graph_scope=scope_factory)
    exec(compile(changed, inspect.getsourcefile(execute), 'exec'), namespace)
    return namespace[execute.__name__]
