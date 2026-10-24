"""Capture full native producer call arguments without credentials or network.

The snapshot producer and task adapter are executed; the paid provider boundary
is replaced by a capture callback. This is not a claim of HTTP transmission.
"""
import argparse
import copy
import hashlib
import json
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from native_pair_probe import run


def capture(home, scratch):
    pair = run(home)
    cfg = json.loads((home / 'EVALUATOR_CONFIG.json').read_text())
    sys.path.insert(0, str(Path(cfg['task_root']) / 'controller_adapter'))
    from morec_ro_execution import search_adapter
    import auto_search_space as native
    from recclaw_core.research_line import provider
    from recclaw_core.experiments.helix_abc_v1.campaign_runtime import campaign_development_validation_profile_ref
    from run_native_json import include_schema
    inputs = json.loads((home / 'INPUTS.json').read_text())
    space = native.load_search_space_package(Path(inputs['space']['package_root']),
        expected_source=native.SearchSpaceSource.AUTO_GENERATED,
        expected_digest=inputs['space']['digest'], require_frozen=False)
    adapter = search_adapter(cfg, space)
    state = pickle.loads((home / 'search/campaign/ROUND_13_CHECKPOINT.pkl').read_bytes()).state_before
    index = state.context.frontier['incumbent_round_index']
    roots = [Path(p) for p in state.candidate_root_by_capability.values()
             if f'round-{index:04d}' in Path(p).parts]
    if len(roots) != 1:
        raise ValueError('Ambiguous incumbent')
    sources = {name: (roots[0] / name).read_text() for name in
               ('main_code.py', 'models.py', 'diagnosis_tools.py',
                'self_evolverec_interface.py', 'morec_initialization.py')}
    appendix = '\nIdentical current incumbent source for this task:\n' + json.dumps(sources, sort_keys=True)
    captured = {}

    class Captured(BaseException):
        pass

    def call(**kwargs):
        kwargs = include_schema(kwargs)
        # All input strings and schema are retained, but credential paths/objects
        # are excluded from export. There are no credentials in mapping config.
        captured[key] = {k: v for k, v in kwargs.items()
                         if k in ('prompt', 'schema', 'model', 'requested_model',
                                  'max_output_tokens', 'output_token_ceiling',
                                  'reasoning_effort', 'wire_api', 'token_ceiling')}
        captured[key]['argument_names'] = sorted(kwargs)
        raise Captured()

    def deny(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
            raise RuntimeError('Capture cannot access network or execute workers')
    sys.addaudithook(deny)
    for arm, views in pair['views'].items():
        for role, view in views.items():
            key = f'{arm}/{role}'
            projected = copy.deepcopy(view)
            projected['objective']['space'] = adapter.task_context()
            producer = adapter.build_provider_research_producer(
                config_source={'source_kind': 'offline_capture'},
                call_root=scratch / arm / role,
                session_id='history-value-paired-capture', provider_call=call,
                proposal_seed=762001, maximum_physical_attempts=1,
                proposal_lane_by_role={r: provider.PROPOSAL_LANE_OPEN_SPEC for r in views},
                frozen_profile_ref=campaign_development_validation_profile_ref(),
                proposal_output_token_ceiling_total_per_context=20000,
                proposal_context_appendix=appendix)
            with patch.object(provider, 'project_provider_context_view', lambda c: projected):
                try:
                    producer(role, state.context.producer_view(role))
                except Captured:
                    pass
            if key not in captured:
                raise ValueError(f'Native callback was not reached: {key}')
    old = json.loads((home / 'PROVIDER.json').read_text())
    return {'stage': 'NATIVE_FULL_PRODUCER_ARGUMENTS_CAPTURED',
            'paid_calls': 0, 'http_sent': False, 'requests': captured,
            'historical_price_table': old['api_limits']['model_prices'],
            'historical_input_limit': old['api_limits']['max_input_tokens'],
            'incumbent_source_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
            'space_digest': space.digest}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='history-value-capture-') as scratch:
        print(json.dumps(capture(args.home, Path(scratch)), ensure_ascii=False, default=str))
