"""Shared launch plumbing only; search state remains owned by each native method."""
import json
import multiprocessing
import os
from pathlib import Path
import sys

from graph_native_inputs import bind_imports
from graph_native_adapter import START, END


def inspection(inputs, method):
    launch = inputs['launch']
    return dict(status='INSPECTED_NO_PROVIDER_NO_WORKER', method=method,
                identity=inputs['evaluator']['run_identity'], native_python=inputs['native_python'],
                root_sha256=inputs['root_sha256'], input_hashes=inputs['hashes'],
                limits=dict(genuine=12, reference_usd=7, gpu_seconds=21600, active_seconds=86400),
                output=launch['output'], native_configuration_imported=False)


def initialize_runtime(inputs, *, marked=False):
    """Called under the existing launch lock, only for an explicit execution."""
    evaluator, launch = inputs['evaluator'], inputs['launch']
    bind_imports(evaluator)
    from active_wall import ActiveWallClock
    from graph_candidate_evaluator import resources_for
    from evaluation_ledger import EvaluationLedger
    ActiveWallClock.create(evaluator['active_wall_path'], evaluator['run_identity'])
    root = Path(launch['output']).resolve()
    (root / 'cache').mkdir(exist_ok=True)
    # Native Shinka imports create ~/docker_cache. Bind its actual cache home
    # before importing native modules, using this run's existing NAS output.
    os.environ.update(HOME=str(root), TMPDIR=str(root), XDG_CACHE_HOME=str(root / 'cache'),
                      E1_RQ1_PROVIDER_CONFIG=str(Path(evaluator['provider_config']).resolve()),
                      E1_RQ1_EVALUATOR_CONFIG=str(Path(launch['evaluator_config']).resolve()),
                      OPENAI_API_KEY='routed-by-frozen-main-transport')
    multiprocessing.set_start_method('spawn', force=False)
    os.sched_setaffinity(0, set(launch['cpus']))
    carrier = root / 'root.py'
    source = ((START + inputs['parent'] + ('\n' if not inputs['parent'].endswith('\n') else '') + END) if marked else inputs['parent']).encode('utf-8')
    if carrier.exists():
        if carrier.read_bytes() != source:
            raise ValueError('Persisted native root carrier changed; retain its original run')
    else:
        carrier.write_bytes(source)
    resources = resources_for(evaluator)
    ledger = EvaluationLedger(root / 'EVALUATION_LEDGER.json',
                              root_sha256=inputs['root_sha256'], identity=evaluator['run_identity'])
    from graph_candidate_evaluator import run_program
    run_program(evaluator)
    return root, carrier, resources, ledger


def require_execution_environment(inputs):
    # Resolving symlinks would hide which virtual environment is active.
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(inputs['native_python']):
        raise ValueError('Execute with the interpreter read from native_preparation')
    root = Path(inputs['launch']['output'])
    if (root / 'PAUSED.json').exists():
        raise RuntimeError('Original interruption is unresolved; reconcile its preserved records before resume')


def settle_result(inputs, result, resources, ledger):
    from durable_evaluation import write_json
    state = ledger.snapshot()
    api = resources.api.summary()
    result = dict(result, genuine_count=state['genuine_count'], common_winner=state['winner'],
                  api=api, test_accessed=False, input_hashes=inputs['hashes'])
    reason = str(result.get('reason', '')).lower()
    interrupted = (api.get('unheld_unknown_usage_failures') or api.get('fatal_provider_statuses')
                   or any(word in reason for word in
                          ('interrupted', 'unsettled', 'unresolved', 'external', 'transport')))
    if interrupted:
        result['status'] = 'PAUSED_REQUIRES_ORIGINAL_STATE_RECONCILIATION'
        target = 'PAUSED.json'
    else:
        target = 'RESULT.json'
    write_json(Path(inputs['launch']['output']) / target, result)
    return result


def print_result(inputs, result):
    print(json.dumps(dict(status=result['status'], genuine_count=result['genuine_count'],
                          output=inputs['launch']['output'])))
