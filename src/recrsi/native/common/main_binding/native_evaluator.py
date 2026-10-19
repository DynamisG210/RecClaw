"""BERT evaluation on the common fixed task; no AutoSpace dependency for RH."""
import hashlib
import json
import math
from pathlib import Path
import sys


def create_evaluator(config):
    sys.path[:0] = [str(Path(__file__).resolve().parent), config['comparison_root'], str(Path(config['comparison_root']).parent)]
    from active_wall import ActiveWallClock
    from main_resources import MainResources
    from openevolve_provider import _settings_and_ledger
    from bert_worker_runtime import BertRuntime
    from evaluation_ledger import EvaluationLedger, MainResultConsumer
    from durable_evaluation import write_json
    from program_evaluator import ProgramEvaluator
    root = Path(config['output'])
    ActiveWallClock.create(config['active_wall_path'], config['run_identity'])
    settings, api = _settings_and_ledger(config['provider_config'])
    if settings['run_identity'] != config['run_identity'] or Path(settings['resource_root']) != root:
        raise ValueError('BERT provider resources differ')
    resources = MainResources(root, config['run_identity'], config['active_wall_path'], api)
    source = Path(config['root_result_source'])
    from root_equivalence import reference_root
    baseline, equivalence = reference_root(config)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    expected = dict(source_sha256=sha(config['parent_source']), seed=config['train_seed'],
        recipe_sha256=sha(config['training_config']), manifest_sha256=sha(config['search_manifest']))
    if not all(math.isfinite(baseline[k]) and 0 <= baseline[k] <= 1 for k in ('ndcg_at_5', 'recall_at_5')):
        raise ValueError('Root lacks valid NDCG@5/Recall@5')
    physical = json.loads(Path(config['root_gpu_ledger']).read_text())
    if any(not job['finished_ns'] for job in physical['jobs'].values()):
        raise ValueError('Referenced root physical cost remains unresolved')
    carry = dict(source=str(source), source_sha256=sha(source),
        gpu_seconds=sum(j['gpu_seconds'] for j in physical['jobs'].values()),
        active_seconds=baseline['wall_time_secs'], physical_execution_repeated=False)
    with resources.gpu.transaction() as state:
        if state.get('root_reference_cost', carry) != carry:
            raise ValueError('Referenced root GPU charge changed')
        state['root_reference_cost'] = carry
    with resources.clock._locked():
        state = resources.clock._read()
        if state.get('root_reference_cost', carry) != carry:
            raise ValueError('Referenced root active charge changed')
        state['root_reference_cost'] = carry
        resources.clock._write(state)
    envelope = dict(identity=dict(source_sha256=expected['source_sha256'], training_seed=config['train_seed'],
        epochs=config['epochs'], protocol_digest=config['protocol_digest'], manifest_sha256=expected['manifest_sha256']),
        outcome=baseline, reused_from=str(source), reference_cost=carry,
        default_equivalence=equivalence)
    if (root/'ROOT_RESULT.json').exists():
        if json.loads((root/'ROOT_RESULT.json').read_text()) != envelope:
            raise ValueError('Prepared root envelope changed')
    else:
        write_json(root/'ROOT_RESULT.json', envelope)
    runtime = BertRuntime(config, resources, baseline)
    ledger = EvaluationLedger(root/'EVALUATION_LEDGER.json', root_sha256=runtime.parent_sha, identity=config['run_identity'])
    if not any(row['root'] for row in ledger.snapshot()['results']):
        pending = [(token, row) for token, row in ledger.snapshot()['pending'].items()
                   if row['source_sha256'] == runtime.parent_sha]
        if len(pending) > 1:
            raise ValueError('Multiple unresolved root reservations')
        token = pending[0][0] if pending else ledger.reserve(runtime.parent_sha)
        ledger.finish(token, valid=True, primary_metric=baseline['ndcg_at_5'], admitted=resources.admit_output(), evidence=baseline)
    return MainResultConsumer(ProgramEvaluator(runtime, worker_timeout_seconds=config['worker_timeout_seconds']),
        ledger, resources.admit_output, resources.require_capacity)
