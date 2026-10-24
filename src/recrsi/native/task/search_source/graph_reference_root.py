"""Bind a measured graph root to one native search's original resource budget.

Shared input/accounting only; no generation, selection, Provider or worker call.
"""
import hashlib
import json
from pathlib import Path

from graph_program import primary_score


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_root(config, resources):
    from evaluation_ledger import EvaluationLedger
    source = Path(config['root_calibration'])
    if sha(source) != config['root_calibration_sha256']:
        raise ValueError('Referenced root calibration identity changed')
    envelope = read(source)
    if envelope['status'] != 'SUCCESS':
        raise ValueError('Calibration did not deliver an admitted complete root')
    result = envelope['result']
    original = read(config['root_binding'])
    binding = read(config['binding'])
    if sha(config['root_binding']) != result['binding_sha256']:
        raise ValueError('Calibration binding differs from the measured root')
    # Mapping becomes known after root; scientific training/data inputs cannot
    # change while retaining its score or cost. New settings need a new root.
    for key in ('adapter','task_id','line','dataset','root_source_sha256',
                'model_config','training','evaluation','metric_binding_version'):
        if binding.get(key) != original.get(key):
            raise ValueError('New scientific settings require a new measured root: '+key)
    # Relocation changes paths, not data identity. Dependency/source identities
    # are checked separately by the runtime closure.
    if ({k:v['sha256'] for k,v in binding['data_files'].items()} !=
            {k:v['sha256'] for k,v in original['data_files'].items()}
            or binding['dev_candidate_pool']['sha256'] != original['dev_candidate_pool']['sha256']):
        raise ValueError('New scientific settings require a new measured root: data')
    if (binding['status'] != 'FROZEN' or not binding.get('catalog_mapping_sha256')
            or binding['catalog_mapping_sha256'] != result['catalog_mapping_sha256']):
        raise ValueError('Search must bind the actual root catalog mapping')
    if (result['source_sha256'] != sha(config['parent_source'])
            or result['source_sha256'] != binding['root_source_sha256']
            or result['role'] != 'root' or result['train_seed'] != config['train_seed']
            or result['evaluation_seed'] != binding['evaluation']['seed']
            or not result['checkpoint_reload_verified']):
        raise ValueError('Measured root source/seed/checkpoint differs')
    score = primary_score(result, binding)
    physical = read(config['root_gpu_ledger'])
    from root_contract import validate_root_receipt
    measured = validate_root_receipt(envelope, physical)
    seconds = measured['gpu_seconds']
    carry = dict(source=str(source), source_sha256=sha(source), gpu_seconds=seconds,
        active_seconds=measured['active_seconds'], physical_execution_repeated=False)
    with resources.gpu.transaction() as state:
        if state.get('root_reference_cost', carry) != carry:
            raise ValueError('Existing root GPU reference charge differs')
        state['root_reference_cost'] = carry
    with resources.clock._locked():
        state = resources.clock._read()
        if state.get('root_reference_cost', carry) != carry:
            raise ValueError('Existing root active reference charge differs')
        state['root_reference_cost'] = carry
        resources.clock._write(state)
    ledger = EvaluationLedger(Path(config['output'])/'EVALUATION_LEDGER.json',
        root_sha256=result['source_sha256'], identity=config['run_identity'])
    baseline = dict(metrics={'combined_score':score, 'ndcg_at_5':score},
        artifacts=dict(valid=True, source_sha256=result['source_sha256'],
                       native_evidence=result, root_reference_cost=carry))
    recovered = ledger.recoverable(result['source_sha256'])
    if recovered is None:
        token = ledger.reserve(result['source_sha256'])
        admitted = resources.admit_output()
        ledger.ready(token, baseline, admitted=admitted)
    else:
        if recovered['result'] != baseline:
            raise ValueError('Original root delivery differs')
        token, admitted = recovered['token'], recovered['admitted']
    ledger.finish(token, valid=True, primary_metric=score, admitted=admitted, evidence=result)
    return baseline, ledger
