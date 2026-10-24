"""Bind isolated arm configs to an assigned finite allocation; no execution."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def prepare(old, allocation_path, output, remote, credential):
    allocation = read(allocation_path)
    if allocation['allocation_id'] not in ('history_value_same_incumbent_pair_20261024_v1',
                                         'history_value_same_incumbent_pair_20261024_v2',
                                         'history_value_same_incumbent_v2_round14_20261024_phase2'):
        raise ValueError('Wrong allocation')
    seed = allocation.get('search_seed', 762001)
    prefix = allocation.get('campaign_prefix', 'history-value-b01-after12')
    if allocation['allocation_id'] != 'history_value_same_incumbent_pair_20261024_v1' and (seed != 762002 or prefix != 'history-value-b01-after12-s762002'):
        raise ValueError('V2 allocation must bind the preregistered next seed and new identity')
    raw_allocation = allocation_path.read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'ALLOCATION.json').write_bytes(raw_allocation)
    for arm in ('retain', 'reset'):
        root = remote / arm / 'search'
        identity = dict(run=prefix + '-' + arm,
                        phase='HISTORY_VALUE_SAME_INCUMBENT', method='ro', arm=arm,
                        block=1, search_seed=seed, train_seed=51001)
        provider = read(old / 'PROVIDER.json')
        limits = dict(provider['api_limits'], max_normalized_u=allocation['api_usd_cap_per_arm']/14)
        provider.update(resource_root=str(root), api_ledger=str(root/'API_EVENTS.jsonl'),
            active_wall_path=str(root/'ACTIVE_WALL.json'), run_identity=identity,
            api_config=credential, model_api_configs={m: credential for m in limits['model_prices']},
            api_limits=limits, ledger_module_dir=str(remote/'runtime'))
        budget = dict(schema='recrsi.history-value-budget.v1', consumer=identity,
            allocation_path=str(remote/'ALLOCATION.json'),
            allocation_sha256=hashlib.sha256(raw_allocation).hexdigest(), output=str(root),
            caps=dict(genuine=allocation['max_valid_candidate_fits_per_arm'],
                      reference_api_usd=allocation['api_usd_cap_per_arm'],
                      execution_seconds=allocation['gpu_seconds_cap_per_arm'],
                      active_seconds=allocation['queue_admission_window_seconds']),
            original_api_limits=read(old/'BUDGET.json')['original_api_limits'],
            space_reference=None, physical_limits=dict(training=6, qualification=12))
        save(output/arm/'BUDGET.json', budget)
        cfg = read(old/'EVALUATOR_CONFIG.json')
        cfg.update(output=str(root), main_binding=str(remote/'runtime'), run_identity=identity, cpus=[], gpu_uuid='ADMISSION_REQUIRED',
            provider_config=str(remote/arm/'PROVIDER.json'),
            active_wall_path=str(root/'ACTIVE_WALL.json'),
            e2_budget_manifest=str(remote/arm/'BUDGET.json'),
            e2_budget_sha256=hashlib.sha256((output/arm/'BUDGET.json').read_bytes()).hexdigest(),
            worker_timeout_seconds=allocation['fit_wall_seconds_cap'])
        launch = read(old/'LAUNCH.json')
        launch.update(campaign_id=identity['run'], output=str(root), cpus=[],
            search_seed=identity['search_seed'], purpose=identity['phase'],
            evaluator_config=str(remote/arm/'EVALUATOR_CONFIG.json'))
        save(output/arm/'PROVIDER.json', provider)
        save(output/arm/'EVALUATOR_CONFIG.json', cfg)
        save(output/arm/'LAUNCH.json', launch)
    return {'stage': 'FINITE_CONFIGS_BOUND_NOT_LAUNCHED', 'endpoint': allocation['endpoint'],
            'api_cap_per_arm': allocation['api_usd_cap_per_arm'],
            'gpu_seconds_per_arm': allocation['gpu_seconds_cap_per_arm'],
            'search_seed': seed, 'training_seed': 51001,
            'physical_admission': 'unbound; no GPU or CPU assigned', 'paid_calls': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('old', type=Path)
    p.add_argument('allocation', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('remote', type=PurePosixPath)
    p.add_argument('credential')
    a = p.parse_args()
    receipt = prepare(a.old, a.allocation, a.output, a.remote, a.credential)
    save(a.output/'PREPARED.json', receipt)
    print(json.dumps(receipt))
