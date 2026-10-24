"""Recompute candidate/frontier effects and all-in costs from a native export."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def analyze(data):
    result = {'schema': 'recrsi.history-pair-analysis.v1', 'arms': {}}
    for arm, raw in data['arms'].items():
        stage = raw['stage']
        evaluations = [r for r in stage['evaluation']['results'] if r['genuine']]
        if len(evaluations) != 3 or not all(r['valid'] for r in evaluations):
            raise ValueError('Expected the complete three-valid-FIT stage')
        trajectory = raw['frontier']['effect_trajectory']
        if len(trajectory) != 3:
            raise ValueError('Missing scientific opportunity trajectory')
        values = [r['primary_metric'] for r in evaluations]
        initial = trajectory[0]['candidate_value'] - trajectory[0]['frontier_delta']
        curve = [initial]
        for value in values:
            curve.append(max(curve[-1], value))
        if not math.isclose(curve[-1], raw['frontier']['incumbent_ndcg@5']):
            raise ValueError('Recomputed frontier differs from native frontier')
        match = {m['api_call_id']: m for m in raw['used_call_matches']}
        costs = {'accepted_stage_calls': 0., 'stage_schema_failures': 0.,
                 'unmatched_success_calls': 0.}
        for event in raw['api_events']:
            if event['event'] != 'SUCCESS':
                continue
            price = stage['api']['model_prices_usd'][event['model']]
            cost = (event['input_tokens'] * price['input'] +
                    event['output_tokens'] * price['output']) / 1e6
            mapped = match.get(event['call_id'])
            category = ('unmatched_success_calls' if mapped is None else
                        'accepted_stage_calls' if mapped['native_status'] == 'SUCCESS'
                        else 'stage_schema_failures')
            costs[category] += cost
        if not math.isclose(sum(costs.values()), stage['api']['normalized_cost_usd']):
            raise ValueError('Cost partition differs from original ledger')
        initial_inputs = [b for b in raw['producer_boundaries'] if b['round'] == 13]
        if len(initial_inputs) != 4:
            raise ValueError('Missing initial producer inputs')
        result['arms'][arm] = {
            'candidate_values': values, 'candidate_mean': statistics.mean(values),
            'candidate_best': max(values), 'initial_incumbent': initial,
            'retained_frontier_curve': curve, 'frontier_gain': curve[-1] - initial,
            'source_hashes': [r['source_sha256'] for r in evaluations],
            'reference_api_cost_partition_usd': costs,
            'all_in_reference_api_usd': sum(costs.values()),
            'unresolved_reserved_usd': stage['api']['reserved_reference_usd'],
            'new_gpu_seconds': sum(j.get('gpu_seconds', 0.) for j in stage['gpu']['jobs'].values()),
            'initial_history_counts_by_role': {b['role']: b['context']['memory']['history_counts']
                                               for b in initial_inputs},
            'test_accessed': stage['test_accessed'],
        }
    left, right = (result['arms'][a] for a in ('retain', 'reset'))
    if not math.isclose(left['initial_incumbent'], right['initial_incumbent']):
        raise ValueError('Different starting incumbents')
    result['candidate_mean_difference'] = left['candidate_mean'] - right['candidate_mean']
    result['candidate_best_difference'] = left['candidate_best'] - right['candidate_best']
    result['initial_input_comparison'] = {}
    contexts = {arm: {b['role']: b['context'] for b in raw['producer_boundaries']
                      if b['round'] == 13} for arm, raw in data['arms'].items()}
    for role in contexts['retain']:
        a, b = contexts['retain'][role], contexts['reset'][role]
        result['initial_input_comparison'][role] = {
            'contract_equal': a['contract'] == b['contract'],
            'objective_equal': a['objective'] == b['objective'],
            'incumbent_equal': a['state']['incumbent'] == b['state']['incumbent'],
            'construction_parent_options_equal': a['state']['construction_parent_options'] ==
                                                  b['state']['construction_parent_options'],
            'reset_history_counts_zero': all(v == 0 for v in b['memory']['history_counts'].values()),
        }
    result['strict_history_only_initial_input_equivalence'] = all(
        all(checks.values()) for checks in result['initial_input_comparison'].values())
    result['scope'] = 'One historical state, one paired prospective search seed; descriptive DEV results, not independent replication or TEST.'
    result['cost_scope'] = 'Reference API prices, all attempts retained; GPU includes new qualification and training, excludes reused historical root.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_export', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = args.native_export.read_bytes()
    result = analyze(json.loads(raw.decode('utf-8-sig')))
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))
