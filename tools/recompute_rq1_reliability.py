"""Recompute RQ1 search-level statistics without promoting snapshot links to verified TEST provenance."""
import argparse
import collections
import hashlib
import json
import statistics
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def identity(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--paper', type=Path, required=True)
    p.add_argument('--bert-observed', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source = a.legacy / 'outputs/rq1_api_gain_cost_20261020/seven_groups/analysis_with_gpu.json'
    equal = a.paper / 'data/rq1_equal_api_20261023.json'
    table_path = a.paper / 'data/rq1_supplied_20261018.json'
    protocol_path = a.paper / 'data/cohort_protocol_20261023.json'
    data, table, protocol = read(source), read(table_path), read(protocol_path)
    repeats = protocol['cohort']['bert_electronics_source_repeats']
    raw = data['searches']
    retained = [r for r in raw if r['group'] != 'BERT_Electronics' or r['repeat'] in repeats]
    excluded = [r for r in raw if r not in retained]
    keys = [(r['group'], r['method'], r['repeat']) for r in retained]
    assert len(keys) == len(set(keys)), 'Duplicate search identity'
    grouped = collections.defaultdict(list)
    for r in retained:
        grouped[(r['group'], r['method'])].append(r)
    stats = []
    for (group, method), rows in sorted(grouped.items()):
        stats.append({'group': group, 'method': method, 'n_search_records': len(rows),
                      'source_repeats': [r['repeat'] for r in rows],
                      'reported_final_evaluations_per_search': [r['n_final_fits'] for r in rows],
                      'ndcg5_mean': statistics.mean(r['ndcg5'] for r in rows),
                      'ndcg5_sample_sd': statistics.stdev(r['ndcg5'] for r in rows) if len(rows)>1 else None,
                      'hr5_mean': statistics.mean(r['hr5'] for r in rows),
                      'hr5_sample_sd': statistics.stdev(r['hr5'] for r in rows) if len(rows)>1 else None,
                      'reference_api_usd_sum': sum(r['api_usd'] for r in rows),
                      'unknown_excluded_count': sum(r.get('unknown_excluded', 0) or 0 for r in rows),
                      'remote_excluded_count': sum(r.get('remote_excluded', 0) or 0 for r in rows),
                      'evidence_class': 'recomputed_from_search_summary; not independently verified fit-level chain'})
    observed = read(a.bert_observed)
    provenance = table['embedded_provenance'][0]['content']
    assert identity(a.bert_observed)['sha256'] == provenance['source']['sha256']
    bert_rows = []
    for r in observed['rows']:
        s, t = r['selection'], r['TEST']
        bert_rows.append({'search_id': r['unit'], 'selection_rule': s['rule'],
                         'selected_source_sha256': s['source_sha256'],
                         'native_ndcg_winner_sha256': s['native_raw_ndcg_winner_unchanged'],
                         'matches_native_ndcg_winner': s['source_sha256'] == s['native_raw_ndcg_winner_unchanged'],
                         'raw_ledger_sha256': s['raw_ledger_sha256'],
                         'dev_ndcg5': s['raw_ndcg_at_5'], 'test': t,
                         'historical_test_exposure': s['historical_TEST_exposure'],
                         'test_feedback_to_search_reported': s['TEST_feedback_to_search'],
                         'checkpoint_test_reused': r['final_TEST_reused'],
                         'test_gpu_seconds': r['final_TEST_gpu_seconds'],
                         'unknown_usage_failures': r['unknown_usage_failures'],
                         'evidence_class': 'hash-matched adopted OBSERVED snapshot; TEST file/source binding not rehashed remotely'})
    bert_stats = {m: {'mean': statistics.mean(r['test']['metrics'][m] for r in bert_rows),
                       'sample_sd': statistics.stdev(r['test']['metrics'][m] for r in bert_rows)}
                  for m in ['ndcg_at_5', 'recall_at_5']}
    values = table['table_state']['values']
    assert f"{bert_stats['ndcg_at_5']['mean']:.4f}" == values['d3|m0|g0|s6']
    assert f"{bert_stats['recall_at_5']['mean']:.4f}" == values['d3|m1|g0|s6']
    model_ids = {'MoRec':'g1','BERT':'g0','NGCF':'g2','SASRec':'g3'}
    dataset_ids = {'CDs':'d0','Electronics':'d1','Office':'d2','ML1M':'d3'}
    method_ids = {'rh':'s1','deep':'s2','openevolve':'s3','shinka':'s4','se':'s5','ro':'s6'}
    tex_path = a.paper / 'tables/RQ1_main_table.tex'
    current_values = {}
    tex_model_ids = {**model_ids, 'BERT4Rec':'g0'}
    tex_methods = {'Seed':'s0','RecHarness':'s1','DeepEvolve':'s2','OpenEvolve':'s3',
                   'ShinkaEvolve':'s4','Self-EvolveRec':'s5','RecRSI (Ours)':'s6'}
    group_id = None
    for line in tex_path.read_text(encoding='utf-8').splitlines():
        match = re.search(r'\\multirow\{7\}\{\*\}\{([^}]+)\}', line)
        if match:
            group_id = tex_model_ids[match.group(1)]
        parts = line.split('&')
        if len(parts) != 10 or parts[1].strip() not in tex_methods:
            continue
        method_id = tex_methods[parts[1].strip()]
        for i, part in enumerate(parts[2:]):
            nums = re.findall(r'0\.\d{4}', part)
            assert len(nums) == 1, line
            current_values[f'd{i//2}|m{i%2}|{group_id}|{method_id}'] = nums[0]
    assert len(current_values) == 224
    stale_cells = [{'cell':k,'legacy_json':v,'current_tex':current_values[k]}
                   for k,v in values.items() if current_values[k] != v]
    comparisons = []
    for row in stats:
        model, dataset = row['group'].split('_')
        for metric, mid in [('ndcg5','m0'), ('hr5','m1')]:
            key = '|'.join([dataset_ids[dataset],mid,model_ids[model],method_ids[row['method']]])
            actual = format(row[metric+'_mean'], '.4f')
            comparisons.append({'cell':key,'group':row['group'],'method':row['method'],
                                'metric':metric,'recomputed':actual,'paper':current_values[key],
                                'matches':actual == current_values[key]})
    report = {'inputs': [identity(x) for x in [source,equal,table_path,protocol_path,tex_path,a.bert_observed]],
              'input_search_records': len(raw), 'retained_search_records': len(retained),
              'excluded_records': [{'group':r['group'],'method':r['method'],'repeat':r['repeat']} for r in excluded],
              'selection_basis': 'Author-adopted BERT source repeats from cohort_protocol; no quality-based selection here',
              'stale_embedded_table_cells':stale_cells, 'statistics': stats, 'paper_cell_comparisons': comparisons,
              'search_inventory': [{k:r.get(k) for k in ['group','method','repeat','ndcg5','hr5','n_final_fits',
                 'cost_source','cost_source_sha256','api_usd','remote_excluded','unknown_excluded',
                 'unclosed_reservations','gpu_source','gpu_job_seconds','gpu_root_reference_seconds',
                 'unknown_gpu_job_ids','note']} for r in retained], 'bert_ml1m_ro_mapping': bert_rows, 'bert_ml1m_ro_statistics': bert_stats,
              'cost_scope': {'gpu': data['gpu_policy'], 'equal_api': read(equal)['cost'], 'actual_total_cost': 'UNKNOWN; exclusions cannot be treated as zero'},
              'remaining_gaps': ['Nine of sixteen main-table settings, including SASRec, absent from this seven-task snapshot',
                                 'Summary final-evaluation counts do not verify source/checkpoint/evaluator/seed bindings',
                                 'BERT ML1M has one checkpoint-only TEST per search, not three refits',
                                 'BERT ML1M composite selection rule and historical TEST exposure require explicit disclosure',
                                 'Remote failures/UNKNOWN and final FIT/TEST cost excluded from this search-cost estimand'],
              'new_training': 0, 'new_test_evaluations': 0, 'new_api_calls': 0}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'RQ1_RECOMPUTED.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'input':len(raw),'retained':len(retained),'task_method_groups':len(stats),'bert':bert_stats,
                      'paper_cells_matched':sum(r['matches'] for r in comparisons), 'paper_cells_total':len(comparisons), 'native_selection_mismatches':[r['search_id'] for r in bert_rows if not r['matches_native_ndcg_winner']]}))

if __name__ == '__main__':
    main()
