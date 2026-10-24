"""Recompute complete-state contrasts and connect supplied decisions to outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--original-csv', type=Path, required=True)
p.add_argument('--decision-csv', type=Path, required=True)
p.add_argument('--native-root', type=Path, required=True)
p.add_argument('--recovered-receipts', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
cells = rows(a.original_csv)
contrasts = []
for state in (2, 3, 18, 6, 10, 8, 12):
    arms = {r['feedback']:r for r in cells if int(r['state']) == state}
    s, m = arms['Summary'], arms['Mechanism']
    contrasts.append({'state':state,'batch':s['batch'],'summary_action':s['action'],
        'mechanism_action':m['action'],'delta':float(m['ndcg5'])-float(s['ndcg5']),
        'action_label_differs':s['action'] != m['action']})
deltas = [r['delta'] for r in contrasts]
chains = []
for r in rows(a.decision_csv):
    original = next(c for c in cells if c['state'] == r['state'] and c['feedback'] == r['arm'])
    assert float(original['ndcg5']) == float(r['result.ndcg5'])
    marker = a.native_root.name + '/'
    path = a.native_root / r['result.terminal_result_path'].split(marker,1)[1]
    if not path.is_file() and a.recovered_receipts:
        path = a.recovered_receipts / f"state_{r['state']}_{r['arm']}.json"
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual is not None:
        assert actual == r['result.terminal_result_sha256'], path
    chains.append({'state':int(r['state']),'arm':r['arm'],
        'feedback':r['short_steps.feedback_gist'],'decision':r['short_steps.native_decision'],
        'source_change':r['short_steps.actual_source_change'],
        'candidate_ndcg5':float(r['result.ndcg5']),
        'retained_frontier_ndcg5':float(r['result.retained_frontier_ndcg5']),
        'exceeds_retained_frontier':float(r['result.ndcg5']) > float(r['result.retained_frontier_ndcg5']),
        'terminal_receipt_local_path':str(path),'terminal_receipt_verified':actual is not None,
        'terminal_receipt_sha256':r['result.terminal_result_sha256'],
        'source_sha256':r['source_sha256'],'worker_result_sha256':r['worker_result_sha256']})
result = {'evidence_class':'offline analysis of original observations',
    'state_contrasts':contrasts,'mean_signed_ndcg5_delta':mean(deltas),'median_signed_ndcg5_delta':median(deltas),
    'leave_one_state_out':[{'omitted_state':r['state'],'mean_signed_delta':mean(x['delta'] for x in contrasts if x is not r)} for r in contrasts],
    'batch_means':{b:mean(r['delta'] for r in contrasts if r['batch']==b) for b in dict.fromkeys(r['batch'] for r in contrasts)},
    'different_action_labels':sum(r['action_label_differs'] for r in contrasts),
    'decision_source_result_chains':chains,
    'interpretation_boundaries':['Action labels are coarse; different labels are not a causal mediation test.',
        'Leave-one-out uses all omissions, not selection of a favorable subset.',
        'Next-candidate improvements and retained-frontier improvements are different outcomes.',
        'Unverified local terminal receipts require retrieval before claiming native chain verification.'],
    'input_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.original_csv,a.decision_csv)},
    'new_calls':0,'new_fits':0,'new_cost_usd':0}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('mean_signed_ndcg5_delta','median_signed_ndcg5_delta','batch_means','different_action_labels')}))
print(json.dumps({'native_receipts_verified':sum(r['terminal_receipt_verified'] for r in chains),
    'leave_one_out_range':[min(r['mean_signed_delta'] for r in result['leave_one_state_out']),max(r['mean_signed_delta'] for r in result['leave_one_state_out'])],
    'differential_frontier_updates':sum(r['exceeds_retained_frontier'] for r in chains)}))
