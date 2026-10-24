"""Reconcile frozen and continued RQ1 records using hash-identified archives."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--archives', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = [r for r in json.loads(a.comparison.read_text())['rows']
            if r['status'] == 'CHANGED_SINCE_FREEZE']
    wanted = {r[k] for r in rows for k in ['expected', 'actual']}
    names = {Path(r['remote_path']).name for r in rows}
    contents, locations = {}, {}
    for archive in sorted(a.archives.glob('*_download/*.tar.gz')):
        with tarfile.open(archive) as t:
            for member in t:
                if not member.isfile() or Path(member.name).name not in names:
                    continue
                b = t.extractfile(member).read()
                h = sha(b)
                if h in wanted:
                    contents[h] = b
                    locations.setdefault(h, []).append({'archive': str(archive), 'member': member.name})
    reports = []
    for r in rows:
        item = {**r, 'old_locations': locations.get(r['expected'], []),
                'new_locations': locations.get(r['actual'], [])}
        if not item['old_locations'] or not item['new_locations']:
            item['reconciliation'] = 'ARCHIVE_MISSING'
            reports.append(item)
            continue
        old, new = contents[r['expected']], contents[r['actual']]
        name = Path(r['remote_path']).name
        if name.endswith('.pkl'):
            item['reconciliation'] = 'BOTH_SERIALIZED_STATES_RECOVERED_NOT_EXECUTED'
        elif name.endswith('.jsonl'):
            x, y = ([json.loads(z) for z in b.splitlines()] for b in [old, new])
            item.update(old_events=len(x), new_events=len(y), old_bytes_prefix=new.startswith(old),
                        old_events_prefix=y[:len(x)] == x,
                        original_last_timestamp_ns=max(z.get('timestamp_ns', 0) for z in x),
                        added_first_timestamp_ns=min((z.get('timestamp_ns', 0) for z in y[len(x):]), default=None))
            item['reconciliation'] = 'APPENDED_EVENTS' if item['old_events_prefix'] else 'EVENTS_CHANGED'
        else:
            x, y = json.loads(old), json.loads(new)
            item['changed_top_level_fields'] = [k for k in sorted(x.keys() | y.keys()) if x.get(k) != y.get(k)]
            item['reconciliation'] = 'BOTH_SNAPSHOTS_RECOVERED'
            if name == 'EVALUATION_LEDGER.json':
                item.update(old_genuine=x['genuine_count'], new_genuine=y['genuine_count'],
                            old_results_prefix=y['results'][:len(x['results'])] == x['results'],
                            contract_unchanged=x['contract'] == y['contract'])
                item['reconciliation'] = 'APPENDED_EVALUATIONS' if item['old_results_prefix'] else 'EVALUATIONS_CHANGED'
            elif name == 'RESULT.json':
                keys = ['normalized_cost_usd', 'raw_reported_reference_usd', 'unknown_usage_failures',
                        'unresolved_reference_hold_usd', 'requests']
                item.update(winner_unchanged=x['common_winner'] == y['common_winner'],
                            original_cost={k:x['api'].get(k) for k in keys},
                            continued_cost={k:y['api'].get(k) for k in keys})
            elif name == 'BALANCED_DEV_FREEZE87.json':
                keys = ['rule', 'source_sha256', 'selection_key', 'raw_ndcg_at_5', 'raw_hr_at_5', 'TEST_used']
                item.update(selection_unchanged=all(x.get(k) == y.get(k) for k in keys),
                            original_selection={k:x.get(k) for k in keys},
                            original_frozen_ns=x['frozen_ns'], continued_frozen_ns=y['frozen_ns'])
            elif name == 'GPU_JOB_BUDGET.json':
                item.update(old_jobs=len(x['jobs']), new_jobs=len(y['jobs']),
                            old_jobs_unchanged=all(y['jobs'].get(k)==v for k,v in x['jobs'].items()),
                            old_metered_gpu_seconds=sum(v.get('gpu_seconds',0) or 0 for v in x['jobs'].values()),
                            new_metered_gpu_seconds=sum(v.get('gpu_seconds',0) or 0 for v in y['jobs'].values()),
                            root_reference_unchanged=x.get('root_reference_cost')==y.get('root_reference_cost'))
            elif name == 'ACTIVE_WALL.json':
                item['original_exclusions_prefix'] = y['exclusions'][:len(x['exclusions'])] == x['exclusions']
                item['original_snapshot'] = x
                item['continued_snapshot'] = y
            elif name == 'CAMPAIGN_STATE.json':
                item['old_active_campaign'] = x['state']['active_profile']['campaign_id']
                item['new_active_campaign'] = y['state']['active_profile']['campaign_id']
        reports.append(item)
    out = {'input_sha256':sha(a.comparison.read_bytes()), 'records':reports,
           'scope':'Historical snapshots and continuation; costs are reported reference USD, not invoices',
           'new_api_calls':0, 'new_fit':0, 'new_test':0}
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'recovered_pairs':sum(bool(r['old_locations'] and r['new_locations']) for r in reports),
                      'records':len(reports), 'checks':[{k:r[k] for k in ['search_id','reconciliation','old_results_prefix',
                      'old_events_prefix','winner_unchanged','selection_unchanged','old_jobs_unchanged'] if k in r} for r in reports]},indent=2))


if __name__ == '__main__':
    main()
