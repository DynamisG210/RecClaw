"""Join original NGCF DEV-selected sources to all indexed CDs/Office TEST refits."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for key in ('selection', 'tests', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    selections = json.loads(a.selection.read_text())['rows']
    tests = json.loads(a.tests.read_text())['test_results']
    rows = []
    for s in selections:
        assert s['source_matches'] and s['value_matches']
        for task in ('cds', 'office'):
            prefix = f"{task}-{s['method']}-{s['repeat']:02d}-"
            selected = sorted((t for t in tests if t['slot'].startswith(prefix)), key=lambda t: t['slot'])
            assert [int(t['slot'].rsplit('-', 1)[1]) for t in selected] == [61001, 61002, 61003]
            assert all(t['source_sha256'] == s['selected_source'] for t in selected)
            values = [t['metrics']['ndcg@5'] for t in selected]
            rows.append(dict(method=s['method'], source_search_repeat=s['repeat'], target=task,
                             source_sha256=s['selected_source'], source_dev_ndcg5=s['value'],
                             target_test_mean=statistics.mean(values), target_refit_sd=statistics.stdev(values),
                             records=selected))
    summary = []
    for task in ('cds', 'office'):
        for method in sorted({r['method'] for r in rows}):
            group = [r for r in rows if r['target']==task and r['method']==method]
            assert len(group)==3
            best_dev = max(group, key=lambda r:r['source_dev_ndcg5'])
            best_target = max(group, key=lambda r:r['target_test_mean'])
            summary.append(dict(target=task, method=method,
                                mean_test=statistics.mean(r['target_test_mean'] for r in group),
                                sd_across_source_searches=statistics.stdev(r['target_test_mean'] for r in group),
                                highest_source_dev_repeat=best_dev['source_search_repeat'],
                                highest_target_test_repeat=best_target['source_search_repeat'],
                                top_repeat_agrees=best_dev['source_search_repeat']==best_target['source_search_repeat']))
    result=dict(rows=rows,summary=summary,logical_test_records=sum(len(r['records']) for r in rows),
                inputs={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ('selection','tests')},
                scope='Original cohort only. Source DEV rank vs target TEST rank is descriptive; no target TEST-based selection, no replacement RO1, no new target searches. Index linkage does not itself verify physical checkpoint bytes.',
                new_api=0,new_fit=0,new_test=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
