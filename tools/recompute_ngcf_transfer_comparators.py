"""Verify frozen NGCF transfer comparator TESTs and recompute source/refit variation."""
import argparse
import collections
import hashlib
import io
import json
import re
import shlex
import statistics
import subprocess
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for name in ['results', 'sources', 'table', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--host', required=True)
    a = p.parse_args()
    manifest = json.loads(a.results.read_text())
    sources = {(r['method'], r['repeat']): r for r in json.loads(a.sources.read_text())}
    rows = [r for r in manifest['test_results'] if r['slot'].split('-')[0] in ['cds', 'office'] and r['slot'].split('-')[1] != 'ro']
    command = 'tar -cf - -- ' + ' '.join(shlex.quote(r['path']) for r in rows)
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', a.host, command], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    groups = collections.defaultdict(list)
    native = []
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
        for row in rows:
            raw = archive.extractfile(row['path'].lstrip('/')).read()
            assert hashlib.sha256(raw).hexdigest() == row['sha256'], row['path']
            d = json.loads(raw)
            dataset, method, repeat, seed = row['slot'].split('-')
            repeat, seed = int(repeat), int(seed)
            assert d['result']['metrics'] == row['metrics']
            assert d['source_sha256'] == row['source_sha256'] == sources[method, repeat]['source_sha256']
            assert d['checkpoint_sha256'] == row['checkpoint_sha256']
            assert d['result']['seed'] == seed and d['result']['status'] == 'SUCCESS'
            assert d['result']['training_steps'] == 0
            native.append(dict(path=row['path'], sha256=row['sha256'], data=d))
            groups[dataset, method, repeat].append((seed, d))
    searches = []
    for (dataset, method, repeat), rs in sorted(groups.items()):
        assert sorted(x[0] for x in rs) == [61001, 61002, 61003]
        metrics = {}
        for m in ['ndcg@5', 'hit@5']:
            vals = [x[1]['result']['metrics'][m] for x in rs]
            metrics[m] = dict(mean=statistics.mean(vals), refit_sd=statistics.stdev(vals), values=vals)
        searches.append(dict(dataset=dataset, method=method, repeat=repeat, metrics=metrics))
    section = a.table.read_text(encoding='utf-8').split('{NGCF}')[1].split('\\midrule')[0]
    labels = dict(rh='RecHarness', deep='DeepEvolve', openevolve='OpenEvolve', shinka='ShinkaEvolve', se='Self-EvolveRec')
    cells = []
    for dataset, method in sorted({(s['dataset'], s['method']) for s in searches}):
        rs = [s for s in searches if (s['dataset'], s['method']) == (dataset, method)]
        assert sorted(s['repeat'] for s in rs) == [1, 2, 3]
        line = next(line for line in section.splitlines() if labels[method] in line)
        table = re.findall(r'0\.\d{4}', line)
        metrics = {}
        for j, m in enumerate(['ndcg@5', 'hit@5']):
            means = [s['metrics'][m]['mean'] for s in rs]
            mean = statistics.mean(means)
            expected = table[(0 if dataset == 'cds' else 4)+j]
            metrics[m] = dict(mean=mean, source_search_mean_sd=statistics.stdev(means), mean_within_source_refit_sd=statistics.mean(s['metrics'][m]['refit_sd'] for s in rs), table=expected, table_matches=f'{mean:.4f}' == expected)
        cells.append(dict(dataset=dataset, method=method, metrics=metrics))
    result = dict(input_hashes={k: hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ['results','sources','table']}, native=native, searches=searches, cells=cells, new_api=0, new_fit=0, new_test=0, scope='Five comparator methods, CDs/Office transfer from frozen source searches. RO replacement cohort excluded explicitly. No new target searches. Source/checkpoint identities matched to manifests; physical checkpoint bytes not checked.')
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(native=len(native), groups=len(searches), table_matches=sum(m['table_matches'] for c in cells for m in c['metrics'].values()), cells=cells), indent=2))


if __name__ == '__main__':
    main()
