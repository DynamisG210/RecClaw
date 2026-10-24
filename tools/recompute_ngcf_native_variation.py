"""Recompute NGCF source-search and refit variation from retained native TESTs."""
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
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.snapshot.read_text(encoding='utf-8-sig'))
    command = 'tar -cf - -- ' + ' '.join(shlex.quote(r['path']) for r in rows)
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', a.host, command], capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode())
    groups = collections.defaultdict(list)
    checks = []
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
        for r in rows:
            raw = archive.extractfile(r['path'].lstrip('/')).read()
            d = json.loads(raw)
            assert hashlib.sha256(raw).hexdigest() == r['sha256'] and d == r['data'], r['path']
            identity = d['gpu_cost']['contract']['identity']
            assert identity['method'] == 'ro' and identity['repeat'] == r['repeat']
            assert identity['train_seed'] == r['seed'] == d['result']['seed']
            assert d['result']['status'] == 'SUCCESS' and d['result']['training_steps'] == 0
            checks.append(dict(path=r['path'], sha256=r['sha256'], matches=True))
            groups[r['dataset'], r['repeat']].append(r)
    searches = []
    for (dataset, repeat), rs in sorted(groups.items()):
        assert sorted(r['seed'] for r in rs) == [61001, 61002, 61003]
        assert len({r['data']['source_sha256'] for r in rs}) == 1
        metrics = {}
        for m in ['ndcg@5', 'hit@5']:
            values = [r['data']['result']['metrics'][m] for r in rs]
            metrics[m] = dict(mean=statistics.mean(values), refit_sd=statistics.stdev(values), values=values)
        searches.append(dict(dataset=dataset, source_repeat=repeat,
                             source_sha256=rs[0]['data']['source_sha256'], metrics=metrics))
    table_text = a.table.read_text(encoding='utf-8')
    section = table_text.split('{NGCF}')[1].split('\\midrule')[0]
    line = next(line for line in section.splitlines() if 'RecRSI' in line)
    table_values = re.findall(r'0\.\d{4}', line)
    cells = []
    for dataset, offset in [('CDs', 0), ('MovieLens-1M', 6)]:
        rs = [s for s in searches if s['dataset'] == dataset]
        assert sorted(s['source_repeat'] for s in rs) == [1, 2, 3]
        metrics = {}
        for j, m in enumerate(['ndcg@5', 'hit@5']):
            means = [s['metrics'][m]['mean'] for s in rs]
            mean = statistics.mean(means)
            metrics[m] = dict(mean=mean, source_search_mean_sd=statistics.stdev(means),
                              mean_within_source_refit_sd=statistics.mean(s['metrics'][m]['refit_sd'] for s in rs),
                              table=table_values[offset+j], table_matches=f'{mean:.4f}' == table_values[offset+j])
        cells.append(dict(dataset=dataset, metrics=metrics))
    result = dict(snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),
                  table_sha256=hashlib.sha256(a.table.read_bytes()).hexdigest(),
                  native_checks=checks, logical_test_slots=len(rows), unique_physical_tests=len({r['path'] for r in rows}),
                  source_search_groups=searches, cells=cells,
                  interpretation='Adopted replacement RO1 plus original RO2/RO3. CDs transfers the same three ML1M-selected programs; CDs is not three new target searches. TEST calls perform zero training steps; refits occurred upstream. Source/checkpoint hashes here are recorded identities, not independent byte verification.',
                  new_api=0, new_fit=0, new_test=0)
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(native_tests=len(checks), cells=cells), indent=2))


if __name__ == '__main__':
    main()
