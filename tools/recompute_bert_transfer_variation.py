"""Verify retained BERT transfer TEST records and separate source-search/refit variation."""
import argparse
import collections
import hashlib
import io
import json
import shlex
import statistics
import subprocess
import tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    records=json.loads(a.snapshot.read_text(encoding='utf-8-sig'))
    command='tar -cf - -- '+' '.join(shlex.quote(r['path']) for r in records)
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.host,command],capture_output=True)
    assert proc.returncode==0,proc.stderr.decode()
    checks=[];groups=collections.defaultdict(list)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
        for r in records:
            b=archive.extractfile(r['path'].lstrip('/')).read()
            data=json.loads(b)
            ok=hashlib.sha256(b).hexdigest()==r['sha256'] and data==r['data']
            checks.append(dict(path=r['path'],sha256=hashlib.sha256(b).hexdigest(),matches=ok))
            assert ok,r['path']
            assert data['source_sha256']==r['source_sha256'] and data['seed']==r['seed']
            groups[r['dataset'],r['method'],r['repeat']].append(r)
    searches=[]
    for (dataset,method,repeat),rows in sorted(groups.items()):
        assert len({r['source_sha256'] for r in rows})==1
        assert len({r['seed'] for r in rows})==len(rows)
        metrics={}
        for m in ['ndcg_at_5','recall_at_5']:
            vals=[r['data']['metrics'][m] for r in rows]
            metrics[m]=dict(mean=statistics.mean(vals),refit_sd=statistics.stdev(vals),values=vals)
        searches.append(dict(dataset=dataset,method=method,source_repeat=repeat,
                             source_sha256=rows[0]['source_sha256'],refit_seeds=[r['seed'] for r in rows],metrics=metrics))
    cells=[]
    for dataset,method in sorted({(r['dataset'],r['method']) for r in searches}):
        rs=[r for r in searches if r['dataset']==dataset and r['method']==method]
        metrics={}
        for m in ['ndcg_at_5','recall_at_5']:
            vals=[r['metrics'][m]['mean'] for r in rs]
            metrics[m]=dict(mean=statistics.mean(vals),source_search_mean_sd=statistics.stdev(vals),
                            mean_within_source_refit_sd=statistics.mean(r['metrics'][m]['refit_sd'] for r in rs))
        cells.append(dict(dataset=dataset,method=method,source_repeats=[r['source_repeat'] for r in rs],metrics=metrics))
    result=dict(snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),native_checks=checks,
                source_search_groups=searches,cells=cells,new_target_searches=0,new_api=0,new_fit=0,new_test=0,
                interpretation='Existing transferred source-search winners; variation of transferred source-search means and within-source refits are distinct. No independent target search inferred.')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(native_tests=len(checks),source_search_groups=len(searches),cells=cells),indent=2))

if __name__=='__main__':
    main()
