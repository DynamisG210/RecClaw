"""Verify original NGCF transfer FIT/TEST bytes and account their separate GPU costs."""
import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for k in ('joined', 'index', 'out'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--host', required=True)
    a = p.parse_args()
    joined = json.loads(a.joined.read_text())
    fits = json.loads(a.index.read_text())['fit_results']
    pairs = []
    for row in joined['rows']:
        for t in row['records']:
            seed = int(t['slot'].rsplit('-',1)[1])
            matched = [f for f in fits if f['seed']==seed and f['source_sha256']==t['source_sha256'] and f['checkpoint_sha256']==t['checkpoint_sha256']]
            assert len(matched)==1, t['slot']
            pairs.append((t,matched[0]))
    paths = sorted({r['path'] for pair in pairs for r in pair})
    proc = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.host,'tar -cf - -- '+' '.join(map(shlex.quote,paths))],capture_output=True)
    assert proc.returncode==0, proc.stderr.decode()
    rows=[]
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
        for t,f in pairs:
            records=[]
            for r in (t,f):
                raw=archive.extractfile(r['path'].lstrip('/')).read()
                assert hashlib.sha256(raw).hexdigest()==r['sha256'],r['path']
                records.append(json.loads(raw))
            td,fd=records
            native=fd['result']['artifacts']['native_evidence']
            worker=native['worker']
            assert worker['checkpoint_sha256']==td['checkpoint_sha256']==t['checkpoint_sha256']
            assert worker['source_sha256']==td['source_sha256']==t['source_sha256']
            assert worker['train_seed']==fd['seed']==int(t['slot'].rsplit('-',1)[1])
            assert fd['test_accessed'] is False and worker['test_accessed'] is False
            assert td['result']['training_steps']==0 and td['result']['metrics']==t['metrics']
            rows.append(dict(slot=t['slot'],fit_path=f['path'],test_path=t['path'],
                             fit_sha256=f['sha256'],test_sha256=t['sha256'],
                             source_sha256=t['source_sha256'],checkpoint=worker['checkpoint'],
                             checkpoint_sha256=worker['checkpoint_sha256'],
                             final_fit_gpu_seconds=native['physical_cost']['gpu_seconds'],
                             test_gpu_seconds=td['gpu_cost']['gpu_seconds'],
                             checkpoint_verification=worker.get('checkpoint_verification')))
    groups=[]
    for task,method in sorted({tuple(r['slot'].split('-')[:2]) for r in rows}):
        subset=[r for r in rows if r['slot'].startswith(task+'-'+method+'-')]
        groups.append(dict(task=task,method=method,refits=len(subset),
                           final_fit_gpu_hours=sum(r['final_fit_gpu_seconds'] for r in subset)/3600,
                           test_gpu_hours=sum(r['test_gpu_seconds'] for r in subset)/3600))
    result=dict(rows=rows,groups=groups,verified_files=len(paths),
                inputs={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ("joined","index")},
                scope='Original complete CDs/Office cohort. Final FIT and TEST cost separate from search; successful matched records only, not failed/UNKNOWN total or invoice. Checkpoint hashes joined across original native records; checkpoint physical bytes not reread.',
                new_api=0,new_fit=0,new_test=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified_files=len(paths),linked=len(rows),groups=groups),indent=2))


if __name__=='__main__':
    main()
