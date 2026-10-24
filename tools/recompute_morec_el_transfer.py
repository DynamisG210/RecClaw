"""Recompute retained MoRec Electronics transfer/refit TEST evidence."""
import argparse
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
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'));analysis=json.loads(a.analysis.read_text(encoding='utf-8-sig'))
    assert hashlib.sha256(a.plan.read_bytes()).hexdigest()==analysis['plan_sha256']
    root=plan['fit_plan'].rsplit('/',1)[0]
    pins={r[k]:r[k+'_sha256'] for r in plan['rows'] for k in ['checkpoint','fit_result']}
    paths=[f"{root}/controlled_test/{r['method']}/{r['seed']}/TEST_RESULT.json" for r in plan['rows']]
    proc=subprocess.run(['ssh','-o','BatchMode=yes',a.host,'tar -cf - -- '+' '.join(map(shlex.quote,paths))],capture_output=True)
    assert proc.returncode==0,proc.stderr.decode()
    hashes=subprocess.run(['ssh','-o','BatchMode=yes',a.host,'bash','-s'],input=('\n'.join('sha256sum -- '+shlex.quote(k) for k in pins)+'\n').encode(),capture_output=True)
    actual={s[66:]:s[:64] for s in hashes.stdout.decode().splitlines() if len(s)>66}
    rows=[]
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as t:
        for r,path in zip(plan['rows'],paths):
            b=t.extractfile(path.lstrip('/')).read();test=json.loads(b)
            old=next(x for x in analysis['rows'] if x['method']==r['method'] and x['seed']==r['seed'])
            checks=dict(test_hash=hashlib.sha256(b).hexdigest()==old['result_sha256'],metrics=test['metrics']==old['metrics'],
                        checkpoint_bytes=actual.get(r['checkpoint'])==r['checkpoint_sha256'],
                        fit_result_bytes=actual.get(r['fit_result'])==r['fit_result_sha256'],
                        test_checkpoint=test['checkpoint_sha256']==r['checkpoint_sha256'])
            rows.append(dict(method=r['method'],seed=r['seed'],metrics=test['metrics'],checks=checks,path=path))
    cells=[]
    for method in sorted({r['method'] for r in rows}):
        rs=[r for r in rows if r['method']==method];metrics={}
        for m in ['ndcg@5','hr@5']:
            values=[r['metrics'][m] for r in rs]
            metrics[m]=dict(mean=statistics.mean(values),refit_sd=statistics.stdev(values),values=values)
        cells.append(dict(method=method,metrics=metrics,target_searches=0,refit_seeds=[r['seed'] for r in rs]))
    result=dict(inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.plan,a.analysis]},rows=rows,cells=cells,
                remote_pins=[dict(path=k,expected=v,actual=actual.get(k),matches=v==actual.get(k)) for k,v in pins.items()],
                new_api=0,new_fit=0,new_test=0,interpretation='Each method transfers one selected source algorithm and refits three seeds; this is not three independent target searches.')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),checks_passed=sum(sum(r['checks'].values()) for r in rows),pins_matched=sum(r['matches'] for r in result['remote_pins']),cells=cells),indent=2))

if __name__=='__main__':
    main()
