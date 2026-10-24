"""Read-only hash verification of a frozen MoRec final-evaluation plan."""
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--native-tests',type=Path,required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    snapshot=json.loads(a.snapshot.read_text(encoding='utf-8-sig'))
    plan_sha=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    assert plan_sha==snapshot['evidence']['final_plan_sha256']
    pins=dict(plan['input_pins'])
    for r in plan['rows']:
        for path,h in r['pinned_inputs'].items():
            assert path not in pins or pins[path]==h
            pins[path]=h
        pins[r['checkpoint']]=r['checkpoint_sha256']
    commands='\n'.join('sha256sum -- '+shlex.quote(x) for x in pins)+'\n'
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.host,'bash','-s'],input=commands.encode(),capture_output=True)
    actual={line[66:]:line[:64] for line in proc.stdout.decode().splitlines() if len(line)>66}
    hashes=[dict(path=k,expected=v,actual=actual.get(k),match=v==actual.get(k)) for k,v in pins.items()]
    rows=[]
    for f in plan['rows']:
        r=next(r for r in snapshot['rows'] if all(r[k]==f[k] for k in ['dataset','system','repeat']))
        b=(a.native_tests/(r['result_sha256']+'.json')).read_bytes();test=json.loads(b)
        checks={'test_bytes_match':hashlib.sha256(b).hexdigest()==r['result_sha256'],
                'pinned_files_match':all(actual.get(k)==v for k,v in f['pinned_inputs'].items()),
                'checkpoint_bytes_match':actual.get(f['checkpoint'])==f['checkpoint_sha256'],
                'test_checkpoint_matches':test['checkpoint_sha256']==f['checkpoint_sha256'],
                'test_source_matches':test['source_sha256']==f['source_sha256'],
                'test_metrics_match':test['metrics']==r['metrics']}
        rows.append(dict(dataset=f['dataset'],system=f['system'],repeat=f['repeat'],checks=checks,
                         root_fallback=f['ROOTfallback'],cached=r['cached'],home=f['home'],
                         checkpoint=f['checkpoint'],source_sha256=f['source_sha256']))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=a.host,plan_sha256=plan_sha,
                snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),hashes=hashes,rows=rows,
                stderr=proc.stderr.decode(),exit_code=proc.returncode,new_api=0,new_fit=0,new_test=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pins=len(hashes),matching_pins=sum(r['match'] for r in hashes),
                         chains=len(rows),complete_chains=sum(all(r['checks'].values()) for r in rows),
                         mismatches=[r for r in hashes if not r['match']])))

if __name__=='__main__':
    main()
