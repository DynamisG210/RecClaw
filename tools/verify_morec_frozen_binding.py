"""Bind MoRec frozen search inputs to measured remote bytes and native TEST."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--native-tests',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    frozen=read(a.binding/'all3_frozen_inputs.json')
    plan=read(a.binding/'final_plan.json')
    review=read(a.review)
    remote=read(a.binding/'REMOTE_PINS.json')
    pins={r['path']:r for r in remote['rows']}
    rows=[]
    for f in frozen['rows']:
        t=next(r for r in review['rows'] if r['system']=='RO' and r['repeat']==f['repeat'])
        b=(a.native_tests/(t['result_sha256']+'.json')).read_bytes()
        test=json.loads(b)
        fitpath=a.binding/('FIT_PLAN_'+str(f['repeat'])+'.json')
        fit=read(fitpath)
        source=fit['sources'][f['fit_source_key']]
        source_checks={name:pins[source['path']+'/'+name]['actual']==h for name,h in source['sha256'].items()}
        checks={
            'fit_plan_bytes_match_frozen_pin':hashlib.sha256(fitpath.read_bytes()).hexdigest()==f['pinned_inputs'][f['fit_plan']],
            'candidate_files_match_fit_plan':all(source_checks.values()),
            'checkpoint_bytes_match_frozen_pin':pins[f['checkpoint']]['actual']==f['checkpoint_sha256'],
            'test_bytes_match_adopted_review':hashlib.sha256(b).hexdigest()==t['result_sha256'],
            'test_checkpoint_matches_frozen_search':test['checkpoint_sha256']==f['checkpoint_sha256'],
            'test_source_matches_frozen_search':test['source_sha256']==f['source_sha256'],
            'test_plan_matches':test['plan_sha256']==hashlib.sha256((a.binding/'final_plan.json').read_bytes()).hexdigest(),
            'plan_binds_original_freeze':plan['freeze_inputs_sha256']==hashlib.sha256((a.binding/'all3_frozen_inputs.json').read_bytes()).hexdigest(),
            'test_metrics_match_review':test['metrics']==t['metrics'],
            'test_seed_matches_train_seed':test['seed']==f['train_seed'],
            'checkpoint_evaluation_only':test['training'] is False,
        }
        rows.append(dict(repeat=f['repeat'],search_seed=f['search_seed'],train_seed=f['train_seed'],
                         eval_seed=f['eval_seed'],source_sha256=f['source_sha256'],checkpoint_sha256=f['checkpoint_sha256'],
                         checks=checks,source_file_checks=source_checks,fit_seeds=fit['fit_seeds']))
    result=dict(rows=rows,remote_pin_count=len(pins),remote_pins_matched=sum(r['match'] for r in pins.values()),
                inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.review,a.binding/'REMOTE_PINS.json',a.binding/'all3_frozen_inputs.json',a.binding/'final_plan.json']},
                new_api=0,new_fit=0,new_test=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'chains':[(r['repeat'],sum(r['checks'].values()),len(r['checks'])) for r in rows],
                      'candidate_file_checks':sum(sum(r['source_file_checks'].values()) for r in rows),
                      'remote_pins_matched':result['remote_pins_matched']}))

if __name__=='__main__':
    main()
