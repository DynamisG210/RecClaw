"""Bind frozen feedback outputs to matched four-role successor context views.

No provider call or candidate selection. The native downstream renderer consumes
these context views; its request/template/seed binding is a separate step.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(encode(x)).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
    content=encode(x)+b'\n'
    if p.exists():assert p.read_bytes()==content, p
    else:p.write_bytes(content)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
roles=('mechanism_composer','falsification_designer','lineage_refiner','frontier_architect')
bound=[];pending=[]
for state in (2,3,18,6,10,8,12):
    outputs={arm:a.root/'outputs'/f'state_{state}'/arm/'FEEDBACK.json' for arm in ('Summary','Mechanism')}
    if not all(path.is_file() for path in outputs.values()):
        pending.append(state);continue
    facts=read(a.root/'payloads'/f'state_{state}'/'COMMON_FACTS.json')
    feedback={arm:read(path) for arm,path in outputs.items()}
    assert all(x['common_facts_sha256']==digest(facts) for x in feedback.values())
    base=deepcopy(facts['prior_history'])
    # The original request projection is already cut at the frozen observation.
    # Preserve it in all arms, changing only role identity and latest feedback.
    base['memory'].pop('latest_experiment_feedback',None)
    for arm in ('Raw','Summary','Mechanism'):
        representation=None if arm=='Raw' else feedback[arm]['response']['proposals'][0]
        home=a.root/'successor_contexts'/f'state_{state}'/arm
        home.mkdir(parents=True,exist_ok=True)
        records=[]
        for role in roles:
            view=deepcopy(base)
            view['context']['role']=role
            view['memory']['latest_experiment_feedback']={'round':state,
                'raw_observation':facts['observation'],'representation':representation}
            stripped=deepcopy(view)
            del stripped['memory']['latest_experiment_feedback']
            stripped['context']['role']=base['context']['role']
            assert stripped==base
            path=home/f'{role}.json';save(path,view)
            records.append({'role':role,'context_sha256':digest(view),'bytes':len(encode(view))})
        binding={'state':state,'arm':arm,'common_facts_sha256':digest(facts),
            'base_context_sha256':digest(base),'representation_sha256':digest(representation),
            'feedback_file_sha256':None if arm=='Raw' else hashlib.sha256(outputs[arm].read_bytes()).hexdigest(),
            'roles':records,'status':'FROZEN_CONTEXT_VIEWS_NOT_YET_PROVIDER_REQUESTS',
            'new_api_calls':0,'new_candidates':0,'new_fits':0}
        save(home/'BINDING.json',binding);bound.append(binding)
report={'paired_states_bound':len(bound)//3,'arms_bound':len(bound),'role_views':len(bound)*4,
    'pending_states':pending,'bindings':bound,
    'remaining':'Apply native proposal template/space/seed/selector and measure complete requests before downstream execution.'}
(a.root/'SUCCESSOR_CONTEXT_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('paired_states_bound','arms_bound','role_views','pending_states')}))
