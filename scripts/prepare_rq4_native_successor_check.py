"""Prepare isolated, network-denied native frozen-state successor payload checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--state',type=int,default=2,choices=(2,3,18,6,10,8,12))
p.add_argument('--original-state',type=Path,required=True)
p.add_argument('--feedback-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--search-seed',type=int,required=True)
p.add_argument('--api-config',type=Path,required=True,help='Prospective common provider route; no credentials copied')
a=p.parse_args()
S=a.original_state.resolve();A=a.output.resolve();V=S.parent if a.state==18 else S.parent.parent;H=V/'rq2_effective_full_five_arm_v1'
assert S.name == ('rq4_feedback_action_space_v1_20261008' if a.state==18 else f'state_{a.state:02d}'), 'State directory must match selected state'
A.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for name in (('COMMON_M18_RAW.json','PREFIX_RESULT.json','PROSPECTIVE_CONTRACT.json') if a.state==18 else ('COMMON_RAW.json','HISTORY_MANIFEST.json','PREFIX_RESULT.json','PROSPECTIVE_CONTRACT.json')):
    shutil.copyfile(S/name,A/name)
pins={}
for arm in ('Raw','Summary','Mechanism'):
    home=A/arm;home.mkdir()
    for name in ('EVALUATOR_CONFIG.json','LAUNCH.json','BUDGET.json','PROVIDER.json','INPUTS.json'):
        data=read(S/arm/name)
        if name=='LAUNCH.json':
            data['campaign_id']=f'rq4-successor-state{a.state}-20261024-'+arm
            data['search_seed']=a.search_seed
        if name=='PROVIDER.json':
            data['api_config']=str(a.api_config)
            data['model_api_configs']={model:str(a.api_config) for model in data['model_api_configs']}
            data['run_identity']['search_seed']=a.search_seed
        if name=='BUDGET.json':data['consumer']['search_seed']=a.search_seed
        if name=='EVALUATOR_CONFIG.json':data['run_identity']['search_seed']=a.search_seed
        write(home/name,data)
    if arm!='Raw':
        source=a.feedback_root/f'outputs/state_{a.state}'/arm/'FEEDBACK.json'
        pins[arm]=hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source,home/'PAID_FEEDBACK.json')
source=(S/'entry.py').read_text()
header="A=Path(__file__).resolve().parent;V=A.parent.parent;H=V/'rq2_effective_full_five_arm_v1';R=A/'runtime';Q=V.parent"
if a.state==18:header=header.replace('V=A.parent.parent','V=A.parent')
assert header in source
source=source.replace(header,f"A=Path(__file__).resolve().parent;V=Path({str(V)!r});H=Path({str(H)!r});R=Path({str(S/'runtime')!r});Q=V.parent\nsys.path.insert(0,{str(S)!r})")
needle=" home=A/args.arm;cfg=read(home/'EVALUATOR_CONFIG.json');"
assert needle in source
source=source.replace(needle," assert args.check, 'Offline-only: paid calls and training are prohibited'\n"+needle)
needle="representation=None if args.arm=='Raw' else {'offline_preflight_only':True,'no_paid_feedback_generated':True}"
assert needle in source
source=source.replace(needle,"representation=None if args.arm=='Raw' else read(home/'PAID_FEEDBACK.json')['response']['proposals'][0]")
compile(source,str(A/'check.py'),'exec');(A/'check.py').write_text(source)
write(A/'PREPARATION.json',{'original_entry_sha256':hashlib.sha256((S/'entry.py').read_bytes()).hexdigest(),
    'offline_entry_sha256':hashlib.sha256(source.encode()).hexdigest(),'paid_feedback_sha256':pins,
    'search_seed':a.search_seed,'api_config':str(a.api_config),'route_change':'Common prospective route; original offline capture retained separately','state':a.state,'order_basis':'original frozen order [2,3,18,6,10,8,12]; not chosen by new output or quality',
    'native_network_worker_denial_preserved':True,'new_api_calls':0,'new_fits':0})
print(json.dumps({'output':str(A),'status':'OFFLINE_NATIVE_ENTRY_PREPARED','paid_feedback_pins':pins}))
