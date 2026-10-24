"""Prepare an allocated native RQ4 successor block without launching calls or workers."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--original-state',type=Path,required=True);p.add_argument('--feedback-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--api-config',type=Path,required=True);a=p.parse_args()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
grant=read(a.allocation);assert grant['api_usd_cap']==3.7 and grant['search_seed']==763006 and grant['max_new_valid_fits']==3
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_rq4_native_successor_check.py')),'--original-state',str(a.original_state),'--feedback-root',str(a.feedback_root),'--output',str(a.output),'--search-seed','763006','--api-config',str(a.api_config)],check=True)
A=a.output.resolve();S=a.original_state.resolve();shutil.copyfile(a.allocation,A/'ALLOCATION.json');shutil.copytree(S/'runtime',A/'runtime')
# This private runtime reads only this stage's finite, independently funded declarations.
f=A/'runtime/e2_budget_binding.py';s=f.read_text();old="{'genuine': 1, 'reference_api_usd': 11.813253199999998, 'execution_seconds': 15397.553288171997, 'active_seconds': 162934.11051936884}"
caps={'genuine':1,'reference_api_usd':3.7/3,'execution_seconds':1800,'active_seconds':14400}
assert old in s;s=s.replace(old,repr(caps));s=s.replace('5400 incremental GPU seconds','1800 incremental GPU seconds');f.write_text(s)
for arm in grant['arms']:
    home=A/arm;out=home/'search';out.mkdir();budget=read(home/'BUDGET.json');budget.update(caps=caps,output=str(out),space_reference=None)
    budget['original_api_limits']['max_requests']=6
    write(home/'BUDGET.json',budget)
    provider=read(home/'PROVIDER.json');provider.update(resource_root=str(out),api_ledger=str(out/'API_EVENTS.jsonl'),active_wall_path=str(out/'ACTIVE_WALL.json'),ledger_module_dir=str(A/'runtime'))
    provider['api_limits'].update(max_requests=6,max_normalized_u=caps['reference_api_usd']/14)
    write(home/'PROVIDER.json',provider)
    cfg=read(home/'EVALUATOR_CONFIG.json');cfg.update(output=str(out),provider_config=str(home/'PROVIDER.json'),active_wall_path=str(out/'ACTIVE_WALL.json'),e2_budget_manifest=str(home/'BUDGET.json'),e2_budget_sha256=sha(home/'BUDGET.json'),worker_timeout_seconds=1800)
    write(home/'EVALUATOR_CONFIG.json',cfg)
    if arm!='Raw':write(home/'FROZEN_FEEDBACK.json',{'representation':read(home/'PAID_FEEDBACK.json')['response']['proposals'][0],'common_raw_sha256':sha(A/'COMMON_RAW.json'),'only_observation_round':2})
s=(A/'check.py').read_text();s=s.replace(f'R=Path({str(S/"runtime")!r})',f'R=Path({str(A/"runtime")!r})')
s=s.replace(" assert args.check, 'Offline-only: paid calls and training are prohibited'", " assert read(A/'ALLOCATION.json')['allocation_id']=='rq4_state2_three_arm_successor_20261024_v1'")
start=s.index('    if not args.check:\n     import contextvars');end=s.index('    if args.cpu and args.expected_wire:',start)
s=s[:start]+s[end:]
s=s.replace("set({'Raw':[60,61],'Summary':[60,61],'Mechanism':[60,61]}[args.arm] if args.cpu else cfg['cpus'])","set(cfg['cpus'])")
# The CPU phase builds candidates only. Native qualification interception denies workers.
compile(s,str(A/'entry.py'),'exec');(A/'entry.py').write_text(s)
shutil.copyfile(S/'COMPLETED_SOURCE_INVENTORY.json',A/'COMPLETED_SOURCE_INVENTORY.json')
files={str(p):sha(p) for p in A.rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
write(A/'EXECUTION_BINDING.json',{'contract_sha256':sha(A/'PROSPECTIVE_CONTRACT.json'),'allocation_sha256':sha(A/'ALLOCATION.json'),'files':files})
print(json.dumps({'stage':'PREPARED_NOT_EXECUTED','root':str(A),'per_arm_api_cap':caps['reference_api_usd'],'calls_cap_per_arm':6,'fits_per_arm':1,'gpu_seconds_per_arm':1800,'new_calls':0,'new_fits':0}))
