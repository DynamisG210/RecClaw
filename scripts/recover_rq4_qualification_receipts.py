"""Recover completed qualification receipts after private-runtime watcher import failure."""
import argparse,hashlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();A=a.root.resolve();R=A/'runtime'
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
paths=read(R/'ORIGINAL_PATHS.json');old=paths[0];paths[0]=str(R);write(R/'ORIGINAL_PATHS.json',paths);sys.path[:0]=paths
from e2_budget_binding import bind_config
from gpu_job_budget import GPUJobBudget
records=[]
for arm in ('Raw','Summary','Mechanism'):
 h=A/arm;cfg=read(h/'EVALUATOR_CONFIG.json');cfg['main_binding']=str(R);bind_config(cfg)
 budget=GPUJobBudget(h/'search/GPU_JOB_BUDGET.json',cfg['run_identity']);state=budget.snapshot()
 backup=h/'search/GPU_BUDGET_BEFORE_RECEIPT_RECOVERY.json'
 if not backup.exists():write(backup,state)
 for token,row in state['jobs'].items():
  if row['finished_ns'] is not None:continue
  job=read(Path(row['control_job']));out=Path(job['run_root']);exitp=Path(job['exit_record']);ex=read(exitp);qual=read(out/'RO_SHARED_QUALIFICATION_SMOKE_RESULT.json')
  assert ex['exit_code']==0 and qual['status']=='SUCCESS' and qual['text_initialization_exact_match']
  assert not Path('/proc',str(row['worker_pid'])).exists(),'Worker identity unresolved'
  # Trusted launcher writes EXIT after completion; add one second conservatively.
  end=max(exitp.stat().st_mtime_ns,(out/'RO_SHARED_QUALIFICATION_SMOKE_RESULT.json').stat().st_mtime_ns)+1_000_000_000
  assert row['started_ns']<end<__import__('time').time_ns()
  budget.finish(token,finished_ns=end,exit_code=0,stopped_by_common_limit=False)
  records.append({'arm':arm,'token':token,'exit_path':str(exitp),'exit_sha256':hashlib.sha256(exitp.read_bytes()).hexdigest(),'charged_seconds_upper':(end-row['started_ns'])/1e9,'basis':'Trusted launcher EXIT receipt mtime plus one second; conservative reconstructed upper bound, not original pidfd measurement'})
 # Retain already successful qualifications; do not replay them.
 ready=read(h/'INFRA000001_CPU_ROUND_003_QUALIFICATION_READY.json');source=Path(ready['candidate_root'])
 matching=[]
 for qp in (h/'search/campaign/innovation_candidates/round-0003').glob('*/.e2*/RO_SHARED_QUALIFICATION_SMOKE_RESULT.json'):
  jobstate=read(qp.parent/'MAIN_QUALIFICATION_PROCESS.json');args=jobstate['command'];cand=Path(args[args.index('--candidate-root')+1])
  if all(hashlib.sha256((cand/n).read_bytes()).hexdigest()==v for n,v in ready['source_sha256'].items()):matching.append(str(qp))
 assert matching
 write(h/'QUALIFICATION_REUSE_BINDING.json',{'original_candidate':str(source),'source_sha256':ready['source_sha256'],'successful_matching_receipts':matching,'new_qualification_workers':0,'engineering_fault_excluded_from_scientific_opportunities':True})
write(A/'WATCHER_RECOVERY.json',{'old_runtime_import':old,'new_runtime_import':str(R),'rows':records,'api_repair_cost_retained':True,'no_budget_reset':True})
binding=read(A/'EXECUTION_BINDING.json');binding['files'][str(R/'ORIGINAL_PATHS.json')]=hashlib.sha256((R/'ORIGINAL_PATHS.json').read_bytes()).hexdigest();write(A/'EXECUTION_BINDING.json',binding)
print(json.dumps({'recovered_jobs':len(records),'charged_seconds_upper':sum(r['charged_seconds_upper'] for r in records),'runtime_budget_module':__import__('e2_budget_binding').__file__}))
