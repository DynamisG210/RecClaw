"""Seal completed RQ4 recovery receipts and reconcile effective opportunities without replay."""
import argparse, hashlib, json, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
rows=[];manifest=[]
for arm in ('Raw','Summary','Mechanism'):
 h=a.root/arm;r=json.loads((h/'RECOVERED_FIT_RESULT.json').read_text());b=json.loads((h/'QUALIFICATION_REUSE_BINDING.json').read_text());w=Path(r['result']['artifacts']['output_dir']);source=Path(b['original_candidate'])
 assert r['result']['status']=='SUCCESS' and not r['result']['test_accessed']
 assert r['gpu']['reserved_seconds']==0 and r['api']['open_reservations']==0
 for n,d in r['source_sha256'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==d
 paths=[(h/'RECOVERED_FIT_RESULT.json','result.json'),(h/'QUALIFICATION_REUSE_BINDING.json','qualification_binding.json')]
 paths += [(source/n,'source/'+n) for n in r['source_sha256']]
 paths += [(f,'worker/'+str(f.relative_to(w))) for f in w.rglob('*') if f.is_file()]
 paths += [(h/'search'/n,'ledger/'+n) for n in ('GPU_JOB_BUDGET.json','GPU_WORKER_EVENTS.jsonl','API_EVENTS.jsonl','ACTIVE_WALL.json') if (h/'search'/n).exists()]
 for f,rel in paths:
  dest=a.output/arm/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
  manifest.append({'source':str(f),'sealed':str(dest.relative_to(a.output)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size})
 jobs=list(r['gpu']['jobs'].values());train=[j for j in jobs if j['kind']=='training'];assert len(train)==1 and train[0]['exit_code']==0
 rows.append({'arm':arm,'effective_scientific_opportunities':1,'valid_candidate_fits':1,'test_fits':0,'qualification_workers':sum(j['kind']=='qualification' for j in jobs),'engineering_supervision_failures_excluded':2,'ndcg5':r['result']['metrics']['ndcg@5'],'hr5':r['result']['metrics']['hr@5'],'training_gpu_seconds':sum(j['gpu_seconds'] for j in train),'qualification_seconds_conservative':sum(j['gpu_seconds'] for j in jobs if j['kind']=='qualification'),'historical_root_reference_seconds':r['gpu']['root_reference_cost']['gpu_seconds'],'api_calls':r['api']['logical_calls'],'api_reference_usd':r['api']['normalized_cost_usd'],'unknown_calls':r['api']['unknown_usage_failures'],'checkpoint_files':[str(f.relative_to(w)) for f in w.rglob('*') if f.suffix in ('.pt','.pth','.ckpt')]})
summary={'stage':'COMPLETE_THREE_VALID_DEV_FITS','state':2,'rows':rows,'valid_fits':3,'test_fits':0,'api_reference_usd':sum(r['api_reference_usd'] for r in rows),'new_gpu_seconds_including_qualification_upper':sum(r['training_gpu_seconds']+r['qualification_seconds_conservative'] for r in rows),'primary_mechanism_minus_summary':rows[2]['ndcg5']-rows[1]['ndcg5'],'summary_minus_raw':rows[1]['ndcg5']-rows[0]['ndcg5'],'accounting':'Versioned engineering-excluded settlement; raw records unchanged; historical root reference not new computation','checkpoint_curve_availability':'Frozen main_code keeps best_state in memory; no per-epoch metrics or checkpoint serialization. Do not reconstruct or replay successful FITs.'}
(a.output/'SETTLEMENT.json').write_text(json.dumps(summary,indent=2));(a.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(summary))
