"""Seal native successor results, matching candidate hashes, curves and checkpoint identities."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[];manifest=[]
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
round_id=read(a.root/'PROSPECTIVE_CONTRACT.json')['next_round']
for arm in ('Raw','Summary','Mechanism'):
 h=a.root/arm;r=read(h/f'INFRA000001_GPU_ROUND_{round_id:03d}_RESULT.json');ready=read(h/f'INFRA000001_CPU_ROUND_{round_id:03d}_QUALIFICATION_READY.json');new=[x for x in r['evaluation']['results'] if x.get('genuine')];assert len(new)==1
 e=new[0];n=e['result']['artifacts']['native_evidence'];assert n['status']=='SUCCESS' and not n['test_accessed'];w=Path(n['artifacts']['output_dir']);curve=read(w/'EPOCH_METRICS.json');assert curve['selected_metrics']['ndcg@5']==e['primary_metric'];assert curve['curve'];assert (w/'BEST_CHECKPOINT.pt').stat().st_size>0
 candidate=Path(ready['candidate_root'])
 for name,d in ready['source_sha256'].items():assert sha(candidate/name)==d
 paths=[(h/f'INFRA000001_GPU_ROUND_{round_id:03d}_RESULT.json','RESULT.json'),(h/f'INFRA000001_CPU_ROUND_{round_id:03d}_QUALIFICATION_READY.json','READY.json')]+[(candidate/n,'source/'+n) for n in ready['source_sha256']]+[(f,'worker/'+f.name) for f in w.iterdir() if f.is_file()]
 for f,rel in paths:
  rec={'arm':arm,'source':str(f),'sha256':sha(f),'bytes':f.stat().st_size}
  if f.suffix!='.pt':
   dest=a.output/arm/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);rec['local_relative']=str(dest.relative_to(a.output))
  else:rec['checkpoint_preserved_remote']=True
  manifest.append(rec)
 jobs=list(r['gpu']['jobs'].values());assert all(j['exit_code']==0 for j in jobs);assert r['gpu']['reserved_seconds']==0
 rows.append({'arm':arm,'valid_fits':1,'ndcg5':e['primary_metric'],'retained_frontier':r['evaluation']['winner']['primary_metric'],'curve_evaluations':len(curve['curve']),'last_evaluated_epoch':curve['curve'][-1]['epoch'],'selected_metrics':curve['selected_metrics'],'api_usd':r['api']['normalized_cost_usd'],'api_calls':r['api']['logical_calls'],'unknown_calls':r['api']['unknown_usage_failures'],'held_usd':r['api']['unresolved_reference_hold_usd'],'new_gpu_seconds':sum(j['gpu_seconds'] for j in jobs),'training_gpu_seconds':sum(j['gpu_seconds'] for j in jobs if j['kind']=='training'),'qualification_gpu_seconds':sum(j['gpu_seconds'] for j in jobs if j['kind']=='qualification'),'historical_root_reference_seconds':r['gpu']['root_reference_cost']['gpu_seconds'],'source_sha256':ready['source_sha256']})
result={'state':round_id-1,'valid_candidate_fits':3,'test_fits':0,'rows':rows,'api_usd':sum(x['api_usd'] for x in rows),'new_gpu_seconds':sum(x['new_gpu_seconds'] for x in rows),'mechanism_minus_summary':rows[2]['ndcg5']-rows[1]['ndcg5'],'summary_minus_raw':rows[1]['ndcg5']-rows[0]['ndcg5'],'checkpoint_policy':'Hashed original remote checkpoints retained; receipt/log/curve/source copies sealed; no retraining'}
(a.output/'SETTLEMENT.json').write_text(json.dumps(result,indent=2));(a.output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(result))

