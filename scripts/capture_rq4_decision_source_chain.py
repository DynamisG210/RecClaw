"""Capture the complete native proposal pool and candidate source chain, without quality selection."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for arm in ('Raw','Summary','Mechanism'):
 h=a.root/arm;round_id=read(a.root/'PROSPECTIVE_CONTRACT.json')['next_round'];ready=h/f'INFRA000001_CPU_ROUND_{round_id:03d}_QUALIFICATION_READY.json';pool=h/'NATIVE_PROPOSAL_OUTCOMES.json'
 if not ready.exists():continue
 target=a.output/arm;target.mkdir(exist_ok=True);binding=read(ready);source=Path(binding['candidate_root'])
 for name,digest in binding['source_sha256'].items():
  assert sha(source/name)==digest
  shutil.copyfile(source/name,target/name)
 shutil.copyfile(pool,target/pool.name);shutil.copyfile(ready,target/ready.name)
 proposals=[]
 for row in read(pool):
  o=row['outcome'];concept=o.get('provenance',{}).get('conceptual_binding',{})
  proposals.append({'digest':row['digest'],'role':o['producer_role'],'failure_code':o['failure_code'],'primitives':concept.get('primitive_ids'),'spec':o.get('spec')})
 rows.append({'arm':arm,'candidate_root':str(source),'source_sha256':binding['source_sha256'],'pool_sha256':sha(pool),'complete_pool':proposals,'quality':'NOT_READ_BY_THIS_CAPTURE'})
(a.output/'DECISION_SOURCE_CHAIN.json').write_text(json.dumps({'rows':rows,'selection':'all completed arms in frozen order','new_api':0,'new_fit':0},indent=2)+'\n')
print(json.dumps({'arms':len(rows),'proposals':sum(len(r['complete_pool']) for r in rows)}))
