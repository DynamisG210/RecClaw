"""Repair zero-call engineering artifacts and isolate child runtime imports."""
import json,shutil,hashlib,time,sys,subprocess
from pathlib import Path
h=Path(sys.argv[1]);o=Path(sys.argv[2])
queue=json.loads((h/'QUEUE_CURRENT.json').read_text())
assert not Path('/proc',str(queue['pid']),'cmdline').exists(), 'Do not change a live queue'
receipt={'time_ns':time.time_ns(),'api_calls':0,'fits':0,'changes':[]}
for arm in ('retain','reset'):
 root=h/arm/'search'
 assert not (root/'API_EVENTS.jsonl').exists(), 'Paid bytes require event-specific recovery'
 assert not json.loads((root/'GPU_JOB_BUDGET.json').read_text())['jobs']
 c=root/'campaign';m=c/'ROUND_13_ATTEMPT_MANIFEST.json'
 if m.exists():
  manifest=json.loads(m.read_text());assert manifest['attempts']==[]
  archive=root/'ENGINEERING_PRECALL_ARTIFACTS';archive.mkdir(exist_ok=False)
  for f in (m,c/'ROUND_13_STARTED.json',root/'STAGE_STARTED.json'):
   if f.exists():f.rename(archive/f.name)
  receipt['changes'].append({'arm':arm,'archived_empty_attempt':manifest})
 cfgpath=h/arm/'EVALUATOR_CONFIG.json';cfg=json.loads(cfgpath.read_text())
 shutil.copy2(cfgpath,h/arm/'PRE_CHILD_RUNTIME_EVALUATOR_CONFIG.json')
 cfg['main_binding']=str(h/'runtime');cfgpath.write_text(json.dumps(cfg,indent=2))
for src in (o/'runtime').glob('*.py'):
 target=h/'runtime'/src.name
 if not target.exists():shutil.copy2(src,target)
paths=json.loads((o/'runtime/ORIGINAL_PATHS.json').read_text());paths=[str(h/'runtime')]+[p for p in paths if p!=str(o/'runtime')]
(h/'runtime/ORIGINAL_PATHS.json').write_text(json.dumps(paths))
receipt['runtime_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (h/'runtime').iterdir() if p.is_file()}
(h/'CHILD_RUNTIME_REPAIR.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'repaired':True,'runtime_files':len(receipt['runtime_hashes'])}))
