"""Evaluate the original paid candidate after receipt-backed qualification recovery; no provider calls."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--attempt',type=int,required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--gpu-fd',type=int,required=True);a=p.parse_args();A=Path(__file__).resolve().parent;R=A/'runtime';read=lambda p:json.loads(p.read_text())
h=A/a.arm;cfg=read(h/'EVALUATOR_CONFIG.json');cfg['main_binding']=str(R)
fd=os.fstat(a.gpu_fd);lock=Path(read(A/'PLACEMENT.json')['gpu_lock']).stat();assert (fd.st_dev,fd.st_ino)==(lock.st_dev,lock.st_ino);os.set_inheritable(a.gpu_fd,False)
assert a.attempt==2 and a.round==3
binding=read(h/'QUALIFICATION_REUSE_BINDING.json');source=Path(binding['original_candidate']);files={n:(source/n).read_text() for n in binding['source_sha256']}
assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==v for n,v in binding['source_sha256'].items())
for path in binding['successful_matching_receipts']:assert read(Path(path))['status']=='SUCCESS'
sys.path[:0]=read(R/'ORIGINAL_PATHS.json');os.sched_setaffinity(0,set(cfg['cpus']))
from e2_budget_binding import bind_config
bind_config(cfg)
from morec_main_evaluator import run_files,resources_for
assert Path(__import__('e2_budget_binding').__file__).parent==R
start=h/'INFRA000002_GPU_ROUND_003_STARTED.json'
with start.open('x') as f:json.dump({'pid':os.getpid(),'time_ns':time.time_ns(),'source_sha256':binding['source_sha256'],'qualification_reused':True,'no_provider_calls':True},f)
result=run_files(cfg,files);resources=resources_for(cfg)
with (h/'RECOVERED_FIT_RESULT.json').open('x') as f:json.dump({'result':result,'gpu':resources.gpu.snapshot(),'api':resources.api.summary(),'source_sha256':binding['source_sha256'],'qualification_reused':True,'no_provider_calls':True},f,indent=2)
print(json.dumps({'arm':a.arm,'status':result.get('status'),'metrics':result.get('metrics')}),flush=True)
