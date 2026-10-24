"""Advance each prepared RQ4 candidate through existing shared GPU/CPU admission."""
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--priority-dir',type=Path,required=True);p.add_argument('--capacity-dir',type=Path,required=True);p.add_argument('--slots',type=Path,required=True);p.add_argument('--required-mib',type=int,required=True);a=p.parse_args();sys.path[:0]=[str(a.priority_dir),str(a.capacity_dir)]
from gpu_cpu_capacity import occupied
from priority_capacity import gpu_capacity
with (a.root/'GPU_CONTROLLER.lock').open('a+b') as owner:
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for arm in ('Raw','Summary','Mechanism'):
        home=a.root/arm;cfg=json.loads((home/'EVALUATOR_CONFIG.json').read_text());cpus=set(cfg['cpus']);gpu=cfg['gpu_uuid']
        if list(home.glob('INFRA*_GPU_ROUND_003_STARTED.json')):raise RuntimeError('Existing GPU attempt requires diagnosis; no automatic replay')
        deadline=time.monotonic()+14400
        while time.monotonic()<deadline:
            if not (home/'INFRA000001_CPU_ROUND_003_QUALIFICATION_READY.json').exists():time.sleep(15);continue
            with ExitStack() as stack:
                try:
                    paths={a.slots/f'CPU{min(cpus):02d}_{max(cpus):02d}.lock',a.slots/f'CPU{min(cpus)}_{max(cpus)}.lock'}
                    for path in a.slots.glob('CPU*.lock'):
                        m=re.fullmatch(r'CPU(\d+)_(\d+)\.lock',path.name)
                        if m and cpus & set(range(int(m[1]),int(m[2])+1)):paths.add(path)
                    for path in sorted(paths):
                        lock=stack.enter_context(path.open('a+b'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if cpus & occupied():raise BlockingIOError()
                    env=dict(os.environ,LD_LIBRARY_PATH=cfg['driver_root'])
                    inventory=subprocess.check_output([str(Path(cfg['driver_root'])/'nvidia-smi'),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],env=env,text=True)
                    choices=sorted((int(free.strip()),ident.strip()) for ident,free in (line.split(',') for line in inventory.splitlines()))
                    admitted=False;reasons=[]
                    for free,ident in reversed(choices):
                        if free<a.required_mib:continue
                        try:receipt=stack.enter_context(gpu_capacity(a.slots,ident,cfg['driver_root'],a.required_mib,'rq4_state2_three_arm_successor_20261024_v1'))
                        except (BlockingIOError,RuntimeError) as error:
                            reasons.append({'gpu':ident,'reason':str(error)});continue
                        gpu=ident;admitted=True;break
                    if not admitted:
                        (a.root/'WAITING_CAPACITY.json').write_text(json.dumps({'time_ns':time.time_ns(),'required_mib':a.required_mib,'reasons':reasons,'cpu_released_between_checks':True}))
                        raise BlockingIOError()
                    lockpath=a.slots/(gpu+'.lock');gpu_handle=stack.enter_context(lockpath.open('a+b'));fcntl.flock(gpu_handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except (BlockingIOError,RuntimeError):pass
                else:
                    placement=a.root/'PLACEMENT.json';body={'gpu_lock':str(lockpath)}
                    placement.write_text(json.dumps(body))
                    old_gpu=cfg['gpu_uuid'];cfg['gpu_uuid']=gpu
                    config_path=home/'EVALUATOR_CONFIG.json';config_path.write_text(json.dumps(cfg,indent=2)+'\n')
                    binding_path=a.root/'EXECUTION_BINDING.json';binding=json.loads(binding_path.read_text());binding['files'][str(config_path)]=hashlib.sha256(config_path.read_bytes()).hexdigest();binding_path.write_text(json.dumps(binding,indent=2)+'\n')
                    (home/'PHYSICAL_PLACEMENT.json').write_text(json.dumps({'old_gpu':old_gpu,'admitted_gpu':gpu,'time_ns':time.time_ns(),'scientific_inputs_changed':False}))
                    command=[sys.executable,'-B',str(a.root/'entry.py'),'--arm',arm,'--attempt','1','--round','3','--gpu-fd',str(gpu_handle.fileno())]
                    (home/'GPU_ADMISSION.json').write_text(json.dumps({'receipt':receipt,'cpu_locks':[str(p) for p in sorted(paths)],'command':command,'time_ns':time.time_ns()}))
                    with (home/'GPU_RUN.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu_handle.fileno(),))
                    if result.returncode:raise RuntimeError(f'{arm} native GPU failure; inspect retained log')
                    print(json.dumps({'arm':arm,'gpu_phase_returned':True}),flush=True);break
            time.sleep(15)
        else:raise RuntimeError('Finite candidate/admission wait expired')
