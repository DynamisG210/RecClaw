"""Advance each prepared RQ4 candidate through existing shared GPU/CPU admission."""
import argparse,fcntl,json,re,subprocess,sys,time
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
                    receipt=stack.enter_context(gpu_capacity(a.slots,gpu,cfg['driver_root'],a.required_mib,'rq4_state2_three_arm_successor_20261024_v1'))
                    lockpath=a.slots/(gpu+'.lock');gpu_handle=stack.enter_context(lockpath.open('a+b'));fcntl.flock(gpu_handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except (BlockingIOError,RuntimeError):pass
                else:
                    placement=a.root/'PLACEMENT.json';body={'gpu_lock':str(lockpath)}
                    if placement.exists():assert json.loads(placement.read_text())==body
                    else:placement.write_text(json.dumps(body))
                    command=[sys.executable,'-B',str(a.root/'entry.py'),'--arm',arm,'--attempt','1','--round','3','--gpu-fd',str(gpu_handle.fileno())]
                    (home/'GPU_ADMISSION.json').write_text(json.dumps({'receipt':receipt,'cpu_locks':[str(p) for p in sorted(paths)],'command':command,'time_ns':time.time_ns()}))
                    with (home/'GPU_RUN.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(gpu_handle.fileno(),))
                    if result.returncode:raise RuntimeError(f'{arm} native GPU failure; inspect retained log')
                    print(json.dumps({'arm':arm,'gpu_phase_returned':True}),flush=True);break
            time.sleep(15)
        else:raise RuntimeError('Finite candidate/admission wait expired')
