"""Run allocated successor CPU candidate generation under existing shared CPU leases."""
import argparse,fcntl,json,os,re,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--priority-dir',type=Path,required=True);p.add_argument('--slots',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.priority_dir));from gpu_cpu_capacity import occupied
with (a.root/'CPU_CONTROLLER.lock').open('a+b') as owner:
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for arm in ('Raw','Summary','Mechanism'):
        home=a.root/arm
        if list(home.glob('INFRA*_CPU_ROUND_003_STARTED.json')):raise RuntimeError('Existing paid attempt; diagnose before resume')
        cpus=set(json.loads((home/'EVALUATOR_CONFIG.json').read_text())['cpus'])
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            with ExitStack() as leases:
                try:
                    paths={a.slots/f'CPU{min(cpus):02d}_{max(cpus):02d}.lock',a.slots/f'CPU{min(cpus)}_{max(cpus)}.lock'}
                    for path in a.slots.glob('CPU*.lock'):
                        match=re.fullmatch(r'CPU(\d+)_(\d+)\.lock',path.name)
                        if match and cpus & set(range(int(match[1]),int(match[2])+1)):paths.add(path)
                    for path in sorted(paths):
                        handle=leases.enter_context(path.open('a+b'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if cpus & occupied():raise BlockingIOError()
                except BlockingIOError:
                    pass
                else:
                    command=[sys.executable,'-B',str(a.root/'entry.py'),'--arm',arm,'--attempt','1','--round','3','--cpu']
                    (home/'CPU_ADMISSION.json').write_text(json.dumps({'cpus':sorted(cpus),'locks':[str(p) for p in sorted(paths)],'command':command,'started_ns':time.time_ns()}))
                    with (home/'CPU_RUN.log').open('x') as log:
                        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:raise RuntimeError(f'{arm} native CPU failure; inspect retained log')
                    print(json.dumps({'arm':arm,'cpu_phase_returned':True}),flush=True)
                    break
            time.sleep(15)
        else:raise RuntimeError('No shared CPU slot within finite admission window')
