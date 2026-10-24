"""Local production-host checks; no provider requests or model training."""
import json
import os
from pathlib import Path
import platform
import subprocess
from .config import read, resolve
from .prepare import WORKER_IDENTITY


def doctor(environment):
    path = Path(environment).resolve()
    env = read(path)
    checks = {}
    checks['platform'] = dict(ok=platform.system()=='Linux' and platform.machine()=='x86_64',
                              detail=platform.platform())
    for key in ('worker_python','recbole_root','driver_root','resource_slots','api_config'):
        try:
            resolve(env[key], path.parent, interpreter=key=='worker_python')
            checks[key] = dict(ok=True)
        except (ValueError, FileNotFoundError) as exc:
            checks[key] = dict(ok=False, detail=str(exc))
    cpus=env['cpus']
    checks['cpus']=dict(ok=len(set(cpus))==4 and hasattr(os,'sched_getaffinity') and set(cpus)<=os.sched_getaffinity(0))
    commands = {}
    if checks['worker_python']['ok']:
        python=resolve(env['worker_python'],path.parent,interpreter=True)
        commands['dependencies']=[python,'-c',WORKER_IDENTITY]
        commands['cuda']=[python,'-c',"import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__)"]
    if checks['driver_root']['ok']:
        commands['gpu']=[str(Path(resolve(env['driver_root'],path.parent))/'nvidia-smi'),'-i',env['gpu_uuid'],'--query-gpu=uuid,memory.free','--format=csv,noheader']
    # The same privileged interpreter is used by the namespace launcher. The
    # child exits immediately after verifying isolated namespace creation.
    commands['namespaces']=['sudo','-n','/usr/bin/python3','-I','-c',
        "import ctypes,os; c=ctypes.CDLL(None,use_errno=True); r=c.unshare(0x00020000|0x08000000|0x40000000); assert r==0, os.strerror(ctypes.get_errno()); print('mount,ipc,network namespaces available')"]
    commands['landlock']=['/usr/bin/python3','-I','-c',
        "import ctypes,os; c=ctypes.CDLL(None,use_errno=True); abi=c.syscall(444,0,0,1); assert abi>=1, os.strerror(ctypes.get_errno()); print('Landlock ABI',abi)"]
    unified=Path('/sys/fs/cgroup/cgroup.controllers').exists()
    memory=(Path('/sys/fs/cgroup/cgroup.subtree_control') if unified else Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'))
    checks['memory_cgroup']=dict(ok=memory.exists() and (not unified or 'memory' in memory.read_text().split()),
                                 detail='cgroup v2' if unified else 'cgroup v1')
    child_env=dict(os.environ)
    if checks['driver_root']['ok']:
        child_env['LD_LIBRARY_PATH']=resolve(env['driver_root'],path.parent)+':'+child_env.get('LD_LIBRARY_PATH','')
    for key, command in commands.items():
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=30,env=child_env)
            checks[key]=dict(ok=result.returncode==0, detail=(result.stdout if result.returncode==0 else result.stderr)[-2000:])
        except (OSError,subprocess.TimeoutExpired) as exc:
            checks[key]=dict(ok=False,detail=str(exc))
    return dict(ready=all(c['ok'] for c in checks.values()),checks=checks)
