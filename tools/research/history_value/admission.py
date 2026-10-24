"""Use the existing shared GPU admission and both canonical CPU lock aliases."""
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def admitted(home, cfg, shared):
    sys.path[:0] = [str(Path(shared['priority_source']).parent),
                   str(Path(shared['capacity_source']).parent)]
    from priority_capacity import gpu_capacity
    from gpu_cpu_capacity import occupied
    slots = Path(shared['slot_root'])
    driver = Path(cfg['driver_root'])
    deadline = time.monotonic() + 86400
    check_output = subprocess.check_output
    def bounded_query(*args, **kwargs):
        kwargs.setdefault('timeout', 10)
        return check_output(*args, **kwargs)
    while time.monotonic() < deadline:
        busy = occupied()
        env = dict(os.environ, LD_LIBRARY_PATH=str(driver))
        gpus = bounded_query([str(driver/'nvidia-smi'), '--query-gpu=uuid',
            '--format=csv,noheader'], env=env, text=True).splitlines()
        for first in range(0, os.cpu_count()-3, 4):
            cpus = set(range(first, first+4))
            if cpus & busy:
                continue
            for gpu in gpus:
                stack = ExitStack()
                try:
                    names = {f'CPU{first:02d}_{first+3:02d}.lock', f'CPU{first}_{first+3}.lock'}
                    for name in sorted(names):
                        lock = stack.enter_context((slots/name).open('a+b'))
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if cpus & occupied():
                        stack.close()
                        break
                    # A hung driver query must release our CPU/GPU locks and
                    # allow another physical GPU to be considered.
                    with patch.object(subprocess, 'check_output', bounded_query):
                        lease = stack.enter_context(gpu_capacity(slots, gpu.strip(), driver,
                                                                 9728, 'history_value'))
                except (BlockingIOError, RuntimeError, subprocess.TimeoutExpired) as error:
                    (home/'ADMISSION_LAST_REJECTION.json').write_text(json.dumps({
                        'gpu': gpu.strip(), 'time_ns': time.time_ns(), 'error': str(error)}))
                    stack.close()
                    continue
                with stack:
                    cfg.update(cpus=sorted(cpus), gpu_uuid=gpu.strip())
                    (home/'ADMISSION_LIVE.json').write_text(json.dumps({
                        'pid': os.getpid(), 'cpus': sorted(cpus), 'gpu': gpu.strip(),
                        'reservation': lease, 'scope': 'one synchronous arm, children must exit before release'}))
                    yield cfg
                    return
        (home/'ADMISSION_WAIT.json').write_text(json.dumps({'updated_ns': time.time_ns(),
            'status': 'WAITING_ACTUAL_CAPACITY_NO_GPU_CPU_HELD'}))
        time.sleep(15)
    raise RuntimeError('Finite admission window exhausted')
