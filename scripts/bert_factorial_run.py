"""Run one preallocated historical factorial cell with native metering/admission."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    allocation = json.loads(Path(cfg['allocation']).read_text())
    root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('a+b') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'ATTEMPT.json').exists():
            raise RuntimeError('Existing attempt requires evidence-based recovery; no replay')
        sys.path[:0] = [cfg['binding'], str(Path(cfg['core'])/'src'), cfg['priority_dir'], cfg['rq2_runtime'], str(Path(cfg['rq2_runtime'])/'binder'), str(Path(cfg['binding']).parent), str(Path(cfg['binding']).parent/'comparison'), str(Path(allocation['shared_admission']['capacity_source']).parent)]
        from priority_capacity import gpu_capacity
        from gpu_cpu_capacity import occupied
        from active_wall import ActiveWallClock
        from gpu_job_budget import GPUJobBudget
        from bert_worker_runtime import BertRuntime
        hashes = {}
        for filename, expected in cfg['pins'].items():
            actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError('Input identity mismatch: ' + filename)
            hashes[filename] = actual
        manifest = json.loads((Path(cfg['data_root'])/'manifest.json').read_text())
        for name in ('train', 'dev'):
            path = Path(cfg['data_root'])/(name+'.jsonl')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != manifest[name+'_sha256']:
                raise ValueError('Data identity mismatch: ' + name)
            hashes[str(path)] = actual
        for directory, names in ((cfg['binding'], ('bert_worker_runtime.py', 'worker_job_time.py', 'worker_watch.py', 'gpu_job_budget.py', 'active_wall.py')),
                                 (cfg['priority_dir'], ('priority_capacity.py', 'gpu_cpu_capacity.py'))):
            for name in names:
                path = Path(directory)/name
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(root/'BINDING.json', dict(hashes=hashes, config=cfg, allocation=allocation))
        deadline = datetime.fromisoformat(allocation['created_utc'].replace('Z', '+00:00')).timestamp()+allocation['queue_admission_window_seconds']
        slots = Path(allocation['shared_admission']['slot_root'])
        original_affinity = os.sched_getaffinity(0)
        while time.time() < deadline:
            with ExitStack() as stack:
                busy = occupied()
                cpus = None
                for lo in range(0, 64, 4):
                    group = set(range(lo, lo+4))
                    if group & busy or not group <= original_affinity:
                        continue
                    leases = ExitStack()
                    try:
                        for name in sorted({f'CPU{lo:02d}_{lo+3:02d}.lock', f'CPU{lo}_{lo+3}.lock'}):
                            handle = leases.enter_context((slots/name).open('a+b'))
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        if group & occupied():
                            leases.close()
                            continue
                    except BlockingIOError:
                        leases.close()
                        continue
                    stack.enter_context(leases)
                    cpus = group
                    break
                if cpus:
                    env = dict(os.environ, LD_LIBRARY_PATH=cfg['driver_dir'])
                    rows = subprocess.check_output([str(Path(cfg['driver_dir'])/'nvidia-smi'), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], env=env, text=True)
                    choices = sorted((int(free.strip()), gpu.strip()) for gpu, free in (line.split(',') for line in rows.splitlines()))
                    for free, gpu in reversed(choices):
                        if free < allocation['required_gpu_mib']:
                            continue
                        try:
                            stack.enter_context(gpu_capacity(slots, gpu, cfg['driver_dir'], allocation['required_gpu_mib'], allocation['allocation_id']))
                        except (RuntimeError, BlockingIOError):
                            continue
                        cfg['gpu_uuid'] = gpu
                        os.sched_setaffinity(0, cpus)
                        identity = allocation['allocation_id']
                        class Resources:
                            def require_capacity(self):
                                self.clock.require_capacity()
                                if self.gpu.remaining_gpu_secs() < 1:
                                    raise RuntimeError('Finite GPU allowance unavailable')
                            def admit_output(self):
                                return self.clock.admit_output() and self.gpu.admit_output()
                        resources = Resources()
                        resources.clock = ActiveWallClock.create(root/'ACTIVE_WALL.json', identity, limit_seconds=allocation['cumulative_active_worker_seconds_cap'])
                        resources.gpu = GPUJobBudget(root/'GPU_BUDGET.json', identity, max_gpu_seconds=allocation['gpu_seconds_cap'])
                        save(root/'ATTEMPT.json', dict(pid=os.getpid(), birth_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19], started_ns=time.time_ns(), gpu=gpu, cpus=sorted(cpus), allocation_id=identity, command=sys.argv))
                        source = Path(cfg['candidate']).read_text()
                        try:
                            result = BertRuntime(cfg, resources, None)._train(cfg['candidate'], source, allocation['fit_wall_seconds_cap'])
                            save(root/'RESULT.json', result)
                        except BaseException as error:
                            save(root/'FAILURE.json', dict(type=type(error).__name__, error=str(error), time_ns=time.time_ns(), scientific_classification='UNDIAGNOSED', gpu_budget=resources.gpu.snapshot()))
                            raise
                        finally:
                            save(root/'COST.json', dict(gpu=resources.gpu.snapshot(), active=resources.clock.snapshot(), api_usd=0, gpu_invoice='UNKNOWN'))
                        return
            save(root/'QUEUE.json', dict(stage='WAITING_SHARED_CAPACITY', checked_ns=time.time_ns(), holding_resources=False))
            time.sleep(10)
        raise RuntimeError('Finite queue admission window expired')


if __name__ == '__main__':
    main()
