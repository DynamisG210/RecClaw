"""Durable protected original-root calibration only. Owner assigns the slot.

No candidate admission or Provider is available here. Native search consumers
must separately qualify candidates and charge the measured root reference cost.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_runtime(closure):
    if sys.version != closure['python_version'] or sha(sys.executable) != closure['python_binary_sha256']:
        raise ValueError('Worker interpreter differs from the captured runtime')
    for path, expected in closure['source_files'].items():
        if sha(path) != expected:
            raise ValueError('Runtime source changed: ' + path)
    for expected in closure['dependencies']:
        dist = importlib.metadata.distribution(expected['name'])
        actual = dict(name=dist.metadata['Name'], version=dist.version)
        for filename, key in [('RECORD', 'record_sha256'), ('METADATA', 'metadata_sha256')]:
            data = dist.read_text(filename)
            actual[key] = None if data is None else hashlib.sha256(data.encode()).hexdigest()
        if actual != expected:
            raise ValueError('Dependency metadata changed: ' + expected['name'])


def setup(config):
    if sha(config['closure']) != config['closure_sha256']:
        raise ValueError('Runtime closure identity differs')
    closure = read(config['closure'])
    verify_runtime(closure)
    for path, expected in config['assets'].items():
        if sha(path) != expected:
            raise ValueError('Calibration asset changed: ' + path)
    sys.path[:0] = [closure['main_binding'], closure['comparison_root']]
    return closure


class NoProviderRoot:
    """Root-only zero API cost: this launcher never loads a provider client."""
    def summary(self):
        return dict(unheld_unknown_usage_failures=0, fatal_provider_statuses=[], normalized_cost_usd=0.)

    def admit_output(self):
        return True


def make_boundary(config, closure, resources):
    from gagc.e2_boundary import E2Boundary
    from worker_job_time import TimedWorkerBoundary
    from worker_watch import append_event

    class MeteredGraphBoundary(E2Boundary):
        _durable_watch = TimedWorkerBoundary._durable_watch

        def popen(self, command, **kwargs):
            self.active_clock.resume()
            self.active_clock.require_capacity()
            requested = 21600
            token, allowance = self.resource_budget.reserve(requested, kind='graph_root_calibration')
            # Bootstrap interruption deliberately retains unresolved reservation.
            process = super().popen(command, **kwargs)
            job_path = Path(process.args[-1])
            started_ns = Path(read(job_path)['worker_pid_path']).stat().st_mtime_ns
            self.resource_budget.started(token, started_ns=started_ns,
                worker_pid=process.worker_pid, control_job=job_path, gpu_uuid=self.gpu_uuid)
            job_id = uuid.uuid4().hex
            append_event(self.timing_path, dict(event='START', job_id=job_id,
                started_ns=started_ns, pid=process.worker_pid, gpu_uuid=self.gpu_uuid,
                cards=1, control_job=str(job_path), basis='DEVICE_BOUND_NATIVE_WORKER_EXECUTION',
                clock='HOST_REALTIME_NS'))
            return self._durable_watch(process, token, allowance, requested, started_ns, job_id)

    boundary = MeteredGraphBoundary(worker_python=Path(closure['python']),
        driver_root=Path(config['driver_root']), control_root=Path(config['output'])/'control',
        launcher=Path(closure['comparison_root'])/'helpers/se_namespace_exec.py',
        gpu_uuid=config['gpu_uuid'], cpus=config['cpus'])
    boundary.resource_entry = HERE/'graph_confined_entry.py'
    boundary.resource_budget, boundary.active_clock = resources.gpu, resources.clock
    boundary.timing_path = Path(config['output'])/'GPU_WORKER_EVENTS.jsonl'
    return boundary


def termination(result, gpu, clock, exit_code):
    if gpu['gpu_seconds'] >= gpu['contract']['max_gpu_seconds']:
        return 'GPU_LIMIT'
    if not clock.admit_output():
        return 'ACTIVE_LIMIT'
    if exit_code != 0 or result is None:
        return 'WORKER_FAILURE'
    return result['stop_reason']


def evaluate_root(config, closure):
    from active_wall import ActiveWallClock
    from main_resources import MainResources
    from evaluation_ledger import EvaluationLedger
    from durable_evaluation import write_json
    from resource_stop import SearchResourceStop
    from graph_worker import inspect_binding
    from graph_program import primary_score

    root = Path(config['output'])
    binding = read(config['binding'])
    inspect_binding(binding, execute=True)
    if sha(config['parent_source']) != binding['root_source_sha256']:
        raise ValueError('Only the bound original root may use this no-Provider launcher')
    clock_path = root/'ACTIVE_WALL.json'
    ActiveWallClock.create(clock_path, config['identity'])
    resources = MainResources(root, config['identity'], clock_path, NoProviderRoot())
    ledger = EvaluationLedger(root/'EVALUATION_LEDGER.json',
        root_sha256=binding['root_source_sha256'], identity=config['identity'])
    resources.require_capacity()
    token = ledger.reserve(binding['root_source_sha256'])
    resources.require_reserved_evaluation_capacity(binding['root_source_sha256'])
    outer = root/'attempts/root'
    outer.mkdir(parents=True, exist_ok=False)
    boundary = make_boundary(config, closure, resources)
    command = [closure['python'], '-B', '-u', str(HERE/'graph_worker.py'),
        '--binding', config['binding'], '--candidate', config['parent_source'],
        '--expected-source-sha', binding['root_source_sha256'], '--output', str(outer/'worker'),
        '--train-seed', str(binding['train_seed']), '--device', 'cuda', '--role', 'root', '--execute']
    env = dict(PATH='/usr/bin:/bin', CUDA_VISIBLE_DEVICES=config['gpu_uuid'],
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        HOME=str(outer), TMPDIR=str(outer), PYTHONHASHSEED=str(binding['train_seed']))
    readonly = [HERE/'graph_worker.py', HERE/'graph_confined_entry.py',
                Path(config['binding']), Path(config['parent_source']), Path(binding['recbole_root'])]
    readonly += [Path(row['path']) for row in binding['data_files'].values()]
    exit_code, error = None, None
    try:
        with (outer/'worker.log').open('ab') as log:
            process = boundary.popen(command, cwd=outer, env=env, readonly=readonly,
                writable=[outer], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            exit_code = process.wait()
    except SearchResourceStop as exc:
        error = str(exc)
    except Exception as exc:
        error = repr(exc)
    gpu = resources.gpu.snapshot()
    result_path = outer/'worker/RESULT.json'
    result = read(result_path) if result_path.exists() else None
    reason = termination(result, gpu, resources.clock, exit_code)
    # No valid root is delivered from a timeout, incomplete fit or unresolved cost.
    valid = (error is None and exit_code == 0 and result is not None
             and reason in ('EPOCH_LIMIT', 'EARLY_STOP') and gpu['reserved_seconds'] == 0)
    if valid:
        receipt = read(outer/'BOUNDARY.json')
        expected = dict(uid=65534, no_new_privs='1', memory_limit_bytes=32 * 1024 ** 3,
            cpu_affinity=sorted(config['cpus']), network='isolated', network_interfaces=['lo'],
            torch_allocator_limit_bytes=9 * 1024 ** 3, torch_threads=4)
        if any(receipt.get(key) != value for key, value in expected.items()):
            valid, error, reason = False, 'Protected boundary receipt mismatch', 'BOUNDARY_FAILURE'
    score = primary_score(result, binding) if valid else None
    admitted = bool(valid and resources.admit_output())
    cost = dict(gpu_seconds=gpu['gpu_seconds'], active_seconds=resources.clock.snapshot()['active_seconds'],
                physical_execution_repeated=False, reference_charge_required=True)
    evidence = dict(phase='ROOT_CALIBRATION', termination=reason, exit_code=exit_code,
        error=error, physical_cost=cost, unresolved_gpu_seconds=gpu['reserved_seconds'],
        worker_result=str(result_path), boundary=str(outer/'BOUNDARY.json'),
        curve=str(outer/'worker/LEARNING_CURVE.jsonl'), binding_sha256=sha(config['binding']),
        closure_sha256=config['closure_sha256'], test_accessed=False)
    value = dict(status='SUCCESS' if valid and admitted else 'NOT_ADMITTED', result=result,
        artifacts=dict(source_sha256=binding['root_source_sha256'], native_evidence=evidence))
    ledger.ready(token, value, admitted=admitted)
    ledger.finish(token, valid=valid, primary_metric=score, admitted=admitted, evidence=evidence)
    write_json(root/'ROOT_CALIBRATION.json', value)
    return value


def execute_request(path, lock_fd):
    os.set_inheritable(lock_fd, False)
    request = read(path)
    # No program/candidate path can enter this trusted original-root entry.
    if request['source'] is not None:
        raise ValueError('Root calibration does not accept candidates')
    config = request['config']
    closure = setup(config)
    from durable_evaluation import write_json
    from resource_stop import SearchResourceStop
    try:
        value = evaluate_root(config, closure)
        result = dict(status='RETURNED', value=value)
    except SearchResourceStop as exc:
        result = dict(status='RESOURCE_STOP', message=str(exc))
    except Exception as exc:
        result = dict(status='ERROR', error_type=type(exc).__name__, message=str(exc))
    write_json(Path(path).parent/'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.request:
        execute_request(args.request, args.lock_fd)
        return
    config = read(args.config)
    closure = setup(config)
    from graph_worker import inspect_binding
    inspect_binding(read(config['binding']), execute=True)
    if not args.execute:
        print(json.dumps(dict(runtime_verified=True, training_started=False,
                              allocation_assigned=bool(config.get('gpu_uuid') and config.get('cpus')))))
        return
    if (not str(config.get('gpu_uuid', '')).startswith('GPU-')
            or len(set(config.get('cpus', []))) != 4
            or any(type(cpu) is not int or cpu < 0 for cpu in config['cpus'])):
        raise ValueError('Owner must bind a free GPU UUID and four distinct CPU cores')
    from durable_evaluation import run_native_job
    result = run_native_job(config, command_factory=lambda request, fd: [closure['python'],
        '-B', str(HERE/'graph_root_launcher.py'), '--request', str(request), '--lock-fd', str(fd)])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
