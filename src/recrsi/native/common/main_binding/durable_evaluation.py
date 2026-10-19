"""Keep the existing native evaluation alive across controller/pool exits.

This is infrastructure, not another research role or trainer. A held OS lock is
inherited by the detached evaluator; reconnecting callers wait for that same job.
An evaluator/host crash without a result remains unresolved, never relaunched.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


def job_request(config, program_path):
    source = None if program_path is None else Path(program_path).read_text(encoding='utf-8')
    # Native editors normalize line endings. An otherwise exact frozen parent
    # remains the already measured root, not another training opportunity.
    if source is not None and config.get('parent_source'):
        parent = Path(config['parent_source'])
        if source == parent.read_text(encoding='utf-8'):
            source = parent.read_bytes().decode('utf-8')
    digest = 'root' if source is None else hashlib.sha256(source.encode()).hexdigest()
    root = Path(config['output']).resolve() / 'native_jobs' / digest
    root.mkdir(parents=True, exist_ok=True)
    # Include the entire trusted binding, not only a candidate hash. Source is
    # copied before dispatch so an upstream temporary file may safely disappear.
    request = dict(config=config, source=source)
    return root, request


def _command(request_path, lock_fd):
    return [sys.executable, '-B', str(Path(__file__).resolve()),
            '--request', str(request_path), '--lock-fd', str(lock_fd)]


def run_native_job(config, program_path=None, *, command_factory=_command):
    root, request = job_request(config, program_path)
    request_path, result_path = root/'request.json', root/'result.json'
    # The child inherits this already-held open-file-description lock. There is
    # no PID reuse test or launch-then-publish race, and no controller-owned pipe.
    with (root/'running.lock').open('a+b') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.1)
        if request_path.exists():
            if json.loads(request_path.read_text()) != request:
                raise ValueError('Saved native job belongs to another source/configuration')
            if not result_path.exists():
                raise RuntimeError(f'Native evaluator interrupted without a delivered result: {root}')
        else:
            write_json(request_path, request)
            with (root/'evaluator.log').open('a') as log:
                # Only the trusted evaluator owns the inherited descriptor.
                process = subprocess.Popen(command_factory(request_path, lock.fileno()),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),), start_new_session=True)
            # close(), not LOCK_UN: explicitly unlocking would also release the
            # child's shared open-file-description lock.
            lock.close()
            with (root/'running.lock').open('a+b') as waiting:
                while True:
                    try:
                        fcntl.flock(waiting, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(.1)
            process.wait()
            if not result_path.exists():
                raise RuntimeError(f'Native evaluator interrupted without a delivered result: {root}')
    result = json.loads(result_path.read_text())
    if result['status'] == 'RESOURCE_STOP':
        from resource_stop import SearchResourceStop
        raise SearchResourceStop(result['message'])
    if result['status'] != 'RETURNED':
        raise RuntimeError(f"Native evaluator {result['error_type']}: {result['message']}; {root}")
    return result['value']


def execute_request(request_path, lock_fd):
    # Close-on-exec prevents model/worker descendants retaining the job lock.
    os.set_inheritable(lock_fd, False)
    sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
    from resource_stop import SearchResourceStop
    from native_evaluator import create_evaluator
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    try:
        evaluator = create_evaluator(request['config'])
        if request['source'] is None:
            value = evaluator.runtime.baseline
        else:
            candidate = request_path.parent/'program.py'
            candidate.write_text(request['source'], encoding='utf-8')
            value = evaluator.evaluate(candidate)
        result = dict(status='RETURNED', value=value)
    except SearchResourceStop as stop:
        result = dict(status='RESOURCE_STOP', message=str(stop))
    except Exception as error:
        result = dict(status='ERROR', error_type=type(error).__name__, message=str(error))
    # MainResultConsumer has already persisted original admission/count/order;
    # the result file is delivery only, never a fresh metric/admission decision.
    write_json(request_path.parent/'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    execute_request(args.request, args.lock_fd)
