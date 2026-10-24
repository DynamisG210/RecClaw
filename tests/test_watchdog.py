"""CPU-only regression for prepared watchdogs; never admits a real GPU job."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

NATIVE = Path(__file__).resolve().parents[1] / 'src/recrsi/native'
BINDINGS = ['common/main_binding', 'task/search_source/private_binding',
            'task/research_line/private_binding']


def test_watchdog_copies_ship_the_same_owned_signal_helper():
    original = (NATIVE / 'task/research_line/private_operations/kill_owned.py').read_bytes()
    for binding in BINDINGS:
        assert (NATIVE / binding / 'kill_owned.py').read_bytes() == original


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux pidfd / E2 sudo boundary')
@pytest.mark.parametrize('binding', BINDINGS)
@pytest.mark.parametrize('mode', ['deadline', 'active_wall', 'natural'])
def test_prepared_watchdog_settles_owned_worker(tmp_path, binding, mode):
    if not shutil.which('sudo') or subprocess.run(
            ['sudo', '-n', 'true'], capture_output=True).returncode:
        pytest.skip('Requires the existing passwordless E2 sudo contract')
    # Copy just the prepared binding: do not accidentally import another lane.
    copied = tmp_path / 'binding'
    shutil.copytree(NATIVE / binding, copied)
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                             str(copied), mode, str(binding == BINDINGS[0])], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt['settled'] and receipt['mode'] == mode
    assert receipt['exit_code'] == (0 if mode == 'natural' else -9)


def exercise(binding, mode, finite_only):
    finite_only = finite_only == "True"
    import tempfile
    sys.path.insert(0, binding)
    import worker_watch as watch
    from active_wall import ActiveWallClock
    from gpu_job_budget import GPUJobBudget
    # /tmp is traversable by nobody; pytest's private parent directory is not.
    with tempfile.TemporaryDirectory(prefix='recrsi-watchdog-cpu-') as directory:
        root = Path(directory)
        root.chmod(0o777)
        pidfile = root / 'worker.pid'
        code = ('import os,time;open(' + repr(str(pidfile)) +
                ',"w").write(str(os.getpid()));time.sleep(' +
                ('0.3' if mode == 'natural' else '10') + ')')
        process = subprocess.Popen(['sudo', '-n', '-u', 'nobody', '/usr/bin/python3', '-I', '-c', code])
        pid = birth = fd = None
        helper = Path(binding) / 'kill_owned.py'
        try:
            for _ in range(100):
                if pidfile.exists() and pidfile.read_text():
                    break
                time.sleep(.01)
            pid = int(pidfile.read_text())
            birth = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
            fd = watch.open_pidfd(pid)
            if mode == 'deadline':
                wrong = subprocess.run(['sudo', '-n', '/usr/bin/python3', '-I', str(helper),
                                        str(pid), str(birth + 1)], capture_output=True)
                assert wrong.returncode != 0 and Path(f'/proc/{pid}').exists()
            identity = {'run': root.name, 'purpose': 'CPU_ONLY_SYNTHETIC_LEDGER'}
            # The legacy common binding is finite-only; private bindings also support None.
            budget = GPUJobBudget(root / 'budget.json', identity, max_gpu_seconds=10 if finite_only else None)
            requested = 1 if mode == 'deadline' else 10 if finite_only else None
            token, allowance = budget.reserve(requested, kind='cpu_fixture')
            started = time.time_ns()
            budget.started(token, started_ns=started, worker_pid=pid,
                           control_job=root / 'cpu-job', gpu_uuid='NONE')
            ActiveWallClock.create(root / 'clock.json', identity,
                                   limit_seconds=.3 if mode == 'active_wall' else 10 if finite_only else None)
            config = dict(budget_path=str(budget.path), budget_contract=budget.contract,
                          clock_path=str(root / 'clock.json'), clock_identity=identity,
                          token=token, allowance=allowance, requested=requested,
                          started_ns=started, timing_path=str(root / 'events.jsonl'),
                          job_id='fixture', result_path=str(root / 'result.json'))
            # Exercise the actual subprocess entry and inherited original pidfd.
            (root / 'config.json').write_text(json.dumps(config))
            watcher = subprocess.run([sys.executable, '-B', str(Path(binding) / 'worker_watch.py'),
                                      str(root / 'config.json'), str(fd)],
                                     pass_fds=(fd,), capture_output=True, text=True, timeout=5)
            assert watcher.returncode == 0, watcher.stderr
            rc = process.wait(timeout=3)
            row = json.loads(budget.path.read_text())['jobs'][token]
            assert row['finished_ns'] is not None and row['exit_code'] is None
            assert row['stopped_by_common_limit'] == (mode == 'active_wall')
            assert 0 <= row['gpu_seconds'] < 4
            budget.record_wait_status(token, rc)
            events = [json.loads(x) for x in (root / 'events.jsonl').read_text().splitlines()]
            assert len(events) == 1 and events[0]['event'] == 'END'
            print(json.dumps(dict(settled=True, mode=mode, exit_code=rc)))
        finally:
            if pid and birth and process.poll() is None:
                subprocess.run(['sudo', '-n', '/usr/bin/python3', '-I', str(helper),
                                str(pid), str(birth)], check=True)
                process.wait(timeout=3)
            if fd is not None:
                os.close(fd)


if __name__ == '__main__':
    exercise(*sys.argv[1:])
