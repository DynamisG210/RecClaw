"""Persist worker exit/deadline observation independently of its controller.

Receives the already-open Linux pidfd for exactly the owned worker. No candidate
imports, model calls, training restarts, or reconstructed/zeroed historical cost.
"""
import json
import fcntl
import ctypes
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from active_wall import ActiveWallClock
from gpu_job_budget import GPUJobBudget


def _pidfd_syscall(number, *args):
    # This bundled Python lacks pidfd wrappers although the verified x86_64
    # Linux5.15 host supports them. Use its ABI, not a PID-reuse-unsafe fallback.
    # https://github.com/torvalds/linux/blob/v5.15/arch/x86/entry/syscalls/syscall_64.tbl
    if os.uname().machine != 'x86_64':
        raise RuntimeError('Prepared worker supervisor requires Linux x86_64')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def signal_owned_worker(pidfd):
    """Use the E2 sudo contract only when the original pidfd lacks permission."""
    try:
        _pidfd_syscall(424, ctypes.c_int(pidfd), ctypes.c_int(signal.SIGKILL),
                       ctypes.c_void_p(), ctypes.c_uint(0))
    except PermissionError:
        def target_pid():
            lines = Path(f'/proc/self/fdinfo/{pidfd}').read_text().splitlines()
            return int(next(line.split(':', 1)[1] for line in lines if line.startswith('Pid:')))

        pid = target_pid()
        if pid < 0:
            return
        try:
            birth = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        except FileNotFoundError:
            return
        if target_pid() != pid:
            return  # The original worker exited while its identity was read.
        subprocess.run(['sudo', '-n', '/usr/bin/python3', '-I',
                        str(Path(__file__).with_name('kill_owned.py')), str(pid), str(birth)],
                       check=True, timeout=10)


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(event, sort_keys=True) + '\n')
        stream.flush()


def observe(config, pidfd):
    budget = GPUJobBudget(config['budget_path'], **config['budget_contract'])
    clock = ActiveWallClock(config['clock_path'], config['clock_identity'])
    stopped = False
    poller = select.poll()
    if pidfd >= 0:
        poller.register(pidfd, select.POLLIN)
    while pidfd >= 0 and not poller.poll(50):
        expired_gpu = (time.time_ns()-config['started_ns'])/1e9 >= config['allowance']
        active = clock.snapshot()
        expired_active = active['stopped'] or active['remaining_seconds'] <= 0
        if expired_gpu or expired_active:
            stopped = expired_active or config['allowance'] < config['requested']
            try:
                signal_owned_worker(pidfd)
            except ProcessLookupError:
                pass
    finished_ns = time.time_ns()
    # pidfd reports physical death, not the wait status owned by the original
    # parent. Keep that status unknown until the parent supplies its real wait.
    budget.finish(config['token'], finished_ns=finished_ns, exit_code=None,
                  stopped_by_common_limit=stopped)
    result = dict(finished_ns=finished_ns, stopped_by_common_limit=stopped,
        exit_code=None, observation_resolution_seconds=.05, basis='OWNED_WORKER_PIDFD_EXIT')
    append_event(config['timing_path'], dict(event='END', job_id=config['job_id'], **result))
    path = Path(config['result_path'])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, sort_keys=True))
    temporary.replace(path)
    return result


if __name__ == '__main__':
    config = json.loads(Path(sys.argv[1]).read_text())
    pidfd = int(sys.argv[2])
    try:
        observe(config, pidfd)
    finally:
        if pidfd >= 0:
            os.close(pidfd)
