"""Private NGCF watcher: original accounting, privileged owned-worker signal."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    support = json.loads((here / 'SUPPORT.json').read_bytes())
    sys.path.insert(0, support['main_binding'])
    spec = importlib.util.spec_from_file_location('original_worker_watch', Path(support['main_binding']) / 'worker_watch.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    config, fd = json.loads(Path(sys.argv[1]).read_bytes()), int(sys.argv[2])
    pid, ticks = None, None
    if fd >= 0:
        info = Path('/proc/self/fdinfo/' + str(fd)).read_text().splitlines()
        pid = int(next(line.split(':', 1)[1] for line in info if line.startswith('Pid:')))
        if pid > 0:
            try:
                ticks = int(Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19])
            except FileNotFoundError:
                pass
    native_syscall = original._pidfd_syscall

    def signal_with_permission(number, *args):
        try:
            return native_syscall(number, *args)
        except PermissionError:
            if number != 424 or pid is None or ticks is None:
                raise
            subprocess.run(['sudo', '-n', '/usr/bin/python3', '-I', str(here / 'kill_owned.py'),
                            str(pid), str(ticks)], check=True, timeout=10)
            return 0

    original._pidfd_syscall = signal_with_permission
    try:
        original.observe(config, fd)
    finally:
        if fd >= 0:
            os.close(fd)


if __name__ == '__main__': main()
