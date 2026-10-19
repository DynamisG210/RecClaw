import os, re, fcntl
from pathlib import Path

def occupied_cpus():
    busy = set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            affinity = os.sched_getaffinity(int(process.name))
            if len(affinity) <= 8 and (process/'cmdline').read_bytes():
                busy.update(affinity)
        except (OSError, PermissionError):
            pass
    return busy

def claim_test_cpu(slot_root):
    # Match the existing shared CPU dispatcher: actual affinity plus all aliases.
    busy = occupied_cpus()
    for first in range(0, 64, 4):
        cpus = set(range(first, first+4))
        if cpus & busy:
            continue
        paths = {slot_root/f'CPU{first:02d}_{first+3:02d}.lock',
                 slot_root/f'CPU{first}_{first+3}.lock'}
        for existing in slot_root.glob('CPU*.lock'):
            match = re.fullmatch(r'CPU(\d+)_(\d+)\.lock', existing.name)
            if match and cpus.intersection(range(int(match[1]), int(match[2])+1)):
                paths.add(existing)
        handles = []
        try:
            for path in sorted(paths):
                handle = path.open('a+b'); handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if cpus & occupied_cpus():
                raise BlockingIOError()
            return sorted(cpus), handles
        except BlockingIOError:
            for handle in handles:
                handle.close()
    return None
