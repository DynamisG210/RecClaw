"""Stage-scoped capacity admission compatible with existing exclusive GPU locks.

Consumers hold the canonical lock shared and serialize capacity reservations under
a short admission lock. Unattributed GPU memory is never discounted. A reservation
is released only when the synchronous stage has returned and its children exited.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def snapshot(driver_root, gpu):
    env = dict(os.environ, LD_LIBRARY_PATH=str(driver_root) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    command = [str(Path(driver_root) / 'nvidia-smi'), '--id=' + gpu]
    free = int(subprocess.check_output(command + ['--query-gpu=memory.free', '--format=csv,noheader,nounits'], env=env, text=True).strip())
    text = subprocess.check_output(command + ['--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], env=env, text=True)
    usage = {}
    for line in text.splitlines():
        pid, memory = (v.strip() for v in line.split(','))
        usage[int(pid)] = int(memory)
    return free, usage


def descendants(root):
    """Only verified process lineage earns a reservation's materialized credit."""
    parents, aliases = {}, {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)
            pid = int(path.name)
            parents[pid] = int(fields['PPid'])
            aliases[pid] = {pid} | set(map(int, fields.get('NSpid', '').split()))
        except (OSError, KeyError, ValueError):
            continue
    owned = {root}
    while True:
        added = {pid for pid, parent in parents.items() if parent in owned} - owned
        if not added:
            break
        owned.update(added)
    return set().union(*(aliases.get(pid, {pid}) for pid in owned))


@contextmanager
def gpu_capacity(slot_root, gpu, driver_root, required_mib, task):
    if required_mib <= 0:
        raise ValueError('An explicit positive peak/cap plus margin is required')
    root = Path(slot_root)
    directory = root / (gpu + '.capacity')
    directory.mkdir(exist_ok=True)
    with (root / (gpu + '.lock')).open('a+b') as canonical:
        # Old EX consumers still exclude us; new cooperating consumers share.
        fcntl.flock(canonical, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with (directory / 'admission.lock').open('a+b') as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            live = []
            for receipt in directory.glob('*.json'):
                lock_path = receipt.with_suffix('.lock')
                with lock_path.open('a+b') as probe:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        live.append(json.loads(receipt.read_text()))
                    else:
                        receipt.unlink()
                        lock_path.unlink()
            free, usage = snapshot(driver_root, gpu)
            withheld = sum(max(0, r['required_mib'] - sum(usage.get(pid, 0) for pid in descendants(r['pid']))) for r in live)
            if free - withheld < required_mib:
                raise RuntimeError('Insufficient capacity: free=%d unmaterialized=%d requested=%d MiB' % (free, withheld, required_mib))
            receipt = directory / (uuid.uuid4().hex + '.json')
            lock_path = receipt.with_suffix('.lock')
            lease = lock_path.open('a+b')
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record = dict(pid=os.getpid(), task=task, required_mib=required_mib, free_mib=free, unmaterialized_mib=withheld, created_ns=time.time_ns())
            receipt.write_text(json.dumps(record))
        try:
            yield record
        finally:
            with (directory / 'admission.lock').open('a+b') as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                receipt.unlink()
                lease.close()
                lock_path.unlink()

