"""CPU allocation shared by baseline and search controllers."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import re


@contextmanager
def claim_cpus(slots, cpus):
    import fcntl
    slots = Path(slots)
    with ExitStack() as stack:
        # Per-core locks also serialize overlapping noncontiguous allocations.
        paths = {slots / ('core-%d.lock' % cpu) for cpu in cpus}
        paths.update({slots/('CPU%02d_%02d.lock' % (min(cpus),max(cpus))),
                      slots/('CPU%d_%d.lock' % (min(cpus),max(cpus)))})
        for path in slots.glob('CPU*.lock'):
            match = re.fullmatch(r'CPU(\d+)_(\d+)\.lock', path.name)
            if match and cpus.intersection(range(int(match[1]),int(match[2])+1)):
                paths.add(path)
        for path in sorted(paths):
            handle = stack.enter_context(path.open('a+b'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
