"""Run the registered two-arm stage serially; never replay a started arm."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('home', type=Path)
    p.add_argument('original', type=Path)
    a = p.parse_args()
    with (a.home/'QUEUE.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm in ('retain', 'reset'):
            if (a.home/arm/'search/STAGE_STARTED.json').exists():
                raise RuntimeError('Started arm requires explicit diagnosis, not replay')
            (a.home/'QUEUE_PROGRESS.json').write_text(json.dumps({
                'pid': os.getpid(), 'arm': arm, 'stage': 'RUNNING_OR_WAITING_ADMISSION',
                'updated_ns': time.time_ns()}))
            with (a.home/(arm+'_PROCESS.log')).open('xb') as log:
                result = subprocess.run([sys.executable, '-B', '-u', str(a.home/'runner.py'),
                    str(a.home), str(a.original), arm], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                (a.home/'QUEUE_ERROR.json').write_text(json.dumps({
                    'arm': arm, 'returncode': result.returncode, 'updated_ns': time.time_ns()}))
                raise SystemExit(result.returncode)
        (a.home/'QUEUE_COMPLETE.json').write_text(json.dumps({'finished_ns': time.time_ns()}))
