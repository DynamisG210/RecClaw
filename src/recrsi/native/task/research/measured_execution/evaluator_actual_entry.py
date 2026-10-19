"""Hold new native evaluator processes until the cold runtime is accepted.

Already running evaluator/worker processes do not execute this entry again.
The native job, paid response and reservation remain pending, with raw wall time.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

N = Path(__file__).resolve().parent
REFERENCE_SHA = 'e528d41ce8863fbf74dd30f01076b43739d5f667e426773a9a3900a62a74a963'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    request = Path(sys.argv[sys.argv.index('--request') + 1]).resolve(strict=True)
    reference = N / 'evaluator_actual_entry_before_acceleration_gate.py'
    assert sha(reference) == REFERENCE_SHA
    gate = N / 'evaluation_acceleration_v1'
    gate.mkdir(exist_ok=True)
    identity = hashlib.sha256(str(request).encode()).hexdigest()
    receipt = gate / ('PENDING_' + identity + '.json')
    record = dict(status='WAITING_BEFORE_NEW_FIT_FOR_ACCEPTED_EVAL_RUNTIME',
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  request=str(request), request_sha256=sha(request),
                  pid=os.getpid(), parent_pid=os.getppid(),
                  reference_entry_sha256=REFERENCE_SHA,
                  new_API=0, new_training=0, GPU_reserved_by_gate=0,
                  scientific_opportunity_reset=False, wall_time_refunded=False)
    if receipt.exists():
        old = json.loads(receipt.read_bytes())
        assert old['request_sha256'] == record['request_sha256']
    else:
        with receipt.open('x') as f:
            json.dump(record, f, ensure_ascii=True, indent=2)
    release_path = gate / 'RELEASE.json'
    while not release_path.exists():
        time.sleep(5)
    release = json.loads(release_path.read_bytes())
    assert release['status'] == 'EQUIVALENT_DEV_ACCELERATION_ACCEPTED'
    assert release['reference_entry_sha256'] == REFERENCE_SHA
    assert release['no_TEST_development'] and release['scientific_protocol_unchanged']
    for path, digest in release['pins'].items():
        assert sha(path) == digest, path
    assert sha(request) == record['request_sha256']
    entry = Path(release['entry'])
    assert str(entry) in release['pins']
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
