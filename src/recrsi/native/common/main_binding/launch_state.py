"""Single native search owner and idempotent delivery of a completed run.

Uses the existing native checkpoint, not a substitute research state. Checkpoint
loading alone does not restore pending requests or the native RNG/worker state.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

from durable_evaluation import write_json


def check_native_checkpoint(path, output):
    path = Path(path).resolve(strict=True)
    checkpoints = (Path(output)/'native/checkpoints').resolve()
    if path.parent != checkpoints:
        raise ValueError('Checkpoint is not from this run native checkpoint directory')
    metadata = json.loads((path/'metadata.json').read_text())
    programs = {}
    for file in (path/'programs').glob('*.json'):
        value = json.loads(file.read_text())
        if value['id'] in programs or not isinstance(value['code'], str):
            raise ValueError('Native checkpoint has duplicate or invalid programs')
        programs[value['id']] = value
    references = set(metadata['archive'])
    for island in metadata['islands']:
        references.update(island)
    references.update(x for x in metadata['island_best_programs'] if x is not None)
    if metadata['best_program_id'] is not None:
        references.add(metadata['best_program_id'])
    if not programs or references - programs.keys():
        # Upstream load otherwise skips unreadable programs/missing metadata and
        # reconstructs a partial archive, silently changing search history.
        raise ValueError('Native checkpoint is missing saved search programs')
    if type(metadata['last_iteration']) is not int or metadata['last_iteration'] < 0:
        raise ValueError('Native checkpoint has no valid iteration')
    return path


@contextmanager
def launch_state(output, input_hashes, *, checkpoint=None, native_resume_check=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'SEARCH_OWNER.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This native search already has an active owner') from error
        started, result_path = output/'STARTED.json', output/'RESULT.json'
        if started.exists() and json.loads(started.read_text())['input_hashes'] != input_hashes:
            raise ValueError('Run inputs differ from its original launch')
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['input_hashes'] != input_hashes:
                raise ValueError('Completed result belongs to different run inputs')
            yield result
            return
        if native_resume_check is not None and checkpoint is not None:
            raise ValueError('Use one native checkpoint protocol, not two')
        if started.exists() and native_resume_check is not None and (output/'native').exists():
            native_resume_check(output)
        elif started.exists() and checkpoint is None and (output/'native').exists():
            raise RuntimeError('Interrupted native search needs its original checkpoint; do not start afresh')
        if checkpoint is not None:
            if not started.exists():
                raise ValueError('Cannot import another run checkpoint into a fresh search')
            check_native_checkpoint(checkpoint, output)
        elif not started.exists():
            write_json(started, dict(input_hashes=input_hashes, started_ns=time.time_ns()))
        # Before OpenEvolve construction there is no native search state to
        # replay: reconnect the existing durable root job, retaining its clock.
        yield None
