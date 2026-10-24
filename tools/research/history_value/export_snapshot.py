"""Read one trusted native RQ3 checkpoint; no campaign execution or writes.

Run with the original controller environment and pass its recorded run directory.
The stdout export is local preparation evidence, not a runnable campaign clone.
"""
import argparse
import dataclasses
import hashlib
import json
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value).__name__)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export(home, round_index):
    sys.path[:0] = json.loads((home.parent / 'ORIGINAL_RUNTIME_PATHS.json').read_text())
    path = home / 'search/campaign' / f'ROUND_{round_index:02d}_CHECKPOINT.pkl'
    raw = path.read_bytes()
    record = pickle.loads(raw)  # Only original, trusted experiment artifacts.
    state = record.state_before
    context = plain(state.context)
    index = context['frontier']['incumbent_round_index']
    cid = context['frontier']['incumbent_candidate_id']
    trace_path = path.parent / f'ROUND_{index:02d}_TRACE.json'
    trace = json.loads(trace_path.read_bytes())
    assert any(a['candidate_id'] == cid for a in trace['summary']['attempts'])
    roots = [Path(p) for p in state.candidate_root_by_capability.values()
             if f'round-{index:04d}' in Path(p).parts]
    if len(roots) != 1:
        raise ValueError('Ambiguous incumbent source: bind successful attempt explicitly')
    files = {}
    for name in ('main_code.py', 'models.py', 'diagnosis_tools.py',
                 'self_evolverec_interface.py', 'morec_initialization.py'):
        data = (roots[0] / name).read_bytes()
        files[name] = {'sha256': sha(data), 'content': data.decode('utf-8')}
    inputs = json.loads((home / 'INPUTS.json').read_bytes())
    space = {}
    for name, expected in inputs['space']['file_sha256'].items():
        data = (Path(inputs['space']['package_root']) / name).read_bytes()
        if sha(data) != expected:
            raise ValueError(f'Frozen space bytes changed: {name}')
        space[name] = json.loads(data)
    return {'schema': 'recrsi.history-snapshot.v1', 'context': context,
            'incumbent_files': files, 'mechanism_space': space,
            'frozen_inputs': inputs,
            'provenance': {'checkpoint': str(path), 'checkpoint_sha256': sha(raw),
                           'incumbent_trace_sha256': sha(trace_path.read_bytes()),
                           'incumbent_root': str(roots[0]),
                           'cutoff': 'state_before; excludes this round and all later outcomes',
                           'exposure': 'historical development states and later outcomes previously seen; not independent unseen evidence'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=Path)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.home, args.round), ensure_ascii=False, sort_keys=True))
