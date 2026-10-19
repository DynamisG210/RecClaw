"""Original evaluator flow with the accepted cold evaluation runtime staged."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

N = Path(__file__).resolve().parent
B = N.parent.parent / 'research_line'
sys.path[:0] = [str(B), str(B / 'future_graph_composition'), str(N)]
import current_evaluator_entry
import continuation_binding
from evaluator_measured_entry import staged


def accepted_release():
    release_path = N / 'evaluation_acceleration_v1/RELEASE.json'
    release = json.loads(release_path.read_bytes())
    assert release['status'] == 'EQUIVALENT_DEV_ACCELERATION_ACCEPTED'
    for path, digest in release['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    return release, hashlib.sha256(release_path.read_bytes()).hexdigest()


@contextmanager
def accelerated_stage(module, corrected):
    release, release_sha = accepted_release()
    run_stage = staged(module)
    original_copy = continuation_binding.copy_runtime_module

    def copy_module(path, runtime, kind, checker):
        original_copy(path, runtime, kind, checker)
        if kind == 'training' and path.name == 'sampled_eval.py':
            original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            assert original_sha == release['reference_sampled_eval_sha256']
            replacements = {'sampled_eval.py': release['sampled_eval']}
            for name, source in replacements.items():
                assert source in release['pins']
                (runtime / name).write_bytes(Path(source).read_bytes())
            receipt = dict(runtime_version=release['runtime_version'],
                           release_sha256=release_sha,
                           original_sampled_eval_sha256=original_sha,
                           installed={name: release['pins'][source]
                                      for name, source in replacements.items()},
                           candidate_source_changed=False,
                           training_steps_changed=False,
                           users_candidates_and_frequency_changed=False)
            (runtime / 'EVALUATION_RUNTIME.json').write_text(
                json.dumps(receipt, ensure_ascii=True, indent=2))

    run_stage.__globals__['_copy_runtime_module'] = copy_module
    with patch.object(module, 'run_stage', run_stage):
        yield


if __name__ == '__main__':
    with patch.object(continuation_binding, 'evaluator_scope', accelerated_stage):
        current_evaluator_entry.main()
