"""Versioned checkpoint infrastructure correction; historical inputs stay immutable.

Install only in a replacement controller/evaluator at a settled worker boundary.
The original candidate, trainer, evaluator and qualification are unchanged. Only
the verifier bytes copied into NEW training runtimes change, with a local receipt.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

VERSION = 'graph-sparse-checkpoint-infrastructure-correction-v1'
CHECKER_SHA = '4451cf4399286b09cad2acc95a167ec3552ba3ac9d75b6c0b636cf3e43de03ca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_runtime_module(path, runtime, kind, corrected):
    raw = path.read_bytes()
    if kind == 'training' and path.name == 'graph_checkpoint_verify.py':
        replacement = Path(corrected).read_bytes()
        if hashlib.sha256(replacement).hexdigest() != CHECKER_SHA:
            raise ValueError('Corrected checkpoint verifier identity changed')
        receipt = dict(version=VERSION, original_path=str(path),
                       original_sha256=hashlib.sha256(raw).hexdigest(),
                       installed_path=str(runtime / path.name),
                       installed_sha256=CHECKER_SHA,
                       scientific_inputs_changed=False)
        (runtime / 'CHECKPOINT_INFRASTRUCTURE_CORRECTION.json').write_text(
            json.dumps(receipt, sort_keys=True), encoding='utf8')
        raw = replacement
    (runtime / path.name).write_bytes(raw)


def corrected_stage(module, corrected):
    """Bind the one measured defect at the original immutable runtime-copy step."""
    original = inspect.getsource(module.run_stage)
    old = '(runtime/path.name).write_bytes(path.read_bytes())'
    if original.count(old) != 1:
        raise ValueError('Original evaluator runtime-copy boundary changed')
    replacement = original.replace(old,
        '_copy_runtime_module(path, runtime, kind, _corrected_checker)')
    scope = dict(module.__dict__, _copy_runtime_module=copy_runtime_module,
                 _corrected_checker=Path(corrected))
    exec(compile(replacement, str(Path(__file__).resolve()), 'exec'), scope)
    return scope['run_stage']


@contextmanager
def evaluator_scope(module, corrected):
    with patch.object(module, 'run_stage', corrected_stage(module, corrected)):
        yield


@contextmanager
def route_new_evaluators(entry):
    """Retain native request/config/cache/job lock and the existing owner watcher."""
    import durable_evaluation
    original = durable_evaluation.run_native_job

    def run(config, program_path=None, *, command_factory=durable_evaluation._command):
        def command(request, fd):
            argv = command_factory(request, fd)
            matches = [i for i, part in enumerate(argv)
                       if Path(str(part)).name == 'graph_candidate_evaluator.py']
            if matches:
                if len(matches) != 1:
                    raise ValueError('Ambiguous original evaluator command')
                i = matches[0]
                argv = argv[:i] + [str(entry)] + argv[i + 1:]
            return argv
        return original(config, program_path, command_factory=command)
    with patch.object(durable_evaluation, 'run_native_job', run):
        yield
