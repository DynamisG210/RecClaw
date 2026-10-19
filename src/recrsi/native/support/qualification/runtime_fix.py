"""Observed graph integration repairs; frozen scientific worker code stays intact."""
import fcntl
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OLD_SETTLEMENT = "if gpu['reserved_seconds']:\n        raise RuntimeError('Physical graph work has unresolved timing; no new attempt')"
NEW_SETTLEMENT = "if gpu['jobs'][process.graph_budget_token].get('finished_ns') is None:\n        raise RuntimeError('This physical graph job has unresolved timing; no new attempt')"


def qualified_config(config, source):
    """Carry the same source's already-bound RO score contract into its cached qualifier."""
    if 'qualification_score_contract' in config:
        return config
    raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    request = Path(config['output']) / 'graph_qualifications' / digest / 'native_jobs/root/request.json'
    if not request.exists():
        return config
    saved = read(request)['config']
    cached = saved['graph_config']
    contract = cached.get('qualification_score_contract')
    if contract is None:
        return config
    if saved['source'].encode() != raw or {k:v for k,v in cached.items() if k!='qualification_score_contract'} != config:
        raise ValueError('Cached qualification source or scientific configuration differs')
    if contract['source_sha256'] != digest:
        raise ValueError('Cached score contract belongs to a different source')
    return dict(config, qualification_score_contract=contract)


def functions(module):
    original = inspect.getsource(module.run_stage)
    assert original.count(OLD_SETTLEMENT) == 1
    fixed = original.replace(OLD_SETTLEMENT, NEW_SETTLEMENT)
    scope = dict(module.__dict__)
    from resource_stop import SearchResourceStop
    scope['SearchResourceStop'] = SearchResourceStop
    exec(compile(fixed, str(HERE / 'runtime_fix.py') + ':job_local', 'exec'), scope)
    tail = fixed[fixed.index('    if not resources.admit_output():'):]
    finish = 'def finish_existing(config,binding,resources,source,outer,output,kind,code,process):\n' + tail
    exec(compile(finish, str(HERE / 'runtime_fix.py') + ':existing_receipt', 'exec'), scope)
    return scope['run_stage'], scope['finish_existing']


def install_worker():
    import graph_candidate_evaluator as module
    if getattr(module, '_job_local_fixed', False):
        return
    m = read(HERE / 'SOURCE.json')
    assert sha(module.__file__) == m['evaluator_sha256']
    stage, _ = functions(module)
    module.run_stage = stage
    original = inspect.getsource(module.evaluate_source)
    start = original.index('    resources.require_capacity()\n    try: token=ledger.reserve(digest)')
    end = original.index('    try:\n        qualify_source', start)
    replacement = '''    pending=[(token,row) for token,row in ledger.snapshot()['pending'].items()
             if row['source_sha256']==digest and 'result' not in row]
    if pending:
        if len(pending)!=1: raise RuntimeError('Ambiguous original pending evaluation')
        token,row=pending[0]
        resources.require_reserved_evaluation_capacity(digest)
        trial=Path(config['output'])/'graph_candidates'/token
        source=trial/'candidate.py'
        if sha(source)!=digest or (trial/'training').exists():
            raise RuntimeError('Original pending source/training needs reconciliation')
    else:
        resources.require_capacity()
        try: token=ledger.reserve(digest)
        except EvaluationLimitReached as error: raise SearchResourceStop(str(error)) from error
        trial=Path(config['output'])/'graph_candidates'/token;trial.mkdir(parents=True,exist_ok=False)
        source=trial/'candidate.py';source.write_bytes(Path(source_path).read_bytes())
'''
    scope = dict(module.__dict__)
    exec(compile(original[:start] + replacement + original[end:],
                 str(HERE / 'runtime_fix.py') + ':pending_token', 'exec'), scope)
    module.evaluate_source = scope['evaluate_source']
    import graph_qualification
    original_qualify = graph_qualification.qualify_program
    def qualify(config, source):
        return original_qualify(qualified_config(config, source), source)
    graph_qualification.qualify_program = qualify
    module._job_local_fixed = True


def install_adapter():
    import graph_native_adapter as adapter
    if getattr(adapter.single_source, '_terminal_newline_fixed', False):
        return
    original = adapter.single_source

    def single_source(program, parent):
        # Exactly one missing transport-final LF. Nothing else is normalized.
        if program.startswith(adapter.START) and program.endswith(adapter.END[:-1]):
            program += '\n'
        return original(program, parent)
    single_source._terminal_newline_fixed = True
    adapter.single_source = single_source


def protect_owned_evaluations(config):
    """Keep native concurrency and watchdog; a live metered durable child is not hung."""
    from shinka.core.async_runner import ShinkaEvolveRunner
    import graph_native_adapter as adapter
    install_adapter()
    original = ShinkaEvolveRunner._is_job_hung
    if getattr(original, '_durable_owned', False):
        return
    parent = Path(config['parent_source']).read_text()

    def is_hung(self, job, now=None):
        if not original(self, job, now):
            return False
        source = adapter.single_source(Path(job.exec_fname).read_text(), parent)
        digest = hashlib.sha256(source.encode()).hexdigest()
        lock = Path(config['output']) / 'native_jobs' / digest / 'running.lock'
        if lock.exists():
            with lock.open('r+b') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
        return True
    is_hung._durable_owned = True
    ShinkaEvolveRunner._is_job_hung = is_hung


def worker_main():
    assert sys.argv[1] == '--entry'
    entry = Path(sys.argv[2]).resolve()
    m = read(HERE / 'SOURCE.json')
    assert str(entry.parent) == m['source']
    assert entry.name in ('graph_candidate_evaluator.py', 'graph_qualification.py')
    sys.path.insert(0, str(entry.parent))
    request = Path(sys.argv[sys.argv.index('--request') + 1])
    job = read(request)['config']
    config = job.get('graph_config', job)
    sys.path[:0] = [config['main_binding'], config['comparison_root'], m['operations']]
    import owned_evaluator
    owned_evaluator.install()
    install_worker()
    name = entry.stem
    module = __import__(name)
    sys.argv = [str(entry)] + sys.argv[3:]
    module.main()


if __name__ == '__main__':
    worker_main()
