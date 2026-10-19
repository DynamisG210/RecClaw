"""Route this recovery's original evaluators to its private watcher only."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def install():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from budget_binding import install_budget
    install_budget()
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from current_resource_policy import install_api_policy
    install_api_policy()
    here = Path(__file__).resolve().parent
    support = json.loads((here / 'SUPPORT.json').read_bytes())
    # The private executable has the same basename; bind utility imports to the
    # original module before importing its unchanged TimedWorkerBoundary.
    spec = importlib.util.spec_from_file_location('worker_watch', Path(support['main_binding']) / 'worker_watch.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['worker_watch'] = module
    spec.loader.exec_module(module)
    import durable_evaluation
    import worker_job_time
    # The original function derives its watcher path from its module __file__.
    # No shared file is edited; the original function and accounting stay intact.
    worker_job_time.__file__ = str(here / 'worker_job_time.py')
    original = durable_evaluation.run_native_job

    def run(config, program_path=None, *, command_factory=durable_evaluation._command):
        def command(request, fd):
            argv = command_factory(request, fd)
            matches = [(i, p) for i,p in enumerate(argv) if Path(str(p)).name in
                       ('graph_candidate_evaluator.py', 'graph_qualification.py')]
            if matches:
                assert len(matches) == 1
                index, entry = matches[0]
                argv = argv[:index] + [str(here / 'owned_evaluator.py'), '--entry', str(entry)] + argv[index+1:]
            return argv
        return original(config, program_path, command_factory=command)
    durable_evaluation.run_native_job = run


def main():
    assert sys.argv[1] == '--entry'
    entry = Path(sys.argv[2]).resolve()
    sys.argv = [str(entry)] + sys.argv[3:]
    request = json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
    config = request['config'].get('graph_config', request['config'])
    sys.path.insert(0, str(entry.parent))
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from current_resource_policy import prepare_evaluator
    prepare_evaluator()
    from graph_candidate_evaluator import configure
    configure(config)
    install()
    importlib.import_module(entry.stem).main()


if __name__ == '__main__': main()
