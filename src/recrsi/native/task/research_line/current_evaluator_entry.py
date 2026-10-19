"""Actual Full evaluator entry with current timers and original checkpoint repair."""
import importlib,json,sys
import current_resource_policy
from pathlib import Path

def main():
    request=json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
    config=request['config'];branch=Path(config['output']).parents[1]
    entries=[Path(p) for p in config['assets'] if Path(p).name=='graph_candidate_evaluator.py']
    assert len(entries)==1
    entry=entries[0].resolve(strict=True);sys.argv[0]=str(entry)
    sys.path[:0]=[str(branch),str(entry.parent),str(branch/'future_graph_composition'),str(branch/'private_operations')]
    from current_resource_policy import prepare_evaluator
    prepare_evaluator()
    import graph_candidate_evaluator as evaluator
    evaluator.configure(config)
    import owned_evaluator
    owned_evaluator.install()
    import continuation_binding
    if json.loads(Path(config['binding']).read_bytes())['line']=='lightgcn':
        from plain_worker_line import install
        install(continuation_binding)
    with continuation_binding.evaluator_scope(evaluator,branch/'future_graph_composition/graph_checkpoint_verify.py'):
        importlib.import_module(entry.stem).main()

if __name__=='__main__':main()
