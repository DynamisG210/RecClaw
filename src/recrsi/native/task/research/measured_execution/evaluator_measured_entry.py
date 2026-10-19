"""Same frozen worker bytes staged into the actual confined runtime."""
import importlib,inspect,json,sys
from pathlib import Path
N=Path(__file__).resolve().parent
B=N.parent.parent/'research_line'
sys.path.insert(0,str(B/'future_graph_composition'))
from continuation_binding import copy_runtime_module

def staged(module):
    source=inspect.getsource(module.run_stage)
    old="modules=[path for path in assets if path.parent==HERE and path.suffix=='.py']"
    assert source.count(old)==1
    source=source.replace(old,"modules=[path for path in assets if (path.parent==HERE or path==script) and path.suffix=='.py']")
    old='(runtime/path.name).write_bytes(path.read_bytes())';assert source.count(old)==1
    source=source.replace(old,'_copy_runtime_module(path,runtime,kind,_corrected_checker)')
    scope=dict(module.__dict__,_copy_runtime_module=copy_runtime_module,_corrected_checker=B/'future_graph_composition/graph_checkpoint_verify.py')
    exec(compile(source,str(N/'evaluator_measured_entry.py')+':runtime_copy','exec'),scope)
    return scope['run_stage']

def main():
    request=json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes());config=request['config']
    entries=[Path(p) for p in config['assets'] if Path(p).name=='graph_candidate_evaluator.py'];assert len(entries)==1
    entry=entries[0].resolve(strict=True);sys.argv[0]=str(entry);sys.path.insert(0,str(entry.parent))
    import graph_candidate_evaluator as evaluator
    evaluator.configure(config);sys.path.insert(0,str(B/'private_operations'))
    import owned_evaluator
    owned_evaluator.install();evaluator.run_stage=staged(evaluator)
    evaluator.main()
if __name__=='__main__':main()
