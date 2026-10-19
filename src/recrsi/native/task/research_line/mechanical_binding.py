"""Private original mechanical binding; resume validates the same qualification cache."""
import sys,hashlib
from pathlib import Path

def install_contract(helper):
    sys.path.insert(0,str(Path(helper).parent))
    import runtime_fix
    import graph_native_adapter as adapter
    original=adapter.evaluate_source
    def evaluate(config,source):
        parent=Path(config['parent_source']).read_text()
        source=adapter.single_source(source,parent)
        if source!=parent:
            digest=hashlib.sha256(source.encode()).hexdigest()
            candidate=Path(config['output'])/'programs'/(digest+'.py')
            candidate.parent.mkdir(parents=True,exist_ok=True)
            if candidate.exists():assert candidate.read_bytes()==source.encode()
            else:candidate.write_bytes(source.encode())
            config=runtime_fix.qualified_config(config,candidate)
        return original(config,source)
    adapter.evaluate_source=evaluate
