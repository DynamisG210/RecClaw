"""Reuse accepted short SQLite storage mapping; keep logical request identities."""
import hashlib,json
from functools import wraps
from pathlib import Path

def install(output):
    from recclaw_core.experiments.helix_abc_v1 import fresh_r1
    if getattr(fresh_r1,'_graph_short_storage',False):return
    native=fresh_r1.bounded_provider_call
    output=Path(output)
    @wraps(native)
    def bounded_provider_call(**kwargs):
        original=Path(kwargs['call_root']);relative=original.relative_to(output)
        key=hashlib.sha256(str(relative).encode()).hexdigest()
        target=output/'broker_calls'/key
        record=dict(original_relative_path=str(relative),broker_relative_path=str(target.relative_to(output)))
        path=output/'broker_call_paths'/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert json.loads(path.read_bytes())==record
        else:path.write_text(json.dumps(record,indent=2)+'\n')
        return native(**dict(kwargs,call_root=target))
    fresh_r1.bounded_provider_call=bounded_provider_call
    fresh_r1._graph_short_storage=True
