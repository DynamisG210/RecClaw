import importlib.util,sys
from pathlib import Path
def install_budget():
    root=Path(__file__).resolve().parent.parent/'private_binding'
    for name in ('evaluation_ledger','active_wall','gpu_job_budget','main_resources','worker_watch','worker_job_time'):
        path=root/(name+'.py')
        old=sys.modules.get(name)
        if old is not None and Path(old.__file__)==path:continue
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
