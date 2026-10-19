"""One fresh revision validation, original first-repeat seeds; main dispatch only."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def wait_for_slot(handle,output,*,limit=21600,now=time.monotonic,sleep=time.sleep):
    started=now();status=Path(output)/'SPACE_SLOT_STATE.json'
    def record(value):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(status=value,wait_seconds=now()-started,limit_seconds=limit,charged_to_active_wall=True)))
        temp.replace(status)
    record('SPACE_READY_WAITING_GPU')
    while True:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            if now()-started>=limit:
                record('GPU_SLOT_WAIT_EXHAUSTED')
                raise TimeoutError('GPU_SLOT_WAIT_INTERRUPTED: retain generated space and charged costs; director reconciliation required')
            sleep(max(0,min(10,limit-(now()-started))))
        else:
            record('SPACE_READY_GPU_ACQUIRED');return

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def setup(m,repeat=1,active_launch=None):
    S=Path(m['source']);sys.path[:0]=[str(S),str(S/'ro'),str(H/'private_operations')]
    load('graph_native_inputs',H/'graph_native_inputs.py')
    from graph_native_inputs import read_inputs,bind_imports
    from current_resource_policy import prepare_evaluator
    prepare_evaluator()
    x=read_inputs(active_launch or H/f'repeat{repeat:02d}/LAUNCH.json','ro');bind_imports(x['evaluator'])
    owned=load('owned_evaluator',H/'private_operations/owned_evaluator.py');owned.install()
    module=load('_revision_native_entry',S/'ro/run_graph_ro_main.py')
    module.configure(x)
    from current_resource_policy import bind_research_config
    bind_research_config(module)
    from pilot_limits import bind
    bind(module)
    # Reuse the accepted mechanical qualification binding. No old candidate
    # content, space, history, broker cache or campaign is loaded.
    base=load('_existing_mechanical_binding',m['mechanical_binding'])
    base.install_contract(m['helper'])
    # One inner composition hook; the outer scientific/owner wrappers remain native.
    sys.path.insert(0,str(H/'future_graph_composition'))
    from composition import bind_execute, outer_scope
    provider=read(x['evaluator']['provider_config'])
    original=bind_execute(module.execute,H/'future_graph_composition',
                          provider['api_ledger'],Path(x['evaluator']['output'])/'repair_evidence',outer=False)
    requirements=load('_revision_requirements',m['requirements'])
    context=load('_revision_comparators',H/'root_context.py')
    def execute(inputs,resources):
        import graph_search_adapter as adapter
        from graph_ro_generation import freeze
        VERSION=m['version']
        from matched_arms import matched_arm
        import learning_components as comparison_policy_module
        from broker_failure import preserve_schema_failures
        from validation_boundary import stop_on_contract_failure
        freeze(Path(inputs['launch']['output'])/'METHOD_REVISION.json',dict(version=VERSION,
            evidence_class='FROZEN_GRAPH_SEARCH_VALIDATION',
            comparison_arm=m['comparison_arm'], comparison_protocol=m['comparison_protocol'], generation_arm=m['generation_arm'],
            paired_repeat=inputs['launch']['repeat'],comparator_version=context.INPUT_VERSION,
            root_refit=False,paid_history_reused=False))
        with outer_scope(H/'future_graph_composition',Path(inputs['evaluator']['output'])/'repair_evidence'), requirements.task_proposal_requirements(adapter),context.graph_root_context(adapter):
            from open_mechanism import open_mechanisms
            from fresh_counterproposal import fresh_counterproposal_scope
            import graph_ro_generation
            with fresh_counterproposal_scope(), open_mechanisms(adapter), matched_arm(adapter, comparison_policy_module, m['comparison_arm']), preserve_schema_failures(), stop_on_contract_failure(inputs['launch']['output']):
                return original(inputs,resources)
    module.execute=execute
    from runtime_bindings import install
    install(x['evaluator']['output'])
    return module,x

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--repeat',type=int,choices=(1,),required=True)
    p.add_argument('--gpu-uuid');p.add_argument('--cpus',type=int,nargs=4)
    args=p.parse_args();m=read(H/'MANIFEST.json')
    if args.execute and m.get('execution_release') != 'MAIN_RELEASED_WITH_ACTUAL_BUDGET':
        raise RuntimeError('Selection comparison inputs or actual-charge budget not released')
    for path,digest in m['pins'].items():assert sha(path)==digest,path
    bound=H/f'repeat{args.repeat:02d}'
    config=read(bound/'EVALUATOR_CONFIG.json');launch=read(bound/'LAUNCH.json')
    with ExitStack() as stack:
        active_launch=bound/'LAUNCH.json'
        if args.execute:
            assert args.gpu_uuid and args.gpu_uuid.startswith('GPU-') and len(set(args.cpus or []))==4
            lock=stack.enter_context((Path(m['slot_root'])/(args.gpu_uuid+'.lock')).open('a+b'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execution=bound/'execution'
            active_launch=execution/'LAUNCH.json'
            if execution.exists():
                config=read(execution/'EVALUATOR_CONFIG.json');launch=read(active_launch)
                if config['gpu_uuid']!=args.gpu_uuid or config['cpus']!=args.cpus:
                    raise ValueError('Resume requires the frozen physical placement')
            else:
                execution.mkdir()
                config.update(gpu_uuid=args.gpu_uuid,cpus=args.cpus)
                launch.update(cpus=args.cpus,evaluator_config=str(execution/'EVALUATOR_CONFIG.json'))
                (execution/'EVALUATOR_CONFIG.json').write_text(json.dumps(config,indent=2))
                active_launch.write_text(json.dumps(launch,indent=2))
                (execution/'HARDWARE_DELTA.json').write_text(json.dumps(dict(gpu_uuid=args.gpu_uuid,cpus=args.cpus,scientific_inputs_unchanged=True)))
            os.sched_setaffinity(0,set(args.cpus))
        module,x=setup(m,args.repeat,active_launch)
        import graph_autospace
        from pilot_limits import pilot_scope
        with pilot_scope():
            sys.argv=[module.__file__,'--config',str(active_launch),'--common-binding',m['source']]+(['--execute'] if args.execute else [])
            module.main()

if __name__=='__main__':
    from qualification_metadata import entry_metadata
    with entry_metadata(sys.modules[__name__]):main()
