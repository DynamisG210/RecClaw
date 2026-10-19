"""Fresh finite measured-computation family, original execution and protocol."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
N=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeat',type=int,choices=(1,2,3),required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    release=read(N/'RELEASE.json')
    for path,expected in release['pins'].items():
        if sha(path)!=expected:raise ValueError('Frozen input changed: '+path)
    H=N.parent;B=Path(release['original_branch']);S=Path(release['original_source'])
    launch_path=N/f'repeat{args.repeat:02d}/LAUNCH.json';launch=read(launch_path)
    config=read(launch['evaluator_config']);provider_cfg=read(config['provider_config'])
    sys.path[:0]=[str(N),str(H/'measured_execution'),str(B/'private_operations'),str(H/'declaration'),str(H),str(H.parent),
        provider_cfg['ledger_module_dir'],str(S),str(S/'ro'),config['main_binding'],config['comparison_root'],str(B)]
    from budget_binding import install_budget
    install_budget()
    from measured_campaign import install_conservative_ledger,campaign_scope
    install_conservative_ledger()
    from endpoint2_only_runtime import install
    install(config['provider_config'])
    spec=importlib.util.spec_from_file_location('_measured_original_entry',B/'entry.py')
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry;spec.loader.exec_module(entry)
    m=dict(read(B/'MANIFEST.json'),version=release['version'],comparison_protocol=release['version'])
    from qualification_metadata import entry_metadata
    with entry_metadata(entry):
        module,inputs=entry.setup(m,args.repeat,launch_path)
        from actual_route_repair import install as repair_route
        repair_route()
        from prospective_revision import install as install_revision
        install_revision()
        from fresh_schema_continuation import install as install_schema
        install_schema()

        from recclaw_core.research_line import standalone
        import source_bound_context,counterproposal_decision,ro_main_provider
        import measured_context,measured_selector
        measured_context.install(source_bound_context)
        original_choice=measured_selector.load_original(B/'counterproposal_decision.py')
        original_compose=standalone.compose_standalone_campaign
        def compose(config,*,provider_call,launch,search_space_adapter,resume):
            source_bound_context.attach(search_space_adapter,Path(inputs['launch']['output'])/'source_bound_evidence')
            return original_compose(config,provider_call=provider_call,launch=launch,
                search_space_adapter=search_space_adapter,resume=resume)
        original_run=standalone.StandaloneResearchComposition.run
        consumer=N/f'repeat{args.repeat:02d}/CONSUMER_MANIFEST.json'
        def run(c,*a,**kw):
            with campaign_scope(c,consumer,sha(consumer),ro_main_provider):return original_run(c,*a,**kw)
        from pilot_limits import pilot_scope
        with patch.object(counterproposal_decision,'adapter_class',lambda base:measured_selector.adapter_class(base,original_choice)),\
             patch.object(standalone,'compose_standalone_campaign',compose),\
             patch.object(standalone.StandaloneResearchComposition,'run',run),pilot_scope():
            sys.argv=[module.__file__,'--config',str(launch_path),'--common-binding',str(S)]+(['--execute'] if args.execute else [])
            module.main()

if __name__=='__main__':main()
