"""Native fixed SELECT/COMPOSE/full-feedback RO on the shared graph executor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def configure(inputs):
    config=inputs['evaluator'];provider=inputs['provider']
    core=Path(config['core']).resolve()
    sys.path[:0]=[str(HERE),str(core/'src'),str(core),str(HERE/'policy')]
    os.environ.update(RECCLAW_RECBOLE_ROOT=inputs['binding']['recbole_root'],RECCLAW_PYTHON_EXECUTABLE=config['worker_python'],RECCLAW_API_CONFIG=provider['api_config'])
    from graph_ro_contract import register_graph_contract
    register_graph_contract()
    # Bind this delivered transport before the shared launcher prepends paths.
    import ro_main_provider
    import run_native_json
    import no_cross_round_measurement
    import learning_components
    if Path(ro_main_provider.__file__).resolve()!=HERE/'ro_main_provider.py':
        raise RuntimeError('Graph RO imported a different native transport')
    for module in (no_cross_round_measurement,learning_components):
        if Path(module.__file__).resolve()!=HERE/'policy'/(module.__name__+'.py'):
            raise RuntimeError('Graph RO imported a different policy module: '+module.__name__)

def research_config(inputs):
    from graph_ro_metric import baseline_source
    from graph_ro_contract import GRAPH_SPLIT,GRAPH_EVALUATOR
    from graph_autospace import digest,METHOD
    from graph_search_adapter import REQUIREMENTS,DEPENDENCIES
    from recclaw_core.research_line.standalone import StandaloneResearchConfig,MAX_STANDALONE_ROUNDS
    launch,config,binding=inputs['launch'],inputs['evaluator'],inputs['binding']
    root=json.loads(Path(config['root_calibration']).read_bytes())['result']
    protocol=dict(task_id=binding['task_id'],training=binding['training'],evaluation=binding['evaluation'],
        data={k:v['sha256'] for k,v in binding['data_files'].items()},catalog_mapping_sha256=binding['catalog_mapping_sha256'],train_seed=config['train_seed'])
    protocol_digest=digest(protocol);protocol_ref=binding['task_id']+':graph-development-protocol'
    source=baseline_source(metrics=root['metrics'],source_ref=config['root_calibration'],source_sha256=inputs['hashes']['root'],
        comparator_ref=binding['task_id']+':original-root',comparator_digest=inputs['root_sha256'],protocol_digest=protocol_digest,seed=config['train_seed'])
    profile=dict(profile_id=binding['task_id']+':graph-model-loss',profile_kind='OFFLINE_TOPN',
        profile_digest=digest(dict(method=METHOD,protocol=protocol,parent_sha256=inputs['root_sha256'])))
    return StandaloneResearchConfig(repo_root=Path(config['core']),run_root=Path(config['output'])/'campaign',
        api_config_source=Path(inputs['provider']['api_config']),campaign_id=launch['campaign_id'],baseline_source=source,
        dataset=binding['dataset'],seed=config['train_seed'],search_seed=launch['search_seed'],epochs=binding['training']['epochs'],
        native_eval_step=binding['training']['eval_step'],native_stopping_step=binding['training']['patience'],
        timeout_seconds=config['worker_timeout_seconds'],watchdog_seconds=config['worker_timeout_seconds'],final_worker_ceiling_seconds=config['worker_timeout_seconds'],
        cuda_visible_devices=config['gpu_uuid'],research_mode='portfolio',native_resource_observation=True,
        round_count=MAX_STANDALONE_ROUNDS,observation_seed_schedule=(config['train_seed'],)*MAX_STANDALONE_ROUNDS,
        attempt_scheduler=True,max_attempts_per_round=1,implementation_total_token_ceiling_per_call=148000,
        proposal_output_token_ceiling_total_per_slot=20000,implementation_output_token_ceiling_total_per_candidate=80000,
        evaluator=GRAPH_EVALUATOR,split=GRAPH_SPLIT,frozen_profile_ref=profile,protocol_ref=protocol_ref,protocol_digest=protocol_digest,
        protocol_requirements=REQUIREMENTS,compatibility_requirements=REQUIREMENTS,available_dependencies=DEPENDENCIES,
        execution_purpose=binding['task_id']+':'+config['phase'])

def execute(inputs,resources):
    from graph_ro_generation import generate,freeze
    from graph_search_adapter import GraphSearchAdapter,graph_materializer_abi
    from graph_ro_execution import GraphROExecution
    from ro_main_provider import native_ro_provider
    from run_native_json import include_schema,json_transport
    from learning_components import FrozenSearchPolicy
    from recclaw_core.research_line.standalone import compose_standalone_campaign
    from recclaw_core.experiments.helix_abc_v1.lab_api_broker import LabApiCanaryBrokerV1
    space=generate(inputs);config=inputs['evaluator'];launch=inputs['launch'];output=Path(config['output'])
    adapter=GraphSearchAdapter(config,launch,inputs['facts'],space)
    native_config=research_config(inputs)
    original=LabApiCanaryBrokerV1.sealed_request_identity
    def identity(broker,**kwargs):return json_transport(original(broker,**kwargs))
    with native_ro_provider(config['provider_config']) as native_call:
        def provider_call(**kwargs):
            key=hashlib.sha256(json.dumps([str(kwargs['call_root']),kwargs['logical_call_id'],kwargs['session_id']]).encode()).hexdigest()
            path=output/'request_resource_context'/(key+'.json')
            if path.exists(): budget=json.loads(path.read_bytes())
            else:
                api=resources.api.summary()
                budget=dict(total_api_budget_usd=7,total_recorded_spent_usd=api['normalized_cost_usd'],
                    remaining_api_budget_usd=max(0,7-api['normalized_cost_usd']),scope='Fresh AutoSpace plus search share MAIN budget')
                freeze(path,budget)
            kwargs['prompt']+='\nCommon original graph facts:\n'+json.dumps(inputs['facts'],ensure_ascii=False)+'\nCommon literature:\n'+json.dumps(inputs['corpus'],ensure_ascii=False)+'\nActual API budget:\n'+json.dumps(budget)
            return native_call(**include_schema(kwargs))
        with graph_materializer_abi(),patch.object(LabApiCanaryBrokerV1,'sealed_request_identity',identity):
            composition=compose_standalone_campaign(native_config,provider_call=provider_call,launch=GraphROExecution(config),
                search_space_adapter=adapter,resume=native_config.run_root.exists())
            policy=FrozenSearchPolicy(composition).install()
            try: rounds=composition.run()
            finally:policy.close()
    return dict(status='NATIVE_RETURNED',returned_rounds=len(rounds),native_state_ref=str(native_config.run_root/'CAMPAIGN_STATE.json'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--common-binding',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();sys.path.insert(0,str(args.common_binding.resolve()))
    from graph_native_inputs import read_inputs,bind_imports
    from graph_native_launch import inspection,require_execution_environment,initialize_runtime,settle_result,print_result
    inputs=read_inputs(args.config,'ro');bind_imports(inputs['evaluator']);configure(inputs)
    from graph_ro_generation import prepare
    prepare(inputs);research_config(inputs)
    if not args.execute:
        print(json.dumps(inspection(inputs,'ro')));return
    require_execution_environment(inputs)
    from launch_state import launch_state
    from resource_stop import SearchResourceStop
    with launch_state(Path(inputs['launch']['output']),inputs['hashes']) as completed:
        if completed is not None:print_result(inputs,completed);return
        root,carrier,resources,ledger=initialize_runtime(inputs)
        try:result=execute(inputs,resources)
        except SearchResourceStop as error:result=dict(status='RESOURCE_STOP',reason=str(error))
        except Exception as error:
            import traceback
            (root/'EXECUTION_ERROR.log').write_text(traceback.format_exc())
            result=dict(status='EXECUTION_ERROR',reason=str(error),error_type=type(error).__name__)
        result=settle_result(inputs,result,resources,ledger)
    print_result(inputs,result)

if __name__=='__main__':main()
