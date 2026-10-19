"""Prospective common graph input binding for all six native controllers."""
import hashlib
import json
from pathlib import Path
import sys

PRICES={'gpt-5.6-luna':dict(input=.2,output=1.2,cache_read=.2,cache_creation=.2),
        'gpt-5.6-terra':dict(input=2.,output=12.,cache_read=2.,cache_creation=2.)}
EMBEDDING='text-embedding-3-small'
EMBEDDING_PRICES={EMBEDDING:dict(input=.02,output=0.,cache_read=.02,cache_creation=.02)}
CONTEXT_RULE=dict(history='common_train_only',ratings='common_train_only',
    mean_item_rating='computed_from_common_train_only',timestamps='train_seconds_converted_to_native_milliseconds',
    metadata=['item_id'],missing_fields=['title','release_year','category','review_text','price','store'],
    simulator_user_pool='all_nonempty_common_train_histories',simulator_user_count=20,
    simulator_seed_derivation='recclaw.external-comparison.e2-sim20.v1',
    mapping='measured_root_catalog_mapping',availability='same_raw_facts_available_to_all_six_methods')


def read_json(path): return json.loads(Path(path).read_bytes())
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_imports(evaluator):
    from graph_candidate_evaluator import configure
    return configure(evaluator,worker_process=False)


def read_inputs(path,method):
    launch=read_json(path);evaluator=read_json(launch['evaluator_config'])
    if method not in ('ro','rh','openevolve','shinka','deep','se') or launch['method']!=method:
        raise ValueError('Explicit native graph method required')
    binding=read_json(evaluator['binding']);identity=evaluator['run_identity']
    from graph_metric_contract import require_binding
    require_binding(binding)
    if (identity['environment']!=binding['task_id'] or identity['method']!=method
            or identity['train_seed']!=evaluator['train_seed']
            or identity['search_seed']!=launch['search_seed']
            or launch['training_seed']!=evaluator['train_seed'] or binding['evaluation']['seed']!=72026):
        raise ValueError('Graph method/task/train/search/evaluation identity mismatch')
    if evaluator['phase']=='FORMAL_MAIN':
        repeat=launch['repeat']
        if type(repeat) is not int or not 1<=repeat<=3 or launch['search_seed']!=41000+repeat or evaluator['train_seed']!=51000+repeat:
            raise ValueError('Formal graph MAIN retains original three paired seeds')
    else: raise ValueError('Prospective @5 consumer is formal-only; retain development @10 runtime')
    if method!='ro' and any(launch.get(key) is not None for key in ('frozen_autospace','autospace_contract','autospace_seed')):
        raise ValueError('Other native methods must not consume private RO generated spaces')
    provider=read_json(evaluator['provider_config']);limits=provider['api_limits']
    expected=dict(model_prices=PRICES,normalized_unit_usd=14,max_normalized_u=.5,
        max_requests=sys.maxsize,max_input_tokens=128000,max_output_tokens=20000,input_estimate_margin=4096)
    if any(limits.get(key)!=value for key,value in expected.items()): raise ValueError('MAIN API budget/pricing changed')
    if limits.get('auxiliary_model_prices',{})!=(EMBEDDING_PRICES if method=='shinka' else {}):
        raise ValueError('Only Shinka retains the approved embedding exception')
    if (provider['run_identity']!=identity or Path(launch['output']).resolve()!=Path(evaluator['output']).resolve()
            or Path(provider['resource_root']).resolve()!=Path(evaluator['output']).resolve()
            or Path(provider['active_wall_path']).resolve()!=Path(evaluator['active_wall_path']).resolve()
            or launch['cpus']!=evaluator['cpus']): raise ValueError('All native activities must share run resources')
    root=read_json(evaluator['root_calibration'])
    if (root['status']!='SUCCESS' or root['result'].get('primary_metric')!='NDCG@5' or root['result']['train_seed']!=evaluator['train_seed']
            or root['result']['catalog_mapping_sha256']!=binding['catalog_mapping_sha256']
            or root['result']['source_sha256']!=sha256(evaluator['parent_source'])
            or sha256(evaluator['root_calibration'])!=evaluator['root_calibration_sha256']):
        raise ValueError('Completed paired-seed measured root required')
    facts=read_json(launch['shared_facts']);corpus=read_json(launch['literature'])
    if (sha256(launch['shared_facts'])!=evaluator['common_facts_sha256']
            or sha256(launch['literature'])!=evaluator['literature_sha256']
            or facts['task_id']!=binding['task_id'] or facts['simulation_context']!=CONTEXT_RULE
            or facts['data_sha256']!={k:v['sha256'] for k,v in binding['data_files'].items()}
            or facts['catalog_mapping_sha256']!=binding['catalog_mapping_sha256']):
        raise ValueError('Graph common original facts or literature differ')
    if not corpus.get('documents'): raise ValueError('Common complete literature cards required')
    native_python=Path(launch['native_python']).absolute()
    if not native_python.is_file(): raise FileNotFoundError(native_python)
    for filename,expected in launch['native_source_sha256'].items():
        if sha256(filename)!=expected: raise ValueError('Pinned native source changed: '+filename)
    for filename,expected in evaluator['assets'].items():
        if sha256(filename)!=expected: raise ValueError('Graph binding source changed: '+filename)
    inputs={'launch':path,'evaluator':launch['evaluator_config'],'provider':evaluator['provider_config'],
        'binding':evaluator['binding'],'parent':evaluator['parent_source'],'root':evaluator['root_calibration'],
        'facts':launch['shared_facts'],'literature':launch['literature']}
    return dict(launch=launch,evaluator=evaluator,provider=provider,binding=binding,facts=facts,corpus=corpus,
        parent=Path(evaluator['parent_source']).read_bytes().decode('utf-8'),
        hashes={key:sha256(filename) for key,filename in inputs.items()},native_python=str(native_python),
        root_sha256=sha256(evaluator['parent_source']))


def task_prompt(inputs):
    from graph_program import common_task_text
    return ('\n'+common_task_text(inputs['binding'])+
        '\nEdit the complete candidate.py model source using your native research/edit mechanism. '
        'All generation/debugging/diagnosis/embedding/qualification/fit costs count toward the common '
        '12 effective evaluations excluding failures/root/cache,36 proposal and physical training safety limits with explicit incomplete status if exhausted,7 reference USD of successful API usage; failed requests remain recorded but do not consume this admission allowance. No active-wall, cumulative GPU or physical-stage time cap. Scientific epochs and early stopping remain unchanged; all actual costs remain recorded. '
        'The root is g0 but its full measured reference cost is charged. No test access. '
        'The original train/static facts and literature below are shared equally, not generated opponent research.\n'
        +json.dumps(inputs['facts'],ensure_ascii=False)+'\n'+json.dumps(inputs['corpus'],ensure_ascii=False))
