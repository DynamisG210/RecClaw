"""Load actual selected graph state and evolved diagnostics in a confined worker."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(request):
    import torch
    from graph_worker import config_values,load_model,clear_cache
    from recbole.config import Config
    from recbole.data import create_dataset,data_preparation
    from recbole.data.interaction import Interaction
    from recbole.utils import init_seed
    binding=json.loads(Path(request['binding']).read_bytes())
    context=json.loads(Path(request['context']).read_bytes())
    evidence=request['evidence']
    if (sha(request['candidate'])!=evidence['source_sha256']
            or sha(evidence['checkpoint'])!=evidence['checkpoint_sha256']
            or evidence['catalog_mapping_sha256']!=binding['catalog_mapping_sha256']
            or evidence['train_seed']!=request['train_seed'] or evidence['test_accessed'] is not False):
        raise ValueError('Observation must use the actual selected model/source/checkpoint')
    if context['catalog_mapping_sha256']!=binding['catalog_mapping_sha256']:
        raise ValueError('Simulation users and graph model mapping differ')
    for key in ('train','item'):
        if context['source_sha256'][key]!=binding['data_files'][key]['sha256']:
            raise ValueError('Observation context changed')
    output=Path(request['output']);output.mkdir(exist_ok=False)
    os.chdir(output)
    data=output/'data'/binding['dataset'];data.mkdir(parents=True)
    for name in ('train','user','item'):
        asset=binding['data_files'][name]
        if sha(asset['path'])!=asset['sha256']: raise ValueError('Train/static input changed')
        shutil.copyfile(asset['path'],data/Path(asset['path']).name)
    model_class=load_model(request['candidate'])
    values=config_values(binding,output,request['train_seed'],request['device'])
    values['benchmark_filename']=['train','train','train']
    argv=sys.argv
    try:
        sys.argv=[argv[0]]
        config=Config(model=model_class,dataset=binding['dataset'],config_dict=values)
    finally: sys.argv=argv
    init_seed(request['train_seed'],True)
    dataset=create_dataset(config)
    from graph_fixed_mapping import apply_root_mapping
    apply_root_mapping(dataset,json.loads(Path(binding['catalog_mapping_path']).read_bytes()),binding['catalog_mapping_sha256'])
    train,a,b=data_preparation(config,dataset);del a,b
    mapping={field:list(map(str,dataset.field2id_token[field])) for field in ('user_id','item_id')}
    if hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()!=binding['catalog_mapping_sha256']:
        raise ValueError('Observation mapping drift')
    model=model_class(config,train._dataset).to(config['device'])
    saved=torch.load(evidence['checkpoint'],map_location=config['device'],weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.eval();clear_cache(model)
    spec=importlib.util.spec_from_file_location('graph_diagnostic_candidate',request['diagnosis'])
    diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
    diagnosis=diagnostic.DiagnosisProbe(model,config['device']).run_full_diagnosis(train)
    recommendations={}
    with torch.no_grad():
        for user in context['simulator_users']:
            interaction=Interaction({model.USER_ID:torch.tensor([int(user)],device=config['device'])})
            scores=model.full_sort_predict(interaction).reshape(-1).clone()
            scores[0]=-torch.inf
            history=context['user_train']['History'][user]
            scores[torch.tensor(history,device=config['device'])]=-torch.inf
            if torch.isfinite(scores).sum()<20:
                raise ValueError('Native simulation requires20 eligible catalog items')
            recommendations[user]=torch.topk(scores,20).indices.cpu().tolist()
    result=dict(status='SUCCESS',observation=dict(diagnosis=diagnosis,
        diagnosis_prompt=diagnostic.diagnosis_interpreter_prompt(diagnosis),recommendations=recommendations,
        checkpoint_sha256=evidence['checkpoint_sha256'],diagnosis_sha256=sha(request['diagnosis']),
        fit_performed=False,optimizer_steps=0,strict_state_dict_load=True,
        test_accessed=False,validation_rows_accessed=False,metric_role='native qualitative feedback only'))
    json.dumps(result,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_bytes())
    binding=json.loads(Path(request['binding']).read_bytes());sys.path.insert(0,binding['recbole_root'])
    try: result=observe(request)
    except Exception as error: result=dict(status='CANDIDATE_FAILURE',failure_stage='native_diagnosis',error=repr(error))
    args.result.write_text(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
