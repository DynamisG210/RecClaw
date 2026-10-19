"""Train-only one-batch model ABI check inside the metered graph boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def run(binding, source, output, seed, device, score_contract=None):
    from graph_candidate_contract import qualify_source
    from graph_worker import config_values, load_model, clear_cache, write_json
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset,data_preparation
    from recbole.utils import init_seed
    qualify_source(source.read_text(encoding='utf-8'))
    if score_contract is not None and score_contract['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Scoring contract belongs to a different candidate source')
    output.mkdir(parents=True,exist_ok=False)
    os.chdir(output)
    data=output/'data'/binding['dataset'];data.mkdir(parents=True)
    for name in ('train','user','item'):
        asset=binding['data_files'][name]
        if hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest()!=asset['sha256']:
            raise ValueError('Qualification input changed: '+name)
        shutil.copyfile(asset['path'],data/Path(asset['path']).name)
    model_class=load_model(source)
    values=config_values(binding,output,seed,device)
    # Qualification has no dev loader or validation score. RecBole's mandatory
    # three-way API receives TRAIN aliases, discarded immediately below.
    values['benchmark_filename']=['train','train','train']
    saved_argv=sys.argv
    try:
        sys.argv=[sys.argv[0]]
        config=Config(model=model_class,dataset=binding['dataset'],config_dict=values)
    finally: sys.argv=saved_argv
    init_seed(seed,True)
    dataset=create_dataset(config)
    from graph_fixed_mapping import apply_root_mapping
    frozen_mapping=json.loads(Path(binding['catalog_mapping_path']).read_bytes())
    apply_root_mapping(dataset,frozen_mapping,binding['catalog_mapping_sha256'])
    train,alias1,alias2=data_preparation(config,dataset)
    del alias1,alias2
    mapping={field:list(map(str,dataset.field2id_token[field])) for field in ('user_id','item_id')}
    mapping_sha=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
    if mapping_sha!=binding['catalog_mapping_sha256']:
        raise ValueError('Train qualification mapping differs from the frozen root')
    init_seed(seed,True)
    model=model_class(config,train._dataset).to(config['device'])
    batch=next(iter(train)).to(config['device'])
    model.train()
    loss=model.calculate_loss(batch)
    score_checks = None
    if score_contract is not None:
        from graph_score_connection import check_graph_score_contract
        score_checks=check_graph_score_contract(model,loss,batch,score_contract)
    loss=sum(loss) if isinstance(loss,tuple) else loss
    if loss.ndim or not torch.isfinite(loss): raise ValueError('Loss must be finite scalar')
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError('Missing or nonfinite train gradients')
    model.zero_grad(set_to_none=True)
    model.eval();clear_cache(model)
    probe=batch[:2]
    with torch.no_grad():
        point=model.predict(probe)
        scores=model.full_sort_predict(probe).reshape(len(probe),dataset.item_num)
        if point.numel()!=len(probe) or not torch.isfinite(point).all() or not torch.isfinite(scores).all():
            raise ValueError('Invalid predict/full-sort score shape or values')
        selected=scores.gather(1,probe[model.ITEM_ID].reshape(-1,1)).reshape(-1)
        torch.testing.assert_close(point.reshape(-1),selected,rtol=1e-5,atol=1e-6)
    checkpoint=output/'qualification_state.pt'
    torch.save(model.state_dict(),checkpoint)
    restored=model_class(config,train._dataset).to(config['device'])
    restored.load_state_dict(torch.load(checkpoint,map_location=config['device'],weights_only=True))
    restored.eval();clear_cache(restored)
    with torch.no_grad():
        torch.testing.assert_close(scores,restored.full_sort_predict(probe).reshape_as(scores),rtol=1e-5,atol=1e-6)
    result=dict(status='QUALIFIED',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        catalog_mapping_sha256=mapping_sha,train_seed=seed,optimizer_steps=0,
        forward_backward_batches=1,validation_rows_accessed=False,test_accessed=False,
        checkpoint_reload_verified=True)
    if score_contract is not None:
        result['score_contract_digest'] = hashlib.sha256(json.dumps(score_contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        result['score_connection_checks'] = score_checks
    write_json(output/'QUALIFICATION.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('binding','candidate','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--train-seed',type=int,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],required=True)
    parser.add_argument('--score-contract-json')
    args=parser.parse_args()
    binding=json.loads(args.binding.read_bytes())
    sys.path.insert(0,binding['recbole_root'])
    # Trusted dependencies are environment failures, never model feedback.
    try:
        import graph_candidate_contract, graph_worker, graph_fixed_mapping
        import torch
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        result=dict(status='INFRASTRUCTURE_FAILURE',error=repr(error),
                    failure_stage='trusted_bootstrap',test_accessed=False,
                    validation_rows_accessed=False)
        (args.output/'QUALIFICATION.json').write_text(json.dumps(result))
        raise
    try:
        result=run(binding,args.candidate.resolve(strict=True),args.output.resolve(),args.train_seed,args.device,
                   json.loads(args.score_contract_json) if args.score_contract_json else None)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        result=dict(status='CANDIDATE_FAILURE',error=repr(error),failure_stage='qualification',
                    test_accessed=False,validation_rows_accessed=False)
        (args.output/'QUALIFICATION.json').write_text(json.dumps(result))
    print(json.dumps(result))


if __name__=='__main__': main()
