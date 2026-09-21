"""Exact state/graph checks and deterministic CPU checkpoint replay.

Do not compare two fresh CUDA sparse reductions with a near-zero relative
tolerance: such a comparison conflates kernel accumulation order with reload.
No dev/test metric or optimizer work is performed by this postfit check.
"""
import copy


def verify_checkpoint(model_class,config,train_dataset,current_model,saved,probe):
    import torch
    from graph_worker import clear_cache
    cpu_state={name:value.detach().cpu() for name,value in saved['state_dict'].items()}
    current_model.load_state_dict(saved['state_dict'],strict=True)
    current=current_model.state_dict()
    if set(current)!=set(cpu_state): raise ValueError('Checkpoint state keys differ')
    for name,expected in cpu_state.items():
        if current[name].dtype!=expected.dtype or not torch.equal(current[name].detach().cpu(),expected):
            raise ValueError('Checkpoint reload tensor is not exact: '+name)
    cpu_config=copy.deepcopy(config)
    cpu_config['device']=torch.device('cpu');cpu_config['use_gpu']=False
    threads=torch.get_num_threads();deterministic=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:
        left=model_class(cpu_config,train_dataset).cpu()
        right=model_class(cpu_config,train_dataset).cpu()
        for model in (left,right):
            model.load_state_dict(cpu_state,strict=True)
            for name,expected in cpu_state.items():
                if not torch.equal(model.state_dict()[name],expected):
                    raise ValueError('CPU checkpoint reload tensor is not exact: '+name)
        def sparse_graphs(model):
            result={}
            for module_name,module in model.named_modules():
                tensors=dict(vars(module));tensors.update(module._buffers)
                for name,value in tensors.items():
                    if isinstance(value,torch.Tensor) and value.layout!=torch.strided:
                        result[module_name+'.'+name]=value.detach().cpu().to_sparse_coo().coalesce()
            return result
        originals=sparse_graphs(current_model);a,b=sparse_graphs(left),sparse_graphs(right)
        if set(originals)!=set(a) or set(a)!=set(b): raise ValueError('Checkpoint graph keys differ')
        for name,original in originals.items():
            for other in (a[name],b[name]):
                if (original.shape!=other.shape or not torch.equal(original.indices(),other.indices())
                        or not torch.equal(original.values(),other.values())):
                    raise ValueError('Reconstructed checkpoint graph is not exact: '+name)
        outputs=[]
        with torch.no_grad():
            for model in (left,left,right,right):
                model.eval();clear_cache(model)
                outputs.append(model.full_sort_predict(probe.to('cpu')).detach().clone())
        if not all(torch.equal(outputs[0],value) for value in outputs[1:]):
            raise ValueError('Deterministic CPU checkpoint replay differs')
        return dict(mode='EXACT_STATE_GRAPH_AND_CPU_REPLAY',state_tensors=len(cpu_state),
            sparse_graphs=sorted(a),cpu_threads=1,repeated_forwards=4,exact=True,
            compared_elements=outputs[0].numel(),max_abs_difference=0.,optimizer_steps=0,
            new_validation_evaluations=0,test_accessed=False)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
