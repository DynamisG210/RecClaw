"""Sample an existing TRAIN loss/backward/update; never run an extra one."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
from unittest.mock import patch

SAMPLE_EPOCHS = (0, 4, 19, 49, 99, 199, 299)

def stats(value):
    import torch
    x = value.detach().float().reshape(-1)
    if not x.numel(): return {'elements': 0}
    return {'elements': x.numel(), 'mean': float(x.mean()),
            'rms': float(x.square().mean().sqrt()), 'min': float(x.min()),
            'max': float(x.max()), 'nonzero': int(torch.count_nonzero(x))}

@contextmanager
def observe_step(model, optimizer, output, source_sha, epoch):
    """Observe the first real optimizer step in seven fixed epochs, no RNG use."""
    if epoch not in SAMPLE_EPOCHS:
        yield
        return
    import torch
    import torch.nn.functional as F
    record = dict(schema='graph.actual-train-signal.v2', source_sha256=source_sha,
                  epoch=epoch, extra_forward=0, extra_backward=0, extra_update=0,
                  scope='One original TRAIN minibatch; observations are not causal ablations.',
                  functional_calls=[], module_calls={}, parameter_groups={})
    hooks=[]; before={}; done=False; in_loss=False
    original_loss=model.calculate_loss
    def loss(*args, **kwargs):
        nonlocal in_loss
        if done: return original_loss(*args, **kwargs)
        in_loss=True
        try: result=original_loss(*args, **kwargs)
        finally: in_loss=False
        parts=result if isinstance(result,tuple) else (result,)
        record['returned_loss_components']=[float(x.detach()) for x in parts]
        record['loss_semantics']='Use exact source; tuple position does not assign ranking/regularization semantics.'
        return result
    def function(name, original):
        def call(x,*args,**kwargs):
            result=original(x,*args,**kwargs)
            if in_loss and not done and x.requires_grad and x.ndim <= 1 and x.numel() <= 4096:
                row=dict(function=name, operand=stats(x), semantic='Source-bound operand, not automatically a ranking margin')
                record['functional_calls'].append(row)
                def grad(g):
                    row['actual_operand_gradient']=stats(g)
                    return None
                hooks.append(x.register_hook(grad))
            return result
        return call
    def module_hook(name):
        def call(module,args,result):
            if in_loss and not done:
                row=record['module_calls'].setdefault(name,dict(type=type(module).__name__,calls=0))
                row['calls']+=1
                if torch.is_tensor(result): row['last_output']=stats(result)
        return call
    for name,module in model.named_modules():
        if name and isinstance(module,(torch.nn.Linear,torch.nn.Embedding,torch.nn.Dropout,torch.nn.ReLU,torch.nn.LeakyReLU,torch.nn.Tanh)):
            hooks.append(module.register_forward_hook(module_hook(name)))
    def pre(opt,args,kwargs):
        if done:return
        with torch.no_grad():
            for name,p in model.named_parameters():
                row=dict(parameter=stats(p), gradient=None if p.grad is None else stats(p.grad))
                record['parameter_groups'][name]=row
                if p.grad is not None:
                    before[name]=(p.detach().clone(),p.grad.detach().clone())
    def post(opt,args,kwargs):
        nonlocal done
        if done:return
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name not in before:continue
                old,g=before[name];delta=p.detach()-old
                row=record['parameter_groups'][name];row['actual_update']=stats(delta)
                # Chunk-free float32 summaries avoid double-size embedding copies.
                dot=float((-g*delta).sum());den=float(g.norm()*delta.norm())
                row['update_cosine_negative_gradient']=dot/den if den else None
        record['optimizer']='Actual existing optimizer state; gradient scale does not predict Adam update.'
        with Path(output).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=True,allow_nan=False)+'\n')
        done=True;before.clear()
        for h in hooks:h.remove()
    step_hooks=[optimizer.register_step_pre_hook(pre),optimizer.register_step_post_hook(post)]
    try:
        with patch.object(model,'calculate_loss',loss),patch.object(F,'softplus',function('softplus',F.softplus)),patch.object(F,'logsigmoid',function('logsigmoid',F.logsigmoid)):
            yield
    finally:
        for h in hooks+step_hooks:h.remove()

def summarize(path, source_sha):
    import hashlib
    path=Path(path)
    if not path.exists():return {'status':'NOT_MEASURED'}
    blob=path.read_bytes();rows=[json.loads(x) for x in blob.splitlines() if x]
    if any(r['source_sha256']!=source_sha for r in rows):raise ValueError('Foreign TRAIN source')
    def extent(values):
        values=[x for x in values if x is not None]
        return dict(first=values[0],last=values[-1],min=min(values),max=max(values)) if values else None
    groups={}
    for name in sorted({k for r in rows for k in r['parameter_groups']}):
        entries=[r['parameter_groups'][name] for r in rows if name in r['parameter_groups']]
        groups[name]={kind:{key:extent([e.get(kind,{}).get(key) for e in entries if e.get(kind)]) for key in ('rms','nonzero')} for kind in ('parameter','gradient','actual_update')}
    return dict(status='MEASURED_ORIGINAL_TRAIN',path=str(path),sha256=hashlib.sha256(blob).hexdigest(),
                epochs=[r['epoch'] for r in rows],parameter_groups=groups,
                functional_samples=[dict(epoch=r['epoch'],calls=r['functional_calls']) for r in rows],
                module_types_seen=sorted({v['type'] for r in rows for v in r['module_calls'].values()}),
                limits='Sampled executed paths; absence is not global inactivity. No new counterfactual training or measured causal effect.')
