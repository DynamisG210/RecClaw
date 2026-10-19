"""Model-only eligibility policy, in addition to the existing OS boundary.

This AST policy is not a Python security sandbox. Candidate execution always
requires the network-denied filesystem/device/process boundary.
"""
import ast

ALLOWED_MODULES = ('numpy', 'scipy.sparse', 'torch.nn', 'torch.linalg', 'torch.sparse', 'math', 'typing', 'collections')
FORBIDDEN_CALLS = {'open','eval','exec','compile','__import__','globals','locals','vars',
                   'input','breakpoint','getattr','setattr','delattr'}
FORBIDDEN_ATTRIBUTES = {'load','save','loadtxt','savetxt','fromfile','tofile','memmap',
    'load_state_dict','manual_seed','seed','set_num_threads','set_default_device',
    'set_default_dtype','set_per_process_memory_fraction','system','popen','spawn',
    'fork','Trainer','Config','evaluate','fit','validation','valid_data','test_data'}


def qualify_source(source):
    tree = ast.parse(source)
    models = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='FreshCandidateModel']
    if len(models) != 1:
        raise ValueError('Exactly one complete FreshCandidateModel is required')
    classes = {node.name:node for node in tree.body if isinstance(node,ast.ClassDef)}
    pending, visited, methods = [models[0]], set(), set()
    while pending:
        current = pending.pop()
        if current.name in visited: continue
        visited.add(current.name)
        methods.update(node.name for node in current.body if isinstance(node,ast.FunctionDef))
        pending.extend(classes[base.id] for base in current.bases
                       if isinstance(base,ast.Name) and base.id in classes)
    if not {'__init__','calculate_loss','predict','full_sort_predict'}.issubset(methods):
        raise ValueError('Missing graph model ABI method')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            if isinstance(node,ast.ImportFrom) and any(alias.name=='*' or alias.name in FORBIDDEN_ATTRIBUTES
                                                       for alias in node.names):
                raise ValueError('Wildcard or runtime/state import outside model ABI')
            modules = [alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if isinstance(node,ast.ImportFrom) and node.level:
                raise ValueError('Candidate must be a self-contained model module')
            for module in modules:
                if module=='recbole.model.abstract_recommender' and isinstance(node,ast.ImportFrom):
                    if any(alias.name not in ('GeneralRecommender', 'SequentialRecommender') for alias in node.names):
                        raise ValueError('Only recommendation model base classes may be imported here')
                elif module in ('recbole.model.layers', 'recbole.model.loss') and isinstance(node,ast.ImportFrom):
                    allowed = {'recbole.model.layers': {'TransformerEncoder'}, 'recbole.model.loss': {'BPRLoss'}}
                    if any(alias.name not in allowed[module] for alias in node.names):
                        raise ValueError('Only the declared model computation components may be imported')
                elif module=='recbole.utils' and isinstance(node,ast.ImportFrom):
                    if any(alias.name!='InputType' for alias in node.names):
                        raise ValueError('Only InputType may be imported from recbole.utils')
                elif module!='torch' and not any(module==root or module.startswith(root+'.') for root in ALLOWED_MODULES):
                    raise ValueError('Import outside graph model computation: '+module)
        if isinstance(node,ast.Name) and node.id in FORBIDDEN_CALLS:
            raise ValueError('Candidate cannot access dynamic execution or host I/O: '+node.id)
        if isinstance(node,ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES or (node.attr.startswith('__') and node.attr!='__init__'):
                raise ValueError('Candidate cannot alter trusted runtime or load external state: '+node.attr)
        if isinstance(node,(ast.Global,ast.Nonlocal)):
            raise ValueError('Candidate cannot alter ambient execution state')
    return dict(status='ELIGIBLE_SOURCE', model_class='FreshCandidateModel',
                mutable_scope='model computation and loss', security_boundary='EXTERNAL_OS_REQUIRED')
