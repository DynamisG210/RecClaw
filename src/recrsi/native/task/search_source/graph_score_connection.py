"""Graph-RO score declarations and narrowly scoped loss/score connectivity checks.
No discovery of all model parameters, free-text inference, or effectiveness claim.
"""
import ast
import hashlib
import json

def contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def score_request(primitives, selected_spec_digest, selected_ids):
    entries = []
    for primitive in primitives:
        kind = primitive.get('score_parameterization')
        scoring = primitive['slot_id'] == 'SCORING_TRANSFORM'
        if kind not in (('LEARNED', 'FIXED') if scoring else ('NOT_SCORING',)):
            raise ValueError('UNKNOWN_SCORE_CONTRACT: missing or inconsistent accepted primitive declaration')
        if scoring:
            entries.append(dict(primitive_id=primitive['primitive_id'], effect_phase=primitive['effect_phase'],
                                score_parameterization=kind))
    return dict(selected_spec_digest=selected_spec_digest, selected_primitive_ids=list(selected_ids), entries=entries)

def bind_source_contract(source, request):
    tree = ast.parse(source.decode('utf-8'))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'GRAPH_SCORE_CONTRACT' for target in node.targets)]
    if len(values) != 1:
        raise ValueError('GRAPH_SCORE_CONTRACT must be one literal module declaration, including explicit empty entries when not applicable')
    declaration = ast.literal_eval(values[0])
    if isinstance(declaration, dict) and isinstance(declaration.get('entries'), list) and any(
            isinstance(row, dict) and 'parameter_targets' in row for row in declaration['entries']):
        return _bind_target_declaration(declaration, source, request)
    if not isinstance(declaration, dict) or set(declaration) != {'selected_spec_digest', 'selected_primitive_ids', 'entries'}:
        raise ValueError('Scoring declaration has invalid fields')
    for key in ('selected_spec_digest', 'selected_primitive_ids'):
        if declaration[key] != request[key]:
            raise ValueError('Scoring declaration differs from selected spec/primitive identity')
    declared = declaration['entries']
    if not isinstance(declared, list) or len(declared) != len(request['entries']):
        raise ValueError('Scoring declaration omits or adds a selected scoring primitive')
    rows = {}; used_names = set()
    for row in declared:
        if not isinstance(row, dict) or set(row) != {'primitive_id', 'parameter_names', 'recommendation_loss_index'}:
            raise ValueError('Scoring entry must declare names and loss component, not override trusted phase/kind')
        if row['primitive_id'] in rows:
            raise ValueError('Duplicate scoring primitive declaration')
        names = row['parameter_names']; index = row['recommendation_loss_index']
        if not isinstance(names, list) or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names):
            raise ValueError('Invalid declared scoring parameter names')
        if used_names.intersection(names):
            raise ValueError('One parameter declaration cannot stand for multiple scoring primitives')
        used_names.update(names)
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError('Invalid recommendation loss component')
        rows[row['primitive_id']] = row
    entries = []
    for expected in request['entries']:
        if expected['primitive_id'] not in rows:
            raise ValueError('Missing selected scoring primitive')
        row = rows[expected['primitive_id']]
        if bool(row['parameter_names']) != (expected['score_parameterization'] == 'LEARNED'):
            raise ValueError('Learned score cannot omit parameters; fixed score must not claim learned parameters')
        entries.append({**expected, **row})
    return dict(schema='graph-ro-score-contract.v2', source_sha256=hashlib.sha256(source).hexdigest(),
                selected_spec_digest=request['selected_spec_digest'], selected_primitive_ids=request['selected_primitive_ids'],
                status='REQUIRED' if entries else 'NOT_APPLICABLE', entries=entries)


def _bind_target_declaration(declaration, source, request):
    """Prospective explicit target contract; never reinterpret legacy names."""
    if not isinstance(declaration, dict) or set(declaration) != {'selected_spec_digest', 'selected_primitive_ids', 'entries'}:
        raise ValueError('Scoring declaration has invalid fields')
    for key in ('selected_spec_digest', 'selected_primitive_ids'):
        if declaration[key] != request[key]:
            raise ValueError('Scoring declaration differs from selected spec/primitive identity')
    declared = declaration['entries']
    if not isinstance(declared, list) or len(declared) != len(request['entries']):
        raise ValueError('Scoring declaration omits or adds a selected scoring primitive')
    rows = {}; used_targets = set()
    for row in declared:
        if not isinstance(row, dict) or set(row) != {'primitive_id', 'parameter_targets', 'recommendation_loss_index'}:
            raise ValueError('Typed scoring entry requires explicit targets and loss component')
        if row['primitive_id'] in rows:
            raise ValueError('Duplicate scoring primitive declaration')
        targets = row['parameter_targets']; index = row['recommendation_loss_index']
        if not isinstance(targets, list):
            raise ValueError('Scoring targets must be a list')
        for target in targets:
            if (not isinstance(target, dict) or set(target) != {'kind', 'path'}
                    or target['kind'] not in ('PARAMETER', 'MODULE')
                    or not isinstance(target['path'], str) or not target['path']):
                raise ValueError('Invalid typed scoring target')
            identity = (target['kind'], target['path'])
            if identity in used_targets:
                raise ValueError('Duplicate scoring target: ' + target['path'])
            used_targets.add(identity)
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError('Invalid recommendation loss component')
        rows[row['primitive_id']] = row
    entries = []
    for expected in request['entries']:
        if expected['primitive_id'] not in rows:
            raise ValueError('Missing selected scoring primitive')
        row = rows[expected['primitive_id']]
        if bool(row['parameter_targets']) != (expected['score_parameterization'] == 'LEARNED'):
            raise ValueError('Learned score cannot omit targets; fixed score must not claim learned targets')
        entries.append({**expected, **row})
    return dict(schema='graph-ro-score-contract.v3', source_sha256=hashlib.sha256(source).hexdigest(),
                selected_spec_digest=request['selected_spec_digest'], selected_primitive_ids=request['selected_primitive_ids'],
                status='REQUIRED' if entries else 'NOT_APPLICABLE', entries=entries)


def resolve_score_targets(model, targets, used_parameters):
    """Resolve only declared targets. Missing/empty/alias/overlap never drops a leaf."""
    parameters = []; resolutions = []
    for target in targets:
        path = target['path']; kind = target['kind']
        try:
            if kind == 'PARAMETER':
                leaves = [(path, model.get_parameter(path))]
            elif kind == 'MODULE':
                module = model.get_submodule(path)
                leaves = [(path + '.' + name, parameter) for name, parameter in
                          module.named_parameters(recurse=True, remove_duplicate=False)]
                if not leaves:
                    raise ValueError('EMPTY score module target: ' + path)
            else:
                raise ValueError('Unknown score target kind: ' + str(kind))
        except AttributeError as error:
            raise ValueError('UNRESOLVED score target ' + kind + ':' + path + ': ' + str(error)) from error
        for name, parameter in leaves:
            if id(parameter) in used_parameters:
                raise ValueError('OVERLAPPING score parameter: ' + name)
            used_parameters.add(id(parameter))
            parameters.append((name, parameter))
        resolutions.append(dict(kind=kind, path=path, resolved_parameter_names=[name for name, _ in leaves]))
    return parameters, resolutions


def check_graph_score_contract(model, loss, batch, contract):
    import torch
    if contract['status'] == 'NOT_APPLICABLE':
        return {'status': 'NOT_APPLICABLE', 'entries': []}
    results = []
    used_parameters = set()
    for entry in contract['entries']:
        phase = entry['effect_phase']; kind = entry['score_parameterization']
        result = {'primitive_id': entry['primitive_id'], 'effect_phase': phase, 'score_parameterization': kind}
        if kind == 'FIXED':
            results.append({**result, 'status': 'FIXED_DECLARATION_ONLY', 'mathematics_verified': False}); continue
        parameters = None
        if contract['schema'] == 'graph-ro-score-contract.v3':
            parameters, resolutions = resolve_score_targets(model, entry['parameter_targets'], used_parameters)
            result['target_resolution'] = resolutions
        if phase == 'INFERENCE_ONLY':
            results.append({**result, 'status': 'INFERENCE_ONLY_DECLARATION', 'training_connection_required': False,
                            'mathematics_verified': False}); continue
        if phase not in ('TRAIN_ONLY', 'TRAIN_AND_INFERENCE'):
            raise ValueError('Unknown score effect phase')
        index = entry['recommendation_loss_index']
        recommendation = loss if index is None else loss[index]
        if not isinstance(recommendation, torch.Tensor) or recommendation.ndim != 0:
            raise ValueError('The declared recommendation loss must be a scalar tensor')
        # A named parameter must connect to both the promised training loss and,
        # when declared, the actual predict API. Unrelated regularizer gradients do not suffice.
        prediction = model.predict(batch).sum() if phase == 'TRAIN_AND_INFERENCE' else None
        checks = []
        if parameters is None:
            parameters = ((name, model.get_parameter(name)) for name in entry['parameter_names'])
        for name, parameter in parameters:
            def gradient(value):
                if not parameter.requires_grad or not value.requires_grad:
                    return None
                return torch.autograd.grad(value, (parameter,), allow_unused=True, retain_graph=True)[0]
            train_grad = gradient(recommendation)
            score_grad = gradient(prediction) if prediction is not None else None
            required = [train_grad] + ([score_grad] if prediction is not None else [])
            if any(g is None for g in required):
                raise ValueError('DISCONNECTED declared score parameter: ' + entry['primitive_id'] + ':' + name)
            if any(not torch.isfinite(g).all() for g in required):
                raise ValueError('NONFINITE declared score parameter gradient: ' + name)
            checks.append({'parameter': name, 'training_connected': True,
                           'prediction_connected': True if prediction is not None else 'NOT_REQUIRED',
                           'nonzero_on_this_batch': bool(torch.count_nonzero(train_grad))})
        results.append({**result, 'status': 'CONNECTED', 'parameters': checks, 'mathematics_verified': False})
    return {'status': 'CONNECTED' if all(x['status'] == 'CONNECTED' for x in results) else 'DECLARATIONS_WITH_LIMITED_CHECKS',
            'entries': results}
