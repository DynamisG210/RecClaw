"""Graph-RO score declarations and narrowly scoped loss/score connectivity checks.
No discovery of all model parameters, free-text inference, or effectiveness claim.
"""
import ast
import hashlib
import json
from graph_score_connection import _bind_target_declaration

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
        required = {'primitive_id', 'parameter_names', 'recommendation_loss_index'}
        echoed = {'effect_phase', 'score_parameterization'}
        if not isinstance(row, dict) or not required <= set(row) or set(row) - required - echoed:
            raise ValueError('Scoring entry has missing or unknown fields')
        expected_rows = [entry for entry in request['entries'] if entry['primitive_id'] == row['primitive_id']]
        if len(expected_rows) != 1:
            raise ValueError('Scoring entry must identify exactly one trusted request entry')
        for key in echoed.intersection(row):
            if row[key] != expected_rows[0][key]:
                raise ValueError('Scoring entry conflicts with trusted phase/kind')
        row = {key: row[key] for key in required}
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

from graph_score_connection import check_graph_score_contract
