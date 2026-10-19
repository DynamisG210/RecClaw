"""Prospective provider information boundary; never starts an experiment.

Install inside the existing branch scope before composition. DISCOVER receives
facts and ABI, not an implementation to edit. REFINE is byte-for-byte unchanged.
Native evidence, parent bindings, selection, materialization and physics remain
outside this presentation boundary.
"""
import copy
import json
from contextlib import contextmanager
from unittest.mock import patch

VERSION = 'graph-independent-computation-discovery-v1'
CONTEXT_MARKER = 'Use this complete role-scoped Research Context JSON as the authoritative input for this call:'

ABI = {
    'entrypoint': 'candidate.py:FreshCandidateModel(config, train_dataset)',
    'base_class': 'recbole.model.abstract_recommender.GeneralRecommender; call its __init__(config, dataset)',
    'input_type': 'recbole.utils.InputType.PAIRWISE',
    'inherited_attributes': ['n_users', 'n_items', 'USER_ID', 'ITEM_ID', 'NEG_ITEM_ID', 'device'],
    'train_dataset': 'RecBole training Dataset only; inter_matrix(form="coo") gives user-item training interactions. Its shape includes padding ID 0.',
    'interaction_fields': {'user_id': 'user indices', 'item_id': 'positive item indices', 'neg_item_id': 'worker-sampled negative item indices'},
    'calculate_loss': 'Accept a pairwise Interaction; return a differentiable scalar tensor or tuple of scalar losses. Declare the recommendation loss index if returning a tuple.',
    'predict': 'Accept user/item Interaction; return one score per pair.',
    'full_sort_predict': 'Accept user Interaction; return batch-by-n_items scores or its flattened equivalent on model device, including padding column. Worker masks padding and training positives.',
    'state': 'Register trainable parameters and persistent tensors for device placement and state_dict reload. No external file I/O or dynamic execution. Do not change seeds or training/evaluation runtime.',
    'cache': 'Worker clears restore_user_e and restore_item_e before training and prediction. Any inference cache must respect training and checkpoint reload.',
    'computation': 'Choose complete model computation and loss. Configuration supplies values, not a requirement to retain any original computation. No starter algorithm is supplied.',
}

DISCOVERY_INSTRUCTION = '''You are the assigned RecClaw research role, constructing an independent complete graph recommendation model and loss for the supplied task.
The original root is a measured comparator and identity reference, not an implementation starting point. No root or previous candidate source, mechanism narrative, or inherited strongest-parent editing instruction is supplied in this discovery phase. Do not invent their contents or claim an isolated component effect from the numerical outcome record.
Develop one coherent full computation hypothesis using the task facts and shared literature. Specify its recommendation mechanism, competing explanation, predicted outcome and falsifier. Choose architecture and loss together where necessary. Neither addition, subtraction, structural distance nor novelty receives a bonus. The objective is competitive full-sort development NDCG@5.
Submit complete candidate.py in program.source without GRAPH_SCORE_CONTRACT. Source is delivered unchanged, with deterministic score metadata appended. Keep the provided ABI, dependencies, compatibility tokens and resource constraints. Use GraphCandidate for model and base_model_config, config_json "{}", current_profile_expressibility_claim NOT_EXPRESSIBLE, and implementation_requirements [].
Choose the exact original-root parent_ref in the supplied registry; research_action is null or DIRECT_IMPROVEMENT with task_ref null. This is comparator binding, not an inheritance requirement. Do not return parent_source_sha256 or closest_parent. Explain a new full computation, not an unobserved edit to hidden source.
Declare the entire scoring computation via program.score_effect. UNCHANGED may not be inferred from an unavailable original implementation; describe your actual scorer using FIXED or LEARNED as appropriate. Use score_parameter_targets: PARAMETER names one exact registered leaf; MODULE explicitly names a module/container and requires every recursive registered leaf to pass the original connectivity checks. Do not list unused leaves or expect inactive leaves to be dropped. recommendation_loss_index is null for a scalar loss. No second model rewrites your program.
Failures have no fabricated metric; root and cached work do not count as new experiments. Numerical history describes whole candidates only. All original evidence remains archived outside this discovery view. Subsequent REFINE steps receive their own measured branch source and feedback.
'''

# Only identifiers and measured values cross from previous research into discovery.
# Free-form explanations/axis credit must not reintroduce an inherited algorithm.
IDENTIFIERS = {'candidate_id', 'source_sha256', 'entrypoint_source_sha256',
               'parent_candidate_id', 'parent_source_sha256', 'child_source_sha256'}
NUMERICAL = {'primary_metric', 'development_metrics', 'metrics', 'measured_cost',
             'child_measured_cost', 'child_value', 'parent_value', 'child_minus_parent',
             'comparator_delta', 'root_delta', 'frontier_delta', 'available_index',
             'genuine', 'valid', 'root', 'metric_bearing', 'round_index'}


def numeric_tree(value):
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: numeric_tree(v) for k, v in value.items()
                if isinstance(v, (int, float, bool, dict, list)) or v is None}
    if isinstance(value, list):
        return [numeric_tree(v) for v in value
                if isinstance(v, (int, float, bool, dict, list)) or v is None]
    raise TypeError('Non-numerical historical value')


def measured_outcomes(context):
    records = {}
    def visit(value):
        if isinstance(value, dict):
            row = {k: copy.deepcopy(v) for k, v in value.items() if k in IDENTIFIERS}
            row.update({k: numeric_tree(v) for k, v in value.items()
                        if k in NUMERICAL and not isinstance(v, str)})
            # Status is a bounded code, not a free-form failure explanation.
            for k in ('status', 'child_status', 'failure_stage'):
                v = value.get(k)
                if isinstance(v, str) and v.replace('_', '').isalnum() and len(v) <= 64:
                    row[k] = v
            if any(k in row for k in IDENTIFIERS) and len(row) > 1:
                records[json.dumps(row, sort_keys=True)] = row
            for v in value.values():
                visit(v)
        elif isinstance(value, list):
            for v in value:
                visit(v)
    for field in ('state', 'memory', 'objective'):
        visit(context.get(field, {}))
    return list(records.values())


def discovery_context(context):
    old = context['objective']['space']
    route = old['branch_allocation']
    if route['mode'] != 'DISCOVER':
        return copy.deepcopy(context)
    keep = ('task', 'entrypoint', 'allowed_program_files', 'candidate_universe',
            'primary_metric', 'test_access', 'parent_id', 'parent_ref_registry',
            'candidate_mechanism_contract', 'branch_allocation')
    space = {k: copy.deepcopy(old[k]) for k in keep if k in old}
    space['discovery_abi'] = copy.deepcopy(ABI)
    space['input_policy'] = VERSION
    objective = {k: copy.deepcopy(context['objective'][k])
                 for k in ('baseline', 'direction', 'metric', 'claim_ceiling', 'open_axes', 'target_axes')
                 if k in context['objective']}
    objective['space'] = space
    return dict(schema=context['schema'], context=copy.deepcopy(context['context']),
                contract=copy.deepcopy(context['contract']), objective=objective,
                memory={'whole_candidate_numerical_outcomes': measured_outcomes(context)},
                state={'construction_reference': route['parent_id']})


def project_prompt(prompt, authority_notice):
    marker = prompt.index(CONTEXT_MARKER)
    start = prompt.index('{', marker + len(CONTEXT_MARKER))
    context, size = json.JSONDecoder().raw_decode(prompt[start:])
    if context['objective']['space']['branch_allocation']['mode'] != 'DISCOVER':
        return prompt
    # The native caller adds original facts, literature, actual budget and schema
    # after this boundary. Archived fully rendered prompts keep that suffix too.
    suffix = prompt[start + size:]
    return (DISCOVERY_INSTRUCTION + authority_notice + '\n\n' + CONTEXT_MARKER + '\n'
            + json.dumps(discovery_context(context), ensure_ascii=False, sort_keys=True)
            + suffix)


def install_producer(producer, authority_notice):
    native_call, native_provider = producer._call, producer.provider_call
    def call(role, view, **options):
        # Includes both modes: no cache namespace can silently reuse an older
        # experiment's response after changing discovery information.
        with patch.object(producer, 'call_root', producer.call_root / VERSION):
            return native_call(role, view, **options)
    def provider_call(**options):
        return native_provider(**{**options, 'prompt': project_prompt(options['prompt'], authority_notice)})
    producer._call = call
    producer.provider_call = provider_call
    return producer


@contextmanager
def independent_scope():
    import branch_search
    import program_adapter
    from discovery_contract import OLD_DEFINITION_NOTICE
    with branch_search.branch_scope():
        base = program_adapter.ProgramAdapter.build_provider_research_producer
        def build(adapter, **kwargs):
            wrapped = base(adapter, **kwargs)
            install_producer(wrapped._producer, OLD_DEFINITION_NOTICE)
            return wrapped
        with patch.object(program_adapter.ProgramAdapter, 'build_provider_research_producer', build):
            yield
