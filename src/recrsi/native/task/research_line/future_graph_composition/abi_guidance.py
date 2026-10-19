"""Clarify an existing worker ABI; neither rewrite nor repair paid programs."""
from contextlib import contextmanager
from unittest.mock import patch

VERSION = 'graph-class-level-input-type-and-config-api-guidance-v2'
GUIDANCE = (
    ' FreshCandidateModel must declare input_type = InputType.PAIRWISE at class '
    'scope, with InputType imported from recbole.utils. RecBole reads this '
    'attribute on the model class before calling its constructor. Assigning '
    'self.input_type inside __init__ is too late, and GeneralRecommender does '
    'not supply this declaration. Do not add loss_type to config_json as a '
    'workaround; that field is not required when class input_type is declared. '
    'This declares the existing pairwise data interface, not a prescribed loss formula.'
)

CONFIG_API_GUIDANCE = " The constructor config is a RecBole Config object, not a dict. Read declared values with config['name']; use ('name' in config) to test presence before applying a default. Config does not implement dict.get: do not call config.get(...). Keep valid zero/False values; do not replace them using an or-default expression. Do not mutate the worker config or introduce a compatibility wrapper."
GUIDANCE += CONFIG_API_GUIDANCE

SEQUENTIAL_GUIDANCE = (
    ' Use SequentialRecommender with the POINTWISE input interface derived by RecBole from frozen loss_type=CE. '
    'Training interactions contain item_id_list, item_length and next-item item_id; '
    'the sequential worker supplies no neg_item_id. Keep the frozen CE loss interface '
    'and score each supplied history prefix. DEV targets are unavailable to model code.'
)


def model_abi(binding):
    import independent_discovery
    abi = dict(independent_discovery.ABI)
    if (binding or {}).get('adapter') == 'recbole-sequential-fixed100-v1':
        abi.update(base_class='recbole.model.abstract_recommender.SequentialRecommender; call __init__(config, dataset)',
            input_type='InputType.POINTWISE, derived by RecBole from frozen loss_type=CE; an explicit class attribute must agree',
            inherited_attributes=['n_items','USER_ID','ITEM_ID','ITEM_SEQ','ITEM_SEQ_LEN','max_seq_length','device'],
            train_dataset='TRAIN chronological prefix/next-item examples only; item_id_list contains mapped item histories.',
            interaction_fields={'user_id':'user indices','item_id_list':'right-padded chronological prefixes',
                                'item_length':'non-padding lengths','item_id':'next-item training targets'},
            calculate_loss='Differentiable CE next-item loss on training prefixes, or tuple with declared recommendation loss index.',
            predict='Score item_id given item_id_list and item_length.',
            full_sort_predict='Score all n_items from item_id_list and item_length only; no next-item labels.')
    else:
        abi['input_type'] = 'Class attribute: input_type = InputType.PAIRWISE; import from recbole.utils; required before construction.'
    abi['config_access'] = CONFIG_API_GUIDANCE.strip()
    return abi


@contextmanager
def abi_scope(binding=None):
    import independent_discovery
    abi = model_abi(binding)
    sequential = (binding or {}).get('adapter') == 'recbole-sequential-fixed100-v1'
    instruction = independent_discovery.DISCOVERY_INSTRUCTION
    if sequential:
        instruction = instruction.replace('graph recommendation', 'sequential recommendation').replace('full-sort development', 'fixed-100 development')
    guidance = SEQUENTIAL_GUIDANCE + CONFIG_API_GUIDANCE if sequential else GUIDANCE
    with patch.object(independent_discovery, 'ABI', abi), patch.object(
            independent_discovery, 'DISCOVERY_INSTRUCTION', instruction + guidance):
        yield
