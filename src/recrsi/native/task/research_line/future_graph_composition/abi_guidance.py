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

@contextmanager
def abi_scope():
    import independent_discovery
    abi = dict(independent_discovery.ABI)
    abi['input_type'] = 'Class attribute: input_type = InputType.PAIRWISE; import from recbole.utils; required before construction.'
    abi['config_access'] = CONFIG_API_GUIDANCE.strip()
    with patch.object(independent_discovery, 'ABI', abi), patch.object(
            independent_discovery, 'DISCOVERY_INSTRUCTION',
            independent_discovery.DISCOVERY_INSTRUCTION + GUIDANCE):
        yield
