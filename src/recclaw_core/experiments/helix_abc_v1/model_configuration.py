"""Machine-owned constructor binding for explicitly declared model config APIs."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from functools import wraps


MODEL_CONFIG_MAPPING_REQUIREMENT = "MODEL_CONSTRUCTOR_CONFIG_PYTHON_MAPPING_V1"
MODEL_CONFIG_MAPPING_CONTRACT = {
    "type": "builtins.dict",
    "values": "Effective framework configuration values, copied without coercion or added defaults.",
    "missing_keys": "config[key] raises KeyError; config.get(key, default) returns default only for an absent key.",
    "present_values": "Explicit False, zero and None stay unchanged; they are not missing keys.",
    "ownership": "The framework retains its original Config. Candidate-owned mechanism settings belong in the model implementation; the fixed external recipe is unchanged.",
}
_BINDING_MODULE = "recclaw_core.experiments.helix_abc_v1.model_configuration"
_BINDING_ALIAS = "_recclaw_bind_model_config_mapping"


def bind_model_config_mapping(model_class):
    """Adapt only a declared model constructor; leave framework Config intact."""
    original_init = model_class.__init__
    if getattr(original_init, "_recclaw_model_config_mapping", False):
        return model_class

    @wraps(original_init)
    def initialize(self, config, dataset):
        values = config if isinstance(config, Mapping) else config.final_config_dict
        original_init(self, dict(values), dataset)

    initialize._recclaw_model_config_mapping = True
    model_class.__init__ = initialize
    return model_class


def bind_model_configuration_source(source: str, class_name: str = "FreshCandidateModel") -> str:
    """Append the same constructor binding to both native materialization paths."""
    tree = ast.parse(source)
    remove_lines = set()
    for node in tree.body:
        owned_import = (
            isinstance(node, ast.ImportFrom) and node.module == _BINDING_MODULE
            and len(node.names) == 1
            and node.names[0].name == "bind_model_config_mapping"
            and node.names[0].asname == _BINDING_ALIAS
        )
        owned_call = (
            isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == class_name
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
            and node.value.func.id == _BINDING_ALIAS
        )
        if owned_import or owned_call:
            remove_lines.update(range(node.lineno, node.end_lineno + 1))
    body = ''.join(line for index, line in enumerate(source.splitlines(keepends=True), 1)
                   if index not in remove_lines).rstrip()
    return (body + '\n\n'
        f'from {_BINDING_MODULE} import bind_model_config_mapping as {_BINDING_ALIAS}\n'
        f'{class_name} = {_BINDING_ALIAS}({class_name})\n')
