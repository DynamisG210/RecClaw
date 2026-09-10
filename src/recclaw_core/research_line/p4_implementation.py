"""Mechanical P4 package binding; the researcher owns only residual mathematics."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

PACKAGE_SOURCE = "from .candidate import FreshCandidateModel\n"
TRAINER_SOURCE = "from recclaw_core.research_line.p4_trainer import P4OperatorTrainer\n\nclass FreshCandidateTrainer(P4OperatorTrainer):\n    def _build_optimizer(self, **kwargs):\n        return super()._build_optimizer(**kwargs)\n"
MODEL_SCAFFOLD = '''
from recclaw_core.experiments.helix_abc_v1.p4_fagsp_parent import FrozenFaGSPResidualModelBase

class FreshCandidateModel(FrozenFaGSPResidualModelBase):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.to(self.device)
        initialize_residual(self, dict(config.final_config_dict), self.p4_residual_inputs())
        self._recclaw_p4_fit_completed = True

    def _mechanism_residual_scores(self):
        return residual_scores(self, self.p4_residual_inputs())
'''

FITTED_MODEL_SCAFFOLD = '''
from recclaw_core.experiments.helix_abc_v1.p4_fagsp_parent import FrozenFaGSPResidualModelBase
from recclaw_core.research_line.p4_implementation import initialize_fitted_operator, fitted_operator_residual

class FreshCandidateModel(FrozenFaGSPResidualModelBase):
    __recclaw_p4_hook_abi__ = 2

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.to(self.device)
        initialize_fitted_operator(self, dict(config.final_config_dict), fit_residual, residual_scores)
        self._recclaw_p4_fit_completed = True

    def _mechanism_residual_scores(self):
        return fitted_operator_residual(self)
'''


def _operator_state(model):
    return {key: getattr(model, f"_p4_operator_state_{i}")
            for i, key in enumerate(model._p4_operator_state_keys)}


def _evaluate_operator(model):
    return model._p4_operator_function(
        model._p4_action_coefficients, _operator_state(model),
        model._p4_operator_config, model.p4_residual_inputs(),
    )


def initialize_fitted_operator(model, config, fit_function, score_function):
    """Own fitting isolation, coefficient registration and static dependencies."""
    import torch
    from .p4_runtime import p4_fit_mode, p4_fitting_inputs

    # No model/self or full-scoring view is passed to the fitting hook.
    fitted = fit_function(dict(config), p4_fitting_inputs(model, config))
    if not isinstance(fitted, Mapping) or set(fitted) != {"coefficients", "state"}:
        raise ValueError("fit_residual must return coefficients and a tensor state mapping")
    if not isinstance(fitted["state"], Mapping):
        raise ValueError("fitted operator state must be a tensor mapping")
    device = model._recclaw_frozen_parent.device
    coefficients = torch.as_tensor(fitted["coefficients"], device=device)
    if not coefficients.is_floating_point():
        raise ValueError("operator coefficients must be a floating-point tensor")
    if config["p4_zero_action"] == "COEFFICIENTS_ZERO" and not coefficients.numel():
        raise ValueError("declared zero action requires actual operator coefficients")
    model._p4_operator_config = dict(config)
    model._p4_operator_function = score_function
    model._p4_operator_state_keys = tuple(fitted["state"])
    for i, value in enumerate(fitted["state"].values()):
        model.register_buffer(f"_p4_operator_state_{i}", torch.as_tensor(value, device=device))
    if p4_fit_mode(config) == "BPR_COEFFICIENTS":
        model._p4_action_coefficients = torch.nn.Parameter(coefficients)
    else:
        model.register_buffer("_p4_action_coefficients", coefficients)
        with torch.no_grad():
            model.register_buffer("_p4_operator_cache", _evaluate_operator(model))


def fitted_operator_residual(model):
    if hasattr(model, "_p4_operator_cache"):
        return model._p4_operator_cache
    return _evaluate_operator(model)


def _check_coefficient_zero_action(model, expected):
    """Exercise the real enabled operator, including coefficient-dependent cache."""
    import torch

    coefficients = model._p4_action_coefficients
    original = coefficients.detach().clone()
    cached = hasattr(model, "_p4_operator_cache")
    try:
        coefficients.zero_()
        if cached:
            model._p4_operator_cache = _evaluate_operator(model)
        torch.testing.assert_close(
            model._score_matrix(), expected, rtol=0, atol=1e-6,
            msg="P4 declared coefficient-zero action does not recover the frozen parent on the enabled score path",
        )
    finally:
        coefficients.copy_(original)
        if cached:
            model._p4_operator_cache = _evaluate_operator(model)


def bind_p4_response(response: Mapping[str, Any], *, hook_abi: int = 1) -> dict[str, Any]:
    files = {row["path"]: row["content"] for row in response["files"]}
    expected = {"recclaw_ext/__init__.py", "recclaw_ext/trainer.py", "recclaw_ext/candidate.py"}
    if set(files) != expected or len(response["files"]) != len(expected):
        raise ValueError("P4 response must contain exactly the three candidate package files")
    files["recclaw_ext/__init__.py"] = PACKAGE_SOURCE
    files["recclaw_ext/trainer.py"] = TRAINER_SOURCE
    source = files["recclaw_ext/candidate.py"]
    if hook_abi == 1:
        hooks = (("initialize_residual", ["self", "parameters", "inputs"]), ("residual_scores", ["self", "inputs"]))
        scaffold = MODEL_SCAFFOLD
    elif hook_abi == 2:
        hooks = (("fit_residual", ["parameters", "inputs"]),
                 ("residual_scores", ["coefficients", "state", "parameters", "inputs"]))
        scaffold = FITTED_MODEL_SCAFFOLD
    else:
        raise ValueError("unknown P4 hook ABI")
    tree = ast.parse(source)
    owned = ast.parse(scaffold).body
    # Native repair receives materialized source. Rebind our exact complete
    # scaffold without asking the Implementer to remove framework wiring.
    if (len(tree.body) >= len(owned)
            and [ast.dump(node) for node in tree.body[-len(owned):]] == [ast.dump(node) for node in owned]):
        source = "".join(source.splitlines(keepends=True)[:tree.body[-len(owned)].lineno - 1])
        tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, arguments in hooks:
        node = functions.get(name)
        if node is None or [arg.arg for arg in node.args.args] != arguments or node.decorator_list:
            raise ValueError(f"P4 requires undecorated {name}({', '.join(arguments)})")
    if any(isinstance(node, (ast.Name, ast.FunctionDef, ast.ClassDef))
           and (getattr(node, "id", None) or getattr(node, "name", None)) in {"FreshCandidateModel", "FrozenFaGSPResidualModelBase"}
           for node in ast.walk(tree)):
        raise ValueError("P4 response must contain residual hooks, not the machine-owned model class")
    files["recclaw_ext/candidate.py"] = source.rstrip() + "\n" + scaffold
    return {**dict(response), "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
            "files": [{"path": path, "content": files[path]} for path in sorted(files)]}


def p4_unit_check(model, config, dataset):
    import torch
    from recbole.data.interaction import Interaction
    from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import validate_p4_sparse_spectral_model_contract
    from recclaw_core.experiments.helix_abc_v1.p4_fagsp_parent import frozen_fagsp_scores_from_dataset

    validate_p4_sparse_spectral_model_contract(model, config, dataset)
    expected = torch.from_numpy(frozen_fagsp_scores_from_dataset(dataset)).to(model._recclaw_frozen_parent.device)
    enabled = model._recclaw_mechanism_enabled
    try:
        with torch.no_grad():
            model.set_mechanism_enabled(False)
            off = model._score_matrix()
            torch.testing.assert_close(off, expected, rtol=0, atol=1e-6)
            model.set_mechanism_enabled(True)
            on = model._score_matrix()
            if (on - off).abs().max().item() <= 1e-7:
                raise AssertionError("P4 residual is not behaviorally live")
            users = torch.arange(model.n_users, device=on.device)
            full = model.full_sort_predict(Interaction({model.USER_ID: users})).reshape(on.shape)
            torch.testing.assert_close(full, on)
            if (getattr(model, "__recclaw_p4_hook_abi__", 1) == 2
                    and model._p4_operator_config["p4_zero_action"] == "COEFFICIENTS_ZERO"):
                _check_coefficient_zero_action(model, expected)
                torch.testing.assert_close(model._score_matrix(), on)
    finally:
        model.set_mechanism_enabled(enabled)
