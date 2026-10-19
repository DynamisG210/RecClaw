"""Compiler-owned mechanics for parent-preserving DiffRec programs.

The declarative Implementer owns only the typed mechanism hooks selected by
the compiled graph.  This module owns the frozen DiffRec lifecycle around
those hooks: construction, schedule, target, objective, posterior solver,
score, and trajectory witness.  Keeping those operations here prevents a
free-form implementation from silently changing an undeclared parent slot.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SPACE_ID = "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1"
_VELOCITY_PRIMITIVE = "dynamics.behavior_guided_velocity"
_FLOW_OBJECTIVE_PRIMITIVE = "objective.flow_matching_velocity"
_VP_FORWARD_PRIMITIVE = "forward.gaussian_variance_preserving"
_PARENT_STATE_PRIMITIVE = "state.full_interaction_vector"
_PARENT_DYNAMICS_PRIMITIVE = "dynamics.time_conditioned_mlp"
_PARENT_OBJECTIVE_PRIMITIVE = "objective.predict_clean_state"
_PARENT_SCHEDULE_PRIMITIVE = "schedule.linear"
_PARENT_SOLVER_PRIMITIVE = "solver.deterministic_reduced_step"
_PARENT_SCORE_PRIMITIVE = "score.recovered_logits"
_VELOCITY_CHANGED_SLOTS = frozenset(
    {"GENERATIVE_DYNAMICS", "DENOISING_FLOW_OBJECTIVE"}
)
_PARENT_X0_UNCONDITIONED_CHANGED_SLOTS = frozenset({"GENERATIVE_DYNAMICS"})
_PARENT_X0_CONDITIONED_CHANGED_SLOTS = frozenset(
    {"GENERATIVE_DYNAMICS", "CONDITIONING"}
)
_STATE_ONLY_CHANGED_SLOTS = frozenset({"STATE_REPRESENTATION"})
_STATE_HOOK = "recclaw_diffusion_state"
_DECODE_HOOK = "recclaw_diffusion_decode"
_CONDITION_HOOK = "recclaw_diffusion_condition"
_FIELD_HOOK = "recclaw_predict_field"
_FIELD_INPUT_HOOK = "recclaw_diffusion_field"
_PARENT_STATE_PARAMETERS = {"value_encoding": "BINARY"}
_PARENT_FORWARD_PARAMETERS = {
    "steps": 5,
    "noise_scale": 0.001,
    "beta_fixed": True,
    "fixed_first_beta": 0.00001,
}
_PARENT_SCHEDULE_PARAMETERS = {"start": 0.0005, "end": 0.005}
_PARENT_DYNAMICS_PARAMETERS = {
    "layers": 1,
    "hidden_dimension": 300,
    "time_embedding_dimension": 10,
    "time_fusion": "CONCAT",
    "activation": "TANH",
    "dropout": 0.5,
    "normalize_input": False,
}
_PARENT_OBJECTIVE_PARAMETERS = {
    "loss": "WEIGHTED_MSE",
    "time_weighting": "SNR",
    "timestep_sampling": "LOSS_SECOND_MOMENT_AFTER_WARMUP",
    "history_num_per_term": 10,
    "uniform_mixture_probability": 0.001,
}
_PARENT_SOLVER_PARAMETERS = {
    "steps": 5,
    "spacing": "UNIFORM",
    "sampling_steps": 0,
    "sampling_noise": False,
    "start_state": "OBSERVED_INTERACTION_STATE",
}
_PARENT_SCORE_PARAMETERS = {"temperature": 1.0}


class DiffusionFlowMechanismBindingError(ValueError):
    """The compiled P7 program cannot use the parent-preserving scaffold."""


def _changed_slot_ids(value: Sequence[Any]) -> frozenset[str]:
    slots: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise DiffusionFlowMechanismBindingError(
                "declared changed slots must contain mappings"
            )
        slot_id = row.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id:
            raise DiffusionFlowMechanismBindingError(
                "declared changed slot lacks a normalized slot_id"
            )
        slots.add(slot_id)
    return frozenset(slots)


def _slot_components(
    component_specs: Mapping[str, Any], slot_id: str
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        spec
        for spec in component_specs.values()
        if isinstance(spec, Mapping) and spec.get("slot_id") == slot_id
    )


def _single_primitive(
    component_specs: Mapping[str, Any],
    *,
    slot_id: str,
    primitive_id: str,
) -> Mapping[str, Any] | None:
    matches = tuple(
        spec
        for spec in _slot_components(component_specs, slot_id)
        if spec.get("primitive_id") == primitive_id
    )
    if len(matches) > 1:
        raise DiffusionFlowMechanismBindingError(
            f"slot {slot_id} contains duplicate {primitive_id} components"
        )
    return matches[0] if matches else None


def diffusion_flow_profile_contract(
    component_specs: Mapping[str, Any],
    declared_changed_slots: Sequence[Any],
) -> dict[str, Any] | None:
    """Return compiler-owned DiffRec mechanics, or None for the exact parent."""

    if not isinstance(component_specs, Mapping):
        raise TypeError("component_specs must be a mapping")
    if not isinstance(declared_changed_slots, (tuple, list)):
        raise TypeError("declared_changed_slots must be a sequence")
    changed_slots = _changed_slot_ids(declared_changed_slots)
    exact_parent_components = (
        (
            "STATE_REPRESENTATION",
            _PARENT_STATE_PRIMITIVE,
            _PARENT_STATE_PARAMETERS,
        ),
        ("FORWARD_PATH", _VP_FORWARD_PRIMITIVE, _PARENT_FORWARD_PARAMETERS),
        ("TIME_SCHEDULE", _PARENT_SCHEDULE_PRIMITIVE, _PARENT_SCHEDULE_PARAMETERS),
        (
            "GENERATIVE_DYNAMICS",
            _PARENT_DYNAMICS_PRIMITIVE,
            _PARENT_DYNAMICS_PARAMETERS,
        ),
        (
            "DENOISING_FLOW_OBJECTIVE",
            _PARENT_OBJECTIVE_PRIMITIVE,
            _PARENT_OBJECTIVE_PARAMETERS,
        ),
        ("RECOVERY_SOLVER", _PARENT_SOLVER_PRIMITIVE, _PARENT_SOLVER_PARAMETERS),
        ("SCORE_HEAD", _PARENT_SCORE_PRIMITIVE, _PARENT_SCORE_PARAMETERS),
    )
    exact_parent = (
        not changed_slots
        and not _slot_components(component_specs, "CONDITIONING")
        and all(
            len(_slot_components(component_specs, slot_id)) == 1
            and (
                component := _single_primitive(
                    component_specs,
                    slot_id=slot_id,
                    primitive_id=primitive_id,
                )
            )
            is not None
            and isinstance(component.get("parameters"), Mapping)
            and dict(component["parameters"]) == expected_parameters
            for slot_id, primitive_id, expected_parameters in exact_parent_components
        )
    )
    if exact_parent:
        return None

    required_single_slots = (
        "STATE_REPRESENTATION",
        "FORWARD_PATH",
        "TIME_SCHEDULE",
        "GENERATIVE_DYNAMICS",
        "DENOISING_FLOW_OBJECTIVE",
        "RECOVERY_SOLVER",
        "SCORE_HEAD",
    )
    if any(
        len(_slot_components(component_specs, slot_id)) != 1
        for slot_id in required_single_slots
    ):
        raise DiffusionFlowMechanismBindingError(
            "canonical DiffRec field binding requires exactly one component per parent slot"
        )

    state_component = _slot_components(component_specs, "STATE_REPRESENTATION")[0]
    forward_component = _slot_components(component_specs, "FORWARD_PATH")[0]
    schedule_component = _slot_components(component_specs, "TIME_SCHEDULE")[0]
    dynamics = _slot_components(component_specs, "GENERATIVE_DYNAMICS")[0]
    objective = _slot_components(component_specs, "DENOISING_FLOW_OBJECTIVE")[0]
    solver_component = _slot_components(component_specs, "RECOVERY_SOLVER")[0]
    score_component = _slot_components(component_specs, "SCORE_HEAD")[0]
    conditions = _slot_components(component_specs, "CONDITIONING")
    priors = _slot_components(component_specs, "PRIOR")
    parent_state = _single_primitive(
        component_specs,
        slot_id="STATE_REPRESENTATION",
        primitive_id=_PARENT_STATE_PRIMITIVE,
    )
    parent_forward = _single_primitive(
        component_specs,
        slot_id="FORWARD_PATH",
        primitive_id=_VP_FORWARD_PRIMITIVE,
    )
    parent_schedule = _single_primitive(
        component_specs,
        slot_id="TIME_SCHEDULE",
        primitive_id=_PARENT_SCHEDULE_PRIMITIVE,
    )
    parent_solver = _single_primitive(
        component_specs,
        slot_id="RECOVERY_SOLVER",
        primitive_id=_PARENT_SOLVER_PRIMITIVE,
    )
    parent_score = _single_primitive(
        component_specs,
        slot_id="SCORE_HEAD",
        primitive_id=_PARENT_SCORE_PRIMITIVE,
    )

    state_parameters = state_component.get("parameters")
    dynamics_parameters = dynamics.get("parameters")
    objective_parameters = objective.get("parameters")
    forward_parameters = forward_component.get("parameters")
    schedule_parameters = schedule_component.get("parameters")
    solver_parameters = solver_component.get("parameters")
    score_parameters = score_component.get("parameters")
    if not all(
        isinstance(value, Mapping)
        for value in (
            state_parameters,
            dynamics_parameters,
            objective_parameters,
            forward_parameters,
            schedule_parameters,
            solver_parameters,
            score_parameters,
        )
    ):
        raise DiffusionFlowMechanismBindingError(
            "state, field, objective, and parent-mechanics parameters must be mappings"
        )
    assert isinstance(state_parameters, Mapping)
    assert isinstance(dynamics_parameters, Mapping)
    assert isinstance(objective_parameters, Mapping)
    assert isinstance(forward_parameters, Mapping)
    assert isinstance(schedule_parameters, Mapping)
    assert isinstance(solver_parameters, Mapping)
    assert isinstance(score_parameters, Mapping)
    dynamics_primitive = dynamics.get("primitive_id")
    objective_primitive = objective.get("primitive_id")
    path_math_changes = changed_slots & {"FORWARD_PATH", "PRIOR"}
    if path_math_changes or priors:
        raise DiffusionFlowMechanismBindingError(
            "the parent-x0 profile freezes the exact Gaussian-VP forward path "
            "with its clean-state objective and posterior solver; an alternate "
            "forward path or prior requires a compiler-supported matched target, "
            "inverse, and recovery solver"
        )
    state_only_graph = changed_slots == _STATE_ONLY_CHANGED_SLOTS
    if state_only_graph:
        if conditions:
            raise DiffusionFlowMechanismBindingError(
                "state-only binding cannot add an undeclared condition"
        )
        if (
            parent_forward is None
            or parent_schedule is None
            or parent_solver is None
            or parent_score is None
            or dynamics_primitive != _PARENT_DYNAMICS_PRIMITIVE
            or dict(dynamics_parameters) != _PARENT_DYNAMICS_PARAMETERS
            or objective_primitive != _PARENT_OBJECTIVE_PRIMITIVE
            or dict(objective_parameters) != _PARENT_OBJECTIVE_PARAMETERS
        ):
            raise DiffusionFlowMechanismBindingError(
                "state-only binding requires the exact parent forward path, dynamics, "
                "objective, and solver"
            )
        if (
            state_component.get("primitive_id") == _PARENT_STATE_PRIMITIVE
            and dict(state_parameters) == _PARENT_STATE_PARAMETERS
        ):
            raise DiffusionFlowMechanismBindingError(
                "declared state representation change is an exact-parent no-op"
            )
        if dict(forward_parameters) != _PARENT_FORWARD_PARAMETERS:
            raise DiffusionFlowMechanismBindingError(
                "state-only binding requires the exact parent Gaussian-VP parameters"
            )
        model_hooks = (_STATE_HOOK,)
        if state_component.get("primitive_id") in {
            "state.spectral_graph_coordinates", "state.learned_latent_autoencoder",
        }:
            model_hooks += (_DECODE_HOOK,)
        objective_mode = "PARENT_X0"
        conditioned = False
        metric = None
        weighting = None
        history_num_per_term = int(objective_parameters["history_num_per_term"])
        uniform_mixture_probability = float(
            objective_parameters["uniform_mixture_probability"]
        )
        mechanical_binding = "STATE_PARENT_X0_SOLVER_V1"
    else:
        if (
            parent_state is None
            or parent_forward is None
            or parent_schedule is None
            or parent_solver is None
            or parent_score is None
        ):
            raise DiffusionFlowMechanismBindingError(
                "DiffRec field binding requires the exact declared parent state, forward, "
                "schedule, solver, and score components"
            )
        if dict(state_parameters) != _PARENT_STATE_PARAMETERS:
            raise DiffusionFlowMechanismBindingError(
                "unchanged full-interaction state semantics differ from the DiffRec parent"
            )

        is_velocity_flow = (
            dynamics_primitive == _VELOCITY_PRIMITIVE
            or objective_primitive == _FLOW_OBJECTIVE_PRIMITIVE
        )
        if is_velocity_flow:
            if not (
                dynamics_primitive == _VELOCITY_PRIMITIVE
                and objective_primitive == _FLOW_OBJECTIVE_PRIMITIVE
            ):
                raise DiffusionFlowMechanismBindingError(
                    "behavior-guided velocity and flow-matching objective must be bound together"
                )
            if changed_slots != _VELOCITY_CHANGED_SLOTS or conditions:
                raise DiffusionFlowMechanismBindingError(
                    "the VP velocity binding permits only the declared GENERATIVE_DYNAMICS "
                    "and DENOISING_FLOW_OBJECTIVE changes"
                )
            if dynamics_parameters.get("field_parameterization") != "VELOCITY":
                raise DiffusionFlowMechanismBindingError(
                    "flow-matching VP binding requires field_parameterization=VELOCITY"
                )
            metric = objective_parameters.get("metric")
            weighting = objective_parameters.get("weighting")
            if metric not in {"L2", "HUBER"} or weighting not in {
                "UNIFORM",
                "ENDPOINT_HEAVY",
                "PATH_SPEED",
            }:
                raise DiffusionFlowMechanismBindingError(
                    "flow-matching metric or weighting is not compiler-supported"
                )
            objective_mode = "VP_VELOCITY"
            conditioned = False
            history_num_per_term = None
            uniform_mixture_probability = None
            mechanical_binding = "VP_VELOCITY_PARENT_SOLVER_V1"
        elif objective_primitive == _PARENT_OBJECTIVE_PRIMITIVE:
            if changed_slots == _PARENT_X0_UNCONDITIONED_CHANGED_SLOTS and not conditions:
                conditioned = False
            elif (
                changed_slots == _PARENT_X0_CONDITIONED_CHANGED_SLOTS
                and len(conditions) == 1
            ):
                # The condition is a separate executable graph node.  Its
                # algorithm belongs to the candidate; the scaffold routes its
                # declared inputs and consumes its output in the field.
                conditioned = True
            else:
                raise DiffusionFlowMechanismBindingError(
                    "parent-x0 field binding requires a declared dynamics change with "
                    "zero conditions, or dynamics plus exactly one compiled condition"
                )
            if (
                not conditioned
                and dynamics_primitive == _PARENT_DYNAMICS_PRIMITIVE
                and dict(dynamics_parameters) == _PARENT_DYNAMICS_PARAMETERS
            ):
                raise DiffusionFlowMechanismBindingError(
                    "parent-x0 dynamics declaration is an exact-parent no-op"
                )
            if dict(objective_parameters) != _PARENT_OBJECTIVE_PARAMETERS:
                raise DiffusionFlowMechanismBindingError(
                    "parent-x0 target, weighting, sampler, or history semantics differ "
                    "from the DiffRec parent"
                )
            objective_mode = "PARENT_X0"
            metric = None
            weighting = None
            history_num_per_term = int(objective_parameters["history_num_per_term"])
            uniform_mixture_probability = float(
                objective_parameters["uniform_mixture_probability"]
            )
            mechanical_binding = "PARENT_X0_FIELD_PARENT_SOLVER_V1"
        else:
            raise DiffusionFlowMechanismBindingError(
                "P7 field graph has no compiler-owned objective binding"
            )
        field_method = (
            _FIELD_INPUT_HOOK if dynamics.get("custom_component_id") else _FIELD_HOOK
        )
        model_hooks = (
            (_CONDITION_HOOK, field_method) if conditioned else (field_method,)
        )

    if dict(solver_parameters) != _PARENT_SOLVER_PARAMETERS:
        raise DiffusionFlowMechanismBindingError(
            "unchanged deterministic solver parameters differ from the DiffRec parent"
        )
    solver_steps = int(solver_parameters.get("steps", -2))
    forward_steps = int(forward_parameters.get("steps", -1))
    if forward_steps != _PARENT_FORWARD_PARAMETERS["steps"]:
        raise DiffusionFlowMechanismBindingError(
            "forward steps differ from the frozen DiffRec parent"
        )
    noise_scale = forward_parameters.get("noise_scale")
    beta_fixed = forward_parameters.get("beta_fixed")
    fixed_first_beta = forward_parameters.get("fixed_first_beta")
    schedule_start = schedule_parameters.get("start")
    schedule_end = schedule_parameters.get("end")
    if (
        isinstance(noise_scale, bool)
        or not isinstance(noise_scale, (int, float))
        or float(noise_scale) != _PARENT_FORWARD_PARAMETERS["noise_scale"]
        or beta_fixed is not True
        or isinstance(fixed_first_beta, bool)
        or not isinstance(fixed_first_beta, (int, float))
        or float(fixed_first_beta)
        != _PARENT_FORWARD_PARAMETERS["fixed_first_beta"]
        or isinstance(schedule_start, bool)
        or not isinstance(schedule_start, (int, float))
        or isinstance(schedule_end, bool)
        or not isinstance(schedule_end, (int, float))
        or {
            "start": float(schedule_start),
            "end": float(schedule_end),
        }
        != _PARENT_SCHEDULE_PARAMETERS
    ):
        raise DiffusionFlowMechanismBindingError(
            "unchanged VP noise, fixed-beta, or linear-schedule semantics drifted"
        )
    temperature = score_parameters.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) != _PARENT_SCORE_PARAMETERS["temperature"]
    ):
        raise DiffusionFlowMechanismBindingError(
            "unchanged recovered-logit score temperature must remain 1.0"
        )
    return {
        "objective_mode": objective_mode,
        "mechanical_binding": mechanical_binding,
        "model_hooks": model_hooks,
        "conditioned": conditioned,
        "condition_component": dict(conditions[0]) if conditioned else None,
        "field_component": dict(dynamics),
        "state_dimension": (
            int(state_parameters["rank"])
            if state_only_graph
            and state_component.get("primitive_id") == "state.spectral_graph_coordinates"
            else int(state_parameters["latent_dimension"])
            if state_only_graph
            and state_component.get("primitive_id") == "state.learned_latent_autoencoder"
            else None
        ),
        "reconstruction_weight": (
            float(state_parameters["reconstruction_weight"])
            if state_only_graph
            and state_component.get("primitive_id") == "state.learned_latent_autoencoder"
            else 0.0
        ),
        "parent_component_ids": {
            "state": state_component["component_id"],
            "forward": forward_component["component_id"],
            "time": schedule_component["component_id"],
        },
        "metric": metric,
        "weighting": weighting,
        "history_num_per_term": history_num_per_term,
        "uniform_mixture_probability": uniform_mixture_probability,
        "solver_steps": solver_steps,
        "sampling_steps": int(solver_parameters["sampling_steps"]),
        "noise_scale": float(noise_scale),
        "fixed_first_beta": float(fixed_first_beta),
        "schedule_start": float(schedule_start),
        "schedule_end": float(schedule_end),
    }


def _exact_diffrec_parent(model_class: type[Any]) -> type[Any]:
    """Resolve the first real DiffRec implementation below FreshCandidateModel."""

    required = {
        "__init__",
        "calculate_loss",
        "p_sample",
        "p_mean_variance",
        "predict",
        "full_sort_predict",
        "q_sample",
        "q_posterior_mean_variance",
        "_extract_into_tensor",
        "sample_timesteps",
        "reweight_loss",
        "update_Lt_history",
    }
    for candidate in model_class.__mro__[1:]:
        if required.issubset(candidate.__dict__):
            return candidate
    raise DiffusionFlowMechanismBindingError(
        "FreshCandidateModel does not directly preserve the exact DiffRec parent"
    )


def bind_diffusion_flow_model_class(
    model_class: type[Any],
    component_specs: Mapping[str, Any],
    declared_changed_slots: Sequence[Any],
) -> type[Any]:
    """Bind exact DiffRec lifecycle mechanics around compiled local hooks.

    Only the exact healthy parent is returned unchanged.  Every other P7
    profile graph must have compiler-owned target and recovery mechanics;
    unsupported graphs fail here rather than falling back to stable-method
    rewrites.
    """

    if not isinstance(model_class, type):
        raise TypeError("model_class must be a class")
    contract = diffusion_flow_profile_contract(
        component_specs, declared_changed_slots
    )
    if contract is None:
        return model_class

    parent_class = _exact_diffrec_parent(model_class)
    required_hook_names = tuple(contract["model_hooks"])
    mechanism_hooks: dict[str, Any] = {}
    for hook_name in required_hook_names:
        hook = model_class.__dict__.get(hook_name)
        if not callable(hook):
            raise DiffusionFlowMechanismBindingError(
                f"compiled P7 mechanism requires callable {hook_name}"
            )
        mechanism_hooks[hook_name] = hook
    field_hook = mechanism_hooks.get(_FIELD_HOOK)
    field_input_hook = mechanism_hooks.get(_FIELD_INPUT_HOOK)
    state_hook = mechanism_hooks.get(_STATE_HOOK)
    decode_hook = mechanism_hooks.get(_DECODE_HOOK)
    condition_hook = mechanism_hooks.get(_CONDITION_HOOK)
    initialize_hook = model_class.__dict__.get("recclaw_initialize_mechanism")
    if initialize_hook is not None and not callable(initialize_hook):
        raise DiffusionFlowMechanismBindingError(
            "recclaw_initialize_mechanism must be callable"
        )

    objective_mode = str(contract["objective_mode"])
    conditioned = bool(contract["conditioned"])
    condition_component = contract["condition_component"]
    field_component = contract["field_component"]
    state_dimension = contract["state_dimension"]
    reconstruction_weight = contract["reconstruction_weight"]
    parent_component_ids = contract["parent_component_ids"]
    metric = None if contract["metric"] is None else str(contract["metric"])
    weighting = (
        None if contract["weighting"] is None else str(contract["weighting"])
    )
    history_num_per_term = contract["history_num_per_term"]
    uniform_mixture_probability = contract["uniform_mixture_probability"]
    solver_steps = int(contract["solver_steps"])
    sampling_steps = int(contract["sampling_steps"])
    noise_scale = float(contract["noise_scale"])
    fixed_first_beta = float(contract["fixed_first_beta"])
    schedule_start = float(contract["schedule_start"])
    schedule_end = float(contract["schedule_end"])
    mechanical_binding = str(contract["mechanical_binding"])

    class CompilerBoundDiffusionFlowModel(model_class):  # type: ignore[misc, valid-type]
        __recclaw_diffusion_flow_mechanical_binding__ = mechanical_binding
        __recclaw_diffusion_condition_component__ = condition_component

        def __init__(self, config: Any, dataset: Any) -> None:
            # The outer machine-owned lifecycle wrapper is the sole initializer
            # dispatcher.  It calls this constructor and then invokes the
            # optional mechanism hook exactly once after cardinality binding.
            model_class.__init__(self, config, dataset)
            self.recclaw_state_dimension = (
                self.n_items if state_dimension is None else state_dimension
            )
            if state_dimension is not None:
                # State coordinates change the DNN endpoints, never the catalog
                # cardinality or the parent's hidden/time/activation mechanics.
                self.mlp = type(self.mlp)(
                    dims=[state_dimension, *config["dims_dnn"], state_dimension],
                    emb_size=self.emb_size,
                    time_type="cat",
                    norm=self.norm,
                    act_func=self.mlp_act_func,
                ).to(self.device)
            if int(self.steps) != solver_steps:
                raise DiffusionFlowMechanismBindingError(
                    "runtime DiffRec steps differ from the compiled parent solver"
                )
            if int(self.sampling_steps) != sampling_steps:
                raise DiffusionFlowMechanismBindingError(
                    "runtime sampling_steps differ from the compiled parent solver"
                )
            if bool(self.sampling_noise):
                raise DiffusionFlowMechanismBindingError(
                    "compiled parent solver is deterministic but runtime sampling_noise is true"
                )
            if (
                self.noise_schedule != "linear"
                or float(self.noise_min) != schedule_start
                or float(self.noise_max) != schedule_end
                or float(self.noise_scale) != noise_scale
                or self.beta_fixed is not True
            ):
                raise DiffusionFlowMechanismBindingError(
                    "runtime VP linear schedule/noise semantics differ from the compiled parent"
                )
            import torch

            expected_betas = torch.linspace(
                schedule_start * noise_scale,
                schedule_end * noise_scale,
                solver_steps,
                device=self.betas.device,
                dtype=self.betas.dtype,
            )
            expected_betas[0] = fixed_first_beta
            if not torch.allclose(
                self.betas,
                expected_betas,
                rtol=1.0e-7,
                atol=1.0e-12,
            ):
                raise DiffusionFlowMechanismBindingError(
                    "runtime VP beta path differs from the compiled linear schedule"
                )
            if objective_mode == "PARENT_X0" and (
                getattr(getattr(self, "mean_type", None), "name", None) != "START_X"
                or getattr(self, "reweight", None) is not True
                or int(getattr(self, "history_num_per_term", -1))
                != int(history_num_per_term)
            ):
                raise DiffusionFlowMechanismBindingError(
                    "runtime DiffRec x0, SNR reweighting, or loss-history semantics "
                    "differ from the compiled parent objective"
                )

        def recclaw_vp_coefficients(self, timestep: Any, shape: Any) -> tuple[Any, Any]:
            alpha = parent_class._extract_into_tensor(
                self, self.sqrt_alphas_cumprod, timestep, shape
            )
            sigma = parent_class._extract_into_tensor(
                self, self.sqrt_one_minus_alphas_cumprod, timestep, shape
            )
            return alpha, sigma

        def recclaw_vp_velocity_target(
            self, x_start: Any, noise: Any, timestep: Any
        ) -> Any:
            # This is the VP v-parameterization paired exactly with q_sample:
            # x_t = alpha*x_0 + sigma*eps; v = alpha*eps - sigma*x_0.
            alpha, sigma = self.recclaw_vp_coefficients(timestep, x_start.shape)
            return alpha * noise - sigma * x_start

        def recclaw_vp_xstart_from_velocity(
            self, state: Any, velocity: Any, timestep: Any
        ) -> Any:
            # The inverse rotation is exact because alpha**2 + sigma**2 = 1.
            alpha, sigma = self.recclaw_vp_coefficients(timestep, state.shape)
            return alpha * state - sigma * velocity

        def _recclaw_validate_local_tensor(
            self,
            value: Any,
            reference: Any,
            hook_name: str,
        ) -> Any:
            import torch

            if (
                not isinstance(value, torch.Tensor)
                or value.shape != reference.shape
                or value.device != reference.device
                or not torch.isfinite(value).all().item()
            ):
                raise DiffusionFlowMechanismBindingError(
                    f"{hook_name} must return one finite tensor with the parent state "
                    "shape and device"
                )
            return value

        def _recclaw_bound_state(self, history: Any) -> Any:
            if state_hook is None:
                return history
            state = state_hook(self, history)
            if state_dimension is not None:
                import torch

                if (
                    not isinstance(state, torch.Tensor)
                    or state.shape != (history.shape[0], state_dimension)
                    or state.device != history.device
                    or not torch.isfinite(state).all().item()
                ):
                    raise DiffusionFlowMechanismBindingError(
                        "recclaw_diffusion_state must return batch-aligned "
                        "coordinates at the declared spectral rank or latent dimension"
                    )
                return state
            return self._recclaw_validate_local_tensor(
                state, history, _STATE_HOOK
            )

        def _recclaw_component_inputs(
            self, component: Any, history: Any, clean_state: Any,
            state: Any, timestep: Any, condition: Any = None,
        ) -> dict[str, Any]:
            data_attributes = {
                "TRAIN_ITEM_GRAPH": "recclaw_item_graph",
                "TRAIN_SPECTRAL_BASIS": "recclaw_spectral_basis",
                "TRAIN_STATISTICS": "recclaw_item_popularity",
            }
            parent_outputs = {
                (parent_component_ids["state"], "state"): clean_state,
                (parent_component_ids["forward"], "state"): state,
                (parent_component_ids["forward"], "time"): timestep,
                (parent_component_ids["time"], "time"): timestep,
            }
            if condition_component is not None:
                parent_outputs[(condition_component["component_id"], "condition")] = condition
            inputs = {}
            for edge in component["inputs"]:
                source = edge["source"]
                if source["kind"] == "DATA":
                    role = source["data_role"]
                    value = (
                        history if role == "TRAIN_USER_INTERACTION_SIGNAL"
                        else getattr(self, data_attributes[role])
                    )
                else:
                    value = parent_outputs[(source["component_id"], source["output_port"])]
                inputs[edge["port"]] = value
            return inputs

        def _recclaw_bound_condition(
            self, history: Any, clean_state: Any, state: Any, timestep: Any,
            *, training: bool,
        ) -> Any:
            if condition_hook is None:
                return None
            import torch

            inputs = self._recclaw_component_inputs(
                condition_component, history, clean_state, state, timestep
            )
            condition = condition_hook(
                self, inputs, dict(condition_component["parameters"]), training=training
            )
            if (
                not isinstance(condition, torch.Tensor)
                or condition.device != state.device
                or not torch.isfinite(condition).all().item()
            ):
                raise DiffusionFlowMechanismBindingError(
                    "the condition node must return a finite tensor on the field device"
                )
            return condition

        def _recclaw_bound_field(
            self,
            state: Any,
            timestep: Any,
            condition: Any = None,
            *,
            context: Any = None,
            training: bool = False,
        ) -> Any:
            import torch

            if field_input_hook is not None:
                if context is None:
                    raise DiffusionFlowMechanismBindingError(
                        "the declared field inputs require the active batch history"
                    )
                inputs = self._recclaw_component_inputs(
                    field_component, *context, state, timestep, condition
                )
                field = field_input_hook(
                    self, inputs, dict(field_component["parameters"]), training=training
                )
            else:
                field = (
                    # DiffRec owns the DNN instance; it is not a class method.
                    self.mlp(state, timestep)
                    if field_hook is None
                    else field_hook(self, state, timestep, condition)
                )
            if (
                not isinstance(field, torch.Tensor)
                or field.shape != state.shape
                or not torch.isfinite(field).all().item()
            ):
                raise DiffusionFlowMechanismBindingError(
                    "the bound DiffRec field must return one finite tensor with the state shape"
                )
            return field

        def calculate_loss(self, interaction: Any) -> Any:
            import torch
            import torch.nn.functional as functional

            user = interaction[self.USER_ID]
            history = self.get_rating_matrix(user)
            clean_state = self._recclaw_bound_state(history)
            batch_size = clean_state.shape[0]
            timestep, probability = parent_class.sample_timesteps(
                self,
                batch_size,
                clean_state.device,
                "uniform" if objective_mode == "VP_VELOCITY" else "importance",
                *(
                    ()
                    if objective_mode == "VP_VELOCITY"
                    else (float(uniform_mixture_probability),)
                ),
            )
            noise = torch.randn_like(clean_state)
            state = parent_class.q_sample(self, clean_state, timestep, noise)
            field = self._recclaw_bound_field(
                state,
                timestep,
                self._recclaw_bound_condition(
                    history, clean_state, state, timestep, training=True
                ),
                context=(history, clean_state),
                training=True,
            )
            if objective_mode == "VP_VELOCITY":
                target = self.recclaw_vp_velocity_target(
                    clean_state, noise, timestep
                )
                if metric == "HUBER":
                    error = functional.smooth_l1_loss(
                        field, target, reduction="none"
                    ).mean(dim=tuple(range(1, field.ndim)))
                else:
                    error = ((field - target) ** 2).mean(
                        dim=tuple(range(1, field.ndim))
                    )
                if weighting == "PATH_SPEED":
                    width = max(math.prod(target.shape[1:]), 1)
                    weight = target.flatten(1).norm(dim=1) / math.sqrt(width)
                elif weighting == "ENDPOINT_HEAVY":
                    denominator = float(max(int(self.steps) - 1, 1))
                    phase = timestep.to(dtype=error.dtype) / denominator
                    weight = 1.0 + (2.0 * phase - 1.0).abs()
                else:
                    weight = torch.ones_like(error)
                reloss = error * weight
                self.recclaw_velocity_norms = (
                    field.detach().flatten(1).norm(dim=1)
                )
                self.recclaw_path_alignment = functional.cosine_similarity(
                    field.detach().flatten(1),
                    target.detach().flatten(1),
                    dim=1,
                    eps=1.0e-8,
                )
            else:
                target = clean_state
                error = ((field - target) ** 2).mean(
                    dim=tuple(range(1, field.ndim))
                )
                reloss = parent_class.reweight_loss(
                    self,
                    clean_state,
                    state,
                    error,
                    timestep,
                    target,
                    field,
                    clean_state.device,
                )
                parent_class.update_Lt_history(self, timestep, reloss)
            loss = (reloss / probability.to(reloss.dtype)).mean()
            if reconstruction_weight != 0.0:
                # Train the autoencoder on clean history once, independently
                # of diffusion timestep sampling and SNR reweighting.
                reconstructed = self._recclaw_validate_local_tensor(
                    decode_hook(self, clean_state), history, _DECODE_HOOK
                )
                loss = loss + reconstruction_weight * (
                    (reconstructed - history) ** 2
                ).mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite compiler-owned DiffRec field loss")
            return loss

        def p_mean_variance(self, state: Any, timestep: Any) -> dict[str, Any]:
            active_batch = getattr(self, "_recclaw_active_condition", None)
            context = active_batch[:2] if active_batch is not None else None
            condition = (
                self._recclaw_bound_condition(
                    *context, state, timestep, training=False
                )
                if active_batch is not None and active_batch[2] else None
            )
            field = self._recclaw_bound_field(
                state,
                timestep,
                condition,
                context=context,
            )
            predicted_start = (
                self.recclaw_vp_xstart_from_velocity(state, field, timestep)
                if objective_mode == "VP_VELOCITY"
                else field
            )
            mean, variance, log_variance = parent_class.q_posterior_mean_variance(
                self,
                x_start=predicted_start,
                x_t=state,
                t=timestep,
            )
            self.recclaw_field_evaluations = (
                int(getattr(self, "recclaw_field_evaluations", 0)) + 1
            )
            return {
                "mean": mean,
                "variance": variance,
                "log_variance": log_variance,
                "pred_xstart": predicted_start,
            }

        def p_sample(self, x_start: Any) -> Any:
            self.recclaw_field_evaluations = 0
            # The loop, timestep order, optional initial corruption, and
            # posterior transition remain byte-behavioral parent mechanics.
            clean_state = self._recclaw_bound_state(x_start)
            missing = object()
            previous = getattr(self, "_recclaw_active_condition", missing)
            self._recclaw_active_condition = (
                (x_start, clean_state, True)
                if conditioned or field_input_hook is not None else None
            )
            try:
                recovered = parent_class.p_sample(self, clean_state)
                if decode_hook is None:
                    return recovered
                return self._recclaw_validate_local_tensor(
                    decode_hook(self, recovered), x_start, _DECODE_HOOK
                )
            finally:
                if previous is missing:
                    delattr(self, "_recclaw_active_condition")
                else:
                    self._recclaw_active_condition = previous

        def full_sort_predict(self, interaction: Any) -> Any:
            return parent_class.full_sort_predict(self, interaction)

        def predict(self, interaction: Any) -> Any:
            return parent_class.predict(self, interaction)

        def recclaw_recovery_trajectory(
            self,
            interaction: Any,
            *,
            condition_enabled: bool,
            seed: int,
        ) -> Any:
            import torch

            user = interaction[self.USER_ID]
            x_start = self.get_rating_matrix(user)
            clean_state = self._recclaw_bound_state(x_start)
            state = clean_state
            generator = torch.Generator(device=state.device)
            generator.manual_seed(int(seed))
            if int(self.sampling_steps) != 0:
                initial_timestep = torch.full(
                    (state.shape[0],),
                    int(self.sampling_steps) - 1,
                    device=state.device,
                    dtype=torch.long,
                )
                initial_noise = torch.randn(
                    state.shape,
                    device=state.device,
                    dtype=state.dtype,
                    generator=generator,
                )
                state = parent_class.q_sample(
                    self, state, initial_timestep, initial_noise
                )
            states = []
            self.recclaw_field_evaluations = 0
            missing = object()
            previous = getattr(self, "_recclaw_active_condition", missing)
            self._recclaw_active_condition = (
                (x_start, clean_state, condition_enabled)
                if (conditioned and condition_enabled) or field_input_hook is not None else None
            )
            try:
                for index in range(int(self.steps) - 1, -1, -1):
                    timestep = torch.full(
                        (state.shape[0],),
                        index,
                        device=state.device,
                        dtype=torch.long,
                    )
                    transition = self.p_mean_variance(state, timestep)
                    state = transition["mean"]
                    if not torch.isfinite(state).all().item():
                        raise FloatingPointError(
                            "nonfinite canonical recovery trajectory"
                        )
                    states.append(state)
            finally:
                if previous is missing:
                    delattr(self, "_recclaw_active_condition")
                else:
                    self._recclaw_active_condition = previous
            return torch.stack(states, dim=0)

    CompilerBoundDiffusionFlowModel.__name__ = model_class.__name__
    CompilerBoundDiffusionFlowModel.__qualname__ = model_class.__qualname__
    CompilerBoundDiffusionFlowModel.__module__ = model_class.__module__
    return CompilerBoundDiffusionFlowModel


def validate_masked_spectral_condition(model: Any, history: Any) -> None:
    """Falsify missing masks on the actual candidate hook during qualification."""
    import torch

    spec = model.__recclaw_diffusion_condition_component__
    parameters = dict(spec["parameters"])
    basis = model.recclaw_spectral_basis
    history = history[:8].clone()
    devices = [history.device.index] if history.is_cuda else []
    previous_training = model.training
    try:
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            cpu_rng = torch.get_rng_state()
            device_rng = torch.cuda.get_rng_state(history.device) if history.is_cuda else None

            def condition(value: Any, overrides: Mapping[str, Any], training: bool) -> Any:
                torch.set_rng_state(cpu_rng)
                if device_rng is not None:
                    torch.cuda.set_rng_state(device_rng, history.device)
                return model.recclaw_diffusion_condition(
                    {"history": value.clone(), "basis": basis},
                    {**parameters, **overrides}, training=training,
                )

            unmasked = history @ basis
            no_dropout = {"history_mask_probability": 0.0, "unconditional_probability": 0.0}
            for training, overrides in ((False, {}), (True, no_dropout)):
                observed = condition(history, overrides, training)
                if observed.shape != unmasked.shape or not torch.allclose(observed, unmasked, rtol=1e-5, atol=1e-6):
                    raise DiffusionFlowMechanismBindingError(
                        "masked spectral condition must project the unmasked history "
                        "during evaluation and when both dropout probabilities are zero"
                    )
            for parameter in ("history_mask_probability", "unconditional_probability"):
                overrides = {**no_dropout, parameter: 1.0}
                personal = condition(history, overrides, True)
                empty = condition(torch.zeros_like(history), overrides, True)
                if personal.shape != empty.shape or not torch.allclose(personal, empty, rtol=1e-5, atol=1e-6):
                    raise DiffusionFlowMechanismBindingError(
                        f"condition still depends on personal history when {parameter}=1"
                    )
    finally:
        model.train(previous_training)


__all__ = [
    "DiffusionFlowMechanismBindingError",
    "bind_diffusion_flow_model_class",
    "diffusion_flow_profile_contract",
]
