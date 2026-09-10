"""Compiler-owned SSD4Rec sequence lifecycle for declared mechanism hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_REPRESENTATION_HOOK = "recclaw_sequence_representation"
_STATE_HOOK = "recclaw_sequence_state"
_PRE_BACKBONE_SLOTS = frozenset({"TEMPORAL_POSITION_ENCODING"})
_BACKBONE_SLOTS = frozenset({"SEQUENCE_BACKBONE"})
_POST_BACKBONE_SLOTS = frozenset(
    {"INTEREST_ROUTING", "STATE_UPDATE_GATING"}
)
_SUPPORTED_SEQUENCE_SLOTS = (
    _PRE_BACKBONE_SLOTS
    | _POST_BACKBONE_SLOTS
)
_PARENT_BACKBONE_PRIMITIVE = (
    "backbone.bidirectional_prefix_reversal_state_space_duality"
)
_PARENT_BACKBONE_PARAMETERS = {
    "layers": 2,
    "state_dimension": 64,
    "head_dimension": 16,
    "expansion": 2,
    "local_convolution": 4,
    "backward_weight": 0.1,
    "direction_parameter_sharing": "SHARED",
    "reverse_output_alignment": "AS_EMITTED_REVERSE_ORDER",
    "post_mixer": "RESIDUAL_FFN",
    "ffn_multiplier": 4.0,
}
_EVENT_TIME_ROLE = "TRAIN_PREFIX_TIMESTAMP_SEQUENCE"
_QUERY_TIME_ROLE = "QUERY_TIMESTAMP"
_ITEM_SEQUENCE_ROLE = "TRAIN_PREFIX_ITEM_SEQUENCE"
_USER_ROLE = "USER_ID"
_SECONDS_PER_DAY = 86400.0
_TEMPORAL_CONTEXT = "_recclaw_machine_owned_sequential_temporal_context"


class SequentialScalingMechanismBindingError(ValueError):
    """The compiled P5 program cannot use the parent-preserving scaffold."""


def _seconds_to_days(seconds: Any) -> Any:
    return seconds / _SECONDS_PER_DAY


def _train_interval_days(config: Any, dataset: Any) -> Any:
    """Normalize same-user adjacent TRAIN target gaps, without prefix weighting."""
    import numpy as np
    import torch

    interaction = dataset.inter_feat
    users = interaction[dataset.uid_field].detach().cpu().numpy()
    # Preserve the loader's timestamp precision; promote only before subtraction,
    # exactly as for runtime gaps and the existing TRAIN quantile population.
    timestamps = interaction["timestamp"].detach().cpu().numpy().astype(np.float64)
    chronology = interaction[config["TIME_FIELD"]].detach().cpu().numpy()
    order = np.lexsort((np.arange(users.shape[0]), chronology, users))
    ordered_users = users[order]
    ordered_timestamps = timestamps[order]
    gaps = np.diff(ordered_timestamps)[ordered_users[1:] == ordered_users[:-1]]
    if np.any(gaps < 0) or not np.isfinite(gaps).all():
        raise SequentialScalingMechanismBindingError(
            "TRAIN chronological timestamp gaps must be finite and nonnegative"
        )
    return torch.from_numpy(_seconds_to_days(gaps))


def _changed_slot_ids(value: Sequence[Any]) -> frozenset[str]:
    slots: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise SequentialScalingMechanismBindingError(
                "declared changed slots must contain mappings"
            )
        slot_id = item.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id:
            raise SequentialScalingMechanismBindingError(
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


def _declared_inputs(
    component_specs: Mapping[str, Any], changed_slots: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    ports: set[str] = set()
    data_roles: set[str] = set()
    for spec in component_specs.values():
        if not isinstance(spec, Mapping) or spec.get("slot_id") not in changed_slots:
            continue
        for item in spec.get("inputs", ()):
            if not isinstance(item, Mapping):
                continue
            port = item.get("port")
            if isinstance(port, str):
                ports.add(port)
            source = item.get("source")
            if (
                isinstance(source, Mapping)
                and source.get("kind") == "DATA"
                and isinstance(source.get("data_role"), str)
            ):
                data_roles.add(str(source["data_role"]))
    return frozenset(ports), frozenset(data_roles)


def _stage_hooks(slots: frozenset[str]) -> tuple[str, ...]:
    hooks: list[str] = []
    if slots & _PRE_BACKBONE_SLOTS:
        hooks.append(_REPRESENTATION_HOOK)
    if slots & _POST_BACKBONE_SLOTS:
        hooks.append(_STATE_HOOK)
    return tuple(hooks)


def sequential_scaling_profile_contract(
    component_specs: Mapping[str, Any],
    declared_changed_slots: Sequence[Any],
) -> tuple[str, ...] | None:
    """Return hooks mutable in this delta; inherited hooks remain parent-owned."""

    if not isinstance(component_specs, Mapping):
        raise TypeError("component_specs must be a mapping")
    if not isinstance(declared_changed_slots, (tuple, list)):
        raise TypeError("declared_changed_slots must be a sequence")
    changed_slots = _changed_slot_ids(declared_changed_slots)
    if changed_slots & _BACKBONE_SLOTS:
        raise SequentialScalingMechanismBindingError(
            "the parent-preserving P5 profile cannot replace the exact frozen "
            "BiSSD backbone; a full backbone rewrite must use CUSTOM_MODEL"
        )
    if not changed_slots <= _SUPPORTED_SEQUENCE_SLOTS:
        raise SequentialScalingMechanismBindingError(
            "compiled changed slots lack a canonical sequential profile hook"
        )
    component_slots = frozenset(
        str(spec.get("slot_id"))
        for spec in component_specs.values()
        if isinstance(spec, Mapping) and isinstance(spec.get("slot_id"), str)
    )
    if not changed_slots <= component_slots:
        raise SequentialScalingMechanismBindingError(
            "compiled changed slots lack executable component specifications"
        )
    if not changed_slots and not component_slots & _SUPPORTED_SEQUENCE_SLOTS:
        return None
    parent_backbones = _slot_components(component_specs, "SEQUENCE_BACKBONE")
    if (
        len(parent_backbones) != 1
        or parent_backbones[0].get("primitive_id") != _PARENT_BACKBONE_PRIMITIVE
        or not isinstance(parent_backbones[0].get("parameters"), Mapping)
        or dict(parent_backbones[0]["parameters"]) != _PARENT_BACKBONE_PARAMETERS
    ):
        raise SequentialScalingMechanismBindingError(
            "parent-preserving P5 mechanics require the exact frozen BiSSD "
            "backbone as the active construction anchor"
        )

    return _stage_hooks(changed_slots) or None


def bind_sequential_scaling_model_class(
    model_class: type[Any],
    component_specs: Mapping[str, Any],
    declared_changed_slots: Sequence[Any],
) -> type[Any]:
    """Bind graph-staged hooks into the exact parent sequence lifecycle."""

    if not isinstance(model_class, type):
        raise TypeError("model_class must be a class")
    if not isinstance(component_specs, Mapping):
        raise TypeError("component_specs must be a mapping")
    if not isinstance(declared_changed_slots, (tuple, list)):
        raise TypeError("declared_changed_slots must be a sequence")
    sequential_scaling_profile_contract(component_specs, declared_changed_slots)
    # Execution follows the complete active graph. The current delta controls
    # Implementer ownership only; retained stages and their DATA inputs stay live.
    contract = _stage_hooks(frozenset(
        str(spec.get("slot_id"))
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
    ))
    if not contract:
        return model_class
    hooks = {name: model_class.__dict__.get(name) for name in contract}
    missing_hooks = sorted(name for name, hook in hooks.items() if not callable(hook))
    if missing_hooks:
        raise SequentialScalingMechanismBindingError(
            "compiled sequence mechanics require " + ", ".join(missing_hooks)
        )
    representation_hook = hooks.get(_REPRESENTATION_HOOK)
    state_hook = hooks.get(_STATE_HOOK)
    parent_loss = getattr(model_class, "calculate_loss")
    parent_predict = getattr(model_class, "predict")
    parent_full_sort = getattr(model_class, "full_sort_predict")
    _, representation_roles = _declared_inputs(
        component_specs, _PRE_BACKBONE_SLOTS
    )
    state_ports, state_roles = _declared_inputs(
        component_specs, _POST_BACKBONE_SLOTS
    )
    data_roles = representation_roles | state_roles
    uses_event_time = _EVENT_TIME_ROLE in data_roles
    uses_query_time = _QUERY_TIME_ROLE in data_roles
    uses_item_sequence = (
        "items" in state_ports or _ITEM_SEQUENCE_ROLE in state_roles
    )
    uses_user = "user" in state_ports or _USER_ROLE in state_roles
    uses_representation = "representation" in state_ports
    representation_uses_event_time = _EVENT_TIME_ROLE in representation_roles
    representation_uses_query_time = _QUERY_TIME_ROLE in representation_roles
    state_uses_event_time = _EVENT_TIME_ROLE in state_roles
    state_uses_query_time = _QUERY_TIME_ROLE in state_roles

    def runtime_context(self: Any, interaction: Any) -> dict[str, Any]:
        return {
            "timestamps": interaction["timestamp_list"] if uses_event_time else None,
            "query_timestamp": interaction["timestamp"] if uses_query_time else None,
            "item_seq": interaction[self.ITEM_SEQ] if uses_item_sequence else None,
            "user_ids": interaction[self.USER_ID] if uses_user else None,
        }

    def call_parent(self: Any, method: Any, interaction: Any) -> Any:
        missing = object()
        previous = getattr(self, _TEMPORAL_CONTEXT, missing)
        setattr(self, _TEMPORAL_CONTEXT, runtime_context(self, interaction))
        try:
            return method(self, interaction)
        finally:
            if previous is missing:
                delattr(self, _TEMPORAL_CONTEXT)
            else:
                setattr(self, _TEMPORAL_CONTEXT, previous)

    class CompilerBoundSequentialScalingModel(model_class):  # type: ignore[misc, valid-type]
        __recclaw_sequential_scaling_mechanical_binding__ = (
            "GRAPH_STAGED_ACTIVE_PARENT_ANCHOR_V3"
        )

        def forward(self, item_seq: Any, item_seq_len: Any) -> Any:
            import torch

            values = self.item_embedding(item_seq)
            if self.norm_embedding:
                values = self.dropout(values)
                values = self.layer_norm(values)
            positions = torch.arange(
                item_seq.shape[1], device=item_seq.device
            ).unsqueeze(0)
            lengths = item_seq_len.to(device=item_seq.device, dtype=torch.long)
            valid_mask = positions < lengths.unsqueeze(1)
            values = values * valid_mask.unsqueeze(-1).to(values.dtype)
            elapsed_days = values.new_zeros(item_seq.shape)
            query_elapsed_days = None
            context = getattr(self, _TEMPORAL_CONTEXT, None)
            if (uses_event_time or uses_query_time or uses_user) and context is None:
                raise SequentialScalingMechanismBindingError(
                    "declared time/user mechanics require an interaction-owned context"
                )
            if uses_event_time:
                timestamps = context["timestamps"].to(device=item_seq.device)
                if timestamps.ndim != 2 or timestamps.shape[1] < item_seq.shape[1]:
                    raise SequentialScalingMechanismBindingError(
                        "timestamp_list is not aligned with the padded item sequence"
                    )
                timestamps = timestamps[:, : item_seq.shape[1]].to(torch.float64)
                if item_seq.shape[1] > 1:
                    adjacent_valid = valid_mask[:, 1:]
                    raw_seconds = timestamps[:, 1:] - timestamps[:, :-1]
                    valid_seconds = raw_seconds[adjacent_valid]
                    if bool((valid_seconds < 0).any()) or not bool(
                        torch.isfinite(valid_seconds).all()
                    ):
                        raise SequentialScalingMechanismBindingError(
                            "valid chronological timestamp gaps must be finite and nonnegative"
                        )
                    adjacent_days = values.new_zeros(raw_seconds.shape).masked_scatter(
                        adjacent_valid,
                        _seconds_to_days(valid_seconds).to(values.dtype),
                    )
                    elapsed_days = torch.cat(
                        (values.new_zeros((values.shape[0], 1)), adjacent_days),
                        dim=1,
                    )
                if uses_query_time:
                    query = context["query_timestamp"].to(
                        device=item_seq.device, dtype=torch.float64
                    ).reshape(-1)
                    last_time = timestamps.gather(
                        1, (lengths - 1).unsqueeze(1)
                    ).squeeze(1)
                    query_seconds = query - last_time
                    if bool((query_seconds < 0).any()) or not bool(
                        torch.isfinite(query_seconds).all()
                    ):
                        raise SequentialScalingMechanismBindingError(
                            "valid query timestamp gaps must be finite and nonnegative"
                        )
                    query_elapsed_days = _seconds_to_days(query_seconds).to(
                        values.dtype
                    )
            backbone_input = values
            if representation_hook is not None:
                backbone_input = representation_hook(
                    self,
                    values,
                    lengths,
                    elapsed_days if representation_uses_event_time else None,
                    (
                        query_elapsed_days
                        if representation_uses_query_time
                        else None
                    ),
                    valid_mask,
                )
            if (
                not isinstance(backbone_input, torch.Tensor)
                or backbone_input.shape != values.shape
            ):
                raise SequentialScalingMechanismBindingError(
                    "pre-backbone representation must preserve [batch,length,hidden]"
                )
            backbone_input = backbone_input * valid_mask.unsqueeze(-1).to(
                backbone_input.dtype
            )

            features = backbone_input
            for layer in self.bissd_layers:
                features = layer(features, lengths)
            if not isinstance(features, torch.Tensor) or features.shape != values.shape:
                raise SequentialScalingMechanismBindingError(
                    "sequence backbone must preserve [batch,length,hidden]"
                )
            features = features * valid_mask.unsqueeze(-1).to(features.dtype)

            if state_hook is not None:
                source_items = (
                    context["item_seq"]
                    if context is not None and context.get("item_seq") is not None
                    else item_seq if uses_item_sequence else None
                )
                user_ids = (
                    context["user_ids"]
                    if context is not None and context.get("user_ids") is not None
                    else None
                )
                residual = state_hook(
                    self,
                    features,
                    backbone_input if uses_representation else None,
                    source_items,
                    lengths,
                    elapsed_days if state_uses_event_time else None,
                    query_elapsed_days if state_uses_query_time else None,
                    valid_mask,
                    user_ids,
                )
                if not isinstance(residual, torch.Tensor) or residual.shape != values.shape:
                    raise SequentialScalingMechanismBindingError(
                        "post-backbone residual must preserve [batch,length,hidden]"
                    )
                features = features + residual
                features = features * valid_mask.unsqueeze(-1).to(features.dtype)
            return self.gather_indexes(features, lengths - 1)

        def calculate_loss(self, interaction: Any) -> Any:
            return call_parent(self, parent_loss, interaction)

        def predict(self, interaction: Any) -> Any:
            return call_parent(self, parent_predict, interaction)

        def full_sort_predict(self, interaction: Any) -> Any:
            return call_parent(self, parent_full_sort, interaction)

    initializer = getattr(model_class, "recclaw_initialize_mechanism", None)
    if (uses_event_time or uses_query_time) and callable(initializer):
        def initialize_with_train_days(self: Any, config: Any, dataset: Any) -> None:
            self.recclaw_train_interval_days = _train_interval_days(config, dataset)
            initializer(self, config, dataset)

        CompilerBoundSequentialScalingModel.recclaw_initialize_mechanism = (
            initialize_with_train_days
        )

    CompilerBoundSequentialScalingModel.__name__ = model_class.__name__
    CompilerBoundSequentialScalingModel.__qualname__ = model_class.__qualname__
    CompilerBoundSequentialScalingModel.__module__ = model_class.__module__
    return CompilerBoundSequentialScalingModel


__all__ = [
    "SequentialScalingMechanismBindingError",
    "bind_sequential_scaling_model_class",
    "sequential_scaling_profile_contract",
]
