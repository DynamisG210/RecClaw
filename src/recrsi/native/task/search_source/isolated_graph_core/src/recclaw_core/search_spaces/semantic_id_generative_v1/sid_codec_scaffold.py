"""Machine-owned codec boundary for frozen LETTER semantic IDs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_FROZEN_LETTER_TOKENIZER = "tokenizer.frozen_letter_collision_suffix"
_PARALLEL_VALID_ASSIGNMENT = "decode.parallel_valid_assignment"
_BINDING_ATTRIBUTE = "_recclaw_frozen_letter_sid_codec"


def _codec_shape(
    component_specs: Mapping[str, Any],
) -> tuple[int, int] | None:
    tokenizer = next(
        (
            spec
            for spec in component_specs.values()
            if isinstance(spec, Mapping)
            and spec.get("primitive_id") == _FROZEN_LETTER_TOKENIZER
            and isinstance(spec.get("parameters"), Mapping)
        ),
        None,
    )
    parallel_assignment = any(
        isinstance(spec, Mapping)
        and spec.get("primitive_id") == _PARALLEL_VALID_ASSIGNMENT
        for spec in component_specs.values()
    )
    if tokenizer is None or not parallel_assignment:
        return None
    parameters = tokenizer["parameters"]
    return (
        int(parameters["codes_per_level"]),
        int(parameters["semantic_prefix_levels"]) + 1,
    )


def absolute_frozen_letter_sids(
    semantic_ids: Any,
    *,
    codes_per_level: int,
    depth: int,
) -> Any:
    """Convert complete position-local tuples to the catalog's absolute IDs.

    Parallel heads naturally emit one zero-based class per position.  The
    frozen catalog, including the official autoregressive parent, uses one
    shared vocabulary with a distinct offset for each position.  Already
    absolute tuples and invalid sentinel tuples pass through unchanged.
    """

    import torch

    if not isinstance(semantic_ids, torch.Tensor):
        raise TypeError("semantic IDs must be a torch.Tensor")
    if semantic_ids.ndim < 2 or semantic_ids.shape[-1] != depth:
        raise ValueError(f"semantic IDs must end in {depth} positions")
    local_tuple = (
        (semantic_ids >= 0) & (semantic_ids < codes_per_level)
    ).all(dim=-1, keepdim=True)
    offsets = (
        torch.arange(depth, device=semantic_ids.device, dtype=semantic_ids.dtype)
        * codes_per_level
        + 1
    )
    return torch.where(local_tuple, semantic_ids + offsets, semantic_ids)


def bind_frozen_letter_sid_codec_model_class(
    model_class: type,
    component_specs: Mapping[str, Any],
) -> type:
    """Own the local-head to absolute-catalog conversion at one stable seam."""

    shape = _codec_shape(component_specs)
    if shape is None or getattr(model_class, _BINDING_ATTRIBUTE, None) == shape:
        return model_class
    codes_per_level, depth = shape

    class _FrozenLetterSidCodecModel(model_class):
        def recclaw_generate_semantic_ids(self, interaction: Any) -> Any:
            semantic_ids = super().recclaw_generate_semantic_ids(interaction)
            return absolute_frozen_letter_sids(
                semantic_ids,
                codes_per_level=codes_per_level,
                depth=depth,
            )

        def recclaw_resolve_semantic_ids(self, semantic_ids: Any) -> Any:
            absolute_ids = absolute_frozen_letter_sids(
                semantic_ids,
                codes_per_level=codes_per_level,
                depth=depth,
            )
            return super().recclaw_resolve_semantic_ids(absolute_ids)

    setattr(_FrozenLetterSidCodecModel, _BINDING_ATTRIBUTE, shape)
    _FrozenLetterSidCodecModel.__name__ = model_class.__name__
    _FrozenLetterSidCodecModel.__qualname__ = model_class.__qualname__
    _FrozenLetterSidCodecModel.__module__ = model_class.__module__
    return _FrozenLetterSidCodecModel


__all__ = [
    "absolute_frozen_letter_sids",
    "bind_frozen_letter_sid_codec_model_class",
]
