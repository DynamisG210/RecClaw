"""Machine-owned tensor geometry for LIGER product quantizers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_PRODUCT_QUANTIZERS = {
    "tokenizer.product_quantization",
    "tokenizer.orthogonally_preconditioned_product_quantization",
}
_PROJECTION_DIMENSION = 128
_BINDING_ATTRIBUTE = "_recclaw_liger_product_geometry"


def _product_shape(
    component_specs: Mapping[str, Any],
) -> tuple[int, int] | None:
    tokenizer = next(
        (
            spec
            for spec in component_specs.values()
            if isinstance(spec, Mapping)
            and spec.get("primitive_id") in _PRODUCT_QUANTIZERS
            and isinstance(spec.get("parameters"), Mapping)
        ),
        None,
    )
    if tokenizer is None:
        return None
    subspaces = int(tokenizer["parameters"]["subspaces"])
    return subspaces, _PROJECTION_DIMENSION // subspaces


def partition_liger_hidden(
    hidden: Any,
    *,
    subspaces: int,
    subspace_dimension: int,
) -> Any:
    """Partition only the final 128-wide LIGER projection dimension."""

    import torch

    if not isinstance(hidden, torch.Tensor):
        raise TypeError("LIGER hidden state must be a torch.Tensor")
    if hidden.shape[-1] != subspaces * subspace_dimension:
        raise ValueError("LIGER product geometry requires a 128-wide hidden state")
    return hidden.reshape(*hidden.shape[:-1], subspaces, subspace_dimension)


def rotate_liger_hidden(
    hidden: Any,
    rotation: Any,
    *,
    subspaces: int,
    subspace_dimension: int,
) -> Any:
    """Apply one declared rotation to each product subspace."""

    import torch

    blocks = partition_liger_hidden(
        hidden,
        subspaces=subspaces,
        subspace_dimension=subspace_dimension,
    )
    if not isinstance(rotation, torch.Tensor) or tuple(rotation.shape) != (
        subspaces,
        subspace_dimension,
        subspace_dimension,
    ):
        raise ValueError("LIGER block rotation has the wrong product geometry")
    return torch.einsum("...sd,sde->...se", blocks, rotation)


def gather_liger_codebook(
    codebook: Any,
    assignments: Any,
    *,
    subspaces: int,
    subspace_dimension: int,
) -> Any:
    """Gather one code vector per product subspace without flattening it."""

    import torch

    if not isinstance(codebook, torch.Tensor) or (
        codebook.ndim != 3
        or codebook.shape[0] != subspaces
        or codebook.shape[-1] != subspace_dimension
    ):
        raise ValueError("LIGER codebook has the wrong product geometry")
    if not isinstance(assignments, torch.Tensor) or (
        assignments.ndim < 1 or assignments.shape[-1] != subspaces
    ):
        raise ValueError("LIGER assignments must end in the product subspaces")
    subspace_index = torch.arange(
        subspaces,
        device=assignments.device,
        dtype=torch.long,
    ).reshape(*([1] * (assignments.ndim - 1)), subspaces)
    return codebook[subspace_index, assignments.long()]


def flatten_liger_blocks(
    blocks: Any,
    *,
    subspaces: int,
    subspace_dimension: int,
) -> Any:
    """Restore the parent 128-wide boundary after block-local operations."""

    import torch

    if not isinstance(blocks, torch.Tensor) or tuple(blocks.shape[-2:]) != (
        subspaces,
        subspace_dimension,
    ):
        raise ValueError("LIGER blocks have the wrong product geometry")
    return blocks.reshape(*blocks.shape[:-2], subspaces * subspace_dimension)


def bind_liger_product_quantizer_model_class(
    model_class: type,
    component_specs: Mapping[str, Any],
) -> type:
    """Expose compiler-owned product geometry to candidate mechanism code."""

    shape = _product_shape(component_specs)
    if shape is None or getattr(model_class, _BINDING_ATTRIBUTE, None) == shape:
        return model_class
    subspaces, subspace_dimension = shape

    class _LigerProductGeometryModel(model_class):
        def recclaw_liger_partition_hidden(self, hidden: Any) -> Any:
            return partition_liger_hidden(
                hidden,
                subspaces=subspaces,
                subspace_dimension=subspace_dimension,
            )

        def recclaw_liger_rotate_hidden(self, hidden: Any, rotation: Any) -> Any:
            return rotate_liger_hidden(
                hidden,
                rotation,
                subspaces=subspaces,
                subspace_dimension=subspace_dimension,
            )

        def recclaw_liger_gather_codebook(
            self,
            codebook: Any,
            assignments: Any,
        ) -> Any:
            return gather_liger_codebook(
                codebook,
                assignments,
                subspaces=subspaces,
                subspace_dimension=subspace_dimension,
            )

        def recclaw_liger_flatten_blocks(self, blocks: Any) -> Any:
            return flatten_liger_blocks(
                blocks,
                subspaces=subspaces,
                subspace_dimension=subspace_dimension,
            )

    setattr(_LigerProductGeometryModel, _BINDING_ATTRIBUTE, shape)
    _LigerProductGeometryModel.__name__ = model_class.__name__
    _LigerProductGeometryModel.__qualname__ = model_class.__qualname__
    _LigerProductGeometryModel.__module__ = model_class.__module__
    return _LigerProductGeometryModel


__all__ = [
    "bind_liger_product_quantizer_model_class",
    "flatten_liger_blocks",
    "gather_liger_codebook",
    "partition_liger_hidden",
    "rotate_liger_hidden",
]
