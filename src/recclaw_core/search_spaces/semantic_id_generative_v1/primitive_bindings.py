"""Executable ownership for Semantic-ID primitives.

The exact LIGER parent owns only its complete frozen component signatures.
Every other available affordance is implementable only through the existing
compiler-owned full-source route. Three affordances remain suspended because
their required frozen inputs or teacher ABI do not exist in the active profile.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


class PrimitiveBindingUnavailable(ValueError):
    """A primitive was declared without an executable source-ownership route."""


EXACT_LIGER_PARENT_COMPONENT_SHA256 = {
    "feature.frozen_text_encoder": "46e6bc96972d36f8e1d32240a0e29b5c1a16db26949b0e291f47e5bd870e8d87",
    "tokenizer.frozen_letter_collision_suffix": "471b36e9d3c407188a7b9d268ba450bb2122373a13f287e79c600fc1bf888ce6",
    "context.sid_history_with_frozen_content": "f1c92abf7e72b59d331a559a100cc558b0303e0a607b5c60402c83e38f2ad65d",
    "generator.t5_encoder_decoder": "88cf82a794c21288b38d720b5af7f89613718b02c64b894bcf19a6eb1dcb5ce0",
    "decode.autoregressive_beam_then_invalid_drop": "b5bcd3d2348beb83c0baad41da53c5f1b3512e02d4cb7ec9ebbcdb10f567a060",
    "resolution.invalid_sid_drop_lookup": "dbeb66c6222b59f517fc288ab28f16d73ff35537f07a21804ccaa45bef561943",
    "retrieval.generated_legal_dense_rerank_to_score": "554e3edfdc867cf0e994b74d074c4850ebd95573649a5d4cbf0c83420bec01cd",
    "objective.token_cross_entropy": "413e2f665badb0efb45584254e64a55681d3ab6aa364703b79509e155f5cb3c3",
    "objective.train_seen_full_catalog_dense_cross_entropy": "0d380c8f4a41380ca5faf8ad6a9779ae2682f7275ae550ba1dbd470bc717e62e",
}

SUSPENDED_PRIMITIVES = frozenset(
    {
        "feature.semantic_collaborative_fusion",
        "feature.recommendation_native_structured_field_autoencoder",
        "alignment.dual_collaborative_distillation",
    }
)

# Populated from the Provider's declarative AXES during module initialization.
EXECUTABLE_PRIMITIVE_BINDINGS: dict[str, str] = {}


def _component_sha256(component: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(component),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def executable_axes(axes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Expose every routed affordance and omit only explicitly suspended ones."""

    projected = []
    EXECUTABLE_PRIMITIVE_BINDINGS.clear()
    for axis in axes:
        value = dict(axis)
        primitives = []
        for primitive in axis["primitives"]:
            primitive_id = str(primitive["primitive_id"])
            if primitive_id in SUSPENDED_PRIMITIVES:
                continue
            ownership = (
                "EXACT_LIGER_PARENT_IF_COMPONENT_SIGNATURE_MATCHES_ELSE_FULL_SOURCE_REQUIRED"
                if primitive_id in EXACT_LIGER_PARENT_COMPONENT_SHA256
                else "FULL_SOURCE_REQUIRED"
            )
            EXECUTABLE_PRIMITIVE_BINDINGS[primitive_id] = ownership
            primitives.append({**dict(primitive), "runtime_binding": ownership})
        value["primitives"] = primitives
        projected.append(value)
    return tuple(projected)


def component_binding_projection(
    components: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Resolve source ownership from each complete component signature."""

    result: dict[str, str] = {}
    for component in components:
        primitive_id = component.get("primitive_id")
        if primitive_id is None:
            result[str(component["component_id"])] = "FULL_SOURCE_REQUIRED"
            continue
        primitive_id = str(primitive_id)
        if primitive_id not in EXECUTABLE_PRIMITIVE_BINDINGS:
            raise PrimitiveBindingUnavailable(
                "semantic-ID primitive has no executable binding: " + primitive_id
            )
        exact_digest = EXACT_LIGER_PARENT_COMPONENT_SHA256.get(primitive_id)
        owner = (
            "EXACT_LIGER_PARENT"
            if exact_digest is not None and _component_sha256(component) == exact_digest
            else "FULL_SOURCE_REQUIRED"
        )
        result[str(component["component_id"])] = owner
    return {key: result[key] for key in sorted(result)}


__all__ = [
    "EXECUTABLE_PRIMITIVE_BINDINGS",
    "EXACT_LIGER_PARENT_COMPONENT_SHA256",
    "PrimitiveBindingUnavailable",
    "SUSPENDED_PRIMITIVES",
    "component_binding_projection",
    "executable_axes",
]
