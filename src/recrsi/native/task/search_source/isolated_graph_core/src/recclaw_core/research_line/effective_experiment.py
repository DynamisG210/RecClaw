"""Execution-level identities for strict BL-ICF research experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.mechanism_space import CompileStatus, compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw


def effective_experiment_identity(program: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact behavior and parameter-free family identities.

    The compiler digest preserves declared primitive spelling.  This identity
    additionally normalizes the algebraic MF cosine aliases observed in the
    canary while retaining every resolved component, custom definition,
    operator, and parameter that can change execution.
    """

    report = compile_program(deep_thaw(program))
    if (
        report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION
        or report.resolved_ir is None
    ):
        raise ValueError("effective identity requires a valid strict BL-ICF program")
    ir = deep_thaw(report.resolved_ir)
    payload = program.get("program_payload")
    rows = ir.get("components") if isinstance(ir, Mapping) else None
    order = ir.get("component_order") if isinstance(ir, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not isinstance(order, Sequence)
        or isinstance(order, (str, bytes))
    ):
        raise ValueError("effective identity requires complete resolved mechanism IR")
    components_by_id = {
        str(row.get("component_id") or ""): dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("component_id") or "")
    }
    component_order = tuple(str(value) for value in order)
    if not component_order or set(component_order) != set(components_by_id):
        raise ValueError("resolved mechanism component order is incomplete")
    aliases = {
        component_id: f"component_{index:03d}"
        for index, component_id in enumerate(component_order)
    }
    primitive_ids = {
        str(row.get("primitive_id") or "")
        for row in components_by_id.values()
        if str(row.get("primitive_id") or "")
    }
    has_untransformed_mf_path = (
        "encoder.none_mf" in primitive_ids
        and not any(
            value.startswith(("message.", "propagation.", "fusion."))
            for value in primitive_ids
        )
    )
    cosine_score_ids = {
        "score.cosine_similarity",
        "score.normalized_dot_product",
    }
    normalize_mf_cosine_alias = has_untransformed_mf_path and (
        bool(primitive_ids & cosine_score_ids)
        or (
            "embedding.normalized" in primitive_ids
            and "score.dot_product" in primitive_ids
        )
    )

    def normalized_primitive(value: str) -> str:
        if value == "score.normalized_dot_product":
            return "score.cosine_similarity"
        if normalize_mf_cosine_alias and value == "score.dot_product":
            return "score.cosine_similarity"
        if normalize_mf_cosine_alias and value == "embedding.normalized":
            return "embedding.independent_user_item"
        return value

    scalar_component_ref_keys = frozenset(
        {
            "component_id",
            "source_component_id",
            "target_component_id",
        }
    )
    sequence_component_ref_keys = frozenset(
        {
            "component_ids",
            "remove_component_ids",
            "replacements",
            "targets",
        }
    )

    def normalize_refs(value: Any, *, key_context: str | None = None) -> Any:
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, child in value.items():
                normalized[str(key)] = normalize_refs(
                    child,
                    key_context=str(key),
                )
            return canonical_value(normalized)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return canonical_value(
                [
                    (
                        aliases[str(item)]
                        if key_context in sequence_component_ref_keys
                        and str(item) in aliases
                        else normalize_refs(item)
                    )
                    for item in value
                ]
            )
        if (
            isinstance(value, str)
            and (
                key_context in scalar_component_ref_keys
                or (
                    isinstance(key_context, str)
                    and key_context.endswith("_component_id")
                )
            )
            and value in aliases
        ):
            return aliases[value]
        return canonical_value(value)

    custom_family_by_id: dict[str, str] = {}
    for definition in ir.get("custom_component_definitions") or []:
        if not isinstance(definition, Mapping):
            continue
        custom_id = definition.get("custom_component_id")
        if not isinstance(custom_id, str) or not custom_id:
            continue
        custom_family_by_id[custom_id] = sha256_digest(
            canonical_value(
                {
                    key: definition.get(key)
                    for key in (
                        "slot_id",
                        "mathematical_definition",
                        "algorithm_definition",
                        "input_ports",
                        "output_ports",
                        "allowed_read_roles",
                    )
                }
            )
        )

    effective_components: list[dict[str, Any]] = []
    family_components: list[dict[str, Any]] = []
    for component_id in component_order:
        source = components_by_id[component_id]
        primitive_id = normalized_primitive(
            str(source.get("primitive_id") or "")
        )
        effective = {
            key: normalize_refs(value)
            for key, value in source.items()
            if key != "component_id"
        }
        effective["component_id"] = aliases[component_id]
        if primitive_id:
            effective["primitive_id"] = primitive_id
        effective_components.append(canonical_value(effective))
        family_component = {
            "slot_id": source.get("slot_id"),
            "primitive_id": primitive_id or None,
            "inputs": normalize_refs(source.get("inputs") or []),
        }
        custom_id = source.get("custom_component_id")
        if isinstance(custom_id, str) and custom_id:
            family_component["custom_component_semantics"] = (
                custom_family_by_id[custom_id]
            )
        else:
            family_component["custom_component_id"] = None
        family_components.append(canonical_value(family_component))

    family_operators = [
        canonical_value(
            {
                "operator_id": str(row.get("operator_id") or ""),
                "targets": sorted(
                    normalize_refs(
                        row.get("targets") or [],
                        key_context="targets",
                    )
                ),
                "replacements": sorted(
                    normalize_refs(
                        row.get("replacements") or [],
                        key_context="replacements",
                    )
                ),
            }
        )
        for row in (ir.get("architecture_operators") or [])
        if isinstance(row, Mapping) and str(row.get("operator_id") or "")
    ]
    family_operators.sort(
        key=lambda row: (
            row["operator_id"],
            tuple(row["targets"]),
            tuple(row["replacements"]),
        )
    )

    effective_operators = []
    for row in ir.get("architecture_operators") or []:
        if not isinstance(row, Mapping):
            continue
        scientific_row = dict(row)
        parameters = scientific_row.get("parameters")
        if isinstance(parameters, Mapping):
            scientific_parameters = dict(parameters)
            scientific_parameters.pop("source_ownership_mode", None)
            scientific_row["parameters"] = scientific_parameters
        effective_operators.append(normalize_refs(scientific_row))

    descriptor = canonical_value(
        {
            "schema": "recclaw.bl-icf.effective-experiment.v1",
            "construction_mode": payload.get("construction_mode"),
            "input_semantics": payload.get("input_semantics"),
            "components": effective_components,
            "custom_component_definitions": normalize_refs(
                ir.get("custom_component_definitions") or []
            ),
            "architecture_operators": effective_operators,
            "algebraic_normalizations": (
                ["UNTRANSFORMED_MF_COSINE_GEOMETRY"]
                if normalize_mf_cosine_alias
                else []
            ),
        }
    )
    family_descriptor_value = {
        "schema": "recclaw.bl-icf.effective-experiment-family.v2",
        "components": family_components,
        "architecture_operators": family_operators,
        "removed_slots": sorted(
            str(slot_id) for slot_id in (payload.get("removed_slots") or [])
        ),
    }
    if custom_family_by_id:
        family_descriptor_value["custom_component_families"] = sorted(
            custom_family_by_id.values()
        )
    family_descriptor = canonical_value(family_descriptor_value)
    return canonical_value(
        {
            "effective_experiment_digest": sha256_digest(descriptor),
            "effective_family_digest": sha256_digest(family_descriptor),
            "primitive_ids": tuple(sorted(primitive_ids)),
        }
    )
