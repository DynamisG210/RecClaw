"""Bind MoRec authoring to its existing axis contract, without repairing drafts.

This module controls future AutoSpace authoring, not the sealed search policy.
It preserves the original schema, family counts and final assembly checks.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from jsonschema import Draft202012Validator


AUTHORING_VERSION = "morec-autospace-main-axis-contract-v1"
AUTHORING_GUIDANCE = """MOREC_AXIS_TARGET_BINDING_V1
The legal_mutation_targets of each slot are stricter than the global mutable-file
list. Plan each family and its implementation inside its own slot's targets.
For main_code.py:train_update_kernel, the target is the existing statements
between E2_SHARED_MUTABLE_TRAIN_UPDATE_START/END inside main, not a new function
or permission to change the outer loop. A supporting baseline anchor in another
file is evidence about a called quantity, not permission to mutate that file.
Keep the planned target, actual effect path, runtime route and baseline anchors
consistent. Do not rename an axis or target to disguise a different intervention.

The objective remains useful algorithmic change, not schema compliance. Keep
three genuinely different proposed mechanism families per axis and the original
cross-axis composition opportunities. Shared filenames do not make mechanisms
equivalent. Distinguish changes to sampled supervision, to the objective, and to
the update rule by their causal effect; do not fill separate axes with aliases
of one idea. All families must be realizable from the supplied baseline within
the declared region, including quantities that the unchanged call interface
actually exposes. Do not move an implementation elsewhere merely to fit a label.

The Critic must check these same targets and routes, in addition to its existing
novelty, reachability, composition and quality review. Judge effects relative to
the supplied baseline: inference ordering alone does not establish training
loss/gradient equivalence. Do not remove a feasible training intervention solely
because a postprocessing step preserves inference ranking. Preserve byte-frozen
text initialization and all original evaluation/data restrictions. Expected gains
are hypotheses; only later common-dev results and feedback establish ability.
"""


def bind_generator_schema(
    schema: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Add per-slot enums; do not mutate cached schemas or scientific contracts."""
    from auto_space_v161 import validate_contract_design

    design = validate_contract_design(contract)
    bound = deepcopy(dict(schema))
    axis_item = bound["properties"]["space_plan"]["properties"]["axes"]["items"]
    primitive_item = bound["properties"]["primitives"]["items"]
    for slot in design["slots"]:
        condition = {
            "required": ["slot_id"],
            "properties": {"slot_id": {"const": slot["slot_id"]}},
        }
        targets = list(slot["legal_mutation_targets"])
        axis_item.setdefault("allOf", []).append({
            "if": deepcopy(condition),
            "then": {"properties": {"mechanism_families": {"items": {
                "properties": {"intended_legal_mutation_target": {"enum": targets}}
            }}}},
        })
        primitive_item.setdefault("allOf", []).append({
            "if": deepcopy(condition),
            "then": {"properties": {
                "legal_mutation_targets": {"items": {"enum": targets}},
                "runtime_reachability": {"properties": {"mutation_target": {"enum": targets}}},
                "effect_path": {"properties": {"mutation_target": {"enum": targets}}},
            }},
        })
    Draft202012Validator.check_schema(bound)
    return bound


def validate_generator_bindings(draft: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Reject the observed failure before spending a Critic call.

The route/anchor checks mirror existing assembly conditions. Supporting model
anchors remain allowed for kernel interventions; no additional scientific family
whitelist, performance gate or automatic target remapping is introduced.
"""
    errors = list(Draft202012Validator(schema).iter_errors(draft))
    if errors:
        details = [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errors[:12]]
        raise ValueError("AutoSpace axis contract rejected: " + "; ".join(details))

    def anchor_key(anchor):
        return anchor["path"], anchor["symbol"], anchor["kind"]

    for primitive in draft["primitives"]:
        identity = primitive["primitive_id"]
        targets = primitive["legal_mutation_targets"]
        anchors = {anchor_key(a) for a in primitive["baseline_anchors"]}
        reach = primitive["runtime_reachability"]
        effect = primitive["effect_path"]
        if (reach["mutation_target"] not in targets or effect["mutation_target"] not in targets):
            raise ValueError(f"{identity}: disconnected mutation target")
        if (anchor_key(reach["baseline_anchor"]) not in anchors
                or anchor_key(effect["baseline_anchor"]) != anchor_key(reach["baseline_anchor"])):
            raise ValueError(f"{identity}: disconnected or inconsistent effect/runtime anchor")
        if "models.py" in targets and not any(a[0] == "models.py" for a in anchors):
            raise ValueError(f"{identity}: missing executed model anchor")
        if ("main_code.py:train_update_kernel" in targets
                and ("main_code.py", "main", "function") not in anchors):
            raise ValueError(f"{identity}: missing executed train-update anchor")
