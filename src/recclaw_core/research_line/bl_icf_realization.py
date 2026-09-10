"""Public strict BL-ICF mechanism-to-realization seam.

The canonical BL-ICF compiler owns mechanism identity.  Candidate package
identity owns implementation provenance.  This module binds the two without
allowing either identity to substitute for the other.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeAlias

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    InnovationSpineError,
    MaterializedCandidate,
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
)
from recclaw_core.experiments.helix_abc_v1.realization_identity import (
    bl_icf_scientific_mechanism_program,
    bl_icf_search_space_conformance,
    open_spec_realization_identity,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    IdeaModeV1,
    OpenResearchSpecV1,
    RealizationModeV1,
)
from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.mechanism_space.contracts import CompileReportV1, CompileStatus


ImplementationProviderCallback: TypeAlias = Callable[
    [Mapping[str, Any]], Mapping[str, Any]
]

_PROFILE_FIELDS = frozenset({"profile_id", "profile_kind", "profile_digest"})
_PROVIDER_BUDGET_FIELDS = frozenset(
    {"max_provider_calls", "implementation_token_ceiling"}
)
_PARENT_BINDING_FIELDS = frozenset({"candidate_id", "program_digest"})


class CompilerBoundRealizationFailureClassV1(str, Enum):
    CONTRACT = "CONTRACT"
    COMPILE = "COMPILE"
    PROVIDER = "PROVIDER"
    IMPLEMENTATION = "IMPLEMENTATION"
    PACKAGE = "PACKAGE"


class CompilerBoundRealizationError(ValueError):
    """Typed internal failure used by the strict reusable seam."""

    def __init__(
        self,
        *,
        failure_class: CompilerBoundRealizationFailureClassV1,
        reason_code: str,
        message: str,
        compile_report: CompileReportV1 | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.reason_code = reason_code
        self.compile_report = compile_report


def _normalized_frozen_profile_ref(
    value: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _PROFILE_FIELDS:
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_FROZEN_PROFILE_REF",
            message=(
                "frozen_profile_ref must contain exactly profile_id, "
                "profile_kind, and profile_digest"
            ),
        )
    profile_id = str(value["profile_id"])
    profile_kind = str(value["profile_kind"])
    if (
        not profile_id
        or profile_id != profile_id.strip()
        or profile_kind != "OFFLINE_TOPN"
    ):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_FROZEN_PROFILE_REF",
            message="frozen_profile_ref must identify an OFFLINE_TOPN profile",
        )
    try:
        digest = validate_sha256(
            str(value["profile_digest"]),
            field_name="frozen_profile_ref.profile_digest",
        )
    except ValueError as error:
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_FROZEN_PROFILE_REF",
            message=str(error),
        ) from error
    return canonical_value(
        {
            "profile_id": profile_id,
            "profile_kind": profile_kind,
            "profile_digest": digest,
        }
    )


def _normalized_parent_binding(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, str] | tuple[dict[str, str], ...] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        normalized = tuple(
            _normalized_parent_binding(item) for item in value
        )
        if any(not isinstance(item, Mapping) for item in normalized):
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                reason_code="INVALID_PARENT_BINDING",
                message="parent bindings must contain complete parent mappings",
            )
        parents = tuple(dict(item) for item in normalized)
        if len({canonical_json_bytes(item) for item in parents}) != len(parents):
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                reason_code="INVALID_PARENT_BINDING",
                message="parent bindings must be unique",
            )
        if not parents:
            return None
        return parents[0] if len(parents) == 1 else parents
    if not isinstance(value, Mapping) or set(value) != _PARENT_BINDING_FIELDS:
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_PARENT_BINDING",
            message=(
                "parent_binding must contain exactly candidate_id and "
                "program_digest"
            ),
        )
    candidate_id = str(value["candidate_id"])
    if not candidate_id or candidate_id != candidate_id.strip():
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_PARENT_BINDING",
            message="parent_binding.candidate_id must be normalized and non-empty",
        )
    try:
        program_digest = validate_sha256(
            str(value["program_digest"]),
            field_name="parent_binding.program_digest",
        )
    except ValueError as error:
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_PARENT_BINDING",
            message=str(error),
        ) from error
    return canonical_value(
        {"candidate_id": candidate_id, "program_digest": program_digest}
    )


def parent_binding_for_mechanism_program(
    mechanism_program: Mapping[str, Any],
) -> dict[str, str] | tuple[dict[str, str], ...] | None:
    """Return strict lineage parents declared by a program, if any."""

    if not isinstance(mechanism_program, Mapping):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_MECHANISM_PROGRAM",
            message="mechanism_program must be a mapping",
        )
    parent_refs = mechanism_program.get("program_payload", {}).get("parent_refs")
    if not isinstance(parent_refs, (tuple, list)):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_PARENT_BINDING",
            message="MechanismProgram parent_refs must be a sequence",
        )
    return _normalized_parent_binding(parent_refs)


def _parent_refs(
    parent_binding: Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
    | None,
) -> list[dict[str, str]]:
    normalized = _normalized_parent_binding(parent_binding)
    if normalized is None:
        return []
    if isinstance(normalized, Mapping):
        return [dict(normalized)]
    return [dict(item) for item in normalized]


def _custom_read_roles(spec: OpenResearchSpecV1) -> tuple[str, ...]:
    text = " ".join(
        (
            spec.hypothesis,
            spec.mechanism_change,
            *spec.implementation_requirements,
            *spec.compatibility_requirements,
        )
    ).lower()
    roles = {"USER_ID", "ITEM_ID", "TRAIN_INTERACTIONS"}
    if any(token in text for token in ("graph", "neighbor", "propagat")):
        roles.add("TRAIN_USER_ITEM_GRAPH")
    if any(token in text for token in ("item-item", "item item")):
        roles.add("TRAIN_ITEM_ITEM_GRAPH")
    if any(token in text for token in ("user-user", "user user")):
        roles.add("TRAIN_USER_USER_GRAPH")
    if any(token in text for token in ("statistic", "frequency", "popularity")):
        roles.add("TRAIN_DERIVED_STATISTICS")
    if "prototype" in text:
        roles.add("TRAIN_DERIVED_PROTOTYPES")
    if any(token in text for token in ("spectral", "singular", "svd")):
        roles.add("TRAIN_DERIVED_SPECTRAL_VIEW")
    return tuple(sorted(roles))


def _relative_cost(spec: OpenResearchSpecV1) -> str:
    text = (spec.resource_hypothesis or "").lower()
    if any(token in text for token in ("very high", "multi-gpu", "distributed")):
        return "VERY_HIGH"
    if any(token in text for token in ("high", "quadratic", "dense")):
        return "HIGH"
    if any(token in text for token in ("low", "cheap", "lightweight")):
        return "LOW"
    return "MEDIUM"


def mechanism_program_from_open_spec(
    spec: OpenResearchSpecV1,
    *,
    frozen_profile_ref: Mapping[str, Any],
    parent_binding: Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Project one OpenSpec into canonical BL-ICF ``CUSTOM_MODEL`` syntax.

    ``frozen_profile_ref`` must come from the arm's sealed campaign/training/
    partition profile.  This function never reads the test fixture profile.
    """

    if not isinstance(spec, OpenResearchSpecV1):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_OPEN_SPEC",
            message="spec must be OpenResearchSpecV1",
        )
    profile_ref = _normalized_frozen_profile_ref(frozen_profile_ref)
    parent = _normalized_parent_binding(parent_binding)
    anchor = next(
        item for item in executable_mechanisms() if item.mechanism_id == "BPR_MF"
    )
    # ``executable_mechanisms`` is process-cached.  Never mutate its bootstrap
    # program while constructing a growing-between-cycles custom candidate.
    program = copy.deepcopy(deep_thaw(anchor.mechanism_program))
    payload = program["program_payload"]
    mathematical_definition = (
        "OpenSpec mechanism definition: "
        + spec.mechanism_change
        + ". Causal chain: "
        + " -> ".join(spec.causal_chain or (spec.hypothesis,))
        + ". Mechanism-off condition: "
        + (spec.mechanism_off_definition or spec.falsifier)
        + "."
    )
    execution_contract = canonical_json_bytes(
        dict(spec.execution_contract or {})
    ).decode("utf-8")
    algorithm_definition = (
        "Implement the declared candidate-local collaborative-filtering "
        "mechanism, preserve the frozen scorer/objective/training boundary, "
        "and expose the resolved executable contract. Requirements: "
        + "; ".join(spec.implementation_requirements)
        + ". Execution contract: "
        + execution_contract
        + "."
    )
    cost = _relative_cost(spec)
    payload.update(
        {
            "construction_mode": "CUSTOM_MODEL",
            "research_question": spec.research_question or spec.hypothesis,
            "core_hypothesis": spec.hypothesis,
            "changed_slots": [
                {"slot_id": "ENCODER", "change_role": "CORE"}
            ],
            "removed_slots": [],
            "parent_refs": _parent_refs(parent),
            "custom_components": [
                {
                    "custom_component_id": "open_spec_encoder",
                    "slot_id": "ENCODER",
                    "mathematical_definition": mathematical_definition,
                    "algorithm_definition": algorithm_definition,
                    "input_ports": [
                        {
                            "port": "representation",
                            "types": ["bl_icf/embedding"],
                            "minimum": 1,
                        }
                    ],
                    "output_ports": [
                        {
                            "port": "representation",
                            "type": "bl_icf/representation",
                        }
                    ],
                    "allowed_read_roles": _custom_read_roles(spec),
                    "family_boundary_justification": (
                        "The candidate uses only frozen train-side collaborative "
                        "inputs and preserves the BL-ICF representation-to-score "
                        "interface and offline Top-N protocol."
                    ),
                    "minimal_implementation": {
                        "entrypoint_role": "MODEL",
                        "file_roles": ["MODEL", "CONFIG", "TEST"],
                        "steps": tuple(spec.implementation_requirements),
                    },
                    "matched_control_rationale": spec.matched_control_requirement,
                    "ablation": (
                        spec.mechanism_off_definition or spec.falsifier
                    ),
                    "failure_modes": (
                        spec.falsifier,
                        spec.competing_explanation,
                    ),
                    "estimated_cost": cost,
                }
            ],
            "architecture_operators": [
                {
                    "operator_id": "synthesize_custom_model",
                    "targets": [],
                    "replacements": ["encoder"],
                    "parameters": {},
                    "rationale": (
                        "Materialize the OpenSpec as a candidate-local custom "
                        "model while preserving the frozen BL-ICF boundary."
                    ),
                }
            ],
            "mechanism_explanation": spec.mechanism_change,
            "matched_control": {
                "control_ref": spec.closest_parent or "BL_ICF_MATCHED_CONTROL",
                "rationale": spec.matched_control_requirement,
            },
            "ablation_plan": [
                {
                    "ablation_id": "remove_open_spec_mechanism",
                    "remove_component_ids": ["encoder"],
                    "expected_observation": (
                        spec.mechanism_off_definition or spec.falsifier
                    ),
                }
            ],
            "failure_modes": [spec.falsifier, spec.competing_explanation],
            "implementation_plan": tuple(spec.implementation_requirements),
            "expected_effects": {
                "relevance": "; ".join(spec.discriminative_predictions),
                "robustness": spec.falsifier,
                "coverage": spec.expected_evidence[0],
                "efficiency": spec.resource_hypothesis or "Measure worker cost.",
            },
            "estimated_cost": {
                "relative_training_compute": cost,
                "relative_memory": cost,
                "precompute_required": any(
                    token in (spec.resource_hypothesis or "").lower()
                    for token in ("precompute", "offline index")
                ),
            },
        }
    )
    payload["components"][1] = {
        "component_id": "encoder",
        "slot_id": "ENCODER",
        "custom_component_id": "open_spec_encoder",
        "inputs": [
            {
                "port": "representation",
                "source": {
                    "kind": "COMPONENT",
                    "component_id": "embedding",
                    "output_port": "embedding",
                },
            }
        ],
        "parameters": {},
    }
    program["profile_ref"] = profile_ref
    return canonical_value(program)


def _implementation_source_ownership(
    payload: Mapping[str, Any],
    *,
    parent_binding: Mapping[str, Any]
    | tuple[Mapping[str, Any], ...]
    | None,
) -> dict[str, Any]:
    """Compile the narrowest parent-relative implementation ownership.

    Source ownership is an implementation responsibility, not a scientific
    mechanism choice.  When one exact parent and one supported local slot are
    declared, keep the parent orchestration machine-owned for composition and
    custom-component changes.  ``ARCHITECTURE_REWRITE`` is itself a scientific
    structural decision, so it retains the high-permission full-source path,
    as do unfilled removals, unsupported, and multi-slot programs.
    """

    full_source = {"mode": "FULL_SOURCE_V1"}
    operators = tuple(payload.get("architecture_operators", ()))

    def reject_local(message: str) -> None:
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_SOURCE_OWNERSHIP",
            message=message,
        )

    changed_slots = sorted(
        {
            str(item.get("slot_id"))
            for item in payload.get("changed_slots", ())
            if isinstance(item, Mapping)
        }
    )
    local_slot = changed_slots[0] if len(changed_slots) == 1 else None
    # removed_slots belongs to the complete program: a replaced builtin may
    # remain listed after its slot has a live custom component in the parent.
    unfilled_removals = set(payload.get("removed_slots", ())) - {
        item.get("slot_id") for item in payload.get("components", ())
        if isinstance(item, Mapping)
    }
    compiler_local = bool(
        isinstance(parent_binding, Mapping)
        and payload.get("construction_mode") != "ARCHITECTURE_REWRITE"
        and (not payload.get("removed_slots") or (
            local_slot == "PRIMARY_OBJECTIVE" and not unfilled_removals
        ))
        and local_slot in {"ENCODER", "PROPAGATION_AGGREGATION", "SCORE_HEAD", "PRIMARY_OBJECTIVE"}
    )
    if not compiler_local:
        return full_source

    assert local_slot is not None

    components = tuple(payload.get("components", ()))
    component_slots = {
        str(item.get("component_id")): str(item.get("slot_id"))
        for item in components
        if isinstance(item, Mapping)
        and isinstance(item.get("component_id"), str)
        and isinstance(item.get("slot_id"), str)
    }
    local_component_ids = sorted(
        component_id
        for component_id, slot_id in component_slots.items()
        if slot_id == local_slot
    )
    if not local_component_ids:
        reject_local(f"parent-local ownership has no {local_slot} component")
    score_trains_objective = False
    if local_slot == "SCORE_HEAD":
        if len(local_component_ids) != 1:
            return full_source
        consumers = [
            item for item in components
            if isinstance(item, Mapping)
            and any(
                isinstance(port, Mapping)
                and isinstance(port.get("source"), Mapping)
                and port["source"].get("component_id") in local_component_ids
                for port in item.get("inputs", ())
            )
        ]
        # Native objectives keep their own realization. This BPR scaffold is
        # selected only by the declared score-to-objective edge, not by family.
        if any(item.get("primitive_id") != "objective.bpr" for item in consumers):
            return full_source
        score_trains_objective = bool(consumers)
    custom_components = tuple(
        item
        for item in payload.get("custom_components", ())
        if isinstance(item, Mapping) and item.get("slot_id") == local_slot
    )
    declared_custom_ids: set[str] = set()
    if local_slot == "ENCODER":
        hook_input_ports = ["representation"]
        hook_point = "PRE_PROPAGATION_REPRESENTATION_V1"
        method_name = "recclaw_encode_representation"
    elif local_slot == "PROPAGATION_AGGREGATION":
        hook_input_ports = ["layer_index", "message", "prior", "relation"]
        hook_point = "POST_PROPAGATION_MESSAGE_V1"
        method_name = "recclaw_propagation_aggregation"
    elif local_slot == "PRIMARY_OBJECTIVE":
        hook_input_ports = ["interaction", "representations"]
        hook_point = "PRIMARY_LOSS_WITH_PARENT_VIEWS_V1"
        method_name = "recclaw_primary_objective"
    else:
        hook_input_ports = ["item_representation", "pairwise", "user_representation"]
        hook_point = "BPR_AND_EVALUATION_SCORE_V1" if score_trains_objective else "EVALUATION_ONLY_SCORE_V1"
        method_name = "recclaw_score_head"

    for custom in custom_components:
        custom_id = custom.get("custom_component_id")
        output_ports = tuple(custom.get("output_ports", ()))
        if not isinstance(custom_id, str) or not custom_id:
            reject_local("custom component identity is invalid")
        if (
            len(output_ports) != 1
            or not isinstance(output_ports[0], Mapping)
            or output_ports[0].get("type") != (
                "bl_icf/objective" if local_slot == "PRIMARY_OBJECTIVE"
                else "core/user_item_relevance_score" if local_slot == "SCORE_HEAD"
                else "bl_icf/representation"
            )
        ):
            reject_local(
                "the parent-local hook requires one representation-typed output"
            )
        declared_custom_ids.add(custom_id)

    instantiated_custom_ids: set[str] = set()
    for component in components:
        if (
            not isinstance(component, Mapping)
            or component.get("slot_id") != local_slot
        ):
            continue
        custom_id = component.get("custom_component_id")
        if custom_id is None:
            continue
        if (
            custom_id not in declared_custom_ids
            or component.get("slot_id") != local_slot
        ):
            reject_local(
                "custom component instances must align to the declared local hook"
            )
        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            reject_local("custom encoder component identity is invalid")
        instantiated_custom_ids.add(str(custom_id))
    if instantiated_custom_ids != declared_custom_ids:
        reject_local("custom component definitions and instances must align")

    def reference_slot(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        if value.startswith("slot:"):
            return value.removeprefix("slot:")
        return component_slots.get(value)

    for operator in operators:
        if not isinstance(operator, Mapping):
            reject_local("parent-local operator footprint is invalid")
        footprint = tuple(operator.get("targets", ())) + tuple(
            operator.get("replacements", ())
        )
        if not footprint:
            # A valid compiled program with one changed slot has only one
            # possible write scope for a targetless operator.  Scientific
            # component ports and the machine hook ABI are intentionally
            # different layers, so source ownership must not infer more from
            # their names.
            continue
        footprint_slots = {reference_slot(item) for item in footprint}
        if (
            operator.get("operator_id") == "synthesize_custom_model"
            and footprint_slots != {local_slot}
        ) or (
            local_slot in footprint_slots
            and footprint_slots != {local_slot}
        ):
            reject_local(
                f"every operator reference must resolve to {local_slot}"
            )
    return {
        "mode": "PARENT_LOCAL_SLOT_HOOK_V1",
        "model_hook": {
            "component_ids": local_component_ids,
            "hook_point": hook_point,
            "input_ports": hook_input_ports,
            "method_name": method_name,
            "output_port": (
                "objective" if local_slot == "PRIMARY_OBJECTIVE"
                else "score" if local_slot == "SCORE_HEAD" else "representation"
            ),
            "slot_id": local_slot,
        },
    }


@dataclass(frozen=True, slots=True)
class CompilerBoundMechanismV1:
    mechanism_program: Mapping[str, Any]
    compiler_candidate_id: str
    compile_report_digest: str
    mechanism_program_digest: str
    mechanism_semantics_digest: str
    semantic_identity_ref: str
    space_identity: Mapping[str, str]
    search_space_id: str
    search_space_digest: str
    ordered_primitive_ids_count: int
    ordered_primitive_ids_digest_algorithm: str
    ordered_primitive_ids_digest: str
    fixed_fallback: bool
    profile_ref: Mapping[str, str]
    parent_binding: Mapping[str, str] | tuple[Mapping[str, str], ...] | None
    source_ownership: Mapping[str, Any]
    required_capabilities: tuple[str, ...]
    resolved_ir: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)

    def implementation_payload(self) -> dict[str, Any]:
        """Return the complete compiler output consumed by the Implementer."""

        payload = self.mechanism_program["program_payload"]
        components = tuple(payload["components"])
        custom_components = tuple(payload.get("custom_components", ()))
        operators = tuple(payload.get("architecture_operators", ()))
        implementation_binding = {
            "architecture_operator_ids": sorted(
                {str(item["operator_id"]) for item in operators}
            ),
            "component_specs": {
                str(item["component_id"]): canonical_value(dict(item))
                for item in components
            },
            "component_ids": sorted(
                {str(item["component_id"]) for item in components}
            ),
            "custom_component_ids": sorted(
                {
                    str(item["custom_component_id"])
                    for item in custom_components
                }
            ),
            "primitive_ids": sorted(
                {
                    str(item["primitive_id"])
                    for item in components
                    if item.get("primitive_id") is not None
                }
            ),
            "source_ownership": self.source_ownership,
        }
        return canonical_value(
            {
                "schema": "recclaw.compiled-mechanism-implementation.v1",
                "compiler_candidate_id": self.compiler_candidate_id,
                "mechanism_program": self.mechanism_program,
                "mechanism_program_digest": self.mechanism_program_digest,
                "mechanism_semantics_digest": self.mechanism_semantics_digest,
                "required_capabilities": self.required_capabilities,
                "resolved_ir": self.resolved_ir,
                "space_identity": self.space_identity,
                "implementation_binding": implementation_binding,
            }
        )


def compile_bl_icf_candidate_source(
    *,
    open_spec: OpenResearchSpecV1 | None,
    frozen_profile_ref: Mapping[str, Any],
    mechanism_program: Mapping[str, Any] | None = None,
    parent_binding: Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
    | None = None,
) -> CompilerBoundMechanismV1:
    """Compile caller-supplied strict syntax or an OpenSpec projection."""

    profile_ref = _normalized_frozen_profile_ref(frozen_profile_ref)
    parent = _normalized_parent_binding(parent_binding)
    program = (
        mechanism_program_from_open_spec(
            open_spec,
            frozen_profile_ref=profile_ref,
            parent_binding=parent,
        )
        if mechanism_program is None and open_spec is not None
        else canonical_value(dict(mechanism_program))
        if mechanism_program is not None
        else None
    )
    if program is None:
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="CANDIDATE_SOURCE_MISSING",
            message="provide a strict MechanismProgram or an OpenSpec",
        )
    if canonical_value(program.get("profile_ref")) != profile_ref:
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="FROZEN_PROFILE_IDENTITY_DRIFT",
            message="MechanismProgram profile_ref differs from frozen_profile_ref",
        )
    expected_parent_refs = _parent_refs(parent)
    if canonical_value(program.get("program_payload", {}).get("parent_refs")) != (
        canonical_value(expected_parent_refs)
    ):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="PARENT_BINDING_DRIFT",
            message="MechanismProgram parent_refs differs from parent_binding",
        )
    report = compile_program(deep_thaw(program))
    if (
        report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION
        or report.candidate_id is None
        or report.mechanism_program_digest is None
        or report.mechanism_semantics_digest is None
        or report.space_identity is None
        or report.resolved_ir is None
    ):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.COMPILE,
            reason_code="BL_ICF_COMPILE_REJECTED",
            message="MechanismProgram is not strict VALID_NEEDS_IMPLEMENTATION",
            compile_report=report,
        )
    scientific_program = bl_icf_scientific_mechanism_program(program)
    scientific_report = (
        report
        if scientific_program == program
        else compile_program(deep_thaw(scientific_program))
    )
    if (
        scientific_report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION
        or scientific_report.mechanism_semantics_digest is None
    ):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.COMPILE,
            reason_code="BL_ICF_SCIENTIFIC_COMPILE_REJECTED",
            message=(
                "MechanismProgram scientific projection is not strict "
                "VALID_NEEDS_IMPLEMENTATION"
            ),
            compile_report=scientific_report,
        )
    conformance = bl_icf_search_space_conformance(
        space_identity=report.space_identity.to_dict(),
        profile_ref=profile_ref,
        fixed_fallback=False,
    )
    source_ownership = _implementation_source_ownership(
        program["program_payload"],
        parent_binding=parent,
    )
    resolved_ir = canonical_value(deep_thaw(report.resolved_ir))
    resolved_ir["mechanism_semantics_digest"] = (
        scientific_report.mechanism_semantics_digest
    )
    return CompilerBoundMechanismV1(
        mechanism_program=program,
        compiler_candidate_id=report.candidate_id,
        compile_report_digest=sha256_digest(report.to_dict()),
        mechanism_program_digest=report.mechanism_program_digest,
        mechanism_semantics_digest=(
            scientific_report.mechanism_semantics_digest
        ),
        semantic_identity_ref=f"bl-icf-mechanism:{report.candidate_id}",
        space_identity=report.space_identity.to_dict(),
        search_space_id=conformance["search_space_id"],
        search_space_digest=conformance["search_space_digest"],
        ordered_primitive_ids_count=conformance[
            "ordered_primitive_ids_count"
        ],
        ordered_primitive_ids_digest_algorithm=(
            conformance["ordered_primitive_ids_digest_algorithm"]
        ),
        ordered_primitive_ids_digest=conformance[
            "ordered_primitive_ids_digest"
        ],
        fixed_fallback=conformance["fixed_fallback"],
        profile_ref=conformance["profile_ref"],
        parent_binding=parent,
        source_ownership=canonical_value(source_ownership),
        required_capabilities=tuple(report.required_capabilities),
        resolved_ir=canonical_value(resolved_ir),
    )


@dataclass(frozen=True, slots=True)
class MechanismImplementationCompanionV1:
    """Minimum arm-owned metadata needed to implement a strict program.

    Search remains the full BL-ICF ``MechanismProgram``.  This companion only
    supplies identities and runtime wiring that are not part of that search
    syntax; it does not select, simplify, or rewrite the mechanism.
    """

    protocol_ref: str
    protocol_digest: str
    context_ref: str
    context_digest: str
    current_profile_ref: str
    current_profile_digest: str
    producer_role: str
    compatibility_requirements: tuple[str, ...]
    execution_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "protocol_ref",
            "context_ref",
            "current_profile_ref",
            "producer_role",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise CompilerBoundRealizationError(
                    failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                    reason_code="INVALID_IMPLEMENTATION_COMPANION",
                    message=f"{field_name} must be normalized and non-empty",
                )
        for field_name in (
            "protocol_digest",
            "context_digest",
            "current_profile_digest",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    validate_sha256(
                        getattr(self, field_name),
                        field_name=field_name,
                    ),
                )
            except ValueError as error:
                raise CompilerBoundRealizationError(
                    failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                    reason_code="INVALID_IMPLEMENTATION_COMPANION",
                    message=str(error),
                ) from error
        requirements = tuple(
            sorted({str(item) for item in self.compatibility_requirements})
        )
        if not requirements or any(
            not item or item != item.strip() for item in requirements
        ):
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                reason_code="INVALID_IMPLEMENTATION_COMPANION",
                message="compatibility_requirements must be non-empty normalized strings",
            )
        object.__setattr__(self, "compatibility_requirements", requirements)
        if not isinstance(self.execution_contract, Mapping):
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                reason_code="INVALID_IMPLEMENTATION_COMPANION",
                message="execution_contract must be a mapping",
            )
        object.__setattr__(
            self,
            "execution_contract",
            canonical_value(dict(self.execution_contract)),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ProviderMechanismProgramProposalV1:
    """Provider-authored research syntax; machine ABI is constructed downstream."""

    producer_role: str
    mechanism_program: Mapping[str, Any]
    implementation_research: Mapping[str, Any]
    resolution_facts: Mapping[str, Any]
    parent_binding: Mapping[str, Any] | tuple[Mapping[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.producer_role, str)
            or not self.producer_role
            or self.producer_role != self.producer_role.strip()
        ):
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                reason_code="INVALID_PROVIDER_PROGRAM_PROPOSAL",
                message="producer_role must be normalized and non-empty",
            )
        for field_name in (
            "mechanism_program",
            "implementation_research",
            "resolution_facts",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise CompilerBoundRealizationError(
                    failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                    reason_code="INVALID_PROVIDER_PROGRAM_PROPOSAL",
                    message=f"{field_name} must be a mapping",
                )
            object.__setattr__(self, field_name, canonical_value(dict(value)))
        normalized_parent = _normalized_parent_binding(self.parent_binding)
        declared_parent = parent_binding_for_mechanism_program(
            self.mechanism_program
        )
        if declared_parent != normalized_parent:
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                reason_code="PARENT_BINDING_DRIFT",
                message=(
                    "Provider proposal parent_binding differs from "
                    "MechanismProgram parent_refs"
                ),
            )
        object.__setattr__(self, "parent_binding", normalized_parent)

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def implementation_spec_for_compiled_program(
    compiler: CompilerBoundMechanismV1,
    *,
    companion: MechanismImplementationCompanionV1,
) -> OpenResearchSpecV1:
    """Build an origin-blind implementation spec from compiler-bound IR.

    This is the minimal adapter for Provider-direct strict-program search.
    Mechanism content comes exclusively from the submitted program and its
    compiler-resolved identity; the companion contributes only external frozen
    identities, compatibility, and executable wiring.
    """

    if not isinstance(compiler, CompilerBoundMechanismV1):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_COMPILER_BINDING",
            message="compiler must be CompilerBoundMechanismV1",
        )
    if not isinstance(companion, MechanismImplementationCompanionV1):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_IMPLEMENTATION_COMPANION",
            message="companion must be MechanismImplementationCompanionV1",
        )
    payload = compiler.mechanism_program["program_payload"]
    changed_slots = tuple(
        str(item["slot_id"]) for item in payload["changed_slots"]
    )
    mechanism_change = str(payload["mechanism_explanation"])
    hypothesis = str(payload["core_hypothesis"])
    failure_modes = tuple(str(item) for item in payload["failure_modes"])
    implementation_plan = tuple(
        str(item) for item in payload["implementation_plan"]
    )
    matched_control = payload["matched_control"]
    ablation = payload["ablation_plan"][0]
    effects = tuple(
        dict.fromkeys(str(value) for value in payload["expected_effects"].values())
    )
    cost = payload["estimated_cost"]
    parent_id: str | None = None
    if isinstance(compiler.parent_binding, Mapping):
        parent_id = str(compiler.parent_binding["candidate_id"])
    elif compiler.parent_binding is not None:
        parent_id = ",".join(
            str(item["candidate_id"]) for item in compiler.parent_binding
        )
    return OpenResearchSpecV1(
        hypothesis=hypothesis,
        mechanism_change=mechanism_change,
        competing_explanation=failure_modes[0],
        matched_control_requirement=str(matched_control["rationale"]),
        implementation_requirements=implementation_plan,
        expected_evidence=effects,
        falsifier=str(ablation["expected_observation"]),
        compatibility_requirements=companion.compatibility_requirements,
        protocol_ref=companion.protocol_ref,
        protocol_digest=companion.protocol_digest,
        context_ref=companion.context_ref,
        context_digest=companion.context_digest,
        current_profile_ref=companion.current_profile_ref,
        current_profile_digest=companion.current_profile_digest,
        producer_role=companion.producer_role,
        high_change_justification=(
            f"Compiler-bound {payload['construction_mode']} changes slots: "
            + ", ".join(changed_slots)
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
        idea_mode=IdeaModeV1.FRONTIER_HYPOTHESIS,
        research_question=str(payload["research_question"]),
        observed_failure_mode="NOT_OBSERVED",
        closest_parent=parent_id,
        minimal_testable_wedge=mechanism_change,
        causal_chain=(hypothesis, mechanism_change),
        discriminative_predictions=effects,
        mechanism_off_definition=str(ablation["expected_observation"]),
        resource_hypothesis=(
            "training_compute="
            + str(cost["relative_training_compute"])
            + "; memory="
            + str(cost["relative_memory"])
            + "; precompute="
            + str(bool(cost["precompute_required"])).lower()
        ),
        realization_mode=(
            RealizationModeV1.PARENT_PRESERVING
            if compiler.parent_binding is not None
            else RealizationModeV1.NON_NESTED
        ),
        execution_contract=companion.execution_contract,
    )


def _validate_explicit_open_spec_companion(
    compiler: CompilerBoundMechanismV1,
    open_spec: OpenResearchSpecV1,
) -> None:
    payload = compiler.mechanism_program["program_payload"]
    if (
        open_spec.hypothesis != payload["core_hypothesis"]
        or open_spec.mechanism_change != payload["mechanism_explanation"]
        or open_spec.matched_control_requirement
        != payload["matched_control"]["rationale"]
        or set(open_spec.implementation_requirements)
        != set(payload["implementation_plan"])
    ):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="OPEN_SPEC_MECHANISM_BINDING_DRIFT",
            message="explicit OpenSpec companion differs from MechanismProgram semantics",
        )


@dataclass(frozen=True, slots=True)
class CompilerBoundRealizedCandidateV1:
    compiler: CompilerBoundMechanismV1
    open_spec_ref: str
    open_spec_digest: str
    executable_entrypoint: str
    model: str
    base_model_config: str
    config: Mapping[str, Any]
    candidate_package_ref: str
    candidate_package_digest: str
    candidate_root_ref: str
    candidate_root_digest: str
    source_tree_digest: str
    implementation_receipt_ref: str
    implementation_receipt_digest: str
    realization_identity_ref: str
    realization_semantics_digest: str
    binding: Mapping[str, Any]
    binding_ref: str
    binding_digest: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def bind_materialized_bl_icf_candidate(
    *,
    compiler: CompilerBoundMechanismV1,
    open_spec: OpenResearchSpecV1,
    materialized: MaterializedCandidate,
) -> CompilerBoundRealizedCandidateV1:
    """Bind an existing package to its independently compiled mechanism."""

    if not isinstance(compiler, CompilerBoundMechanismV1):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_COMPILER_BINDING",
            message="compiler must be CompilerBoundMechanismV1",
        )
    if not isinstance(open_spec, OpenResearchSpecV1):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_OPEN_SPEC",
            message="open_spec must be OpenResearchSpecV1",
        )
    if not isinstance(materialized, MaterializedCandidate):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="INVALID_MATERIALIZED_CANDIDATE",
            message="materialized must be MaterializedCandidate",
        )
    package = materialized.package
    if (
        package.research_spec_ref != open_spec.spec_id
        or package.research_spec_digest != open_spec.digest
        or open_spec.execution_contract is None
    ):
        raise CompilerBoundRealizationError(
            failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
            reason_code="SPEC_PACKAGE_BINDING_DRIFT",
            message="CandidatePackage is not bound to the compiled OpenSpec",
        )
    realization_ref, realization_digest = open_spec_realization_identity(
        open_spec,
        candidate_package_ref=package.package_id,
        candidate_package_digest=package.digest,
        candidate_root_ref=package.candidate_root_ref,
        candidate_root_digest=package.candidate_root_digest,
        source_tree_digest=package.source_tree_digest,
        executable_entrypoint=package.executable_entrypoint,
        execution_contract=open_spec.execution_contract,
    )
    contract = canonical_value(dict(open_spec.execution_contract))
    binding = canonical_value(
        {
            "schema": "recclaw.compiler-bound-realization-binding.v1",
            "compiler_candidate_id": compiler.compiler_candidate_id,
            "compile_report_digest": compiler.compile_report_digest,
            "mechanism_program_digest": compiler.mechanism_program_digest,
            "mechanism_semantics_digest": compiler.mechanism_semantics_digest,
            "semantic_identity_ref": compiler.semantic_identity_ref,
            "space_identity": compiler.space_identity,
            "search_space_id": compiler.search_space_id,
            "search_space_digest": compiler.search_space_digest,
            "ordered_primitive_ids_count": (
                compiler.ordered_primitive_ids_count
            ),
            "ordered_primitive_ids_digest_algorithm": (
                compiler.ordered_primitive_ids_digest_algorithm
            ),
            "ordered_primitive_ids_digest": (
                compiler.ordered_primitive_ids_digest
            ),
            "fixed_fallback": compiler.fixed_fallback,
            "profile_ref": compiler.profile_ref,
            "parent_binding": compiler.parent_binding,
            "open_spec_ref": open_spec.spec_id,
            "open_spec_digest": open_spec.digest,
            "candidate_package_ref": package.package_id,
            "candidate_package_digest": package.digest,
            "candidate_root_ref": package.candidate_root_ref,
            "candidate_root_digest": package.candidate_root_digest,
            "source_tree_digest": package.source_tree_digest,
            "implementation_receipt_ref": package.implementation_receipt_ref,
            "implementation_receipt_digest": package.implementation_receipt_digest,
            "executable_entrypoint": package.executable_entrypoint,
            "model": contract["model"],
            "base_model_config": contract["base_model_config"],
            "config": contract["config"],
            "realization_identity_ref": realization_ref,
            "realization_semantics_digest": realization_digest,
        }
    )
    binding_digest = sha256_digest(binding)
    return CompilerBoundRealizedCandidateV1(
        compiler=compiler,
        open_spec_ref=open_spec.spec_id,
        open_spec_digest=open_spec.digest,
        executable_entrypoint=package.executable_entrypoint,
        model=str(contract["model"]),
        base_model_config=str(contract["base_model_config"]),
        config=canonical_value(dict(contract["config"])),
        candidate_package_ref=package.package_id,
        candidate_package_digest=package.digest,
        candidate_root_ref=package.candidate_root_ref,
        candidate_root_digest=package.candidate_root_digest,
        source_tree_digest=package.source_tree_digest,
        implementation_receipt_ref=package.implementation_receipt_ref,
        implementation_receipt_digest=package.implementation_receipt_digest,
        realization_identity_ref=realization_ref,
        realization_semantics_digest=realization_digest,
        binding=binding,
        binding_ref=content_id("recclaw-compiler-bound-realization-v1", binding),
        binding_digest=binding_digest,
    )


@dataclass(frozen=True, slots=True)
class CompilerBoundRealizationFailureV1:
    failure_class: CompilerBoundRealizationFailureClassV1
    reason_code: str
    message: str
    compile_report: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class CompilerBoundRealizationResultV1:
    realized_candidate: CompilerBoundRealizedCandidateV1 | None
    failure: CompilerBoundRealizationFailureV1 | None

    def __post_init__(self) -> None:
        if (self.realized_candidate is None) == (self.failure is None):
            raise ValueError("result requires exactly one candidate or failure")

    @property
    def succeeded(self) -> bool:
        return self.realized_candidate is not None

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def _failure_result(
    error: CompilerBoundRealizationError,
) -> CompilerBoundRealizationResultV1:
    return CompilerBoundRealizationResultV1(
        realized_candidate=None,
        failure=CompilerBoundRealizationFailureV1(
            failure_class=error.failure_class,
            reason_code=error.reason_code,
            message=str(error),
            compile_report=(
                error.compile_report.to_dict()
                if error.compile_report is not None
                else None
            ),
        ),
    )


def realize_bl_icf_candidate(
    *,
    open_spec: OpenResearchSpecV1 | None,
    implementation_companion: MechanismImplementationCompanionV1 | None,
    frozen_profile_ref: Mapping[str, Any],
    provider_callback: ImplementationProviderCallback,
    provider_budget: Mapping[str, Any],
    candidate_root: Path,
    candidate_root_ref: str,
    implementer_policy: SharedImplementerPolicy,
    mechanism_program: Mapping[str, Any] | None = None,
    parent_binding: Mapping[str, Any]
    | Sequence[Mapping[str, Any]]
    | None = None,
    exact_parent_bundle: Mapping[str, Any] | None = None,
) -> CompilerBoundRealizationResultV1:
    """One-call public seam for strict compiler-bound realization.

    The provider callback receives the existing blind implementer request.
    Provider/model retries remain an outer orchestration concern so every
    physical call and cost can be recorded by the owning arm.
    """

    try:
        compiler = compile_bl_icf_candidate_source(
            open_spec=open_spec,
            frozen_profile_ref=frozen_profile_ref,
            mechanism_program=mechanism_program,
            parent_binding=parent_binding,
        )
        if open_spec is None:
            if implementation_companion is None:
                raise CompilerBoundRealizationError(
                    failure_class=(
                        CompilerBoundRealizationFailureClassV1.CONTRACT
                    ),
                    reason_code="IMPLEMENTATION_COMPANION_MISSING",
                    message=(
                        "a direct MechanismProgram requires an implementation "
                        "companion or explicit OpenSpec companion"
                    ),
                )
            implementation_spec = implementation_spec_for_compiled_program(
                compiler,
                companion=implementation_companion,
            )
        else:
            if implementation_companion is not None:
                raise CompilerBoundRealizationError(
                    failure_class=(
                        CompilerBoundRealizationFailureClassV1.CONTRACT
                    ),
                    reason_code="AMBIGUOUS_IMPLEMENTATION_COMPANION",
                    message="provide OpenSpec or minimal companion, not both",
                )
            implementation_spec = open_spec
            _validate_explicit_open_spec_companion(compiler, implementation_spec)
        if (
            not isinstance(provider_budget, Mapping)
            or set(provider_budget) != _PROVIDER_BUDGET_FIELDS
            or provider_budget.get("max_provider_calls") != 1
            or provider_budget.get("implementation_token_ceiling")
            != implementer_policy.implementation_token_ceiling
        ):
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.CONTRACT,
                reason_code="PROVIDER_BUDGET_POLICY_DRIFT",
                message=(
                    "provider_budget must bind one physical call and the exact "
                    "SharedImplementerPolicy token ceiling"
                ),
            )
        request = build_shared_implementer_request(
            implementation_spec,
            policy=implementer_policy,
            compiled_mechanism=compiler.implementation_payload(),
            exact_parent_bundle=exact_parent_bundle,
        )
        try:
            response = provider_callback(request)
        except Exception as error:
            raise CompilerBoundRealizationError(
                failure_class=CompilerBoundRealizationFailureClassV1.PROVIDER,
                reason_code=type(error).__name__,
                message=str(error),
            ) from error
        materialized = materialize_candidate_package(
            implementation_spec,
            policy=implementer_policy,
            implementation_response=response,
            candidate_root=Path(candidate_root),
            candidate_root_ref=candidate_root_ref,
            compiled_mechanism=compiler.implementation_payload(),
            exact_parent_bundle=exact_parent_bundle,
        )
        realized = bind_materialized_bl_icf_candidate(
            compiler=compiler,
            open_spec=implementation_spec,
            materialized=materialized,
        )
        return CompilerBoundRealizationResultV1(
            realized_candidate=realized,
            failure=None,
        )
    except CompilerBoundRealizationError as error:
        return _failure_result(error)
    except InnovationSpineError as error:
        failure_class = (
            CompilerBoundRealizationFailureClassV1.IMPLEMENTATION
            if error.failure_class == "IMPLEMENTATION"
            else CompilerBoundRealizationFailureClassV1.PACKAGE
        )
        return CompilerBoundRealizationResultV1(
            realized_candidate=None,
            failure=CompilerBoundRealizationFailureV1(
                failure_class=failure_class,
                reason_code=error.reason_code,
                message=str(error),
            ),
        )


__all__ = [
    "CompilerBoundMechanismV1",
    "CompilerBoundRealizationError",
    "CompilerBoundRealizationFailureClassV1",
    "CompilerBoundRealizationFailureV1",
    "CompilerBoundRealizationResultV1",
    "CompilerBoundRealizedCandidateV1",
    "ImplementationProviderCallback",
    "MechanismImplementationCompanionV1",
    "ProviderMechanismProgramProposalV1",
    "bind_materialized_bl_icf_candidate",
    "compile_bl_icf_candidate_source",
    "implementation_spec_for_compiled_program",
    "mechanism_program_from_open_spec",
    "parent_binding_for_mechanism_program",
    "realize_bl_icf_candidate",
]
