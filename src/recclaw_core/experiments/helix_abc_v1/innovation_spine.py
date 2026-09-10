"""Deterministic OpenSpec-to-CandidatePackage boundary for Research Line vNext."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .canonical import (
    canonical_value,
    content_id,
    sha256_digest,
    validate_relative_artifact_path,
    validate_sha256,
)
from .epoch_sampler_scaffold import required_train_data_fit_roles
from .innovation_recbole_adapter import (
    candidate_tree_identity,
    snapshot_candidate_tree,
)
from .vnext_contracts import CandidatePackageV1, OpenResearchSpecV1
from .model_configuration import (
    MODEL_CONFIG_MAPPING_CONTRACT,
    MODEL_CONFIG_MAPPING_REQUIREMENT,
    bind_model_configuration_source,
)


_BLIND_SPEC_FIELDS = (
    "causal_chain",
    "closest_parent",
    "compatibility_requirements",
    "competing_explanation",
    "current_profile_digest",
    "current_profile_expressibility_claim",
    "current_profile_ref",
    "expected_evidence",
    "falsifier",
    "high_change_justification",
    "hypothesis",
    "idea_mode",
    "implementation_requirements",
    "matched_control_requirement",
    "mechanism_off_definition",
    "mechanism_change",
    "minimal_testable_wedge",
    "protocol_digest",
    "protocol_ref",
    "realization_mode",
    "research_question",
    "resource_hypothesis",
    "discriminative_predictions",
)
_FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "arm",
        "arm_code",
        "candidate_origin",
        "context_digest",
        "context_ref",
        "controller",
        "controller_id",
        "metric_observation",
        "metric_values",
        "origin",
        "outcome",
        "outcomes",
        "producer",
        "producer_id",
        "producer_role",
        "result",
        "results",
        "research_spec_digest",
        "research_spec_ref",
        "spec_id",
    }
)
_CANDIDATE_PREFIXES = ("recclaw_ext/", "tests/")
FULL_SOURCE_RESPONSE_MODE = "STRICT_JSON_FULL_FILE_CONTENTS"
PARENT_METHOD_PATCH_RESPONSE_MODE = "PARENT_METHOD_PATCH_V1"
PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE = "PARENT_LOCAL_SLOT_PATCH_V1"
PROFILE_MODEL_HOOKS_RESPONSE_MODE = "PROFILE_MODEL_HOOKS_V1"
SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE = "SCAFFOLDED_FULL_SOURCE_V1"
_COMPILED_MECHANISM_FIELDS = frozenset(
    {
        "compiler_candidate_id",
        "implementation_binding",
        "mechanism_program",
        "mechanism_program_digest",
        "mechanism_semantics_digest",
        "required_capabilities",
        "resolved_ir",
        "schema",
        "space_identity",
    }
)
_IMPLEMENTATION_BINDING_SEQUENCE_FIELDS = frozenset(
    {
        "architecture_operator_ids",
        "component_ids",
        "custom_component_ids",
        "primitive_ids",
    }
)
_IMPLEMENTATION_BINDING_MAPPING_FIELDS = frozenset({"component_specs"})
_IMPLEMENTATION_BINDING_OPTIONAL_MAPPING_FIELDS = frozenset(
    {"source_ownership"}
)
_IMPLEMENTATION_BINDING_FIELDS = (
    _IMPLEMENTATION_BINDING_SEQUENCE_FIELDS
    | _IMPLEMENTATION_BINDING_MAPPING_FIELDS
)
_SOURCE_BINDING_CONSTANTS = {
    "RECCLAW_COMPILER_CANDIDATE_ID": "compiler_candidate_id",
    "RECCLAW_MECHANISM_PROGRAM_DIGEST": "mechanism_program_digest",
    "RECCLAW_MECHANISM_SEMANTICS_DIGEST": "mechanism_semantics_digest",
}
_SOURCE_BINDING_SEQUENCE_CONSTANTS = {
    "RECCLAW_IMPLEMENTED_ARCHITECTURE_OPERATOR_IDS": "architecture_operator_ids",
    "RECCLAW_IMPLEMENTED_COMPONENT_IDS": "component_ids",
    "RECCLAW_IMPLEMENTED_CUSTOM_COMPONENT_IDS": "custom_component_ids",
    "RECCLAW_IMPLEMENTED_PRIMITIVE_IDS": "primitive_ids",
}
_SOURCE_BINDING_MAPPING_CONSTANTS = {
    "RECCLAW_IMPLEMENTED_COMPONENT_SPECS": "component_specs",
}
_MECHANISM_BEHAVIOR_ACCEPTANCE = {
    "propagation.user_item_specific_depth": (
        "FreshCandidateModel.computer must apply distinct user and item propagation "
        "depths or layer-aggregation weights; one shared depth followed by one "
        "shared layer mean is not user-item-specific depth."
    ),
    "objective.partial_auc_surrogate": (
        "The training objective must select or score-dependently weight a top/partial "
        "negative tail (for example topk, quantile, or sorted hard negatives); plain "
        "single-negative softplus/logsigmoid pairwise mean is ordinary BPR."
    ),
    "ssl.objective.info_nce": (
        "Each mini-batch InfoNCE call must form contrastive rows and negatives from "
        "the current batch entity IDs or an explicitly bounded sampled subset; do "
        "not build an all-user/all-item square logits matrix on every mini-batch."
    ),
}


def _declared_propagation_depths(
    component_specs: Mapping[str, Any] | None,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(component_specs, Mapping):
        return ()
    supported = {"propagation.symmetric_normalization"}
    declared: list[tuple[str, int]] = []
    for component in component_specs.values():
        if not isinstance(component, Mapping):
            continue
        primitive_id = component.get("primitive_id")
        parameters = component.get("parameters")
        depth = parameters.get("depth") if isinstance(parameters, Mapping) else None
        if (
            primitive_id in supported
            and isinstance(depth, int)
            and not isinstance(depth, bool)
            and depth > 0
        ):
            declared.append((str(primitive_id), depth))
    return tuple(sorted(declared))


def _propagation_depth_acceptance(primitive_id: str, depth: int) -> str:
    return (
        f"{primitive_id} declares depth={depth}; each propagation evaluation must "
        f"execute exactly {depth} adjacency propagation hop(s) and may aggregate "
        f"only the initial representation plus hops 1 through {depth}."
    )


def _proven_returned_sparse_hops(function: ast.AST) -> int | None:
    """Return a hop depth only for a simple, explicit returned sparse chain."""

    def names(node: ast.AST | None) -> set[str]:
        if node is None:
            return set()
        return {
            item.id if isinstance(item, ast.Name) else item.attr
            for item in ast.walk(node)
            if isinstance(item, (ast.Name, ast.Attribute))
        }

    def sparse_mm_calls(node: ast.AST) -> tuple[ast.Call, ...]:
        return tuple(
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "mm"
            and "sparse" in names(item.func.value)
        )

    top_level_assignments = tuple(
        node
        for node in getattr(function, "body", ())
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    )
    top_level_sparse_count = sum(
        len(sparse_mm_calls(node.value)) for node in top_level_assignments
    )
    if top_level_sparse_count == 0:
        return None
    if top_level_sparse_count != len(sparse_mm_calls(function)):
        # Loops, branches, and nested helper definitions need path-sensitive
        # reasoning.  Leave equivalent implementations unclassified.
        return None

    hop_by_name: dict[str, int] = {}
    for node in top_level_assignments:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = set().union(*(names(target) for target in targets))
        calls = sparse_mm_calls(node.value)
        if calls:
            call_depths = []
            for call in calls:
                if len(call.args) < 2:
                    return None
                source_names = names(call.args[1])
                source_depths = [
                    hop_by_name[name]
                    for name in source_names
                    if name in hop_by_name
                ]
                call_depths.append(1 + max(source_depths, default=0))
            depth = max(call_depths)
        else:
            dependency_depths = [
                hop_by_name[name]
                for name in names(node.value)
                if name in hop_by_name
            ]
            if not dependency_depths:
                continue
            depth = max(dependency_depths)
        for target_name in target_names:
            hop_by_name[target_name] = depth

    returned_depths = [
        hop_by_name[name]
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        for name in names(node.value)
        if name in hop_by_name
    ]
    return max(returned_depths) if returned_depths else None


class InnovationSpineError(RuntimeError):
    """Typed implementation/package failure at the blind materialization boundary."""

    def __init__(
        self,
        *,
        failure_class: str,
        reason_code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if failure_class not in {"IMPLEMENTATION", "PACKAGE"}:
            raise ValueError("InnovationSpineError failure_class is invalid")
        super().__init__(message)
        self.failure_class = failure_class
        self.reason_code = reason_code
        self.details = canonical_value(details) if details is not None else None


@dataclass(frozen=True, slots=True)
class SharedImplementerPolicy:
    """Common non-outcome service inputs shared by every implementation source."""

    allowed_files: tuple[str, ...]
    dependency_identity_ref: str
    dependency_identity_digest: str
    runtime_identity_ref: str
    runtime_identity_digest: str
    prompt_digest: str
    tool_policy_digest: str
    implementation_token_ceiling: int
    execution_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            paths = tuple(
                validate_relative_artifact_path(str(path))
                for path in self.allowed_files
            )
        except Exception as error:
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_WRITE_ALLOWLIST",
                message="allowed_files contains an invalid relative path",
            ) from error
        if (
            not paths
            or len(set(paths)) != len(paths)
            or any(not path.startswith(_CANDIDATE_PREFIXES) for path in paths)
            or not any(path.startswith("recclaw_ext/") for path in paths)
        ):
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_WRITE_ALLOWLIST",
                message=(
                    "allowed_files must be unique candidate-local paths and "
                    "include an entrypoint source under recclaw_ext/"
                ),
            )
        object.__setattr__(self, "allowed_files", tuple(sorted(paths)))
        for field_name in (
            "dependency_identity_ref",
            "runtime_identity_ref",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise InnovationSpineError(
                    failure_class="PACKAGE",
                    reason_code="INVALID_IDENTITY_REFERENCE",
                    message=f"{field_name} must be a normalized non-empty string",
                )
        for field_name in (
            "dependency_identity_digest",
            "runtime_identity_digest",
            "prompt_digest",
            "tool_policy_digest",
        ):
            try:
                validate_sha256(getattr(self, field_name), field_name=field_name)
            except Exception as error:
                raise InnovationSpineError(
                    failure_class="PACKAGE",
                    reason_code="INVALID_IDENTITY_DIGEST",
                    message=f"{field_name} must be a SHA-256 digest",
                ) from error
        if (
            not isinstance(self.implementation_token_ceiling, int)
            or isinstance(self.implementation_token_ceiling, bool)
            or self.implementation_token_ceiling < 1
        ):
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_TOKEN_CEILING",
                message="implementation_token_ceiling must be positive",
            )
        if self.execution_contract is not None and not isinstance(
            self.execution_contract, Mapping
        ):
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_EXECUTION_CONTRACT",
                message="execution_contract must be a mapping when supplied",
            )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MaterializedCandidate:
    """Successful local materialization and its RC0 CandidatePackageV1."""

    package: CandidatePackageV1
    blind_projection: Mapping[str, Any]
    shared_request: Mapping[str, Any]
    implementation_receipt: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "blind_projection": self.blind_projection,
                "implementation_receipt": self.implementation_receipt,
                "package": self.package.canonical_dict(),
                "shared_request": self.shared_request,
            }
        )


def _implementation_failure(
    reason_code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> InnovationSpineError:
    return InnovationSpineError(
        failure_class="IMPLEMENTATION",
        reason_code=reason_code,
        message=message,
        details=details,
    )


def _candidate_ownership(*method_names: str) -> dict[str, tuple[str, ...]]:
    """Identify candidate-local symbols that can repair a behavior failure."""

    return {
        "implicated_methods": tuple(
            dict.fromkeys(name for name in method_names if name.isidentifier())
        ),
        "implicated_files": ("recclaw_ext/candidate.py",),
    }


def _package_failure(reason_code: str, message: str) -> InnovationSpineError:
    return InnovationSpineError(
        failure_class="PACKAGE",
        reason_code=reason_code,
        message=message,
    )


def origin_blind_projection(spec: OpenResearchSpecV1) -> dict[str, Any]:
    """Project semantic implementation inputs without source/context identity."""

    if not isinstance(spec, OpenResearchSpecV1):
        raise _implementation_failure(
            "INVALID_OPEN_SPEC",
            "origin-blind projection requires OpenResearchSpecV1",
        )
    source = spec.to_dict()
    projection = canonical_value(
        {
            field_name: source[field_name]
            for field_name in _BLIND_SPEC_FIELDS
            if field_name in source
        }
    )
    _reject_forbidden_request_keys(projection)
    return projection


def _reject_forbidden_request_keys(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            field_name = str(key)
            if field_name.lower() in _FORBIDDEN_REQUEST_KEYS:
                raise _implementation_failure(
                    "BLIND_REQUEST_IDENTITY_OR_OUTCOME_LEAK",
                    "blind request contains forbidden field "
                    + ".".join((*path, field_name)),
                )
            _reject_forbidden_request_keys(
                item,
                path=(*path, field_name),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden_request_keys(
                item,
                path=(*path, str(index)),
            )


def profile_model_hooks_for_program(
    *,
    search_space_id: str,
    program_payload: Mapping[str, Any],
    component_specs: Any,
) -> tuple[str, ...] | None:
    """Return the compiler-owned hooks for one mechanically closed profile graph."""

    profile_spaces = frozenset(
        {
            "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1",
            "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1",
        }
    )
    if search_space_id not in profile_spaces:
        return None
    if program_payload.get("construction_mode") == "CUSTOM_MODEL":
        changed = {
            item.get("slot_id") for item in program_payload.get("changed_slots", ())
        }
        components = program_payload.get("components", ())
        # A local custom node does not transfer the unchanged parent lifecycle
        # to free-form implementation. The profile contract below checks that
        # its existing hooks can actually own this graph.
        if search_space_id == "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1":
            if not changed or not changed <= {
                "TEMPORAL_POSITION_ENCODING", "INTEREST_ROUTING", "STATE_UPDATE_GATING",
            }:
                return None
        elif (
            changed != {"GENERATIVE_DYNAMICS"}
            or {item.get("slot_id") for item in components} != {
                "STATE_REPRESENTATION", "FORWARD_PATH", "TIME_SCHEDULE",
                "GENERATIVE_DYNAMICS", "DENOISING_FLOW_OBJECTIVE",
                "RECOVERY_SOLVER", "SCORE_HEAD",
            }
        ):
            return None
    if program_payload.get("removed_slots"):
        raise ValueError(
            "UNSUPPORTED_PROFILE_MECHANICAL_BINDING: profile candidate with "
            "removed slots has no compiler-owned outer lifecycle"
        )
    if isinstance(component_specs, Mapping):
        normalized_component_specs = dict(component_specs)
    elif isinstance(component_specs, (tuple, list)):
        normalized_component_specs = {}
        for spec in component_specs:
            component_id = (
                spec.get("component_id") if isinstance(spec, Mapping) else None
            )
            if (
                not isinstance(component_id, str)
                or not component_id
                or component_id in normalized_component_specs
            ):
                raise ValueError(
                    "UNSUPPORTED_PROFILE_MECHANICAL_BINDING: profile components "
                    "require unique normalized component_id values"
                )
            normalized_component_specs[component_id] = spec
    else:
        raise ValueError(
            "UNSUPPORTED_PROFILE_MECHANICAL_BINDING: profile component "
            "specifications must be a mapping or sequence"
        )
    if search_space_id == "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1":
        from recclaw_core.search_spaces.sequential_scaling_v1.mechanism_scaffold import (
            SequentialScalingMechanismBindingError,
            sequential_scaling_profile_contract,
        )

        try:
            return sequential_scaling_profile_contract(
                normalized_component_specs,
                program_payload.get("changed_slots", ()),
            )
        except SequentialScalingMechanismBindingError as error:
            raise ValueError(
                "UNSUPPORTED_PROFILE_MECHANICAL_BINDING: " + str(error)
            ) from error

    from recclaw_core.search_spaces.diffusion_flow_cf_v1.mechanism_scaffold import (
        DiffusionFlowMechanismBindingError,
        diffusion_flow_profile_contract,
    )

    try:
        contract = diffusion_flow_profile_contract(
            normalized_component_specs,
            program_payload.get("changed_slots", ()),
        )
    except DiffusionFlowMechanismBindingError as error:
        raise ValueError(
            "UNSUPPORTED_PROFILE_MECHANICAL_BINDING: " + str(error)
        ) from error
    return None if contract is None else tuple(contract["model_hooks"])


def _expected_profile_model_hooks(
    value: Mapping[str, Any],
    component_specs: Mapping[str, Any],
) -> tuple[str, ...] | None:
    payload = value["mechanism_program"].get("program_payload", {})
    if not isinstance(payload, Mapping):
        return None
    try:
        return profile_model_hooks_for_program(
            search_space_id=str(value["space_identity"].get("search_space_id", "")),
            program_payload=payload,
            component_specs=component_specs,
        )
    except ValueError:
        return None


def _normalized_compiled_mechanism(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _COMPILED_MECHANISM_FIELDS:
        raise _implementation_failure(
            "COMPILED_MECHANISM_FIELDS_INVALID",
            "compiled_mechanism does not match the exact implementation boundary",
        )
    if value.get("schema") != "recclaw.compiled-mechanism-implementation.v1":
        raise _implementation_failure(
            "COMPILED_MECHANISM_SCHEMA_INVALID",
            "compiled_mechanism schema is unsupported",
        )
    candidate_id = value.get("compiler_candidate_id")
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or candidate_id != candidate_id.strip()
    ):
        raise _implementation_failure(
            "COMPILED_MECHANISM_IDENTITY_INVALID",
            "compiler_candidate_id must be normalized and non-empty",
        )
    for field_name in (
        "mechanism_program_digest",
        "mechanism_semantics_digest",
    ):
        try:
            validate_sha256(str(value.get(field_name)), field_name=field_name)
        except Exception as error:
            raise _implementation_failure(
                "COMPILED_MECHANISM_IDENTITY_INVALID",
                f"{field_name} must be a SHA-256 digest",
            ) from error
    if not isinstance(value.get("mechanism_program"), Mapping):
        raise _implementation_failure(
            "COMPILED_MECHANISM_PROGRAM_INVALID",
            "mechanism_program must be the complete compiler input mapping",
        )
    if not isinstance(value.get("resolved_ir"), Mapping):
        raise _implementation_failure(
            "COMPILED_MECHANISM_IR_INVALID",
            "resolved_ir must be the complete compiler-resolved mapping",
        )
    if not isinstance(value.get("space_identity"), Mapping):
        raise _implementation_failure(
            "COMPILED_MECHANISM_SPACE_INVALID",
            "space_identity must be the compiler-bound search-space identity",
        )
    capabilities = value.get("required_capabilities")
    if not isinstance(capabilities, (tuple, list)) or any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in capabilities
    ):
        raise _implementation_failure(
            "COMPILED_MECHANISM_CAPABILITIES_INVALID",
            "required_capabilities must contain normalized strings",
        )
    binding = value.get("implementation_binding")
    if (
        not isinstance(binding, Mapping)
        or not _IMPLEMENTATION_BINDING_FIELDS.issubset(binding)
        or not set(binding).issubset(
            _IMPLEMENTATION_BINDING_FIELDS
            | _IMPLEMENTATION_BINDING_OPTIONAL_MAPPING_FIELDS
        )
    ):
        raise _implementation_failure(
            "COMPILED_MECHANISM_BINDING_INVALID",
            "implementation_binding does not cover the exact program structure",
        )
    for field_name in sorted(_IMPLEMENTATION_BINDING_SEQUENCE_FIELDS):
        items = binding[field_name]
        if not isinstance(items, (tuple, list)) or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in items
        ):
            raise _implementation_failure(
                "COMPILED_MECHANISM_BINDING_INVALID",
                f"implementation_binding.{field_name} must contain normalized strings",
            )
        if list(items) != sorted(set(items)):
            raise _implementation_failure(
                "COMPILED_MECHANISM_BINDING_INVALID",
                f"implementation_binding.{field_name} must be sorted and unique",
            )
    component_specs = binding["component_specs"]
    if not isinstance(component_specs, Mapping) or list(component_specs) != list(
        binding["component_ids"]
    ):
        raise _implementation_failure(
            "COMPILED_MECHANISM_BINDING_INVALID",
            "implementation_binding.component_specs must exactly cover component_ids",
        )
    for component_id, component_spec in component_specs.items():
        if (
            not isinstance(component_spec, Mapping)
            or component_spec.get("component_id") != component_id
            or not isinstance(component_spec.get("parameters"), Mapping)
            or not isinstance(component_spec.get("slot_id"), str)
            or not component_spec.get("slot_id")
        ):
            raise _implementation_failure(
                "COMPILED_MECHANISM_BINDING_INVALID",
                "implementation_binding.component_specs contains an invalid component",
            )
    source_ownership = binding.get("source_ownership")
    if source_ownership is None:
        derived_hooks = _expected_profile_model_hooks(value, component_specs)
        payload = value["mechanism_program"].get("program_payload", {})
        changed_slots = (
            payload.get("changed_slots", ()) if isinstance(payload, Mapping) else ()
        )
        search_space_id = value["space_identity"].get("search_space_id")
        if derived_hooks is not None:
            source_ownership = {
                "mode": PROFILE_MODEL_HOOKS_RESPONSE_MODE,
                "model_hooks": list(derived_hooks),
            }
            effective_binding = dict(binding)
            effective_binding["source_ownership"] = source_ownership
            effective_value = dict(value)
            effective_value["implementation_binding"] = effective_binding
            value = effective_value
            binding = effective_binding
        elif (
            search_space_id
            in {
                "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1",
                "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1",
            }
            and isinstance(payload, Mapping)
            and payload.get("construction_mode") != "CUSTOM_MODEL"
            and changed_slots
        ):
            raise _implementation_failure(
                "UNSUPPORTED_PROFILE_MECHANICAL_BINDING",
                "profile COMPOSITION lacks a compiler-owned hook for its changed slots",
            )
    if source_ownership is not None:
        if not isinstance(source_ownership, Mapping):
            raise _implementation_failure(
                "COMPILED_MECHANISM_BINDING_INVALID",
                "implementation_binding.source_ownership must be a mapping",
            )
        ownership_mode = source_ownership.get("mode")
        if ownership_mode == "FULL_SOURCE_V1":
            if set(source_ownership) != {"mode"}:
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BINDING_INVALID",
                    "full-source ownership contains unexpected fields",
                )
        elif ownership_mode == "PARENT_LOCAL_SLOT_HOOK_V1":
            if set(source_ownership) != {"mode", "model_hook"}:
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BINDING_INVALID",
                    "parent-local ownership fields are invalid",
                )
            hook = source_ownership.get("model_hook")
            expected_hook_fields = {
                "component_ids",
                "hook_point",
                "input_ports",
                "method_name",
                "output_port",
                "slot_id",
            }
            if not isinstance(hook, Mapping) or set(hook) != expected_hook_fields:
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BINDING_INVALID",
                    "parent-local model hook fields are invalid",
                )
            hook_components = hook.get("component_ids")
            input_ports = hook.get("input_ports")
            hook_specs = {
                "PRIMARY_OBJECTIVE": {
                    "hook_point": "PRIMARY_LOSS_WITH_PARENT_VIEWS_V1",
                    "input_ports": ["interaction", "representations"],
                    "method_name": "recclaw_primary_objective",
                },
                "ENCODER": {
                    "hook_point": "PRE_PROPAGATION_REPRESENTATION_V1",
                    "input_ports": ["representation"],
                    "method_name": "recclaw_encode_representation",
                },
                "PROPAGATION_AGGREGATION": {
                    "hook_point": "POST_PROPAGATION_MESSAGE_V1",
                    "input_ports": [
                        "layer_index",
                        "message",
                        "prior",
                        "relation",
                    ],
                    "method_name": "recclaw_propagation_aggregation",
                },
                "SCORE_HEAD": {
                    "hook_point": hook.get("hook_point") if hook.get("hook_point") in {
                        "BPR_AND_EVALUATION_SCORE_V1", "EVALUATION_ONLY_SCORE_V1"
                    } else None,
                    "input_ports": ["item_representation", "pairwise", "user_representation"],
                    "method_name": "recclaw_score_head",
                },
            }
            hook_slot = hook.get("slot_id")
            hook_spec = hook_specs.get(hook_slot)
            if (
                hook_spec is None
                or hook_spec["hook_point"] is None
                or not isinstance(hook_components, (tuple, list))
                or not hook_components
                or any(
                    not isinstance(item, str) or not item
                    for item in hook_components
                )
                or list(hook_components) != sorted(set(hook_components))
                or any(
                    item not in component_specs
                    or component_specs[item].get("slot_id") != hook_slot
                    for item in hook_components
                )
                or not isinstance(input_ports, (tuple, list))
                or any(not isinstance(item, str) or not item for item in input_ports)
                or list(input_ports) != sorted(set(input_ports))
                or list(input_ports) != hook_spec["input_ports"]
                or hook.get("hook_point") != hook_spec["hook_point"]
                or hook.get("method_name") != hook_spec["method_name"]
                or hook.get("output_port") != (
                    "objective" if hook_slot == "PRIMARY_OBJECTIVE"
                    else "score" if hook_slot == "SCORE_HEAD" else "representation"
                )
            ):
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BINDING_INVALID",
                    "parent-local model hook does not match its encoder components",
                )
            program_payload = value["mechanism_program"].get(
                "program_payload", {}
            )
            changed_slots = sorted(
                {
                    str(item.get("slot_id"))
                    for item in program_payload.get("changed_slots", ())
                    if isinstance(item, Mapping)
                }
            )
            if (
                changed_slots != [hook_slot]
                or (program_payload.get("removed_slots") and (
                    hook_slot != "PRIMARY_OBJECTIVE" or
                    set(program_payload.get("removed_slots", ())) - {
                        item.get("slot_id") for item in program_payload.get("components", ())
                        if isinstance(item, Mapping)
                    }
                ))
            ):
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BINDING_INVALID",
                    "parent-local ownership exceeds the declared program delta",
                )
        elif ownership_mode == PROFILE_MODEL_HOOKS_RESPONSE_MODE:
            if set(source_ownership) != {"mode", "model_hooks"}:
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BINDING_INVALID",
                    "profile model-hook ownership fields are invalid",
                )
            model_hooks = source_ownership.get("model_hooks")
            expected_hooks = _expected_profile_model_hooks(value, component_specs)
            if (
                expected_hooks is None
                or not isinstance(model_hooks, (tuple, list))
                or tuple(model_hooks) != expected_hooks
            ):
                raise _implementation_failure(
                    "UNSUPPORTED_PROFILE_MECHANICAL_BINDING",
                    "profile model hooks do not match the compiled changed slots",
                )
        else:
            raise _implementation_failure(
                "COMPILED_MECHANISM_BINDING_INVALID",
                "implementation_binding.source_ownership mode is unsupported",
            )
    normalized = canonical_value(dict(value))
    _reject_forbidden_request_keys(normalized)
    return normalized


def _blind_execution_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove observed Producer provenance before the shared implementer call."""

    contract = dict(value)
    config = contract.get("config")
    if isinstance(config, Mapping) and "producer_role" in config:
        blind_config = dict(config)
        blind_config.pop("producer_role", None)
        contract["config"] = blind_config
    return canonical_value(contract)


def _normalized_exact_parent_bundle(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    required = {
        "candidate_id",
        "capability_ref",
        "files",
        "instruction",
        "program_digest",
        "source_tree_digest",
    }
    if (
        not isinstance(value, Mapping)
        or not required.issubset(value)
        or not set(value).issubset(required | {"execution_contract"})
    ):
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent bundle fields are invalid",
        )
    parent_contract = value.get("execution_contract")
    if parent_contract is not None and (
        not isinstance(parent_contract, Mapping)
        or set(parent_contract)
        != {"capability_family", "model", "base_model_config", "config"}
        or not isinstance(parent_contract.get("config"), Mapping)
    ):
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent execution contract is invalid",
        )
    if value.get("instruction") != "CLONE_EXACT_PARENT_AND_LOCAL_PATCH":
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent bundle must require clone-and-local-patch",
        )
    for field_name in ("candidate_id", "capability_ref"):
        field = value.get(field_name)
        if not isinstance(field, str) or not field or field != field.strip():
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                f"exact parent {field_name} is invalid",
            )
    for field_name in ("program_digest", "source_tree_digest"):
        try:
            validate_sha256(value.get(field_name), field_name=field_name)
        except Exception as error:
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                f"exact parent {field_name} is invalid",
            ) from error
    files = value.get("files")
    if not isinstance(files, (tuple, list)) or not files:
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent bundle has no files",
        )
    normalized_files: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"content", "path", "sha256"}:
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                "exact parent file fields are invalid",
            )
        path = validate_relative_artifact_path(str(item["path"]))
        content = item["content"]
        digest = item["sha256"]
        if (
            path in seen
            or not isinstance(content, str)
            or hashlib.sha256(content.encode("utf-8")).hexdigest() != digest
        ):
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                "exact parent file bytes or SHA-256 are invalid",
            )
        seen.add(path)
        normalized_files.append(
            {"content": content, "path": path, "sha256": str(digest)}
        )
    return canonical_value(
        {
            **dict(value),
            "files": sorted(normalized_files, key=lambda item: item["path"]),
        }
    )


def _refresh_machine_owned_parent_trainer(
    *,
    compiled_mechanism: Mapping[str, Any] | None,
    exact_parent_bundle: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Bind resumed P6 candidates to the current machine-owned trainer."""

    parent = _normalized_exact_parent_bundle(exact_parent_bundle)
    if parent is None or compiled_mechanism is None:
        return parent
    space_identity = compiled_mechanism.get("space_identity")
    if (
        not isinstance(space_identity, Mapping)
        or space_identity.get("search_space_id")
        != "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1"
        or not parent["capability_ref"].startswith(
            "parent:semantic-id-generative:"
        )
    ):
        return parent

    trainer_path = (
        Path(__file__).resolve().parents[2]
        / "search_spaces/semantic_id_generative_v1/assets/"
        "liger_parent_trainer.py.tmpl"
    )
    trainer_source = trainer_path.read_text(encoding="utf-8")
    trainer_digest = hashlib.sha256(trainer_source.encode("utf-8")).hexdigest()
    files = []
    trainer_found = False
    for item in parent["files"]:
        row = dict(item)
        if row["path"] == "recclaw_ext/trainer.py":
            trainer_found = True
            row = {
                "content": trainer_source,
                "path": row["path"],
                "sha256": trainer_digest,
            }
        files.append(row)
    if not trainer_found:
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "semantic-ID exact parent lacks its machine-owned trainer",
        )
    identity_rows = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": len(item["content"].encode("utf-8")),
        }
        for item in files
    ]
    return canonical_value(
        {
            **parent,
            "files": files,
            "source_tree_digest": sha256_digest(
                {"files": sorted(identity_rows, key=lambda row: row["path"])}
            ),
        }
    )


_SEMANTIC_ID_PARENT_OUTER_METHODS = frozenset(
    {
        "calculate_loss",
        "full_sort_predict",
        "predict",
        "recclaw_generate_semantic_ids",
        "recclaw_generation_batch",
        "recclaw_resolve_semantic_ids",
        "recclaw_target_ranks",
    }
)
_SEMANTIC_ID_PARENT_SLOT_HOOKS = {
    "USER_CONTEXT_ENCODER": frozenset({"recclaw_prepare_liger_batch"}),
    "GENERATIVE_OBJECTIVE": frozenset({"recclaw_additional_objective"}),
    "DECODING_STRATEGY": frozenset(
        {"recclaw_generation_kwargs", "recclaw_postprocess_generated"}
    ),
}
_SEMANTIC_ID_PARENT_HOOK_SIGNATURES = {
    "recclaw_initialize_mechanism": (
        "def recclaw_initialize_mechanism(self, config, dataset):\n"
        "    pass\n"
    ),
    "recclaw_prepare_liger_batch": (
        "def recclaw_prepare_liger_batch(self, input_batch):\n"
        "    pass\n"
    ),
    "recclaw_generation_kwargs": (
        "def recclaw_generation_kwargs(self, input_batch):\n"
        "    pass\n"
    ),
    "recclaw_postprocess_generated": (
        "def recclaw_postprocess_generated(self, generated, input_batch):\n"
        "    pass\n"
    ),
    "recclaw_additional_objective": (
        "def recclaw_additional_objective("
        "self, interaction, outputs, logits, targets):\n"
        "    pass\n"
    ),
}


def _semantic_id_parent_hook_contract(
    compiled_mechanism: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Keep unchanged P6 machinery while exposing the declared research methods."""

    space_identity = compiled_mechanism.get("space_identity")
    if (
        not isinstance(space_identity, Mapping)
        or space_identity.get("search_space_id")
        != "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1"
    ):
        return None
    program = compiled_mechanism.get("mechanism_program")
    payload = program.get("program_payload") if isinstance(program, Mapping) else None
    if not isinstance(payload, Mapping) or payload.get("construction_mode") == "CUSTOM_MODEL":
        return None
    stage_methods = any(
        isinstance(component, Mapping)
        and str(component.get("primitive_id", "")).startswith("staging.")
        for component in payload.get("components", ())
    ) or "OPTIMIZATION_STAGING" in payload.get("removed_slots", ())
    if not stage_methods and {"POST_TRAINING_MUTATION", "TRAINING_STAGE_MUTATION"}.intersection(
        compiled_mechanism.get("required_capabilities", ())
    ):
        # A post-training objective without an explicit stage remains full-source.
        # Do not invent its missing schedule or turn it into an additive loss.
        return None
    changed_slots = frozenset(
        str(item.get("slot_id"))
        for item in payload.get("changed_slots", ())
        if isinstance(item, Mapping) and isinstance(item.get("slot_id"), str)
    )
    allowed_parent_hooks = {"recclaw_initialize_mechanism"}
    from recclaw_core.search_spaces.semantic_id_generative_v1.semantic_decode_scaffold import (
        machine_owned_catalog_beam,
    )
    builtin_catalog_beam = machine_owned_catalog_beam({
        str(index): component
        for index, component in enumerate(payload.get("components", ()))
    }) is not None
    forbidden_outer_methods = set(_SEMANTIC_ID_PARENT_OUTER_METHODS)
    if stage_methods:
        allowed_parent_hooks.add("calculate_loss")
        forbidden_outer_methods.remove("calculate_loss")
    if builtin_catalog_beam:
        forbidden_outer_methods.update(_SEMANTIC_ID_PARENT_SLOT_HOOKS["DECODING_STRATEGY"])
    required_all: set[str] = set()
    required_any: list[tuple[str, ...]] = []
    for slot_id, hooks in _SEMANTIC_ID_PARENT_SLOT_HOOKS.items():
        if slot_id not in changed_slots:
            continue
        if (slot_id == "DECODING_STRATEGY" and builtin_catalog_beam) or (
            slot_id == "GENERATIVE_OBJECTIVE" and stage_methods
        ):
            continue
        allowed_parent_hooks.update(hooks)
        if slot_id == "DECODING_STRATEGY":
            required_any.append(tuple(sorted(hooks)))
        else:
            required_all.update(hooks)
    return {
        "allowed_parent_hooks": sorted(allowed_parent_hooks),
        "forbidden_outer_methods": sorted(forbidden_outer_methods),
        "machine_owned_catalog_beam": builtin_catalog_beam,
        "required_all": sorted(required_all),
        "required_any": [list(group) for group in required_any],
        **({"owns_training_stages": True} if stage_methods else {}),
    }


def _semantic_id_machine_decode_contract(compiled: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if compiled["space_identity"].get("search_space_id") != "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1":
        return None
    changed = {
        row["slot_id"] for row in compiled["mechanism_program"]["program_payload"].get("changed_slots", ())
    }
    if changed.intersection({"ITEM_RESOLUTION", "DENSE_RETRIEVAL_CORRECTION"}):
        return None
    from recclaw_core.search_spaces.semantic_id_generative_v1.semantic_decode_scaffold import _decode_contract

    return _decode_contract(compiled["implementation_binding"]["component_specs"])


def _implementation_response_mode(
    *,
    compiled_mechanism: Mapping[str, Any] | None,
    exact_parent_bundle: Mapping[str, Any] | None,
) -> str:
    if compiled_mechanism is None:
        return FULL_SOURCE_RESPONSE_MODE
    binding = compiled_mechanism.get("implementation_binding")
    source_ownership = (
        binding.get("source_ownership")
        if isinstance(binding, Mapping)
        else None
    )
    program = compiled_mechanism.get("mechanism_program")
    payload = (
        program.get("program_payload")
        if isinstance(program, Mapping)
        else None
    )
    if (
        isinstance(source_ownership, Mapping)
        and source_ownership.get("mode") == "FULL_SOURCE_V1"
    ):
        search_space_id = (
            compiled_mechanism.get("space_identity", {}).get("search_space_id")
            if isinstance(compiled_mechanism.get("space_identity"), Mapping)
            else None
        )
        if (
            exact_parent_bundle is not None
            and search_space_id == "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1"
            and _semantic_id_parent_hook_contract(compiled_mechanism) is not None
        ):
            return SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE
        # The search-space compiler owns this decision.  In particular, a
        # multi-slot COMPOSITION cannot be reduced to the narrower parent
        # method-patch ABI merely because exact parent bytes are available.
        return FULL_SOURCE_RESPONSE_MODE
    if (
        isinstance(source_ownership, Mapping)
        and source_ownership.get("mode") == "PARENT_LOCAL_SLOT_HOOK_V1"
    ):
        if exact_parent_bundle is None:
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                "parent-local source ownership requires exact parent bytes",
            )
        hook = source_ownership["model_hook"]
        if hook["slot_id"] == "PRIMARY_OBJECTIVE":
            from recclaw_core.research_line.bl_icf_objective_scaffold import (
                parent_objective_is_bindable, retained_components_match,
            )

            source = next((row["content"] for row in exact_parent_bundle["files"]
                           if row["path"] == "recclaw_ext/candidate.py"), "")
            if not parent_objective_is_bindable(source) or not retained_components_match(
                source, binding["component_specs"],
                require_metadata=bool(set(payload.get("removed_slots", ())) - {"PRIMARY_OBJECTIVE"}),
            ):
                return FULL_SOURCE_RESPONSE_MODE
        if hook["slot_id"] == "SCORE_HEAD":
            from recclaw_core.research_line.bl_icf_score_scaffold import parent_score_is_bindable

            source = next((row["content"] for row in exact_parent_bundle["files"]
                           if row["path"] == "recclaw_ext/candidate.py"), "")
            if not parent_score_is_bindable(
                source, train_objective=hook["hook_point"] == "BPR_AND_EVALUATION_SCORE_V1"
            ):
                # Keep this mechanism available through the original full
                # implementation path, not a failed/retried local patch.
                return FULL_SOURCE_RESPONSE_MODE
        return PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE
    if (
        isinstance(source_ownership, Mapping)
        and source_ownership.get("mode") == PROFILE_MODEL_HOOKS_RESPONSE_MODE
    ):
        if exact_parent_bundle is None:
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                "profile model hooks require exact parent bytes",
            )
        return PROFILE_MODEL_HOOKS_RESPONSE_MODE
    if exact_parent_bundle is None:
        return FULL_SOURCE_RESPONSE_MODE
    if (
        isinstance(payload, Mapping)
        and payload.get("construction_mode") == "COMPOSITION"
    ):
        return PARENT_METHOD_PATCH_RESPONSE_MODE
    return FULL_SOURCE_RESPONSE_MODE


def build_shared_implementer_request(
    spec: OpenResearchSpecV1,
    *,
    policy: SharedImplementerPolicy,
    compiled_mechanism: Mapping[str, Any] | None = None,
    exact_parent_bundle: Mapping[str, Any] | None = None,
    research_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only payload visible to a shared implementation service."""

    if not isinstance(policy, SharedImplementerPolicy):
        raise _implementation_failure(
            "INVALID_SHARED_POLICY",
            "shared implementer policy has the wrong type",
        )
    projection = origin_blind_projection(spec)
    normalized_compiled = _normalized_compiled_mechanism(compiled_mechanism)
    normalized_parent = _refresh_machine_owned_parent_trainer(
        compiled_mechanism=normalized_compiled,
        exact_parent_bundle=exact_parent_bundle,
    )
    identity_preimage: dict[str, Any] = {"blind_research_spec": projection}
    if normalized_compiled is not None:
        identity_preimage["compiled_mechanism"] = normalized_compiled
    if normalized_parent is not None:
        identity_preimage["exact_parent_bundle"] = normalized_parent
    projection_digest = sha256_digest(identity_preimage)
    service_policy: dict[str, Any] = {
        "candidate_local_write_only": True,
        "dependency_identity_digest": policy.dependency_identity_digest,
        "dependency_identity_ref": policy.dependency_identity_ref,
        "implementation_token_ceiling": policy.implementation_token_ceiling,
        "prompt_digest": policy.prompt_digest,
        "response_mode": _implementation_response_mode(
            compiled_mechanism=normalized_compiled,
            exact_parent_bundle=normalized_parent,
        ),
        "runtime_identity_digest": policy.runtime_identity_digest,
        "runtime_identity_ref": policy.runtime_identity_ref,
        "tool_policy_digest": policy.tool_policy_digest,
    }
    if MODEL_CONFIG_MAPPING_REQUIREMENT in spec.implementation_requirements:
        service_policy["model_constructor_config"] = MODEL_CONFIG_MAPPING_CONTRACT
    if (
        normalized_compiled is not None
        and service_policy["response_mode"]
        == PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE
    ):
        service_policy["parent_local_source_ownership"] = (
            normalized_compiled["implementation_binding"]["source_ownership"]
        )
    if (
        normalized_compiled is not None
        and service_policy["response_mode"] == PROFILE_MODEL_HOOKS_RESPONSE_MODE
    ):
        service_policy["profile_model_source_ownership"] = (
            normalized_compiled["implementation_binding"]["source_ownership"]
        )
        service_policy["profile_model_delta_contract"] = (
            "Implement the mutable model_hooks with new recclaw_ helpers as needed. "
            "Retained hooks and their helpers keep their parent implementation; helpers "
            "used only by mutable hooks may change. Unchanged "
            "copies may be omitted. recclaw_initialize_mechanism supplies only the "
            "current initialization delta: the compiler runs inherited initialization "
            "first, then this delta. Explicit new values may override inherited values."
        )
    if (
        normalized_compiled is not None
        and service_policy["response_mode"]
        == PARENT_METHOD_PATCH_RESPONSE_MODE
    ):
        service_policy["parent_method_source_ownership"] = {
            "model_methods": sorted(
                _parent_method_model_signatures(normalized_compiled)
            ),
            "trainer_methods": sorted(_TRAINER_PATCH_SIGNATURES),
        }
    if (
        normalized_compiled is not None
        and service_policy["response_mode"]
        == SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE
    ):
        parent_hook_contract = _semantic_id_parent_hook_contract(
            normalized_compiled
        )
        assert parent_hook_contract is not None
        service_policy["machine_owned_outer_abi"] = {
            "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
            "model_base": "official LIGER/TIGER T5 parent",
            "complete_sid_scoring": {
                "call": "self.recclaw_complete_sid_token_log_probs(outputs, sids)",
                "inputs": (
                    "outputs is the live positive parent forward output; sids is "
                    "self.item_sids[item_ids], catalog-absolute tokens [B,4] or [B,K,4]. "
                    "The additive hook targets argument is item IDs [B], not SID tokens."
                ),
                "returns": (
                    "Differentiable per-token log p(sid_t | history, sid_<t), same "
                    "shape as sids. Each code uses its own shifted decoder prefix; "
                    "the exact positive encoder state and history attention mask "
                    "are reused without recomputing the encoder or context hook."
                ),
                "ownership": (
                    "Machine-owned helper; call it, do not redefine it. The mechanism "
                    "owns negative sampling, score reduction and objective formula. "
                    "outputs.logits conditions on the positive SID prefix and cannot "
                    "score a different complete SID by gathering its tokens."
                ),
            },
            "trainer_lifecycle": (
                "research-owned stage methods within the unchanged evaluation protocol"
                if parent_hook_contract.get("owns_training_stages")
                else "official RecBole LIGER adapter"
            ),
            **(
                {"trainer_methods": sorted(_SEMANTIC_ID_STAGE_TRAINER_SIGNATURES)}
                if parent_hook_contract.get("owns_training_stages") else {}
            ),
            "mechanism_file": "recclaw_ext/candidate.py",
            "parent_hook_contract": parent_hook_contract,
            "mechanism_methods": [
                *parent_hook_contract["allowed_parent_hooks"],
                "mechanism-local recclaw_* helpers",
            ],
            "module_helpers": "recclaw_* or RECCLAW_*",
            "forbidden_mechanical_methods": [
                "__init__",
                "forward",
                *parent_hook_contract["forbidden_outer_methods"],
            ],
        }
    if normalized_compiled is not None:
        binding = normalized_compiled.get("implementation_binding")
        component_specs = (
            binding.get("component_specs")
            if isinstance(binding, Mapping)
            else {}
        )
        slots = {
            str(spec.get("slot_id"))
            for spec in component_specs.values()
            if isinstance(component_specs, Mapping)
            and isinstance(spec, Mapping)
            and isinstance(spec.get("slot_id"), str)
        }
        service_policy["objective_parameter_semantics"] = {
            "geometry_regularization_declared": (
                "GEOMETRY_REGULARIZATION" in slots
            ),
            "objective_weight_target": "PRIMARY_OBJECTIVE_ONLY",
            "regularization_weight_source": (
                "GEOMETRY_REGULARIZATION_COMPONENT_ONLY"
            ),
        }
        primitive_ids = {
            str(value)
            for value in (
                binding.get("primitive_ids", ())
                if isinstance(binding, Mapping)
                else ()
            )
        }
        if normalized_compiled["space_identity"].get("search_space_id") == "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1":
            outer = service_policy.setdefault("machine_owned_outer_abi", {})
            outer["model_metadata"] = {
                "type": "ModelType.SEQUENTIAL", "input_type": "InputType.POINTWISE",
                "owner": "compiler; model architecture and class base remain research choices in full-source mode",
            }
            decode_contract = _semantic_id_machine_decode_contract(normalized_compiled)
            if decode_contract is not None:
                outer["generation_hook"] = {
                    "signature": "recclaw_generation_batch(self, interaction, targets)",
                    "returns": "(semantic_ids, query_state, history)",
                    "semantic_ids": (
                        "generated position-local SID tokens [B,K,4] from the declared parallel decoder"
                        if decode_contract[3] else
                        "generated catalog-absolute SID tokens [B,K,4] from the declared generator"
                    ),
                    "query_state": "the loss-trained dense query [B,128]",
                    "history": "the actual conditioning item history [B,L]; never the current target",
                    "targets": "shape/device context only; target labels cannot condition generation",
                    "owner": "exact parent in scaffolded mode; research model in full-source mode",
                }
                outer["decode_methods"] = [
                    "recclaw_resolve_semantic_ids", "recclaw_target_ranks",
                    "_hybrid_scores", "predict", "full_sort_predict",
                ]
        mechanism_semantics: dict[str, Any] = {}
        if "generator.t5_encoder_decoder" in primitive_ids:
            mechanism_semantics["generator.t5_encoder_decoder"] = {
                "encoder_only_call": (
                    "Use self.encoder(...) or self.get_encoder()(...) when only "
                    "encoder outputs are needed."
                ),
                "forward_call_contract": (
                    "Every T5 forward call must include labels, decoder_input_ids, "
                    "or decoder_inputs_embeds; never call self(...) with only "
                    "encoder inputs."
                ),
                "generation_call": (
                    "Use generate(...) for autoregressive SID decoding."
                ),
            }
        if "ssl.objective.cross_layer_info_nce" in primitive_ids:
            mechanism_semantics["ssl.objective.cross_layer_info_nce"] = {
                "contrast_unit": "UNIQUE_ENTITY_ID",
                "duplicate_entity_policy": "SAME_ENTITY_NEVER_NEGATIVE",
                "required_behavior": (
                    "Repeated rows for one user or item identify the same positive "
                    "entity; deduplicate them or use a multi-positive mask rather "
                    "than treating row position as entity identity."
                ),
            }
        curriculum_declared = any(
            isinstance(component, Mapping)
            and component.get("slot_id") == "NEGATIVE_SAMPLER"
            and isinstance(component.get("parameters"), Mapping)
            and component["parameters"].get("hardness") == "CURRICULUM"
            for component in (
                component_specs.values()
                if isinstance(component_specs, Mapping)
                else ()
            )
        )
        if curriculum_declared:
            mechanism_semantics["sampler.hardness.curriculum"] = {
                "epoch_progression_causal": True,
                "model_state_causal": True,
                "same_epoch_cached": True,
            }
        if mechanism_semantics:
            service_policy["compiled_mechanism_semantics"] = mechanism_semantics
    if policy.execution_contract is not None:
        service_policy["execution_contract"] = _blind_execution_contract(
            policy.execution_contract
        )
    request_payload: dict[str, Any] = {
        "blind_candidate_id": "innovation-candidate-" + projection_digest[:24],
        "blind_research_spec": projection,
        "candidate_local_write_allowlist": policy.allowed_files,
        "schema": (
            "recclaw.shared-implementer-request.v2"
            if normalized_compiled is not None
            else "recclaw.shared-implementer-request.v1"
        ),
        "service_policy": service_policy,
    }
    if normalized_compiled is not None:
        request_payload["compiled_mechanism"] = normalized_compiled
    if normalized_parent is not None:
        request_payload["exact_parent_bundle"] = normalized_parent
    if research_feedback:
        request_payload["research_feedback"] = canonical_value(dict(research_feedback))
    request = canonical_value(request_payload)
    _reject_forbidden_request_keys(request)
    return request


_CLOSED_FORM_SOLVER_CALL_SUFFIXES = (
    ".solve",
    ".inv",
    ".inverse",
    ".pinv",
    ".cholesky",
    ".lu_factor",
    ".ldl_factor",
    ".factorized",
)
_INITIALIZATION_METHOD_NAMES = (
    "__init__",
    "recclaw_initialize_mechanism",
)


def _fresh_candidate_mro_methods(
    tree: ast.Module,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Resolve methods from the executable local candidate hierarchy only.

    Exact-parent bundles may carry several sibling model families with the same
    method names.  Module-wide name collection lets a later sibling masquerade
    as ``FreshCandidateModel``.  Keep the first implementation encountered on
    the candidate's own local class/base path and ignore unrelated siblings.
    """

    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    candidate = classes.get("FreshCandidateModel")
    if candidate is None:
        # Legacy non-conversion entrypoints do not necessarily use the fixed
        # FreshCandidateModel name. Preserve their previous family-neutral
        # behavior while the compiled conversion path uses the exact MRO.
        return {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    visited: set[str] = set()

    def collect(class_node: ast.ClassDef) -> None:
        if class_node.name in visited:
            return
        visited.add(class_node.name)
        for node in class_node.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name not in methods
            ):
                methods[node.name] = node
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                collect(classes[base.id])

    collect(candidate)
    return methods


def _fresh_candidate_execution_graph(
    tree: ast.Module,
) -> tuple[
    dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    dict[str, set[str]],
]:
    """Resolve the candidate MRO plus module helpers it actually calls.

    Recommendation models commonly keep score/loss algebra in module-level
    helpers such as ``_pairwise_bpr``. Omitting those helpers makes a valid
    inherited execution path appear absent. Unrelated sibling-model methods
    remain excluded by the candidate-MRO resolver.
    """

    methods = _fresh_candidate_mro_methods(tree)
    candidate_present = any(
        isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"
        for node in tree.body
    )
    callables: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = dict(methods)
    module_keys: dict[str, str] = {}
    if candidate_present:
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = f"module::{node.name}"
            module_keys[node.name] = key
            callables[key] = node

    def method_receiver(value: ast.AST) -> bool:
        return (
            isinstance(value, ast.Name)
            and value.id in {"self", "super"}
        ) or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "super"
        )

    calls: dict[str, set[str]] = {}
    for name, implementation in callables.items():
        targets: set[str] = set()
        for call in (
            node for node in ast.walk(implementation) if isinstance(node, ast.Call)
        ):
            if (
                isinstance(call.func, ast.Attribute)
                and method_receiver(call.func.value)
                and call.func.attr in methods
            ):
                targets.add(call.func.attr)
            elif isinstance(call.func, ast.Name):
                module_key = module_keys.get(call.func.id)
                if module_key is not None:
                    targets.add(module_key)
        calls[name] = targets
    return callables, calls


def _closed_form_solver_facts(
    methods: Mapping[str, ast.AST],
    *,
    dotted_name: Callable[[ast.AST], str],
    reachable_from: Callable[[str], set[str]],
) -> tuple[set[str], set[str], set[str]]:
    """Return solver methods, precomputed cache attrs, and effectful consumers."""

    solver_methods: set[str] = set()
    direct_solver_methods: set[str] = set()
    for method_name, method in methods.items():
        direct_solver = any(
            isinstance(node, ast.Call)
            and dotted_name(node.func).lower().endswith(
                _CLOSED_FORM_SOLVER_CALL_SUFFIXES
            )
            for node in ast.walk(method)
        )
        iterative_solver = (
            "solve" in method_name.lower()
            and any(
                isinstance(node, (ast.For, ast.While)) for node in ast.walk(method)
            )
            and any(
                isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
                for node in ast.walk(method)
            )
        )
        if direct_solver or iterative_solver:
            solver_methods.add(method_name)
        if direct_solver:
            direct_solver_methods.add(method_name)

    precompute_methods = {
        name
        for name in methods
        if name in _INITIALIZATION_METHOD_NAMES or "precompute" in name.lower()
    }
    precompute_reachable = set().union(
        *(reachable_from(name) for name in precompute_methods)
    )

    def expression_sources(
        method_name: str,
        expression: ast.AST,
        seen_names: frozenset[str] = frozenset(),
    ) -> tuple[set[str], set[str]]:
        method = methods[method_name]
        parameters = {
            argument.arg
            for argument in (
                *method.args.posonlyargs,
                *method.args.args,
                *method.args.kwonlyargs,
            )
            if argument.arg not in {"self", "cls"}
        }
        local_values: dict[str, list[ast.AST]] = {}
        for node in ast.walk(method):
            value = getattr(node, "value", None)
            targets = (
                tuple(node.targets)
                if isinstance(node, ast.Assign)
                else (node.target,)
                if isinstance(node, ast.AnnAssign)
                else ()
            )
            if not isinstance(value, ast.AST):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    local_values.setdefault(target.id, []).append(value)
        attributes = {
            node.attr
            for node in ast.walk(expression)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }
        source_parameters: set[str] = set()
        for node in ast.walk(expression):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            if node.id in parameters:
                source_parameters.add(node.id)
                continue
            values = local_values.get(node.id, ())
            if len(values) != 1 or node.id in seen_names:
                continue
            nested_attrs, nested_params = expression_sources(
                method_name,
                values[0],
                seen_names | {node.id},
            )
            attributes.update(nested_attrs)
            source_parameters.update(nested_params)
        return attributes, source_parameters

    class ValueSourceCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()
            self.attributes: set[str] = set()

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # Matrix extent is metadata, not evidence that matrix values feed
            # the derived solver system (for example ``sp.eye(x.shape[0])``).
            if node.attr in {"shape", "ndim", "size"}:
                return
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                self.attributes.add(node.attr)
                return
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                self.names.add(node.id)

    def value_sources(
        expression: ast.AST,
        *,
        method_name: str | None = None,
        seen_names: frozenset[str] = frozenset(),
    ) -> tuple[set[str], set[str]]:
        collector = ValueSourceCollector()
        collector.visit(expression)
        if method_name is None:
            return collector.attributes, collector.names
        method = methods[method_name]
        for name in tuple(collector.names):
            if name in seen_names:
                continue
            bindings: list[ast.AST] = []
            for node in ast.walk(method):
                value = getattr(node, "value", None)
                targets = (
                    tuple(node.targets)
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                    if isinstance(node, ast.AnnAssign)
                    else ()
                )
                if not isinstance(value, ast.AST):
                    continue
                if getattr(node, "lineno", 0) >= getattr(expression, "lineno", 0):
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in targets
                ):
                    bindings.append(value)
            if len(bindings) != 1:
                continue
            nested_attributes, nested_names = value_sources(
                bindings[0],
                method_name=method_name,
                seen_names=seen_names | {name},
            )
            collector.names.remove(name)
            collector.attributes.update(nested_attributes)
            collector.names.update(nested_names)
        return collector.attributes, collector.names

    def parameter_sources(
        callee_name: str,
        parameter_name: str,
        seen: frozenset[tuple[str, str]] = frozenset(),
    ) -> set[str]:
        identity = (callee_name, parameter_name)
        if identity in seen:
            return set()
        callee = methods[callee_name]
        positional = [
            argument.arg
            for argument in (*callee.args.posonlyargs, *callee.args.args)
            if argument.arg not in {"self", "cls"}
        ]
        resolved: set[str] = set()
        for caller_name in precompute_reachable:
            for node in ast.walk(methods[caller_name]):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr == callee_name
                ):
                    continue
                argument: ast.AST | None = None
                if parameter_name in positional:
                    position = positional.index(parameter_name)
                    if position < len(node.args):
                        argument = node.args[position]
                if argument is None:
                    argument = next(
                        (
                            keyword.value
                            for keyword in node.keywords
                            if keyword.arg == parameter_name
                        ),
                        None,
                    )
                if argument is None:
                    continue
                attributes, parameters = expression_sources(caller_name, argument)
                resolved.update(attributes)
                for caller_parameter in parameters:
                    resolved.update(
                        parameter_sources(
                            caller_name,
                            caller_parameter,
                            seen | {identity},
                        )
                    )
        return resolved

    solver_input_attrs: set[str] = set()
    for method_name in solver_methods:
        for node in ast.walk(methods[method_name]):
            if not (
                isinstance(node, ast.Call)
                and dotted_name(node.func).lower().endswith(
                    _CLOSED_FORM_SOLVER_CALL_SUFFIXES
                )
            ):
                continue
            attributes, parameters = expression_sources(method_name, node)
            solver_input_attrs.update(attributes)
            for parameter in parameters:
                solver_input_attrs.update(parameter_sources(method_name, parameter))
        if method_name not in direct_solver_methods:
            attributes, _names = value_sources(methods[method_name])
            solver_input_attrs.update(attributes)
    precomputed_attributes = {
        target.attr
        for method_name in precompute_reachable
        for node in ast.walk(methods[method_name])
        for target in (
            tuple(node.targets)
            if isinstance(node, ast.Assign)
            else (node.target,)
            if isinstance(node, ast.AnnAssign)
            else ()
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    unbound_solver_inputs = solver_input_attrs - precomputed_attributes
    if unbound_solver_inputs:
        raise _implementation_failure(
            "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
            "closed-form solver input is not bound to precomputed relation/view "
            f"state: {sorted(unbound_solver_inputs)}",
            details=_candidate_ownership("__init__"),
        )

    attribute_values: dict[str, list[tuple[str, ast.AST]]] = {}
    for method_name in precompute_reachable:
        for node in ast.walk(methods[method_name]):
            value = getattr(node, "value", None)
            targets = (
                tuple(node.targets)
                if isinstance(node, ast.Assign)
                else (node.target,)
                if isinstance(node, ast.AnnAssign)
                else ()
            )
            if not isinstance(value, ast.AST):
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attribute_values.setdefault(target.attr, []).append(
                        (method_name, value)
                    )

    def has_relation_value_provenance(
        attribute: str,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if attribute in seen:
            return False
        bindings = attribute_values.get(attribute, ())
        if len(bindings) != 1:
            return False
        method_name, value = bindings[0]
        source_attributes, source_names = value_sources(
            value,
            method_name=method_name,
        )
        method = methods[method_name]
        external_value_parameters = {
            argument.arg
            for argument in (
                *method.args.posonlyargs,
                *method.args.args,
                *method.args.kwonlyargs,
            )
            if argument.arg not in {"self", "cls", "config"}
        }
        if source_names & external_value_parameters:
            return True
        return any(
            has_relation_value_provenance(
                source_attribute,
                seen | {attribute},
            )
            for source_attribute in source_attributes
        )

    if solver_input_attrs and not any(
        has_relation_value_provenance(attribute)
        for attribute in solver_input_attrs
    ):
        raise _implementation_failure(
            "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
            "closed-form solver input has no declared relation/view value "
            "provenance",
            details=_candidate_ownership("__init__"),
        )

    cached_solver_attrs: set[str] = set()
    for method_name in precompute_reachable:
        method = methods[method_name]
        solver_names: set[str] = set()
        for node in ast.walk(method):
            value = getattr(node, "value", None)
            targets = (
                tuple(node.targets)
                if isinstance(node, ast.Assign)
                else (node.target,)
                if isinstance(node, ast.AnnAssign)
                else ()
            )
            if not isinstance(value, ast.AST):
                continue
            calls_solver = any(
                isinstance(child, ast.Call)
                and (
                    (
                        isinstance(child.func, ast.Attribute)
                        and isinstance(child.func.value, ast.Name)
                        and child.func.value.id == "self"
                        and child.func.attr in solver_methods
                    )
                    or dotted_name(child.func).lower().endswith(
                        _CLOSED_FORM_SOLVER_CALL_SUFFIXES
                    )
                )
                for child in ast.walk(value)
            )
            uses_solver_value = any(
                isinstance(child, ast.Name) and child.id in solver_names
                for child in ast.walk(value)
            )
            if not (calls_solver or uses_solver_value):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    solver_names.add(target.id)
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    cached_solver_attrs.add(target.attr)
    effectful_consumers: set[str] = set()
    for method_name, method in methods.items():
        direct_cached_call = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in cached_solver_attrs
            for node in ast.walk(method)
        )
        returned_cached_value = any(
            isinstance(node, ast.Return)
            and node.value is not None
            and any(
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and child.attr in cached_solver_attrs
                for child in ast.walk(node.value)
            )
            for node in ast.walk(method)
        )
        if direct_cached_call or returned_cached_value:
            effectful_consumers.add(method_name)
    return solver_methods, cached_solver_attrs, effectful_consumers


def _validate_declared_slot_execution_paths(
    tree: ast.Module,
    *,
    primitive_ids: tuple[str, ...],
    component_specs: Mapping[str, Any] | None,
    precompute_required: bool = False,
    execution_config: Mapping[str, Any] | None = None,
) -> None:
    """Check slot semantics on the actual loss and evaluation call paths."""

    if not isinstance(component_specs, Mapping):
        return
    components = tuple(
        item for item in component_specs.values() if isinstance(item, Mapping)
    )
    declared_primitives = {str(item) for item in primitive_ids}
    slots = {
        str(item.get("slot_id"))
        for item in components
        if isinstance(item.get("slot_id"), str)
    }
    methods, calls = _fresh_candidate_execution_graph(tree)
    initialization_methods = tuple(
        name for name in _INITIALIZATION_METHOD_NAMES if name in methods
    )
    roots = ("calculate_loss", "predict", "full_sort_predict")
    present_roots = tuple(root for root in roots if root in methods)
    if not present_roots:
        return
    if len(present_roots) != len(roots):
        missing = tuple(root for root in roots if root not in methods)
        raise _implementation_failure(
            "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
            "compiled model is missing required execution path(s): "
            + ", ".join(missing),
            details=_candidate_ownership(*missing),
        )

    def dotted_name(value: ast.AST) -> str:
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            prefix = dotted_name(value.value)
            return f"{prefix}.{value.attr}" if prefix else value.attr
        return ""

    def reachable_from(root: str) -> set[str]:
        reachable = {root}
        pending = [root]
        while pending:
            caller = pending.pop()
            for callee in calls.get(caller, ()):
                if callee in methods and callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
        return reachable

    denoising_weights = {
        float(component["parameters"]["weight"])
        for component in components
        if component.get("slot_id") == "DENOISING_LONG_TAIL"
        and isinstance(component.get("parameters"), Mapping)
        and isinstance(component["parameters"].get("weight"), (int, float))
        and not isinstance(component["parameters"].get("weight"), bool)
    }
    if len(denoising_weights) == 1:
        expected_weight = next(iter(denoising_weights))
        expected_config_keys = {
            str(key)
            for key, value in (
                execution_config.items()
                if isinstance(execution_config, Mapping)
                else ()
            )
            if isinstance(key, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) == expected_weight
        }
        if len(expected_config_keys) > 1:
            raise _implementation_failure(
                "COMPILED_MECHANISM_CONFIG_BINDING_AMBIGUOUS",
                "multiple execution-config keys match the compiled component weight",
                details=_candidate_ownership("__init__", "calculate_loss"),
            )
        loss_methods = reachable_from("calculate_loss")

        def is_weight_source(node: ast.AST) -> bool:
            return any(
                (
                    isinstance(child, ast.Constant)
                    and isinstance(child.value, (int, float))
                    and not isinstance(child.value, bool)
                    and float(child.value) == expected_weight
                )
                or (
                    isinstance(child, ast.Constant)
                    and child.value in expected_config_keys
                )
                for child in ast.walk(node)
            )

        weight_bindings: set[tuple[str, str]] = set()
        for method_name in loss_methods | set(initialization_methods):
            method = methods.get(method_name)
            if method is None:
                continue
            for node in ast.walk(method):
                value = getattr(node, "value", None)
                targets = (
                    tuple(node.targets)
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                    if isinstance(node, ast.AnnAssign)
                    else ()
                )
                if not isinstance(value, ast.AST) or not is_weight_source(value):
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        weight_bindings.add(("name", target.id))
                    elif (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        weight_bindings.add(("self", target.attr))

        def uses_weight_binding(node: ast.AST) -> bool:
            return is_weight_source(node) or any(
                ("name", child.id) in weight_bindings
                if isinstance(child, ast.Name)
                else (
                    ("self", child.attr) in weight_bindings
                    if isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"
                    else False
                )
                for child in ast.walk(node)
            )

        has_weighted_loss_term = any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and (uses_weight_binding(node.left) or uses_weight_binding(node.right))
            for method_name in loss_methods
            for node in ast.walk(methods[method_name])
        )
        if not has_weighted_loss_term:
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "declared DENOISING_LONG_TAIL weight is not bound to an "
                "executable loss term",
                details=_candidate_ownership("calculate_loss"),
            )

    if precompute_required and any(
        component.get("primitive_id") == "efficiency.closed_form_solver"
        for component in components
    ):
        (
            solver_methods,
            cached_solver_attrs,
            solver_consumer_methods,
        ) = _closed_form_solver_facts(
            methods,
            dotted_name=dotted_name,
            reachable_from=reachable_from,
        )
        solver_reachable = set().union(
            *(reachable_from(root) for root in (*roots, *initialization_methods))
        )
        if not solver_methods.intersection(solver_reachable):
            direct_execution_methods = (
                set.intersection(
                    *(set(calls.get(root, ())) for root in roots)
                )
                & set(methods)
            )
            execution_owners = tuple(sorted(direct_execution_methods)) or roots
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "declared closed-form solver has no reachable solve, inverse, "
                "factorization, or iterative solver computation",
                details=_candidate_ownership("__init__", *execution_owners),
            )
        execution_reachable = {
            root: reachable_from(root) for root in roots
        }
        recomputing_roots = tuple(
            root
            for root, root_reachable in execution_reachable.items()
            if solver_methods.intersection(root_reachable)
        )
        if recomputing_roots:
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "precompute-required closed-form solver is recomputed on a "
                "training or scoring path instead of reusing cached state",
                details=_candidate_ownership("__init__", *recomputing_roots),
            )

        if not cached_solver_attrs:
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "precompute-required closed-form solver does not materialize "
                "its result into cached model state",
                details=_candidate_ownership("__init__"),
            )
        for root, reachable in execution_reachable.items():
            if not solver_consumer_methods.intersection(reachable):
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                    "precompute-required closed-form solver cache is not reused "
                    f"by {root}",
                    details=_candidate_ownership(root),
                )

    reachable = {root: reachable_from(root) for root in roots}

    def operation_families(method_names: set[str]) -> set[str]:
        families: set[str] = set()
        for method_name in method_names:
            method = methods[method_name]
            for node in ast.walk(method):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                    families.add("DOT")
                if not isinstance(node, ast.Call):
                    continue
                name = dotted_name(node.func).lower()
                leaf = name.rsplit(".", 1)[-1]
                if name.endswith((".normalize", ".cosine_similarity")):
                    families.add("NORMALIZED")
                if any(
                    token in name
                    for token in ("cdist", "distance", "dist", "poincare", "hyperbol")
                ):
                    families.add("DISTANCE")
                # Sparse/dense ``mm`` is also the ordinary propagation ABI and
                # is not, by itself, evidence that the declared scorer reaches
                # this root.  Keep only unambiguous score operations here.
                if name.endswith((".matmul", ".bmm", ".einsum", ".dot")):
                    families.add("DOT")
                if leaf in {"sum", "mean"} and any(
                    (
                        isinstance(item, ast.BinOp)
                        and isinstance(item.op, ast.Mult)
                    )
                    or (
                        isinstance(item, ast.Call)
                        and dotted_name(item.func).lower().rsplit(".", 1)[-1]
                        in {"mul", "multiply"}
                    )
                    for item in ast.walk(node)
                ):
                    families.add("DOT")
        return families

    score_components = tuple(
        item
        for item in components
        if item.get("slot_id") == "SCORE_HEAD"
        and str(item.get("primitive_id")) in declared_primitives
    )
    if score_components:
        score_primitives = {
            str(item.get("primitive_id", "")).lower() for item in score_components
        }
        normalized_score = any(
            token in primitive
            for primitive in score_primitives
            for token in ("cosine", "normalized_dot")
        )
        distance_score = any(
            token in primitive
            for primitive in score_primitives
            for token in ("distance", "hyperbol", "poincare")
        )
        bilinear_score = any(
            "bilinear" in primitive for primitive in score_primitives
        )
        dot_score = normalized_score or any(
            primitive.endswith((".dot_product", ".dot"))
            for primitive in score_primitives
        )
        families_by_root = {
            root: operation_families(reachable[root]) for root in roots
        }
        expected_family = "DISTANCE" if distance_score else "DOT" if dot_score else None
        family_mismatch_roots = tuple(
            root
            for root, families in families_by_root.items()
            if expected_family is not None and expected_family not in families
        )
        if family_mismatch_roots:
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "declared score/representation semantics are not shared by loss, predict, and full-sort paths",
                details=_candidate_ownership(*family_mismatch_roots),
            )

        if bilinear_score:
            parameterized_score_attributes: set[str] = set()
            for initializer_name in initialization_methods:
                initializer = methods[initializer_name]
                for node in ast.walk(initializer):
                    target: ast.AST | None = None
                    value: ast.AST | None = None
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        target, value = node.targets[0], node.value
                    elif isinstance(node, ast.AnnAssign):
                        target, value = node.target, node.value
                    if not (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and isinstance(value, ast.Call)
                    ):
                        continue
                    constructor = dotted_name(value.func).lower()
                    if constructor.endswith(
                        (".bilinear", ".linear", ".parameter")
                    ):
                        parameterized_score_attributes.add(target.attr)

            def has_bilinear_transform(method_names: set[str]) -> bool:
                return any(
                    (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                        and node.func.attr in parameterized_score_attributes
                    )
                    or (
                        isinstance(node, ast.BinOp)
                        and isinstance(node.op, ast.MatMult)
                        and any(
                            isinstance(item, ast.Attribute)
                            and isinstance(item.value, ast.Name)
                            and item.value.id == "self"
                            and item.attr in parameterized_score_attributes
                            for item in ast.walk(node)
                        )
                    )
                    for method_name in method_names
                    for node in ast.walk(methods[method_name])
                )

            missing_bilinear_roots = tuple(
                root
                for root in roots
                if not has_bilinear_transform(reachable[root])
            )
            if missing_bilinear_roots:
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                    "declared bilinear score head lacks a parameterized transform on every score path",
                    details=_candidate_ownership(*missing_bilinear_roots),
                )
        missing_normalized_roots = tuple(
            root
            for root, families in families_by_root.items()
            if normalized_score and "NORMALIZED" not in families
        )
        if missing_normalized_roots:
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "declared normalized score head is not applied on every score path",
                details=_candidate_ownership(*missing_normalized_roots),
            )
        missing_distance_roots = tuple(
            root
            for root, families in families_by_root.items()
            if distance_score and "DISTANCE" not in families
        )
        if missing_distance_roots:
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "declared distance score head is not applied on every score path",
                details=_candidate_ownership(*missing_distance_roots),
            )

    epoch_sampler_specs = tuple(
        item
        for item in components
        if item.get("slot_id") == "NEGATIVE_SAMPLER"
        and str(item.get("primitive_id")) in declared_primitives
        and isinstance(item.get("parameters"), Mapping)
        and item["parameters"].get("refresh_frequency") == "EPOCH"
    )

    def has_compiler_owned_uniform_sampler_binding() -> bool:
        if not epoch_sampler_specs or any(
            item.get("primitive_id") != "sampler.uniform"
            for item in epoch_sampler_specs
        ):
            return False
        exact_import = any(
            isinstance(node, ast.ImportFrom)
            and node.module
            == "recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold"
            and any(
                alias.name == "bind_epoch_sampler_model_class"
                and alias.asname == "_recclaw_bind_epoch_sampler_model_class"
                for alias in node.names
            )
            for node in tree.body
        )
        exact_binding = any(
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "FreshCandidateModel"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_recclaw_bind_epoch_sampler_model_class"
            and len(node.value.args) == 2
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "FreshCandidateModel"
            and isinstance(node.value.args[1], ast.Name)
            and node.value.args[1].id == "RECCLAW_IMPLEMENTED_COMPONENT_SPECS"
            and not node.value.keywords
            for node in tree.body
        )
        return exact_import and exact_binding

    if (
        epoch_sampler_specs
        and "recclaw_sampler_step" not in reachable["calculate_loss"]
        and not has_compiler_owned_uniform_sampler_binding()
    ):
        raise _implementation_failure(
            "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
            "declared negative sampler does not feed the loss path",
            details=_candidate_ownership("calculate_loss"),
        )
    if "PRIMARY_OBJECTIVE" in slots and "calculate_loss" not in methods:
        raise _implementation_failure(
            "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
            "declared primary objective does not feed the loss path",
            details=_candidate_ownership("calculate_loss"),
        )


def _validate_compiled_mechanism_behavior(
    tree: ast.Module,
    *,
    primitive_ids: tuple[str, ...],
    component_specs: Mapping[str, Any] | None = None,
    precompute_required: bool = False,
    execution_config: Mapping[str, Any] | None = None,
) -> None:
    """Apply the one family-neutral mechanism truth boundary."""

    specs = component_specs if isinstance(component_specs, Mapping) else {}
    geometry_declared = any(
        isinstance(spec, Mapping)
        and spec.get("slot_id") == "GEOMETRY_REGULARIZATION"
        for spec in specs.values()
    )
    objective_weights = {
        float(spec["parameters"]["weight"])
        for spec in specs.values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "PRIMARY_OBJECTIVE"
        and isinstance(spec.get("parameters"), Mapping)
        and isinstance(spec["parameters"].get("weight"), (int, float))
        and not isinstance(spec["parameters"].get("weight"), bool)
    }
    candidate_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FreshCandidateModel"
        ),
        None,
    )
    if objective_weights and not geometry_declared and candidate_class is not None:
        for node in ast.walk(candidate_class):
            targets = (
                tuple(node.targets)
                if isinstance(node, ast.Assign)
                else (node.target,)
                if isinstance(node, ast.AnnAssign)
                else ()
            )
            if not any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "reg_weight"
                for target in targets
            ):
                continue
            value = getattr(node, "value", None)
            if value is None:
                continue
            literals = {
                float(item.value)
                for item in ast.walk(value)
                if isinstance(item, ast.Constant)
                and isinstance(item.value, (int, float))
                and not isinstance(item.value, bool)
            }
            objective_weight_subscript = any(
                isinstance(item, ast.Subscript)
                and isinstance(item.slice, ast.Constant)
                and item.slice.value == "weight"
                for item in ast.walk(value)
            )
            if literals & objective_weights or objective_weight_subscript:
                raise _implementation_failure(
                    "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                    "primary objective weight is routed into reg_weight without "
                    "a GEOMETRY_REGULARIZATION component",
                )
    _validate_declared_mechanism_observability(tree, component_specs)
    _validate_declared_slot_execution_paths(
        tree,
        primitive_ids=primitive_ids,
        component_specs=component_specs,
        precompute_required=precompute_required,
        execution_config=execution_config,
    )


def validate_mechanism_source_behavior(
    source: str,
    *,
    primitive_ids: tuple[str, ...],
    component_specs: Mapping[str, Any] | None = None,
    precompute_required: bool = False,
    execution_config: Mapping[str, Any] | None = None,
) -> None:
    """Recheck the compiled behavior claims against immutable executed source."""

    try:
        tree = ast.parse(source, filename="executed-entrypoint.py")
    except SyntaxError as error:
        raise _implementation_failure(
            "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
            "executed entrypoint source is not valid Python",
        ) from error
    _validate_compiled_mechanism_behavior(
        tree,
        primitive_ids=primitive_ids,
        component_specs=component_specs,
        precompute_required=precompute_required,
        execution_config=execution_config,
    )


def _validate_source_compiled_binding(
    response: Mapping[str, Any],
    *,
    entrypoint: str,
    compiled_mechanism: Mapping[str, Any] | None,
    execution_config: Mapping[str, Any] | None = None,
) -> None:
    normalized = _normalized_compiled_mechanism(compiled_mechanism)
    if normalized is None:
        return
    entrypoint_path = entrypoint.split(":", 1)[0].replace(".", "/") + ".py"
    source = next(
        (
            item["content"]
            for item in response["files"]
            if item["path"] == entrypoint_path
        ),
        None,
    )
    if not isinstance(source, str):
        raise _implementation_failure(
            "COMPILED_MECHANISM_SOURCE_BINDING_MISSING",
            "entrypoint source is missing the compiler binding",
        )
    try:
        tree = ast.parse(source, filename=entrypoint_path)
    except SyntaxError as error:
        raise _implementation_failure(
            "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
            "entrypoint source is not valid Python",
        ) from error
    constants: dict[str, Any] = {}
    for node in tree.body:
        name: str | None = None
        expression: ast.expr | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            expression = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            expression = node.value
        if name is None or expression is None:
            continue
        try:
            constants[name] = ast.literal_eval(expression)
        except (ValueError, TypeError):
            continue
    for constant_name, field_name in _SOURCE_BINDING_CONSTANTS.items():
        if constants.get(constant_name) != normalized[field_name]:
            raise _implementation_failure(
                "COMPILED_MECHANISM_SOURCE_BINDING_MISMATCH",
                f"entrypoint source must bind exact {constant_name}",
            )
    binding = normalized["implementation_binding"]
    for constant_name, field_name in _SOURCE_BINDING_SEQUENCE_CONSTANTS.items():
        observed = constants.get(constant_name)
        if (
            not isinstance(observed, (tuple, list))
            or any(not isinstance(item, str) for item in observed)
            or sorted(observed) != sorted(binding[field_name])
        ):
            raise _implementation_failure(
                "COMPILED_MECHANISM_SOURCE_BINDING_MISMATCH",
                f"entrypoint source must bind exact {constant_name}",
            )
    for constant_name, field_name in _SOURCE_BINDING_MAPPING_CONSTANTS.items():
        observed = constants.get(constant_name)
        if not isinstance(observed, Mapping) or canonical_value(observed) != canonical_value(
            binding[field_name]
        ):
            raise _implementation_failure(
                "COMPILED_MECHANISM_SOURCE_BINDING_MISMATCH",
                f"entrypoint source must bind exact {constant_name}",
            )
    _validate_compiled_mechanism_behavior(
        tree,
        primitive_ids=tuple(binding["primitive_ids"]),
        component_specs=binding["component_specs"],
        precompute_required=(
            normalized.get("mechanism_program", {})
            .get("program_payload", {})
            .get("estimated_cost", {})
            .get("precompute_required")
            is True
        ),
        execution_config=execution_config,
    )


def _validated_entrypoint(value: Any, *, allowed_files: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value.count(":") != 1:
        raise _implementation_failure(
            "INVALID_ENTRYPOINT",
            "entrypoint must use module.path:ClassName",
        )
    module_name, class_name = value.split(":", 1)
    if (
        not module_name.startswith("recclaw_ext.")
        or not class_name.isidentifier()
        or any(not part.isidentifier() for part in module_name.split("."))
    ):
        raise _implementation_failure(
            "INVALID_ENTRYPOINT",
            "entrypoint must identify a candidate-local recclaw_ext class",
        )
    source_path = module_name.replace(".", "/") + ".py"
    if source_path not in allowed_files:
        raise _implementation_failure(
            "ENTRYPOINT_OUTSIDE_ALLOWLIST",
            "entrypoint source is outside the exact write allowlist",
        )
    return value


def bind_normalized_embedding_model_class(
    model_class: type,
    component_specs: Mapping[str, Any],
) -> type:
    """Bind declared USER/ITEM identity tables to the canonical L2 view."""

    roles = {
        source.get("data_role")
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "EMBEDDING"
        and spec.get("primitive_id") == "embedding.normalized"
        for port in spec.get("inputs", ())
        if isinstance(port, Mapping)
        for source in (port.get("source"),)
        if isinstance(source, Mapping) and source.get("kind") == "DATA"
    }
    attributes = tuple(
        name
        for role, name in (
            ("USER_ID", "user_embedding"),
            ("ITEM_ID", "item_embedding"),
        )
        if role in roles
    )
    if not attributes:
        return model_class
    if (
        getattr(model_class, "_recclaw_normalized_embedding_attributes", ())
        == attributes
    ):
        return model_class

    import torch
    from torch.nn.utils import parametrize

    class _L2Weight(torch.nn.Module):
        def forward(self, weight: Any) -> Any:
            return torch.nn.functional.normalize(weight, p=2.0, dim=-1)

    class _NormalizedEmbeddingModel(model_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for attribute in attributes:
                embedding = getattr(self, attribute)
                if not isinstance(embedding, torch.nn.Embedding):
                    raise TypeError(
                        f"{attribute} must be torch.nn.Embedding for embedding.normalized"
                    )
                if getattr(embedding, "_recclaw_l2_weight_bound", False):
                    continue
                parametrize.register_parametrization(
                    embedding,
                    "weight",
                    _L2Weight(),
                )
                object.__setattr__(embedding, "_recclaw_l2_weight_bound", True)

    _NormalizedEmbeddingModel.__name__ = model_class.__name__
    _NormalizedEmbeddingModel.__qualname__ = model_class.__qualname__
    _NormalizedEmbeddingModel._recclaw_normalized_embedding_attributes = attributes
    return _NormalizedEmbeddingModel


def _normalize_canonical_identity_cardinality(
    class_node: ast.ClassDef,
    component_specs: Mapping[str, Any],
) -> None:
    """Give the compiler ownership of RecBole identity-table cardinality.

    ``user_num`` and ``item_num`` already include RecBole's padding identity.
    When an Implementer sizes a declared canonical USER/ITEM table with an
    off-by-one expression, that same expression commonly becomes the model's
    ID-domain bound. Normalize that mechanical expression once while leaving
    non-canonical mechanism embeddings and all initialization choices intact.
    """

    declared_roles = {
        str(source.get("data_role"))
        for spec in component_specs.values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "EMBEDDING"
        and spec.get("primitive_id") == "embedding.independent_user_item"
        and spec.get("custom_component_id") is None
        for port in spec.get("inputs", ())
        if isinstance(port, Mapping)
        for source in (port.get("source"),)
        if isinstance(source, Mapping) and source.get("kind") == "DATA"
    }
    field_roles = {
        field_name: role
        for role, field_name in (
            ("USER_ID", "user_embedding"),
            ("ITEM_ID", "item_embedding"),
        )
        if role in declared_roles
    }
    if not field_roles:
        return

    initialization_methods = tuple(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _INITIALIZATION_METHOD_NAMES
    )
    if not initialization_methods:
        return

    def self_field(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        return None

    def embedding_call(node: ast.AST) -> ast.Call | None:
        if not isinstance(node, ast.Call):
            return None
        if (
            isinstance(node.func, ast.Name) and node.func.id == "Embedding"
        ) or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "Embedding"
        ):
            return node
        return None

    def set_cardinality_argument(call: ast.Call, value: ast.expr) -> None:
        if call.args:
            call.args[0] = value
            return
        for keyword in call.keywords:
            if keyword.arg == "num_embeddings":
                keyword.value = value
                return

    def exact_cardinality(role: str) -> ast.Attribute:
        return ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="n_users" if role == "USER_ID" else "n_items",
            ctx=ast.Load(),
        )

    def simple_assignment(node: ast.AST) -> tuple[str | None, ast.Call | None]:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            return self_field(node.targets[0]), embedding_call(node.value)
        if isinstance(node, ast.AnnAssign):
            return self_field(node.target), embedding_call(node.value)
        return None, None

    dataset_parameters = {
        method.args.args[2].arg
        for method in initialization_methods
        if len(method.args.args) >= 3
    }

    def cardinality_role(node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            owner = node.value.id if isinstance(node.value, ast.Name) else None
            if owner == "self":
                role = {
                    "n_users": "USER_ID",
                    "n_items": "ITEM_ID",
                }.get(node.attr)
                return role if role in declared_roles else None
            if owner in dataset_parameters:
                role = {
                    "n_users": "USER_ID",
                    "user_num": "USER_ID",
                    "n_items": "ITEM_ID",
                    "item_num": "ITEM_ID",
                }.get(node.attr)
                return role if role in declared_roles else None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and not node.keywords
        ):
            return cardinality_role(node.args[0])
        return None

    def off_by_one_role(node: ast.AST) -> str | None:
        if not isinstance(node, ast.BinOp):
            return None
        if (
            isinstance(node.op, (ast.Add, ast.Sub))
            and cardinality_role(node.left) is not None
            and isinstance(node.right, ast.Constant)
            and node.right.value == 1
        ):
            return cardinality_role(node.left)
        if (
            isinstance(node.op, ast.Add)
            and isinstance(node.left, ast.Constant)
            and node.left.value == 1
            and cardinality_role(node.right) is not None
        ):
            return cardinality_role(node.right)
        return None

    def normalized_bound(node: ast.expr) -> ast.expr:
        role = off_by_one_role(node)
        if role is None:
            return node
        return ast.copy_location(exact_cardinality(role), node)

    def normalized_shape(node: ast.expr) -> ast.expr:
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            node.elts = [normalized_shape(item) for item in node.elts]
            return node
        return normalized_bound(node)

    class _CanonicalIdentityCardinality(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            if embedding_call(node) is not None:
                # Canonical tables are normalized by their assignment below;
                # all other Embedding calls remain mechanism-owned.
                return node
            visited = self.generic_visit(node)
            if not isinstance(visited, ast.Call):
                return visited
            function_name = (
                visited.func.id
                if isinstance(visited.func, ast.Name)
                else visited.func.attr
                if isinstance(visited.func, ast.Attribute)
                else None
            )
            if function_name in {"range", "arange"} and visited.args:
                stop_index = 0 if len(visited.args) == 1 else 1
                visited.args[stop_index] = normalized_bound(
                    visited.args[stop_index]
                )
            shape_functions = {
                "empty",
                "eye",
                "identity",
                "new_empty",
                "new_ones",
                "new_zeros",
                "ones",
                "rand",
                "randn",
                "zeros",
            }
            if function_name in shape_functions:
                visited.args = [normalized_shape(item) for item in visited.args]
            elif function_name in {"full", "new_full"} and visited.args:
                visited.args[0] = normalized_shape(visited.args[0])
            elif function_name == "sparse_coo_tensor" and len(visited.args) >= 3:
                visited.args[2] = normalized_shape(visited.args[2])
            for keyword in visited.keywords:
                if keyword.arg in {"shape", "size"}:
                    keyword.value = normalized_shape(keyword.value)
            return visited

        def visit_Slice(self, node: ast.Slice) -> ast.AST:
            visited = self.generic_visit(node)
            if isinstance(visited, ast.Slice) and visited.upper is not None:
                visited.upper = normalized_bound(visited.upper)
            return visited

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            field_name, call = simple_assignment(node)
            if call is not None and field_name not in field_roles:
                # Mechanism-specific tables may intentionally own an extra
                # sentinel or prototype row; they are not RecBole ID tables.
                return node
            visited = self.generic_visit(node)
            if isinstance(visited, ast.Assign):
                field_name, call = simple_assignment(visited)
                role = field_roles.get(field_name or "")
                if role is not None and call is not None:
                    set_cardinality_argument(call, exact_cardinality(role))
            return visited

        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
            field_name, call = simple_assignment(node)
            if call is not None and field_name not in field_roles:
                return node
            visited = self.generic_visit(node)
            if isinstance(visited, ast.AnnAssign):
                field_name, call = simple_assignment(visited)
                role = field_roles.get(field_name or "")
                if role is not None and call is not None:
                    set_cardinality_argument(call, exact_cardinality(role))
            return visited

    _CanonicalIdentityCardinality().visit(class_node)


_MODEL_PATCH_SIGNATURES = {
    "calculate_loss": "def calculate_loss(self, interaction):\n    pass\n",
    "predict": "def predict(self, interaction):\n    pass\n",
    "full_sort_predict": "def full_sort_predict(self, interaction):\n    pass\n",
}
_BL_ICF_MODEL_PATCH_SIGNATURES = {
    "recclaw_primary_objective": (
        "def recclaw_primary_objective(self, interaction, representations):\n    pass\n"
    ),
    "recclaw_initialize_primary_objective": (
        "def recclaw_initialize_primary_objective(self, config, dataset):\n    pass\n"
    ),
    "recclaw_encode_representation": (
        "def recclaw_encode_representation(self, representation):\n"
        "    pass\n"
    ),
    "recclaw_propagation_aggregation": (
        "def recclaw_propagation_aggregation("
        "self, message, prior, relation, layer_index):\n"
        "    pass\n"
    ),
}
_TRAINER_PATCH_SIGNATURES = {
    "_build_optimizer": "def _build_optimizer(self, **kwargs):\n    pass\n",
    "_train_epoch": (
        "def _train_epoch(self, train_data, epoch_idx, loss_func=None, "
        "show_progress=False):\n"
        "    pass\n"
    ),
}
_SEMANTIC_ID_STAGE_TRAINER_SIGNATURES = {
    **_TRAINER_PATCH_SIGNATURES,
    "fit": (
        "def fit(self, train_data, valid_data=None, verbose=True, saved=True, "
        "show_progress=False, callback_fn=None):\n    pass\n"
    ),
}


def _parent_method_model_signatures(
    compiled_mechanism: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return the stable model methods owned by one exact-parent patch."""

    signatures = dict(_MODEL_PATCH_SIGNATURES)
    search_space_id = (
        compiled_mechanism["space_identity"].get("search_space_id")
        if isinstance(compiled_mechanism, Mapping)
        and isinstance(compiled_mechanism.get("space_identity"), Mapping)
        else None
    )
    if search_space_id == "BL_ICF_MECHANISM_SPACE_V1":
        signatures.update(_BL_ICF_MODEL_PATCH_SIGNATURES)
    if (
        search_space_id == "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1"
    ):
        signatures["forward"] = (
            "def forward(self, item_seq, item_seq_len):\n    pass\n"
        )
    return signatures


def _bind_semantic_id_trainer_stage(
    parent_source: str, compiled: Mapping[str, Any],
) -> str:
    stage_specs = [
        spec
        for spec in compiled["implementation_binding"]["component_specs"].values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "OPTIMIZATION_STAGING"
    ]
    removed_staging = "OPTIMIZATION_STAGING" in compiled["mechanism_program"][
        "program_payload"
    ].get("removed_slots", ())
    if not stage_specs and not removed_staging:
        return parent_source
    if len(stage_specs) > 1:
        raise _implementation_failure(
            "PARENT_METHOD_PATCH_ABI_DRIFT",
            "compiler-owned optimization staging must resolve to one typed spec",
        )
    stage_spec = canonical_value(
        {
            "parameters": stage_specs[0]["parameters"],
            "primitive_id": stage_specs[0]["primitive_id"],
        }
    ) if stage_specs else None
    try:
        trainer_tree = ast.parse(
            parent_source,
            filename="recclaw_ext/trainer.py",
        )
    except SyntaxError as error:
        raise _implementation_failure(
            "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
            "parent trainer source is not valid Python",
        ) from error
    assignment = ast.Assign(
        targets=[
            ast.Name(
                id="RECCLAW_OPTIMIZATION_STAGE_SPEC",
                ctx=ast.Store(),
            )
        ],
        value=ast.parse(repr(stage_spec), mode="eval").body,
    )
    for index, node in enumerate(trainer_tree.body):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "RECCLAW_OPTIMIZATION_STAGE_SPEC"
        ):
            trainer_tree.body[index] = ast.copy_location(assignment, node)
            break
    else:
        insert_at = next(
            (
                index
                for index, node in enumerate(trainer_tree.body)
                if isinstance(node, ast.ClassDef)
            ),
            len(trainer_tree.body),
        )
        trainer_tree.body.insert(insert_at, assignment)
    ast.fix_missing_locations(trainer_tree)
    return ast.unparse(trainer_tree) + "\n"


def _expand_exact_parent_method_patch(
    response: Mapping[str, Any],
    *,
    exact_parent_bundle: Mapping[str, Any],
    compiled_mechanism: Mapping[str, Any] | None = None,
    allow_mechanism_helpers: bool = False,
    lock_trainer_lifecycle: bool = False,
    model_signatures: Mapping[str, str] | None = None,
    trainer_signatures: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge explicit research methods into exact parent-owned source files."""

    compiled = _normalized_compiled_mechanism(compiled_mechanism)

    if not isinstance(response, Mapping) or set(response) != {
        "entrypoint",
        "files",
        "implementation_summary",
    }:
        raise _implementation_failure(
            "PARENT_METHOD_PATCH_FIELDS_INVALID",
            "parent method patch fields do not match the shared boundary",
        )
    if response.get("entrypoint") != "recclaw_ext.candidate:FreshCandidateModel":
        raise _implementation_failure(
            "MACHINE_OWNED_ENTRYPOINT_DRIFT",
            "parent method patches use the exact parent model entrypoint",
        )
    patch_files = response.get("files")
    if not isinstance(patch_files, list) or not patch_files:
        raise _implementation_failure(
            "PARENT_METHOD_PATCH_EMPTY",
            "parent method patch must contain model or trainer methods",
        )
    allowed = {
        "recclaw_ext/candidate.py": (
            "FreshCandidateModel",
            (
                dict(model_signatures)
                if model_signatures is not None
                else _parent_method_model_signatures(compiled)
            ),
        ),
        "recclaw_ext/trainer.py": (
            "FreshCandidateTrainer",
            (
                dict(trainer_signatures) if trainer_signatures is not None
                else {} if lock_trainer_lifecycle else _TRAINER_PATCH_SIGNATURES
            ),
        ),
    }
    parent = _normalized_exact_parent_bundle(exact_parent_bundle)
    assert parent is not None
    parent_files = {
        str(item["path"]): str(item["content"]) for item in parent["files"]
    }
    if not set(allowed).issubset(parent_files):
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent bundle lacks the stable model or trainer source",
        )

    patches: dict[str, list[str]] = {}
    package_marker_seen = False
    for item in patch_files:
        if not isinstance(item, Mapping) or set(item) != {"content", "path"}:
            raise _implementation_failure(
                "PARENT_METHOD_PATCH_FILE_INVALID",
                "method patch files require only path and content",
            )
        path = item.get("path")
        content = item.get("content")
        if path == "recclaw_ext/__init__.py":
            if (
                package_marker_seen
                or not isinstance(content, str)
                or content != parent_files.get(path)
            ):
                raise _implementation_failure(
                    "PARENT_METHOD_PATCH_FILE_INVALID",
                    "package marker must exactly preserve the parent package bytes",
                )
            package_marker_seen = True
            continue
        if (
            path not in allowed
            or not isinstance(content, str)
            or not content
        ):
            raise _implementation_failure(
                "PARENT_METHOD_PATCH_FILE_INVALID",
                "method patch path and content must match the shared boundary",
            )
        patches.setdefault(str(path), []).append(content)

    training_primitives = () if compiled is None else tuple(
        primitive_id
        for primitive_id in compiled["implementation_binding"]["primitive_ids"]
        if primitive_id.startswith("training.")
    )
    if not training_primitives and not lock_trainer_lifecycle and trainer_signatures is None:
        patches.pop("recclaw_ext/trainer.py", None)
    if not patches:
        raise _implementation_failure(
            "PARENT_METHOD_PATCH_EMPTY",
            "parent method patch contains no applicable model or trainer methods",
        )

    def canonical_method(
        method: ast.FunctionDef,
        signatures: Mapping[str, str],
    ) -> ast.FunctionDef:
        if method.decorator_list:
            raise _implementation_failure(
                "PARENT_METHOD_PATCH_ABI_DRIFT",
                "method patch decorators cannot replace parent-owned ABI",
            )
        signature = signatures.get(method.name)
        if signature is None:
            return method
        fixed = ast.parse(signature).body[0]
        assert isinstance(fixed, ast.FunctionDef)
        original_positional = method.args.posonlyargs + method.args.args
        fixed_positional = fixed.args.posonlyargs + fixed.args.args
        if (
            len(original_positional) != len(fixed_positional)
            or len(method.args.kwonlyargs) != len(fixed.args.kwonlyargs)
            or (method.args.vararg is None) != (fixed.args.vararg is None)
            or (method.args.kwarg is None) != (fixed.args.kwarg is None)
        ):
            raise _implementation_failure(
                "PARENT_METHOD_PATCH_ABI_DRIFT",
                f"method patch signature does not match {method.name}",
            )
        argument_renames = {
            original.arg: normalized.arg
            for original, normalized in zip(
                original_positional,
                fixed_positional,
                strict=False,
            )
            if original.arg != normalized.arg
        }

        class _RenameCanonicalArguments(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                replacement = argument_renames.get(node.id)
                if replacement is None:
                    return node
                return ast.copy_location(
                    ast.Name(id=replacement, ctx=node.ctx),
                    node,
                )

        renamer = _RenameCanonicalArguments()
        fixed.body = [renamer.visit(node) for node in method.body]
        fixed.type_comment = method.type_comment
        return fixed

    def merge_fragments(
        parent_source: str,
        patch_sources: list[str],
        *,
        path: str,
        class_name: str,
        signatures: Mapping[str, str],
    ) -> str:
        try:
            parent_tree = ast.parse(parent_source, filename=path)
        except SyntaxError as error:
            raise _implementation_failure(
                "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
                "parent method patch is not valid Python",
            ) from error

        def helper_names(node: ast.AST) -> tuple[str, ...] | None:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return ()
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return ()
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                return (node.name,)
            if isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) for target in node.targets
            ):
                return tuple(target.id for target in node.targets)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                return (node.target.id,)
            return None

        parent_names = {
            node.name
            for node in parent_tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        parent_names.update(
            target.id
            for node in parent_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        )
        parent_class = next(
            (
                node
                for node in parent_tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if parent_class is None:
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                "exact parent source lacks its stable entrypoint class",
            )
        parsed_fragments: list[tuple[str, ast.Module, ast.ClassDef]] = []
        for fragment_index, patch_source in enumerate(patch_sources):
            try:
                patch_tree = ast.parse(
                    patch_source,
                    filename=f"{path}:patch:{fragment_index}",
                )
            except SyntaxError as error:
                raise _implementation_failure(
                    "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
                    "parent method patch is not valid Python",
                ) from error
            patch_classes = [
                node
                for node in patch_tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ]
            if (
                len(patch_classes) != 1
                or patch_classes[0].bases
                or patch_classes[0].keywords
                or patch_classes[0].decorator_list
            ):
                raise _implementation_failure(
                    "PARENT_METHOD_PATCH_ABI_DRIFT",
                    "method patch must contain one unbased parent entrypoint class",
                )
            parsed_fragments.append(
                (
                    ast.dump(patch_tree, include_attributes=False),
                    patch_tree,
                    patch_classes[0],
                )
            )

        # Canonical AST ordering makes the output independent of Provider file order.
        parsed_fragments.sort(key=lambda item: item[0])
        methods: list[ast.FunctionDef] = []
        method_names: set[str] = set()
        helper_nodes: list[ast.stmt] = []
        claimed_helpers: set[str] = set()
        for _, patch_tree, patch_class in parsed_fragments:
            fragment_helpers = [
                node for node in patch_tree.body if node is not patch_class
            ]
            if fragment_helpers and not allow_mechanism_helpers:
                raise _implementation_failure(
                    "PARENT_METHOD_PATCH_ABI_DRIFT",
                    "method patch cannot add module-level source",
                )
            for helper in fragment_helpers:
                names = helper_names(helper)
                if names is None or any(
                    not (
                        name.startswith("recclaw_")
                        or name.startswith("RECCLAW_")
                    )
                    for name in names
                ):
                    raise _implementation_failure(
                        "SCAFFOLDED_MECHANISM_NAMESPACE_DRIFT",
                        "mechanism helpers must stay in the recclaw_ namespace",
                    )
                if parent_names.intersection(names) or claimed_helpers.intersection(
                    names
                ):
                    raise _implementation_failure(
                        "SCAFFOLDED_MECHANISM_NAMESPACE_DRIFT",
                        "mechanism helper collides with a stable parent symbol "
                        "or another mechanism helper",
                    )
                claimed_helpers.update(names)
                helper_nodes.append(helper)
            for node in patch_class.body:
                if not isinstance(node, ast.FunctionDef):
                    raise _implementation_failure(
                        "PARENT_METHOD_PATCH_ABI_DRIFT",
                        "method patch class may contain methods only",
                    )
                if node.name in method_names:
                    raise _implementation_failure(
                        "PARENT_METHOD_PATCH_ABI_DRIFT",
                        "method patch repeats a method name across fragments; "
                        "each stable parent symbol or recclaw_ mechanism symbol "
                        "may be owned once",
                    )
                if (
                    (
                        lock_trainer_lifecycle
                        and path == "recclaw_ext/trainer.py"
                        and node.name in signatures
                    )
                    or (
                        node.name not in signatures
                        and not node.name.startswith("recclaw_")
                    )
                ):
                    raise _implementation_failure(
                        "PARENT_METHOD_PATCH_ABI_DRIFT",
                        "method patch attempted to own a stable parent symbol",
                    )
                if any(
                    isinstance(descendant, ast.Attribute)
                    and descendant.attr.startswith("recclaw_")
                    and isinstance(descendant.value, ast.Call)
                    and isinstance(descendant.value.func, ast.Name)
                    and descendant.value.func.id == "super"
                    for descendant in ast.walk(node)
                ):
                    raise _implementation_failure(
                        "SCAFFOLDED_MECHANISM_SUPER_HELPER_INVALID",
                        "recclaw_ mechanism state cannot be recovered from the "
                        "TIGER/T5 parent through super().recclaw_*",
                    )
                method_names.add(node.name)
                methods.append(canonical_method(node, signatures))
        if not methods:
            raise _implementation_failure(
                "PARENT_METHOD_PATCH_EMPTY",
                "method patch class contains no executable methods",
            )
        by_name = {method.name: method for method in methods}
        merged_body: list[ast.stmt] = []
        replaced: set[str] = set()
        for node in parent_class.body:
            if isinstance(node, ast.FunctionDef) and node.name in by_name:
                merged_body.append(by_name[node.name])
                replaced.add(node.name)
            else:
                merged_body.append(node)
        merged_body.extend(
            method for method in methods if method.name not in replaced
        )
        parent_class.body = merged_body
        parent_tree.body.extend(helper_nodes)
        ast.fix_missing_locations(parent_tree)
        return ast.unparse(parent_tree) + "\n"

    if (lock_trainer_lifecycle or trainer_signatures is not None) and compiled is not None:
        parent_files["recclaw_ext/trainer.py"] = _bind_semantic_id_trainer_stage(
            parent_files["recclaw_ext/trainer.py"], compiled
        )

    for path, patch_sources in patches.items():
        class_name, signatures = allowed[path]
        parent_files[path] = merge_fragments(
            parent_files[path],
            patch_sources,
            path=path,
            class_name=class_name,
            signatures=signatures,
        )
    return canonical_value(
        {
            "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
            "files": [
                {"path": path, "content": content}
                for path, content in sorted(parent_files.items())
            ],
            "implementation_summary": response["implementation_summary"],
        }
    )


def _self_attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    current = node
    names: list[str] = []
    while True:
        if isinstance(current, ast.Subscript):
            current = current.value
            continue
        if isinstance(current, ast.Attribute):
            names.append(current.attr)
            current = current.value
            continue
        if isinstance(current, ast.Name) and current.id == "self":
            return tuple(reversed(names))
        return None


def _parent_owned_state_writes(fragment_class: ast.ClassDef) -> tuple[str, ...]:
    """Find candidate writes outside the reserved recclaw_* extension namespace."""

    parent_owned_writes: set[str] = set()
    for node in ast.walk(fragment_class):
        targets: tuple[ast.AST, ...] = ()
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        elif isinstance(node, (ast.AugAssign, ast.Delete)):
            targets = (
                (node.target,)
                if isinstance(node, ast.AugAssign)
                else tuple(node.targets)
            )
        for target in targets:
            for descendant in ast.walk(target):
                chain = _self_attribute_chain(descendant)
                if chain and not chain[0].startswith("recclaw_"):
                    parent_owned_writes.add(".".join(chain))
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "self"
        ):
            attribute = (
                node.args[1].value
                if isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                else "<dynamic>"
            )
            if not attribute.startswith("recclaw_"):
                parent_owned_writes.add(attribute)
        call_chain = _self_attribute_chain(node.func)
        if (
            call_chain
            and len(call_chain) > 1
            and call_chain[-1].endswith("_")
            and not call_chain[0].startswith("recclaw_")
        ):
            parent_owned_writes.add(".".join(call_chain[:-1]))
        if (
            call_chain
            in {("register_buffer",), ("register_parameter",), ("add_module",)}
            and node.args
        ):
            registered = (
                node.args[0].value
                if isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                else "<dynamic>"
            )
            if not registered.startswith("recclaw_"):
                parent_owned_writes.add(registered)
        if (
            call_chain
            and call_chain[-1] == "load_state_dict"
            and not call_chain[0].startswith("recclaw_")
        ):
            parent_owned_writes.add(
                ".".join(call_chain[:-1]) or "<whole_parent_state>"
            )
    return tuple(sorted(parent_owned_writes))


def _reject_parent_owned_state_writes(
    fragment_class: ast.ClassDef,
    *,
    error_code: str,
    boundary: str,
) -> None:
    parent_owned_writes = _parent_owned_state_writes(fragment_class)
    if parent_owned_writes:
        raise _implementation_failure(
            error_code,
            f"{boundary} attempted to write parent-owned state: "
            + ", ".join(parent_owned_writes),
        )


def _expand_scaffolded_full_source(
    response: Mapping[str, Any],
    *,
    exact_parent_bundle: Mapping[str, Any],
    compiled_mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep P6 full mechanism freedom inside a machine-owned exact-parent ABI."""

    compiled = _normalized_compiled_mechanism(compiled_mechanism)
    assert compiled is not None
    parent_hook_contract = _semantic_id_parent_hook_contract(compiled)
    stage_methods = bool(parent_hook_contract and parent_hook_contract.get("owns_training_stages"))
    owned_methods: set[str] = set()
    for item in response.get("files", ()):
        if (
            not isinstance(item, Mapping)
            or item.get("path") != "recclaw_ext/candidate.py"
            or not isinstance(item.get("content"), str)
        ):
            continue
        try:
            fragment = ast.parse(str(item["content"]))
        except SyntaxError:
            continue
        candidate_classes = [
            class_node
            for class_node in fragment.body
            if isinstance(class_node, ast.ClassDef)
            and class_node.name == "FreshCandidateModel"
        ]
        for class_node in candidate_classes:
            if not stage_methods:
                _reject_parent_owned_state_writes(
                    class_node,
                    error_code="SEMANTIC_ID_ACTIVE_PARENT_STATE_OVERRIDE",
                    boundary="parent-preserving semantic-ID fragment",
                )
            owned_methods.update(
                node.name
                for node in class_node.body
                if isinstance(node, ast.FunctionDef)
            )

    if parent_hook_contract is not None:
        forbidden = set(parent_hook_contract["forbidden_outer_methods"])
        attempted_outer = sorted(owned_methods & forbidden)
        if attempted_outer:
            raise _implementation_failure(
                "SEMANTIC_ID_ACTIVE_PARENT_GENERATOR_OVERRIDE",
                "parent-preserving semantic-ID fragments cannot replace the "
                "machine-owned T5 loss/generation/scoring lifecycle: "
                + ", ".join(attempted_outer),
            )
        known_parent_hooks = set().union(
            *_SEMANTIC_ID_PARENT_SLOT_HOOKS.values()
        )
        allowed_parent_hooks = set(parent_hook_contract["allowed_parent_hooks"])
        undeclared_parent_hooks = sorted(
            (owned_methods & known_parent_hooks) - allowed_parent_hooks
        )
        if undeclared_parent_hooks:
            raise _implementation_failure(
                "SEMANTIC_ID_PARENT_HOOK_SCOPE_INVALID",
                "semantic-ID fragment owns a parent hook outside its declared "
                "changed slots: " + ", ".join(undeclared_parent_hooks),
            )
        missing_all = set(parent_hook_contract["required_all"]) - owned_methods
        missing_any = [
            set(group)
            for group in parent_hook_contract["required_any"]
            if not owned_methods.intersection(group)
        ]
        if missing_all or missing_any:
            requirements = sorted(missing_all)
            requirements.extend(
                "one of " + "/".join(sorted(group)) for group in missing_any
            )
            raise _implementation_failure(
                "SEMANTIC_ID_PARENT_HOOK_MISSING",
                "parent-local semantic-ID changes require "
                + ", ".join(requirements),
            )
    return _expand_exact_parent_method_patch(
        response,
        exact_parent_bundle=exact_parent_bundle,
        compiled_mechanism=compiled_mechanism,
        allow_mechanism_helpers=True,
        lock_trainer_lifecycle=not stage_methods,
        trainer_signatures=_SEMANTIC_ID_STAGE_TRAINER_SIGNATURES if stage_methods else None,
        model_signatures=(
            {
                name: (
                    _MODEL_PATCH_SIGNATURES[name] if name == "calculate_loss"
                    else _SEMANTIC_ID_PARENT_HOOK_SIGNATURES[name]
                )
                for name in parent_hook_contract["allowed_parent_hooks"]
            }
            if parent_hook_contract is not None
            else None
        ),
    )


_PROFILE_MODEL_HOOK_SIGNATURES = {
    "recclaw_diffusion_forward": (
        "def recclaw_diffusion_forward(self, clean_state, timestep, noise, "
        "history, prior=None):\n"
        "    pass\n"
    ),
    "recclaw_diffusion_prior": (
        "def recclaw_diffusion_prior(self, history):\n"
        "    pass\n"
    ),
    "recclaw_diffusion_state": (
        "def recclaw_diffusion_state(self, history):\n"
        "    pass\n"
    ),
    "recclaw_diffusion_decode": (
        "def recclaw_diffusion_decode(self, state):\n"
        "    pass\n"
    ),
    "recclaw_diffusion_condition": (
        "def recclaw_diffusion_condition(self, inputs, parameters, *, training):\n"
        "    pass\n"
    ),
    "recclaw_predict_field": (
        "def recclaw_predict_field(self, state, timestep, condition=None):\n"
        "    pass\n"
    ),
    "recclaw_diffusion_field": (
        "def recclaw_diffusion_field(self, inputs, parameters, *, training):\n"
        "    raise NotImplementedError\n"
    ),
    "recclaw_sequence_representation": (
        "def recclaw_sequence_representation(self, item_embeddings, item_seq_len, "
        "elapsed_days, query_elapsed_days, valid_mask):\n"
        "    pass\n"
    ),
    "recclaw_sequence_state": (
        "def recclaw_sequence_state(self, parent_state, item_embeddings, item_seq, "
        "item_seq_len, elapsed_days, query_elapsed_days, valid_mask, "
        "user_ids=None):\n"
        "    pass\n"
    ),
}


def _compose_hook_initializers(
    fragment_tree: ast.Module,
    *,
    parent_initializer: ast.FunctionDef,
    compiled: Mapping[str, Any],
    parent: Mapping[str, Any],
    parent_method_names: set[str],
    error_code: str,
) -> None:
    """Run inherited mechanism initialization before the current delta once."""
    fragment_classes = [
        node for node in fragment_tree.body if isinstance(node, ast.ClassDef)
    ]
    fragment_by_name = {
        node.name: node for node in fragment_classes[0].body
        if isinstance(node, ast.FunctionDef)
    }
    initializer_name = "recclaw_initialize_mechanism"
    current_initializer = fragment_by_name[initializer_name]
    initializer_suffix = sha256_digest(
        {
            "compiler_candidate_id": compiled["compiler_candidate_id"],
            "parent_source_tree_digest": parent["source_tree_digest"],
        }
    )[:16]
    parent_initializer_name = (
        f"recclaw_parent_initializer_{initializer_suffix}"
    )
    current_initializer_name = (
        f"recclaw_current_initializer_{initializer_suffix}"
    )
    generated_names = {parent_initializer_name, current_initializer_name}
    if generated_names & (parent_method_names | set(fragment_by_name)):
        raise _implementation_failure(
            error_code,
            "machine-owned initializer namespace collides with parent source",
        )

    def initializer_copy(
        source: ast.FunctionDef,
        name: str,
    ) -> ast.FunctionDef:
        copied = ast.parse(
            "def recclaw_initialize_mechanism(self, config, dataset):\n"
            "    pass\n"
        ).body[0]
        assert isinstance(copied, ast.FunctionDef)
        copied.name = name
        copied.body = copy.deepcopy(source.body)
        ast.copy_location(copied, source)
        return copied

    parent_initializer_copy = initializer_copy(
        parent_initializer,
        parent_initializer_name,
    )
    current_initializer_copy = initializer_copy(
        current_initializer,
        current_initializer_name,
    )
    combined_initializer = ast.parse(
        "def recclaw_initialize_mechanism(self, config, dataset):\n"
        f"    self.{parent_initializer_name}(config, dataset)\n"
        f"    self.{current_initializer_name}(config, dataset)\n"
    ).body[0]
    assert isinstance(combined_initializer, ast.FunctionDef)
    ast.copy_location(combined_initializer, current_initializer)
    combined_body: list[ast.stmt] = []
    for node in fragment_classes[0].body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == initializer_name
        ):
            combined_body.extend(
                (
                    parent_initializer_copy,
                    current_initializer_copy,
                    combined_initializer,
                )
            )
        else:
            combined_body.append(node)
    fragment_classes[0].body = combined_body


def _expand_profile_model_hooks(
    response: Mapping[str, Any],
    *,
    exact_parent_bundle: Mapping[str, Any],
    compiled_mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    """Move only compiler-declared mechanism hooks into the exact parent."""

    compiled = _normalized_compiled_mechanism(compiled_mechanism)
    if compiled is None:
        raise _implementation_failure(
            "PROFILE_MODEL_SOURCE_OWNERSHIP_MISSING",
            "profile model-hook materialization requires compiler scope",
        )
    ownership = compiled["implementation_binding"].get("source_ownership")
    if (
        not isinstance(ownership, Mapping)
        or ownership.get("mode") != PROFILE_MODEL_HOOKS_RESPONSE_MODE
    ):
        raise _implementation_failure(
            "PROFILE_MODEL_SOURCE_OWNERSHIP_MISSING",
            "compiled mechanism did not select profile model hooks",
        )
    hook_names = tuple(ownership["model_hooks"])
    patch_files = response.get("files") if isinstance(response, Mapping) else None
    if (
        not isinstance(patch_files, list)
        or len(patch_files) != 1
        or not isinstance(patch_files[0], Mapping)
        or patch_files[0].get("path") != "recclaw_ext/candidate.py"
        or not isinstance(patch_files[0].get("content"), str)
    ):
        raise _implementation_failure(
            "PROFILE_MODEL_HOOK_FILE_INVALID",
            "profile hooks require exactly one candidate method fragment",
        )
    try:
        fragment_tree = ast.parse(
            str(patch_files[0]["content"]),
            filename="recclaw_ext/candidate.py:profile-model-hooks",
        )
    except SyntaxError as error:
        raise _implementation_failure(
            "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
            "profile model-hook fragment is not valid Python",
        ) from error
    fragment_classes = [
        node for node in fragment_tree.body if isinstance(node, ast.ClassDef)
    ]
    methods = (
        [node for node in fragment_classes[0].body if isinstance(node, ast.FunctionDef)]
        if len(fragment_classes) == 1
        else []
    )
    method_names = {method.name for method in methods}
    if (
        len(fragment_tree.body) != 1
        or len(fragment_classes) != 1
        or fragment_classes[0].name != "FreshCandidateModel"
        or len(methods) != len(fragment_classes[0].body)
        or not set(hook_names) <= method_names
        or any(not method.name.startswith("recclaw_") for method in methods)
    ):
        raise _implementation_failure(
            "PROFILE_MODEL_HOOK_ABI_DRIFT",
            "Implementer may own only declared profile hooks and recclaw helpers",
        )
    _reject_parent_owned_state_writes(
        fragment_classes[0],
        error_code="PROFILE_MODEL_HOOK_ABI_DRIFT",
        boundary="profile model-hook fragment",
    )
    parent = _normalized_exact_parent_bundle(exact_parent_bundle)
    assert parent is not None
    parent_source = next(
        row["content"] for row in parent["files"]
        if row["path"] == "recclaw_ext/candidate.py"
    )
    parent_tree = ast.parse(parent_source)
    parent_methods = _fresh_candidate_mro_methods(parent_tree)
    _, parent_calls = _fresh_candidate_execution_graph(parent_tree)
    initializer_name = "recclaw_initialize_mechanism"
    # A stage hook cannot become a helper just because it shares the namespace.
    undeclared_hooks = set(_PROFILE_MODEL_HOOK_SIGNATURES) - set(hook_names)
    protected = set()
    pending = list(undeclared_hooks | set(_MODEL_PATCH_SIGNATURES) | {initializer_name})
    while pending:
        name = pending.pop()
        if name in protected or name in hook_names:
            continue
        protected.add(name)
        pending.extend(parent_calls.get(name, ()))
    protected.discard(initializer_name)
    echoes = set()
    for method in methods:
        if method.name not in protected:
            continue
        inherited = parent_methods.get(method.name)
        if inherited is not None and ast.dump(
            method, include_attributes=False
        ) == ast.dump(inherited, include_attributes=False):
            echoes.add(method.name)
            continue
        raise _implementation_failure(
            "PROFILE_MODEL_HOOK_ABI_DRIFT",
            "profile hook delta cannot replace retained method " + method.name
            + "; use a new recclaw_ helper for the mutable hooks",
        )
    fragment_classes[0].body = [node for node in methods if node.name not in echoes]
    current_initializer = next(
        (node for node in methods if node.name == initializer_name), None
    )
    parent_initializer = parent_methods.get(initializer_name)
    if current_initializer is not None and parent_initializer is not None:
        if ast.dump(current_initializer, include_attributes=False) == ast.dump(
            parent_initializer, include_attributes=False
        ):
            fragment_classes[0].body.remove(current_initializer)
        else:
            _compose_hook_initializers(
                fragment_tree,
                parent_initializer=parent_initializer,
                compiled=compiled,
                parent=parent,
                parent_method_names=set(parent_methods),
                error_code="PROFILE_MODEL_HOOK_ABI_DRIFT",
            )
    ast.fix_missing_locations(fragment_tree)
    response = {
        **dict(response),
        "files": [{
            "path": "recclaw_ext/candidate.py",
            "content": ast.unparse(fragment_tree) + "\n",
        }],
    }
    return _expand_exact_parent_method_patch(
        response,
        exact_parent_bundle=exact_parent_bundle,
        compiled_mechanism=compiled,
        lock_trainer_lifecycle=True,
        model_signatures={
            name: _PROFILE_MODEL_HOOK_SIGNATURES[name] for name in hook_names
        },
    )


def _expand_exact_parent_local_slot_patch(
    response: Mapping[str, Any],
    *,
    exact_parent_bundle: Mapping[str, Any],
    compiled_mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize one compiler-scoped extension inside the exact parent.

    The Implementer owns only the declared typed hook and mechanism helpers.
    The machine keeps parent orchestration, untouched mechanisms and trainer
    behavior, then inserts the hook at the compiler-selected seam. An objective
    hook replaces the primary loss while retaining the auxiliary loss terms.
    """

    compiled = _normalized_compiled_mechanism(compiled_mechanism)
    if compiled is None:
        raise _implementation_failure(
            "PARENT_LOCAL_SOURCE_OWNERSHIP_MISSING",
            "parent-local materialization requires compiled source ownership",
        )
    ownership = compiled["implementation_binding"].get("source_ownership")
    if (
        not isinstance(ownership, Mapping)
        or ownership.get("mode") != "PARENT_LOCAL_SLOT_HOOK_V1"
    ):
        raise _implementation_failure(
            "PARENT_LOCAL_SOURCE_OWNERSHIP_MISSING",
            "compiled mechanism did not select parent-local source ownership",
        )
    hook = ownership["model_hook"]
    method_name = str(hook["method_name"])
    primary_objective = hook["slot_id"] == "PRIMARY_OBJECTIVE"

    patch_files = response.get("files") if isinstance(response, Mapping) else None
    if (
        not isinstance(patch_files, list)
        or len(patch_files) != 1
        or not isinstance(patch_files[0], Mapping)
        or patch_files[0].get("path") != "recclaw_ext/candidate.py"
        or not isinstance(patch_files[0].get("content"), str)
    ):
        raise _implementation_failure(
            "PARENT_LOCAL_SLOT_PATCH_FILE_INVALID",
            "parent-local extension requires one model fragment",
        )
    try:
        fragment_tree = ast.parse(
            str(patch_files[0]["content"]),
            filename="recclaw_ext/candidate.py:parent-local-slot-patch",
        )
    except SyntaxError as error:
        raise _implementation_failure(
            "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
            "parent-local fragment is not valid Python",
        ) from error
    fragment_classes = [
        node for node in fragment_tree.body if isinstance(node, ast.ClassDef)
    ]
    fragment_methods = (
        [
            node
            for node in fragment_classes[0].body
            if isinstance(node, ast.FunctionDef)
        ]
        if len(fragment_classes) == 1
        else []
    )
    if (
        len(fragment_tree.body) != 1
        or len(fragment_classes) != 1
        or fragment_classes[0].name != "FreshCandidateModel"
        or len(fragment_methods) != len(fragment_classes[0].body)
        or method_name not in {item.name for item in fragment_methods}
        or any(
            item.name != "recclaw_initialize_mechanism"
            and not item.name.startswith("recclaw_")
            for item in fragment_methods
        )
    ):
        raise _implementation_failure(
            "PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
            "parent-local fragment may own only the declared hook and recclaw helpers",
        )

    _reject_parent_owned_state_writes(
        fragment_classes[0],
        error_code="PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
        boundary="parent-local fragment",
    )

    fragment_by_name = {item.name: item for item in fragment_methods}
    if len(fragment_by_name) != len(fragment_methods):
        raise _implementation_failure(
            "PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
            "parent-local fragment contains duplicate method names",
        )
    reserved_hook_names = {
        "recclaw_encode_representation",
        "recclaw_propagation_aggregation",
        "recclaw_score_head",
        "recclaw_primary_objective",
    }
    if any(
        name in reserved_hook_names and name != method_name
        for name in fragment_by_name
    ):
        raise _implementation_failure(
            "PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
            "parent-local fragment may define only its declared model hook",
        )

    parent = _normalized_exact_parent_bundle(exact_parent_bundle)
    assert parent is not None
    parent_model_source = next(
        (
            str(item["content"])
            for item in parent["files"]
            if item["path"] == "recclaw_ext/candidate.py"
        ),
        None,
    )
    if parent_model_source is None:
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent bundle lacks its model source",
        )
    try:
        parent_model_tree = ast.parse(
            parent_model_source,
            filename="recclaw_ext/candidate.py:exact-parent",
        )
    except SyntaxError as error:
        raise _implementation_failure(
            "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
            "exact parent model is not valid Python",
        ) from error
    parent_classes = {
        node.name: node
        for node in parent_model_tree.body
        if isinstance(node, ast.ClassDef)
    }
    parent_entrypoint = parent_classes.get("FreshCandidateModel")
    if parent_entrypoint is None:
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent source lacks FreshCandidateModel",
        )

    def inherited_method(
        classes_by_name: Mapping[str, ast.ClassDef],
        class_node: ast.ClassDef,
        name: str,
        seen: frozenset[str] = frozenset(),
    ) -> ast.FunctionDef | None:
        if class_node.name in seen:
            return None
        direct = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(direct) == 1:
            return direct[0]
        if direct:
            return None
        next_seen = seen | {class_node.name}
        inherited = [
            method
            for base in class_node.bases
            if isinstance(base, ast.Name) and base.id in classes_by_name
            for method in (
                inherited_method(
                    classes_by_name,
                    classes_by_name[base.id],
                    name,
                    next_seen,
                ),
            )
            if method is not None
        ]
        return inherited[0] if len(inherited) == 1 else None

    def inherited_method_names(
        classes_by_name: Mapping[str, ast.ClassDef],
        class_node: ast.ClassDef,
        seen: frozenset[str] = frozenset(),
    ) -> set[str]:
        if class_node.name in seen:
            return set()
        next_seen = seen | {class_node.name}
        names = {
            node.name
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        }
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in classes_by_name:
                names.update(
                    inherited_method_names(
                        classes_by_name,
                        classes_by_name[base.id],
                        next_seen,
                    )
                )
        return names

    parent_method_names = inherited_method_names(parent_classes, parent_entrypoint)
    initializer_name = (
        "recclaw_initialize_primary_objective" if primary_objective
        else "recclaw_initialize_mechanism"
    )
    replaceable_names = {method_name, initializer_name}
    collisions = sorted(
        name
        for name in fragment_by_name
        if name in parent_method_names and name not in replaceable_names
    )
    if collisions:
        raise _implementation_failure(
            "PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
            "parent-local fragment attempted to replace an existing helper: "
            + ", ".join(collisions),
        )

    patch_response = dict(response)
    if primary_objective:
        # This initializer belongs to the replaced objective, unlike the shared
        # initialization delta for retained representation mechanisms.
        if "recclaw_initialize_mechanism" in fragment_by_name:
            raise _implementation_failure(
                "PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
                "primary objective initialization uses recclaw_initialize_primary_objective",
            )
        if initializer_name not in fragment_by_name:
            fragment_classes[0].body.append(ast.parse(
                f"def {initializer_name}(self, config, dataset):\n    pass\n"
            ).body[0])
        ast.fix_missing_locations(fragment_tree)
        patch_response["files"] = [{
            "path": "recclaw_ext/candidate.py",
            "content": ast.unparse(fragment_tree) + "\n",
        }]
    current_initializer = fragment_by_name.get(initializer_name)
    if not primary_objective and current_initializer is not None and initializer_name in parent_method_names:
        parent_initializer = inherited_method(
            parent_classes,
            parent_entrypoint,
            initializer_name,
        )
        if parent_initializer is None:
            raise _implementation_failure(
                "PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
                "exact parent has no unique mechanism initializer",
            )
        _compose_hook_initializers(
            fragment_tree,
            parent_initializer=parent_initializer,
            compiled=compiled,
            parent=parent,
            parent_method_names=parent_method_names,
            error_code="PARENT_LOCAL_SLOT_PATCH_ABI_DRIFT",
        )
        ast.fix_missing_locations(fragment_tree)
        patch_response["files"] = [
            {
                "path": "recclaw_ext/candidate.py",
                "content": ast.unparse(fragment_tree) + "\n",
            }
        ]

    expanded = _expand_exact_parent_method_patch(
        patch_response,
        exact_parent_bundle=exact_parent_bundle,
        compiled_mechanism=compiled_mechanism,
    )
    files = [dict(item) for item in expanded["files"]]
    model_index = next(
        (
            index
            for index, item in enumerate(files)
            if item.get("path") == "recclaw_ext/candidate.py"
        ),
        None,
    )
    if model_index is None:
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent bundle lacks its model source",
        )
    model_source = files[model_index].get("content")
    if not isinstance(model_source, str):
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent model source is invalid",
        )
    try:
        model_tree = ast.parse(model_source, filename="recclaw_ext/candidate.py")
    except SyntaxError as error:
        raise _implementation_failure(
            "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
            "expanded exact-parent model is not valid Python",
        ) from error

    classes = {
        node.name: node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef)
    }
    entrypoint = classes.get("FreshCandidateModel")
    if entrypoint is None:
        raise _implementation_failure(
            "EXACT_PARENT_BUNDLE_INVALID",
            "exact parent source lacks FreshCandidateModel",
        )

    if primary_objective:
        from recclaw_core.research_line.bl_icf_objective_scaffold import bind_parent_objective

        bind_parent_objective(
            entrypoint, resolve_method=lambda name: inherited_method(classes, entrypoint, name),
        )
        ast.fix_missing_locations(model_tree)
        files[model_index]["content"] = ast.unparse(model_tree) + "\n"
        return canonical_value({**dict(expanded), "files": files})

    if hook["slot_id"] == "SCORE_HEAD":
        from recclaw_core.research_line.bl_icf_score_scaffold import bind_parent_score

        try:
            bind_parent_score(
                model_tree, entrypoint,
                resolve_method=lambda name: inherited_method(classes, entrypoint, name),
                train_objective=hook["hook_point"] == "BPR_AND_EVALUATION_SCORE_V1",
            )
        except ValueError as error:
            raise _implementation_failure("PARENT_LOCAL_SCORE_SEAM_UNSUPPORTED", str(error)) from error
        ast.fix_missing_locations(model_tree)
        files[model_index]["content"] = ast.unparse(model_tree) + "\n"
        return canonical_value({**dict(expanded), "files": files})

    inherited_forward = inherited_method(classes, entrypoint, "forward")
    if inherited_forward is None:
        raise _implementation_failure(
            "PARENT_LOCAL_HOOK_POINT_UNAVAILABLE",
            "exact parent has no unique inherited forward method",
        )
    forward = copy.deepcopy(inherited_forward)

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    existing_hook_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method_name
    ]
    if len(existing_hook_calls) > 1:
        raise _implementation_failure(
            "PARENT_LOCAL_HOOK_POINT_AMBIGUOUS",
            "exact parent invokes the declared local hook more than once",
        )
    existing_hook_call = (
        existing_hook_calls[0] if existing_hook_calls else None
    )

    hook_point = str(hook["hook_point"])
    if hook_point == "PRE_PROPAGATION_REPRESENTATION_V1":

        def embedding_weight(node: ast.AST, role: str) -> bool:
            return (
                isinstance(node, ast.Attribute)
                and node.attr == "weight"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == f"{role}_embedding"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
            )

        anchors: list[tuple[int, ast.Assign, str]] = []
        for index, statement in enumerate(forward.body):
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or not isinstance(statement.value, ast.Call)
                or dotted_name(statement.value.func)
                not in {"torch.cat", "torch.concat"}
                or not statement.value.args
                or not isinstance(statement.value.args[0], (ast.List, ast.Tuple))
            ):
                continue
            elements = statement.value.args[0].elts
            if (
                len(elements) == 2
                and any(embedding_weight(item, "user") for item in elements)
                and any(embedding_weight(item, "item") for item in elements)
            ):
                anchors.append((index, statement, statement.targets[0].id))
        if len(anchors) != 1:
            raise _implementation_failure(
                "PARENT_LOCAL_HOOK_POINT_UNAVAILABLE",
                "exact parent has no unique user-item representation seam",
            )
        anchor_index, anchor, representation_name = anchors[0]
        propagation_calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and dotted_name(node.func) in {"torch.sparse.mm", "torch.spmm"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == representation_name
        ]
        if not propagation_calls:
            raise _implementation_failure(
                "PARENT_LOCAL_HOOK_POINT_UNAVAILABLE",
                "exact parent has no propagation path after the encoder seam",
            )
        if existing_hook_call is not None:
            hook_statement = (
                forward.body[anchor_index + 1]
                if anchor_index + 1 < len(forward.body)
                else None
            )
            if not (
                isinstance(hook_statement, ast.Assign)
                and len(hook_statement.targets) == 1
                and isinstance(hook_statement.targets[0], ast.Name)
                and hook_statement.targets[0].id == representation_name
                and hook_statement.value is existing_hook_call
                and not existing_hook_call.keywords
                and len(existing_hook_call.args) == 1
                and isinstance(existing_hook_call.args[0], ast.Name)
                and existing_hook_call.args[0].id == representation_name
            ):
                raise _implementation_failure(
                    "PARENT_LOCAL_HOOK_POINT_AMBIGUOUS",
                    "existing encoder hook is not the compiler-owned seam",
                )
        else:
            hook_assignment = ast.Assign(
                targets=[ast.Name(id=representation_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=method_name,
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=representation_name, ctx=ast.Load())],
                    keywords=[],
                ),
            )
            ast.copy_location(hook_assignment, anchor)
            forward.body.insert(anchor_index + 1, hook_assignment)
    elif hook_point == "POST_PROPAGATION_MESSAGE_V1":
        direct_seams: list[tuple[ast.For, ast.Assign, ast.Call]] = []
        reused_seams: list[tuple[ast.For, ast.Assign, ast.Call]] = []
        for loop in (
            node for node in ast.walk(forward) if isinstance(node, ast.For)
        ):
            if not isinstance(loop.target, ast.Name):
                continue
            for statement in loop.body:
                if not (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and isinstance(statement.value, ast.Call)
                ):
                    continue
                candidate = statement.value
                if (
                    dotted_name(candidate.func)
                    in {"torch.sparse.mm", "torch.spmm"}
                    and len(candidate.args) >= 2
                    and isinstance(candidate.args[1], ast.Name)
                    and statement.targets[0].id == candidate.args[1].id
                ):
                    direct_seams.append((loop, statement, candidate))
                    continue
                if candidate is not existing_hook_call:
                    continue
                if (
                    candidate.keywords
                    or len(candidate.args) != 4
                    or not isinstance(candidate.args[0], ast.Call)
                    or dotted_name(candidate.args[0].func)
                    not in {"torch.sparse.mm", "torch.spmm"}
                    or len(candidate.args[0].args) < 2
                    or not isinstance(candidate.args[0].args[1], ast.Name)
                    or not isinstance(candidate.args[1], ast.Name)
                    or statement.targets[0].id != candidate.args[1].id
                    or candidate.args[0].args[1].id != candidate.args[1].id
                    or ast.dump(
                        candidate.args[0].args[0],
                        include_attributes=False,
                    )
                    != ast.dump(candidate.args[2], include_attributes=False)
                    or not isinstance(candidate.args[3], ast.Name)
                    or candidate.args[3].id != loop.target.id
                ):
                    continue
                reused_seams.append((loop, statement, candidate.args[0]))
        if existing_hook_call is not None:
            if len(reused_seams) != 1 or direct_seams:
                raise _implementation_failure(
                    "PARENT_LOCAL_HOOK_POINT_AMBIGUOUS",
                    "existing propagation hook is not the unique compiler-owned seam",
                )
        else:
            if len(direct_seams) != 1:
                raise _implementation_failure(
                    "PARENT_LOCAL_HOOK_POINT_UNAVAILABLE",
                    "exact parent has no unique iterative propagation message seam",
                )
            loop, propagation_assignment, sparse_message = direct_seams[0]
            relation = sparse_message.args[0]
            prior = sparse_message.args[1]
            wrapped_message = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=method_name,
                    ctx=ast.Load(),
                ),
                args=[
                    copy.deepcopy(sparse_message),
                    copy.deepcopy(prior),
                    copy.deepcopy(relation),
                    copy.deepcopy(loop.target),
                ],
                keywords=[],
            )
            ast.copy_location(wrapped_message, sparse_message)
            propagation_assignment.value = wrapped_message
    else:
        raise _implementation_failure(
            "PARENT_LOCAL_HOOK_POINT_UNAVAILABLE",
            "compiled parent-local hook point is unsupported",
        )
    entrypoint.body = [
        node
        for node in entrypoint.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "forward")
    ]
    entrypoint.body.append(forward)
    ast.fix_missing_locations(model_tree)
    files[model_index]["content"] = ast.unparse(model_tree) + "\n"
    return canonical_value({**dict(expanded), "files": files})


def _compiler_owned_source_binding(
    response: Mapping[str, Any],
    *,
    compiled_mechanism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Inject exact compiler constants and stable lifecycle bindings."""

    normalized = _normalized_compiled_mechanism(compiled_mechanism)
    if normalized is None:
        return canonical_value(dict(response))
    if not isinstance(response, Mapping) or not isinstance(response.get("files"), list):
        return canonical_value(dict(response))
    binding = normalized["implementation_binding"]
    source_ownership = binding.get("source_ownership")
    owns_profile_model_hooks = bool(
        isinstance(source_ownership, Mapping)
        and source_ownership.get("mode") == PROFILE_MODEL_HOOKS_RESPONSE_MODE
    )
    search_space_id = normalized["space_identity"].get("search_space_id")
    owns_strict_early_stopping = search_space_id in {
        "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1",
        "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1",
        "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1",
    }
    program_payload = normalized["mechanism_program"].get("program_payload", {})
    efficiency = (
        program_payload.get("efficiency", {})
        if isinstance(program_payload, Mapping)
        else {}
    )
    precompute_required_solver = bool(
        isinstance(efficiency, Mapping)
        and efficiency.get("closed_form_solver")
        and efficiency.get("precompute_required") is True
    )
    changed_slots = (
        program_payload.get("changed_slots", ())
        if isinstance(program_payload, Mapping)
        else ()
    )
    changed_slot_ids = {
        str(item.get("slot_id"))
        for item in changed_slots
        if isinstance(item, Mapping)
    }
    epoch_sampler_specs = tuple(
        spec
        for spec in binding["component_specs"].values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "NEGATIVE_SAMPLER"
        and isinstance(spec.get("parameters"), Mapping)
        and spec["parameters"].get("refresh_frequency") == "EPOCH"
    )
    scored_batch_false_negative_specs = tuple(
        spec
        for spec in binding["component_specs"].values()
        if isinstance(spec, Mapping)
        and spec.get("slot_id") == "NEGATIVE_SAMPLER"
        and spec.get("primitive_id") == "sampler.false_negative_aware"
        and isinstance(spec.get("parameters"), Mapping)
        and spec["parameters"].get("hardness") in {"DYNAMIC", "CURRICULUM"}
        and spec["parameters"].get("refresh_frequency") == "BATCH"
    )
    # Scored BATCH samplers select every batch, but curriculum still needs the
    # same machine-owned epoch signal.  The trainer wrapper supplies only that
    # stable lifecycle fact; scientific scoring and selection remain hooks.
    owns_epoch_sampler_lifecycle = bool(
        epoch_sampler_specs or scored_batch_false_negative_specs
    )
    owns_train_data_fit_lifecycle = bool(
        required_train_data_fit_roles(binding["component_specs"])
    )
    owns_normalized_embedding_binding = any(
        isinstance(spec, Mapping)
        and spec.get("slot_id") == "EMBEDDING"
        and spec.get("primitive_id") == "embedding.normalized"
        for spec in binding["component_specs"].values()
    )
    primitive_ids = set(binding["primitive_ids"])
    owns_frozen_letter_sid_codec = {
        "tokenizer.frozen_letter_collision_suffix",
        "decode.parallel_valid_assignment",
    } <= primitive_ids
    owns_liger_product_geometry = bool(
        {
            "tokenizer.product_quantization",
            "tokenizer.orthogonally_preconditioned_product_quantization",
        }
        & primitive_ids
    )
    owns_semantic_id_decode = _semantic_id_machine_decode_contract(normalized) is not None
    exact_values = {
        "RECCLAW_COMPILER_CANDIDATE_ID": normalized["compiler_candidate_id"],
        "RECCLAW_MECHANISM_PROGRAM_DIGEST": normalized["mechanism_program_digest"],
        "RECCLAW_MECHANISM_SEMANTICS_DIGEST": normalized["mechanism_semantics_digest"],
        "RECCLAW_IMPLEMENTED_COMPONENT_IDS": binding["component_ids"],
        "RECCLAW_IMPLEMENTED_COMPONENT_SPECS": binding["component_specs"],
        "RECCLAW_IMPLEMENTED_PRIMITIVE_IDS": binding["primitive_ids"],
        "RECCLAW_IMPLEMENTED_CUSTOM_COMPONENT_IDS": binding["custom_component_ids"],
        "RECCLAW_IMPLEMENTED_ARCHITECTURE_OPERATOR_IDS": binding[
            "architecture_operator_ids"
        ],
        "RECCLAW_DECLARED_CHANGED_SLOTS": changed_slots,
        "RECCLAW_PRECOMPUTE_REQUIRED_CLOSED_FORM_SOLVER": (
            precompute_required_solver
        ),
    }
    entrypoint = response.get("entrypoint")
    entrypoint_class = (
        entrypoint.rsplit(":", 1)[1]
        if isinstance(entrypoint, str)
        and entrypoint.startswith("recclaw_ext.candidate:")
        and entrypoint.count(":") == 1
        else None
    )
    files: list[dict[str, Any]] = []
    found = False
    for item in response["files"]:
        if not isinstance(item, Mapping):
            files.append(item)
            continue
        path = item.get("path")
        if path == "recclaw_ext/trainer.py":
            trainer_source = item.get("content")
            if not isinstance(trainer_source, str):
                files.append(dict(item))
                continue
            if search_space_id == "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1":
                trainer_source = _bind_semantic_id_trainer_stage(trainer_source, normalized)
            try:
                trainer_tree = ast.parse(
                    trainer_source,
                    filename="recclaw_ext/trainer.py",
                )
            except SyntaxError as error:
                raise _implementation_failure(
                    "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
                    "trainer source is not valid Python",
                ) from error

            # Replace owned imports and their actual alias calls together;
            # other helpers from these scaffold modules remain candidate code.
            compiler_trainer_binding_names: set[str] = set()
            has_compiler_trainer_import = False
            for module, function_name in (
                ("recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold", "bind_epoch_sampler_trainer_class"),
                ("recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold", "bind_train_data_fit_trainer_class"),
                ("recclaw_core.experiments.helix_abc_v1.trainer_lifecycle_scaffold", "bind_strict_early_stopping_trainer_class"),
            ):
                compiler_trainer_binding_names.update(
                    (function_name, "_recclaw_" + function_name)
                )
                for node in trainer_tree.body:
                    if isinstance(node, ast.ImportFrom) and node.module == module:
                        aliases = [alias for alias in node.names if alias.name == function_name]
                        has_compiler_trainer_import |= bool(aliases)
                        compiler_trainer_binding_names.update(
                            alias.asname or alias.name for alias in aliases
                        )
                        node.names = [alias for alias in node.names if alias.name != function_name]
            has_compiler_trainer_binding = has_compiler_trainer_import or any(
                (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "FreshCandidateTrainer"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id
                    in compiler_trainer_binding_names
                )
                for node in trainer_tree.body
            )
            if (
                not owns_epoch_sampler_lifecycle
                and not owns_train_data_fit_lifecycle
                and not owns_strict_early_stopping
                and not has_compiler_trainer_binding
            ):
                files.append(dict(item))
                continue

            class _RemoveImplementationOwnedRefresh(ast.NodeTransformer):
                def visit_Call(self, node: ast.Call) -> ast.AST:
                    visited = self.generic_visit(node)
                    if (
                        isinstance(visited, ast.Call)
                        and isinstance(visited.func, ast.Attribute)
                        and visited.func.attr == "recclaw_sampler_refresh"
                    ):
                        return ast.copy_location(ast.Constant(value=None), visited)
                    return visited

            if owns_epoch_sampler_lifecycle:
                trainer_tree = _RemoveImplementationOwnedRefresh().visit(
                    trainer_tree
                )
            trainer_tree.body = [
                node
                for node in trainer_tree.body
                if not (
                    isinstance(node, ast.ImportFrom)
                    and not node.names
                )
                and not (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "FreshCandidateTrainer"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id
                    in compiler_trainer_binding_names
                )
            ]
            if owns_epoch_sampler_lifecycle:
                trainer_tree.body.extend(
                    ast.parse(
                        "from recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold "
                        "import bind_epoch_sampler_trainer_class as "
                        "_recclaw_bind_epoch_sampler_trainer_class\n"
                        "FreshCandidateTrainer = "
                        "_recclaw_bind_epoch_sampler_trainer_class(FreshCandidateTrainer)\n"
                    ).body
                )
            if owns_train_data_fit_lifecycle:
                trainer_tree.body.extend(
                    ast.parse(
                        "from recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold "
                        "import bind_train_data_fit_trainer_class as "
                        "_recclaw_bind_train_data_fit_trainer_class\n"
                        "FreshCandidateTrainer = "
                        "_recclaw_bind_train_data_fit_trainer_class(FreshCandidateTrainer)\n"
                    ).body
                )
            if owns_strict_early_stopping:
                trainer_tree.body.extend(
                    ast.parse(
                        "from recclaw_core.experiments.helix_abc_v1."
                        "trainer_lifecycle_scaffold import "
                        "bind_strict_early_stopping_trainer_class as "
                        "_recclaw_bind_strict_early_stopping_trainer_class\n"
                        "FreshCandidateTrainer = "
                        "_recclaw_bind_strict_early_stopping_trainer_class("
                        "FreshCandidateTrainer)\n"
                    ).body
                )
            ast.fix_missing_locations(trainer_tree)
            files.append(
                {
                    **dict(item),
                    "content": ast.unparse(trainer_tree) + "\n",
                }
            )
            continue
        if path != "recclaw_ext/candidate.py":
            files.append(dict(item) if isinstance(item, Mapping) else item)
            continue
        found = True
        source = item.get("content")
        if not isinstance(source, str):
            files.append(dict(item))
            continue
        try:
            tree = ast.parse(source, filename="recclaw_ext/candidate.py")
        except SyntaxError as error:
            raise _implementation_failure(
                "IMPLEMENTATION_SOURCE_SYNTAX_INVALID",
                "entrypoint source is not valid Python",
            ) from error

        if _semantic_id_parent_hook_contract(normalized) is not None:
            from recclaw_core.search_spaces.semantic_id_generative_v1.semantic_decode_scaffold import (
                preserve_liger_context_input_gradients,
            )

            preserve_liger_context_input_gradients(tree)

        # RecBole owns the model input ABI.  Implementers may express the
        # corresponding enum member as its wire spelling, but the executable
        # class must expose the real InputType object.  Preserve the selected
        # mode (including POINTWISE/LISTWISE); this is syntax normalization,
        # not a research-policy choice.
        input_type_members = {
            "listwise": "LISTWISE",
            "pairwise": "PAIRWISE",
            "pointwise": "POINTWISE",
        }
        input_type_normalized = False
        for class_node in (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and entrypoint_class is not None
            and node.name == entrypoint_class
        ):
            _normalize_canonical_identity_cardinality(
                class_node,
                binding["component_specs"],
            )
            for statement in class_node.body:
                target: ast.expr | None = None
                value: ast.expr | None = None
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                ):
                    target = statement.targets[0]
                    value = statement.value
                elif isinstance(statement, ast.AnnAssign):
                    target = statement.target
                    value = statement.value
                if (
                    isinstance(target, ast.Name)
                    and target.id == "input_type"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value.casefold() in input_type_members
                ):
                    statement.value = ast.Attribute(
                        value=ast.Name(id="InputType", ctx=ast.Load()),
                        attr=input_type_members[value.value.casefold()],
                        ctx=ast.Load(),
                    )
                    input_type_normalized = True
        if input_type_normalized and not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "recbole.utils"
            and any(
                alias.name == "InputType"
                and alias.asname in {None, "InputType"}
                for alias in node.names
            )
            for node in tree.body
        ):
            import_at = next(
                (
                    index
                    for index, node in enumerate(tree.body)
                    if not isinstance(node, (ast.Import, ast.ImportFrom))
                ),
                len(tree.body),
            )
            tree.body.insert(
                import_at,
                ast.ImportFrom(
                    module="recbole.utils",
                    names=[ast.alias(name="InputType")],
                    level=0,
                ),
            )
        # Replace compiler-owned class bindings as a pair, including aliases.
        # Removing an import while retaining its differently named call turns
        # valid Implementer source into a NameError before qualification.
        compiler_binding_names: set[str] = set()
        for module, function_name in (
            ("recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold", "bind_epoch_sampler_model_class"),
            ("recclaw_core.experiments.helix_abc_v1.innovation_spine", "bind_normalized_embedding_model_class"),
            (
                "recclaw_core.search_spaces.semantic_id_generative_v1.sid_codec_scaffold",
                "bind_frozen_letter_sid_codec_model_class",
            ),
            (
                "recclaw_core.search_spaces.semantic_id_generative_v1.product_quantizer_scaffold",
                "bind_liger_product_quantizer_model_class",
            ),
            (
                "recclaw_core.search_spaces.semantic_id_generative_v1.semantic_decode_scaffold",
                "bind_semantic_id_decode_model_class",
            ),
            (
                "recclaw_core.search_spaces.semantic_id_generative_v1.semantic_decode_scaffold",
                "bind_semantic_id_model_metadata",
            ),
            (
                "recclaw_core.search_spaces.diffusion_flow_cf_v1.mechanism_scaffold",
                "bind_diffusion_flow_model_class",
            ),
            (
                "recclaw_core.search_spaces.sequential_scaling_v1.mechanism_scaffold",
                "bind_sequential_scaling_model_class",
            ),
        ):
            compiler_binding_names.update((function_name, "_recclaw_" + function_name))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == module:
                    compiler_binding_names.update(
                        alias.asname or alias.name
                        for alias in node.names if alias.name == function_name
                    )
                    node.names = [alias for alias in node.names if alias.name != function_name]
        body = [
            node
            for node in tree.body
            if not (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and (
                    node.targets[0].id
                    if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    else node.target.id
                    if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    else None
                )
                in exact_values
            )
            and not (
                isinstance(node, ast.ImportFrom)
                and not node.names
            )
            and not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "FreshCandidateModel"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in compiler_binding_names
            )
        ]
        assignments = [
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.parse(repr(canonical_value(value)), mode="eval").body,
            )
            for name, value in exact_values.items()
        ]
        insert_at = next(
            (index for index, node in enumerate(body) if isinstance(node, ast.ClassDef)),
            len(body),
        )
        body[insert_at:insert_at] = assignments
        if (
            entrypoint_class is not None
            and entrypoint_class != "FreshCandidateModel"
            and any(
                isinstance(node, ast.ClassDef) and node.name == entrypoint_class
                for node in body
            )
        ):
            body.append(
                ast.Assign(
                    targets=[
                        ast.Name(
                            id="FreshCandidateModel",
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Name(
                        id=entrypoint_class,
                        ctx=ast.Load(),
                    ),
                )
            )
        if owns_profile_model_hooks:
            if search_space_id == "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1":
                profile_binding_source = (
                    "from recclaw_core.search_spaces.sequential_scaling_v1."
                    "mechanism_scaffold import bind_sequential_scaling_model_class as "
                    "_recclaw_bind_sequential_scaling_model_class\n"
                    "FreshCandidateModel = _recclaw_bind_sequential_scaling_model_class("
                    "FreshCandidateModel, RECCLAW_IMPLEMENTED_COMPONENT_SPECS, "
                    "RECCLAW_DECLARED_CHANGED_SLOTS)\n"
                )
            elif search_space_id == "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1":
                profile_binding_source = (
                    "from recclaw_core.search_spaces.diffusion_flow_cf_v1."
                    "mechanism_scaffold import bind_diffusion_flow_model_class as "
                    "_recclaw_bind_diffusion_flow_model_class\n"
                    "FreshCandidateModel = _recclaw_bind_diffusion_flow_model_class("
                    "FreshCandidateModel, RECCLAW_IMPLEMENTED_COMPONENT_SPECS, "
                    "RECCLAW_DECLARED_CHANGED_SLOTS)\n"
                )
            else:
                raise _implementation_failure(
                    "UNSUPPORTED_PROFILE_MECHANICAL_BINDING",
                    "profile model hooks have no compiler-owned binder",
                )
            body.extend(ast.parse(profile_binding_source).body)
        body.extend(
            ast.parse(
                "from recclaw_core.experiments.helix_abc_v1.epoch_sampler_scaffold "
                "import bind_epoch_sampler_model_class as "
                "_recclaw_bind_epoch_sampler_model_class\n"
                "FreshCandidateModel = _recclaw_bind_epoch_sampler_model_class("
                "FreshCandidateModel, RECCLAW_IMPLEMENTED_COMPONENT_SPECS)\n"
            ).body
        )
        if owns_normalized_embedding_binding:
            body.extend(
                ast.parse(
                    "from recclaw_core.experiments.helix_abc_v1.innovation_spine "
                    "import bind_normalized_embedding_model_class as "
                    "_recclaw_bind_normalized_embedding_model_class\n"
                    "FreshCandidateModel = "
                    "_recclaw_bind_normalized_embedding_model_class("
                    "FreshCandidateModel, RECCLAW_IMPLEMENTED_COMPONENT_SPECS)\n"
                ).body
            )
        if owns_frozen_letter_sid_codec:
            body.extend(
                ast.parse(
                    "from recclaw_core.search_spaces.semantic_id_generative_v1."
                    "sid_codec_scaffold import "
                    "bind_frozen_letter_sid_codec_model_class as "
                    "_recclaw_bind_frozen_letter_sid_codec_model_class\n"
                    "FreshCandidateModel = "
                    "_recclaw_bind_frozen_letter_sid_codec_model_class("
                    "FreshCandidateModel, RECCLAW_IMPLEMENTED_COMPONENT_SPECS)\n"
                ).body
            )
        if owns_liger_product_geometry:
            body.extend(
                ast.parse(
                    "from recclaw_core.search_spaces.semantic_id_generative_v1."
                    "product_quantizer_scaffold import "
                    "bind_liger_product_quantizer_model_class as "
                    "_recclaw_bind_liger_product_quantizer_model_class\n"
                    "FreshCandidateModel = "
                    "_recclaw_bind_liger_product_quantizer_model_class("
                    "FreshCandidateModel, RECCLAW_IMPLEMENTED_COMPONENT_SPECS)\n"
                ).body
            )
        if normalized["space_identity"].get("search_space_id") == "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1":
            body.extend(ast.parse(
                "from recclaw_core.search_spaces.semantic_id_generative_v1."
                "semantic_decode_scaffold import bind_semantic_id_model_metadata "
                "as _recclaw_bind_semantic_id_model_metadata\n"
                "FreshCandidateModel = _recclaw_bind_semantic_id_model_metadata(FreshCandidateModel)\n"
            ).body)
        if owns_semantic_id_decode:
            body.extend(
                ast.parse(
                    "from recclaw_core.search_spaces.semantic_id_generative_v1."
                    "semantic_decode_scaffold import "
                    "bind_semantic_id_decode_model_class as "
                    "_recclaw_bind_semantic_id_decode_model_class\n"
                    "FreshCandidateModel = "
                    "_recclaw_bind_semantic_id_decode_model_class("
                    "FreshCandidateModel, RECCLAW_IMPLEMENTED_COMPONENT_SPECS)\n"
                ).body
            )
        tree.body = body
        ast.fix_missing_locations(tree)
        files.append({**dict(item), "content": ast.unparse(tree) + "\n"})
    if not found:
        return canonical_value(dict(response))
    return canonical_value(
        {
            **dict(response),
            "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
            "files": files,
        }
    )


def _validate_exact_parent_full_source_diff(
    response: Mapping[str, Any],
    *,
    exact_parent_bundle: Mapping[str, Any],
) -> None:
    """Validate a compiler-free full-source child against one exact parent.

    Open search spaces can own a full model source without a BL compiler.  The
    stable package ABI remains machine-owned: every parent file must be
    returned, non-entrypoint files must stay byte-exact, and the executable
    source must contain a real delta.  This branch does not infer or repair the
    scientific mechanism.
    """

    parent = _normalized_exact_parent_bundle(exact_parent_bundle)
    assert parent is not None
    parent_files = {
        str(item["path"]): str(item["content"])
        for item in parent["files"]
    }
    child_files = {
        str(item["path"]): str(item["content"])
        for item in response["files"]
    }
    if set(child_files) != set(parent_files):
        raise _implementation_failure(
            "EXACT_PARENT_FILE_SET_DRIFT",
            "full-source child must preserve the exact parent package file set",
        )
    entrypoint_path = (
        str(response["entrypoint"]).split(":", 1)[0].replace(".", "/") + ".py"
    )
    if entrypoint_path not in parent_files:
        raise _implementation_failure(
            "EXACT_PARENT_ENTRYPOINT_DRIFT",
            "full-source child entrypoint is outside the exact parent package",
        )
    changed_paths = tuple(
        sorted(
            path
            for path, content in child_files.items()
            if content != parent_files[path]
        )
    )
    if changed_paths != (entrypoint_path,):
        raise _implementation_failure(
            "EXACT_PARENT_FULL_SOURCE_DIFF_INVALID",
            "compiler-free full-source child must change only its executable source",
            details={"changed_paths": changed_paths},
        )


def _validate_declared_mechanism_observability(
    tree: ast.Module,
    component_specs: Mapping[str, Any] | None,
) -> None:
    """Require declared structural data to reach an observable model path."""

    if not isinstance(component_specs, Mapping):
        return
    components = tuple(
        item for item in component_specs.values() if isinstance(item, Mapping)
    )
    declared_roles = {
        str(source.get("data_role"))
        for component in components
        for input_spec in component.get("inputs", ())
        if isinstance(input_spec, Mapping)
        for source in (input_spec.get("source"),)
        if isinstance(source, Mapping) and source.get("kind") == "DATA"
    }
    component_by_id = {
        str(component.get("component_id")): component
        for component in components
        if isinstance(component.get("component_id"), str)
    }
    outgoing: dict[str, set[str]] = {}
    for component_id, component in component_by_id.items():
        for input_spec in component.get("inputs", ()):
            if not isinstance(input_spec, Mapping):
                continue
            source = input_spec.get("source")
            if isinstance(source, Mapping) and source.get("kind") == "COMPONENT":
                source_id = source.get("component_id")
                if isinstance(source_id, str):
                    outgoing.setdefault(source_id, set()).add(component_id)
    structural_sources = {
        component_id
        for component_id, component in component_by_id.items()
        if component.get("slot_id") == "RELATION_VIEW"
        or any(
            isinstance(input_spec, Mapping)
            and isinstance(input_spec.get("source"), Mapping)
            and input_spec["source"].get("kind") == "DATA"
            and (
                str(input_spec["source"].get("data_role", "")).endswith("_GRAPH")
                or "RELATION" in str(input_spec["source"].get("data_role", ""))
                or "ADJACENCY" in str(input_spec["source"].get("data_role", ""))
            )
            for input_spec in component.get("inputs", ())
        )
    }
    structural_operators = {
        component_id
        for component_id, component in component_by_id.items()
        if component.get("slot_id") in {"MESSAGE", "PROPAGATION_AGGREGATION"}
        or (
            component.get("slot_id") == "ENCODER"
            and component.get("primitive_id") != "encoder.none_mf"
            and any(
                isinstance(input_spec, Mapping)
                and isinstance(input_spec.get("source"), Mapping)
                and input_spec["source"].get("kind") == "COMPONENT"
                and input_spec["source"].get("component_id") in structural_sources
                for input_spec in component.get("inputs", ())
            )
        )
    }
    score_ids = {
        component_id
        for component_id, component in component_by_id.items()
        if component.get("slot_id") == "SCORE_HEAD"
    }
    training_ids = {
        component_id
        for component_id, component in component_by_id.items()
        if component.get("slot_id") == "TRAINING_PROCEDURE"
    }
    execution_sinks = score_ids | training_ids
    unconsumed_sources = []
    for source_id in structural_sources:
        pending = [source_id]
        visited = {source_id}
        reaches_execution = False
        while pending:
            current = pending.pop()
            if current in execution_sinks:
                reaches_execution = True
                break
            for successor in outgoing.get(current, ()):
                if successor not in visited:
                    visited.add(successor)
                    pending.append(successor)
        if not reaches_execution:
            unconsumed_sources.append(source_id)
    if unconsumed_sources:
        raise _implementation_failure(
            "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
            "declared relation/graph components have no score or training consumer: "
            + ", ".join(sorted(unconsumed_sources)),
        )
    structural = False
    for source_id in structural_sources:
        pending = [(source_id, source_id in structural_operators)]
        visited = set(pending)
        while pending:
            current, seen_structural = pending.pop()
            if current in score_ids and seen_structural:
                structural = True
                break
            for successor in outgoing.get(current, ()):
                state = (
                    successor,
                    seen_structural or successor in structural_operators,
                )
                if state not in visited:
                    visited.add(state)
                    pending.append(state)
        if structural:
            break
    if not structural:
        return

    def dotted_name(value: ast.AST) -> str:
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            prefix = dotted_name(value.value)
            return f"{prefix}.{value.attr}" if prefix else value.attr
        return ""

    def is_structural_operation(node: ast.AST) -> bool:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return True
        if not isinstance(node, ast.Call):
            return False
        name = dotted_name(node.func).lower()
        return name.endswith(
            (
                "sparse.mm",
                ".spmm",
                ".propagate",
                ".message_and_aggregate",
                ".index_add",
                ".index_add_",
                ".scatter",
                ".scatter_add",
            )
        )

    methods, calls = _fresh_candidate_execution_graph(tree)
    structural_methods = {
        name
        for name, method in methods.items()
        if any(is_structural_operation(node) for node in ast.walk(method))
    }
    def reachable_from(root: str) -> set[str]:
        reachable = {root} if root in methods else set()
        pending = list(reachable)
        while pending:
            caller = pending.pop()
            for callee in calls.get(caller, ()):
                if callee in methods and callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
        return reachable

    if any(
        component.get("primitive_id") == "efficiency.closed_form_solver"
        for component in components
    ):
        (
            _solver_methods,
            _cached_solver_attrs,
            solver_consumer_methods,
        ) = _closed_form_solver_facts(
            methods,
            dotted_name=dotted_name,
            reachable_from=reachable_from,
        )
        structural_methods.update(solver_consumer_methods)

    required_paths = {
        method_name: reachable_from(method_name)
        for method_name in ("calculate_loss", "predict", "full_sort_predict")
    }
    missing_paths = tuple(
        method_name
        for method_name, reachable in required_paths.items()
        if not structural_methods.intersection(reachable)
    )
    if missing_paths:
        raise _implementation_failure(
            "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
            "declared train-graph/relation structure must enter loss, predict, "
            "and full-sort score paths; missing " + ", ".join(missing_paths),
            details=_candidate_ownership(*missing_paths),
        )

    reachable_structural_methods = set().union(
        *(
            structural_methods.intersection(reachable)
            for reachable in required_paths.values()
        )
    )
    for method_name in reachable_structural_methods:
        method = methods[method_name]
        for loop in (
            node for node in ast.walk(method) if isinstance(node, (ast.For, ast.While))
        ):
            for node in ast.walk(loop):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Name)
                    and node.targets[0].id == node.value.id
                ):
                    raise _implementation_failure(
                        "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                        "declared iterative mechanism contains an identity recurrence instead of a live update",
                        details=_candidate_ownership(method_name),
                    )

def _validate_implementation_response(
    response: Mapping[str, Any],
    *,
    policy: SharedImplementerPolicy,
    compiled_mechanism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_compiled = _normalized_compiled_mechanism(compiled_mechanism)
    contract_config = (
        policy.execution_contract.get("config", {})
        if isinstance(policy.execution_contract, Mapping)
        else {}
    )
    required_denoise_contract_weight: float | None = None
    required_denoise_contract_keys: frozenset[str] = frozenset()
    if normalized_compiled is not None and isinstance(
        policy.execution_contract, Mapping
    ):
        binding = normalized_compiled.get("implementation_binding")
        component_specs = (
            binding.get("component_specs")
            if isinstance(binding, Mapping)
            else {}
        )
        denoising_weights = {
            float(component["parameters"]["weight"])
            for component in component_specs.values()
            if isinstance(component_specs, Mapping)
            and isinstance(component, Mapping)
            and component.get("slot_id") == "DENOISING_LONG_TAIL"
            and isinstance(component.get("parameters"), Mapping)
            and isinstance(component["parameters"].get("weight"), (int, float))
            and not isinstance(component["parameters"].get("weight"), bool)
        }
        if len(denoising_weights) == 1 and isinstance(contract_config, Mapping):
            expected_weight = next(iter(denoising_weights))
            required_denoise_contract_keys = frozenset(
                str(key)
                for key, value in contract_config.items()
                if isinstance(key, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) == expected_weight
            )
            if required_denoise_contract_keys:
                required_denoise_contract_weight = expected_weight
    if not isinstance(response, Mapping) or set(response) != {
        "entrypoint",
        "files",
        "implementation_summary",
    }:
        raise _implementation_failure(
            "IMPLEMENTATION_RESPONSE_FIELDS_INVALID",
            "implementation response fields do not match the shared boundary",
        )
    summary = response["implementation_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise _implementation_failure(
            "IMPLEMENTATION_SUMMARY_MISSING",
            "implementation_summary must be non-empty",
        )
    entrypoint = _validated_entrypoint(
        response["entrypoint"],
        allowed_files=policy.allowed_files,
    )
    if (
        compiled_mechanism is not None
        and "recclaw_ext/candidate.py" in policy.allowed_files
        and entrypoint != "recclaw_ext.candidate:FreshCandidateModel"
    ):
        raise _implementation_failure(
            "MACHINE_OWNED_ENTRYPOINT_DRIFT",
            "compiled candidates use the exact machine-owned RecBole entrypoint",
        )
    raw_files = response["files"]
    if not isinstance(raw_files, list) or not 1 <= len(raw_files) <= 5:
        raise _implementation_failure(
            "IMPLEMENTATION_FILE_COUNT_INVALID",
            "implementation response must provide between one and five files",
        )
    normalized_files: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"content", "path"}:
            raise _implementation_failure(
                "IMPLEMENTATION_FILE_FIELDS_INVALID",
                "implementation file fields must be path and content",
            )
        try:
            path = validate_relative_artifact_path(str(item["path"]))
        except Exception as error:
            raise _implementation_failure(
                "IMPLEMENTATION_FILE_INVALID",
                "implementation file path is not a safe relative artifact path",
            ) from error
        content = item["content"]
        if (
            path in seen
            or path not in policy.allowed_files
            or not isinstance(content, str)
            or not content
            or "\x00" in content
        ):
            raise _implementation_failure(
                "IMPLEMENTATION_FILE_INVALID",
                "implementation file is duplicated, forbidden, or invalid UTF-8 text",
            )
        seen.add(path)
        normalized_files.append({"content": content, "path": path})
    if not seen or not seen.issubset(set(policy.allowed_files)):
        raise _implementation_failure(
            "IMPLEMENTATION_FILE_SET_MISMATCH",
            "implementation response must materialize candidate-local files only",
        )
    entrypoint_path = response["entrypoint"].split(":", 1)[0].replace(".", "/") + ".py"
    if entrypoint_path not in seen:
        raise _implementation_failure(
            "ENTRYPOINT_SOURCE_MISSING",
            "implementation response must include its entrypoint source",
        )
    execution_entrypoints: dict[str, str] = {}
    if isinstance(contract_config, Mapping):
        for field_name, value in contract_config.items():
            if (
                isinstance(field_name, str)
                and field_name.endswith("_entrypoint")
                and isinstance(value, str)
                and value.startswith("recclaw_ext.")
            ):
                validated = _validated_entrypoint(
                    value,
                    allowed_files=policy.allowed_files,
                )
                module_name, _class_name = validated.split(":", 1)
                execution_entrypoints[validated] = (
                    module_name.replace(".", "/") + ".py"
                )
    required_package_files = set(execution_entrypoints.values())
    conversion_package = "recclaw_ext/candidate.py" in policy.allowed_files
    if conversion_package:
        required_package_files.update(
            {"recclaw_ext/__init__.py", "recclaw_ext/candidate.py"}
        )
    missing = tuple(sorted(required_package_files - seen))
    candidate_entrypoint_drift = conversion_package and not str(
        response["entrypoint"]
    ).startswith("recclaw_ext.candidate:")
    if missing or candidate_entrypoint_drift:
        missing_execution = tuple(
            path for path in missing if path in execution_entrypoints.values()
        )
        raise _implementation_failure(
            (
                "EXECUTION_ENTRYPOINT_SOURCE_MISSING"
                if missing_execution
                else "REQUIRED_PACKAGE_FILES_MISSING"
            ),
            "conversion package lacks required candidate-local source: "
            + ", ".join(missing),
            details={
                "implicated_file": (
                    missing_execution[0]
                    if missing_execution
                    else missing[0]
                    if missing
                    else "recclaw_ext/candidate.py"
                ),
                "missing_files": missing,
            },
        )
    normalized_response = canonical_value(
        {
            "entrypoint": entrypoint,
            "files": sorted(normalized_files, key=lambda item: item["path"]),
            "implementation_summary": summary.strip(),
        }
    )
    _validate_source_compiled_binding(
        normalized_response,
        entrypoint=entrypoint,
        compiled_mechanism=compiled_mechanism,
        execution_config=contract_config,
    )
    if required_denoise_contract_weight is not None:
        source = next(
            item["content"]
            for item in normalized_response["files"]
            if item["path"] == entrypoint_path
        )
        tree = ast.parse(source, filename=entrypoint_path)
        methods, calls = _fresh_candidate_execution_graph(tree)
        reachable = {
            "calculate_loss",
            *_INITIALIZATION_METHOD_NAMES,
        } & set(methods)
        pending = list(reachable)
        while pending:
            for callee in calls.get(pending.pop(), ()):
                if callee in methods and callee not in reachable:
                    reachable.add(callee)
                    pending.append(callee)
        if not any(
            isinstance(node, ast.Constant)
            and node.value in required_denoise_contract_keys
            for method_name in reachable
            for node in ast.walk(methods[method_name])
        ):
            raise _implementation_failure(
                "COMPILED_MECHANISM_BEHAVIOR_MISMATCH",
                "compiled denoising weight is not consumed by the executable "
                "DENOISING_LONG_TAIL loss path",
            )
    return normalized_response


def _validate_exact_parent_child_diff(
    response: Mapping[str, Any],
    *,
    exact_parent_bundle: Mapping[str, Any],
    compiled_mechanism: Mapping[str, Any],
) -> None:
    """Verify exact-parent bytes/file provenance without guessing mechanism legality."""

    parent = _normalized_exact_parent_bundle(exact_parent_bundle)
    assert parent is not None
    parent_files = {str(item["path"]): str(item["content"]) for item in parent["files"]}
    child_files = {str(item["path"]): str(item["content"]) for item in response["files"]}
    if set(parent_files) != set(child_files):
        raise _implementation_failure(
            "EXACT_PARENT_FILE_SET_DRIFT",
            "lineage child must clone the exact parent file set",
        )
    # Compiler-owned source attribution and the ordinary disposable qualifier
    # remain the mechanism truth boundary. A handwritten AST diff taxonomy is
    # not authoritative enough to reject valid architecture, scoring, or
    # post-hoc mechanisms; lineage credit is decided from executed evidence.


def _ensure_fresh_root(root: Path, *, blind_candidate_id: str) -> None:
    if root.name != blind_candidate_id:
        raise _package_failure(
            "CANDIDATE_ROOT_IDENTITY_MISMATCH",
            "candidate root name must equal the blind candidate identity",
        )
    if os.path.lexists(root):
        raise _package_failure(
            (
                "CANDIDATE_ROOT_SYMLINK"
                if root.is_symlink()
                else "CANDIDATE_ROOT_NOT_FRESH"
            ),
            "candidate root must not exist before materialization",
        )
    parent = root.parent
    if not parent.is_dir() or parent.is_symlink():
        raise _package_failure(
            "CANDIDATE_ROOT_PARENT_INVALID",
            "candidate root parent must be an existing non-symlink directory",
        )


def _exclusive_write(root: Path, *, relative: str, content: str) -> None:
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts[:-1]:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise _package_failure(
                    "CANDIDATE_PACKAGE_PATH_UNSAFE",
                    "candidate package directory is not a real directory",
                )
            continue
        current.mkdir(mode=0o700)
    target = current / parts[-1]
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        payload = content.encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_candidate_package(
    spec: OpenResearchSpecV1,
    *,
    policy: SharedImplementerPolicy,
    implementation_response: Mapping[str, Any],
    candidate_root: Path,
    candidate_root_ref: str,
    compiled_mechanism: Mapping[str, Any] | None = None,
    exact_parent_bundle: Mapping[str, Any] | None = None,
    research_feedback: Mapping[str, Any] | None = None,
) -> MaterializedCandidate:
    """Materialize one local response into a fresh root and RC0 package."""

    request = build_shared_implementer_request(
        spec,
        policy=policy,
        compiled_mechanism=compiled_mechanism,
        exact_parent_bundle=exact_parent_bundle,
        research_feedback=research_feedback,
    )
    exact_parent_bundle = request.get("exact_parent_bundle")
    normalized_response = canonical_value(dict(implementation_response))
    if (
        policy.execution_contract is not None
        and policy.execution_contract.get("base_model_config") == "P4SparseSpectral"
    ):
        from recclaw_core.research_line.p4_implementation import bind_p4_response

        try:
            normalized_response = canonical_value(bind_p4_response(
                normalized_response,
                hook_abi=policy.execution_contract.get("config", {}).get("p4_hook_abi", 1),
            ))
        except (SyntaxError, ValueError) as error:
            raise _implementation_failure(
                "P4_RESIDUAL_ABI_INVALID", str(error),
                details={"implicated_files": ["recclaw_ext/candidate.py"]},
            ) from error
    if (
        request["service_policy"]["response_mode"]
        == PARENT_METHOD_PATCH_RESPONSE_MODE
    ):
        if exact_parent_bundle is None:
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                "parent method patch mode requires the exact parent source",
            )
        normalized_response = _expand_exact_parent_method_patch(
            implementation_response,
            exact_parent_bundle=exact_parent_bundle,
            compiled_mechanism=compiled_mechanism,
        )
    elif (
        request["service_policy"]["response_mode"]
        == SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE
    ):
        if exact_parent_bundle is None or not isinstance(
            compiled_mechanism, Mapping
        ):
            raise _implementation_failure(
                "EXACT_PARENT_BUNDLE_INVALID",
                "scaffolded semantic-ID source requires exact parent and compiler scope",
            )
        normalized_response = _expand_scaffolded_full_source(
            implementation_response,
            exact_parent_bundle=exact_parent_bundle,
            compiled_mechanism=compiled_mechanism,
        )
    elif (
        request["service_policy"]["response_mode"]
        == PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE
    ):
        if exact_parent_bundle is None or not isinstance(
            compiled_mechanism, Mapping
        ):
            raise _implementation_failure(
                "PARENT_LOCAL_SOURCE_OWNERSHIP_MISSING",
                "parent-local slot patch requires exact parent and compiler scope",
            )
        normalized_response = _expand_exact_parent_local_slot_patch(
            implementation_response,
            exact_parent_bundle=exact_parent_bundle,
            compiled_mechanism=compiled_mechanism,
        )
    elif (
        request["service_policy"]["response_mode"]
        == PROFILE_MODEL_HOOKS_RESPONSE_MODE
    ):
        if exact_parent_bundle is None or not isinstance(
            compiled_mechanism, Mapping
        ):
            raise _implementation_failure(
                "PROFILE_MODEL_SOURCE_OWNERSHIP_MISSING",
                "profile model hooks require exact parent and compiler scope",
            )
        normalized_response = _expand_profile_model_hooks(
            implementation_response,
            exact_parent_bundle=exact_parent_bundle,
            compiled_mechanism=compiled_mechanism,
        )
    response = _validate_implementation_response(
        _compiler_owned_source_binding(
            normalized_response,
            compiled_mechanism=compiled_mechanism,
        ),
        policy=policy,
        compiled_mechanism=compiled_mechanism,
    )
    if MODEL_CONFIG_MAPPING_REQUIREMENT in spec.implementation_requirements:
        module_name, class_name = response["entrypoint"].split(":", 1)
        entrypoint_path = module_name.replace(".", "/") + ".py"
        response = canonical_value({
            **response,
            "files": [
                {**item, "content": bind_model_configuration_source(item["content"], class_name)}
                if item["path"] == entrypoint_path else item
                for item in response["files"]
            ],
        })
    if exact_parent_bundle is not None:
        if isinstance(compiled_mechanism, Mapping):
            _validate_exact_parent_child_diff(
                response,
                exact_parent_bundle=exact_parent_bundle,
                compiled_mechanism=compiled_mechanism,
            )
        else:
            _validate_exact_parent_full_source_diff(
                response,
                exact_parent_bundle=exact_parent_bundle,
            )
    root = candidate_root.absolute()
    _ensure_fresh_root(
        root,
        blind_candidate_id=str(request["blind_candidate_id"]),
    )
    if (
        not isinstance(candidate_root_ref, str)
        or not candidate_root_ref
        or candidate_root_ref != candidate_root_ref.strip()
    ):
        raise _package_failure(
            "CANDIDATE_ROOT_REF_INVALID",
            "candidate_root_ref must be a normalized non-empty string",
        )

    created_root = False
    try:
        root.mkdir(mode=0o700)
        created_root = True
        for item in response["files"]:
            _exclusive_write(
                root,
                relative=str(item["path"]),
                content=str(item["content"]),
            )
        manifest = snapshot_candidate_tree(root)
        package_allowed_files = tuple(
            str(item["path"]) for item in response["files"]
        )
        if (
            any(row["path"] not in package_allowed_files for row in manifest)
            or tuple(row["path"] for row in manifest) != package_allowed_files
            or any((root / row["path"]).is_symlink() for row in manifest)
        ):
            raise _package_failure(
                "MATERIALIZED_TREE_MISMATCH",
                "materialized tree differs from the exact allowlist",
            )
        source_tree_digest, candidate_root_digest = candidate_tree_identity(
            root,
            candidate_root_ref=candidate_root_ref,
        )
        projection = origin_blind_projection(spec)
        projection_digest = sha256_digest(projection)
        implementation_receipt = canonical_value(
            {
                "blind_candidate_id": request["blind_candidate_id"],
                "entrypoint": response["entrypoint"],
                "request_digest": sha256_digest(request),
                "response_digest": sha256_digest(response),
                "schema": (
                    "recclaw.shared-implementer-materialization-receipt.v1"
                ),
                "source_tree_digest": source_tree_digest,
                "written_files": tuple(
                    {
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                    for row in manifest
                ),
            }
        )
        receipt_digest = sha256_digest(implementation_receipt)
        receipt_ref = content_id(
            "recclaw-implementation-receipt-v1",
            implementation_receipt,
        )
        package = CandidatePackageV1(
            research_spec_ref=spec.spec_id,
            research_spec_digest=spec.digest,
            protocol_ref=spec.protocol_ref,
            protocol_digest=spec.protocol_digest,
            source_tree_digest=source_tree_digest,
            candidate_root_ref=candidate_root_ref,
            candidate_root_digest=candidate_root_digest,
            executable_entrypoint=str(response["entrypoint"]),
            allowed_files=package_allowed_files,
            dependency_identity_ref=policy.dependency_identity_ref,
            dependency_identity_digest=policy.dependency_identity_digest,
            runtime_identity_ref=policy.runtime_identity_ref,
            runtime_identity_digest=policy.runtime_identity_digest,
            implementation_receipt_ref=receipt_ref,
            implementation_receipt_digest=receipt_digest,
            origin_blind_projection_digest=projection_digest,
        )
        return MaterializedCandidate(
            package=package,
            blind_projection=projection,
            shared_request=request,
            implementation_receipt=implementation_receipt,
        )
    except InnovationSpineError:
        if created_root:
            shutil.rmtree(root)
        raise
    except OSError as error:
        if created_root:
            shutil.rmtree(root)
        raise _package_failure(
            "CANDIDATE_PACKAGE_FILESYSTEM_FAILURE",
            type(error).__name__,
        ) from error
    except Exception as error:
        if created_root:
            shutil.rmtree(root)
        raise _package_failure(
            "CANDIDATE_PACKAGE_IDENTITY_FAILURE",
            type(error).__name__,
        ) from error


__all__ = [
    "InnovationSpineError",
    "MaterializedCandidate",
    "SharedImplementerPolicy",
    "bind_normalized_embedding_model_class",
    "build_shared_implementer_request",
    "materialize_candidate_package",
    "origin_blind_projection",
]
