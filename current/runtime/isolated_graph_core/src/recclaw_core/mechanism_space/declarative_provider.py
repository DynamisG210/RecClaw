"""Shared compiler for package-owned declarative mechanism families.

This module is deliberately not a dynamic plugin system.  Every family is a
Python-owned immutable specification and must still be added to the closed
catalog explicitly.  The shared engine only centralizes strict schema, graph,
type, change-budget, protocol, and evidence-obligation validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .canonical import deep_freeze, deep_thaw, domain_sha256, snapshot_json
from .contracts import CompileDiagnostic, CompileReportV1, CompileStatus, SpaceIdentity
from .spectral_selection import spectral_index_ranges

_ENGINE_VERSION = "recclaw.declarative-mechanism-provider.v1"
# Search-space identity tracks compiler semantics, not incidental source bytes.
# Bump this token only when the declarative language or materialized meaning
# changes.  A changed value intentionally requires fresh P5/P6/P7 READY identities.
_ENGINE_SEMANTIC_TOKEN = (
    "3d49a6045dc37dd89fd4da5f6c1b57d87a680da9d10f6240877ff378932e1c09"
)
_CLAIM_CEILING = "DEVELOPMENT_ONLY_SINGLE_PROTOCOL_NO_GENERAL_CLAIM"
_CONSTRUCTION_MODES = ("COMPOSITION", "ARCHITECTURE_REWRITE", "CUSTOM_MODEL")
_CHANGE_ROLES = ("CORE", "SUPPORT")
_MAX_HIGH_PERMISSION_CHANGED_SLOTS = 8
_COST_LEVELS = ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")
_FILE_ROLES = (
    "MODEL",
    "TOKENIZER",
    "RELATION_BUILDER",
    "LOSS",
    "SAMPLER",
    "TRAINER",
    "DECODER",
    "RERANKER",
    "CONFIG",
    "TEST",
)


def train_spectral_basis_rank_max(
    execution_contract: Mapping[str, Any] | None,
) -> int | None:
    """Derive the spectral solver's rank ceiling from frozen cardinalities."""

    if not isinstance(execution_contract, Mapping):
        return None
    config = execution_contract.get("config")
    if not isinstance(config, Mapping):
        return None
    snapshot: Any = None
    frozen_profile = config.get("frozen_profile")
    if isinstance(frozen_profile, Mapping):
        frozen_fields = frozen_profile.get("frozen_fields")
        if isinstance(frozen_fields, Mapping):
            snapshot = frozen_fields.get("dataset_snapshot")
    if not isinstance(snapshot, Mapping):
        # Older direct compiler callers supplied this before READY launches
        # carried the canonical frozen-family profile envelope.
        snapshot = config.get("dataset_snapshot")
    if not isinstance(snapshot, Mapping):
        return None
    users = snapshot.get("users")
    items = snapshot.get("items")
    if (
        not isinstance(users, int)
        or isinstance(users, bool)
        or users < 1
        or not isinstance(items, int)
        or isinstance(items, bool)
        or items < 1
    ):
        return None
    return min(users, items) - 1


def _execution_bound_data_diagnostics(
    payload: Mapping[str, Any],
    execution_contract: Mapping[str, Any] | None,
) -> tuple[CompileDiagnostic, ...]:
    if execution_contract is None and not any(
        component.get("primitive_id") == "state.spectral_graph_coordinates"
        or "spectral_index_ranges" in component["parameters"]
        for component in payload["components"]
    ):
        return ()
    spectral_consumers: list[tuple[str, Any]] = []
    selections: dict[str, tuple[tuple[int, int], ...]] = {}
    diagnostics: list[CompileDiagnostic] = []
    for component in payload["components"]:
        if any(
            edge["source"]["kind"] == "DATA"
            and edge["source"]["data_role"] == "TRAIN_SPECTRAL_BASIS"
            for edge in component["inputs"]
        ):
            spectral_consumers.append(
                (str(component["component_id"]), component["parameters"].get("rank"))
            )
            component_id, rank = spectral_consumers[-1]
            parameters = component["parameters"]
            ranges = parameters.get("spectral_index_ranges", ((0, rank),))
            if component.get("primitive_id") == "state.spectral_graph_coordinates":
                ranges = parameters.get("spectral_index_ranges")
            try:
                selections[component_id] = spectral_index_ranges(rank, ranges)
            except ValueError as error:
                diagnostics.append(CompileDiagnostic(
                    "TRAIN_SPECTRAL_BASIS_SELECTION_INVALID", str(error),
                    path=f"/program_payload/components/{component_id}/parameters",
                ))
    if not spectral_consumers:
        return ()

    maximum = train_spectral_basis_rank_max(execution_contract)
    if maximum is None and execution_contract is not None:
        return (
            CompileDiagnostic(
                "TRAIN_SPECTRAL_BASIS_CARDINALITY_UNAVAILABLE",
                "TRAIN_SPECTRAL_BASIS requires frozen train-user and non-padding-item cardinalities",
                path=(
                    "/execution_contract/config/frozen_profile/frozen_fields/"
                    "dataset_snapshot"
                ),
                expected={"users": "positive integer", "items": "positive integer"},
            ),
        )

    for component_id, rank in spectral_consumers:
        if (
            maximum is not None
            and isinstance(rank, int)
            and not isinstance(rank, bool)
            and rank > maximum
        ):
            diagnostics.append(
                CompileDiagnostic(
                    "TRAIN_SPECTRAL_BASIS_RANK_INFEASIBLE",
                    "declared spectral rank exceeds the execution-bound effective maximum",
                    path=f"/program_payload/components/{component_id}/parameters/rank",
                    expected={"maximum": maximum},
                    actual=rank,
                )
            )
        elif (
            maximum is not None
            and component_id in selections
            and selections[component_id][-1][1] > maximum
        ):
            diagnostics.append(CompileDiagnostic(
                "TRAIN_SPECTRAL_BASIS_DEPTH_INFEASIBLE",
                "selected columns require a spectral depth above the execution-bound maximum",
                path=f"/program_payload/components/{component_id}/parameters/spectral_index_ranges",
                expected={"maximum": maximum},
                actual=selections[component_id][-1][1],
            ))
    ranks = {
        rank
        for _component_id, rank in spectral_consumers
        if isinstance(rank, int) and not isinstance(rank, bool)
    }
    if len(ranks) > 1:
        diagnostics.append(
            CompileDiagnostic(
                "TRAIN_SPECTRAL_BASIS_RANK_CONFLICT",
                "all consumers of the shared TRAIN_SPECTRAL_BASIS must declare one rank",
                path="/program_payload/components",
                expected={"shared_rank": True},
                actual={
                    component_id: rank
                    for component_id, rank in spectral_consumers
                },
            )
        )
    elif len(set(selections.values())) > 1:
        diagnostics.append(CompileDiagnostic(
            "TRAIN_SPECTRAL_BASIS_SELECTION_CONFLICT",
            "all consumers of one shared TRAIN_SPECTRAL_BASIS must select the same columns",
            path="/program_payload/components",
            actual={key: list(value) for key, value in selections.items()},
        ))
    return tuple(diagnostics)


@dataclass(frozen=True, slots=True)
class DeclarativeFamilySpec:
    """Immutable package specification consumed by the shared compiler."""

    search_space_id: str
    search_space_version: str
    family_id: str
    family_version: str
    provider_id: str
    candidate_prefix: str
    scientific_object: str
    supported_profile_kinds: tuple[str, ...]
    allowed_data_roles: Mapping[str, str]
    forbidden_data_roles: tuple[str, ...]
    frozen_protocol_fields: tuple[str, ...]
    output_type: str
    output_slots: tuple[str, ...]
    required_slots: tuple[str, ...]
    type_compatibility: tuple[tuple[str, str], ...]
    axes: tuple[Mapping[str, Any], ...]
    operators: tuple[Mapping[str, Any], ...]
    capability_families: tuple[str, ...]
    qualification_contract: Mapping[str, Any]
    episode_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_data_roles", deep_freeze(self.allowed_data_roles))
        object.__setattr__(self, "axes", tuple(deep_freeze(item) for item in self.axes))
        object.__setattr__(self, "operators", tuple(deep_freeze(item) for item in self.operators))
        object.__setattr__(self, "qualification_contract", deep_freeze(self.qualification_contract))
        object.__setattr__(self, "episode_contract", deep_freeze(self.episode_contract))
        outcome_classes = tuple(self.episode_contract.get("outcome_classes", ()))
        outcome_lanes = self.episode_contract.get("outcome_memory_lanes", {})
        slot_update_keys = self.episode_contract.get("slot_update_keys", {})
        primitive_update_keys = self.episode_contract.get("primitive_update_keys", {})
        negative_update_keys = tuple(self.episode_contract.get("negative_update_keys", ()))
        legal_lanes = {
            "MECHANISM_POSITIVE",
            "MECHANISM_NEGATIVE",
            "OPTIMIZATION_DIAGNOSTIC",
            "PROTOCOL_DIAGNOSTIC",
            "RESOURCE_DIAGNOSTIC",
            "INCONCLUSIVE",
        }
        axis_ids = {str(item["slot_id"]) for item in self.axes}
        primitive_ids = {
            str(primitive_spec["primitive_id"])
            for axis_spec in self.axes
            for primitive_spec in axis_spec["primitives"]
        }
        if not outcome_classes or set(outcome_lanes) != set(outcome_classes):
            raise ValueError("episode outcome classes require an exact memory-lane mapping")
        if set(outcome_lanes.values()) - legal_lanes:
            raise ValueError("episode outcome memory lane is not recognized")
        if not negative_update_keys or len(negative_update_keys) != len(set(negative_update_keys)):
            raise ValueError("episode negative_update_keys must be non-empty and unique")
        if set(slot_update_keys) != axis_ids:
            raise ValueError("episode slot_update_keys must cover every family axis exactly")
        for slot_id, keys in slot_update_keys.items():
            if not keys or set(keys) - set(negative_update_keys):
                raise ValueError(f"episode slot update keys are invalid for {slot_id}")
        if set(primitive_update_keys) - primitive_ids:
            raise ValueError("episode primitive update keys reference an unknown primitive")
        for primitive_id, keys in primitive_update_keys.items():
            if not keys or set(keys) - set(negative_update_keys):
                raise ValueError(
                    f"episode primitive update keys are invalid for {primitive_id}"
                )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "engine_version": _ENGINE_VERSION,
            "engine_source_sha256": _ENGINE_SEMANTIC_TOKEN,
            "search_space_id": self.search_space_id,
            "search_space_version": self.search_space_version,
            "family_id": self.family_id,
            "family_version": self.family_version,
            "provider_id": self.provider_id,
            "candidate_prefix": self.candidate_prefix,
            "scientific_object": self.scientific_object,
            "supported_profile_kinds": list(self.supported_profile_kinds),
            "allowed_data_roles": deep_thaw(self.allowed_data_roles),
            "forbidden_data_roles": list(self.forbidden_data_roles),
            "frozen_protocol_fields": list(self.frozen_protocol_fields),
            "output_type": self.output_type,
            "output_slots": list(self.output_slots),
            "required_slots": list(self.required_slots),
            "type_compatibility": [list(item) for item in self.type_compatibility],
            "axes": deep_thaw(self.axes),
            "operators": deep_thaw(self.operators),
            "capability_families": list(self.capability_families),
            "qualification_contract": deep_thaw(self.qualification_contract),
            "episode_contract": deep_thaw(self.episode_contract),
        }


def primitive(
    primitive_id: str,
    slot_id: str,
    *,
    inputs: Sequence[tuple[str, Sequence[str], int]] = (),
    outputs: Sequence[tuple[str, str]] = (),
    parameters: Mapping[str, Any] | None = None,
    capabilities: Sequence[str] = (),
    causal_effect: str,
    failure_signal: str,
    resource_effect: str,
    implementation_status: str = "LOCAL_IMPLEMENTATION_REQUIRED",
) -> dict[str, Any]:
    """Construct one typed research primitive without model-name aliases."""

    return {
        "primitive_id": primitive_id,
        "slot_id": slot_id,
        "input_ports": [
            {"port": port, "accepted_types": list(types), "minimum": minimum}
            for port, types, minimum in inputs
        ],
        "output_ports": [{"port": port, "type": type_ref} for port, type_ref in outputs],
        "parameter_schema": dict(parameters or _empty_parameters()),
        "capabilities": sorted(set(capabilities)),
        "causal_effect": causal_effect,
        "failure_signal": failure_signal,
        "resource_effect": resource_effect,
        "implementation_status": implementation_status,
    }


def axis(
    slot_id: str,
    *,
    research_role: str,
    interaction_guidance: str,
    allow_multiple: bool,
    primitives: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "slot_id": slot_id,
        "research_role": research_role,
        "interaction_guidance": interaction_guidance,
        "allow_multiple": allow_multiple,
        "primitives": [dict(item) for item in primitives],
    }


def operator(
    operator_id: str,
    *,
    capabilities: Sequence[str],
    requires_target: bool,
    requires_replacement: bool,
    research_use: str,
) -> dict[str, Any]:
    return {
        "operator_id": operator_id,
        "capabilities": sorted(set(capabilities)),
        "requires_target": requires_target,
        "requires_replacement": requires_replacement,
        "research_use": research_use,
    }


def object_parameters(
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(required),
    }


def _empty_parameters() -> dict[str, Any]:
    return object_parameters({})


def standard_operators(
    *,
    structure_capability: str,
    efficiency_capability: str,
    training_capability: str,
) -> tuple[dict[str, Any], ...]:
    return (
        operator(
            "add_component",
            capabilities=(structure_capability,),
            requires_target=False,
            requires_replacement=False,
            research_use="Add one typed mechanism while preserving the matched parent path.",
        ),
        operator(
            "replace_component",
            capabilities=(structure_capability,),
            requires_target=True,
            requires_replacement=True,
            research_use="Replace an existing causal component with a declared alternative.",
        ),
        operator(
            "split_path",
            capabilities=(structure_capability,),
            requires_target=True,
            requires_replacement=True,
            research_use="Create explicit short/long, sparse/residual, or local/global branches.",
        ),
        operator(
            "fuse_paths",
            capabilities=(structure_capability,),
            requires_target=True,
            requires_replacement=True,
            research_use="Fuse declared branches with an ablatable fusion mechanism.",
        ),
        operator(
            "precompute_operator",
            capabilities=(efficiency_capability,),
            requires_target=True,
            requires_replacement=False,
            research_use="Move a train-derived operator out of the update loop without changing data access.",
        ),
        operator(
            "alternate_optimization",
            capabilities=(training_capability,),
            requires_target=True,
            requires_replacement=False,
            research_use="Alternate declared stages whose budgets remain separately measured.",
        ),
        operator(
            "synthesize_custom_model",
            capabilities=("CUSTOM_MODEL_IMPLEMENTATION",),
            requires_target=False,
            requires_replacement=False,
            research_use="Use the disciplined custom escape only for a genuinely missing causal mechanism.",
        ),
    )


def _schema_diagnostics(
    instance: Any,
    schema: Mapping[str, Any],
    *,
    code: str,
) -> list[CompileDiagnostic]:
    schema_value = deep_thaw(schema)
    try:
        Draft202012Validator.check_schema(schema_value)
    except SchemaError as exc:
        return [CompileDiagnostic("PACKAGE_SCHEMA_INVALID", "package schema is invalid", actual=type(exc).__name__)]
    diagnostics: list[CompileDiagnostic] = []
    for error in sorted(
        Draft202012Validator(schema_value).iter_errors(instance),
        key=lambda item: list(item.absolute_path),
    ):
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        diagnostics.append(CompileDiagnostic(code, error.message, path=path if path != "/" else "/program_payload"))
    return diagnostics


def _topological_order(
    component_ids: Sequence[str], edges: Mapping[str, set[str]]
) -> tuple[list[str], bool]:
    indegree = {component_id: 0 for component_id in component_ids}
    downstream: dict[str, set[str]] = defaultdict(set)
    for target, sources in edges.items():
        indegree[target] += len(sources)
        for source in sources:
            downstream[source].add(target)
    queue = deque(sorted(item for item, count in indegree.items() if count == 0))
    result: list[str] = []
    while queue:
        current = queue.popleft()
        result.append(current)
        for target in sorted(downstream.get(current, ())):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return result, len(result) != len(component_ids)


class DeclarativeMechanismSpaceProvider:
    """Closed Provider implementation for an explicitly catalogued family."""

    def __init__(self, spec: DeclarativeFamilySpec):
        self._spec = spec
        self.provider_id = spec.provider_id
        axes: dict[str, Mapping[str, Any]] = {}
        primitives: dict[str, Mapping[str, Any]] = {}
        for axis_spec in spec.axes:
            slot_id = str(axis_spec["slot_id"])
            if slot_id in axes:
                raise ValueError(f"duplicate axis: {slot_id}")
            axes[slot_id] = axis_spec
            for item in axis_spec["primitives"]:
                primitive_id = str(item["primitive_id"])
                if primitive_id in primitives:
                    raise ValueError(f"duplicate primitive: {primitive_id}")
                if item["slot_id"] != slot_id:
                    raise ValueError(f"primitive slot mismatch: {primitive_id}")
                primitives[primitive_id] = item
        operators = {str(item["operator_id"]): item for item in spec.operators}
        if len(operators) != len(spec.operators):
            raise ValueError("duplicate declarative architecture operator")
        self._axes = MappingProxyType(axes)
        self._primitives = MappingProxyType(primitives)
        self._operators = MappingProxyType(operators)
        self._compatibility = frozenset(spec.type_compatibility)
        self._schema = deep_freeze(self._build_payload_schema())

    @property
    def spec(self) -> DeclarativeFamilySpec:
        return self._spec

    def identity(self) -> SpaceIdentity:
        digest = domain_sha256("recclaw.declarative-search-space-closure.v1", self._spec.identity_payload())
        return SpaceIdentity(
            search_space_id=self._spec.search_space_id,
            search_space_version=self._spec.search_space_version,
            search_space_digest=digest,
            family_id=self._spec.family_id,
            family_version=self._spec.family_version,
            provider_id=self.provider_id,
        )

    def prompt_projection(self) -> Mapping[str, Any]:
        identity = self.identity()
        return deep_freeze(
            {
                "projection_version": "recclaw.declarative-prompt-projection.v1",
                "space_identity": identity.to_dict(),
                "scientific_object": self._spec.scientific_object,
                "allowed_data_roles": [
                    {"role_id": key, "type_ref": self._spec.allowed_data_roles[key], "derivation_scope": "TRAIN_ONLY"}
                    for key in sorted(self._spec.allowed_data_roles)
                ],
                "forbidden_data_roles": list(self._spec.forbidden_data_roles),
                "frozen_protocol_fields": list(self._spec.frozen_protocol_fields),
                "output_contract": {"type_ref": self._spec.output_type, "candidate_universe_effect": "PRESERVE"},
                "axes": [deep_thaw(self._axes[key]) for key in sorted(self._axes)],
                "architecture_operators": [deep_thaw(self._operators[key]) for key in sorted(self._operators)],
                "proposal_obligations": {
                    "core_changes": 1,
                    "maximum_support_changes": 1,
                    "requires_matched_control": True,
                    "requires_mechanism_off_ablation": True,
                    "requires_discriminating_predictions": True,
                    "requires_failure_classification": True,
                    "static_candidate_menu_forbidden": True,
                    "hidden_fallback_forbidden": True,
                },
                "qualification_contract": deep_thaw(self._spec.qualification_contract),
                "episode_contract": deep_thaw(self._spec.episode_contract),
                "claim_ceiling": _CLAIM_CEILING,
            }
        )

    def program_schema(self) -> Mapping[str, Any]:
        identity = self.identity()
        return deep_freeze(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "record_type": {"const": "MECHANISM_PROGRAM_ENVELOPE"},
                    "kernel_schema_version": {"const": "recclaw.mechanism-space.kernel.v1"},
                    "search_space_id": {"const": identity.search_space_id},
                    "search_space_digest": {"const": identity.search_space_digest},
                    "family_id": {"const": identity.family_id},
                    "family_version": {"const": identity.family_version},
                    "profile_ref": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "profile_id": {"type": "string", "minLength": 1},
                            "profile_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                            "profile_kind": {"enum": list(self._spec.supported_profile_kinds)},
                        },
                        "required": ["profile_id", "profile_digest", "profile_kind"],
                    },
                    "program_payload": deep_thaw(self._schema),
                },
                "required": [
                    "record_type",
                    "kernel_schema_version",
                    "search_space_id",
                    "search_space_digest",
                    "family_id",
                    "family_version",
                    "profile_ref",
                    "program_payload",
                ],
            }
        )

    def compile(
        self,
        envelope: Mapping[str, Any],
        *,
        execution_contract: Mapping[str, Any] | None = None,
    ) -> CompileReportV1:
        identity = self.identity()
        payload = snapshot_json(envelope["program_payload"])
        diagnostics = _schema_diagnostics(payload, self._schema, code="FAMILY_PROGRAM_SCHEMA_INVALID")
        profile_kind = envelope["profile_ref"]["profile_kind"]
        if profile_kind not in self._spec.supported_profile_kinds:
            diagnostics.append(
                CompileDiagnostic(
                    "UNSUPPORTED_PROFILE",
                    "profile kind is outside this family contract",
                    path="/profile_ref/profile_kind",
                    expected=list(self._spec.supported_profile_kinds),
                    actual=profile_kind,
                )
            )
        if diagnostics:
            return CompileReportV1(CompileStatus.INVALID, diagnostics=tuple(diagnostics), space_identity=identity)
        diagnostics.extend(
            _execution_bound_data_diagnostics(payload, execution_contract)
        )
        if diagnostics:
            return CompileReportV1(
                CompileStatus.INVALID,
                diagnostics=tuple(diagnostics),
                space_identity=identity,
            )

        components = {str(item["component_id"]): item for item in payload["components"]}
        if len(components) != len(payload["components"]):
            diagnostics.append(CompileDiagnostic("DUPLICATE_COMPONENT_ID", "component_id values must be unique", path="/program_payload/components"))
        custom_specs = {str(item["custom_component_id"]): item for item in payload["custom_components"]}
        if len(custom_specs) != len(payload["custom_components"]):
            diagnostics.append(CompileDiagnostic("DUPLICATE_CUSTOM_COMPONENT_ID", "custom_component_id values must be unique", path="/program_payload/custom_components"))

        component_specs: dict[str, Mapping[str, Any]] = {}
        component_outputs: dict[str, dict[str, str]] = {}
        slot_components: dict[str, list[str]] = defaultdict(list)
        required_capabilities: set[str] = set()
        used_custom: set[str] = set()
        for component_id, component in components.items():
            slot_id = str(component["slot_id"])
            slot_components[slot_id].append(component_id)
            spec: Mapping[str, Any] | None
            if "primitive_id" in component:
                primitive_id = str(component["primitive_id"])
                spec = self._primitives.get(primitive_id)
                if spec is None:
                    diagnostics.append(CompileDiagnostic("UNKNOWN_PRIMITIVE", "primitive is not in this family registry", path=f"/program_payload/components/{component_id}/primitive_id", actual=primitive_id))
                    continue
                if spec["slot_id"] != slot_id:
                    diagnostics.append(CompileDiagnostic("PRIMITIVE_SLOT_MISMATCH", "primitive is instantiated in the wrong slot", path=f"/program_payload/components/{component_id}/slot_id", expected=spec["slot_id"], actual=slot_id))
                for error in _schema_diagnostics(component["parameters"], spec["parameter_schema"], code="PRIMITIVE_PARAMETERS_INVALID"):
                    diagnostics.append(CompileDiagnostic(error.code, error.message, path=f"/program_payload/components/{component_id}/parameters{error.path}"))
                required_capabilities.update(spec["capabilities"])
            else:
                custom_id = str(component["custom_component_id"])
                custom = custom_specs.get(custom_id)
                if custom is None:
                    diagnostics.append(CompileDiagnostic("UNKNOWN_CUSTOM_COMPONENT", "custom component is undeclared", path=f"/program_payload/components/{component_id}/custom_component_id", actual=custom_id))
                    continue
                used_custom.add(custom_id)
                if custom["slot_id"] != slot_id:
                    diagnostics.append(CompileDiagnostic("CUSTOM_COMPONENT_SLOT_MISMATCH", "custom declaration and instance use different slots", path=f"/program_payload/components/{component_id}/slot_id"))
                spec = {
                    "input_ports": custom["input_ports"],
                    "output_ports": custom["output_ports"],
                    "capabilities": ["CUSTOM_MODEL_IMPLEMENTATION"],
                    "allowed_read_roles": custom["allowed_read_roles"],
                }
                required_capabilities.add("CUSTOM_MODEL_IMPLEMENTATION")
                implementation = custom["minimal_implementation"]
                if implementation["entrypoint_role"] not in implementation["file_roles"]:
                    diagnostics.append(CompileDiagnostic("CUSTOM_ENTRYPOINT_ROLE_NOT_DECLARED", "custom entrypoint role must appear in file_roles", path=f"/program_payload/custom_components/{custom_id}/minimal_implementation"))
            component_specs[component_id] = spec
            component_outputs[component_id] = {str(item["port"]): str(item["type"]) for item in spec["output_ports"]}

        unused_custom = sorted(set(custom_specs) - used_custom)
        if unused_custom:
            diagnostics.append(CompileDiagnostic("UNUSED_CUSTOM_COMPONENT", "every custom declaration must be instantiated", path="/program_payload/custom_components", actual=unused_custom))
        self._validate_change_budget(payload, diagnostics)
        self._validate_slots(slot_components, diagnostics)
        self._validate_operators(payload, set(components), required_capabilities, diagnostics)

        edges: dict[str, set[str]] = defaultdict(set)
        used_data_roles: set[str] = set()
        for component_id, component in components.items():
            spec = component_specs.get(component_id)
            if spec is None:
                continue
            counts = Counter(str(item["port"]) for item in component["inputs"])
            port_specs = {str(item["port"]): item for item in spec["input_ports"]}
            for port, count in counts.items():
                if port not in port_specs:
                    diagnostics.append(CompileDiagnostic("UNKNOWN_INPUT_PORT", "component supplies an undeclared input port", path=f"/program_payload/components/{component_id}/inputs", actual=port))
            for port, port_spec in port_specs.items():
                if counts.get(port, 0) < int(port_spec["minimum"]):
                    diagnostics.append(CompileDiagnostic("MISSING_REQUIRED_INPUT", "component lacks a required typed input", path=f"/program_payload/components/{component_id}/inputs", expected={"port": port, "minimum": port_spec["minimum"]}, actual=counts.get(port, 0)))
            for input_item in component["inputs"]:
                port = str(input_item["port"])
                if port not in port_specs:
                    continue
                source = input_item["source"]
                provided: str | None = None
                if source["kind"] == "DATA":
                    role = str(source["data_role"])
                    used_data_roles.add(role)
                    provided = self._spec.allowed_data_roles.get(role)
                    if provided is None:
                        diagnostics.append(CompileDiagnostic("FORBIDDEN_OR_UNKNOWN_DATA_ROLE", "data role is outside the family contract", path=f"/program_payload/components/{component_id}/inputs", actual=role))
                    elif "custom_component_id" in component and role not in spec.get("allowed_read_roles", ()):
                        diagnostics.append(CompileDiagnostic("CUSTOM_DATA_ROLE_NOT_DECLARED", "custom component reads a role absent from allowed_read_roles", path=f"/program_payload/components/{component_id}/inputs", actual=role))
                else:
                    source_id = str(source["component_id"])
                    output_port = str(source["output_port"])
                    if source_id not in component_outputs:
                        diagnostics.append(CompileDiagnostic("UNKNOWN_SOURCE_COMPONENT", "input references an unknown component", path=f"/program_payload/components/{component_id}/inputs", actual=source_id))
                    elif output_port not in component_outputs[source_id]:
                        diagnostics.append(CompileDiagnostic("UNKNOWN_SOURCE_PORT", "input references an unknown output port", path=f"/program_payload/components/{component_id}/inputs", actual=output_port))
                    else:
                        provided = component_outputs[source_id][output_port]
                        edges[component_id].add(source_id)
                if provided is not None and not any(self._type_compatible(provided, str(accepted)) for accepted in port_specs[port]["accepted_types"]):
                    diagnostics.append(CompileDiagnostic("INPUT_TYPE_MISMATCH", "typed mechanism edge is incompatible", path=f"/program_payload/components/{component_id}/inputs", expected=list(port_specs[port]["accepted_types"]), actual=provided))

        order, has_cycle = _topological_order(tuple(components), edges)
        if has_cycle:
            diagnostics.append(CompileDiagnostic("MECHANISM_GRAPH_CYCLE", "mechanism component graph must be acyclic", path="/program_payload/components"))
        output_components = [component_id for slot in self._spec.output_slots for component_id in slot_components.get(slot, ())]
        if not any(self._spec.output_type in component_outputs.get(item, {}).values() for item in output_components):
            diagnostics.append(CompileDiagnostic("OUTPUT_CONTRACT_MISMATCH", "family output slot does not produce the frozen output type", expected=self._spec.output_type, actual=output_components))
        declared_roles = set(payload["declared_data_roles"])
        if declared_roles != used_data_roles:
            diagnostics.append(CompileDiagnostic("DECLARED_DATA_ROLES_MISMATCH", "declared_data_roles must exactly match data roles used by the graph", path="/program_payload/declared_data_roles", expected=sorted(used_data_roles), actual=sorted(declared_roles)))
        objective_components = [
            component_id
            for component_id, outputs in component_outputs.items()
            if any(type_ref.endswith("/objective") for type_ref in outputs.values())
        ]
        evidence_sinks = sorted(set(output_components) | set(objective_components))
        connected_to_output = set(evidence_sinks)
        queue = deque(evidence_sinks)
        while queue:
            current = queue.popleft()
            for source_id in sorted(edges.get(current, ())):
                if source_id not in connected_to_output:
                    connected_to_output.add(source_id)
                    queue.append(source_id)
        core_slots = {str(item["slot_id"]) for item in payload["changed_slots"] if item["change_role"] == "CORE"}
        disconnected_core = sorted(
            component_id
            for slot_id in core_slots
            for component_id in slot_components.get(slot_id, ())
            if component_id not in connected_to_output
        )
        if disconnected_core:
            diagnostics.append(CompileDiagnostic("CORE_MECHANISM_NOT_ON_OUTPUT_PATH", "core mechanism must be causally connected to the family output", path="/program_payload/changed_slots", actual=disconnected_core))
        for index, ablation in enumerate(payload["ablation_plan"]):
            unknown = sorted(set(ablation["remove_component_ids"]) - set(components))
            if unknown:
                diagnostics.append(CompileDiagnostic("UNKNOWN_ABLATION_COMPONENT", "ablation must remove final mechanism components", path=f"/program_payload/ablation_plan/{index}/remove_component_ids", actual=unknown))
        core_component_ids = {
            component_id
            for slot_id in core_slots
            for component_id in slot_components.get(slot_id, ())
        }
        if not any(
            core_component_ids.intersection(ablation["remove_component_ids"])
            for ablation in payload["ablation_plan"]
        ):
            diagnostics.append(
                CompileDiagnostic(
                    "CORE_MECHANISM_ABLATION_MISSING",
                    "at least one ablation must switch off a component in the declared core slot",
                    path="/program_payload/ablation_plan",
                    expected=sorted(core_component_ids),
                )
            )

        protocol_impact = payload["protocol_impact"]
        protocol_branch = protocol_impact["status"] == "BRANCH_REQUIRED"
        if protocol_impact["status"] == "UNCHANGED" and protocol_impact["requested_changes"]:
            diagnostics.append(CompileDiagnostic("PROTOCOL_IMPACT_INCONSISTENT", "UNCHANGED cannot request frozen protocol changes", path="/program_payload/protocol_impact"))
        if protocol_branch and not protocol_impact["requested_changes"]:
            diagnostics.append(CompileDiagnostic("PROTOCOL_BRANCH_REASON_MISSING", "BRANCH_REQUIRED must name changed frozen fields", path="/program_payload/protocol_impact/requested_changes"))
        unknown_capabilities = sorted(required_capabilities - set(self._spec.capability_families))
        if unknown_capabilities:
            diagnostics.append(CompileDiagnostic("UNKNOWN_CAPABILITY", "compiler derived capability outside family policy", actual=unknown_capabilities))

        program_digest = domain_sha256(f"{self._spec.family_id}.mechanism-program.v1", envelope)
        candidate_id = f"{self._spec.candidate_prefix}_{program_digest[:20]}"
        semantic_digest: str | None = None
        resolved_ir: dict[str, Any] | None = None
        if not diagnostics and not has_cycle and len(component_specs) == len(components):
            semantic_digest = self._semantic_digest(
                components,
                component_specs,
                custom_specs,
                order,
                payload,
                envelope,
            )
            resolved_ir = {
                "ir_version": "recclaw.declarative-resolved-mechanism-ir.v1",
                "authority": "NONE",
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
                "candidate_id": candidate_id,
                "search_space_id": identity.search_space_id,
                "search_space_digest": identity.search_space_digest,
                "family_id": identity.family_id,
                "profile_ref": snapshot_json(envelope["profile_ref"]),
                "mechanism_program_digest": program_digest,
                "mechanism_semantics_digest": semantic_digest,
                "component_order": order,
                "components": [snapshot_json(components[item]) for item in order],
                "custom_component_definitions": [snapshot_json(custom_specs[item]) for item in sorted(custom_specs)],
                "architecture_operators": snapshot_json(payload["architecture_operators"]),
                "changed_slots": snapshot_json(payload["changed_slots"]),
                "required_capabilities": sorted(required_capabilities),
                "implementation_requirement": "CANDIDATE_LOCAL_IMPLEMENTATION_AND_QUALIFICATION_REQUIRED",
                "candidate_root": f"recclaw_ext/generated/{candidate_id}/",
                "qualification_contract": deep_thaw(self._spec.qualification_contract),
                "episode_contract": deep_thaw(self._spec.episode_contract),
                "protocol_impact": snapshot_json(protocol_impact),
                "claim_ceiling": _CLAIM_CEILING,
            }
        if diagnostics:
            return CompileReportV1(CompileStatus.INVALID, diagnostics=tuple(diagnostics), space_identity=identity, candidate_id=candidate_id, mechanism_semantics_digest=semantic_digest, mechanism_program_digest=program_digest, required_capabilities=tuple(sorted(required_capabilities)), resolved_ir=resolved_ir)
        if protocol_branch:
            return CompileReportV1(
                CompileStatus.PROTOCOL_BRANCH_REQUIRED,
                diagnostics=(CompileDiagnostic("PROTOCOL_BRANCH_REQUIRED", "proposal changes frozen family protocol and is not runnable in this profile", path="/program_payload/protocol_impact/requested_changes", actual=protocol_impact["requested_changes"]),),
                space_identity=identity,
                candidate_id=candidate_id,
                mechanism_semantics_digest=semantic_digest,
                mechanism_program_digest=program_digest,
                required_capabilities=tuple(sorted(required_capabilities)),
                resolved_ir=resolved_ir,
            )
        return CompileReportV1(CompileStatus.VALID_NEEDS_IMPLEMENTATION, space_identity=identity, candidate_id=candidate_id, mechanism_semantics_digest=semantic_digest, mechanism_program_digest=program_digest, required_capabilities=tuple(sorted(required_capabilities)), resolved_ir=resolved_ir)

    def _type_compatible(self, provided: str, accepted: str) -> bool:
        return provided == accepted or (provided, accepted) in self._compatibility

    def _validate_change_budget(self, payload: Mapping[str, Any], diagnostics: list[CompileDiagnostic]) -> None:
        changes = payload["changed_slots"]
        slots = [str(item["slot_id"]) for item in changes]
        if len(slots) != len(set(slots)):
            diagnostics.append(CompileDiagnostic("DUPLICATE_CHANGED_SLOT", "a slot may appear only once in changed_slots", path="/program_payload/changed_slots"))
        core_count = sum(item["change_role"] == "CORE" for item in changes)
        support_count = sum(item["change_role"] == "SUPPORT" for item in changes)
        if payload["construction_mode"] == "COMPOSITION":
            if core_count != 1 or support_count > 1:
                diagnostics.append(CompileDiagnostic("CHANGE_BUDGET_INVALID", "COMPOSITION requires exactly one core and at most one support change", path="/program_payload/changed_slots", expected={"core": 1, "maximum_support": 1}, actual={"core": core_count, "support": support_count}))
        elif core_count < 1:
            diagnostics.append(CompileDiagnostic("CHANGE_BUDGET_INVALID", "high-permission construction requires at least one core change", path="/program_payload/changed_slots", expected={"minimum_core": 1, "maximum_total": _MAX_HIGH_PERMISSION_CHANGED_SLOTS}, actual={"core": core_count, "support": support_count}))
        custom_count = len(payload["custom_components"])
        synthesize = any(item["operator_id"] == "synthesize_custom_model" for item in payload["architecture_operators"])
        if payload["construction_mode"] == "CUSTOM_MODEL":
            if custom_count == 0 or not synthesize:
                diagnostics.append(CompileDiagnostic("CUSTOM_MODEL_ESCAPE_INCOMPLETE", "CUSTOM_MODEL requires a custom declaration and synthesize_custom_model operator", path="/program_payload"))
        elif custom_count or synthesize:
            diagnostics.append(CompileDiagnostic("CUSTOM_MODEL_MODE_REQUIRED", "custom mechanisms require CUSTOM_MODEL construction mode", path="/program_payload/construction_mode"))

    def _validate_slots(self, slot_components: Mapping[str, list[str]], diagnostics: list[CompileDiagnostic]) -> None:
        for slot_id in self._spec.required_slots:
            if not slot_components.get(slot_id):
                diagnostics.append(CompileDiagnostic("REQUIRED_SLOT_MISSING", "family mechanism graph lacks a required slot", path="/program_payload/components", expected=slot_id))
        for slot_id, component_ids in slot_components.items():
            axis_spec = self._axes.get(slot_id)
            if axis_spec is None:
                diagnostics.append(CompileDiagnostic("UNKNOWN_SLOT", "component uses a slot outside the family registry", path="/program_payload/components", actual=slot_id))
            elif not axis_spec["allow_multiple"] and len(component_ids) > 1:
                diagnostics.append(CompileDiagnostic("SLOT_CARDINALITY_EXCEEDED", "slot permits at most one component", path="/program_payload/components", expected=1, actual={slot_id: len(component_ids)}))

    def _validate_operators(self, payload: Mapping[str, Any], component_ids: set[str], capabilities: set[str], diagnostics: list[CompileDiagnostic]) -> None:
        removed_slots = set(payload["removed_slots"])
        for index, item in enumerate(payload["architecture_operators"]):
            operator_spec = self._operators.get(str(item["operator_id"]))
            if operator_spec is None:
                diagnostics.append(CompileDiagnostic("UNKNOWN_ARCHITECTURE_OPERATOR", "operator is outside the family grammar", path=f"/program_payload/architecture_operators/{index}/operator_id", actual=item["operator_id"]))
                continue
            capabilities.update(operator_spec["capabilities"])
            if operator_spec["requires_target"] and not item["targets"]:
                diagnostics.append(CompileDiagnostic("OPERATOR_TARGET_REQUIRED", "operator requires at least one target", path=f"/program_payload/architecture_operators/{index}/targets"))
            if operator_spec["requires_replacement"] and not item["replacements"]:
                diagnostics.append(CompileDiagnostic("OPERATOR_REPLACEMENT_REQUIRED", "operator requires an explicit replacement", path=f"/program_payload/architecture_operators/{index}/replacements"))
            for target in item["targets"]:
                if str(target).startswith("slot:"):
                    if str(target).split(":", 1)[1] not in removed_slots:
                        diagnostics.append(CompileDiagnostic("REMOVED_SLOT_NOT_DECLARED", "slot target must also be declared removed", path=f"/program_payload/architecture_operators/{index}/targets", actual=target))
                elif target not in component_ids:
                    diagnostics.append(CompileDiagnostic("UNKNOWN_OPERATOR_TARGET", "operator target is not a final component", path=f"/program_payload/architecture_operators/{index}/targets", actual=target))
            for replacement in item["replacements"]:
                if replacement not in component_ids:
                    diagnostics.append(CompileDiagnostic("UNKNOWN_OPERATOR_REPLACEMENT", "operator replacement is not a final component", path=f"/program_payload/architecture_operators/{index}/replacements", actual=replacement))

    def _semantic_digest(
        self,
        components: Mapping[str, Mapping[str, Any]],
        specs: Mapping[str, Mapping[str, Any]],
        custom_specs: Mapping[str, Mapping[str, Any]],
        order: Sequence[str],
        payload: Mapping[str, Any],
        envelope: Mapping[str, Any],
    ) -> str:
        hashes: dict[str, str] = {}
        for component_id in order:
            component = components[component_id]
            inputs: list[dict[str, Any]] = []
            for input_item in component["inputs"]:
                source = input_item["source"]
                source_identity: Any = (
                    {"data_role": source["data_role"]}
                    if source["kind"] == "DATA"
                    else {"component_semantics": hashes[source["component_id"]], "output_port": source["output_port"]}
                )
                inputs.append({"port": input_item["port"], "source": source_identity})
            inputs.sort(key=lambda item: (item["port"], str(item["source"])))
            if "primitive_id" in component:
                implementation = {"primitive_id": component["primitive_id"]}
            else:
                custom = custom_specs[str(component["custom_component_id"])]
                implementation = {
                    "custom_component_semantics": {
                        "mathematical_definition": custom[
                            "mathematical_definition"
                        ],
                        "algorithm_definition": custom["algorithm_definition"],
                        "input_ports": sorted(
                            (
                                {
                                    **dict(port),
                                    "accepted_types": sorted(
                                        port["accepted_types"]
                                    ),
                                }
                                for port in custom["input_ports"]
                            ),
                            key=lambda port: port["port"],
                        ),
                        "output_ports": sorted(
                            custom["output_ports"],
                            key=lambda port: (port["port"], port["type"]),
                        ),
                        "allowed_read_roles": sorted(
                            custom["allowed_read_roles"]
                        ),
                    }
                }
            hashes[component_id] = domain_sha256(
                "recclaw.declarative-component-semantics.v1",
                {
                    "implementation": implementation,
                    "slot_id": component["slot_id"],
                    "inputs": inputs,
                    "parameters": component["parameters"],
                    "outputs": deep_thaw(specs[component_id]["output_ports"]),
                },
            )
        def operator_target(value: Any) -> Any:
            if isinstance(value, str) and value in hashes:
                return {"component_semantics": hashes[value]}
            return value

        semantic_operators = sorted(
            (
                {
                    "operator_id": item["operator_id"],
                    "targets": sorted(
                        (operator_target(value) for value in item["targets"]),
                        key=str,
                    ),
                    "replacements": sorted(
                        (
                            operator_target(value)
                            for value in item["replacements"]
                        ),
                        key=str,
                    ),
                    "parameters": item["parameters"],
                }
                for item in payload["architecture_operators"]
            ),
            key=str,
        )
        return domain_sha256(
            f"{self._spec.family_id}.mechanism-semantics.v1",
            {
                "profile_ref": envelope["profile_ref"],
                "search_space_digest": self.identity().search_space_digest,
                "construction_mode": payload["construction_mode"],
                "component_semantics": sorted(hashes.values()),
                "changed_slots": sorted(payload["changed_slots"], key=lambda item: (item["slot_id"], item["change_role"])),
                "removed_slots": sorted(payload["removed_slots"]),
                "operators": semantic_operators,
                "protocol_impact": payload["protocol_impact"],
            },
        )

    def _build_payload_schema(self) -> dict[str, Any]:
        slots = sorted(self._axes)
        roles = sorted(self._spec.allowed_data_roles)
        operators = sorted(self._operators)
        type_pattern = "^[a-z][a-z0-9_]*/[a-z0-9_]+$"
        component_source = {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"kind": {"const": "DATA"}, "data_role": {"enum": roles}},
                    "required": ["kind", "data_role"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"const": "COMPONENT"},
                        "component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                        "output_port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                    },
                    "required": ["kind", "component_id", "output_port"],
                },
            ]
        }
        port_input = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "source": component_source,
            },
            "required": ["port", "source"],
        }
        component = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "slot_id": {"enum": slots},
                "primitive_id": {"type": "string", "pattern": "^[a-z][a-z0-9_.-]{2,127}$"},
                "custom_component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "inputs": {"type": "array", "items": port_input},
                "parameters": {"type": "object"},
            },
            "required": ["component_id", "slot_id", "inputs", "parameters"],
            "oneOf": [
                {"required": ["primitive_id"], "not": {"required": ["custom_component_id"]}},
                {"required": ["custom_component_id"], "not": {"required": ["primitive_id"]}},
            ],
        }
        custom_component = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "custom_component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "slot_id": {"enum": slots},
                "mathematical_definition": {"type": "string", "minLength": 20},
                "algorithm_definition": {"type": "string", "minLength": 20},
                "input_ports": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                            "accepted_types": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "pattern": type_pattern}},
                            "minimum": {"type": "integer", "minimum": 0, "maximum": 8},
                        },
                        "required": ["port", "accepted_types", "minimum"],
                    },
                },
                "output_ports": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                            "type": {"type": "string", "pattern": type_pattern},
                        },
                        "required": ["port", "type"],
                    },
                },
                "allowed_read_roles": {"type": "array", "uniqueItems": True, "items": {"enum": roles}},
                "family_boundary_justification": {"type": "string", "minLength": 20},
                "minimal_implementation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "entrypoint_role": {"enum": list(_FILE_ROLES)},
                        "file_roles": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": list(_FILE_ROLES)}},
                        "steps": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 5}},
                    },
                    "required": ["entrypoint_role", "file_roles", "steps"],
                },
                "matched_control_rationale": {"type": "string", "minLength": 12},
                "ablation": {"type": "string", "minLength": 12},
                "failure_modes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 8}},
                "estimated_cost": {"enum": list(_COST_LEVELS)},
            },
            "required": [
                "custom_component_id", "slot_id", "mathematical_definition", "algorithm_definition",
                "input_ports", "output_ports", "allowed_read_roles", "family_boundary_justification",
                "minimal_implementation", "matched_control_rationale", "ablation", "failure_modes", "estimated_cost",
            ],
        }
        text_signal = {"type": "string", "minLength": 8, "maxLength": 4000}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"const": f"recclaw.{self._spec.family_id.lower().replace('_', '-')}.mechanism-program.v1"},
                "family_contract_id": {"const": self._spec.family_id},
                "construction_mode": {"enum": list(_CONSTRUCTION_MODES)},
                "parent_refs": {
                    "type": "array", "uniqueItems": True,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {"candidate_id": {"type": "string", "minLength": 1}, "program_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
                        "required": ["candidate_id", "program_digest"],
                    },
                },
                "research_question": {"type": "string", "minLength": 12, "maxLength": 2000},
                "core_hypothesis": text_signal,
                "declared_data_roles": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": roles}},
                "components": {"type": "array", "minItems": 3, "maxItems": 96, "items": component},
                "architecture_operators": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "operator_id": {"enum": operators},
                            "targets": {"type": "array", "uniqueItems": True, "items": {"type": "string", "pattern": "^(slot:[A-Z][A-Z0-9_]{0,63}|[a-z][a-z0-9_]{0,63})$"}},
                            "replacements": {"type": "array", "uniqueItems": True, "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"}},
                            "parameters": {"type": "object"},
                            "rationale": text_signal,
                        },
                        "required": ["operator_id", "targets", "replacements", "parameters", "rationale"],
                    },
                },
                "changed_slots": {
                    "type": "array", "minItems": 1, "maxItems": _MAX_HIGH_PERMISSION_CHANGED_SLOTS, "uniqueItems": True,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {"slot_id": {"enum": slots}, "change_role": {"enum": list(_CHANGE_ROLES)}},
                        "required": ["slot_id", "change_role"],
                    },
                },
                "removed_slots": {"type": "array", "uniqueItems": True, "items": {"enum": slots}},
                "custom_components": {"type": "array", "maxItems": 16, "items": custom_component},
                "mechanism_explanation": {"type": "string", "minLength": 20, "maxLength": 4000},
                "expected_effects": {
                    "type": "object", "additionalProperties": False,
                    "properties": {key: text_signal for key in ("relevance", "efficiency", "robustness", "coverage")},
                    "required": ["relevance", "efficiency", "robustness", "coverage"],
                },
                "matched_control": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"control_ref": {"type": "string", "minLength": 1}, "rationale": {"type": "string", "minLength": 12}},
                    "required": ["control_ref", "rationale"],
                },
                "ablation_plan": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "ablation_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                            "remove_component_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"}},
                            "expected_observation": text_signal,
                        },
                        "required": ["ablation_id", "remove_component_ids", "expected_observation"],
                    },
                },
                "discriminating_predictions": {
                    "type": "array", "minItems": 1, "maxItems": 4,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {"metric_or_probe": text_signal, "if_supported": text_signal, "if_refuted": text_signal},
                        "required": ["metric_or_probe", "if_supported", "if_refuted"],
                    },
                },
                "failure_interpretation": {
                    "type": "object", "additionalProperties": False,
                    "properties": {key: text_signal for key in ("mechanism_failure", "optimization_failure", "protocol_failure", "resource_failure")},
                    "required": ["mechanism_failure", "optimization_failure", "protocol_failure", "resource_failure"],
                },
                "implementation_plan": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 5}},
                "resource_contract": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "relative_training_compute": {"enum": list(_COST_LEVELS)},
                        "relative_memory": {"enum": list(_COST_LEVELS)},
                        "precompute_required": {"type": "boolean"},
                        "separate_budget_stages": {"type": "array", "uniqueItems": True, "items": {"type": "string", "minLength": 2}},
                    },
                    "required": ["relative_training_compute", "relative_memory", "precompute_required", "separate_budget_stages"],
                },
                "claim_ceiling": {"const": _CLAIM_CEILING},
                "protocol_impact": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "status": {"enum": ["UNCHANGED", "BRANCH_REQUIRED"]},
                        "requested_changes": {"type": "array", "uniqueItems": True, "items": {"enum": list(self._spec.frozen_protocol_fields)}},
                    },
                    "required": ["status", "requested_changes"],
                },
            },
            "required": [
                "schema_version", "family_contract_id", "construction_mode", "parent_refs", "research_question",
                "core_hypothesis", "declared_data_roles", "components", "architecture_operators", "changed_slots",
                "removed_slots", "custom_components", "mechanism_explanation", "expected_effects", "matched_control",
                "ablation_plan", "discriminating_predictions", "failure_interpretation", "implementation_plan",
                "resource_contract", "claim_ceiling", "protocol_impact",
            ],
        }


__all__ = [
    "DeclarativeFamilySpec",
    "DeclarativeMechanismSpaceProvider",
    "axis",
    "object_parameters",
    "operator",
    "primitive",
    "standard_operators",
    "train_spectral_basis_rank_max",
]
