"""Closed compiler for ``BL_ICF_MECHANISM_SPACE_V1``.

The compiler validates a code-free mechanism program.  It neither imports nor
executes candidate code and it does not grant execution authority.  Its output
is a development-only resolved IR plus compiler-derived capability needs.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from recclaw_core.mechanism_space.canonical import (
    deep_freeze,
    deep_thaw,
    domain_sha256,
    load_json_bytes,
    snapshot_json,
)
from recclaw_core.mechanism_space.contracts import (
    CompileDiagnostic,
    CompileReportV1,
    CompileStatus,
    SpaceIdentity,
)

SPACE_ID = "BL_ICF_MECHANISM_SPACE_V1"
FAMILY_ID = "BL_ICF_V1"
PROVIDER_ID = "recclaw.search-space-provider.bl-icf.v1"
_RESOURCE_PACKAGE = "recclaw_core.search_spaces.bl_icf_v1.resources"
_GENERIC_RESOURCE_PACKAGE = "recclaw_core.mechanism_space.resources"

_BL_RESOURCE_NAMES = (
    "search_space_manifest_v1.json",
    "family_contract_v1.json",
    "primitive_registry_v1.json",
    "mechanism_grammar_v1.json",
    "capability_policy_v1.json",
    "mechanism_program_v1.schema.json",
)
_GENERIC_RESOURCE_NAMES = (
    "mechanism_program_envelope_v1.schema.json",
    "candidate_execution_binding_v1.schema.json",
)
_KERNEL_SOURCE_NAMES = (
    "canonical.py",
    "contracts.py",
    "kernel.py",
)

_CUSTOM_SLOT_CAPABILITIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "RELATION_VIEW": ("RELATION_MUTATION",),
        "EMBEDDING": ("CONFIG_MUTATION",),
        "ENCODER": ("GRAPH_OPERATOR_MUTATION",),
        "MESSAGE": ("GRAPH_OPERATOR_MUTATION",),
        "PROPAGATION_AGGREGATION": ("GRAPH_OPERATOR_MUTATION",),
        "RELATION_DECOMPOSITION": ("RELATION_MUTATION",),
        "FUSION_ROUTING": ("GRAPH_OPERATOR_MUTATION",),
        "SCORE_HEAD": ("CONFIG_MUTATION",),
        "PRIMARY_OBJECTIVE": ("LOSS_MUTATION",),
        "NEGATIVE_SAMPLER": ("SAMPLER_MUTATION",),
        "SELF_SUPERVISION": ("AUXILIARY_OBJECTIVE_MUTATION",),
        "GEOMETRY_REGULARIZATION": ("AUXILIARY_OBJECTIVE_MUTATION",),
        "DENOISING_LONG_TAIL": ("LOSS_MUTATION",),
        "TRAINING_PROCEDURE": ("TRAINING_ALGORITHM_MUTATION",),
        "EFFICIENCY_APPROXIMATION": ("EFFICIENCY_REWRITE",),
        "POSTHOC_RERANK": ("POSTHOC_RERANK",),
    }
)


@dataclass(frozen=True, slots=True)
class _BLResources:
    manifest: Mapping[str, Any]
    family: Mapping[str, Any]
    registry: Mapping[str, Any]
    grammar: Mapping[str, Any]
    capability_policy: Mapping[str, Any]
    program_schema: Mapping[str, Any]
    closure: Mapping[str, Any]
    primitives: Mapping[str, Mapping[str, Any]]
    axes: Mapping[str, Mapping[str, Any]]
    operators: Mapping[str, Mapping[str, Any]]
    data_roles: Mapping[str, str]
    type_compatibility: frozenset[tuple[str, str]]


def _resource_bytes(package: str, name: str) -> bytes:
    return resources.files(package).joinpath(name).read_bytes()


def _relative_resource_key(package: str, name: str) -> str:
    return package.replace(".", "/") + "/" + name


@lru_cache(maxsize=1)
def _load_resources() -> _BLResources:
    closure = load_json_bytes(_resource_bytes(_RESOURCE_PACKAGE, "closure_v1.json"))
    expected_hashes = closure.get("resource_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("BL-ICF closure resource_sha256 is malformed")

    raw_documents: dict[str, bytes] = {}
    for package, names in (
        (_RESOURCE_PACKAGE, _BL_RESOURCE_NAMES),
        (_GENERIC_RESOURCE_PACKAGE, _GENERIC_RESOURCE_NAMES),
    ):
        for name in names:
            key = _relative_resource_key(package, name)
            raw = _resource_bytes(package, name)
            raw_documents[key] = raw
            if hashlib.sha256(raw).hexdigest() != expected_hashes.get(key):
                raise ValueError(f"BL-ICF resource exact-byte mismatch: {key}")
    if set(raw_documents) != set(expected_hashes):
        raise ValueError("BL-ICF resource closure is not bidirectionally complete")

    provider_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if provider_sha256 != closure.get("provider_source_sha256"):
        raise ValueError("BL-ICF provider source is not bound to the resource closure")
    kernel_root = Path(__file__).resolve().parents[2] / "mechanism_space"
    kernel_source_hashes = {
        f"recclaw_core/mechanism_space/{name}": hashlib.sha256(
            (kernel_root / name).read_bytes()
        ).hexdigest()
        for name in _KERNEL_SOURCE_NAMES
    }
    if kernel_source_hashes != closure.get("kernel_source_sha256"):
        raise ValueError("BL-ICF generic kernel source is not bound to the resource closure")
    runtime_dependencies = closure.get("runtime_dependencies")
    if not isinstance(runtime_dependencies, dict):
        raise ValueError("BL-ICF runtime dependency closure is malformed")
    observed_dependencies: dict[str, str] = {}
    for package, expected_version in runtime_dependencies.items():
        try:
            observed_dependencies[str(package)] = version(str(package))
        except PackageNotFoundError as exc:
            raise ValueError(f"BL-ICF runtime dependency is missing: {package}") from exc
        if observed_dependencies[str(package)] != expected_version:
            raise ValueError(
                f"BL-ICF runtime dependency version mismatch: {package}"
            )
    identity_payload = {
        "kernel_source_sha256": kernel_source_hashes,
        "provider_id": closure.get("provider_id"),
        "provider_source_sha256": provider_sha256,
        "resource_sha256": expected_hashes,
        "runtime_dependencies": observed_dependencies,
        "search_space_id": closure.get("search_space_id"),
        "search_space_version": closure.get("search_space_version"),
    }
    observed_space_digest = domain_sha256(
        "recclaw.search-space.resource-closure.v1", identity_payload
    )
    if observed_space_digest != closure.get("search_space_digest"):
        raise ValueError("BL-ICF search-space digest does not match the package closure")

    documents = {key: load_json_bytes(raw) for key, raw in raw_documents.items()}
    prefix = _relative_resource_key(_RESOURCE_PACKAGE, "")
    manifest = documents[prefix + "search_space_manifest_v1.json"]
    family = documents[prefix + "family_contract_v1.json"]
    registry = documents[prefix + "primitive_registry_v1.json"]
    grammar = documents[prefix + "mechanism_grammar_v1.json"]
    capability_policy = documents[prefix + "capability_policy_v1.json"]
    program_schema = documents[prefix + "mechanism_program_v1.schema.json"]

    if manifest.get("search_space_id") != SPACE_ID or manifest.get("family_id") != FAMILY_ID:
        raise ValueError("BL-ICF manifest identity mismatch")
    if manifest.get("provider_id") != PROVIDER_ID:
        raise ValueError("BL-ICF provider identity mismatch")
    if family.get("family_id") != FAMILY_ID:
        raise ValueError("BL-ICF family identity mismatch")
    if registry.get("axis_count") != 16 or len(registry.get("axes", [])) != 16:
        raise ValueError("BL-ICF primitive registry must close exactly 16 axes")
    if grammar.get("cross_family_composition") != "DENY_UNLESS_REGISTERED_BRIDGE":
        raise ValueError("BL-ICF cross-family composition must fail closed")
    if capability_policy.get("authority") != "NONE" or capability_policy.get("formal_acceptance") is not False:
        raise ValueError("BL-ICF capability policy must remain development-only")

    primitives: dict[str, Mapping[str, Any]] = {}
    axes: dict[str, Mapping[str, Any]] = {}
    for axis in registry["axes"]:
        slot_id = str(axis["slot_id"])
        if slot_id in axes:
            raise ValueError(f"duplicate BL-ICF axis: {slot_id}")
        axes[slot_id] = deep_freeze(axis)
        for primitive in axis["primitives"]:
            primitive_id = str(primitive["primitive_id"])
            if primitive_id in primitives:
                raise ValueError(f"duplicate BL-ICF primitive: {primitive_id}")
            if primitive.get("slot_id") != slot_id:
                raise ValueError(f"BL-ICF primitive slot mismatch: {primitive_id}")
            primitives[primitive_id] = deep_freeze(primitive)
    if len(primitives) != registry.get("primitive_count"):
        raise ValueError("BL-ICF primitive count mismatch")

    operators: dict[str, Mapping[str, Any]] = {}
    for operator in grammar["operators"]:
        operator_id = str(operator["operator_id"])
        if operator_id in operators:
            raise ValueError(f"duplicate BL-ICF operator: {operator_id}")
        operators[operator_id] = deep_freeze(operator)

    data_roles = {
        str(item["role_id"]): str(item["type_ref"])
        for item in family["allowed_data_roles"]
    }
    compatibility = frozenset(
        (str(item["provided"]), str(item["accepted_as"]))
        for item in family["type_compatibility"]
    )
    return _BLResources(
        manifest=deep_freeze(manifest),
        family=deep_freeze(family),
        registry=deep_freeze(registry),
        grammar=deep_freeze(grammar),
        capability_policy=deep_freeze(capability_policy),
        program_schema=deep_freeze(program_schema),
        closure=deep_freeze(closure),
        primitives=MappingProxyType(primitives),
        axes=MappingProxyType(axes),
        operators=MappingProxyType(operators),
        data_roles=MappingProxyType(data_roles),
        type_compatibility=compatibility,
    )


def _schema_diagnostics(instance: Any, schema: Mapping[str, Any]) -> list[CompileDiagnostic]:
    schema_value = deep_thaw(schema)
    try:
        Draft202012Validator.check_schema(schema_value)
    except SchemaError as exc:
        return [
            CompileDiagnostic(
                "PACKAGE_SCHEMA_INVALID",
                "package-owned BL-ICF schema is invalid",
                actual=type(exc).__name__,
            )
        ]
    diagnostics: list[CompileDiagnostic] = []
    for error in sorted(
        Draft202012Validator(schema_value).iter_errors(instance),
        key=lambda item: list(item.absolute_path),
    ):
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        diagnostics.append(
            CompileDiagnostic(
                "BL_ICF_SCHEMA_INVALID",
                error.message,
                path=path if path != "/" else "/program_payload",
            )
        )
    return diagnostics


def _type_compatible(provided: str, accepted: str, contract: _BLResources) -> bool:
    return provided == accepted or (provided, accepted) in contract.type_compatibility


def _topological_order(
    component_ids: Sequence[str], edges: Mapping[str, set[str]]
) -> tuple[list[str], bool]:
    indegree = {component_id: 0 for component_id in component_ids}
    downstream: dict[str, set[str]] = defaultdict(set)
    for target, sources in edges.items():
        indegree[target] += len(sources)
        for source in sources:
            downstream[source].add(target)
    queue = deque(sorted(component_id for component_id, count in indegree.items() if count == 0))
    result: list[str] = []
    while queue:
        current = queue.popleft()
        result.append(current)
        for target in sorted(downstream.get(current, ())):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return result, len(result) != len(component_ids)


def _source_type(
    source: Mapping[str, Any],
    *,
    component_outputs: Mapping[str, Mapping[str, str]],
    contract: _BLResources,
) -> tuple[str | None, str | None]:
    if source["kind"] == "DATA":
        role = str(source["data_role"])
        return contract.data_roles.get(role), None
    source_id = str(source["component_id"])
    port = str(source["output_port"])
    outputs = component_outputs.get(source_id)
    if outputs is None:
        return None, "UNKNOWN_SOURCE_COMPONENT"
    if port not in outputs:
        return None, "UNKNOWN_SOURCE_PORT"
    return outputs[port], None


def _component_semantic_hashes(
    components: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
    order: Sequence[str],
    custom_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for component_id in order:
        component = components[component_id]
        inputs: list[dict[str, Any]] = []
        for item in component["inputs"]:
            source = item["source"]
            if source["kind"] == "DATA":
                source_identity: Any = {"data_role": source["data_role"]}
            else:
                source_identity = {
                    "component_semantics": hashes[source["component_id"]],
                    "output_port": source["output_port"],
                }
            inputs.append({"port": item["port"], "source": source_identity})
        inputs.sort(key=lambda item: (item["port"], str(item["source"])))
        if "primitive_id" in component:
            implementation = {"primitive_id": component["primitive_id"]}
        else:
            custom = dict(custom_specs[component["custom_component_id"]])
            custom.pop("custom_component_id", None)
            implementation = {"custom_component": custom}
        semantic_value = {
            "implementation": implementation,
            "inputs": inputs,
            "parameters": component["parameters"],
            "slot_id": component["slot_id"],
            "output_contract": [
                {"port": item["port"], "type": item["type"]}
                for item in specs[component_id]["output_ports"]
            ],
        }
        hashes[component_id] = domain_sha256(
            "recclaw.bl-icf.component-semantics.v1", semantic_value
        )
    return hashes


class BL_ICFProvider:
    provider_id = PROVIDER_ID

    def identity(self) -> SpaceIdentity:
        contract = _load_resources()
        return SpaceIdentity(
            search_space_id=SPACE_ID,
            search_space_version=str(contract.manifest["search_space_version"]),
            search_space_digest=str(contract.closure["search_space_digest"]),
            family_id=FAMILY_ID,
            family_version=str(contract.manifest["family_version"]),
            provider_id=PROVIDER_ID,
        )

    def prompt_projection(self) -> Mapping[str, Any]:
        contract = _load_resources()
        axes = []
        for slot_id in sorted(contract.axes):
            axis = contract.axes[slot_id]
            primitives = []
            for item in axis["primitives"]:
                primitives.append(
                    {
                        "primitive_id": item["primitive_id"],
                        "slot_id": item["slot_id"],
                        "input_ports": deep_thaw(item["input_ports"]),
                        "output_ports": deep_thaw(item["output_ports"]),
                        "parameter_schema": deep_thaw(item["parameter_schema"]),
                        "capabilities": list(item["capabilities"]),
                    }
                )
            axes.append(
                {
                    "slot_id": slot_id,
                    "allow_multiple": axis["allow_multiple"],
                    "primitives": primitives,
                }
            )
        return deep_freeze(
            {
                "projection_version": contract.manifest["prompt_projection_version"],
                "space_identity": self.identity().to_dict(),
                "scientific_object": contract.family["scientific_object"],
                "allowed_data_roles": deep_thaw(contract.family["allowed_data_roles"]),
                "forbidden_data_roles": list(contract.family["forbidden_data_roles"]),
                "output_contract": deep_thaw(contract.family["output_contract"]),
                "frozen_protocol_fields": list(contract.family["frozen_protocol_fields"]),
                "axes": axes,
                "architecture_operators": [
                    {
                        "operator_id": item["operator_id"],
                        "requires_target": item["requires_target"],
                        "requires_replacement": item["requires_replacement"],
                    }
                    for item in contract.grammar["operators"]
                ],
                "custom_mechanism_requirements": [
                    "mathematical_definition",
                    "algorithm_definition",
                    "typed_input_output_ports",
                    "family_boundary_justification",
                    "minimal_implementation",
                    "matched_control",
                    "ablation",
                    "failure_modes",
                    "estimated_cost",
                ],
                "claim_ceiling": contract.family["claim_ceiling"],
            }
        )

    def program_schema(self) -> Mapping[str, Any]:
        contract = _load_resources()
        envelope_schema = load_json_bytes(
            _resource_bytes(
                _GENERIC_RESOURCE_PACKAGE,
                "mechanism_program_envelope_v1.schema.json",
            )
        )
        schema = snapshot_json(envelope_schema)
        schema["properties"]["search_space_id"] = {"const": SPACE_ID}
        schema["properties"]["search_space_digest"] = {
            "const": self.identity().search_space_digest
        }
        schema["properties"]["family_id"] = {"const": FAMILY_ID}
        schema["properties"]["family_version"] = {
            "const": self.identity().family_version
        }
        schema["properties"]["program_payload"] = deep_thaw(
            contract.program_schema
        )
        return deep_freeze(schema)

    def compile(self, envelope: Mapping[str, Any]) -> CompileReportV1:
        contract = _load_resources()
        identity = self.identity()
        payload = snapshot_json(envelope["program_payload"])
        diagnostics = _schema_diagnostics(payload, contract.program_schema)
        profile_ref = envelope["profile_ref"]
        if profile_ref["profile_kind"] not in contract.family["supported_profile_kinds"]:
            diagnostics.append(
                CompileDiagnostic(
                    "UNSUPPORTED_PROFILE",
                    "BL-ICF v1 only supports the OFFLINE_TOPN scientific profile",
                    path="/profile_ref/profile_kind",
                    expected=list(contract.family["supported_profile_kinds"]),
                    actual=profile_ref["profile_kind"],
                )
            )
        if diagnostics:
            return CompileReportV1(
                CompileStatus.INVALID,
                diagnostics=tuple(diagnostics),
                space_identity=identity,
            )

        components = {str(item["component_id"]): item for item in payload["components"]}
        if len(components) != len(payload["components"]):
            diagnostics.append(
                CompileDiagnostic(
                    "DUPLICATE_COMPONENT_ID",
                    "component_id values must be unique",
                    path="/program_payload/components",
                )
            )
        custom_specs = {
            str(item["custom_component_id"]): item for item in payload["custom_components"]
        }
        if len(custom_specs) != len(payload["custom_components"]):
            diagnostics.append(
                CompileDiagnostic(
                    "DUPLICATE_CUSTOM_COMPONENT_ID",
                    "custom_component_id values must be unique",
                    path="/program_payload/custom_components",
                )
            )

        component_specs: dict[str, Mapping[str, Any]] = {}
        component_outputs: dict[str, dict[str, str]] = {}
        required_capabilities: set[str] = set()
        needs_implementation = False
        used_custom_ids: set[str] = set()
        slot_components: dict[str, list[str]] = defaultdict(list)

        for component_id, component in components.items():
            slot_id = str(component["slot_id"])
            slot_components[slot_id].append(component_id)
            if "primitive_id" in component:
                primitive_id = str(component["primitive_id"])
                spec = contract.primitives.get(primitive_id)
                if spec is None:
                    diagnostics.append(
                        CompileDiagnostic(
                            "UNKNOWN_PRIMITIVE",
                            "primitive is not present in the package-owned registry",
                            path=f"/program_payload/components/{component_id}/primitive_id",
                            actual=primitive_id,
                        )
                    )
                    continue
                if spec["slot_id"] != slot_id:
                    diagnostics.append(
                        CompileDiagnostic(
                            "PRIMITIVE_SLOT_MISMATCH",
                            "primitive is instantiated in the wrong mechanism slot",
                            path=f"/program_payload/components/{component_id}/slot_id",
                            expected=spec["slot_id"],
                            actual=slot_id,
                        )
                    )
                parameter_errors = _schema_diagnostics(component["parameters"], spec["parameter_schema"])
                for item in parameter_errors:
                    diagnostics.append(
                        CompileDiagnostic(
                            "PRIMITIVE_PARAMETERS_INVALID",
                            item.message,
                            path=f"/program_payload/components/{component_id}/parameters{item.path}",
                        )
                    )
                required_capabilities.update(spec["capabilities"])
                if spec["implementation_status"] != "BUILTIN_RUNTIME":
                    needs_implementation = True
            else:
                custom_id = str(component["custom_component_id"])
                custom = custom_specs.get(custom_id)
                if custom is None:
                    diagnostics.append(
                        CompileDiagnostic(
                            "UNKNOWN_CUSTOM_COMPONENT",
                            "component references an undeclared custom mechanism",
                            path=f"/program_payload/components/{component_id}/custom_component_id",
                            actual=custom_id,
                        )
                    )
                    continue
                used_custom_ids.add(custom_id)
                if custom["slot_id"] != slot_id:
                    diagnostics.append(
                        CompileDiagnostic(
                            "CUSTOM_COMPONENT_SLOT_MISMATCH",
                            "custom component declaration and instance use different slots",
                            path=f"/program_payload/components/{component_id}/slot_id",
                            expected=custom["slot_id"],
                            actual=slot_id,
                        )
                    )
                spec = {
                    "primitive_id": f"custom.{custom_id}",
                    "slot_id": slot_id,
                    "input_ports": [
                        {
                            "port": item["port"],
                            "accepted_types": list(item["types"]),
                            "minimum": item["minimum"],
                        }
                        for item in custom["input_ports"]
                    ],
                    "output_ports": custom["output_ports"],
                    "parameter_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {},
                    },
                    "capabilities": ["CUSTOM_MODEL_IMPLEMENTATION", *_CUSTOM_SLOT_CAPABILITIES[slot_id]],
                    "implementation_status": "CUSTOM_IMPLEMENTATION_REQUIRED",
                    "tags": [],
                }
                if component["parameters"]:
                    diagnostics.append(
                        CompileDiagnostic(
                            "CUSTOM_PARAMETERS_UNDECLARED",
                            "v1 custom component parameters must be included in its mathematical definition",
                            path=f"/program_payload/components/{component_id}/parameters",
                            expected={},
                            actual=component["parameters"],
                        )
                    )
                required_capabilities.update(spec["capabilities"])
                needs_implementation = True
            component_specs[component_id] = spec
            component_outputs[component_id] = {
                str(item["port"]): str(item["type"]) for item in spec["output_ports"]
            }

        unused_custom = sorted(set(custom_specs) - used_custom_ids)
        if unused_custom:
            diagnostics.append(
                CompileDiagnostic(
                    "UNUSED_CUSTOM_COMPONENT",
                    "every custom mechanism declaration must be instantiated",
                    path="/program_payload/custom_components",
                    actual=unused_custom,
                )
            )

        edges: dict[str, set[str]] = defaultdict(set)
        for component_id, component in components.items():
            spec = component_specs.get(component_id)
            if spec is None:
                continue
            port_specs = {str(item["port"]): item for item in spec["input_ports"]}
            counts: Counter[str] = Counter()
            for input_item in component["inputs"]:
                port_name = str(input_item["port"])
                counts[port_name] += 1
                port_spec = port_specs.get(port_name)
                if port_spec is None:
                    diagnostics.append(
                        CompileDiagnostic(
                            "UNKNOWN_INPUT_PORT",
                            "component input names a port not declared by its primitive",
                            path=f"/program_payload/components/{component_id}/inputs/{port_name}",
                            actual=port_name,
                        )
                    )
                    continue
                source = input_item["source"]
                source_type, source_error = _source_type(
                    source, component_outputs=component_outputs, contract=contract
                )
                if source_error:
                    diagnostics.append(
                        CompileDiagnostic(
                            source_error,
                            "component input references an unknown source component or output port",
                            path=f"/program_payload/components/{component_id}/inputs/{port_name}",
                            actual=deep_thaw(source),
                        )
                    )
                    continue
                if source["kind"] == "COMPONENT":
                    edges[component_id].add(str(source["component_id"]))
                accepted_types = [str(value) for value in port_spec["accepted_types"]]
                if source_type is not None and not any(
                    _type_compatible(source_type, accepted, contract) for accepted in accepted_types
                ):
                    diagnostics.append(
                        CompileDiagnostic(
                            "PORT_TYPE_MISMATCH",
                            "component input source type is not accepted by the target port",
                            path=f"/program_payload/components/{component_id}/inputs/{port_name}",
                            expected=accepted_types,
                            actual=source_type,
                        )
                    )
            for port_name, port_spec in port_specs.items():
                if counts[port_name] < int(port_spec["minimum"]):
                    diagnostics.append(
                        CompileDiagnostic(
                            "REQUIRED_INPUT_MISSING",
                            "component does not satisfy the primitive input cardinality",
                            path=f"/program_payload/components/{component_id}/inputs",
                            expected={"port": port_name, "minimum": port_spec["minimum"]},
                            actual=counts[port_name],
                        )
                    )

        order, has_cycle = _topological_order(list(components), edges)
        if has_cycle:
            diagnostics.append(
                CompileDiagnostic(
                    "COMPONENT_GRAPH_CYCLE",
                    "mechanism component graph must be acyclic",
                    path="/program_payload/components",
                )
            )

        for slot_id, component_ids in slot_components.items():
            axis = contract.axes.get(slot_id)
            if axis is not None and not axis["allow_multiple"] and len(component_ids) > 1:
                diagnostics.append(
                    CompileDiagnostic(
                        "SLOT_CARDINALITY_EXCEEDED",
                        "mechanism slot permits at most one component",
                        path=f"/program_payload/components/{slot_id}",
                        expected=1,
                        actual=len(component_ids),
                    )
                )
        for required_slot in ("ENCODER", "SCORE_HEAD"):
            if len(slot_components.get(required_slot, ())) != 1:
                diagnostics.append(
                    CompileDiagnostic(
                        "REQUIRED_SLOT_MISSING",
                        "BL-ICF programs require exactly one encoder and score head",
                        path="/program_payload/components",
                        expected=required_slot,
                        actual=len(slot_components.get(required_slot, ())),
                    )
                )

        score_ids = slot_components.get("SCORE_HEAD", [])
        if score_ids:
            outputs = component_outputs.get(score_ids[0], {})
            if "core/user_item_relevance_score" not in outputs.values():
                diagnostics.append(
                    CompileDiagnostic(
                        "OUTPUT_CONTRACT_MISMATCH",
                        "score head must preserve the frozen candidate universe relevance-score contract",
                        path=f"/program_payload/components/{score_ids[0]}",
                    )
                )

        diagnostics.extend(
            self._validate_change_budget(payload, contract)
        )
        operator_caps, operator_diagnostics = self._validate_operators(
            payload, components, contract
        )
        required_capabilities.update(operator_caps)
        diagnostics.extend(operator_diagnostics)
        diagnostics.extend(self._validate_encoder_regime(payload, components, slot_components))
        builtin_runtime_materialization = self._builtin_runtime_materialization(
            payload, components
        )
        if builtin_runtime_materialization is None:
            needs_implementation = True

        protocol_impact = payload["protocol_impact"]
        protocol_branch = protocol_impact["status"] == "BRANCH_REQUIRED"
        if protocol_impact["status"] == "UNCHANGED" and protocol_impact["requested_changes"]:
            diagnostics.append(
                CompileDiagnostic(
                    "PROTOCOL_IMPACT_INCONSISTENT",
                    "UNCHANGED protocol impact cannot contain requested changes",
                    path="/program_payload/protocol_impact",
                )
            )
        if protocol_impact["status"] == "BRANCH_REQUIRED" and not protocol_impact["requested_changes"]:
            diagnostics.append(
                CompileDiagnostic(
                    "PROTOCOL_BRANCH_REASON_MISSING",
                    "BRANCH_REQUIRED must identify the frozen protocol fields involved",
                    path="/program_payload/protocol_impact/requested_changes",
                )
            )

        allowed_capabilities = set(contract.capability_policy["capability_families"])
        unknown_capabilities = sorted(required_capabilities - allowed_capabilities)
        if unknown_capabilities:
            diagnostics.append(
                CompileDiagnostic(
                    "UNKNOWN_CAPABILITY",
                    "compiler derived a capability outside the package-owned policy",
                    actual=unknown_capabilities,
                )
            )

        program_digest = domain_sha256(
            "recclaw.bl-icf.mechanism-program.v1", envelope
        )
        candidate_id = "bl1_" + program_digest[:20]
        semantics_digest: str | None = None
        resolved_ir: dict[str, Any] | None = None
        if not diagnostics and not has_cycle and len(component_specs) == len(components):
            structural_hashes = _component_semantic_hashes(
                components, component_specs, order, custom_specs
            )
            semantic_operators = []
            for item in payload["architecture_operators"]:
                def semantic_ref(value: str) -> str:
                    return structural_hashes.get(value, value)

                semantic_operators.append(
                    {
                        "operator_id": item["operator_id"],
                        "targets": sorted(semantic_ref(value) for value in item["targets"]),
                        "replacements": sorted(semantic_ref(value) for value in item["replacements"]),
                        "parameters": item["parameters"],
                    }
                )
            semantic_value = {
                "construction_mode": payload["construction_mode"],
                "input_semantics": payload["input_semantics"],
                "component_semantics": sorted(structural_hashes.values()),
                "operators": sorted(
                    semantic_operators,
                    key=lambda item: (item["operator_id"], str(item["targets"]), str(item["replacements"])),
                ),
                "changed_slots": sorted(
                    payload["changed_slots"], key=lambda item: (item["slot_id"], item["change_role"])
                ),
                "removed_slots": sorted(payload["removed_slots"]),
                "protocol_impact": payload["protocol_impact"],
                "profile_ref": envelope["profile_ref"],
                "search_space_digest": identity.search_space_digest,
            }
            semantics_digest = domain_sha256(
                "recclaw.bl-icf.mechanism-semantics.v1", semantic_value
            )
            resolved_ir = {
                "ir_version": "recclaw.bl-icf.resolved-mechanism-ir.v1",
                "authority": "NONE",
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
                "candidate_id": candidate_id,
                "search_space_id": identity.search_space_id,
                "search_space_digest": identity.search_space_digest,
                "family_id": identity.family_id,
                "profile_ref": snapshot_json(envelope["profile_ref"]),
                "mechanism_program_digest": program_digest,
                "mechanism_semantics_digest": semantics_digest,
                "component_order": order,
                "components": [snapshot_json(components[item]) for item in order],
                "architecture_operators": snapshot_json(payload["architecture_operators"]),
                "required_capabilities": sorted(required_capabilities),
                "implementation_requirement": (
                    "LOCAL_IMPLEMENTATION_REQUIRED" if needs_implementation else "BUILTIN_RUNTIME"
                ),
                "builtin_runtime_adapter": (
                    builtin_runtime_materialization["adapter_id"]
                    if builtin_runtime_materialization
                    else None
                ),
                "builtin_runtime_materialization": builtin_runtime_materialization,
                "candidate_root": f"recclaw_ext/generated/{candidate_id}/",
                "protocol_impact": snapshot_json(protocol_impact),
                "claim_ceiling": payload["claim_ceiling"],
            }

        if diagnostics:
            return CompileReportV1(
                CompileStatus.INVALID,
                diagnostics=tuple(diagnostics),
                space_identity=identity,
                candidate_id=candidate_id,
                mechanism_semantics_digest=semantics_digest,
                mechanism_program_digest=program_digest,
                required_capabilities=tuple(sorted(required_capabilities)),
                resolved_ir=resolved_ir,
            )
        if protocol_branch:
            return CompileReportV1(
                CompileStatus.PROTOCOL_BRANCH_REQUIRED,
                diagnostics=(
                    CompileDiagnostic(
                        "PROTOCOL_BRANCH_REQUIRED",
                        "proposal changes frozen protocol semantics and is not runnable in BL-ICF v1",
                        path="/program_payload/protocol_impact/requested_changes",
                        actual=protocol_impact["requested_changes"],
                    ),
                ),
                space_identity=identity,
                candidate_id=candidate_id,
                mechanism_semantics_digest=semantics_digest,
                mechanism_program_digest=program_digest,
                required_capabilities=tuple(sorted(required_capabilities)),
                resolved_ir=resolved_ir,
            )
        return CompileReportV1(
            CompileStatus.VALID_NEEDS_IMPLEMENTATION if needs_implementation else CompileStatus.VALID_WIRED,
            space_identity=identity,
            candidate_id=candidate_id,
            mechanism_semantics_digest=semantics_digest,
            mechanism_program_digest=program_digest,
            required_capabilities=tuple(sorted(required_capabilities)),
            resolved_ir=resolved_ir,
        )

    @staticmethod
    def _builtin_runtime_materialization(
        payload: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        if payload["custom_components"] or payload["architecture_operators"]:
            return None
        components_by_primitive = {
            str(component.get("primitive_id") or ""): component
            for component in components.values()
        }
        primitive_ids = set(components_by_primitive)
        bpr_signature = {
            "embedding.independent_user_item",
            "encoder.none_mf",
            "score.dot_product",
            "sampler.uniform",
            "objective.bpr",
            "training.adam",
        }
        lightgcn_signature = {
            "relation.user_item_bipartite",
            "embedding.independent_user_item",
            "encoder.explicit_message_passing",
            "message.identity",
            "propagation.symmetric_normalization",
            "fusion.layer_weighted_sum",
            "score.dot_product",
            "sampler.uniform",
            "objective.bpr",
            "training.adam",
        }
        if primitive_ids != bpr_signature and primitive_ids != lightgcn_signature:
            return None

        sampler = components_by_primitive["sampler.uniform"]["parameters"]
        supported_sampler = {
            "false_negative_policy": "ALLOW_UNKNOWN",
            "hardness": "NONE",
            "purpose": ["RANKING_SIGNAL"],
            "refresh_frequency": "BATCH",
            "replacement": True,
        }
        if any(sampler.get(key) != value for key, value in supported_sampler.items()):
            return None
        objective = components_by_primitive["objective.bpr"]["parameters"]
        if objective.get("weight", 1.0) != 1.0:
            return None

        embedding = components_by_primitive["embedding.independent_user_item"][
            "parameters"
        ]
        training = components_by_primitive["training.adam"]["parameters"]
        overrides: dict[str, Any] = {
            "embedding_size": int(embedding.get("dimension", 64)),
            "learning_rate": float(training.get("learning_rate", 0.001)),
            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num": int(sampler["negative_count"]),
                "alpha": 1.0,
                "dynamic": False,
                "candidate_num": 0,
            },
        }
        if primitive_ids == bpr_signature:
            adapter_id = "recclaw.builtin-runner.bpr.v1"
            base_model = "BPR"
            mechanism_config = "configs/bpr.yaml"
        else:
            adapter_id = "recclaw.builtin-runner.lightgcn.v1"
            base_model = "LightGCN"
            mechanism_config = "configs/lightgcn.yaml"
            propagation = components_by_primitive[
                "propagation.symmetric_normalization"
            ]["parameters"]
            overrides["n_layers"] = int(propagation.get("depth", 2))
        return {
            "adapter_id": adapter_id,
            "adapter_version": "1.0.0",
            "base_model": base_model,
            "mechanism_config": mechanism_config,
            "mechanism_overrides": overrides,
            "protocol_config_source": "EXECUTION_BINDING",
            "runner_contract": "recbole.run_recbole.config-chain.v1",
        }

    @staticmethod
    def _validate_change_budget(
        payload: Mapping[str, Any], contract: _BLResources
    ) -> list[CompileDiagnostic]:
        diagnostics: list[CompileDiagnostic] = []
        changes = payload["changed_slots"]
        slot_ids = [str(item["slot_id"]) for item in changes]
        if len(set(slot_ids)) != len(slot_ids):
            diagnostics.append(
                CompileDiagnostic(
                    "DUPLICATE_CHANGED_SLOT",
                    "a mechanism slot can appear only once in changed_slots",
                    path="/program_payload/changed_slots",
                )
            )
        core_count = sum(item["change_role"] == "CORE" for item in changes)
        support_count = sum(item["change_role"] == "SUPPORT" for item in changes)
        if core_count != 1:
            diagnostics.append(
                CompileDiagnostic(
                    "CORE_CHANGE_COUNT_INVALID",
                    "every candidate requires exactly one core mechanism change",
                    path="/program_payload/changed_slots",
                    expected=1,
                    actual=core_count,
                )
            )
        if payload["construction_mode"] == "COMPOSITION":
            limits = contract.grammar["ordinary_candidate_limits"]
            if support_count > int(limits["support_changes"]):
                diagnostics.append(
                    CompileDiagnostic(
                        "ORDINARY_CHANGE_BUDGET_EXCEEDED",
                        "ordinary composition permits one core and at most one support change",
                        path="/program_payload/changed_slots",
                        expected=deep_thaw(limits),
                        actual={"core_changes": core_count, "support_changes": support_count},
                    )
                )
        custom_count = len(payload["custom_components"])
        synthesize = any(
            item["operator_id"] == "synthesize_custom_model"
            for item in payload["architecture_operators"]
        )
        if payload["construction_mode"] == "CUSTOM_MODEL":
            if custom_count == 0 or not synthesize:
                diagnostics.append(
                    CompileDiagnostic(
                        "CUSTOM_MODEL_ESCAPE_INCOMPLETE",
                        "CUSTOM_MODEL requires a custom component and synthesize_custom_model operator",
                        path="/program_payload",
                    )
                )
        elif custom_count or synthesize:
            diagnostics.append(
                CompileDiagnostic(
                    "CUSTOM_MODEL_MODE_REQUIRED",
                    "custom components are only valid in CUSTOM_MODEL construction mode",
                    path="/program_payload/construction_mode",
                )
            )
        return diagnostics

    @staticmethod
    def _validate_operators(
        payload: Mapping[str, Any],
        components: Mapping[str, Mapping[str, Any]],
        contract: _BLResources,
    ) -> tuple[set[str], list[CompileDiagnostic]]:
        capabilities: set[str] = set()
        diagnostics: list[CompileDiagnostic] = []
        component_ids = set(components)
        removed_slots = set(payload["removed_slots"])
        for index, item in enumerate(payload["architecture_operators"]):
            operator_id = str(item["operator_id"])
            spec = contract.operators.get(operator_id)
            if spec is None:
                diagnostics.append(
                    CompileDiagnostic(
                        "UNKNOWN_ARCHITECTURE_OPERATOR",
                        "operator is not registered in the package-owned grammar",
                        path=f"/program_payload/architecture_operators/{index}/operator_id",
                        actual=operator_id,
                    )
                )
                continue
            capabilities.update(spec["capabilities"])
            if spec["requires_target"] and not item["targets"]:
                diagnostics.append(
                    CompileDiagnostic(
                        "OPERATOR_TARGET_REQUIRED",
                        "architecture operator requires at least one target",
                        path=f"/program_payload/architecture_operators/{index}/targets",
                    )
                )
            if spec["requires_replacement"] and not item["replacements"]:
                diagnostics.append(
                    CompileDiagnostic(
                        "OPERATOR_REPLACEMENT_REQUIRED",
                        "architecture operator requires an explicit replacement",
                        path=f"/program_payload/architecture_operators/{index}/replacements",
                    )
                )
            for target in item["targets"]:
                if str(target).startswith("slot:"):
                    slot_id = str(target).split(":", 1)[1]
                    if slot_id not in removed_slots:
                        diagnostics.append(
                            CompileDiagnostic(
                                "REMOVED_SLOT_NOT_DECLARED",
                                "slot target must also appear in removed_slots",
                                path=f"/program_payload/architecture_operators/{index}/targets",
                                actual=target,
                            )
                        )
                elif target not in component_ids:
                    diagnostics.append(
                        CompileDiagnostic(
                            "UNKNOWN_OPERATOR_TARGET",
                            "operator target is neither a final component nor a declared removed slot",
                            path=f"/program_payload/architecture_operators/{index}/targets",
                            actual=target,
                        )
                    )
            for replacement in item["replacements"]:
                if replacement not in component_ids:
                    diagnostics.append(
                        CompileDiagnostic(
                            "UNKNOWN_OPERATOR_REPLACEMENT",
                            "operator replacement must reference a final component",
                            path=f"/program_payload/architecture_operators/{index}/replacements",
                            actual=replacement,
                        )
                    )
        return capabilities, diagnostics

    @staticmethod
    def _validate_encoder_regime(
        payload: Mapping[str, Any],
        components: Mapping[str, Mapping[str, Any]],
        slot_components: Mapping[str, list[str]],
    ) -> list[CompileDiagnostic]:
        diagnostics: list[CompileDiagnostic] = []
        encoder_ids = slot_components.get("ENCODER", [])
        if len(encoder_ids) != 1:
            return diagnostics
        encoder = components[encoder_ids[0]]
        primitive_id = str(encoder.get("primitive_id") or "custom")
        propagation_ids = [
            str(components[item].get("primitive_id") or "custom")
            for item in slot_components.get("PROPAGATION_AGGREGATION", [])
        ]
        explicit_propagation = [item for item in propagation_ids if item != "propagation.none"]
        message_count = len(slot_components.get("MESSAGE", []))
        objective_ids = {
            str(components[item].get("primitive_id") or "custom")
            for item in slot_components.get("PRIMARY_OBJECTIVE", [])
        }
        relation_ids = {
            str(components[item].get("primitive_id") or "custom")
            for item in slot_components.get("RELATION_VIEW", [])
        }
        efficiency_ids = {
            str(components[item].get("primitive_id") or "custom")
            for item in slot_components.get("EFFICIENCY_APPROXIMATION", [])
        }
        if primitive_id == "encoder.none_mf" and (message_count or explicit_propagation):
            diagnostics.append(
                CompileDiagnostic(
                    "MF_REGIME_HAS_PROPAGATION",
                    "NONE/MF encoder cannot retain explicit message passing",
                    path="/program_payload/components",
                )
            )
        if primitive_id == "encoder.explicit_message_passing" and (
            message_count == 0 or not explicit_propagation
        ):
            diagnostics.append(
                CompileDiagnostic(
                    "EXPLICIT_PROPAGATION_INCOMPLETE",
                    "explicit message-passing encoder requires message and propagation components",
                    path="/program_payload/components",
                )
            )
        if primitive_id in {"encoder.precomputed_graph_filter", "encoder.closed_form_filter"}:
            if not relation_ids or not (
                {"efficiency.graph_filter_precomputation", "efficiency.closed_form_solver"}
                & efficiency_ids
            ):
                diagnostics.append(
                    CompileDiagnostic(
                        "GRAPH_FILTER_REGIME_INCOMPLETE",
                        "graph-filter regimes require a train-derived relation and precompute/closed-form component",
                        path="/program_payload/components",
                    )
                )
        if primitive_id == "encoder.fixed_point_constraint":
            required_relations = {
                "relation.user_item_bipartite",
                "relation.item_item_cooccurrence_topk",
            }
            required_objectives = {
                "objective.user_item_structure_constraint",
                "objective.item_item_relation_constraint",
            }
            if message_count or explicit_propagation:
                diagnostics.append(
                    CompileDiagnostic(
                        "FIXED_POINT_RETAINS_MESSAGE_PASSING",
                        "fixed-point constraint regime must remove explicit message passing",
                        path="/program_payload/components",
                    )
                )
            if not required_relations.issubset(relation_ids) or not required_objectives.issubset(objective_ids):
                diagnostics.append(
                    CompileDiagnostic(
                        "FIXED_POINT_RELATION_CLOSURE_INCOMPLETE",
                        "fixed-point regime requires explicit user-item and top-K item-item constraints",
                        path="/program_payload/components",
                        expected={
                            "relations": sorted(required_relations),
                            "objectives": sorted(required_objectives),
                        },
                        actual={
                            "relations": sorted(relation_ids),
                            "objectives": sorted(objective_ids),
                        },
                    )
                )
            if "efficiency.remove_message_passing" not in efficiency_ids:
                diagnostics.append(
                    CompileDiagnostic(
                        "FIXED_POINT_REMOVAL_NOT_EXPLICIT",
                        "fixed-point regime must encode message-passing removal as an efficiency rewrite",
                        path="/program_payload/components",
                    )
                )
        return diagnostics


PROVIDER = BL_ICFProvider()

__all__ = ["BL_ICFProvider", "PROVIDER"]
