"""Fail-closed Research Line adapter for V2R4 declarative families."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.realization_identity import (
    open_spec_realization_identity,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    profile_model_hooks_for_program,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    OpenSpecSearchCandidateV1,
    SearchCandidateBindingV1,
    SearchExecutableProfileV1,
    bind_search_candidate,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    IdeaModeV1,
    OpenResearchSpecV1,
    RealizationModeV1,
)
from recclaw_core.mechanism_space import CompileStatus
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.mechanism_space.declarative_provider import (
    DeclarativeMechanismSpaceProvider,
)

from .bl_icf_realization import (
    MechanismImplementationCompanionV1,
    ProviderMechanismProgramProposalV1,
)
from .interfaces import ProducerOutcome
from .frozen_family_profile import validate_frozen_family_profile
from .search_space_adapter import (
    ConfirmationResolutionKindV1,
    ConfirmationResolutionV1,
    SearchSpaceExecutionBindingV1,
)


_FULL_SOURCE_PARENT_SPACES = {
    "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1",
    "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1",
    "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1",
}

_PROFILE_MODEL_HOOK_MODE = "PROFILE_MODEL_HOOKS_V1"
_PROFILE_MODEL_HOOK_SPACES = frozenset(
    {
        "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1",
        "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1",
    }
)
def _profile_model_hook_ownership(
    *,
    search_space_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Compile only profile mechanics with a complete machine-owned outer ABI."""

    model_hooks = profile_model_hooks_for_program(
        search_space_id=search_space_id,
        program_payload=payload,
        component_specs=payload.get("components", ()),
    )
    if model_hooks is None:
        return None
    return {
        "mode": _PROFILE_MODEL_HOOK_MODE,
        "model_hooks": list(model_hooks),
    }


def _declarative_source_ownership(
    *,
    search_space_id: str,
    payload: Mapping[str, Any],
    resolved_ir: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Keep the exact frozen parent for ordinary local composition."""

    construction_mode = payload.get("construction_mode")
    has_custom_synthesis = any(
        isinstance(item, Mapping)
        and item.get("operator_id") == "synthesize_custom_model"
        for item in payload.get("architecture_operators", ())
    )
    if search_space_id in _PROFILE_MODEL_HOOK_SPACES:
        if construction_mode == "CUSTOM_MODEL":
            if has_custom_synthesis:
                try:
                    ownership = _profile_model_hook_ownership(
                        search_space_id=search_space_id, payload=payload
                    )
                except ValueError:
                    ownership = None
                if ownership is not None:
                    return ownership
                return {"mode": "FULL_SOURCE_V1"}
            raise ValueError(
                "UNSUPPORTED_PROFILE_MECHANICAL_BINDING: CUSTOM_MODEL requires "
                "an explicit synthesize_custom_model operator"
            )
        return _profile_model_hook_ownership(
            search_space_id=search_space_id,
            payload=payload,
        )
    if (
        search_space_id == "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1"
        and construction_mode == "CUSTOM_MODEL"
    ):
        if has_custom_synthesis:
            return {"mode": "FULL_SOURCE_V1"}
        raise ValueError(
            "UNSUPPORTED_PROFILE_MECHANICAL_BINDING: CUSTOM_MODEL requires "
            "an explicit synthesize_custom_model operator"
        )
    if (
        search_space_id == "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1"
        and resolved_ir.get("source_ownership") == "FULL_SOURCE_REQUIRED"
    ):
        return {"mode": "FULL_SOURCE_V1"}
    profile_ownership = _profile_model_hook_ownership(
        search_space_id=search_space_id,
        payload=payload,
    )
    if profile_ownership is not None:
        return profile_ownership
    if (
        search_space_id in _FULL_SOURCE_PARENT_SPACES
        and payload.get("construction_mode") == "COMPOSITION"
    ):
        return None
    if search_space_id in _FULL_SOURCE_PARENT_SPACES:
        return {"mode": "FULL_SOURCE_V1"}
    return None


@dataclass(frozen=True, slots=True)
class DeclarativeCompilerBindingV1:
    mechanism_program: Mapping[str, Any]
    compiler_candidate_id: str
    compile_report_digest: str
    mechanism_program_digest: str
    mechanism_semantics_digest: str
    semantic_identity_ref: str
    space_identity: Mapping[str, str]
    required_capabilities: tuple[str, ...]
    resolved_ir: Mapping[str, Any]

    def implementation_payload(self) -> dict[str, Any]:
        payload = self.mechanism_program["program_payload"]
        components = tuple(payload["components"])
        custom_components = tuple(payload.get("custom_components", ()))
        operators = tuple(payload.get("architecture_operators", ()))
        source_ownership = _declarative_source_ownership(
            search_space_id=str(self.space_identity["search_space_id"]),
            payload=payload,
            resolved_ir=self.resolved_ir,
        )
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
        }
        if source_ownership is not None:
            implementation_binding["source_ownership"] = source_ownership
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


def _mechanism_semantic_identity_ref(
    *,
    search_space_id: str,
    candidate_id: str,
) -> str:
    """Namespace a declarative-family semantic identity without BL changes."""

    return f"mechanism:{search_space_id}:{candidate_id}"


def _search_space_conformance(
    provider: DeclarativeMechanismSpaceProvider,
    *,
    space_identity: Mapping[str, Any],
    profile_ref: Mapping[str, Any],
    fixed_fallback: bool = False,
) -> dict[str, Any]:
    """Bind one non-BL provider's package identity and prompt-visible order."""

    if not isinstance(fixed_fallback, bool):
        raise ValueError("fixed_fallback must be a boolean")
    identity = canonical_value(dict(space_identity))
    expected_identity = canonical_value(provider.identity().to_dict())
    if identity != expected_identity:
        raise ValueError("search-space identity differs from its provider")
    projection = provider.prompt_projection()
    primitive_ids = tuple(
        str(primitive["primitive_id"])
        for axis in projection["axes"]
        for primitive in axis["primitives"]
    )
    payload = ("\n".join(primitive_ids) + "\n").encode("utf-8")
    return canonical_value(
        {
            "search_space_id": identity["search_space_id"],
            "search_space_digest": validate_sha256(
                str(identity["search_space_digest"]),
                field_name="space_identity.search_space_digest",
            ),
            "ordered_primitive_ids_count": len(primitive_ids),
            "ordered_primitive_ids_digest_algorithm": (
                "SHA256_UTF8_LF_TERMINATED_ID_LIST_V1"
            ),
            "ordered_primitive_ids_digest": hashlib.sha256(payload).hexdigest(),
            "profile_ref": canonical_value(dict(profile_ref)),
            "fixed_fallback": fixed_fallback,
        }
    )


def _compile_binding(
    provider: DeclarativeMechanismSpaceProvider,
    mechanism_program: Mapping[str, Any],
    frozen_profile_ref: Mapping[str, Any],
    *,
    execution_contract: Mapping[str, Any] | None = None,
) -> DeclarativeCompilerBindingV1:
    program = canonical_value(dict(mechanism_program))
    if program.get("search_space_id") != provider.identity().search_space_id:
        raise ValueError("mechanism program belongs to a different search space")
    if canonical_value(program.get("profile_ref")) != canonical_value(dict(frozen_profile_ref)):
        raise ValueError("mechanism program profile_ref differs from the frozen profile")
    compile_kwargs = (
        {"execution_contract": execution_contract}
        if execution_contract is not None
        and "TRAIN_SPECTRAL_BASIS" in provider.spec.allowed_data_roles
        else {}
    )
    report = provider.compile(deep_thaw(program), **compile_kwargs)
    if (
        report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION
        or report.candidate_id is None
        or report.mechanism_program_digest is None
        or report.mechanism_semantics_digest is None
        or report.space_identity is None
        or report.resolved_ir is None
    ):
        raise ValueError("declarative mechanism program is not strict VALID_NEEDS_IMPLEMENTATION")
    semantic_ref = _mechanism_semantic_identity_ref(
        search_space_id=report.space_identity.search_space_id,
        candidate_id=report.candidate_id,
    )
    return DeclarativeCompilerBindingV1(
        mechanism_program=program,
        compiler_candidate_id=report.candidate_id,
        compile_report_digest=sha256_digest(report.to_dict()),
        mechanism_program_digest=report.mechanism_program_digest,
        mechanism_semantics_digest=report.mechanism_semantics_digest,
        semantic_identity_ref=semantic_ref,
        space_identity=canonical_value(report.space_identity.to_dict()),
        required_capabilities=tuple(report.required_capabilities),
        resolved_ir=canonical_value(deep_thaw(report.resolved_ir)),
    )


def _full_execution_contract(
    value: Mapping[str, Any],
    *,
    provider: DeclarativeMechanismSpaceProvider,
    profile_ref: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("execution_contract must be a mapping")
    contract = dict(value)
    config = contract.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("execution_contract.config must be a mapping")
    validate_frozen_family_profile(
        provider,
        profile_ref=profile_ref,
        execution_config=config,
    )
    # Preserve runtime ownership through the existing config projection so a
    # missing declared dependency cannot be charged to a new research idea.
    dependencies = contract.pop("required_dependencies", ())
    contract["config"] = {
        **dict(config),
        "recclaw_runtime_dependencies": list(dependencies),
        "recclaw_trainer_entrypoint": "recclaw_ext.trainer:FreshCandidateTrainer",
    }
    return canonical_value(contract)


def _implementation_spec(
    compiler: DeclarativeCompilerBindingV1,
    companion: MechanismImplementationCompanionV1,
) -> OpenResearchSpecV1:
    payload = compiler.mechanism_program["program_payload"]
    predictions = tuple(
        f"{item['metric_or_probe']}: supported={item['if_supported']}; refuted={item['if_refuted']}"
        for item in payload["discriminating_predictions"]
    )
    effects = tuple(dict.fromkeys(str(value) for value in payload["expected_effects"].values()))
    failure = payload["failure_interpretation"]
    resource = payload["resource_contract"]
    parent_ids = tuple(str(item["candidate_id"]) for item in payload["parent_refs"])
    return OpenResearchSpecV1(
        hypothesis=str(payload["core_hypothesis"]),
        mechanism_change=str(payload["mechanism_explanation"]),
        competing_explanation=str(failure["mechanism_failure"]),
        matched_control_requirement=str(payload["matched_control"]["rationale"]),
        implementation_requirements=tuple(str(item) for item in payload["implementation_plan"]),
        expected_evidence=effects,
        falsifier=str(payload["ablation_plan"][0]["expected_observation"]),
        compatibility_requirements=companion.compatibility_requirements,
        protocol_ref=companion.protocol_ref,
        protocol_digest=companion.protocol_digest,
        context_ref=companion.context_ref,
        context_digest=companion.context_digest,
        current_profile_ref=companion.current_profile_ref,
        current_profile_digest=companion.current_profile_digest,
        producer_role=companion.producer_role,
        high_change_justification=(
            f"Compiler-bound {payload['construction_mode']} changes "
            + ", ".join(str(item["slot_id"]) for item in payload["changed_slots"])
        ),
        current_profile_expressibility_claim=CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE,
        idea_mode=IdeaModeV1.FRONTIER_HYPOTHESIS,
        research_question=str(payload["research_question"]),
        observed_failure_mode="NOT_OBSERVED",
        closest_parent=",".join(parent_ids) if parent_ids else None,
        minimal_testable_wedge=str(payload["mechanism_explanation"]),
        causal_chain=(str(payload["core_hypothesis"]), str(payload["mechanism_explanation"])),
        discriminative_predictions=predictions,
        mechanism_off_definition=str(payload["ablation_plan"][0]["expected_observation"]),
        resource_hypothesis=(
            f"training_compute={resource['relative_training_compute']}; "
            f"memory={resource['relative_memory']}; "
            f"precompute={str(resource['precompute_required']).lower()}; "
            f"separate_stages={','.join(resource['separate_budget_stages']) or 'none'}"
        ),
        realization_mode=(RealizationModeV1.PARENT_PRESERVING if parent_ids else RealizationModeV1.NON_NESTED),
        execution_contract=companion.execution_contract,
    )


def _effective_identity(compiler: DeclarativeCompilerBindingV1) -> dict[str, Any]:
    rows = {
        str(item["component_id"]): item
        for item in compiler.resolved_ir["components"]
    }
    custom_semantics = {
        str(item["custom_component_id"]): sha256_digest(
            {
                "slot_id": item.get("slot_id"),
                "mathematical_definition": item.get(
                    "mathematical_definition"
                ),
                "algorithm_definition": item.get("algorithm_definition"),
                "input_ports": sorted(
                    (
                        {
                            **dict(port),
                            "accepted_types": sorted(port["accepted_types"]),
                        }
                        for port in item.get("input_ports", ())
                    ),
                    key=lambda port: port["port"],
                ),
                "output_ports": sorted(
                    item.get("output_ports", ()),
                    key=lambda port: (port["port"], port["type"]),
                ),
                "allowed_read_roles": sorted(
                    item.get("allowed_read_roles", ())
                ),
            }
        )
        for item in compiler.resolved_ir.get("custom_component_definitions", ())
    }
    component_families: dict[str, str] = {}
    primitive_ids: set[str] = set()
    for component_id in compiler.resolved_ir["component_order"]:
        row = rows[str(component_id)]
        primitive_id = row.get("primitive_id")
        custom_id = row.get("custom_component_id")
        if isinstance(primitive_id, str):
            primitive_ids.add(primitive_id)
        inputs = []
        for edge in row["inputs"]:
            source = edge["source"]
            if source["kind"] == "DATA":
                source_identity = {"data_role": source["data_role"]}
            else:
                source_identity = {
                    "component_family": component_families[
                        str(source["component_id"])
                    ],
                    "output_port": source["output_port"],
                }
            inputs.append({"port": edge["port"], "source": source_identity})
        inputs.sort(key=lambda item: (str(item["port"]), str(item["source"])))
        component_families[str(component_id)] = sha256_digest(
            {
                "slot_id": row["slot_id"],
                "primitive_id": primitive_id,
                "custom_component_semantics": (
                    custom_semantics[str(custom_id)]
                    if isinstance(custom_id, str)
                    else None
                ),
                "inputs": inputs,
            }
        )

    def operator_target(value: Any) -> Any:
        if isinstance(value, str) and value in component_families:
            return {"component_family": component_families[value]}
        return value

    family_payload = canonical_value({
        "search_space_id": compiler.space_identity["search_space_id"],
        "search_space_digest": compiler.space_identity["search_space_digest"],
        "construction_mode": compiler.mechanism_program["program_payload"][
            "construction_mode"
        ],
        "component_families": sorted(component_families.values()),
        "architecture_operators": sorted(
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
                }
                for item in compiler.resolved_ir["architecture_operators"]
            ),
            key=str,
        ),
        "removed_slots": sorted(
            compiler.mechanism_program["program_payload"]["removed_slots"]
        ),
        "custom_component_families": sorted(custom_semantics.values()),
    })
    family_digest = sha256_digest(family_payload)
    return canonical_value(
        {
            "effective_family_digest": family_digest,
            "effective_experiment_digest": sha256_digest(
                {
                    "family_digest": family_digest,
                    "mechanism_semantics_digest": (
                        compiler.mechanism_semantics_digest
                    ),
                    "profile_ref": compiler.mechanism_program["profile_ref"],
                }
            ),
            "primitive_ids": tuple(sorted(primitive_ids)),
            "mechanism_program_digest": compiler.mechanism_program_digest,
            "mechanism_semantics_digest": compiler.mechanism_semantics_digest,
            "search_space_id": compiler.space_identity["search_space_id"],
            "search_space_digest": compiler.space_identity["search_space_digest"],
        }
    )


class DeclarativeSearchSpaceAdapterV1:
    """One adapter class whose semantics remain owned by one family Provider."""

    def __init__(
        self,
        provider: DeclarativeMechanismSpaceProvider,
        *,
        adapter_id: str,
        execution_contract: Mapping[str, Any] | None = None,
        implementation_template_source: str | Path | None = None,
        focused_language: Mapping[str, Any] | None = None,
    ):
        self.provider = provider
        self.adapter_id = adapter_id
        self.execution_contract = (
            None
            if execution_contract is None
            else canonical_value(dict(execution_contract))
        )
        self.implementation_template_source = implementation_template_source
        self.focused_language = (
            None
            if focused_language is None
            else canonical_value(dict(focused_language))
        )

    @property
    def supported_frozen_profile_kinds(self) -> tuple[str, ...]:
        """Expose the exact family-owned standalone profile kinds."""

        return self.provider.spec.supported_profile_kinds

    def build_provider_research_producer(self, **kwargs: Any) -> Any:
        """Bind the shared Provider transport to this family's strict program."""

        if self.execution_contract is None:
            raise ValueError(
                "declarative runtime adapter requires an execution_contract"
            )
        from .declarative_provider_runtime import DeclarativeResearchProducer

        return DeclarativeResearchProducer(
            declarative_provider=self.provider,
            execution_contract=self.execution_contract,
            focused_language=self.focused_language,
            **kwargs,
        )

    def build_provider_implementer(self, **kwargs: Any) -> Any:
        """Use the origin-blind implementer with a family-neutral strict prompt."""

        from .provider import ProviderImplementerGateway

        if self.implementation_template_source is not None:
            kwargs["implementation_template_source"] = (
                self.implementation_template_source
            )
        return ProviderImplementerGateway(**kwargs)

    def qualification_unit_check(
        self,
        *,
        base_model_config: str,
        execution_contract: Mapping[str, Any],
    ) -> Any:
        del base_model_config
        from .declarative_qualification import (
            declarative_behavioral_unit_check,
        )

        return declarative_behavioral_unit_check(
            execution_contract=execution_contract
        )

    def search_space_conformance(
        self, *, profile_ref: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return _search_space_conformance(
            self.provider,
            space_identity=self.provider.identity().to_dict(),
            profile_ref=profile_ref,
            fixed_fallback=False,
        )

    def resolve_confirmation(
        self,
        kind: str,
        primary_binding: Any,
        context: Mapping[str, Any],
    ) -> ConfirmationResolutionV1:
        phase = context.get("phase")
        if phase == "PROJECT_PROVIDER_PROPOSAL":
            return self._project_provider_proposal(primary_binding, context)
        if phase in {"IDENTIFY_INNOVATION", "PREPARE_INNOVATION"}:
            outcome = primary_binding
            frozen_profile_ref = context.get("frozen_profile_ref")
            if not isinstance(outcome, ProducerOutcome) or not isinstance(frozen_profile_ref, Mapping) or not isinstance(outcome.source_mechanism_program, Mapping):
                return ConfirmationResolutionV1(ConfirmationResolutionKindV1.UNSUPPORTED, reason="DECLARATIVE_INNOVATION_CONTEXT_INCOMPLETE")
            compiler = _compile_binding(
                self.provider,
                outcome.source_mechanism_program,
                frozen_profile_ref,
                execution_contract=self.execution_contract,
            )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "producer_outcome": outcome,
                    "compiler_binding": compiler,
                    "mechanism_program": compiler.mechanism_program,
                    "compiled_implementation": compiler.implementation_payload(),
                    "semantic_identity_ref": compiler.semantic_identity_ref,
                    "semantic_identity_digest": compiler.mechanism_semantics_digest,
                    "effective_identity": _effective_identity(compiler),
                    "candidate_id": compiler.compiler_candidate_id,
                    "mechanism_id": compiler.compiler_candidate_id,
                    "parent_available": bool(compiler.mechanism_program["program_payload"]["parent_refs"]),
                },
            )
        if phase == "PACKAGE_QUALIFIED_INNOVATION":
            return self._package_qualified(context)
        if phase == "MATERIALIZE_DISCOVERY":
            resolution = context.get("resolution")
            profile = context.get("execution_profile")
            proposal = getattr(primary_binding, "source_proposal", None)
            capability_ref = getattr(resolution, "resolved_current_capability_ref", None)
            if proposal is None or profile is None or capability_ref is None:
                return ConfirmationResolutionV1(ConfirmationResolutionKindV1.UNSUPPORTED, reason="DECLARATIVE_DISCOVERY_BINDING_CONTEXT_INCOMPLETE")
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={"native_binding": bind_search_candidate(profile=profile, proposal=proposal, capability_ref=capability_ref)},
            )
        if phase in {"DECLARE_FOLLOWUP", "GUARD_CONTROL", "MATCH_PROPOSAL"}:
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.NEEDS_PROPOSAL,
                reason="DECLARATIVE_EXACT_CONTROL_MUST_BE_PROPOSED_AND_QUALIFIED",
            )
        return ConfirmationResolutionV1(
            ConfirmationResolutionKindV1.UNSUPPORTED,
            reason="DECLARATIVE_CONFIRMATION_PHASE_UNSUPPORTED",
        )

    def _project_provider_proposal(
        self,
        proposal: Any,
        context: Mapping[str, Any],
    ) -> ConfirmationResolutionV1:
        research_context = context.get("research_context")
        producer_role = context.get("producer_role")
        bindings = context.get("producer_bindings")
        frozen_profile_ref = context.get("frozen_profile_ref")
        if research_context is None or not isinstance(producer_role, str) or not isinstance(bindings, Mapping) or not isinstance(frozen_profile_ref, Mapping):
            return ConfirmationResolutionV1(ConfirmationResolutionKindV1.UNSUPPORTED, reason="DECLARATIVE_PROVIDER_PROJECTION_CONTEXT_INCOMPLETE")
        if not isinstance(proposal, ProviderMechanismProgramProposalV1):
            from .producers import _project_result

            spec, facts, source_proposal, program, companion = _project_result(
                producer_role,
                proposal,
                bindings=bindings,
                frozen_profile_ref=frozen_profile_ref,
            )
        else:
            if proposal.producer_role != producer_role:
                raise ValueError("strict program proposal role differs from assigned Producer")
            compiler = _compile_binding(
                self.provider,
                proposal.mechanism_program,
                frozen_profile_ref,
                execution_contract=self.execution_contract,
            )
            companion = MechanismImplementationCompanionV1(
                protocol_ref=str(bindings["protocol_ref"]),
                protocol_digest=str(bindings["protocol_digest"]),
                context_ref=str(bindings["context_ref"]),
                context_digest=str(bindings["context_digest"]),
                current_profile_ref=str(bindings["current_profile_ref"]),
                current_profile_digest=str(bindings["current_profile_digest"]),
                producer_role=producer_role,
                compatibility_requirements=tuple(bindings["compatibility_requirements"]),
                execution_contract=_full_execution_contract(
                    self.execution_contract,
                    provider=self.provider,
                    profile_ref=compiler.mechanism_program["profile_ref"],
                ),
            )
            spec = _implementation_spec(compiler, companion)
            facts = canonical_value(dict(proposal.resolution_facts))
            required_fact_fields = {
                "requested_current_semantics_digest",
                "capability_diff",
                "high_change_dimensions",
                "required_dependencies",
                "required_budget",
            }
            if set(facts) != required_fact_fields or facts["requested_current_semantics_digest"] is not None or not facts["capability_diff"] or not facts["high_change_dimensions"]:
                raise ValueError("strict program resolution facts do not prove an unexpressible capability")
            source_proposal = None
            program = compiler.mechanism_program
        outcome = ProducerOutcome(
            producer_role=producer_role,
            context_ref=research_context.context_ref,
            context_digest=research_context.digest,
            spec=spec,
            resolution_facts=facts,
            source_proposal=source_proposal,
            source_mechanism_program=program,
            implementation_companion=companion,
            provenance={
                "producer_role": producer_role,
                "context_ref": research_context.context_ref,
                "context_digest": research_context.digest,
                "spec_id": spec.spec_id,
                "spec_digest": spec.digest,
                "source_mechanism_program_digest": sha256_digest(program) if program is not None else None,
                "search_space_id": self.provider.identity().search_space_id,
                "status": "PRODUCED",
            },
        )
        return ConfirmationResolutionV1(ConfirmationResolutionKindV1.EXACT_BINDING, binding={"producer_outcome": outcome})

    def _package_qualified(self, context: Mapping[str, Any]) -> ConfirmationResolutionV1:
        prepared = context.get("prepared_innovation")
        materialized = context.get("materialized")
        qualification = context.get("qualification")
        capability = context.get("capability")
        policy = context.get("implementer_policy")
        utility = context.get("utility_features")
        evidence = context.get("feature_evidence")
        if not isinstance(prepared, Mapping) or any(item is None for item in (materialized, qualification, capability, policy)) or not isinstance(utility, Mapping) or not isinstance(evidence, Mapping):
            return ConfirmationResolutionV1(ConfirmationResolutionKindV1.UNSUPPORTED, reason="DECLARATIVE_QUALIFIED_INNOVATION_CONTEXT_INCOMPLETE")
        outcome = prepared["producer_outcome"]
        compiler: DeclarativeCompilerBindingV1 = prepared["compiler_binding"]
        contract = policy.execution_contract
        if contract is None:
            raise ValueError("qualified declarative candidate lacks execution contract")
        package = materialized.package
        realization_ref, realization_digest = open_spec_realization_identity(
            outcome.spec,
            candidate_package_ref=package.package_id,
            candidate_package_digest=package.digest,
            candidate_root_ref=package.candidate_root_ref,
            candidate_root_digest=package.candidate_root_digest,
            source_tree_digest=package.source_tree_digest,
            executable_entrypoint=package.executable_entrypoint,
            execution_contract=contract,
        )
        conformance = _search_space_conformance(
            self.provider,
            space_identity=compiler.space_identity,
            profile_ref=compiler.mechanism_program["profile_ref"],
            fixed_fallback=False,
        )
        parent_refs = compiler.mechanism_program["program_payload"]["parent_refs"]
        parent_binding: Any = None if not parent_refs else parent_refs[0] if len(parent_refs) == 1 else parent_refs
        realization_binding = canonical_value(
            {
                "schema": "recclaw.compiler-bound-realization-binding.v1",
                "compiler_candidate_id": compiler.compiler_candidate_id,
                "compile_report_digest": compiler.compile_report_digest,
                "mechanism_program_digest": compiler.mechanism_program_digest,
                "mechanism_semantics_digest": compiler.mechanism_semantics_digest,
                "semantic_identity_ref": compiler.semantic_identity_ref,
                "space_identity": compiler.space_identity,
                **conformance,
                "parent_binding": parent_binding,
                "open_spec_ref": outcome.spec.spec_id,
                "open_spec_digest": outcome.spec.digest,
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
        search_candidate = OpenSpecSearchCandidateV1(
            spec=outcome.spec,
            capability_ref=capability.capability_id,
            capability_digest=capability.digest,
            candidate_package_ref=package.package_id,
            candidate_package_digest=package.digest,
            candidate_root_ref=package.candidate_root_ref,
            candidate_root_digest=package.candidate_root_digest,
            source_tree_digest=package.source_tree_digest,
            executable_entrypoint=package.executable_entrypoint,
            execution_contract=contract,
            mechanism_program=compiler.mechanism_program,
            compiler_candidate_id=compiler.compiler_candidate_id,
            compile_report_digest=compiler.compile_report_digest,
            mechanism_program_digest=compiler.mechanism_program_digest,
            semantic_identity_ref=compiler.semantic_identity_ref,
            semantic_identity_digest=compiler.mechanism_semantics_digest,
            realization_identity_ref=realization_ref,
            realization_semantics_digest=realization_digest,
            implementation_receipt_ref=package.implementation_receipt_ref,
            implementation_receipt_digest=package.implementation_receipt_digest,
            realization_binding_ref=content_id("recclaw-compiler-bound-realization-v1", realization_binding),
            realization_binding_digest=sha256_digest(realization_binding),
            qualification_receipt_ref=qualification.receipt.receipt_id,
            qualification_receipt_digest=qualification.receipt.digest,
            mechanism_axis=str(context.get("mechanism_axis", "")),
            utility_features=SearchUtilityFeaturesV1(**dict(utility)),
            feature_evidence=RouterFeatureEvidenceV1(**dict(evidence)),
        )
        return ConfirmationResolutionV1(ConfirmationResolutionKindV1.EXACT_BINDING, binding={"search_candidate": search_candidate})

    def validate_execution_binding(self, binding: SearchSpaceExecutionBindingV1) -> None:
        if binding.adapter_id != self.adapter_id:
            raise ValueError("execution binding targets a different adapter")
        native = binding.native_binding
        if not isinstance(native, SearchCandidateBindingV1):
            raise ValueError("declarative execution requires SearchCandidateBindingV1")
        proposal = native.proposal
        mechanism_program = getattr(proposal, "mechanism_program", None)
        if not isinstance(mechanism_program, Mapping) or mechanism_program.get("search_space_id") != self.provider.identity().search_space_id:
            raise ValueError("execution candidate belongs to a different search space")
        compile_kwargs = (
            {"execution_contract": self.execution_contract}
            if self.execution_contract is not None
            and "TRAIN_SPECTRAL_BASIS"
            in self.provider.spec.allowed_data_roles
            else {}
        )
        report = self.provider.compile(
            deep_thaw(mechanism_program),
            **compile_kwargs,
        )
        if not report.is_valid or report.mechanism_semantics_digest != binding.semantic_identity_digest:
            raise ValueError("execution binding differs from compiler semantics")
        if binding.binding_digest != native.digest or binding.candidate_id != proposal.candidate_id:
            raise ValueError("execution binding differs from Search candidate identity")

    def effective_identity(self, binding: SearchSpaceExecutionBindingV1) -> Mapping[str, Any]:
        self.validate_execution_binding(binding)
        program = binding.native_binding.proposal.mechanism_program
        compiler = _compile_binding(
            self.provider,
            program,
            program["profile_ref"],
            execution_contract=self.execution_contract,
        )
        return _effective_identity(compiler)

    def execution_recipe(self, binding: SearchSpaceExecutionBindingV1) -> Mapping[str, Any]:
        self.validate_execution_binding(binding)
        context = binding.execution_context
        profile = context.get("profile")
        qualified = context.get("qualified_execution")
        evaluator = context.get("evaluator")
        split = context.get("split")
        if not isinstance(profile, SearchExecutableProfileV1) or not isinstance(qualified, Mapping) or not isinstance(evaluator, Mapping) or not isinstance(split, str) or not split:
            raise ValueError("declarative execution context is incomplete")
        native = binding.native_binding
        entry = profile.entry(native.capability_ref)
        if entry.capability_digest != native.capability_digest or entry.executable_entrypoint != native.executable_entrypoint or entry.semantic_identity_digest != native.mechanism_semantics_digest:
            raise ValueError("active profile entry differs from Search binding")
        program = native.proposal.mechanism_program
        contract = native.proposal.execution_contract
        config = contract.get("config") if isinstance(contract, Mapping) else None
        if not isinstance(config, Mapping):
            raise ValueError("qualified execution contract lacks config")
        frozen_profile = validate_frozen_family_profile(
            self.provider,
            profile_ref=program["profile_ref"],
            execution_config=config,
        )
        if config.get("profile_kind") not in self.provider.spec.supported_profile_kinds:
            raise ValueError("execution config profile_kind differs from family contract")
        dataset = config.get("dataset")
        protocol_digest = config.get("protocol_digest")
        if not isinstance(dataset, str) or not dataset or not isinstance(protocol_digest, str) or len(protocol_digest) != 64:
            raise ValueError("execution config must bind dataset and protocol_digest")
        return canonical_value(
            {
                **dict(qualified),
                "capability_ref": entry.capability_ref,
                "capability_digest": entry.capability_digest,
                "profile_ref": profile.profile_ref,
                "profile_digest": profile.profile_digest,
                "entrypoint": native.executable_entrypoint,
                "mechanism_id": native.proposal.mechanism_id,
                "mechanism_semantics_digest": native.mechanism_semantics_digest,
                "search_space_id": program["search_space_id"],
                "search_space_digest": program["search_space_digest"],
                "profile_kind": config["profile_kind"],
                "dataset": dataset,
                "protocol_digest": protocol_digest,
                "frozen_profile": frozen_profile,
                "episode_contract": dict(self.provider.spec.episode_contract),
                "qualification_contract": dict(self.provider.spec.qualification_contract),
                "split": split,
                "evaluator": dict(evaluator),
                "execution_role": "CANDIDATE",
                "hidden_fallback": False,
            }
        )


__all__ = ["DeclarativeCompilerBindingV1", "DeclarativeSearchSpaceAdapterV1"]
