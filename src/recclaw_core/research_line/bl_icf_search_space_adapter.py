"""Strict BL-ICF implementation of the generic feedback search-space port."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.compilation_cache import (
    compile_campaign_program,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    MECHANISM_AXIS_UNIVERSE_V1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
    canonical_mechanism_axis,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    OpenSpecSearchCandidateV1,
    adapt_current_search_profile,
    bind_search_candidate,
)
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.search_spaces.bl_icf_v1.provider import BL_ICFProvider

from .bootstrap import bootstrap_search_pool
from .effective_experiment import effective_experiment_identity
from .execution import execution_recipe_for_search_binding
from .interfaces import ProducerOutcome
from .single_parent_search import (
    PARENT_BASE_MODEL_CONFIG,
    bound_parent_from_context,
    is_bl_icf_single_parent_context,
)
from .bl_icf_realization import (
    bind_materialized_bl_icf_candidate,
    compile_bl_icf_candidate_source,
    implementation_spec_for_compiled_program,
    parent_binding_for_mechanism_program,
)
from .search_space_adapter import (
    ConfirmationResolutionKindV1,
    ConfirmationResolutionV1,
    SearchSpaceExecutionBindingV1,
)


_FROZEN_LIGHTGCNPP_RUNTIME_DEFAULTS = {
    "embedding_size": 64,
    "n_layers": 2,
    "alpha": 0.4,
    "beta": 0.1,
    "gamma": 0.0,
    "reg_weight": 0.0001,
}


def _single_parent_execution_contract(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Carry the frozen LightGCN++ ABI into every parent-local candidate."""

    return canonical_value(
        {
            **dict(value),
            "base_model_config": PARENT_BASE_MODEL_CONFIG,
            "config": {
                **_FROZEN_LIGHTGCNPP_RUNTIME_DEFAULTS,
                **dict(value["config"]),
            },
        }
    )


def bl_icf_effective_identity_for_program(
    program: Mapping[str, Any],
) -> Mapping[str, Any]:
    """BL-only compatibility entrypoint for pre-adapter realization code."""

    return effective_experiment_identity(program)


def _single_parent_source(
    context: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    research_context = context.get("research_context")
    knowledge_base = getattr(research_context, "knowledge_base", None)
    if not isinstance(knowledge_base, Mapping) and isinstance(
        research_context, Mapping
    ):
        knowledge_base = research_context.get("knowledge_base")
    if not isinstance(knowledge_base, Mapping):
        return None
    baseline_context = knowledge_base.get("baseline_context")
    if not isinstance(baseline_context, Mapping):
        return None
    return bound_parent_from_context(baseline_context)


def _uses_single_parent_context(context: Mapping[str, Any]) -> bool:
    research_context = context.get("research_context")
    knowledge_base = getattr(research_context, "knowledge_base", None)
    if not isinstance(knowledge_base, Mapping) and isinstance(
        research_context, Mapping
    ):
        knowledge_base = research_context.get("knowledge_base")
    if not isinstance(knowledge_base, Mapping):
        return False
    return is_bl_icf_single_parent_context(
        knowledge_base.get("baseline_context")
    )


def _lineage_parent_binding_from_context(
    context: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    lineage_parent = context.get("lineage_parent_mechanism_program")
    if not isinstance(lineage_parent, Mapping):
        return None
    candidate_id = lineage_parent.get("candidate_id")
    program_digest = lineage_parent.get("program_digest")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return None
    if not isinstance(program_digest, str):
        return None
    return canonical_value(
        {
            "candidate_id": candidate_id,
            "program_digest": validate_sha256(
                program_digest,
                field_name="lineage_parent_mechanism_program.program_digest",
            ),
        }
    )


def _lineage_parent_binding_from_outcome(
    outcome: ProducerOutcome,
) -> Mapping[str, Any] | None:
    provenance = outcome.provenance
    if not isinstance(provenance, Mapping):
        return None
    binding = provenance.get("lineage_parent_binding")
    if not isinstance(binding, Mapping):
        return None
    candidate_id = binding.get("candidate_id")
    program_digest = binding.get("program_digest")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return None
    if not isinstance(program_digest, str):
        return None
    return canonical_value(
        {
            "candidate_id": candidate_id,
            "program_digest": validate_sha256(
                program_digest,
                field_name="outcome.provenance.lineage_parent_binding.program_digest",
            ),
        }
    )


def _construction_parent_binding(
    context: Mapping[str, Any],
    *,
    outcome: ProducerOutcome | None = None,
) -> Mapping[str, Any] | None:
    lineage_binding = _lineage_parent_binding_from_context(context)
    if lineage_binding is None and outcome is not None:
        lineage_binding = _lineage_parent_binding_from_outcome(outcome)
    if lineage_binding is not None:
        return lineage_binding
    frozen_parent = _single_parent_source(context)
    frozen_binding = (
        frozen_parent.get("binding")
        if isinstance(frozen_parent, Mapping)
        else None
    )
    return (
        canonical_value(dict(frozen_binding))
        if isinstance(frozen_binding, Mapping)
        else None
    )


def _construction_parent_label(
    binding: Mapping[str, Any],
    frozen_parent: Mapping[str, Any] | None,
) -> str:
    frozen_binding = frozen_parent.get("binding") if frozen_parent is not None else None
    if canonical_value(binding) == canonical_value(frozen_binding):
        return "LightGCN++"
    return str(binding["candidate_id"])


_PROFILE_CONTROL_BINDING_SCHEMA = (
    "recclaw.research-line.active-profile-control-binding.v1"
)


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _declared_ablation(
    mechanism_program: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    payload = mechanism_program.get("program_payload", mechanism_program)
    if not isinstance(payload, Mapping):
        return None
    raw_plan = payload.get("ablation_plan")
    if (
        isinstance(raw_plan, (str, bytes))
        or not isinstance(raw_plan, (tuple, list))
        or not raw_plan
        or not isinstance(raw_plan[0], Mapping)
    ):
        return None
    ablation = dict(raw_plan[0])
    removed = ablation.get("remove_component_ids")
    if (
        not isinstance(ablation.get("ablation_id"), str)
        or not str(ablation["ablation_id"]).strip()
        or isinstance(removed, (str, bytes))
        or not isinstance(removed, (tuple, list))
        or not removed
        or not all(isinstance(item, str) and item.strip() for item in removed)
    ):
        raise ValueError(
            "mechanism ablation plan lacks an executable component identity"
        )
    return canonical_value(ablation)


def _declared_confirmation_bindings(
    *,
    selected: Any,
    context: Any,
    mechanism_program: Mapping[str, Any],
) -> dict[str, Any]:
    source = getattr(selected, "source_proposal", None)
    if source is None:
        return {}
    plan = source.matched_control_plan
    candidate_id = plan.comparator_candidate_id
    program_digest = plan.comparator_program_digest
    if candidate_id is None or program_digest is None:
        return {}
    if plan.protocol_digest != context.protocol_digest:
        raise ValueError(
            "matched-control plan protocol differs from active Research context"
        )
    validated_program_digest = validate_sha256(
        program_digest,
        field_name="matched_control_plan.comparator_program_digest",
    )
    catalog = executable_mechanisms()
    matches = tuple(
        mechanism
        for mechanism in catalog
        if mechanism.candidate_id == candidate_id
        and mechanism.mechanism_program_digest == validated_program_digest
    )
    if not matches:
        return {}
    if len(matches) != 1:
        raise ValueError(
            "matched-control identity is not unique in the executable catalog"
        )
    control = matches[0]
    profile_binding = canonical_value(
        {
            "schema": _PROFILE_CONTROL_BINDING_SCHEMA,
            "binding_origin": "PACKAGE_CONTROL_CATALOG",
            "catalog_ref": "BL_ICF_EXECUTABLE_CONTROL_CATALOG_V1",
            "catalog_digest": sha256_digest(
                tuple(
                    (
                        item.candidate_id,
                        item.mechanism_semantics_digest,
                        item.mechanism_program_digest,
                    )
                    for item in catalog
                )
            ),
            "protocol_digest": context.protocol_digest,
            "candidate_id": _required_text(
                candidate_id,
                field="matched_control_plan.comparator_candidate_id",
            ),
            "semantic_identity_ref": f"bl-icf-mechanism:{control.mechanism_id}",
            "semantic_identity_digest": control.mechanism_semantics_digest,
            "mechanism_id": control.mechanism_id,
            "bound_program_digest": validated_program_digest,
            "changed_axis": _required_text(plan.changed_axis, field="changed_axis"),
        }
    )
    binding = canonical_value(
        {
            "candidate_id": control.candidate_id,
            "candidate_semantic_digest": control.mechanism_semantics_digest,
            "mechanism_program": control.mechanism_program,
            "mechanism_program_digest": control.mechanism_program_digest,
            "binding_origin": "PACKAGE_CONTROL_CATALOG",
            "bound_program_digest": control.mechanism_program_digest,
            "control_identity": profile_binding,
        }
    )
    bindings: dict[str, Any] = {"matched_control_binding": binding}
    ablation = _declared_ablation(mechanism_program)
    source_matches = tuple(
        mechanism
        for mechanism in catalog
        if canonical_value(mechanism.mechanism_program)
        == canonical_value(mechanism_program)
    )
    if (
        ablation is not None
        and len(source_matches) == 1
        and source_matches[0].mechanism_id != control.mechanism_id
        and source_matches[0].base_mechanism_id == control.mechanism_id
    ):
        bindings["mechanism_off_binding"] = canonical_value(
            {
                **binding,
                "ablation": ablation,
                "ablation_source": "DECLARED_MECHANISM_PROGRAM",
                "ablation_realization": "MATCHED_PARENT_CAPABILITY",
                "source_mechanism_id": source_matches[0].mechanism_id,
                "source_mechanism_semantics_digest": (
                    source_matches[0].mechanism_semantics_digest
                ),
            }
        )
    return bindings


def _resolve_declared_control_binding(
    *,
    selected: Any,
    context: Any,
    mechanism_program: Mapping[str, Any],
    requested_kind: str,
) -> Mapping[str, Any] | None:
    if requested_kind not in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
        raise ValueError("requested control kind is outside the BL adapter contract")
    bindings = _declared_confirmation_bindings(
        selected=selected,
        context=context,
        mechanism_program=mechanism_program,
    )
    ordered = (
        ("matched_control_binding", "MATCHED_CONTROL"),
        ("mechanism_off_binding", "MECHANISM_OFF"),
    )
    if requested_kind == "MECHANISM_OFF":
        ordered = (ordered[1],)
    for key, operation in ordered:
        binding = bindings.get(key)
        if not isinstance(binding, Mapping):
            continue
        if (
            operation == "MATCHED_CONTROL"
            and binding.get("mechanism_program_digest")
            == sha256_digest(mechanism_program)
        ):
            continue
        required = binding.get("candidate_id")
        if operation == "MECHANISM_OFF":
            ablation = binding.get("ablation")
            if isinstance(ablation, Mapping):
                required = ablation.get("ablation_id", required)
        if not isinstance(required, str) or not required.strip():
            raise ValueError(
                "declared control binding lacks a required executable identity"
            )
        return canonical_value(
            {
                "operation": operation,
                "required_seed_or_control": required,
                "execution_binding": binding,
            }
        )
    return None


def _materialize_verification_control(
    *,
    pending_task: Mapping[str, Any],
    context: Any,
    policy: Any,
) -> Mapping[str, Any] | None:
    task_record = pending_task.get("task_record")
    task_payload = task_record if isinstance(task_record, Mapping) else pending_task
    executable = pending_task.get("execution_binding")
    if not isinstance(executable, Mapping):
        task_metadata = (
            task_record.get("metadata")
            if isinstance(task_record, Mapping)
            else pending_task.get("metadata")
        )
        if isinstance(task_metadata, Mapping):
            executable = task_metadata.get("execution_binding")
    if not isinstance(executable, Mapping) or executable.get("binding_origin") not in {
        "PACKAGE_CONTROL_CATALOG",
        "ACTIVE_PROFILE",
    }:
        return None
    task_program = pending_task.get(
        "mechanism_program", task_payload.get("mechanism_program")
    )
    task_program_digest = pending_task.get(
        "mechanism_program_digest", task_payload.get("mechanism_program_digest")
    )
    task_semantics = pending_task.get(
        "candidate_semantic_digest", task_payload.get("candidate_semantic_digest")
    )
    task_candidate_id = pending_task.get(
        "candidate_id", task_payload.get("candidate_id")
    )
    if (
        not isinstance(task_program, Mapping)
        or not isinstance(task_program_digest, str)
        or not isinstance(task_semantics, str)
        or not isinstance(task_candidate_id, str)
        or executable.get("candidate_id") != task_candidate_id
        or executable.get("mechanism_program_digest") != task_program_digest
        or executable.get("candidate_semantic_digest") != task_semantics
    ):
        return None
    task_report = compile_campaign_program(task_program)
    if (
        not task_report.is_valid
        or task_report.mechanism_program_digest != task_program_digest
        or task_report.mechanism_semantics_digest != task_semantics
    ):
        return None
    profile = adapt_current_search_profile(campaign_id=context.campaign_id)
    if (
        profile.protocol_ref != context.protocol_ref
        or profile.protocol_digest != context.protocol_digest
    ):
        raise ValueError("BL control catalog protocol differs from Research context")
    control_context = replace(
        context,
        active_profile_ref=profile.profile_ref,
        active_profile_digest=profile.profile_digest,
    )
    proposals = bootstrap_search_pool(
        control_context,
        profile,
        policy,
        max_proposals=len(profile.entries),
    )
    matches = []
    for proposal in proposals:
        proposal_program = deep_thaw(proposal.mechanism_program)
        report = compile_campaign_program(proposal_program)
        if (
            report.is_valid
            and report.mechanism_program_digest == task_program_digest
            and report.mechanism_semantics_digest == task_semantics
            and canonical_value(proposal_program) == canonical_value(task_program)
        ):
            matches.append(proposal)
    if len(matches) != 1:
        return None
    proposal = matches[0]
    entry = next(
        (
            item
            for item in profile.entries
            if item.semantic_identity_ref
            == f"bl-icf-mechanism:{proposal.mechanism_id}"
            and item.semantic_identity_digest == task_semantics
        ),
        None,
    )
    if entry is None:
        return None
    return {
        "execution_profile": profile,
        "native_binding": bind_search_candidate(
            profile=profile,
            proposal=proposal,
            capability_ref=entry.capability_ref,
        ),
    }


class BlIcfSearchSpaceAdapterV1:
    """Compatibility adapter over the existing strict BL-ICF functions."""

    adapter_id = "recclaw.search-space-adapter.bl-icf.v1"
    supported_frozen_profile_kinds = ("OFFLINE_TOPN",)

    def primitive_slots(self, primitive_ids: Sequence[str]) -> Mapping[str, str]:
        requested = {str(primitive_id) for primitive_id in primitive_ids}
        result: dict[str, str] = {}
        for axis in BL_ICFProvider().prompt_projection()["axes"]:
            slot_id = str(axis["slot_id"])
            policy_axis = canonical_mechanism_axis(slot_id)
            if policy_axis is None:
                continue
            for primitive in axis["primitives"]:
                primitive_id = str(primitive["primitive_id"])
                if primitive_id in requested:
                    result[primitive_id] = policy_axis
        return canonical_value(result)

    def resolve_confirmation(
        self,
        kind: str,
        primary_binding: Any,
        context: Mapping[str, Any],
    ) -> ConfirmationResolutionV1:
        phase = context.get("phase")
        if phase in {"IDENTIFY_INNOVATION", "PREPARE_INNOVATION"}:
            outcome = primary_binding
            frozen_profile_ref = context.get("frozen_profile_ref")
            frozen_parent = _single_parent_source(context)
            if _uses_single_parent_context(context) and frozen_parent is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_FROZEN_PARENT_UNAVAILABLE",
                )
            if not isinstance(outcome, ProducerOutcome) or not isinstance(
                frozen_profile_ref, Mapping
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_INNOVATION_CONTEXT_INCOMPLETE",
                )
            construction_parent_binding = _construction_parent_binding(
                context,
                outcome=outcome,
            )
            if isinstance(construction_parent_binding, Mapping):
                supplied_programs = [
                    item
                    for item in (
                        outcome.source_mechanism_program,
                        getattr(outcome.source_proposal, "mechanism_program", None),
                    )
                    if isinstance(item, Mapping)
                ]
                if not supplied_programs:
                    return ConfirmationResolutionV1(
                        ConfirmationResolutionKindV1.UNSUPPORTED,
                        reason="BL_ICF_TYPED_SINGLE_PARENT_PROGRAM_REQUIRED",
                    )
                if any(
                    canonical_value(
                        parent_binding_for_mechanism_program(item)
                    )
                    != canonical_value(construction_parent_binding)
                    for item in supplied_programs
                ):
                    return ConfirmationResolutionV1(
                        ConfirmationResolutionKindV1.UNSUPPORTED,
                        reason="BL_ICF_CONSTRUCTION_PARENT_BINDING_MISMATCH",
                    )
                if (
                    outcome.spec is not None
                    and isinstance(outcome.spec.execution_contract, Mapping)
                ):
                    execution_contract = _single_parent_execution_contract(
                        outcome.spec.execution_contract
                    )
                    companion = outcome.implementation_companion
                    outcome = replace(
                        outcome,
                        spec=replace(
                            outcome.spec,
                            execution_contract=canonical_value(
                                execution_contract
                            ),
                            closest_parent=_construction_parent_label(
                                construction_parent_binding,
                                frozen_parent,
                            ),
                        ),
                        implementation_companion=(
                            replace(
                                companion,
                                execution_contract=canonical_value(
                                    execution_contract
                                ),
                            )
                            if companion is not None
                            else None
                        ),
                    )
            direct_program = outcome.source_mechanism_program
            compiler = (
                compile_bl_icf_candidate_source(
                    open_spec=None,
                    frozen_profile_ref=frozen_profile_ref,
                    mechanism_program=direct_program,
                    parent_binding=parent_binding_for_mechanism_program(
                        direct_program
                    ),
                )
                if direct_program is not None
                else None
            )
            if compiler is not None:
                companion = outcome.implementation_companion
                if companion is None:
                    raise ValueError(
                        "Provider-direct MechanismProgram lacks its implementation companion"
                    )
                outcome = replace(
                    outcome,
                    spec=implementation_spec_for_compiled_program(
                        compiler,
                        companion=companion,
                    ),
                )
            source = outcome.source_proposal
            if source is not None:
                compiler = compile_bl_icf_candidate_source(
                    open_spec=None,
                    frozen_profile_ref=frozen_profile_ref,
                    mechanism_program=deep_thaw(source.mechanism_program),
                    parent_binding=parent_binding_for_mechanism_program(
                        source.mechanism_program
                    ),
                )
            elif compiler is None:
                compiler = compile_bl_icf_candidate_source(
                    open_spec=outcome.spec,
                    frozen_profile_ref=frozen_profile_ref,
                )
            program = (
                deep_thaw(source.mechanism_program)
                if source is not None
                else deep_thaw(compiler.mechanism_program)
            )
            identity = bl_icf_effective_identity_for_program(program)
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "producer_outcome": outcome,
                    "compiler_binding": compiler,
                    "mechanism_program": program,
                    "compiled_implementation": compiler.implementation_payload(),
                    "semantic_identity_ref": compiler.semantic_identity_ref,
                    "semantic_identity_digest": compiler.mechanism_semantics_digest,
                    "effective_identity": identity,
                    "candidate_id": (
                        source.candidate_id
                        if source is not None
                        else compiler.compiler_candidate_id
                    ),
                    "mechanism_id": (
                        source.mechanism_id
                        if source is not None
                        else compiler.compiler_candidate_id
                    ),
                    "parent_available": (
                        parent_binding_for_mechanism_program(program) is not None
                        or bool(outcome.spec and outcome.spec.closest_parent)
                    ),
                },
            )
        if phase == "PACKAGE_QUALIFIED_INNOVATION":
            prepared = context.get("prepared_innovation")
            materialized = context.get("materialized")
            qualification = context.get("qualification")
            capability = context.get("capability")
            policy = context.get("implementer_policy")
            utility = context.get("utility_features")
            evidence = context.get("feature_evidence")
            if not isinstance(prepared, Mapping) or any(
                item is None
                for item in (materialized, qualification, capability, policy)
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_QUALIFIED_INNOVATION_CONTEXT_INCOMPLETE",
                )
            outcome = prepared["producer_outcome"]
            source = outcome.source_proposal
            compiler = prepared["compiler_binding"]
            if source is not None:
                search_candidate = (
                    replace(
                        source,
                        mechanism_program=deep_thaw(source.mechanism_program),
                        utility_features=SearchUtilityFeaturesV1(**dict(utility)),
                        feature_evidence=RouterFeatureEvidenceV1(**dict(evidence)),
                    )
                    if isinstance(utility, Mapping) and isinstance(evidence, Mapping)
                    else source
                )
            else:
                contract = policy.execution_contract
                if contract is None:
                    raise ValueError(
                        "qualified OpenSpec lacks its resolved execution contract"
                    )
                realization = bind_materialized_bl_icf_candidate(
                    compiler=compiler,
                    open_spec=outcome.spec,
                    materialized=materialized,
                )
                if not isinstance(utility, Mapping) or not isinstance(
                    evidence, Mapping
                ):
                    return ConfirmationResolutionV1(
                        ConfirmationResolutionKindV1.UNSUPPORTED,
                        reason="BL_ICF_QUALIFIED_INNOVATION_FEATURES_MISSING",
                    )
                search_candidate = OpenSpecSearchCandidateV1(
                    spec=outcome.spec,
                    capability_ref=capability.capability_id,
                    capability_digest=capability.digest,
                    candidate_package_ref=materialized.package.package_id,
                    candidate_package_digest=materialized.package.digest,
                    candidate_root_ref=materialized.package.candidate_root_ref,
                    candidate_root_digest=materialized.package.candidate_root_digest,
                    source_tree_digest=materialized.package.source_tree_digest,
                    executable_entrypoint=materialized.package.executable_entrypoint,
                    execution_contract=contract,
                    mechanism_program=compiler.mechanism_program,
                    compiler_candidate_id=compiler.compiler_candidate_id,
                    compile_report_digest=compiler.compile_report_digest,
                    mechanism_program_digest=compiler.mechanism_program_digest,
                    semantic_identity_ref=prepared["semantic_identity_ref"],
                    semantic_identity_digest=prepared["semantic_identity_digest"],
                    realization_identity_ref=realization.realization_identity_ref,
                    realization_semantics_digest=realization.realization_semantics_digest,
                    implementation_receipt_ref=realization.implementation_receipt_ref,
                    implementation_receipt_digest=realization.implementation_receipt_digest,
                    realization_binding_ref=realization.binding_ref,
                    realization_binding_digest=realization.binding_digest,
                    qualification_receipt_ref=qualification.receipt.receipt_id,
                    qualification_receipt_digest=qualification.receipt.digest,
                    mechanism_axis=str(context.get("mechanism_axis", "")),
                    utility_features=SearchUtilityFeaturesV1(**dict(utility)),
                    feature_evidence=RouterFeatureEvidenceV1(**dict(evidence)),
                )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={"search_candidate": search_candidate},
            )
        if phase == "PROJECT_PROVIDER_PROPOSAL":
            from .producers import _project_result

            research_context = context.get("research_context")
            producer_role = context.get("producer_role")
            bindings = context.get("producer_bindings")
            frozen_profile_ref = context.get("frozen_profile_ref")
            if (
                research_context is None
                or not isinstance(producer_role, str)
                or not isinstance(bindings, Mapping)
                or not isinstance(frozen_profile_ref, Mapping)
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_PROVIDER_PROJECTION_CONTEXT_INCOMPLETE",
                )
            frozen_parent = _single_parent_source(context)
            single_parent_mode = _uses_single_parent_context(context)
            if single_parent_mode and frozen_parent is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_FROZEN_PARENT_UNAVAILABLE",
                )
            direct_program = getattr(primary_binding, "mechanism_program", None)
            if single_parent_mode and not isinstance(direct_program, Mapping):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_TYPED_SINGLE_PARENT_PROGRAM_REQUIRED",
                )
            if single_parent_mode:
                supplied_parent = parent_binding_for_mechanism_program(
                    direct_program
                )
                expected_parent = _construction_parent_binding(context)
                if canonical_value(supplied_parent) != canonical_value(
                    expected_parent
                ):
                    return ConfirmationResolutionV1(
                        ConfirmationResolutionKindV1.UNSUPPORTED,
                        reason="BL_ICF_CONSTRUCTION_PARENT_BINDING_MISMATCH",
                    )
            spec, facts, proposal, program, companion = _project_result(
                producer_role,
                primary_binding,
                bindings=bindings,
                frozen_profile_ref=frozen_profile_ref,
            )
            if single_parent_mode and isinstance(spec.execution_contract, Mapping):
                execution_contract = _single_parent_execution_contract(
                    spec.execution_contract
                )
                spec = replace(
                    spec,
                    execution_contract=canonical_value(execution_contract),
                    closest_parent=_construction_parent_label(
                        expected_parent,
                        frozen_parent,
                    ),
                )
                if companion is not None:
                    companion = replace(
                        companion,
                        execution_contract=canonical_value(execution_contract),
                    )
            outcome = ProducerOutcome(
                producer_role=producer_role,
                context_ref=research_context.context_ref,
                context_digest=research_context.digest,
                spec=spec,
                resolution_facts=facts,
                source_proposal=proposal,
                source_mechanism_program=program,
                implementation_companion=companion,
                provenance={
                    "producer_role": producer_role,
                    "context_ref": research_context.context_ref,
                    "context_digest": research_context.digest,
                    "spec_id": spec.spec_id,
                    "spec_digest": spec.digest,
                    "source_proposal_digest": (
                        proposal.digest if proposal is not None else None
                    ),
                    "source_mechanism_program_digest": (
                        sha256_digest(program) if program is not None else None
                    ),
                    "status": "PRODUCED",
                },
            )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={"producer_outcome": outcome},
            )
        if phase == "MATERIALIZE_DISCOVERY":
            resolution = context.get("resolution")
            profile = context.get("execution_profile")
            proposal = getattr(primary_binding, "source_proposal", None)
            capability_ref = getattr(
                resolution, "resolved_current_capability_ref", None
            )
            if proposal is None or profile is None or capability_ref is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_DISCOVERY_BINDING_CONTEXT_INCOMPLETE",
                )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "native_binding": bind_search_candidate(
                        profile=profile,
                        proposal=proposal,
                        capability_ref=capability_ref,
                    )
                },
            )
        if kind not in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.UNSUPPORTED,
                reason="BL_ICF_CONFIRMATION_KIND_UNSUPPORTED",
            )
        if phase == "MATERIALIZE_VERIFICATION":
            pending_task = context.get("pending_task")
            research_context = context.get("research_context")
            policy = context.get("policy")
            if not isinstance(pending_task, Mapping) or research_context is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_VERIFICATION_CONTEXT_INCOMPLETE",
                )
            materialized = _materialize_verification_control(
                pending_task=pending_task,
                context=research_context,
                policy=policy,
            )
            if materialized is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="BL_ICF_VERIFICATION_BINDING_UNAVAILABLE",
                )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding=materialized,
            )
        if phase in {"DECLARE_FOLLOWUP", "GUARD_CONTROL"}:
            research_context = context.get("research_context")
            mechanism_program = context.get("mechanism_program")
            if research_context is None or not isinstance(
                mechanism_program, Mapping
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.NEEDS_PROPOSAL,
                    reason="BL_ICF_CONTROL_CONTEXT_INCOMPLETE",
                )
            resolved = _resolve_declared_control_binding(
                selected=primary_binding,
                context=research_context,
                mechanism_program=mechanism_program,
                requested_kind=kind,
            )
            if resolved is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.NEEDS_PROPOSAL,
                    reason="BL_ICF_DECLARED_CONTROL_NEEDS_PROPOSAL",
                )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding=resolved,
            )
        if phase != "MATCH_PROPOSAL":
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.UNSUPPORTED,
                reason="BL_ICF_CONFIRMATION_PHASE_UNSUPPORTED",
            )
        pending_task = context.get("pending_task")
        next_test = context.get("next_discriminative_test")
        if not isinstance(pending_task, Mapping) or not isinstance(next_test, str):
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.UNSUPPORTED,
                reason="BL_ICF_CONFIRMATION_CONTEXT_INCOMPLETE",
            )
        if not _confirmation_candidate_matches(
            pending_task=pending_task,
            selected_outcome=primary_binding,
            next_discriminative_test=next_test,
        ):
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.NEEDS_PROPOSAL,
                reason="BL_ICF_EXACT_CONFIRMATION_NOT_PROPOSED",
            )
        source = getattr(primary_binding, "source_proposal", None)
        spec = getattr(primary_binding, "spec", None)
        candidate_id = getattr(source, "candidate_id", None)
        if candidate_id is None and spec is not None:
            candidate_id = getattr(spec, "spec_id", None)
        opaque = canonical_value(
            {
                "adapter_id": self.adapter_id,
                "candidate_id": candidate_id,
                "outcome_digest": getattr(primary_binding, "digest", None),
                "task_id": pending_task.get("task_id"),
                "operation": kind,
            }
        )
        return ConfirmationResolutionV1(
            ConfirmationResolutionKindV1.EXACT_BINDING,
            binding=opaque,
        )

    def validate_execution_binding(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> None:
        self.execution_recipe(binding)

    def effective_identity(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]:
        return bl_icf_effective_identity_for_program(
            binding.native_binding.proposal.mechanism_program
        )

    def execution_recipe(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]:
        context = binding.execution_context
        return execution_recipe_for_search_binding(
            binding.native_binding,
            profile=context["profile"],
            qualified_execution=context.get("qualified_execution"),
            evaluator=context["evaluator"],
            split=context["split"],
        )


def default_search_space_adapter() -> BlIcfSearchSpaceAdapterV1:
    return BlIcfSearchSpaceAdapterV1()


def _complete_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in (
            "slot_id",
            "dimension",
            "mechanism_dimension",
            "changed_axis",
            "axis",
            "mechanism_axis",
        ):
            item = value.get(key)
            if isinstance(item, str) and item.strip() and item.strip() not in result:
                result.append(item.strip())
        return tuple(result)
    if not isinstance(value, (tuple, list)):
        return ()
    result = []
    for item in value:
        for identity in _complete_values(item):
            if identity not in result:
                result.append(identity)
    return tuple(result)


def _program_values(program: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(program, Mapping):
        return ()
    payload = program.get("program_payload", program)
    if not isinstance(payload, Mapping):
        return ()
    declared: list[str] = []
    raw: list[str] = []
    for key in (
        "mechanism_axis_footprint",
        "changed_axis_footprint",
        "changed_axes",
        "changed_dimensions",
        "high_change_dimensions",
        "changed_slots",
    ):
        for identity in _complete_values(payload.get(key, ())):
            if identity not in declared:
                declared.append(identity)
            if identity not in MECHANISM_AXIS_UNIVERSE_V1 and identity not in raw:
                raw.append(identity)
    return tuple(raw or declared)


def _axis_footprint(outcome: Any) -> tuple[str, ...]:
    resolution = outcome.resolution_facts
    programs = (
        outcome.source_mechanism_program,
        (
            outcome.source_proposal.mechanism_program
            if outcome.source_proposal is not None
            else None
        ),
    )
    program_values: list[str] = []
    for program in programs:
        for identity in _program_values(program):
            if identity not in program_values:
                program_values.append(identity)
    resolution_values: list[str] = []
    for key in (
        "mechanism_axis_footprint",
        "changed_axis_footprint",
        "changed_axes",
        "changed_dimensions",
    ):
        for identity in _complete_values(resolution.get(key, ())):
            if identity not in resolution_values:
                resolution_values.append(identity)
    partial_raw = tuple(
        value
        for value in resolution_values
        if value not in MECHANISM_AXIS_UNIVERSE_V1
    )
    program_raw = tuple(
        value for value in program_values if value not in MECHANISM_AXIS_UNIVERSE_V1
    )
    if program_values:
        return tuple(program_raw or partial_raw or program_values)
    if resolution_values:
        return tuple(partial_raw or resolution_values)
    result = list(_complete_values(resolution.get("high_change_dimensions", ())))
    if not result:
        source = outcome.source_proposal
        result.extend(
            _complete_values(
                source.mechanism_axis
                if source is not None
                else resolution.get("mechanism_axis", ())
            )
        )
    return tuple(result)


def _confirmation_candidate_matches(
    *,
    pending_task: Mapping[str, Any],
    selected_outcome: Any,
    next_discriminative_test: str,
) -> bool:
    """Own the strict BL-ICF structural match behind the generic port."""

    del next_discriminative_test
    task_record = pending_task.get("task_record")
    if not isinstance(task_record, Mapping):
        return False
    metadata = task_record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if metadata.get("execution_state") != "AWAITING_CANDIDATE_BINDING":
        return False
    operation = getattr(task_record.get("operation"), "value", task_record.get("operation"))
    if operation not in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
        return False
    source = getattr(selected_outcome, "source_proposal", None)
    spec = getattr(selected_outcome, "spec", None)
    if spec is None:
        return False
    # A source-less OpenSpec can be the Provider's direct, compiler-bound
    # confirmation proposal.  Its parent/control prose is not sufficient on
    # its own; the program checks below must prove the requested operation.
    program = (
        source.mechanism_program
        if source is not None
        else getattr(selected_outcome, "source_mechanism_program", None)
    )
    if not isinstance(program, Mapping):
        return False
    if (
        source is None
        and metadata.get("binding_requirement") == "EXACT_EFFECTIVE_IDENTITY"
        and isinstance(task_record.get("mechanism_program_digest"), str)
        and sha256_digest(program) == task_record["mechanism_program_digest"]
        and task_record.get("protocol_digest") == spec.protocol_digest
    ):
        # The queued task already owns the exact executable program identity.
        # A source-less Provider result that copies those bytes is the requested
        # binding even when the program's internal parent is its legitimate BL
        # base rather than the frontier candidate that requested the control.
        return True
    anchor_candidate = metadata.get("frontier_candidate_id")
    if source is not None and source.candidate_id == anchor_candidate:
        return False
    requested_parent = task_record.get("parent_candidate_id")
    payload = program.get("program_payload", program)
    payload = payload if isinstance(payload, Mapping) else {}
    candidate_parents = {
        str(value)
        for value in (
            source.parent_candidate_id if source is not None else None,
            getattr(spec, "closest_parent", None),
        )
        if isinstance(value, str) and value
    }
    parent_refs = payload.get("parent_refs")
    if isinstance(parent_refs, (tuple, list)):
        candidate_parents.update(
            str(item["candidate_id"])
            for item in parent_refs
            if isinstance(item, Mapping)
            and isinstance(item.get("candidate_id"), str)
            and item["candidate_id"]
        )
    frontier_parent_ids: set[str] = set()
    if source is None:
        task_program = task_record.get("mechanism_program")
        task_program = task_program if isinstance(task_program, Mapping) else {}
        semantic_ref = task_program.get("semantic_identity_ref")
        frontier_parent_ids = {
            str(value)
            for value in (
                anchor_candidate,
                task_record.get("candidate_id"),
                (
                    semantic_ref.rsplit(":", 1)[-1]
                    if isinstance(semantic_ref, str) and semantic_ref
                    else None
                ),
            )
            if isinstance(value, str) and value
        }
    raw_control = payload.get("matched_control")
    raw_control = raw_control if isinstance(raw_control, Mapping) else {}
    direct_control_ref = raw_control.get("control_ref")
    exact_parent_control_refs = set(frontier_parent_ids)
    if isinstance(requested_parent, str) and requested_parent:
        exact_parent_control_refs.add(requested_parent)
    direct_frontier_control = (
        source is None
        and isinstance(direct_control_ref, str)
        and direct_control_ref in exact_parent_control_refs
    )
    if isinstance(requested_parent, str) and requested_parent:
        if requested_parent not in candidate_parents:
            return False
    elif (
        source is None
        and not direct_frontier_control
        and not candidate_parents.intersection(frontier_parent_ids)
    ):
        return False

    control_plan = source.matched_control_plan.to_dict() if source is not None else {}
    comparator_values = {
        value
        for value in (
            control_plan.get("comparator_candidate_id"),
            control_plan.get("comparator_program_digest"),
            control_plan.get("protocol_digest"),
            payload.get("comparator_identity"),
            raw_control.get("control_ref"),
            raw_control.get("comparator_identity"),
            raw_control.get("comparator_candidate_id"),
            raw_control.get("comparator_program_digest"),
        )
        if isinstance(value, str) and value
    }
    if (
        task_record.get("comparator_identity") not in comparator_values
        and not direct_frontier_control
    ):
        return False
    if operation == "MATCHED_CONTROL":
        if not (
            control_plan.get("comparator_candidate_id")
            or control_plan.get("comparator_program_digest")
            or raw_control.get("control_ref")
            or raw_control.get("comparator_identity")
            or raw_control.get("comparator_candidate_id")
            or raw_control.get("comparator_program_digest")
        ):
            return False
    elif not (payload.get("ablation_plan") or payload.get("ablation")):
        return False

    requested_footprint = _complete_values(
        metadata.get("mechanism_axis_footprint", ())
    )
    candidate_footprint = _axis_footprint(selected_outcome)
    if not requested_footprint or not set(requested_footprint).issubset(
        set(candidate_footprint)
    ):
        return False
    if source is None:
        removed_slots = set(_complete_values(payload.get("removed_slots", ())))
        component_slots = {
            str(component.get("slot_id"))
            for component in payload.get("components", ())
            if isinstance(component, Mapping)
            and isinstance(component.get("slot_id"), str)
            and component.get("slot_id")
        }
        requested_slots = set(requested_footprint)
        if operation == "MECHANISM_OFF":
            if not requested_slots.issubset(removed_slots):
                return False
            if requested_slots.intersection(component_slots):
                return False
        else:
            provenance = getattr(selected_outcome, "provenance", None)
            provenance = provenance if isinstance(provenance, Mapping) else {}
            program_digest = provenance.get("source_mechanism_program_digest")
            try:
                validate_sha256(
                    program_digest,
                    field_name="provenance.source_mechanism_program_digest",
                )
            except (TypeError, ValueError):
                return False
            changed_slots = payload.get("changed_slots", ())
            if not isinstance(changed_slots, (tuple, list)) or not changed_slots:
                return False
            if not all(
                isinstance(item, Mapping)
                and isinstance(item.get("slot_id"), str)
                and item.get("slot_id")
                for item in changed_slots
            ):
                return False
            declared_changed_slots = {
                str(item["slot_id"]) for item in changed_slots
            }
            if provenance.get("status") != "PRODUCED":
                return False
            if not direct_frontier_control:
                return False
            if declared_changed_slots != requested_slots:
                return False
            if not requested_slots.issubset(component_slots):
                return False
            if requested_slots.intersection(removed_slots):
                return False
    requested_observables = metadata.get("expected_observable", ())
    if isinstance(requested_observables, str):
        requested_observables = (requested_observables,)
    if not isinstance(requested_observables, (tuple, list)):
        return False
    requested_falsifier = metadata.get("falsifier")
    if source is not None and not set(requested_observables).issubset(
        set(spec.expected_evidence)
    ):
        return False
    if source is not None and requested_falsifier != spec.falsifier:
        return False
    if source is None and (
        not spec.expected_evidence
        or not isinstance(spec.falsifier, str)
        or not spec.falsifier.strip()
        or not (payload.get("ablation_plan") or payload.get("ablation"))
    ):
        return False
    if task_record.get("protocol_digest") != spec.protocol_digest:
        return False
    contrast = metadata.get("core_mechanism_contrast")
    if isinstance(contrast, str) and contrast:
        declared_contrast = payload.get(
            "core_mechanism_contrast",
            payload.get("mechanism_contrast", spec.mechanism_change),
        )
        if source is not None and declared_contrast != contrast:
            return False
        if source is None and (
            not isinstance(declared_contrast, str) or not declared_contrast.strip()
        ):
            return False
    return True


__all__ = ["BlIcfSearchSpaceAdapterV1", "default_search_space_adapter"]
