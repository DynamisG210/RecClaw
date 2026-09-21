"""Existing native conceptual binding/qualification consumption, graph identity only."""
from __future__ import annotations
import json
from typing import Any, Mapping
from auto_search_space import ConceptualSearchCandidateBindingV1
class GraphNativeSearchCandidate:
    """Graph candidate record consumed by RO's non-BL-ICF routing branch."""

    schema = "recclaw.external-comparison.graph-native-search-candidate.v1"

    def __init__(self, **values: Any):
        self.__dict__.update(values)

    @property
    def digest(self) -> str:
        from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest

        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value

        payload = dict(self.__dict__)
        for name in ("utility_features", "feature_evidence"):
            value = payload.get(name)
            if hasattr(value, "to_dict"):
                payload[name] = value.to_dict()
        return canonical_value({"schema": self.schema, **payload})


class GraphProviderContextAdapter:
    """Preserve the RO Producer while replacing only its task context."""

    def __init__(self, producer: Any, adapter: "GraphRoSearchSpaceAdapter"):
        self._producer = producer
        self._adapter = adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._producer, name)

    def __call__(self, producer_role: str, context_view: Mapping[str, Any]) -> Any:
        return self._producer(producer_role, self._adapter.provider_context(context_view))

    def call_with_namespace(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
        *,
        logical_namespace: str,
    ) -> Any:
        return self._producer.call_with_namespace(
            producer_role,
            self._adapter.provider_context(context_view),
            logical_namespace=logical_namespace,
        )


class GraphImplementerContextAdapter:
    """Preserve the RO Implementer while supplying the shared initial program."""

    def __init__(self, implementer: Any, adapter: "GraphRoSearchSpaceAdapter"):
        self._implementer = implementer
        self._adapter = adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._implementer, name)

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        projected = self._adapter.implementation_context(request)
        response = self._implementer(projected)
        return self._adapter.canonicalize_implementation_response(response, projected)

    def call_with_namespace(
        self,
        request: Mapping[str, Any],
        *,
        logical_namespace: str,
    ) -> Mapping[str, Any]:
        projected = self._adapter.implementation_context(request)
        response = self._implementer.call_with_namespace(
            projected,
            logical_namespace=logical_namespace,
        )
        return self._adapter.canonicalize_implementation_response(response, projected)


class GraphNativeMethods:
    def bind_conceptual_proposal(
        self,
        *,
        proposal_ref: str,
        proposal_digest: str,
        primitive_ids: list[str] | tuple[str, ...],
        operator_id: str | None = None,
        target_primitive_id: str | None = None,
        replacement_primitive_id: str | None = None,
        custom_components: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] = (),
    ) -> Any:
        """Bind a proposal to AutoSpace before implementation or qualification."""

        return self.conceptual_space.bind_proposal(
            proposal_ref=proposal_ref,
            proposal_digest=proposal_digest,
            primitive_ids=primitive_ids,
            operator_id=operator_id,
            target_primitive_id=target_primitive_id,
            replacement_primitive_id=replacement_primitive_id,
            custom_components=custom_components,
        )


    def _trusted_conceptual_binding(
        self, mechanism_program: Mapping[str, Any]
    ) -> ConceptualSearchCandidateBindingV1:
        conceptual = mechanism_program.get("conceptual_mechanism_program")
        if not isinstance(conceptual, Mapping):
            raise ValueError("Graph_CONCEPTUAL_SEARCH_BINDING_MISSING")
        if (
            conceptual.get("proposal_ref") != mechanism_program.get("spec_ref")
            or conceptual.get("proposal_digest") != mechanism_program.get("spec_digest")
        ):
            raise ValueError("Graph_CONCEPTUAL_SEARCH_BINDING_IDENTITY_DRIFT")
        if conceptual.get("search_space_digest") != self.conceptual_space.digest:
            raise ValueError("Graph_CONCEPTUAL_SEARCH_SPACE_IDENTITY_DRIFT")
        rebound = self.bind_conceptual_proposal(
            proposal_ref=str(conceptual.get("proposal_ref", "")),
            proposal_digest=str(conceptual.get("proposal_digest", "")),
            primitive_ids=tuple(conceptual.get("primitive_ids", ())),
            operator_id=conceptual.get("operator_id"),
            target_primitive_id=conceptual.get("target_primitive_id"),
            replacement_primitive_id=conceptual.get("replacement_primitive_id"),
            custom_components=tuple(conceptual.get("custom_components", ())),
        )
        if rebound.as_dict() != dict(conceptual):
            raise ValueError("Graph_CONCEPTUAL_SEARCH_BINDING_IDENTITY_DRIFT")
        return rebound


    def _prepare_native_innovation(self, outcome: Any) -> Any:
        """Create the Graph identity that replaces RO's BL-ICF compiler identity."""

        from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest

        spec = outcome.spec
        candidate_id = f"cand-graph-{spec.digest[:24]}"
        mechanism_id = f"graph-native-{spec.digest[:24]}"
        axis = "architecture"
        for dimension in outcome.resolution_facts.get("high_change_dimensions", ()):
            if isinstance(dimension, str) and dimension.strip():
                axis = dimension.strip()
                break
        program = {
            "schema": "recclaw.external-comparison.graph-native-mechanism.v1",
            "candidate_id": candidate_id,
            "producer_role": outcome.producer_role,
            "spec_ref": spec.spec_id,
            "spec_digest": spec.digest,
            "mechanism_id": mechanism_id,
            "mechanism_axis": axis,
            "execution_contract": spec.execution_contract,
            "mechanism_hypothesis": spec.hypothesis,
            "mechanism_change": spec.mechanism_change,
        }
        conceptual_binding = outcome.provenance.get("conceptual_binding")
        if not isinstance(conceptual_binding, Mapping):
            raise ValueError("Graph_CONCEPTUAL_SEARCH_BINDING_MISSING")
        if conceptual_binding.get("search_space_digest") != self.conceptual_space.digest:
            raise ValueError("Graph_CONCEPTUAL_SEARCH_SPACE_IDENTITY_DRIFT")
        primitive_ids = tuple(conceptual_binding.get("primitive_ids", ()))
        rebound = self.bind_conceptual_proposal(
            proposal_ref=spec.spec_id,
            proposal_digest=spec.digest,
            primitive_ids=primitive_ids,
            operator_id=conceptual_binding.get("operator_id"),
            target_primitive_id=conceptual_binding.get("target_primitive_id"),
            replacement_primitive_id=conceptual_binding.get("replacement_primitive_id"),
            custom_components=tuple(conceptual_binding.get("custom_components", ())),
        )
        program["conceptual_mechanism_program"] = rebound.as_dict()
        semantic_digest = sha256_digest(
            {"adapter": self.adapter_id, "program": program}
        )
        program["semantic_identity_ref"] = f"graph-native-mechanism:{candidate_id}"
        program["semantic_identity_digest"] = semantic_digest
        effective_identity = {
            "effective_experiment_digest": sha256_digest(
                {
                    "adapter": self.adapter_id,
                    "candidate": semantic_digest,
                    "task": self.task_id,
                }
            ),
            "effective_family_digest": self.conceptual_space.family_digest,
            "primitive_ids": primitive_ids,
        }
        return {
            "producer_outcome": outcome,
            "compiler_binding": None,
            "compiled_implementation": None,
            "mechanism_program": program,
            "semantic_identity_ref": program["semantic_identity_ref"],
            "semantic_identity_digest": semantic_digest,
            "effective_identity": effective_identity,
            "candidate_id": candidate_id,
            "mechanism_id": mechanism_id,
            "mechanism_axis": axis,
            "parent_available": bool(spec.closest_parent),
        }


    def _native_candidate(
        self,
        prepared: Mapping[str, Any],
        outcome: Any,
        utility: Mapping[str, Any],
        evidence: Mapping[str, Any],
        *,
        capability: Any,
        materialized: Any,
        qualification: Any,
    ) -> Any:
        from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
        from recclaw_core.experiments.helix_abc_v1.research_contracts import (
            RouterFeatureEvidenceV1,
            SearchUtilityFeaturesV1,
        )

        spec = outcome.spec
        candidate_id = str(prepared["candidate_id"])
        utility_value = SearchUtilityFeaturesV1(**dict(utility))
        evidence_value = RouterFeatureEvidenceV1(
            **{
                **dict(evidence),
                "llm_diagnostic": (
                    evidence.get("llm_diagnostic", utility_value)
                    if isinstance(evidence.get("llm_diagnostic"), SearchUtilityFeaturesV1)
                    else SearchUtilityFeaturesV1(
                        **dict(evidence.get("llm_diagnostic", utility))
                    )
                ),
            }
        )
        values = {
            "spec": spec,
            "candidate_id": candidate_id,
            "producer_role": outcome.producer_role,
            "mechanism_id": str(prepared["mechanism_id"]),
            "mechanism_axis": str(prepared["mechanism_axis"]),
            "mechanism_program": prepared["mechanism_program"],
            "utility_features": utility_value,
            "feature_evidence": evidence_value,
            "semantic_identity_ref": prepared["semantic_identity_ref"],
            "semantic_identity_digest": prepared["semantic_identity_digest"],
            "compile_report_digest": None,
            "mechanism_program_digest": sha256_digest(prepared["mechanism_program"]),
            "compiler_candidate_id": candidate_id,
            "candidate_package_ref": materialized.package.package_id,
            "candidate_package_digest": materialized.package.digest,
            "candidate_root_ref": materialized.package.candidate_root_ref,
            "candidate_root_digest": materialized.package.candidate_root_digest,
            "source_tree_digest": materialized.package.source_tree_digest,
            "executable_entrypoint": materialized.package.executable_entrypoint,
            "execution_contract": spec.execution_contract,
            "capability_ref": capability.capability_id,
            "capability_digest": capability.digest,
            "qualification_receipt_ref": qualification.receipt.receipt_id,
            "qualification_receipt_digest": qualification.receipt.digest,
            "parent_candidate_id": None,
            "realization_identity_ref": prepared["semantic_identity_ref"],
            "realization_semantics_digest": prepared["semantic_identity_digest"],
            "realization_binding_ref": prepared["semantic_identity_ref"],
            "realization_binding_digest": prepared["semantic_identity_digest"],
            "candidate_label": spec.hypothesis,
            "mechanism_hypothesis": spec.hypothesis,
            "competing_hypothesis": spec.competing_explanation,
            "predicted_outcome_signature": "; ".join(spec.expected_evidence),
            "failure_mode": spec.falsifier,
        }
        return GraphNativeSearchCandidate(**values)


    def bind_search_candidate(self, *, profile: Any, proposal: Any, capability_ref: str) -> Any:
        """Bind an Graph candidate without invoking RO's BL-ICF compiler."""

        from recclaw_core.experiments.helix_abc_v1.search_adapter import (
            SearchCandidateBindingV1,
            SearchProfileEntryOriginV1,
        )

        entry = profile.entry(capability_ref)
        if entry.origin is not SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
            raise ValueError("Graph candidates require a qualified registry entry")
        program = proposal.mechanism_program
        semantic_digest = program.get("semantic_identity_digest")
        if semantic_digest != entry.semantic_identity_digest:
            raise ValueError("Graph candidate semantic identity drift")
        return SearchCandidateBindingV1(
            proposal=proposal,
            capability_ref=entry.capability_ref,
            capability_digest=entry.capability_digest,
            executable_entrypoint=entry.executable_entrypoint,
            entry_origin=entry.origin,
            mechanism_semantics_digest=semantic_digest,
        )


    def resolve_confirmation(
        self,
        kind: str,
        primary_binding: Any,
        context: Mapping[str, Any],
    ) -> Any:
        from recclaw_core.research_line.search_space_adapter import (
            ConfirmationResolutionKindV1,
            ConfirmationResolutionV1,
        )

        if context.get("phase") in {
            "IDENTIFY_INNOVATION",
            "PREPARE_INNOVATION",
        }:
            from recclaw_core.research_line.interfaces import ProducerOutcome

            if not isinstance(primary_binding, ProducerOutcome) or primary_binding.spec is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="Graph_INNOVATION_CONTEXT_INCOMPLETE",
                )
            prepared = self._prepare_native_innovation(primary_binding)
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding=prepared,
            )

        if context.get("phase") == "PROJECT_PROVIDER_PROPOSAL":
            from recclaw_core.experiments.helix_abc_v1.canonical import (
                sha256_digest,
            )
            from recclaw_core.research_line.interfaces import ProducerOutcome
            from recclaw_core.research_line.producers import _project_result

            research_context = context.get("research_context")
            producer_role = context.get("producer_role")
            bindings = context.get("producer_bindings")
            frozen_profile_ref = context.get("frozen_profile_ref")
            if (
                research_context is None
                or not isinstance(producer_role, str)
                or not isinstance(bindings, Mapping)
                or not isinstance(frozen_profile_ref, Mapping)
                or not isinstance(primary_binding, Mapping)
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="Graph_PROVIDER_PROJECTION_CONTEXT_INCOMPLETE",
                )

            draft = dict(primary_binding)
            conceptual_primitive_ids = draft.pop("conceptual_primitive_ids", None)
            conceptual_operator_id = draft.pop("conceptual_operator_id", None)
            conceptual_target_primitive_id = draft.pop("conceptual_target_primitive_id", None)
            conceptual_replacement_primitive_id = draft.pop("conceptual_replacement_primitive_id", None)
            conceptual_custom_components = draft.pop("conceptual_custom_components", None)
            if not isinstance(conceptual_primitive_ids, list) or not isinstance(
                conceptual_custom_components, list
            ) or not isinstance(conceptual_operator_id, str):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="Graph_CONCEPTUAL_PRIMITIVE_BINDING_MISSING",
                )
            if (
                draft.get("idea_mode") == "DIAGNOSIS_DRIVEN"
                and draft.get("observed_failure_mode") in (None, "NOT_OBSERVED")
            ):
                draft["idea_mode"] = "FRONTIER_HYPOTHESIS"
                draft["observed_failure_mode"] = None
            execution_contract = draft.get("execution_contract")
            if isinstance(execution_contract, Mapping) and "config_json" in execution_contract:
                try:
                    config = json.loads(str(execution_contract["config_json"]))
                except (TypeError, ValueError) as error:
                    return ConfirmationResolutionV1(
                        ConfirmationResolutionKindV1.UNSUPPORTED,
                        reason=f"Graph_EXECUTION_CONTRACT_INVALID:{type(error).__name__}",
                    )
                if not isinstance(config, Mapping):
                    return ConfirmationResolutionV1(
                        ConfirmationResolutionKindV1.UNSUPPORTED,
                        reason="Graph_EXECUTION_CONTRACT_CONFIG_INVALID",
                    )
                draft["execution_contract"] = {
                    "capability_family": execution_contract.get("capability_family"),
                    "model": execution_contract.get("model"),
                    "base_model_config": execution_contract.get("base_model_config"),
                    "config": dict(config),
                }
            try:
                spec, facts, source_proposal, program, companion = _project_result(
                    producer_role,
                    draft,
                    bindings=bindings,
                    frozen_profile_ref=frozen_profile_ref,
                )
                from dataclasses import replace
                # Generic RecBole projection injects a candidate trainer; graph owns no trainer hook.
                spec = replace(spec, execution_contract={**spec.execution_contract, "config": {}})
            except Exception as error:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason=(
                        f"Graph_PROVIDER_PROJECTION_FAILED:{type(error).__name__}:"
                        f"{error}"
                    ),
                )
            try:
                conceptual_binding = self.bind_conceptual_proposal(
                    proposal_ref=spec.spec_id,
                    proposal_digest=spec.digest,
                    primitive_ids=conceptual_primitive_ids,
                    operator_id=conceptual_operator_id,
                    target_primitive_id=conceptual_target_primitive_id,
                    replacement_primitive_id=conceptual_replacement_primitive_id,
                    custom_components=tuple(conceptual_custom_components),
                )
            except ValueError as error:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason=f"Graph_CONCEPTUAL_BINDING_INVALID:{error}",
                )
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
                    "source_mechanism_program_digest": (
                        sha256_digest(program) if program is not None else None
                    ),
                    "search_space_id": self.conceptual_space.search_space_id,
                    "search_space_digest": self.conceptual_space.digest,
                    "conceptual_binding": conceptual_binding.as_dict(),
                    "status": "PRODUCED",
                },
            )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={"producer_outcome": outcome},
            )

        if context.get("phase") == "PACKAGE_QUALIFIED_INNOVATION":
            prepared = context.get("prepared_innovation")
            outcome = prepared.get("producer_outcome") if isinstance(prepared, Mapping) else None
            utility = context.get("utility_features")
            evidence = context.get("feature_evidence")
            if (
                not isinstance(prepared, Mapping)
                or outcome is None
                or not isinstance(utility, Mapping)
                or not isinstance(evidence, Mapping)
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="Graph_QUALIFIED_INNOVATION_CONTEXT_INCOMPLETE",
                )
            capability = context.get("capability")
            materialized = context.get("materialized")
            qualification = context.get("qualification")
            if capability is None or materialized is None or qualification is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="Graph_QUALIFIED_INNOVATION_CONTEXT_INCOMPLETE",
                )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "search_candidate": self._native_candidate(
                        prepared,
                        outcome,
                        utility,
                        evidence,
                        capability=capability,
                        materialized=materialized,
                        qualification=qualification,
                    )
                },
            )

        return ConfirmationResolutionV1(
            ConfirmationResolutionKindV1.NEEDS_PROPOSAL,
            reason=f"{kind} requires a fresh graph candidate.py implementation",
        )


    def validate_execution_binding(self, binding: Any) -> None:
        from recclaw_core.experiments.helix_abc_v1.search_adapter import (
            SearchCandidateBindingV1,
        )

        if binding.adapter_id != self.adapter_id:
            raise ValueError("execution binding targets a different adapter")
        native = binding.native_binding
        if not isinstance(native, SearchCandidateBindingV1):
            raise ValueError("Graph execution requires SearchCandidateBindingV1")
        if (
            binding.binding_digest != native.digest
            or binding.candidate_id != native.proposal.candidate_id
            or binding.semantic_identity_digest
            != native.mechanism_semantics_digest
        ):
            raise ValueError("Graph execution binding identity drift")


    def effective_identity(self, binding: Any) -> Mapping[str, Any]:
        from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest

        self.validate_execution_binding(binding)
        return {
            "effective_experiment_digest": sha256_digest(
                {
                    "adapter": self.adapter_id,
                    "candidate": binding.semantic_identity_digest,
                    "task": self.task_id,
                }
            ),
            "effective_family_digest": self.conceptual_space.family_digest,
        }


    def execution_recipe(self, binding: Any) -> Mapping[str, Any]:
        from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value
        from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
            validate_execution_recipe,
        )

        self.validate_execution_binding(binding)
        context = binding.execution_context
        profile = context["profile"]
        entry = profile.entry(binding.native_binding.capability_ref)
        recipe = dict(context["qualified_execution"])
        recipe["config"] = {**dict(recipe.get("config", {})), "dataset": self.facts["dataset"]}
        recipe.update(
            {
                "capability_ref": entry.capability_ref,
                "capability_digest": entry.capability_digest,
                "profile_ref": profile.profile_ref,
                "profile_digest": profile.profile_digest,
                "entrypoint": binding.native_binding.executable_entrypoint,
                "mechanism_id": binding.native_binding.proposal.mechanism_id,
                "mechanism_semantics_digest": binding.native_binding.mechanism_semantics_digest,
                "dataset": self.facts["dataset"],
                "split": context["split"],
                "evaluator": context["evaluator"],
                "execution_role": "CANDIDATE",
            }
        )
        validate_execution_recipe(recipe)
        return canonical_value(recipe)


