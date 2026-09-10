"""Native compact-MultVAE search-space binding for E1 Research Line."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    OpenSpecProjectionError,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.model_configuration import (
    MODEL_CONFIG_MAPPING_REQUIREMENT,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    QualifiedSearchCandidateProtocolV1,
    SearchCandidateBindingV1,
    SearchProfileEntryOriginV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
)
from recclaw_core.research_line.interfaces import ProducerOutcome, ResearchContext

from .e1_native_bridge import (
    E1_ALLOWED_FILES,
    E1_CAPABILITY_FAMILY,
    E1_ENTRYPOINT,
    E1_MODEL,
    E1_PARENT_ID,
    E1_PROTOCOL_DIGEST,
    E1_PROTOCOL_REF,
    E1_TRAINING_CONFIG,
    parent_package_identity,
)
from .search_space_adapter import (
    ConfirmationResolutionKindV1,
    ConfirmationResolutionV1,
    SearchSpaceExecutionBindingV1,
)


E1_ADAPTER_ID = "recclaw.search-space-adapter.e1-multvae.v1"
E1_PROFILE_ID = E1_ADAPTER_ID
E1_MECHANISM_SPACE_ID = "E1_MULTVAE_NATIVE_V1"
E1_MECHANISM_LANGUAGE_ID = "recclaw.e1.multvae-native-language.v1"
E1_PARENT_CAPABILITY_REF = "recclaw.e1.parent-capability.v1"
E1_COMPATIBILITY_REQUIREMENTS = (
    "general collaborative filtering",
    "offline top-n evaluation",
    "train-only fitting",
    "user-wise implicit-feedback input",
)
E1_PROTOCOL_REQUIREMENTS = (
    "frozen dataset and split",
    "full-sort NDCG@10",
    *E1_COMPATIBILITY_REQUIREMENTS,
)
E1_IMPLEMENTATION_REQUIREMENTS = (
    MODEL_CONFIG_MAPPING_REQUIREMENT,
    "return the complete three-file recclaw_ext package",
    "define recclaw_ext.models.e1_multvae:FreshCandidateModel",
    "preserve the RecBole GeneralRecommender user-wise interaction ABI",
    "make the declared mechanism live in calculate_loss and/or full_sort_predict",
    "use no candidate-owned trainer, evaluator, data reader, or heldout input",
)

_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
E1_NATIVE_LANGUAGE_PATH = _RESOURCE_ROOT / "e1_multvae_native_language_v1.json"
E1_PROPOSAL_PROMPT_PATH = _RESOURCE_ROOT / "e1_multvae_proposal_prompt_v1.txt"
E1_DIRECTOR_PROMPT_PATH = _RESOURCE_ROOT / "e1_multvae_director_prompt_v1.txt"
E1_IMPLEMENTER_PROMPT_PATH = _RESOURCE_ROOT / "e1_multvae_implementer_prompt_v1.txt"
E1_PROPOSAL_SCHEMA_PATH = (
    _RESOURCE_ROOT / "e1_multvae_proposal_response_v1.schema.json"
)
E1_IMPLEMENTATION_SCHEMA_PATH = (
    _RESOURCE_ROOT / "e1_multvae_implementation_response_v1.schema.json"
)
_PARENT_ROOT = Path(__file__).resolve().parents[3] / "e1_native_parent"


@lru_cache(maxsize=1)
def e1_native_mechanism_language() -> Mapping[str, Any]:
    document = json.loads(E1_NATIVE_LANGUAGE_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != E1_MECHANISM_LANGUAGE_ID
        or document.get("language_id") != E1_MECHANISM_LANGUAGE_ID
        or document.get("research_profile_id") != E1_PROFILE_ID
        or not isinstance(document.get("causal_zones"), list)
        or not document["causal_zones"]
    ):
        raise ValueError("E1 native mechanism language identity is invalid")
    normalized = canonical_value(dict(document))
    return canonical_value(
        {**dict(normalized), "projection_digest": sha256_digest(normalized)}
    )


E1_PARENT_MECHANISM_PROGRAM = canonical_value(
    {
        "schema": "recclaw.e1.multvae-mechanism-program.v1",
        "profile_id": E1_PROFILE_ID,
        "protocol_ref": E1_PROTOCOL_REF,
        "protocol_digest": E1_PROTOCOL_DIGEST,
        "program_payload": {
            "construction_mode": "FROZEN_PARENT",
            "parent_refs": (),
            "mechanism_axis_footprint": (
                "INPUT_REPRESENTATION",
                "POSTERIOR_PRIOR",
                "BOTTLENECK_GEOMETRY",
                "DECODER_SCORING",
                "OBJECTIVE_REGULARIZATION",
                "OPTIMIZATION_DYNAMICS",
                "INFERENCE_CALIBRATION",
            ),
            "mechanism": (
                "pilot-known RH parent: input dropout 0.125 before L2, "
                "compact multinomial VAE with a 384-wide "
                "encoder/decoder, 128-dimensional Gaussian latent, and "
                "linearly annealed KL"
            ),
            "execution_contract": {
                "capability_family": E1_CAPABILITY_FAMILY,
                "model": E1_MODEL,
                "base_model_config": E1_MODEL,
                "config": E1_TRAINING_CONFIG,
            },
        },
    }
)
E1_PARENT_PROGRAM_DIGEST = sha256_digest(E1_PARENT_MECHANISM_PROGRAM)


def e1_profile_manifest() -> Mapping[str, Any]:
    language = e1_native_mechanism_language()
    return canonical_value(
        {
            "schema": "recclaw.e1.multvae-profile.v1",
            "adapter_id": E1_ADAPTER_ID,
            "mechanism_space": E1_MECHANISM_SPACE_ID,
            "mechanism_language_id": E1_MECHANISM_LANGUAGE_ID,
            "mechanism_language_digest": language["projection_digest"],
            "parent_id": E1_PARENT_ID,
            "parent_program_digest": E1_PARENT_PROGRAM_DIGEST,
            "parent_package": parent_package_identity(),
            "protocol_ref": E1_PROTOCOL_REF,
            "protocol_digest": E1_PROTOCOL_DIGEST,
            "entrypoint": E1_ENTRYPOINT,
            "model": E1_MODEL,
            "training_config": E1_TRAINING_CONFIG,
        }
    )


def e1_frozen_profile_ref() -> Mapping[str, Any]:
    return canonical_value(
        {
            "profile_id": E1_PROFILE_ID,
            "profile_digest": sha256_digest(e1_profile_manifest()),
            "profile_kind": "OFFLINE_TOPN",
        }
    )


def e1_baseline_context(
    *,
    ndcg_at_10: float,
    recall_at_10: float,
    result_ref: str,
    result_sha256: str,
    seed: int = 54201,
) -> Mapping[str, Any]:
    """Build the one-parent context from an already calibrated dev result."""

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (ndcg_at_10, recall_at_10)
    ):
        raise ValueError("E1 baseline metrics must be finite numbers")
    if not isinstance(result_ref, str) or not result_ref.strip():
        raise ValueError("E1 baseline result_ref must be non-empty")
    result_digest = validate_sha256(
        result_sha256,
        field_name="E1 baseline result_sha256",
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError("E1 baseline seed must be positive")
    package = parent_package_identity()
    files = []
    for relative_path in E1_ALLOWED_FILES:
        payload = (_PARENT_ROOT / relative_path).read_bytes()
        files.append(
            canonical_value(
                {
                    "path": relative_path,
                    "content": payload.decode("utf-8"),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        )
    return canonical_value(
        {
            "schema": "recclaw.research-baseline-context.v2",
            "research_profile_id": E1_PROFILE_ID,
            "mechanism_space": E1_MECHANISM_SPACE_ID,
            "mechanism_language_id": E1_MECHANISM_LANGUAGE_ID,
            "mechanism_language": e1_native_mechanism_language(),
            "metric": "NDCG@10",
            "split": "data/dev",
            "parent_anchor": {
                "name": "Compact MultVAE",
                "base_model_config": E1_MODEL,
                "mechanism_summary": E1_PARENT_MECHANISM_PROGRAM[
                    "program_payload"
                ]["mechanism"],
                "binding": {
                    "candidate_id": E1_PARENT_ID,
                    "program_digest": E1_PARENT_PROGRAM_DIGEST,
                },
                "mechanism_program": E1_PARENT_MECHANISM_PROGRAM,
                "paired_metric": {
                    "name": "NDCG@10",
                    "value": float(ndcg_at_10),
                    "recall_at_10": float(recall_at_10),
                    "seed": seed,
                    "source_ref": result_ref.strip(),
                    "source_sha256": result_digest,
                    "partition_role": "DEVELOPMENT_VALIDATION",
                },
                "source_bundle": {
                    "candidate_id": E1_PARENT_ID,
                    "capability_ref": E1_PARENT_CAPABILITY_REF,
                    "program_digest": E1_PARENT_PROGRAM_DIGEST,
                    "source_tree_digest": package["source_tree_digest"],
                    "instruction": "CLONE_EXACT_PARENT_AND_LOCAL_PATCH",
                    "files": files,
                },
            },
            "search_objective": (
                "Discover a faithful mechanism that improves development "
                "NDCG@10 over the exact compact MultVAE parent."
            ),
            "decision_semantics": {
                "development_feedback_only": True,
                "frozen_root_remains_comparator": True,
                "construction_parent_selection": "EXPLICIT_AVAILABLE_PARENT",
                "outer_heldout_access": "FORBIDDEN",
            },
        }
    )


def _expected_parent_binding(
    research_context: ResearchContext,
    explicit_lineage: Mapping[str, Any] | None = None,
    *,
    selected_parent: str | None = None,
    construction_parent_options: tuple[Mapping[str, Any], ...] | None = None,
) -> Mapping[str, str]:
    if construction_parent_options is not None:
        selected = tuple(option for option in construction_parent_options
                         if option.get("candidate_id") == selected_parent)
        if len(selected) != 1:
            raise OpenSpecProjectionError("E1 selected construction parent is unavailable or ambiguous")
        return canonical_value({
            "candidate_id": selected[0]["candidate_id"],
            "program_digest": validate_sha256(
                str(selected[0].get("program_digest")),
                field_name="E1 selected parent program_digest"),
        })
    sources: list[Any] = []
    if isinstance(explicit_lineage, Mapping):
        sources.append(explicit_lineage)
    sources.append(research_context.frontier.get("lineage_parent_binding"))
    baseline = research_context.knowledge_base.get("baseline_context")
    if isinstance(baseline, Mapping):
        parent = baseline.get("parent_anchor")
        if isinstance(parent, Mapping):
            sources.append(parent.get("binding"))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        candidate_id = source.get("candidate_id")
        program_digest = source.get("program_digest")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            continue
        try:
            digest = validate_sha256(
                str(program_digest),
                field_name="E1 parent program_digest",
            )
        except ValueError:
            continue
        return canonical_value(
            {
                "candidate_id": candidate_id.strip(),
                "program_digest": digest,
            }
        )
    raise OpenSpecProjectionError("E1 exact construction parent is unavailable")


def _validate_provider_draft(
    draft: Mapping[str, Any],
    *,
    producer_role: str,
    expected_parent: Mapping[str, str],
    bindings: Mapping[str, Any],
) -> None:
    if draft.get("producer_role") != producer_role:
        raise OpenSpecProjectionError("E1 proposal producer_role drift")
    if draft.get("closest_parent") != expected_parent["candidate_id"]:
        raise OpenSpecProjectionError(
            "E1 proposal closest_parent differs from active construction parent"
        )
    if draft.get("current_profile_expressibility_claim") != (
        CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE.value
    ):
        raise OpenSpecProjectionError("E1 proposal must require a fresh capability")
    if set(draft.get("compatibility_requirements", ())) != set(
        E1_COMPATIBILITY_REQUIREMENTS
    ):
        raise OpenSpecProjectionError("E1 compatibility requirements drift")
    if set(bindings.get("compatibility_requirements", ())) != set(
        E1_COMPATIBILITY_REQUIREMENTS
    ):
        raise OpenSpecProjectionError("E1 Provider bindings drift")
    contract = draft.get("execution_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "capability_family",
        "model",
        "base_model_config",
        "config",
    }:
        raise OpenSpecProjectionError("E1 execution contract is incomplete")
    if (
        contract.get("capability_family") != E1_CAPABILITY_FAMILY
        or contract.get("model") != E1_MODEL
        or contract.get("base_model_config") != E1_MODEL
        or contract.get("config") != {}
    ):
        raise OpenSpecProjectionError("E1 execution contract changes fixed substrate")


def _coerce_spec(
    result: Mapping[str, Any],
    *,
    producer_role: str,
    research_context: ResearchContext,
    bindings: Mapping[str, Any],
    explicit_lineage: Mapping[str, Any] | None,
    construction_parent_options: tuple[Mapping[str, Any], ...] | None = None,
) -> tuple[OpenResearchSpecV1, Mapping[str, Any]]:
    draft = dict(result)
    # A display-name alias always means the root, never the current best.
    baseline = research_context.knowledge_base.get("baseline_context", {})
    anchor = baseline.get("parent_anchor", {})
    if draft.get("closest_parent") == anchor.get("name"):
        draft["closest_parent"] = anchor["binding"]["candidate_id"]
    expected_parent = _expected_parent_binding(
        research_context, explicit_lineage, selected_parent=draft.get("closest_parent"),
        construction_parent_options=construction_parent_options)
    _validate_provider_draft(
        draft,
        producer_role=producer_role,
        expected_parent=expected_parent,
        bindings=bindings,
    )
    draft["implementation_requirements"] = tuple(dict.fromkeys((
        *draft.get("implementation_requirements", ()),
        *E1_IMPLEMENTATION_REQUIREMENTS,
    )))
    draft["compatibility_requirements"] = E1_COMPATIBILITY_REQUIREMENTS
    draft["execution_contract"] = {
        "capability_family": E1_CAPABILITY_FAMILY,
        "model": E1_MODEL,
        "base_model_config": E1_MODEL,
        "config": E1_TRAINING_CONFIG,
    }
    spec, facts = project_open_producer_draft(
        draft,
        bindings=bindings,
        strict_resolution_contract=True,
    )
    if (
        spec.protocol_ref != research_context.protocol_ref
        or spec.protocol_digest != research_context.protocol_digest
    ):
        raise OpenSpecProjectionError("E1 proposal protocol identity drift")
    return spec, facts


def _mechanism_program(
    spec: OpenResearchSpecV1,
    facts: Mapping[str, Any],
    parent_binding: Mapping[str, str],
) -> Mapping[str, Any]:
    dimensions = tuple(str(item) for item in facts["high_change_dimensions"])
    if spec.realization_mode is None:
        raise ValueError("E1 proposal lacks realization_mode")
    return canonical_value(
        {
            "schema": "recclaw.e1.multvae-mechanism-program.v1",
            "profile_id": E1_PROFILE_ID,
            "protocol_ref": E1_PROTOCOL_REF,
            "protocol_digest": E1_PROTOCOL_DIGEST,
            "producer_role": spec.producer_role,
            "program_payload": {
                "construction_mode": spec.realization_mode.value,
                "parent_refs": (parent_binding,),
                "mechanism_axis_footprint": dimensions,
                "hypothesis": spec.hypothesis,
                "mechanism_change": spec.mechanism_change,
                "minimal_testable_wedge": spec.minimal_testable_wedge,
                "causal_chain": spec.causal_chain,
                "competing_explanation": spec.competing_explanation,
                "discriminative_predictions": spec.discriminative_predictions,
                "mechanism_off_definition": spec.mechanism_off_definition,
                "falsifier": spec.falsifier,
                "resource_hypothesis": spec.resource_hypothesis,
                "execution_contract": spec.execution_contract,
            },
        }
    )


def _candidate_identity(
    program: Mapping[str, Any],
) -> tuple[str, str, str]:
    semantic_digest = sha256_digest(
        {
            "mechanism_space": E1_MECHANISM_SPACE_ID,
            "protocol_digest": E1_PROTOCOL_DIGEST,
            "program": program,
        }
    )
    candidate_id = "e1-candidate-" + semantic_digest[:24]
    return candidate_id, f"e1-multvae-mechanism:{candidate_id}", semantic_digest


def _effective_identity(program: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = program.get("program_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("E1 mechanism program lacks program_payload")
    mechanism_change = payload.get("mechanism_change")
    if not isinstance(mechanism_change, str) or not mechanism_change.strip():
        raise ValueError("E1 mechanism program lacks mechanism_change")
    axes = tuple(str(item) for item in payload.get("mechanism_axis_footprint", ()))
    return canonical_value(
        {
            "effective_experiment_digest": sha256_digest(
                {
                    "mechanism_space": E1_MECHANISM_SPACE_ID,
                    "protocol_digest": E1_PROTOCOL_DIGEST,
                    "program": program,
                }
            ),
            "effective_family_digest": sha256_digest(
                {
                    "mechanism_space": E1_MECHANISM_SPACE_ID,
                    "parent_refs": payload.get("parent_refs"),
                    "axes": axes,
                    "mechanism_change": mechanism_change.strip(),
                }
            ),
            "primitive_ids": axes,
        }
    )


@dataclass(frozen=True, slots=True)
class E1MultVAEQualifiedCandidateV1:
    spec: OpenResearchSpecV1
    candidate_id: str
    producer_role: str
    mechanism_id: str
    mechanism_axis: str
    mechanism_program: Mapping[str, Any]
    parent_candidate_id: str
    utility_features: SearchUtilityFeaturesV1
    feature_evidence: RouterFeatureEvidenceV1
    semantic_identity_ref: str
    semantic_identity_digest: str
    mechanism_program_digest: str
    candidate_package_ref: str
    candidate_package_digest: str
    candidate_root_ref: str
    candidate_root_digest: str
    source_tree_digest: str
    execution_contract: Mapping[str, Any]
    capability_ref: str
    capability_digest: str
    executable_entrypoint: str
    qualification_receipt_ref: str
    qualification_receipt_digest: str

    schema = "recclaw.e1.multvae-qualified-candidate.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, OpenResearchSpecV1):
            raise TypeError("E1 candidate requires OpenResearchSpecV1")
        program = canonical_value(dict(self.mechanism_program))
        if self.mechanism_program_digest != sha256_digest(program):
            raise ValueError("E1 candidate mechanism program digest drift")
        contract = canonical_value(dict(self.execution_contract))
        if contract != canonical_value(dict(self.spec.execution_contract or {})):
            raise ValueError("E1 candidate execution contract drift")
        if (
            self.spec.protocol_ref != E1_PROTOCOL_REF
            or self.spec.protocol_digest != E1_PROTOCOL_DIGEST
            or self.executable_entrypoint != E1_ENTRYPOINT
        ):
            raise ValueError("E1 candidate fixed protocol or entrypoint drift")
        for field_name in (
            "semantic_identity_digest",
            "candidate_package_digest",
            "candidate_root_digest",
            "source_tree_digest",
            "capability_digest",
            "qualification_receipt_digest",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        object.__setattr__(self, "mechanism_program", program)
        object.__setattr__(self, "execution_contract", contract)

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            method = getattr(value, "to_dict", None)
            return method() if callable(method) else value

        return canonical_value(
            {
                "schema": self.schema,
                **{
                    field.name: encode(getattr(self, field.name))
                    for field in fields(self)
                },
            }
        )


class E1MultVAESearchSpaceAdapterV1:
    """Bind E1 without changing the native ResearchCampaign policy."""

    def __init__(self, *, worker_boundary=None, resource_probe_runner=None,
                 pilot_context: str = ""):
        self.qualification_executor = worker_boundary.qualify if worker_boundary else None
        self.process_launcher = worker_boundary.training_popen if worker_boundary else None
        self.resource_probe_runner = resource_probe_runner
        self.proposal_context_appendix = pilot_context

    adapter_id = E1_ADAPTER_ID
    supported_frozen_profile_kinds = ("OFFLINE_TOPN",)
    implementation_requirements = E1_IMPLEMENTATION_REQUIREMENTS
    compatibility_requirements = E1_COMPATIBILITY_REQUIREMENTS
    protocol_requirements = E1_PROTOCOL_REQUIREMENTS
    implementation_allowed_files = E1_ALLOWED_FILES
    proposal_prompt_source = E1_PROPOSAL_PROMPT_PATH
    proposal_schema_source = E1_PROPOSAL_SCHEMA_PATH
    implementation_prompt_source = E1_IMPLEMENTER_PROMPT_PATH
    implementation_schema_source = E1_IMPLEMENTATION_SCHEMA_PATH

    def select_innovation_outcome(
        self,
        candidates: tuple[tuple[ProducerOutcome, Any], ...],
    ) -> str | None:
        """Return the sole portfolio decision allowed to execute in E1."""

        frontier = tuple(
            outcome
            for outcome, _resolution in candidates
            if outcome.producer_role == "frontier_architect"
        )
        return frontier[0].digest if len(frontier) == 1 else None

    def resolve_confirmation(
        self,
        kind: str,
        primary_binding: Any,
        context: Mapping[str, Any],
    ) -> ConfirmationResolutionV1:
        phase = context.get("phase")
        if phase == "PROJECT_PROVIDER_PROPOSAL":
            if not isinstance(primary_binding, Mapping):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_PROVIDER_RESULT_IS_NOT_OPEN_DRAFT",
                )
            try:
                research_context = context["research_context"]
                if not isinstance(research_context, ResearchContext):
                    raise TypeError("research_context is invalid")
                spec, facts = _coerce_spec(
                    primary_binding,
                    producer_role=str(context["producer_role"]),
                    research_context=research_context,
                    bindings=context["producer_bindings"],
                    explicit_lineage=context.get(
                        "lineage_parent_mechanism_program"
                    ),
                    construction_parent_options=context.get("construction_parent_options"),
                )
            except (KeyError, TypeError, ValueError) as error:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason=(
                        "E1_OPEN_SPEC_PROJECTION_FAILED:"
                        f"{type(error).__name__}:{error}"
                    ),
                )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "producer_outcome": ProducerOutcome(
                        producer_role=spec.producer_role,
                        context_ref=spec.context_ref,
                        context_digest=spec.context_digest,
                        spec=spec,
                        resolution_facts=facts,
                        provenance={
                            "producer_role": spec.producer_role,
                            "status": "PRODUCED",
                            "search_space_profile": E1_PROFILE_ID,
                            "lineage_parent_binding": _expected_parent_binding(
                                research_context,
                                context.get("lineage_parent_mechanism_program"),
                                selected_parent=spec.closest_parent,
                                construction_parent_options=context.get("construction_parent_options")),
                        },
                    )
                },
            )

        if phase in {"IDENTIFY_INNOVATION", "PREPARE_INNOVATION"}:
            outcome = primary_binding
            research_context = context.get("research_context")
            if (
                not isinstance(outcome, ProducerOutcome)
                or outcome.spec is None
                or not isinstance(research_context, ResearchContext)
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_INNOVATION_OUTCOME_INCOMPLETE",
                )
            try:
                parent = _expected_parent_binding(
                    research_context, outcome.provenance.get("lineage_parent_binding"),
                    selected_parent=outcome.spec.closest_parent,
                    construction_parent_options=context.get("construction_parent_options"))
                if outcome.spec.closest_parent != parent["candidate_id"]:
                    raise ValueError("E1 prepared parent identity drift")
                recorded_parent = outcome.provenance.get("lineage_parent_binding")
                if isinstance(recorded_parent, Mapping) and recorded_parent != parent:
                    raise ValueError("E1 selected parent binding changed after projection")
                program = _mechanism_program(
                    outcome.spec,
                    outcome.resolution_facts,
                    parent,
                )
                candidate_id, semantic_ref, semantic_digest = _candidate_identity(
                    program
                )
            except (TypeError, ValueError) as error:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason=f"E1_INNOVATION_PREPARATION_FAILED:{error}",
                )
            dimensions = tuple(outcome.resolution_facts["high_change_dimensions"])
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "producer_outcome": outcome,
                    "compiler_binding": None,
                    "mechanism_program": program,
                    "compiled_implementation": None,
                    "semantic_identity_ref": semantic_ref,
                    "semantic_identity_digest": semantic_digest,
                    "effective_identity": _effective_identity(program),
                    "candidate_id": candidate_id,
                    "mechanism_id": candidate_id,
                    "parent_available": True,
                    "mechanism_axis": dimensions[0],
                },
            )

        if phase == "PACKAGE_QUALIFIED_INNOVATION":
            prepared = context.get("prepared_innovation")
            materialized = context.get("materialized")
            capability = context.get("capability")
            utility = context.get("utility_features")
            evidence = context.get("feature_evidence")
            if (
                not isinstance(prepared, Mapping)
                or materialized is None
                or capability is None
                or not isinstance(utility, Mapping)
                or not isinstance(evidence, Mapping)
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_QUALIFIED_INNOVATION_CONTEXT_INCOMPLETE",
                )
            outcome = prepared.get("producer_outcome")
            if not isinstance(outcome, ProducerOutcome) or outcome.spec is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_QUALIFIED_OUTCOME_INVALID",
                )
            package = materialized.package
            if (
                package.package_id != capability.candidate_package_ref
                or package.digest != capability.candidate_package_digest
                or package.executable_entrypoint != E1_ENTRYPOINT
                or tuple(package.allowed_files) != tuple(sorted(E1_ALLOWED_FILES))
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_QUALIFIED_PACKAGE_IDENTITY_DRIFT",
                )
            written_files = materialized.implementation_receipt.get(
                "written_files"
            )
            if not isinstance(written_files, (tuple, list)):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_QUALIFIED_PACKAGE_MANIFEST_MISSING",
                )
            written_sha256 = {
                str(row.get("path")): str(row.get("sha256"))
                for row in written_files
                if isinstance(row, Mapping)
            }
            parent_sha256 = parent_package_identity()["file_sha256"]
            if any(
                written_sha256.get(path) != parent_sha256[path]
                for path in E1_ALLOWED_FILES[:2]
            ):
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_FIXED_PACKAGE_INIT_BYTES_DRIFT",
                )
            program = prepared["mechanism_program"]
            parent_refs = program["program_payload"]["parent_refs"]
            candidate = E1MultVAEQualifiedCandidateV1(
                spec=outcome.spec,
                candidate_id=str(prepared["candidate_id"]),
                producer_role=outcome.producer_role,
                mechanism_id=str(prepared["mechanism_id"]),
                mechanism_axis=str(
                    prepared.get("mechanism_axis")
                    or context.get("mechanism_axis")
                ),
                mechanism_program=program,
                parent_candidate_id=str(parent_refs[0]["candidate_id"]),
                utility_features=SearchUtilityFeaturesV1(**dict(utility)),
                feature_evidence=RouterFeatureEvidenceV1(**dict(evidence)),
                semantic_identity_ref=str(prepared["semantic_identity_ref"]),
                semantic_identity_digest=str(
                    prepared["semantic_identity_digest"]
                ),
                mechanism_program_digest=sha256_digest(program),
                candidate_package_ref=capability.candidate_package_ref,
                candidate_package_digest=capability.candidate_package_digest,
                candidate_root_ref=package.candidate_root_ref,
                candidate_root_digest=package.candidate_root_digest,
                source_tree_digest=package.source_tree_digest,
                execution_contract=outcome.spec.execution_contract,
                capability_ref=capability.capability_id,
                capability_digest=capability.digest,
                executable_entrypoint=package.executable_entrypoint,
                qualification_receipt_ref=capability.qualification_receipt_ref,
                qualification_receipt_digest=(
                    capability.qualification_receipt_digest
                ),
            )
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={"search_candidate": candidate},
            )

        if phase == "MATERIALIZE_DISCOVERY":
            outcome = primary_binding
            resolution = context.get("resolution")
            profile = context.get("execution_profile")
            proposal = getattr(outcome, "source_proposal", None)
            capability_ref = getattr(
                resolution,
                "resolved_current_capability_ref",
                None,
            )
            if proposal is None or profile is None or capability_ref is None:
                return ConfirmationResolutionV1(
                    ConfirmationResolutionKindV1.UNSUPPORTED,
                    reason="E1_DISCOVERY_BINDING_CONTEXT_INCOMPLETE",
                )
            entry = profile.entry(capability_ref)
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.EXACT_BINDING,
                binding={
                    "native_binding": SearchCandidateBindingV1(
                        proposal=proposal,
                        capability_ref=entry.capability_ref,
                        capability_digest=entry.capability_digest,
                        executable_entrypoint=entry.executable_entrypoint,
                        entry_origin=entry.origin,
                        mechanism_semantics_digest=entry.semantic_identity_digest,
                    )
                },
            )

        if phase in {
            "DECLARE_FOLLOWUP",
            "MATCH_PROPOSAL",
            "MATERIALIZE_VERIFICATION",
            "GUARD_CONTROL",
        }:
            return ConfirmationResolutionV1(
                ConfirmationResolutionKindV1.NEEDS_PROPOSAL,
                reason="E1_CONTROL_REQUIRES_EXPLICIT_QUALIFIED_CANDIDATE",
            )
        return ConfirmationResolutionV1(
            ConfirmationResolutionKindV1.UNSUPPORTED,
            reason="E1_CONFIRMATION_PHASE_UNSUPPORTED",
        )

    def matches_exact_parent_bundle(
        self,
        outcome: ProducerOutcome,
        exact_parent_bundle: Mapping[str, Any],
    ) -> bool:
        parent = (outcome.provenance.get("lineage_parent_binding")
                  if isinstance(outcome, ProducerOutcome) else None)
        return bool(
            isinstance(outcome, ProducerOutcome)
            and outcome.spec is not None
            and outcome.spec.closest_parent
            == exact_parent_bundle.get("candidate_id")
            and (not isinstance(parent, Mapping) or parent == {
                "candidate_id": exact_parent_bundle.get("candidate_id"),
                "program_digest": exact_parent_bundle.get("program_digest"),
            })
            and tuple(
                sorted(
                    str(row.get("path"))
                    for row in exact_parent_bundle.get("files", ())
                    if isinstance(row, Mapping)
                )
            )
            == tuple(sorted(E1_ALLOWED_FILES))
        )

    def validate_lineage_candidate(self, candidate: Any) -> None:
        if not isinstance(candidate, E1MultVAEQualifiedCandidateV1):
            raise TypeError("lineage candidate is not an E1 candidate")
        if candidate.mechanism_program_digest != sha256_digest(
            candidate.mechanism_program
        ):
            raise ValueError("E1 lineage program digest drift")
        parent_refs = candidate.mechanism_program["program_payload"][
            "parent_refs"
        ]
        if (
            not isinstance(parent_refs, (tuple, list))
            or len(parent_refs) != 1
            or parent_refs[0].get("candidate_id")
            != candidate.parent_candidate_id
        ):
            raise ValueError("E1 lineage parent binding drift")

    def qualification_unit_check(
        self, *, base_model_config: str, execution_contract: Mapping[str, Any]
    ) -> None:
        # MechanicalRecBoleAdapter owns the user-wise API and smoke checks;
        # the BL pairwise behavioral closure does not apply to E1.
        return None

    def validate_execution_binding(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> None:
        if binding.adapter_id != self.adapter_id:
            raise ValueError("E1 adapter binding id mismatch")
        self.execution_recipe(binding)

    def effective_identity(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]:
        proposal = binding.native_binding.proposal
        program = getattr(proposal, "mechanism_program", None)
        if not isinstance(program, Mapping):
            raise ValueError("E1 binding lacks mechanism_program")
        return _effective_identity(program)

    def execution_recipe(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]:
        native = binding.native_binding
        proposal = native.proposal
        context = binding.execution_context
        qualified_execution = context.get("qualified_execution")
        if (
            native.entry_origin is not SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
            or not isinstance(proposal, QualifiedSearchCandidateProtocolV1)
            or not isinstance(qualified_execution, Mapping)
        ):
            raise ValueError("E1 execution requires one qualified native candidate")
        recipe = dict(qualified_execution)
        recipe.update(
            {
                "capability_family": E1_CAPABILITY_FAMILY,
                "capability_ref": native.capability_ref,
                "capability_digest": native.capability_digest,
                "profile_ref": context["profile"].profile_ref,
                "profile_digest": context["profile"].profile_digest,
                "entrypoint": E1_ENTRYPOINT,
                "model": E1_MODEL,
                "base_model_config": E1_MODEL,
                "config": E1_TRAINING_CONFIG,
                "mechanism_id": proposal.mechanism_id,
                "mechanism_semantics_digest": native.mechanism_semantics_digest,
                "dataset": COMMON_DATASET,
                "split": DEVELOPMENT_SPLIT,
                "evaluator": DEVELOPMENT_EVALUATOR,
                "evaluator_digest": sha256_digest(DEVELOPMENT_EVALUATOR),
                "execution_role": "CANDIDATE",
                "e1_protocol_ref": E1_PROTOCOL_REF,
                "e1_protocol_digest": E1_PROTOCOL_DIGEST,
            }
        )
        normalized = canonical_value(recipe)
        validate_execution_recipe(normalized)
        return normalized


def default_e1_multvae_adapter() -> E1MultVAESearchSpaceAdapterV1:
    return E1MultVAESearchSpaceAdapterV1()


__all__ = [
    "E1_ADAPTER_ID",
    "E1_COMPATIBILITY_REQUIREMENTS",
    "E1_IMPLEMENTATION_REQUIREMENTS",
    "E1_MECHANISM_LANGUAGE_ID",
    "E1_MECHANISM_SPACE_ID",
    "E1MultVAEQualifiedCandidateV1",
    "E1MultVAESearchSpaceAdapterV1",
    "default_e1_multvae_adapter",
    "e1_baseline_context",
    "e1_frozen_profile_ref",
    "e1_native_mechanism_language",
    "e1_profile_manifest",
]
