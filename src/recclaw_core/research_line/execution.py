"""Pure Search binding projections into common execution and evidence lanes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from recclaw_core.helix.scientific_attribution import (
    NOT_AVAILABLE,
    SearchUtilityEventV2,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    CampaignRuntimeError,
    execution_recipe_for_program,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    OpenSpecSearchCandidateV1,
    SearchCandidateBindingV1,
    SearchExecutableProfileV1,
    SearchProfileEntryOriginV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import CandidateProposalV4
from recclaw_core.experiments.helix_abc_v1.scientific_episode import (
    FrozenComparisonIdentityV1,
    ScientificEpisodeClosureV1,
    close_scientific_episode,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    EpisodeEvidenceClassV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)
from recclaw_core.research_line.interfaces import ProducerOutcome, ResearchContext


_QUALIFIED_EXECUTION_FIELDS = frozenset(
    {
        "capability_family",
        "model",
        "base_model_config",
        "config",
        "entrypoint_source_sha256",
        "candidate_package_ref",
        "candidate_package_digest",
        "candidate_root_ref",
        "candidate_root_digest",
        "candidate_source_tree_digest",
    }
)


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be normalized and non-empty")
    return value


def _require_ref_digest(
    value: Mapping[str, Any],
    *,
    ref_field: str,
    digest_field: str,
) -> tuple[str, str]:
    ref = _require_text(value.get(ref_field), field_name=ref_field)
    try:
        digest = validate_sha256(
            value.get(digest_field),
            field_name=digest_field,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    return ref, digest


def _assert_identity(
    recipe: Mapping[str, Any],
    *,
    field_name: str,
    expected: Any,
) -> None:
    if field_name in recipe and canonical_value(recipe[field_name]) != canonical_value(expected):
        raise ValueError(f"execution recipe {field_name} identity drift")


def _binding_entry(
    binding: SearchCandidateBindingV1,
    profile: SearchExecutableProfileV1,
) -> Any:
    if not isinstance(binding, SearchCandidateBindingV1):
        raise ValueError("binding must be SearchCandidateBindingV1")
    if not isinstance(profile, SearchExecutableProfileV1):
        raise ValueError("profile must be SearchExecutableProfileV1")
    try:
        entry = profile.entry(binding.capability_ref)
    except Exception as error:
        raise ValueError("binding capability is not in the supplied profile") from error
    if (
        entry.capability_digest != binding.capability_digest
        or entry.executable_entrypoint != binding.executable_entrypoint
        or entry.origin is not binding.entry_origin
        or entry.semantic_identity_digest != binding.mechanism_semantics_digest
    ):
        raise ValueError("Search binding/profile identity drift")
    return entry


def _overlay_common_binding_identity(
    recipe: Mapping[str, Any],
    *,
    binding: SearchCandidateBindingV1,
    profile: SearchExecutableProfileV1,
) -> dict[str, Any]:
    entry = _binding_entry(binding, profile)
    expected = {
        "capability_ref": entry.capability_ref,
        "capability_digest": entry.capability_digest,
        "profile_ref": profile.profile_ref,
        "profile_digest": profile.profile_digest,
        "entrypoint": binding.executable_entrypoint,
        "mechanism_id": binding.proposal.mechanism_id,
        "mechanism_semantics_digest": binding.mechanism_semantics_digest,
        "dataset": COMMON_DATASET,
        "split": COMMON_SPLIT,
        "evaluator": COMMON_EVALUATOR,
        "execution_role": "CANDIDATE",
    }
    for field_name, expected_value in expected.items():
        _assert_identity(
            recipe,
            field_name=field_name,
            expected=expected_value,
        )
    result = dict(recipe)
    result.update(expected)
    return canonical_value(result)


def execution_recipe_for_search_binding(
    binding: SearchCandidateBindingV1,
    *,
    profile: SearchExecutableProfileV1,
    qualified_execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one Search binding into an explicit common runner recipe."""

    entry = _binding_entry(binding, profile)
    if entry.origin is SearchProfileEntryOriginV1.FIXED_66:
        if qualified_execution is not None:
            raise ValueError("fixed Search bindings cannot carry qualified execution inputs")
        if not isinstance(binding.proposal, CandidateProposalV4):
            raise ValueError("fixed Search bindings require a legacy BL-ICF proposal")
        try:
            recipe = execution_recipe_for_program(
                deep_thaw(binding.proposal.mechanism_program),
            )
        except CampaignRuntimeError as error:
            raise ValueError("fixed Search binding is outside the executable catalog") from error
        if recipe.get("mechanism_id") != binding.proposal.mechanism_id:
            raise ValueError("fixed Search mechanism identity drift")
        if recipe.get("mechanism_semantics_digest") != binding.mechanism_semantics_digest:
            raise ValueError("fixed Search semantics identity drift")
        if recipe.get("entrypoint") != binding.executable_entrypoint:
            raise ValueError("fixed Search entrypoint identity drift")
        if recipe.get("base_mechanism_id") is None:
            raise ValueError("fixed Search recipe lacks base mechanism identity")
        projected = dict(recipe)
        projected["capability_family"] = recipe["base_mechanism_id"]
    else:
        if qualified_execution is None:
            raise ValueError(
                "QUALIFIED_REGISTRY binding requires explicit qualified execution inputs"
            )
        supplied = _require_mapping(
            qualified_execution,
            field_name="qualified_execution",
        )
        supplied_fields = set(supplied)
        missing = sorted(_QUALIFIED_EXECUTION_FIELDS - supplied_fields)
        extra = sorted(supplied_fields - _QUALIFIED_EXECUTION_FIELDS)
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if extra:
                detail.append("extra=" + ",".join(extra))
            raise ValueError(
                "qualified_execution must contain exactly the explicit execution inputs ("
                + "; ".join(detail)
                + ")"
            )
        projected = dict(supplied)

    projected = _overlay_common_binding_identity(
        projected,
        binding=binding,
        profile=profile,
    )
    try:
        validate_execution_recipe(projected)
    except Exception as error:
        raise ValueError(str(error)) from error
    return canonical_value(projected)


def _finite_metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _cost_projection(candidate_run: Mapping[str, Any]) -> dict[str, Any]:
    """Project bounded, measured execution-cost evidence.

    The lossless telemetry stays on the immutable physical candidate run.  A
    Search utility event is long-lived memory input, so copying every batch
    record here would repeatedly embed the full training trace.  Keep its
    digest plus the explicit reservation/device/process evidence instead.
    ``reserved_gpu_worker_seconds`` retains its launcher-defined reservation
    semantics and is never inferred from generic wall time or a prediction.
    """

    projected = {
        field_name: candidate_run[field_name]
        for field_name in (
            "wall_time_ms",
            "parent_process_interval",
            "resource_telemetry_sha256",
            "resource_prediction",
            "cuda_visible_devices",
            "gpu_reservation_status",
            "reserved_gpu_worker_seconds",
            "reserved_gpu_worker_seconds_semantics",
            "training_device_evidence",
            "device_evidence_validation",
        )
        if field_name in candidate_run
    }
    reservation = candidate_run.get("gpu_reservation_evidence")
    if isinstance(reservation, Mapping):
        projected["gpu_reservation_evidence"] = dict(reservation)
    return canonical_value(projected)


def _failure_detail(
    *,
    failure_class: ResearchFailureClassV1,
    candidate_run: Mapping[str, Any],
    incumbent_observation: Mapping[str, Any],
) -> dict[str, Any]:
    return canonical_value(
        {
            "candidate_run": dict(candidate_run),
            "failure_class": failure_class.value,
            "incumbent_observation": dict(incumbent_observation),
        }
    )


def project_common_execution_feedback(
    *,
    context: ResearchContext,
    selected_outcome: ProducerOutcome,
    binding: SearchCandidateBindingV1,
    candidate_run: Mapping[str, Any],
    incumbent_observation: Mapping[str, Any],
    metric_contract_digest: str,
    observation_seed: str,
    next_discriminative_test: str,
) -> tuple[
    SearchUtilityEventV2,
    FrozenComparisonIdentityV1,
    TypedResearchEpisodeV1 | None,
    ScientificEpisodeClosureV1,
    dict[str, Any] | None,
]:
    """Project one supplied execution observation into Search and Episode lanes."""

    if not isinstance(context, ResearchContext):
        raise ValueError("context must be ResearchContext")
    if not isinstance(selected_outcome, ProducerOutcome):
        raise ValueError("selected_outcome must be ProducerOutcome")
    if selected_outcome.spec is None:
        raise ValueError("selected_outcome must carry a successful OpenSpec")
    if not isinstance(binding, SearchCandidateBindingV1):
        raise ValueError("binding must be SearchCandidateBindingV1")
    candidate_run = _require_mapping(candidate_run, field_name="candidate_run")
    incumbent_observation = _require_mapping(
        incumbent_observation,
        field_name="incumbent_observation",
    )
    if (
        selected_outcome.context_ref != context.context_ref
        or selected_outcome.context_digest != context.digest
    ):
        raise ValueError("selected Search outcome is not bound to context/binding")
    if isinstance(binding.proposal, CandidateProposalV4):
        if (
            selected_outcome.source_proposal is None
            or selected_outcome.source_proposal.candidate_id
            != binding.proposal.candidate_id
        ):
            raise ValueError("selected legacy Search outcome differs from its binding")
    elif isinstance(binding.proposal, OpenSpecSearchCandidateV1):
        source_spec = binding.proposal.spec.to_dict()
        selected_spec = selected_outcome.spec.to_dict()
        for field_name in (
            "context_ref",
            "context_digest",
            "current_profile_ref",
            "current_profile_digest",
            "current_profile_expressibility_claim",
        ):
            source_spec.pop(field_name, None)
            selected_spec.pop(field_name, None)
        if source_spec != selected_spec:
            raise ValueError("selected OpenSpec Search outcome differs from its lineage")
    else:
        raise ValueError("selected Search binding candidate type is unsupported")
    _require_text(observation_seed, field_name="observation_seed")
    _require_text(
        next_discriminative_test,
        field_name="next_discriminative_test",
    )
    try:
        metric_contract_digest = validate_sha256(
            metric_contract_digest,
            field_name="metric_contract_digest",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if metric_contract_digest != sha256_digest(COMMON_EVALUATOR):
        raise ValueError("metric_contract_digest is not the common evaluator identity")

    binding_ref, binding_digest = _require_ref_digest(
        candidate_run,
        ref_field="experiment_binding_ref",
        digest_field="experiment_binding_digest",
    )
    comparator_ref, comparator_digest = _require_ref_digest(
        incumbent_observation,
        ref_field="comparator_ref",
        digest_field="comparator_digest",
    )
    identity = FrozenComparisonIdentityV1(
        campaign_id=context.campaign_id,
        context_ref=context.context_ref,
        context_digest=context.digest,
        executable_capability_ref=binding.capability_ref,
        executable_capability_digest=binding.capability_digest,
        executable_profile_ref=context.active_profile_ref,
        executable_profile_digest=context.active_profile_digest,
        experiment_binding_ref=binding_ref,
        experiment_binding_digest=binding_digest,
        comparator_ref=comparator_ref,
        comparator_digest=comparator_digest,
        protocol_ref=context.protocol_ref,
        protocol_digest=context.protocol_digest,
    )

    status = str(candidate_run.get("exit_status", ""))
    candidate_metrics = candidate_run.get("metrics")
    candidate_metric = (
        _finite_metric(candidate_metrics.get("ndcg@10"))
        if isinstance(candidate_metrics, Mapping)
        else None
    )
    incumbent_metric = _finite_metric(incumbent_observation.get("frozen_ndcg@10"))
    if status == "SUCCESS" and candidate_metric is not None and incumbent_metric is not None:
        failure_class = ResearchFailureClassV1.INCONCLUSIVE
        comparator_delta: float | str = candidate_metric - incumbent_metric
        runnable_observation = "RUNNABLE"
        blocker = "NONE"
    else:
        if status == "RESOURCE_CENSORED":
            failure_class = ResearchFailureClassV1.RESOURCE
        elif status == "RUNTIME_FAILURE":
            failure_class = ResearchFailureClassV1.RUNTIME
        elif status == "SUCCESS":
            failure_class = ResearchFailureClassV1.OUTCOME_MISSING
        else:
            failure_class = ResearchFailureClassV1.RUNTIME
        comparator_delta = NOT_AVAILABLE
        runnable_observation = "NOT_RUNNABLE"
        blocker = failure_class.value

    event = SearchUtilityEventV2(
        candidate_semantic_digest=binding.mechanism_semantics_digest,
        candidate_id=binding.proposal.candidate_id,
        mechanism_axis=binding.proposal.mechanism_axis,
        common_outcome_class=status or failure_class.value,
        runnable_observation=runnable_observation,
        comparator_delta=comparator_delta,
        metric_contract_digest=metric_contract_digest,
        resource_cost_projection=_cost_projection(candidate_run),
        typed_blocker_class=blocker,
        observation_seed=str(observation_seed),
    )

    if failure_class is ResearchFailureClassV1.INCONCLUSIVE:
        outcome_mapping = canonical_value(
            {
                "candidate_run": dict(candidate_run),
                "incumbent_observation": dict(incumbent_observation),
                "metric_contract_digest": metric_contract_digest,
                "observation_seed": observation_seed,
            }
        )
        cost_mapping = _cost_projection(candidate_run)
        outcome_digest = sha256_digest(outcome_mapping)
        cost_digest = sha256_digest(cost_mapping)
        outcome_ref = content_id(
            "recclaw-search-execution-outcome-v1",
            outcome_mapping,
        )
        cost_ref = content_id(
            "recclaw-search-execution-cost-v1",
            cost_mapping,
        )
        episode = TypedResearchEpisodeV1(
            campaign_id=context.campaign_id,
            context_ref=context.context_ref,
            context_digest=context.digest,
            hypothesis=selected_outcome.spec.hypothesis,
            executable_capability_ref=binding.capability_ref,
            executable_capability_digest=binding.capability_digest,
            executable_profile_ref=context.active_profile_ref,
            executable_profile_digest=context.active_profile_digest,
            experiment_binding_ref=binding_ref,
            experiment_binding_digest=binding_digest,
            comparator_ref=comparator_ref,
            comparator_digest=comparator_digest,
            outcome_ref=outcome_ref,
            outcome_digest=outcome_digest,
            cost_ref=cost_ref,
            cost_digest=cost_digest,
            protocol_ref=context.protocol_ref,
            protocol_digest=context.protocol_digest,
            evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
            experiment_executed=True,
            mechanism_interpretation="NOT_ADJUDICATED",
            competing_explanation=selected_outcome.spec.competing_explanation,
            failure_class=ResearchFailureClassV1.INCONCLUSIVE,
            mechanism_negative_evidence=False,
            next_discriminative_test=next_discriminative_test,
            qualification_receipt_ref=None,
            qualification_receipt_digest=None,
            qualification_evidence_used_as_scientific=False,
        )
        closure = close_scientific_episode(
            comparison_identity=identity,
            failure_class=ResearchFailureClassV1.INCONCLUSIVE,
            episode=episode,
            observed_outcome_ref=outcome_ref,
            observed_outcome_digest=outcome_digest,
        )
        return event, identity, episode, closure, None

    failure_detail = _failure_detail(
        failure_class=failure_class,
        candidate_run=candidate_run,
        incumbent_observation=incumbent_observation,
    )
    failure_detail_digest = sha256_digest(failure_detail)
    failure_detail_ref = content_id(
        "recclaw-search-execution-failure-v1",
        failure_detail,
    )
    closure = close_scientific_episode(
        comparison_identity=identity,
        failure_class=failure_class,
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        failure_detail_ref=failure_detail_ref,
        failure_detail_digest=failure_detail_digest,
    )
    return event, identity, None, closure, failure_detail


__all__ = [
    "execution_recipe_for_search_binding",
    "project_common_execution_feedback",
]
