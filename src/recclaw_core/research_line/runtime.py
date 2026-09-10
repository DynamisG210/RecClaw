"""Small production composition for one complete Research Line round."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    ProviderUnavailableError, provider_failure_is_external, retry_eligible,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import LabApiRequestAdmissionError

from recclaw_core.experiments.helix_abc_v1.capability_admission import (
    VersionedCapabilityRegistry,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_scientific_profile_ref,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (
    MAX_REPAIR_TURNS,
    build_mechanical_repair_request,
    is_mechanical_repair_failure,
    normalize_candidate_runtime_imports,
    scope_mechanical_repair_source,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
    P4_SPARSE_SPECTRAL_EVALUATOR,
    P4_SPARSE_SPECTRAL_SPLIT,
    ExperimentBindingError,
    ExperimentBindingV1,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.idea_quality import (
    admit_research_innovation_candidate,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    MechanicalRecBoleAdapterError,
    MechanicalRecBoleAdapterV1,
    MechanicalQualificationRun,
    RecBoleQualificationFixture,
    snapshot_candidate_tree,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    InnovationSpineError,
    MaterializedCandidate,
    PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE,
    PARENT_METHOD_PATCH_RESPONSE_MODE,
    PROFILE_MODEL_HOOKS_RESPONSE_MODE,
    SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE,
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
    validate_mechanism_source_behavior,
)
from recclaw_core.experiments.helix_abc_v1.next_fresh_profile import (
    NextFreshProfileBuildManifest,
)
from recclaw_core.experiments.helix_abc_v1.meta_control import (
    MetaControlActivationReceiptV1,
    MetaControlPromotionDecisionV1,
    MetaControlShadowEvaluationV1,
    MetaControlUpdateProposalV1,
    activate_promoted_control_policy,
    build_meta_update_proposal,
    materialize_proposed_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    project_candidate_proposal_v4,
)
from recclaw_core.experiments.helix_abc_v1.producer_opportunity import (
    acquire_producer_opportunity,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedResearchPolicyV1,
    _next_acquisition_parameters,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DISCOVERY_PRODUCERS,
    MECHANISM_AXIS_UNIVERSE_V1,
    canonical_mechanism_axis,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
    reclassify_progressing_timeout_profile,
    resource_probe_execution_key,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    ExperimentAcquisitionResultV1,
    FrozenExperimentSlateV1,
    OpenSpecSearchCandidateV1,
    QualifiedSearchCandidateProtocolV1,
    SearchCandidateBindingV1,
    SearchExecutableProfileV1,
    SearchProfileEntryOriginV1,
    activate_next_fresh_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
    predecessor_executable_entries,
    route_frozen_experiment_slate,
)
from recclaw_core.experiments.helix_abc_v1.scientific_episode import (
    close_scientific_episode,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityKindV1,
    CapabilityResolutionResultV1,
    CapabilityResolutionV1,
    CurrentProfileExpressibilityV1,
    ExecutableProfileVNext,
    ProfileBuildReceiptV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
    ResearchFailureClassV1,
)
from .lineage_identity import exact_parent_candidate_id
from recclaw_core.experiments.helix_abc_v1.vnext_orchestration import (
    admit_local_qualification,
    build_local_next_fresh_profile,
    qualify_local_innovation_candidate,
    resolve_producer_outcomes,
)
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.mechanism_space import compile_program

from .execution import (
    project_common_execution_feedback,
)
from .search_space_adapter import (
    ConfirmationResolutionKindV1,
    ConfirmationResolutionV1,
    SearchSpaceAdapter,
    SearchSpaceExecutionBindingV1,
)
from .interfaces import (
    research_producer_roles,
    ProducerOutcome,
    ResearchContext,
    ResearchTaskOperationV2,
    ResearchTaskQueueV2,
    ResearchTaskRecordV2,
    ResearchTaskStatusV2,
    observation_updates_search_utility,
    project_provider_context_view,
)
from .single_parent_search import (
    bound_parent_from_context,
    exact_parent_bundle_from_context,
    is_single_parent_context,
    single_parent_research_axes,
)
from .interpreter import (
    EpisodeInterpretation,
    MissingSearchInterpretation,
    _behavior,
    _meta_aggregate,
    interpret_missing_search_opportunity,
    interpret_scientific_diagnostic,
    interpret_typed_research_episode,
    interpret_verification_diagnostic,
    interpret_verification_episode,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_resolver_environment,
)
from .portfolio import PortfolioAttemptFailureV2, PortfolioCandidateV2
from .portfolio_profile_builder import PortfolioCandidateProfileV2
from .producers import (
    ResearchProducer,
    compiler_owned_execution_contract,
    compiler_owned_generic_execution_contract,
    produce_research_specs,
)
from .profile_source import (
    ResearchProfileRecordV1,
    ResearchProfileSourceV1,
    lineage_identity_digest,
)
from .provider import (
    RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING,
    _inherit_unchanged_parent_slots,
    _latest_research_traces_by_role,
)


_NEXT_DEVELOPMENT_SEED = "NEXT_DEVELOPMENT_SEED"


def _default_search_space_adapter() -> SearchSpaceAdapter:
    from .bl_icf_search_space_adapter import default_search_space_adapter

    return default_search_space_adapter()


def _resolve_confirmation(
    adapter: SearchSpaceAdapter,
    *,
    pending_task: Mapping[str, Any],
    primary_binding: Any,
    next_discriminative_test: str,
) -> ConfirmationResolutionV1:
    task_record = pending_task.get("task_record")
    operation = (
        task_record.get("operation")
        if isinstance(task_record, Mapping)
        else pending_task.get("operation")
    )
    kind = str(getattr(operation, "value", operation) or "")
    resolution = adapter.resolve_confirmation(
        kind,
        primary_binding,
        {
            "phase": "MATCH_PROPOSAL",
            "pending_task": pending_task,
            "next_discriminative_test": next_discriminative_test,
        },
    )
    if not isinstance(resolution, ConfirmationResolutionV1):
        raise TypeError(
            "SearchSpaceAdapter.resolve_confirmation must return "
            "ConfirmationResolutionV1"
        )
    return resolution


def _materialize_discovery_binding(
    adapter: SearchSpaceAdapter,
    *,
    outcome: ProducerOutcome,
    resolution: CapabilityResolutionV1,
    profile: SearchExecutableProfileV1,
) -> SearchCandidateBindingV1:
    materialized = adapter.resolve_confirmation(
        "DISCOVERY",
        outcome,
        {
            "phase": "MATERIALIZE_DISCOVERY",
            "resolution": resolution,
            "execution_profile": profile,
        },
    )
    if (
        not isinstance(materialized, ConfirmationResolutionV1)
        or materialized.kind is not ConfirmationResolutionKindV1.EXACT_BINDING
        or not isinstance(materialized.binding, Mapping)
        or not isinstance(
            materialized.binding.get("native_binding"),
            SearchCandidateBindingV1,
        )
    ):
        raise ValueError("adapter did not materialize the discovery binding")
    binding = materialized.binding["native_binding"]
    if (
        outcome.source_proposal is None
        or binding.proposal != outcome.source_proposal
        or binding.capability_ref != resolution.resolved_current_capability_ref
    ):
        raise ValueError("adapter discovery binding differs from Resolver identity")
    entry = profile.entry(binding.capability_ref)
    if (
        binding.capability_digest != entry.capability_digest
        or binding.executable_entrypoint != entry.executable_entrypoint
        or binding.entry_origin != entry.origin
        or binding.mechanism_semantics_digest != entry.semantic_identity_digest
    ):
        raise ValueError("adapter discovery binding differs from active profile")
    return binding


def _declare_followup_availability(
    adapter: SearchSpaceAdapter,
    *,
    selected_outcome: ProducerOutcome,
    context: ResearchContext,
    mechanism_program: Mapping[str, Any],
) -> Mapping[str, str]:
    availability: dict[str, str] = {}
    for kind in ("MATCHED_CONTROL", "MECHANISM_OFF"):
        resolution = adapter.resolve_confirmation(
            kind,
            selected_outcome,
            {
                "phase": "DECLARE_FOLLOWUP",
                "research_context": context,
                "mechanism_program": mechanism_program,
            },
        )
        if not isinstance(resolution, ConfirmationResolutionV1):
            raise TypeError(
                "SearchSpaceAdapter.resolve_confirmation must return "
                "ConfirmationResolutionV1"
            )
        availability[kind] = resolution.kind.value
    return canonical_value(availability)


def _exact_confirmation_candidate_ids(
    pending_task: Mapping[str, Any] | None,
) -> frozenset[str]:
    if not isinstance(pending_task, Mapping):
        return frozenset()
    values = pending_task.get("confirmation_candidate_ids", ())
    if isinstance(values, str) or not isinstance(values, (tuple, list)):
        return frozenset()
    return frozenset(str(value) for value in values if isinstance(value, str))


def _search_space_execution_binding(
    adapter: SearchSpaceAdapter,
    binding: SearchCandidateBindingV1,
    confirmation_binding: Mapping[str, Any] | None,
    *,
    execution_context: Mapping[str, Any] | None = None,
) -> SearchSpaceExecutionBindingV1:
    return SearchSpaceExecutionBindingV1(
        adapter_id=adapter.adapter_id,
        binding_ref=content_id(
            "recclaw-search-space-execution-binding-v1",
            {
                "adapter_id": adapter.adapter_id,
                "binding_digest": binding.digest,
                "confirmation_binding": confirmation_binding,
            },
        ),
        binding_digest=binding.digest,
        candidate_id=binding.proposal.candidate_id,
        semantic_identity_digest=binding.mechanism_semantics_digest,
        native_binding=binding,
        execution_context={
            "confirmation_binding": confirmation_binding,
            **dict(execution_context or {}),
        },
    )


def _adapter_execution_attestation(
    binding: SearchCandidateBindingV1,
    *,
    search_space_adapter: SearchSpaceAdapter,
    confirmation_binding: Mapping[str, Any] | None,
    execution_context: Mapping[str, Any],
) -> tuple[SearchSpaceExecutionBindingV1, Mapping[str, Any], Mapping[str, Any]]:
    adapter_binding = _search_space_execution_binding(
        search_space_adapter,
        binding,
        confirmation_binding,
        execution_context=execution_context,
    )
    search_space_adapter.validate_execution_binding(adapter_binding)
    identity = search_space_adapter.effective_identity(adapter_binding)
    recipe = search_space_adapter.execution_recipe(adapter_binding)
    if not isinstance(identity, Mapping):
        raise TypeError("SearchSpaceAdapter.effective_identity must return a mapping")
    if not isinstance(recipe, Mapping):
        raise TypeError("SearchSpaceAdapter.execution_recipe must return a mapping")
    normalized_identity = canonical_value(dict(identity))
    for field_name in (
        "effective_experiment_digest",
        "effective_family_digest",
    ):
        value = normalized_identity.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                "SearchSpaceAdapter effective identity lacks " + field_name
            )
    return (
        adapter_binding,
        normalized_identity,
        canonical_value(dict(recipe)),
    )


def _adapter_route_evidence(
    adapter_binding: SearchSpaceExecutionBindingV1,
    effective_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    confirmation_binding = adapter_binding.execution_context.get(
        "confirmation_binding"
    )
    evidence = {
            "adapter_id": adapter_binding.adapter_id,
            "binding_ref": adapter_binding.binding_ref,
            "binding_digest": adapter_binding.binding_digest,
            "candidate_id": adapter_binding.candidate_id,
            "semantic_identity_digest": (
                adapter_binding.semantic_identity_digest
            ),
            "effective_experiment_digest": effective_identity[
                "effective_experiment_digest"
            ],
            "effective_family_digest": effective_identity[
                "effective_family_digest"
            ],
            "opaque_binding": confirmation_binding,
        }
    primitive_ids = effective_identity.get("primitive_ids")
    if isinstance(primitive_ids, (tuple, list)):
        evidence["primitive_ids"] = tuple(str(item) for item in primitive_ids)
    return canonical_value(evidence)


def _trace_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_trace_value(item) for item in value)
    for method_name in ("to_dict", "canonical_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return _trace_value(method())
    if is_dataclass(value):
        return {
            field.name: _trace_value(getattr(value, field.name, None))
            for field in fields(value)
        }
    return canonical_value(value)


def _trace_dataclass(value: Any) -> dict[str, Any]:
    return canonical_value(
        {
            field.name: _trace_value(getattr(value, field.name, None))
            for field in fields(value)
        }
    )


_PREBINDING_RETRY_ORDINAL = 1
_PREBINDING_RETRY_NAMESPACE = "prebinding-retry-1"
_PREBINDING_RETRY_STATUSES = frozenset({"PENDING", "SUCCEEDED", "EXHAUSTED"})
_TRANSPORT_RETRY_NAMESPACE_PREFIX = "transport-retry-"


def _normalize_prebinding_retry(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("prebinding_retry must be a canonical mapping")
    normalized = canonical_value(dict(value))
    if normalized.get("ordinal") != _PREBINDING_RETRY_ORDINAL:
        raise ValueError("prebinding retry ordinal is outside the frozen budget")
    if normalized.get("status") not in _PREBINDING_RETRY_STATUSES:
        raise ValueError("prebinding retry status is invalid")
    if normalized.get("retry_logical_namespace") != _PREBINDING_RETRY_NAMESPACE:
        raise ValueError("prebinding retry logical namespace is invalid")
    usage_charge = normalized.get("usage_charge")
    if (
        isinstance(usage_charge, bool)
        or not isinstance(usage_charge, int)
        or usage_charge < 0
    ):
        raise ValueError("prebinding retry usage_charge is invalid")
    for key in ("first_outcomes", "first_provider_traces"):
        if not isinstance(normalized.get(key), (tuple, list)):
            raise ValueError(f"prebinding retry {key} are invalid")
    return normalized


def _validate_prebinding_retry_identity(
    value: Mapping[str, Any],
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    budget_snapshot: Mapping[str, Any],
) -> None:
    expected = {
        "context_digest": context.digest,
        "profile_ref": active_profile.profile_ref,
        "profile_digest": active_profile.profile_digest,
        "budget_snapshot_digest": sha256_digest(budget_snapshot),
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            raise ValueError(f"prebinding retry {field_name} drift")


def _provider_failure_reason_code(trace: Mapping[str, Any]) -> str | None:
    """Read only the typed reason exposed by the Provider trace contract."""

    failure = trace.get("failure")
    if not isinstance(failure, Mapping):
        return None
    reason = failure.get("reason_code")
    if isinstance(reason, str):
        return reason
    detail = failure.get("error_detail")
    if isinstance(detail, Mapping) and isinstance(detail.get("reason_code"), str):
        return str(detail["reason_code"])
    return None


def _prepared_producer_roles(prepared: PreparedResearchRoundV1) -> tuple[str, ...]:
    metadata = (getattr(prepared.producer_outcomes[0], "provenance", {})
                if prepared.producer_outcomes else {})
    return research_producer_roles(metadata)


def _all_scheduled_token_ceiling_failures(
    producer_outcomes: Sequence[ProducerOutcome],
    provider_traces: Sequence[Mapping[str, Any]],
    *,
    scheduled_roles: Sequence[str] = DISCOVERY_PRODUCERS,
) -> bool:
    if len(producer_outcomes) != len(scheduled_roles):
        return False
    if {item.producer_role for item in producer_outcomes} != set(scheduled_roles):
        return False
    if any(item.spec is not None for item in producer_outcomes):
        return False
    traces = tuple(
        item
        for item in provider_traces
        if item.get("kind") == "research_producer"
    )
    return len(traces) == len(scheduled_roles) and all(
        _provider_failure_reason_code(item) == "TOKEN_CEILING" for item in traces
    )


def _provider_usage_charge(
    traces: Sequence[Mapping[str, Any]],
    producer: ResearchProducer,
) -> int:
    ceiling = getattr(
        producer,
        "total_token_ceiling",
        RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING,
    )
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 1:
        ceiling = RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING
    total = 0
    for trace in traces:
        transport_ceiling = trace.get("transport_ceiling")
        trace_ceiling = (
            transport_ceiling.get("total_tokens")
            if isinstance(transport_ceiling, Mapping)
            else None
        )
        if (
            isinstance(trace_ceiling, bool)
            or not isinstance(trace_ceiling, int)
            or trace_ceiling < 1
        ):
            trace_ceiling = ceiling
        attempts = trace.get("attempts")
        if isinstance(attempts, (tuple, list)) and attempts:
            for attempt in attempts:
                if (isinstance(attempt, Mapping)
                    and attempt.get("billed_tokens") == 0
                    and attempt.get("usage_charge_basis") == "CONFIRMED_UNBILLED_EXTERNAL_FAILURE"):
                    continue
                charge = (
                    attempt.get("billed_tokens")
                    if isinstance(attempt, Mapping)
                    else None
                )
                if (
                    isinstance(charge, int)
                    and not isinstance(charge, bool)
                    and charge > 0
                ):
                    total += charge
                else:
                    total += trace_ceiling
            continue
        usage = trace.get("usage")
        billed = usage.get("billed_tokens") if isinstance(usage, Mapping) else None
        failure = trace.get("failure")
        if (isinstance(failure, Mapping)
            and failure.get("billed_tokens") == 0
            and failure.get("usage_charge_basis") == "CONFIRMED_UNBILLED_EXTERNAL_FAILURE"):
            continue
        failure_billed = (
            failure.get("billed_tokens")
            if isinstance(failure, Mapping)
            else None
        )
        for charge in (billed, failure_billed):
            if (
                isinstance(charge, int)
                and not isinstance(charge, bool)
                and charge > 0
            ):
                total += charge
                break
        else:
            total += trace_ceiling
    return total


def _namespaced_retry_producer(producer: ResearchProducer) -> ResearchProducer | None:
    method = getattr(producer, "call_with_namespace", None)
    if not callable(method):
        return None

    def call(producer_role: str, context_view: Mapping[str, Any]) -> Any:
        return method(
            producer_role,
            context_view,
            logical_namespace=_PREBINDING_RETRY_NAMESPACE,
        )

    return call


def _next_transport_retry_namespace(
    provider_traces: Sequence[Mapping[str, Any]],
) -> str:
    """Derive the next crash-stable Provider transport identity."""

    marker = f":{_TRANSPORT_RETRY_NAMESPACE_PREFIX}"
    highest = 0
    for trace in provider_traces:
        if not isinstance(trace, Mapping):
            continue
        logical_call_id = trace.get("logical_call_id")
        if not isinstance(logical_call_id, str):
            continue
        _prefix, separator, suffix = logical_call_id.partition(marker)
        if not separator:
            continue
        raw_ordinal = suffix.split(":", 1)[0]
        if raw_ordinal.isdigit():
            highest = max(highest, int(raw_ordinal))
    return f"{_TRANSPORT_RETRY_NAMESPACE_PREFIX}{highest + 1}"


def _transport_retry_producer(
    producer: ResearchProducer,
    provider_traces: Sequence[Mapping[str, Any]],
) -> ResearchProducer:
    """Use a fresh durable broker identity without changing scientific state."""

    method = getattr(producer, "call_with_namespace", None)
    if not callable(method):
        return producer
    logical_namespace = _next_transport_retry_namespace(provider_traces)
    resume = getattr(producer, "resume_transport_call", None)
    latest_by_role = _latest_research_traces_by_role(provider_traces)

    def call(producer_role: str, context_view: Mapping[str, Any]) -> Any:
        trace = latest_by_role.get(producer_role)
        if callable(resume) and trace is not None:
            return resume(
                producer_role, context_view, logical_namespace=logical_namespace,
                trace=trace,
            )
        return method(
            producer_role,
            context_view,
            logical_namespace=logical_namespace,
        )

    return call


def _is_implementation_admission_failure(failure: Mapping[str, Any]) -> bool:
    return failure.get("failure_class") == "PROVIDER_ADMISSION" or (
        failure.get("failure_class") == "PROVIDER"
        and failure.get("stage") == "PROVIDER"
        and failure.get("reason_code") == "BudgetExhausted"
    )


def _innovation_provider_failure(
    innovation: InnovationLaneResult | None,
) -> Mapping[str, Any] | None:
    if innovation is not None and innovation.attempts:
        failure = innovation.attempts[-1].get("failure")
        if (
            isinstance(failure, Mapping)
            and (
                failure.get("failure_class") == "PROVIDER_EXTERNAL"
                or _is_implementation_admission_failure(failure)
            )
        ):
            return failure
    return None


def _prepared_has_external_implementation_failure(
    prepared: PreparedResearchRoundV1,
) -> bool:
    return (
        not prepared.search_bindings
        and _innovation_provider_failure(prepared.innovation) is not None
    )


def _provider_attempt_is_terminal(trace: Mapping[str, Any]) -> bool:
    for attempt in trace.get("attempts", ()):
        if not isinstance(attempt, Mapping):
            continue
        if str(attempt.get("failure_class") or "") in {
            "AUTHENTICATION_ERROR",
            "CLI_CONTRACT_ERROR",
        }:
            return True
        if str(attempt.get("reason_code") or "") in {
            "CONTENT_JSON_DECODE",
            "ENVELOPE_JSON_DECODE",
            "SCHEMA_VALIDATION",
        }:
            return True
    return False


def _prepared_has_terminal_provider_failure(
    prepared: PreparedResearchRoundV1,
) -> bool:
    return bool(
        not prepared.search_bindings
        and prepared.innovation is None
        and any(
            _provider_attempt_is_terminal(trace)
            for trace in prepared.provider_traces
            if isinstance(trace, Mapping)
        )
    )


_LEGACY_ORIGINAL_IDENTITY_COLLISION = (
    "logical API call identity differs from stored bytes"
)


def _is_canonical_sha256(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return validate_sha256(value, field_name=field_name) == value
    except ValueError:
        return False


def _prepared_is_legacy_original_identity_collision(
    prepared: PreparedResearchRoundV1,
) -> bool:
    """Recognize the sealed-request migration boundary observed at R46.

    This is deliberately narrower than a generic local broker failure.  The
    checkpoint was produced after an Original-matched logical identity found
    a different historical request in the immutable broker; the derivative
    roles then failed only because that one batched response was unavailable.
    """

    outcomes = tuple(prepared.producer_outcomes)
    if (
        getattr(prepared, "prebinding_retry", None) is not None
        or len(outcomes) != len(DISCOVERY_PRODUCERS)
        or tuple(item.producer_role for item in outcomes)
        != tuple(DISCOVERY_PRODUCERS)
        or any(
            item.spec is not None
            or item.failure_code != "PRODUCER_CALL_FAILED"
            for item in outcomes
        )
        or outcomes[0].failure_detail
        != "FreshR1Error: Original-matched proposal Provider call failed"
        or any(
            outcome.failure_detail != f"KeyError: '{role}'"
            for outcome, role in zip(
                outcomes[1:], DISCOVERY_PRODUCERS[1:], strict=True
            )
        )
        or len(prepared.provider_traces) != 1
    ):
        return False

    trace = prepared.provider_traces[0]
    if not isinstance(trace, Mapping):
        return False
    attempts = trace.get("attempts")
    sealed = trace.get("sealed_request")
    receipt = trace.get("receipt")
    failure = trace.get("failure")
    logical_call_id = trace.get("logical_call_id")
    session_id = trace.get("session_id")
    if (
        trace.get("kind") != "original_matched_producer"
        or not isinstance(logical_call_id, str)
        or ":original-matched:" not in logical_call_id
        or not isinstance(session_id, str)
        or not session_id
        or not isinstance(attempts, (tuple, list))
        or len(attempts) != 1
        or not isinstance(sealed, Mapping)
        or sealed.get("schema") != "recclaw.provider.sealed-request.v1"
        or sealed.get("logical_call_id") != logical_call_id
        or sealed.get("proposal_generation_session_id") != session_id
        or sealed.get("expected_proposal_count") != len(DISCOVERY_PRODUCERS)
        or not isinstance(sealed.get("request_payload"), Mapping)
        or not sealed.get("request_payload")
        or not isinstance(receipt, Mapping)
        or not isinstance(failure, Mapping)
    ):
        return False

    attempt = attempts[0]
    if not isinstance(attempt, Mapping):
        return False
    sealed_digest = sealed.get("request_digest")
    historical_digest = attempt.get("request_digest")
    if (
        not _is_canonical_sha256(
            sealed_digest,
            field_name="legacy_original.sealed_request.request_digest",
        )
        or not _is_canonical_sha256(
            historical_digest,
            field_name="legacy_original.historical_request_digest",
        )
        or sealed_digest == historical_digest
    ):
        return False

    physical_call_count = attempt.get("physical_call_count")
    failure_physical_call_count = failure.get("physical_call_count")
    return bool(
        attempt.get("status") == "FAILED"
        and attempt.get("failure_class") == "LOCAL_BROKER_FAILURE"
        and attempt.get("message") == _LEGACY_ORIGINAL_IDENTITY_COLLISION
        and not isinstance(physical_call_count, bool)
        and physical_call_count == 0
        and attempt.get("reason_code") is None
        and attempt.get("provider_role") == "original_matched_proposal"
        and attempt.get("requested_model") == "gpt-5.4"
        and receipt.get("status") == "FAILED"
        and receipt.get("logical_call_id") == logical_call_id
        and receipt.get("proposal_generation_session_id") == session_id
        and receipt.get("provider_role") == "original_matched_proposal"
        and receipt.get("requested_model") == "gpt-5.4"
        and receipt.get("request_digest") == historical_digest
        and failure.get("failure_class") == "LOCAL_BROKER_FAILURE"
        and failure.get("message") == _LEGACY_ORIGINAL_IDENTITY_COLLISION
        and not isinstance(failure_physical_call_count, bool)
        and failure_physical_call_count == 0
        and failure.get("reason_code") is None
        and failure.get("request_digest") == historical_digest
    )


def _prepared_has_transient_producer_failure(
    prepared: PreparedResearchRoundV1,
) -> bool:
    """Recognize only an unfinished Provider boundary, not typed rejection."""
    if (
        prepared.search_bindings
        or prepared.innovation is not None
        or not prepared.producer_outcomes
    ):
        return False
    outcomes = tuple(prepared.producer_outcomes)
    scheduled_roles = _prepared_producer_roles(prepared)
    complete_scheduled_failure = bool(
        len(outcomes) == len(scheduled_roles)
        and {outcome.producer_role for outcome in outcomes}
        == set(scheduled_roles)
    )
    all_calls_failed = all(
        outcome.spec is None
        and outcome.failure_code == "PRODUCER_CALL_FAILED"
        for outcome in outcomes
    )
    standard_traces = tuple(
        trace
        for trace in prepared.provider_traces
        if isinstance(trace, Mapping)
        and trace.get("kind") == "research_producer"
    )
    latest_trace_by_role = _latest_research_traces_by_role(prepared.provider_traces)
    retryable_roles = {
        role for role, trace in latest_trace_by_role.items()
        if retry_eligible(trace.get("failure") or {})
        or (trace.get("failure") or {}).get("failure_class") == "CONNECTIVITY_ERROR"
    }
    if _prepared_has_terminal_provider_failure(prepared) and not retryable_roles:
        return False
    all_provider_receipts_failed = bool(
        len(standard_traces) == len(scheduled_roles)
    ) and all(
        isinstance(trace, Mapping)
        and isinstance(trace.get("receipt"), Mapping)
        and trace["receipt"].get("status") == "FAILED"
        for trace in standard_traces
    )
    ordinary_transient = complete_scheduled_failure and all_calls_failed and (
        all(
            "Provider call failed" in str(outcome.failure_detail or "")
            for outcome in outcomes
        )
        or all_provider_receipts_failed
    )
    unfinished_started_boundary = bool(
        not prepared.provider_traces
        and len(outcomes) == len(scheduled_roles)
        and {outcome.producer_role for outcome in outcomes}
        == set(scheduled_roles)
        and all(
            outcome.spec is None
            and outcome.failure_code == "PRODUCER_CALL_FAILED"
            for outcome in outcomes
        )
    )
    return bool(
        ordinary_transient
        or unfinished_started_boundary
        or _prepared_is_legacy_original_identity_collision(prepared)
        or any(
            outcome.spec is None
            and outcome.failure_code == "PRODUCER_CALL_FAILED"
            and latest_trace_by_role.get(outcome.producer_role, {}).get("receipt", {}).get("status") == "FAILED"
            and outcome.producer_role in retryable_roles
            for outcome in outcomes
        )
    )


def _unfinished_producer_roles(prepared: PreparedResearchRoundV1) -> frozenset[str]:
    """An unreturned call can resume; a returned invalid proposal needs new research."""
    if (
        prepared.search_bindings or prepared.innovation is not None
    ):
        return frozenset()
    if any(
        trace.get("kind") == "original_matched_producer"
        for trace in prepared.provider_traces if isinstance(trace, Mapping)
    ):
        # Original has a single batched call and its own recovery identity.
        return frozenset(DISCOVERY_PRODUCERS) if _prepared_has_transient_producer_failure(prepared) else frozenset()
    traces = _latest_research_traces_by_role(prepared.provider_traces)
    return frozenset(
        outcome.producer_role for outcome in prepared.producer_outcomes
        if outcome.spec is None
        and outcome.failure_code in {"PRODUCER_CALL_FAILED", "PRODUCER_REQUEST_FAILED"}
        and traces.get(outcome.producer_role, {}).get("receipt", {}).get("status") != "SUCCESS"
        and (
            not _provider_attempt_is_terminal(traces.get(outcome.producer_role, {}))
            or provider_failure_is_external(traces.get(outcome.producer_role, {}).get("failure"))
        )
    )


def _prepared_has_unfinished_producer_failure(prepared: PreparedResearchRoundV1) -> bool:
    return bool(_unfinished_producer_roles(prepared))


class ImplementerGateway(Protocol):
    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ExperimentRunner(Protocol):
    def __call__(
        self,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
    ) -> Mapping[str, Any]: ...


class EvidenceGuardPort(Protocol):
    """Research-owned boundary for the helix Evidence Guard bridge.

    The runtime deliberately depends on this small mapping contract instead
    of importing ``helix`` or the Guard implementation.  A concrete bridge
    must return the candidate identity and stage in every adjudication; an
    invalid or missing response is fail-closed before scientific state can
    advance.
    """

    def pre_run(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        observation_seed: str,
    ) -> Mapping[str, Any]: ...

    def post_run(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        candidate_run: Mapping[str, Any],
        closure: Any,
        observation_seed: str,
        active_task: Mapping[str, Any] | None = None,
        remaining_metric_opportunities: int | None = None,
    ) -> Mapping[str, Any]: ...

    def adjust_portfolio_information(
        self,
        candidates: Sequence[PortfolioCandidateV2],
    ) -> Sequence[PortfolioCandidateV2]: ...


class CandidateHandoffFactory(Protocol):
    """Build the complete, outcome-blind handoff for a frozen round."""

    def __call__(
        self,
        *,
        context: ResearchContext,
        active_profile: SearchExecutableProfileV1,
        resolutions: Sequence[
            tuple[ProducerOutcome, CapabilityResolutionV1 | None]
        ],
        search_bindings: Sequence[SearchCandidateBindingV1],
        qualified_execution_by_capability: Mapping[str, Mapping[str, Any]],
        candidate_root_by_capability: Mapping[str, str | Path],
        resource_profile_by_capability: Mapping[str, Mapping[str, Any]],
    ) -> Sequence["RoundCandidateHandoffV1"]: ...


class InnovationResourceProbe(Protocol):
    def __call__(
        self,
        *,
        candidate_root: Path,
        source_path: Path,
        entrypoint: str,
        source_sha256: str,
        execution_recipe: Mapping[str, Any],
        probe_root: Path,
        candidate_ref: str,
        candidate_package_digest: str,
    ) -> Mapping[str, Any]: ...


class MetaShadowReplay(Protocol):
    def __call__(
        self,
        *,
        context: ResearchContext,
        champion_policy: VersionedResearchPolicyV1,
        challenger_policy: VersionedResearchPolicyV1,
        search_memory: Any,
    ) -> Mapping[str, Any]: ...


SearchCandidate = CandidateProposalV4 | QualifiedSearchCandidateProtocolV1


@dataclass(frozen=True, slots=True)
class InnovationRuntimeInputs:
    """Inputs with immediate consumers in implementation, admission, or activation."""

    implementer: ImplementerGateway
    policy: SharedImplementerPolicy
    candidate_parent: Path
    fixture_factory: Callable[
        [SharedImplementerPolicy, int, Path], RecBoleQualificationFixture
    ]
    unit_check_factory: Callable[
        [SharedImplementerPolicy], Callable[[Any, Any, Any], None]
    ]
    capability_kind: CapabilityKindV1
    capability_version: str
    registry_version: str
    predecessor_registry_ref: str
    predecessor_registry_digest: str
    profile_version: str
    fresh_campaign_id: str
    resource_admission_required: bool = False
    resource_probe: InnovationResourceProbe | None = None
    resource_probe_parent: Path | None = None
    resource_structural_context: Mapping[str, Any] | None = None
    qualification_executor: Callable[..., MechanicalQualificationRun] | None = None
    native_training_cadence: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        contract = self.policy.execution_contract
        required = {"capability_family", "model", "base_model_config", "config"}
        if contract is not None and (
            not isinstance(contract, Mapping) or set(contract) != required
        ):
            raise ValueError(
                "Innovation execution_contract, when static, must contain exactly the four qualified execution fields"
            )
        if not isinstance(self.resource_admission_required, bool):
            raise ValueError("resource_admission_required must be a boolean")
        if self.native_training_cadence is not None:
            cadence = dict(self.native_training_cadence)
            if set(cadence) != {"epochs", "eval_step", "stopping_step"} or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in cadence.values()
            ):
                raise ValueError("native_training_cadence requires positive epochs/eval_step/stopping_step")
            object.__setattr__(self, "native_training_cadence", canonical_value(cadence))
        configured = self.resource_probe is not None or self.resource_probe_parent is not None
        if self.resource_admission_required and not configured:
            raise ValueError(
                "resource-admitted Innovation requires a disposable resource probe"
            )
        if configured and (
            not callable(self.resource_probe)
            or not isinstance(self.resource_probe_parent, Path)
        ):
            raise ValueError(
                "resource_probe and resource_probe_parent must be supplied together"
            )
        if self.resource_probe_parent is not None:
            object.__setattr__(
                self,
                "resource_probe_parent",
                self.resource_probe_parent.resolve(),
            )
        if self.resource_structural_context is not None and not isinstance(
            self.resource_structural_context, Mapping
        ):
            raise ValueError("resource_structural_context must be a mapping")
        object.__setattr__(
            self,
            "resource_structural_context",
            canonical_value(dict(self.resource_structural_context or {})),
        )


@dataclass(frozen=True, slots=True)
class IdeaAcquisitionResult:
    """Minimal policy-consuming acquisition trace for OpenSpec Innovation."""

    pool_digest: str
    ranked_spec_digests: tuple[str, ...]
    selected_spec_digest: str
    score_by_spec_digest: Mapping[str, float]
    policy_digest: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MetaResearchInputs:
    """The fresh-campaign boundary for one outcome-driven Meta update."""

    offline_replay: MetaShadowReplay
    next_campaign_id: str

    def __post_init__(self) -> None:
        if not self.next_campaign_id.strip():
            raise ValueError("next_campaign_id must be non-empty")


@dataclass(frozen=True, slots=True)
class MetaResearchResult:
    update_proposal: MetaControlUpdateProposalV1
    challenger_policy: VersionedResearchPolicyV1
    offline_replay: Mapping[str, Any] | None
    shadow_evaluation: MetaControlShadowEvaluationV1 | None
    promotion_decision: MetaControlPromotionDecisionV1
    activated_policy: VersionedResearchPolicyV1 | None
    activation_receipt: MetaControlActivationReceiptV1 | None

    def to_dict(self) -> dict[str, Any]:
        return _trace_dataclass(self)


def _innovation_attempt_proposal_digest(attempt: Mapping[str, Any]) -> str | None:
    """Read the frozen selection identity, including already saved attempts.

    An adapter may bind parent labels or executable metadata after acquisition;
    that effective implementation spec is not a new research opportunity.
    """
    acquisition = attempt.get("idea_acquisition")
    if isinstance(acquisition, Mapping):
        return acquisition["selected_spec_digest"]
    return attempt.get("spec_digest")


@dataclass(frozen=True, slots=True)
class InnovationLaneResult:
    selected_outcome: ProducerOutcome
    resolution: CapabilityResolutionV1
    idea_acquisition: IdeaAcquisitionResult
    attempts: tuple[Mapping[str, Any], ...]
    materialized: MaterializedCandidate | None
    qualification: MechanicalQualificationRun | None
    capability: QualifiedCapabilityV1 | None
    registry: VersionedCapabilityRegistry | None
    profile_manifest: NextFreshProfileBuildManifest | None
    next_profile: ExecutableProfileVNext | None
    profile_receipt: ProfileBuildReceiptV1 | None
    qualified_execution: Mapping[str, Any] | None
    search_candidate: SearchCandidate | None
    candidate_root: str | None
    fresh_campaign_id: str
    resource_profile: Mapping[str, Any] | None = None
    quality_admission: Mapping[str, Any] | None = None

    @property
    def admitted(self) -> bool:
        if self.quality_admission is not None:
            return bool(self.quality_admission.get("admitted") is True)
        return self.capability is not None

    @property
    def mechanically_qualified(self) -> bool:
        return self.capability is not None

    @property
    def resource_admitted(self) -> bool:
        return bool(
            self.quality_admission is not None
            and self.quality_admission.get("admitted") is True
            and self.quality_admission.get("status") == "RESOURCE_ADMITTED"
        )

    @property
    def activation_ready(self) -> bool:
        return (
            self.admitted
            and self.next_profile is not None
            and self.search_candidate is not None
        )

    @property
    def training_ready(self) -> bool:
        return self.activation_ready and (
            self.resource_admitted
            or (self.qualified_execution or {}).get("native_training_cadence") is not None
        )

    @property
    def candidate_attempt_count(self) -> int:
        """Count distinct candidates that entered implementation/qualification.

        Proposal-semantic and effective-identity duplicates are rejected before
        an implementation root exists.  They are useful generation feedback,
        but they must not exhaust the fixed budget for real candidate attempts.
        Neither may worker-transient, shared-infrastructure, or recovery stops:
        those leave the unchanged candidate without a scientific outcome.
        """

        def consumes_candidate_attempt(item: Mapping[str, Any]) -> bool:
            raw_scope = item.get("failure_scope")
            failure = item.get("failure")
            if raw_scope is None and isinstance(failure, Mapping):
                raw_scope = failure.get("failure_scope")
            return str(raw_scope or "").upper() not in {
                "WORKER_TRANSIENT",
                "SHARED_INFRASTRUCTURE",
                "RECOVERY",
            }

        return len(
            {
                _innovation_attempt_proposal_digest(item)
                for item in self.attempts
                if _innovation_attempt_proposal_digest(item) is not None
                and isinstance(item.get("candidate_root"), str)
                and bool(str(item["candidate_root"]).strip())
                and consumes_candidate_attempt(item)
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = _trace_dataclass(self)
        payload["selected_outcome"]["digest"] = self.selected_outcome.digest
        payload["idea_acquisition"]["digest"] = self.idea_acquisition.digest
        return canonical_value(payload)


@dataclass(frozen=True, slots=True)
class InitialSearchPoolResult:
    """Qualification-only bootstrap for the first frozen Search profile."""

    active_profile: SearchExecutableProfileV1
    proposals: tuple[CandidateProposalV4, ...]
    open_candidates: tuple[QualifiedSearchCandidateProtocolV1, ...]
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]]
    candidate_root_by_capability: Mapping[str, str]
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]]
    innovations: tuple[InnovationLaneResult, ...]


@dataclass(frozen=True, slots=True)
class RoundCandidateHandoffV1:
    """Identity-bound, outcome-blind inputs for one current-round candidate.

    ``portfolio_candidate`` remains a Router feature record.  The qualified
    execution, candidate-local root, and resource profile are separate
    physical-admission inputs and are carried together only at this explicit
    runtime boundary.
    """

    candidate_id: str
    binding_digest: str
    portfolio_candidate: PortfolioCandidateV2
    qualified_execution: Mapping[str, Any] | None = None
    candidate_root_path: str | None = None
    resource_profile: Mapping[str, Any] | None = None
    portfolio_profile: PortfolioCandidateProfileV2 | None = None
    profile_digest: str | None = None
    source_schema_version: str | None = None
    source_ref: str | None = None
    source_digest: str | None = None
    lineage_identity_digest: str | None = None
    durable_evidence_digests: Mapping[str, str] | None = None

    schema = "recclaw.research-line.round-candidate-handoff.v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or self.candidate_id != self.candidate_id.strip()
        ):
            raise ValueError("candidate_id must be normalized and non-empty")
        try:
            binding_digest = validate_sha256(
                self.binding_digest,
                field_name="binding_digest",
            )
        except (TypeError, ValueError) as error:
            raise ValueError("binding_digest must be a sha256 digest") from error
        object.__setattr__(self, "binding_digest", binding_digest)
        if not isinstance(self.portfolio_candidate, PortfolioCandidateV2):
            raise ValueError("portfolio_candidate must be PortfolioCandidateV2")
        if self.portfolio_candidate.candidate_id != self.candidate_id:
            raise ValueError("handoff candidate_id differs from portfolio candidate")
        if self.qualified_execution is not None:
            if not isinstance(self.qualified_execution, Mapping):
                raise ValueError("qualified_execution must be a mapping")
            object.__setattr__(
                self,
                "qualified_execution",
                canonical_value(dict(self.qualified_execution)),
            )
        root = self.candidate_root_path
        if root is not None:
            if isinstance(root, Path):
                root = str(root.resolve())
            elif isinstance(root, str) and root.strip():
                root = str(Path(root).resolve())
            else:
                raise ValueError("candidate_root_path must be a non-empty path")
            object.__setattr__(self, "candidate_root_path", root)
        if self.resource_profile is not None:
            if not isinstance(self.resource_profile, Mapping):
                raise ValueError("resource_profile must be a mapping")
            object.__setattr__(
                self,
                "resource_profile",
                canonical_value(dict(self.resource_profile)),
            )
        profile = self.portfolio_profile
        source_fields = (
            self.profile_digest,
            self.source_schema_version,
            self.source_ref,
            self.source_digest,
            self.lineage_identity_digest,
        )
        if profile is not None and not isinstance(
            profile,
            PortfolioCandidateProfileV2,
        ):
            raise ValueError(
                "portfolio_profile must be PortfolioCandidateProfileV2"
            )
        if any(value is not None for value in source_fields):
            if profile is None or any(value is None for value in source_fields):
                raise ValueError(
                    "profile/source identity must be complete when supplied"
                )
            try:
                profile_digest = validate_sha256(
                    self.profile_digest,
                    field_name="profile_digest",
                )
                source_digest = validate_sha256(
                    self.source_digest,
                    field_name="source_digest",
                )
                lineage_digest = validate_sha256(
                    self.lineage_identity_digest,
                    field_name="lineage_identity_digest",
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "profile/source identity fields must be sha256 digests"
                ) from error
            if profile_digest != profile.profile_digest:
                raise ValueError("profile_digest differs from portfolio_profile")
            if lineage_digest != lineage_identity_digest(profile.candidate):
                raise ValueError("lineage_identity_digest differs from profile")
            object.__setattr__(self, "profile_digest", profile_digest)
            object.__setattr__(self, "source_digest", source_digest)
            object.__setattr__(self, "lineage_identity_digest", lineage_digest)
            if (
                not isinstance(self.source_schema_version, str)
                or not self.source_schema_version.strip()
                or not isinstance(self.source_ref, str)
                or not self.source_ref.strip()
            ):
                raise ValueError(
                    "source_schema_version and source_ref must be non-empty"
                )
            object.__setattr__(
                self,
                "source_schema_version",
                self.source_schema_version.strip(),
            )
            object.__setattr__(self, "source_ref", self.source_ref.strip())
        elif profile is not None:
            raise ValueError(
                "portfolio_profile requires profile/source identity fields"
            )
        if self.durable_evidence_digests is not None:
            if profile is None or any(value is None for value in source_fields):
                raise ValueError(
                    "durable evidence digests require a complete profile/source identity"
                )
            if not isinstance(self.durable_evidence_digests, Mapping):
                raise ValueError("durable_evidence_digests must be a mapping")
            normalized_evidence_digests: dict[str, str] = {}
            for key, value in self.durable_evidence_digests.items():
                try:
                    normalized_evidence_digests[str(key)] = validate_sha256(
                        value,
                        field_name=f"durable_evidence_digests.{key}",
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "durable_evidence_digests must contain SHA-256 values"
                    ) from error
            object.__setattr__(
                self,
                "durable_evidence_digests",
                canonical_value(normalized_evidence_digests),
            )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value({
                "schema": self.schema,
                "candidate_id": self.candidate_id,
                "binding_digest": self.binding_digest,
                "portfolio_candidate": self.portfolio_candidate.to_dict(),
                "qualified_execution": self.qualified_execution,
                "candidate_root_path": self.candidate_root_path,
                "resource_profile": self.resource_profile,
                **(
                    {
                        "portfolio_profile": self.portfolio_profile.to_dict(),
                        "profile_digest": self.profile_digest,
                        "source_schema_version": self.source_schema_version,
                        "source_ref": self.source_ref,
                        "source_digest": self.source_digest,
                        "lineage_identity_digest": self.lineage_identity_digest,
                        **(
                            {
                                "durable_evidence_digests": self.durable_evidence_digests
                            }
                            if self.durable_evidence_digests is not None
                            else {}
                        ),
                    }
                    if self.portfolio_profile is not None
                    else {}
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedResearchRoundV1:
    """All non-physical preparation that must be reused on round resume."""

    context_digest: str
    profile_ref: str
    profile_digest: str
    producer_outcomes: tuple[ProducerOutcome, ...]
    carryover_outcomes: tuple[ProducerOutcome, ...]
    resolutions: tuple[tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...]
    deferred_innovation_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    deferred_search_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    search_acquisition: ExperimentAcquisitionResultV1 | None
    search_slate: FrozenExperimentSlateV1 | None
    search_bindings: tuple[SearchCandidateBindingV1, ...]
    search_pairs: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...]
    open_search_pairs: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1, QualifiedSearchCandidateProtocolV1],
        ...,
    ]
    innovation: InnovationLaneResult | None
    metric_contract_digest: str
    observation_seed: str
    next_discriminative_test: str
    provider_traces: tuple[Mapping[str, Any], ...]
    confirmation_seed: str | None = None
    portfolio_candidates: tuple[PortfolioCandidateV2, ...] = ()
    candidate_handoffs: tuple[RoundCandidateHandoffV1, ...] = ()
    prebinding_retry: Mapping[str, Any] | None = None

    schema = "recclaw.research-line.prepared-round.v1"

    def __post_init__(self) -> None:
        if self.prebinding_retry is not None:
            object.__setattr__(
                self,
                "prebinding_retry",
                _normalize_prebinding_retry(self.prebinding_retry),
            )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "context_digest": self.context_digest,
            "profile_ref": self.profile_ref,
            "profile_digest": self.profile_digest,
            "producer_outcomes": tuple(
                _trace_value(item) for item in self.producer_outcomes
            ),
            "carryover_outcomes": tuple(
                _trace_value(item) for item in self.carryover_outcomes
            ),
            "resolutions": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.resolutions
            ),
            "deferred_innovation_outcomes": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.deferred_innovation_outcomes
            ),
            "deferred_search_outcomes": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.deferred_search_outcomes
            ),
            "search_acquisition": _trace_value(self.search_acquisition),
            "search_slate": _trace_value(self.search_slate),
            "search_bindings": tuple(
                _trace_value(item) for item in self.search_bindings
            ),
            "search_pairs": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                }
                for outcome, resolution in self.search_pairs
            ),
            "open_search_pairs": tuple(
                {
                    "outcome": _trace_value(outcome),
                    "resolution": _trace_value(resolution),
                    "candidate": _trace_value(candidate),
                }
                for outcome, resolution, candidate in self.open_search_pairs
            ),
            "innovation": _trace_value(self.innovation),
            "metric_contract_digest": self.metric_contract_digest,
            "observation_seed": self.observation_seed,
            "next_discriminative_test": self.next_discriminative_test,
            "confirmation_seed": getattr(self, "confirmation_seed", None),
            "provider_traces": _trace_value(self.provider_traces),
            "portfolio_candidates": tuple(
                item.to_dict() for item in self.portfolio_candidates
            ),
        }
        prebinding_retry = getattr(self, "prebinding_retry", None)
        if prebinding_retry is not None:
            payload["prebinding_retry"] = prebinding_retry
            payload["prebinding_retry_digest"] = sha256_digest(prebinding_retry)
        handoffs = tuple(getattr(self, "candidate_handoffs", ()) or ())
        if handoffs:
            payload["candidate_handoffs"] = tuple(
                item.to_dict() for item in handoffs
            )
        return canonical_value(payload)


_RECOVERABLE_RESOURCE_FAILURE_SCOPES = frozenset(
    {"WORKER_TRANSIENT", "SHARED_INFRASTRUCTURE", "RECOVERY"}
)


def _reclassify_innovation_progressing_timeout(
    innovation: InnovationLaneResult,
    resource_probe_parent: Path,
) -> InnovationLaneResult | None:
    """Close a persisted measured resource stop without rerunning a candidate."""

    if not isinstance(innovation.resource_profile, Mapping):
        return None
    semantic_digests = tuple(
        dict.fromkeys(
            str(item["mechanism_semantics_digest"])
            for item in reversed(innovation.attempts)
            if isinstance(item, Mapping)
            and isinstance(item.get("mechanism_semantics_digest"), str)
            and len(str(item["mechanism_semantics_digest"])) == 64
        )
    )
    probe_roots: list[Path] = [
        Path(item["resource_probe_root"])
        for item in innovation.attempts
        if isinstance(item, Mapping)
        and isinstance(item.get("resource_probe_root"), str)
        and Path(item["resource_probe_root"]).is_dir()
    ]
    for digest in semantic_digests:
        probe_roots.extend(
            path
            for path in resource_probe_parent.rglob(
                f"probe-{digest[:16]}"
            )
            if path.is_dir()
        )
    profile = reclassify_progressing_timeout_profile(
        innovation.resource_profile,
        probe_roots=tuple(dict.fromkeys(probe_roots)),
    )
    if profile is None:
        return None
    prior_quality_admission = innovation.quality_admission
    if not isinstance(prior_quality_admission, Mapping):
        return None
    quality_admission = canonical_value(
        {
            **dict(prior_quality_admission),
            "status": str(profile.get("status") or "RESOURCE_PROBE_FAILED"),
        }
    )
    failure = _resource_admission_failure(
        resource_profile=profile,
        quality_admission=quality_admission,
    )
    attempts: list[Mapping[str, Any]] = []
    updated = False
    for item in innovation.attempts:
        item_mapping = dict(item)
        item_profile = item_mapping.get("resource_profile")
        item_diagnostic = (
            item_profile.get("resource_probe_diagnostic")
            if isinstance(item_profile, Mapping)
            else None
        )
        if (
            isinstance(item_profile, Mapping)
            and isinstance(item_diagnostic, Mapping)
            and str(item_diagnostic.get("error_type") or "").upper()
            in {
                "PROBE_TIMEOUT",
                "RESOURCE_CENSORED",
                "RESOURCE_PROBE_INTERRUPTED_RECOVERY",
                "RESOURCE_PROBE_LEGACY_INTERVAL_UNAVAILABLE",
            }
            and item_profile.get("candidate_ref")
            == profile.get("candidate_ref")
        ):
            item_mapping.update(
                {
                    "resource_profile": profile,
                    "failure": failure,
                }
            )
            updated = True
        attempts.append(canonical_value(item_mapping))
    if not updated:
        return None
    return replace(
        innovation,
        attempts=tuple(attempts),
        resource_profile=profile,
        quality_admission=quality_admission,
    )


def _innovation_resource_failure_scope(
    innovation: InnovationLaneResult | None,
) -> str | None:
    """Return the probe-owned failure scope for a blocked Innovation candidate."""

    if innovation is None or innovation.activation_ready:
        return None
    sources: list[Mapping[str, Any]] = []
    # Historical infrastructure failures remain evidence, not the current
    # failure scope. A fresh cost measurement or implementation failure must
    # not inherit an old external interruption and retry forever.
    for attempt in innovation.attempts[-1:]:
        if not isinstance(attempt, Mapping):
            continue
        failure = attempt.get("failure")
        if isinstance(failure, Mapping):
            sources.append(failure)
        profile = attempt.get("resource_profile")
        if isinstance(profile, Mapping):
            sources.append(profile)
    if isinstance(innovation.resource_profile, Mapping):
        sources.append(innovation.resource_profile)
    for source in sources:
        diagnostic = source.get("resource_probe_diagnostic")
        raw_scope = source.get("failure_scope")
        if raw_scope is None and isinstance(diagnostic, Mapping):
            raw_scope = diagnostic.get("failure_scope")
        if raw_scope is None:
            detail = source.get("detail")
            if isinstance(detail, Mapping):
                raw_scope = detail.get("failure_scope")
        scope = str(raw_scope or "").upper()
        if scope in {
            "CANDIDATE_LOCAL",
            "LINEAGE_COMPUTE_PATTERN",
            "WORKER_TRANSIENT",
            "SHARED_INFRASTRUCTURE",
            "RECOVERY",
        }:
            return scope
    return None


def _prepared_has_recoverable_resource_failure(
    prepared: PreparedResearchRoundV1,
) -> bool:
    """Whether preparation stopped on infrastructure, not candidate merit."""

    return (
        not prepared.search_bindings
        and _innovation_resource_failure_scope(prepared.innovation)
        in _RECOVERABLE_RESOURCE_FAILURE_SCOPES
    )


def _prepared_has_untried_candidate_local_innovation(
    prepared: PreparedResearchRoundV1,
    *,
    attempt_budget: int | None = None,
) -> bool:
    """Whether a durable prepared slate still owns runnable Innovation work."""

    innovation = prepared.innovation
    if innovation is None or innovation.activation_ready or prepared.search_bindings:
        return False
    candidate_attempts = tuple(
        item
        for item in innovation.attempts
        if isinstance(item, Mapping)
        and isinstance(item.get("candidate_root"), str)
        and bool(str(item["candidate_root"]).strip())
    )
    if not candidate_attempts:
        return False
    last_failure = candidate_attempts[-1].get("failure")
    if (
        not isinstance(last_failure, Mapping)
        or _innovation_resource_failure_scope(innovation)
        in _RECOVERABLE_RESOURCE_FAILURE_SCOPES
        or (
            last_failure.get("failure_scope") != "CANDIDATE_LOCAL"
            # Exhausted implementation repair (including a no-op) also
            # retires this candidate, not the rest of its frozen slate.
            and last_failure.get("failure_class") != "IMPLEMENTATION"
        )
    ):
        return False
    if (
        attempt_budget is not None
        and innovation.candidate_attempt_count >= attempt_budget
    ):
        return False
    consumed_spec_digests = {
        _innovation_attempt_proposal_digest(item)
        for item in innovation.attempts
        if isinstance(item, Mapping)
        and isinstance(_innovation_attempt_proposal_digest(item), str)
    }
    return any(
        resolution is not None
        and resolution.resolution
        is CapabilityResolutionResultV1.INNOVATION_REQUIRED
        and outcome.spec is not None
        and outcome.spec.digest not in consumed_spec_digests
        for outcome, resolution in prepared.resolutions
    )


@dataclass(frozen=True, slots=True)
class RoundAttemptV1:
    """One immutable engineering/scientific attempt inside a Research round.

    The existing ``search_acquisition``/``candidate_run`` fields on
    :class:`ResearchRoundResult` remain the compatibility projection for the
    metric-bearing attempt.  This record is the lossless in-round view used by
    the scheduler and campaign persistence layer.
    """

    attempt_index: int
    acquisition: ExperimentAcquisitionResultV1 | None
    binding: SearchCandidateBindingV1
    execution_recipe: Mapping[str, Any]
    candidate_run: Mapping[str, Any]
    engineering_disposition: str
    search_space_attestation: Mapping[str, Any] | None = None
    failure_scope: str | None = None
    failure_detail: Mapping[str, Any] | None = None
    observation_ref: str | None = None
    observation_digest: str | None = None
    resource_prediction: Mapping[str, Any] | None = None
    assigned_deadline_seconds: float | None = None
    live_health_decisions: tuple[Mapping[str, Any], ...] = ()
    diagnostic_feedback: Mapping[str, Any] | None = None
    diagnostic_successor_context: Mapping[str, Any] | None = None
    diagnostic_policy_successor: Mapping[str, Any] | None = None
    diagnostic_search_memory_snapshot: Mapping[str, Any] | None = None
    evidence_pre_adjudication: Mapping[str, Any] | None = None
    evidence_post_adjudication: Mapping[str, Any] | None = None

    schema = "recclaw.research-line.round-attempt.v1"

    def __post_init__(self) -> None:
        if isinstance(self.attempt_index, bool) or self.attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        if not isinstance(self.binding, SearchCandidateBindingV1):
            raise ValueError("binding must be SearchCandidateBindingV1")
        if not isinstance(self.execution_recipe, Mapping):
            raise ValueError("execution_recipe must be a mapping")
        if not isinstance(self.candidate_run, Mapping):
            raise ValueError("candidate_run must be a mapping")
        if not isinstance(self.engineering_disposition, str) or not self.engineering_disposition:
            raise ValueError("engineering_disposition must be non-empty")
        object.__setattr__(self, "execution_recipe", canonical_value(dict(self.execution_recipe)))
        object.__setattr__(self, "candidate_run", canonical_value(dict(self.candidate_run)))
        if self.search_space_attestation is not None:
            object.__setattr__(
                self,
                "search_space_attestation",
                canonical_value(dict(self.search_space_attestation)),
            )
        if self.failure_detail is not None:
            object.__setattr__(
                self,
                "failure_detail",
                canonical_value(dict(self.failure_detail)),
            )
        if self.resource_prediction is not None:
            object.__setattr__(
                self,
                "resource_prediction",
                canonical_value(dict(self.resource_prediction)),
            )
        object.__setattr__(
            self,
            "live_health_decisions",
            tuple(canonical_value(dict(item)) for item in self.live_health_decisions),
        )
        if self.diagnostic_feedback is not None:
            object.__setattr__(
                self,
                "diagnostic_feedback",
                canonical_value(dict(self.diagnostic_feedback)),
            )
        for field_name in (
            "diagnostic_successor_context",
            "diagnostic_policy_successor",
            "diagnostic_search_memory_snapshot",
            "evidence_pre_adjudication",
            "evidence_post_adjudication",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    canonical_value(dict(value)),
                )

    @property
    def candidate_id(self) -> str:
        return self.binding.proposal.candidate_id

    @property
    def metric_bearing(self) -> bool:
        return self.engineering_disposition == "METRIC_BEARING"

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
                "schema": self.schema,
                "attempt_index": self.attempt_index,
                "acquisition": _trace_value(self.acquisition),
                "binding": self.binding.canonical_dict(),
                "candidate_id": self.candidate_id,
                "execution_recipe": self.execution_recipe,
                "candidate_run": self.candidate_run,
                "engineering_disposition": self.engineering_disposition,
                "failure_scope": self.failure_scope,
                "failure_detail": self.failure_detail,
                "observation_ref": self.observation_ref,
                "observation_digest": self.observation_digest,
                "resource_prediction": self.resource_prediction,
                "assigned_deadline_seconds": self.assigned_deadline_seconds,
                "live_health_decisions": self.live_health_decisions,
                "diagnostic_feedback": self.diagnostic_feedback,
                "diagnostic_successor_context": self.diagnostic_successor_context,
                "diagnostic_policy_successor": self.diagnostic_policy_successor,
                "diagnostic_search_memory_snapshot": (
                    self.diagnostic_search_memory_snapshot
                ),
            }
        if self.evidence_pre_adjudication is not None:
            payload["evidence_pre_adjudication"] = self.evidence_pre_adjudication
        if self.evidence_post_adjudication is not None:
            payload["evidence_post_adjudication"] = self.evidence_post_adjudication
        return canonical_value(payload)


@dataclass(frozen=True, slots=True)
class ResearchRoundResult:
    """Queryable Context-to-feedback trace for one experiment opportunity."""

    context: ResearchContext
    active_profile: SearchExecutableProfileV1
    producer_outcomes: tuple[ProducerOutcome, ...]
    carryover_outcomes: tuple[ProducerOutcome, ...]
    resolutions: tuple[tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...]
    deferred_innovation_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    deferred_search_outcomes: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ]
    search_acquisition: ExperimentAcquisitionResultV1 | None
    innovation: InnovationLaneResult | None
    selected_outcome: ProducerOutcome | None
    execution_recipe: Mapping[str, Any] | None
    candidate_run: Mapping[str, Any] | None
    interpretation: EpisodeInterpretation | MissingSearchInterpretation | None
    provider_traces: tuple[Mapping[str, Any], ...]
    meta_research: MetaResearchResult | None = None
    attempts: tuple[RoundAttemptV1, ...] = ()
    metric_bearing_attempt_index: int | None = None
    attempt_scheduler_enabled: bool = False
    incomplete_reason: str | None = None
    prepared: PreparedResearchRoundV1 | None = None
    evidence_pre_trace: tuple[Mapping[str, Any], ...] = ()
    evidence_post_trace: tuple[Mapping[str, Any], ...] = ()

    @property
    def successor_context(self) -> ResearchContext:
        if self.interpretation is None:
            raise ValueError("round without closed execution feedback has no successor Context")
        return self.interpretation.successor_context

    @property
    def candidate_attempt_summaries(self) -> tuple[Mapping[str, Any], ...]:
        """Return durable candidate outcomes aggregated across generations."""

        feedback = getattr(self.interpretation, "feedback_projection", None)
        if not isinstance(feedback, Mapping) or feedback.get("schema") != (
            "recclaw.research-line.candidate-no-metric-feedback.v1"
        ):
            return ()
        rows = feedback.get("candidate_failures", ())
        if not isinstance(rows, (tuple, list)):
            return ()
        return tuple(
            canonical_value(dict(item))
            for item in rows
            if isinstance(item, Mapping)
        )

    @property
    def attempt_count(self) -> int:
        feedback = getattr(self.interpretation, "feedback_projection", None)
        if isinstance(feedback, Mapping):
            slot_attempt_count = feedback.get("slot_attempt_count")
            if (
                isinstance(slot_attempt_count, int)
                and not isinstance(slot_attempt_count, bool)
                and slot_attempt_count >= 0
            ):
                return slot_attempt_count
        summaries = self.candidate_attempt_summaries
        return len(summaries) if summaries else len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        payload = _trace_dataclass(self)
        payload["context"].update(
            context_ref=self.context.context_ref,
            context_digest=self.context.digest,
        )
        for projected, outcome in zip(
            payload["producer_outcomes"], self.producer_outcomes, strict=True
        ):
            projected["digest"] = outcome.digest
        for projected, outcome in zip(
            payload["carryover_outcomes"], self.carryover_outcomes, strict=True
        ):
            projected["digest"] = outcome.digest
        if self.search_acquisition is not None:
            payload["search_acquisition"]["route_trace"]["digest"] = (
                self.search_acquisition.route_trace.digest
            )
            binding = self.search_acquisition.selected_binding
            if binding is not None:
                payload["search_acquisition"]["selected_binding"]["digest"] = (
                    binding.digest
                )
        payload["selected_outcome_digest"] = (
            self.selected_outcome.digest if self.selected_outcome is not None else None
        )
        candidate_attempt_summaries = self.candidate_attempt_summaries
        if candidate_attempt_summaries:
            payload["candidate_attempt_summaries"] = candidate_attempt_summaries
            payload["attempt_digests"] = tuple(
                sha256_digest(item) for item in candidate_attempt_summaries
            )
        else:
            payload["attempt_digests"] = tuple(item.digest for item in self.attempts)
        payload["metric_bearing_attempt_index"] = self.metric_bearing_attempt_index
        payload["attempt_scheduler_enabled"] = self.attempt_scheduler_enabled
        payload["incomplete_reason"] = self.incomplete_reason
        payload["prepared_digest"] = self.prepared.digest if self.prepared else None
        if self.evidence_pre_trace:
            payload["evidence_pre_trace"] = self.evidence_pre_trace
        else:
            payload.pop("evidence_pre_trace", None)
        if self.evidence_post_trace:
            payload["evidence_post_trace"] = self.evidence_post_trace
        else:
            payload.pop("evidence_post_trace", None)
        return canonical_value(payload)

    @property
    def has_metric_bearing_attempt(self) -> bool:
        if self.metric_bearing_attempt_index is not None:
            return True
        # Compatibility projection for callers that still use the old
        # single-attempt shape and do not populate ``attempts``.
        return (
            not self.attempt_scheduler_enabled
            and self.candidate_run is not None
            and self.interpretation is not None
            and getattr(self.interpretation, "episode", None) is not None
        )


def _trace_starts(*owners: Any) -> tuple[tuple[Any, int], ...]:
    starts: list[tuple[Any, int]] = []
    seen: set[int] = set()
    for owner in owners:
        if owner is None or id(owner) in seen:
            continue
        traces = getattr(owner, "call_traces", None)
        if isinstance(traces, (tuple, list)):
            starts.append((owner, len(traces)))
            seen.add(id(owner))
    return tuple(starts)


def _new_provider_traces(
    starts: Sequence[tuple[Any, int]],
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for owner, start in starts:
        traces = getattr(owner, "call_traces", ())
        if not isinstance(traces, (tuple, list)):
            continue
        for trace in traces[start:]:
            if isinstance(trace, Mapping):
                result.append(canonical_value(dict(trace)))
    return tuple(result)


def bindings_for_context(
    context: ResearchContext,
    *,
    active_profile: SearchExecutableProfileV1,
    implementation_requirements: Sequence[str],
    compatibility_requirements: Sequence[str],
) -> dict[str, Any]:
    """Build Producer bindings from the actually active profile, fixed or fresh."""

    return canonical_value(
        {
            "context_ref": context.context_ref,
            "context_digest": context.digest,
            "protocol_ref": context.protocol_ref,
            "protocol_digest": context.protocol_digest,
            "current_profile_ref": context.active_profile_ref,
            "current_profile_digest": context.active_profile_digest,
            "current_capability_semantics": tuple(
                entry.semantic_identity_digest for entry in active_profile.entries
            ),
            "implementation_requirements": tuple(implementation_requirements),
            "compatibility_requirements": tuple(compatibility_requirements),
        }
    )


def resolver_environment_for_profile(
    profile: SearchExecutableProfileV1,
    *,
    available_dependencies: Sequence[str],
    budget_limits: Mapping[str, int],
    protocol_requirements: Sequence[str],
) -> dict[str, Any]:
    """Expose every active entry, including admitted capabilities, to Resolver."""

    return canonical_value(
        {
            "current_profile_ref": profile.profile_ref,
            "current_profile_digest": profile.profile_digest,
            "current_capabilities": tuple(
                {
                    "capability_ref": entry.capability_ref,
                    "capability_digest": entry.capability_digest,
                    "semantics_digest": entry.semantic_identity_digest,
                }
                for entry in profile.entries
            ),
            "protocol_ref": profile.protocol_ref,
            "protocol_digest": profile.protocol_digest,
            "protocol_requirements": tuple(protocol_requirements),
            "available_dependencies": tuple(available_dependencies),
            "budget_limits": dict(budget_limits),
        }
    )


def _carryover_outcomes(
    context: ResearchContext,
    proposals: Sequence[CandidateProposalV4],
    bindings: Mapping[str, Any],
) -> tuple[ProducerOutcome, ...]:
    outcomes = []
    for proposal in proposals:
        spec, facts = project_candidate_proposal_v4(proposal, bindings=bindings)
        outcomes.append(
            ProducerOutcome(
                producer_role=proposal.producer_role,
                context_ref=context.context_ref,
                context_digest=context.digest,
                spec=spec,
                resolution_facts=facts,
                source_proposal=proposal,
            )
        )
    return tuple(outcomes)


def _carryover_open_outcomes(
    context: ResearchContext,
    candidates: Sequence[QualifiedSearchCandidateProtocolV1],
) -> tuple[tuple[ProducerOutcome, QualifiedSearchCandidateProtocolV1], ...]:
    """Rebind admitted OpenSpec lineage to the fresh Context for Search."""

    result = []
    for candidate in candidates:
        source = candidate.spec
        spec = replace(
            source,
            context_ref=context.context_ref,
            context_digest=context.digest,
            current_profile_ref=context.active_profile_ref,
            current_profile_digest=context.active_profile_digest,
            current_profile_expressibility_claim=(
                CurrentProfileExpressibilityV1.EXPRESSIBLE
            ),
        )
        outcome = ProducerOutcome(
            producer_role=spec.producer_role,
            context_ref=context.context_ref,
            context_digest=context.digest,
            spec=spec,
            resolution_facts={
                "requested_current_semantics_digest": (
                    candidate.semantic_identity_digest
                ),
                "capability_diff": (),
                "high_change_dimensions": (),
                "required_dependencies": (),
                "required_budget": {},
            },
            source_proposal=None,
        )
        result.append((outcome, candidate))
    return tuple(result)


def _source_files(response: Mapping[str, Any]) -> dict[str, str]:
    files = response.get("files", ())
    if not isinstance(files, (tuple, list)):
        return {}
    return {
        str(item["path"]): str(item["content"])
        for item in files
        if isinstance(item, Mapping) and "path" in item and "content" in item
    }


def _materialization_failure(error: InnovationSpineError) -> dict[str, Any]:
    return canonical_value(
        {
            "stage": "SCHEMA",
            "failure_class": error.failure_class,
            "reason_code": error.reason_code,
            "message": str(error),
            **(
                dict(error.details)
                if getattr(error, "details", None) is not None
                else {}
            ),
        }
    )


def _qualified_execution(
    policy: SharedImplementerPolicy,
    materialized: MaterializedCandidate,
    capability: QualifiedCapabilityV1,
) -> dict[str, Any]:
    contract = policy.execution_contract
    if not isinstance(contract, Mapping):
        raise ValueError("admitted capability requires an explicit execution_contract")
    required = {"capability_family", "model", "base_model_config", "config"}
    if set(contract) != required:
        raise ValueError(
            "execution_contract must contain capability_family, model, base_model_config, and config"
        )
    source_path = (
        materialized.package.executable_entrypoint.split(":", 1)[0].replace(".", "/")
        + ".py"
    )
    manifest = materialized.implementation_receipt["written_files"]
    source = next((row for row in manifest if row["path"] == source_path), None)
    if source is None:
        raise ValueError("candidate entrypoint source is absent from its package manifest")
    behavior_source_rows = tuple(
        sorted(
            (
                {
                    "sha256": str(row["sha256"]),
                    "size_bytes": int(row["size_bytes"]),
                }
                for row in manifest
                if isinstance(row, Mapping)
                and str(row.get("path", "")).startswith("recclaw_ext/")
                and str(row.get("path", "")).endswith(".py")
            ),
            key=lambda row: (row["sha256"], row["size_bytes"]),
        )
    )
    if not behavior_source_rows:
        raise ValueError("candidate package contains no behavior source files")
    candidate_source_content_digest = sha256_digest(
        {
            "schema": "recclaw.candidate-behavior-source-content.v1",
            "files": behavior_source_rows,
        }
    )
    return canonical_value(
        {
            **dict(contract),
            "entrypoint_source_sha256": source["sha256"],
            "candidate_package_ref": capability.candidate_package_ref,
            "candidate_package_digest": capability.candidate_package_digest,
            "candidate_root_ref": materialized.package.candidate_root_ref,
            "candidate_root_digest": materialized.package.candidate_root_digest,
            "candidate_source_tree_digest": capability.source_tree_digest,
            "candidate_source_content_digest": candidate_source_content_digest,
        }
    )


def _semantic_duplicate_preimplementation_admission(
    *,
    spec: OpenResearchSpecV1,
    semantic_identity_digest: str,
) -> dict[str, Any]:
    """Record an outcome-blind duplicate without implementation or probing."""

    return canonical_value(
        {
            "admitted": False,
            "candidate_ref": None,
            "candidate_package_digest": None,
            "effect_fields_consumed": [],
            "feature_evidence": {"semantic_duplicate": True},
            "held_out_reads": 0,
            "mechanism_effect_updates": 0,
            "outcome_fields_consumed": [],
            "phase": "POST_PROPOSAL_PRE_IMPLEMENTATION",
            "resource_profile_digest": None,
            "schema": "recclaw.research-line.innovation-quality-admission.v1",
            "semantic_identity_digest": semantic_identity_digest,
            "spec_digest": spec.digest,
            "status": "SEMANTIC_DUPLICATE",
        }
    )


def _resource_probe_recipe(
    *,
    current_profile: SearchExecutableProfileV1,
    capability: QualifiedCapabilityV1,
    qualified_execution: Mapping[str, Any],
    entrypoint: str,
    mechanism_id: str,
    mechanism_semantics_digest: str,
    evaluator: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """Bind the disposable probe to the exact realized package and protocol.

    The candidate is not in the active profile until resource admission, so
    the probe records the active profile as its predecessor boundary while
    binding the new capability/package/source identities explicitly.
    """

    execution_config = qualified_execution.get("config")
    contract_dataset = (
        execution_config.get("dataset", COMMON_DATASET)
        if isinstance(execution_config, Mapping)
        else COMMON_DATASET
    )
    recipe = canonical_value(
        {
            **dict(qualified_execution),
            "capability_ref": capability.capability_id,
            "capability_digest": capability.digest,
            "profile_ref": current_profile.profile_ref,
            "profile_digest": current_profile.profile_digest,
            "entrypoint": entrypoint,
            "mechanism_id": mechanism_id,
            "mechanism_semantics_digest": mechanism_semantics_digest,
            "dataset": contract_dataset,
            "split": split,
            "evaluator": canonical_value(dict(evaluator)),
            "execution_role": "CANDIDATE",
        }
    )
    validate_execution_recipe(recipe)
    return recipe


def _acquire_innovation_spec(
    candidates: Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]],
    *,
    research_policy: VersionedResearchPolicyV1,
    search_space_adapter: SearchSpaceAdapter,
    frozen_profile_ref: Mapping[str, Any],
    research_context: ResearchContext | None = None,
    enforce_unbound_confirmation: bool = True,
    outcome_selector: (
        Callable[
            [Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]]],
            str | None,
        ]
        | None
    ) = None,
) -> tuple[ProducerOutcome, CapabilityResolutionV1, IdeaAcquisitionResult] | None:
    """Choose one executable research opportunity after mechanical checks.

    Original-matched runs retain their injected controller selector.  The
    common Research Line keeps ordinary and high-innovation Producers as peer
    opportunity sources: it covers available roles before following the
    effect-learned Producer allocation.  It never parses or hand-ranks the
    proposed mechanism itself.
    """

    mechanically_feasible = tuple(
        (outcome, resolution)
        for outcome, resolution in candidates
        if outcome.spec is not None
    )
    if not mechanically_feasible:
        return None

    parameters = dict(research_policy.acquisition_parameters)
    if research_context is not None:
        global_memory = research_context.scientific_memory.get(
            "global_memory", {}
        )
        raw_queue = (
            global_memory.get("task_queue")
            if isinstance(global_memory, Mapping)
            else None
        )
        stale_task = parameters.get("research_task_record")
        stale_task_id = (
            stale_task.get("task_id")
            if isinstance(stale_task, Mapping)
            else None
        )
        durable_task = (
            ResearchTaskQueueV2.from_dict(raw_queue).get(stale_task_id)
            if isinstance(stale_task_id, str)
            else None
        )
        if durable_task is not None and durable_task.status in {
            ResearchTaskStatusV2.SATISFIED,
            ResearchTaskStatusV2.CLOSED,
        }:
            parameters.pop("research_task_record", None)

    task_record = (
        parameters.get("research_task_record")
        if isinstance(parameters.get("research_task_record"), Mapping)
        else None
    )
    task_metadata = (
        task_record.get("metadata")
        if isinstance(task_record, Mapping)
        and isinstance(task_record.get("metadata"), Mapping)
        else {}
    )
    exact_confirmation_required = bool(
        enforce_unbound_confirmation
        and isinstance(task_record, Mapping)
        and task_record.get("operation") in {"MATCHED_CONTROL", "MECHANISM_OFF"}
        and task_record.get("status") in {"PENDING", "ACTIVE"}
        and task_metadata.get("execution_state")
        == "AWAITING_CANDIDATE_BINDING"
    )
    eligible: list[tuple[ProducerOutcome, CapabilityResolutionV1]] = []
    for outcome, resolution in mechanically_feasible:
        assert outcome.spec is not None
        if exact_confirmation_required:
            confirmation = _resolve_confirmation(
                search_space_adapter,
                pending_task={
                    "task_id": task_record.get("task_id"),
                    "task_record": task_record,
                },
                primary_binding=outcome,
                next_discriminative_test=str(
                    parameters.get("next_discriminative_task", "")
                ),
            )
            if confirmation.kind is not ConfirmationResolutionKindV1.EXACT_BINDING:
                continue
        if _innovation_identity_digests(
            outcome,
            search_space_adapter=search_space_adapter,
            frozen_profile_ref=frozen_profile_ref,
            research_context=research_context,
        ) is None:
            continue
        eligible.append((outcome, resolution))
    if not eligible:
        return None

    if outcome_selector is not None:
        selected_digest = outcome_selector(tuple(eligible))
        selected = tuple(
            pair for pair in eligible if pair[0].digest == selected_digest
        )
        if len(selected) != 1:
            return None
        selected_outcome, selected_resolution = selected[0]
        ranked_pairs = selected
    else:
        selected_digest, ranked_digests = _select_common_producer_opportunity(
            eligible,
            research_policy=research_policy,
            research_context=research_context,
        )
        ranked_by_digest = {
            outcome.digest: (outcome, resolution)
            for outcome, resolution in eligible
        }
        ranked_pairs = tuple(
            ranked_by_digest[digest]
            for digest in ranked_digests
        )
        selected_outcome, selected_resolution = ranked_by_digest[
            selected_digest
        ]
    assert selected_outcome.spec is not None
    acquisition = IdeaAcquisitionResult(
        pool_digest=sha256_digest(
            tuple(
                {
                    "producer_outcome_digest": outcome.digest,
                    "resolution_digest": resolution.digest,
                }
                for outcome, resolution in eligible
            )
        ),
        ranked_spec_digests=tuple(
            outcome.spec.digest
            for outcome, _resolution in ranked_pairs
            if outcome.spec is not None
        ),
        selected_spec_digest=selected_outcome.spec.digest,
        score_by_spec_digest={},
        policy_digest=research_policy.digest,
    )
    return selected_outcome, selected_resolution, acquisition


def _select_common_producer_opportunity(
    candidates: Sequence[tuple[ProducerOutcome, Any]],
    *,
    research_policy: VersionedResearchPolicyV1,
    research_context: ResearchContext | None,
) -> tuple[str, tuple[str, ...]]:
    """Schedule one peer Producer without inspecting mechanism semantics."""

    acquisition_parameters = dict(research_policy.acquisition_parameters)
    raw_effect_rates = acquisition_parameters.get("producer_useful_rates")
    effect_rates = (
        dict(raw_effect_rates)
        if isinstance(raw_effect_rates, (Mapping, tuple, list))
        else None
    )
    allocation = (
        {role: float(effect_rates[role]) for role in DISCOVERY_PRODUCERS}
        if isinstance(effect_rates, Mapping)
        and set(effect_rates) == set(DISCOVERY_PRODUCERS)
        else dict(research_policy.producer_token_allocation)
    )
    parent_ranked = tuple(
        pair
        for _index, pair in sorted(
            enumerate(candidates),
            key=lambda item: (
                -float(allocation.get(item[1][0].producer_role, 0.0)),
                item[0],
            ),
        )
    )
    opportunity = acquire_producer_opportunity(
        parent_ranked_candidate_ids=tuple(
            outcome.digest for outcome, _resolution in parent_ranked
        ),
        parent_decision_digest=sha256_digest(
            {
                "research_policy_digest": research_policy.digest,
                "ranked_outcome_digests": tuple(
                    outcome.digest
                    for outcome, _resolution in parent_ranked
                ),
            }
        ),
        producer_role_by_candidate_id={
            outcome.digest: outcome.producer_role
            for outcome, _resolution in parent_ranked
        },
        prior_selected_roles=_prior_producer_opportunity_roles(
            research_context
        ),
    )
    return (
        opportunity.selected_candidate_id,
        opportunity.ranked_candidate_ids,
    )


def _prior_producer_opportunity_roles(
    research_context: ResearchContext | None,
) -> tuple[str, ...]:
    """Recover one role entry per previously selected research candidate."""

    if research_context is None:
        return ()
    global_memory = research_context.scientific_memory.get(
        "global_memory", {}
    )
    raw_state = research_context.scientific_memory.get(
        "producer_opportunity_state",
        (
            global_memory.get("producer_opportunity_state")
            if isinstance(global_memory, Mapping)
            else None
        ),
    )
    if isinstance(raw_state, Mapping):
        count = raw_state.get("opportunity_count")
        raw_counts = raw_state.get("role_counts")
        recent = tuple(raw_state.get("recent_roles", ()))
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            and isinstance(raw_counts, Mapping)
            and all(role in DISCOVERY_PRODUCERS for role in recent)
        ):
            counts = {
                role: int(raw_counts.get(role, 0))
                for role in DISCOVERY_PRODUCERS
            }
            recent_counts = {
                role: recent.count(role) for role in DISCOVERY_PRODUCERS
            }
            prefix = tuple(
                role
                for role in DISCOVERY_PRODUCERS
                for _ in range(counts[role] - recent_counts[role])
            )
            reconstructed = (*prefix, *recent)
            if len(reconstructed) == count:
                return reconstructed
    records: dict[tuple[Any, ...], tuple[tuple[int, int, int], str]] = {}

    def remember(
        row: Mapping[str, Any],
        *,
        fallback_role: str | None = None,
    ) -> None:
        if row.get("execution_lane") == "AUXILIARY_VERIFICATION":
            return
        role = row.get("producer_role", fallback_role)
        if role not in DISCOVERY_PRODUCERS:
            return
        round_index = int(row.get("round_index", 0) or 0)
        generation = int(row.get("discovery_generation", 0) or 0)
        attempt = int(row.get("attempt", 0) or 0)
        identity = next(
            (
                row.get(field_name)
                for field_name in (
                    "candidate_id",
                    "spec_digest",
                    "outcome_digest",
                )
                if isinstance(row.get(field_name), str)
                and row.get(field_name)
            ),
            sha256_digest(row),
        )
        key = (round_index, str(identity), str(role))
        records.setdefault(
            key,
            ((round_index, generation, attempt), str(role)),
        )

    for row in _round_attempt_memory(research_context):
        remember(row)
    for source in (research_context.scientific_memory, global_memory):
        observations = (
            source.get("executed_observations", ())
            if isinstance(source, Mapping)
            else ()
        )
        if isinstance(observations, (tuple, list)):
            for row in observations:
                if isinstance(row, Mapping):
                    remember(row)
    role_memory = research_context.scientific_memory.get("by_role", {})
    if isinstance(role_memory, Mapping):
        for role in DISCOVERY_PRODUCERS:
            memory = role_memory.get(role)
            history = (
                memory.get("execution_history", ())
                if isinstance(memory, Mapping)
                else ()
            )
            if isinstance(history, (tuple, list)):
                for row in history:
                    if isinstance(row, Mapping):
                        remember(row, fallback_role=role)
    return tuple(
        role
        for _order, role in sorted(
            records.values(),
            key=lambda item: item[0],
        )
    )


def _innovation_axis_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (tuple, list)):
        return ()
    result: list[str] = []
    for item in value:
        axis = canonical_mechanism_axis(item)
        if axis is not None and axis not in result:
            result.append(axis)
    return tuple(result)


def _innovation_complete_values(value: Any) -> tuple[str, ...]:
    """Preserve raw BL-ICF slot/dimension identities for candidate context."""

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
            if isinstance(item, str) and item.strip():
                text = item.strip()
                if text not in result:
                    result.append(text)
        return tuple(result)
    if not isinstance(value, (tuple, list)):
        return ()
    result: list[str] = []
    for item in value:
        for identity in _innovation_complete_values(item):
            if identity not in result:
                result.append(identity)
    return tuple(result)


def _innovation_program_values(
    program: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Prefer raw changed dimensions over a policy-axis projection."""

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
        identities = _innovation_complete_values(payload.get(key, ()))
        for identity in identities:
            if identity not in declared:
                declared.append(identity)
            if identity not in MECHANISM_AXIS_UNIVERSE_V1 and identity not in raw:
                raw.append(identity)
    return tuple(raw or declared)


def _innovation_non_policy_values(value: Any) -> tuple[str, ...]:
    return tuple(
        identity
        for identity in _innovation_complete_values(value)
        if identity not in MECHANISM_AXIS_UNIVERSE_V1
    )


def _innovation_axis_footprint(outcome: ProducerOutcome) -> tuple[str, ...]:
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
        for identity in _innovation_program_values(program):
            if identity not in program_values:
                program_values.append(identity)

    resolution_values: list[str] = []
    for key in (
        "mechanism_axis_footprint",
        "changed_axis_footprint",
        "changed_axes",
        "changed_dimensions",
    ):
        for identity in _innovation_complete_values(resolution.get(key, ())):
            if identity not in resolution_values:
                resolution_values.append(identity)
    partial_raw_values = tuple(
        dict.fromkeys(
            (
                *_innovation_non_policy_values(resolution_values),
            )
        )
    )
    program_raw_values = tuple(
        identity
        for identity in program_values
        if identity not in MECHANISM_AXIS_UNIVERSE_V1
    )
    if program_values:
        # Raw compiled program dimensions are authoritative.  A resolution
        # policy scalar is not an additional raw mechanism change.
        return tuple(program_raw_values or partial_raw_values or program_values)
    if resolution_values:
        return tuple(partial_raw_values or resolution_values)

    result = list(
        _innovation_complete_values(resolution.get("high_change_dimensions", ()))
    )
    # The proposal axis is a policy scalar in the usual path.  Use it only
    # when the executable program/resolution exposed no more specific raw
    # changed identity, otherwise it would duplicate that identity through an
    # alias such as TRAINING_PROCEDURE -> self_supervision.
    if not result:
        if outcome.source_proposal is not None:
            result.extend(
                _innovation_complete_values(outcome.source_proposal.mechanism_axis)
            )
        else:
            result.extend(
                _innovation_complete_values(
                    outcome.resolution_facts.get("mechanism_axis", ())
                )
            )
    return tuple(result)


def _construction_parent_context(outcome: ProducerOutcome) -> dict[str, Any]:
    option = outcome.provenance.get("construction_parent_option")
    if isinstance(option, Mapping):
        return {
            "construction_parent_options": (option,),
            "lineage_parent_mechanism_program": option,
        }
    return {}


def _innovation_identity_digests(
    outcome: ProducerOutcome,
    *,
    search_space_adapter: SearchSpaceAdapter,
    frozen_profile_ref: Mapping[str, Any],
    research_context: ResearchContext | None = None,
) -> tuple[str, str] | None:
    try:
        resolved = search_space_adapter.resolve_confirmation(
            "INNOVATION",
            outcome,
            {
                "phase": "IDENTIFY_INNOVATION",
                "frozen_profile_ref": frozen_profile_ref,
                "research_context": research_context,
                **_construction_parent_context(outcome),
            },
        )
        if (
            resolved.kind is not ConfirmationResolutionKindV1.EXACT_BINDING
            or not isinstance(resolved.binding, Mapping)
            or not isinstance(resolved.binding.get("effective_identity"), Mapping)
        ):
            return None
        identity = resolved.binding["effective_identity"]
        return (
            str(identity["effective_family_digest"]),
            str(identity["effective_experiment_digest"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


_DECLARED_COST_ORDINAL = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}


def _declared_program_cost(outcome: ProducerOutcome) -> float | None:
    """Project the Provider program's pre-outcome cost declaration to [0, 1]."""

    program = outcome.source_mechanism_program
    if not isinstance(program, Mapping):
        return None
    payload = program.get("program_payload")
    if not isinstance(payload, Mapping):
        return None
    estimated = payload.get("estimated_cost")
    if not isinstance(estimated, Mapping):
        estimated = payload.get("resource_contract")
    if not isinstance(estimated, Mapping):
        return None
    values: list[float] = []
    for field_name in ("relative_training_compute", "relative_memory"):
        value = _DECLARED_COST_ORDINAL.get(
            str(estimated.get(field_name, "")).upper()
        )
        if value is not None:
            values.append(value)
    precompute = estimated.get("precompute_required")
    if isinstance(precompute, bool):
        values.append(float(precompute))
    return sum(values) / len(values) if values else None


def _innovation_estimated_cost(outcome: ProducerOutcome) -> float:
    """Combine implementation spend and declared mechanism execution cost."""

    budget = dict(outcome.resolution_facts.get("required_budget", {}))
    token_cost = min(
        1.0,
        max(0.0, float(budget.get("implementation_token_ceiling", 0)) / 25_000.0),
    )
    declared_cost = _declared_program_cost(outcome)
    if declared_cost is None:
        return token_cost
    return min(1.0, max(0.0, 0.5 * token_cost + 0.5 * declared_cost))


def _implementation_policy(
    inputs: InnovationRuntimeInputs,
    outcome: ProducerOutcome,
    *,
    compiled_mechanism: Mapping[str, Any] | None = None,
    exact_parent_bundle: Mapping[str, Any] | None = None,
) -> SharedImplementerPolicy:
    assert outcome.spec is not None
    policy = inputs.policy
    if isinstance(compiled_mechanism, Mapping):
        space_identity = compiled_mechanism.get("space_identity")
        search_space_id = (
            space_identity.get("search_space_id")
            if isinstance(space_identity, Mapping)
            else None
        )
        template_for_space = getattr(
            inputs.implementer,
            "implementation_template_for_search_space",
            None,
        )
        if isinstance(search_space_id, str) and callable(template_for_space):
            policy = replace(
                policy,
                prompt_digest=sha256_digest(
                    template_for_space(search_space_id)
                ),
            )
    contract = outcome.spec.execution_contract
    if contract is None:
        if policy.execution_contract is None:
            raise ValueError("selected Innovation OpenSpec lacks an execution contract")
        contract = policy.execution_contract
    if compiled_mechanism is not None and outcome.spec.execution_contract is not None:
        space_identity = compiled_mechanism.get("space_identity")
        search_space_id = (
            space_identity.get("search_space_id")
            if isinstance(space_identity, Mapping)
            else None
        )
        if not isinstance(search_space_id, str) or not search_space_id:
            raise ValueError("compiled mechanism lacks search_space_id")
        if (
            search_space_id == "BL_ICF_MECHANISM_SPACE_V1"
            and outcome.implementation_companion is not None
        ):
            contract = compiler_owned_execution_contract(
                {
                    "base_model_config": contract["base_model_config"],
                    "mechanism_config": dict(contract["config"]),
                }
            )
        else:
            contract_config = contract.get("config")
            declarative_frozen_contract = (
                isinstance(contract_config, Mapping)
                and isinstance(
                    contract_config.get("recclaw_verified_assets"), Mapping
                )
                and isinstance(contract_config.get("frozen_profile"), Mapping)
            )
            if declarative_frozen_contract:
                if (
                    contract_config.get("recclaw_trainer_entrypoint")
                    != "recclaw_ext.trainer:FreshCandidateTrainer"
                ):
                    raise ValueError(
                        "declarative execution contract has a non-canonical trainer"
                    )
                contract = canonical_value(dict(contract))
            else:
                contract = compiler_owned_generic_execution_contract(contract)
    if exact_parent_bundle is not None:
        parent_contract = exact_parent_bundle.get("execution_contract")
        if isinstance(parent_contract, Mapping):
            # Explicit research config is the delta; omitted values inherit.
            # Component.parameters is not a config ownership map (custom
            # mechanisms legitimately declare their config only in the contract).
            # Never silently cancel an explicit delta by guessing its owner.
            contract = canonical_value(
                {
                    **dict(contract),
                    "config": {
                        **dict(parent_contract["config"]),
                        **dict(contract["config"]),
                    },
                }
            )
    return replace(policy, execution_contract=contract)


def _implementation_outcome(
    outcome: ProducerOutcome, contract: Mapping[str, Any] | None,
) -> ProducerOutcome:
    """Resolve executable metadata without changing the selected research intent.

    Selection, retries and confirmation keep the original proposal identity.
    Every implementation artifact is instead derived from this effective spec.
    """
    assert outcome.spec is not None
    # Legacy fixed candidates have no enriched executable spec. Their existing
    # adapter remains responsible for binding; do not convert that public path.
    if (outcome.spec.execution_contract is None or contract is None
            or contract == outcome.spec.execution_contract):
        return outcome
    companion = outcome.implementation_companion
    return replace(
        outcome,
        spec=replace(outcome.spec, execution_contract=contract),
        implementation_companion=(
            replace(companion, execution_contract=contract)
            if companion is not None else None
        ),
    )


def _open_mechanism_axis(
    outcome: ProducerOutcome,
    *,
    research_context: ResearchContext | None = None,
) -> str:
    baseline_context = (
        research_context.knowledge_base.get("baseline_context", {})
        if research_context is not None
        else {}
    )
    declared_profile_axes = (
        single_parent_research_axes(baseline_context)
        if isinstance(baseline_context, Mapping)
        else ()
    )
    context_policy = (
        research_context.policy
        if research_context is not None
        and isinstance(getattr(research_context, "policy", None), Mapping)
        else {}
    )
    raw_policy_axes = context_policy.get("mechanism_axis_targeting", ())
    policy_axes = (
        tuple(raw_policy_axes)
        if isinstance(raw_policy_axes, (tuple, list))
        and all(isinstance(axis, str) for axis in raw_policy_axes)
        else ()
    )
    profile_axes = (
        declared_profile_axes
        if len(policy_axes) == len(declared_profile_axes)
        and len(set(policy_axes)) == len(policy_axes)
        and set(policy_axes) == set(declared_profile_axes)
        else ()
    )

    def policy_axis(value: Any) -> str | None:
        if isinstance(value, str) and value in profile_axes:
            return value
        legacy = canonical_mechanism_axis(value)
        if legacy is not None:
            return legacy
        return None

    source = outcome.source_proposal
    program = (
        source.mechanism_program
        if source is not None
        else outcome.source_mechanism_program
    )
    payload = (
        program.get("program_payload", program)
        if isinstance(program, Mapping)
        else {}
    )
    changed_slots = payload.get("changed_slots", ()) if isinstance(payload, Mapping) else ()
    if isinstance(changed_slots, (tuple, list)):
        for change_role in ("CORE", "SUPPORT"):
            for change in changed_slots:
                if not isinstance(change, Mapping) or change.get("change_role") != change_role:
                    continue
                axis = policy_axis(change.get("slot_id"))
                if axis is not None:
                    return axis
    dimensions = tuple(outcome.resolution_facts.get("high_change_dimensions", ()))
    axis = next(
        (
            policy_axis(dimension)
            for dimension in dimensions
            if policy_axis(dimension) is not None
        ),
        "",
    )
    if axis:
        return axis
    declared_axis = (
        source.mechanism_axis
        if source is not None
        else outcome.resolution_facts.get("mechanism_axis")
    )
    return str(declared_axis) if declared_axis in profile_axes else ""


def _qualified_open_features(
    outcome: ProducerOutcome,
    *,
    parent_available: bool,
) -> tuple[SearchUtilityFeaturesV1, RouterFeatureEvidenceV1]:
    estimated_cost = _innovation_estimated_cost(outcome)
    utility = SearchUtilityFeaturesV1(
        runnable_probability=1.0,
        useful_signal=0.5,
        # Qualification proves executability, not effect.  Keep the effect
        # prior neutral while giving an as-yet-unobserved high-change
        # realization its actual first-experiment information value.
        frontier_potential=1.0,
        information_gain=1.0,
        cost=estimated_cost,
        blocker_risk=0.0,
    )
    evidence = RouterFeatureEvidenceV1(
        compile_valid=True,
        handler_available=True,
        materializer_available=True,
        blocker_rate=0.0,
        semantic_duplicate=False,
        parent_available=parent_available,
        mechanism_depth=len(outcome.spec.causal_chain) if outcome.spec else 0,
        estimated_cost=estimated_cost,
        llm_diagnostic=utility,
    )
    return utility, evidence


def _resource_performance_repair_failure(
    *,
    resource_profile: Mapping[str, Any],
    quality_admission: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Describe one outcome-blind implementation-efficiency repair.

    A completed bounded GPU probe may violate an explicit implementation
    efficiency contract. Insufficient allocated time alone does not establish
    an implementation defect, especially on a shared device.
    """

    prediction = resource_profile.get("prediction")
    process = resource_profile.get("probe_process")
    efficiency_comparison = quality_admission.get(
        "implementation_efficiency_comparison"
    )
    efficiency_status = quality_admission.get("status")
    repeated_timeout_risk = (
        efficiency_status
        in {
            "RESOURCE_DEFERRED_IMPLEMENTATION_EFFICIENCY_NOT_IMPROVED",
            "RESOURCE_DEFERRED_IMPLEMENTATION_EFFICIENCY_MECHANISM_DRIFT",
        }
        and isinstance(efficiency_comparison, Mapping)
    )
    deferred_rows = resource_profile.get("deferred")
    typed_resource_infeasible = isinstance(deferred_rows, (tuple, list)) and any(
        isinstance(item, Mapping)
        and item.get("reason")
        in {
            "PREDICTED_GPU_MEMORY_RESERVE_ENVELOPE",
            "PROJECTED_CAMPAIGN_CHECKPOINT_STORAGE",
        }
        for item in deferred_rows
    )
    profile_efficiency_infeasible = (
        isinstance(deferred_rows, (tuple, list))
        and any(
            isinstance(item, Mapping)
            and item.get("reason")
            == "PROFILE_IMPLEMENTATION_EFFICIENCY_ENVELOPE"
            for item in deferred_rows
        )
        and isinstance(prediction, Mapping)
        and prediction.get("profile_resource_efficiency_envelope_exceeded")
        is True
    )
    # Allocated worker time is a resource constraint, not an algorithm-efficiency
    # contract. Only the explicit pre-candidate envelope justifies this repair.
    measured_window_infeasible = (
        quality_admission.get("status") == "RESOURCE_DEFERRED"
        and resource_profile.get("status")
        in {"RESOURCE_ADMITTED", "RESOURCE_DEFERRED"}
        and not typed_resource_infeasible
        and isinstance(prediction, Mapping)
        and profile_efficiency_infeasible
    )
    if (
        not (repeated_timeout_risk or measured_window_infeasible)
        or not isinstance(prediction, Mapping)
        or prediction.get("model")
        != "FIXED_BATCH_THROUGHPUT_NATIVE_EARLY_STOP_EXTRAPOLATION_V5"
        or not isinstance(process, Mapping)
        or process.get("status") != "RESULT"
        or process.get("exit_code") != 0
        or process.get("process_isolated") is not True
    ):
        return None
    ceiling = prediction.get("worker_ceiling_seconds")
    detail = (
        dict(efficiency_comparison)
        if repeated_timeout_risk
        else {
            "native_early_stop_budget_required_seconds": prediction.get(
                "native_early_stop_budget_required_seconds"
            ),
            "full_requested_epoch_estimated_wall_time_seconds": prediction.get(
                "full_requested_epoch_estimated_wall_time_seconds"
            ),
            "profile_resource_efficiency_envelope": prediction.get(
                "profile_resource_efficiency_envelope"
            ),
            "worker_ceiling_seconds": ceiling,
        }
    )
    return canonical_value(
        {
            "stage": "RESOURCE_PROBE",
            "failure_class": "IMPLEMENTATION",
            "implicated_file": "recclaw_ext/candidate.py",
            "repair_scope": "IMPLEMENTATION_RESOURCE_BEHAVIOR_ONLY",
            "preserve_mechanism_program": True,
            "reason_code": (
                "IMPLEMENTATION_EFFICIENCY_REPAIR_CHANGED_MECHANISM"
                if efficiency_status
                == "RESOURCE_DEFERRED_IMPLEMENTATION_EFFICIENCY_MECHANISM_DRIFT"
                else "MEASURED_IMPLEMENTATION_THROUGHPUT_NOT_IMPROVED"
                if repeated_timeout_risk
                else "MEASURED_IMPLEMENTATION_THROUGHPUT_EXCEEDS_ENVELOPE"
            ),
            "message": (
                "The bounded isolated GPU probe completed, but measured "
                "implementation throughput exceeds the Profile-owned execution "
                "efficiency envelope. Optimize the computational "
                "realization only. Preserve the exact mechanism program, all "
                "implemented primitive and architecture-operator identities, "
                "the RecBole API, and training/evaluation semantics. Remove "
                "avoidable Python per-example loops, repeated full-graph "
                "recomputation, and quadratic all-node intermediates using "
                "semantics-preserving vectorization, caching, or minibatching."
            ),
            "detail": detail,
        }
    )


def _resource_admission_failure(
    *,
    resource_profile: Mapping[str, Any],
    quality_admission: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    performance_failure = _resource_performance_repair_failure(
        resource_profile=resource_profile,
        quality_admission=quality_admission,
    )
    if performance_failure is not None:
        return performance_failure
    if quality_admission.get("admitted") is True:
        return None
    deferred_rows = resource_profile.get("deferred")
    deferred_rows = (
        tuple(item for item in deferred_rows if isinstance(item, Mapping))
        if isinstance(deferred_rows, (tuple, list))
        else ()
    )
    typed_deferred = next(
        (
            item
            for item in deferred_rows
            if item.get("reason")
            in {
                "PREDICTED_GPU_MEMORY_RESERVE_ENVELOPE",
                "PROJECTED_CAMPAIGN_CHECKPOINT_STORAGE",
            }
        ),
        None,
    )
    if typed_deferred is not None:
        reason_code = str(typed_deferred["reason"])
        prediction = resource_profile.get("prediction")
        prediction = prediction if isinstance(prediction, Mapping) else {}
        fields = (
            (
                "peak_memory_observed_mib",
                "peak_memory_prediction_mib",
                "peak_memory_candidate_ceiling_mib",
            )
            if reason_code == "PREDICTED_GPU_MEMORY_RESERVE_ENVELOPE"
            else (
                "checkpoint_storage_bytes",
                "projected_campaign_checkpoint_bytes",
                "storage_available_after_probe_cleanup_bytes",
            )
        )
        detail = {
            field: prediction[field]
            for field in fields
            if isinstance(prediction.get(field), (int, float))
            and not isinstance(prediction.get(field), bool)
        }
        detail["resource_disposition"] = str(
            typed_deferred.get("resource_disposition") or "RESOURCE_INFEASIBLE"
        )
        if reason_code == "PROJECTED_CAMPAIGN_CHECKPOINT_STORAGE":
            return canonical_value(
                {
                    "stage": "RESOURCE_PROBE",
                    "failure_class": "INFRASTRUCTURE",
                    "failure_scope": "SHARED_INFRASTRUCTURE",
                    "reason_code": reason_code,
                    "message": (
                        "The current writable root cannot retain the measured "
                        "serialized state for the remaining campaign. This is a "
                        "storage placement or capacity failure, not candidate "
                        "implementation work. Preserve the exact candidate and "
                        "continue it on a writable root with sufficient capacity."
                    ),
                    "detail": detail,
                }
            )
        return canonical_value(
            {
                "stage": "RESOURCE_PROBE",
                "failure_class": "IMPLEMENTATION",
                "reason_code": reason_code,
                "repair_scope": "IMPLEMENTATION_RESOURCE_BEHAVIOR_ONLY",
                "preserve_mechanism_program": True,
                "message": (
                    "The bounded resource probe measured an executable resource "
                    "envelope that is not sustainable for this campaign. Preserve "
                    "the mechanism program, formula, architecture, and real mechanism "
                    "hyperparameters; repair only the implementation representation, "
                    "memory lifetime, or checkpoint serialization identified by the "
                    "typed reason and compact measured detail."
                ),
                "detail": detail,
            }
        )
    budget_deferred = next(
        (item for item in deferred_rows if item.get("reason")
         == "NATIVE_EARLY_STOP_WINDOW_EXCEEDS_EXECUTION_BUDGET"),
        None,
    )
    if budget_deferred is not None:
        prediction = resource_profile.get("prediction") or {}
        return canonical_value({
            "stage": "RESOURCE_PROBE",
            "failure_class": "INFRASTRUCTURE",
            "failure_scope": "SHARED_INFRASTRUCTURE",
            "reason_code": "NATIVE_EARLY_STOP_WINDOW_EXCEEDS_EXECUTION_BUDGET",
            "message": (
                "The measured native execution window exceeds the allocated budget. "
                "Preserve the qualified candidate and completed probe for resource "
                "coordination; this alone does not establish an implementation defect."
            ),
            "detail": {key: prediction.get(key) for key in (
                "native_early_stop_budget_required_seconds",
                "worker_ceiling_seconds",
                "full_requested_epoch_estimated_wall_time_seconds",
            )},
        })
    profile_status = str(resource_profile.get("status") or "RESOURCE_PROFILE_INVALID")
    reason_code = str(quality_admission.get("status") or profile_status)
    if reason_code == "RESOURCE_PROFILE_INVALID":
        return canonical_value({
            "stage": "RESOURCE_PROBE",
            "failure_class": "INFRASTRUCTURE",
            "failure_scope": "SHARED_INFRASTRUCTURE",
            "reason_code": reason_code,
            "message": (
                "The machine-produced resource profile does not satisfy its "
                "consumer contract. Preserve the qualified candidate and "
                "completed probe; repair the resource interface before retrying "
                "admission. This is not candidate implementation work."
            ),
        })
    diagnostic = resource_profile.get("resource_probe_diagnostic")
    if not isinstance(diagnostic, Mapping):
        diagnostic = resource_profile.get("diagnostic")
    detail = dict(diagnostic) if isinstance(diagnostic, Mapping) else None
    diagnostic_phase = str(
        diagnostic.get("phase", "") if isinstance(diagnostic, Mapping) else ""
    )
    diagnostic_type = str(
        diagnostic.get("error_type", "")
        if isinstance(diagnostic, Mapping)
        else ""
    )
    diagnostic_message = str(
        diagnostic.get("message", "")
        if isinstance(diagnostic, Mapping)
        else ""
    )
    raw_failure_scope = resource_profile.get("failure_scope")
    if raw_failure_scope is None and isinstance(diagnostic, Mapping):
        raw_failure_scope = diagnostic.get("failure_scope")
    failure_scope = str(raw_failure_scope or "").upper()
    candidate_local_oom = (
        failure_scope == "CANDIDATE_LOCAL"
        and (
            diagnostic_type.lower()
            in {"outofmemoryerror", "torch.outofmemoryerror"}
            or "out of memory" in diagnostic_message.lower()
        )
    )
    if candidate_local_oom:
        memory_detail = dict(detail or {})
        probe_summary = resource_profile.get("probe")
        if isinstance(probe_summary, Mapping):
            memory_detail["probe"] = dict(probe_summary)
        return canonical_value(
            {
                "stage": "RESOURCE_PROBE",
                "failure_class": "IMPLEMENTATION",
                "reason_code": "MEASURED_IMPLEMENTATION_MEMORY_EXCEEDS_DEVICE",
                "failure_scope": "CANDIDATE_LOCAL",
                "repair_scope": "IMPLEMENTATION_RESOURCE_BEHAVIOR_ONLY",
                "preserve_mechanism_program": True,
                "message": (
                    "The isolated probe proved that this candidate's executable "
                    "representation exhausts device memory. Preserve the exact "
                    "mechanism program, architecture, real mechanism parameters, "
                    "and frozen widths; replace only the implementation state "
                    "representation or memory lifetime with a semantically "
                    "equivalent bounded-memory realization."
                ),
                "detail": memory_detail,
            }
        )
    if failure_scope in {"WORKER_TRANSIENT", "SHARED_INFRASTRUCTURE"}:
        return canonical_value(
            {
                "stage": "RESOURCE_PROBE",
                "failure_class": failure_scope,
                "reason_code": diagnostic_type or profile_status,
                "failure_scope": failure_scope,
                "message": (
                    "The isolated resource probe stopped at a recoverable worker "
                    "or shared-infrastructure boundary. Preserve this candidate "
                    "and generation for the next execution attempt."
                ),
                "detail": detail,
            }
        )
    if failure_scope != "CANDIDATE_LOCAL" and (
        diagnostic_phase == "RESOURCE_PROBE_RECOVERY"
        or diagnostic_type == "RESOURCE_PROBE_INTERRUPTED_RECOVERY"
    ):
        return canonical_value(
            {
                "stage": "RESOURCE_PROBE",
                "failure_class": "RESOURCE",
                "reason_code": (
                    diagnostic_type or "RESOURCE_PROBE_INTERRUPTED_RECOVERY"
                ),
                "failure_scope": "RECOVERY",
                "message": str(
                    diagnostic.get("message")
                    if isinstance(diagnostic, Mapping)
                    else "interrupted resource probe recovery"
                ),
                "detail": detail,
            }
        )
    return canonical_value(
        {
            "stage": "RESOURCE_PROBE",
            "failure_class": "IMPLEMENTATION",
            "reason_code": reason_code,
            **({"failure_scope": failure_scope} if failure_scope else {}),
            "message": (
                "The isolated resource probe did not admit this implementation. "
                "Preserve the research mechanism, formula, architecture, and real "
                "mechanism hyperparameters; repair only its executable resource "
                "behavior using the typed probe reason and measured telemetry."
            ),
            "detail": detail,
        }
    )


def _create_candidate_attempt_parent(
    candidate_parent: Path,
    attempt_index: int,
) -> Path:
    """Create a fresh physical directory for one logical attempt.

    A campaign can be resumed from a pre-outcome snapshot whose logical
    attempt manifest is still empty while an interrupted implementation has
    already created its candidate directory.  Preserve that evidence and use
    a deterministic resume suffix without changing the logical attempt index.
    """

    base_name = f"attempt-{attempt_index:02d}"
    attempt_parent = Path(candidate_parent) / base_name
    resume_index = 0
    while attempt_parent.exists():
        resume_index += 1
        attempt_parent = (
            Path(candidate_parent)
            / f"{base_name}-resume-{resume_index:02d}"
        )
    attempt_parent.mkdir(parents=True, exist_ok=False)
    return attempt_parent


def _selected_implementation_feedback(context: ResearchContext | None, mechanism_change: str) -> Mapping[str, Any]:
    if context is None:
        return {}
    scientific_memory = context.scientific_memory
    if not any(isinstance(scientific_memory.get(key), Mapping) for key in ("global_memory", "global")):
        # Initial and older contexts keep shared history directly in memory.
        scientific_memory = {"global_memory": {
            key: value for key, value in scientific_memory.items() if key != "by_role"
        }}
    view = project_provider_context_view({
        "round_index": context.round_index,
        "scientific_memory": scientific_memory,
        "research_portfolio": [{"mechanism_change": mechanism_change}],
    })
    memory = view["memory"]
    feedback: dict[str, Any] = {"comparison_basis": memory["comparison_basis"]}
    for name in ("recent_experiments", "related_mechanism_experiments"):
        packed = memory.get(name, {})
        rows = [dict(zip(packed["attempt_fields"], values)) for values in packed.get("attempts", ())]
        # Implementation consumes scientific/cost facts, never arm or role credit.
        feedback[name] = [{key: value for key, value in row.items() if key in {
            "round", "mechanism", "ndcg@10", "delta", "cost", "failure", "next_task", "parent", "frontier_delta", "efficiency", "evidence", "confounds",
        }} for row in rows]
    guard = view.get("state", {}).get("evidence_guard")
    if guard:
        feedback["evidence_guard"] = guard
    return canonical_value(feedback)


def _implementation_efficiency_repair_request(
    *,
    request: Mapping[str, Any],
    semantic_identity_digest: str,
    structural_context: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Repair a selected exact experiment; a new hypothesis remains discovery."""

    source = structural_context.get("implementation_efficiency_repair_source_files")
    facts = structural_context.get("implementation_efficiency_repair_context")
    prior_contract = structural_context.get("implementation_efficiency_repair_execution_contract")
    if (
        structural_context.get("implementation_efficiency_repair_semantic_digest") != semantic_identity_digest
        or not isinstance(source, Mapping) or not source
        or not isinstance(facts, Mapping)
        or not isinstance(prior_contract, Mapping)
        or canonical_value(prior_contract) != request["service_policy"].get("execution_contract")
    ):
        return None
    return build_mechanical_repair_request(
        request,
        {
            "stage": "RESOURCE_PROBE",
            "failure_class": "IMPLEMENTATION",
            "reason_code": "MEASURED_IMPLEMENTATION_THROUGHPUT_EXCEEDS_ENVELOPE",
            "implicated_file": "recclaw_ext/candidate.py",
            "repair_scope": "IMPLEMENTATION_RESOURCE_BEHAVIOR_ONLY",
            "preserve_mechanism_program": True,
            "message": "The selected exact experiment was resource-censored. Improve its computational realization, preserving mechanism, effective configuration and the frozen training protocol. Return the request's declared response mode.",
            "detail": dict(facts),
        },
        current_source=source,
        repair_attempt=1,
    )


def _matching_exact_parent_bundle(
    outcome: ProducerOutcome,
    compiled_mechanism: Any,
    exact_parent_bundle: Mapping[str, Any] | None,
    *,
    frozen_single_parent: bool = False,
    search_space_adapter: SearchSpaceAdapter | None = None,
) -> Mapping[str, Any] | None:
    if exact_parent_bundle is None:
        return None
    if not frozen_single_parent and not (
        isinstance(outcome.provenance, Mapping)
        and outcome.provenance.get("lineage_parent_visibility") is True
    ):
        return None
    if not isinstance(compiled_mechanism, Mapping):
        matcher = getattr(
            search_space_adapter,
            "matches_exact_parent_bundle",
            None,
        )
        if callable(matcher) and matcher(outcome, exact_parent_bundle):
            return exact_parent_bundle
        return None
    payload = compiled_mechanism.get("mechanism_program", {}).get(
        "program_payload", {}
    )
    if (
        isinstance(payload, Mapping)
        and payload.get("construction_mode") == "CUSTOM_MODEL"
        and not frozen_single_parent
    ):
        return None
    parent_refs = (
        payload.get("parent_refs") if isinstance(payload, Mapping) else None
    )
    expected = canonical_value(
        [
            {
                "candidate_id": exact_parent_bundle.get("candidate_id"),
                "program_digest": exact_parent_bundle.get("program_digest"),
            }
        ]
    )
    if canonical_value(parent_refs) != expected:
        return None
    return exact_parent_bundle


def _lineage_parent_bundle_requested(
    outcome: ProducerOutcome,
    compiled_mechanism: Any,
) -> bool:
    """Return whether this producer was explicitly given an exact parent."""

    if not (
        isinstance(outcome.provenance, Mapping)
        and outcome.provenance.get("lineage_parent_visibility") is True
        and isinstance(compiled_mechanism, Mapping)
    ):
        return False
    parent_refs = (
        compiled_mechanism.get("mechanism_program", {})
        .get("program_payload", {})
        .get("parent_refs")
    )
    return bool(parent_refs)


def _run_innovation_lane(
    candidates: Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]],
    *,
    current_profile: SearchExecutableProfileV1,
    current_slate_ref: str,
    current_slate_digest: str,
    research_policy: VersionedResearchPolicyV1,
    search_space_adapter: SearchSpaceAdapter,
    research_context: ResearchContext | None = None,
    inputs: InnovationRuntimeInputs,
    evaluator: Mapping[str, Any],
    split: str,
    frozen_profile_ref: Mapping[str, Any],
    implementation_call_limit: int = MAX_REPAIR_TURNS + 1,
    candidate_attempt_limit: int = 1,
    allow_candidate_reroute: bool = False,
    enforce_unbound_confirmation: bool = True,
    outcome_selector: (
        Callable[
            [Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]]],
            str | None,
        ]
        | None
    ) = None,
    attempted_semantic_identities: frozenset[str] = frozenset(),
    attempted_effective_experiment_identities: frozenset[str] = frozenset(),
    _prior_attempts: Sequence[Mapping[str, Any]] = (),
    _reroute_index: int = 0,
    _forced_first_gateway_request: Mapping[str, Any] | None = None,
    _qualified_retry: InnovationLaneResult | None = None,
    _provider_retry_request: Mapping[str, Any] | None = None,
    _provider_retry_namespace: str | None = None,
    _provider_retry_remaining_calls: int | None = None,
    exact_parent_bundle: Mapping[str, Any] | None = None,
    exact_parent_bundle_loader: Callable[[ProducerOutcome | None], Mapping[str, Any] | None] | None = None,
) -> InnovationLaneResult | None:
    if (
        isinstance(implementation_call_limit, bool)
        or not isinstance(implementation_call_limit, int)
        or implementation_call_limit < 1
    ):
        raise ValueError("implementation_call_limit must be a positive integer")
    if (
        isinstance(candidate_attempt_limit, bool)
        or not isinstance(candidate_attempt_limit, int)
        or candidate_attempt_limit < 1
    ):
        raise ValueError("candidate_attempt_limit must be a positive integer")
    acquired = _acquire_innovation_spec(
        candidates,
        research_policy=research_policy,
        search_space_adapter=search_space_adapter,
        frozen_profile_ref=frozen_profile_ref,
        research_context=research_context,
        enforce_unbound_confirmation=enforce_unbound_confirmation,
        outcome_selector=outcome_selector,
    )
    if acquired is None:
        return None
    selected_outcome, resolution, idea_acquisition = acquired
    assert selected_outcome.spec is not None
    prepared_resolution = search_space_adapter.resolve_confirmation(
        "INNOVATION",
        selected_outcome,
        {
            "phase": "PREPARE_INNOVATION",
            "frozen_profile_ref": frozen_profile_ref,
            "research_context": research_context,
            **_construction_parent_context(selected_outcome),
        },
    )
    if (
        prepared_resolution.kind
        is not ConfirmationResolutionKindV1.EXACT_BINDING
        or not isinstance(prepared_resolution.binding, Mapping)
    ):
        raise ValueError("SearchSpaceAdapter could not prepare Innovation")
    prepared_innovation = prepared_resolution.binding
    # Keep acquisition/rerouting on the frozen Producer identity. The adapter's
    # bound parent and executable metadata belong to implementation only.
    prepared_outcome = prepared_innovation["producer_outcome"]
    source_proposal = prepared_outcome.source_proposal
    compiler_binding = prepared_innovation.get("compiler_binding")
    semantic_identity_ref = str(prepared_innovation["semantic_identity_ref"])
    semantic_identity_digest = str(
        prepared_innovation["semantic_identity_digest"]
    )
    mechanism_program = prepared_innovation["mechanism_program"]
    effective_identity = prepared_innovation["effective_identity"]
    effective_experiment_digest = str(
        effective_identity["effective_experiment_digest"]
    )
    effective_family_digest = str(effective_identity["effective_family_digest"])
    attempts: list[Mapping[str, Any]] = [
        canonical_value(dict(item)) for item in _prior_attempts
    ]
    effective_duplicate = (
        effective_experiment_digest in attempted_effective_experiment_identities
    )
    semantic_duplicate = (
        semantic_identity_digest in attempted_semantic_identities
        or any(
            entry.semantic_identity_digest == semantic_identity_digest
            for entry in current_profile.entries
        )
    )
    repair_context = inputs.resource_structural_context or {}
    selected_measured_repair = (
        repair_context.get("implementation_efficiency_repair_semantic_digest") == semantic_identity_digest
        and bool(repair_context.get("implementation_efficiency_repair_source_files"))
        and bool(repair_context.get("implementation_efficiency_repair_context"))
    )
    if (semantic_duplicate or effective_duplicate) and not selected_measured_repair:
        quality_admission = _semantic_duplicate_preimplementation_admission(
            spec=selected_outcome.spec,
            semantic_identity_digest=(
                effective_experiment_digest
                if effective_duplicate
                else semantic_identity_digest
            ),
        )
        duplicate_reason = (
            "EFFECTIVE_EXPERIMENT_DUPLICATE"
            if effective_duplicate
            else "SEMANTIC_DUPLICATE"
        )
        attempts.append(
            canonical_value(
                {
                    "attempt": len(attempts),
                    "producer_role": selected_outcome.producer_role,
                    "spec_digest": selected_outcome.spec.digest,
                    "core_mechanism_contrast": selected_outcome.spec.mechanism_change,
                    "next_discriminative_task": selected_outcome.spec.falsifier,
                    "candidate_id": (
                        source_proposal.candidate_id
                        if source_proposal is not None
                        else prepared_innovation["candidate_id"]
                    ),
                    "mechanism_semantics_digest": semantic_identity_digest,
                    "effective_experiment_digest": effective_experiment_digest,
                    "effective_family_digest": effective_family_digest,
                    "primitive_ids": effective_identity["primitive_ids"],
                    "idea_acquisition": idea_acquisition.to_dict(),
                    "failure": {
                        "stage": "POST_PROPOSAL_PRE_IMPLEMENTATION",
                        "failure_class": "SEMANTIC_IDENTITY",
                        "reason_code": duplicate_reason,
                        "message": (
                            "candidate execution semantics were already attempted "
                            "in this exploration campaign"
                        ),
                    },
                    "quality_admission": quality_admission,
                }
            )
        )
        blocked_result = InnovationLaneResult(
            selected_outcome=selected_outcome,
            resolution=resolution,
            idea_acquisition=idea_acquisition,
            attempts=tuple(attempts),
            materialized=None,
            qualification=None,
            capability=None,
            registry=None,
            profile_manifest=None,
            next_profile=None,
            profile_receipt=None,
            qualified_execution=None,
            search_candidate=None,
            candidate_root=None,
            fresh_campaign_id=inputs.fresh_campaign_id,
            resource_profile=None,
            quality_admission=quality_admission,
        )
        remaining = tuple(
            pair
            for pair in candidates
            if pair[0].digest != selected_outcome.digest
        )
        if (
            allow_candidate_reroute
            and remaining
            and _reroute_index + 1 < candidate_attempt_limit
        ):
            next_result = _run_innovation_lane(
                remaining,
                current_profile=current_profile,
                current_slate_ref=current_slate_ref,
                current_slate_digest=current_slate_digest,
                research_policy=research_policy,
                search_space_adapter=search_space_adapter,
                research_context=research_context,
                inputs=inputs,
                evaluator=evaluator,
                split=split,
                frozen_profile_ref=frozen_profile_ref,
                implementation_call_limit=implementation_call_limit,
                candidate_attempt_limit=candidate_attempt_limit,
                allow_candidate_reroute=True,
                enforce_unbound_confirmation=enforce_unbound_confirmation,
                outcome_selector=outcome_selector,
                attempted_semantic_identities=(
                    attempted_semantic_identities | {semantic_identity_digest}
                ),
                attempted_effective_experiment_identities=(
                    attempted_effective_experiment_identities
                    | {effective_experiment_digest}
                ),
                _prior_attempts=blocked_result.attempts,
                _reroute_index=_reroute_index + 1,
                exact_parent_bundle=exact_parent_bundle,
                exact_parent_bundle_loader=exact_parent_bundle_loader,
            )
            return next_result if next_result is not None else blocked_result
        return blocked_result
    compiled_implementation = prepared_innovation.get("compiled_implementation")
    candidate_parent_bundle = exact_parent_bundle
    frozen_single_parent_requested = bool(
        research_context is not None
        and isinstance(research_context.knowledge_base, Mapping)
        and is_single_parent_context(
            research_context.knowledge_base.get("baseline_context")
        )
    )
    if (
        candidate_parent_bundle is None
        and exact_parent_bundle_loader is not None
        and (
            frozen_single_parent_requested
            or _lineage_parent_bundle_requested(
                selected_outcome,
                compiled_implementation,
            )
        )
    ):
        candidate_parent_bundle = exact_parent_bundle_loader(selected_outcome)
    selected_parent_bundle = _matching_exact_parent_bundle(
        selected_outcome,
        compiled_implementation,
        candidate_parent_bundle,
        frozen_single_parent=frozen_single_parent_requested,
        search_space_adapter=search_space_adapter,
    )
    compiled_program = (
        compiled_implementation.get("mechanism_program")
        if isinstance(compiled_implementation, Mapping)
        else None
    )
    compiled_payload = (
        compiled_program.get("program_payload")
        if isinstance(compiled_program, Mapping)
        else None
    )
    construction_mode = (
        compiled_payload.get("construction_mode")
        if isinstance(compiled_payload, Mapping)
        else None
    )
    if (
        frozen_single_parent_requested
        and construction_mode != "CUSTOM_MODEL"
        and selected_parent_bundle is None
    ):
        raise ValueError(
            "single-parent preserving candidate lacks the exact active construction parent source"
        )
    implementer_policy = _implementation_policy(
        inputs,
        prepared_outcome,
        compiled_mechanism=(
            compiled_implementation
            if isinstance(compiled_implementation, Mapping)
            else None
        ),
        exact_parent_bundle=selected_parent_bundle,
    )
    implementation_outcome = _implementation_outcome(
        prepared_outcome, implementer_policy.execution_contract,
    )
    implementation_spec = implementation_outcome.spec
    prepared_innovation = {
        **prepared_innovation, "producer_outcome": implementation_outcome,
    }
    implementation_feedback = _selected_implementation_feedback(
        research_context, selected_outcome.spec.mechanism_change
    )
    request = (
        dict(_qualified_retry.materialized.shared_request)
        if _qualified_retry is not None and _qualified_retry.materialized is not None
        else build_shared_implementer_request(
            implementation_spec,
            policy=implementer_policy,
            compiled_mechanism=compiled_implementation,
            exact_parent_bundle=selected_parent_bundle,
            research_feedback=implementation_feedback,
        )
    )
    forced_first_gateway_request = _forced_first_gateway_request
    if forced_first_gateway_request is None:
        forced_first_gateway_request = _implementation_efficiency_repair_request(
            request=request,
            semantic_identity_digest=semantic_identity_digest,
            structural_context=inputs.resource_structural_context or {},
        )
    parent_scoped_patch = (
        request["service_policy"]["response_mode"]
        in {
            PARENT_METHOD_PATCH_RESPONSE_MODE,
            PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE,
            PROFILE_MODEL_HOOKS_RESPONSE_MODE,
            SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE,
        }
    )
    blind_candidate_id = str(request["blind_candidate_id"])
    candidate_attempts: list[Mapping[str, Any]] = []
    response: Mapping[str, Any] | None = None
    materialized: MaterializedCandidate | None = None
    qualification: MechanicalQualificationRun | None = None
    candidate_root: Path | None = None

    if _qualified_retry is not None:
        # Qualification belongs to this immutable materialization, not to a
        # fresh Implementer request. An infrastructure-only resource retry
        # resumes the completed stage; it is not another implementation.
        materialized = _qualified_retry.materialized
        qualification = _qualified_retry.qualification
        if (
            materialized is None or qualification is None
            or qualification.receipt.status is not QualificationStatusV1.PASS
            or _qualified_retry.capability is None
            or _qualified_retry.registry is None
            or _qualified_retry.qualified_execution is None
            or _qualified_retry.candidate_root is None
            or not attempts
            or idea_acquisition.selected_spec_digest
            != _qualified_retry.idea_acquisition.selected_spec_digest
        ):
            raise ValueError("qualified resource retry lacks its completed candidate stage")
        candidate_root = Path(_qualified_retry.candidate_root)
        for row in materialized.implementation_receipt["written_files"]:
            source_path = candidate_root / row["path"]
            if (
                not source_path.is_file()
                or hashlib.sha256(source_path.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise ValueError("qualified resource retry candidate source changed")
        if any(
            _qualified_retry.qualified_execution.get(key) != value
            for key, value in dict(implementer_policy.execution_contract or {}).items()
        ):
            raise ValueError("qualified resource retry execution contract changed")
        current_spec_digest = attempts[-1]["spec_digest"]
        candidate_attempts = [
            item for item in attempts
            if item.get("spec_digest") == current_spec_digest
            and not item.get("resource_retry", False)
        ]
        response = {"files": [
            {"path": path, "content": (candidate_root / path).read_text(encoding="utf-8")}
            for path in materialized.package.allowed_files
        ]}
        attempts.append(canonical_value({
            **{key: value for key, value in attempts[-1].items()
               if key not in {"failure", "resource_profile", "quality_admission"}},
            "attempt": len(attempts), "resource_retry": True, "failure": None,
        }))

    first_repair_attempt = (
        int(_provider_retry_request.get("repair_attempt", 0))
        if _provider_retry_request is not None else 0
    )
    if _provider_retry_request is not None:
        forced_first_gateway_request = _provider_retry_request
    repair_stop = (
        first_repair_attempt + _provider_retry_remaining_calls
        if _provider_retry_remaining_calls is not None
        else min(MAX_REPAIR_TURNS + 1, implementation_call_limit)
    )
    for repair_attempt in range(
        first_repair_attempt,
        0 if _qualified_retry is not None else repair_stop
    ):
        gateway_request = (
            forced_first_gateway_request
            if repair_attempt == first_repair_attempt
            and forced_first_gateway_request is not None
            else
            request
            if repair_attempt == 0
            else build_mechanical_repair_request(
                request,
                candidate_attempts[-1]["failure"],
                current_source=_source_files(response or {}),
                repair_attempt=repair_attempt,
            )
        )
        recorded_repair_attempt = int(
            gateway_request.get("repair_attempt", repair_attempt)
        )
        attempt_index = len(attempts)
        try:
            retry_method = getattr(inputs.implementer, "call_with_namespace", None)
            if (
                _provider_retry_request is not None
                and _provider_retry_namespace is not None
                and repair_attempt == first_repair_attempt
                and callable(retry_method)
            ):
                response = retry_method(
                    gateway_request, logical_namespace=_provider_retry_namespace,
                )
            else:
                response = inputs.implementer(gateway_request)
        except Exception as error:  # external implementer boundary
            attempt_record = canonical_value(
                {
                    "attempt": attempt_index,
                    "repair_attempt": recorded_repair_attempt,
                    "producer_role": selected_outcome.producer_role,
                    "spec_digest": selected_outcome.spec.digest,
                    "candidate_id": blind_candidate_id,
                    "mechanism_semantics_digest": semantic_identity_digest,
                    "effective_experiment_digest": effective_experiment_digest,
                    "effective_family_digest": effective_family_digest,
                    "primitive_ids": effective_identity["primitive_ids"],
                    "idea_acquisition": idea_acquisition.to_dict(),
                    "failure": {
                        "stage": "PROVIDER",
                        "failure_class": "PROVIDER",
                        "reason_code": type(error).__name__,
                        "message": str(error),
                        **({
                            "failure_class": "PROVIDER_EXTERNAL",
                            "provider_failure": error.result.failure,
                            "sealed_request": error.result.sealed_request,
                            "gateway_request": gateway_request,
                            "remaining_calls": repair_stop - repair_attempt,
                        } if isinstance(error, ProviderUnavailableError) else {}),
                        **({
                            "failure_class": "PROVIDER_ADMISSION",
                            "reason_code": type(error.__cause__ or error).__name__,
                            "gateway_request": gateway_request,
                            "remaining_calls": repair_stop - repair_attempt,
                        } if isinstance(error, LabApiRequestAdmissionError) else {}),
                    },
                }
            )
            attempts.append(attempt_record)
            candidate_attempts.append(attempt_record)
            break
        if not parent_scoped_patch:
            normalized_source = normalize_candidate_runtime_imports(
                _source_files(response)
            )
            response = canonical_value(
                {
                    **dict(response),
                    "files": [
                        {"content": content, "path": path}
                        for path, content in normalized_source.items()
                    ],
                }
            )
        # Compose method-local revisions even for compiler-owned parent hooks.
        # Fragment syntax/ABI repairs without method ownership stay with the binder.
        if (
            recorded_repair_attempt > 0
            and compiled_implementation is not None
            and (
                not parent_scoped_patch
                or gateway_request.get("repair_context", {}).get("implicated_methods")
            )
        ):
            repair_context = gateway_request.get("repair_context")
            try:
                if not isinstance(repair_context, Mapping):
                    raise ValueError("mechanical repair context is missing")
                scoped_source = scope_mechanical_repair_source(
                    current_source=repair_context.get("current_source_files", {}),
                    repaired_source=_source_files(response),
                    failure=repair_context,
                    compiled_mechanism=compiled_implementation,
                    exact_parent_bundle=(
                        gateway_request.get("exact_parent_bundle")
                        if gateway_request["service_policy"]["response_mode"]
                        == PROFILE_MODEL_HOOKS_RESPONSE_MODE else None
                    ),
                )
                response = canonical_value(
                    {
                        **dict(response),
                        "files": [
                            {"content": content, "path": path}
                            for path, content in scoped_source.items()
                        ],
                    }
                )
            except ValueError as error:
                current_repair_source = repair_context.get("current_source_files", {})
                if not isinstance(current_repair_source, Mapping):
                    current_repair_source = {}
                response = canonical_value(
                    {
                        **dict(response),
                        "files": [
                            {"content": str(content), "path": str(path)}
                            for path, content in current_repair_source.items()
                        ],
                    }
                )
                scope_reason = str(error).split(":", 1)[0]
                if scope_reason not in {
                    "MECHANICAL_REPAIR_DEPENDENCY_UNRESOLVED",
                    "MECHANICAL_REPAIR_SCOPE_DRIFT",
                    "MECHANICAL_REPAIR_SYMBOL_MAPPING_AMBIGUOUS",
                }:
                    scope_reason = "MECHANICAL_REPAIR_SCOPE_DRIFT"
                attempt_record = canonical_value(
                    {
                        "attempt": len(attempts),
                        "repair_attempt": recorded_repair_attempt,
                        "producer_role": selected_outcome.producer_role,
                        "spec_digest": selected_outcome.spec.digest,
                        "candidate_id": blind_candidate_id,
                        "mechanism_semantics_digest": semantic_identity_digest,
                        "effective_experiment_digest": effective_experiment_digest,
                        "effective_family_digest": effective_family_digest,
                        "primitive_ids": effective_identity["primitive_ids"],
                        "idea_acquisition": idea_acquisition.to_dict(),
                        "failure": {
                            "stage": "STATIC_VALIDATION",
                            "failure_class": "IMPLEMENTATION",
                            "reason_code": scope_reason,
                            "message": str(error),
                        },
                    }
                )
                attempts.append(attempt_record)
                candidate_attempts.append(attempt_record)
                continue
        attempt_parent = _create_candidate_attempt_parent(
            inputs.candidate_parent,
            attempt_index,
        )
        candidate_root = attempt_parent / blind_candidate_id
        root_ref = content_id(
            "recclaw-innovation-candidate-root-v1",
            {
                "campaign_id": current_profile.campaign_id,
                "round": attempt_index,
                "blind_candidate_id": blind_candidate_id,
            },
        )
        try:
            fixture = inputs.fixture_factory(implementer_policy, attempt_index, candidate_root)
            unit_check = inputs.unit_check_factory(implementer_policy)
            if inputs.resource_admission_required or inputs.native_training_cadence is not None:
                materialized = materialize_candidate_package(
                    implementation_spec,
                    policy=implementer_policy,
                    implementation_response=response,
                    candidate_root=candidate_root,
                    candidate_root_ref=root_ref,
                    compiled_mechanism=compiled_implementation,
                    exact_parent_bundle=selected_parent_bundle,
                    research_feedback=implementation_feedback,
                )
                if recorded_repair_attempt > 0 and not parent_scoped_patch:
                    # Judge a full-source repair after the authoritative binder,
                    # before spending another qualification/probe. Raw omission
                    # of machine-owned source is not a computational change.
                    previous_source = gateway_request["repair_context"]["current_source_files"]
                    actual_source = {
                        path: (candidate_root / path).read_text(encoding="utf-8")
                        for path in materialized.package.allowed_files
                    }
                    if actual_source == previous_source:
                        raise InnovationSpineError(
                            failure_class="IMPLEMENTATION",
                            reason_code="MECHANICAL_REPAIR_SCOPE_DRIFT",
                            message="MECHANICAL_REPAIR_NO_EFFECT: materialized revision is unchanged",
                        )
                qualify = inputs.qualification_executor or MechanicalRecBoleAdapterV1().qualify_disposable
                qualification = qualify(
                    materialized.package,
                    research_spec=implementation_spec,
                    candidate_root=candidate_root,
                    fixture=fixture,
                    # The disposable adapter already runs the shared model/API
                    # unit stage. Arbitrary caller closures are intentionally
                    # not executed in the spawned process.
                    unit_check=None,
                )
            else:
                materialized, qualification = qualify_local_innovation_candidate(
                    implementation_spec,
                    policy=implementer_policy,
                    implementation_response=response,
                    candidate_root=candidate_root,
                    candidate_root_ref=root_ref,
                    fixture=fixture,
                    unit_check=unit_check,
                    compiled_mechanism=compiled_implementation,
                    exact_parent_bundle=selected_parent_bundle,
                    research_feedback=implementation_feedback,
                )
            failure = qualification.failure_detail
            if not parent_scoped_patch:
                # Repairs operate on the code qualification actually executed,
                # including compiler-owned bindings, not the raw Provider draft.
                # This same response also supplies the resource-repair boundary.
                response = canonical_value(
                    {
                        **dict(response),
                        "files": [
                            {
                                "path": path,
                                "content": (candidate_root / path).read_text(
                                    encoding="utf-8"
                                ),
                            }
                            for path in materialized.package.allowed_files
                        ],
                    }
                )
        except InnovationSpineError as error:
            materialized = None
            qualification = None
            failure = _materialization_failure(error)
        attempt_record = canonical_value(
            {
                "attempt": attempt_index,
                "repair_attempt": recorded_repair_attempt,
                "producer_role": selected_outcome.producer_role,
                "spec_digest": selected_outcome.spec.digest,
                "core_mechanism_contrast": selected_outcome.spec.mechanism_change,
                "next_discriminative_task": selected_outcome.spec.falsifier,
                "candidate_id": blind_candidate_id,
                "mechanism_semantics_digest": semantic_identity_digest,
                "effective_experiment_digest": effective_experiment_digest,
                "effective_family_digest": effective_family_digest,
                "primitive_ids": effective_identity["primitive_ids"],
                "idea_acquisition": idea_acquisition.to_dict(),
                "candidate_root": str(candidate_root),
                "qualification": (
                    qualification.to_dict() if qualification is not None else None
                ),
                "failure": failure,
            }
        )
        attempts.append(attempt_record)
        candidate_attempts.append(attempt_record)
        if (
            qualification is not None
            and qualification.receipt.status is QualificationStatusV1.PASS
        ):
            break
        if failure is None or not is_mechanical_repair_failure(failure):
            break

    if (
        materialized is None
        or qualification is None
        or qualification.receipt.status is not QualificationStatusV1.PASS
        or candidate_root is None
    ):
        failed_result = InnovationLaneResult(
            selected_outcome=selected_outcome,
            resolution=resolution,
            idea_acquisition=idea_acquisition,
            attempts=tuple(attempts),
            materialized=materialized,
            qualification=qualification,
            capability=None,
            registry=None,
            profile_manifest=None,
            next_profile=None,
            profile_receipt=None,
            qualified_execution=None,
            search_candidate=None,
            candidate_root=(str(candidate_root) if candidate_root is not None else None),
            fresh_campaign_id=inputs.fresh_campaign_id,
        )
        remaining = tuple(
            pair
            for pair in candidates
            if pair[0].digest != selected_outcome.digest
        )
        if (
            allow_candidate_reroute
            and remaining
            and _reroute_index + 1 < candidate_attempt_limit
            and _innovation_provider_failure(failed_result) is None
        ):
            reroute_suffix = f"candidate-reroute-{_reroute_index + 1:02d}"
            reroute_inputs = replace(
                inputs,
                candidate_parent=inputs.candidate_parent / reroute_suffix,
                resource_probe_parent=(
                    inputs.resource_probe_parent / reroute_suffix
                    if inputs.resource_probe_parent is not None
                    else None
                ),
            )
            next_result = _run_innovation_lane(
                remaining,
                current_profile=current_profile,
                current_slate_ref=current_slate_ref,
                current_slate_digest=current_slate_digest,
                research_policy=research_policy,
                search_space_adapter=search_space_adapter,
                research_context=research_context,
                inputs=reroute_inputs,
                evaluator=evaluator,
                split=split,
                frozen_profile_ref=frozen_profile_ref,
                implementation_call_limit=implementation_call_limit,
                candidate_attempt_limit=candidate_attempt_limit,
                allow_candidate_reroute=True,
                enforce_unbound_confirmation=enforce_unbound_confirmation,
                outcome_selector=outcome_selector,
                attempted_semantic_identities=(
                    attempted_semantic_identities | {semantic_identity_digest}
                ),
                attempted_effective_experiment_identities=(
                    attempted_effective_experiment_identities
                    | {effective_experiment_digest}
                ),
                _prior_attempts=failed_result.attempts,
                _reroute_index=_reroute_index + 1,
                exact_parent_bundle=exact_parent_bundle,
                exact_parent_bundle_loader=exact_parent_bundle_loader,
            )
            return next_result if next_result is not None else failed_result
        return failed_result

    capability, registry = (
        (_qualified_retry.capability, _qualified_retry.registry)
        if _qualified_retry is not None
        else admit_local_qualification(
            implementation_spec,
            materialized,
            qualification,
            capability_kind=inputs.capability_kind,
            capability_version=inputs.capability_version,
            semantic_identity_ref=semantic_identity_ref,
            semantic_identity_digest=semantic_identity_digest,
            registry_version=inputs.registry_version,
            predecessor_registry_ref=inputs.predecessor_registry_ref,
            predecessor_registry_digest=inputs.predecessor_registry_digest,
        )
    )
    qualified_execution = (
        dict(_qualified_retry.qualified_execution)
        if _qualified_retry is not None
        else _qualified_execution(implementer_policy, materialized, capability)
    )
    if inputs.native_training_cadence is not None:
        # Protocol configuration, not a measured resource prediction. Keep it
        # with the durable qualified candidate so carryover/recovery retains it.
        qualified_execution["native_training_cadence"] = dict(inputs.native_training_cadence)
    resource_profile: Mapping[str, Any] | None = None
    quality_admission: Mapping[str, Any] | None = None
    if inputs.resource_probe is not None:
        assert inputs.resource_probe_parent is not None
        source_relative = (
            materialized.package.executable_entrypoint.split(":", 1)[0]
            .replace(".", "/")
            + ".py"
        )
        source_path = candidate_root / source_relative
        mechanism_id = str(prepared_innovation["mechanism_id"])
        probe_recipe = _resource_probe_recipe(
            current_profile=current_profile,
            capability=capability,
            qualified_execution=qualified_execution,
            entrypoint=materialized.package.executable_entrypoint,
            mechanism_id=mechanism_id,
            mechanism_semantics_digest=semantic_identity_digest,
            evaluator=evaluator,
            split=split,
        )
        probe_root = (
            inputs.resource_probe_parent
            / f"probe-{resource_probe_execution_key(probe_recipe)[:16]}"
        )
        resource_profile = canonical_value(
            dict(
                inputs.resource_probe(
                    candidate_root=candidate_root,
                    source_path=source_path,
                    entrypoint=materialized.package.executable_entrypoint,
                    source_sha256=str(
                        qualified_execution["entrypoint_source_sha256"]
                    ),
                    execution_recipe=probe_recipe,
                    probe_root=probe_root,
                    candidate_ref=capability.capability_id,
                    candidate_package_digest=capability.candidate_package_digest,
                )
            )
        )
        attempts[-1] = canonical_value({
            **dict(attempts[-1]), "resource_probe_root": str(probe_root),
        })
        resource_structural_context = dict(inputs.resource_structural_context or {})
        # The old timeout constrains a repair of that mechanism, not fresh research.
        prior_repair_semantic = resource_structural_context.get("implementation_efficiency_repair_semantic_digest")
        if isinstance(prior_repair_semantic, str) and prior_repair_semantic and prior_repair_semantic != semantic_identity_digest:
            resource_structural_context.pop("implementation_efficiency_repair_context", None)
        resource_structural_context["candidate_semantic_digest"] = (
            semantic_identity_digest
        )
        quality_admission = admit_research_innovation_candidate(
            spec=implementation_spec,
            resolution=resolution,
            qualification=qualification,
            resource_profile=resource_profile,
            structural_context=resource_structural_context,
            semantic_duplicate=any(
                entry.semantic_identity_digest == semantic_identity_digest
                for entry in current_profile.entries
            ),
        )
        if (
            inputs.resource_admission_required
            and quality_admission.get("admitted") is not True
        ):
            performance_failure = _resource_admission_failure(
                resource_profile=resource_profile,
                quality_admission=quality_admission,
            )
            attempts[-1] = canonical_value(
                {
                    **dict(attempts[-1]),
                    "resource_profile": resource_profile,
                    "quality_admission": quality_admission,
                    "failure": performance_failure,
                }
            )
            blocked_result = InnovationLaneResult(
                selected_outcome=selected_outcome,
                resolution=resolution,
                idea_acquisition=idea_acquisition,
                attempts=tuple(attempts),
                materialized=materialized,
                qualification=qualification,
                capability=capability,
                registry=registry,
                profile_manifest=None,
                next_profile=None,
                profile_receipt=None,
                qualified_execution=qualified_execution,
                search_candidate=None,
                candidate_root=str(candidate_root),
                fresh_campaign_id=inputs.fresh_campaign_id,
                resource_profile=resource_profile,
                quality_admission=quality_admission,
            )
            if (
                _innovation_resource_failure_scope(blocked_result)
                in _RECOVERABLE_RESOURCE_FAILURE_SCOPES
            ):
                # Infrastructure did not evaluate the candidate.  Keep this
                # exact outcome as the recoverable generation boundary instead
                # of spending the opportunity on an Implementer rewrite or peer.
                return blocked_result
            if (
                performance_failure is not None
                and is_mechanical_repair_failure(performance_failure)
                and len(candidate_attempts) < implementation_call_limit
            ):
                if _qualified_retry is not None:
                    # A newly measured candidate-local cost problem is a real
                    # implementation repair, using today's ownership contract.
                    request = build_shared_implementer_request(
                        implementation_spec, policy=implementer_policy,
                        compiled_mechanism=compiled_implementation,
                        exact_parent_bundle=selected_parent_bundle,
                        research_feedback=implementation_feedback,
                    )
                resource_repair_request = build_mechanical_repair_request(
                    request,
                    performance_failure,
                    current_source=_source_files(response or {}),
                    repair_attempt=len(candidate_attempts),
                )
                prediction = resource_profile.get("prediction")
                prior_epoch_ms = (
                    prediction.get("estimated_epoch_wall_time_ms")
                    if isinstance(prediction, Mapping)
                    else None
                )
                repair_structural_context = {
                    "implementation_efficiency_repair_context": {
                        "schema": (
                            "recclaw.research-line.implementation-efficiency-"
                            "repair-context.v1"
                        ),
                        "reason_code": "MEASURED_WORKER_RESOURCE_CEILING",
                        "next_attempt_scope": "IMPLEMENTATION_EFFICIENCY_ONLY",
                        "mean_train_epoch_wall_time_ms": prior_epoch_ms,
                        "mean_eval_epoch_wall_time_ms": 0.0,
                    },
                    "implementation_efficiency_repair_spec_digest": (
                        selected_outcome.spec.digest
                    ),
                    "implementation_efficiency_repair_semantic_digest": (
                        semantic_identity_digest
                    ),
                }
                repair_inputs = replace(
                    inputs,
                    candidate_parent=inputs.candidate_parent
                    / "resource-performance-repair",
                    resource_probe_parent=(
                        inputs.resource_probe_parent
                        / "resource-performance-repair"
                        if inputs.resource_probe_parent is not None
                        else None
                    ),
                    resource_structural_context=repair_structural_context,
                )
                repaired_result = _run_innovation_lane(
                    ((selected_outcome, resolution),),
                    current_profile=current_profile,
                    current_slate_ref=current_slate_ref,
                    current_slate_digest=current_slate_digest,
                    research_policy=research_policy,
                    search_space_adapter=search_space_adapter,
                    research_context=research_context,
                    inputs=repair_inputs,
                    evaluator=evaluator,
                    split=split,
                    frozen_profile_ref=frozen_profile_ref,
                    implementation_call_limit=1,
                    candidate_attempt_limit=1,
                    allow_candidate_reroute=False,
                    enforce_unbound_confirmation=enforce_unbound_confirmation,
                    outcome_selector=outcome_selector,
                    attempted_semantic_identities=attempted_semantic_identities,
                    attempted_effective_experiment_identities=(
                        attempted_effective_experiment_identities
                    ),
                    _prior_attempts=blocked_result.attempts,
                    _reroute_index=_reroute_index,
                    _forced_first_gateway_request=resource_repair_request,
                    exact_parent_bundle=exact_parent_bundle,
                    exact_parent_bundle_loader=exact_parent_bundle_loader,
                )
                if repaired_result is not None and repaired_result.activation_ready:
                    return repaired_result
                if repaired_result is not None:
                    blocked_result = repaired_result
                if _innovation_provider_failure(blocked_result) is not None:
                    return blocked_result
            remaining = tuple(
                pair
                for pair in candidates
                if pair[0].digest != selected_outcome.digest
            )
            if (
                allow_candidate_reroute
                and remaining
                and _reroute_index + 1 < candidate_attempt_limit
            ):
                reroute_suffix = f"candidate-reroute-{_reroute_index + 1:02d}"
                reroute_inputs = replace(
                    inputs,
                    candidate_parent=inputs.candidate_parent / reroute_suffix,
                    resource_probe_parent=(
                        inputs.resource_probe_parent / reroute_suffix
                        if inputs.resource_probe_parent is not None
                        else None
                    ),
                )
                next_result = _run_innovation_lane(
                    remaining,
                    current_profile=current_profile,
                    current_slate_ref=current_slate_ref,
                    current_slate_digest=current_slate_digest,
                    research_policy=research_policy,
                    search_space_adapter=search_space_adapter,
                    research_context=research_context,
                    inputs=reroute_inputs,
                    evaluator=evaluator,
                    split=split,
                    frozen_profile_ref=frozen_profile_ref,
                    implementation_call_limit=implementation_call_limit,
                    candidate_attempt_limit=candidate_attempt_limit,
                    allow_candidate_reroute=True,
                    enforce_unbound_confirmation=enforce_unbound_confirmation,
                    outcome_selector=outcome_selector,
                    attempted_semantic_identities=(
                        attempted_semantic_identities | {semantic_identity_digest}
                    ),
                    attempted_effective_experiment_identities=(
                        attempted_effective_experiment_identities
                        | {effective_experiment_digest}
                    ),
                    _prior_attempts=blocked_result.attempts,
                    _reroute_index=_reroute_index + 1,
                    exact_parent_bundle=exact_parent_bundle,
                    exact_parent_bundle_loader=exact_parent_bundle_loader,
                )
                return next_result if next_result is not None else blocked_result
            return blocked_result
    if quality_admission is not None:
        utility = dict(quality_admission["utility_features"])
        feature_evidence = dict(quality_admission["feature_evidence"])
    elif source_proposal is None:
        default_utility, default_evidence = _qualified_open_features(
            selected_outcome,
            parent_available=bool(prepared_innovation["parent_available"]),
        )
        utility = default_utility.to_dict()
        feature_evidence = default_evidence.to_dict()
    else:
        utility = None
        feature_evidence = None
    packaged = search_space_adapter.resolve_confirmation(
        "INNOVATION",
        selected_outcome,
        {
            "phase": "PACKAGE_QUALIFIED_INNOVATION",
            "prepared_innovation": prepared_innovation,
            "materialized": materialized,
            "qualification": qualification,
            "capability": capability,
            "implementer_policy": implementer_policy,
            "mechanism_axis": _open_mechanism_axis(
                selected_outcome,
                research_context=research_context,
            ),
            "utility_features": utility,
            "feature_evidence": feature_evidence,
        },
    )
    if (
        packaged.kind is not ConfirmationResolutionKindV1.EXACT_BINDING
        or not isinstance(packaged.binding, Mapping)
        or packaged.binding.get("search_candidate") is None
    ):
        raise ValueError("SearchSpaceAdapter could not package qualified Innovation")
    search_candidate = packaged.binding["search_candidate"]
    manifest, next_profile, receipt = build_local_next_fresh_profile(
        registry,
        profile_version=inputs.profile_version,
        predecessor_profile_ref=current_profile.profile_ref,
        predecessor_profile_digest=current_profile.profile_digest,
        current_campaign_slate_ref=current_slate_ref,
        current_campaign_slate_digest=current_slate_digest,
        predecessor_executable_entries=predecessor_executable_entries(current_profile),
        compatibility_requirements=selected_outcome.spec.compatibility_requirements,
    )
    return InnovationLaneResult(
        selected_outcome=selected_outcome,
        resolution=resolution,
        idea_acquisition=idea_acquisition,
        attempts=tuple(attempts),
        materialized=materialized,
        qualification=qualification,
        capability=capability,
        registry=registry,
        profile_manifest=manifest,
        next_profile=next_profile,
        profile_receipt=receipt,
        qualified_execution=qualified_execution,
        search_candidate=search_candidate,
        candidate_root=str(candidate_root),
        fresh_campaign_id=inputs.fresh_campaign_id,
        resource_profile=resource_profile,
        quality_admission=quality_admission,
    )


def _initial_pool_failure_summaries(
    attempts: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Keep one concise final qualification/resource failure per candidate."""

    by_spec: dict[str, Mapping[str, Any]] = {}
    for attempt in attempts:
        spec_digest = attempt.get("spec_digest")
        failure = attempt.get("failure")
        if spec_digest is None or not isinstance(failure, Mapping):
            continue
        by_spec[str(spec_digest)] = canonical_value(
            {
                "producer_role": attempt.get("producer_role"),
                "spec_digest": str(spec_digest),
                "candidate_id": attempt.get("candidate_id"),
                "failure": {
                    key: failure.get(key)
                    for key in (
                        "stage",
                        "failure_class",
                        "reason_code",
                        "message",
                    )
                    if failure.get(key) is not None
                },
            }
        )
    return tuple(by_spec[key] for key in sorted(by_spec))


def stage_initial_search_pool(
    *,
    context: ResearchContext,
    bootstrap_profile: SearchExecutableProfileV1,
    final_campaign_id: str,
    producer: ResearchProducer,
    producer_bindings: Mapping[str, Any],
    resolver_environment: Mapping[str, Any],
    policy: VersionedResearchPolicyV1,
    search_space_adapter: SearchSpaceAdapter,
    innovation_inputs: InnovationRuntimeInputs,
    evaluator: Mapping[str, Any],
    split: str,
    frozen_profile_ref: Mapping[str, Any],
    implementation_call_limit: int,
    maximum_candidates: int = 4,
    minimum_candidates: int = 1,
) -> InitialSearchPoolResult:
    """Qualify a multi-candidate pool before counted round one.

    This is the explicit Innovation -> next-cycle boundary required by the
    dual-lane architecture.  It performs Provider ideation, implementation,
    qualification, and capability admission only.  It never invokes the
    experiment runner, interpreter, frontier update, or round counter.
    """

    knowledge_base = getattr(context, "knowledge_base", None)
    baseline_context = (
        knowledge_base.get("baseline_context")
        if isinstance(knowledge_base, Mapping)
        else None
    )
    if is_single_parent_context(baseline_context):
        raise ValueError(
            "single-parent search starts directly from its frozen parent and "
            "does not use the legacy multi-candidate bootstrap pool"
        )

    if final_campaign_id == bootstrap_profile.campaign_id:
        raise ValueError("bootstrap and counted campaign identities must differ")
    if minimum_candidates < 1 or maximum_candidates < minimum_candidates:
        raise ValueError("invalid initial Search pool bounds")

    innovations: list[InnovationLaneResult] = []
    failed_attempts: list[Mapping[str, Any]] = []
    attempted_spec_digests: set[str] = set()
    attempted_semantic_identities: set[str] = set()
    attempted_effective_experiment_identities: set[str] = set()
    batch_context = context
    batch_bindings = dict(producer_bindings)

    # The initial four-Producer batch retains its existing behavior.  When it
    # yields zero executable candidates, make one bounded same-round follow-up
    # batch with the real failures in shared Research memory.
    for batch_index in range(2):
        outcomes = produce_research_specs(
            batch_context,
            producer,
            batch_bindings,
            frozen_profile_ref=frozen_profile_ref,
        )
        resolved = resolve_producer_outcomes(
            outcomes,
            environment=resolver_environment,
        )
        remaining = [
            (outcome, resolution)
            for outcome, resolution in resolved
            if resolution is not None
            and resolution.resolution
            is CapabilityResolutionResultV1.INNOVATION_REQUIRED
            and outcome.spec is not None
            and outcome.spec.digest not in attempted_spec_digests
        ]
        bootstrap_slate = canonical_value(
            {
                "schema": "recclaw.initial-search-pool.v1",
                "context_digest": batch_context.digest,
                "profile_digest": bootstrap_profile.profile_digest,
                "candidate_count": len(remaining),
            }
        )
        slate_ref = content_id(
            "recclaw-initial-search-pool-v1", bootstrap_slate
        )
        slate_digest = sha256_digest(bootstrap_slate)
        batch_inputs = (
            innovation_inputs
            if batch_index == 0
            else replace(
                innovation_inputs,
                candidate_parent=(
                    innovation_inputs.candidate_parent
                    / "supplemental-batch-01"
                ),
                resource_probe_parent=(
                    innovation_inputs.resource_probe_parent
                    / "supplemental-batch-01"
                    if innovation_inputs.resource_probe_parent is not None
                    else None
                ),
            )
        )
        batch_target = (
            maximum_candidates if batch_index == 0 else minimum_candidates
        )

        while remaining and len(innovations) < batch_target:
            index = len(innovations) + 1
            candidate_parent = (
                batch_inputs.candidate_parent
                / f"bootstrap-candidate-{index:02d}"
            )
            resource_probe_parent = (
                batch_inputs.resource_probe_parent
                / f"bootstrap-candidate-{index:02d}"
                if batch_inputs.resource_probe_parent is not None
                else None
            )
            staged_inputs = replace(
                batch_inputs,
                candidate_parent=candidate_parent,
                resource_probe_parent=resource_probe_parent,
                capability_version=(
                    f"{final_campaign_id}:bootstrap-capability-{index:02d}-v1"
                ),
                registry_version=(
                    f"{final_campaign_id}:bootstrap-single-registry-{index:02d}-v1"
                ),
                predecessor_registry_ref=bootstrap_profile.profile_ref,
                predecessor_registry_digest=bootstrap_profile.profile_digest,
                profile_version=(
                    f"{final_campaign_id}:bootstrap-single-profile-{index:02d}-v1"
                ),
                fresh_campaign_id=(
                    f"{final_campaign_id}:bootstrap-qualified-{index:02d}"
                ),
            )
            innovation = _run_innovation_lane(
                remaining,
                current_profile=bootstrap_profile,
                current_slate_ref=slate_ref,
                current_slate_digest=slate_digest,
                research_policy=policy,
                search_space_adapter=search_space_adapter,
                research_context=batch_context,
                inputs=staged_inputs,
                evaluator=evaluator,
                split=split,
                frozen_profile_ref=frozen_profile_ref,
                implementation_call_limit=implementation_call_limit,
                candidate_attempt_limit=len(remaining),
                allow_candidate_reroute=True,
                enforce_unbound_confirmation=False,
                attempted_semantic_identities=frozenset(
                    attempted_semantic_identities
                ),
                attempted_effective_experiment_identities=frozenset(
                    attempted_effective_experiment_identities
                ),
            )
            if innovation is None:
                break
            failed_attempts.extend(innovation.attempts)
            for item in innovation.attempts:
                if _innovation_attempt_proposal_digest(item) is not None:
                    attempted_spec_digests.add(_innovation_attempt_proposal_digest(item))
                if item.get("mechanism_semantics_digest") is not None:
                    attempted_semantic_identities.add(
                        str(item["mechanism_semantics_digest"])
                    )
                if item.get("effective_experiment_digest") is not None:
                    attempted_effective_experiment_identities.add(
                        str(item["effective_experiment_digest"])
                    )
            remaining = [
                pair
                for pair in remaining
                if pair[0].spec is not None
                and pair[0].spec.digest not in attempted_spec_digests
            ]
            if not innovation.activation_ready:
                continue
            innovations.append(innovation)

        if len(innovations) >= minimum_candidates or batch_index == 1:
            break
        failure_summaries = _initial_pool_failure_summaries(failed_attempts)
        memory = dict(context.scientific_memory)
        global_memory = dict(memory.get("global_memory", {}))
        global_memory["initial_pool_failure_feedback"] = failure_summaries
        batch_context = replace(
            context,
            scientific_memory={
                **memory,
                "discovery_generation": batch_index + 1,
                "global_memory": global_memory,
            },
        )
        batch_bindings = {
            **dict(producer_bindings),
            "context_ref": batch_context.context_ref,
            "context_digest": batch_context.digest,
            "protocol_ref": batch_context.protocol_ref,
            "protocol_digest": batch_context.protocol_digest,
            "current_profile_ref": batch_context.active_profile_ref,
            "current_profile_digest": batch_context.active_profile_digest,
        }

    if len(innovations) < minimum_candidates:
        return InitialSearchPoolResult(
            active_profile=bootstrap_profile,
            proposals=(),
            open_candidates=(),
            qualified_execution_by_capability={},
            candidate_root_by_capability={},
            resource_profile_by_capability={},
            innovations=(),
        )

    capabilities = tuple(
        innovation.capability
        for innovation in innovations
        if innovation.capability is not None
    )
    registry = VersionedCapabilityRegistry.build(
        registry_version=f"{final_campaign_id}:bootstrap-registry-v1",
        predecessor_registry_ref=bootstrap_profile.profile_ref,
        predecessor_registry_digest=bootstrap_profile.profile_digest,
        protocol_ref=bootstrap_profile.protocol_ref,
        protocol_digest=bootstrap_profile.protocol_digest,
        capabilities=capabilities,
    )
    compatibility_requirements = tuple(
        sorted(
            {
                requirement
                for innovation in innovations
                for requirement in (
                    innovation.selected_outcome.spec.compatibility_requirements
                    if innovation.selected_outcome.spec is not None
                    else ()
                )
            }
        )
    )
    _manifest, next_profile, _receipt = build_local_next_fresh_profile(
        registry,
        profile_version=f"{final_campaign_id}:bootstrap-profile-v1",
        predecessor_profile_ref=bootstrap_profile.profile_ref,
        predecessor_profile_digest=bootstrap_profile.profile_digest,
        current_campaign_slate_ref=slate_ref,
        current_campaign_slate_digest=slate_digest,
        predecessor_executable_entries=predecessor_executable_entries(
            bootstrap_profile
        ),
        compatibility_requirements=compatibility_requirements,
    )
    active_profile = activate_next_fresh_search_profile(
        predecessor=bootstrap_profile,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=final_campaign_id,
    )

    proposals: list[CandidateProposalV4] = []
    open_candidates: list[QualifiedSearchCandidateProtocolV1] = []
    qualified: dict[str, Mapping[str, Any]] = {}
    roots: dict[str, str] = {}
    resources: dict[str, Mapping[str, Any]] = {}
    for innovation in innovations:
        assert innovation.capability is not None
        assert innovation.qualified_execution is not None
        assert innovation.search_candidate is not None
        assert innovation.candidate_root is not None
        capability_ref = innovation.capability.capability_id
        qualified[capability_ref] = innovation.qualified_execution
        roots[capability_ref] = innovation.candidate_root
        if innovation.resource_profile is not None:
            resources[capability_ref] = innovation.resource_profile
        if isinstance(innovation.search_candidate, CandidateProposalV4):
            proposals.append(innovation.search_candidate)
        else:
            open_candidates.append(innovation.search_candidate)

    return InitialSearchPoolResult(
        active_profile=active_profile,
        proposals=tuple(proposals),
        open_candidates=tuple(open_candidates),
        qualified_execution_by_capability=canonical_value(qualified),
        candidate_root_by_capability=canonical_value(roots),
        resource_profile_by_capability=canonical_value(resources),
        innovations=tuple(innovations),
    )


def _closed_meta_effect_evidence(
    interpretation: EpisodeInterpretation | MissingSearchInterpretation,
    aggregate: Mapping[str, Any],
) -> bool:
    quality = aggregate.get("producer_quality_scores")
    return (
        isinstance(interpretation, EpisodeInterpretation)
        and interpretation.episode is not None
        and interpretation.episode.experiment_executed
        and interpretation.closure.outcome_ref is not None
        and interpretation.closure.outcome_digest is not None
        and isinstance(quality, Mapping)
        and bool(quality)
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in quality.values()
        )
    )


def _run_meta_research(
    *,
    context: ResearchContext,
    interpretation: EpisodeInterpretation | MissingSearchInterpretation,
    inputs: MetaResearchInputs,
) -> MetaResearchResult:
    champion = interpretation.policy_successor
    search_memory = interpretation.search_memory_snapshot
    outcome_aggregate = dict(champion.acquisition_parameters)
    proposal = build_meta_update_proposal(
        policy=champion,
        search_memory=search_memory,
        outcome_aggregate=outcome_aggregate,
    )
    challenger = materialize_proposed_control_policy(
        parent=champion,
        proposal=proposal,
    )
    has_effect_evidence = _closed_meta_effect_evidence(
        interpretation,
        outcome_aggregate,
    )
    decision = MetaControlPromotionDecisionV1(
        proposal_digest=proposal.digest,
        evaluation_digest=sha256_digest(
            {
                "proposal_digest": proposal.digest,
                "producer_quality_scores": outcome_aggregate.get(
                    "producer_quality_scores"
                ),
            }
        ),
        parent_policy_digest=champion.digest,
        challenger_policy_digest=challenger.digest,
        verdict="PROMOTE" if has_effect_evidence else "HOLD",
        activation_boundary="NEXT_CAMPAIGN",
    )
    activated_policy = None
    activation_receipt = None
    if decision.verdict == "PROMOTE":
        activated_policy, activation_receipt = activate_promoted_control_policy(
            parent=champion,
            proposal=proposal,
            decision=decision,
            campaign_id=inputs.next_campaign_id,
        )
    return MetaResearchResult(
        update_proposal=proposal,
        challenger_policy=challenger,
        offline_replay=None,
        shadow_evaluation=None,
        promotion_decision=decision,
        activated_policy=activated_policy,
        activation_receipt=activation_receipt,
    )


def _validate_experiment_binding(
    binding: ExperimentBindingV1,
    *,
    recipe: Mapping[str, Any],
    observation_seed: str,
) -> None:
    expected_recipe_digest = sha256_digest(recipe)
    if binding.execution_recipe_digest != expected_recipe_digest:
        raise ValueError("runner Experiment Binding is not bound to the selected recipe")
    if str(binding.seed) != observation_seed:
        raise ValueError("runner Experiment Binding seed differs from the frozen seed")
    recipe_fields = (
        "capability_family",
        "capability_ref",
        "capability_digest",
        "profile_ref",
        "profile_digest",
        "mechanism_id",
        "candidate_package_ref",
        "candidate_package_digest",
        "candidate_root_ref",
        "candidate_root_digest",
        "candidate_source_tree_digest",
        "entrypoint",
        "entrypoint_source_sha256",
        "model",
        "base_model_config",
        "config",
        "dataset",
        "split",
        "evaluator",
        "execution_role",
        "comparator_ref",
        "comparator_digest",
    )
    mismatched = [
        field_name
        for field_name in recipe_fields
        if canonical_value(getattr(binding, field_name))
        != canonical_value(recipe.get(field_name))
    ]
    for field_name in ("execution_purpose", "run_id"):
        if field_name in recipe and getattr(binding, field_name) != recipe[field_name]:
            mismatched.append(field_name)
    if "seed" in recipe and int(recipe["seed"]) != binding.seed:
        mismatched.append("seed")
    if mismatched:
        raise ValueError(
            "runner Experiment Binding differs from the selected recipe for: "
            + ", ".join(mismatched)
        )


def _guard_response(
    value: Any,
    *,
    stage: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Normalize one bridge response without importing the bridge package."""

    if isinstance(value, Mapping):
        response = dict(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        response = dict(to_dict()) if callable(to_dict) else {}
    if not response:
        raise ValueError(f"Evidence Guard {stage} response is not a mapping")
    returned_candidate_id = response.get("candidate_id")
    if returned_candidate_id is not None and str(returned_candidate_id) != candidate_id:
        raise ValueError(f"Evidence Guard {stage} candidate identity drift")
    returned_stage = response.get("stage")
    if returned_stage is not None:
        returned_stage = getattr(returned_stage, "value", returned_stage)
        if str(returned_stage) != stage:
            raise ValueError(f"Evidence Guard {stage} response stage drift")
    response.setdefault("candidate_id", candidate_id)
    response.setdefault("stage", stage)
    response["status"] = str(
        getattr(response.get("status", ""), "value", response.get("status", ""))
    ).upper()
    return canonical_value(response)


def _guard_protocol_drift(response: Mapping[str, Any]) -> bool:
    # V13's nested current-attempt classification is authoritative.  A
    # protocol attestation mismatch can accompany an engineering or invalid
    # attempt and must not be relabelled as protocol drift here.
    projection = _guard_control_projection(response)
    summary = projection.get("evidence_summary") if projection is not None else None
    if isinstance(summary, Mapping):
        current_attempt = summary.get("current_attempt_class")
        if isinstance(current_attempt, str) and current_attempt.strip():
            return current_attempt.upper() == "PROTOCOL_DRIFT"

    # Only the attestation's actual failed checks are a valid legacy fallback;
    # status aliases and router risk are not protocol evidence.
    attestation = response.get("protocol_attestation")
    if isinstance(attestation, Mapping):
        failed_checks = attestation.get("failed_checks")
        if isinstance(failed_checks, (tuple, list, set, frozenset)):
            protocol_checks = {
                "dataset",
                "dataset_manifest",
                "split",
                "evaluator",
                "seed",
                "epochs",
                "comparator",
                "execution_role",
            }
            return bool(protocol_checks.intersection(str(item) for item in failed_checks))
    return False


def _guard_control_projection(response: Mapping[str, Any]) -> Mapping[str, Any] | None:
    projection = response.get("control_projection")
    return projection if isinstance(projection, Mapping) else None


def _guard_contract_complete(response: Mapping[str, Any]) -> bool:
    """Require the V13 control projection before any scientific credit."""

    projection = _guard_control_projection(response)
    if projection is None:
        return False
    summary = projection.get("evidence_summary")
    promotion = projection.get("development_promotion")
    if not isinstance(summary, Mapping) or not isinstance(promotion, str):
        return False
    conclusion = summary.get("scientific_conclusion_strength")
    confidence_weight = projection.get("confidence_weight")
    evidence_count = summary.get("evidence_count")
    required_count = summary.get("required_seed_count")
    current_attempt = summary.get("current_attempt_class")
    return (
        isinstance(conclusion, str)
        and bool(conclusion.strip())
        and isinstance(evidence_count, int)
        and not isinstance(evidence_count, bool)
        and isinstance(required_count, int)
        and not isinstance(required_count, bool)
        and required_count > 0
        and isinstance(confidence_weight, (int, float))
        and not isinstance(confidence_weight, bool)
        and math.isfinite(float(confidence_weight))
        and 0.0 <= float(confidence_weight) <= 1.0
        and bool(promotion.strip())
        and isinstance(current_attempt, str)
        and current_attempt
        in {"VALID_METRIC", "ENGINEERING_FAILURE", "PROTOCOL_DRIFT", "INVALID"}
    )


def _guard_confidence_weight(response: Mapping[str, Any]) -> float:
    projection = _guard_control_projection(response)
    summary = projection.get("evidence_summary") if projection is not None else None
    if not isinstance(projection, Mapping) or not isinstance(summary, Mapping):
        return 0.0
    if summary.get("current_attempt_class") != "VALID_METRIC":
        return 0.0
    value = projection.get("confidence_weight")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return 0.0
    return float(value)


def _guard_promotion_allowed(response: Mapping[str, Any]) -> bool:
    projection = _guard_control_projection(response)
    if projection is None:
        return False
    summary = projection.get("evidence_summary")
    promotion = projection.get("development_promotion")
    if not isinstance(summary, Mapping) or not isinstance(promotion, str):
        return False
    state = summary.get("scientific_conclusion_strength")
    attempt = summary.get("current_attempt_class")
    return (
        state == "SUPPORTED"
        and attempt == "VALID_METRIC"
        and promotion == "ALLOW_DEVELOPMENT_PROMOTION"
    )


def _guard_claim_state(response: Mapping[str, Any]) -> str | None:
    projection = _guard_control_projection(response)
    summary = projection.get("evidence_summary") if projection is not None else None
    if (
        isinstance(summary, Mapping)
        and summary.get("scientific_conclusion_strength") is not None
    ):
        return str(summary["scientific_conclusion_strength"])
    return None


def _guard_current_attempt_class(response: Mapping[str, Any]) -> str | None:
    projection = _guard_control_projection(response)
    summary = projection.get("evidence_summary") if projection is not None else None
    if isinstance(summary, Mapping):
        value = summary.get("current_attempt_class")
        if isinstance(value, str) and value.strip():
            return value.upper()
    return None


def _guard_meta_update_allowed(response: Mapping[str, Any]) -> bool:
    """Keep native Research meta-learning for every exact valid metric."""

    fusion = response.get("fusion")
    if not isinstance(fusion, Mapping) or fusion.get("meta_update_allowed") is not True:
        return False
    return bool(
        _guard_promotion_allowed(response)
        or (
            response.get("research_update_mode")
            == "PRESERVE_NATIVE_RESEARCH"
            and _guard_current_attempt_class(response) == "VALID_METRIC"
        )
    )


def _guard_validation_directive(response: Mapping[str, Any]) -> Mapping[str, Any]:
    directive = response.get("validation_directive")
    if isinstance(directive, Mapping):
        return directive
    projection = _guard_control_projection(response)
    if not isinstance(projection, Mapping):
        return {}
    return {
        "action": projection.get("next_seed_allocation"),
        "next_seed": projection.get("next_seed"),
        "missing_seed_count": (
            projection.get("evidence_summary", {}).get("missing_seed_count", 1)
            if isinstance(projection.get("evidence_summary"), Mapping)
            else 1
        ),
    }


def _guard_next_seed_is_valid(
    *,
    next_seed: Any,
    current_seed: str,
    observation_seed_schedule: Sequence[str] | None,
) -> bool:
    if next_seed is None or observation_seed_schedule is None:
        return False
    schedule = tuple(str(item) for item in observation_seed_schedule)
    current = str(current_seed)
    candidate = str(next_seed)
    try:
        current_index = schedule.index(current)
        next_index = schedule.index(candidate)
    except ValueError:
        return False
    return next_index > current_index


def _guard_next_frozen_schedule_seed(
    *,
    current_seed: str,
    observation_seed_schedule: Sequence[str] | None,
) -> str | None:
    """Return the exact immediate successor in the frozen validation schedule."""

    if observation_seed_schedule is None:
        return None
    schedule = tuple(str(item) for item in observation_seed_schedule)
    current = str(current_seed)
    try:
        current_index = schedule.index(current)
    except ValueError:
        return None
    next_index = current_index + 1
    if next_index >= len(schedule):
        return None
    return schedule[next_index]


def _guard_seed_was_already_validated(
    *,
    context: ResearchContext,
    binding: SearchCandidateBindingV1,
    seed: Any,
) -> bool:
    candidate_seed = str(seed)
    memory = context.scientific_memory
    global_memory = memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    observations = global_memory.get(
        "executed_observations",
        memory.get("executed_observations", ()),
    )
    for observation in observations if isinstance(observations, (tuple, list)) else ():
        if (
            isinstance(observation, Mapping)
            and observation.get("candidate_semantic_digest")
            == binding.mechanism_semantics_digest
            and str(observation.get("observation_seed")) == candidate_seed
            and (
                observation.get("metric_bearing") is True
                or observation.get("evidence_domain") == "SCIENTIFIC_EPISODE"
            )
        ):
            return True
    search_observations = global_memory.get(
        "search_observations",
        memory.get("search_observations", ()),
    )
    for observation in (
        search_observations if isinstance(search_observations, (tuple, list)) else ()
    ):
        if (
            isinstance(observation, Mapping)
            and observation.get("candidate_semantic_digest")
            == binding.mechanism_semantics_digest
            and str(observation.get("observation_seed")) == candidate_seed
            and (
                observation.get("metric_bearing") is True
                or observation.get("evidence_domain") == "SCIENTIFIC_EPISODE"
            )
        ):
            return True
    raw_queue = global_memory.get("task_queue", memory.get("task_queue"))
    queue = ResearchTaskQueueV2.from_dict(raw_queue)
    return any(
        task.candidate_semantic_digest == binding.mechanism_semantics_digest
        and candidate_seed in task.evidence_present
        for task in queue.tasks
    )


def _guard_seed_advice_is_valid(
    *,
    response: Mapping[str, Any],
    context: ResearchContext,
    binding: SearchCandidateBindingV1,
    current_seed: str,
    observation_seed_schedule: Sequence[str] | None,
) -> bool:
    directive = _guard_validation_directive(response)
    action = str(directive.get("action", "NONE"))
    if action not in {"ENQUEUE_NEW_SEED", "NEXT_UNSEEN_SEED_SAME_BUDGET"}:
        return True
    next_seed = directive.get("next_seed")
    return _guard_next_seed_is_valid(
        next_seed=next_seed,
        current_seed=current_seed,
        observation_seed_schedule=observation_seed_schedule,
    ) and not _guard_seed_was_already_validated(
        context=context,
        binding=binding,
        seed=next_seed,
    )


def _unsupported_guard_seed_task(
    task: ResearchTaskRecordV2,
    *,
    directive: Mapping[str, Any],
    reason: str,
    observation_seed_schedule: Sequence[str] | None,
) -> ResearchTaskRecordV2:
    """Retain an unexecutable Guard seed request as a closed non-metric task."""

    return replace(
        task,
        metadata={
            **task.metadata,
            "task_resolution": canonical_value(
                {
                    "status": "CLOSED",
                    "reason": reason,
                    "evidence_class": "NON_METRIC_TASK_RESOLUTION",
                    "original_directive": directive,
                    "frozen_observation_seed_schedule": tuple(
                        str(seed) for seed in (observation_seed_schedule or ())
                    ),
                }
            ),
        },
    )


def _guard_task_from_response(
    *,
    response: Mapping[str, Any],
    context: ResearchContext,
    binding: SearchCandidateBindingV1,
    recipe: Mapping[str, Any],
    comparator_identity: str,
    confidence_weight: float,
    observation_seed: str,
    observation_seed_schedule: Sequence[str] | None,
) -> ResearchTaskRecordV2 | None:
    """Consume a complete V13 task, or materialize only same-candidate seed work.

    Matched-control and mechanism-off work must arrive with the already
    catalog-bound executable identity.  Research never turns a bare control
    label into executable code here; the existing binding/interpreter path
    remains authoritative for that resolution.
    """

    if isinstance(response.get("source_control_attribution"), Mapping):
        # A Guard control consumes one bounded discriminative task.  Its result
        # updates the source attribution state below; it must not recursively
        # create a validation lane for the control candidate.
        return None

    task_value: Any = response.get("research_task")
    if task_value is None:
        fusion = response.get("fusion")
        if isinstance(fusion, Mapping):
            task_value = fusion.get("research_task")
    if isinstance(task_value, Mapping):
        payload = dict(task_value)
        if payload.get("operation") is not None:
            operation = str(getattr(payload.get("operation"), "value", payload["operation"]))
            task = ResearchTaskRecordV2.from_dict(payload)
            if operation in {
                ResearchTaskOperationV2.NEW_SEED.value,
                ResearchTaskOperationV2.REPRODUCE.value,
            } and not _guard_next_seed_is_valid(
                next_seed=payload.get("required_seed_or_control"),
                current_seed=observation_seed,
                observation_seed_schedule=observation_seed_schedule,
            ):
                return _unsupported_guard_seed_task(
                    task,
                    directive=payload,
                    reason="UNSUPPORTED_OUTSIDE_FROZEN_SEED_SCHEDULE",
                    observation_seed_schedule=observation_seed_schedule,
                )
            if operation in {
                ResearchTaskOperationV2.NEW_SEED.value,
                ResearchTaskOperationV2.REPRODUCE.value,
            } and _guard_seed_was_already_validated(
                context=context,
                binding=binding,
                seed=payload.get("required_seed_or_control"),
            ):
                return _unsupported_guard_seed_task(
                    task,
                    directive=payload,
                    reason="UNSUPPORTED_SEED_ALREADY_VALIDATED",
                    observation_seed_schedule=observation_seed_schedule,
                )
            return task
        task_type = payload.get("task_type")
        operation_by_type = {
            "VALIDATE_SAME_CANDIDATE": ResearchTaskOperationV2.NEW_SEED.value,
            "RUN_MATCHED_CONTROL": ResearchTaskOperationV2.MATCHED_CONTROL.value,
            "RUN_ABLATION": ResearchTaskOperationV2.MECHANISM_OFF.value,
            "REPAIR_IMPLEMENTATION": ResearchTaskOperationV2.REPAIR.value,
            "PROTOCOL_BRANCH_DIAGNOSTIC": ResearchTaskOperationV2.MOVE_ON.value,
        }
        if task_type in operation_by_type:
            required = payload.get("required_seed_or_control")
            program = payload.get("mechanism_program")
            if not all(
                isinstance(payload.get(name), str)
                for name in (
                    "task_id",
                    "candidate_id",
                    "candidate_semantic_digest",
                    "mechanism_program_digest",
                    "comparator_identity",
                    "protocol_digest",
                )
            ) or not isinstance(required, str) or not isinstance(program, Mapping):
                raise ValueError(
                    "Evidence Guard typed task lacks complete executable binding"
                )
            operation = operation_by_type[task_type]
            task = ResearchTaskRecordV2.from_dict(
                {
                    **payload,
                    "operation": operation,
                }
            )
            if operation in {
                ResearchTaskOperationV2.NEW_SEED.value,
                ResearchTaskOperationV2.REPRODUCE.value,
            } and not _guard_next_seed_is_valid(
                next_seed=required,
                current_seed=observation_seed,
                observation_seed_schedule=observation_seed_schedule,
            ):
                return _unsupported_guard_seed_task(
                    task,
                    directive=payload,
                    reason="UNSUPPORTED_OUTSIDE_FROZEN_SEED_SCHEDULE",
                    observation_seed_schedule=observation_seed_schedule,
                )
            if operation in {
                ResearchTaskOperationV2.NEW_SEED.value,
                ResearchTaskOperationV2.REPRODUCE.value,
            } and _guard_seed_was_already_validated(
                context=context,
                binding=binding,
                seed=required,
            ):
                return _unsupported_guard_seed_task(
                    task,
                    directive=payload,
                    reason="UNSUPPORTED_SEED_ALREADY_VALIDATED",
                    observation_seed_schedule=observation_seed_schedule,
                )
            return task

    directive = _guard_validation_directive(response)
    if not directive:
        return None
    action = str(directive.get("action", "NONE"))
    if action not in {"ENQUEUE_NEW_SEED", "NEXT_UNSEEN_SEED_SAME_BUDGET"}:
        # A matched-control/mechanism-off directive without a complete typed
        # task is an interface failure, not permission to invent a binding.
        return None
    next_seed = directive.get("next_seed")
    if next_seed is None:
        return None
    program = binding.proposal.mechanism_program
    program_digest = sha256_digest(program)
    candidate_id = binding.proposal.candidate_id
    required_seed = str(next_seed)
    task_id = sha256_digest(
        {
            "operation": ResearchTaskOperationV2.NEW_SEED.value,
            "candidate_id": candidate_id,
            "candidate_semantic_digest": binding.mechanism_semantics_digest,
            "mechanism_program_digest": program_digest,
            "required_seed_or_control": required_seed,
            "comparator_identity": comparator_identity,
            "protocol_digest": context.protocol_digest,
        }
    )
    allocation = response.get("allocation_decision")
    allocation_metadata = (
        {
            "helix_allocation_action_id": allocation.get("action_id"),
            "helix_allocation_policy_digest": allocation.get("policy_digest"),
            "helix_information_gain_score": allocation.get(
                "information_gain_score"
            ),
        }
        if isinstance(allocation, Mapping)
        and allocation.get("action") == "REPLICATE"
        and isinstance(allocation.get("action_id"), str)
        else {}
    )
    task = ResearchTaskRecordV2(
        task_id=task_id,
        operation=ResearchTaskOperationV2.NEW_SEED,
        candidate_id=candidate_id,
        candidate_semantic_digest=binding.mechanism_semantics_digest,
        mechanism_program_digest=program_digest,
        parent_candidate_id=exact_parent_candidate_id(binding.proposal),
        comparator_identity=comparator_identity,
        protocol_digest=context.protocol_digest,
        required_seed_or_control=required_seed,
        priority=confidence_weight,
        created_round=context.round_index,
        mechanism_program=program,
        missing_seed_count=int(directive.get("missing_seed_count", 1)),
        metadata={
            "guard_source": "EVIDENCE_GUARD",
            "guard_validation_action": action,
            "guard_confidence_weight": confidence_weight,
            "execution_recipe_digest": sha256_digest(recipe),
            **allocation_metadata,
        },
    )
    if not _guard_next_seed_is_valid(
        next_seed=next_seed,
        current_seed=observation_seed,
        observation_seed_schedule=observation_seed_schedule,
    ):
        return _unsupported_guard_seed_task(
            task,
            directive=directive,
            reason="UNSUPPORTED_OUTSIDE_FROZEN_SEED_SCHEDULE",
            observation_seed_schedule=observation_seed_schedule,
        )
    if _guard_seed_was_already_validated(
        context=context,
        binding=binding,
        seed=next_seed,
    ):
        return _unsupported_guard_seed_task(
            task,
            directive=directive,
            reason="UNSUPPORTED_SEED_ALREADY_VALIDATED",
            observation_seed_schedule=observation_seed_schedule,
        )
    return task


def _guard_bind_control_task(
    *,
    response: Mapping[str, Any],
    context: ResearchContext,
    selected_outcome: ProducerOutcome,
    binding: SearchCandidateBindingV1,
    comparator_identity: str,
    observation_seed: str,
    observation_seed_schedule: Sequence[str] | None = None,
    search_space_adapter: SearchSpaceAdapter,
) -> ResearchTaskRecordV2 | None:
    """Bind a Guard control intent through the v27 Research interpreter."""

    projection = _guard_control_projection(response)
    if not isinstance(projection, Mapping):
        return None
    allocation = response.get("allocation_decision")
    if (
        isinstance(allocation, Mapping)
        and allocation.get("action") != "CONTROL"
        and allocation.get("reason") != "CONTROL_BINDING_REQUIRED"
    ):
        return None
    requested_kind = projection.get("requested_control_kind")
    if requested_kind == "NONE" or requested_kind is None:
        return None
    if requested_kind not in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
        raise ValueError("Evidence Guard requested an unsupported control kind")
    resolution = search_space_adapter.resolve_confirmation(
        str(requested_kind),
        selected_outcome,
        {
            "phase": "GUARD_CONTROL",
            "research_context": context,
            "mechanism_program": binding.proposal.mechanism_program,
        },
    )
    if not isinstance(resolution, ConfirmationResolutionV1):
        raise TypeError(
            "SearchSpaceAdapter.resolve_confirmation must return "
            "ConfirmationResolutionV1"
        )
    if resolution.kind is ConfirmationResolutionKindV1.NEEDS_PROPOSAL:
        operation = ResearchTaskOperationV2(str(requested_kind))
        source_program = binding.proposal.mechanism_program
        source_program_digest = sha256_digest(source_program)
        required = "AWAITING_CONTROL_PROPOSAL"
        task_id = sha256_digest(
            {
                "operation": operation.value,
                "candidate_id": binding.proposal.candidate_id,
                "candidate_semantic_digest": binding.mechanism_semantics_digest,
                "required_seed_or_control": required,
                "comparator_identity": comparator_identity,
                "protocol_digest": context.protocol_digest,
                "guard_source_candidate": binding.proposal.candidate_id,
            }
        )
        return ResearchTaskRecordV2(
            task_id=task_id,
            operation=operation,
            candidate_id=binding.proposal.candidate_id,
            candidate_semantic_digest=binding.mechanism_semantics_digest,
            mechanism_program_digest=source_program_digest,
            parent_candidate_id=exact_parent_candidate_id(binding.proposal),
            comparator_identity=comparator_identity,
            protocol_digest=context.protocol_digest,
            required_seed_or_control=required,
            priority=_guard_confidence_weight(response),
            created_round=context.round_index,
            evidence_present=(str(observation_seed),),
            missing_seed_count=1,
            mechanism_program=source_program,
            producer_role=selected_outcome.producer_role,
            provenance_digest=(
                selected_outcome.source_proposal.digest
                if selected_outcome.source_proposal is not None
                else None
            ),
            metadata={
                "guard_source": "EVIDENCE_GUARD",
                "guard_requested_control_kind": requested_kind,
                "execution_state": "AWAITING_CANDIDATE_BINDING",
                "binding_requirement": resolution.kind.value,
                "frontier_candidate_id": binding.proposal.candidate_id,
                "frontier_candidate_semantic_digest": (
                    binding.mechanism_semantics_digest
                ),
                "frontier_candidate_program_digest": source_program_digest,
                "guard_source_mechanism_axis": binding.proposal.mechanism_axis,
                "adapter_resolution_reason": resolution.reason,
            },
        )
    if resolution.kind is not ConfirmationResolutionKindV1.EXACT_BINDING:
        return None
    resolved = resolution.binding
    if not isinstance(resolved, Mapping):
        raise ValueError("exact Guard control resolution lacks an opaque binding")
    executable = resolved.get("execution_binding")
    if not isinstance(executable, Mapping):
        raise ValueError("Research control resolver returned no executable binding")
    operation = ResearchTaskOperationV2(str(resolved["operation"]))
    required = str(resolved["required_seed_or_control"])
    program = executable.get("mechanism_program")
    if not isinstance(program, Mapping):
        raise ValueError("Research control resolver returned no mechanism program")
    candidate_id = str(executable["candidate_id"])
    semantic_digest = str(executable["candidate_semantic_digest"])
    program_digest = str(executable["mechanism_program_digest"])
    source_program_digest = sha256_digest(binding.proposal.mechanism_program)
    summary = response.get("evidence_summary", {})
    same_seed_attribution = bool(summary.get("discriminative_question"))
    verification_seed = _guard_validation_directive(response).get("next_seed")
    verification_seed_source = "GUARD_VALIDATION_DIRECTIVE"
    if same_seed_attribution:
        future_seeds = tuple(str(seed) for seed in (observation_seed_schedule or ()))[context.round_index:]
        if str(observation_seed) not in future_seeds:
            # Keep the question in research context, without reserving an
            # impossible control or inventing another seed/worker opportunity.
            return None
        verification_seed = str(observation_seed)
        verification_seed_source = "SAME_SEED_CONTROL_IN_EXISTING_FUTURE_SLOT"
    elif verification_seed is None:
        verification_seed = _guard_next_frozen_schedule_seed(
            current_seed=observation_seed,
            observation_seed_schedule=observation_seed_schedule,
        )
        verification_seed_source = "FROZEN_OBSERVATION_SCHEDULE_SUCCESSOR"
    if not same_seed_attribution and not _guard_next_seed_is_valid(
        next_seed=verification_seed,
        current_seed=observation_seed,
        observation_seed_schedule=observation_seed_schedule,
    ):
        raise ValueError(
            "Evidence Guard control lacks a preregistered future verification seed"
        )
    allocation_source_semantic_digest = (
        allocation.get("candidate_semantic_digest")
        if isinstance(allocation, Mapping)
        else None
    )
    allocation_source_program_digest = (
        allocation.get("mechanism_program_digest")
        if isinstance(allocation, Mapping)
        else None
    )
    frontier_semantic_digest = (
        str(allocation_source_semantic_digest)
        if isinstance(allocation_source_semantic_digest, str)
        else binding.mechanism_semantics_digest
    )
    frontier_program_digest = (
        str(allocation_source_program_digest)
        if isinstance(allocation_source_program_digest, str)
        else source_program_digest
    )
    raw_source_summary = projection.get("evidence_summary")
    source_summary = canonical_value(
        {
            **(
                dict(raw_source_summary)
                if isinstance(raw_source_summary, Mapping)
                else {}
            ),
            "candidate_id": binding.proposal.candidate_id,
            "candidate_semantic_digest": frontier_semantic_digest,
            "mechanism_program_digest": frontier_program_digest,
            "protocol_digest": (
                raw_source_summary.get("protocol_digest", context.protocol_digest)
                if isinstance(raw_source_summary, Mapping)
                else context.protocol_digest
            ),
            "comparator_identity": (
                raw_source_summary.get("comparator_identity", comparator_identity)
                if isinstance(raw_source_summary, Mapping)
                else comparator_identity
            ),
            "mechanism_axis": binding.proposal.mechanism_axis,
        }
    )
    task_id = sha256_digest(
        {
            "operation": operation.value,
            "candidate_id": candidate_id,
            "candidate_semantic_digest": semantic_digest,
            "mechanism_program_digest": program_digest,
            "required_seed_or_control": required,
            "comparator_identity": comparator_identity,
            "protocol_digest": context.protocol_digest,
            "guard_source_candidate": binding.proposal.candidate_id,
        }
    )
    return ResearchTaskRecordV2(
        task_id=task_id,
        operation=operation,
        candidate_id=candidate_id,
        candidate_semantic_digest=semantic_digest,
        mechanism_program_digest=program_digest,
        parent_candidate_id=exact_parent_candidate_id(binding.proposal),
        comparator_identity=comparator_identity,
        protocol_digest=context.protocol_digest,
        required_seed_or_control=required,
        priority=_guard_confidence_weight(response),
        created_round=context.round_index,
        evidence_present=(str(observation_seed),),
        missing_seed_count=1,
        mechanism_program=program,
        producer_role=selected_outcome.producer_role,
        provenance_digest=(
            selected_outcome.source_proposal.digest
            if selected_outcome.source_proposal is not None
            else None
        ),
        metadata={
            "guard_source": "EVIDENCE_GUARD",
            "guard_requested_control_kind": requested_kind,
            "frontier_candidate_id": binding.proposal.candidate_id,
            "frontier_candidate_semantic_digest": frontier_semantic_digest,
            "frontier_mechanism_program_digest": frontier_program_digest,
            "guard_source_mechanism_axis": binding.proposal.mechanism_axis,
            "guard_source_evidence_summary": source_summary,
            "verification_seed": str(verification_seed),
            "verification_seed_source": verification_seed_source,
            "guard_control_relation": executable.get("control_identity"),
            "execution_binding": executable,
            **(
                {
                    "helix_allocation_action_id": allocation.get("action_id"),
                    "helix_allocation_policy_digest": allocation.get(
                        "policy_digest"
                    ),
                    "helix_information_gain_score": allocation.get(
                        "information_gain_score"
                    ),
                }
                if isinstance(allocation, Mapping)
                and isinstance(allocation.get("action_id"), str)
                else {}
            ),
        },
    )


def _guard_active_task(
    pending_task: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return only the queue record actually scheduled for this round."""

    if not isinstance(pending_task, Mapping):
        return None
    resolution = pending_task.get("task_resolution")
    if isinstance(resolution, Mapping) and resolution.get("status") == "CLOSED":
        return None
    task = pending_task.get("task_record")
    return canonical_value(dict(task)) if isinstance(task, Mapping) else None


def _guard_context_update(
    *,
    interpretation: EpisodeInterpretation,
    before: ResearchContext,
    response: Mapping[str, Any],
    binding: SearchCandidateBindingV1,
    task: ResearchTaskRecordV2 | None,
    evidence_domain: str,
    policy_before: VersionedResearchPolicyV1 | None = None,
) -> EpisodeInterpretation:
    """Route typed Guard metadata through existing context/queue memory."""

    successor = interpretation.successor_context
    frontier = dict(successor.frontier)
    state = _guard_claim_state(response)
    promotion_allowed = _guard_promotion_allowed(response)
    current_attempt = _guard_current_attempt_class(response)
    preserve_native_research = (
        response.get("research_update_mode") == "PRESERVE_NATIVE_RESEARCH"
        and current_attempt == "VALID_METRIC"
    )
    allocation = response.get("allocation_decision")
    allocation_is_relevant = bool(
        isinstance(allocation, Mapping)
        and allocation.get("decision_relevant") is True
    )
    source_control_attribution = response.get("source_control_attribution")
    non_supported_science = bool(
        evidence_domain == "SCIENTIFIC_EPISODE"
        and not promotion_allowed
        and not preserve_native_research
    )
    diagnostic_attempt = current_attempt in {
        "ENGINEERING_FAILURE",
        "PROTOCOL_DRIFT",
        "INVALID",
    }
    preserve_trusted_state = non_supported_science or diagnostic_attempt
    preserve_policy = preserve_trusted_state
    if preserve_trusted_state:
        # The ordinary interpreter computes a full scientific frontier/policy
        # before Guard adjudication.  A preliminary, negative, or inconclusive
        # signal, or a Guard diagnostic, may be retained as diagnostic memory,
        # but must not leak into any trusted frontier bank
        # (global/family/parent/control/confirmation) or policy successor.
        frontier = dict(before.frontier)
    memory = (
        dict(before.scientific_memory)
        if preserve_trusted_state
        else dict(successor.scientific_memory)
    )
    raw_global = memory.get("global_memory")
    global_memory = dict(raw_global) if isinstance(raw_global, Mapping) else {}
    if (
        preserve_trusted_state
        or preserve_native_research
        or isinstance(source_control_attribution, Mapping)
    ):
        successor_global_raw = successor.scientific_memory.get("global_memory")
        successor_global = (
            successor_global_raw
            if isinstance(successor_global_raw, Mapping)
            else {}
        )
        if isinstance(successor_global.get("task_queue"), Mapping):
            # Frontier/policy learning remains Research-native, but marginal
            # validation/control allocation has one owner. Preserve lifecycle
            # transitions for tasks that existed before this round; do not
            # also import the interpreter's generic follow-up bundle. Guard's
            # value-of-information action, if any, is enqueued below with an
            # exact seed/control identity. This prevents one observation from
            # spawning both native and Guard confirmation queues.
            before_global_raw = before.scientific_memory.get("global_memory")
            before_global = (
                before_global_raw
                if isinstance(before_global_raw, Mapping)
                else {}
            )
            prior_queue = ResearchTaskQueueV2.from_dict(
                before_global.get(
                    "task_queue", before.scientific_memory.get("task_queue")
                )
            )
            successor_queue = ResearchTaskQueueV2.from_dict(
                successor_global["task_queue"]
            )
            retained_tasks = [
                successor_queue.get(prior.task_id) or prior
                for prior in prior_queue.tasks
            ]
            task_resolution = (
                task.metadata.get("task_resolution")
                if task is not None
                else None
            )
            guard_has_no_executable_replacement = bool(
                isinstance(task_resolution, Mapping)
                and task_resolution.get("status") == "CLOSED"
            )
            if guard_has_no_executable_replacement or (
                task is None
                and preserve_trusted_state
                and not preserve_native_research
            ):
                # If Guard cannot authorize its requested action (notably an
                # unavailable new seed), a missing or already-closed Guard task
                # is not an executable replacement and
                # must not erase same-seed native controls produced by the
                # interpreter.  Preserve only generic control intents here;
                # NEW_SEED/REPRODUCE remain excluded from discovery.
                retained_ids = {item.task_id for item in retained_tasks}
                native_event = interpretation.search_utility_event
                native_seed = (
                    native_event.observation_seed
                    if native_event is not None
                    else None
                )
                retained_tasks.extend(
                    item
                    for item in successor_queue.tasks
                    if item.task_id not in retained_ids
                    and item.operation
                    in {
                        ResearchTaskOperationV2.MATCHED_CONTROL,
                        ResearchTaskOperationV2.MECHANISM_OFF,
                    }
                    and item.metadata.get("execution_state")
                    == "AWAITING_CANDIDATE_BINDING"
                    and native_seed in item.evidence_present
                )
            global_memory["task_queue"] = ResearchTaskQueueV2(
                tuple(retained_tasks)
            ).to_dict()
    weight = _guard_confidence_weight(response)
    guard_record = canonical_value(
        {
            "round_index": before.round_index,
            "candidate_id": binding.proposal.candidate_id,
            "candidate_semantic_digest": binding.mechanism_semantics_digest,
            "evidence_domain": evidence_domain,
            "claim_state": state,
            "promotion_allowed": promotion_allowed,
            "confidence_weight": weight,
            "response_digest": sha256_digest(response),
        }
    )
    records = list(global_memory.get("evidence_guard_records", ()))
    records.append(guard_record)
    global_memory["evidence_guard_records"] = tuple(records[-128:])
    global_memory["frontier"] = canonical_value(frontier)
    if evidence_domain == "SCIENTIFIC_EPISODE":
        if not promotion_allowed and not preserve_native_research:
            preliminary = list(global_memory.get("preliminary_search_credit", ()))
            preliminary.append(guard_record)
            global_memory["preliminary_search_credit"] = tuple(preliminary[-128:])
            if state in {"PRELIMINARY_NONPOSITIVE", "REFUTED"}:
                negative = list(global_memory.get("mechanism_negative_evidence", ()))
                negative.append(
                    canonical_value(
                        {
                            **guard_record,
                            "evidence_class": "MECHANISM_NEGATIVE",
                            "negative_evidence_weight": weight,
                        }
                    )
                )
                global_memory["mechanism_negative_evidence"] = tuple(negative[-128:])
        if non_supported_science:
            event = interpretation.search_utility_event
            if event is not None:
                weighted_observation = canonical_value(
                    {
                        "round_index": before.round_index,
                        "candidate_id": event.candidate_id,
                        "candidate_semantic_digest": event.candidate_semantic_digest,
                        "observation_seed": event.observation_seed,
                        "common_outcome_class": event.common_outcome_class,
                        "mechanism_axis": event.mechanism_axis,
                        "comparator_delta": event.comparator_delta,
                        "evidence_domain": "SCIENTIFIC_EPISODE",
                        "mechanism_effect_update_allowed": False,
                        "guard_confidence_weight": weight,
                        "guard_claim_state": state,
                    }
                )
                executed = list(global_memory.get("executed_observations", ()))
                executed.append(weighted_observation)
                global_memory["executed_observations"] = tuple(executed[-128:])
                search_observations = list(global_memory.get("search_observations", ()))
                search_observations.append(
                    canonical_value(
                        {
                            **weighted_observation,
                            "event_id": sha256_digest(weighted_observation),
                        }
                    )
                )
                global_memory["search_observations"] = tuple(
                    search_observations[-128:]
                )
        observations = list(global_memory.get("executed_observations", ()))
        if observations:
            observations[-1] = canonical_value(
                {
                    **dict(observations[-1]),
                    "guard_confidence_weight": weight,
                    "guard_claim_state": state,
                    "guard_promotion_allowed": promotion_allowed,
                    "guard_research_update_mode": response.get("research_update_mode"),
                }
            )
            global_memory["executed_observations"] = tuple(observations)
        search_observations = list(global_memory.get("search_observations", ()))
        if search_observations:
            search_observations[-1] = canonical_value(
                {
                    **dict(search_observations[-1]),
                    "guard_confidence_weight": weight,
                    "guard_claim_state": state,
                    "guard_promotion_allowed": promotion_allowed,
                    "guard_research_update_mode": response.get("research_update_mode"),
                }
            )
            global_memory["search_observations"] = tuple(search_observations)
    else:
        global_memory["latest_guard_engineering_record"] = guard_record
        diagnostics = list(global_memory.get("guard_diagnostic_memory", ()))
        diagnostics.append(guard_record)
        global_memory["guard_diagnostic_memory"] = tuple(diagnostics[-128:])
    if allocation_is_relevant:
        global_memory["latest_guard_feedback"] = canonical_value(dict(response))
    elif preserve_native_research:
        # No physical action does not mean no useful evidence. Keep the
        # observation and its research meaning, but not stale execution advice.
        global_memory["latest_guard_feedback"] = canonical_value({
            "evidence_summary": response.get(
                "evidence_summary", (_guard_control_projection(response) or {}).get("evidence_summary", {})
            ),
            "research_update_mode": "PRESERVE_NATIVE_RESEARCH",
        })
    else:
        # Engineering-only evidence has its own diagnostic path above.
        global_memory.pop("latest_guard_feedback", None)
    if isinstance(source_control_attribution, Mapping):
        attribution_record = canonical_value(dict(source_control_attribution))
        attribution_digest = sha256_digest(attribution_record)
        prior_attributions = list(
            global_memory.get("mechanism_attribution_observations", ())
        )
        if all(
            not isinstance(item, Mapping)
            or sha256_digest(item) != attribution_digest
            for item in prior_attributions
        ):
            prior_attributions.append(attribution_record)
        global_memory["mechanism_attribution_observations"] = tuple(
            prior_attributions[-128:]
        )
        global_memory["latest_mechanism_attribution"] = attribution_record
    if task is not None:
        queue = ResearchTaskQueueV2.from_dict(
            global_memory.get("task_queue", memory.get("task_queue"))
        )
        prior_task = queue.get(task.task_id)
        queue = queue.enqueue(task)
        task_resolution = task.metadata.get("task_resolution")
        guard_task_transition = None
        if (
            isinstance(task_resolution, Mapping)
            and task_resolution.get("status") == "CLOSED"
        ):
            close_reason = str(
                task_resolution.get(
                    "reason",
                    "CLOSED_BY_NON_METRIC_TASK_RESOLUTION",
                )
            )
            queue = queue.close(task.task_id, reason=close_reason)
            guard_task_transition = canonical_value(
                {
                    "satisfied_task_ids": (),
                    "closed_task_ids": (task.task_id,),
                    "created_task_ids": (
                        (task.task_id,) if prior_task is None else ()
                    ),
                    "deferred_requirements": (),
                    "metric_bearing_evidence": False,
                    "resolution": task_resolution,
                }
            )
        durable_task = queue.get(task.task_id)
        global_memory["task_queue"] = queue.to_dict()
        global_memory["latest_feedback"] = {
            **dict(global_memory.get("latest_feedback", {})),
            "guard_task": (
                durable_task.to_dict() if durable_task is not None else task.to_dict()
            ),
            "task_queue_head": (
                queue.select_next().to_dict() if queue.select_next() is not None else None
            ),
            **(
                {"guard_task_transition": guard_task_transition}
                if guard_task_transition is not None
                else {}
            ),
        }
    memory["global_memory"] = canonical_value(global_memory)
    # Keep the V1 top-level aliases synchronized with the V2 global projection;
    # legacy ranking/producer consumers must see the same Guard task and
    # evidence records on the next normal round.
    memory["search_memory_head"] = global_memory.get("search_memory_head")
    memory["latest_feedback"] = global_memory.get("latest_feedback")
    memory["task_queue"] = global_memory.get("task_queue")
    memory["executed_observations"] = global_memory.get("executed_observations", ())
    memory["search_observations"] = global_memory.get("search_observations", ())
    successor = replace(
        successor,
        frontier=canonical_value(frontier),
        scientific_memory=canonical_value(memory),
        policy=(
            policy_before.to_dict()
            if preserve_policy and policy_before is not None
            else successor.policy
        ),
    )
    return replace(
        interpretation,
        policy_successor=(
            policy_before
            if preserve_policy and policy_before is not None
            else interpretation.policy_successor
        ),
        successor_context=successor,
    )


def _reproject_guarded_acquisition(
    *,
    interpretation: EpisodeInterpretation,
    producer_outcomes: Sequence[ProducerOutcome],
    selected_outcome: ProducerOutcome,
    route_metadata: Mapping[str, Any],
    task_record: ResearchTaskRecordV2,
    required_probability: float,
) -> EpisodeInterpretation:
    """Project Guard-enriched memory through the existing acquisition policy."""

    outcomes = {
        outcome.producer_role: outcome
        for outcome in producer_outcomes
    }
    aggregate = _meta_aggregate(
        outcomes,
        selected_outcome.producer_role,
        selected_outcome,
        interpretation.search_utility_event,
        required_probability,
        interpretation.policy_successor,
        scientific_episode=True,
        context=interpretation.successor_context,
        route=route_metadata,
        episode=interpretation.episode,
        task_record=task_record,
    )
    successor_policy = replace(
        interpretation.policy_successor,
        acquisition_parameters=_next_acquisition_parameters(
            interpretation.policy_successor,
            aggregate,
        ),
    )
    successor_context = replace(
        interpretation.successor_context,
        policy=successor_policy.to_dict(),
    )
    return replace(
        interpretation,
        policy_successor=successor_policy,
        successor_context=successor_context,
        behavior_after=_behavior(successor_context, successor_policy),
    )


def _neutralize_undeclared_parent_program_deltas(
    context: ResearchContext,
    candidates: Sequence[QualifiedSearchCandidateProtocolV1],
) -> tuple[ResearchContext, bool]:
    """Promote explicit append-only fidelity evidence into utility exclusions."""

    del candidates

    memory = dict(context.scientific_memory)
    raw_global = memory.get("global_memory")
    global_memory = dict(raw_global) if isinstance(raw_global, Mapping) else {}
    raw_evidence = global_memory.get("implementation_fidelity_neutralization_evidence", ())
    evidence = (
        tuple(row for row in raw_evidence if isinstance(row, Mapping))
        if isinstance(raw_evidence, (tuple, list))
        else ()
    )
    existing = [
        canonical_value(dict(row))
        for row in global_memory.get("utility_neutralizations", ())
        if isinstance(row, Mapping)
    ]
    existing_digests = {sha256_digest(row) for row in existing}
    appended = False
    for item in evidence:
        semantic_digest = item.get("candidate_semantic_digest")
        seed = item.get("observation_seed")
        round_index = item.get("round_index")
        reason = item.get("reason")
        if (
            not isinstance(semantic_digest, str)
            or not semantic_digest
            or not isinstance(seed, (str, int))
            or isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            continue
        record = canonical_value(
            {
                "round_index": round_index,
                "candidate_semantic_digest": semantic_digest,
                "observation_seed": seed,
                "classification": "IMPLEMENTATION_FIDELITY_CONFOUNDED",
                "reason": reason.strip(),
                "search_utility_update_allowed": False,
                "measurement_retained": True,
            }
        )
        digest = sha256_digest(record)
        if digest not in existing_digests:
            existing.append(record)
            existing_digests.add(digest)
            appended = True
    if not appended:
        return context, False
    global_memory["utility_neutralizations"] = tuple(existing[-128:])
    memory["global_memory"] = canonical_value(global_memory)
    return replace(context, scientific_memory=canonical_value(memory)), True


def normalize_fidelity_utility_state(
    context: ResearchContext,
    policy: VersionedResearchPolicyV1,
) -> tuple[ResearchContext, VersionedResearchPolicyV1, bool]:
    """Materialize audited fidelity exclusions before a round is identified."""

    normalized_context, evidence_changed = _neutralize_undeclared_parent_program_deltas(
        context,
        (),
    )
    normalized_global = normalized_context.scientific_memory.get("global_memory")
    normalized_global = (
        normalized_global if isinstance(normalized_global, Mapping) else {}
    )
    has_fidelity_exclusion = any(
        isinstance(row, Mapping)
        and row.get("classification") == "IMPLEMENTATION_FIDELITY_CONFOUNDED"
        and row.get("search_utility_update_allowed") is False
        and row.get("measurement_retained") is True
        for row in normalized_global.get("utility_neutralizations", ())
    )
    if not has_fidelity_exclusion:
        return context, policy, False
    acquisition_parameters = dict(policy.acquisition_parameters)
    policy_changed = False
    for field_name in (
        "producer_useful_rates",
        "producer_quality_scores",
        "family_quality_scores",
        "experiment_quality_scores",
        "axis_scores",
    ):
        if field_name in acquisition_parameters:
            acquisition_parameters.pop(field_name)
            policy_changed = True
    if (
        not evidence_changed
        and not policy_changed
        and canonical_value(normalized_context.policy)
        == canonical_value(policy.to_dict())
    ):
        return context, policy, False
    normalized_policy = replace(
        policy,
        acquisition_parameters=acquisition_parameters,
    )
    normalized_context = replace(
        normalized_context,
        policy=normalized_policy.to_dict(),
    )
    return normalized_context, normalized_policy, True


def _search_ranking_inputs(
    context: ResearchContext,
) -> tuple[
    tuple[tuple[str, str], ...],
    Mapping[str, Any] | None,
    Mapping[str, float],
]:
    memory = context.scientific_memory
    global_memory = memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    raw_observations = global_memory.get(
        "executed_observations",
        memory.get("executed_observations", ()),
    )
    raw_neutralizations = global_memory.get("utility_neutralizations", ())
    utility_neutralizations = (
        tuple(row for row in raw_neutralizations if isinstance(row, Mapping))
        if isinstance(raw_neutralizations, (tuple, list))
        else ()
    )
    pairs: list[tuple[str, str]] = []
    effects_by_axis: dict[str, list[float]] = {}
    if isinstance(raw_observations, (tuple, list)):
        for observation in raw_observations:
            if not isinstance(observation, Mapping):
                continue
            semantic_digest = observation.get("candidate_semantic_digest")
            observation_seed = observation.get("observation_seed")
            if not isinstance(semantic_digest, str) or not isinstance(
                observation_seed, str
            ):
                continue
            pair = (semantic_digest, observation_seed)
            if pair not in pairs:
                pairs.append(pair)
            effect = observation.get("comparator_delta")
            domain = observation.get("evidence_domain")
            if domain in {"RESOURCE_SEARCH", "GUARD_ENGINEERING", "PROTOCOL_DRIFT"}:
                continue
            causal_effect_allowed = (
                observation.get("causal_credit_allowed") is True
                and observation.get("mechanism_effect_update_allowed") is not False
            )
            # A completed development metric is directional search evidence
            # even when single-seed claim wording remains INCONCLUSIVE.  The
            # explicit bit is written by repair08; the SUCCESS fallback makes
            # already-sealed pre-repair observations usable after resume.
            descriptive_search_allowed = observation_updates_search_utility(
                observation,
                neutralizations=utility_neutralizations,
            )
            # Directional utility authority is stronger than legacy causal
            # flags: an explicitly neutralized measurement remains retained
            # but cannot update any axis even if an old row says causal=True.
            if not descriptive_search_allowed:
                continue
            raw_footprint = observation.get("mechanism_axis_footprint", ())
            footprint = (
                tuple(
                    dict.fromkeys(
                        str(item).strip()
                        for item in raw_footprint
                        if str(item).strip()
                    )
                )
                if isinstance(raw_footprint, (tuple, list))
                else ()
            )
            if not footprint:
                axis = observation.get("mechanism_axis")
                if isinstance(axis, str) and axis.strip():
                    footprint = (axis.strip(),)
            if (
                footprint
                and isinstance(effect, (int, float))
                and not isinstance(effect, bool)
            ):
                weight = (
                    1.0
                    if observation.get("guard_research_update_mode")
                    == "PRESERVE_NATIVE_RESEARCH"
                    else observation.get("guard_confidence_weight", 1.0)
                )
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    weight = 1.0
                weighted_effect = (
                    float(effect)
                    * min(1.0, max(0.0, float(weight)))
                    / len(footprint)
                )
                for axis in footprint:
                    effects_by_axis.setdefault(axis, []).append(weighted_effect)
                # Preserve the existing coarse proposal-family signal used by
                # the portfolio router while adding the more truthful
                # mechanism-footprint signal consumed by Provider context.
                coarse_axis = observation.get("mechanism_axis")
                if (
                    isinstance(coarse_axis, str)
                    and coarse_axis.strip()
                    and coarse_axis.strip() not in footprint
                ):
                    effects_by_axis.setdefault(coarse_axis.strip(), []).append(
                        float(effect) * min(1.0, max(0.0, float(weight)))
                    )
    raw_queue = global_memory.get("task_queue", memory.get("task_queue"))
    queued = ResearchTaskQueueV2.from_dict(raw_queue).select_next()
    if queued is not None:
        pending_task = _pending_task_projection(queued)
    else:
        latest_feedback = global_memory.get(
            "latest_feedback",
            memory.get("latest_feedback"),
        )
        pending_task = (
            latest_feedback.get("research_task_slot")
            if isinstance(latest_feedback, Mapping)
            else None
        )
    raw_attributions = global_memory.get("mechanism_attribution_observations", ())
    if isinstance(raw_attributions, (tuple, list)):
        for attribution in raw_attributions:
            if not isinstance(attribution, Mapping):
                continue
            if attribution.get("evidence_class") != "DEVELOPMENT_ONLY":
                continue
            if attribution.get("causal_credit_allowed") is not True:
                continue
            state = attribution.get("attribution_state")
            if state not in {"DESCRIPTIVE_SUPPORT", "DESCRIPTIVE_REFUTATION"}:
                continue
            axis = attribution.get("source_mechanism_axis")
            delta = attribution.get("incremental_delta")
            weight = attribution.get("confidence_weight", 0.0)
            if (
                isinstance(axis, str)
                and isinstance(delta, (int, float))
                and not isinstance(delta, bool)
                and isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and math.isfinite(float(delta))
                and math.isfinite(float(weight))
            ):
                effects_by_axis.setdefault(axis, []).append(
                    float(delta) * min(1.0, max(0.0, float(weight)))
                )
    return (
        tuple(pairs),
        pending_task if isinstance(pending_task, Mapping) else None,
        {
            axis: sum(effects) / len(effects)
            for axis, effects in effects_by_axis.items()
            if effects
        },
    )


def _pending_task_projection(task: ResearchTaskRecordV2) -> dict[str, Any]:
    """Carry one durable task through Provider, acquisition, and execution."""

    return {
        **task.prompt_projection(),
        "task_id": task.task_id,
        "candidate_id": task.candidate_id,
        "parent_candidate_id": task.parent_candidate_id,
        "comparator_identity": task.comparator_identity,
        "priority": task.priority,
        "operation": task.operation.value,
        "evidence_present": task.evidence_present,
        "mechanism_program_digest": task.mechanism_program_digest,
        "mechanism_program": (
            canonical_value(task.mechanism_program)
            if isinstance(task.mechanism_program, Mapping)
            else None
        ),
        "verification_seed": task.metadata.get("verification_seed"),
        "execution_binding": task.metadata.get("execution_binding"),
        "task_record": task.to_dict(),
    }


def _feedback_confirmation_task_directive(
    pending_task: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project the executable task identity into search-owned rejection feedback."""

    task_record = pending_task["task_record"]
    metadata = task_record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return canonical_value(
        {
            **{
                field_name: task_record.get(field_name)
                for field_name in (
                    "task_id",
                    "operation",
                    "candidate_id",
                    "candidate_semantic_digest",
                    "mechanism_program_digest",
                    "parent_candidate_id",
                    "comparator_identity",
                    "protocol_digest",
                    "required_seed_or_control",
                    "producer_role",
                )
            },
            **{
                field_name: metadata.get(field_name)
                for field_name in (
                    "execution_state",
                    "binding_requirement",
                    "confirmation_target",
                    "effective_experiment_digest",
                    "effective_family_digest",
                )
            },
            "task_record_digest": sha256_digest(task_record),
        }
    )


def _discovery_feedback_task(
    context: ResearchContext,
    *,
    observation_seed: str,
) -> Mapping[str, Any] | None:
    """Keep controls and replications outside ordinary discovery.

    Exact, already-bound evidence work is handled by the auxiliary verification
    lane.  Unbound control intents remain feedback for later decisions; they do
    not redirect Provider or consume a discovery opportunity.
    """

    return None


def _policy_without_discovery_feedback_task(
    policy: VersionedResearchPolicyV1,
) -> VersionedResearchPolicyV1:
    parameters = dict(policy.acquisition_parameters)
    parameters.pop("research_task_record", None)
    parameters.pop("next_discriminative_task", None)
    return replace(policy, acquisition_parameters=tuple(parameters.items()))


def _close_unsupported_discovery_feedback_task(
    pending_task: Mapping[str, Any],
    *,
    reason: str = "UNSUPPORTED_NO_LEGAL_BINDING",
) -> Mapping[str, Any]:
    return {
        **dict(pending_task),
        "task_resolution": {
            "status": "CLOSED",
            "reason": reason,
            "evidence_class": "NON_METRIC_TASK_RESOLUTION",
        },
    }


def _active_task_directive(
    pending_task: Mapping[str, Any] | None,
    *,
    observation_seed: str,
) -> Mapping[str, Any] | None:
    if not isinstance(pending_task, Mapping) or not isinstance(
        pending_task.get("task_record"), Mapping
    ):
        return None
    task_record = pending_task["task_record"]
    metadata = task_record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return canonical_value(
        {
            **dict(pending_task),
            **{
                field_name: metadata[field_name]
                for field_name in (
                    "execution_state",
                    "binding_requirement",
                    "frontier_candidate_id",
                    "frontier_parent_candidate_id",
                    "frontier_comparator_identity",
                    "next_discriminative_test",
                    "mechanism_axis",
                    "mechanism_axis_footprint",
                    "core_mechanism_contrast",
                )
                if field_name in metadata
            },
            "task_digest": sha256_digest(task_record),
            "observation_seed": observation_seed,
            "protocol_digest": task_record.get("protocol_digest"),
            "execution_eligible_this_round": True,
        }
    )


_EXACT_VERIFICATION_OPERATIONS = {
    "NEW_SEED",
    "REPRODUCE",
    "MATCHED_CONTROL",
    "MECHANISM_OFF",
}


def _exact_verification_task_bypasses_provider(
    task: Mapping[str, Any] | None,
    *,
    observation_seed: str,
    executed_semantic_seed_pairs: Sequence[tuple[str, str]],
) -> bool:
    """Return true only for a fully bound durable verification task."""

    if not isinstance(task, Mapping):
        return False
    operation = task.get("operation")
    if operation not in _EXACT_VERIFICATION_OPERATIONS:
        return False
    for field_name in (
        "task_id",
        "candidate_semantic_digest",
        "mechanism_program_digest",
    ):
        if not isinstance(task.get(field_name), str) or not task[field_name]:
            raise ValueError(f"queued verification task lacks exact {field_name}")
    semantic_digest = str(task["candidate_semantic_digest"])
    if (semantic_digest, observation_seed) in set(executed_semantic_seed_pairs):
        return False
    if operation in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
        if not isinstance(task.get("execution_binding"), Mapping):
            task_record = task.get("task_record")
            task_metadata = (
                task_record.get("metadata")
                if isinstance(task_record, Mapping)
                else {}
            )
            if (
                isinstance(task_metadata, Mapping)
                and task_metadata.get("execution_state")
                == "AWAITING_CANDIDATE_BINDING"
            ):
                # This is a real discriminative intent, but its candidate
                # must be supplied by the next Provider/acquisition pass.
                return False
            raise ValueError("queued control task lacks exact execution binding")
        if not isinstance(task.get("verification_seed"), str) or not task[
            "verification_seed"
        ]:
            raise ValueError(
                "queued control task lacks preregistered verification seed"
            )
        return task["verification_seed"] == observation_seed
    required = task.get("required_seed_or_control")
    if required == _NEXT_DEVELOPMENT_SEED:
        evidence_present = task.get("evidence_present", ())
        if isinstance(evidence_present, str):
            evidence_present = (evidence_present,)
        return observation_seed not in tuple(evidence_present)
    return required == observation_seed


def _verification_feedback_task(
    context: ResearchContext,
    *,
    observation_seed: str,
    executed_semantic_seed_pairs: Sequence[tuple[str, str]],
) -> Mapping[str, Any] | None:
    """Select the next fully bound verification executable on this seed."""

    memory = context.scientific_memory
    global_memory = memory.get("global_memory")
    if not isinstance(global_memory, Mapping):
        global_memory = memory.get("global")
    raw_queue = (
        global_memory.get("task_queue", memory.get("task_queue"))
        if isinstance(global_memory, Mapping)
        else memory.get("task_queue")
    )
    queue = ResearchTaskQueueV2.from_dict(raw_queue)
    eligible = []
    for task in queue.tasks:
        if task.status.value != "PENDING":
            continue
        projection = _pending_task_projection(task)
        if _exact_verification_task_bypasses_provider(
            projection,
            observation_seed=observation_seed,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        ):
            eligible.append(task)
    reserved = [task for task in eligible if task.metadata.get("helix_allocation_action_id")]
    selected = ResearchTaskQueueV2(tuple(reserved or eligible)).select_next()
    return _pending_task_projection(selected) if selected is not None else None


def _package_control_verification_binding(
    *,
    context: ResearchContext,
    policy: VersionedResearchPolicyV1,
    producer_bindings: Mapping[str, Any],
    resolver_environment: Mapping[str, Any],
    pending_task: Mapping[str, Any],
    search_space_adapter: SearchSpaceAdapter,
) -> tuple[
    SearchExecutableProfileV1,
    ProducerOutcome,
    CapabilityResolutionV1,
    SearchCandidateBindingV1,
]:
    """Materialize one exact adapter-owned control without discovery search."""

    task_record = pending_task.get("task_record")
    operation = (
        task_record.get("operation")
        if isinstance(task_record, Mapping)
        else pending_task.get("operation")
    )
    operation = str(getattr(operation, "value", operation) or "")
    materialized = search_space_adapter.resolve_confirmation(
        operation,
        None,
        {
            "phase": "MATERIALIZE_VERIFICATION",
            "pending_task": pending_task,
            "research_context": context,
            "policy": policy,
            "producer_bindings": producer_bindings,
        },
    )
    if not isinstance(materialized, ConfirmationResolutionV1):
        raise TypeError(
            "SearchSpaceAdapter.resolve_confirmation must return "
            "ConfirmationResolutionV1"
        )
    if (
        materialized.kind is not ConfirmationResolutionKindV1.EXACT_BINDING
        or not isinstance(materialized.binding, Mapping)
    ):
        raise ValueError("queued verification task has no exact adapter binding")
    control_profile = materialized.binding.get("execution_profile")
    binding = materialized.binding.get("native_binding")
    if not isinstance(control_profile, SearchExecutableProfileV1) or not isinstance(
        binding, SearchCandidateBindingV1
    ):
        raise ValueError(
            "adapter verification materialization lacks profile/native binding"
        )
    task_program = pending_task.get("mechanism_program")
    task_program_digest = pending_task.get("mechanism_program_digest")
    task_semantics = pending_task.get("candidate_semantic_digest")
    if (
        not isinstance(task_program, Mapping)
        or not isinstance(task_program_digest, str)
        or not isinstance(task_semantics, str)
    ):
        raise ValueError("queued package control has an invalid exact program identity")
    if (
        binding.mechanism_semantics_digest != task_semantics
        or canonical_value(binding.proposal.mechanism_program)
        != canonical_value(task_program)
    ):
        raise ValueError(
            "adapter verification binding differs from the queued exact identity"
        )
    profile_entry = control_profile.entry(binding.capability_ref)
    if (
        control_profile.campaign_id != context.campaign_id
        or binding.capability_digest != profile_entry.capability_digest
        or binding.executable_entrypoint != profile_entry.executable_entrypoint
        or binding.entry_origin != profile_entry.origin
        or control_profile.protocol_ref != context.protocol_ref
        or control_profile.protocol_digest != context.protocol_digest
    ):
        raise ValueError("adapter verification profile differs from Research context")
    control_context = replace(
        context,
        active_profile_ref=control_profile.profile_ref,
        active_profile_digest=control_profile.profile_digest,
    )
    adapter_outcome = materialized.binding.get("producer_outcome")
    outcomes = (
        (adapter_outcome,)
        if isinstance(adapter_outcome, ProducerOutcome)
        else _carryover_outcomes(
            control_context,
            (binding.proposal,),
            producer_bindings,
        )
    )
    if len(outcomes) != 1:
        raise ValueError("package control did not materialize one persisted outcome")
    if (
        outcomes[0].context_ref != control_context.context_ref
        or outcomes[0].context_digest != control_context.digest
        or outcomes[0].source_proposal != binding.proposal
    ):
        raise ValueError("adapter verification outcome differs from Research context")
    resolved = tuple(
        item
            for item in resolve_producer_outcomes(
                outcomes,
                environment=resolver_environment,
        )
        if item[1] is not None
        and item[1].resolution is CapabilityResolutionResultV1.SEARCH_READY
    )
    if len(resolved) != 1:
        raise ValueError("package control is not Search-ready in its immutable catalog")
    outcome, resolution = resolved[0]
    if resolution.resolved_current_capability_ref is None:
        raise ValueError("package control resolution lacks a capability reference")
    return control_profile, outcome, resolution, binding


_METRIC_EXECUTABLE_IDENTITY_FIELDS = (
    "model",
    "base_model_config",
    "config",
    "entrypoint_source_sha256",
    "candidate_source_content_digest",
    "base_mechanism_id",
    "operator_ids",
    "execution_role",
)


def _metric_executable_identity_digest(recipe: Mapping[str, Any]) -> str:
    """Hash behavior-defining execution inputs, excluding renameable labels.

    Candidate IDs, mechanism IDs, capability refs, package refs, profile refs,
    entrypoint module paths, and source-tree paths are intentionally absent:
    changing those labels or moving identical source bytes must not make an
    otherwise byte/config-identical executable count as new research.  The
    name-independent behavior-source content and config remain part of the
    identity, so a real implementation change or parameterized ablation is not
    confused with a replay merely because it shares an entrypoint file.
    """

    projection = {
        field_name: recipe[field_name]
        for field_name in _METRIC_EXECUTABLE_IDENTITY_FIELDS
        if field_name in recipe
    }
    if not all(
        field_name in projection
        for field_name in ("model", "config", "entrypoint_source_sha256")
    ):
        raise ValueError(
            "execution recipe lacks model/config/entrypoint-source identity"
        )
    return sha256_digest(
        {
            "schema": "recclaw.metric-executable-identity.v1",
            "execution": canonical_value(projection),
        }
    )


def _round_attempt_memory(context: ResearchContext) -> tuple[Mapping[str, Any], ...]:
    raw_attempts = context.scientific_memory.get("round_attempts", ())
    global_memory = context.scientific_memory.get("global_memory")
    global_attempts = (
        global_memory.get("round_attempts", ())
        if isinstance(global_memory, Mapping)
        else ()
    )
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in (raw_attempts, global_attempts):
        if not isinstance(raw, (tuple, list)):
            continue
        for attempt in raw:
            if not isinstance(attempt, Mapping):
                continue
            identity = sha256_digest(attempt)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(attempt)
    return tuple(rows)


def _compact_identity_memory(
    context: ResearchContext,
    key: str,
) -> tuple[str, ...]:
    global_memory = context.scientific_memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    values: list[str] = []
    seen: set[str] = set()
    for raw_values in (
        context.scientific_memory.get(key, ()),
        global_memory.get(key, ()),
    ):
        if not isinstance(raw_values, (tuple, list)):
            continue
        for value in raw_values:
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                values.append(value)
    return tuple(values)


def _metric_observation_memory(
    context: ResearchContext,
) -> tuple[Mapping[str, Any], ...]:
    global_memory = context.scientific_memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw_rows in (
        _round_attempt_memory(context),
        context.scientific_memory.get("metric_observation_index", ()),
        global_memory.get("metric_observation_index", ()),
    ):
        if not isinstance(raw_rows, (tuple, list)):
            continue
        for row in raw_rows:
            if not isinstance(row, Mapping):
                continue
            identity = sha256_digest(row)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(row)
    return tuple(rows)


def _existing_exact_feedback_observation_rebind(
    *,
    context: ResearchContext,
    pending_task: Mapping[str, Any] | None,
    innovation: InnovationLaneResult | None,
    exact_confirmation: bool,
    evaluator: Mapping[str, Any],
    split: str,
    observation_seed: str,
    metric_contract_digest: str,
) -> Mapping[str, Any] | None:
    """Reference one prior physical metric for an exact queued control."""

    if not exact_confirmation or innovation is None or not isinstance(
        pending_task, Mapping
    ):
        return None
    task = pending_task.get("task_record")
    task = task if isinstance(task, Mapping) else {}
    operation = getattr(task.get("operation"), "value", task.get("operation"))
    metadata = task.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if (
        operation not in {"MATCHED_CONTROL", "MECHANISM_OFF"}
        or task.get("status") not in {"PENDING", "ACTIVE"}
        or metadata.get("execution_state") != "AWAITING_CANDIDATE_BINDING"
        or task.get("protocol_digest") != context.protocol_digest
    ):
        return None
    global_memory = context.scientific_memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    standalone_identity = global_memory.get("standalone_identity")
    standalone_identity = (
        standalone_identity if isinstance(standalone_identity, Mapping) else {}
    )
    metric_name = evaluator.get("metric")
    dataset = context.budget.get("dataset")
    frozen_split = context.budget.get("evaluation_split")
    if (
        standalone_identity.get("protocol_digest") != context.protocol_digest
        or not isinstance(dataset, str)
        or not dataset
        or frozen_split != split
        or not isinstance(metric_name, str)
        or not metric_name
    ):
        return None
    duplicate_attempt = next(
        (
            item
            for item in reversed(innovation.attempts)
            if isinstance(item, Mapping)
            and isinstance(item.get("failure"), Mapping)
            and item["failure"].get("reason_code")
            in {"EFFECTIVE_EXPERIMENT_DUPLICATE", "EFFECTIVE_FAMILY_DUPLICATE"}
            and isinstance(item.get("effective_experiment_digest"), str)
            and isinstance(item.get("effective_family_digest"), str)
        ),
        None,
    )
    if duplicate_attempt is None:
        return None
    effective_experiment = duplicate_attempt["effective_experiment_digest"]
    effective_family = duplicate_attempt["effective_family_digest"]
    metric_key = metric_name.lower()
    rows = _metric_observation_memory(context)

    def is_metric_row(row: Mapping[str, Any]) -> bool:
        metrics = row.get("development_metrics")
        return bool(
            row.get("metric_bearing") is True
            and row.get("outcome") == "SUCCESS"
            and str(row.get("observation_seed")) == str(observation_seed)
            and isinstance(metrics, Mapping)
            and isinstance(metrics.get(metric_key), (int, float))
            and not isinstance(metrics.get(metric_key), bool)
            and isinstance(row.get("physical_observation_ref"), str)
            and isinstance(row.get("physical_observation_digest"), str)
        )

    matches = tuple(
        row
        for row in rows
        if is_metric_row(row)
        and row.get("effective_experiment_digest") == effective_experiment
        and row.get("effective_family_digest") == effective_family
    )
    unique_matches = {
        str(row["physical_observation_digest"]): row for row in matches
    }
    if len(unique_matches) != 1:
        return None
    observation = next(iter(unique_matches.values()))
    target_candidate = metadata.get("frontier_candidate_id", task.get("candidate_id"))
    target_semantic = metadata.get(
        "frontier_candidate_semantic_digest",
        task.get("candidate_semantic_digest"),
    )
    target_experiment = metadata.get("effective_experiment_digest")
    targets = tuple(
        row
        for row in rows
        if is_metric_row(row)
        and row.get("candidate_id") == target_candidate
        and row.get("candidate_semantic_digest") == target_semantic
        and row.get("effective_experiment_digest") == target_experiment
    )
    unique_targets = {
        str(row["physical_observation_digest"]): row for row in targets
    }
    if len(unique_targets) != 1:
        return None
    target = next(iter(unique_targets.values()))
    observation_value = float(observation["development_metrics"][metric_key])
    target_value = float(target["development_metrics"][metric_key])
    return canonical_value(
        {
            "schema": "recclaw.research-line.existing-observation-rebind.v1",
            "task_id": pending_task.get("task_id", task.get("task_id")),
            "operation": operation,
            "protocol_digest": context.protocol_digest,
            "dataset": dataset,
            "split": split,
            "metric": metric_name,
            "metric_contract_digest": metric_contract_digest,
            "observation_seed": str(observation_seed),
            "effective_experiment_digest": effective_experiment,
            "effective_family_digest": effective_family,
            "observation_candidate_id": observation.get("candidate_id"),
            "observation_candidate_semantic_digest": observation.get(
                "candidate_semantic_digest"
            ),
            "physical_observation_ref": observation["physical_observation_ref"],
            "physical_observation_digest": observation[
                "physical_observation_digest"
            ],
            "observation_round_index": observation.get("round_index"),
            "observation_value": observation_value,
            "target_candidate_id": target.get("candidate_id"),
            "target_candidate_semantic_digest": target.get(
                "candidate_semantic_digest"
            ),
            "target_physical_observation_ref": target[
                "physical_observation_ref"
            ],
            "target_value": target_value,
            "comparator_delta": observation_value - target_value,
            "metric_count_increment": 0,
            "runner_call_count": 0,
        }
    )


def _apply_existing_observation_rebind(
    *,
    interpretation: MissingSearchInterpretation,
    pending_task: Mapping[str, Any],
    rebind: Mapping[str, Any],
) -> MissingSearchInterpretation:
    successor = interpretation.successor_context
    memory = dict(successor.scientific_memory)
    global_memory = memory.get("global_memory")
    global_memory = dict(global_memory) if isinstance(global_memory, Mapping) else {}
    queue = ResearchTaskQueueV2.from_dict(global_memory.get("task_queue"))
    task_id = str(rebind["task_id"])
    task = queue.get(task_id)
    if task is None:
        raw_task = pending_task.get("task_record")
        if not isinstance(raw_task, Mapping):
            return interpretation
        task = ResearchTaskRecordV2.from_dict(raw_task)
        queue = queue.enqueue(task)
    if task.status in {ResearchTaskStatusV2.PENDING, ResearchTaskStatusV2.ACTIVE}:
        metadata = dict(task.metadata)
        metadata.update(
            {
                "execution_state": "BOUND_TO_EXISTING_OBSERVATION",
                "existing_observation_rebind": canonical_value(dict(rebind)),
            }
        )
        queue = queue._replace(replace(task, metadata=metadata))
        queue = queue.satisfy(
            task_id,
            evidence=(str(rebind["observation_seed"]),),
            reason="EXACT_EXISTING_OBSERVATION_REBOUND",
        )
    elif task.status is not ResearchTaskStatusV2.SATISFIED:
        return interpretation
    durable = queue.get(task_id)
    transition = canonical_value(
        {
            "satisfied_task_ids": (task_id,),
            "closed_task_ids": (),
            "created_task_ids": (),
            "deferred_requirements": (),
            "metric_bearing_evidence": True,
            "metric_count_increment": 0,
            "existing_observation_rebind": canonical_value(dict(rebind)),
        }
    )
    feedback = dict(global_memory.get("latest_feedback", {}))
    feedback.update(
        {
            "research_task_slot": None,
            "task_queue_head": (
                queue.select_next().to_dict()
                if queue.select_next() is not None
                else None
            ),
            "task_queue_transition": transition,
            "task_queue_digest": queue.digest,
            "existing_observation_rebind": canonical_value(dict(rebind)),
        }
    )
    prior_rebinds = [
        item
        for item in global_memory.get("existing_observation_rebindings", ())
        if isinstance(item, Mapping) and item.get("task_id") != task_id
    ]
    prior_rebinds.append(canonical_value(dict(rebind)))
    global_memory["existing_observation_rebindings"] = tuple(prior_rebinds[-128:])
    global_memory["latest_existing_observation_rebind"] = canonical_value(
        dict(rebind)
    )
    global_memory["latest_feedback"] = canonical_value(feedback)
    global_memory["task_queue"] = queue.to_dict()
    memory["global_memory"] = canonical_value(global_memory)
    memory["latest_feedback"] = global_memory["latest_feedback"]
    memory["task_queue"] = global_memory["task_queue"]
    rebound = replace(successor, scientific_memory=canonical_value(memory))
    return replace(
        interpretation,
        successor_context=rebound,
        behavior_after=_behavior(rebound, interpretation.policy_successor),
    )


def _attempted_executable_identities(
    context: ResearchContext,
) -> frozenset[str]:
    """Return executable identities whose exact behavior is already resolved.

    A metric or candidate-local deterministic failure retires the exact code
    identity.  Explicit worker-transient and shared-infrastructure failures do
    not: once the shared cause is repaired, the unchanged executable still has
    an unresolved scientific outcome and must remain runnable.
    """

    identities: set[str] = set()
    for attempt in _round_attempt_memory(context):
        if not _attempt_retires_executable_identity(attempt):
            continue
        value = attempt.get("executable_identity_digest")
        if isinstance(value, str) and value:
            identities.add(value)
    identities.update(
        _compact_identity_memory(
            context, "attempted_executable_identity_digests"
        )
    )
    return frozenset(identities)


def _attempt_retires_executable_identity(attempt: Mapping[str, Any]) -> bool:
    """Whether an attempt resolved the exact executable rather than its substrate."""

    if attempt.get("metric_bearing") is True:
        return True
    raw_scope = attempt.get("failure_scope")
    failure = attempt.get("failure")
    if raw_scope is None and isinstance(failure, Mapping):
        raw_scope = failure.get("failure_scope") or failure.get(
            "engineering_scope"
        )
    scope = str(raw_scope).strip().upper() if raw_scope is not None else ""
    return scope not in {"WORKER_TRANSIENT", "SHARED_INFRASTRUCTURE"}


def _attempted_semantic_identities(
    context: ResearchContext,
) -> frozenset[str]:
    """Return every semantic identity attempted in prior exploration rounds."""

    identities: set[str] = set()
    for attempt in _round_attempt_memory(context):
        if attempt.get("metric_bearing") is not True:
            continue
        value = attempt.get("candidate_semantic_digest")
        if isinstance(value, str) and value:
            identities.add(value)
    identities.update(
        _compact_identity_memory(context, "attempted_semantic_identity_digests")
    )
    return frozenset(identities)


def _attempted_effective_experiment_identities(
    context: ResearchContext,
) -> frozenset[str]:
    """Return execution-equivalent BL-ICF experiments already attempted."""

    identities: set[str] = set()
    for attempt in _round_attempt_memory(context):
        if attempt.get("metric_bearing") is not True:
            continue
        value = attempt.get("effective_experiment_digest")
        if isinstance(value, str) and value:
            identities.add(value)
    identities.update(
        _compact_identity_memory(
            context, "attempted_effective_experiment_digests"
        )
    )
    return frozenset(identities)


def _router_eligible_effective_bindings(
    *,
    context: ResearchContext,
    bindings: Sequence[SearchCandidateBindingV1],
    observation_seed: str,
    pending_task: Mapping[str, Any] | None,
    executed_semantic_seed_pairs: tuple[tuple[str, str], ...],
    search_space_adapter: SearchSpaceAdapter,
    confirmation_bindings: Mapping[str, Mapping[str, Any]],
    profile: SearchExecutableProfileV1,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]],
    evaluator: Mapping[str, Any],
    split: str,
) -> tuple[SearchCandidateBindingV1, ...]:
    """Remove already-attempted effective experiments before Router ALLOW."""

    attempted = _attempted_effective_experiment_identities(context)
    eligible: list[SearchCandidateBindingV1] = []
    for binding in bindings:
        qualified_execution = (
            qualified_execution_by_capability.get(binding.capability_ref)
            if binding.entry_origin
            is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
            else None
        )
        _adapter_binding, effective_identity, _recipe = (
            _adapter_execution_attestation(
                binding,
                search_space_adapter=search_space_adapter,
                confirmation_binding=confirmation_bindings.get(
                    binding.proposal.candidate_id
                ),
                execution_context={
                    "profile": profile,
                    "qualified_execution": qualified_execution,
                    "evaluator": evaluator,
                    "split": split,
                },
            )
        )
        if (
            str(effective_identity["effective_experiment_digest"])
            not in attempted
            or _explicit_new_seed_confirmation(
                binding=binding,
                observation_seed=observation_seed,
                pending_task=pending_task,
                executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            )
        ):
            eligible.append(binding)
    return tuple(eligible)


def _explicit_new_seed_confirmation(
    *,
    binding: SearchCandidateBindingV1,
    observation_seed: str,
    pending_task: Mapping[str, Any] | None,
    executed_semantic_seed_pairs: tuple[tuple[str, str], ...],
) -> bool:
    if pending_task is None:
        return False
    task_type = pending_task.get("task_type")
    task_type = getattr(task_type, "value", task_type)
    task_status = pending_task.get("task_status")
    task_status = getattr(task_status, "value", task_status)
    required_seed = pending_task.get("required_seed_or_control")
    seed_matches = required_seed == observation_seed or (
        required_seed == "NEXT_DEVELOPMENT_SEED"
        and (binding.mechanism_semantics_digest, observation_seed)
        not in executed_semantic_seed_pairs
    )
    return (
        task_type == "VALIDATE_SAME_CANDIDATE"
        and task_status in {"PENDING", "ACTIVE"}
        and pending_task.get("candidate_semantic_digest")
        == binding.mechanism_semantics_digest
        and seed_matches
        and (binding.mechanism_semantics_digest, observation_seed)
        not in executed_semantic_seed_pairs
    )


def _repeated_executable_identity(
    *,
    recipe: Mapping[str, Any],
    history: frozenset[str],
) -> str | None:
    identity = _metric_executable_identity_digest(recipe)
    return "executable_identity_digest" if identity in history else None


def _explicit_attempt_budget(
    budget_snapshot: Mapping[str, Any],
    explicit_limit: int | None,
    *,
    required: bool = True,
) -> int | None:
    if explicit_limit is not None:
        value = explicit_limit
    else:
        value = None
        for field_name in (
            "max_attempts_per_round",
            "round_attempt_budget",
            "remaining_attempt_budget",
            "attempt_budget",
        ):
            candidate = budget_snapshot.get(field_name)
            if candidate is not None:
                value = candidate
                break
        if value is None:
            if required:
                raise ValueError(
                    "attempt_scheduler=True requires an explicit frozen "
                    "max_attempts_per_round or round attempt budget"
                )
            return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("round attempt budget must be a non-negative integer")

    return value


def _attempt_budget_limit(
    budget_snapshot: Mapping[str, Any],
    explicit_limit: int | None,
    pool_size: int,
) -> int:
    value = _explicit_attempt_budget(
        budget_snapshot,
        explicit_limit,
        required=True,
    )
    assert value is not None
    return min(value, pool_size)


def _attempt_failure_scope(
    candidate_run: Mapping[str, Any],
    failure_detail: Mapping[str, Any] | None,
) -> str:
    """Read an explicit infrastructure scope without guessing from a status.

    Existing workers do not emit a scope, so their typed resource/runtime
    failures retain the historical candidate-local default.  A worker or
    adapter that knows the failure is shared must say so explicitly; this is
    what prevents a shared outage from quarantining the rest of the pool.
    """

    raw: Any = None
    for field_name in (
        "failure_scope",
        "engineering_scope",
        "failure_scope_class",
    ):
        if candidate_run.get(field_name) is not None:
            raw = candidate_run[field_name]
            break
    if raw is None and isinstance(candidate_run.get("resource_telemetry"), Mapping):
        telemetry = candidate_run["resource_telemetry"]
        for field_name in ("failure_scope", "engineering_scope", "scope"):
            if telemetry.get(field_name) is not None:
                raw = telemetry[field_name]
                break
    if raw is None and isinstance(failure_detail, Mapping):
        raw = failure_detail.get("failure_scope") or failure_detail.get(
            "engineering_scope"
        )
    normalized = str(raw).strip().upper() if raw is not None else ""
    if candidate_run.get("shared_infrastructure_failure") is True:
        normalized = "SHARED_INFRASTRUCTURE"
    if normalized in {
        "CANDIDATE_LOCAL",
        "LINEAGE_COMPUTE_PATTERN",
        "WORKER_TRANSIENT",
        "SHARED_INFRASTRUCTURE",
    }:
        return normalized
    return "CANDIDATE_LOCAL"


def _attempt_consumed_slot_training_budget(attempt: RoundAttemptV1) -> bool:
    """Whether the one formal training run ended without a metric.

    ``RESOURCE_CENSORED`` is emitted by the main runner after it actually ran
    the candidate up to the frozen resource ceiling.  It is useful typed
    negative evidence, but it cannot authorize a second training run in the
    same fixed slot.
    """

    return bool(
        not attempt.metric_bearing
        and attempt.failure_scope != "SHARED_INFRASTRUCTURE"
        and str(attempt.candidate_run.get("exit_status", "")).upper()
        == "RESOURCE_CENSORED"
    )


def _guard_diagnostic_closure(
    *,
    identity: Any,
    binding: SearchCandidateBindingV1,
    candidate_run: Mapping[str, Any],
    response: Mapping[str, Any] | None,
) -> Any:
    """Close a Guard rejection as D0 diagnostic evidence, never as science."""

    current_attempt = _guard_current_attempt_class(response or {})
    detail = canonical_value(
        {
            "reason": "EVIDENCE_GUARD_ATTEMPT_NOT_SCIENTIFIC",
            "candidate_id": binding.proposal.candidate_id,
            "candidate_semantic_digest": binding.mechanism_semantics_digest,
            "current_attempt_class": current_attempt,
            "candidate_run_status": candidate_run.get("run_status"),
            "guard_response_digest": (
                sha256_digest(response) if isinstance(response, Mapping) else None
            ),
        }
    )
    detail_digest = sha256_digest(detail)
    detail_ref = content_id("recclaw-search-execution-failure-v1", detail)
    return close_scientific_episode(
        comparison_identity=identity,
        failure_class=(
            ResearchFailureClassV1.PROTOCOL
            if current_attempt == "PROTOCOL_DRIFT"
            else ResearchFailureClassV1.INTERFACE
        ),
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        failure_detail_ref=detail_ref,
        failure_detail_digest=detail_digest,
    )


def _candidate_run_binding(
    candidate_run: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    observation_seed: str,
) -> ExperimentBindingV1:
    try:
        run_binding = ExperimentBindingV1.from_canonical_dict(
            candidate_run.get("experiment_binding")
        )
    except ExperimentBindingError as error:
        raise ValueError(str(error)) from error
    _validate_experiment_binding(
        run_binding,
        recipe=recipe,
        observation_seed=observation_seed,
    )
    if candidate_run.get("execution_recipe_digest") != sha256_digest(recipe):
        raise ValueError("runner result is not bound to the selected execution recipe")
    if str(candidate_run.get("seed")) != observation_seed:
        raise ValueError("runner result seed differs from the frozen observation seed")
    run_binding_digest = candidate_run.get("experiment_binding_digest")
    if (
        run_binding_digest != run_binding.digest
        or candidate_run.get("experiment_binding_ref") != run_binding.ref
        or candidate_run.get("binding_digest", run_binding_digest) != run_binding_digest
    ):
        raise ValueError("runner result carries an inconsistent Experiment Binding identity")
    return run_binding


def _post_run_implementation_fidelity_failure(
    *,
    binding: SearchCandidateBindingV1,
    recipe: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    candidate_root_by_capability: Mapping[str, str | Path],
) -> Mapping[str, Any] | None:
    """Quarantine a completed result whose executed code falsifies its claims."""

    if (
        binding.entry_origin is not SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
        or str(candidate_run.get("exit_status", "")).upper() != "SUCCESS"
    ):
        return None
    run_experiment_binding = candidate_run.get("experiment_binding")
    binding_candidate_root = (
        run_experiment_binding.get("candidate_root_path")
        if isinstance(run_experiment_binding, Mapping)
        else None
    )
    candidate_root = (
        recipe.get("candidate_root_path")
        or binding_candidate_root
        or candidate_root_by_capability.get(binding.capability_ref)
    )
    metrics = candidate_run.get("metrics")

    def failure(reason_code: str, message: str, **evidence: Any) -> Mapping[str, Any]:
        return canonical_value(
            {
                "stage": "POST_WORKER_IMPLEMENTATION_FIDELITY",
                "failure_class": "IMPLEMENTATION",
                "reason_code": reason_code,
                "message": message,
                "candidate_id": binding.proposal.candidate_id,
                "actual_seed": str(candidate_run.get("seed")),
                "measured_metrics": (
                    dict(metrics) if isinstance(metrics, Mapping) else {}
                ),
                "physical_observation_ref": candidate_run.get(
                    "physical_observation_ref"
                ),
                "physical_observation_digest": candidate_run.get(
                    "physical_observation_digest"
                ),
                "measurement_retained": True,
                "causal_credit_allowed": False,
                **evidence,
            }
        )

    if candidate_root is None:
        return failure(
            "COMPILED_MECHANISM_SOURCE_UNAVAILABLE",
            "qualified successful execution has no exact candidate source root",
        )
    entrypoint_path = recipe["entrypoint"].split(":", 1)[0].replace(".", "/") + ".py"
    source_path = Path(candidate_root) / entrypoint_path
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as error:
        return failure(
            "COMPILED_MECHANISM_SOURCE_UNAVAILABLE",
            "qualified successful execution candidate source cannot be read",
            executed_source_path=entrypoint_path,
            candidate_root_path=str(Path(candidate_root)),
            source_error=type(error).__name__,
        )
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if source_sha256 != recipe.get("entrypoint_source_sha256"):
        raise ValueError("executed candidate source differs from the execution recipe")
    program_payload = binding.proposal.mechanism_program.get("program_payload", {})
    components = (
        program_payload.get("components", ())
        if isinstance(program_payload, Mapping)
        else ()
    )
    primitive_ids = tuple(
        sorted(
            {
                str(component["primitive_id"])
                for component in components
                if isinstance(component, Mapping)
                and isinstance(component.get("primitive_id"), str)
            }
        )
    )
    component_specs = {
        str(component["component_id"]): canonical_value(component)
        for component in components
        if isinstance(component, Mapping)
        and isinstance(component.get("component_id"), str)
    }
    try:
        validate_mechanism_source_behavior(
            source,
            primitive_ids=primitive_ids,
            component_specs=component_specs,
            execution_config=(
                recipe.get("config")
                if isinstance(recipe.get("config"), Mapping)
                else None
            ),
            precompute_required=(
                program_payload.get("estimated_cost", {}).get(
                    "precompute_required"
                )
                is True
                if isinstance(program_payload.get("estimated_cost"), Mapping)
                else False
            ),
        )
    except InnovationSpineError as error:
        if error.reason_code != "COMPILED_MECHANISM_BEHAVIOR_MISMATCH":
            raise
        return failure(
            error.reason_code,
            str(error),
            executed_source_path=entrypoint_path,
            executed_source_sha256=source_sha256,
            declared_primitive_ids=primitive_ids,
        )
    return None


def _implementation_fidelity_diagnostic_closure(
    *,
    identity: Any,
    binding: SearchCandidateBindingV1,
    candidate_run: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> Any:
    detail = canonical_value(
        {
            **dict(failure),
            "candidate_semantic_digest": binding.mechanism_semantics_digest,
            "physical_observation_ref": candidate_run.get(
                "physical_observation_ref"
            ),
            "physical_observation_digest": candidate_run.get(
                "physical_observation_digest"
            ),
        }
    )
    return close_scientific_episode(
        comparison_identity=identity,
        failure_class=ResearchFailureClassV1.IMPLEMENTATION,
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        failure_detail_ref=content_id(
            "recclaw-implementation-fidelity-failure-v1",
            detail,
        ),
        failure_detail_digest=sha256_digest(detail),
    )


def _selected_outcome_for_binding(
    binding: SearchCandidateBindingV1,
    *,
    search_pairs: Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]],
    open_search_pairs: Sequence[
        tuple[ProducerOutcome, CapabilityResolutionV1, QualifiedSearchCandidateProtocolV1]
    ],
) -> ProducerOutcome:
    candidate_id = binding.proposal.candidate_id
    for outcome, _resolution in search_pairs:
        if (
            outcome.source_proposal is not None
            and outcome.source_proposal.candidate_id == candidate_id
        ):
            return outcome
    for outcome, _resolution, candidate in open_search_pairs:
        if candidate.candidate_id == candidate_id:
            return outcome
    raise ValueError("routed binding has no matching Producer outcome")


def _validate_handoff_resource_profile(
    profile: Mapping[str, Any] | None,
    *,
    binding: SearchCandidateBindingV1,
    qualified_execution: Mapping[str, Any] | None,
    require_complete: bool,
    require_explicit_prediction: bool = False,
) -> None:
    if profile is None:
        if require_complete:
            raise ValueError(
                "portfolio handoff lacks the admitted resource profile for "
                + binding.proposal.candidate_id
            )
        return
    if not isinstance(profile, Mapping) or not profile:
        raise ValueError("resource profile must be a non-empty mapping")
    candidate_ref = profile.get("candidate_ref")
    if candidate_ref is not None and candidate_ref != binding.capability_ref:
        raise ValueError("resource profile candidate identity drift")
    if require_complete and candidate_ref != binding.capability_ref:
        raise ValueError("resource profile lacks the binding capability identity")
    if (
        qualified_execution is not None
        and profile.get("candidate_package_digest") is not None
        and profile.get("candidate_package_digest")
        != qualified_execution.get("candidate_package_digest")
    ):
        raise ValueError("resource profile package identity drift")
    if profile.get("effect_fields_consumed") not in (None, (), []):
        raise ValueError("resource profile contains mechanism-effect evidence")
    if profile.get("outcome_fields_consumed") not in (None, (), []):
        raise ValueError("resource profile contains outcome evidence")
    if profile.get("held_out_reads") not in (None, 0):
        raise ValueError("resource profile contains heldout evidence")
    if (
        "mechanism_effect_update_allowed" in profile
        and profile.get("mechanism_effect_update_allowed") is not False
    ):
        raise ValueError("resource profile permits mechanism-effect updates")
    if require_explicit_prediction:
        prediction = profile.get("prediction")
        prediction = prediction if isinstance(prediction, Mapping) else {}
        explicit = tuple(
            value
            for value in (
                profile.get("predicted_gpu_seconds"),
                profile.get("predicted_gpu_worker_seconds"),
                prediction.get("predicted_gpu_seconds"),
                prediction.get("predicted_gpu_worker_seconds"),
            )
            if value is not None
        )
        if not explicit:
            raise ValueError(
                "formal profile handoff requires explicit identity-bound "
                "predicted_gpu_seconds or predicted_gpu_worker_seconds"
            )
        try:
            numeric = tuple(float(value) for value in explicit)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "formal profile handoff GPU-worker prediction is not numeric"
            ) from error
        if any(value <= 0.0 for value in numeric):
            raise ValueError(
                "formal profile handoff GPU-worker prediction must be positive"
            )
        if any(value != numeric[0] for value in numeric[1:]):
            raise ValueError(
                "formal profile handoff GPU-worker prediction fields disagree"
            )


def _durable_evidence_digests(
    *,
    qualified_execution: Mapping[str, Any] | None,
    candidate_root_path: str | None,
    resource_profile: Mapping[str, Any] | None,
) -> dict[str, str]:
    if resource_profile is None:
        raise ValueError("durable evidence digest requires a resource profile")
    digests = {"resource_profile": sha256_digest(resource_profile)}
    if qualified_execution is not None:
        digests["qualified_execution"] = sha256_digest(qualified_execution)
    if candidate_root_path is not None:
        digests["candidate_root_path"] = sha256_digest(
            {"candidate_root_path": str(Path(candidate_root_path).resolve())}
        )
    return canonical_value(digests)


def _binding_parent_id(binding: SearchCandidateBindingV1) -> str | None:
    return exact_parent_candidate_id(binding.proposal)


def _validate_candidate_handoff(
    handoff: RoundCandidateHandoffV1,
    *,
    binding: SearchCandidateBindingV1,
    active_profile: SearchExecutableProfileV1,
    require_complete: bool,
    evaluator: Mapping[str, Any],
    split: str,
    search_space_adapter: SearchSpaceAdapter,
) -> None:
    if not isinstance(handoff, RoundCandidateHandoffV1):
        raise ValueError("candidate handoff must be RoundCandidateHandoffV1")
    if handoff.candidate_id != binding.proposal.candidate_id:
        raise ValueError("candidate handoff candidate identity drift")
    if handoff.binding_digest != binding.digest:
        raise ValueError("candidate handoff binding digest drift")
    if handoff.portfolio_candidate.semantic_digest != binding.mechanism_semantics_digest:
        raise ValueError("candidate handoff portfolio semantic identity drift")
    formal_profile = handoff.portfolio_profile is not None
    if formal_profile:
        if handoff.resource_profile is None:
            raise ValueError("formal profile handoff lacks resource evidence")
        if handoff.portfolio_profile.candidate != handoff.portfolio_candidate:
            raise ValueError("candidate handoff profile/candidate identity drift")
        expected_family = getattr(binding.proposal, "mechanism_axis", None)
        if (
            expected_family is not None
            and handoff.portfolio_candidate.family_id != expected_family
        ):
            raise ValueError("candidate handoff family identity drift")
        expected_parent = _binding_parent_id(binding)
        if handoff.portfolio_candidate.parent_id != expected_parent:
            raise ValueError("candidate handoff parent identity drift")
        if handoff.lineage_identity_digest != lineage_identity_digest(
            handoff.portfolio_candidate
        ):
            raise ValueError("candidate handoff lineage identity drift")
        if handoff.durable_evidence_digests is not None:
            if handoff.durable_evidence_digests != _durable_evidence_digests(
                qualified_execution=handoff.qualified_execution,
                candidate_root_path=handoff.candidate_root_path,
                resource_profile=handoff.resource_profile,
            ):
                raise ValueError("candidate handoff durable evidence digest drift")
    if binding.entry_origin is SearchProfileEntryOriginV1.FIXED_66:
        if handoff.qualified_execution is not None:
            raise ValueError("FIXED_66 handoff cannot carry qualified execution")
        if handoff.candidate_root_path is not None:
            raise ValueError("FIXED_66 handoff cannot carry a candidate root")
        qualified_execution = None
    elif binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
        if handoff.qualified_execution is None:
            raise ValueError("qualified handoff lacks explicit execution inputs")
        if handoff.candidate_root_path is None:
            raise ValueError("qualified handoff lacks a candidate root")
        root = Path(handoff.candidate_root_path)
        if not root.is_dir() or not (root / "recclaw_ext").is_dir():
            raise ValueError(
                "qualified handoff candidate root is unavailable: " + str(root)
            )
        qualified_execution = handoff.qualified_execution
    else:
        raise ValueError("candidate handoff has an unknown Search profile origin")
    try:
        _adapter_execution_attestation(
            binding,
            search_space_adapter=search_space_adapter,
            confirmation_binding=None,
            execution_context={
                "profile": active_profile,
                "qualified_execution": qualified_execution,
                "evaluator": evaluator,
                "split": split,
            },
        )
    except (TypeError, ValueError) as error:
        raise ValueError("candidate handoff execution identity is invalid") from error
    _validate_handoff_resource_profile(
        handoff.resource_profile,
        binding=binding,
        qualified_execution=qualified_execution,
        require_complete=require_complete,
        require_explicit_prediction=formal_profile,
    )


def _validate_candidate_handoffs(
    handoffs: Sequence[RoundCandidateHandoffV1],
    *,
    bindings: Sequence[SearchCandidateBindingV1],
    active_profile: SearchExecutableProfileV1,
    require_complete: bool,
    evaluator: Mapping[str, Any],
    split: str,
    search_space_adapter: SearchSpaceAdapter,
) -> tuple[RoundCandidateHandoffV1, ...]:
    normalized = tuple(handoffs)
    by_candidate: dict[str, RoundCandidateHandoffV1] = {}
    for handoff in normalized:
        if not isinstance(handoff, RoundCandidateHandoffV1):
            raise ValueError("candidate handoff factory returned an invalid record")
        if handoff.candidate_id in by_candidate:
            raise ValueError("candidate handoff candidate identity is duplicated")
        by_candidate[handoff.candidate_id] = handoff
    expected_ids = tuple(binding.proposal.candidate_id for binding in bindings)
    expected_set = set(expected_ids)
    actual_set = set(by_candidate)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(
            "portfolio handoff coverage is not exact (" + "; ".join(detail) + ")"
        )
    ordered: list[RoundCandidateHandoffV1] = []
    for binding in bindings:
        handoff = by_candidate[binding.proposal.candidate_id]
        _validate_candidate_handoff(
            handoff,
            binding=binding,
            active_profile=active_profile,
            require_complete=require_complete,
            evaluator=evaluator,
            split=split,
            search_space_adapter=search_space_adapter,
        )
        ordered.append(handoff)
    return tuple(ordered)


def _static_candidate_handoffs(
    portfolio_candidates: Sequence[PortfolioCandidateV2],
    *,
    bindings: Sequence[SearchCandidateBindingV1],
    active_profile: SearchExecutableProfileV1,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]],
    candidate_root_by_capability: Mapping[str, str | Path],
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]],
    evaluator: Mapping[str, Any],
    split: str,
    search_space_adapter: SearchSpaceAdapter,
) -> tuple[RoundCandidateHandoffV1, ...]:
    by_candidate: dict[str, PortfolioCandidateV2] = {}
    for candidate in portfolio_candidates:
        if candidate.candidate_id in by_candidate:
            raise ValueError("portfolio candidate identity is duplicated")
        by_candidate[candidate.candidate_id] = candidate
    expected_ids = {binding.proposal.candidate_id for binding in bindings}
    if set(by_candidate) != expected_ids:
        missing = sorted(expected_ids - set(by_candidate))
        extra = sorted(set(by_candidate) - expected_ids)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(
            "portfolio candidate coverage is not exact ("
            + "; ".join(detail)
            + ")"
        )
    handoffs = tuple(
        RoundCandidateHandoffV1(
            candidate_id=binding.proposal.candidate_id,
            binding_digest=binding.digest,
            portfolio_candidate=by_candidate[binding.proposal.candidate_id],
            qualified_execution=(
                qualified_execution_by_capability.get(binding.capability_ref)
                if binding.entry_origin
                is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
                else None
            ),
            candidate_root_path=(
                candidate_root_by_capability.get(binding.capability_ref)
                if binding.entry_origin
                is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
                else None
            ),
            resource_profile=resource_profile_by_capability.get(
                binding.capability_ref
            ),
        )
        for binding in bindings
        if binding.proposal.candidate_id in by_candidate
    )
    return _validate_candidate_handoffs(
        handoffs,
        bindings=bindings,
        active_profile=active_profile,
        require_complete=False,
        evaluator=evaluator,
        split=split,
        search_space_adapter=search_space_adapter,
    )


def _candidate_handoffs_for_round(
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    resolutions: Sequence[
        tuple[ProducerOutcome, CapabilityResolutionV1 | None]
    ],
    search_bindings: tuple[SearchCandidateBindingV1, ...],
    portfolio_candidates: tuple[PortfolioCandidateV2, ...],
    research_profile_source: ResearchProfileSourceV1 | None,
    candidate_handoff_factory: CandidateHandoffFactory | None,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None,
    candidate_root_by_capability: Mapping[str, str | Path] | None,
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None,
    evaluator: Mapping[str, Any],
    split: str,
    search_space_adapter: SearchSpaceAdapter,
) -> tuple[RoundCandidateHandoffV1, ...]:
    qualified = dict(qualified_execution_by_capability or {})
    roots = dict(candidate_root_by_capability or {})
    resources = dict(resource_profile_by_capability or {})
    if research_profile_source is not None:
        if not isinstance(research_profile_source, ResearchProfileSourceV1):
            raise ValueError(
                "research_profile_source must be ResearchProfileSourceV1"
            )
        if candidate_handoff_factory is not None:
            raise ValueError(
                "research_profile_source and candidate_handoff_factory are mutually exclusive"
            )
        records = research_profile_source.build_profiles(
            context=context,
            active_profile=active_profile,
            resolutions=tuple(resolutions),
            search_bindings=search_bindings,
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resources,
        )
        if len(records) != len(search_bindings):
            raise ValueError("research profile source did not cover every binding")
        handoffs = tuple(
            RoundCandidateHandoffV1(
                candidate_id=binding.proposal.candidate_id,
                binding_digest=binding.digest,
                portfolio_candidate=record.portfolio_profile.candidate,
                qualified_execution=record.qualified_execution,
                candidate_root_path=record.candidate_root_path,
                resource_profile=record.resource_profile,
                portfolio_profile=record.portfolio_profile,
                profile_digest=record.profile_digest,
                source_schema_version=research_profile_source.schema_version,
                source_ref=research_profile_source.source_ref,
                source_digest=research_profile_source.source_digest,
                lineage_identity_digest=record.lineage_identity_digest,
                durable_evidence_digests=record.durable_evidence_digests,
            )
            for binding, record in zip(search_bindings, records, strict=True)
        )
        handoffs = _validate_candidate_handoffs(
            handoffs,
            bindings=search_bindings,
            active_profile=active_profile,
            require_complete=True,
            evaluator=evaluator,
            split=split,
            search_space_adapter=search_space_adapter,
        )
        if portfolio_candidates:
            expected = {
                item.candidate_id: item.to_dict() for item in portfolio_candidates
            }
            actual = {
                item.candidate_id: item.portfolio_candidate.to_dict()
                for item in handoffs
            }
            if actual != expected:
                raise ValueError(
                    "static portfolio candidates disagree with profile source"
                )
        return handoffs
    if candidate_handoff_factory is not None:
        if not callable(candidate_handoff_factory):
            raise ValueError("candidate_handoff_factory must be callable")
        produced = candidate_handoff_factory(
            context=context,
            active_profile=active_profile,
            resolutions=tuple(resolutions),
            search_bindings=search_bindings,
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resources,
        )
        if isinstance(produced, Mapping):
            produced = tuple(produced.values())
        if isinstance(produced, (str, bytes)) or not isinstance(produced, Sequence):
            raise ValueError(
                "candidate_handoff_factory must return a sequence of handoffs"
            )
        handoffs = _validate_candidate_handoffs(
            tuple(produced),
            bindings=search_bindings,
            active_profile=active_profile,
            require_complete=True,
            evaluator=evaluator,
            split=split,
            search_space_adapter=search_space_adapter,
        )
        for binding, handoff in zip(search_bindings, handoffs, strict=True):
            if binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY:
                expected_execution = qualified.get(binding.capability_ref)
                if (
                    expected_execution is not None
                    and handoff.qualified_execution != expected_execution
                ):
                    raise ValueError("candidate handoff execution state drift")
                expected_root = roots.get(binding.capability_ref)
                if expected_root is not None and (
                    handoff.candidate_root_path is None
                    or Path(handoff.candidate_root_path).resolve()
                    != Path(expected_root).resolve()
                ):
                    raise ValueError("candidate handoff root state drift")
            expected_resource = resources.get(binding.capability_ref)
            if expected_resource is not None and (
                handoff.resource_profile is None
                or canonical_value(dict(handoff.resource_profile))
                != canonical_value(dict(expected_resource))
            ):
                raise ValueError("candidate handoff resource state drift")
        if portfolio_candidates:
            expected = {
                item.candidate_id: item.to_dict() for item in portfolio_candidates
            }
            actual = {
                item.candidate_id: item.portfolio_candidate.to_dict()
                for item in handoffs
            }
            if actual != expected:
                raise ValueError(
                    "static portfolio candidates disagree with dynamic handoff"
                )
        return handoffs
    if not portfolio_candidates:
        return ()
    return _static_candidate_handoffs(
        portfolio_candidates,
        bindings=search_bindings,
        active_profile=active_profile,
        qualified_execution_by_capability=qualified,
        candidate_root_by_capability=roots,
        resource_profile_by_capability=resources,
        evaluator=evaluator,
        split=split,
        search_space_adapter=search_space_adapter,
    )


def _execution_recipe_for_handoff(
    binding: SearchCandidateBindingV1,
    *,
    profile: SearchExecutableProfileV1,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None,
    candidate_root_by_capability: Mapping[str, str | Path] | None,
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None,
    candidate_handoffs: Sequence[RoundCandidateHandoffV1],
    evaluator: Mapping[str, Any],
    split: str,
    search_space_adapter: SearchSpaceAdapter,
    confirmation_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    by_candidate = {item.candidate_id: item for item in candidate_handoffs}
    handoff = by_candidate.get(binding.proposal.candidate_id)
    if candidate_handoffs and handoff is None:
        raise ValueError("selected binding has no candidate handoff")
    qualified_execution = (
        handoff.qualified_execution
        if handoff is not None
        and binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
        else (
            (qualified_execution_by_capability or {}).get(binding.capability_ref)
            if binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
            else None
        )
    )
    search_space_binding, effective_identity, adapter_recipe = (
        _adapter_execution_attestation(
            binding,
            search_space_adapter=search_space_adapter,
            confirmation_binding=confirmation_binding,
            execution_context={
                "profile": profile,
                "qualified_execution": qualified_execution,
                "evaluator": evaluator,
                "split": split,
            },
        )
    )
    recipe = dict(adapter_recipe)
    native_cadence = (qualified_execution or {}).get("native_training_cadence")
    if native_cadence is not None:
        config = dict(recipe["config"])
        mismatched = [
            key for key, value in native_cadence.items()
            if key in config and config[key] != value
        ]
        if mismatched:
            raise ValueError("qualified execution cadence differs from native protocol: "
                             + ", ".join(mismatched))
        recipe["config"] = {**config, **dict(native_cadence)}
    resource_profile = (
        handoff.resource_profile
        if handoff is not None and handoff.resource_profile is not None
        else (resource_profile_by_capability or {}).get(binding.capability_ref)
    )
    if resource_profile is not None:
        prediction = resource_profile.get("prediction")
        config = recipe.get("config")
        if not isinstance(prediction, Mapping) or not isinstance(config, Mapping):
            raise ValueError(
                "qualified resource profile lacks recipe cadence evidence"
            )
        cadence = {
            "epochs": prediction.get("requested_epochs"),
            "eval_step": prediction.get("native_early_stop_eval_step"),
            "stopping_step": prediction.get(
                "native_early_stop_stopping_step"
            ),
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in cadence.values()
        ):
            raise ValueError(
                "qualified resource profile has invalid recipe cadence evidence"
            )
        mismatched = [
            field_name
            for field_name, expected in cadence.items()
            if field_name in config and config[field_name] != expected
        ]
        if mismatched:
            raise ValueError(
                "qualified execution cadence differs from its resource profile: "
                + ", ".join(mismatched)
            )
        recipe["config"] = {**dict(config), **cadence}
    candidate_root = (
        handoff.candidate_root_path
        if handoff is not None and handoff.candidate_root_path is not None
        else (candidate_root_by_capability or {}).get(binding.capability_ref)
        if binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
        else None
    )
    if resource_profile is not None:
        recipe["resource_prediction"] = resource_profile
    if candidate_root is not None:
        recipe["candidate_root_path"] = str(Path(candidate_root).resolve())
    if resource_profile is not None or candidate_root is not None:
        recipe = canonical_value(recipe)
    return (
        recipe,
        effective_identity,
        _adapter_route_evidence(search_space_binding, effective_identity),
    )


def _mechanism_projection_for_binding(
    binding: SearchCandidateBindingV1,
    selected_outcome: ProducerOutcome,
) -> Mapping[str, Any]:
    if selected_outcome.source_proposal is not None:
        return binding.proposal.mechanism_program
    return {
        "schema": "recclaw-open-spec-realization-semantics-e0.v1",
        "semantic_identity_ref": binding.proposal.semantic_identity_ref,
        "semantic_identity_digest": binding.proposal.semantic_identity_digest,
        "research_spec_ref": binding.proposal.spec.spec_id,
        "research_spec_digest": binding.proposal.spec.digest,
        "candidate_package_ref": binding.proposal.candidate_package_ref,
        "candidate_package_digest": binding.proposal.candidate_package_digest,
        "execution_contract": binding.proposal.execution_contract,
        "parent_candidate_id": exact_parent_candidate_id(binding.proposal),
    }


def _route_metadata_for_attempt(
    *,
    acquisition: ExperimentAcquisitionResultV1,
    selected_outcome: ProducerOutcome,
    binding: SearchCandidateBindingV1,
    candidate_handoff: RoundCandidateHandoffV1 | None = None,
    next_discriminative_test: str,
    observation_seed: str,
    failure_scope: str | None = None,
    attempt_index: int | None = None,
    pending_task: Mapping[str, Any] | None = None,
    research_task_record: Mapping[str, Any] | None = None,
    confirmation_seed: str | None = None,
    search_space_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "route_trace_digest": acquisition.route_trace.digest,
        "selected_producer_role": selected_outcome.producer_role,
        "selected_candidate_id": binding.proposal.candidate_id,
        "selected_candidate_semantic_digest": binding.mechanism_semantics_digest,
        "selected_mechanism_axis": binding.proposal.mechanism_axis,
        "required_selected_runnable_probability": (
            binding.proposal.utility_features.runnable_probability
        ),
        "mechanism_program": _mechanism_projection_for_binding(
            binding, selected_outcome
        ),
        "selected_candidate_parent_id": exact_parent_candidate_id(
            binding.proposal
        ),
        "lineage_parent_binding": (
            selected_outcome.provenance.get("lineage_parent_binding")
            if isinstance(selected_outcome.provenance, Mapping)
            else None
        ),
        "lineage_candidate_id": getattr(
            binding.proposal,
            "compiler_candidate_id",
            binding.proposal.candidate_id,
        ),
        "lineage_program_digest": getattr(
            binding.proposal,
            "mechanism_program_digest",
            sha256_digest(binding.proposal.mechanism_program),
        ),
        "comparator_identity": None,
        "required_seed_or_control": observation_seed,
        "next_discriminative_test": next_discriminative_test,
    }
    if isinstance(research_task_record, Mapping):
        metadata["research_task_record"] = canonical_value(
            dict(research_task_record)
        )
    if isinstance(search_space_attestation, Mapping):
        attestation = canonical_value(dict(search_space_attestation))
        metadata["search_space_execution_binding"] = attestation
        for field_name in (
            "adapter_id",
            "binding_ref",
            "binding_digest",
            "effective_experiment_digest",
            "effective_family_digest",
        ):
            metadata[field_name] = attestation.get(field_name)
    if candidate_handoff is not None:
        portfolio_candidate = candidate_handoff.portfolio_candidate
        metadata.update(
            {
                "selected_candidate_family_id": portfolio_candidate.family_id,
                "selected_candidate_parent_id": portfolio_candidate.parent_id,
                "selected_compute_pattern": portfolio_candidate.compute_pattern,
                "selected_resource_admission_state": (
                    portfolio_candidate.resource_admission_state.value
                ),
            }
        )
        resource_profile = candidate_handoff.resource_profile
        profile = candidate_handoff.portfolio_profile
        durable_digests = candidate_handoff.durable_evidence_digests
        resource_digest = (
            durable_digests.get("resource_profile")
            if isinstance(durable_digests, Mapping)
            else None
        )
        if resource_digest is None and resource_profile is not None:
            resource_digest = sha256_digest(resource_profile)
        if profile is not None:
            metadata["selected_portfolio_profile_digest"] = profile.profile_digest
        if resource_digest is not None:
            metadata["selected_resource_evidence_digest"] = resource_digest
    if failure_scope is not None:
        metadata["failure_scope"] = failure_scope
    if attempt_index is not None:
        metadata["round_attempt_index"] = attempt_index
    if confirmation_seed is not None:
        metadata["confirmation_seed"] = confirmation_seed
    if isinstance(pending_task, Mapping):
        metadata["active_task_id"] = pending_task.get("task_id")
        if isinstance(pending_task.get("task_record"), Mapping):
            metadata["active_task_record"] = canonical_value(
                dict(pending_task["task_record"])
            )
            metadata["active_task_record_digest"] = sha256_digest(
                pending_task["task_record"]
            )
        task_type = pending_task.get("task_type")
        task_type = getattr(task_type, "value", task_type)
        same_candidate = (
            pending_task.get("candidate_semantic_digest")
            == binding.mechanism_semantics_digest
            and (
                pending_task.get("candidate_id") is None
                or pending_task.get("candidate_id")
                == binding.proposal.candidate_id
            )
        )
        required = pending_task.get("required_seed_or_control")
        evidence_present = pending_task.get("evidence_present", ())
        if isinstance(evidence_present, str):
            evidence_present = (evidence_present,)
        generic_new_seed = (
            required == _NEXT_DEVELOPMENT_SEED
            and isinstance(evidence_present, (tuple, list))
            and observation_seed not in evidence_present
        )
        if same_candidate and (
            (
                task_type == "VALIDATE_SAME_CANDIDATE"
                and (required == observation_seed or generic_new_seed)
            )
            or task_type == "REPAIR_IMPLEMENTATION"
            or (
                task_type in {"RUN_MATCHED_CONTROL", "RUN_ABLATION"}
                and required == observation_seed
            )
        ):
            metadata["satisfies_task_id"] = pending_task.get("task_id")
        resolution = pending_task.get("task_resolution")
        if isinstance(resolution, Mapping) and resolution.get("status") == "CLOSED":
            metadata["close_task_id"] = pending_task.get("task_id")
            metadata["close_task_reason"] = resolution.get("reason")
        if binding.proposal.candidate_id in _exact_confirmation_candidate_ids(
            pending_task
        ):
            task_record = pending_task.get("task_record")
            task_operation = (
                task_record.get("operation")
                if isinstance(task_record, Mapping)
                else None
            )
            task_operation = getattr(task_operation, "value", task_operation)
            metadata["comparator_identity"] = task_record.get(
                "comparator_identity"
            )
            metadata["task_operation"] = task_operation
            metadata["satisfies_task_id"] = pending_task.get("task_id")
    return metadata


def _run_scheduled_attempts(
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    producer_outcomes: tuple[ProducerOutcome, ...],
    carryover_outcomes: tuple[ProducerOutcome, ...],
    resolutions: tuple[tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...],
    deferred_innovation: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...],
    deferred_search: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...],
    search_acquisition: ExperimentAcquisitionResultV1,
    search_bindings: tuple[SearchCandidateBindingV1, ...],
    search_pairs: tuple[tuple[ProducerOutcome, CapabilityResolutionV1], ...],
    open_search_pairs: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1, QualifiedSearchCandidateProtocolV1]
    ],
    innovation: InnovationLaneResult | None,
    budget_snapshot: Mapping[str, Any],
    router: StrongStaticRouterV1,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    runner: ExperimentRunner,
    incumbent_observation: Mapping[str, Any],
    metric_contract_digest: str,
    observation_seed: str,
    next_discriminative_test: str,
    confirmation_seed: str | None,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None,
    candidate_root_by_capability: Mapping[str, str | Path] | None,
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None,
    meta_research_inputs: MetaResearchInputs | None,
    provider_traces: tuple[Mapping[str, Any], ...],
    prepared_round: PreparedResearchRoundV1,
    max_attempts_per_round: int | None,
    recovered_attempts: Sequence[Mapping[str, Any]],
    executed_semantic_seed_pairs: tuple[tuple[str, str], ...],
    pending_task: Mapping[str, Any] | None,
    mechanism_axis_effects: Mapping[str, float],
    portfolio_candidates: tuple[PortfolioCandidateV2, ...],
    candidate_handoffs: tuple[RoundCandidateHandoffV1, ...],
    evidence_port: EvidenceGuardPort | None,
    observation_seed_schedule: Sequence[str] | None,
    evaluator: Mapping[str, Any],
    split: str,
    search_space_adapter: SearchSpaceAdapter,
    confirmation_bindings: Mapping[str, Mapping[str, Any]],
    preferred_candidate_ids: Sequence[str] = (),
    recovered_evidence_pre_trace: Sequence[Mapping[str, Any]] = (),
    result_active_profile: SearchExecutableProfileV1 | None = None,
    verification_only: bool = False,
) -> ResearchRoundResult:
    """Run a frozen candidate pool until one metric seals or the pool stops.

    ``recovered_attempts`` are replayed from immutable physical observations
    before new routing begins.  Their candidate identities consume the same
    round budget and are removed from the remaining pool, so resuming an
    incomplete round cannot invoke the physical runner twice for one attempt.
    """

    output_active_profile = result_active_profile or active_profile
    outcome_by_candidate: dict[str, ProducerOutcome] = {}
    for outcome, _resolution in search_pairs:
        if outcome.source_proposal is not None:
            outcome_by_candidate[outcome.source_proposal.candidate_id] = outcome
    for outcome, _resolution, candidate in open_search_pairs:
        outcome_by_candidate[candidate.candidate_id] = outcome
    binding_by_candidate = {
        binding.proposal.candidate_id: binding for binding in search_bindings
    }
    handoff_by_candidate = {
        item.candidate_id: item for item in candidate_handoffs
    }
    prepared_slate = prepared_round.search_slate
    if prepared_slate is not None:
        if tuple(prepared_slate.bindings) != tuple(search_bindings):
            raise ValueError("prepared search slate candidate identity drift")
        if (
            search_acquisition is None
            or search_acquisition.slate_digest != prepared_slate.digest
        ):
            raise ValueError("prepared search slate acquisition identity drift")

    limit = _attempt_budget_limit(
        budget_snapshot,
        max_attempts_per_round,
        len(search_bindings),
    )
    recovered: tuple[Mapping[str, Any], ...] = tuple(recovered_attempts)
    if len(recovered) > limit:
        raise ValueError("recovered attempt manifest exceeds the frozen round budget")
    seen_recovered_candidates: set[str] = set()
    for index, row in enumerate(recovered):
        if not isinstance(row, Mapping):
            raise ValueError("recovered attempt manifest row is invalid")
        raw_index = row.get("attempt_index")
        if isinstance(raw_index, bool) or raw_index != index:
            raise ValueError("recovered attempt manifest indices are not contiguous")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("recovered attempt lacks candidate identity")
        if candidate_id != "__LEGACY_SINGLE_ATTEMPT__":
            if candidate_id not in binding_by_candidate:
                raise ValueError("recovered attempt candidate is outside the frozen pool")
            if candidate_id in seen_recovered_candidates:
                raise ValueError("recovered attempt candidate identity is duplicated")
            seen_recovered_candidates.add(candidate_id)
        if not isinstance(row.get("candidate_run"), Mapping):
            raise ValueError("recovered attempt lacks its immutable candidate_run")

    attempts: list[RoundAttemptV1] = []
    attempted_candidate_ids: set[str] = set()
    pre_blocked_candidate_ids: set[str] = set()
    effective_blocked_candidate_ids: set[str] = set()
    identity_blocked_candidate_ids: set[str] = set()
    semantic_blocked_candidate_ids: set[str] = set()
    code_identity_history = _attempted_executable_identities(context)
    semantic_identity_history = _attempted_semantic_identities(context)
    effective_identity_history = _attempted_effective_experiment_identities(context)
    round_code_identities: set[str] = set()
    round_semantic_identities: set[str] = set()
    round_effective_identities: set[str] = set()
    evidence_pre_trace: list[Mapping[str, Any]] = [
        canonical_value(dict(item))
        for item in recovered_evidence_pre_trace
        if isinstance(item, Mapping)
    ]
    evidence_post_trace: list[Mapping[str, Any]] = []
    working_context = context
    working_policy = policy
    last_candidate_negative_interpretation: EpisodeInterpretation | None = None
    acquisition = search_acquisition
    incomplete_reason = (
        "ROUND_ATTEMPT_BUDGET_EXHAUSTED"
        if limit == 0
        else "ROUND_ATTEMPT_POOL_EXHAUSTED"
    )
    guard_terminal = False

    def reroute_same_frozen_slate(
        current: ExperimentAcquisitionResultV1,
        excluded: set[str],
    ) -> ExperimentAcquisitionResultV1:
        trace = current.route_trace
        ranked = tuple(
            candidate_id
            for candidate_id in trace.ranked_candidate_ids
            if candidate_id not in excluded
        )
        selected_id = ranked[0] if ranked else None
        next_trace = replace(
            trace,
            ranked_candidate_ids=ranked,
            selected_candidate_id=selected_id,
            selection_score=(trace.selection_score if selected_id is not None else None),
        )
        return replace(
            current,
            route_trace=next_trace,
            selected_binding=(
                binding_by_candidate.get(selected_id)
                if selected_id is not None
                else None
            ),
        )

    def pre_admit(
        *,
        binding: SearchCandidateBindingV1,
        recipe: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        if evidence_port is None:
            return None
        candidate_id = binding.proposal.candidate_id
        current_guard_identity = getattr(evidence_port, "identity_digest", None)
        # A persisted PRE decision is the replayable identity of a frozen
        # slate admission only while the Guard implementation identity is
        # unchanged.  A bridge migration may re-adjudicate the same prepared
        # candidate without rerunning Producer, Provider, implementation, or
        # qualification work.
        for previous in reversed(evidence_pre_trace):
            if (
                isinstance(previous, Mapping)
                and previous.get("candidate_id") == candidate_id
                and str(previous.get("stage", "PRE")).upper() == "PRE"
            ):
                action_contract = previous.get("action_contract")
                previous_guard_identity = (
                    action_contract.get("guard_bridge_identity_digest")
                    if isinstance(action_contract, Mapping)
                    else None
                )
                if (
                    isinstance(current_guard_identity, str)
                    and previous_guard_identity != current_guard_identity
                ):
                    continue
                return previous
        try:
            response = _guard_response(
                evidence_port.pre_run(
                    recipe=recipe,
                    binding=binding,
                    observation_seed=observation_seed,
                ),
                stage="PRE",
                candidate_id=binding.proposal.candidate_id,
            )
        except Exception as error:
            response = canonical_value(
                {
                    "candidate_id": candidate_id,
                    "stage": "PRE",
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        evidence_pre_trace.append(response)
        return response

    def pre_is_blocked(response: Mapping[str, Any] | None) -> bool:
        if response is None:
            return False
        return str(response.get("status", "ERROR")).upper() not in {
            "ALLOW",
            "ADJUDICATED",
        }

    def route_remaining(
        remaining_bindings: tuple[SearchCandidateBindingV1, ...],
    ) -> ExperimentAcquisitionResultV1:
        if not attempts:
            return (
                acquisition
                if (
                    pre_blocked_candidate_ids
                    or effective_blocked_candidate_ids
                    or identity_blocked_candidate_ids
                    or semantic_blocked_candidate_ids
                )
                else search_acquisition
            )
        (
            current_executed_pairs,
            current_pending_task,
            current_axis_effects,
        ) = _search_ranking_inputs(working_context)
        if not verification_only:
            # Same-round rerouting is still the same discovery decision.  Do
            # not let a failed candidate silently discard or substitute the
            # durable task identity selected before the first attempt.
            current_pending_task = pending_task
        slate = freeze_experiment_slate(
            profile=active_profile,
            bindings=remaining_bindings,
            budget_snapshot=budget_snapshot,
        )
        remaining_candidate_ids = {
            binding.proposal.candidate_id for binding in remaining_bindings
        }
        routed_portfolio = (
            tuple(item.portfolio_candidate for item in candidate_handoffs)
            if candidate_handoffs
            else portfolio_candidates
        )
        remaining_portfolio = tuple(
            item
            for item in routed_portfolio
            if item.candidate_id in remaining_candidate_ids
        )
        return route_frozen_experiment_slate(
            profile=active_profile,
            slate=slate,
            router=router,
            policy_projection=working_policy.to_dict(),
            executed_semantic_seed_pairs=tuple(
                (
                    *current_executed_pairs,
                    *(
                        (item.binding.mechanism_semantics_digest, observation_seed)
                        for item in attempts
                    ),
                )
            ),
            current_observation_seed=observation_seed,
            pending_task=current_pending_task,
            mechanism_axis_effects=current_axis_effects,
            portfolio_candidates=(
                remaining_portfolio if routed_portfolio else None
            ),
            preferred_candidate_ids=preferred_candidate_ids,
            attempt_failures=(
                tuple(
                    PortfolioAttemptFailureV2(
                        candidate_id=item.candidate_id,
                        scope=item.failure_scope or "CANDIDATE_LOCAL",
                        reason=(
                            str(item.failure_detail.get("reason_code"))
                            if isinstance(item.failure_detail, Mapping)
                            and item.failure_detail.get("reason_code") is not None
                            else item.engineering_disposition
                        ),
                        compute_pattern=(
                            handoff_by_candidate[item.candidate_id]
                            .portfolio_candidate.compute_pattern
                            if item.candidate_id in handoff_by_candidate
                            else None
                        ),
                    )
                    for item in attempts
                    if not item.metric_bearing
                )
                if routed_portfolio
                else ()
            ),
        )

    def materialize_attempt(
        *,
        attempt_acquisition: ExperimentAcquisitionResultV1,
        binding: SearchCandidateBindingV1,
        recipe: Mapping[str, Any],
        search_space_attestation: Mapping[str, Any],
        candidate_run: Mapping[str, Any],
        manifest_row: Mapping[str, Any] | None = None,
        evidence_pre: Mapping[str, Any] | None = None,
        implementation_fidelity_failure: Mapping[str, Any] | None = None,
    ) -> tuple[
        RoundAttemptV1,
        ProducerOutcome,
        Any,
        Any,
        Any,
        Any,
        dict[str, Any],
        Mapping[str, Any] | None,
    ]:
        selected_outcome = _selected_outcome_for_binding(
            binding,
            search_pairs=search_pairs,
            open_search_pairs=open_search_pairs,
        )
        event, identity, episode, closure, failure_detail = (
            project_common_execution_feedback(
                context=context,
                selected_outcome=selected_outcome,
                binding=binding,
                candidate_run=candidate_run,
                incumbent_observation=incumbent_observation,
                metric_contract_digest=metric_contract_digest,
                observation_seed=observation_seed,
                next_discriminative_test=next_discriminative_test,
                evaluator=evaluator,
            )
        )
        if implementation_fidelity_failure is not None:
            failure_detail = canonical_value(
                dict(implementation_fidelity_failure)
            )
            episode = None
            closure = _implementation_fidelity_diagnostic_closure(
                identity=identity,
                binding=binding,
                candidate_run=candidate_run,
                failure=failure_detail,
            )
            disposition = "IMPLEMENTATION_FIDELITY_REJECTED"
            failure_scope = "CANDIDATE_LOCAL"
        elif episode is not None:
            disposition = "METRIC_BEARING"
            failure_scope = None
        else:
            failure_scope = _attempt_failure_scope(candidate_run, failure_detail)
            disposition = "ENGINEERING_FAILURE"
        guard_post: Mapping[str, Any] | None = None
        if evidence_port is not None:
            stored_guard_post = (
                manifest_row.get("evidence_post_adjudication")
                if manifest_row is not None
                else None
            )
            if isinstance(stored_guard_post, Mapping):
                guard_post = _guard_response(
                    stored_guard_post,
                    stage="POST",
                    candidate_id=binding.proposal.candidate_id,
                )
            else:
                try:
                    guard_post = _guard_response(
                        evidence_port.post_run(
                            recipe=recipe,
                            binding=binding,
                            candidate_run=candidate_run,
                            closure=closure,
                            observation_seed=observation_seed,
                            active_task=_guard_active_task(pending_task),
                            remaining_metric_opportunities=(
                                max(0, len(observation_seed_schedule) - working_context.round_index)
                                if observation_seed_schedule is not None else None
                            ),
                        ),
                        stage="POST",
                        candidate_id=binding.proposal.candidate_id,
                    )
                except Exception as error:
                    guard_post = canonical_value(
                        {
                            "candidate_id": binding.proposal.candidate_id,
                            "stage": "POST",
                            "status": "ERROR",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
            existing_task = guard_post.get("research_task")
            fusion = guard_post.get("fusion")
            if existing_task is None and isinstance(fusion, Mapping):
                existing_task = fusion.get("research_task")
            if existing_task is None and _guard_contract_complete(guard_post):
                try:
                    bound_control_task = _guard_bind_control_task(
                        response=guard_post,
                        context=working_context,
                        selected_outcome=selected_outcome,
                        binding=binding,
                        comparator_identity=identity.comparator_ref,
                        observation_seed=observation_seed,
                        observation_seed_schedule=observation_seed_schedule,
                        search_space_adapter=search_space_adapter,
                    )
                except Exception as error:
                    guard_post = canonical_value(
                        {
                            "candidate_id": binding.proposal.candidate_id,
                            "stage": "POST",
                            "status": "ERROR",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "guard_response_digest": sha256_digest(guard_post),
                        }
                    )
                else:
                    if bound_control_task is not None:
                        reserve_bound_control = getattr(
                            evidence_port, "reserve_bound_control", None
                        )
                        awaiting_binding = (
                            bound_control_task.metadata.get("execution_state")
                            == "AWAITING_CANDIDATE_BINDING"
                        )
                        if callable(reserve_bound_control) and not awaiting_binding:
                            guard_post = _guard_response(
                                reserve_bound_control(
                                    response=guard_post,
                                    control_task=bound_control_task.to_dict(),
                                    observation_seed=observation_seed,
                                ),
                                stage="POST",
                                candidate_id=binding.proposal.candidate_id,
                            )
                            rebound_control_task = _guard_bind_control_task(
                                response=guard_post,
                                context=working_context,
                                selected_outcome=selected_outcome,
                                binding=binding,
                                comparator_identity=identity.comparator_ref,
                                observation_seed=observation_seed,
                                observation_seed_schedule=observation_seed_schedule,
                                search_space_adapter=search_space_adapter,
                            )
                            if rebound_control_task is None:
                                raise ValueError(
                                    "reserved control lost its exact task binding"
                                )
                            bound_control_task = rebound_control_task
                        guard_post = canonical_value(
                            {
                                **dict(guard_post),
                                "research_task": bound_control_task.to_dict(),
                                "research_task_binding": {
                                    "authority": "RESEARCH_INTERPRETER",
                                    "status": (
                                        "AWAITING_CANDIDATE_BINDING"
                                        if awaiting_binding
                                        else "EXACT_CATALOG_BINDING"
                                    ),
                                    "task_digest": bound_control_task.digest,
                                },
                            }
                        )
                    else:
                        projection = _guard_control_projection(guard_post)
                        requested_kind = (
                            projection.get("requested_control_kind")
                            if isinstance(projection, Mapping)
                            else None
                        )
                        allocation = guard_post.get("allocation_decision")
                        if (
                            isinstance(allocation, Mapping)
                            and allocation.get("action") != "CONTROL"
                        ):
                            requested_kind = None
                        if requested_kind in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
                            # A valid physical observation must not become an
                            # infrastructure failure merely because no
                            # distinct, declared catalog control exists.
                            # Record the bounded attribution result and keep
                            # promotion held without synthesizing a task.
                            guard_post = canonical_value(
                                {
                                    **dict(guard_post),
                                    "research_task_binding": {
                                        "authority": "RESEARCH_INTERPRETER",
                                        "status": (
                                            "UNAVAILABLE_NO_DISTINCT_DECLARED_"
                                            "CATALOG_CONTROL"
                                        ),
                                        "requested_control_kind": requested_kind,
                                    },
                                }
                            )
            evidence_post_trace.append(guard_post)
            post_status = str(guard_post.get("status", "ERROR")).upper()
            guard_usable = (
                _guard_contract_complete(guard_post)
                and post_status in {"ADJUDICATED", "ALLOW", ""}
                and _guard_current_attempt_class(guard_post)
                not in {
                    "ENGINEERING_FAILURE",
                    "GUARD_ENGINEERING_FAILURE",
                    "PROTOCOL_DRIFT",
                    "INVALID",
                    "INVALID_PROTOCOL",
                    "INVALID_METRIC",
                }
                and not _guard_protocol_drift(guard_post)
            )
            if episode is None and not _guard_contract_complete(guard_post):
                disposition = "GUARD_ENGINEERING_FAILURE"
                failure_scope = "SHARED_INFRASTRUCTURE"
            elif episode is None and post_status not in {"ADJUDICATED", "ALLOW", ""}:
                disposition = "PROTOCOL_DRIFT" if _guard_protocol_drift(guard_post) else "GUARD_ENGINEERING_FAILURE"
                failure_scope = "SHARED_INFRASTRUCTURE"
            elif episode is None and _guard_current_attempt_class(guard_post) in {
                "ENGINEERING_FAILURE",
                "GUARD_ENGINEERING_FAILURE",
                "PROTOCOL_DRIFT",
                "INVALID",
                "INVALID_PROTOCOL",
                "INVALID_METRIC",
            }:
                current_attempt = _guard_current_attempt_class(guard_post)
                disposition = (
                    "PROTOCOL_DRIFT"
                    if current_attempt in {"PROTOCOL_DRIFT", "INVALID_PROTOCOL"}
                    else "GUARD_ENGINEERING_FAILURE"
                )
                failure_scope = (
                    "SHARED_INFRASTRUCTURE"
                    if current_attempt
                    in {"PROTOCOL_DRIFT", "INVALID", "INVALID_PROTOCOL"}
                    else _attempt_failure_scope(candidate_run, failure_detail)
                )
            elif episode is None and _guard_protocol_drift(guard_post):
                disposition = "PROTOCOL_DRIFT"
                failure_scope = "SHARED_INFRASTRUCTURE"
            elif guard_usable:
                directive = _guard_validation_directive(guard_post)
                action = str(directive.get("action", "NONE"))
                fusion = guard_post.get("fusion")
                typed_task = guard_post.get("research_task")
                if typed_task is None and isinstance(fusion, Mapping):
                    typed_task = fusion.get("research_task")
                task_binding = guard_post.get("research_task_binding")
                unavailable_control = (
                    isinstance(task_binding, Mapping)
                    and task_binding.get("authority") == "RESEARCH_INTERPRETER"
                    and task_binding.get("status")
                    == "UNAVAILABLE_NO_DISTINCT_DECLARED_CATALOG_CONTROL"
                )
                if episode is None and action in {
                    "MATCHED_CONTROL_OR_ABLATION",
                    "MATCHED_CONTROL_OR_MECHANISM_OFF",
                    "PROTOCOL_BRANCH",
                } and typed_task is None and not unavailable_control:
                    disposition = "GUARD_ENGINEERING_FAILURE"
                    failure_scope = "SHARED_INFRASTRUCTURE"
                elif episode is None and action in {
                    "NEXT_UNSEEN_SEED_SAME_BUDGET",
                    "ENQUEUE_NEW_SEED",
                } and not _guard_seed_advice_is_valid(
                    response=guard_post,
                    context=working_context,
                    binding=binding,
                    current_seed=observation_seed,
                    observation_seed_schedule=observation_seed_schedule,
                ):
                    disposition = "GUARD_ENGINEERING_FAILURE"
                    failure_scope = "SHARED_INFRASTRUCTURE"
        # Once native Research execution produced a valid metric-bearing
        # episode, an Evidence Guard interface or adjudication error remains
        # auxiliary diagnostic evidence.  It must never erase that result.
        observation_ref = candidate_run.get("physical_observation_ref")
        observation_digest = candidate_run.get("physical_observation_digest")
        if manifest_row is not None:
            observation_ref = observation_ref or manifest_row.get("observation_ref")
            observation_digest = observation_digest or manifest_row.get(
                "observation_digest"
            )
            stored_recipe_digest = manifest_row.get("execution_recipe_digest")
            if (
                stored_recipe_digest is not None
                and stored_recipe_digest != sha256_digest(recipe)
            ):
                raise ValueError("recovered attempt recipe identity drift")
            stored_binding_digest = manifest_row.get("binding_digest")
            if (
                stored_binding_digest is not None
                and stored_binding_digest != binding.digest
            ):
                raise ValueError("recovered attempt binding identity drift")
            stored_disposition = manifest_row.get("engineering_disposition")
            if stored_disposition is not None and stored_disposition != disposition:
                raise ValueError("recovered attempt disposition drift")
            stored_scope = manifest_row.get("failure_scope")
            if stored_scope is not None and stored_scope != failure_scope:
                raise ValueError("recovered attempt failure scope drift")
        attempt = RoundAttemptV1(
            attempt_index=len(attempts),
            acquisition=attempt_acquisition,
            binding=binding,
            execution_recipe=recipe,
            candidate_run=candidate_run,
            engineering_disposition=disposition,
            search_space_attestation=search_space_attestation,
            failure_scope=failure_scope,
            failure_detail=failure_detail,
            observation_ref=(str(observation_ref) if observation_ref is not None else None),
            observation_digest=(
                str(observation_digest) if observation_digest is not None else None
            ),
            resource_prediction=(
                candidate_run.get("resource_prediction")
                if isinstance(candidate_run.get("resource_prediction"), Mapping)
                else None
            ),
            assigned_deadline_seconds=(
                float(candidate_run["assigned_deadline_seconds"])
                if isinstance(candidate_run.get("assigned_deadline_seconds"), (int, float))
                and not isinstance(candidate_run.get("assigned_deadline_seconds"), bool)
                else None
            ),
            live_health_decisions=tuple(
                item
                for item in candidate_run.get("live_health_decisions", ())
                if isinstance(item, Mapping)
            ),
            evidence_pre_adjudication=evidence_pre,
            evidence_post_adjudication=guard_post,
        )
        route_metadata = _route_metadata_for_attempt(
            acquisition=attempt_acquisition,
            selected_outcome=selected_outcome,
            binding=binding,
            candidate_handoff=handoff_by_candidate.get(binding.proposal.candidate_id),
            next_discriminative_test=next_discriminative_test,
            observation_seed=observation_seed,
            failure_scope=failure_scope,
            attempt_index=attempt.attempt_index,
            pending_task=pending_task,
            research_task_record=(
                dict(policy.acquisition_parameters).get("research_task_record")
                if verification_only and isinstance(
                    dict(policy.acquisition_parameters).get(
                        "research_task_record"
                    ),
                    Mapping,
                )
                else None
            ),
            confirmation_seed=confirmation_seed,
            search_space_attestation=search_space_attestation,
        )
        if implementation_fidelity_failure is not None:
            route_metadata.update(
                {
                    "task_operation": "MOVE_ON",
                    "close_task_reason": "IMPLEMENTATION_FIDELITY_REJECTED",
                    "implementation_fidelity_failure": failure_detail,
                }
            )
        route_metadata["followup_confirmation_availability"] = (
            _declare_followup_availability(
                search_space_adapter,
                selected_outcome=selected_outcome,
                context=working_context,
                mechanism_program=binding.proposal.mechanism_program,
            )
        )
        if route_metadata.get("comparator_identity") is None:
            route_metadata["comparator_identity"] = identity.comparator_ref
        elif route_metadata["comparator_identity"] != identity.comparator_ref:
            # An adapter-confirmed control retains the Guard's comparator for
            # task closure.  Record the round's incumbent separately instead
            # of rejecting a completed, correctly bound control execution.
            route_metadata["round_comparison_identity"] = identity.comparator_ref
        if guard_post is not None:
            route_metadata.update(
                {
                    "guard_evidence_class": guard_post.get("outcome_class"),
                    "guard_claim_state": _guard_claim_state(guard_post),
                    "guard_confidence_weight": _guard_confidence_weight(guard_post),
                    "guard_promotion_allowed": _guard_promotion_allowed(guard_post),
                }
            )
        return (
            attempt,
            selected_outcome,
            event,
            identity,
            episode,
            closure,
            route_metadata,
            guard_post,
        )

    def record_diagnostic(
        *,
        attempt: RoundAttemptV1,
        selected_outcome: ProducerOutcome,
        event: Any,
        identity: Any,
        closure: Any,
        route_metadata: Mapping[str, Any],
    ) -> None:
        nonlocal working_context, working_policy
        nonlocal last_candidate_negative_interpretation
        if verification_only:
            diagnostic = interpret_verification_diagnostic(
                closure=closure,
                comparison_identity=identity,
                context=working_context,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=working_policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
                frozen_context=context,
            )
        else:
            diagnostic = interpret_scientific_diagnostic(
                closure=closure,
                comparison_identity=identity,
                context=working_context,
                producer_outcomes=producer_outcomes,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=working_policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
                frozen_context=context,
                advance_round=False,
            )
            if attempt.failure_scope in {
                "CANDIDATE_LOCAL",
                "LINEAGE_COMPUTE_PATTERN",
            }:
                required_probability = route_metadata.get(
                    "required_selected_runnable_probability"
                )
                if (
                    not isinstance(required_probability, (int, float))
                    or isinstance(required_probability, bool)
                ):
                    required_probability = 0.5
                outcomes = {
                    outcome.producer_role: outcome
                    for outcome in producer_outcomes
                }
                aggregate = _meta_aggregate(
                    outcomes,
                    selected_outcome.producer_role,
                    selected_outcome,
                    diagnostic.search_utility_event,
                    float(required_probability),
                    diagnostic.policy_successor,
                    scientific_episode=False,
                    context=diagnostic.successor_context,
                    route=route_metadata,
                    episode=None,
                    task_record=None,
                )
                acquisition_parameters = dict(
                    _next_acquisition_parameters(
                        diagnostic.policy_successor,
                        aggregate,
                    )
                )
                effective_identity = attempt.search_space_attestation
                if isinstance(effective_identity, Mapping):
                    for target, source in (
                        ("attempted_family_digests", "effective_family_digest"),
                        (
                            "attempted_experiment_digests",
                            "effective_experiment_digest",
                        ),
                    ):
                        values = list(acquisition_parameters.get(target, ()))
                        value = effective_identity.get(source)
                        if isinstance(value, str) and value and value not in values:
                            values.append(value)
                        acquisition_parameters[target] = tuple(values[-800:])
                successor_policy = replace(
                    diagnostic.policy_successor,
                    acquisition_parameters=tuple(acquisition_parameters.items()),
                )
                successor_context = replace(
                    diagnostic.successor_context,
                    policy=successor_policy.to_dict(),
                )
                diagnostic = replace(
                    diagnostic,
                    policy_successor=successor_policy,
                    successor_context=successor_context,
                    behavior_after=_behavior(
                        successor_context,
                        successor_policy,
                    ),
                )
                last_candidate_negative_interpretation = diagnostic
        guard_post = attempt.evidence_post_adjudication
        if isinstance(guard_post, Mapping):
            diagnostic = _guard_context_update(
                interpretation=diagnostic,
                before=working_context,
                response=guard_post,
                binding=attempt.binding,
                task=None,
                evidence_domain="GUARD_ENGINEERING",
                policy_before=working_policy,
            )
        working_context = diagnostic.successor_context
        working_policy = diagnostic.policy_successor
        diagnostic_feedback = diagnostic.feedback_projection.to_dict()
        if (
            attempt.engineering_disposition
            == "IMPLEMENTATION_FIDELITY_REJECTED"
            and isinstance(attempt.failure_detail, Mapping)
        ):
            diagnostic_feedback = canonical_value(
                {
                    **diagnostic_feedback,
                    "implementation_fidelity_failure": attempt.failure_detail,
                }
            )
        attempts[-1] = replace(
            attempt,
            diagnostic_feedback=diagnostic_feedback,
            diagnostic_successor_context=diagnostic.successor_context.to_dict(),
            diagnostic_policy_successor=diagnostic.policy_successor.to_dict(),
            diagnostic_search_memory_snapshot=(
                diagnostic.search_memory_snapshot.to_dict()
            ),
        )

    def metric_result(
        *,
        attempt: RoundAttemptV1,
        selected_outcome: ProducerOutcome,
        event: Any,
        identity: Any,
        episode: Any,
        closure: Any,
        route_metadata: Mapping[str, Any],
    ) -> ResearchRoundResult:
        if verification_only:
            interpretation = interpret_verification_episode(
                episode=episode,
                comparison_identity=identity,
                closure=closure,
                context=working_context,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=working_policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
                frozen_context=context,
            )
        else:
            interpretation = interpret_typed_research_episode(
                episode=episode,
                comparison_identity=identity,
                closure=closure,
                context=working_context,
                producer_outcomes=producer_outcomes,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=working_policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
                frozen_context=context,
            )
        guard_post = attempt.evidence_post_adjudication
        if isinstance(guard_post, Mapping):
            pre_guard_task_record = dict(
                interpretation.policy_successor.acquisition_parameters
            ).get("research_task_record")
            try:
                guard_task = _guard_task_from_response(
                    response=guard_post,
                    context=working_context,
                    binding=attempt.binding,
                    recipe=attempt.execution_recipe,
                    comparator_identity=identity.comparator_ref,
                    confidence_weight=_guard_confidence_weight(guard_post),
                    observation_seed=observation_seed,
                    observation_seed_schedule=observation_seed_schedule,
                )
            except Exception as error:
                # Auxiliary Evidence Guard task materialization cannot erase a
                # valid Research metric.  Persist the diagnostic and continue
                # the native interpretation without scheduling that action.
                guard_post = canonical_value(
                    {
                        **dict(guard_post),
                        "auxiliary_task_error": {
                            "error_type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                attempt = replace(
                    attempt,
                    evidence_post_adjudication=guard_post,
                )
                attempts[attempt.attempt_index] = attempt
                guard_task = None
            interpretation = _guard_context_update(
                interpretation=interpretation,
                before=working_context,
                response=guard_post,
                binding=attempt.binding,
                task=guard_task,
                evidence_domain="SCIENTIFIC_EPISODE",
                policy_before=working_policy,
            )
            task_record = (
                ResearchTaskRecordV2.from_dict(pre_guard_task_record)
                if isinstance(pre_guard_task_record, Mapping)
                else None
            )
            required_probability = route_metadata.get(
                "required_selected_runnable_probability"
            )
            if (
                not verification_only
                and task_record is not None
                and isinstance(required_probability, (int, float))
                and not isinstance(required_probability, bool)
                and guard_post.get("research_update_mode")
                != "PRESERVE_NATIVE_RESEARCH"
            ):
                interpretation = _reproject_guarded_acquisition(
                    interpretation=interpretation,
                    producer_outcomes=producer_outcomes,
                    selected_outcome=selected_outcome,
                    route_metadata=route_metadata,
                    task_record=task_record,
                    required_probability=float(required_probability),
                )
        meta_research = (
            _run_meta_research(
                context=working_context,
                interpretation=interpretation,
                inputs=meta_research_inputs,
            )
            if (
                meta_research_inputs is not None
                and not verification_only
            )
            else None
        )
        return ResearchRoundResult(
            context=context,
            active_profile=output_active_profile,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation_outcomes=deferred_innovation,
            deferred_search_outcomes=deferred_search,
            search_acquisition=attempt.acquisition,
            innovation=innovation,
            selected_outcome=selected_outcome,
            execution_recipe=attempt.execution_recipe,
            candidate_run=attempt.candidate_run,
            interpretation=interpretation,
            provider_traces=provider_traces,
            meta_research=meta_research,
            attempts=tuple(attempts),
            metric_bearing_attempt_index=attempt.attempt_index,
            attempt_scheduler_enabled=True,
            prepared=prepared_round,
            evidence_pre_trace=tuple(evidence_pre_trace),
            evidence_post_trace=tuple(evidence_post_trace),
        )

    recovered_terminal_stop = False
    for manifest_row in recovered:
        remaining_bindings = tuple(
            binding
            for binding in search_bindings
            if binding.proposal.candidate_id not in attempted_candidate_ids
            and binding.proposal.candidate_id not in pre_blocked_candidate_ids
            and binding.proposal.candidate_id not in effective_blocked_candidate_ids
            and binding.proposal.candidate_id not in identity_blocked_candidate_ids
            and binding.proposal.candidate_id not in semantic_blocked_candidate_ids
        )
        if not remaining_bindings:
            raise ValueError("recovered attempt has no remaining frozen binding")
        attempt_acquisition = route_remaining(remaining_bindings)
        recovered_candidate_id = str(manifest_row["candidate_id"])
        if recovered_candidate_id == "__LEGACY_SINGLE_ATTEMPT__":
            if attempts or attempt_acquisition.selected_binding is None:
                raise ValueError("legacy physical observation is not the initial attempt")
            binding = attempt_acquisition.selected_binding
        else:
            binding = binding_by_candidate[recovered_candidate_id]
            selected_binding = attempt_acquisition.selected_binding
            if (
                selected_binding is None
                or selected_binding.proposal.candidate_id != recovered_candidate_id
            ):
                raise ValueError("recovered attempt route identity drift")
        recipe, effective_identity, search_space_attestation = _execution_recipe_for_handoff(
            binding,
            profile=active_profile,
            qualified_execution_by_capability=qualified_execution_by_capability,
            candidate_root_by_capability=candidate_root_by_capability,
            resource_profile_by_capability=resource_profile_by_capability,
            candidate_handoffs=candidate_handoffs,
            evaluator=evaluator,
            split=split,
            search_space_adapter=search_space_adapter,
            confirmation_binding=confirmation_bindings.get(
                binding.proposal.candidate_id
            ),
        )
        semantic_identity = binding.mechanism_semantics_digest
        effective_digest = str(
            effective_identity["effective_experiment_digest"]
        )
        explicit_confirmation = _explicit_new_seed_confirmation(
            binding=binding,
            observation_seed=observation_seed,
            pending_task=pending_task,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        )
        if (
            (
                effective_digest in effective_identity_history
                and not explicit_confirmation
            )
            or effective_digest in round_effective_identities
        ):
            raise ValueError(
                "recovered attempt duplicates effective execution semantics"
            )
        if (
            (
                semantic_identity in semantic_identity_history
                and not explicit_confirmation
            )
            or semantic_identity in round_semantic_identities
        ):
            raise ValueError(
                "recovered attempt duplicates a semantic identity within exploration"
            )
        round_semantic_identities.add(semantic_identity)
        round_effective_identities.add(effective_digest)
        executable_identity = _metric_executable_identity_digest(recipe)
        if executable_identity in round_code_identities:
            raise ValueError(
                "recovered attempt duplicates an executable identity within the round"
            )
        repeated_code_field = (
            None
            if explicit_confirmation
            else _repeated_executable_identity(
                recipe=recipe,
                history=code_identity_history,
            )
        )
        if repeated_code_field is not None:
            raise ValueError(
                "recovered attempt repeats an attempted executable identity: "
                + repeated_code_field
            )
        round_code_identities.add(executable_identity)
        candidate_run = canonical_value(dict(manifest_row["candidate_run"]))
        _candidate_run_binding(
            candidate_run,
            recipe=recipe,
            observation_seed=observation_seed,
        )
        implementation_fidelity_failure = (
            _post_run_implementation_fidelity_failure(
                binding=binding,
                recipe=recipe,
                candidate_run=candidate_run,
                candidate_root_by_capability=dict(
                    candidate_root_by_capability or {}
                ),
            )
        )
        recovered_pre = manifest_row.get("evidence_pre_adjudication")
        if evidence_port is not None:
            if not isinstance(recovered_pre, Mapping):
                # A process can stop after its immutable physical observation
                # is durable but before the final attempt manifest copies the
                # already-admitted PRE decision. Re-adjudicate the exact
                # frozen binding/recipe here: durable Guard ports key PRE by
                # the full request digest, so this reuses the prior event and
                # cannot replay Provider, qualification, or physical work.
                recovered_pre = pre_admit(
                    binding=binding,
                    recipe=recipe,
                )
            if not isinstance(recovered_pre, Mapping):
                raise ValueError(
                    "recovered guarded attempt could not restore PRE adjudication"
                )
            recovered_pre = _guard_response(
                recovered_pre,
                stage="PRE",
                candidate_id=binding.proposal.candidate_id,
            )
            if str(recovered_pre.get("status", "")).upper() != "ALLOW":
                raise ValueError(
                    "recovered guarded attempt PRE adjudication is not ALLOW"
                )
            if not any(
                item.get("candidate_id") == binding.proposal.candidate_id
                for item in evidence_pre_trace
                if isinstance(item, Mapping)
            ):
                evidence_pre_trace.append(recovered_pre)
        (
            attempt,
            selected_outcome,
            event,
            identity,
            episode,
            closure,
            route_metadata,
            _guard_post,
        ) = materialize_attempt(
            attempt_acquisition=attempt_acquisition,
            binding=binding,
            recipe=recipe,
            search_space_attestation=search_space_attestation,
            candidate_run=candidate_run,
            manifest_row=manifest_row,
            evidence_pre=(
                recovered_pre if isinstance(recovered_pre, Mapping) else None
            ),
            implementation_fidelity_failure=implementation_fidelity_failure,
        )
        attempts.append(attempt)
        attempted_candidate_ids.add(binding.proposal.candidate_id)
        if attempt.engineering_disposition == "PROTOCOL_DRIFT":
            incomplete_reason = "ROUND_GUARD_PROTOCOL_DRIFT"
            guard_terminal = True
            break
        if episode is not None and attempt.metric_bearing:
            return metric_result(
                attempt=attempt,
                selected_outcome=selected_outcome,
                event=event,
                identity=identity,
                episode=episode,
                closure=closure,
                route_metadata=route_metadata,
            )
        record_diagnostic(
            attempt=attempt,
            selected_outcome=selected_outcome,
            event=event,
            identity=identity,
            closure=closure,
            route_metadata=route_metadata,
        )
        if _attempt_consumed_slot_training_budget(attempt):
            incomplete_reason = "ROUND_SLOT_TRAINING_CONSUMED_NO_METRIC"
            break
        # Recovery classifies the immutable physical outcome exactly once.
        # A fresh process may then continue an untried binding from the same
        # prepared slate without replaying either Provider or physical work.
        if attempt.failure_scope in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}:
            incomplete_reason = "ROUND_ATTEMPT_RETRYABLE_ENGINEERING_FAILURE"
            continue
        incomplete_reason = (
            "ROUND_ATTEMPT_SHARED_INFRASTRUCTURE_STOP"
            if attempt.failure_scope == "SHARED_INFRASTRUCTURE"
            else "ROUND_ATTEMPT_WORKER_TRANSIENT_STOP"
        )
        recovered_terminal_stop = not any(
            candidate.proposal.candidate_id not in attempted_candidate_ids
            for candidate in search_bindings
        )
        break

    while (
        len(attempts) < limit
        and not guard_terminal
        and not recovered_terminal_stop
    ):
        remaining_bindings = tuple(
            binding
            for binding in search_bindings
            if binding.proposal.candidate_id not in attempted_candidate_ids
            and binding.proposal.candidate_id not in pre_blocked_candidate_ids
            and binding.proposal.candidate_id not in effective_blocked_candidate_ids
            and binding.proposal.candidate_id not in identity_blocked_candidate_ids
            and binding.proposal.candidate_id not in semantic_blocked_candidate_ids
        )
        if not remaining_bindings:
            incomplete_reason = (
                "ROUND_ALL_EFFECTIVE_EXPERIMENTS_DUPLICATE"
                if effective_blocked_candidate_ids and not attempts
                else "ROUND_ALL_SEMANTIC_IDENTITIES_DUPLICATE"
                if semantic_blocked_candidate_ids and not attempts
                else "ROUND_ALL_CODE_IDENTITIES_DUPLICATE"
                if identity_blocked_candidate_ids and not attempts
                else "ROUND_GUARD_ALL_PRE_BLOCKED"
                if pre_blocked_candidate_ids and not attempts
                else "ROUND_ATTEMPT_POOL_EXHAUSTED_AFTER_ENGINEERING_FAILURE"
            )
            break
        acquisition = route_remaining(remaining_bindings)
        binding = acquisition.selected_binding
        if binding is None:
            incomplete_reason = "ROUND_ATTEMPT_NO_ELIGIBLE_REMAINING_BINDING"
            break
        recipe, effective_identity, search_space_attestation = _execution_recipe_for_handoff(
            binding,
            profile=active_profile,
            qualified_execution_by_capability=qualified_execution_by_capability,
            candidate_root_by_capability=candidate_root_by_capability,
            resource_profile_by_capability=resource_profile_by_capability,
            candidate_handoffs=candidate_handoffs,
            evaluator=evaluator,
            split=split,
            search_space_adapter=search_space_adapter,
            confirmation_binding=confirmation_bindings.get(
                binding.proposal.candidate_id
            ),
        )
        semantic_identity = binding.mechanism_semantics_digest
        effective_digest = str(
            effective_identity["effective_experiment_digest"]
        )
        explicit_confirmation = _explicit_new_seed_confirmation(
            binding=binding,
            observation_seed=observation_seed,
            pending_task=pending_task,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        )
        if (
            (
                effective_digest in effective_identity_history
                and not explicit_confirmation
            )
            or effective_digest in round_effective_identities
        ):
            effective_blocked_candidate_ids.add(binding.proposal.candidate_id)
            incomplete_reason = "ROUND_EFFECTIVE_EXPERIMENT_DUPLICATE"
            acquisition = reroute_same_frozen_slate(
                acquisition,
                attempted_candidate_ids
                | pre_blocked_candidate_ids
                | effective_blocked_candidate_ids
                | identity_blocked_candidate_ids
                | semantic_blocked_candidate_ids,
            )
            continue
        if (
            (
                semantic_identity in semantic_identity_history
                and not explicit_confirmation
            )
            or semantic_identity in round_semantic_identities
        ):
            semantic_blocked_candidate_ids.add(binding.proposal.candidate_id)
            incomplete_reason = "ROUND_SEMANTIC_IDENTITY_DUPLICATE"
            acquisition = reroute_same_frozen_slate(
                acquisition,
                attempted_candidate_ids
                | pre_blocked_candidate_ids
                | effective_blocked_candidate_ids
                | identity_blocked_candidate_ids
                | semantic_blocked_candidate_ids,
            )
            continue
        round_semantic_identities.add(semantic_identity)
        round_effective_identities.add(effective_digest)
        executable_identity = _metric_executable_identity_digest(recipe)
        if executable_identity in round_code_identities:
            identity_blocked_candidate_ids.add(binding.proposal.candidate_id)
            incomplete_reason = "ROUND_CODE_IDENTITY_DUPLICATE"
            acquisition = reroute_same_frozen_slate(
                acquisition,
                attempted_candidate_ids
                | pre_blocked_candidate_ids
                | effective_blocked_candidate_ids
                | identity_blocked_candidate_ids,
            )
            continue
        repeated_code_field = (
            None
            if explicit_confirmation
            else _repeated_executable_identity(
                recipe=recipe,
                history=code_identity_history,
            )
        )
        if repeated_code_field is not None:
            identity_blocked_candidate_ids.add(binding.proposal.candidate_id)
            incomplete_reason = "ROUND_CODE_IDENTITY_DUPLICATE"
            acquisition = reroute_same_frozen_slate(
                acquisition,
                attempted_candidate_ids
                | pre_blocked_candidate_ids
                | effective_blocked_candidate_ids
                | identity_blocked_candidate_ids
                | semantic_blocked_candidate_ids,
            )
            continue
        round_code_identities.add(executable_identity)
        pre_response = pre_admit(binding=binding, recipe=recipe)
        if pre_is_blocked(pre_response):
            pre_blocked_candidate_ids.add(binding.proposal.candidate_id)
            acquisition = reroute_same_frozen_slate(
                acquisition,
                attempted_candidate_ids
                | pre_blocked_candidate_ids
                | effective_blocked_candidate_ids
                | identity_blocked_candidate_ids
                | semantic_blocked_candidate_ids,
            )
            if acquisition.selected_binding is None:
                incomplete_reason = (
                    "ROUND_GUARD_ALL_PRE_BLOCKED"
                    if not attempts
                    else "ROUND_GUARD_REMAINING_PRE_BLOCKED"
                )
                break
            continue
        candidate_run = canonical_value(dict(runner(recipe, binding)))
        _candidate_run_binding(
            candidate_run,
            recipe=recipe,
            observation_seed=observation_seed,
        )
        implementation_fidelity_failure = (
            _post_run_implementation_fidelity_failure(
                binding=binding,
                recipe=recipe,
                candidate_run=candidate_run,
                candidate_root_by_capability=dict(
                    candidate_root_by_capability or {}
                ),
            )
        )
        (
            attempt,
            selected_outcome,
            event,
            identity,
            episode,
            closure,
            route_metadata,
            _guard_post,
        ) = materialize_attempt(
            attempt_acquisition=acquisition,
            binding=binding,
            recipe=recipe,
            search_space_attestation=search_space_attestation,
            candidate_run=candidate_run,
            evidence_pre=pre_response,
            implementation_fidelity_failure=implementation_fidelity_failure,
        )
        attempts.append(attempt)
        attempted_candidate_ids.add(binding.proposal.candidate_id)
        if attempt.engineering_disposition == "PROTOCOL_DRIFT":
            incomplete_reason = "ROUND_GUARD_PROTOCOL_DRIFT"
            guard_terminal = True
            break
        if episode is not None and attempt.metric_bearing:
            return metric_result(
                attempt=attempt,
                selected_outcome=selected_outcome,
                event=event,
                identity=identity,
                episode=episode,
                closure=closure,
                route_metadata=route_metadata,
            )
        record_diagnostic(
            attempt=attempt,
            selected_outcome=selected_outcome,
            event=event,
            identity=identity,
            closure=closure,
            route_metadata=route_metadata,
        )
        if _attempt_consumed_slot_training_budget(attempt):
            incomplete_reason = "ROUND_SLOT_TRAINING_CONSUMED_NO_METRIC"
            break
        if attempt.failure_scope in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}:
            incomplete_reason = "ROUND_ATTEMPT_RETRYABLE_ENGINEERING_FAILURE"
            continue
        incomplete_reason = (
            "ROUND_ATTEMPT_SHARED_INFRASTRUCTURE_STOP"
            if attempt.failure_scope == "SHARED_INFRASTRUCTURE"
            else "ROUND_ATTEMPT_WORKER_TRANSIENT_STOP"
        )
        break

    if (
        attempts
        and len(attempts) >= limit
        and not attempts[-1].metric_bearing
        and not guard_terminal
        and incomplete_reason == "ROUND_ATTEMPT_RETRYABLE_ENGINEERING_FAILURE"
        and attempts[-1].engineering_disposition
        not in {"PROTOCOL_DRIFT", "GUARD_ENGINEERING_FAILURE"}
    ):
        incomplete_reason = "ROUND_ATTEMPT_BUDGET_EXHAUSTED"
    last_attempt = attempts[-1] if attempts else None
    return ResearchRoundResult(
        context=context,
        active_profile=output_active_profile,
        producer_outcomes=producer_outcomes,
        carryover_outcomes=carryover_outcomes,
        resolutions=resolutions,
        deferred_innovation_outcomes=deferred_innovation,
        deferred_search_outcomes=deferred_search,
        search_acquisition=(last_attempt.acquisition if last_attempt is not None else acquisition),
        innovation=innovation,
        selected_outcome=(
            outcome_by_candidate.get(last_attempt.candidate_id)
            if last_attempt is not None
            else None
        ),
        execution_recipe=(last_attempt.execution_recipe if last_attempt is not None else None),
        candidate_run=(last_attempt.candidate_run if last_attempt is not None else None),
        interpretation=last_candidate_negative_interpretation,
        provider_traces=provider_traces,
        meta_research=None,
        attempts=tuple(attempts),
        metric_bearing_attempt_index=None,
        attempt_scheduler_enabled=True,
        incomplete_reason=incomplete_reason,
        prepared=prepared_round,
        evidence_pre_trace=tuple(evidence_pre_trace),
        evidence_post_trace=tuple(evidence_post_trace),
    )


def _missing_search_round_result(
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    producer_outcomes: tuple[ProducerOutcome, ...],
    carryover_outcomes: tuple[ProducerOutcome, ...],
    resolutions: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...
    ],
    deferred_innovation: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ],
    deferred_search: tuple[
        tuple[ProducerOutcome, CapabilityResolutionV1], ...
    ],
    acquisition: ExperimentAcquisitionResultV1 | None,
    innovation: InnovationLaneResult | None,
    provider_traces: tuple[Mapping[str, Any], ...],
    next_discriminative_test: str,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    meta_research_inputs: MetaResearchInputs | None,
    search_ready_count: int,
    innovation_required_count: int,
    attempt_scheduler: bool,
    attempt_budget: int | None,
    prepared: PreparedResearchRoundV1 | None,
    feedback_proposal_retry_required: bool = False,
    feedback_confirmation_rejections: Sequence[Mapping[str, Any]] = (),
    pending_task: Mapping[str, Any] | None = None,
    existing_observation_rebind: Mapping[str, Any] | None = None,
) -> ResearchRoundResult:
    """Record zero-binding diagnostics without consuming a formal seed."""

    if _innovation_provider_failure(innovation) is not None:
        # No worker result and no candidate negative: retain this exact request.
        return ResearchRoundResult(
            context=context, active_profile=active_profile,
            producer_outcomes=producer_outcomes, carryover_outcomes=carryover_outcomes,
            resolutions=resolutions, deferred_innovation_outcomes=deferred_innovation,
            deferred_search_outcomes=deferred_search, search_acquisition=acquisition,
            innovation=innovation, selected_outcome=None, execution_recipe=None,
            candidate_run=None, interpretation=None, provider_traces=provider_traces,
            meta_research=None, attempts=(), metric_bearing_attempt_index=None,
            attempt_scheduler_enabled=attempt_scheduler,
            incomplete_reason=(
                "ROUND_IMPLEMENTATION_ADMISSION_PAUSED"
                if _is_implementation_admission_failure(_innovation_provider_failure(innovation))
                else "ROUND_IMPLEMENTATION_PROVIDER_UNAVAILABLE"
            ), prepared=prepared,
        )
    recoverable_resource_scope = _innovation_resource_failure_scope(innovation)
    innovation_budget_exhausted = (
        attempt_scheduler
        and attempt_budget is not None
        and innovation is not None
        and not innovation.activation_ready
        and innovation.candidate_attempt_count >= attempt_budget
        and recoverable_resource_scope
        not in _RECOVERABLE_RESOURCE_FAILURE_SCOPES
    )

    resolution_classes: dict[str, int] = {}
    producer_failures: dict[str, str] = {}
    for outcome, resolution in resolutions:
        label = (
            resolution.resolution.value
            if resolution is not None
            else "PRODUCER_FAILURE"
        )
        resolution_classes[label] = resolution_classes.get(label, 0) + 1
        if outcome.failure_code is not None:
            producer_failures[outcome.producer_role] = outcome.failure_code
    missing_interpretation = interpret_missing_search_opportunity(
        context=context,
        producer_outcomes=producer_outcomes,
        diagnostic_detail={
            "reason": "NO_LEGAL_SEARCH_BINDING",
            "resolution_classes": resolution_classes,
            "producer_failures": producer_failures,
            "search_ready_count": search_ready_count,
            "deferred_search_count": len(deferred_search),
            "innovation_required_count": innovation_required_count,
            "innovation_candidate_attempt_count": (
                innovation.candidate_attempt_count
                if innovation is not None
                else 0
            ),
            **(
                {
                    "feedback_confirmation_rejections": tuple(
                        canonical_value(dict(item))
                        for item in feedback_confirmation_rejections
                    )
                }
                if feedback_confirmation_rejections
                else {}
            ),
        },
        next_discriminative_test=next_discriminative_test,
        policy=policy,
        memory_writer=memory_writer,
        route_trace_digest=(
            acquisition.route_trace.digest if acquisition is not None else None
        ),
        route_metadata=(
            {
                "active_task_id": pending_task.get("task_id"),
                "active_task_record": pending_task.get("task_record"),
                **(
                    {
                        "close_task_id": pending_task.get("task_id"),
                        "close_task_reason": pending_task["task_resolution"].get(
                            "reason"
                        ),
                    }
                    if isinstance(pending_task.get("task_resolution"), Mapping)
                    and pending_task["task_resolution"].get("status") == "CLOSED"
                    else {}
                ),
            }
            if isinstance(pending_task, Mapping)
            and isinstance(pending_task.get("task_record"), Mapping)
            else None
        ),
    )
    if (
        isinstance(existing_observation_rebind, Mapping)
        and isinstance(pending_task, Mapping)
    ):
        missing_interpretation = _apply_existing_observation_rebind(
            interpretation=missing_interpretation,
            pending_task=pending_task,
            rebind=existing_observation_rebind,
        )
    meta_research = (
        _run_meta_research(
            context=context,
            interpretation=missing_interpretation,
            inputs=meta_research_inputs,
        )
        if meta_research_inputs is not None and not attempt_scheduler
        else None
    )
    return ResearchRoundResult(
        context=context,
        active_profile=active_profile,
        producer_outcomes=producer_outcomes,
        carryover_outcomes=carryover_outcomes,
        resolutions=resolutions,
        deferred_innovation_outcomes=deferred_innovation,
        deferred_search_outcomes=deferred_search,
        search_acquisition=acquisition,
        innovation=innovation,
        selected_outcome=None,
        execution_recipe=None,
        candidate_run=None,
        interpretation=missing_interpretation,
        provider_traces=provider_traces,
        meta_research=meta_research,
        attempts=(),
        metric_bearing_attempt_index=None,
        attempt_scheduler_enabled=attempt_scheduler,
        incomplete_reason=(
            (
                "ROUND_EXISTING_OBSERVATION_REBOUND"
                if existing_observation_rebind is not None
                else "ROUND_ATTEMPT_SHARED_INFRASTRUCTURE_STOP"
                if recoverable_resource_scope == "SHARED_INFRASTRUCTURE"
                else "ROUND_ATTEMPT_WORKER_TRANSIENT_STOP"
                if recoverable_resource_scope
                in {"WORKER_TRANSIENT", "RECOVERY"}
                else "ROUND_ATTEMPT_BUDGET_EXHAUSTED"
                if innovation_budget_exhausted
                else (
                    "ROUND_FEEDBACK_PROPOSAL_RETRY_REQUIRED"
                    if feedback_proposal_retry_required
                    else "ROUND_ATTEMPT_NO_LEGAL_SEARCH_BINDING"
                )
            )
            if attempt_scheduler
            else None
        ),
        prepared=prepared,
    )


def _exact_construction_parent_bundle(
    option: Mapping[str, Any],
    *,
    baseline_context: Mapping[str, Any],
    candidate_roots: Mapping[str, Any],
    allowed_files: tuple[str, ...],
) -> Mapping[str, Any]:
    """Load the selected proposal's parent, including after checkpoint resume."""
    if option["kind"] == "FROZEN_ROOT":
        bundle = exact_parent_bundle_from_context(
            baseline_context,
            allowed_files=allowed_files,
        )
        if bundle is None or any(bundle[key] != option[key] for key in (
            "candidate_id", "program_digest", "source_tree_digest",
        )):
            raise ValueError("selected construction root source is unavailable")
        return bundle
    root = candidate_roots.get(option["capability_ref"])
    if root is None:
        raise ValueError("selected construction parent source is unavailable")
    return canonical_value({
        "candidate_id": option["candidate_id"],
        "program_digest": option["program_digest"],
        "capability_ref": option["capability_ref"],
        "source_tree_digest": option["source_tree_digest"],
        "execution_contract": option["execution_contract"],
        "instruction": "CLONE_EXACT_PARENT_AND_LOCAL_PATCH",
        "files": _verified_candidate_source_files(
            Path(root), option["source_tree_digest"], allowed_files,
        ),
    })


def _construction_parent_options(
    context: ResearchContext,
    candidates: Sequence[QualifiedSearchCandidateProtocolV1],
    candidate_roots: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Expose existing parent identities; source files are loaded only after choice."""
    baseline = context.knowledge_base.get("baseline_context")
    if not is_single_parent_context(baseline):
        return ()
    root = bound_parent_from_context(baseline)
    options: list[Mapping[str, Any]] = []
    if root is not None and isinstance(root.get("source_bundle"), Mapping):
        bundle = root["source_bundle"]
        anchor = baseline["parent_anchor"]
        options.append({
            **dict(root["binding"]),
            "kind": "FROZEN_ROOT",
            "mechanism_program": root["mechanism_program"],
            "execution_contract": root["execution_contract"],
            "capability_ref": bundle["capability_ref"],
            "source_tree_digest": bundle["source_tree_digest"],
            "observed_metric": anchor["paired_metric"],
            "hypothesis": anchor["name"],
        })
    observations = {
        row["candidate_id"]: row
        for row in context.frontier.get("effect_trajectory", ())
        if isinstance(row, Mapping) and isinstance(row.get("candidate_value"), (int, float))
    }
    for candidate in candidates:
        if candidate.capability_ref not in candidate_roots:
            continue
        option = {
            "candidate_id": getattr(candidate, "compiler_candidate_id", candidate.candidate_id),
            "program_digest": candidate.mechanism_program_digest,
            "kind": "EXISTING_CANDIDATE",
            "mechanism_program": candidate.mechanism_program,
            "execution_contract": candidate.execution_contract,
            "capability_ref": candidate.capability_ref,
            "source_tree_digest": candidate.source_tree_digest,
            "hypothesis": candidate.spec.hypothesis,
            "mechanism_change": candidate.spec.mechanism_change,
        }
        observation = observations.get(candidate.candidate_id)
        if observation is not None:
            option["kind"] = "MEASURED_CANDIDATE"
            option["observed_metric"] = {
                "value": observation["candidate_value"],
                "round_index": observation["round_index"],
            }
        options.append(option)
    return tuple(options)


def _lineage_parent_candidate(
    context: ResearchContext,
    carryover_open_candidates: Sequence[QualifiedSearchCandidateProtocolV1],
    *,
    search_space_adapter: SearchSpaceAdapter,
) -> QualifiedSearchCandidateProtocolV1 | None:
    """Resolve the executable candidate that owns the improved frontier."""

    adapter_validator = getattr(
        search_space_adapter,
        "validate_lineage_candidate",
        None,
    )
    # Native adapters validate their own candidate representation below.
    # Without that hook, retain the existing compiler-backed lineage boundary.
    if not callable(adapter_validator):
        carryover_open_candidates = tuple(
            item for item in carryover_open_candidates
            if isinstance(item, OpenSpecSearchCandidateV1)
        )

    def verified_candidate(
        matches: Sequence[QualifiedSearchCandidateProtocolV1],
    ) -> QualifiedSearchCandidateProtocolV1 | None:
        if len(matches) != 1:
            return None
        candidate = matches[0]
        if callable(adapter_validator):
            try:
                adapter_validator(candidate)
            except (TypeError, ValueError):
                return None
            return candidate
        program = canonical_value(dict(candidate.mechanism_program))
        try:
            report = compile_program(deep_thaw(program))
        except Exception:
            return None
        if (
            report.candidate_id != candidate.compiler_candidate_id
            or report.mechanism_program_digest
            != candidate.mechanism_program_digest
        ):
            return None
        return candidate

    binding = context.frontier.get("lineage_parent_binding")
    if isinstance(binding, Mapping):
        candidate_id = binding.get("candidate_id")
        program_digest = binding.get("program_digest")
        if isinstance(candidate_id, str) and isinstance(program_digest, str):
            candidate = verified_candidate(
                tuple(
                    item
                    for item in carryover_open_candidates
                    if getattr(
                        item,
                        "compiler_candidate_id",
                        item.candidate_id,
                    ) == candidate_id
                    and item.mechanism_program_digest == program_digest
                )
            )
            if candidate is not None:
                return candidate

    # Legacy checkpoints may have the effect record but not the explicit
    # lineage binding. Only a real frontier improvement is a construction
    # parent; a merely executable regression remains negative search evidence.
    trajectory = context.frontier.get("effect_trajectory")
    if isinstance(trajectory, (tuple, list)):
        for observation in reversed(trajectory):
            if not isinstance(observation, Mapping):
                continue
            if observation.get("frontier_updated") is not True:
                continue
            candidate_id = observation.get("candidate_id")
            if not isinstance(candidate_id, str):
                continue
            candidate = verified_candidate(
                tuple(
                    item
                    for item in carryover_open_candidates
                    if getattr(item, "candidate_id", None) == candidate_id
                )
            )
            if candidate is not None:
                return candidate

    return None


def _lineage_parent_mechanism_program(
    context: ResearchContext,
    carryover_open_candidates: Sequence[QualifiedSearchCandidateProtocolV1],
    *,
    search_space_adapter: SearchSpaceAdapter,
) -> Mapping[str, Any] | None:
    """Recover the exact measured lineage parent without changing Context identity."""

    candidate = _lineage_parent_candidate(
        context,
        carryover_open_candidates,
        search_space_adapter=search_space_adapter,
    )
    if candidate is None:
        return None
    return canonical_value(
        {
            "candidate_id": getattr(
                candidate,
                "compiler_candidate_id",
                candidate.candidate_id,
            ),
            "program_digest": candidate.mechanism_program_digest,
            "mechanism_program": dict(candidate.mechanism_program),
            "execution_contract": candidate.execution_contract,
        }
    )


def _verified_candidate_source_files(
    root: Path, source_tree_digest: str, allowed_files: Sequence[str],
) -> list[Mapping[str, str]]:
    manifest = snapshot_candidate_tree(root)
    expected_paths = tuple(sorted(str(path) for path in allowed_files))
    if tuple(row["path"] for row in manifest) != expected_paths:
        raise ValueError("lineage parent source file set differs from policy")
    if sha256_digest({"files": manifest}) != source_tree_digest:
        raise ValueError("lineage parent source tree digest drift")
    rows_by_path = {str(row["path"]): row for row in manifest}
    files: list[Mapping[str, str]] = []
    for relative in expected_paths:
        payload = (root / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != rows_by_path[relative]["sha256"]:
            raise ValueError("lineage parent file SHA-256 drift")
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("lineage parent source is not UTF-8") from error
        files.append(
            canonical_value(
                {
                    "content": content,
                    "path": relative,
                    "sha256": rows_by_path[relative]["sha256"],
                }
            )
        )
    return files


def _exact_lineage_parent_bundle(
    context: ResearchContext,
    carryover_open_candidates: Sequence[QualifiedSearchCandidateProtocolV1],
    *,
    candidate_root_by_capability: Mapping[str, str | Path],
    allowed_files: Sequence[str],
    search_space_adapter: SearchSpaceAdapter,
) -> Mapping[str, Any] | None:
    candidate = _lineage_parent_candidate(context, carryover_open_candidates, search_space_adapter=search_space_adapter)
    if candidate is None:
        return None
    root_value = candidate_root_by_capability.get(candidate.capability_ref)
    if root_value is None:
        raise ValueError("lineage parent candidate root is unavailable")
    files = _verified_candidate_source_files(
        Path(root_value), candidate.source_tree_digest, allowed_files,
    )
    return canonical_value(
        {
            "candidate_id": getattr(
                candidate,
                "compiler_candidate_id",
                candidate.candidate_id,
            ),
            "capability_ref": candidate.capability_ref,
            "execution_contract": candidate.execution_contract,
            "files": files,
            "instruction": "CLONE_EXACT_PARENT_AND_LOCAL_PATCH",
            "program_digest": candidate.mechanism_program_digest,
            "source_tree_digest": candidate.source_tree_digest,
        }
    )


def _latest_completed_execution(
    context: ResearchContext, *, allowed_files: Sequence[str],
) -> Mapping[str, Any] | None:
    memory = context.scientific_memory
    global_memory = memory.get("global_memory", memory.get("global", memory))
    attempts = global_memory.get("round_attempts", ())
    for row in reversed(attempts):
        if not isinstance(row, Mapping) or not isinstance(row.get("completed_execution"), Mapping):
            continue
        facts = dict(row["completed_execution"])
        root = facts.pop("candidate_root_path", None)
        tree_digest = facts.pop("candidate_source_tree_digest", None)
        facts.update({key: row[key] for key in ("round_index", "attempt_index", "candidate_id", "compiler_candidate_id") if key in row})
        facts["proposed_mechanism"] = row.get("core_mechanism_contrast")
        facts["measured_cost"] = row.get("measured_execution_cost")
        # Historical code is optional research context. Never substitute the
        # current contents of a changed source tree for the executed version.
        if root and tree_digest and allowed_files:
            try:
                files = _verified_candidate_source_files(Path(root), tree_digest, allowed_files)
                entrypoint = facts.get("configuration", {}).get("entrypoint", "")
                entrypoint_path = entrypoint.split(":", 1)[0].replace(".", "/") + ".py"
                if facts.get("entrypoint_source_sha256") and not any(
                    item["path"] == entrypoint_path and item["sha256"] == facts["entrypoint_source_sha256"]
                    for item in files
                ):
                    raise ValueError("completed execution entrypoint SHA-256 drift")
            except (OSError, ValueError, MechanicalRecBoleAdapterError):
                facts["implementation_status"] = "SOURCE_UNAVAILABLE_OR_CHANGED"
            else:
                facts["implementation"] = {
                    "source_tree_digest": tree_digest,
                    "files": [{"path": item["path"], "content": item["content"]} for item in files],
                }
        return canonical_value(facts)
    return None


def run_research_round(
    *,
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    producer: ResearchProducer,
    producer_bindings: Mapping[str, Any],
    resolver_environment: Mapping[str, Any],
    carryover_proposals: Sequence[CandidateProposalV4],
    carryover_open_candidates: Sequence[QualifiedSearchCandidateProtocolV1] = (),
    budget_snapshot: Mapping[str, Any],
    router: StrongStaticRouterV1,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    runner: ExperimentRunner,
    incumbent_observation: Mapping[str, Any],
    metric_contract_digest: str,
    observation_seed: str,
    next_discriminative_test: str,
    confirmation_seed: str | None = None,
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None = None,
    innovation_inputs: InnovationRuntimeInputs | None = None,
    meta_research_inputs: MetaResearchInputs | None = None,
    attempt_scheduler: bool = False,
    max_attempts_per_round: int | None = None,
    prebinding_token_ceiling_retry: bool = True,
    recovered_attempts: Sequence[Mapping[str, Any]] = (),
    recovered_evidence_pre_trace: Sequence[Mapping[str, Any]] = (),
    prepared_round: PreparedResearchRoundV1 | None = None,
    on_prepared: Callable[[PreparedResearchRoundV1], None] | None = None,
    portfolio_candidates: Sequence[PortfolioCandidateV2] | None = None,
    research_profile_source: ResearchProfileSourceV1 | None = None,
    candidate_handoff_factory: CandidateHandoffFactory | None = None,
    candidate_root_by_capability: Mapping[str, str | Path] | None = None,
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_port: EvidenceGuardPort | None = None,
    observation_seed_schedule: Sequence[str] | None = None,
    evaluator: Mapping[str, Any] = COMMON_EVALUATOR,
    split: str = COMMON_SPLIT,
    frozen_profile_ref: Mapping[str, Any] | None = None,
    round_role: str = "LEGACY_MIXED",
    search_space_adapter: SearchSpaceAdapter | None = None,
    feedback_proposal_generation_exhausted: bool = False,
) -> ResearchRoundResult:
    """Run one four-Producer, dual-lane, one-experiment Research Line round."""

    if round_role not in {"DISCOVERY", "VERIFICATION", "LEGACY_MIXED"}:
        raise ValueError(
            "round_role must be DISCOVERY, VERIFICATION, or LEGACY_MIXED"
        )
    if not isinstance(feedback_proposal_generation_exhausted, bool):
        raise TypeError(
            "feedback_proposal_generation_exhausted must be bool"
        )
    active_search_space_adapter = (
        _default_search_space_adapter()
        if search_space_adapter is None
        else search_space_adapter
    )
    if not isinstance(active_search_space_adapter, SearchSpaceAdapter):
        raise TypeError("search_space_adapter must implement SearchSpaceAdapter")
    normalized_evaluator = canonical_value(dict(evaluator))
    if not (
        (split == COMMON_SPLIT and normalized_evaluator == COMMON_EVALUATOR)
        or (split == P4_SPARSE_SPECTRAL_SPLIT and normalized_evaluator == P4_SPARSE_SPECTRAL_EVALUATOR)
        or (
            split == DEVELOPMENT_SPLIT
            and normalized_evaluator == DEVELOPMENT_EVALUATOR
        )
    ):
        raise ValueError("Research round split/evaluator contract is unsupported")
    if metric_contract_digest != sha256_digest(normalized_evaluator):
        raise ValueError("Research round metric contract digest drift")
    normalized_frozen_profile_ref = canonical_value(
        dict(
            campaign_scientific_profile_ref()
            if frozen_profile_ref is None
            else frozen_profile_ref
        )
    )
    if (
        active_profile.campaign_id != context.campaign_id
        or active_profile.profile_ref != context.active_profile_ref
        or active_profile.profile_digest != context.active_profile_digest
    ):
        raise ValueError("active Search profile is not the Research Context profile")
    if evidence_port is not None and not attempt_scheduler:
        raise ValueError(
            "Evidence Guard requires the bounded Research attempt scheduler"
        )
    if attempt_scheduler:
        _explicit_attempt_budget(
            budget_snapshot,
            max_attempts_per_round,
            required=True,
        )
    if not isinstance(prebinding_token_ceiling_retry, bool):
        raise ValueError("prebinding_token_ceiling_retry must be a boolean")
    normalized_portfolio = tuple(portfolio_candidates or ())
    single_parent_baseline = (
        context.knowledge_base.get("baseline_context")
        if isinstance(context.knowledge_base, Mapping)
        else None
    )
    single_parent_mode = is_single_parent_context(single_parent_baseline)
    lineage_parent_program = _lineage_parent_mechanism_program(
        context,
        carryover_open_candidates,
        search_space_adapter=active_search_space_adapter,
    )
    effective_qualified_execution = dict(qualified_execution_by_capability or {})
    effective_candidate_roots = dict(candidate_root_by_capability or {})
    effective_resource_profiles = dict(resource_profile_by_capability or {})
    construction_parent_options = _construction_parent_options(
        context, carryover_open_candidates, effective_candidate_roots,
    )
    def exact_parent_bundle_loader(outcome: ProducerOutcome | None = None) -> Mapping[str, Any] | None:
        if innovation_inputs is None:
            return None
        option = outcome.provenance.get("construction_parent_option") if outcome is not None else None
        if isinstance(option, Mapping):
            return _exact_construction_parent_bundle(
                option, baseline_context=single_parent_baseline,
                candidate_roots=effective_candidate_roots,
                allowed_files=tuple(innovation_inputs.policy.allowed_files),
            )
        if lineage_parent_program is not None:
            return _exact_lineage_parent_bundle(
                context,
                carryover_open_candidates,
                candidate_root_by_capability=effective_candidate_roots,
                allowed_files=innovation_inputs.policy.allowed_files,
                search_space_adapter=active_search_space_adapter,
            )
        if single_parent_mode:
            assert isinstance(single_parent_baseline, Mapping)
            return exact_parent_bundle_from_context(
                single_parent_baseline,
                allowed_files=tuple(innovation_inputs.policy.allowed_files),
            )
        return None
    def research_parent_program() -> Mapping[str, Any] | None:
        # Research must be able to inspect what its measured parent executes,
        # not infer that solely from the inherited mechanism declaration.
        if lineage_parent_program is None:
            return None
        bundle = exact_parent_bundle_loader()
        if bundle is None:
            return lineage_parent_program
        return {**lineage_parent_program, "source_bundle": bundle}
    latest_completed_execution = _latest_completed_execution(
        context, allowed_files=innovation_inputs.policy.allowed_files if innovation_inputs else (),
    )
    execution_profile = active_profile
    if any(
        not isinstance(item, PortfolioCandidateV2)
        for item in normalized_portfolio
    ):
        raise ValueError(
            "portfolio_candidates must contain PortfolioCandidateV2 records"
        )
    adjust_portfolio = getattr(
        evidence_port, "adjust_portfolio_information", None
    )
    if callable(adjust_portfolio) and round_role == "DISCOVERY":
        adjusted = tuple(adjust_portfolio(normalized_portfolio))
        if (
            len(adjusted) != len(normalized_portfolio)
            or tuple(item.candidate_id for item in adjusted)
            != tuple(item.candidate_id for item in normalized_portfolio)
        ):
            raise ValueError(
                "Evidence Guard portfolio adjustment changed candidate coverage"
            )
        normalized_portfolio = adjusted
    resume_retry_outcomes: tuple[ProducerOutcome, ...] | None = None
    resume_retry_trace_prefix: tuple[Mapping[str, Any], ...] = ()
    resume_transient_trace_prefix: tuple[Mapping[str, Any], ...] = ()
    resume_resource_outcome_digest: str | None = None
    resume_candidate_local_innovation: InnovationLaneResult | None = None
    resume_innovation_attempts: tuple[Mapping[str, Any], ...] = ()
    resume_innovation_candidate_count = 0
    resume_qualified_innovation: InnovationLaneResult | None = None
    resume_provider_request: Mapping[str, Any] | None = None
    resume_provider_namespace: str | None = None
    resume_provider_remaining_calls: int | None = None
    prebinding_retry_mapping: Mapping[str, Any] | None = None

    if prepared_round is not None:
        if not attempt_scheduler:
            raise ValueError("prepared round resume requires attempt_scheduler=True")
        prepared_timeout_reclassified = False
        if (
            innovation_inputs is not None
            and innovation_inputs.resource_probe_parent is not None
            and prepared_round.innovation is not None
        ):
            reclassified_innovation = (
                _reclassify_innovation_progressing_timeout(
                    prepared_round.innovation,
                    innovation_inputs.resource_probe_parent,
                )
            )
            if reclassified_innovation is not None:
                prepared_round = replace(
                    prepared_round,
                    innovation=reclassified_innovation,
                )
                prepared_timeout_reclassified = True
        prepared_retry_mapping = getattr(prepared_round, "prebinding_retry", None)
        prepared_retry_status = (
            str(prepared_retry_mapping["status"])
            if isinstance(prepared_retry_mapping, Mapping)
            else None
        )
        prepared_is_partial_retry = (
            prepared_retry_mapping is not None
            and not prepared_round.resolutions
            and not prepared_round.search_bindings
        )
        prepared_is_provider_retry = (
            _prepared_has_unfinished_producer_failure(prepared_round)
        )
        prepared_is_resource_retry = _prepared_has_recoverable_resource_failure(
            prepared_round
        )
        prepared_is_implementation_retry = (
            _prepared_has_external_implementation_failure(prepared_round)
        )
        prepared_has_untried_candidate_local_innovation = (
            _prepared_has_untried_candidate_local_innovation(
                prepared_round,
                attempt_budget=max_attempts_per_round,
            )
        )
        if prepared_is_provider_retry and not prepared_round.provider_traces:
            prepared_is_provider_retry = callable(
                getattr(producer, "configure_started_round_provider_replay", None)
            )
        if _prepared_has_terminal_provider_failure(prepared_round) and not prepared_is_provider_retry:
            return ResearchRoundResult(
                context=context,
                active_profile=active_profile,
                producer_outcomes=prepared_round.producer_outcomes,
                carryover_outcomes=prepared_round.carryover_outcomes,
                resolutions=prepared_round.resolutions,
                deferred_innovation_outcomes=(
                    prepared_round.deferred_innovation_outcomes
                ),
                deferred_search_outcomes=prepared_round.deferred_search_outcomes,
                search_acquisition=prepared_round.search_acquisition,
                innovation=prepared_round.innovation,
                selected_outcome=None,
                execution_recipe=None,
                candidate_run=None,
                interpretation=None,
                provider_traces=prepared_round.provider_traces,
                meta_research=None,
                attempts=(),
                metric_bearing_attempt_index=None,
                attempt_scheduler_enabled=True,
                incomplete_reason="ROUND_ATTEMPT_NO_LEGAL_SEARCH_BINDING",
                prepared=prepared_round,
            )
        if on_prepared is not None and not (
            prepared_is_partial_retry
            or prepared_is_provider_retry
            or prepared_is_resource_retry
            or prepared_is_implementation_retry
            or prepared_timeout_reclassified
            or prepared_has_untried_candidate_local_innovation
        ):
            raise ValueError("prepared round resume cannot prepare the round again")
        if (
            prepared_round.context_digest != context.digest
            or prepared_round.profile_ref != active_profile.profile_ref
            or prepared_round.profile_digest != active_profile.profile_digest
        ):
            raise ValueError("prepared round context/profile identity drift")
        if (
            prepared_round.search_slate is not None
            and sha256_digest(prepared_round.search_slate.budget_snapshot)
            != sha256_digest(budget_snapshot)
        ):
            raise ValueError("prepared round budget snapshot drift")
        if confirmation_seed != getattr(prepared_round, "confirmation_seed", None):
            raise ValueError("prepared round confirmation seed identity drift")
        if prepared_timeout_reclassified and on_prepared is not None:
            on_prepared(prepared_round)
        if prepared_retry_mapping is not None:
            if not prebinding_token_ceiling_retry:
                raise ValueError(
                    "prepared round contains a disabled prebinding retry"
                )
            _validate_prebinding_retry_identity(
                prepared_retry_mapping,
                context=context,
                active_profile=active_profile,
                budget_snapshot=budget_snapshot,
            )
        if prepared_is_partial_retry:
            if prepared_retry_status == "EXHAUSTED":
                if not prepared_round.carryover_outcomes:
                    return ResearchRoundResult(
                        context=context,
                        active_profile=active_profile,
                        producer_outcomes=prepared_round.producer_outcomes,
                        carryover_outcomes=prepared_round.carryover_outcomes,
                        resolutions=prepared_round.resolutions,
                        deferred_innovation_outcomes=prepared_round.deferred_innovation_outcomes,
                        deferred_search_outcomes=prepared_round.deferred_search_outcomes,
                        search_acquisition=prepared_round.search_acquisition,
                        innovation=prepared_round.innovation,
                        selected_outcome=None,
                        execution_recipe=None,
                        candidate_run=None,
                        interpretation=None,
                        provider_traces=prepared_round.provider_traces,
                        meta_research=None,
                        attempts=(),
                        metric_bearing_attempt_index=None,
                        attempt_scheduler_enabled=True,
                        incomplete_reason="ROUND_PREBINDING_RETRY_EXHAUSTED",
                        prepared=prepared_round,
                    )
                resume_retry_outcomes = tuple(prepared_round.producer_outcomes)
                resume_retry_trace_prefix = tuple(prepared_round.provider_traces)
                prebinding_retry_mapping = prepared_retry_mapping
                prepared_round = None
            if prepared_retry_status == "PENDING":
                retry_producer = _namespaced_retry_producer(producer)
                if retry_producer is None:
                    retry_outcomes: tuple[ProducerOutcome, ...] = ()
                    retry_traces: tuple[Mapping[str, Any], ...] = ()
                else:
                    retry_trace_starts = _trace_starts(producer)
                    retry_outcomes = produce_research_specs(
                        context,
                        retry_producer,
                        producer_bindings,
                        frozen_profile_ref=normalized_frozen_profile_ref,
                        active_task_directive=_active_task_directive(
                            _discovery_feedback_task(
                                context,
                                observation_seed=prepared_round.observation_seed,
                            )
                            if round_role == "DISCOVERY"
                            else None,
                            observation_seed=prepared_round.observation_seed,
                        ),
                        lineage_parent_mechanism_program=research_parent_program(),
                        construction_parent_options=construction_parent_options,
                        latest_completed_execution=latest_completed_execution,
                        research_window_budget=budget_snapshot.get("research_window"),
                        search_space_adapter=active_search_space_adapter,
                    )
                    retry_traces = _new_provider_traces(retry_trace_starts)
                retry_status = (
                    "SUCCEEDED"
                    if any(item.spec is not None for item in retry_outcomes)
                    else "EXHAUSTED"
                )
                retry_mapping = canonical_value(
                    {
                        **dict(prepared_retry_mapping),
                        "status": retry_status,
                        "usage_charge": int(
                            prepared_retry_mapping["usage_charge"]
                        )
                        + _provider_usage_charge(retry_traces, producer),
                        "retry_trace_digest": sha256_digest(retry_traces),
                        "retry_logical_calls_digest": sha256_digest(
                            tuple(
                                str(item.get("logical_call_id"))
                                for item in retry_traces
                                if item.get("logical_call_id") is not None
                            )
                        ),
                    }
                )
                updated_prepared = replace(
                    prepared_round,
                    producer_outcomes=retry_outcomes,
                    provider_traces=tuple(
                        (*prepared_round.provider_traces, *retry_traces)
                    ),
                    prebinding_retry=retry_mapping,
                )
                if on_prepared is not None:
                    on_prepared(updated_prepared)
                if retry_status == "EXHAUSTED":
                    return ResearchRoundResult(
                        context=context,
                        active_profile=active_profile,
                        producer_outcomes=retry_outcomes,
                        carryover_outcomes=updated_prepared.carryover_outcomes,
                        resolutions=(),
                        deferred_innovation_outcomes=(),
                        deferred_search_outcomes=(),
                        search_acquisition=None,
                        innovation=None,
                        selected_outcome=None,
                        execution_recipe=None,
                        candidate_run=None,
                        interpretation=None,
                        provider_traces=updated_prepared.provider_traces,
                        meta_research=None,
                        attempts=(),
                        metric_bearing_attempt_index=None,
                        attempt_scheduler_enabled=True,
                        incomplete_reason="ROUND_PREBINDING_RETRY_EXHAUSTED",
                        prepared=updated_prepared,
                    )
                resume_retry_outcomes = tuple(retry_outcomes)
                resume_retry_trace_prefix = tuple(updated_prepared.provider_traces)
                prebinding_retry_mapping = retry_mapping
                prepared_round = None
            if prepared_retry_status == "SUCCEEDED":
                resume_retry_outcomes = tuple(prepared_round.producer_outcomes)
                resume_retry_trace_prefix = tuple(prepared_round.provider_traces)
                prebinding_retry_mapping = prepared_retry_mapping
                prepared_round = None
        if prepared_is_provider_retry and prepared_round is not None:
            # Complete only unfinished calls. Completed peer research belongs
            # to this exact context and must not be bought or generated again.
            retry_trace_starts = _trace_starts(producer)
            unfinished_roles = _unfinished_producer_roles(prepared_round)
            retry_outcomes = produce_research_specs(
                context,
                _transport_retry_producer(
                    producer,
                    prepared_round.provider_traces,
                ),
                producer_bindings,
                frozen_profile_ref=normalized_frozen_profile_ref,
                active_task_directive=_active_task_directive(
                    _discovery_feedback_task(
                        context,
                        observation_seed=prepared_round.observation_seed,
                    )
                    if round_role == "DISCOVERY"
                    else None,
                    observation_seed=prepared_round.observation_seed,
                ),
                lineage_parent_mechanism_program=research_parent_program(),
                construction_parent_options=construction_parent_options,
                latest_completed_execution=latest_completed_execution,
                research_window_budget=budget_snapshot.get("research_window"),
                search_space_adapter=active_search_space_adapter,
                completed_outcomes=tuple(
                    outcome for outcome in prepared_round.producer_outcomes
                    if outcome.producer_role not in unfinished_roles
                ),
            )
            retry_traces = _new_provider_traces(retry_trace_starts)
            resume_retry_outcomes = tuple(retry_outcomes)
            resume_transient_trace_prefix = tuple(
                (*prepared_round.provider_traces, *retry_traces)
            )
            prepared_round = None
        if prepared_is_implementation_retry and prepared_round is not None:
            assert prepared_round.innovation is not None
            assert innovation_inputs is not None
            failure = _innovation_provider_failure(prepared_round.innovation)
            assert failure is not None
            # Older pre-HTTP budget stops did not save a gateway request. The
            # first implementation can be rebuilt from this exact saved slate.
            resume_provider_request = failure.get("gateway_request")
            resume_provider_remaining_calls = failure.get("remaining_calls")
            resume_retry_outcomes = tuple(prepared_round.producer_outcomes)
            resume_innovation_attempts = tuple(prepared_round.innovation.attempts)
            resume_innovation_candidate_count = replace(
                prepared_round.innovation,
                attempts=tuple(
                    item for item in resume_innovation_attempts
                    if _innovation_attempt_proposal_digest(item)
                    != prepared_round.innovation.idea_acquisition.selected_spec_digest
                ),
            ).candidate_attempt_count
            resume_resource_outcome_digest = next(
                outcome.digest for outcome, resolution in prepared_round.resolutions
                if resolution is not None
                and resolution.digest == prepared_round.innovation.resolution.digest
                and outcome.spec is not None
                and outcome.spec.digest == prepared_round.innovation.idea_acquisition.selected_spec_digest
            )
            resume_retry_trace_prefix = tuple(prepared_round.provider_traces)
            prebinding_retry_mapping = prepared_round.prebinding_retry
            resume_provider_namespace = (
                _next_transport_retry_namespace(prepared_round.provider_traces)
                if failure["failure_class"] == "PROVIDER_EXTERNAL"
                else None
            )
            prepared_round = None
        if prepared_is_resource_retry and prepared_round is not None:
            # Re-enter only the already selected producer outcome.  A fresh
            # probe namespace avoids replaying the failed physical probe while
            # keeping the same logical candidate and discovery generation.
            assert prepared_round.innovation is not None
            resume_retry_outcomes = tuple(prepared_round.producer_outcomes)
            resume_innovation_attempts = tuple(prepared_round.innovation.attempts)
            resume_qualified_innovation = prepared_round.innovation
            # Count consumed candidates before the selected retry. Infrastructure
            # stops do not consume a candidate; a prior local failure of this
            # same spec must not count it twice either.
            resume_innovation_candidate_count = replace(
                prepared_round.innovation,
                attempts=tuple(
                    item for item in resume_innovation_attempts
                    if _innovation_attempt_proposal_digest(item)
                    != _innovation_attempt_proposal_digest(resume_innovation_attempts[-1])
                ),
            ).candidate_attempt_count
            resume_resource_outcome_digest = next(
                outcome.digest
                for outcome, resolution in prepared_round.resolutions
                if resolution is not None
                and resolution.digest
                == prepared_round.innovation.resolution.digest
                and outcome.spec is not None
                and outcome.spec.digest
                == prepared_round.innovation.idea_acquisition.selected_spec_digest
            )
            resume_retry_trace_prefix = tuple(prepared_round.provider_traces)
            prebinding_retry_mapping = prepared_round.prebinding_retry
            if (
                innovation_inputs is not None
                and innovation_inputs.resource_probe_parent is not None
            ):
                innovation_inputs = replace(
                    innovation_inputs,
                    resource_probe_parent=(
                        innovation_inputs.resource_probe_parent
                        / f"recover-{prepared_round.digest[:16]}"
                    ),
                )
            prepared_round = None
        if (
            prepared_round is not None
            and (
                prepared_timeout_reclassified
                or prepared_has_untried_candidate_local_innovation
            )
        ):
            # The frozen candidate now has sufficient persisted physical evidence
            # to be consumed as candidate-local.  Reuse the original Provider
            # slate and continue its untried Innovation candidates in this slot.
            assert prepared_round.innovation is not None
            resume_candidate_local_innovation = prepared_round.innovation
            resume_innovation_attempts = tuple(prepared_round.innovation.attempts)
            resume_innovation_candidate_count = (
                prepared_round.innovation.candidate_attempt_count
            )
            resume_retry_outcomes = tuple(prepared_round.producer_outcomes)
            resume_retry_trace_prefix = tuple(prepared_round.provider_traces)
            prebinding_retry_mapping = prepared_round.prebinding_retry
            prepared_round = None
        empty_prepared_feedback_rejections: tuple[Mapping[str, Any], ...] = ()
        if (
            prepared_round is not None
            and prepared_round.search_acquisition is None
            and not prepared_round.search_bindings
            and round_role == "DISCOVERY"
        ):
            prepared_pending_task = _discovery_feedback_task(
                context,
                observation_seed=prepared_round.observation_seed,
            )
            prepared_confirmations = tuple(
                (
                    outcome,
                    resolution,
                    _resolve_confirmation(
                        active_search_space_adapter,
                        pending_task=prepared_pending_task,
                        primary_binding=outcome,
                        next_discriminative_test=(
                            prepared_round.next_discriminative_test
                        ),
                    ),
                )
                for outcome, resolution in prepared_round.resolutions
                if prepared_pending_task is not None and outcome.spec is not None
            )
            prepared_exact_innovation = any(
                resolution is not None
                and resolution.resolution
                is CapabilityResolutionResultV1.INNOVATION_REQUIRED
                and confirmation.kind is ConfirmationResolutionKindV1.EXACT_BINDING
                for _outcome, resolution, confirmation in prepared_confirmations
            )
            innovation_attempt_by_spec = {
                str(item["spec_digest"]): item
                for item in (
                    prepared_round.innovation.attempts
                    if prepared_round.innovation is not None
                    else ()
                )
                if isinstance(item, Mapping)
                and isinstance(item.get("spec_digest"), str)
            }
            prepared_rejections: list[Mapping[str, Any]] = []
            for outcome, _resolution, confirmation in prepared_confirmations:
                if confirmation.kind is not ConfirmationResolutionKindV1.NEEDS_PROPOSAL:
                    continue
                attempt = innovation_attempt_by_spec.get(outcome.spec.digest)
                prepared_rejections.append(
                    canonical_value(
                        {
                            "outcome_digest": outcome.digest,
                            "candidate_id": (
                                attempt.get("candidate_id")
                                if isinstance(attempt, Mapping)
                                else outcome.source_proposal.candidate_id
                                if outcome.source_proposal is not None
                                else outcome.spec.spec_id
                            ),
                            "candidate_semantic_digest": (
                                attempt.get("mechanism_semantics_digest")
                                if isinstance(attempt, Mapping)
                                else None
                            ),
                            "effective_experiment_digest": (
                                attempt.get("effective_experiment_digest")
                                if isinstance(attempt, Mapping)
                                else None
                            ),
                            "effective_family_digest": (
                                attempt.get("effective_family_digest")
                                if isinstance(attempt, Mapping)
                                else None
                            ),
                            "primitive_ids": (
                                attempt.get("primitive_ids")
                                if isinstance(attempt, Mapping)
                                else None
                            ),
                            "spec_digest": outcome.spec.digest,
                            "producer_role": outcome.producer_role,
                            "resolution_kind": confirmation.kind.value,
                            "reason": confirmation.reason,
                            "task_id": prepared_pending_task.get("task_id"),
                            "original_directive": (
                                _feedback_confirmation_task_directive(
                                    prepared_pending_task
                                )
                            ),
                        }
                    )
                )
            empty_prepared_feedback_rejections = tuple(prepared_rejections)
            if prepared_exact_innovation:
                # The prepared slate predates adapter confirmation of its
                # direct control.  Reuse its Provider outcomes and rebuild only
                # the existing Innovation/Router path under the retained task.
                resume_retry_outcomes = tuple(prepared_round.producer_outcomes)
                resume_retry_trace_prefix = tuple(prepared_round.provider_traces)
                prebinding_retry_mapping = prepared_round.prebinding_retry
                prepared_round = None
        if prepared_round is not None:
            prepared_innovation = prepared_round.innovation
            if (
                prepared_innovation is not None
                and prepared_innovation.activation_ready
                and len(prepared_round.search_bindings) == 1
                and prepared_innovation.search_candidate is not None
                and prepared_round.search_bindings[0].proposal.candidate_id
                == prepared_innovation.search_candidate.candidate_id
            ):
                if (
                    not prepared_innovation.training_ready
                    or prepared_innovation.registry is None
                    or prepared_innovation.next_profile is None
                    or prepared_innovation.capability is None
                    or prepared_innovation.qualified_execution is None
                    or prepared_innovation.candidate_root is None
                ):
                    raise ValueError(
                        "prepared Innovation reroute lacks admitted execution evidence"
                    )
                execution_profile = activate_next_fresh_search_profile(
                    predecessor=active_profile,
                    next_profile=prepared_innovation.next_profile,
                    registry=prepared_innovation.registry,
                    fresh_campaign_id=prepared_innovation.fresh_campaign_id,
                )
                effective_qualified_execution[
                    prepared_innovation.capability.capability_id
                ] = prepared_innovation.qualified_execution
                effective_candidate_roots[
                    prepared_innovation.capability.capability_id
                ] = prepared_innovation.candidate_root
                if prepared_innovation.resource_profile is not None:
                    effective_resource_profiles[
                        prepared_innovation.capability.capability_id
                    ] = prepared_innovation.resource_profile
            prepared_handoffs = tuple(
                getattr(prepared_round, "candidate_handoffs", ()) or ()
            )
            if research_profile_source is not None and not isinstance(
                research_profile_source,
                ResearchProfileSourceV1,
            ):
                raise ValueError(
                    "research_profile_source must be ResearchProfileSourceV1"
                )
            if research_profile_source is not None and prepared_handoffs:
                for handoff in prepared_handoffs:
                    if (
                        handoff.source_schema_version
                        != research_profile_source.schema_version
                        or handoff.source_ref != research_profile_source.source_ref
                        or handoff.source_digest != research_profile_source.source_digest
                    ):
                        raise ValueError(
                            "prepared round profile source identity drift"
                        )
            if (
                research_profile_source is not None
                and not prepared_handoffs
                and prepared_round.portfolio_candidates
            ):
                raise ValueError(
                    "prepared round lacks the complete profile-source handoff"
                )
            prepared_portfolio = tuple(prepared_round.portfolio_candidates)
            if not prepared_portfolio and prepared_handoffs:
                prepared_portfolio = tuple(
                    item.portfolio_candidate for item in prepared_handoffs
                )
            if normalized_portfolio and normalized_portfolio != prepared_portfolio:
                raise ValueError("prepared round portfolio candidate identity drift")
            if prepared_handoffs:
                candidate_handoffs = _validate_candidate_handoffs(
                    prepared_handoffs,
                    bindings=prepared_round.search_bindings,
                    active_profile=execution_profile,
                    require_complete=False,
                    evaluator=normalized_evaluator,
                    split=split,
                    search_space_adapter=active_search_space_adapter,
                )
                handoff_portfolio = tuple(
                    item.portfolio_candidate for item in candidate_handoffs
                )
                if prepared_portfolio and handoff_portfolio != prepared_portfolio:
                    raise ValueError("prepared round handoff portfolio identity drift")
                normalized_portfolio = handoff_portfolio
            elif prepared_round.portfolio_candidates:
                candidate_handoffs = _static_candidate_handoffs(
                    tuple(prepared_round.portfolio_candidates),
                    bindings=prepared_round.search_bindings,
                    active_profile=execution_profile,
                    qualified_execution_by_capability=effective_qualified_execution,
                    candidate_root_by_capability=effective_candidate_roots,
                    resource_profile_by_capability=effective_resource_profiles,
                    evaluator=normalized_evaluator,
                    split=split,
                    search_space_adapter=active_search_space_adapter,
                )
                normalized_portfolio = tuple(
                    item.portfolio_candidate for item in candidate_handoffs
                )
            else:
                candidate_handoffs = ()
                normalized_portfolio = ()
            if prepared_round.search_acquisition is None or not prepared_round.search_bindings:
                prepared_pending_task = (
                    _discovery_feedback_task(
                        context,
                        observation_seed=prepared_round.observation_seed,
                    )
                    if round_role == "DISCOVERY"
                    else None
                )
                return _missing_search_round_result(
                    context=context,
                    active_profile=active_profile,
                    producer_outcomes=prepared_round.producer_outcomes,
                    carryover_outcomes=prepared_round.carryover_outcomes,
                    resolutions=prepared_round.resolutions,
                    deferred_innovation=prepared_round.deferred_innovation_outcomes,
                    deferred_search=prepared_round.deferred_search_outcomes,
                    acquisition=prepared_round.search_acquisition,
                    innovation=prepared_round.innovation,
                    provider_traces=prepared_round.provider_traces,
                    next_discriminative_test=prepared_round.next_discriminative_test,
                    policy=policy,
                    memory_writer=memory_writer,
                    meta_research_inputs=meta_research_inputs,
                    search_ready_count=(
                        len(prepared_round.search_pairs)
                        + len(prepared_round.open_search_pairs)
                    ),
                    innovation_required_count=sum(
                        1
                        for _outcome, resolution in prepared_round.resolutions
                        if resolution is not None
                        and resolution.resolution
                        is CapabilityResolutionResultV1.INNOVATION_REQUIRED
                    ),
                    attempt_scheduler=True,
                    attempt_budget=(
                        int(max_attempts_per_round)
                        if max_attempts_per_round is not None
                        else None
                    ),
                    prepared=(
                        None
                        if empty_prepared_feedback_rejections
                        else prepared_round
                    ),
                    feedback_proposal_retry_required=bool(
                        empty_prepared_feedback_rejections
                    ),
                    feedback_confirmation_rejections=(
                        empty_prepared_feedback_rejections
                    ),
                    pending_task=prepared_pending_task,
                )
            (
                executed_semantic_seed_pairs,
                pending_task,
                mechanism_axis_effects,
            ) = _search_ranking_inputs(context)
            if round_role == "DISCOVERY":
                pending_task = _discovery_feedback_task(
                    context,
                    observation_seed=prepared_round.observation_seed,
                )
            elif round_role == "VERIFICATION":
                pending_task = _verification_feedback_task(
                    context,
                    observation_seed=prepared_round.observation_seed,
                    executed_semantic_seed_pairs=executed_semantic_seed_pairs,
                )
            elif round_role != "VERIFICATION":
                pending_task = None
            prepared_confirmation_bindings: dict[
                str, Mapping[str, Any]
            ] = {}
            prepared_confirmation_rejections: list[Mapping[str, Any]] = []
            prepared_confirmation_needs_proposal = False
            if pending_task is not None:
                prepared_outcome_by_candidate = {
                    **{
                        outcome.source_proposal.candidate_id: outcome
                        for outcome, _resolution in prepared_round.search_pairs
                        if outcome.source_proposal is not None
                    },
                    **{
                        candidate.candidate_id: outcome
                        for outcome, _resolution, candidate in (
                            prepared_round.open_search_pairs
                        )
                    },
                }
                for binding in prepared_round.search_bindings:
                    outcome = prepared_outcome_by_candidate.get(
                        binding.proposal.candidate_id
                    )
                    if outcome is None:
                        continue
                    resolution = _resolve_confirmation(
                        active_search_space_adapter,
                        pending_task=pending_task,
                        primary_binding=outcome,
                        next_discriminative_test=(
                            prepared_round.next_discriminative_test
                        ),
                    )
                    prepared_confirmation_needs_proposal = (
                        prepared_confirmation_needs_proposal
                        or resolution.kind
                        is ConfirmationResolutionKindV1.NEEDS_PROPOSAL
                    )
                    if (
                        resolution.kind
                        is ConfirmationResolutionKindV1.NEEDS_PROPOSAL
                    ):
                        prepared_confirmation_rejections.append(
                            canonical_value(
                                {
                                    "outcome_digest": outcome.digest,
                                    "candidate_id": binding.proposal.candidate_id,
                                    "spec_digest": outcome.spec.digest,
                                    "producer_role": outcome.producer_role,
                                    "resolution_kind": resolution.kind.value,
                                    "reason": resolution.reason,
                                    "task_id": pending_task.get("task_id"),
                                    "original_directive": (
                                        _feedback_confirmation_task_directive(
                                            pending_task
                                        )
                                    ),
                                }
                            )
                        )
                    if (
                        resolution.kind
                        is ConfirmationResolutionKindV1.EXACT_BINDING
                        and resolution.binding is not None
                    ):
                        prepared_confirmation_bindings[
                            binding.proposal.candidate_id
                        ] = resolution.binding
                if prepared_confirmation_needs_proposal and len(
                    prepared_confirmation_bindings
                ) != len(prepared_round.search_bindings):
                    return _missing_search_round_result(
                        context=context,
                        active_profile=active_profile,
                        producer_outcomes=prepared_round.producer_outcomes,
                        carryover_outcomes=prepared_round.carryover_outcomes,
                        resolutions=prepared_round.resolutions,
                        deferred_innovation=(
                            prepared_round.deferred_innovation_outcomes
                        ),
                        deferred_search=prepared_round.deferred_search_outcomes,
                        acquisition=None,
                        innovation=prepared_round.innovation,
                        provider_traces=prepared_round.provider_traces,
                        next_discriminative_test=(
                            prepared_round.next_discriminative_test
                        ),
                        policy=policy,
                        memory_writer=memory_writer,
                        meta_research_inputs=meta_research_inputs,
                        search_ready_count=0,
                        innovation_required_count=0,
                        attempt_scheduler=True,
                        attempt_budget=(
                            int(max_attempts_per_round)
                            if max_attempts_per_round is not None
                            else None
                        ),
                        # This slate has now been consumed as stale evidence.
                        # Do not let Campaign recovery reload it as an
                        # executable pool on the next proposal generation.
                        prepared=None,
                        feedback_confirmation_rejections=tuple(
                            prepared_confirmation_rejections
                        ),
                        feedback_proposal_retry_required=True,
                        pending_task=pending_task,
                    )
                if prepared_confirmation_bindings:
                    pending_task = {
                        **dict(pending_task),
                        "confirmation_candidate_ids": tuple(
                            sorted(prepared_confirmation_bindings)
                        ),
                    }
                else:
                    pending_task = _close_unsupported_discovery_feedback_task(
                        pending_task
                    )
                    policy = _policy_without_discovery_feedback_task(policy)
            frozen_budget_snapshot = (
                prepared_round.search_slate.budget_snapshot
                if prepared_round.search_slate is not None
                else budget_snapshot
            )
            return _run_scheduled_attempts(
                context=context,
                active_profile=execution_profile,
                producer_outcomes=prepared_round.producer_outcomes,
                carryover_outcomes=prepared_round.carryover_outcomes,
                resolutions=prepared_round.resolutions,
                deferred_innovation=prepared_round.deferred_innovation_outcomes,
                deferred_search=prepared_round.deferred_search_outcomes,
                search_acquisition=prepared_round.search_acquisition,
                search_bindings=prepared_round.search_bindings,
                search_pairs=prepared_round.search_pairs,
                open_search_pairs=prepared_round.open_search_pairs,
                innovation=prepared_round.innovation,
                budget_snapshot=frozen_budget_snapshot,
                router=router,
                policy=policy,
                memory_writer=memory_writer,
                runner=runner,
                incumbent_observation=incumbent_observation,
                metric_contract_digest=prepared_round.metric_contract_digest,
                observation_seed=prepared_round.observation_seed,
                next_discriminative_test=prepared_round.next_discriminative_test,
                confirmation_seed=getattr(prepared_round, "confirmation_seed", None),
                qualified_execution_by_capability=effective_qualified_execution,
                candidate_root_by_capability=effective_candidate_roots,
                resource_profile_by_capability=effective_resource_profiles,
                meta_research_inputs=meta_research_inputs,
                provider_traces=prepared_round.provider_traces,
                prepared_round=prepared_round,
                max_attempts_per_round=max_attempts_per_round,
                recovered_attempts=recovered_attempts,
                executed_semantic_seed_pairs=executed_semantic_seed_pairs,
                pending_task=pending_task,
                mechanism_axis_effects=mechanism_axis_effects,
                portfolio_candidates=normalized_portfolio,
                candidate_handoffs=candidate_handoffs,
                evidence_port=evidence_port,
                observation_seed_schedule=observation_seed_schedule,
                evaluator=normalized_evaluator,
                split=split,
                search_space_adapter=active_search_space_adapter,
                confirmation_bindings=prepared_confirmation_bindings,
                recovered_evidence_pre_trace=recovered_evidence_pre_trace,
                result_active_profile=active_profile,
                verification_only=(round_role == "VERIFICATION"),
            )
    (
        executed_semantic_seed_pairs,
        pending_task,
        mechanism_axis_effects,
    ) = _search_ranking_inputs(context)
    if round_role == "DISCOVERY":
        # Search only the executable capabilities frozen before this round.
        # Current-round Innovation may qualify future capabilities, but it
        # cannot replace or clear this Search pool. Evidence informs candidate
        # design without imposing a pending verification/control task on
        # discovery; explicitly scheduled verification uses the other branch.
        pending_task = _discovery_feedback_task(
            context,
            observation_seed=observation_seed,
        )
    elif round_role == "VERIFICATION":
        pending_task = _verification_feedback_task(
            context,
            observation_seed=observation_seed,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        )
    elif round_role == "LEGACY_MIXED":
        # Compatibility callers may consume explicit carryover candidates, but
        # queued verification cannot silently hijack that mixed round. Exact
        # task execution is owned exclusively by VERIFICATION.
        pending_task = None
    replay_producer = (
        getattr(meta_research_inputs.offline_replay, "producer", None)
        if meta_research_inputs is not None
        else None
    )
    trace_starts = _trace_starts(
        producer,
        innovation_inputs.implementer if innovation_inputs is not None else None,
        replay_producer,
    )
    exact_verification_task = (
        _exact_verification_task_bypasses_provider(
            pending_task,
            observation_seed=observation_seed,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        )
        if round_role == "VERIFICATION"
        else False
    )
    unbound_verification_task = (
        round_role == "VERIFICATION"
        and isinstance(pending_task, Mapping)
        and isinstance(pending_task.get("task_record"), Mapping)
        and pending_task["task_record"].get("metadata", {}).get(
            "execution_state"
        )
        == "AWAITING_CANDIDATE_BINDING"
    )
    if round_role == "VERIFICATION" and not (
        exact_verification_task or unbound_verification_task
    ):
        raise ValueError(
            "VERIFICATION requires one exact queued task bound to this observation seed"
        )
    producer_outcomes = (
        resume_retry_outcomes
        if resume_retry_outcomes is not None
        else (
            ()
            if exact_verification_task
            else produce_research_specs(
                context,
                producer,
                producer_bindings,
                frozen_profile_ref=normalized_frozen_profile_ref,
                active_task_directive=_active_task_directive(
                    pending_task,
                    observation_seed=observation_seed,
                ),
                lineage_parent_mechanism_program=research_parent_program(),
                construction_parent_options=construction_parent_options,
                latest_completed_execution=latest_completed_execution,
                research_window_budget=budget_snapshot.get("research_window"),
                search_space_adapter=active_search_space_adapter,
            )
        )
    )
    legacy_carryover_outcomes = _carryover_outcomes(
        context, carryover_proposals, producer_bindings
    )
    open_carryover_pairs = _carryover_open_outcomes(
        context,
        carryover_open_candidates,
    )
    open_candidate_by_outcome_digest = {
        outcome.digest: candidate for outcome, candidate in open_carryover_pairs
    }
    carryover_outcomes = (
        *legacy_carryover_outcomes,
        *(outcome for outcome, _candidate in open_carryover_pairs),
    )
    first_provider_traces = _new_provider_traces(trace_starts)
    pending_provider_traces = tuple((
        *resume_retry_trace_prefix, *resume_transient_trace_prefix, *first_provider_traces,
    ))
    latest_by_role = _latest_research_traces_by_role(pending_provider_traces)
    if attempt_scheduler and any(
        outcome.spec is None
        and provider_failure_is_external(latest_by_role.get(outcome.producer_role, {}).get("failure"))
        for outcome in producer_outcomes
    ):
        # Missing external responses are still research opportunities. Persist
        # the existing portfolio before selecting or implementing a candidate.
        pending_prepared = PreparedResearchRoundV1(
            context_digest=context.digest, profile_ref=active_profile.profile_ref,
            profile_digest=active_profile.profile_digest, producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes, resolutions=(),
            deferred_innovation_outcomes=(), deferred_search_outcomes=(),
            search_acquisition=None, search_slate=None, search_bindings=(),
            search_pairs=(), open_search_pairs=(), innovation=None,
            metric_contract_digest=metric_contract_digest, observation_seed=observation_seed,
            next_discriminative_test=next_discriminative_test, confirmation_seed=confirmation_seed,
            provider_traces=pending_provider_traces, portfolio_candidates=normalized_portfolio,
            candidate_handoffs=(), prebinding_retry=prebinding_retry_mapping,
        )
        if on_prepared is not None:
            on_prepared(pending_prepared)
        return ResearchRoundResult(
            context=context, active_profile=active_profile, producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes, resolutions=(),
            deferred_innovation_outcomes=(), deferred_search_outcomes=(),
            search_acquisition=None, innovation=None, selected_outcome=None,
            execution_recipe=None, candidate_run=None, interpretation=None,
            provider_traces=pending_provider_traces, meta_research=None, attempts=(),
            metric_bearing_attempt_index=None, attempt_scheduler_enabled=True,
            incomplete_reason="ROUND_PRODUCER_PROVIDER_UNAVAILABLE", prepared=pending_prepared,
        )
    if (
        attempt_scheduler
        and prebinding_token_ceiling_retry
        and prebinding_retry_mapping is None
        and resume_retry_outcomes is None
        and _all_scheduled_token_ceiling_failures(
            producer_outcomes,
            first_provider_traces,
            scheduled_roles=research_producer_roles(context.budget),
        )
    ):
        retry_producer = _namespaced_retry_producer(producer)
        if retry_producer is not None:
            pending_retry_mapping = canonical_value(
                {
                    "ordinal": _PREBINDING_RETRY_ORDINAL,
                    "status": "PENDING",
                    "context_digest": context.digest,
                    "profile_ref": active_profile.profile_ref,
                    "profile_digest": active_profile.profile_digest,
                    "budget_snapshot_digest": sha256_digest(budget_snapshot),
                    "first_outcomes": tuple(
                        _trace_value(item) for item in producer_outcomes
                    ),
                    "first_provider_traces": first_provider_traces,
                    "first_provider_trace_digest": sha256_digest(
                        first_provider_traces
                    ),
                    "usage_charge": _provider_usage_charge(
                        first_provider_traces,
                        producer,
                    ),
                    "retry_logical_namespace": _PREBINDING_RETRY_NAMESPACE,
                    "retry_logical_namespace_digest": sha256_digest(
                        _PREBINDING_RETRY_NAMESPACE
                    ),
                }
            )
            pending_prepared = PreparedResearchRoundV1(
                context_digest=context.digest,
                profile_ref=active_profile.profile_ref,
                profile_digest=active_profile.profile_digest,
                producer_outcomes=producer_outcomes,
                carryover_outcomes=carryover_outcomes,
                resolutions=(),
                deferred_innovation_outcomes=(),
                deferred_search_outcomes=(),
                search_acquisition=None,
                search_slate=None,
                search_bindings=(),
                search_pairs=(),
                open_search_pairs=(),
                innovation=None,
                metric_contract_digest=metric_contract_digest,
                observation_seed=observation_seed,
                next_discriminative_test=next_discriminative_test,
                confirmation_seed=confirmation_seed,
                provider_traces=first_provider_traces,
                portfolio_candidates=normalized_portfolio,
                candidate_handoffs=(),
                prebinding_retry=pending_retry_mapping,
            )
            if on_prepared is not None:
                on_prepared(pending_prepared)
            retry_trace_starts = _trace_starts(producer)
            retry_outcomes = produce_research_specs(
                context,
                retry_producer,
                producer_bindings,
                frozen_profile_ref=normalized_frozen_profile_ref,
                active_task_directive=_active_task_directive(
                    pending_task,
                    observation_seed=observation_seed,
                ),
                lineage_parent_mechanism_program=research_parent_program(),
                construction_parent_options=construction_parent_options,
                latest_completed_execution=latest_completed_execution,
                research_window_budget=budget_snapshot.get("research_window"),
                search_space_adapter=active_search_space_adapter,
            )
            retry_traces = _new_provider_traces(retry_trace_starts)
            retry_status = (
                "SUCCEEDED"
                if any(item.spec is not None for item in retry_outcomes)
                else "EXHAUSTED"
            )
            prebinding_retry_mapping = canonical_value(
                {
                    **dict(pending_retry_mapping),
                    "status": retry_status,
                    "usage_charge": int(pending_retry_mapping["usage_charge"])
                    + _provider_usage_charge(retry_traces, producer),
                    "retry_trace_digest": sha256_digest(retry_traces),
                    "retry_logical_calls_digest": sha256_digest(
                        tuple(
                            str(item.get("logical_call_id"))
                            for item in retry_traces
                            if item.get("logical_call_id") is not None
                        )
                    ),
                }
            )
            retry_prepared = replace(
                pending_prepared,
                producer_outcomes=retry_outcomes,
                provider_traces=tuple((*first_provider_traces, *retry_traces)),
                prebinding_retry=prebinding_retry_mapping,
            )
            if on_prepared is not None:
                on_prepared(retry_prepared)
            if retry_status == "EXHAUSTED":
                if not carryover_outcomes:
                    return ResearchRoundResult(
                        context=context,
                        active_profile=active_profile,
                        producer_outcomes=retry_outcomes,
                        carryover_outcomes=carryover_outcomes,
                        resolutions=(),
                        deferred_innovation_outcomes=(),
                        deferred_search_outcomes=(),
                        search_acquisition=None,
                        innovation=None,
                        selected_outcome=None,
                        execution_recipe=None,
                        candidate_run=None,
                        interpretation=None,
                        provider_traces=retry_prepared.provider_traces,
                        meta_research=None,
                        attempts=(),
                        metric_bearing_attempt_index=None,
                        attempt_scheduler_enabled=True,
                        incomplete_reason="ROUND_PREBINDING_RETRY_EXHAUSTED",
                        prepared=retry_prepared,
                    )
            producer_outcomes = retry_outcomes
    resolutions = resolve_producer_outcomes(
        (*carryover_outcomes, *producer_outcomes),
        environment=resolver_environment,
    )
    confirmation_by_outcome_digest: dict[str, ConfirmationResolutionV1] = {}
    exact_confirmation_outcome_digests: frozenset[str] = frozenset()
    feedback_confirmation_rejections: tuple[Mapping[str, Any], ...] = ()
    confirmation_needs_proposal = False
    feedback_confirmation_active = (
        round_role == "DISCOVERY" and pending_task is not None
    )
    if feedback_confirmation_active:
        confirmation_by_outcome_digest = {
            outcome.digest: _resolve_confirmation(
                active_search_space_adapter,
                pending_task=pending_task,
                primary_binding=outcome,
                next_discriminative_test=next_discriminative_test,
            )
            for outcome, resolution in resolutions
            if resolution is not None and outcome.spec is not None
        }
        exact_confirmation_outcome_digests = frozenset(
            digest
            for digest, resolution in confirmation_by_outcome_digest.items()
            if resolution.kind is ConfirmationResolutionKindV1.EXACT_BINDING
        )
        confirmation_needs_proposal = any(
            resolution.kind is ConfirmationResolutionKindV1.NEEDS_PROPOSAL
            for resolution in confirmation_by_outcome_digest.values()
        )
        feedback_confirmation_rejections = tuple(
            canonical_value(
                {
                    "outcome_digest": outcome.digest,
                    "candidate_id": (
                        outcome.source_proposal.candidate_id
                        if outcome.source_proposal is not None
                        else outcome.spec.spec_id
                    ),
                    "spec_digest": outcome.spec.digest,
                    "producer_role": outcome.producer_role,
                    "resolution_kind": confirmation.kind.value,
                    "reason": confirmation.reason,
                    "task_id": pending_task.get("task_id"),
                    "original_directive": _feedback_confirmation_task_directive(
                        pending_task
                    ),
                }
            )
            for outcome, _resolution in resolutions
            for confirmation in (
                confirmation_by_outcome_digest.get(outcome.digest),
            )
            if outcome.spec is not None
            and confirmation is not None
            and confirmation.kind is ConfirmationResolutionKindV1.NEEDS_PROPOSAL
        )
        if not exact_confirmation_outcome_digests:
            if not confirmation_needs_proposal:
                pending_task = _close_unsupported_discovery_feedback_task(
                    pending_task
                )
                policy = _policy_without_discovery_feedback_task(policy)
    innovation_outcome_selector = getattr(
        active_search_space_adapter,
        "select_innovation_outcome",
        None,
    )
    if not callable(innovation_outcome_selector):
        innovation_outcome_selector = getattr(
            router,
            "select_innovation_outcome",
            None,
        )
    if not callable(innovation_outcome_selector):
        innovation_outcome_selector = None
    executable_discovery_outcome_digests: frozenset[str] | None = None
    discovery_outcome_rank: Mapping[str, int] = {}
    if (
        round_role == "DISCOVERY"
        and not exact_confirmation_outcome_digests
        and innovation_outcome_selector is None
    ):
        attempted_effective_experiments = (
            _attempted_effective_experiment_identities(context)
        )
        discovery_opportunities = []
        for outcome, resolution in resolutions:
            if (
                resolution is None
                or outcome.spec is None
                or resolution.resolution not in {
                    CapabilityResolutionResultV1.SEARCH_READY,
                    CapabilityResolutionResultV1.INNOVATION_REQUIRED,
                }
            ):
                continue
            admitted_candidate = open_candidate_by_outcome_digest.get(outcome.digest)
            if admitted_candidate is not None:
                # Carryover resolution facts describe current eligibility, not
                # the admitted program. Use its existing executable binding.
                admitted_binding = bind_search_candidate(
                    profile=active_profile,
                    proposal=admitted_candidate,
                    capability_ref=admitted_candidate.capability_ref,
                )
                _, admitted_identity, _ = _adapter_execution_attestation(
                    admitted_binding,
                    search_space_adapter=active_search_space_adapter,
                    confirmation_binding=None,
                    execution_context={
                        "profile": execution_profile,
                        "qualified_execution": effective_qualified_execution.get(
                            admitted_binding.capability_ref
                        ),
                        "evaluator": normalized_evaluator,
                        "split": split,
                    },
                )
                experiment_digest = str(
                    admitted_identity["effective_experiment_digest"]
                )
            else:
                identity = _innovation_identity_digests(
                    outcome,
                    search_space_adapter=active_search_space_adapter,
                    frozen_profile_ref=normalized_frozen_profile_ref,
                    research_context=context,
                )
                if identity is None:
                    continue
                experiment_digest = identity[1]
            if experiment_digest not in attempted_effective_experiments:
                discovery_opportunities.append((outcome, resolution))
        if discovery_opportunities:
            if resume_resource_outcome_digest is not None:
                opportunity_digests = tuple(
                    outcome.digest for outcome, _resolution in discovery_opportunities
                )
                if resume_resource_outcome_digest not in opportunity_digests:
                    raise ValueError(
                        "recoverable resource candidate is absent from resumed outcomes"
                    )
                selected_digest = resume_resource_outcome_digest
                ranked_digests = (
                    selected_digest,
                    *(
                        digest
                        for digest in opportunity_digests
                        if digest != selected_digest
                    ),
                )
            else:
                selected_digest, ranked_digests = _select_common_producer_opportunity(
                    discovery_opportunities,
                    research_policy=policy,
                    research_context=context,
                )
            opportunity_by_digest = {
                outcome.digest: resolution
                for outcome, resolution in discovery_opportunities
            }
            selected_resolution = opportunity_by_digest[selected_digest]
            executable_discovery_outcome_digests = frozenset(
                digest
                for digest in ranked_digests
                if opportunity_by_digest[digest].resolution
                is selected_resolution.resolution
            )
            discovery_outcome_rank = {
                digest: index for index, digest in enumerate(ranked_digests)
            }
        else:
            executable_discovery_outcome_digests = frozenset()
    search_pairs = tuple(sorted((
        (outcome, resolution)
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
        and outcome.source_proposal is not None
        and (
            executable_discovery_outcome_digests is None
            or outcome.digest in executable_discovery_outcome_digests
        )
    ), key=lambda pair: discovery_outcome_rank.get(pair[0].digest, 0)))
    open_search_pairs = tuple(sorted((
        (outcome, resolution, open_candidate_by_outcome_digest[outcome.digest])
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
        and outcome.digest in open_candidate_by_outcome_digest
        and (
            executable_discovery_outcome_digests is None
            or outcome.digest in executable_discovery_outcome_digests
        )
    ), key=lambda pair: discovery_outcome_rank.get(pair[0].digest, 0)))
    if feedback_confirmation_active and exact_confirmation_outcome_digests:
        # Generic runtime accepts only adapter-confirmed opaque bindings.  It
        # does not inspect the search-space program or infer control lineage.
        search_pairs = tuple(
            pair
            for pair in search_pairs
            if pair[0].digest in exact_confirmation_outcome_digests
        )
        open_search_pairs = tuple(
            pair
            for pair in open_search_pairs
            if pair[0].digest in exact_confirmation_outcome_digests
        )
    elif feedback_confirmation_active and confirmation_needs_proposal:
        # The Provider slate did not contain the exact adapter-owned binding.
        # Preserve the AWAITING task across generations; no unrelated candidate
        # may enter Router merely because the bounded generation ended.
        search_pairs = ()
        open_search_pairs = ()
    deferred_search = tuple(sorted((
        (outcome, resolution)
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
        and outcome.source_proposal is None
        and outcome.digest not in open_candidate_by_outcome_digest
        and (
            executable_discovery_outcome_digests is None
            or outcome.digest in executable_discovery_outcome_digests
        )
    ), key=lambda pair: discovery_outcome_rank.get(pair[0].digest, 0)))
    search_bindings = tuple(
        [
            _materialize_discovery_binding(
                active_search_space_adapter,
                outcome=outcome,
                resolution=resolution,
                profile=active_profile,
            )
            for outcome, resolution in search_pairs
            if outcome.source_proposal is not None
            and resolution.resolved_current_capability_ref is not None
        ]
        + [
            bind_search_candidate(
                profile=active_profile,
                proposal=candidate,
                capability_ref=candidate.capability_ref,
            )
            for _outcome, _resolution, candidate in open_search_pairs
        ]
    )
    discovery_candidate_outcome_digests = {
        **{
            outcome.source_proposal.candidate_id: outcome.digest
            for outcome, _resolution in search_pairs
            if outcome.source_proposal is not None
        },
        **{
            candidate.candidate_id: outcome.digest
            for outcome, _resolution, candidate in open_search_pairs
        },
    }
    discovery_preferred_candidate_ids = tuple(
        candidate_id
        for candidate_id, _outcome_digest in sorted(
            discovery_candidate_outcome_digests.items(),
            key=lambda item: discovery_outcome_rank.get(
                item[1], len(discovery_outcome_rank)
            ),
        )
    )
    confirmation_bindings: dict[str, Mapping[str, Any]] = {}
    if exact_confirmation_outcome_digests:
        outcome_digest_by_candidate = {
            **{
                outcome.source_proposal.candidate_id: outcome.digest
                for outcome, _resolution in search_pairs
                if outcome.source_proposal is not None
            },
            **{
                candidate.candidate_id: outcome.digest
                for outcome, _resolution, candidate in open_search_pairs
            },
        }
        for binding in search_bindings:
            outcome_digest = outcome_digest_by_candidate.get(
                binding.proposal.candidate_id
            )
            resolution = confirmation_by_outcome_digest.get(str(outcome_digest))
            if (
                resolution is not None
                and resolution.kind is ConfirmationResolutionKindV1.EXACT_BINDING
                and resolution.binding is not None
            ):
                confirmation_bindings[binding.proposal.candidate_id] = (
                    resolution.binding
                )
        pending_task = {
            **dict(pending_task),
            "confirmation_candidate_ids": tuple(
                sorted(confirmation_bindings)
            ),
        }
    search_bindings = _router_eligible_effective_bindings(
        context=context,
        bindings=search_bindings,
        observation_seed=observation_seed,
        pending_task=pending_task,
        executed_semantic_seed_pairs=executed_semantic_seed_pairs,
        search_space_adapter=active_search_space_adapter,
        confirmation_bindings=confirmation_bindings,
        profile=execution_profile,
        qualified_execution_by_capability=effective_qualified_execution,
        evaluator=normalized_evaluator,
        split=split,
    )
    if exact_verification_task and not any(
        binding.mechanism_semantics_digest
        == pending_task.get("candidate_semantic_digest")
        for binding in search_bindings
    ):
        (
            execution_profile,
            control_outcome,
            control_resolution,
            control_binding,
        ) = _package_control_verification_binding(
            context=context,
            policy=policy,
            producer_bindings=producer_bindings,
            resolver_environment=resolver_environment,
            pending_task=pending_task,
            search_space_adapter=active_search_space_adapter,
        )
        carryover_outcomes = (control_outcome,)
        resolutions = ((control_outcome, control_resolution),)
        search_pairs = ((control_outcome, control_resolution),)
        open_search_pairs = ()
        deferred_search = ()
        search_bindings = (control_binding,)
    candidate_handoffs = _candidate_handoffs_for_round(
        context=context,
        active_profile=execution_profile,
        resolutions=resolutions,
        search_bindings=search_bindings,
        portfolio_candidates=normalized_portfolio,
        research_profile_source=research_profile_source,
        candidate_handoff_factory=candidate_handoff_factory,
        qualified_execution_by_capability=effective_qualified_execution,
        candidate_root_by_capability=effective_candidate_roots,
        resource_profile_by_capability=effective_resource_profiles,
        evaluator=normalized_evaluator,
        split=split,
        search_space_adapter=active_search_space_adapter,
    )
    if candidate_handoffs:
        normalized_portfolio = tuple(
            item.portfolio_candidate for item in candidate_handoffs
        )
    slate: FrozenExperimentSlateV1 | None = None
    if search_bindings:
        slate = freeze_experiment_slate(
            profile=execution_profile,
            bindings=search_bindings,
            budget_snapshot=budget_snapshot,
        )
        routing_executed_pairs = executed_semantic_seed_pairs
        if exact_verification_task:
            # The exact adapter binding has already been checked against the
            # durable task above.  Prior observations of that semantic family
            # must not make the separately scheduled verification look like
            # an ordinary discovery duplicate to the generic Router.
            routing_executed_pairs = tuple(
                pair
                for pair in executed_semantic_seed_pairs
                if pair[0] != pending_task.get("candidate_semantic_digest")
            )
        acquisition = route_frozen_experiment_slate(
            profile=execution_profile,
            slate=slate,
            router=router,
            policy_projection=policy.to_dict(),
            executed_semantic_seed_pairs=routing_executed_pairs,
            current_observation_seed=observation_seed,
            pending_task=pending_task,
            mechanism_axis_effects=mechanism_axis_effects,
            portfolio_candidates=(
                normalized_portfolio if normalized_portfolio else None
            ),
            preferred_candidate_ids=discovery_preferred_candidate_ids,
        )
    else:
        acquisition = None
    innovation_pairs = tuple(sorted((
        (outcome, resolution)
        for outcome, resolution in resolutions
        if resolution is not None
        and resolution.resolution is CapabilityResolutionResultV1.INNOVATION_REQUIRED
        and (
            executable_discovery_outcome_digests is None
            or outcome.digest in executable_discovery_outcome_digests
        )
    ), key=lambda pair: discovery_outcome_rank.get(pair[0].digest, 0)))
    if resume_candidate_local_innovation is not None:
        consumed_spec_digests = {
            _innovation_attempt_proposal_digest(item)
            for item in resume_candidate_local_innovation.attempts
            if isinstance(item, Mapping)
            and isinstance(_innovation_attempt_proposal_digest(item), str)
        }
        innovation_pairs = tuple(
            pair
            for pair in innovation_pairs
            if pair[0].spec is not None
            and pair[0].spec.digest not in consumed_spec_digests
        )
    if exact_confirmation_outcome_digests:
        innovation_pairs = tuple(
            pair
            for pair in innovation_pairs
            if pair[0].digest in exact_confirmation_outcome_digests
        )
    elif feedback_confirmation_active and confirmation_needs_proposal:
        # The adapter has explicitly said that none of this generation's
        # proposals resolves the retained task. Do not reinterpret an
        # unrelated INNOVATION_REQUIRED outcome as executable work.
        innovation_pairs = ()
    # A selected resource retry narrows execution, not the frozen candidate
    # queue. Retain the other opportunities for native same-slot rerouting.
    frozen_innovation_pairs = innovation_pairs
    if resume_resource_outcome_digest is not None:
        innovation_pairs = tuple(
            pair for pair in innovation_pairs
            if pair[0].digest == resume_resource_outcome_digest
        )
    consumed_innovation_attempts = tuple(
        item for item in resume_innovation_attempts
        if resume_resource_outcome_digest is None
        or _innovation_attempt_proposal_digest(item) not in {
            pair[0].spec.digest for pair in innovation_pairs
            if pair[0].spec is not None
        }
    )
    deferred_innovation = tuple(
        (outcome, resolution)
        for outcome, resolution in innovation_pairs
        if innovation_inputs is None
    )
    if acquisition is not None:
        current_slate_ref = acquisition.slate_ref
        current_slate_digest = acquisition.slate_digest
    else:
        missing_slate = {
            "campaign_id": context.campaign_id,
            "profile_ref": active_profile.profile_ref,
            "profile_digest": active_profile.profile_digest,
            "budget_snapshot": canonical_value(budget_snapshot),
            "search_candidate_count": 0,
        }
        current_slate_ref = content_id(
            "recclaw-missing-search-opportunity-v1",
            missing_slate,
        )
        current_slate_digest = sha256_digest(missing_slate)
    innovation = (
        _run_innovation_lane(
            innovation_pairs,
            current_profile=active_profile,
            current_slate_ref=current_slate_ref,
            current_slate_digest=current_slate_digest,
            research_policy=policy,
            search_space_adapter=active_search_space_adapter,
            research_context=context,
            inputs=innovation_inputs,
            evaluator=normalized_evaluator,
            split=split,
            frozen_profile_ref=normalized_frozen_profile_ref,
            implementation_call_limit=(
                int(
                    context.budget.get(
                        "implementer_logical_calls_max",
                        MAX_REPAIR_TURNS + 1,
                    )
                )
                if attempt_scheduler
                else MAX_REPAIR_TURNS + 1
            ),
            candidate_attempt_limit=(
                int(max_attempts_per_round)
                if attempt_scheduler and max_attempts_per_round is not None
                else 1
            ),
            allow_candidate_reroute=attempt_scheduler,
            # Feedback candidates were already admitted through the injected
            # SearchSpaceAdapter; generic acquisition must not reinterpret
            # their search-space semantics.
            enforce_unbound_confirmation=False,
            outcome_selector=innovation_outcome_selector,
            attempted_semantic_identities=(
                _attempted_semantic_identities(context)
                | frozenset(
                    str(item["mechanism_semantics_digest"])
                    for item in (
                        consumed_innovation_attempts
                    )
                    if isinstance(item, Mapping)
                    and isinstance(item.get("mechanism_semantics_digest"), str)
                )
            ),
            attempted_effective_experiment_identities=(
                _attempted_effective_experiment_identities(context)
                | frozenset(
                    str(item["effective_experiment_digest"])
                    for item in (
                        consumed_innovation_attempts
                    )
                    if isinstance(item, Mapping)
                    and isinstance(item.get("effective_experiment_digest"), str)
                )
            ),
            _prior_attempts=resume_innovation_attempts,
            _reroute_index=resume_innovation_candidate_count,
            _qualified_retry=resume_qualified_innovation,
            _provider_retry_request=resume_provider_request,
            _provider_retry_namespace=resume_provider_namespace,
            _provider_retry_remaining_calls=resume_provider_remaining_calls,
            exact_parent_bundle_loader=exact_parent_bundle_loader,
        )
        if innovation_inputs is not None and innovation_pairs
        else None
    )
    if innovation is None and resume_candidate_local_innovation is not None:
        innovation = resume_candidate_local_innovation
    if innovation is not None:
        attempted_innovation_digests = {
            _innovation_attempt_proposal_digest(item)
            for item in innovation.attempts
            if _innovation_attempt_proposal_digest(item) is not None
        }
        deferred_innovation = tuple(
            pair
            for pair in frozen_innovation_pairs
            if pair[0].spec is not None
            and pair[0].spec.digest not in attempted_innovation_digests
        )
    innovation_confirmation_resolution = (
        confirmation_by_outcome_digest.get(innovation.selected_outcome.digest)
        if innovation is not None
        else None
    )
    innovation_exact_confirmation = (
        innovation_confirmation_resolution is not None
        and innovation_confirmation_resolution.kind
        is ConfirmationResolutionKindV1.EXACT_BINDING
        and innovation_confirmation_resolution.binding is not None
    )
    existing_observation_rebind = _existing_exact_feedback_observation_rebind(
        context=context,
        pending_task=pending_task,
        innovation=innovation,
        exact_confirmation=innovation_exact_confirmation,
        evaluator=normalized_evaluator,
        split=split,
        observation_seed=observation_seed,
        metric_contract_digest=metric_contract_digest,
    )
    if (
        (acquisition is None or acquisition.selected_binding is None)
        and innovation is not None
        and innovation.training_ready
        and innovation.registry is not None
        and innovation.next_profile is not None
        and innovation.capability is not None
        and innovation.search_candidate is not None
        and innovation.qualified_execution is not None
        and innovation.candidate_root is not None
        and (
            not feedback_confirmation_active
            or innovation_exact_confirmation
        )
    ):
        execution_profile = activate_next_fresh_search_profile(
            predecessor=active_profile,
            next_profile=innovation.next_profile,
            registry=innovation.registry,
            fresh_campaign_id=innovation.fresh_campaign_id,
        )
        innovation_binding = bind_search_candidate(
            profile=execution_profile,
            proposal=innovation.search_candidate,
            capability_ref=innovation.capability.capability_id,
        )
        confirmation_resolution = innovation_confirmation_resolution
        if (
            confirmation_resolution is not None
            and confirmation_resolution.kind
            is ConfirmationResolutionKindV1.EXACT_BINDING
            and confirmation_resolution.binding is not None
        ):
            confirmation_bindings[
                innovation_binding.proposal.candidate_id
            ] = confirmation_resolution.binding
            pending_task = {
                **dict(pending_task or {}),
                "confirmation_candidate_ids": tuple(
                    sorted(confirmation_bindings)
                ),
            }
        search_bindings = (innovation_binding,)
        if isinstance(innovation.search_candidate, CandidateProposalV4):
            execution_outcome = innovation.selected_outcome
            search_pairs = ((execution_outcome, innovation.resolution),)
            open_search_pairs = ()
        else:
            execution_outcome = _implementation_outcome(
                innovation.selected_outcome,
                innovation.search_candidate.execution_contract,
            )
            search_pairs = ()
            open_search_pairs = (
                (
                    execution_outcome,
                    innovation.resolution,
                    innovation.search_candidate,
                ),
            )
        effective_qualified_execution[
            innovation.capability.capability_id
        ] = innovation.qualified_execution
        effective_candidate_roots[
            innovation.capability.capability_id
        ] = innovation.candidate_root
        if innovation.resource_profile is not None:
            effective_resource_profiles[
                innovation.capability.capability_id
            ] = innovation.resource_profile
        candidate_handoffs = _candidate_handoffs_for_round(
            context=context,
            active_profile=execution_profile,
            resolutions=((execution_outcome, innovation.resolution),),
            search_bindings=search_bindings,
            portfolio_candidates=(),
            research_profile_source=research_profile_source,
            candidate_handoff_factory=candidate_handoff_factory,
            qualified_execution_by_capability=effective_qualified_execution,
            candidate_root_by_capability=effective_candidate_roots,
            resource_profile_by_capability=effective_resource_profiles,
            evaluator=normalized_evaluator,
            split=split,
            search_space_adapter=active_search_space_adapter,
        )
        normalized_portfolio = tuple(
            item.portfolio_candidate for item in candidate_handoffs
        )
        slate = freeze_experiment_slate(
            profile=execution_profile,
            bindings=search_bindings,
            budget_snapshot=budget_snapshot,
        )
        acquisition = route_frozen_experiment_slate(
            profile=execution_profile,
            slate=slate,
            router=router,
            policy_projection=policy.to_dict(),
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            current_observation_seed=observation_seed,
            pending_task=pending_task,
            mechanism_axis_effects=mechanism_axis_effects,
            portfolio_candidates=(
                normalized_portfolio if normalized_portfolio else None
            ),
        )
    provider_traces = tuple(
        (
            *resume_retry_trace_prefix,
            *resume_transient_trace_prefix,
            *_new_provider_traces(trace_starts),
        )
    )
    prepared = (
        PreparedResearchRoundV1(
            context_digest=context.digest,
            profile_ref=active_profile.profile_ref,
            profile_digest=active_profile.profile_digest,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation_outcomes=deferred_innovation,
            deferred_search_outcomes=deferred_search,
            search_acquisition=acquisition,
            search_slate=slate,
            search_bindings=search_bindings,
            search_pairs=search_pairs,
            open_search_pairs=open_search_pairs,
            innovation=innovation,
            metric_contract_digest=metric_contract_digest,
            observation_seed=observation_seed,
            next_discriminative_test=next_discriminative_test,
            confirmation_seed=confirmation_seed,
            provider_traces=provider_traces,
            portfolio_candidates=normalized_portfolio,
            candidate_handoffs=candidate_handoffs,
            prebinding_retry=prebinding_retry_mapping,
        )
        if attempt_scheduler
        else None
    )
    if attempt_scheduler and prepared is not None and on_prepared is not None:
        # This is the scheduler's durable boundary: the callback must return
        # before the first physical runner invocation below.
        on_prepared(prepared)
    binding = acquisition.selected_binding if acquisition is not None else None
    if binding is None:
        if feedback_confirmation_rejections and innovation is not None:
            innovation_attempt_by_spec = {
                str(item["spec_digest"]): item
                for item in innovation.attempts
                if isinstance(item, Mapping)
                and isinstance(item.get("spec_digest"), str)
            }
            enriched_rejections: list[Mapping[str, Any]] = []
            for item in feedback_confirmation_rejections:
                payload = dict(item)
                attempt = innovation_attempt_by_spec.get(
                    str(item.get("spec_digest"))
                )
                if isinstance(attempt, Mapping):
                    payload.update(
                        candidate_id=attempt.get("candidate_id"),
                        candidate_semantic_digest=attempt.get(
                            "mechanism_semantics_digest"
                        ),
                        effective_experiment_digest=attempt.get(
                            "effective_experiment_digest"
                        ),
                        effective_family_digest=attempt.get(
                            "effective_family_digest"
                        ),
                        primitive_ids=attempt.get("primitive_ids"),
                    )
                enriched_rejections.append(canonical_value(payload))
            feedback_confirmation_rejections = tuple(enriched_rejections)
        # Producer outcomes are bound to the original round Context.  Missing
        # opportunity interpretation must therefore use that same Context and
        # policy pair; any local feedback-task closure only controls routing in
        # this invocation and must not create a mismatched interpreter input.
        return _missing_search_round_result(
            context=context,
            producer_outcomes=producer_outcomes,
            active_profile=active_profile,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation=deferred_innovation,
            deferred_search=deferred_search,
            acquisition=acquisition,
            innovation=innovation,
            provider_traces=provider_traces,
            next_discriminative_test=next_discriminative_test,
            policy=policy,
            memory_writer=memory_writer,
            meta_research_inputs=meta_research_inputs,
            search_ready_count=len(search_pairs) + len(open_search_pairs),
            innovation_required_count=len(innovation_pairs),
            attempt_scheduler=attempt_scheduler,
            attempt_budget=(
                int(max_attempts_per_round)
                if attempt_scheduler and max_attempts_per_round is not None
                else None
            ),
            prepared=prepared,
            feedback_proposal_retry_required=(
                feedback_confirmation_active
                and confirmation_needs_proposal
                and not exact_confirmation_outcome_digests
            ),
            feedback_confirmation_rejections=(
                feedback_confirmation_rejections
            ),
            pending_task=pending_task,
            existing_observation_rebind=existing_observation_rebind,
        )
    if attempt_scheduler:
        return _run_scheduled_attempts(
            context=context,
            active_profile=execution_profile,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=carryover_outcomes,
            resolutions=resolutions,
            deferred_innovation=deferred_innovation,
            deferred_search=deferred_search,
            search_acquisition=acquisition,
            search_bindings=search_bindings,
            search_pairs=search_pairs,
            open_search_pairs=open_search_pairs,
            innovation=innovation,
            budget_snapshot=budget_snapshot,
            router=router,
            policy=policy,
            memory_writer=memory_writer,
            runner=runner,
            incumbent_observation=incumbent_observation,
            metric_contract_digest=metric_contract_digest,
            observation_seed=observation_seed,
            next_discriminative_test=next_discriminative_test,
            confirmation_seed=confirmation_seed,
            qualified_execution_by_capability=effective_qualified_execution,
            candidate_root_by_capability=effective_candidate_roots,
            resource_profile_by_capability=effective_resource_profiles,
            meta_research_inputs=meta_research_inputs,
            provider_traces=provider_traces,
            prepared_round=prepared,
            max_attempts_per_round=max_attempts_per_round,
            recovered_attempts=recovered_attempts,
            executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            pending_task=pending_task,
            mechanism_axis_effects=mechanism_axis_effects,
            portfolio_candidates=normalized_portfolio,
            candidate_handoffs=candidate_handoffs,
            evidence_port=evidence_port,
            observation_seed_schedule=observation_seed_schedule,
            evaluator=normalized_evaluator,
            split=split,
            search_space_adapter=active_search_space_adapter,
            confirmation_bindings=confirmation_bindings,
            preferred_candidate_ids=discovery_preferred_candidate_ids,
            recovered_evidence_pre_trace=recovered_evidence_pre_trace,
            result_active_profile=active_profile,
            verification_only=exact_verification_task,
        )
    selected_outcome = _selected_outcome_for_binding(
        binding,
        search_pairs=search_pairs,
        open_search_pairs=open_search_pairs,
    )
    if selected_outcome.source_proposal is not None:
        mechanism_projection: Mapping[str, Any] = binding.proposal.mechanism_program
    else:
        mechanism_projection = {
            "schema": "recclaw-open-spec-realization-semantics-e0.v1",
            "semantic_identity_ref": binding.proposal.semantic_identity_ref,
            "semantic_identity_digest": binding.proposal.semantic_identity_digest,
            "research_spec_ref": binding.proposal.spec.spec_id,
            "research_spec_digest": binding.proposal.spec.digest,
            "candidate_package_ref": binding.proposal.candidate_package_ref,
            "candidate_package_digest": binding.proposal.candidate_package_digest,
            "execution_contract": binding.proposal.execution_contract,
        }
    recipe, effective_identity, search_space_attestation = _execution_recipe_for_handoff(
        binding,
        profile=execution_profile,
        qualified_execution_by_capability=effective_qualified_execution,
        candidate_root_by_capability=effective_candidate_roots,
        resource_profile_by_capability=effective_resource_profiles,
        candidate_handoffs=candidate_handoffs,
        evaluator=normalized_evaluator,
        split=split,
        search_space_adapter=active_search_space_adapter,
        confirmation_binding=confirmation_bindings.get(
            binding.proposal.candidate_id
        ),
    )
    candidate_run = canonical_value(dict(runner(recipe, binding)))
    try:
        run_experiment_binding = ExperimentBindingV1.from_canonical_dict(
            candidate_run.get("experiment_binding")
        )
    except ExperimentBindingError as error:
        raise ValueError(str(error)) from error
    _validate_experiment_binding(
        run_experiment_binding,
        recipe=recipe,
        observation_seed=observation_seed,
    )
    if candidate_run.get("execution_recipe_digest") != sha256_digest(recipe):
        raise ValueError("runner result is not bound to the selected execution recipe")
    if str(candidate_run.get("seed")) != observation_seed:
        raise ValueError("runner result seed differs from the frozen observation seed")
    run_binding_digest = candidate_run.get("experiment_binding_digest")
    if (
        run_binding_digest != run_experiment_binding.digest
        or candidate_run.get("experiment_binding_ref")
        != run_experiment_binding.ref
        or candidate_run.get("binding_digest", run_binding_digest)
        != run_binding_digest
    ):
        raise ValueError("runner result carries an inconsistent Experiment Binding identity")
    event, identity, episode, closure, _failure_detail = (
        project_common_execution_feedback(
            context=context,
            selected_outcome=selected_outcome,
            binding=binding,
            candidate_run=candidate_run,
            incumbent_observation=incumbent_observation,
            metric_contract_digest=metric_contract_digest,
            observation_seed=observation_seed,
            next_discriminative_test=next_discriminative_test,
            evaluator=normalized_evaluator,
        )
    )
    route_metadata = {
        "route_trace_digest": acquisition.route_trace.digest,
        "selected_producer_role": selected_outcome.producer_role,
        "selected_candidate_id": binding.proposal.candidate_id,
        "selected_candidate_semantic_digest": binding.mechanism_semantics_digest,
        "selected_mechanism_axis": binding.proposal.mechanism_axis,
        "selected_candidate_parent_id": exact_parent_candidate_id(
            binding.proposal
        ),
        "lineage_parent_binding": (
            selected_outcome.provenance.get("lineage_parent_binding")
            if isinstance(selected_outcome.provenance, Mapping)
            else None
        ),
        "required_selected_runnable_probability": (
            binding.proposal.utility_features.runnable_probability
        ),
        "mechanism_program": mechanism_projection,
        "comparator_identity": identity.comparator_ref,
        "required_seed_or_control": observation_seed,
        "next_discriminative_test": next_discriminative_test,
        "search_space_execution_binding": search_space_attestation,
        "adapter_id": search_space_attestation["adapter_id"],
        "binding_ref": search_space_attestation["binding_ref"],
        "binding_digest": search_space_attestation["binding_digest"],
        "effective_experiment_digest": effective_identity[
            "effective_experiment_digest"
        ],
        "effective_family_digest": effective_identity[
            "effective_family_digest"
        ],
        "lineage_candidate_id": getattr(
            binding.proposal,
            "compiler_candidate_id",
            binding.proposal.candidate_id,
        ),
        "lineage_program_digest": getattr(
            binding.proposal,
            "mechanism_program_digest",
            sha256_digest(binding.proposal.mechanism_program),
        ),
    }
    route_metadata["followup_confirmation_availability"] = (
        _declare_followup_availability(
            active_search_space_adapter,
            selected_outcome=selected_outcome,
            context=context,
            mechanism_program=binding.proposal.mechanism_program,
        )
    )
    if isinstance(pending_task, Mapping):
        if isinstance(pending_task.get("task_record"), Mapping):
            route_metadata["active_task_record"] = canonical_value(
                dict(pending_task["task_record"])
            )
            route_metadata["active_task_record_digest"] = sha256_digest(
                pending_task["task_record"]
            )
        route_metadata["active_task_id"] = pending_task.get("task_id")
        if binding.proposal.candidate_id in _exact_confirmation_candidate_ids(
            pending_task
        ):
            task_record = pending_task.get("task_record")
            if isinstance(task_record, Mapping):
                route_metadata["comparator_identity"] = task_record.get(
                    "comparator_identity"
                )
                route_metadata["task_operation"] = task_record.get("operation")
                route_metadata["satisfies_task_id"] = pending_task.get(
                    "task_id"
                )
        resolution = pending_task.get("task_resolution")
        if isinstance(resolution, Mapping) and resolution.get("status") == "CLOSED":
            route_metadata["close_task_id"] = pending_task.get("task_id")
            route_metadata["close_task_reason"] = resolution.get("reason")
    if isinstance(dict(policy.acquisition_parameters).get("research_task_record"), Mapping):
        route_metadata["research_task_record"] = canonical_value(
            dict(dict(policy.acquisition_parameters)["research_task_record"])
        )
    if confirmation_seed is not None:
        route_metadata["confirmation_seed"] = confirmation_seed
    if episode is None:
        if exact_verification_task:
            interpretation = interpret_verification_diagnostic(
                closure=closure,
                comparison_identity=identity,
                context=context,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
            )
        else:
            interpretation = interpret_scientific_diagnostic(
                closure=closure,
                comparison_identity=identity,
                context=context,
                producer_outcomes=producer_outcomes,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
            )
    else:
        if exact_verification_task:
            interpretation = interpret_verification_episode(
                episode=episode,
                comparison_identity=identity,
                closure=closure,
                context=context,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
            )
        else:
            interpretation = interpret_typed_research_episode(
                episode=episode,
                comparison_identity=identity,
                closure=closure,
                context=context,
                producer_outcomes=producer_outcomes,
                route_metadata=route_metadata,
                evaluator_projection=event,
                policy=policy,
                memory_writer=memory_writer,
                selected_outcome=selected_outcome,
            )
    meta_research = (
        _run_meta_research(
            context=context,
            interpretation=interpretation,
            inputs=meta_research_inputs,
        )
        if meta_research_inputs is not None and not exact_verification_task
        else None
    )
    return ResearchRoundResult(
        context=context,
        active_profile=active_profile,
        producer_outcomes=producer_outcomes,
        carryover_outcomes=carryover_outcomes,
        resolutions=resolutions,
        deferred_innovation_outcomes=deferred_innovation,
        deferred_search_outcomes=deferred_search,
        search_acquisition=acquisition,
        innovation=innovation,
        selected_outcome=selected_outcome,
        execution_recipe=recipe,
        candidate_run=candidate_run,
        interpretation=interpretation,
            provider_traces=provider_traces,
        meta_research=meta_research,
    )


def activate_promoted_meta_strategy(
    result: ResearchRoundResult,
    *,
    next_profile: SearchExecutableProfileV1,
) -> tuple[ResearchContext, VersionedResearchPolicyV1]:
    """Bind a promoted strategy to the supplied next-campaign executable profile."""

    meta = result.meta_research
    if (
        result.interpretation is None
        or meta is None
        or meta.activated_policy is None
        or meta.activation_receipt is None
    ):
        raise ValueError("round has no promoted Meta strategy to activate")
    if next_profile.campaign_id != meta.activation_receipt.campaign_id:
        raise ValueError("next Search profile is not the Meta activation campaign")
    successor = result.interpretation.successor_context
    if (
        next_profile.protocol_ref != successor.protocol_ref
        or next_profile.protocol_digest != successor.protocol_digest
    ):
        raise ValueError("Meta activation cannot change the frozen protocol")
    meta_memory = {
        "proposal_digest": meta.update_proposal.digest,
        "effect_evaluation_digest": (
            meta.promotion_decision.evaluation_digest
        ),
        "promotion_decision_digest": meta.promotion_decision.digest,
        "activated_policy_digest": meta.activated_policy.digest,
    }
    context = replace(
        successor,
        campaign_id=next_profile.campaign_id,
        active_profile_ref=next_profile.profile_ref,
        active_profile_digest=next_profile.profile_digest,
        policy=meta.activated_policy.to_dict(),
        scientific_memory={
            **successor.scientific_memory,
            "meta_strategy": meta_memory,
        },
    )
    return context, meta.activated_policy


def activate_staged_innovation(
    result: ResearchRoundResult,
) -> tuple[
    SearchExecutableProfileV1,
    ResearchContext,
    SearchCandidate,
    Mapping[str, Any],
]:
    """Activate an admitted primitive only across the next-campaign boundary."""

    innovation = result.innovation
    if (
        innovation is None
        or not innovation.admitted
        or innovation.next_profile is None
        or innovation.registry is None
        or innovation.qualified_execution is None
        or innovation.search_candidate is None
    ):
        raise ValueError("round has no admitted Innovation capability to activate")
    active = activate_next_fresh_search_profile(
        predecessor=result.active_profile,
        next_profile=innovation.next_profile,
        registry=innovation.registry,
        fresh_campaign_id=innovation.fresh_campaign_id,
    )
    successor = result.successor_context
    memory = dict(successor.scientific_memory)
    frontier_global = successor.frontier.get("global")
    frontier_global = (
        frontier_global if isinstance(frontier_global, Mapping) else {}
    )
    last_observation = frontier_global.get("last_observation")
    last_observation = (
        last_observation if isinstance(last_observation, Mapping) else {}
    )
    if last_observation.get("frontier_updated") is True:
        memory["activated_capability"] = {
            "capability": innovation.capability.canonical_dict(),
            "search_candidate": innovation.search_candidate.to_dict(),
            "qualified_execution": innovation.qualified_execution,
        }
    successor = replace(
        successor,
        campaign_id=active.campaign_id,
        active_profile_ref=active.profile_ref,
        active_profile_digest=active.profile_digest,
        scientific_memory=memory,
    )
    return (
        active,
        successor,
        innovation.search_candidate,
        innovation.qualified_execution,
    )


__all__ = [
    "CandidateHandoffFactory",
    "IdeaAcquisitionResult",
    "InitialSearchPoolResult",
    "InnovationLaneResult",
    "InnovationRuntimeInputs",
    "MetaResearchInputs",
    "MetaResearchResult",
    "PreparedResearchRoundV1",
    "ResearchProfileSourceV1",
    "RoundCandidateHandoffV1",
    "RoundAttemptV1",
    "ResearchRoundResult",
    "SearchCandidate",
    "activate_promoted_meta_strategy",
    "activate_staged_innovation",
    "bindings_for_context",
    "normalize_fidelity_utility_state",
    "resolver_environment_for_profile",
    "run_research_round",
    "stage_initial_search_pool",
]
