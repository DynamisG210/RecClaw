"""The minimal four-Producer edge for one Research Line round."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol, TypeAlias

from recclaw_core.experiments.helix_abc_v1.open_spec import (
    OpenSpecProjectionError,
    project_candidate_proposal_v4,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV4,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import OpenResearchSpecV1
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_scientific_profile_ref,
)
from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value

from .bl_icf_realization import (
    CompilerBoundRealizationError,
    MechanismImplementationCompanionV1,
    ProviderMechanismProgramProposalV1,
    compile_bl_icf_candidate_source,
    implementation_spec_for_compiled_program,
    parent_binding_for_mechanism_program,
)
from .interfaces import (
    ProducerOutcome,
    ResearchContext,
    lineage_parent_binding_from_context_view,
    project_provider_context_view,
    research_producer_roles,
)
from .search_space_adapter import (
    ConfirmationResolutionKindV1,
    ConfirmationResolutionV1,
    SearchSpaceAdapter,
)


ProducerResult: TypeAlias = (
    CandidateProposalV4
    | ProviderMechanismProgramProposalV1
    | Mapping[str, Any]
)


class ResearchProducer(Protocol):
    """Injectable external Producer boundary; Provider invocation stays outside."""

    def __call__(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
    ) -> ProducerResult:
        ...


_BINDING_FIELDS = (
    "context_ref",
    "context_digest",
    "protocol_ref",
    "protocol_digest",
    "current_profile_ref",
    "current_profile_digest",
)


_MACHINE_EXECUTION_CONFIG_FIELDS = frozenset(
    {
        "candidate_id",
        "dataset",
        "entrypoint",
        "epochs",
        "evaluator",
        "model",
        "recclaw_trainer_entrypoint",
        "seed",
        "split",
        "timeout",
        "timeout_seconds",
    }
)


def compiler_owned_execution_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a typed BL-ICF proposal to the stable RecBole ABI.

    Provider output is advisory for mechanism hyperparameters.  Stable runtime
    fields are owned by the campaign/compiler and are therefore discarded here
    rather than turning a scientifically equivalent proposal into a projection
    failure.  The authoritative values are injected by the execution scaffold.
    """

    if not isinstance(value, Mapping):
        raise OpenSpecProjectionError("implementation_research must be a mapping")
    if set(value) != {"base_model_config", "mechanism_config"}:
        raise OpenSpecProjectionError(
            "implementation_research must contain base_model_config and mechanism_config"
        )
    base_model_config = value["base_model_config"]
    if base_model_config not in {"BPR", "LightGCN"}:
        raise OpenSpecProjectionError(
            "implementation_research.base_model_config is not a frozen family"
        )
    mechanism_config = value["mechanism_config"]
    if not isinstance(mechanism_config, Mapping):
        raise OpenSpecProjectionError(
            "implementation_research.mechanism_config must be a mapping"
        )
    runtime_config = {
        str(key): canonical_value(item)
        for key, item in mechanism_config.items()
        if key not in _MACHINE_EXECUTION_CONFIG_FIELDS
    }
    runtime_config["recclaw_trainer_entrypoint"] = (
        "recclaw_ext.trainer:FreshCandidateTrainer"
    )
    return canonical_value(
        {
            "capability_family": "STRICT_BL_ICF_COMPILED_MECHANISM",
            "model": "FreshCandidateModel",
            "base_model_config": base_model_config,
            "config": runtime_config,
        }
    )


def compiler_owned_generic_execution_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind stable runtime fields without replacing native search-space identity."""

    required = {"capability_family", "model", "base_model_config", "config"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise OpenSpecProjectionError(
            "generic execution_contract must contain exactly capability_family, "
            "model, base_model_config, and config"
        )
    capability_family = value["capability_family"]
    if (
        not isinstance(capability_family, str)
        or not capability_family
        or capability_family != capability_family.strip()
    ):
        raise OpenSpecProjectionError(
            "generic execution_contract.capability_family must be a trimmed non-empty string"
        )
    base_model_config = value["base_model_config"]
    if (
        not isinstance(base_model_config, str)
        or not base_model_config
        or not base_model_config.isidentifier()
    ):
        raise OpenSpecProjectionError(
            "generic execution_contract.base_model_config must be a bare model-config identifier"
        )
    mechanism_config = value["config"]
    if not isinstance(mechanism_config, Mapping):
        raise OpenSpecProjectionError(
            "generic execution_contract.config must be a mapping"
        )
    runtime_config = {
        str(key): canonical_value(item)
        for key, item in mechanism_config.items()
        if key not in _MACHINE_EXECUTION_CONFIG_FIELDS
    }
    runtime_config["recclaw_trainer_entrypoint"] = (
        "recclaw_ext.trainer:FreshCandidateTrainer"
    )
    return canonical_value(
        {
            "capability_family": capability_family,
            "model": "FreshCandidateModel",
            "base_model_config": base_model_config,
            "config": runtime_config,
        }
    )


def _validate_bindings(
    context: ResearchContext,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping")

    supplied = dict(bindings)
    expected = {
        "context_ref": context.context_ref,
        "context_digest": context.digest,
        "protocol_ref": context.protocol_ref,
        "protocol_digest": context.protocol_digest,
        "current_profile_ref": context.active_profile_ref,
        "current_profile_digest": context.active_profile_digest,
    }
    missing = [field for field in _BINDING_FIELDS if field not in supplied]
    if missing:
        raise ValueError(
            "bindings missing required OpenSpec identity fields: "
            + ", ".join(missing)
        )
    mismatched = [
        field for field in _BINDING_FIELDS if supplied[field] != expected[field]
    ]
    if mismatched:
        raise ValueError(
            "bindings must match ResearchContext for: "
            + ", ".join(mismatched)
        )
    return supplied


def _project_result(
    producer_role: str,
    result: ProducerResult,
    *,
    bindings: Mapping[str, Any],
    frozen_profile_ref: Mapping[str, Any],
) -> tuple[
    OpenResearchSpecV1,
    dict[str, Any],
    CandidateProposalV4 | None,
    Mapping[str, Any] | None,
    MechanismImplementationCompanionV1 | None,
]:
    if isinstance(result, CandidateProposalV4):
        if result.producer_role != producer_role:
            raise OpenSpecProjectionError(
                "CandidateProposalV4 producer_role does not match assigned role"
            )
        spec, facts = project_candidate_proposal_v4(
            result,
            bindings=bindings,
        )
        return spec, facts, result, None, None

    if isinstance(result, ProviderMechanismProgramProposalV1):
        if result.producer_role != producer_role:
            raise OpenSpecProjectionError(
                "strict MechanismProgram producer_role does not match assigned role"
            )
        try:
            compiler = compile_bl_icf_candidate_source(
                open_spec=None,
                frozen_profile_ref=frozen_profile_ref,
                mechanism_program=result.mechanism_program,
                parent_binding=result.parent_binding,
            )
            companion = MechanismImplementationCompanionV1(
                protocol_ref=str(bindings["protocol_ref"]),
                protocol_digest=str(bindings["protocol_digest"]),
                context_ref=str(bindings["context_ref"]),
                context_digest=str(bindings["context_digest"]),
                current_profile_ref=str(bindings["current_profile_ref"]),
                current_profile_digest=str(bindings["current_profile_digest"]),
                producer_role=producer_role,
                compatibility_requirements=tuple(
                    bindings["compatibility_requirements"]
                ),
                execution_contract=compiler_owned_execution_contract(
                    result.implementation_research
                ),
            )
            spec = implementation_spec_for_compiled_program(
                compiler,
                companion=companion,
            )
        except (CompilerBoundRealizationError, KeyError, TypeError, ValueError) as error:
            raise OpenSpecProjectionError(str(error)) from error
        facts = dict(result.resolution_facts)
        expected_facts = {
            "requested_current_semantics_digest",
            "capability_diff",
            "high_change_dimensions",
            "required_dependencies",
            "required_budget",
        }
        if set(facts) != expected_facts:
            raise OpenSpecProjectionError(
                "strict MechanismProgram resolution_facts has field drift"
            )
        if facts["requested_current_semantics_digest"] is not None:
            raise OpenSpecProjectionError(
                "strict MechanismProgram Innovation source must be NOT_EXPRESSIBLE"
            )
        if not facts["capability_diff"] or not facts["high_change_dimensions"]:
            raise OpenSpecProjectionError(
                "strict MechanismProgram Innovation source lacks capability facts"
            )
        return spec, facts, None, compiler.mechanism_program, companion

    if isinstance(result, Mapping):
        if result.get("producer_role") != producer_role:
            raise OpenSpecProjectionError(
                "open Producer draft producer_role does not match assigned role"
            )
        spec, facts = project_open_producer_draft(
            result,
            bindings=bindings,
            strict_resolution_contract=True,
        )
        if isinstance(spec.execution_contract, Mapping):
            spec = replace(
                spec,
                execution_contract=compiler_owned_generic_execution_contract(
                    spec.execution_contract
                ),
            )
        return spec, facts, None, None, None

    raise OpenSpecProjectionError(
        "Producer must return CandidateProposalV4 or an open Producer draft mapping"
    )


def _failure(
    context: ResearchContext,
    producer_role: str,
    *,
    failure_code: str,
    error: Exception,
) -> ProducerOutcome:
    return ProducerOutcome(
        producer_role=producer_role,
        context_ref=context.context_ref,
        context_digest=context.digest,
        spec=None,
        resolution_facts={
            "failure_code": failure_code,
            "producer_role": producer_role,
        },
        provenance={
            "producer_role": producer_role,
            "context_ref": context.context_ref,
            "context_digest": context.digest,
            "status": "CALL_OR_PROJECTION_FAILURE",
        },
        failure_code=failure_code,
        failure_detail=f"{type(error).__name__}: {error}",
    )


_ACTIVE_TASK_DIRECTIVE_UNSET = object()


def _frontier_portfolio_projection(
    outcomes: list[ProducerOutcome],
) -> tuple[dict[str, Any], ...]:
    """Expose successful peer ideas to the high-innovation actor, compactly."""

    portfolio: list[dict[str, Any]] = []
    for outcome in outcomes:
        spec = outcome.spec
        if spec is None:
            continue
        program = outcome.source_mechanism_program
        if program is None and outcome.source_proposal is not None:
            program = outcome.source_proposal.mechanism_program
        payload = (
            program.get("program_payload")
            if isinstance(program, Mapping)
            else None
        )
        portfolio.append(
            canonical_value(
                {
                    "producer_role": outcome.producer_role,
                    "spec_digest": spec.digest,
                    "hypothesis": spec.hypothesis,
                    "mechanism_change": spec.mechanism_change,
                    "competing_explanation": spec.competing_explanation,
                    "expected_evidence": spec.expected_evidence,
                    "falsifier": spec.falsifier,
                    **(
                        {
                            "construction_mode": payload.get(
                                "construction_mode"
                            ),
                            "changed_slots": payload.get("changed_slots", ()),
                            "estimated_cost": payload.get("estimated_cost"),
                            "expected_effects": payload.get(
                                "expected_effects"
                            ),
                        }
                        if isinstance(payload, Mapping)
                        else {}
                    ),
                }
            )
        )
    return tuple(portfolio)


def _expected_parent_binding_for_result(
    context_view: Mapping[str, Any],
) -> dict[str, str] | tuple[dict[str, str], ...] | None:
    provider_context_view = project_provider_context_view(context_view)
    provider_state = provider_context_view.get("state")
    provider_state = provider_state if isinstance(provider_state, Mapping) else {}
    active_task = provider_state.get("task")
    if (
        isinstance(active_task, Mapping)
        and active_task.get("execution_eligible_this_round") is True
        and active_task.get("binding_requirement")
        == "EXACT_EFFECTIVE_IDENTITY"
    ):
        target_program = active_task.get("target_mechanism_program")
        program_payload = (
            target_program.get("program_payload")
            if isinstance(target_program, Mapping)
            else None
        )
        if (
            isinstance(program_payload, Mapping)
            and isinstance(program_payload.get("parent_refs"), (tuple, list))
        ):
            return parent_binding_for_mechanism_program(target_program)
    lineage_parent = lineage_parent_binding_from_context_view(context_view)
    if lineage_parent is not None:
        return lineage_parent
    frozen_parent = provider_state.get("frozen_parent_binding")
    if isinstance(frozen_parent, Mapping):
        return canonical_value(dict(frozen_parent))
    return None


def _with_construction_parent(
    outcome: ProducerOutcome,
    options: tuple[Mapping[str, Any], ...],
    default_binding: Mapping[str, Any] | None,
) -> ProducerOutcome:
    if not options:
        return outcome
    binding = outcome.provenance.get("lineage_parent_binding") or default_binding
    matches = [option for option in options if isinstance(binding, Mapping) and all(
        option[key] == binding.get(key) for key in ("candidate_id", "program_digest")
    )]
    if len(matches) != 1:
        raise OpenSpecProjectionError("selected construction parent is not available")
    option = matches[0]
    return replace(outcome, provenance={
        **dict(outcome.provenance),
        "lineage_parent_binding": {key: option[key] for key in ("candidate_id", "program_digest")},
        "lineage_parent_visibility": True,
        "construction_parent_option": canonical_value(dict(option)),
    })


def produce_research_specs(
    context: ResearchContext,
    producer: ResearchProducer,
    bindings: Mapping[str, Any],
    *,
    frozen_profile_ref: Mapping[str, Any] | None = None,
    active_task_directive: Mapping[str, Any] | None | object = (
        _ACTIVE_TASK_DIRECTIVE_UNSET
    ),
    lineage_parent_mechanism_program: Mapping[str, Any] | None = None,
    construction_parent_options: tuple[Mapping[str, Any], ...] = (),
    latest_completed_execution: Mapping[str, Any] | None = None,
    research_window_budget: Mapping[str, Any] | None = None,
    search_space_adapter: SearchSpaceAdapter | None = None,
    completed_outcomes: tuple[ProducerOutcome, ...] = (),
) -> tuple[ProducerOutcome, ...]:
    """Complete the portfolio, retaining stages already finished in this context."""

    if not isinstance(context, ResearchContext):
        raise TypeError("context must be ResearchContext")
    normalized_bindings = _validate_bindings(context, bindings)
    normalized_profile_ref = canonical_value(
        dict(
            campaign_scientific_profile_ref()
            if frozen_profile_ref is None
            else frozen_profile_ref
        )
    )

    scheduled_roles = research_producer_roles(context.budget)
    completed = {outcome.producer_role: outcome for outcome in completed_outcomes}
    if len(completed) != len(completed_outcomes) or any(
        role not in scheduled_roles
        or outcome.context_ref != context.context_ref
        or outcome.context_digest != context.digest
        for role, outcome in completed.items()
    ):
        raise ValueError("completed Producer stages belong to a different context")
    outcomes: list[ProducerOutcome] = []
    for producer_role in scheduled_roles:
        if producer_role in completed:
            outcomes.append(completed[producer_role])
            continue
        context_view = context.producer_view(producer_role)
        if research_window_budget is not None:
            context_view = {**context_view, "budget": {
                **context_view["budget"], "research_window": research_window_budget,
            }}
        if latest_completed_execution is not None:
            context_view = {**context_view, "latest_completed_execution": latest_completed_execution}
        if producer_role == "frontier_architect" and len(scheduled_roles) > 1:
            # The high-innovation call sees the three already projected peer
            # ideas but remains an independent executable opportunity rather
            # than replacing or selecting the ordinary search proposals.
            context_view = {
                **context_view,
                "research_portfolio": _frontier_portfolio_projection(
                    outcomes
                ),
            }
        lineage_parent_binding = (
            {
                "candidate_id": lineage_parent_mechanism_program.get("candidate_id"),
                "program_digest": lineage_parent_mechanism_program.get("program_digest"),
            }
            if isinstance(lineage_parent_mechanism_program, Mapping)
            else None
        )
        lineage_parent_visible = bool(
            isinstance(lineage_parent_binding, Mapping)
            and isinstance(lineage_parent_binding.get("candidate_id"), str)
            and lineage_parent_binding.get("candidate_id")
            and isinstance(lineage_parent_binding.get("program_digest"), str)
        )
        if lineage_parent_visible:
            context_view = {
                **context_view,
                "lineage_parent_mechanism_program": canonical_value(
                    dict(lineage_parent_mechanism_program)
                ),
            }
        if active_task_directive is not _ACTIVE_TASK_DIRECTIVE_UNSET:
            # This round-local directive does not mutate ResearchContext
            # identity. An explicit None suppresses an ineligible durable
            # queue head; omitting the argument preserves the legacy view.
            context_view = {
                **context_view,
                "active_task_directive": (
                    canonical_value(dict(active_task_directive))
                    if isinstance(active_task_directive, Mapping)
                    else None
                ),
            }
        expected_parent_binding = _expected_parent_binding_for_result(context_view)
        options = construction_parent_options
        active_task = (
            project_provider_context_view(context_view).get("state", {}).get("task")
            if options else None
        )
        if options and isinstance(active_task, Mapping) and (
            active_task.get("execution_eligible_this_round") is True
            and active_task.get("binding_requirement") == "EXACT_EFFECTIVE_IDENTITY"
        ):
            required = (expected_parent_binding,) if isinstance(expected_parent_binding, Mapping) else expected_parent_binding or ()
            options = tuple(option for option in options if any(all(
                option[key] == binding.get(key) for key in ("candidate_id", "program_digest")
            ) for binding in required))
            if required and not options:
                outcomes.append(_failure(context, producer_role,
                    failure_code="OPEN_SPEC_PROJECTION_FAILED",
                    error=OpenSpecProjectionError("exact task construction parent is unavailable")))
                continue
        if options:
            context_view = {**context_view, "construction_parent_options": options}
        parent_binding_required = expected_parent_binding is not None
        try:
            result = producer(producer_role, context_view)
        except Exception as error:
            typed_failure_code = getattr(error, "failure_code", None)
            outcomes.append(
                _failure(
                    context,
                    producer_role,
                    failure_code=(
                        typed_failure_code
                        if isinstance(typed_failure_code, str)
                        and typed_failure_code
                        else "PRODUCER_CALL_FAILED"
                    ),
                    error=error,
                )
            )
            continue

        try:
            if isinstance(result, ProviderMechanismProgramProposalV1):
                if options:
                    matches = [option for option in options if isinstance(result.parent_binding, Mapping)
                               and all(option[key] == result.parent_binding.get(key)
                                       for key in ("candidate_id", "program_digest"))]
                    if len(matches) != 1:
                        raise OpenSpecProjectionError("selected construction parent is not available")
                    selected_parent = matches[0]
                    expected_parent_binding = result.parent_binding
                elif (
                    result.parent_binding != expected_parent_binding
                    and (
                        parent_binding_required
                        or result.parent_binding is not None
                    )
                ):
                    raise OpenSpecProjectionError(
                        "strict MechanismProgram parent binding differs from "
                        "the activated lineage parent in Research Context"
                    )
                else:
                    selected_parent = None
            else:
                selected_parent = None
            if search_space_adapter is not None:
                projected = search_space_adapter.resolve_confirmation(
                    "DISCOVERY",
                    result,
                    {
                        "phase": "PROJECT_PROVIDER_PROPOSAL",
                        "producer_role": producer_role,
                        "research_context": context,
                        "producer_bindings": normalized_bindings,
                        "frozen_profile_ref": normalized_profile_ref,
                        "lineage_parent_mechanism_program": (
                            selected_parent if selected_parent is not None else (
                                lineage_parent_mechanism_program if lineage_parent_visible else None
                            )
                        ),
                        **({"construction_parent_options": options} if options else {}),
                    },
                )
                if (
                    not isinstance(projected, ConfirmationResolutionV1)
                    or projected.kind
                    is not ConfirmationResolutionKindV1.EXACT_BINDING
                    or not isinstance(projected.binding, Mapping)
                    or not isinstance(
                        projected.binding.get("producer_outcome"), ProducerOutcome
                    )
                ):
                    reason = (
                        projected.reason
                        if isinstance(projected, ConfirmationResolutionV1)
                        else None
                    )
                    raise OpenSpecProjectionError(
                        reason or "adapter did not project the Provider proposal"
                    )
                outcome = projected.binding["producer_outcome"]
                if (
                    outcome.producer_role != producer_role
                    or outcome.context_ref != context.context_ref
                    or outcome.context_digest != context.digest
                ):
                    raise OpenSpecProjectionError(
                        "adapter Provider projection differs from assigned Context"
                    )
                outcomes.append(
                    _with_construction_parent(outcome, options, expected_parent_binding)
                    if options else
                    replace(
                        outcome,
                        provenance={
                            **dict(outcome.provenance),
                            "lineage_parent_visibility": True,
                            "lineage_parent_binding": canonical_value(
                                lineage_parent_binding
                            ),
                        },
                    )
                    if lineage_parent_visible
                    else outcome
                )
                continue
            (
                spec,
                resolution_facts,
                source_proposal,
                source_mechanism_program,
                implementation_companion,
            ) = _project_result(
                producer_role,
                result,
                bindings=normalized_bindings,
                frozen_profile_ref=normalized_profile_ref,
            )
        except OpenSpecProjectionError as error:
            outcomes.append(
                _failure(
                    context,
                    producer_role,
                    failure_code="OPEN_SPEC_PROJECTION_FAILED",
                    error=error,
                )
            )
            continue

        provenance = {
            "producer_role": producer_role,
            "context_ref": context.context_ref,
            "context_digest": context.digest,
            "spec_id": spec.spec_id,
            "spec_digest": spec.digest,
            "source_proposal_digest": (
                source_proposal.digest if source_proposal is not None else None
            ),
            "source_mechanism_program_digest": (
                compile_bl_icf_candidate_source(
                    open_spec=None,
                    frozen_profile_ref=normalized_profile_ref,
                    mechanism_program=source_mechanism_program,
                    parent_binding=parent_binding_for_mechanism_program(
                        source_mechanism_program
                    ),
                ).mechanism_program_digest
                if source_mechanism_program is not None
                else None
            ),
            "status": "PRODUCED",
            "lineage_parent_visibility": lineage_parent_visible,
            "lineage_parent_binding": (
                canonical_value(expected_parent_binding) if options else (
                    canonical_value(lineage_parent_binding) if lineage_parent_visible else None
                )
            ),
        }
        outcomes.append(_with_construction_parent(
            ProducerOutcome(
                producer_role=producer_role,
                context_ref=context.context_ref,
                context_digest=context.digest,
                spec=spec,
                resolution_facts=resolution_facts,
                source_proposal=source_proposal,
                source_mechanism_program=source_mechanism_program,
                implementation_companion=implementation_companion,
                provenance=provenance,
            ), options, expected_parent_binding,
        ))

    if context.budget.get("research_mode") == "director_sequential":
        # Persist the real schedule with successful and failed outcomes alike.
        # Existing checkpoints retain their original four-role interpretation.
        return tuple(replace(outcome, provenance={
            **dict(outcome.provenance), "research_mode": "director_sequential",
        }) for outcome in outcomes)
    return tuple(outcomes)


__all__ = [
    "ResearchProducer",
    "compiler_owned_execution_contract",
    "compiler_owned_generic_execution_contract",
    "produce_research_specs",
]
