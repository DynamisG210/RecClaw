"""Deterministic round-one Search bootstrap from the active BL-ICF profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    ExecutableMechanismV1,
    executable_mechanisms,
    execution_recipe_for_program,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV4,
    DiscoveryCreditV1,
    DiscriminativeExperimentPlanV1,
    ProposalIntentV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.research_science import (
    DeterministicRouterFeatureBuilderV1,
    LineageIndexV1,
    matched_control_plan,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchExecutableEntryV1,
    SearchExecutableProfileV1,
)
from recclaw_core.mechanism_space.canonical import deep_thaw

from .interfaces import ResearchContext


_AXIS_ALIASES = {
    "fusion": "message_transform",
    "negative_sampling": "sampling",
    "regularization": "geometry",
}
_AXIS_FIELDS = (
    "mechanism_axis",
    "axis",
    "unresolved_axes",
    "uncovered_axes",
    "underexplored_axes",
    "axis_gaps",
    "mechanism_axis_gaps",
)


def _axes(value: Any) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, (tuple, list)):
        return ()
    result: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            continue
        normalized = _AXIS_ALIASES.get(item.strip(), item.strip())
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _mapping_axes(value: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for field_name in _AXIS_FIELDS:
        for axis in _axes(value.get(field_name)):
            if axis not in result:
                result.append(axis)
    return tuple(result)


def _context_axis_signals(
    context: ResearchContext,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    question_axes: list[str] = []
    for question in context.unresolved_questions:
        if isinstance(question, Mapping):
            for axis in _mapping_axes(question):
                if axis not in question_axes:
                    question_axes.append(axis)

    frontier_axes = _mapping_axes(context.frontier)
    memory_axes = _mapping_axes(context.scientific_memory)
    return tuple(question_axes), frontier_axes, memory_axes


def _axis_order(
    *,
    policy: VersionedResearchPolicyV1,
    question_axes: Sequence[str],
    frontier_axes: Sequence[str],
    memory_axes: Sequence[str],
    active_axes: Sequence[str],
) -> tuple[str, ...]:
    result: list[str] = []
    for group in (
        _axes(policy.mechanism_axis_targeting),
        question_axes,
        frontier_axes,
        memory_axes,
        active_axes,
    ):
        for axis in group:
            if axis not in result:
                result.append(axis)
    return tuple(result)


def _active_catalog(
    profile: SearchExecutableProfileV1,
) -> tuple[tuple[SearchExecutableEntryV1, ExecutableMechanismV1], ...]:
    by_semantics = {
        mechanism.mechanism_semantics_digest: mechanism
        for mechanism in executable_mechanisms()
    }
    active: list[tuple[SearchExecutableEntryV1, ExecutableMechanismV1]] = []
    for entry in profile.entries:
        mechanism = by_semantics.get(entry.semantic_identity_digest)
        if mechanism is None:
            raise ValueError(
                "active Search profile entry is not an exact BL-ICF catalog mechanism"
            )
        recipe = execution_recipe_for_program(
            deep_thaw(mechanism.mechanism_program)
        )
        if (
            recipe["mechanism_id"] != mechanism.mechanism_id
            or recipe["mechanism_semantics_digest"]
            != entry.semantic_identity_digest
            or recipe["entrypoint"] != entry.executable_entrypoint
        ):
            raise ValueError("active Search profile entry has catalog identity drift")
        active.append((entry, mechanism))
    if not active:
        raise ValueError("active Search profile has no exact BL-ICF mechanisms")
    return tuple(active)


def _diagnostic_features(
    *,
    axis: str,
    policy_axes: Sequence[str],
    question_axes: Sequence[str],
    frontier_axes: Sequence[str],
    memory_axes: Sequence[str],
    estimated_cost: float,
) -> SearchUtilityFeaturesV1:
    """Create pre-outcome structural priors, not observed research evidence."""

    targeted = axis in policy_axes
    question_match = axis in question_axes
    frontier_match = axis in frontier_axes
    memory_match = axis in memory_axes
    return SearchUtilityFeaturesV1(
        runnable_probability=1.0,
        useful_signal=min(1.0, 0.5 + 0.1 * targeted + 0.1 * question_match),
        frontier_potential=min(
            1.0,
            0.5 + 0.1 * targeted + 0.1 * frontier_match,
        ),
        information_gain=min(
            1.0,
            0.5 + 0.1 * question_match + 0.1 * memory_match,
        ),
        cost=estimated_cost,
        blocker_risk=0.0,
    )


def _candidate_sort_key(
    item: tuple[SearchExecutableEntryV1, ExecutableMechanismV1],
) -> tuple[int, str]:
    _entry, mechanism = item
    return (
        0 if mechanism.base_mechanism_id != "BPR_MF" else 1,
        mechanism.mechanism_id,
    )


def _select_catalog_entries(
    active: Sequence[tuple[SearchExecutableEntryV1, ExecutableMechanismV1]],
    *,
    axis_order: Sequence[str],
    max_proposals: int,
) -> tuple[tuple[SearchExecutableEntryV1, ExecutableMechanismV1], ...]:
    by_axis: dict[
        str, list[tuple[SearchExecutableEntryV1, ExecutableMechanismV1]]
    ] = {}
    for item in active:
        by_axis.setdefault(item[1].mechanism_axis, []).append(item)

    selected: list[tuple[SearchExecutableEntryV1, ExecutableMechanismV1]] = []
    selected_ids: set[str] = set()
    for axis in axis_order:
        choices = by_axis.get(axis, ())
        if not choices:
            continue
        item = min(choices, key=_candidate_sort_key)
        if item[1].mechanism_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item[1].mechanism_id)
        if len(selected) == max_proposals:
            break

    if selected and max_proposals > 1 and not any(
        item[1].base_mechanism_id != "BPR_MF" for item in selected
    ):
        selected_axes = {item[1].mechanism_axis for item in selected}
        fallback = next(
            (
                item
                for item in active
                if item[1].base_mechanism_id != "BPR_MF"
                and item[1].mechanism_id not in selected_ids
                and item[1].mechanism_axis not in selected_axes
            ),
            None,
        )
        if fallback is None:
            fallback = next(
                (
                    item
                    for item in active
                    if item[1].base_mechanism_id != "BPR_MF"
                    and item[1].mechanism_id not in selected_ids
                ),
                None,
            )
        if fallback is not None:
            if len(selected) < max_proposals:
                selected.append(fallback)
            else:
                selected[-1] = fallback
    return tuple(selected)


def bootstrap_search_pool(
    context: ResearchContext,
    active_profile: SearchExecutableProfileV1,
    policy: VersionedResearchPolicyV1,
    *,
    max_proposals: int = 4,
) -> tuple[CandidateProposalV4, ...]:
    """Build deterministic round-one typed proposals from the active profile."""

    if not isinstance(context, ResearchContext):
        raise TypeError("context must be ResearchContext")
    if not isinstance(active_profile, SearchExecutableProfileV1):
        raise TypeError("active_profile must be SearchExecutableProfileV1")
    if not isinstance(policy, VersionedResearchPolicyV1):
        raise TypeError("policy must be VersionedResearchPolicyV1")
    if isinstance(max_proposals, bool) or not isinstance(max_proposals, int):
        raise TypeError("max_proposals must be an integer")
    if max_proposals < 1:
        raise ValueError("max_proposals must be positive")
    if context.round_index != 1:
        raise ValueError("Search bootstrap is only defined for round 1")
    if (
        context.campaign_id != active_profile.campaign_id
        or context.active_profile_ref != active_profile.profile_ref
        or context.active_profile_digest != active_profile.profile_digest
        or context.protocol_ref != active_profile.protocol_ref
        or context.protocol_digest != active_profile.protocol_digest
    ):
        raise ValueError("active Search profile is not bound to the Research Context")
    if canonical_value(context.policy) != canonical_value(policy.to_dict()):
        raise ValueError("Research Context policy differs from bootstrap policy")

    active = _active_catalog(active_profile)
    active_axes = tuple(dict.fromkeys(item[1].mechanism_axis for item in active))
    question_axes, frontier_axes, memory_axes = _context_axis_signals(context)
    policy_axes = _axes(policy.mechanism_axis_targeting)
    axis_order = _axis_order(
        policy=policy,
        question_axes=question_axes,
        frontier_axes=frontier_axes,
        memory_axes=memory_axes,
        active_axes=active_axes,
    )
    selected = _select_catalog_entries(
        active,
        axis_order=axis_order,
        max_proposals=max_proposals,
    )
    if not selected:
        return ()

    active_by_id = {
        mechanism.mechanism_id: mechanism for _, mechanism in active
    }
    lineage = LineageIndexV1()
    metric_name = str(context.frozen_goal.get("metric", "the frozen metric"))
    question_hint = next(
        (
            str(question["question"])
            for question in context.unresolved_questions
            if isinstance(question, Mapping)
            and isinstance(question.get("question"), str)
            and question["question"].strip()
        ),
        "the unresolved mechanism question",
    )
    proposals: list[CandidateProposalV4] = []
    for index, (entry, mechanism) in enumerate(selected):
        role = DISCOVERY_PRODUCERS[index % len(DISCOVERY_PRODUCERS)]
        candidate_id = "cand-bootstrap-" + sha256_digest(
            {
                "campaign_id": context.campaign_id,
                "round_index": context.round_index,
                "profile_digest": active_profile.profile_digest,
                "policy_digest": policy.digest,
                "mechanism_semantics_digest": entry.semantic_identity_digest,
            }
        )[:24]
        hypothesis = (
            f"Under {metric_name}, test {mechanism.mechanism_id} on "
            f"the {mechanism.mechanism_axis} axis for: {question_hint}."
        )
        competing = (
            f"The matched {mechanism.base_mechanism_id} control, rather than "
            f"the {mechanism.mechanism_axis} change, explains the frontier signal."
        )
        predicted = (
            f"The {mechanism.mechanism_axis} change produces a distinct "
            "same-protocol signature."
        )
        failure = "The matched control reproduces the predicted signature."
        root_mechanism = active_by_id.get(mechanism.base_mechanism_id)
        if root_mechanism is None:
            raise ValueError(
                "active Search profile lacks the exact catalog parent control"
            )
        control = matched_control_plan(
            lineage=lineage,
            primary_candidate_id=candidate_id,
            parent_candidate_id=None,
            changed_axis=mechanism.mechanism_axis,
            mechanism_hypothesis=hypothesis,
            protocol_digest=active_profile.protocol_digest,
            queued_comparator_candidate_id=root_mechanism.candidate_id,
            queued_comparator_program_digest=root_mechanism.mechanism_program_digest,
        )
        estimated_cost = min(1.0, 0.15 + 0.05 * len(mechanism.operator_ids))
        diagnostic = _diagnostic_features(
            axis=mechanism.mechanism_axis,
            policy_axes=policy_axes,
            question_axes=question_axes,
            frontier_axes=frontier_axes,
            memory_axes=memory_axes,
            estimated_cost=estimated_cost,
        )
        utility, evidence = DeterministicRouterFeatureBuilderV1().build(
            compile_valid=True,
            handler_available=bool(entry.executable_entrypoint),
            materializer_available=True,
            mechanism_id=mechanism.mechanism_id,
            mechanism_depth=0,
            estimated_cost=estimated_cost,
            semantics_digest=entry.semantic_identity_digest,
            parent_available=True,
            lineage=lineage,
            llm_diagnostic=diagnostic,
        )
        discriminative = (
            DiscriminativeExperimentPlanV1(
                competing_hypotheses=(hypothesis, competing),
                predicted_outcome_signature=predicted,
                primary_candidate=candidate_id,
                matched_control_plan=control,
                falsifier=failure,
                next_decision_rule=(
                    "retain the mechanism only if the exact matched control "
                    "does not reproduce the signature"
                ),
            )
            if role == "falsification_designer"
            else None
        )
        proposals.append(
            CandidateProposalV4(
                candidate_id=candidate_id,
                producer_id=f"producer:bootstrap:{role}",
                producer_role=role,
                proposal_intent=(
                    ProposalIntentV1.FALSIFICATION
                    if role == "falsification_designer"
                    else ProposalIntentV1.DISCOVERY
                ),
                discovery_credit=DiscoveryCreditV1.DISCOVERY,
                mechanism_id=mechanism.mechanism_id,
                mechanism_axis=mechanism.mechanism_axis,
                mechanism_program=deep_thaw(mechanism.mechanism_program),
                candidate_label=(
                    f"Round-one bootstrap: {mechanism.mechanism_id}"
                ),
                mechanism_hypothesis=hypothesis,
                competing_hypothesis=competing,
                predicted_outcome_signature=predicted,
                failure_mode=failure,
                utility_features=utility,
                feature_evidence=evidence,
                matched_control_plan=control,
                discriminative_plan=discriminative,
                parent_candidate_id=None,
                assigned_before_call=True,
                post_hoc_relabel=False,
            )
        )
    return tuple(proposals)


__all__ = ["bootstrap_search_pool"]
