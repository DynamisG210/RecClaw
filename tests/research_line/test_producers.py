from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    frozen_search_resolver_environment,
    resolve_capability,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityResolutionResultV1,
    OpenResearchSpecV1,
)
from recclaw_core.research_line.interfaces import ResearchContext
import recclaw_core.research_line.producers as producers_module
from recclaw_core.research_line.producers import produce_research_specs


def _digest(label: str) -> str:
    return sha256_digest({"label": label})


def _context(
    *,
    policy: dict[str, Any] | None = None,
    scientific_memory: dict[str, Any] | None = None,
) -> ResearchContext:
    identity = frozen_search_bindings(
        context_ref="context:seed",
        context_digest=_digest("context-seed"),
    )
    return ResearchContext(
        campaign_id="campaign-producers",
        round_index=1,
        knowledge_base={"mechanisms": ["graph", "self_supervision"]},
        frozen_goal={"metric": "NDCG@10", "direction": "maximize"},
        frontier={"best_value": 0.12, "unresolved_axes": ["propagation"]},
        scientific_memory=scientific_memory or {
            "head": "memory:round-1",
            "by_role": {
                role: {"retrieval_key": f"{role}:round-1"}
                for role in DISCOVERY_PRODUCERS
            },
        },
        unresolved_questions=(
            {"question": "Which mechanism explains the frontier gap?"},
        ),
        policy=policy or {
            "producer_token_allocation": {
                role: 0.25 for role in DISCOVERY_PRODUCERS
            },
            "mechanism_axis_targeting": ("propagation", "objective"),
            "memory_retrieval_policy": "ROLE_SCOPED_PRIOR_ROUND_V1",
        },
        budget={"proposal_calls": 4, "token_fraction": 1.0},
        active_profile_ref=identity["current_profile_ref"],
        active_profile_digest=identity["current_profile_digest"],
        protocol_ref=identity["protocol_ref"],
        protocol_digest=identity["protocol_digest"],
    )


def _bindings(context: ResearchContext) -> dict[str, Any]:
    return frozen_search_bindings(
        context_ref=context.context_ref,
        context_digest=context.digest,
    )


def _environment() -> dict[str, Any]:
    return frozen_search_resolver_environment()


def _draft(
    role: str,
    *,
    semantics_digest: str,
    marker: str,
) -> dict[str, Any]:
    return {
        "producer_role": role,
        "hypothesis": f"{role} hypothesis {marker}",
        "mechanism_change": f"Test the {role} mechanism wedge {marker}.",
        "competing_explanation": "The observed signal is optimization noise.",
        "matched_control_requirement": "Run the matched incumbent under the same protocol.",
        "implementation_requirements": ("candidate-local package",),
        "expected_evidence": ("full-sort NDCG",),
        "falsifier": "The predicted signature is absent.",
        "compatibility_requirements": ("general collaborative filtering",),
        "high_change_justification": "The proposal is a testable research wedge.",
        "current_profile_expressibility_claim": "EXPRESSIBLE",
        "resolution_facts": {
            "requested_current_semantics_digest": semantics_digest,
            "capability_diff": (),
            "high_change_dimensions": (),
            "required_dependencies": (),
            "required_budget": {},
        },
    }


def _captured_inputs(
    context: ResearchContext,
) -> tuple[list[tuple[str, dict[str, Any]]], tuple[Any, ...]]:
    semantics_digest = _environment()["current_capabilities"][0][
        "semantics_digest"
    ]
    calls: list[tuple[str, dict[str, Any]]] = []

    def producer(role: str, view: dict[str, Any]) -> dict[str, Any]:
        calls.append((role, view))
        return _draft(
            role,
            semantics_digest=semantics_digest,
            marker=view["memory"]["retrieval_key"],
        )

    outcomes = produce_research_specs(context, producer, _bindings(context))
    return calls, outcomes


def test_calls_exactly_four_roles_with_shared_context_and_four_outcomes() -> None:
    context = _context()
    calls, outcomes = _captured_inputs(context)

    assert [role for role, _view in calls] == list(DISCOVERY_PRODUCERS)
    assert len(calls) == len(outcomes) == 4
    assert [outcome.producer_role for outcome in outcomes] == list(
        DISCOVERY_PRODUCERS
    )
    assert {view["context_ref"] for _role, view in calls} == {context.context_ref}
    assert {view["context_digest"] for _role, view in calls} == {context.digest}
    assert [view["producer_role"] for _role, view in calls] == list(
        DISCOVERY_PRODUCERS
    )


def test_successes_are_real_open_specs_consumable_by_resolver() -> None:
    context = _context()
    _calls, outcomes = _captured_inputs(context)

    for outcome in outcomes:
        assert outcome.spec is not None
        assert isinstance(outcome.spec, OpenResearchSpecV1)
        resolution = resolve_capability(
            outcome.spec,
            resolution_facts=outcome.resolution_facts,
            environment=_environment(),
        )
        assert resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY


def test_policy_allocation_and_axis_change_alter_callable_input() -> None:
    before_context = _context()
    before_calls, _before_outcomes = _captured_inputs(before_context)

    changed_policy = {
        **before_context.policy,
        "producer_token_allocation": {
            **dict(before_context.policy["producer_token_allocation"]),
            "frontier_architect": 0.4,
        },
        "mechanism_axis_targeting": ("geometry",),
    }
    after_context = replace(before_context, policy=changed_policy)
    after_calls, _after_outcomes = _captured_inputs(after_context)

    before_view = dict(before_calls[-1][1])
    after_view = dict(after_calls[-1][1])
    assert before_view["producer_token_fraction"] == 0.25
    assert after_view["producer_token_fraction"] == 0.4
    assert before_view["mechanism_axis_targeting"] == [
        "propagation",
        "objective",
    ]
    assert after_view["mechanism_axis_targeting"] == ["geometry"]
    assert before_view != after_view


def test_role_scoped_memory_changes_only_the_relevant_callable_input() -> None:
    before_context = _context()
    before_calls, _before_outcomes = _captured_inputs(before_context)

    memory = dict(before_context.scientific_memory)
    by_role = dict(memory["by_role"])
    by_role["frontier_architect"] = {"retrieval_key": "frontier:round-2"}
    after_context = replace(
        before_context,
        scientific_memory={**memory, "by_role": by_role},
    )
    after_calls, _after_outcomes = _captured_inputs(after_context)

    before = {role: view for role, view in before_calls}
    after = {role: view for role, view in after_calls}
    assert before["frontier_architect"]["memory"] != after[
        "frontier_architect"
    ]["memory"]
    assert before["mechanism_composer"]["memory"] == after[
        "mechanism_composer"
    ]["memory"]


def test_one_role_failure_preserves_three_bound_specs_and_role_identity() -> None:
    context = _context()
    semantics_digest = _environment()["current_capabilities"][0][
        "semantics_digest"
    ]
    calls: list[str] = []

    def producer(role: str, view: dict[str, Any]) -> dict[str, Any]:
        calls.append(role)
        if role == "falsification_designer":
            raise RuntimeError("deterministic producer failure")
        return _draft(
            role,
            semantics_digest=semantics_digest,
            marker=view["memory"]["retrieval_key"],
        )

    outcomes = produce_research_specs(context, producer, _bindings(context))
    by_role = {outcome.producer_role: outcome for outcome in outcomes}

    assert calls == list(DISCOVERY_PRODUCERS)
    assert len(outcomes) == 4
    assert by_role["falsification_designer"].failure_code == "PRODUCER_CALL_FAILED"
    assert by_role["falsification_designer"].spec is None
    assert sum(outcome.spec is not None for outcome in outcomes) == 3
    assert all(
        outcome.context_digest == context.digest for outcome in outcomes
    )


def test_role_mismatch_and_cross_context_bindings_cannot_pass_as_specs() -> None:
    context = _context()
    semantics_digest = _environment()["current_capabilities"][0][
        "semantics_digest"
    ]

    def mismatching_producer(role: str, view: dict[str, Any]) -> dict[str, Any]:
        returned_role = (
            "mechanism_composer"
            if role == "frontier_architect"
            else role
        )
        return _draft(
            returned_role,
            semantics_digest=semantics_digest,
            marker=view["memory"]["retrieval_key"],
        )

    outcomes = produce_research_specs(
        context,
        mismatching_producer,
        _bindings(context),
    )
    frontier = next(
        outcome
        for outcome in outcomes
        if outcome.producer_role == "frontier_architect"
    )
    assert frontier.spec is None
    assert frontier.failure_code == "OPEN_SPEC_PROJECTION_FAILED"
    assert all(
        outcome.producer_role in DISCOVERY_PRODUCERS for outcome in outcomes
    )

    wrong_bindings = _bindings(context)
    wrong_bindings["context_digest"] = _digest("different-context")
    with pytest.raises(ValueError, match="must match ResearchContext"):
        produce_research_specs(context, mismatching_producer, wrong_bindings)


def test_unexpected_projection_internal_error_propagates(monkeypatch) -> None:
    context = _context()

    def internal_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("unexpected projection defect")

    monkeypatch.setattr(producers_module, "_project_result", internal_failure)

    with pytest.raises(RuntimeError, match="unexpected projection defect"):
        produce_research_specs(
            context,
            lambda _role, _view: {},
            _bindings(context),
        )
