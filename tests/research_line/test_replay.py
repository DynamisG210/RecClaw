from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    VersionedResearchPolicyV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    adapt_current_search_profile,
)
from recclaw_core.research_line.interfaces import ResearchContext
from recclaw_core.research_line.replay import OfflineProducerReplayV1
from recclaw_core.research_line.runtime import bindings_for_context


_HELIX_TEST_ROOT = (
    Path(__file__).resolve().parents[1] / "experiments" / "helix_abc_v1"
)
sys.path.insert(0, str(_HELIX_TEST_ROOT))

from test_e0_search_adapter import _proposal  # noqa: E402
from test_vnext_local_orchestration import _open_draft  # noqa: E402


def _context(profile: Any, policy: VersionedResearchPolicyV1) -> ResearchContext:
    return ResearchContext(
        campaign_id=profile.campaign_id,
        round_index=1,
        knowledge_base={
            "search_space": "BL-ICF",
            "executable_entries": len(profile.entries),
        },
        frozen_goal={"metric": "NDCG@10", "direction": "maximize"},
        frontier={"incumbent_ndcg@10": 0.40},
        scientific_memory={
            "by_role": {
                role: {"prior": role} for role in DISCOVERY_PRODUCERS
            }
        },
        unresolved_questions=(
            {"question": "which mechanism moves the frontier?"},
        ),
        policy=policy.to_dict(),
        budget={"producer_calls": 4, "experiment_opportunities": 1},
        active_profile_ref=profile.profile_ref,
        active_profile_digest=profile.profile_digest,
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
    )


def _memory():
    return SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY").commit(
        round_index=1,
        expected_predecessor_digest=None,
        beliefs=(),
        route_trace_digest=sha256_digest({"route": "offline-replay-test"}),
        feedback_projection={
            "producer_useful_rates": {
                role: 0.5 for role in DISCOVERY_PRODUCERS
            }
        },
    )


def test_offline_replay_reuses_four_producers_and_policy_changes_metrics() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:offline-replay")
    champion_policy = replace(
        initial_research_policy(),
        memory_retrieval_policy="REPLAY_CHAMPION_V1",
    )
    challenger_policy = replace(
        champion_policy,
        memory_retrieval_policy="REPLAY_CHALLENGER_V1",
    )
    context = _context(profile, champion_policy)
    bindings = bindings_for_context(
        context,
        active_profile=profile,
        implementation_requirements=(
            "RecBole general recommender interface",
            "candidate-local package",
        ),
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
    )
    mechanisms_by_axis = {}
    for mechanism in executable_mechanisms():
        mechanisms_by_axis.setdefault(mechanism.mechanism_axis, mechanism)
    selected_mechanisms = tuple(mechanisms_by_axis.values())[:4]
    calls: list[tuple[str, dict[str, Any]]] = []

    def producer(role: str, view: dict[str, Any]):
        calls.append((role, view))
        is_challenger = (
            view["memory_retrieval_policy"]
            == challenger_policy.memory_retrieval_policy
        )
        index = DISCOVERY_PRODUCERS.index(role)
        mechanism = selected_mechanisms[index if is_challenger else 0]
        proposal = _proposal(
            candidate_id=(
                f"cand-replay-{'challenger' if is_challenger else 'champion'}-{role}"
            ),
            mechanism_id=mechanism.mechanism_id,
            mechanism_axis=mechanism.mechanism_axis,
            mechanism_program=mechanism.mechanism_program,
            protocol_digest=profile.protocol_digest,
            role=role,
        )
        if not is_challenger:
            proposal = replace(
                proposal,
                mechanism_program=canonical_value(proposal.mechanism_program),
                parent_candidate_id="cand-replay-champion-root",
            )
        return proposal

    result = OfflineProducerReplayV1(
        producer=producer,
        producer_bindings=bindings,
        equal_replay_token_charge=137,
        deterministic_directive_replay=True,
    )(
        context=context,
        champion_policy=champion_policy,
        challenger_policy=challenger_policy,
        search_memory=_memory(),
    )

    assert set(result) == {
        "source_context_digest",
        "source_search_memory_digest",
        "champion_policy_digest",
        "challenger_policy_digest",
        "champion",
        "challenger",
        "same_model_prompt_schema_and_contexts",
        "deterministic_directive_replay",
    }
    assert result["source_context_digest"] == context.digest
    assert result["champion_policy_digest"] == champion_policy.digest
    assert result["challenger_policy_digest"] == challenger_policy.digest
    assert result["same_model_prompt_schema_and_contexts"] is True
    assert result["deterministic_directive_replay"] is True

    champion = result["champion"]
    challenger = result["challenger"]
    assert champion.proposal_count == challenger.proposal_count == 4
    assert champion.common_eligible_count == challenger.common_eligible_count == 4
    assert champion.unique_semantics_count == 1
    assert challenger.unique_semantics_count == 4
    assert champion.mechanism_axis_count == 1
    assert challenger.mechanism_axis_count == 4
    assert champion.lineage_root_count == 1
    assert challenger.lineage_root_count == 4
    assert champion.control_count == challenger.control_count == 4
    assert champion.semantic_collision_count == 3
    assert challenger.semantic_collision_count == 0
    assert champion.billed_tokens == challenger.billed_tokens == 137

    assert [role for role, _view in calls] == list(DISCOVERY_PRODUCERS) * 2
    champion_views = dict(calls[:4])
    challenger_views = dict(calls[4:])
    for role in DISCOVERY_PRODUCERS:
        assert champion_views[role]["policy"] != challenger_views[role]["policy"]
        assert (
            champion_views[role]["scientific_memory"]
            == challenger_views[role]["scientific_memory"]
        )
        assert (
            champion_views[role]["context_digest"]
            != challenger_views[role]["context_digest"]
        )
        assert champion_views[role]["protocol_digest"] == profile.protocol_digest
        assert challenger_views[role]["protocol_digest"] == profile.protocol_digest


def test_offline_replay_counts_open_draft_specs_without_typed_proposals() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:open-replay")
    policy = replace(
        initial_research_policy(),
        memory_retrieval_policy="REPLAY_OPEN_DRAFT_V1",
    )
    context = _context(profile, policy)
    bindings = bindings_for_context(
        context,
        active_profile=profile,
        implementation_requirements=(
            "RecBole general recommender interface",
            "candidate-local package",
        ),
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
    )

    def producer(role: str, _view: dict[str, Any]) -> dict[str, Any]:
        draft = _open_draft(producer_role=role)
        draft.update(
            {
                "idea_mode": "FRONTIER_HYPOTHESIS",
                "research_question": "Which open mechanism explains the gap?",
                "observed_failure_mode": None,
                "closest_parent": "parent:open-replay-incumbent",
                "minimal_testable_wedge": "one gated interaction path",
                "causal_chain": ("gate", "pairwise score"),
                "discriminative_predictions": ("the gate changes the signature",),
                "mechanism_off_definition": "remove the interaction gate",
                "resource_hypothesis": "one candidate-local package",
                "realization_mode": "NON_NESTED",
            }
        )
        return draft

    memory = _memory()
    result = OfflineProducerReplayV1(
        producer=producer,
        producer_bindings=bindings,
        equal_replay_token_charge=211,
        deterministic_directive_replay=True,
    )(
        context=context,
        champion_policy=policy,
        challenger_policy=policy,
        search_memory=memory,
    )

    assert result["source_search_memory_digest"] == memory.digest
    assert result["champion"].proposal_count == 4
    assert result["champion"].common_eligible_count == 4
    assert result["champion"].unique_semantics_count == 1
    assert result["champion"].mechanism_axis_count == 1
    assert result["champion"].lineage_root_count == 1
    assert result["champion"].control_count == 4
    assert result["champion"].semantic_collision_count == 3
    assert result["champion"].billed_tokens == 211
