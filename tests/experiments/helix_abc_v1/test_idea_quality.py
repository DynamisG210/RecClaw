from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from recclaw_core.experiments.helix_abc_v1.idea_quality import (
    Q1_CANDIDATE_SLOTS,
    Q1_CONDITIONAL_SENTINEL_PROMPT,
    Q1_MODEL,
    Q1_PROPOSAL_TOKEN_CEILING,
    build_q1_ab_contract,
    derive_q1_conditional_contract_sentinel_schema,
    derive_enriched_proposal_schema,
    q1_provider_contract_static_matrix,
    render_q1_producer_prompt,
    score_preoutcome_testability,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (
    derive_fresh_r2_proposal_schema,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    OpenSpecProjectionError,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    SharedImplementerPolicy,
    build_shared_implementer_request,
)
from recclaw_core.experiments.helix_abc_v1.canonical import canonical_json_bytes
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiBrokerReleaseV1,
    validate_provider_strict_schema,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"


def _bindings() -> dict[str, object]:
    digest = "a" * 64
    return {
        "protocol_ref": "protocol:test",
        "protocol_digest": digest,
        "context_ref": "context:test",
        "context_digest": digest,
        "current_profile_ref": "profile:test",
        "current_profile_digest": digest,
        "implementation_requirements": ["candidate-local package"],
        "compatibility_requirements": ["pairwise input"],
    }


def _enriched_draft(*, mode: str) -> dict[str, object]:
    diagnosis = mode == "DIAGNOSIS_DRIVEN"
    return {
        "producer_role": (
            "falsification_designer" if diagnosis else "frontier_architect"
        ),
        "hypothesis": "A gated residual corrects oversmoothing without replacing BPR.",
        "mechanism_change": "Add one learned residual gate around one propagation step.",
        "competing_explanation": "Any gain comes only from added parameters.",
        "matched_control_requirement": "Match BPR data, split, seed, and budget.",
        "implementation_requirements": ["candidate-local package"],
        "expected_evidence": ["Gate-on differs from gate-off on fixed inputs."],
        "falsifier": "Reject the mechanism account if gate-off is not parent-near.",
        "compatibility_requirements": ["pairwise input"],
        "high_change_justification": "One new trainable propagation gate.",
        "current_profile_expressibility_claim": "NOT_EXPRESSIBLE",
        "resolution_facts": {
            "requested_current_semantics_digest": None,
            "capability_diff": ["learned residual propagation gate"],
            "high_change_dimensions": ["PROPAGATION_MECHANISM"],
            "required_dependencies": [],
            "required_budget": {
                "implementation_token_ceiling": 20000,
                "qualification_gpu_minutes": 10,
                "qualification_wall_minutes": 15,
            },
        },
        "idea_mode": mode,
        "research_question": "Does a residual gate isolate useful propagation from oversmoothing?",
        "observed_failure_mode": (
            "Accepted frontier candidates were resource-censored before effect evidence."
            if diagnosis
            else None
        ),
        "closest_parent": "BPR with one LightGCN-style propagation step",
        "minimal_testable_wedge": "Toggle only the learned residual gate.",
        "causal_chain": [
            "gate suppresses harmful propagated components",
            "parent embeddings retain pairwise ranking signal",
        ],
        "discriminative_predictions": [
            "Gate-on beats a parameter-matched always-on residual if selective suppression matters."
        ],
        "mechanism_off_definition": "Set the residual gate to zero and recover parent scoring.",
        "resource_hypothesis": "One sparse propagation adds linear edge-time and fits 10 GiB.",
        "realization_mode": "PARENT_PRESERVING",
    }


def test_offline_ab_contract_is_symmetric_origin_blind_and_small() -> None:
    contract = build_q1_ab_contract()

    assert contract["model"] == Q1_MODEL
    assert contract["tools"] == []
    assert contract["proposal_token_ceiling"] == Q1_PROPOSAL_TOKEN_CEILING
    assert Q1_PROPOSAL_TOKEN_CEILING == 16_000
    assert contract["proposal_token_ceiling_calibration"] == {
        "current": 16_000,
        "derivation": (
            "2x fresh maximum proposal input tokens (7917) = 15834, "
            "rounded upward to the next thousand"
        ),
        "historical_right_censored": 12_000,
        "outcome_blind": True,
        "same_for_all_arms_and_slots": True,
    }
    assert contract["candidate_slots"] == list(Q1_CANDIDATE_SLOTS)
    assert contract["candidate_count_per_arm"] == 2
    assert contract["selection_budget_per_arm"] == 1
    assert contract["inputs_symmetric_except_contract_enrichment"] is True
    assert contract["selection_frozen_before_implementation_or_qualification"] is True
    assert contract["selection_features"] == [
        "discriminative_value",
        "mechanism_off_executability",
        "parent_clarity",
        "q0r2_resource_feasibility",
        "scientific_testability",
    ]
    assert contract["forbidden_selection_features"] == [
        "implementation_success",
        "ndcg_or_other_effect_metric",
        "qualification_result",
    ]


def test_all_four_proposal_slots_share_the_frozen_broker_release_ceiling() -> None:
    release = LabApiBrokerReleaseV1.create(
        base_url="https://laboratory.invalid/v1",
        model=Q1_MODEL,
        response_schema_digest="a" * 64,
        max_total_tokens_per_call=Q1_PROPOSAL_TOKEN_CEILING,
        timeout_ms=900_000,
    )
    assert release.max_total_tokens_per_call == 16_000
    assert release.retry_count == 0
    assert release.temperature == 0.0
    assert Q1_CANDIDATE_SLOTS == ("diagnosis", "frontier")
    assert build_q1_ab_contract()["candidate_count_per_arm"] == 2


def test_prompts_are_origin_blind_and_only_enriched_contract_differs() -> None:
    template = (RESOURCE_ROOT / "idea_quality_producer_prompt_v1.txt").read_text(
        encoding="utf-8"
    )
    common = {
        "template": template,
        "slot": Q1_CANDIDATE_SLOTS[0],
        "role": "falsification_designer",
        "seed": 55011,
        "context": {"accepted_evidence": "resource censoring without effect update"},
        "profile_catalog": [{"semantics_digest": "b" * 64, "mechanism_summary": "BPR"}],
        "protocol_ref": "protocol:test",
        "protocol_digest": "c" * 64,
        "context_ref": "context:test",
        "context_digest": "d" * 64,
        "profile_ref": "profile:test",
        "profile_digest": "e" * 64,
    }
    baseline = render_q1_producer_prompt(arm="baseline", **common)
    enriched = render_q1_producer_prompt(arm="enriched", **common)

    for prompt in (baseline, enriched):
        lowered = prompt.lower()
        assert "side_a" not in lowered
        assert "side_b" not in lowered
        assert "expected winner" not in lowered
        assert "implementation success" not in lowered
        assert "candidate_ndcg" not in lowered
        assert "observed_ndcg" not in lowered
    assert baseline != enriched
    assert "current OpenSpec fields" in baseline
    assert "minimal_testable_wedge" in enriched
    assert "DIAGNOSIS_DRIVEN" in enriched


def test_real_context_double_braces_are_not_misclassified_as_template_tokens() -> None:
    template = (RESOURCE_ROOT / "idea_quality_producer_prompt_v1.txt").read_text(
        encoding="utf-8"
    )
    prompt = render_q1_producer_prompt(
        template,
        arm="baseline",
        slot="diagnosis",
        role="falsification_designer",
        seed=55011,
        context={"equation": "score={{u,i}} dot product"},
        profile_catalog=[],
        protocol_ref="protocol:test",
        protocol_digest="a" * 64,
        context_ref="context:test",
        context_digest="b" * 64,
        profile_ref="profile:test",
        profile_digest="c" * 64,
    )
    assert "score={{u,i}}" in prompt


def test_enriched_schema_and_existing_projection_enforce_mode_semantics() -> None:
    schema = derive_enriched_proposal_schema()
    proposal = schema["properties"]["proposals"]["items"]
    required = set(proposal["required"])
    assert {
        "idea_mode",
        "research_question",
        "observed_failure_mode",
        "closest_parent",
        "minimal_testable_wedge",
        "causal_chain",
        "discriminative_predictions",
        "mechanism_off_definition",
        "resource_hypothesis",
        "realization_mode",
    } <= required
    assert "allOf" not in proposal
    serialized_schema = canonical_json_bytes(schema)
    for keyword in (b'"allOf"', b'"if"', b'"then"', b'"const"'):
        assert keyword not in serialized_schema

    projected = {}
    for mode in ("DIAGNOSIS_DRIVEN", "FRONTIER_HYPOTHESIS"):
        draft = _enriched_draft(mode=mode)
        response = {
            "schema": "recclaw.research-line.fresh-open-spec-proposal-response.v1",
            "proposals": [draft],
        }
        roundtripped = json.loads(canonical_json_bytes(response))
        jsonschema.validate(roundtripped, schema)
        projected[mode], _facts = project_open_producer_draft(
            roundtripped["proposals"][0], bindings=_bindings()
        )
    diagnosis = projected["DIAGNOSIS_DRIVEN"]
    frontier = projected["FRONTIER_HYPOTHESIS"]
    assert diagnosis.idea_mode.value == "DIAGNOSIS_DRIVEN"
    assert diagnosis.observed_failure_mode is not None
    assert frontier.idea_mode.value == "FRONTIER_HYPOTHESIS"
    assert frontier.observed_failure_mode is None
    assert frontier.minimal_testable_wedge == "Toggle only the learned residual gate."

    invalid = _enriched_draft(mode="DIAGNOSIS_DRIVEN")
    invalid["observed_failure_mode"] = None
    with pytest.raises(OpenSpecProjectionError):
        project_open_producer_draft(invalid, bindings=_bindings())

    invalid = _enriched_draft(mode="FRONTIER_HYPOTHESIS")
    invalid["observed_failure_mode"] = "Invented failure"
    jsonschema.validate(
        {
            "schema": "recclaw.research-line.fresh-open-spec-proposal-response.v1",
            "proposals": [invalid],
        },
        schema,
    )
    with pytest.raises(OpenSpecProjectionError):
        project_open_producer_draft(invalid, bindings=_bindings())

    missing = _enriched_draft(mode="DIAGNOSIS_DRIVEN")
    del missing["minimal_testable_wedge"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {
                "schema": "recclaw.research-line.fresh-open-spec-proposal-response.v1",
                "proposals": [missing],
            },
            schema,
        )


def test_provider_compatible_realization_preserves_baseline_schema_bytes() -> None:
    baseline_bytes = canonical_json_bytes(derive_fresh_r2_proposal_schema())
    assert len(baseline_bytes) == 2885
    assert (
        hashlib.sha256(baseline_bytes).hexdigest()
        == "84e1deee46b6584aa08a1046724157ef03a7492c6e92d9d15b7ddcc45fb9369a"
    )


def test_contract_sentinel_is_small_nonresearch_and_isolates_conditionals() -> None:
    sentinel = derive_q1_conditional_contract_sentinel_schema()
    baseline = build_q1_ab_contract()
    matrix = q1_provider_contract_static_matrix()

    validate_provider_strict_schema(sentinel)
    assert matrix["schemas"]["sentinel"]["conditional_composition"] is True
    assert matrix["schemas"]["sentinel"]["proposal_properties"] == 2
    assert matrix["schemas"]["sentinel"]["canonical_bytes"] < matrix["schemas"][
        "baseline"
    ]["canonical_bytes"]
    assert matrix["research_candidate_generation_allowed"] is False
    assert baseline["model"] == matrix["invariants"]["model"]
    serialized = canonical_json_bytes(sentinel).lower()
    for forbidden in (b"openspec", b"recommender", b"hypothesis", b"mechanism"):
        assert forbidden not in serialized
        assert forbidden not in Q1_CONDITIONAL_SENTINEL_PROMPT.lower().encode()


def test_preoutcome_scoring_rejects_outcome_or_implementation_inputs() -> None:
    spec, _facts = project_open_producer_draft(
        _enriched_draft(mode="FRONTIER_HYPOTHESIS"), bindings=_bindings()
    )
    score = score_preoutcome_testability(spec, q0r2_resource_feasible=True)
    assert score["total"] > 0
    with pytest.raises(ValueError):
        score_preoutcome_testability(
            spec,
            q0r2_resource_feasible=True,
            outcome_features={"ndcg@10": 1.0},
        )


def test_existing_origin_blind_implementer_consumer_receives_enriched_semantics() -> None:
    spec, _facts = project_open_producer_draft(
        _enriched_draft(mode="FRONTIER_HYPOTHESIS"), bindings=_bindings()
    )
    policy = SharedImplementerPolicy(
        allowed_files=("recclaw_ext/__init__.py", "recclaw_ext/candidate.py"),
        dependency_identity_ref="dependency:test",
        dependency_identity_digest="1" * 64,
        runtime_identity_ref="runtime:test",
        runtime_identity_digest="2" * 64,
        prompt_digest="3" * 64,
        tool_policy_digest="4" * 64,
        implementation_token_ceiling=20_000,
    )
    request = build_shared_implementer_request(spec, policy=policy)
    projected = request["blind_research_spec"]

    assert projected["minimal_testable_wedge"] == spec.minimal_testable_wedge
    assert projected["mechanism_off_definition"] == spec.mechanism_off_definition
    assert projected["realization_mode"] == "PARENT_PRESERVING"
    assert "producer_role" not in projected
    assert "context_ref" not in projected
