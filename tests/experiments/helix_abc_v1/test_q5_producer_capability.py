from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recclaw_core.experiments.helix_abc_v1.idea_quality import (  # noqa: E402
    render_q1_producer_prompt,
    score_preoutcome_testability,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    CurrentProfileExpressibilityV1,
    IdeaModeV1,
    OpenResearchSpecV1,
    RealizationModeV1,
)
from run_q5a_idea_feasibility import (  # noqa: E402
    Q5A_AGGREGATE_CONTEXT,
    _render_q5_prompt,
)


def _digest(label: str) -> str:
    return (label.encode("utf-8").hex() + "0" * 64)[:64]


def _spec(
    *,
    mode: IdeaModeV1,
    realization: RealizationModeV1,
    parent: str,
    wedge: str,
    causal_chain: tuple[str, ...],
) -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis="A candidate-local mechanism changes propagation behavior.",
        mechanism_change="Apply one causal operator to the parent scoring path.",
        competing_explanation="The operator may be inert.",
        matched_control_requirement="Keep the parent model and data fixed.",
        implementation_requirements=("Implement the frozen RecBole interface.",),
        expected_evidence=("Loss and full-sort behavior are measurable.",),
        falsifier="Reject if the matched control is not distinguishable.",
        compatibility_requirements=("full-sort NDCG@10",),
        protocol_ref="protocol:q5b",
        protocol_digest=_digest("protocol"),
        context_ref="context:q5b",
        context_digest=_digest("context"),
        current_profile_ref="profile:q5b",
        current_profile_digest=_digest("profile"),
        producer_role="mechanism_composer",
        high_change_justification="The mechanism is candidate-local.",
        current_profile_expressibility_claim=CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE,
        idea_mode=mode,
        research_question="Does the operator alter the parent score for the proposed failure mode?",
        observed_failure_mode=("parent underweights tail items" if mode is IdeaModeV1.DIAGNOSIS_DRIVEN else None),
        closest_parent=parent,
        minimal_testable_wedge=wedge,
        causal_chain=causal_chain,
        discriminative_predictions=("The matched control rejects the mechanism.",),
        mechanism_off_definition=(
            "Disable the operator and recover the equivalent parent loss and scoring path."
            if realization is RealizationModeV1.PARENT_PRESERVING
            else "Use the declared matched parent package as the mechanism-off control."
        ),
        resource_hypothesis="4 GB GPU memory, 3 GPU minutes, 8 wall minutes.",
        realization_mode=realization,
    )


def test_q5_prompt_injects_distinct_role_modes_and_pool_signatures() -> None:
    template = (
        ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources/"
        "idea_quality_producer_prompt_v1.txt"
    ).read_text(encoding="utf-8")
    active = SimpleNamespace(
        protocol_ref="protocol:q5b",
        protocol_digest=_digest("protocol"),
        profile_ref="profile:q5b",
        profile_digest=_digest("profile"),
    )
    common = {
        "template": template,
        "seed": 71001,
        "context": {"q5a_aggregate_facts": Q5A_AGGREGATE_CONTEXT},
        "catalog": [{"profile_ref": "BL_ICF_EXECUTABLE_PROFILE_V2"}],
        "active": active,
        "context_digest": _digest("aggregate"),
        "pool_signatures": ({"causal_operator": "tail gate"},),
    }
    diagnosis = _render_q5_prompt(
        slot="slot-01",
        role="mechanism_composer",
        mode="DIAGNOSIS_DRIVEN",
        **common,
    )
    frontier = _render_q5_prompt(
        slot="slot-05",
        role="frontier_architect",
        mode="FRONTIER_HYPOTHESIS",
        **common,
    )
    assert diagnosis != frontier
    assert "PARENT_PRESERVING" in diagnosis
    assert "NON_NESTED" in frontier
    assert "Compose a genuinely new" in diagnosis
    assert "Propose a new structural frontier" in frontier
    assert "tail gate" in diagnosis
    assert "candidate_specific_results" in diagnosis
    assert "{{" not in diagnosis and "{{" not in frontier
    assert Q5A_AGGREGATE_CONTEXT["candidate_specific_results"] is False


def test_legacy_q1_renderer_supplies_new_shared_template_defaults() -> None:
    template = (
        ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources/"
        "idea_quality_producer_prompt_v1.txt"
    ).read_text(encoding="utf-8")
    rendered = render_q1_producer_prompt(
        template,
        arm="enriched",
        slot="diagnosis",
        role="mechanism_composer",
        seed=55011,
        context={},
        profile_catalog=(),
        protocol_ref="protocol:q1",
        protocol_digest=_digest("protocol-q1"),
        context_ref="context:q1",
        context_digest=_digest("context-q1"),
        profile_ref="profile:q1",
        profile_digest=_digest("profile-q1"),
    )
    assert "Compose a genuinely new" in rendered
    assert "Previously proposed mechanism signatures in this pool" in rendered
    assert "{{ROLE_INSTRUCTION}}" not in rendered
    assert "{{POOL_MECHANISM_SIGNATURES}}" not in rendered


def test_graded_preoutcome_features_are_structural_and_not_all_tied() -> None:
    rich = _spec(
        mode=IdeaModeV1.DIAGNOSIS_DRIVEN,
        realization=RealizationModeV1.PARENT_PRESERVING,
        parent="BL_ICF_EXECUTABLE_PROFILE_V2",
        wedge=(
            "Route a tail tensor through a gated loss and score path; off disables "
            "the gate and recovers the parent with O(batch*embedding) memory."
        ),
        causal_chain=("tail gate", "loss", "score"),
    )
    weak = _spec(
        mode=IdeaModeV1.DIAGNOSIS_DRIVEN,
        realization=RealizationModeV1.NON_NESTED,
        parent="natural-language parent hypothesis",
        wedge="change the model",
        causal_chain=("change",),
    )
    context = {
        "parent_catalog": [{"profile_ref": "BL_ICF_EXECUTABLE_PROFILE_V2"}],
        "pool_signatures": ({"causal_operator": "other operator"},),
        "required_budget": {"gpu_minutes": 10, "wall_minutes": 30},
        "failure_summary": {
            "implementer_success_count": 17,
            "qualifier_failure_taxonomy": {
                "API_OR_TENSOR": 4,
                "UNIT_OR_MECHANISM_OFF": 5,
                "CONSTRUCTION_IMPORT": 3,
            },
        },
        "mode": "DIAGNOSIS_DRIVEN",
    }
    rich_score = score_preoutcome_testability(
        rich, q0r2_resource_feasible=True, structural_context=context
    )
    weak_score = score_preoutcome_testability(
        weak, q0r2_resource_feasible=True, structural_context=context
    )
    assert rich_score["outcome_fields_consumed"] == []
    assert rich_score["features"]["executable_parent"] > weak_score["features"]["executable_parent"]
    assert rich_score["features"]["wedge_specificity"] > weak_score["features"]["wedge_specificity"]
    assert rich_score["features"]["causal_component_count"] > weak_score["features"]["causal_component_count"]
    assert rich_score["features"]["role_mode_fit"] > weak_score["features"]["role_mode_fit"]
    assert rich_score["features"]["parent_family_novelty"] >= weak_score["features"]["parent_family_novelty"]
    assert "causal_operator_novelty" in rich_score["features"]
    assert "qualifier_risk" in rich_score["features"]
    assert "resource_margin" in rich_score["features"]
    assert rich_score["total"] != weak_score["total"]
