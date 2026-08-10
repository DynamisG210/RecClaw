from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerCallV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    SharedImplementerPolicy,
    build_shared_implementer_request,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    CanaryBrokerError,
    validate_provider_strict_schema,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    IdeaModeV1,
    OpenResearchSpecV1,
    RealizationModeV1,
)
from recclaw_core.research_line import provider as provider_module
from recclaw_core.research_line.provider import (
    ProviderImplementerGateway,
    ProviderResearchProducer,
)


def _digest(label: str) -> str:
    return sha256_digest({"fixture": label})


def _config() -> dict[str, Any]:
    return {
        "source_ref": "llm-config:ordered-primary",
        "source_digest": _digest("ordered-primary-config"),
        "release_digest": _digest("provider-release"),
        "model": "gpt-5.4",
        "transport": "injected",
    }


def _context_view(role: str, *, marker: str = "round-1") -> dict[str, Any]:
    return {
        "context_ref": "research-context:campaign-provider:round-1",
        "context_digest": _digest("provider-context"),
        "producer_role": role,
        "knowledge_base": {"mechanism_index": f"kb-{marker}"},
        "frozen_goal": {"research_question": f"goal-{marker}"},
        "frontier": {"unresolved_axis": f"axis-{marker}"},
        "memory": {"role_memory": f"memory-{role}-{marker}"},
        "unresolved_questions": ({"question": f"question-{marker}"},),
        "budget": {"remaining_tokens": f"budget-{marker}"},
        "active_profile_ref": "profile:provider-fixture",
        "active_profile_digest": _digest("provider-profile"),
        "protocol_ref": "protocol:provider-fixture",
        "protocol_digest": _digest("provider-protocol"),
        "producer_token_fraction": 0.25,
        "mechanism_axis_targeting": [f"target-{marker}"],
        "memory_retrieval_policy": "ROLE_SCOPED_TEST",
    }


def _open_proposal(
    role: str,
    *,
    base_model_config: str = "BPR",
) -> dict[str, Any]:
    return {
        "producer_role": role,
        "idea_mode": "FRONTIER_HYPOTHESIS",
        "research_question": (
            f"Can the {role} mechanism change candidate-local pairwise scores "
            "under the frozen BL-ICF protocol?"
        ),
        "observed_failure_mode": None,
        "hypothesis": f"A {role} mechanism changes pairwise ranking.",
        "closest_parent": "BL-ICF executable profile",
        "minimal_testable_wedge": (
            "Add one candidate-local trainable interaction while preserving "
            "the frozen pairwise interface."
        ),
        "causal_chain": [
            "candidate-local interaction changes representation",
            "changed representation alters pairwise score difference",
            "score difference yields a discriminative control prediction",
        ],
        "mechanism_change": "Add a candidate-local trainable interaction.",
        "competing_explanation": "The signal is only inherited BPR capacity.",
        "discriminative_predictions": [
            "the interaction-specific control signature changes",
            "the matched parent lacks that signature",
        ],
        "mechanism_off_definition": (
            "Without the candidate-local interaction, the predicted signature "
            "must disappear."
        ),
        "matched_control_requirement": "Use the matched pairwise incumbent.",
        "implementation_requirements": [
            "RecBole general recommender interface",
            "candidate-local package",
        ],
        "execution_contract": {
            "capability_family": "OPEN_INTERACTION_CUSTOM",
            "model": "FreshCandidateModel",
            "base_model_config": base_model_config,
            "config": {},
        },
        "expected_evidence": [
            "construction and API checks",
            "prospective matched-control falsification only; no outcome claim",
        ],
        "falsifier": "The predicted interaction signature is absent.",
        "compatibility_requirements": [
            "general collaborative filtering",
            "pairwise input",
        ],
        "resource_hypothesis": (
            "One additional candidate-local interaction should fit within the "
            "bounded implementation and qualification budget."
        ),
        "realization_mode": "NON_NESTED",
        "high_change_justification": (
            "The interaction is outside the frozen profile; claim ceiling is "
            "prospective mechanism evidence only, not a metric outcome."
        ),
        "current_profile_expressibility_claim": "NOT_EXPRESSIBLE",
        "resolution_facts": {
            "requested_current_semantics_digest": None,
            "capability_diff": [f"new {role} interaction"],
            "high_change_dimensions": ["INTERACTION_STRUCTURE"],
            "required_dependencies": ["recbole-runtime"],
        "required_budget": {
            "implementation_token_ceiling": 20000,
            "qualification_gpu_minutes": 10,
            "qualification_wall_minutes": 30,
        },
        },
    }


def _proposal_response(
    role: str,
    *,
    base_model_config: str = "BPR",
) -> dict[str, Any]:
    proposal = _open_proposal(role, base_model_config=base_model_config)
    contract = proposal["execution_contract"]
    contract["config_json"] = json.dumps(contract.pop("config"), sort_keys=True)
    return {
        "schema": "recclaw.research-line.fresh-open-spec-proposal-response.v1",
        "proposals": [proposal],
    }


def _implementation_response() -> dict[str, Any]:
    return {
        "schema": "recclaw.research-line.fresh-r1-implementation-response.v1",
        "proposals": [
            {
                "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
                "files": [
                    {"path": "recclaw_ext/__init__.py", "content": "# fixture\n"},
                    {
                        "path": "recclaw_ext/candidate.py",
                        "content": "class FreshCandidateModel:\n    pass\n",
                    },
                ],
                "implementation_summary": "Provider fixture implementation.",
            }
        ],
    }


def _call_result(
    *,
    logical_call_id: str,
    response: dict[str, Any],
    attempts: tuple[dict[str, Any], ...] | None = None,
) -> fresh_r1.ProviderAttemptResult:
    request_digest = _digest(logical_call_id)
    call = CanaryBrokerCallV1(
        logical_call_id=logical_call_id,
        request_digest=request_digest,
        response_digest=sha256_digest(response),
        response=response,
        input_tokens=13,
        cached_input_tokens=2,
        output_tokens=7,
        total_tokens=20,
        latency_ms=11,
        returned_model="gpt-5.4",
    )
    return fresh_r1.ProviderAttemptResult(
        call=call,
        attempts=attempts
        or (
            {
                "ordinal": 1,
                "status": "SUCCESS",
                "request_digest": request_digest,
                "input_tokens": 13,
                "output_tokens": 7,
                "billed_tokens": 20,
            },
        ),
        failure=None,
    )


def _implementation_policy() -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=("recclaw_ext/__init__.py", "recclaw_ext/candidate.py"),
        dependency_identity_ref="dependencies:provider-fixture",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref="runtime:provider-fixture",
        runtime_identity_digest=_digest("runtime"),
        prompt_digest=_digest("implementer-prompt"),
        tool_policy_digest=_digest("tool-policy"),
        implementation_token_ceiling=fresh_r1.IMPLEMENTATION_TOKEN_CEILING,
        execution_contract={
            "base_model_config": "LightGCN",
            "model": "LightGCN",
            "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
            "protocol_ref": "protocol:provider-implementer",
            "protocol_digest": _digest("protocol"),
        },
    )


def test_enriched_provider_draft_projects_to_open_research_spec() -> None:
    draft = _open_proposal("frontier_architect")
    spec, facts = project_open_producer_draft(
        draft,
        bindings={
            "protocol_ref": "protocol:provider-fixture",
            "protocol_digest": _digest("protocol"),
            "context_ref": "context:provider-fixture",
            "context_digest": _digest("context"),
            "current_profile_ref": "profile:provider-fixture",
            "current_profile_digest": _digest("profile"),
            "implementation_requirements": (
                "RecBole general recommender interface",
                "candidate-local package",
            ),
            "compatibility_requirements": (
                "general collaborative filtering",
                "pairwise input",
            ),
        },
        strict_resolution_contract=True,
    )

    assert spec.idea_mode is IdeaModeV1.FRONTIER_HYPOTHESIS
    assert spec.realization_mode is RealizationModeV1.NON_NESTED
    assert spec.research_question is not None
    assert spec.closest_parent == "BL-ICF executable profile"
    assert spec.causal_chain
    assert spec.discriminative_predictions
    assert spec.resource_hypothesis is not None
    assert spec.execution_contract["capability_family"] == "OPEN_INTERACTION_CUSTOM"
    assert "claim ceiling" in spec.high_change_justification
    assert facts["required_budget"]["implementation_token_ceiling"] == 20000


def test_provider_schema_rejects_missing_enriched_field() -> None:
    resource_root = Path(fresh_r1.__file__).resolve().parent / "resources"
    schema = json.loads(
        (resource_root / "research_line_open_spec_proposal_response_v1.schema.json")
        .read_text(encoding="utf-8")
    )
    response = _proposal_response("frontier_architect")
    del response["proposals"][0]["research_question"]

    with pytest.raises(jsonschema.ValidationError):
        fresh_r1.validate_v4_response_contract(response, provider_schema=schema)


def test_provider_schema_is_strict_and_rejects_open_config_object() -> None:
    resource_root = Path(fresh_r1.__file__).resolve().parent / "resources"
    schema = json.loads(
        (resource_root / "research_line_open_spec_proposal_response_v1.schema.json")
        .read_text(encoding="utf-8")
    )

    validate_provider_strict_schema(schema)
    with pytest.raises(CanaryBrokerError, match="allows extra properties"):
        validate_provider_strict_schema(
            {"type": "object", "additionalProperties": True}
        )


def test_research_producer_consumes_complete_role_context_and_separates_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_bounded_provider_call(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        calls.append(kwargs)
        role = str(kwargs["logical_call_id"]).split(":")[-2]
        return _call_result(
            logical_call_id=str(kwargs["logical_call_id"]),
            response=_proposal_response(role),
        )

    monkeypatch.setattr(
        provider_module.fresh_r1,
        "bounded_provider_call",
        fake_bounded_provider_call,
    )
    producer = ProviderResearchProducer(
        config_source=_config(),
        call_root=tmp_path,
        session_id="research-session-1",
    )

    for role in DISCOVERY_PRODUCERS:
        draft = producer(role, _context_view(role))
        assert draft["producer_role"] == role
        assert draft["execution_contract"]["config"] == {}
        assert "config_json" not in draft["execution_contract"]

    assert len(calls) == 4
    assert len({call["logical_call_id"] for call in calls}) == 4
    for call in calls:
        prompt = call["prompt"]
        assert "kb-round-1" in prompt
        assert "goal-round-1" in prompt
        assert "axis-round-1" in prompt
        assert "memory-" in prompt
        assert "budget-round-1" in prompt
        assert "target-round-1" in prompt
        assert "profile:provider-fixture" in prompt
        for field_name in (
            "idea_mode",
            "research_question",
            "closest_parent",
            "causal_chain",
            "discriminative_predictions",
            "resource_hypothesis",
            "realization_mode",
        ):
            assert field_name in prompt
        assert "llm-config:ordered-primary" not in prompt
    assert len(producer.call_traces) == 4


@pytest.mark.parametrize(
    ("wire_value", "normalized_value"),
    (
        ("LightGCN", "LightGCN"),
        ("LightGCN.yaml", "LightGCN"),
        ("LightGCN.yml", "LightGCN"),
    ),
)
def test_provider_research_producer_normalizes_model_config_identifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wire_value: str,
    normalized_value: str,
) -> None:
    def fake_bounded_provider_call(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        role = str(kwargs["logical_call_id"]).split(":")[-2]
        return _call_result(
            logical_call_id=str(kwargs["logical_call_id"]),
            response=_proposal_response(
                role,
                base_model_config=wire_value,
            ),
        )

    monkeypatch.setattr(
        provider_module.fresh_r1,
        "bounded_provider_call",
        fake_bounded_provider_call,
    )
    producer = ProviderResearchProducer(
        config_source=_config(),
        call_root=tmp_path,
        session_id="research-session-model-config",
    )

    proposal = producer("frontier_architect", _context_view("frontier_architect"))

    assert proposal["execution_contract"]["base_model_config"] == normalized_value
    assert proposal["execution_contract"]["config"] == {}
    assert "config_json" not in proposal["execution_contract"]


@pytest.mark.parametrize(
    "wire_value",
    (
        "LightGCN.json",
        "configs/LightGCN.yaml",
        "LightGCN.yaml.bak",
        "Light-GCN",
    ),
)
def test_provider_research_producer_rejects_non_identifier_model_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wire_value: str,
) -> None:
    def fake_bounded_provider_call(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        role = str(kwargs["logical_call_id"]).split(":")[-2]
        return _call_result(
            logical_call_id=str(kwargs["logical_call_id"]),
            response=_proposal_response(
                role,
                base_model_config=wire_value,
            ),
        )

    monkeypatch.setattr(
        provider_module.fresh_r1,
        "bounded_provider_call",
        fake_bounded_provider_call,
    )
    producer = ProviderResearchProducer(
        config_source=_config(),
        call_root=tmp_path,
        session_id="research-session-invalid-model-config",
    )

    with pytest.raises(
        fresh_r1.FreshR1Error,
        match="bare model-config identifier",
    ):
        producer("frontier_architect", _context_view("frontier_architect"))


def test_research_producer_gives_shadow_policies_distinct_logical_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_bounded_provider_call(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        calls.append(kwargs)
        role = str(kwargs["logical_call_id"]).split(":")[-2]
        return _call_result(
            logical_call_id=str(kwargs["logical_call_id"]),
            response=_proposal_response(role),
        )

    monkeypatch.setattr(
        provider_module.fresh_r1,
        "bounded_provider_call",
        fake_bounded_provider_call,
    )
    producer = ProviderResearchProducer(
        config_source=_config(),
        call_root=tmp_path,
        session_id="research-session-shadow",
    )
    role = "frontier_architect"
    view = _context_view(role)

    producer.call_with_namespace(
        role,
        view,
        logical_namespace="offline-replay:champion:aaa",
    )
    producer.call_with_namespace(
        role,
        view,
        logical_namespace="offline-replay:challenger:bbb",
    )

    assert len(calls) == 2
    assert calls[0]["prompt"] == calls[1]["prompt"]
    assert calls[0]["logical_call_id"] != calls[1]["logical_call_id"]
    assert "offline-replay:champion:aaa" in calls[0]["logical_call_id"]
    assert "offline-replay:challenger:bbb" in calls[1]["logical_call_id"]


def test_research_producer_keeps_attempts_usage_and_config_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attempts = (
        {
            "ordinal": 1,
            "status": "FAILED",
            "request_digest": _digest("attempt-1"),
            "input_tokens": 10,
            "output_tokens": 0,
            "billed_tokens": 10,
        },
        {
            "ordinal": 2,
            "status": "SUCCESS",
            "request_digest": _digest("attempt-2"),
            "input_tokens": 13,
            "output_tokens": 7,
            "billed_tokens": 20,
        },
    )

    def fake_bounded_provider_call(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        return _call_result(
            logical_call_id=str(kwargs["logical_call_id"]),
            response=_proposal_response("mechanism_composer"),
            attempts=attempts,
        )

    monkeypatch.setattr(
        provider_module.fresh_r1,
        "bounded_provider_call",
        fake_bounded_provider_call,
    )
    producer = ProviderResearchProducer(
        config_source=_config(),
        call_root=tmp_path,
        session_id="research-session-usage",
    )
    producer("mechanism_composer", _context_view("mechanism_composer"))

    trace = producer.last_call_trace
    assert trace is not None
    assert trace["attempts"] == list(attempts)
    assert trace["usage"] == {
        "input_tokens": 13,
        "cached_input_tokens": 2,
        "output_tokens": 7,
        "billed_tokens": 20,
    }
    assert trace["config_identity"]["source_ref"] == "llm-config:ordered-primary"
    assert trace["config_identity"]["release_digest"] == _config()[
        "release_digest"
    ]


def test_provider_failure_raises_without_fabricating_a_proposal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_bounded_provider_call(**_kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        return fresh_r1.ProviderAttemptResult(
            call=None,
            attempts=(
                {"ordinal": 1, "status": "FAILED", "billed_tokens": 3},
            ),
            failure={"failure_class": "PROVIDER", "reason_code": "TEST_FAILURE"},
        )

    monkeypatch.setattr(
        provider_module.fresh_r1,
        "bounded_provider_call",
        fake_bounded_provider_call,
    )
    producer = ProviderResearchProducer(
        config_source=_config(),
        call_root=tmp_path,
        session_id="research-session-failure",
    )
    with pytest.raises(fresh_r1.FreshR1Error):
        producer("frontier_architect", _context_view("frontier_architect"))
    assert producer.last_call_trace is not None
    assert producer.last_call_trace["receipt"]["status"] == "FAILED"
    assert producer.last_call_trace["failure"]["reason_code"] == "TEST_FAILURE"


def test_implementer_gateway_uses_origin_blind_request_and_returns_materializer_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_bounded_provider_call(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        calls.append(kwargs)
        return _call_result(
            logical_call_id=str(kwargs["logical_call_id"]),
            response=_implementation_response(),
        )

    monkeypatch.setattr(
        provider_module.fresh_r1,
        "bounded_provider_call",
        fake_bounded_provider_call,
    )
    spec = OpenResearchSpecV1(
        hypothesis="A candidate-local mechanism changes ranking.",
        mechanism_change="Add a trainable interaction.",
        competing_explanation="The effect is inherited capacity.",
        matched_control_requirement="Use the frozen pairwise parent.",
        implementation_requirements=("candidate-local package",),
        expected_evidence=("API contract",),
        falsifier="The interaction is absent.",
        compatibility_requirements=("general collaborative filtering",),
        protocol_ref="protocol:provider-implementer",
        protocol_digest=_digest("protocol"),
        context_ref="context:provider-implementer",
        context_digest=_digest("context"),
        current_profile_ref="profile:provider-implementer",
        current_profile_digest=_digest("profile"),
        producer_role="mechanism_composer",
        high_change_justification="Outside the frozen profile.",
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )
    request = build_shared_implementer_request(
        spec,
        policy=_implementation_policy(),
    )
    gateway = ProviderImplementerGateway(
        config_source=_config(),
        call_root=tmp_path,
        session_id="implementer-session-1",
    )

    implementation = gateway(request)

    assert implementation["entrypoint"] == "recclaw_ext.candidate:FreshCandidateModel"
    assert len(calls) == 1
    assert calls[0]["logical_call_id"].endswith(":implementation:0")
    assert calls[0]["call_root"].name == "revision_00"
    prompt = calls[0]["prompt"]
    assert request["blind_candidate_id"] in prompt
    assert "producer_role" not in prompt
    assert "context:provider-implementer" not in prompt
    assert "execution_contract" in prompt
    assert '"base_model_config":"LightGCN"' in prompt
    assert "subclass recbole.model.general_recommender.bpr.BPR" not in prompt
    assert "api_key" not in prompt.lower()
    assert gateway.last_call_trace is not None
    assert gateway.last_call_trace["usage"]["billed_tokens"] == 20

    gateway({**request, "repair_attempt": 1})
    assert calls[1]["logical_call_id"].endswith(":implementation:1")
    assert calls[1]["call_root"].name == "revision_01"


def test_implementer_prompt_has_no_family_default_for_frozen_execution_contract(
    tmp_path: Path,
) -> None:
    del tmp_path
    template_path = (
        Path(fresh_r1.__file__).resolve().parent
        / "resources"
        / "research_line_implementer_prompt_v1.txt"
    )
    template = template_path.read_text(encoding="utf-8")
    for family in ("BPR", "LightGCN", "GeneralRecommender"):
        prompt = fresh_r1.render_implementation_prompt(
            template,
            {
                "blind_candidate_id": "innovation-candidate-family-test",
                "blind_research_spec": {"hypothesis": "fixture"},
                "candidate_local_write_allowlist": (
                    "recclaw_ext/__init__.py",
                    "recclaw_ext/candidate.py",
                ),
                "schema": "recclaw.shared-implementer-request.v1",
                "service_policy": {
                    "execution_contract": {
                        "base_model_config": family,
                        "model": family,
                    }
                },
            },
        )
        assert "execution_contract" in prompt
        assert family in prompt
        assert "subclass recbole.model.general_recommender.bpr.BPR" not in prompt
        assert "frozen BL-ICF protocol" in prompt
        assert "SciPy 1.15.3" in prompt
        assert "dok_matrix._update" in prompt
