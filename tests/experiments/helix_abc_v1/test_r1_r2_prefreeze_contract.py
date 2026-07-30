from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import jsonschema
import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    validate_provider_strict_schema,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    frozen_search_resolver_environment,
    project_open_producer_draft,
    resolve_capability,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityResolutionResultV1,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = (
    ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
SCHEMA_PATH = RESOURCE_ROOT / "fresh_open_spec_proposal_response_v1.schema.json"
SENTINEL_PATH = RESOURCE_ROOT / "fresh_open_spec_schema_probe_payload_v1.json"
SCHEDULE_PATH = RESOURCE_ROOT / "fresh_open_spec_call_schedule_v1.json"
FAILURE_PATH = RESOURCE_ROOT / "fresh_open_spec_failure_policy_v1.json"
TOOL_PATH = RESOURCE_ROOT / "fresh_open_spec_tool_policy_v1.json"
PROMPT_PATH = RESOURCE_ROOT / "fresh_open_spec_proposal_prompt_v1.txt"
POLICY_PATH = ROOT / "docs/research_line/vnext/R1_R2_SCIENTIFIC_POLICY_V1.json"
IDENTITY_PATH = ROOT / "docs/research_line/vnext/R1_R2_FRESH_IDENTITY_PLAN_V1.json"
PROBE_SCRIPT = ROOT / "scripts/probe_fresh_open_spec_endpoint.py"
PROBE_RECEIPT = (
    ROOT
    / "docs/research_line/vnext/"
    "FRESH_OPEN_SPEC_ENDPOINT_PROBE_RECEIPT_V1.json"
)


def _draft(role: str) -> dict[str, object]:
    return {
        "producer_role": role,
        "hypothesis": "A new interaction head changes ranking gradients.",
        "mechanism_change": "Add a candidate-local relation-conditioned interaction head.",
        "competing_explanation": "Any signal comes only from added parameter count.",
        "matched_control_requirement": "Match parameter count without the relation-conditioned gate.",
        "implementation_requirements": [
            "RecBole general recommender interface",
            "candidate-local package",
        ],
        "expected_evidence": [
            "The gated head changes loss and predict outputs versus its matched control."
        ],
        "falsifier": "The matched control reproduces the same output changes.",
        "compatibility_requirements": [
            "general collaborative filtering",
            "pairwise input",
        ],
        "high_change_justification": "The frozen 66 contains no relation-conditioned interaction head.",
        "current_profile_expressibility_claim": "NOT_EXPRESSIBLE",
        "resolution_facts": {
            "requested_current_semantics_digest": None,
            "capability_diff": ["relation-conditioned interaction head"],
            "high_change_dimensions": ["INTERACTION_STRUCTURE"],
            "required_dependencies": ["recbole-runtime"],
            "required_budget": {
                "implementation_token_ceiling": 20_000,
                "qualification_gpu_minutes": 10,
                "qualification_wall_minutes": 30,
            },
        },
    }


def test_fresh_response_schema_is_strict_and_probe_echo_is_pre_authored() -> None:
    schema = json.loads(SCHEMA_PATH.read_bytes())
    sentinel = json.loads(SENTINEL_PATH.read_bytes())

    jsonschema.validators.validator_for(schema).check_schema(schema)
    validate_provider_strict_schema(schema)
    jsonschema.validate(sentinel, schema)
    assert sentinel["proposals"][0]["hypothesis"].startswith(
        "Qualification-only schema sentinel"
    )
    assert set(schema["properties"]) == {"schema", "proposals"}
    proposal_fields = set(
        schema["properties"]["proposals"]["items"]["properties"]
    )
    assert not {
        "candidate_id",
        "mechanism_id",
        "mechanism_program",
        "operator",
        "parameter_overrides",
    } & proposal_fields


@pytest.mark.parametrize(
    "role",
    [
        "mechanism_composer",
        "lineage_refiner",
        "falsification_designer",
        "frontier_architect",
    ],
)
def test_all_existing_producer_roles_mechanically_project_to_innovation_required(
    role: str,
) -> None:
    draft = _draft(role)
    response = {
        "schema": "recclaw.research-line.fresh-open-spec-proposal-response.v1",
        "proposals": [draft],
    }
    schema = json.loads(SCHEMA_PATH.read_bytes())
    jsonschema.validate(response, schema)
    bindings = frozen_search_bindings(
        context_ref="fresh-r1-r2-prefreeze-context-v1",
        context_digest=sha256_digest(
            {"context": "fresh-r1-r2-prefreeze-context-v1"}
        ),
    )
    spec, facts = project_open_producer_draft(draft, bindings=bindings)
    resolution = resolve_capability(
        spec,
        resolution_facts=facts,
        environment=frozen_search_resolver_environment(
            available_dependencies=("recbole-runtime",),
            budget_limits={
                "implementation_token_ceiling": 20_000,
                "qualification_gpu_minutes": 10,
                "qualification_wall_minutes": 30,
            },
        ),
    )

    assert spec.producer_role == role
    assert (
        resolution.resolution
        is CapabilityResolutionResultV1.INNOVATION_REQUIRED
    )
    assert resolution.catalog_fallback_used is False
    assert resolution.no_silent_fallback is True


def test_call_schedule_and_failure_policy_are_fixed_and_balanced() -> None:
    schedule = json.loads(SCHEDULE_PATH.read_bytes())
    failure = json.loads(FAILURE_PATH.read_bytes())
    tool = json.loads(TOOL_PATH.read_bytes())
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    assert schedule["call_count_per_side"] == 8
    assert schedule["proposal_budget_per_side"] == 8
    assert schedule["expected_proposals_per_call"] == 1
    assert schedule["token_budget_per_call"] == 6000
    counts = Counter(
        item["producer_role"] for item in schedule["role_schedule"]
    )
    assert counts == {
        "mechanism_composer": 2,
        "lineage_refiner": 2,
        "falsification_designer": 2,
        "frontier_architect": 2,
    }
    assert failure["no_retry"] is True
    assert failure["retry_count"] == 0
    assert (
        failure["content_not_json_action"]
        == "TERMINAL_CONSUME_PREASSIGNED_SLOT"
    )
    assert failure["replacement_call"] == "FORBIDDEN"
    assert failure["schema_relaxation"] == "FORBIDDEN"
    assert failure["successful_response_selection"] == "FORBIDDEN"
    assert tool["tools"] == []
    assert tool["tool_choice"] == "NONE"
    assert "fixed 66" not in prompt.lower()
    assert "not expressible by the frozen 66-entry profile" in prompt


def test_scientific_policy_and_identity_plan_are_preoutcome_and_isolated() -> None:
    policy = json.loads(POLICY_PATH.read_bytes())
    identity = json.loads(IDENTITY_PATH.read_bytes())
    threshold = policy["threshold_policy"]

    assert threshold["minimum_fresh_specs_per_side"] == 4
    assert threshold["minimum_producer_roles_per_side"] == 2
    assert threshold["minimum_qualified_capabilities_per_side"] == 2
    assert policy["analysis_plan"]["denominator_per_side"] == 8
    assert policy["missingness_policy"]["no_replacement_calls"] is True
    assert policy["missingness_policy"]["no_success_cherry_pick"] is True
    assert policy["qualification_gate"]["manual_candidate_patch_forbidden"] is True
    assert policy["qualification_gate"]["evidence_class"] == "DEVELOPMENT_ONLY"
    assert identity["freshness_assertions"]["legacy_attempt_identity_reuse"] is False

    r1 = identity["r1"]
    r2 = identity["r2"]
    for field in (
        "lineage",
        "memory_namespace",
        "outcome_namespace",
        "root",
        "coordinator_db",
        "candidate_namespace",
        "package_namespace",
    ):
        other = "db" if field == "coordinator_db" else field
        assert r1[field] != r2[other]
    assert set(r1["seed_plan"]["side_a_proposal_seed_by_slot"]).isdisjoint(
        r1["seed_plan"]["side_b_proposal_seed_by_slot"]
    )
    assert r1["side_a"] != r1["side_b"]
    assert (
        r1["side_a"]["origin_field_excluded_from_implementer_and_qualifier"]
        is True
    )
    assert (
        r1["side_b"]["origin_field_excluded_from_implementer_and_qualifier"]
        is True
    )


def test_probe_script_contains_one_transport_call_and_no_candidate_path() -> None:
    tree = ast.parse(PROBE_SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "call_with_session"
    ]
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert len(calls) == 1
    assert not any(
        name.endswith(
            (
                "open_spec",
                "innovation_spine",
                "innovation_recbole_adapter",
                "capability_admission",
            )
        )
        for name in imports
    )


def test_checked_in_probe_receipt_is_single_call_minimal_and_blocked() -> None:
    receipt = json.loads(PROBE_RECEIPT.read_bytes())

    assert receipt["status"] == "BLOCKED"
    assert receipt["classification"] == (
        "HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED"
    )
    assert receipt["physical_provider_calls"] == 1
    assert receipt["retry_count"] == 0
    assert receipt["returned_model"] is None
    assert receipt["authentication_status"] == (
        "UNVERIFIED_AFTER_SCHEMA_REJECTION"
    )
    assert receipt["blocked_fields"] == [
        "endpoint_authentication",
        "exact_fresh_open_spec_schema_support",
        "returned_model",
    ]
    for field in (
        "candidate_admissions",
        "candidate_roots_created",
        "held_out_reads",
        "open_specs_projected",
        "outcomes_consumed",
        "research_candidates_generated",
        "resolver_calls",
        "training_runs",
    ):
        assert receipt[field] == 0
    assert receipt["sensitive_headers_persisted"] is False
    assert receipt["sensitive_values_persisted"] is False
