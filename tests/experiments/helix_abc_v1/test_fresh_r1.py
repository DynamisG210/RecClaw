from __future__ import annotations

from pathlib import Path

import jsonschema

from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    PROTOCOL_REQUIREMENTS,
    evaluate_gate,
    render_implementation_prompt,
    render_proposal_prompt,
    retry_eligible,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"


def test_formal_implementation_schema_is_strict_and_candidate_local() -> None:
    schema = __import__("json").loads(
        (RESOURCE_ROOT / "fresh_r1_implementation_response_v1.schema.json").read_bytes()
    )
    jsonschema.validators.validator_for(schema).check_schema(schema)
    response = {
        "schema": "recclaw.research-line.fresh-r1-implementation-response.v1",
        "proposals": [
            {
                "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
                "files": [
                    {"path": "recclaw_ext/__init__.py", "content": "# fresh\n"},
                    {
                        "path": "recclaw_ext/candidate.py",
                        "content": "class FreshCandidateModel:\n    pass\n",
                    },
                ],
                "implementation_summary": "fixture",
            }
        ],
    }
    jsonschema.validate(response, schema)


def test_prompts_preserve_slot_role_and_blind_implementation_boundary() -> None:
    proposal_template = (
        RESOURCE_ROOT / "fresh_open_spec_proposal_prompt_v1.txt"
    ).read_text(encoding="utf-8")
    rendered = render_proposal_prompt(
        proposal_template,
        side_identity="recclaw-fresh-r1-side-a-v1",
        logical_slot_id="slot-01",
        proposal_seed=41001,
        producer_role="mechanism_composer",
    )
    assert "slot-01" in rendered
    assert "41001" in rendered
    assert "mechanism_composer" in rendered
    assert all(value in rendered for value in PROTOCOL_REQUIREMENTS)
    implementation_template = (
        RESOURCE_ROOT / "fresh_r1_implementer_prompt_v1.txt"
    ).read_text(encoding="utf-8")
    implementation = render_implementation_prompt(
        implementation_template,
        {
            "blind_candidate_id": "innovation-candidate-123",
            "blind_research_spec": {"hypothesis": "new relation"},
            "candidate_local_write_allowlist": ["recclaw_ext/candidate.py"],
            "schema": "recclaw.shared-implementer-request.v1",
            "service_policy": {"candidate_local_write_only": True},
        },
    )
    assert "innovation-candidate-123" in implementation
    assert "side_a" not in implementation
    assert "producer_role" not in implementation


def test_retry_policy_is_exactly_transient_only() -> None:
    assert retry_eligible({"http_status": 408})
    assert retry_eligible({"http_status": 429})
    assert retry_eligible({"http_status": 503})
    assert retry_eligible({"failure_class": "TIMEOUT"})
    assert retry_eligible({"exception_type": "ConnectionResetError"})
    assert not retry_eligible({"http_status": 400})
    assert not retry_eligible({"http_status": 401})
    assert not retry_eligible({"failure_class": "SCHEMA_VALIDATION_FAILURE"})
    assert not retry_eligible({"exception_type": "URLError"})


def test_gate_requires_both_sides_and_real_behavior_change() -> None:
    records = {}
    for side in ("side_a", "side_b"):
        records[side] = [
            {
                "producer_role": "mechanism_composer" if index % 2 == 0 else "frontier_architect",
                "qualification_status": "PASS" if index < 2 else None,
                "real_mechanism_change": index == 0,
                "spec_digest": f"{side}-{index}",
            }
            for index in range(4)
        ]
    assert evaluate_gate(records)["pass"] is True
    records["side_b"][1]["qualification_status"] = "FAIL"
    assert evaluate_gate(records)["pass"] is False
