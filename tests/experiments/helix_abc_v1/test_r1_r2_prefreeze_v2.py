from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
)
from recclaw_core.experiments.helix_abc_v1.prefreeze_v2 import (
    BLOCKED_REL,
    MANIFEST_REL,
    POLICY_REL,
    PROBE_REL,
    V1_BLOCKED_REL,
    V1_BLOCKED_SHA256,
    V1_MANIFEST_REL,
    V1_MANIFEST_SHA256,
    V1_PROBE_REL,
    V1_PROBE_SHA256,
    exact_probe_request_payload,
    expected_retry_policy,
    provider_free_dry_run,
    schema_probe_prompt,
    validate_prefreeze_v2,
    validate_prefreeze_v2_payload,
)
from recclaw_core.experiments.helix_abc_v1.wave2_integration import (
    Wave2IntegrationError,
)


ROOT = Path(__file__).resolve().parents[3]
REPROBE_SCRIPT = ROOT / "scripts/reprobe_fresh_open_spec_endpoint_v2.py"


def test_v1_negative_evidence_remains_byte_identical() -> None:
    expected = {
        V1_MANIFEST_REL: V1_MANIFEST_SHA256,
        V1_BLOCKED_REL: V1_BLOCKED_SHA256,
        V1_PROBE_REL: V1_PROBE_SHA256,
    }
    for relative, digest in expected.items():
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_v2_retry_policy_is_bounded_same_slot_and_fail_closed() -> None:
    policy_path = ROOT / POLICY_REL
    policy = json.loads(policy_path.read_bytes())

    assert policy_path.read_bytes() == canonical_json_bytes(policy)
    assert policy == expected_retry_policy()
    assert policy["proposal_slots_per_side"] == 8
    assert policy["proposal_denominator_per_side"] == 8
    assert policy["retry_limit"] == {
        "deterministic_backoff_ms_after_failure": [1000, 3000],
        "maximum_additional_physical_attempts": 2,
        "maximum_total_physical_attempts_per_slot": 3,
    }
    assert policy["slot_attempt_semantics"]["retry_is_same_slot"] is True
    assert policy["slot_attempt_semantics"]["retry_is_new_proposal"] is False
    assert policy["slot_attempt_semantics"]["same_request_payload_digest_required"] is True
    assert policy["response_contract_rules"]["content_not_json_retry"] == "FORBIDDEN"
    assert policy["exhausted_slot_missingness"]["mechanism_negative_evidence"] is False
    assert policy["special_v2_diagnostic_reprobe"]["retry_count"] == 0


def test_v2_reprobe_payload_is_exact_v1_schema_sentinel_and_no_tools() -> None:
    spec = importlib.util.spec_from_file_location(
        "v1_probe_script",
        ROOT / "scripts/probe_fresh_open_spec_endpoint.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentinel = json.loads((ROOT / module.SENTINEL_PATH.relative_to(ROOT)).read_bytes())
    payload = exact_probe_request_payload(ROOT)

    assert schema_probe_prompt(ROOT) == module._prompt(sentinel)
    assert payload["model"] == "gpt-5.4"
    assert payload["max_tokens"] == module.PROBE_TOKEN_CEILING
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert payload["response_format"]["json_schema"]["strict"] is True
    schema_text = canonical_json_bytes(
        payload["response_format"]["json_schema"]["schema"]
    ).decode("utf-8")
    assert "uniqueItems" in schema_text


def test_v2_manifest_validator_binds_ab_identity_and_is_provider_free() -> None:
    manifest = validate_prefreeze_v2(ROOT)
    call = manifest["future_r1_provider_call_contract"]
    dry_run = provider_free_dry_run(ROOT)

    assert call["proposal_slots_per_side"] == 8
    assert call["proposal_denominator_per_side"] == 8
    assert call["side_a_call_contract_digest"] == call["shared_call_contract_digest"]
    assert call["side_b_call_contract_digest"] == call["shared_call_contract_digest"]
    assert manifest["diagnostic_reprobe"]["maximum_physical_provider_calls"] == 1
    assert manifest["diagnostic_reprobe"]["retry_count"] == 0
    assert dry_run["provider_calls"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["candidate_admissions"] == 0
    assert dry_run["outcomes_consumed"] == 0
    assert dry_run["held_out_reads"] == 0


def test_v2_validator_rejects_retry_scope_or_denominator_mutation() -> None:
    payload = json.loads((ROOT / MANIFEST_REL).read_bytes())
    payload["future_r1_provider_call_contract"]["proposal_denominator_per_side"] = 7
    with pytest.raises(Wave2IntegrationError):
        validate_prefreeze_v2_payload(ROOT, payload)


def test_v2_reprobe_script_has_one_provider_call_and_no_candidate_path() -> None:
    tree = ast.parse(REPROBE_SCRIPT.read_text(encoding="utf-8"))
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


def test_checked_in_v2_outcome_never_claims_research_side_effects() -> None:
    probe_path = ROOT / PROBE_REL
    if not probe_path.exists():
        pytest.skip("V2 re-probe has not been executed")
    probe = json.loads(probe_path.read_bytes())
    assert probe["physical_provider_calls"] == 1
    assert probe["retry_count"] == 0
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
        assert probe[field] == 0
    if (ROOT / BLOCKED_REL).exists():
        blocked = json.loads((ROOT / BLOCKED_REL).read_bytes())
        assert blocked["r1_worker_launch_authorized"] is False
        assert blocked["third_probe_forbidden"] is True
