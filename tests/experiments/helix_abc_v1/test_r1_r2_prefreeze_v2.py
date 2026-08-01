from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
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
    SQLITE_CALL_COLUMNS,
    V3_ATTEMPT_RECEIPT_REL,
    V3_AUTH_REL,
    V3_BLOCKED_REL,
    V3_MANIFEST_REL,
    V3_POLICY_REL,
    V3_READY_REL,
    exact_probe_request_payload,
    expected_retry_policy,
    expected_v3_retry_policy,
    provider_free_dry_run,
    provider_free_v3_dry_run,
    schema_probe_prompt,
    validate_prefreeze_v2,
    validate_prefreeze_v2_payload,
    validate_prefreeze_v3,
    validate_prefreeze_v3_payload,
    verify_v2_seal,
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

    assert len(calls) == 3
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


def test_v2_artifacts_are_sealed_before_v3() -> None:
    sealed = verify_v2_seal(ROOT)
    assert len(sealed) == 6
    for relative, digest in sealed.items():
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_v3_policy_is_exact_bounded_retry_not_new_proposals() -> None:
    policy_path = ROOT / V3_POLICY_REL
    policy = json.loads(policy_path.read_bytes())

    assert policy_path.read_bytes() == canonical_json_bytes(policy)
    assert policy == expected_v3_retry_policy()
    diagnostic = policy["diagnostic_slot"]
    assert diagnostic["maximum_total_physical_attempts"] == 3
    assert diagnostic["maximum_additional_physical_attempts"] == 2
    assert diagnostic["deterministic_backoff_ms_after_failure"] == [1000, 3000]
    assert diagnostic["same_logical_call_id_required"] is True
    assert diagnostic["same_request_payload_digest_required"] is True
    assert diagnostic["successful_response_selection"] == "FORBIDDEN"
    assert policy["proposal_slots_per_side"] == 8
    assert policy["proposal_denominator_per_side"] == 8
    assert policy["missingness"]["mechanism_negative_evidence"] is False


def test_v3_manifest_is_fresh_preoutcome_and_provider_free() -> None:
    manifest = validate_prefreeze_v3(ROOT)
    dry_run = provider_free_v3_dry_run(ROOT)
    contract = manifest["exact_provider_contract"]
    retry = manifest["bounded_retry"]
    future = manifest["future_r1_scientific_contract"]

    assert manifest["attempt_identity"]["distinct_from_v1_v2_attempts"] is True
    assert manifest["attempt_identity"]["old_attempt_call_db_identity_reuse"] is False
    assert contract["model"] == "gpt-5.4"
    assert contract["request_payload_digest"] == sha256_digest(
        exact_probe_request_payload(ROOT)
    )
    assert contract["response_schema_contains_unique_items"] is True
    assert retry["maximum_physical_attempts"] == 3
    assert len(
        {
            item["physical_attempt_identity_digest"]
            for item in retry["physical_attempt_identities"]
        }
    ) == 3
    assert future["proposal_slots_per_side"] == 8
    assert future["proposal_denominator_per_side"] == 8
    assert future["side_a_call_contract_digest"] == future[
        "shared_call_contract_digest"
    ]
    assert future["side_b_call_contract_digest"] == future[
        "shared_call_contract_digest"
    ]
    assert dry_run["provider_calls"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["held_out_reads"] == 0


def test_v3_validator_rejects_attempt_or_denominator_mutation() -> None:
    payload = json.loads((ROOT / V3_MANIFEST_REL).read_bytes())
    payload["future_r1_scientific_contract"]["proposal_denominator_per_side"] = 9
    with pytest.raises(Wave2IntegrationError):
        validate_prefreeze_v3_payload(ROOT, payload)


def test_v3_sqlite_schema_identity_matches_existing_broker_schema() -> None:
    db_path = Path(
        "/root/projects/RecClaw_r1_prefreeze_reprobe_v2_private/broker.sqlite3"
    )
    connection = sqlite3.connect(db_path)
    try:
        columns = tuple(
            str(row[1])
            for row in connection.execute("PRAGMA table_info(calls)").fetchall()
        )
    finally:
        connection.close()
    assert columns == SQLITE_CALL_COLUMNS


def test_checked_in_v3_outcome_is_bounded_ordered_and_side_effect_free() -> None:
    attempt_path = ROOT / V3_ATTEMPT_RECEIPT_REL
    if not attempt_path.exists():
        pytest.skip("V3 Provider diagnostic has not been executed")
    receipt = json.loads(attempt_path.read_bytes())
    attempts = receipt["physical_attempts"]

    assert receipt["physical_provider_calls"] == len(attempts)
    assert 1 <= len(attempts) <= 3
    assert receipt["retry_count"] == len(attempts) - 1
    assert [item["ordinal"] for item in attempts] == list(
        range(1, len(attempts) + 1)
    )
    assert len({item["request_envelope_digest"] for item in attempts}) == 1
    assert len({item["request_payload_digest"] for item in attempts}) == 1
    prior_digest = None
    prior_end_ns = None
    required_gap_ms = 0
    for item in attempts:
        preimage = dict(item)
        observed_digest = preimage.pop("attempt_digest")
        assert preimage["prior_attempt_digest"] == prior_digest
        assert sha256_digest(preimage) == observed_digest
        assert item["monotonic_start_ns"] < item["monotonic_end_ns"]
        if prior_end_ns is not None:
            assert (
                item["monotonic_start_ns"] - prior_end_ns
                >= required_gap_ms * 1_000_000
            )
        prior_digest = observed_digest
        prior_end_ns = item["monotonic_end_ns"]
        required_gap_ms = item["backoff_ms_after_attempt"]
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
    assert not ((ROOT / V3_BLOCKED_REL).exists() and (ROOT / V3_READY_REL).exists())
