from __future__ import annotations

import json
import sqlite3
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.prefreeze_v2 import (
    V4_MANIFEST_REL,
)
from recclaw_core.experiments.helix_abc_v1.prefreeze_v5 import (
    V4_SEALED_DIGESTS,
    V5_LOGICAL_CALL_ID,
    V5_ATTEMPT_RECEIPT_REL,
    V5_BLOCKED_REL,
    V5_PRIVATE_ROOT,
    V5_READY_REL,
    V5_SESSION_ID,
    diagnostic_reason_vocabulary,
    expected_prefreeze_v5_manifest,
    expected_v5_retry_policy,
    provider_free_v5_dry_run,
    validate_prefreeze_v5,
    verify_v4_seal,
    V5_SEALED_DIGESTS,
    V6_AUTH_REL,
    V6_ATTEMPT_RECEIPT_REL,
    V6_BLOCKED_REL,
    V6_MANIFEST_REL,
    V6_POLICY_REL,
    V6_PRIVATE_ROOT,
    V6_READY_REL,
    V6_RELEASE_REL,
    V6_REQUESTED_MODEL_ALIAS,
    V6_REQUIRED_RETURNED_SNAPSHOT,
    expected_prefreeze_v6_manifest,
    expected_v6_provider_release,
    provider_free_v6_dry_run,
    validate_prefreeze_v6,
    validate_v6_exact_model_pair,
    verify_v5_seal,
    V6_SEALED_DIGESTS,
    V7_AUTH_REL,
    V7_ATTEMPT_RECEIPT_REL,
    V7_BLOCKED_REL,
    V7_DIAGNOSTIC_TOKEN_CEILING,
    V7_MANIFEST_REL,
    V7_POLICY_REL,
    V7_PRIVATE_ROOT,
    V7_READY_REL,
    V7_RELEASE_REL,
    exact_v7_probe_request_payload_digest,
    expected_prefreeze_v7_manifest,
    expected_v7_provider_release,
    expected_v7_retry_policy,
    provider_free_v7_dry_run,
    validate_prefreeze_v7,
    verify_v6_seal,
    V7_SEALED_DIGESTS,
    V8_AUTH_REL,
    V8_ATTEMPT_RECEIPT_REL,
    V8_BLOCKED_REL,
    V8_DIAGNOSTIC_TOKEN_CEILING,
    V8_MANIFEST_REL,
    V8_MODEL_SNAPSHOT,
    V8_POLICY_REL,
    V8_PRIVATE_ROOT,
    V8_READY_REL,
    V8_RELEASE_REL,
    exact_v8_probe_request_payload,
    expected_prefreeze_v8_manifest,
    expected_v8_provider_release,
    expected_v8_retry_policy,
    provider_free_v8_dry_run,
    prefreeze_v8_runtime_spec,
    validate_prefreeze_v8,
    validate_v8_exact_snapshot_pair,
    verify_v7_seal,
    V8_SEALED_DIGESTS,
    V9_AUTH_REL,
    V9_ATTEMPT_RECEIPT_REL,
    V9_BLOCKED_REL,
    V9_MANIFEST_REL,
    V9_POLICY_REL,
    V9_PRIVATE_ROOT,
    V9_READY_REL,
    V9_RELEASE_REL,
    expected_prefreeze_v9_manifest,
    expected_v9_provider_release,
    expected_v9_retry_policy,
    prefreeze_v9_runtime_spec,
    provider_free_v9_dry_run,
    validate_prefreeze_v9,
    verify_v8_seal,
)
from scripts.build_r1_r2_prefreeze_v5_artifacts import _contract
from scripts.reprobe_fresh_open_spec_endpoint_v2 import (
    _classify_v5_failure,
    _prefreeze_diagnostic_contract,
)
from recclaw_core.experiments.helix_abc_v1.wave2_integration import (
    Wave2IntegrationError,
)


ROOT = Path(__file__).resolve().parents[3]


def _row(
    *,
    error_type: str,
    http_status: int | None,
    error_detail: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "receipt_json": json.dumps(
            {"http_status": http_status, "receipt_digest": "a" * 64}
        ),
        "error_detail_json": (
            json.dumps(error_detail) if error_detail is not None else None
        ),
        "outcome_json": None,
        "error_type": error_type,
    }


def test_v4_sealed_artifacts_are_exact_bytes() -> None:
    observed = verify_v4_seal(ROOT)
    assert observed == {
        path.as_posix(): digest
        for path, digest in V4_SEALED_DIGESTS.items()
    }
    for relative, digest in V4_SEALED_DIGESTS.items():
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_v5_changes_only_diagnostic_identity_and_observability() -> None:
    manifest = expected_prefreeze_v5_manifest(ROOT)
    v4 = json.loads((ROOT / V4_MANIFEST_REL).read_bytes())
    v4_provider = dict(v4["exact_provider_contract"])
    v5_provider = dict(manifest["exact_provider_contract"])
    for field in (
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    ):
        v4_provider.pop(field)
        v5_provider.pop(field)
    assert v5_provider == v4_provider
    assert manifest["exact_provider_contract"]["logical_call_id"] == V5_LOGICAL_CALL_ID
    assert (
        manifest["exact_provider_contract"]["proposal_generation_session_id"]
        == V5_SESSION_ID
    )
    assert (
        manifest["future_r1_scientific_contract"]
        == v4["future_r1_scientific_contract"]
    )
    assert (
        manifest["v4_scientific_contract_digest"]
        == manifest["v5_scientific_contract_digest"]
    )
    assert manifest["exact_provider_contract"]["model"] == "gpt-5.4"
    assert manifest["pre_outcome_counters"] == {
        "provider_calls": 0,
        "research_candidates_generated": 0,
        "open_specs_projected": 0,
        "resolver_calls": 0,
        "candidate_roots_created": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "training_runs": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
    }


def test_v5_policy_preserves_bounded_retry_and_denominator() -> None:
    policy = expected_v5_retry_policy(ROOT)
    slot = policy["diagnostic_slot"]
    assert slot["maximum_total_physical_attempts"] == 3
    assert slot["maximum_additional_physical_attempts"] == 2
    assert slot["deterministic_backoff_ms_after_failure"] == [1000, 3000]
    assert policy["proposal_slots_per_side"] == 8
    assert policy["proposal_denominator_per_side"] == 8
    assert policy["response_contract_failure"]["retry_eligible"] is False
    assert policy["response_contract_failure"]["manual_patch"] == "FORBIDDEN"


def test_checked_in_v5_artifacts_are_exact_and_provider_free() -> None:
    manifest = validate_prefreeze_v5(ROOT)
    dry_run = provider_free_v5_dry_run(ROOT)
    assert manifest == expected_prefreeze_v5_manifest(ROOT)
    assert dry_run["status"] == "PASS_PROVIDER_FREE_V5_OBSERVABILITY"
    assert dry_run["provider_calls"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["candidate_qualifications"] == 0
    assert dry_run["candidate_admissions"] == 0
    assert dry_run["outcomes_consumed"] == 0
    assert dry_run["held_out_reads"] == 0


@pytest.mark.parametrize(
    "reason_code",
    [item["reason_code"] for item in diagnostic_reason_vocabulary()],
)
def test_each_allowlisted_response_reason_is_terminal(reason_code: str) -> None:
    error_type = (
        "SCHEMA_VALIDATION_FAILURE"
        if reason_code == "SCHEMA_VALIDATION"
        else "RESPONSE_CONTRACT_ERROR"
    )
    evidence = _classify_v5_failure(
        error=ValueError("must-not-be-persisted"),
        row=_row(
            error_type=error_type,
            http_status=200,
            error_detail={"reason_code": reason_code},
        ),
    )
    assert evidence["response_contract_reason_code"] == reason_code
    assert evidence["retry_eligible"] is False


def test_non_allowlisted_or_content_bearing_reason_fails_closed() -> None:
    with pytest.raises(SystemExit, match="allowlisted reason code"):
        _classify_v5_failure(
            error=ValueError("must-not-be-persisted"),
            row=_row(
                error_type="RESPONSE_CONTRACT_ERROR",
                http_status=200,
                error_detail={
                    "reason_code": "UNKNOWN",
                    "raw_content": "forbidden",
                },
            ),
        )


@pytest.mark.parametrize("http_status", [408, 429, 500, 503])
def test_only_frozen_transient_http_classes_are_retry_eligible(
    http_status: int,
) -> None:
    evidence = _classify_v5_failure(
        error=RuntimeError("not persisted"),
        row=_row(
            error_type=f"HTTP_{http_status}",
            http_status=http_status,
            error_detail={"http_status": http_status},
        ),
    )
    assert evidence["retry_eligible"] is True


@pytest.mark.parametrize("http_status", [400, 401, 403, 404, 422])
def test_deterministic_http_classes_are_terminal(http_status: int) -> None:
    evidence = _classify_v5_failure(
        error=RuntimeError("not persisted"),
        row=_row(
            error_type=f"HTTP_{http_status}",
            http_status=http_status,
            error_detail={"http_status": http_status},
        ),
    )
    assert evidence["retry_eligible"] is False


def test_checked_in_v5_outcome_is_one_call_model_mismatch_and_side_effect_free() -> None:
    receipt_path = ROOT / V5_ATTEMPT_RECEIPT_REL
    blocked_path = ROOT / V5_BLOCKED_REL
    assert receipt_path.is_file()
    assert blocked_path.is_file()
    assert not (ROOT / V5_READY_REL).exists()
    receipt = json.loads(receipt_path.read_bytes())
    blocked = json.loads(blocked_path.read_bytes())
    assert receipt_path.read_bytes() == bytes(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True),
        "utf-8",
    )
    assert receipt["status"] == "BLOCKED"
    assert receipt["final_classification"] == "RETURNED_MODEL_MISMATCH"
    assert receipt["response_contract_reason_code"] is None
    assert receipt["physical_provider_calls"] == 1
    assert receipt["retry_count"] == 0
    assert len(receipt["physical_attempts"]) == 1
    attempt = receipt["physical_attempts"][0]
    assert attempt["http_status"] == 200
    assert attempt["retry_eligible"] is False
    assert attempt["returned_model"] != "gpt-5.4"
    assert attempt["local_uniqueness_status"] == "NOT_REACHED"
    assert blocked["status"] == "BLOCKED_PREFREEZE_V5"
    assert blocked["r1_worker_launch_authorized"] is False
    for field in (
        "research_candidates_generated",
        "open_specs_projected",
        "resolver_calls",
        "candidate_roots_created",
        "candidate_qualifications",
        "candidate_admissions",
        "training_runs",
        "outcomes_consumed",
        "held_out_reads",
    ):
        assert receipt[field] == 0
    connection = sqlite3.connect(
        V5_PRIVATE_ROOT / "physical_attempt_01" / "broker.sqlite3"
    )
    try:
        row = connection.execute(
            "SELECT COUNT(*), status, error_type FROM calls"
        ).fetchone()
    finally:
        connection.close()
    assert row == (1, "SUCCESS", None)


def test_v5_sealed_artifacts_remain_exact_bytes() -> None:
    observed = verify_v5_seal(ROOT)
    assert observed == {
        path.as_posix(): digest for path, digest in V5_SEALED_DIGESTS.items()
    }
    for relative, digest in V5_SEALED_DIGESTS.items():
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_v6_exact_model_pair_uses_two_literal_equalities() -> None:
    validate_v6_exact_model_pair(
        requested_model_alias=V6_REQUESTED_MODEL_ALIAS,
        returned_model=V6_REQUIRED_RETURNED_SNAPSHOT,
    )
    for alias, snapshot in (
        ("gpt-5.4 ", V6_REQUIRED_RETURNED_SNAPSHOT),
        ("gpt-5.5", V6_REQUIRED_RETURNED_SNAPSHOT),
        (V6_REQUESTED_MODEL_ALIAS, "gpt-5.4-2026-03-06"),
        (V6_REQUESTED_MODEL_ALIAS, "gpt-5.4"),
    ):
        with pytest.raises(Wave2IntegrationError):
            validate_v6_exact_model_pair(
                requested_model_alias=alias,
                returned_model=snapshot,
            )


@pytest.mark.parametrize(
    "field,mutated",
    [
        ("requested_model_alias", "gpt-5.4 "),
        ("required_returned_snapshot", "gpt-5.4-2026-03-06"),
        ("endpoint_digest", "0" * 64),
        ("credential_config_digest", "1" * 64),
        ("credential_identity_digest", "2" * 64),
        ("response_schema_digest", "3" * 64),
        ("local_uniqueness_contract_digest", "4" * 64),
        ("prompt_digest", "5" * 64),
        ("tool_policy_digest", "6" * 64),
        ("diagnostic_token_budget", 2001),
        ("future_r1_call_count_per_side", 9),
        ("future_r1_proposal_budget_per_side", 9),
        ("future_r1_token_budget_per_call", 6001),
    ],
)
def test_v6_release_digest_is_sensitive_to_every_frozen_binding(
    field: str, mutated: object
) -> None:
    release = expected_v6_provider_release(ROOT)
    expected_digest = release.pop("release_digest")
    changed = deepcopy(release)
    changed[field] = mutated
    assert sha256_digest(release) == expected_digest
    assert sha256_digest(changed) != expected_digest


def test_v6_changes_only_exact_pair_binding_and_shared_digest_identity() -> None:
    v5 = expected_prefreeze_v5_manifest(ROOT)
    v6 = expected_prefreeze_v6_manifest(ROOT)
    exact_same = (
        "endpoint_digest",
        "credential_config_digest",
        "credential_identity_digest",
        "response_schema_digest",
        "sentinel_digest",
        "request_payload_digest",
        "prompt_digest",
        "tool_policy_digest",
        "request_mode",
        "temperature",
        "token_budget",
    )
    for field in exact_same:
        assert v6["exact_provider_contract"][field] == v5[
            "exact_provider_contract"
        ][field]
    assert v6["authorized_v6_protocol_change"] == {
        "only_scientific_protocol_change": (
            "EXACT_REQUEST_ALIAS_TO_EXACT_RETURNED_SNAPSHOT_BINDING"
        ),
        "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
        "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
        "matching": "TWO_EXACT_LITERAL_EQUALITIES_FAIL_CLOSED",
        "generic_alias_acceptance": "FORBIDDEN",
    }
    assert (
        v6["accepted_v5_engineering_history"]["http_error_persistence"]
        == "STATUS_ONLY_NO_PROVIDER_BODY"
    )
    v5_future = dict(v5["future_r1_scientific_contract"])
    v6_future = dict(v6["future_r1_scientific_contract"])
    for field in (
        "shared_call_contract_digest",
        "side_a_call_contract_digest",
        "side_b_call_contract_digest",
    ):
        v5_future.pop(field)
        v6_future.pop(field)
    assert v6_future == v5_future


def test_v6_shared_probe_path_places_local_uniqueness_before_downstream() -> None:
    path = ROOT / "scripts/reprobe_fresh_open_spec_endpoint_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    shared_main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_main_prefreeze_diagnostic"
    )
    calls = {
        (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", "")
        ): node.lineno
        for node in ast.walk(shared_main)
        if isinstance(node, ast.Call)
        and (
            getattr(node.func, "id", None)
            in {"validate_v6_exact_model_pair", "validate_v4_response_contract"}
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "call_with_session"
            )
        )
    }
    assert calls["call_with_session"] < calls["validate_v6_exact_model_pair"]
    assert calls["validate_v6_exact_model_pair"] < calls[
        "validate_v4_response_contract"
    ]
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
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


def test_checked_in_v6_contract_artifacts_are_exact_and_provider_free() -> None:
    for relative in (V6_RELEASE_REL, V6_POLICY_REL, V6_MANIFEST_REL, V6_AUTH_REL):
        assert (ROOT / relative).is_file()
    manifest = validate_prefreeze_v6(ROOT)
    dry_run = provider_free_v6_dry_run(ROOT)
    assert manifest == expected_prefreeze_v6_manifest(ROOT)
    assert dry_run["status"] == "PASS_PROVIDER_FREE_V6_EXACT_MODEL_PAIR"
    assert dry_run["provider_calls"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["candidate_qualifications"] == 0
    assert dry_run["candidate_admissions"] == 0
    assert dry_run["outcomes_consumed"] == 0
    assert dry_run["held_out_reads"] == 0


def test_checked_in_v6_outcome_is_terminal_token_ceiling_and_side_effect_free() -> None:
    attempt_path = ROOT / V6_ATTEMPT_RECEIPT_REL
    blocked_path = ROOT / V6_BLOCKED_REL
    assert attempt_path.is_file()
    assert blocked_path.is_file()
    assert not (ROOT / V6_READY_REL).exists()
    receipt = json.loads(attempt_path.read_bytes())
    blocked = json.loads(blocked_path.read_bytes())
    assert receipt["status"] == "BLOCKED"
    assert receipt["final_classification"] == (
        "HTTP_200_RESPONSE_CONTRACT_TOKEN_CEILING"
    )
    assert receipt["response_contract_reason_code"] == "TOKEN_CEILING"
    assert receipt["physical_provider_calls"] == 1
    assert receipt["retry_count"] == 0
    attempt = receipt["physical_attempts"][0]
    assert attempt["http_status"] == 200
    assert attempt["retry_eligible"] is False
    assert attempt["response_contract_reason_code"] == "TOKEN_CEILING"
    assert attempt["local_uniqueness_status"] == "NOT_REACHED"
    assert blocked["status"] == "BLOCKED_PREFREEZE_V6"
    assert blocked["r1_worker_launch_authorized"] is False
    for field in (
        "research_candidates_generated",
        "open_specs_projected",
        "resolver_calls",
        "candidate_roots_created",
        "candidate_qualifications",
        "candidate_admissions",
        "training_runs",
        "outcomes_consumed",
        "held_out_reads",
    ):
        assert receipt[field] == 0
    connection = sqlite3.connect(
        V6_PRIVATE_ROOT / "physical_attempt_01" / "broker.sqlite3"
    )
    try:
        row = connection.execute(
            "SELECT COUNT(*), status, error_type, error_detail_json FROM calls"
        ).fetchone()
    finally:
        connection.close()
    assert row == (
        1,
        "FAILED",
        "RESPONSE_CONTRACT_ERROR",
        '{"reason_code":"TOKEN_CEILING"}',
    )


def test_v6_sealed_artifacts_remain_exact_bytes() -> None:
    observed = verify_v6_seal(ROOT)
    assert observed == {
        path.as_posix(): digest for path, digest in V6_SEALED_DIGESTS.items()
    }
    for relative, digest in V6_SEALED_DIGESTS.items():
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_v7_is_only_fresh_identity_and_diagnostic_ceiling_delta() -> None:
    v6 = json.loads((ROOT / V6_MANIFEST_REL).read_bytes())
    v7 = expected_prefreeze_v7_manifest(ROOT)
    release = expected_v7_provider_release(ROOT)
    assert V7_DIAGNOSTIC_TOKEN_CEILING == 6000
    assert release["diagnostic_token_budget"] == 6000
    assert release["transport_max_total_tokens_per_call"] == 6000
    assert release["future_r1_token_budget_per_call"] == 6000
    assert v7["future_r1_scientific_contract"] == v6[
        "future_r1_scientific_contract"
    ]
    assert v7["response_contract_equivalence"] == v6[
        "response_contract_equivalence"
    ]
    assert v7["preserved_scientific_identity"] == v6[
        "preserved_scientific_identity"
    ]
    exact_v6 = v6["exact_provider_contract"]
    exact_v7 = v7["exact_provider_contract"]
    for field in (
        "model",
        "requested_model_alias",
        "required_returned_snapshot",
        "endpoint_digest",
        "credential_config_digest",
        "credential_identity_digest",
        "response_schema_digest",
        "sentinel_digest",
        "prompt_digest",
        "tool_policy_digest",
        "request_mode",
        "temperature",
    ):
        assert exact_v7[field] == exact_v6[field]
    assert exact_v7["token_budget"] == 6000
    assert exact_v7["request_payload_digest"] == (
        exact_v7_probe_request_payload_digest(ROOT)
    )
    assert exact_v7["request_payload_digest"] != exact_v6[
        "request_payload_digest"
    ]


def test_v7_retry_policy_keeps_token_ceiling_terminal() -> None:
    policy = expected_v7_retry_policy(ROOT)
    slot = policy["diagnostic_slot"]
    assert slot["maximum_total_physical_attempts"] == 3
    assert slot["maximum_additional_physical_attempts"] == 2
    assert slot["deterministic_backoff_ms_after_failure"] == [1000, 3000]
    assert policy["response_contract_failure"]["retry_eligible"] is False
    assert "SEMANTIC_RESPONSE_CONTRACT_FAILURE" in slot[
        "terminal_no_retry_failure_classes"
    ]


def test_checked_in_v7_contract_artifacts_are_exact_and_provider_free() -> None:
    for relative in (V7_RELEASE_REL, V7_POLICY_REL, V7_MANIFEST_REL, V7_AUTH_REL):
        assert (ROOT / relative).is_file()
    manifest = validate_prefreeze_v7(ROOT)
    dry_run = provider_free_v7_dry_run(ROOT)
    assert manifest == expected_prefreeze_v7_manifest(ROOT)
    assert dry_run["status"] == (
        "PASS_PROVIDER_FREE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
    )
    assert dry_run["provider_calls"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["candidate_qualifications"] == 0
    assert dry_run["candidate_admissions"] == 0
    assert dry_run["outcomes_consumed"] == 0
    assert dry_run["held_out_reads"] == 0


def test_checked_in_v7_outcome_is_terminal_exact_pair_mismatch() -> None:
    attempt_path = ROOT / V7_ATTEMPT_RECEIPT_REL
    blocked_path = ROOT / V7_BLOCKED_REL
    assert attempt_path.is_file()
    assert blocked_path.is_file()
    assert not (ROOT / V7_READY_REL).exists()
    receipt = json.loads(attempt_path.read_bytes())
    blocked = json.loads(blocked_path.read_bytes())
    attempt = receipt["physical_attempts"][0]
    assert receipt["status"] == "BLOCKED"
    assert receipt["physical_provider_calls"] == 1
    assert receipt["retry_count"] == 0
    assert receipt["final_classification"] == "RETURNED_MODEL_PAIR_MISMATCH"
    assert attempt["http_status"] == 200
    assert attempt["returned_model"] == V6_REQUESTED_MODEL_ALIAS
    assert attempt["returned_model"] != V6_REQUIRED_RETURNED_SNAPSHOT
    assert attempt["retry_eligible"] is False
    assert attempt["local_uniqueness_status"] == "NOT_REACHED"
    assert blocked["status"] == "BLOCKED_PREFREEZE_V7"
    assert blocked["r1_worker_launch_authorized"] is False
    for field in (
        "research_candidates_generated",
        "open_specs_projected",
        "resolver_calls",
        "candidate_roots_created",
        "candidate_qualifications",
        "candidate_admissions",
        "training_runs",
        "outcomes_consumed",
        "held_out_reads",
    ):
        assert receipt[field] == 0
    connection = sqlite3.connect(
        V7_PRIVATE_ROOT / "physical_attempt_01" / "broker.sqlite3"
    )
    try:
        row = connection.execute(
            "SELECT COUNT(*),status,error_type,total_tokens,returned_model "
            "FROM calls"
        ).fetchone()
    finally:
        connection.close()
    assert row == (1, "SUCCESS", None, 2408, V6_REQUESTED_MODEL_ALIAS)


def test_v7_sealed_artifacts_remain_exact_bytes_for_v8() -> None:
    observed = verify_v7_seal(ROOT)
    for relative, digest in V7_SEALED_DIGESTS.items():
        assert observed[relative.as_posix()] == digest
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_v8_exact_snapshot_pair_has_no_alias_or_pattern_acceptance() -> None:
    validate_v8_exact_snapshot_pair(
        requested_model=V8_MODEL_SNAPSHOT,
        returned_model=V8_MODEL_SNAPSHOT,
    )
    for requested, returned in (
        ("gpt-5.4", V8_MODEL_SNAPSHOT),
        (V8_MODEL_SNAPSHOT, "gpt-5.4"),
        (V8_MODEL_SNAPSHOT, "gpt-5.4-2026-03-06"),
        (V8_MODEL_SNAPSHOT + " ", V8_MODEL_SNAPSHOT),
    ):
        with pytest.raises(Wave2IntegrationError):
            validate_v8_exact_snapshot_pair(
                requested_model=requested,
                returned_model=returned,
            )


def test_v8_changes_only_authorized_snapshot_identity_and_derived_digests() -> None:
    v7 = json.loads((ROOT / V7_MANIFEST_REL).read_bytes())
    v8 = expected_prefreeze_v8_manifest(ROOT)
    exact7 = dict(v7["exact_provider_contract"])
    exact8 = dict(v8["exact_provider_contract"])
    unchanged = (
        "endpoint_digest",
        "credential_config_digest",
        "credential_identity_digest",
        "response_schema_digest",
        "sentinel_digest",
        "prompt_digest",
        "tool_policy_digest",
        "request_mode",
        "temperature",
        "token_budget",
    )
    for field in unchanged:
        assert exact8[field] == exact7[field]
    assert exact8["model"] == V8_MODEL_SNAPSHOT
    assert exact8["requested_model_literal"] == V8_MODEL_SNAPSHOT
    assert exact8["required_returned_snapshot"] == V8_MODEL_SNAPSHOT
    payload = exact_v8_probe_request_payload(ROOT)
    assert payload["model"] == V8_MODEL_SNAPSHOT
    assert payload["max_tokens"] == V8_DIAGNOSTIC_TOKEN_CEILING == 6000
    assert "tools" not in payload and "functions" not in payload
    assert v8["response_contract_equivalence"] == v7[
        "response_contract_equivalence"
    ]
    assert v8["preserved_scientific_identity"] == v7[
        "preserved_scientific_identity"
    ]
    future7 = dict(v7["future_r1_scientific_contract"])
    future8 = dict(v8["future_r1_scientific_contract"])
    for field in (
        "requested_model_literal",
        "required_returned_snapshot",
        "shared_call_contract_digest",
        "side_a_call_contract_digest",
        "side_b_call_contract_digest",
    ):
        future7.pop(field, None)
        future8.pop(field, None)
    assert future8 == future7


@pytest.mark.parametrize(
    "field,mutated",
    [
        ("requested_model_literal", "gpt-5.4"),
        ("required_returned_snapshot", "gpt-5.4"),
        ("endpoint_digest", "0" * 64),
        ("credential_config_digest", "1" * 64),
        ("credential_identity_digest", "2" * 64),
        ("response_schema_digest", "3" * 64),
        ("local_uniqueness_contract_digest", "4" * 64),
        ("prompt_digest", "5" * 64),
        ("tool_policy_digest", "6" * 64),
        ("diagnostic_token_budget", 5999),
        ("future_r1_token_budget_per_call", 5999),
        ("future_r1_call_count_per_side", 7),
        ("future_r1_proposal_budget_per_side", 7),
    ],
)
def test_v8_release_digest_is_sensitive_to_every_identity_binding(
    field: str, mutated: object
) -> None:
    release = expected_v8_provider_release(ROOT)
    expected_digest = release.pop("release_digest")
    changed = deepcopy(release)
    changed[field] = mutated
    assert sha256_digest(release) == expected_digest
    assert sha256_digest(changed) != expected_digest


def test_v8_policy_keeps_all_contract_identity_failures_terminal() -> None:
    policy = expected_v8_retry_policy(ROOT)
    vocabulary = {
        item["reason_code"]: item for item in policy[
            "response_contract_reason_vocabulary"
        ]
    }
    assert vocabulary["RETURNED_MODEL_TYPE_OR_EMPTY"]["retry_eligible"] == (
        "FALSE_TERMINAL_RESPONSE_CONTRACT"
    )
    assert policy["exact_snapshot_identity_failure"]["retry_eligible"] is False
    assert policy["diagnostic_slot"]["maximum_total_physical_attempts"] == 3
    assert policy["diagnostic_slot"][
        "deterministic_backoff_ms_after_failure"
    ] == [1000, 3000]


def test_v8_builder_and_probe_consume_one_shared_version_spec() -> None:
    expected = prefreeze_v8_runtime_spec()
    assert _contract(8) == expected
    assert _prefreeze_diagnostic_contract(8) == expected


def test_v8_missing_returned_model_reason_is_terminal_in_probe_policy() -> None:
    evidence = _classify_v5_failure(
        error=ValueError("must-not-be-persisted"),
        row=_row(
            error_type="RESPONSE_CONTRACT_ERROR",
            http_status=200,
            error_detail={"reason_code": "RETURNED_MODEL_TYPE_OR_EMPTY"},
        ),
        version=8,
    )
    assert evidence["classification"] == (
        "HTTP_200_RESPONSE_CONTRACT_RETURNED_MODEL_TYPE_OR_EMPTY"
    )
    assert evidence["retry_eligible"] is False


def test_checked_in_v8_contract_artifacts_are_exact_and_provider_free() -> None:
    for relative in (V8_RELEASE_REL, V8_POLICY_REL, V8_MANIFEST_REL, V8_AUTH_REL):
        assert (ROOT / relative).is_file()
    manifest = validate_prefreeze_v8(ROOT)
    dry_run = provider_free_v8_dry_run(ROOT)
    assert manifest == expected_prefreeze_v8_manifest(ROOT)
    assert dry_run["status"] == (
        "PASS_PROVIDER_FREE_V8_EXACT_SNAPSHOT_REQUEST"
    )
    assert dry_run["provider_calls"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["candidate_qualifications"] == 0
    assert dry_run["candidate_admissions"] == 0
    assert dry_run["outcomes_consumed"] == 0
    assert dry_run["held_out_reads"] == 0


def test_v8_outcome_is_three_transient_503s_exhausted_and_side_effect_free() -> None:
    attempt_path = ROOT / V8_ATTEMPT_RECEIPT_REL
    blocked_path = ROOT / V8_BLOCKED_REL
    assert attempt_path.is_file()
    assert blocked_path.is_file()
    assert not (ROOT / V8_READY_REL).exists()
    receipt = json.loads(attempt_path.read_bytes())
    blocked = json.loads(blocked_path.read_bytes())
    assert receipt["status"] == "BLOCKED"
    assert receipt["physical_provider_calls"] == 3
    assert receipt["retry_count"] == 2
    assert receipt["final_classification"] == (
        "HTTP_503_TRANSIENT_PROVIDER_ERROR"
    )
    assert receipt["termination_reason"] == "TRANSIENT_ATTEMPTS_EXHAUSTED"
    assert receipt["response_contract_reason_code"] is None
    assert [item["backoff_ms_after_attempt"] for item in receipt["physical_attempts"]] == [
        1000,
        3000,
        0,
    ]
    assert all(
        item["classification"] == "HTTP_503_TRANSIENT_PROVIDER_ERROR"
        and item["http_status"] == 503
        and item["returned_model"] is None
        for item in receipt["physical_attempts"]
    )
    assert blocked["status"] == "BLOCKED_PREFREEZE_V8"
    assert blocked["r1_worker_launch_authorized"] is False
    for ordinal in (1, 2, 3):
        connection = sqlite3.connect(
            V8_PRIVATE_ROOT
            / f"physical_attempt_{ordinal:02d}"
            / "broker.sqlite3"
        )
        try:
            row = connection.execute(
                "SELECT COUNT(*),status,error_type,error_detail_json,"
                "response_json,returned_model FROM calls"
            ).fetchone()
        finally:
            connection.close()
        assert row == (
            1,
            "FAILED",
            "HTTP_503",
            '{"http_status":503}',
            None,
            None,
        )
    for field in (
        "research_candidates_generated",
        "open_specs_projected",
        "resolver_calls",
        "candidate_roots_created",
        "candidate_qualifications",
        "candidate_admissions",
        "training_runs",
        "outcomes_consumed",
        "held_out_reads",
    ):
        assert receipt[field] == 0


def test_v8_sealed_artifacts_remain_exact_bytes_for_v9() -> None:
    observed = verify_v8_seal(ROOT)
    for relative, digest in V8_SEALED_DIGESTS.items():
        assert observed[relative.as_posix()] == digest
        assert bytes_sha256((ROOT / relative).read_bytes()) == digest


def test_v9_is_only_availability_identity_and_derived_digest_delta() -> None:
    v8 = json.loads((ROOT / V8_MANIFEST_REL).read_bytes())
    v9 = expected_prefreeze_v9_manifest(ROOT)
    exact8 = dict(v8["exact_provider_contract"])
    exact9 = dict(v9["exact_provider_contract"])
    for field in (
        "model",
        "requested_model_literal",
        "required_returned_snapshot",
        "model_identity_pair_digest",
        "transport_provider_release_digest",
        "endpoint_digest",
        "credential_config_digest",
        "credential_identity_digest",
        "response_schema_digest",
        "sentinel_digest",
        "request_payload_digest",
        "prompt_digest",
        "tool_policy_digest",
        "request_mode",
        "temperature",
        "token_budget",
    ):
        assert exact9[field] == exact8[field]
    assert v9["future_r1_scientific_contract"] == v8[
        "future_r1_scientific_contract"
    ]
    assert v9["response_contract_equivalence"] == v8[
        "response_contract_equivalence"
    ]
    assert v9["preserved_scientific_identity"] == v8[
        "preserved_scientific_identity"
    ]
    assert v9["provider_returned_model_evidence"] == v8[
        "provider_returned_model_evidence"
    ]
    assert v9["authorized_v9_availability_recheck"][
        "scientific_contract_change"
    ] is False
    assert v9["authorized_v9_availability_recheck"][
        "v10_or_unbounded_retry_authorized"
    ] is False


def test_v9_release_and_policy_change_only_version_predecessor_identity() -> None:
    release8 = expected_v8_provider_release(ROOT)
    release9 = expected_v9_provider_release(ROOT)
    for value in (release8, release9):
        value.pop("schema", None)
        value.pop("release_digest", None)
        value.pop("availability_predecessor", None)
    assert release9 == release8
    policy8 = expected_v8_retry_policy(ROOT)
    policy9 = expected_v9_retry_policy(ROOT)
    for value in (policy8, policy9):
        value.pop("schema", None)
        value.pop("inherited_v7_policy_ref", None)
        value.pop("inherited_v7_policy_digest", None)
        value.pop("inherited_v8_policy_ref", None)
        value.pop("inherited_v8_policy_digest", None)
        value["diagnostic_slot"].pop("slot_id", None)
    assert policy9 == policy8


def test_v9_builder_and_probe_consume_one_shared_version_spec() -> None:
    expected = prefreeze_v9_runtime_spec()
    assert _contract(9) == expected
    assert _prefreeze_diagnostic_contract(9) == expected


def test_checked_in_v9_contract_artifacts_are_exact_and_provider_free() -> None:
    for relative in (V9_RELEASE_REL, V9_POLICY_REL, V9_MANIFEST_REL, V9_AUTH_REL):
        assert (ROOT / relative).is_file()
    manifest = validate_prefreeze_v9(ROOT)
    dry_run = provider_free_v9_dry_run(ROOT)
    assert manifest == expected_prefreeze_v9_manifest(ROOT)
    assert dry_run["status"] == "PASS_PROVIDER_FREE_V9_AVAILABILITY_RECHECK"
    assert dry_run["provider_calls"] == 0
    assert dry_run["training_runs"] == 0
    assert dry_run["candidate_qualifications"] == 0
    assert dry_run["candidate_admissions"] == 0
    assert dry_run["outcomes_consumed"] == 0
    assert dry_run["held_out_reads"] == 0


def test_v9_outcome_is_hard_resource_block_after_three_transient_503s() -> None:
    attempt_path = ROOT / V9_ATTEMPT_RECEIPT_REL
    blocked_path = ROOT / V9_BLOCKED_REL
    assert attempt_path.is_file()
    assert blocked_path.is_file()
    assert not (ROOT / V9_READY_REL).exists()
    receipt = json.loads(attempt_path.read_bytes())
    blocked = json.loads(blocked_path.read_bytes())
    assert receipt["status"] == "BLOCKED"
    assert receipt["physical_provider_calls"] == 3
    assert receipt["retry_count"] == 2
    assert receipt["final_classification"] == (
        "HTTP_503_TRANSIENT_PROVIDER_ERROR"
    )
    assert receipt["termination_reason"] == "TRANSIENT_ATTEMPTS_EXHAUSTED"
    assert [item["backoff_ms_after_attempt"] for item in receipt["physical_attempts"]] == [
        1000,
        3000,
        0,
    ]
    assert all(
        item["classification"] == "HTTP_503_TRANSIENT_PROVIDER_ERROR"
        and item["http_status"] == 503
        and item["returned_model"] is None
        for item in receipt["physical_attempts"]
    )
    assert blocked["status"] == "HARD_BLOCKED_RESOURCE_PROVIDER_UNAVAILABLE"
    assert blocked["r1_worker_launch_authorized"] is False
    for ordinal in (1, 2, 3):
        connection = sqlite3.connect(
            V9_PRIVATE_ROOT
            / f"physical_attempt_{ordinal:02d}"
            / "broker.sqlite3"
        )
        try:
            row = connection.execute(
                "SELECT COUNT(*),status,error_type,error_detail_json,"
                "response_json,returned_model FROM calls"
            ).fetchone()
        finally:
            connection.close()
        assert row == (
            1,
            "FAILED",
            "HTTP_503",
            '{"http_status":503}',
            None,
            None,
        )
    for field in (
        "research_candidates_generated",
        "open_specs_projected",
        "resolver_calls",
        "candidate_roots_created",
        "candidate_qualifications",
        "candidate_admissions",
        "training_runs",
        "outcomes_consumed",
        "held_out_reads",
    ):
        assert receipt[field] == 0
