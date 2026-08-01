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
)
from scripts.reprobe_fresh_open_spec_endpoint_v2 import _classify_v5_failure
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
        and node.name == "_main_v5_or_v6"
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
