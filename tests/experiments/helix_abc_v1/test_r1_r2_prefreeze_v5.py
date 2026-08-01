from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
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
)
from scripts.reprobe_fresh_open_spec_endpoint_v2 import _classify_v5_failure


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
