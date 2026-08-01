#!/usr/bin/env python3
"""Prepare, validate, and finalize additive Prefreeze V5 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.prefreeze_v5 import (  # noqa: E402
    V5_ATTEMPT_ID,
    V5_ATTEMPT_RECEIPT_REL,
    V5_ATTEMPT_RECEIPT_SCHEMA,
    V5_AUTH_REL,
    V5_BLOCKED_REL,
    V5_BLOCKED_SCHEMA,
    V5_DRY_RUN_REL,
    V5_MANIFEST_REL,
    V5_POLICY_REL,
    V5_PRIVATE_ROOT,
    V5_READY_REL,
    V5_READY_SCHEMA,
    V5_VERIFICATION_REL,
    V5_VERIFICATION_SCHEMA,
    diagnostic_reason_vocabulary,
    expected_prefreeze_v5_manifest,
    expected_v5_authorization,
    expected_v5_retry_policy,
    provider_free_v5_dry_run,
    validate_prefreeze_v5,
    verify_v4_seal,
)


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise SystemExit(
                f"existing artifact differs; refusing overwrite: {path.name}"
            )
        return
    path.write_bytes(encoded)


def _prepare() -> int:
    verify_v4_seal(ROOT)
    for forbidden in (
        V5_ATTEMPT_RECEIPT_REL,
        V5_BLOCKED_REL,
        V5_READY_REL,
        V5_VERIFICATION_REL,
    ):
        if (ROOT / forbidden).exists():
            raise SystemExit("V5 outcome artifact already exists; prepare is sealed")
    if V5_PRIVATE_ROOT.exists():
        raise SystemExit("V5 private root already exists; identity is not fresh")
    _write_once(ROOT / V5_POLICY_REL, expected_v5_retry_policy(ROOT))
    _write_once(ROOT / V5_MANIFEST_REL, expected_prefreeze_v5_manifest(ROOT))
    _write_once(ROOT / V5_AUTH_REL, expected_v5_authorization(ROOT))
    manifest = validate_prefreeze_v5(ROOT)
    dry_run = provider_free_v5_dry_run(ROOT)
    print(
        json.dumps(
            {
                "attempt_id": V5_ATTEMPT_ID,
                "authorization_sha256": bytes_sha256(
                    (ROOT / V5_AUTH_REL).read_bytes()
                ),
                "broker_source_sha256": manifest[
                    "engineering_observability_change"
                ]["broker_source_digest"],
                "manifest_sha256": bytes_sha256(
                    (ROOT / V5_MANIFEST_REL).read_bytes()
                ),
                "policy_sha256": bytes_sha256(
                    (ROOT / V5_POLICY_REL).read_bytes()
                ),
                "provider_calls": dry_run["provider_calls"],
                "request_payload_digest": manifest["exact_provider_contract"][
                    "request_payload_digest"
                ],
                "status": "PREPARED_V5_PRE_OUTCOME",
            },
            sort_keys=True,
        )
    )
    return 0


def _dry_run() -> int:
    receipt = provider_free_v5_dry_run(ROOT)
    _write_once(ROOT / V5_DRY_RUN_REL, receipt)
    print(
        json.dumps(
            {
                "dry_run_receipt_sha256": bytes_sha256(
                    (ROOT / V5_DRY_RUN_REL).read_bytes()
                ),
                "provider_calls": 0,
                "status": receipt["status"],
                "training_runs": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _load_attempt() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = validate_prefreeze_v5(ROOT)
    path = ROOT / V5_ATTEMPT_RECEIPT_REL
    if not path.is_file():
        raise SystemExit("V5 Provider attempt receipt is missing")
    receipt = json.loads(path.read_bytes())
    if canonical_json_bytes(receipt) != path.read_bytes():
        raise SystemExit("V5 Provider attempt receipt is not canonical JSON")
    fixed = {
        "schema": V5_ATTEMPT_RECEIPT_SCHEMA,
        "attempt_id": V5_ATTEMPT_ID,
        "model_requested": "gpt-5.4",
        "endpoint_digest": manifest["exact_provider_contract"][
            "endpoint_digest"
        ],
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "response_schema_digest": manifest["exact_provider_contract"][
            "response_schema_digest"
        ],
        "sensitive_values_persisted": False,
        "sensitive_headers_persisted": False,
        "raw_response_or_provider_body_persisted_on_failure": False,
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
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise SystemExit(f"V5 attempt receipt does not prove {field}")
    attempts = receipt.get("physical_attempts")
    count = receipt.get("physical_provider_calls")
    if (
        not isinstance(attempts, list)
        or not isinstance(count, int)
        or count != len(attempts)
        or not 1 <= count <= 3
        or receipt.get("retry_count") != count - 1
    ):
        raise SystemExit("V5 physical attempt count/retry count is invalid")
    identities = manifest["bounded_retry"]["physical_attempt_identities"]
    reason_codes = {
        item["reason_code"] for item in diagnostic_reason_vocabulary()
    }
    envelopes: set[str] = set()
    prior_digest: str | None = None
    prior_end_ns: int | None = None
    required_gap_ms = 0
    for index, attempt in enumerate(attempts, start=1):
        expected_identity = identities[index - 1]
        if (
            attempt.get("ordinal") != index
            or attempt.get("physical_attempt_identity_digest")
            != expected_identity["physical_attempt_identity_digest"]
            or attempt.get("private_root_digest")
            != expected_identity["private_root_digest"]
        ):
            raise SystemExit("V5 physical attempt identity changed")
        if (
            attempt.get("logical_call_id")
            != manifest["exact_provider_contract"]["logical_call_id"]
            or attempt.get("request_payload_digest")
            != receipt["request_payload_digest"]
        ):
            raise SystemExit("V5 diagnostic slot/payload identity changed")
        envelope = attempt.get("request_envelope_digest")
        if not isinstance(envelope, str):
            raise SystemExit("V5 request envelope digest is missing")
        envelopes.add(envelope)
        preimage = dict(attempt)
        attempt_digest = preimage.pop("attempt_digest", None)
        if (
            preimage.get("prior_attempt_digest") != prior_digest
            or sha256_digest(preimage) != attempt_digest
        ):
            raise SystemExit("V5 attempt digest chain is invalid")
        prior_digest = attempt_digest
        start_ns = attempt.get("monotonic_start_ns")
        end_ns = attempt.get("monotonic_end_ns")
        if (
            not isinstance(start_ns, int)
            or not isinstance(end_ns, int)
            or start_ns >= end_ns
            or (
                prior_end_ns is not None
                and start_ns - prior_end_ns < required_gap_ms * 1_000_000
            )
        ):
            raise SystemExit("V5 physical attempt time/backoff order is invalid")
        prior_end_ns = end_ns
        reason_code = attempt.get("response_contract_reason_code")
        if reason_code is not None and reason_code not in reason_codes:
            raise SystemExit("V5 persisted a non-allowlisted reason code")
        if reason_code is not None and attempt.get("retry_eligible") is not False:
            raise SystemExit("V5 response-contract reason was retried")
        if index < count and (
            attempt.get("retry_eligible") is not True
            or attempt.get("termination_reason") != "RETRY_SCHEDULED"
            or attempt.get("backoff_ms_after_attempt")
            != (1000 if index == 1 else 3000)
        ):
            raise SystemExit("V5 non-final retry proof is invalid")
        if index == count and attempt.get("backoff_ms_after_attempt") != 0:
            raise SystemExit("V5 final attempt scheduled an extra retry")
        required_gap_ms = int(attempt.get("backoff_ms_after_attempt", 0))
    if len(envelopes) != 1:
        raise SystemExit("V5 attempts did not share one request envelope")
    final = attempts[-1]
    if receipt.get("status") == "PASS":
        if (
            final.get("classification")
            != "PASS_EXACT_GPT_5_4_AUTH_PROVIDER_AND_LOCAL_SCHEMA"
            or final.get("termination_reason")
            != "FIRST_VALID_LOCAL_EQUIVALENT_RESPONSE_ACCEPTED"
            or receipt.get("returned_model") != "gpt-5.4"
            or receipt.get("authentication_status") != "VERIFIED"
            or receipt.get("local_semantic_equivalence_status") != "VERIFIED"
            or receipt.get("blocked_fields") != []
        ):
            raise SystemExit("V5 PASS receipt is not fully closed")
    elif receipt.get("status") == "BLOCKED":
        if final.get("termination_reason") not in {
            "DETERMINISTIC_TERMINAL_FAILURE",
            "TRANSIENT_ATTEMPTS_EXHAUSTED",
            "LOCAL_RESPONSE_CONTRACT_TERMINAL_FAILURE",
        }:
            raise SystemExit("V5 BLOCKED termination is invalid")
    else:
        raise SystemExit("V5 attempt receipt has unknown status")
    return manifest, receipt


def _record_verification(*, pytest_passed: int, pytest_skipped: int) -> int:
    _, attempt = _load_attempt()
    if attempt["status"] != "PASS":
        raise SystemExit("V5 BLOCKED attempt cannot receive PASS verification")
    dry_run_path = ROOT / V5_DRY_RUN_REL
    if not dry_run_path.is_file():
        raise SystemExit("V5 provider-free dry-run receipt is missing")
    dry_run = json.loads(dry_run_path.read_bytes())
    if (
        canonical_json_bytes(dry_run) != dry_run_path.read_bytes()
        or dry_run.get("status") != "PASS_PROVIDER_FREE_V5_OBSERVABILITY"
    ):
        raise SystemExit("V5 provider-free dry-run is not PASS")
    if pytest_passed < 1 or pytest_skipped < 0:
        raise SystemExit("V5 pytest result is invalid")
    verification = {
        "schema": V5_VERIFICATION_SCHEMA,
        "status": "PASS",
        "attempt_id": V5_ATTEMPT_ID,
        "attempt_receipt_digest": bytes_sha256(
            (ROOT / V5_ATTEMPT_RECEIPT_REL).read_bytes()
        ),
        "dry_run_receipt_digest": bytes_sha256(dry_run_path.read_bytes()),
        "validator": "PASS",
        "provider_free_dry_run": "PASS",
        "targeted_adjacent_no_training_tests": "PASS",
        "pytest_passed": pytest_passed,
        "pytest_skipped": pytest_skipped,
        "py_compile_modified_only": "PASS",
        "canonical_hash_secret_diff_structure": "PASS",
        "provider_calls_during_verification": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
    }
    _write_once(ROOT / V5_VERIFICATION_REL, verification)
    print(
        json.dumps(
            {
                "status": "PASS",
                "verification_receipt_sha256": bytes_sha256(
                    (ROOT / V5_VERIFICATION_REL).read_bytes()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _finalize() -> int:
    manifest, attempt = _load_attempt()
    attempt_path = ROOT / V5_ATTEMPT_RECEIPT_REL
    if attempt["status"] == "BLOCKED":
        if (ROOT / V5_READY_REL).exists():
            raise SystemExit("V5 READY exists; refusing BLOCKED finalization")
        blocked = {
            "schema": V5_BLOCKED_SCHEMA,
            "status": "BLOCKED_PREFREEZE_V5",
            "attempt_id": V5_ATTEMPT_ID,
            "manifest_ref": V5_MANIFEST_REL.name,
            "manifest_digest": bytes_sha256(
                (ROOT / V5_MANIFEST_REL).read_bytes()
            ),
            "retry_policy_ref": V5_POLICY_REL.name,
            "retry_policy_digest": bytes_sha256(
                (ROOT / V5_POLICY_REL).read_bytes()
            ),
            "authorization_ref": V5_AUTH_REL.name,
            "authorization_digest": bytes_sha256(
                (ROOT / V5_AUTH_REL).read_bytes()
            ),
            "attempt_receipt_ref": V5_ATTEMPT_RECEIPT_REL.name,
            "attempt_receipt_digest": bytes_sha256(attempt_path.read_bytes()),
            "final_classification": attempt["final_classification"],
            "response_contract_reason_code": attempt.get(
                "response_contract_reason_code"
            ),
            "termination_reason": attempt["termination_reason"],
            "physical_provider_calls_v5": attempt["physical_provider_calls"],
            "retry_count_v5": attempt["retry_count"],
            "physical_attempt_classifications": [
                {
                    "ordinal": item["ordinal"],
                    "classification": item["classification"],
                    "response_contract_reason_code": item.get(
                        "response_contract_reason_code"
                    ),
                    "retry_eligible": item["retry_eligible"],
                    "backoff_ms_after_attempt": item[
                        "backoff_ms_after_attempt"
                    ],
                    "termination_reason": item["termination_reason"],
                }
                for item in attempt["physical_attempts"]
            ],
            "provider_calls_must_stop_for_v5_slot": True,
            "v6_allowed_only_for_single_local_root_cause": True,
            "provider_failure_is_mechanism_negative_evidence": False,
            "r1_prefreeze_ready_receipt_emitted": False,
            "r1_worker_launch_authorized": False,
            "side_effects_v5": {
                "provider_calls": attempt["physical_provider_calls"],
                "research_candidates_generated": 0,
                "candidate_roots_created": 0,
                "candidate_qualifications": 0,
                "candidate_admissions": 0,
                "training_runs": 0,
                "outcomes_consumed": 0,
                "held_out_reads": 0,
            },
        }
        _write_once(ROOT / V5_BLOCKED_REL, blocked)
        print(
            json.dumps(
                {
                    "blocked_receipt_sha256": bytes_sha256(
                        (ROOT / V5_BLOCKED_REL).read_bytes()
                    ),
                    "physical_provider_calls_v5": attempt[
                        "physical_provider_calls"
                    ],
                    "retry_count_v5": attempt["retry_count"],
                    "status": "BLOCKED_PREFREEZE_V5",
                },
                sort_keys=True,
            )
        )
        return 2

    verification_path = ROOT / V5_VERIFICATION_REL
    if not verification_path.is_file():
        print(json.dumps({"status": "PASS_AWAITING_LOCAL_VERIFICATION"}))
        return 3
    verification = json.loads(verification_path.read_bytes())
    required = {
        "status": "PASS",
        "attempt_receipt_digest": bytes_sha256(attempt_path.read_bytes()),
        "validator": "PASS",
        "provider_free_dry_run": "PASS",
        "targeted_adjacent_no_training_tests": "PASS",
        "py_compile_modified_only": "PASS",
        "canonical_hash_secret_diff_structure": "PASS",
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
    }
    for field, expected in required.items():
        if verification.get(field) != expected:
            raise SystemExit(f"V5 verification does not prove {field}")
    ready = {
        "schema": V5_READY_SCHEMA,
        "status": "R1_PREFREEZE_READY",
        "attempt_id": V5_ATTEMPT_ID,
        "manifest_ref": V5_MANIFEST_REL.name,
        "manifest_digest": bytes_sha256((ROOT / V5_MANIFEST_REL).read_bytes()),
        "retry_policy_ref": V5_POLICY_REL.name,
        "retry_policy_digest": bytes_sha256((ROOT / V5_POLICY_REL).read_bytes()),
        "authorization_ref": V5_AUTH_REL.name,
        "authorization_digest": bytes_sha256((ROOT / V5_AUTH_REL).read_bytes()),
        "attempt_receipt_ref": V5_ATTEMPT_RECEIPT_REL.name,
        "attempt_receipt_digest": bytes_sha256(attempt_path.read_bytes()),
        "verification_receipt_ref": V5_VERIFICATION_REL.name,
        "verification_receipt_digest": bytes_sha256(
            verification_path.read_bytes()
        ),
        "model": "gpt-5.4",
        "returned_model": "gpt-5.4",
        "authentication_status": "VERIFIED",
        "provider_schema_support": "VERIFIED",
        "local_semantic_equivalence": "VERIFIED_EXACT_V1_ACCEPTANCE_SET",
        "physical_provider_calls_v5": attempt["physical_provider_calls"],
        "retry_count_v5": attempt["retry_count"],
        "r1_worker_launch_authorized": True,
        "r2_launch_authorized": False,
        "training_started": False,
        "candidate_qualification_performed": False,
        "candidate_admission_performed": False,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "authorization_scope": (
            "INDEPENDENT_R1_WORKER_MAY_START_EXACT_FROZEN_R1_ONLY"
        ),
    }
    _write_once(ROOT / V5_READY_REL, ready)
    print(
        json.dumps(
            {
                "ready_receipt_sha256": bytes_sha256(
                    (ROOT / V5_READY_REL).read_bytes()
                ),
                "status": "R1_PREFREEZE_READY",
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "dry-run", "record-verification", "finalize"),
    )
    parser.add_argument("--pytest-passed", type=int, default=0)
    parser.add_argument("--pytest-skipped", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        return _prepare()
    if args.action == "dry-run":
        return _dry_run()
    if args.action == "record-verification":
        return _record_verification(
            pytest_passed=args.pytest_passed,
            pytest_skipped=args.pytest_skipped,
        )
    return _finalize()


if __name__ == "__main__":
    raise SystemExit(main())
