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
    V6_ATTEMPT_ID,
    V6_ATTEMPT_RECEIPT_REL,
    V6_ATTEMPT_RECEIPT_SCHEMA,
    V6_AUTH_REL,
    V6_BLOCKED_REL,
    V6_BLOCKED_SCHEMA,
    V6_DRY_RUN_REL,
    V6_MANIFEST_REL,
    V6_POLICY_REL,
    V6_PRIVATE_ROOT,
    V6_READY_REL,
    V6_READY_SCHEMA,
    V6_RELEASE_REL,
    V6_REQUESTED_MODEL_ALIAS,
    V6_REQUIRED_RETURNED_SNAPSHOT,
    V6_VERIFICATION_REL,
    V6_VERIFICATION_SCHEMA,
    expected_prefreeze_v6_manifest,
    expected_v6_authorization,
    expected_v6_provider_release,
    expected_v6_retry_policy,
    provider_free_v6_dry_run,
    validate_prefreeze_v6,
    verify_v5_seal,
    V7_ATTEMPT_ID,
    V7_ATTEMPT_RECEIPT_REL,
    V7_ATTEMPT_RECEIPT_SCHEMA,
    V7_AUTH_REL,
    V7_BLOCKED_REL,
    V7_BLOCKED_SCHEMA,
    V7_DIAGNOSTIC_TOKEN_CEILING,
    V7_DRY_RUN_REL,
    V7_MANIFEST_REL,
    V7_POLICY_REL,
    V7_PRIVATE_ROOT,
    V7_READY_REL,
    V7_READY_SCHEMA,
    V7_RELEASE_REL,
    V7_VERIFICATION_REL,
    V7_VERIFICATION_SCHEMA,
    expected_prefreeze_v7_manifest,
    expected_v7_authorization,
    expected_v7_provider_release,
    expected_v7_retry_policy,
    provider_free_v7_dry_run,
    prefreeze_v8_runtime_spec,
    validate_prefreeze_v7,
    verify_v6_seal,
)


def _contract(version: int) -> dict[str, Any]:
    if version == 8:
        return prefreeze_v8_runtime_spec()
    if version == 5:
        return {
            "label": "V5",
            "attempt_id": V5_ATTEMPT_ID,
            "attempt_rel": V5_ATTEMPT_RECEIPT_REL,
            "attempt_schema": V5_ATTEMPT_RECEIPT_SCHEMA,
            "auth_rel": V5_AUTH_REL,
            "blocked_rel": V5_BLOCKED_REL,
            "blocked_schema": V5_BLOCKED_SCHEMA,
            "dry_run_rel": V5_DRY_RUN_REL,
            "manifest_rel": V5_MANIFEST_REL,
            "policy_rel": V5_POLICY_REL,
            "private_root": V5_PRIVATE_ROOT,
            "ready_rel": V5_READY_REL,
            "ready_schema": V5_READY_SCHEMA,
            "verification_rel": V5_VERIFICATION_REL,
            "verification_schema": V5_VERIFICATION_SCHEMA,
            "validate": validate_prefreeze_v5,
            "dry_run": provider_free_v5_dry_run,
            "requested_model": "gpt-5.4",
            "required_returned_model": "gpt-5.4",
            "pass_classification": (
                "PASS_EXACT_GPT_5_4_AUTH_PROVIDER_AND_LOCAL_SCHEMA"
            ),
            "pass_status": "PASS_PROVIDER_FREE_V5_OBSERVABILITY",
            "verify_predecessor": verify_v4_seal,
            "expected_policy": expected_v5_retry_policy,
            "expected_manifest": expected_prefreeze_v5_manifest,
            "expected_authorization": expected_v5_authorization,
            "expected_release": None,
            "release_rel": None,
        }
    if version == 6:
        return {
            "label": "V6",
            "attempt_id": V6_ATTEMPT_ID,
            "attempt_rel": V6_ATTEMPT_RECEIPT_REL,
            "attempt_schema": V6_ATTEMPT_RECEIPT_SCHEMA,
            "auth_rel": V6_AUTH_REL,
            "blocked_rel": V6_BLOCKED_REL,
            "blocked_schema": V6_BLOCKED_SCHEMA,
            "dry_run_rel": V6_DRY_RUN_REL,
            "manifest_rel": V6_MANIFEST_REL,
            "policy_rel": V6_POLICY_REL,
            "private_root": V6_PRIVATE_ROOT,
            "ready_rel": V6_READY_REL,
            "ready_schema": V6_READY_SCHEMA,
            "release_rel": V6_RELEASE_REL,
            "verification_rel": V6_VERIFICATION_REL,
            "verification_schema": V6_VERIFICATION_SCHEMA,
            "validate": validate_prefreeze_v6,
            "dry_run": provider_free_v6_dry_run,
            "requested_model": V6_REQUESTED_MODEL_ALIAS,
            "required_returned_model": V6_REQUIRED_RETURNED_SNAPSHOT,
            "pass_classification": (
                "PASS_EXACT_GPT_5_4_ALIAS_SNAPSHOT_AUTH_PROVIDER_AND_LOCAL_SCHEMA"
            ),
            "pass_status": "PASS_PROVIDER_FREE_V6_EXACT_MODEL_PAIR",
            "verify_predecessor": verify_v5_seal,
            "expected_policy": expected_v6_retry_policy,
            "expected_manifest": expected_prefreeze_v6_manifest,
            "expected_authorization": expected_v6_authorization,
            "expected_release": expected_v6_provider_release,
        }
    if version == 7:
        return {
            "label": "V7",
            "attempt_id": V7_ATTEMPT_ID,
            "attempt_rel": V7_ATTEMPT_RECEIPT_REL,
            "attempt_schema": V7_ATTEMPT_RECEIPT_SCHEMA,
            "auth_rel": V7_AUTH_REL,
            "blocked_rel": V7_BLOCKED_REL,
            "blocked_schema": V7_BLOCKED_SCHEMA,
            "diagnostic_token_ceiling": V7_DIAGNOSTIC_TOKEN_CEILING,
            "dry_run_rel": V7_DRY_RUN_REL,
            "manifest_rel": V7_MANIFEST_REL,
            "policy_rel": V7_POLICY_REL,
            "private_root": V7_PRIVATE_ROOT,
            "ready_rel": V7_READY_REL,
            "ready_schema": V7_READY_SCHEMA,
            "release_rel": V7_RELEASE_REL,
            "verification_rel": V7_VERIFICATION_REL,
            "verification_schema": V7_VERIFICATION_SCHEMA,
            "validate": validate_prefreeze_v7,
            "dry_run": provider_free_v7_dry_run,
            "requested_model": V6_REQUESTED_MODEL_ALIAS,
            "required_returned_model": V6_REQUIRED_RETURNED_SNAPSHOT,
            "pass_classification": (
                "PASS_EXACT_GPT_5_4_ALIAS_SNAPSHOT_AUTH_PROVIDER_AND_LOCAL_SCHEMA"
            ),
            "pass_status": (
                "PASS_PROVIDER_FREE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
            ),
            "verify_predecessor": verify_v6_seal,
            "expected_policy": expected_v7_retry_policy,
            "expected_manifest": expected_prefreeze_v7_manifest,
            "expected_authorization": expected_v7_authorization,
            "expected_release": expected_v7_provider_release,
        }
    raise SystemExit("unsupported Prefreeze artifact version")


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise SystemExit(
                f"existing artifact differs; refusing overwrite: {path.name}"
            )
        return
    path.write_bytes(encoded)


def _prepare(version: int) -> int:
    contract = _contract(version)
    contract["verify_predecessor"](ROOT)
    for forbidden in (
        contract["attempt_rel"],
        contract["blocked_rel"],
        contract["ready_rel"],
        contract["verification_rel"],
    ):
        if (ROOT / forbidden).exists():
            raise SystemExit(
                f"{contract['label']} outcome artifact already exists; prepare is sealed"
            )
    if contract["private_root"].exists():
        raise SystemExit(
            f"{contract['label']} private root already exists; identity is not fresh"
        )
    expected_release = contract["expected_release"]
    if expected_release is not None:
        _write_once(ROOT / contract["release_rel"], expected_release(ROOT))
    policy = contract["expected_policy"](ROOT)
    manifest_payload = contract["expected_manifest"](ROOT)
    authorization = contract["expected_authorization"](ROOT)
    _write_once(ROOT / contract["policy_rel"], policy)
    _write_once(ROOT / contract["manifest_rel"], manifest_payload)
    _write_once(ROOT / contract["auth_rel"], authorization)
    manifest = contract["validate"](ROOT)
    dry_run = contract["dry_run"](ROOT)
    print(
        json.dumps(
            {
                "attempt_id": contract["attempt_id"],
                "authorization_sha256": bytes_sha256(
                    (ROOT / contract["auth_rel"]).read_bytes()
                ),
                "manifest_sha256": bytes_sha256(
                    (ROOT / contract["manifest_rel"]).read_bytes()
                ),
                "policy_sha256": bytes_sha256(
                    (ROOT / contract["policy_rel"]).read_bytes()
                ),
                "provider_calls": dry_run["provider_calls"],
                "request_payload_digest": manifest["exact_provider_contract"][
                    "request_payload_digest"
                ],
                "status": f"PREPARED_{contract['label']}_PRE_OUTCOME",
            },
            sort_keys=True,
        )
    )
    return 0


def _dry_run(version: int) -> int:
    contract = _contract(version)
    receipt = contract["dry_run"](ROOT)
    _write_once(ROOT / contract["dry_run_rel"], receipt)
    print(
        json.dumps(
            {
                "dry_run_receipt_sha256": bytes_sha256(
                    (ROOT / contract["dry_run_rel"]).read_bytes()
                ),
                "provider_calls": 0,
                "status": receipt["status"],
                "training_runs": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _load_attempt(version: int) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _contract(version)
    label = contract["label"]
    manifest = contract["validate"](ROOT)
    path = ROOT / contract["attempt_rel"]
    if not path.is_file():
        raise SystemExit(f"{label} Provider attempt receipt is missing")
    receipt = json.loads(path.read_bytes())
    if canonical_json_bytes(receipt) != path.read_bytes():
        raise SystemExit(f"{label} Provider attempt receipt is not canonical JSON")
    fixed = {
        "schema": contract["attempt_schema"],
        "attempt_id": contract["attempt_id"],
        "model_requested": contract["requested_model"],
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
    identity_receipt_fields = contract.get("identity_receipt_fields")
    if identity_receipt_fields is not None:
        fixed.update(identity_receipt_fields)
        fixed["provider_release_digest"] = manifest[
            "exact_provider_contract"
        ]["provider_release_digest"]
    elif version >= 6:
        fixed.update(
            {
                "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
                "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
                "provider_release_digest": manifest[
                    "exact_provider_contract"
                ]["provider_release_digest"],
            }
        )
    if "diagnostic_token_ceiling" in contract:
        fixed["diagnostic_token_ceiling"] = contract[
            "diagnostic_token_ceiling"
        ]
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise SystemExit(f"{label} attempt receipt does not prove {field}")
    attempts = receipt.get("physical_attempts")
    count = receipt.get("physical_provider_calls")
    if (
        not isinstance(attempts, list)
        or not isinstance(count, int)
        or count != len(attempts)
        or not 1 <= count <= 3
        or receipt.get("retry_count") != count - 1
    ):
        raise SystemExit(f"{label} physical attempt count/retry count is invalid")
    identities = manifest["bounded_retry"]["physical_attempt_identities"]
    reason_codes = {
        item["reason_code"]
        for item in diagnostic_reason_vocabulary(
            include_returned_model_identity=version >= 8
        )
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
            raise SystemExit(f"{label} physical attempt identity changed")
        if (
            attempt.get("logical_call_id")
            != manifest["exact_provider_contract"]["logical_call_id"]
            or attempt.get("request_payload_digest")
            != receipt["request_payload_digest"]
        ):
            raise SystemExit(f"{label} diagnostic slot/payload identity changed")
        envelope = attempt.get("request_envelope_digest")
        if not isinstance(envelope, str):
            raise SystemExit(f"{label} request envelope digest is missing")
        envelopes.add(envelope)
        preimage = dict(attempt)
        attempt_digest = preimage.pop("attempt_digest", None)
        if (
            preimage.get("prior_attempt_digest") != prior_digest
            or sha256_digest(preimage) != attempt_digest
        ):
            raise SystemExit(f"{label} attempt digest chain is invalid")
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
            raise SystemExit(f"{label} physical attempt time/backoff order is invalid")
        prior_end_ns = end_ns
        reason_code = attempt.get("response_contract_reason_code")
        if reason_code is not None and reason_code not in reason_codes:
            raise SystemExit(f"{label} persisted a non-allowlisted reason code")
        if reason_code is not None and attempt.get("retry_eligible") is not False:
            raise SystemExit(f"{label} response-contract reason was retried")
        if index < count and (
            attempt.get("retry_eligible") is not True
            or attempt.get("termination_reason") != "RETRY_SCHEDULED"
            or attempt.get("backoff_ms_after_attempt")
            != (1000 if index == 1 else 3000)
        ):
            raise SystemExit(f"{label} non-final retry proof is invalid")
        if index == count and attempt.get("backoff_ms_after_attempt") != 0:
            raise SystemExit(f"{label} final attempt scheduled an extra retry")
        required_gap_ms = int(attempt.get("backoff_ms_after_attempt", 0))
    if len(envelopes) != 1:
        raise SystemExit(f"{label} attempts did not share one request envelope")
    final = attempts[-1]
    if receipt.get("status") == "PASS":
        if (
            final.get("classification")
            != contract["pass_classification"]
            or final.get("termination_reason")
            != "FIRST_VALID_LOCAL_EQUIVALENT_RESPONSE_ACCEPTED"
            or receipt.get("returned_model")
            != contract["required_returned_model"]
            or receipt.get("authentication_status") != "VERIFIED"
            or receipt.get("local_semantic_equivalence_status") != "VERIFIED"
            or receipt.get("blocked_fields") != []
        ):
            raise SystemExit(f"{label} PASS receipt is not fully closed")
    elif receipt.get("status") == "BLOCKED":
        if final.get("termination_reason") not in {
            "DETERMINISTIC_TERMINAL_FAILURE",
            "TRANSIENT_ATTEMPTS_EXHAUSTED",
            "LOCAL_RESPONSE_CONTRACT_TERMINAL_FAILURE",
        }:
            raise SystemExit(f"{label} BLOCKED termination is invalid")
    else:
        raise SystemExit(f"{label} attempt receipt has unknown status")
    return manifest, receipt


def _record_verification(
    *, version: int, pytest_passed: int, pytest_skipped: int
) -> int:
    contract = _contract(version)
    label = contract["label"]
    _, attempt = _load_attempt(version)
    if attempt["status"] != "PASS":
        raise SystemExit(f"{label} BLOCKED attempt cannot receive PASS verification")
    dry_run_path = ROOT / contract["dry_run_rel"]
    if not dry_run_path.is_file():
        raise SystemExit(f"{label} provider-free dry-run receipt is missing")
    dry_run = json.loads(dry_run_path.read_bytes())
    if (
        canonical_json_bytes(dry_run) != dry_run_path.read_bytes()
        or dry_run.get("status") != contract["pass_status"]
    ):
        raise SystemExit(f"{label} provider-free dry-run is not PASS")
    if pytest_passed < 1 or pytest_skipped < 0:
        raise SystemExit(f"{label} pytest result is invalid")
    verification = {
        "schema": contract["verification_schema"],
        "status": "PASS",
        "attempt_id": contract["attempt_id"],
        "attempt_receipt_digest": bytes_sha256(
            (ROOT / contract["attempt_rel"]).read_bytes()
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
    _write_once(ROOT / contract["verification_rel"], verification)
    print(
        json.dumps(
            {
                "status": "PASS",
                "verification_receipt_sha256": bytes_sha256(
                    (ROOT / contract["verification_rel"]).read_bytes()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _finalize(version: int) -> int:
    contract = _contract(version)
    label = contract["label"]
    manifest, attempt = _load_attempt(version)
    attempt_path = ROOT / contract["attempt_rel"]
    if attempt["status"] == "BLOCKED":
        if (ROOT / contract["ready_rel"]).exists():
            raise SystemExit(f"{label} READY exists; refusing BLOCKED finalization")
        blocked = {
            "schema": contract["blocked_schema"],
            "status": f"BLOCKED_PREFREEZE_{label}",
            "attempt_id": contract["attempt_id"],
            "manifest_ref": contract["manifest_rel"].name,
            "manifest_digest": bytes_sha256(
                (ROOT / contract["manifest_rel"]).read_bytes()
            ),
            "retry_policy_ref": contract["policy_rel"].name,
            "retry_policy_digest": bytes_sha256(
                (ROOT / contract["policy_rel"]).read_bytes()
            ),
            "authorization_ref": contract["auth_rel"].name,
            "authorization_digest": bytes_sha256(
                (ROOT / contract["auth_rel"]).read_bytes()
            ),
            "attempt_receipt_ref": contract["attempt_rel"].name,
            "attempt_receipt_digest": bytes_sha256(attempt_path.read_bytes()),
            "final_classification": attempt["final_classification"],
            "response_contract_reason_code": attempt.get(
                "response_contract_reason_code"
            ),
            "termination_reason": attempt["termination_reason"],
            f"physical_provider_calls_v{version}": attempt[
                "physical_provider_calls"
            ],
            f"retry_count_v{version}": attempt["retry_count"],
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
            f"provider_calls_must_stop_for_v{version}_slot": True,
            "provider_failure_is_mechanism_negative_evidence": False,
            "r1_prefreeze_ready_receipt_emitted": False,
            "r1_worker_launch_authorized": False,
            f"side_effects_v{version}": {
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
        if version == 5:
            blocked["v6_allowed_only_for_single_local_root_cause"] = True
        _write_once(ROOT / contract["blocked_rel"], blocked)
        print(
            json.dumps(
                {
                    "blocked_receipt_sha256": bytes_sha256(
                        (ROOT / contract["blocked_rel"]).read_bytes()
                    ),
                    f"physical_provider_calls_v{version}": attempt[
                        "physical_provider_calls"
                    ],
                    f"retry_count_v{version}": attempt["retry_count"],
                    "status": f"BLOCKED_PREFREEZE_{label}",
                },
                sort_keys=True,
            )
        )
        return 2

    verification_path = ROOT / contract["verification_rel"]
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
            raise SystemExit(f"{label} verification does not prove {field}")
    ready = {
        "schema": contract["ready_schema"],
        "status": "R1_PREFREEZE_READY",
        "attempt_id": contract["attempt_id"],
        "manifest_ref": contract["manifest_rel"].name,
        "manifest_digest": bytes_sha256(
            (ROOT / contract["manifest_rel"]).read_bytes()
        ),
        "retry_policy_ref": contract["policy_rel"].name,
        "retry_policy_digest": bytes_sha256(
            (ROOT / contract["policy_rel"]).read_bytes()
        ),
        "authorization_ref": contract["auth_rel"].name,
        "authorization_digest": bytes_sha256(
            (ROOT / contract["auth_rel"]).read_bytes()
        ),
        "attempt_receipt_ref": contract["attempt_rel"].name,
        "attempt_receipt_digest": bytes_sha256(attempt_path.read_bytes()),
        "verification_receipt_ref": contract["verification_rel"].name,
        "verification_receipt_digest": bytes_sha256(
            verification_path.read_bytes()
        ),
        "model": contract["requested_model"],
        "returned_model": contract["required_returned_model"],
        "required_returned_snapshot": contract["required_returned_model"],
        "diagnostic_token_ceiling": contract.get("diagnostic_token_ceiling"),
        "authentication_status": "VERIFIED",
        "provider_schema_support": "VERIFIED",
        "local_semantic_equivalence": "VERIFIED_EXACT_V1_ACCEPTANCE_SET",
        f"physical_provider_calls_v{version}": attempt[
            "physical_provider_calls"
        ],
        f"retry_count_v{version}": attempt["retry_count"],
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
    identity_fields = contract.get("identity_receipt_fields")
    if identity_fields is None:
        ready["requested_model_alias"] = contract["requested_model"]
    else:
        ready.update(identity_fields)
    _write_once(ROOT / contract["ready_rel"], ready)
    print(
        json.dumps(
            {
                "ready_receipt_sha256": bytes_sha256(
                    (ROOT / contract["ready_rel"]).read_bytes()
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
    parser.add_argument("--version", type=int, choices=(5, 6, 7, 8), default=5)
    args = parser.parse_args()
    if args.action == "prepare":
        return _prepare(args.version)
    if args.action == "dry-run":
        return _dry_run(args.version)
    if args.action == "record-verification":
        return _record_verification(
            version=args.version,
            pytest_passed=args.pytest_passed,
            pytest_skipped=args.pytest_skipped,
        )
    return _finalize(args.version)


if __name__ == "__main__":
    raise SystemExit(main())
