"""Fail-closed Prefreeze V5 observability contract and validator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .canonical import bytes_sha256, canonical_json_bytes, canonical_value, sha256_digest
from .lab_api_broker import LabApiResponseFailureReasonV1
from .prefreeze_v2 import (
    DOC_ROOT_REL,
    SQLITE_CALL_COLUMNS,
    V4_ATTEMPT_RECEIPT_REL,
    V4_AUTH_REL,
    V4_BLOCKED_REL,
    V4_DRY_RUN_REL,
    V4_MANIFEST_REL,
    V4_NEGATIVE_FIXTURE_REL,
    V4_POLICY_REL,
    V4_PROVIDER_SCHEMA_REL,
    V4_RELEASE_REL,
    V4_VALIDATOR_REL,
    V4_VALID_FIXTURE_REL,
    verify_v1_seal,
    verify_v2_seal,
    verify_v3_seal,
)
from .wave2_integration import Wave2IntegrationError


V4_HEAD = "45aa25211322d9f765e28447d5d1e885ccf3ee70"
V4_PARENT = "68014e714d6636268fc36b259bf13498f158e4cd"
V4_TREE = "cf2b333b2930be78557b8e0502e6db995ae42aa7"

V4_SEALED_DIGESTS: dict[Path, str] = {
    V4_MANIFEST_REL: "932c72d875eb629fa617243244d72b64ad7c5d186fce45d46390a754a06e56dc",
    V4_POLICY_REL: "21f62cdb87e795ebfc611bfe10752bcc1f37e0220a7d5033e421434ee73e490f",
    V4_AUTH_REL: "599911d0b19833b7e5e9baad2c60b0b35f461fc98e4e22b3d0ff101021879323",
    V4_RELEASE_REL: "26eee7ac8e10130f3bdac88ef2e8975f9bf4394853d618c302d7b8dddf12378a",
    V4_DRY_RUN_REL: "c298ecb3a99ff3b8224bb92a19197b233af4d2db32fd2adf9951869e89a1b870",
    V4_ATTEMPT_RECEIPT_REL: "caf07471016017b76cce3bbc58ff4986b2485fe2b0c1cdd3dfe5f6c597c6df4a",
    V4_BLOCKED_REL: "2ab74bdcf79f23712d01fd95550a72596d1fd71be6ce9af8cdbee5e38c49c180",
    V4_PROVIDER_SCHEMA_REL: "1a03214b9e9b27036d0350398c3e8ab17fd6158c638aa75ad5354b041c3c7e5b",
    V4_VALID_FIXTURE_REL: "05cc80fa34c99145955dc41075b4f3cb9c33013f034cfcbafc1778ec4b742064",
    V4_NEGATIVE_FIXTURE_REL: "ff4a98243e0846189623061e3bbb8765450af386047ab2af23263867fc611ea8",
    V4_VALIDATOR_REL: "caa20a24cfea3e1ca45be3a5fb785a5e761ee6197b1c16bf87b491e261b08f9c",
}

V5_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v5-20260801"
V5_MANIFEST_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-attempt.v5"
V5_POLICY_SCHEMA = "recclaw.research-line.r1-provider-retry-policy.v5"
V5_AUTH_SCHEMA = "recclaw.research-line.prefreeze-v5-authorization.v1"
V5_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v5-provider-attempt-receipt.v1"
)
V5_DRY_RUN_SCHEMA = "recclaw.research-line.prefreeze-v5-dry-run-receipt.v1"
V5_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v5-blocked-receipt.v1"
V5_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v5"
V5_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v5-verification-receipt.v1"
)

V5_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v5-observability-diagnostic-slot"
V5_SESSION_ID = (
    "fresh-open-spec-prefreeze-v5-observability-diagnostic-slot-session"
)
V5_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v5_private")

V5_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V5.json"
V5_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V5.json"
V5_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V5_AUTHORIZATION.json"
V5_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V5.json"
)
V5_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V5_BLOCKED_RECEIPT.json"
V5_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V5_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V5_DRY_RUN_RECEIPT.json"
V5_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V5_VERIFICATION_RECEIPT.json"
BROKER_SOURCE_REL = Path(
    "src/recclaw_core/experiments/helix_abc_v1/lab_api_broker.py"
)


def _repo_ref(relative: Path) -> str:
    return relative.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Wave2IntegrationError(f"JSON resource is not an object: {path}")
    if canonical_json_bytes(value) != path.read_bytes():
        raise Wave2IntegrationError(f"JSON resource is not canonical: {path}")
    return value


def _load_exact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _read_json(path)
    if canonical_value(observed) != canonical_value(expected):
        raise Wave2IntegrationError(f"fail-closed artifact mismatch: {path}")
    return observed


def v5_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V5 physical attempt ordinal must be 1..3")
    return V5_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v4_seal(repo_root: Path) -> dict[str, str]:
    verify_v1_seal(repo_root)
    verify_v2_seal(repo_root)
    verify_v3_seal(repo_root)
    for relative, digest in V4_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V4 bytes changed: {relative.as_posix()}"
            )
    return {
        relative.as_posix(): digest
        for relative, digest in V4_SEALED_DIGESTS.items()
    }


def diagnostic_reason_vocabulary() -> list[dict[str, str]]:
    stages = {
        "ENVELOPE_JSON_DECODE": "HTTP200_ENVELOPE_DECODE",
        "ENVELOPE_SHAPE": "HTTP200_ENVELOPE_SHAPE",
        "CHOICES_SHAPE": "HTTP200_CHOICES_SHAPE",
        "CHOICE_SHAPE": "HTTP200_CHOICE_SHAPE",
        "MESSAGE_SHAPE": "HTTP200_MESSAGE_SHAPE",
        "MESSAGE_CONTENT_TYPE_OR_EMPTY": "HTTP200_MESSAGE_CONTENT",
        "CONTENT_JSON_DECODE": "HTTP200_CONTENT_JSON_DECODE",
        "SCHEMA_VALIDATION": "HTTP200_STRICT_SCHEMA_VALIDATION",
        "PROPOSAL_COUNT": "HTTP200_PROPOSAL_COUNT",
        "USAGE_SHAPE": "HTTP200_USAGE_SHAPE",
        "TOKEN_USAGE_TYPE": "HTTP200_TOKEN_USAGE_TYPE",
        "TOKEN_CEILING": "HTTP200_TOKEN_CEILING",
    }
    enum_values = {item.value for item in LabApiResponseFailureReasonV1}
    if set(stages) != enum_values:
        raise Wave2IntegrationError("V5 reason vocabulary differs from Broker enum")
    return [
        {
            "reason_code": reason,
            "stage": stages[reason],
            "retry_eligible": "FALSE_TERMINAL_RESPONSE_CONTRACT",
        }
        for reason in sorted(stages)
    ]


def expected_v5_retry_policy(repo_root: Path) -> dict[str, Any]:
    v4_policy = _read_json(repo_root / V4_POLICY_REL)
    return {
        "schema": V5_POLICY_SCHEMA,
        "inherited_v4_policy_ref": _repo_ref(V4_POLICY_REL),
        "inherited_v4_policy_digest": V4_SEALED_DIGESTS[V4_POLICY_REL],
        "inherited_v4_policy_semantics_digest": sha256_digest(v4_policy),
        "proposal_slots_per_side": 8,
        "proposal_denominator_per_side": 8,
        "diagnostic_slot": {
            **deepcopy(v4_policy["diagnostic_slot"]),
            "slot_id": "PREFREEZE_V5_OBSERVABILITY_DIAGNOSTIC",
        },
        "response_contract_reason_vocabulary": diagnostic_reason_vocabulary(),
        "response_contract_failure": {
            "retry_eligible": False,
            "manual_patch": "FORBIDDEN",
            "successful_response_selection": "FORBIDDEN",
            "schema_relaxation": "FORBIDDEN",
            "candidate_admission": "FORBIDDEN",
            "mechanism_negative_evidence": False,
        },
        "missingness": deepcopy(v4_policy["missingness"]),
    }


def expected_prefreeze_v5_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v4_seal(repo_root)
    v4_manifest = _read_json(repo_root / V4_MANIFEST_REL)
    policy = expected_v5_retry_policy(repo_root)
    exact_contract = deepcopy(v4_manifest["exact_provider_contract"])
    exact_contract.update(
        {
            "logical_call_id": V5_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V5_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V5_OBSERVABILITY_DIAGNOSTIC",
        }
    )
    physical_identities = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V5_ATTEMPT_ID,
                    "diagnostic_slot": "PREFREEZE_V5_OBSERVABILITY_DIAGNOSTIC",
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v5_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v5_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    future_contract = deepcopy(v4_manifest["future_r1_scientific_contract"])
    return {
        "schema": V5_MANIFEST_SCHEMA,
        "attempt_identity": {
            "attempt_id": V5_ATTEMPT_ID,
            "base_commit": V4_HEAD,
            "base_parent": V4_PARENT,
            "base_tree": V4_TREE,
            "pre_outcome": True,
            "distinct_from_v1_v2_v3_v4_attempts": True,
            "old_attempt_call_session_db_identity_reuse": False,
        },
        "sealed_predecessor_evidence": {
            "v4_manifest_digest": V4_SEALED_DIGESTS[V4_MANIFEST_REL],
            "v4_attempt_digest": V4_SEALED_DIGESTS[V4_ATTEMPT_RECEIPT_REL],
            "v4_blocked_digest": V4_SEALED_DIGESTS[V4_BLOCKED_REL],
            "v4_provider_schema_digest": V4_SEALED_DIGESTS[V4_PROVIDER_SCHEMA_REL],
            "v4_local_validator_digest": V4_SEALED_DIGESTS[V4_VALIDATOR_REL],
            "v1_v2_v3_v4_preservation": (
                "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
            ),
        },
        "engineering_observability_change": {
            "only_change": "ALLOWLISTED_CONTENT_FREE_HTTP200_PARSE_REASON",
            "broker_source_ref": _repo_ref(BROKER_SOURCE_REL),
            "broker_source_digest": bytes_sha256(
                (repo_root / BROKER_SOURCE_REL).read_bytes()
            ),
            "reason_vocabulary": diagnostic_reason_vocabulary(),
            "reason_vocabulary_digest": sha256_digest(
                diagnostic_reason_vocabulary()
            ),
            "sqlite_schema_change": False,
            "persistence_field": "calls.error_detail_json.reason_code",
            "raw_response_persisted_on_failure": False,
            "provider_body_persisted_on_failure": False,
            "exception_text_persisted": False,
            "prompt_secret_header_endpoint_literal_persisted": False,
            "success_path_changed": False,
            "schema_validation_changed": False,
            "token_accounting_changed": False,
            "provider_release_semantics_changed": False,
            "retry_eligibility_changed": False,
        },
        "exact_provider_contract": exact_contract,
        "response_contract_equivalence": deepcopy(
            v4_manifest["response_contract_equivalence"]
        ),
        "bounded_retry": {
            "policy_ref": _repo_ref(V5_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "maximum_physical_attempts": 3,
            "maximum_retry_count": 2,
            "deterministic_backoff_ms": [1000, 3000],
            "physical_attempt_identities": physical_identities,
            "sqlite_calls_schema_digest": sha256_digest(
                {"table": "calls", "columns": list(SQLITE_CALL_COLUMNS)}
            ),
        },
        "future_r1_scientific_contract": future_contract,
        "v4_scientific_contract_digest": sha256_digest(
            v4_manifest["future_r1_scientific_contract"]
        ),
        "v5_scientific_contract_digest": sha256_digest(future_contract),
        "pre_outcome_counters": {
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
        },
        "r1_worker_launch_authorized": False,
    }


def expected_v5_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v5_manifest(repo_root)
    policy = expected_v5_retry_policy(repo_root)
    return {
        "schema": V5_AUTH_SCHEMA,
        "status": "AUTHORIZED_ONE_V5_OBSERVABILITY_DIAGNOSTIC_SLOT",
        "attempt_id": V5_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V5_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V5_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "broker_source_digest": manifest["engineering_observability_change"][
            "broker_source_digest"
        ],
        "reason_vocabulary_digest": manifest[
            "engineering_observability_change"
        ]["reason_vocabulary_digest"],
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v5(repo_root: Path) -> dict[str, Any]:
    verify_v4_seal(repo_root)
    policy = _load_exact(
        repo_root / V5_POLICY_REL,
        expected_v5_retry_policy(repo_root),
    )
    manifest = _load_exact(
        repo_root / V5_MANIFEST_REL,
        expected_prefreeze_v5_manifest(repo_root),
    )
    _load_exact(
        repo_root / V5_AUTH_REL,
        expected_v5_authorization(repo_root),
    )
    v4_manifest = _read_json(repo_root / V4_MANIFEST_REL)
    v4_contract = v4_manifest["exact_provider_contract"]
    v5_contract = manifest["exact_provider_contract"]
    identity_fields = {
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    }
    if {
        key: value for key, value in v5_contract.items() if key not in identity_fields
    } != {
        key: value for key, value in v4_contract.items() if key not in identity_fields
    }:
        raise Wave2IntegrationError("V5 changed the exact V4 Provider payload contract")
    if (
        manifest["future_r1_scientific_contract"]
        != v4_manifest["future_r1_scientific_contract"]
        or manifest["v4_scientific_contract_digest"]
        != manifest["v5_scientific_contract_digest"]
    ):
        raise Wave2IntegrationError("V5 changed the frozen R1 scientific contract")
    future = manifest["future_r1_scientific_contract"]
    shared = future["shared_call_contract_digest"]
    if (
        future["side_a_call_contract_digest"] != shared
        or future["side_b_call_contract_digest"] != shared
        or future["proposal_slots_per_side"] != 8
        or future["proposal_denominator_per_side"] != 8
    ):
        raise Wave2IntegrationError("V5 A/B symmetry or denominator changed")
    if policy["response_contract_reason_vocabulary"] != diagnostic_reason_vocabulary():
        raise Wave2IntegrationError("V5 policy reason vocabulary changed")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V5 is not pre-outcome")
    return manifest


def provider_free_v5_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v5(repo_root)
    return {
        "schema": V5_DRY_RUN_SCHEMA,
        "status": "PASS_PROVIDER_FREE_V5_OBSERVABILITY",
        "attempt_id": V5_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_v4_seals_verified": True,
        "exact_v4_payload_contract_verified": True,
        "v4_response_semantics_verified_unchanged": True,
        "ab_call_contract_symmetry_verified": True,
        "allowlisted_reason_vocabulary_verified": True,
        "sqlite_schema_unchanged": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


__all__ = [
    "BROKER_SOURCE_REL",
    "V5_ATTEMPT_ID",
    "V5_ATTEMPT_RECEIPT_REL",
    "V5_ATTEMPT_RECEIPT_SCHEMA",
    "V5_AUTH_REL",
    "V5_BLOCKED_REL",
    "V5_BLOCKED_SCHEMA",
    "V5_DRY_RUN_REL",
    "V5_LOGICAL_CALL_ID",
    "V5_MANIFEST_REL",
    "V5_POLICY_REL",
    "V5_PRIVATE_ROOT",
    "V5_READY_REL",
    "V5_READY_SCHEMA",
    "V5_SESSION_ID",
    "V5_VERIFICATION_REL",
    "V5_VERIFICATION_SCHEMA",
    "diagnostic_reason_vocabulary",
    "expected_prefreeze_v5_manifest",
    "expected_v5_authorization",
    "expected_v5_retry_policy",
    "provider_free_v5_dry_run",
    "v5_physical_root",
    "validate_prefreeze_v5",
    "verify_v4_seal",
]
