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
    V1_MANIFEST_REL,
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
    exact_v4_probe_request_payload,
    verify_v1_seal,
    verify_v2_seal,
    verify_v3_seal,
)
from .wave2_integration import Wave2IntegrationError


V4_HEAD = "45aa25211322d9f765e28447d5d1e885ccf3ee70"
V4_PARENT = "68014e714d6636268fc36b259bf13498f158e4cd"
V4_TREE = "cf2b333b2930be78557b8e0502e6db995ae42aa7"

V5_HEAD = "d0fd84ce8174a4bfc913d98e1d0fa58e7480f363"
V5_PARENT = "45aa25211322d9f765e28447d5d1e885ccf3ee70"
V5_TREE = "72c0a36c1ebde8ffc80368cba624b4886e3fdf3d"

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

V5_SEALED_DIGESTS: dict[Path, str] = {
    V5_MANIFEST_REL: "e4b5468c2f0711cfc0b7d52330efa4240cc26bc41c0d0a45b8416bc01da3b936",
    V5_POLICY_REL: "5bfa9a9f11cf817f8f6c1a38392f514e0826665d93899256f9c7daf8b5120749",
    V5_AUTH_REL: "2ff14b4aa657743a3c76ad3738e20bc5e8f6e5b42f6f552f6b553c339b25af3d",
    V5_DRY_RUN_REL: "c6ee04e0ddb5e0636f861ca5c5dfe7b2f57e84b6bb283d49b84a26f62c86869f",
    V5_ATTEMPT_RECEIPT_REL: "8f6b74382d11b967d2b67f7771ad37b4bdb5c3a5e92446b885a44d1eb4779fed",
    V5_BLOCKED_REL: "5254d1d517f0f521d44e9fccf300f025f754cdab0d157a89d7ae7685b9ee8031",
}

V6_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v6-20260801"
V6_MANIFEST_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-attempt.v6"
V6_POLICY_SCHEMA = "recclaw.research-line.r1-provider-retry-policy.v6"
V6_AUTH_SCHEMA = "recclaw.research-line.prefreeze-v6-authorization.v1"
V6_RELEASE_SCHEMA = "recclaw.research-line.provider-release-contract.v6"
V6_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v6-provider-attempt-receipt.v1"
)
V6_DRY_RUN_SCHEMA = "recclaw.research-line.prefreeze-v6-dry-run-receipt.v1"
V6_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v6-blocked-receipt.v1"
V6_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v6"
V6_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v6-verification-receipt.v1"
)

V6_REQUESTED_MODEL_ALIAS = "gpt-5.4"
V6_REQUIRED_RETURNED_SNAPSHOT = "gpt-5.4-2026-03-05"
V6_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v6-exact-model-pair-slot"
V6_SESSION_ID = "fresh-open-spec-prefreeze-v6-exact-model-pair-slot-session"
V6_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v6_private")

V6_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V6.json"
V6_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V6.json"
V6_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V6.json"
V6_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V6_AUTHORIZATION.json"
V6_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V6.json"
)
V6_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V6_BLOCKED_RECEIPT.json"
V6_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V6_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V6_DRY_RUN_RECEIPT.json"
V6_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V6_VERIFICATION_RECEIPT.json"

V6_HEAD = "6d10e44baf73824d6e69770e4632aea38eb79de4"
V6_PARENT = "d0fd84ce8174a4bfc913d98e1d0fa58e7480f363"
V6_TREE = "46666c6af830e2e4aafdea8e17be0a99bdf2ee6c"
V6_SEALED_DIGESTS: dict[Path, str] = {
    V6_RELEASE_REL: "d480d615096e5a62bfa22f89407ef160012657df3230a0aeade9401ce2453f44",
    V6_POLICY_REL: "58e6cf43c7c85f08cdefa0d05ee9e71526db0e9fa239dae4f86bfc05ea294194",
    V6_MANIFEST_REL: "b9afcfbe38f3f2a5ecb013f477a0181776dd9314739868b761843d53a7ff57f2",
    V6_AUTH_REL: "3e0d884569836a84678f12800ef8105e2450f60a70a4d25c996cbe5e99529907",
    V6_DRY_RUN_REL: "3f0f5bee76266b71a49ebcfd3583e6c49277a0ddc0fdd6fa189d75d74116e6a5",
    V6_ATTEMPT_RECEIPT_REL: "c721e773ee0e06f49d62ad7c3d91ab4208a1c8c9b10eb44fc6f404ad5198a801",
    V6_BLOCKED_REL: "5072e2b73e2e3e53e9546a3cf56970fe0e6f360aa43e91d29ed339c38d2e25b4",
}

V7_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v7-20260801"
V7_DIAGNOSTIC_TOKEN_CEILING = 6000
V7_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v7-provider-attempt-receipt.v1"
)
V7_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v7-blocked-receipt.v1"
V7_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v7"
V7_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v7-verification-receipt.v1"
)
V7_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v7-ceiling-alignment-slot"
V7_SESSION_ID = "fresh-open-spec-prefreeze-v7-ceiling-alignment-slot-session"
V7_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v7_private")
V7_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V7.json"
V7_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V7.json"
V7_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V7.json"
V7_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V7_AUTHORIZATION.json"
V7_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V7.json"
)
V7_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V7_BLOCKED_RECEIPT.json"
V7_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V7_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V7_DRY_RUN_RECEIPT.json"
V7_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V7_VERIFICATION_RECEIPT.json"


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


def v6_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V6 physical attempt ordinal must be 1..3")
    return V6_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v5_seal(repo_root: Path) -> dict[str, str]:
    """Prove that V1--V5 accepted evidence remains byte-identical."""

    verify_v4_seal(repo_root)
    for relative, digest in V5_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V5 bytes changed: {relative.as_posix()}"
            )
    broker_digest = bytes_sha256((repo_root / BROKER_SOURCE_REL).read_bytes())
    if broker_digest != "3e255e71e128011e6533b259fcb3455da378b9ecb398862aed3d5eac8ae00441":
        raise Wave2IntegrationError("accepted V5 Broker source bytes changed")
    return {
        relative.as_posix(): digest
        for relative, digest in V5_SEALED_DIGESTS.items()
    }


def validate_v6_exact_model_pair(
    *, requested_model_alias: str, returned_model: str
) -> None:
    """Accept only the two exact, pre-frozen model identity literals."""

    if requested_model_alias != V6_REQUESTED_MODEL_ALIAS:
        raise Wave2IntegrationError("V6 requested model alias mismatch")
    if returned_model != V6_REQUIRED_RETURNED_SNAPSHOT:
        raise Wave2IntegrationError("V6 returned model snapshot mismatch")


def _v6_preserved_scientific_identity(repo_root: Path) -> dict[str, Any]:
    v1 = _read_json(repo_root / V1_MANIFEST_REL)
    preserved = {
        key: deepcopy(v1[key])
        for key in (
            "open_spec_contract",
            "r1_identity",
            "r2_identity",
            "runtime_identity",
            "evidence_policy",
            "shared_implementation",
        )
    }
    return {
        "source_manifest_ref": _repo_ref(V1_MANIFEST_REL),
        "source_manifest_digest": bytes_sha256(
            (repo_root / V1_MANIFEST_REL).read_bytes()
        ),
        "preserved_fields": sorted(preserved),
        "preserved_fields_digest": sha256_digest(preserved),
        "seed_root_db_candidate_package_outcome_memory_lineage_unchanged": True,
        "runtime_dependency_missingness_gate_threshold_analysis_unchanged": True,
        "shared_origin_blind_implementer_qualifier_unchanged": True,
        "manual_patch_prohibition_unchanged": True,
    }


def expected_v6_provider_release(repo_root: Path) -> dict[str, Any]:
    """Return the exact-pair release binding without changing transport bytes."""

    verify_v5_seal(repo_root)
    v5 = _read_json(repo_root / V5_MANIFEST_REL)
    v1 = _read_json(repo_root / V1_MANIFEST_REL)
    exact = v5["exact_provider_contract"]
    proposal = v1["shared_proposal_call"]
    preimage = {
        "schema": V6_RELEASE_SCHEMA,
        "transport_release_ref": exact["provider_release_ref"],
        "transport_release_artifact_digest": V4_SEALED_DIGESTS[V4_RELEASE_REL],
        "transport_release_contract_digest": exact["provider_release_digest"],
        "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
        "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
        "endpoint_digest": exact["endpoint_digest"],
        "credential_config_digest": exact["credential_config_digest"],
        "credential_identity_digest": exact["credential_identity_digest"],
        "response_schema_digest": exact["response_schema_digest"],
        "local_uniqueness_contract_digest": v5[
            "response_contract_equivalence"
        ]["local_uniqueness_contract_digest"],
        "prompt_digest": exact["prompt_digest"],
        "tool_policy_digest": exact["tool_policy_digest"],
        "request_mode": exact["request_mode"],
        "temperature": exact["temperature"],
        "diagnostic_slot_call_count": 1,
        "diagnostic_expected_proposals": 1,
        "diagnostic_token_budget": exact["token_budget"],
        "transport_max_total_tokens_per_call": proposal["token_budget"],
        "future_r1_call_count_per_side": proposal["call_count"],
        "future_r1_proposal_budget_per_side": proposal[
            "proposal_budget_per_side"
        ],
        "future_r1_expected_proposals_per_slot": proposal[
            "expected_proposals_per_call"
        ],
        "future_r1_token_budget_per_call": proposal["token_budget"],
        "alias_or_snapshot_matching": "EXACT_LITERAL_ONLY",
        "prefix_regex_startswith_or_arbitrary_snapshot": "FORBIDDEN",
        "silent_alias_fallback_model_or_endpoint_change": "FORBIDDEN",
    }
    return {**preimage, "release_digest": sha256_digest(preimage)}


def expected_v6_retry_policy(repo_root: Path) -> dict[str, Any]:
    policy = deepcopy(_read_json(repo_root / V5_POLICY_REL))
    policy["schema"] = V6_POLICY_SCHEMA
    policy["inherited_v5_policy_ref"] = _repo_ref(V5_POLICY_REL)
    policy["inherited_v5_policy_digest"] = V5_SEALED_DIGESTS[V5_POLICY_REL]
    policy.pop("inherited_v4_policy_ref", None)
    policy.pop("inherited_v4_policy_digest", None)
    policy.pop("inherited_v4_policy_semantics_digest", None)
    policy["diagnostic_slot"]["slot_id"] = "PREFREEZE_V6_EXACT_MODEL_PAIR"
    policy["exact_model_pair_failure"] = {
        "failure_class": "RETURNED_MODEL_PAIR_MISMATCH",
        "retry_eligible": False,
        "manual_patch": "FORBIDDEN",
        "successful_response_selection": "FORBIDDEN",
        "candidate_admission": "FORBIDDEN",
        "mechanism_negative_evidence": False,
    }
    return policy


def expected_prefreeze_v6_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v5_seal(repo_root)
    v5 = _read_json(repo_root / V5_MANIFEST_REL)
    release = expected_v6_provider_release(repo_root)
    policy = expected_v6_retry_policy(repo_root)
    exact = deepcopy(v5["exact_provider_contract"])
    exact.update(
        {
            "model": V6_REQUESTED_MODEL_ALIAS,
            "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
            "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
            "model_identity_pair_digest": sha256_digest(
                {
                    "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
                    "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
                }
            ),
            "transport_provider_release_digest": exact[
                "provider_release_digest"
            ],
            "provider_release_ref": _repo_ref(V6_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "logical_call_id": V6_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V6_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V6_EXACT_MODEL_PAIR",
        }
    )
    physical_identities = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V6_ATTEMPT_ID,
                    "diagnostic_slot": "PREFREEZE_V6_EXACT_MODEL_PAIR",
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v6_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v6_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    future = deepcopy(v5["future_r1_scientific_contract"])
    shared_call_digest = sha256_digest(
        {
            "inherited_v5_shared_call_contract_digest": future[
                "shared_call_contract_digest"
            ],
            "v6_provider_release_contract_digest": release["release_digest"],
        }
    )
    future.update(
        {
            "shared_call_contract_digest": shared_call_digest,
            "side_a_call_contract_digest": shared_call_digest,
            "side_b_call_contract_digest": shared_call_digest,
        }
    )
    return {
        "schema": V6_MANIFEST_SCHEMA,
        "attempt_identity": {
            "attempt_id": V6_ATTEMPT_ID,
            "base_commit": V5_HEAD,
            "base_parent": V5_PARENT,
            "base_tree": V5_TREE,
            "pre_outcome": True,
            "distinct_from_v1_v2_v3_v4_v5_attempts": True,
            "old_attempt_call_session_db_identity_reuse": False,
        },
        "sealed_predecessor_evidence": {
            "v5_manifest_digest": V5_SEALED_DIGESTS[V5_MANIFEST_REL],
            "v5_attempt_digest": V5_SEALED_DIGESTS[V5_ATTEMPT_RECEIPT_REL],
            "v5_blocked_digest": V5_SEALED_DIGESTS[V5_BLOCKED_REL],
            "v1_v2_v3_v4_v5_preservation": (
                "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
            ),
        },
        "accepted_v5_engineering_history": {
            "http_200_parse_failure_observability": (
                "ALLOWLISTED_CONTENT_FREE_REASON_CODE"
            ),
            "http_error_persistence": "STATUS_ONLY_NO_PROVIDER_BODY",
            "sqlite_schema_changed": False,
            "accepted_v5_artifacts_rewritten": False,
        },
        "authorized_v6_protocol_change": {
            "only_scientific_protocol_change": (
                "EXACT_REQUEST_ALIAS_TO_EXACT_RETURNED_SNAPSHOT_BINDING"
            ),
            "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
            "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
            "matching": "TWO_EXACT_LITERAL_EQUALITIES_FAIL_CLOSED",
            "generic_alias_acceptance": "FORBIDDEN",
        },
        "provider_release_contract": release,
        "exact_provider_contract": exact,
        "response_contract_equivalence": deepcopy(
            v5["response_contract_equivalence"]
        ),
        "bounded_retry": {
            "policy_ref": _repo_ref(V6_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "maximum_physical_attempts": 3,
            "maximum_retry_count": 2,
            "deterministic_backoff_ms": [1000, 3000],
            "physical_attempt_identities": physical_identities,
            "sqlite_calls_schema_digest": v5["bounded_retry"][
                "sqlite_calls_schema_digest"
            ],
        },
        "preserved_scientific_identity": _v6_preserved_scientific_identity(
            repo_root
        ),
        "future_r1_scientific_contract": future,
        "inherited_v5_scientific_contract_digest": sha256_digest(
            v5["future_r1_scientific_contract"]
        ),
        "v6_scientific_contract_digest": sha256_digest(future),
        "pre_outcome_counters": deepcopy(v5["pre_outcome_counters"]),
        "r1_worker_launch_authorized": False,
    }


def expected_v6_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v6_manifest(repo_root)
    policy = expected_v6_retry_policy(repo_root)
    release = expected_v6_provider_release(repo_root)
    return {
        "schema": V6_AUTH_SCHEMA,
        "status": "AUTHORIZED_ONE_V6_EXACT_MODEL_PAIR_DIAGNOSTIC_SLOT",
        "attempt_id": V6_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V6_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V6_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V6_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
        "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
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


def validate_prefreeze_v6(repo_root: Path) -> dict[str, Any]:
    verify_v5_seal(repo_root)
    release = _load_exact(
        repo_root / V6_RELEASE_REL, expected_v6_provider_release(repo_root)
    )
    policy = _load_exact(
        repo_root / V6_POLICY_REL, expected_v6_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V6_MANIFEST_REL, expected_prefreeze_v6_manifest(repo_root)
    )
    _load_exact(repo_root / V6_AUTH_REL, expected_v6_authorization(repo_root))
    exact = manifest["exact_provider_contract"]
    validate_v6_exact_model_pair(
        requested_model_alias=exact["requested_model_alias"],
        returned_model=exact["required_returned_snapshot"],
    )
    if (
        exact["provider_release_digest"] != release["release_digest"]
        or exact["request_payload_digest"]
        != _read_json(repo_root / V5_MANIFEST_REL)["exact_provider_contract"][
            "request_payload_digest"
        ]
    ):
        raise Wave2IntegrationError("V6 release or request payload binding changed")
    future = manifest["future_r1_scientific_contract"]
    shared = future["shared_call_contract_digest"]
    if (
        future["side_a_call_contract_digest"] != shared
        or future["side_b_call_contract_digest"] != shared
        or future["proposal_slots_per_side"] != 8
        or future["proposal_denominator_per_side"] != 8
    ):
        raise Wave2IntegrationError("V6 A/B symmetry or denominator changed")
    if (
        policy["diagnostic_slot"]["maximum_total_physical_attempts"] != 3
        or policy["diagnostic_slot"]["deterministic_backoff_ms_after_failure"]
        != [1000, 3000]
        or policy["exact_model_pair_failure"]["retry_eligible"] is not False
    ):
        raise Wave2IntegrationError("V6 retry or model-pair failure policy changed")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V6 is not pre-outcome")
    return manifest


def provider_free_v6_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v6(repo_root)
    return {
        "schema": V6_DRY_RUN_SCHEMA,
        "status": "PASS_PROVIDER_FREE_V6_EXACT_MODEL_PAIR",
        "attempt_id": V6_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_v4_v5_seals_verified": True,
        "exact_requested_alias_verified": True,
        "exact_required_returned_snapshot_verified": True,
        "pair_bound_provider_release_verified": True,
        "exact_v5_request_payload_verified": True,
        "local_uniqueness_before_downstream_verified": True,
        "ab_call_contract_symmetry_verified": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def v7_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V7 physical attempt ordinal must be 1..3")
    return V7_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v6_seal(repo_root: Path) -> dict[str, str]:
    verify_v5_seal(repo_root)
    for relative, digest in V6_SEALED_DIGESTS.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V6 bytes changed: {relative.as_posix()}"
            )
    return {
        relative.as_posix(): digest
        for relative, digest in V6_SEALED_DIGESTS.items()
    }


def exact_v7_probe_request_payload_digest(repo_root: Path) -> str:
    payload = exact_v4_probe_request_payload(repo_root)
    payload["max_tokens"] = V7_DIAGNOSTIC_TOKEN_CEILING
    return sha256_digest(payload)


def expected_v7_provider_release(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    release = deepcopy(_read_json(repo_root / V6_RELEASE_REL))
    release.pop("release_digest")
    release["schema"] = "recclaw.research-line.provider-release-contract.v7"
    release["diagnostic_token_budget"] = V7_DIAGNOSTIC_TOKEN_CEILING
    return {**release, "release_digest": sha256_digest(release)}


def expected_v7_retry_policy(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    policy = deepcopy(_read_json(repo_root / V6_POLICY_REL))
    policy["schema"] = "recclaw.research-line.r1-provider-retry-policy.v7"
    policy.pop("inherited_v5_policy_ref", None)
    policy.pop("inherited_v5_policy_digest", None)
    policy["inherited_v6_policy_ref"] = _repo_ref(V6_POLICY_REL)
    policy["inherited_v6_policy_digest"] = V6_SEALED_DIGESTS[V6_POLICY_REL]
    policy["diagnostic_slot"]["slot_id"] = (
        "PREFREEZE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
    )
    return policy


def expected_prefreeze_v7_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    v6 = _read_json(repo_root / V6_MANIFEST_REL)
    release = expected_v7_provider_release(repo_root)
    policy = expected_v7_retry_policy(repo_root)
    manifest = deepcopy(v6)
    manifest["schema"] = "recclaw.research-line.r1-r2-prefreeze-attempt.v7"
    manifest["attempt_identity"] = {
        "attempt_id": V7_ATTEMPT_ID,
        "base_commit": V6_HEAD,
        "base_parent": V6_PARENT,
        "base_tree": V6_TREE,
        "pre_outcome": True,
        "distinct_from_v1_v2_v3_v4_v5_v6_attempts": True,
        "old_attempt_call_session_db_identity_reuse": False,
    }
    manifest["sealed_predecessor_evidence"] = {
        "v6_manifest_digest": V6_SEALED_DIGESTS[V6_MANIFEST_REL],
        "v6_attempt_digest": V6_SEALED_DIGESTS[V6_ATTEMPT_RECEIPT_REL],
        "v6_blocked_digest": V6_SEALED_DIGESTS[V6_BLOCKED_REL],
        "v1_v2_v3_v4_v5_v6_preservation": (
            "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL"
        ),
    }
    manifest["authorized_v7_infrastructure_alignment"] = {
        "decision": "CONTRACT_EQUIVALENT_INFRASTRUCTURE_FIX",
        "only_contract_delta": (
            "DIAGNOSTIC_TOKEN_CEILING_2000_TO_FROZEN_6000"
        ),
        "prior_diagnostic_token_ceiling": 2000,
        "diagnostic_token_ceiling": V7_DIAGNOSTIC_TOKEN_CEILING,
        "transport_and_future_r1_token_ceiling": 6000,
        "research_proposal_budget_changed": False,
        "schema_semantics_model_pair_or_uniqueness_changed": False,
    }
    manifest["provider_release_contract"] = release
    exact = manifest["exact_provider_contract"]
    exact.update(
        {
            "provider_release_ref": _repo_ref(V7_RELEASE_REL),
            "provider_release_artifact_digest": bytes_sha256(
                canonical_json_bytes(release)
            ),
            "provider_release_digest": release["release_digest"],
            "request_payload_digest": exact_v7_probe_request_payload_digest(
                repo_root
            ),
            "token_budget": V7_DIAGNOSTIC_TOKEN_CEILING,
            "logical_call_id": V7_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V7_SESSION_ID,
            "diagnostic_slot_id": (
                "PREFREEZE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
            ),
        }
    )
    identities = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V7_ATTEMPT_ID,
                    "diagnostic_slot": (
                        "PREFREEZE_V7_DIAGNOSTIC_CEILING_ALIGNMENT"
                    ),
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v7_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v7_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    manifest["bounded_retry"].update(
        {
            "policy_ref": _repo_ref(V7_POLICY_REL),
            "policy_digest": bytes_sha256(canonical_json_bytes(policy)),
            "physical_attempt_identities": identities,
        }
    )
    future = manifest["future_r1_scientific_contract"]
    manifest.pop("inherited_v5_scientific_contract_digest", None)
    manifest.pop("v6_scientific_contract_digest", None)
    manifest["inherited_v6_scientific_contract_digest"] = sha256_digest(
        v6["future_r1_scientific_contract"]
    )
    manifest["v7_scientific_contract_digest"] = sha256_digest(future)
    return manifest


def expected_v7_authorization(repo_root: Path) -> dict[str, Any]:
    manifest = expected_prefreeze_v7_manifest(repo_root)
    policy = expected_v7_retry_policy(repo_root)
    release = expected_v7_provider_release(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v7-authorization.v1",
        "status": "AUTHORIZED_ONE_V7_CEILING_ALIGNMENT_DIAGNOSTIC_SLOT",
        "attempt_id": V7_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V7_MANIFEST_REL),
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "retry_policy_ref": _repo_ref(V7_POLICY_REL),
        "retry_policy_digest": bytes_sha256(canonical_json_bytes(policy)),
        "provider_release_ref": _repo_ref(V7_RELEASE_REL),
        "provider_release_artifact_digest": bytes_sha256(
            canonical_json_bytes(release)
        ),
        "provider_release_contract_digest": release["release_digest"],
        "requested_model_alias": V6_REQUESTED_MODEL_ALIAS,
        "required_returned_snapshot": V6_REQUIRED_RETURNED_SNAPSHOT,
        "diagnostic_token_ceiling": V7_DIAGNOSTIC_TOKEN_CEILING,
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


def validate_prefreeze_v7(repo_root: Path) -> dict[str, Any]:
    verify_v6_seal(repo_root)
    release = _load_exact(
        repo_root / V7_RELEASE_REL, expected_v7_provider_release(repo_root)
    )
    policy = _load_exact(
        repo_root / V7_POLICY_REL, expected_v7_retry_policy(repo_root)
    )
    manifest = _load_exact(
        repo_root / V7_MANIFEST_REL, expected_prefreeze_v7_manifest(repo_root)
    )
    _load_exact(repo_root / V7_AUTH_REL, expected_v7_authorization(repo_root))
    v6 = _read_json(repo_root / V6_MANIFEST_REL)
    exact = manifest["exact_provider_contract"]
    validate_v6_exact_model_pair(
        requested_model_alias=exact["requested_model_alias"],
        returned_model=exact["required_returned_snapshot"],
    )
    allowed_exact_delta = {
        "provider_release_ref",
        "provider_release_artifact_digest",
        "provider_release_digest",
        "request_payload_digest",
        "token_budget",
        "logical_call_id",
        "proposal_generation_session_id",
        "diagnostic_slot_id",
    }
    if {
        key: value
        for key, value in exact.items()
        if key not in allowed_exact_delta
    } != {
        key: value
        for key, value in v6["exact_provider_contract"].items()
        if key not in allowed_exact_delta
    }:
        raise Wave2IntegrationError("V7 changed a non-diagnostic Provider field")
    if (
        exact["token_budget"] != V7_DIAGNOSTIC_TOKEN_CEILING
        or release["diagnostic_token_budget"]
        != release["transport_max_total_tokens_per_call"]
        or release["diagnostic_token_budget"]
        != release["future_r1_token_budget_per_call"]
    ):
        raise Wave2IntegrationError("V7 diagnostic ceiling is not exact 6000")
    if (
        manifest["future_r1_scientific_contract"]
        != v6["future_r1_scientific_contract"]
        or manifest["response_contract_equivalence"]
        != v6["response_contract_equivalence"]
        or manifest["preserved_scientific_identity"]
        != v6["preserved_scientific_identity"]
    ):
        raise Wave2IntegrationError("V7 changed the R1 or response contract")
    if (
        policy["diagnostic_slot"]["maximum_total_physical_attempts"] != 3
        or policy["diagnostic_slot"]["deterministic_backoff_ms_after_failure"]
        != [1000, 3000]
        or policy["response_contract_failure"]["retry_eligible"] is not False
        or "SEMANTIC_RESPONSE_CONTRACT_FAILURE"
        not in policy["diagnostic_slot"]["terminal_no_retry_failure_classes"]
    ):
        raise Wave2IntegrationError("V7 retry/terminal policy changed")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V7 is not pre-outcome")
    return manifest


def provider_free_v7_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v7(repo_root)
    return {
        "schema": "recclaw.research-line.prefreeze-v7-dry-run-receipt.v1",
        "status": "PASS_PROVIDER_FREE_V7_DIAGNOSTIC_CEILING_ALIGNMENT",
        "attempt_id": V7_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_v4_v5_v6_seals_verified": True,
        "only_diagnostic_ceiling_delta_verified": True,
        "diagnostic_transport_future_ceiling_equal_6000": True,
        "future_r1_contract_exact_v6_verified": True,
        "exact_model_pair_verified": True,
        "strict_schema_sentinel_local_uniqueness_verified": True,
        "token_ceiling_terminal_verified": True,
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
    "V5_SEALED_DIGESTS",
    "V6_ATTEMPT_ID",
    "V6_ATTEMPT_RECEIPT_REL",
    "V6_ATTEMPT_RECEIPT_SCHEMA",
    "V6_AUTH_REL",
    "V6_BLOCKED_REL",
    "V6_BLOCKED_SCHEMA",
    "V6_DRY_RUN_REL",
    "V6_LOGICAL_CALL_ID",
    "V6_MANIFEST_REL",
    "V6_POLICY_REL",
    "V6_PRIVATE_ROOT",
    "V6_READY_REL",
    "V6_READY_SCHEMA",
    "V6_RELEASE_REL",
    "V6_REQUESTED_MODEL_ALIAS",
    "V6_REQUIRED_RETURNED_SNAPSHOT",
    "V6_SESSION_ID",
    "V6_VERIFICATION_REL",
    "V6_VERIFICATION_SCHEMA",
    "expected_prefreeze_v6_manifest",
    "expected_v6_authorization",
    "expected_v6_provider_release",
    "expected_v6_retry_policy",
    "provider_free_v6_dry_run",
    "v6_physical_root",
    "validate_prefreeze_v6",
    "validate_v6_exact_model_pair",
    "verify_v5_seal",
    "V6_SEALED_DIGESTS",
    "V7_ATTEMPT_ID",
    "V7_ATTEMPT_RECEIPT_REL",
    "V7_ATTEMPT_RECEIPT_SCHEMA",
    "V7_AUTH_REL",
    "V7_BLOCKED_REL",
    "V7_BLOCKED_SCHEMA",
    "V7_DIAGNOSTIC_TOKEN_CEILING",
    "V7_DRY_RUN_REL",
    "V7_LOGICAL_CALL_ID",
    "V7_MANIFEST_REL",
    "V7_POLICY_REL",
    "V7_PRIVATE_ROOT",
    "V7_READY_REL",
    "V7_READY_SCHEMA",
    "V7_RELEASE_REL",
    "V7_SESSION_ID",
    "V7_VERIFICATION_REL",
    "V7_VERIFICATION_SCHEMA",
    "exact_v7_probe_request_payload_digest",
    "expected_prefreeze_v7_manifest",
    "expected_v7_authorization",
    "expected_v7_provider_release",
    "expected_v7_retry_policy",
    "provider_free_v7_dry_run",
    "v7_physical_root",
    "validate_prefreeze_v7",
    "verify_v6_seal",
]
