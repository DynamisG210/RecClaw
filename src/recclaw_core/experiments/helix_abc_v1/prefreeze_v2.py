"""Fail-closed Prefreeze V2/V3 identity and provider-free validation.

Prefreeze V2 is an additive, pre-outcome amendment.  It seals the accepted V1
negative evidence and changes only the future R1 provider retry policy.  The
one diagnostic re-probe authorized by V2 is not an R1 proposal slot and never
uses that retry policy.

Prefreeze V3 is another additive attempt.  It reuses the exact V1 physical
request payload and applies the bounded transient retry policy frozen by V2.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .canonical import bytes_sha256, canonical_json_bytes, canonical_value, sha256_digest
from .wave2_integration import GPT_5_4_MODEL_DIGEST, Wave2IntegrationError
from .v4_response_contract import (
    V1_UNIQUE_SCHEMA_PATHS,
    V4_LOCAL_ARRAY_PATHS,
    V4_UNIQUENESS_CONTRACT,
    V4_UNIQUENESS_CONTRACT_DIGEST,
    V4LocalUniquenessError,
    derive_v4_provider_schema,
    validate_v4_response_contract,
    validate_v4_schema_derivation,
)


V1_HEAD = "34357c6d7fe2a2a3c5fa77d17f6f57a9190bd9ce"
V1_PARENT = "fbcb410e9bdf5052ed48a7e54152eca3917c8dd9"
V1_TREE = "e9d02e8a7e89d20e70ee06c6e12ab87f6edd2a6d"
V1_MANIFEST_SHA256 = "baad034df0bf423bf2ffc90847848d461be5fdbef1d8b4ff8d68386d6e508ab4"
V1_BLOCKED_SHA256 = "c491f2f1776f912b4ea1cab5d7db66ddc2d83d6ea2a138c9dc9e09d6c0e7fef3"
V1_PROBE_SHA256 = "35c38fff6802a11d41f06a45258dce0df9fb712854030add7246b7111815e370"

ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v2-20260801"
MANIFEST_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-attempt.v2"
POLICY_SCHEMA = "recclaw.research-line.r1-provider-retry-policy.v2"
AUTH_SCHEMA = "recclaw.research-line.prefreeze-v2-reprobe-authorization.v1"
DRY_RUN_SCHEMA = "recclaw.research-line.prefreeze-v2-dry-run-receipt.v1"
PROBE_SCHEMA = "recclaw.research-line.fresh-open-spec-endpoint-reprobe.v2"
BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v2-blocked-receipt.v1"
READY_SCHEMA = "recclaw.research-line.r1-prefreeze-v2-ready-receipt.v1"

MODEL = "gpt-5.4"
PROBE_TOKEN_CEILING = 2000
LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v2-diagnostic-reprobe"
SESSION_ID = "fresh-open-spec-prefreeze-v2-diagnostic-reprobe-session"
PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_reprobe_v2_private")

DOC_ROOT_REL = Path("docs/research_line/vnext")
RESOURCE_ROOT_REL = Path("src/recclaw_core/experiments/helix_abc_v1/resources")
V1_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST.json"
V1_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_BLOCKED_RECEIPT.json"
V1_PROBE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_PROBE_RECEIPT_V1.json"
RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V1.json"
POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V2.json"
MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V2.json"
AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V2_REPROBE_AUTHORIZATION.json"
PROBE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_REPROBE_RECEIPT_V2.json"
BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V2_BLOCKED_RECEIPT.json"
READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_V2_READY_RECEIPT.json"
DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V2_DRY_RUN_RECEIPT.json"
SCHEMA_REL = RESOURCE_ROOT_REL / "fresh_open_spec_proposal_response_v1.schema.json"
SENTINEL_REL = RESOURCE_ROOT_REL / "fresh_open_spec_schema_probe_payload_v1.json"
PROMPT_REL = RESOURCE_ROOT_REL / "fresh_open_spec_proposal_prompt_v1.txt"
TOOL_REL = RESOURCE_ROOT_REL / "fresh_open_spec_tool_policy_v1.json"
SCHEDULE_REL = RESOURCE_ROOT_REL / "fresh_open_spec_call_schedule_v1.json"


def _repo_ref(relative: Path) -> str:
    return f"repo:{relative.as_posix()}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Wave2IntegrationError(f"invalid JSON resource: {path.name}") from exc
    if not isinstance(value, dict):
        raise Wave2IntegrationError(f"JSON resource must be an object: {path.name}")
    return value


def _canonical_file(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if path.read_bytes() != canonical_json_bytes(value):
        raise Wave2IntegrationError(f"resource is not canonical JSON: {path.name}")
    return value


def verify_v1_seal(repo_root: Path) -> dict[str, str]:
    """Verify the three accepted V1 artifacts without rewriting any byte."""

    expected = {
        V1_MANIFEST_REL: V1_MANIFEST_SHA256,
        V1_BLOCKED_REL: V1_BLOCKED_SHA256,
        V1_PROBE_REL: V1_PROBE_SHA256,
    }
    for relative, digest in expected.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(f"sealed V1 bytes changed: {relative.as_posix()}")
    return {relative.as_posix(): digest for relative, digest in expected.items()}


def expected_retry_policy() -> dict[str, Any]:
    """Return the complete future-R1 retry policy authorized by the user."""

    return {
        "schema": POLICY_SCHEMA,
        "scope": "FUTURE_FRESH_R1_PROPOSAL_PROVIDER_CALLS_ONLY",
        "proposal_slots_per_side": 8,
        "proposal_denominator_per_side": 8,
        "ab_call_contract_symmetry_required": True,
        "slot_attempt_semantics": {
            "retry_is_same_slot": True,
            "same_logical_call_id_required": True,
            "same_request_payload_digest_required": True,
            "retry_is_new_proposal": False,
            "proposal_denominator_change": "FORBIDDEN",
            "successful_response_selection": "FORBIDDEN",
            "record_every_physical_attempt": True,
            "record_attempt_count": True,
            "record_failure_class": True,
        },
        "retry_limit": {
            "maximum_additional_physical_attempts": 2,
            "maximum_total_physical_attempts_per_slot": 3,
            "deterministic_backoff_ms_after_failure": [1000, 3000],
        },
        "retryable_failure_classes": [
            "NETWORK_TIMEOUT",
            "NETWORK_RESET",
            "HTTP_408",
            "HTTP_429",
            "HTTP_5XX",
            "PROVIDER_OVERLOAD",
        ],
        "terminal_no_retry_failure_classes": [
            "HTTP_400_SCHEMA_CONTRACT",
            "HTTP_401_AUTH",
            "HTTP_403_AUTH",
            "OTHER_DETERMINISTIC_HTTP_4XX",
            "CONTENT_NOT_JSON",
            "JSON_PARSE_FAILURE",
            "SCHEMA_VALIDATION_FAILURE",
            "SEMANTIC_RESPONSE_CONTRACT_FAILURE",
        ],
        "response_contract_rules": {
            "schema_relaxation": "FORBIDDEN",
            "manual_response_patch": "FORBIDDEN",
            "replacement_proposal_call": "FORBIDDEN",
            "content_not_json_retry": "FORBIDDEN",
            "pick_successful_attempt": "FORBIDDEN",
        },
        "exhausted_slot_missingness": {
            "slot_consumed": True,
            "candidate_status": "NO_CANDIDATE",
            "analysis_class": "MISSING_PROVIDER_OR_ENGINEERING_FAILURE",
            "mechanism_negative_evidence": False,
            "replacement_slot": "FORBIDDEN",
        },
        "special_v2_diagnostic_reprobe": {
            "policy_applies": False,
            "maximum_physical_attempts": 1,
            "retry_count": 0,
        },
    }


def schema_probe_prompt(repo_root: Path) -> str:
    """Build the exact pre-authored V1 sentinel prompt, without research text."""

    sentinel = _read_json(repo_root / SENTINEL_REL)
    return (
        "This is a one-time transport, authentication, exact-model, and strict-"
        "JSON-schema qualification probe. It is not research ideation. Echo the "
        "following JSON object exactly, with no additions, omissions, prose, "
        "markdown, or tool use. The response is a pre-authored sentinel and must "
        "never be projected into OpenResearchSpecV1, resolved, implemented, "
        "qualified, admitted, counted, trained, or interpreted as a candidate:\n"
        + canonical_json_bytes(sentinel).decode("utf-8")
    )


def exact_probe_request_payload(repo_root: Path) -> dict[str, Any]:
    """Return the physical HTTP JSON body used by both V1 and V2 probes."""

    schema = _read_json(repo_root / SCHEMA_REL)
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": schema_probe_prompt(repo_root)}],
        "temperature": 0.0,
        "max_tokens": PROBE_TOKEN_CEILING,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "recclaw_campaign_proposals",
                "strict": True,
                "schema": schema,
            },
        },
    }


def exact_probe_request_payload_digest(repo_root: Path) -> str:
    return sha256_digest(exact_probe_request_payload(repo_root))


def expected_prefreeze_manifest(repo_root: Path) -> dict[str, Any]:
    """Construct the only admitted V2 pre-outcome manifest."""

    verify_v1_seal(repo_root)
    v1_manifest = _read_json(repo_root / V1_MANIFEST_REL)
    v1_probe = _read_json(repo_root / V1_PROBE_REL)
    release = _read_json(repo_root / RELEASE_REL)
    policy_bytes = canonical_json_bytes(expected_retry_policy())
    schema_bytes = (repo_root / SCHEMA_REL).read_bytes()
    schema = _read_json(repo_root / SCHEMA_REL)
    if "uniqueItems" not in canonical_json_bytes(schema).decode("utf-8"):
        raise Wave2IntegrationError("frozen V1 schema no longer contains uniqueItems")
    inherited = v1_manifest["shared_proposal_call"]
    future_call = {
        "model_name": MODEL,
        "model_digest": GPT_5_4_MODEL_DIGEST,
        "endpoint_digest": v1_probe["endpoint_digest"],
        "provider_release_digest": release["release_digest"],
        "prompt_digest": inherited["prompt_digest"],
        "tool_policy_digest": inherited["tool_digest"],
        "response_schema_digest": bytes_sha256(schema_bytes),
        "call_schedule_digest": inherited["call_schedule_digest"],
        "token_budget_per_call": 6000,
        "proposal_slots_per_side": 8,
        "proposal_budget_per_side": 8,
        "proposal_denominator_per_side": 8,
        "expected_proposals_per_slot": 1,
        "retry_policy_digest": bytes_sha256(policy_bytes),
        "maximum_additional_physical_attempts_per_slot": 2,
        "maximum_total_physical_attempts_per_slot": 3,
        "retry_payload_identity": "EXACT_SAME_SLOT_AND_REQUEST_PAYLOAD_DIGEST",
        "replacement_proposal": "FORBIDDEN",
        "successful_response_selection": "FORBIDDEN",
        "schema_relaxation": "FORBIDDEN",
        "manual_response_patch": "FORBIDDEN",
    }
    shared_digest = sha256_digest(future_call)
    return {
        "schema": MANIFEST_SCHEMA,
        "attempt_identity": {
            "attempt_id": ATTEMPT_ID,
            "base_commit": V1_HEAD,
            "base_parent": V1_PARENT,
            "base_tree": V1_TREE,
            "pre_outcome": True,
            "distinct_from_v1_attempt": True,
            "v1_attempt_reuse": False,
        },
        "sealed_v1_evidence": {
            "manifest_ref": _repo_ref(V1_MANIFEST_REL),
            "manifest_sha256": V1_MANIFEST_SHA256,
            "blocked_receipt_ref": _repo_ref(V1_BLOCKED_REL),
            "blocked_receipt_sha256": V1_BLOCKED_SHA256,
            "probe_receipt_ref": _repo_ref(V1_PROBE_REL),
            "probe_receipt_sha256": V1_PROBE_SHA256,
            "blocker_classification": "HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED",
            "preservation": "SEALED_BYTE_IDENTICAL_NO_REWRITE",
        },
        "scientific_contract": {
            "identity_threshold_analysis_runtime_inherited_from_v1": True,
            "inherited_v1_manifest_digest": V1_MANIFEST_SHA256,
            "fresh_open_spec_only": True,
            "fixed_66_candidate_allowed": False,
            "producer_rewrite_forbidden": True,
            "minimum_fresh_specs_per_side": 4,
            "minimum_producer_roles_per_side": 2,
            "shared_origin_blind_implementer_qualifier": True,
            "manual_candidate_patch_forbidden": True,
            "qualification_evidence_class": "DEVELOPMENT_ONLY",
            "outcome_held_out_access_before_r1": "FORBIDDEN",
        },
        "future_r1_provider_call_contract": {
            **future_call,
            "shared_call_contract_digest": shared_digest,
            "side_a_call_contract_digest": shared_digest,
            "side_b_call_contract_digest": shared_digest,
        },
        "future_r1_retry_policy": {
            "ref": _repo_ref(POLICY_REL),
            "digest": bytes_sha256(policy_bytes),
        },
        "diagnostic_reprobe": {
            "scope": "ONE_NON_RESEARCH_DETERMINISM_REPROBE_ONLY",
            "logical_call_id": LOGICAL_CALL_ID,
            "proposal_generation_session_id": SESSION_ID,
            "private_root_digest": sha256_digest({"path": PRIVATE_ROOT.as_posix()}),
            "model": MODEL,
            "endpoint_digest": v1_probe["endpoint_digest"],
            "credential_config_digest": v1_probe["credential_config_digest"],
            "credential_identity_digest": v1_probe["credential_identity_digest"],
            "provider_release_digest": release["release_digest"],
            "response_schema_digest": bytes_sha256(schema_bytes),
            "response_schema_contains_unique_items": True,
            "sentinel_digest": sha256_digest(_read_json(repo_root / SENTINEL_REL)),
            "request_payload_digest": exact_probe_request_payload_digest(repo_root),
            "request_mode": "SINGLE_JSON_SCHEMA_NO_TOOLS",
            "maximum_physical_provider_calls": 1,
            "retry_count": 0,
            "research_candidate_generation": "FORBIDDEN",
        },
        "pre_outcome_counters": {
            "provider_calls": 0,
            "research_candidates_generated": 0,
            "open_specs_projected": 0,
            "resolver_calls": 0,
            "candidate_roots_created": 0,
            "candidate_admissions": 0,
            "training_runs": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
        },
        "ready_condition": (
            "REPROBE_PASS_AND_FAIL_CLOSED_VALIDATOR_DRY_RUN_TEST_HASH_SECRET_DIFF_"
            "STRUCTURE_CHECKS_ALL_PASS"
        ),
        "r1_worker_launch_authorized": False,
    }


def expected_reprobe_authorization(repo_root: Path) -> dict[str, Any]:
    manifest_bytes = canonical_json_bytes(expected_prefreeze_manifest(repo_root))
    policy_bytes = canonical_json_bytes(expected_retry_policy())
    return {
        "schema": AUTH_SCHEMA,
        "status": "AUTHORIZED_ONE_DIAGNOSTIC_REPROBE_ONLY",
        "attempt_id": ATTEMPT_ID,
        "manifest_ref": _repo_ref(MANIFEST_REL),
        "manifest_digest": bytes_sha256(manifest_bytes),
        "retry_policy_ref": _repo_ref(POLICY_REL),
        "retry_policy_digest": bytes_sha256(policy_bytes),
        "v1_blocked_receipt_ref": _repo_ref(V1_BLOCKED_REL),
        "v1_blocked_receipt_digest": V1_BLOCKED_SHA256,
        "request_payload_digest": exact_probe_request_payload_digest(repo_root),
        "maximum_physical_provider_calls": 1,
        "retry_count": 0,
        "provider_calls_before_authorization": 0,
        "research_candidates_generated": 0,
        "training_runs": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


def _load_exact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _canonical_file(path)
    normalized = canonical_value(expected)
    if observed != normalized:
        raise Wave2IntegrationError(f"fail-closed V2 resource mismatch: {path.name}")
    return observed


def validate_prefreeze_v2_payload(
    repo_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an in-memory manifest against the one admitted V2 value."""

    observed = canonical_value(payload)
    expected = canonical_value(expected_prefreeze_manifest(repo_root))
    if observed != expected:
        raise Wave2IntegrationError("fail-closed Prefreeze V2 manifest mismatch")
    return observed


def validate_prefreeze_v2(repo_root: Path) -> dict[str, Any]:
    """Validate policy, manifest, authorization, V1 seal, and A/B symmetry."""

    verify_v1_seal(repo_root)
    _load_exact(repo_root / POLICY_REL, expected_retry_policy())
    manifest = _load_exact(
        repo_root / MANIFEST_REL,
        expected_prefreeze_manifest(repo_root),
    )
    validate_prefreeze_v2_payload(repo_root, manifest)
    _load_exact(
        repo_root / AUTH_REL,
        expected_reprobe_authorization(repo_root),
    )
    call = manifest["future_r1_provider_call_contract"]
    shared = call["shared_call_contract_digest"]
    if call["side_a_call_contract_digest"] != shared or call["side_b_call_contract_digest"] != shared:
        raise Wave2IntegrationError("Prefreeze V2 A/B call digests differ")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V2 is not pre-outcome")
    return manifest


def provider_free_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v2(repo_root)
    return {
        "schema": DRY_RUN_SCHEMA,
        "status": "PASS_PROVIDER_FREE_VALIDATION",
        "attempt_id": ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_seal_verified": True,
        "ab_call_contract_symmetry_verified": True,
        "retry_policy_verified": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


# Prefreeze V3 is intentionally implemented in the existing validator module;
# it does not introduce another Broker, transport boundary, or state service.
V2_HEAD = "3e2ac024347f6950f0414b39974cd8e46fd5e654"
V2_PARENT = V1_HEAD
V2_TREE = "2438bfd66f5818994ab7db226a0942cbc5ebc037"
V2_MANIFEST_SHA256 = "c6d40629b3597a52054cbd1b26d76b3c61345d9f8d9dc838387c6f5122f2af6f"
V2_POLICY_SHA256 = "b07e26b469c5975a4def21476dcda4cd245c41decfc558a06bf7f7d84b9941b3"
V2_AUTH_SHA256 = "38cab44e1eafafddebabac348cdf15b5a8c17c32855f026f7621cd3c2ccbb3d3"
V2_PROBE_SHA256 = "24c325c6c03636d7e4886f51db9d72c8e9a86ebd539b28ff09fb710395917f50"
V2_BLOCKED_SHA256 = "41e5f94afdaedaa81a4ae43b07c9120d6345f2625dd7e6717d502ac0f83b625f"
V2_DRY_RUN_SHA256 = "2c0fd926f051eb3f6a6a24a22932bc6823be35ba9087358492670c3004660231"

V3_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v3-20260801"
V3_MANIFEST_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-attempt.v3"
V3_POLICY_SCHEMA = "recclaw.research-line.r1-provider-retry-policy.v3"
V3_AUTH_SCHEMA = "recclaw.research-line.prefreeze-v3-reprobe-authorization.v1"
V3_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v3-provider-attempt-receipt.v1"
)
V3_DRY_RUN_SCHEMA = "recclaw.research-line.prefreeze-v3-dry-run-receipt.v1"
V3_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v3-blocked-receipt.v1"
V3_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-v3-ready-receipt.v1"

V3_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v3-diagnostic-slot"
V3_SESSION_ID = "fresh-open-spec-prefreeze-v3-diagnostic-slot-session"
V3_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_reprobe_v3_private")

V3_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V3.json"
V3_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V3.json"
V3_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V3_REPROBE_AUTHORIZATION.json"
V3_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V3.json"
)
V3_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V3_BLOCKED_RECEIPT.json"
V3_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_V3_READY_RECEIPT.json"
V3_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V3_DRY_RUN_RECEIPT.json"
V3_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V3_VERIFICATION_RECEIPT.json"

SQLITE_CALL_COLUMNS = (
    "logical_call_id",
    "request_digest",
    "response_digest",
    "response_json",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_ms",
    "returned_model",
    "status",
    "error_type",
    "error_detail_json",
    "proposal_generation_session_id",
    "receipt_digest",
    "receipt_json",
    "closure_receipt_json",
    "outcome_json",
    "broker_release_digest",
)


def v3_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V3 physical attempt ordinal must be 1..3")
    return V3_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v2_seal(repo_root: Path) -> dict[str, str]:
    """Verify every committed V2 manifest/policy/receipt artifact."""

    expected = {
        MANIFEST_REL: V2_MANIFEST_SHA256,
        POLICY_REL: V2_POLICY_SHA256,
        AUTH_REL: V2_AUTH_SHA256,
        PROBE_REL: V2_PROBE_SHA256,
        BLOCKED_REL: V2_BLOCKED_SHA256,
        DRY_RUN_REL: V2_DRY_RUN_SHA256,
    }
    for relative, digest in expected.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V2 bytes changed: {relative.as_posix()}"
            )
    return {relative.as_posix(): digest for relative, digest in expected.items()}


def expected_v3_retry_policy() -> dict[str, Any]:
    """Bind V2 retry semantics to the one V3 diagnostic slot."""

    inherited = expected_retry_policy()
    return {
        "schema": V3_POLICY_SCHEMA,
        "inherited_v2_policy_ref": _repo_ref(POLICY_REL),
        "inherited_v2_policy_digest": V2_POLICY_SHA256,
        "future_r1_policy_unchanged": True,
        "proposal_slots_per_side": inherited["proposal_slots_per_side"],
        "proposal_denominator_per_side": inherited[
            "proposal_denominator_per_side"
        ],
        "diagnostic_slot": {
            "slot_id": "PREFREEZE_V3_ENDPOINT_SCHEMA_DIAGNOSTIC",
            "same_logical_call_id_required": True,
            "same_request_payload_digest_required": True,
            "initial_physical_attempts": 1,
            "maximum_additional_physical_attempts": 2,
            "maximum_total_physical_attempts": 3,
            "deterministic_backoff_ms_after_failure": [1000, 3000],
            "retryable_failure_classes": inherited[
                "retryable_failure_classes"
            ],
            "terminal_no_retry_failure_classes": inherited[
                "terminal_no_retry_failure_classes"
            ],
            "first_valid_strict_response_action": "ACCEPT_AND_STOP",
            "successful_response_selection": "FORBIDDEN",
            "extra_call_after_valid_response": "FORBIDDEN",
            "local_receipt_failure_action": "STOP_WITHOUT_PROVIDER_RETRY",
        },
        "missingness": inherited["exhausted_slot_missingness"],
        "response_contract_rules": inherited["response_contract_rules"],
    }


def expected_prefreeze_v3_manifest(repo_root: Path) -> dict[str, Any]:
    """Construct the only admitted V3 pre-outcome identity."""

    verify_v1_seal(repo_root)
    verify_v2_seal(repo_root)
    v1_probe = _read_json(repo_root / V1_PROBE_REL)
    v2_manifest = _read_json(repo_root / MANIFEST_REL)
    release = _read_json(repo_root / RELEASE_REL)
    schema_bytes = (repo_root / SCHEMA_REL).read_bytes()
    schema = _read_json(repo_root / SCHEMA_REL)
    if "uniqueItems" not in canonical_json_bytes(schema).decode("utf-8"):
        raise Wave2IntegrationError("frozen V1 schema no longer contains uniqueItems")
    policy_bytes = canonical_json_bytes(expected_v3_retry_policy())
    physical_roots = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V3_ATTEMPT_ID,
                    "diagnostic_slot": "PREFREEZE_V3_ENDPOINT_SCHEMA_DIAGNOSTIC",
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v3_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v3_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    future = v2_manifest["future_r1_provider_call_contract"]
    return {
        "schema": V3_MANIFEST_SCHEMA,
        "attempt_identity": {
            "attempt_id": V3_ATTEMPT_ID,
            "base_commit": V2_HEAD,
            "base_parent": V2_PARENT,
            "base_tree": V2_TREE,
            "pre_outcome": True,
            "distinct_from_v1_v2_attempts": True,
            "old_attempt_call_db_identity_reuse": False,
        },
        "sealed_negative_evidence": {
            "v1_manifest_digest": V1_MANIFEST_SHA256,
            "v1_probe_receipt_digest": V1_PROBE_SHA256,
            "v1_blocked_receipt_digest": V1_BLOCKED_SHA256,
            "v1_failure_class": "HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED",
            "v2_manifest_digest": V2_MANIFEST_SHA256,
            "v2_policy_digest": V2_POLICY_SHA256,
            "v2_authorization_digest": V2_AUTH_SHA256,
            "v2_probe_receipt_digest": V2_PROBE_SHA256,
            "v2_blocked_receipt_digest": V2_BLOCKED_SHA256,
            "v2_dry_run_digest": V2_DRY_RUN_SHA256,
            "v2_failure_class": (
                "HTTP_503_TRANSIENT_PROVIDER_SERVICE_UNAVAILABLE"
            ),
            "preservation": "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL",
        },
        "exact_provider_contract": {
            "model": MODEL,
            "model_digest": GPT_5_4_MODEL_DIGEST,
            "endpoint_digest": v1_probe["endpoint_digest"],
            "credential_config_digest": v1_probe[
                "credential_config_digest"
            ],
            "credential_identity_digest": v1_probe[
                "credential_identity_digest"
            ],
            "provider_release_digest": release["release_digest"],
            "response_schema_digest": bytes_sha256(schema_bytes),
            "response_schema_contains_unique_items": True,
            "sentinel_digest": sha256_digest(
                _read_json(repo_root / SENTINEL_REL)
            ),
            "request_payload_digest": exact_probe_request_payload_digest(
                repo_root
            ),
            "request_mode": "SINGLE_JSON_SCHEMA_NO_TOOLS",
            "logical_call_id": V3_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V3_SESSION_ID,
            "diagnostic_slot_id": (
                "PREFREEZE_V3_ENDPOINT_SCHEMA_DIAGNOSTIC"
            ),
        },
        "bounded_retry": {
            "policy_ref": _repo_ref(V3_POLICY_REL),
            "policy_digest": bytes_sha256(policy_bytes),
            "maximum_physical_attempts": 3,
            "maximum_retry_count": 2,
            "deterministic_backoff_ms": [1000, 3000],
            "physical_attempt_identities": physical_roots,
            "sqlite_calls_schema_digest": sha256_digest(
                {"table": "calls", "columns": list(SQLITE_CALL_COLUMNS)}
            ),
        },
        "future_r1_scientific_contract": {
            "shared_call_contract_digest": future[
                "shared_call_contract_digest"
            ],
            "side_a_call_contract_digest": future[
                "side_a_call_contract_digest"
            ],
            "side_b_call_contract_digest": future[
                "side_b_call_contract_digest"
            ],
            "proposal_slots_per_side": 8,
            "proposal_denominator_per_side": 8,
            "retry_is_proposal": False,
            "provider_failure_analysis": (
                "MISSING_PROVIDER_OR_ENGINEERING_FAILURE"
            ),
            "provider_failure_is_mechanism_negative_evidence": False,
        },
        "prohibited_actions": {
            "model_or_endpoint_change": "FORBIDDEN",
            "schema_relaxation": "FORBIDDEN",
            "manual_response_patch": "FORBIDDEN",
            "candidate_or_qualification_or_admission": "FORBIDDEN",
            "training_or_outcome_or_held_out": "FORBIDDEN",
            "additional_attempt_after_terminal_state": "FORBIDDEN",
        },
        "pre_outcome_counters": {
            "provider_calls": 0,
            "research_candidates_generated": 0,
            "open_specs_projected": 0,
            "resolver_calls": 0,
            "candidate_roots_created": 0,
            "candidate_admissions": 0,
            "training_runs": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
        },
        "ready_condition": (
            "FIRST_VALID_STRICT_RESPONSE_AND_VALIDATOR_DRY_RUN_TEST_CANONICAL_"
            "HASH_SECRET_DIFF_STRUCTURE_ALL_PASS"
        ),
        "r1_worker_launch_authorized": False,
    }


def expected_v3_authorization(repo_root: Path) -> dict[str, Any]:
    manifest_bytes = canonical_json_bytes(
        expected_prefreeze_v3_manifest(repo_root)
    )
    policy_bytes = canonical_json_bytes(expected_v3_retry_policy())
    return {
        "schema": V3_AUTH_SCHEMA,
        "status": "AUTHORIZED_ONE_BOUNDED_DIAGNOSTIC_SLOT",
        "attempt_id": V3_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V3_MANIFEST_REL),
        "manifest_digest": bytes_sha256(manifest_bytes),
        "retry_policy_ref": _repo_ref(V3_POLICY_REL),
        "retry_policy_digest": bytes_sha256(policy_bytes),
        "request_payload_digest": exact_probe_request_payload_digest(
            repo_root
        ),
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v3_payload(
    repo_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    observed = canonical_value(payload)
    expected = canonical_value(expected_prefreeze_v3_manifest(repo_root))
    if observed != expected:
        raise Wave2IntegrationError("fail-closed Prefreeze V3 manifest mismatch")
    return observed


def validate_prefreeze_v3(repo_root: Path) -> dict[str, Any]:
    """Validate V3 identity, sealed predecessors, retry, and A/B science."""

    verify_v1_seal(repo_root)
    verify_v2_seal(repo_root)
    _load_exact(repo_root / V3_POLICY_REL, expected_v3_retry_policy())
    manifest = _load_exact(
        repo_root / V3_MANIFEST_REL,
        expected_prefreeze_v3_manifest(repo_root),
    )
    validate_prefreeze_v3_payload(repo_root, manifest)
    _load_exact(
        repo_root / V3_AUTH_REL,
        expected_v3_authorization(repo_root),
    )
    future = manifest["future_r1_scientific_contract"]
    shared = future["shared_call_contract_digest"]
    if (
        future["side_a_call_contract_digest"] != shared
        or future["side_b_call_contract_digest"] != shared
    ):
        raise Wave2IntegrationError("Prefreeze V3 A/B call digests differ")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V3 is not pre-outcome")
    return manifest


def provider_free_v3_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v3(repo_root)
    return {
        "schema": V3_DRY_RUN_SCHEMA,
        "status": "PASS_PROVIDER_FREE_VALIDATION",
        "attempt_id": V3_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_seals_verified": True,
        "ab_call_contract_symmetry_verified": True,
        "bounded_retry_policy_verified": True,
        "sqlite_schema_identity_frozen": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }


V3_HEAD = "68014e714d6636268fc36b259bf13498f158e4cd"
V3_PARENT = V2_HEAD
V3_TREE = "1d58175f7331f46abd9af20d07bd9690bea20167"
V3_MANIFEST_SHA256 = "91b1a6829f6b3996602a053d4bbb5f884d3677a8c8264f5f982d24ea30f9ab5b"
V3_POLICY_SHA256 = "eca1fbc7a684ca5293f5686b6881ef7fba54252c25a1a1fd2bc6b8efacc63d90"
V3_AUTH_SHA256 = "3722674ae27be2e95deb9ca052a2f4227c32010057d3ac3e86d27508d3532bc0"
V3_ATTEMPT_SHA256 = "2c431d913a4b87f46f2d4a9275947c41b84cf12ceb9126bf8ed95e76a752dde6"
V3_BLOCKED_SHA256 = "bcd9fe54df6189f14e7d84b81817b279350d566861081e9718c284b7823ab9a9"
V3_DRY_RUN_SHA256 = "1f1db1450720174a00ff9801035609b183e06a37f0d5375e6bdae2596e08e134"
V1_SCHEMA_SHA256 = "d328eefa658a78471ac81fcfff441fccad07ccb89cabb620c7f7cf7a6c617456"

V4_ATTEMPT_ID = "recclaw-r1-r2-prefreeze-v4-20260801"
V4_MANIFEST_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-attempt.v4"
V4_POLICY_SCHEMA = "recclaw.research-line.r1-provider-retry-policy.v4"
V4_AUTH_SCHEMA = "recclaw.research-line.prefreeze-v4-authorization.v1"
V4_ATTEMPT_RECEIPT_SCHEMA = (
    "recclaw.research-line.prefreeze-v4-provider-attempt-receipt.v1"
)
V4_DRY_RUN_SCHEMA = "recclaw.research-line.prefreeze-v4-dry-run-receipt.v1"
V4_BLOCKED_SCHEMA = "recclaw.research-line.prefreeze-v4-blocked-receipt.v1"
V4_READY_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v4"
V4_VERIFICATION_SCHEMA = (
    "recclaw.research-line.prefreeze-v4-verification-receipt.v1"
)

V4_LOGICAL_CALL_ID = "fresh-open-spec-prefreeze-v4-schema-qualification-slot"
V4_SESSION_ID = (
    "fresh-open-spec-prefreeze-v4-schema-qualification-slot-session"
)
V4_PRIVATE_ROOT = Path("/root/projects/RecClaw_r1_prefreeze_probe_v4_private")

V4_POLICY_REL = DOC_ROOT_REL / "R1_PROVIDER_RETRY_POLICY_V4.json"
V4_MANIFEST_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_MANIFEST_V4.json"
V4_AUTH_REL = DOC_ROOT_REL / "PREFREEZE_V4_AUTHORIZATION.json"
V4_ATTEMPT_RECEIPT_REL = (
    DOC_ROOT_REL / "FRESH_OPEN_SPEC_ENDPOINT_ATTEMPT_RECEIPT_V4.json"
)
V4_BLOCKED_REL = DOC_ROOT_REL / "PREFREEZE_V4_BLOCKED_RECEIPT.json"
V4_READY_REL = DOC_ROOT_REL / "R1_PREFREEZE_READY_RECEIPT.json"
V4_DRY_RUN_REL = DOC_ROOT_REL / "R1_R2_PREFREEZE_V4_DRY_RUN_RECEIPT.json"
V4_VERIFICATION_REL = DOC_ROOT_REL / "PREFREEZE_V4_VERIFICATION_RECEIPT.json"
V4_RELEASE_REL = DOC_ROOT_REL / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V4.json"

V4_PROVIDER_SCHEMA_REL = (
    RESOURCE_ROOT_REL / "fresh_open_spec_proposal_response_v4_provider.schema.json"
)
V4_VALID_FIXTURE_REL = (
    RESOURCE_ROOT_REL / "fresh_open_spec_v4_valid_fixture.json"
)
V4_NEGATIVE_FIXTURE_REL = (
    RESOURCE_ROOT_REL / "fresh_open_spec_v4_duplicate_arrays_negative_fixture.json"
)
V4_VALIDATOR_REL = Path(
    "src/recclaw_core/experiments/helix_abc_v1/v4_response_contract.py"
)


def v4_physical_root(ordinal: int) -> Path:
    if ordinal not in {1, 2, 3}:
        raise Wave2IntegrationError("V4 physical attempt ordinal must be 1..3")
    return V4_PRIVATE_ROOT / f"physical_attempt_{ordinal:02d}"


def verify_v3_seal(repo_root: Path) -> dict[str, str]:
    expected = {
        V3_MANIFEST_REL: V3_MANIFEST_SHA256,
        V3_POLICY_REL: V3_POLICY_SHA256,
        V3_AUTH_REL: V3_AUTH_SHA256,
        V3_ATTEMPT_RECEIPT_REL: V3_ATTEMPT_SHA256,
        V3_BLOCKED_REL: V3_BLOCKED_SHA256,
        V3_DRY_RUN_REL: V3_DRY_RUN_SHA256,
    }
    for relative, digest in expected.items():
        path = repo_root / relative
        if not path.is_file() or bytes_sha256(path.read_bytes()) != digest:
            raise Wave2IntegrationError(
                f"sealed V3 bytes changed: {relative.as_posix()}"
            )
    schema_path = repo_root / SCHEMA_REL
    if bytes_sha256(schema_path.read_bytes()) != V1_SCHEMA_SHA256:
        raise Wave2IntegrationError("sealed V1 response schema bytes changed")
    return {relative.as_posix(): digest for relative, digest in expected.items()}


def expected_v4_provider_schema(repo_root: Path) -> dict[str, Any]:
    v1_schema = _read_json(repo_root / SCHEMA_REL)
    return derive_v4_provider_schema(v1_schema)


def expected_v4_valid_fixture(repo_root: Path) -> dict[str, Any]:
    return canonical_value(_read_json(repo_root / SENTINEL_REL))


def expected_v4_negative_fixture(repo_root: Path) -> dict[str, Any]:
    fixture = expected_v4_valid_fixture(repo_root)
    proposal = fixture["proposals"][0]
    for path in V4_LOCAL_ARRAY_PATHS:
        current: Any = proposal
        for part in path[:-1]:
            current = current[part]
        values = current[path[-1]]
        duplicate = (
            values[0]
            if values
            else "qualification-only local uniqueness negative fixture"
        )
        current[path[-1]] = [duplicate, deepcopy(duplicate)]
    return fixture


def expected_v4_release(repo_root: Path) -> dict[str, Any]:
    release = _read_json(repo_root / RELEASE_REL)
    preimage = dict(release)
    preimage.pop("release_digest", None)
    preimage["response_schema_digest"] = bytes_sha256(
        canonical_json_bytes(expected_v4_provider_schema(repo_root))
    )
    return {**preimage, "release_digest": sha256_digest(preimage)}


def exact_v4_probe_request_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": schema_probe_prompt(repo_root)}
        ],
        "temperature": 0.0,
        "max_tokens": PROBE_TOKEN_CEILING,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "recclaw_campaign_proposals",
                "strict": True,
                "schema": expected_v4_provider_schema(repo_root),
            },
        },
    }


def exact_v4_probe_request_payload_digest(repo_root: Path) -> str:
    return sha256_digest(exact_v4_probe_request_payload(repo_root))


def expected_v4_retry_policy() -> dict[str, Any]:
    inherited = expected_v3_retry_policy()
    diagnostic = inherited["diagnostic_slot"]
    return {
        "schema": V4_POLICY_SCHEMA,
        "inherited_v3_policy_ref": _repo_ref(V3_POLICY_REL),
        "inherited_v3_policy_digest": V3_POLICY_SHA256,
        "proposal_slots_per_side": 8,
        "proposal_denominator_per_side": 8,
        "diagnostic_slot": {
            **diagnostic,
            "slot_id": "PREFREEZE_V4_SCHEMA_QUALIFICATION",
        },
        "local_uniqueness_failure": {
            "failure_class": (
                "LOCAL_ARRAY_UNIQUENESS_CONTRACT_FAILURE"
            ),
            "retry_eligible": False,
            "manual_patch": "FORBIDDEN",
            "successful_response_selection": "FORBIDDEN",
            "candidate_admission": "FORBIDDEN",
            "mechanism_negative_evidence": False,
        },
        "missingness": inherited["missingness"],
    }


def expected_prefreeze_v4_manifest(repo_root: Path) -> dict[str, Any]:
    verify_v1_seal(repo_root)
    verify_v2_seal(repo_root)
    verify_v3_seal(repo_root)
    v1_schema = _read_json(repo_root / SCHEMA_REL)
    v4_schema = expected_v4_provider_schema(repo_root)
    validate_v4_schema_derivation(
        v1_schema=v1_schema,
        v4_provider_schema=v4_schema,
    )
    v4_schema_bytes = canonical_json_bytes(v4_schema)
    policy_bytes = canonical_json_bytes(expected_v4_retry_policy())
    release = expected_v4_release(repo_root)
    v3_manifest = _read_json(repo_root / V3_MANIFEST_REL)
    physical_roots = [
        {
            "ordinal": ordinal,
            "physical_attempt_identity_digest": sha256_digest(
                {
                    "attempt_id": V4_ATTEMPT_ID,
                    "diagnostic_slot": "PREFREEZE_V4_SCHEMA_QUALIFICATION",
                    "ordinal": ordinal,
                    "private_root_digest": sha256_digest(
                        {"path": v4_physical_root(ordinal).as_posix()}
                    ),
                }
            ),
            "private_root_digest": sha256_digest(
                {"path": v4_physical_root(ordinal).as_posix()}
            ),
        }
        for ordinal in (1, 2, 3)
    ]
    inherited_call_digest = v3_manifest[
        "future_r1_scientific_contract"
    ]["shared_call_contract_digest"]
    v4_call_preimage = {
        "inherited_call_contract_digest": inherited_call_digest,
        "provider_response_schema_digest": bytes_sha256(v4_schema_bytes),
        "local_uniqueness_contract_digest": V4_UNIQUENESS_CONTRACT_DIGEST,
        "retry_policy_digest": bytes_sha256(policy_bytes),
        "proposal_slots_per_side": 8,
        "proposal_denominator_per_side": 8,
    }
    shared_call_digest = sha256_digest(v4_call_preimage)
    return {
        "schema": V4_MANIFEST_SCHEMA,
        "attempt_identity": {
            "attempt_id": V4_ATTEMPT_ID,
            "base_commit": V3_HEAD,
            "base_parent": V3_PARENT,
            "base_tree": V3_TREE,
            "pre_outcome": True,
            "distinct_from_v1_v2_v3_attempts": True,
            "old_attempt_call_db_identity_reuse": False,
        },
        "sealed_predecessor_evidence": {
            "v1_manifest_digest": V1_MANIFEST_SHA256,
            "v1_probe_digest": V1_PROBE_SHA256,
            "v1_blocked_digest": V1_BLOCKED_SHA256,
            "v2_manifest_digest": V2_MANIFEST_SHA256,
            "v2_probe_digest": V2_PROBE_SHA256,
            "v2_blocked_digest": V2_BLOCKED_SHA256,
            "v3_manifest_digest": V3_MANIFEST_SHA256,
            "v3_attempt_digest": V3_ATTEMPT_SHA256,
            "v3_blocked_digest": V3_BLOCKED_SHA256,
            "preservation": "SEALED_WORKING_AND_COMMITTED_BYTES_IDENTICAL",
        },
        "response_contract_equivalence": {
            "v1_schema_ref": _repo_ref(SCHEMA_REL),
            "v1_schema_digest": V1_SCHEMA_SHA256,
            "v4_provider_schema_ref": _repo_ref(V4_PROVIDER_SCHEMA_REL),
            "v4_provider_schema_digest": bytes_sha256(v4_schema_bytes),
            "only_provider_schema_change": (
                "DELETE_EXACT_SIX_UNIQUE_ITEMS_TRUE"
            ),
            "removed_schema_paths": [
                "/" + "/".join(path) for path in V1_UNIQUE_SCHEMA_PATHS
            ],
            "local_response_paths": [
                "/proposals/*/" + "/".join(path)
                for path in V4_LOCAL_ARRAY_PATHS
            ],
            "local_validator_ref": (
                _repo_ref(V4_VALIDATOR_REL)
                + ":validate_v4_response_contract"
            ),
            "local_validator_code_digest": bytes_sha256(
                (repo_root / V4_VALIDATOR_REL).read_bytes()
            ),
            "local_uniqueness_contract": V4_UNIQUENESS_CONTRACT,
            "local_uniqueness_contract_digest": (
                V4_UNIQUENESS_CONTRACT_DIGEST
            ),
            "execution_order": [
                "COMPLETE_STRICT_V4_PROVIDER_SCHEMA",
                "LOCAL_SIX_PATH_CANONICAL_DEEP_UNIQUENESS",
                "OPEN_SPEC_PROJECTION_OR_ANY_DOWNSTREAM_USE",
            ],
            "local_failure_class": (
                "LOCAL_ARRAY_UNIQUENESS_CONTRACT_FAILURE"
            ),
            "local_failure_retry_eligible": False,
            "semantic_equivalence": "EXACT_V1_ACCEPTANCE_SET",
            "valid_fixture_ref": _repo_ref(V4_VALID_FIXTURE_REL),
            "valid_fixture_digest": bytes_sha256(
                canonical_json_bytes(expected_v4_valid_fixture(repo_root))
            ),
            "negative_fixture_ref": _repo_ref(V4_NEGATIVE_FIXTURE_REL),
            "negative_fixture_digest": bytes_sha256(
                canonical_json_bytes(expected_v4_negative_fixture(repo_root))
            ),
        },
        "exact_provider_contract": {
            "model": MODEL,
            "model_digest": GPT_5_4_MODEL_DIGEST,
            "endpoint_digest": release["endpoint_digest"],
            "credential_config_digest": _read_json(
                repo_root / V1_PROBE_REL
            )["credential_config_digest"],
            "credential_identity_digest": _read_json(
                repo_root / V1_PROBE_REL
            )["credential_identity_digest"],
            "provider_release_ref": _repo_ref(V4_RELEASE_REL),
            "provider_release_digest": release["release_digest"],
            "response_schema_digest": bytes_sha256(v4_schema_bytes),
            "sentinel_digest": sha256_digest(
                expected_v4_valid_fixture(repo_root)
            ),
            "request_payload_digest": exact_v4_probe_request_payload_digest(
                repo_root
            ),
            "prompt_digest": bytes_sha256(
                (repo_root / PROMPT_REL).read_bytes()
            ),
            "tool_policy_digest": bytes_sha256(
                (repo_root / TOOL_REL).read_bytes()
            ),
            "request_mode": "SINGLE_JSON_SCHEMA_NO_TOOLS",
            "temperature": 0.0,
            "token_budget": PROBE_TOKEN_CEILING,
            "logical_call_id": V4_LOGICAL_CALL_ID,
            "proposal_generation_session_id": V4_SESSION_ID,
            "diagnostic_slot_id": "PREFREEZE_V4_SCHEMA_QUALIFICATION",
        },
        "bounded_retry": {
            "policy_ref": _repo_ref(V4_POLICY_REL),
            "policy_digest": bytes_sha256(policy_bytes),
            "maximum_physical_attempts": 3,
            "maximum_retry_count": 2,
            "deterministic_backoff_ms": [1000, 3000],
            "physical_attempt_identities": physical_roots,
            "sqlite_calls_schema_digest": sha256_digest(
                {"table": "calls", "columns": list(SQLITE_CALL_COLUMNS)}
            ),
        },
        "future_r1_scientific_contract": {
            "shared_call_contract_digest": shared_call_digest,
            "side_a_call_contract_digest": shared_call_digest,
            "side_b_call_contract_digest": shared_call_digest,
            "proposal_slots_per_side": 8,
            "proposal_denominator_per_side": 8,
            "retry_is_proposal": False,
            "fixed_66_or_static_candidate_fallback": "FORBIDDEN",
            "provider_failure_analysis": (
                "MISSING_PROVIDER_OR_ENGINEERING_FAILURE"
            ),
            "provider_failure_is_mechanism_negative_evidence": False,
            "shared_origin_blind_implementer_qualifier": True,
            "qualification_evidence_class": "DEVELOPMENT_ONLY",
        },
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


def expected_v4_authorization(repo_root: Path) -> dict[str, Any]:
    manifest_bytes = canonical_json_bytes(
        expected_prefreeze_v4_manifest(repo_root)
    )
    policy_bytes = canonical_json_bytes(expected_v4_retry_policy())
    return {
        "schema": V4_AUTH_SCHEMA,
        "status": "AUTHORIZED_ONE_V4_SCHEMA_QUALIFICATION_SLOT",
        "attempt_id": V4_ATTEMPT_ID,
        "manifest_ref": _repo_ref(V4_MANIFEST_REL),
        "manifest_digest": bytes_sha256(manifest_bytes),
        "retry_policy_ref": _repo_ref(V4_POLICY_REL),
        "retry_policy_digest": bytes_sha256(policy_bytes),
        "provider_schema_ref": _repo_ref(V4_PROVIDER_SCHEMA_REL),
        "provider_schema_digest": bytes_sha256(
            canonical_json_bytes(expected_v4_provider_schema(repo_root))
        ),
        "local_uniqueness_contract_digest": V4_UNIQUENESS_CONTRACT_DIGEST,
        "request_payload_digest": exact_v4_probe_request_payload_digest(
            repo_root
        ),
        "maximum_physical_attempts": 3,
        "maximum_retry_count": 2,
        "deterministic_backoff_ms": [1000, 3000],
        "provider_calls_before_authorization": 0,
        "candidate_training_outcome_held_out_before_authorization": 0,
        "r1_worker_launch_authorized": False,
    }


def validate_prefreeze_v4_payload(
    repo_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    observed = canonical_value(payload)
    expected = canonical_value(expected_prefreeze_v4_manifest(repo_root))
    if observed != expected:
        raise Wave2IntegrationError("fail-closed Prefreeze V4 manifest mismatch")
    return observed


def validate_prefreeze_v4(repo_root: Path) -> dict[str, Any]:
    verify_v1_seal(repo_root)
    verify_v2_seal(repo_root)
    verify_v3_seal(repo_root)
    _load_exact(repo_root / V4_POLICY_REL, expected_v4_retry_policy())
    _load_exact(repo_root / V4_RELEASE_REL, expected_v4_release(repo_root))
    _load_exact(
        repo_root / V4_PROVIDER_SCHEMA_REL,
        expected_v4_provider_schema(repo_root),
    )
    _load_exact(
        repo_root / V4_VALID_FIXTURE_REL,
        expected_v4_valid_fixture(repo_root),
    )
    _load_exact(
        repo_root / V4_NEGATIVE_FIXTURE_REL,
        expected_v4_negative_fixture(repo_root),
    )
    manifest = _load_exact(
        repo_root / V4_MANIFEST_REL,
        expected_prefreeze_v4_manifest(repo_root),
    )
    validate_prefreeze_v4_payload(repo_root, manifest)
    _load_exact(
        repo_root / V4_AUTH_REL,
        expected_v4_authorization(repo_root),
    )
    response_contract = manifest["response_contract_equivalence"]
    if bytes_sha256((repo_root / V4_VALIDATOR_REL).read_bytes()) != (
        response_contract["local_validator_code_digest"]
    ):
        raise Wave2IntegrationError("V4 local validator code bytes changed")
    future = manifest["future_r1_scientific_contract"]
    shared = future["shared_call_contract_digest"]
    if (
        future["side_a_call_contract_digest"] != shared
        or future["side_b_call_contract_digest"] != shared
    ):
        raise Wave2IntegrationError("Prefreeze V4 A/B call digests differ")
    if any(manifest["pre_outcome_counters"].values()):
        raise Wave2IntegrationError("Prefreeze V4 is not pre-outcome")
    valid = _read_json(repo_root / V4_VALID_FIXTURE_REL)
    negative = _read_json(repo_root / V4_NEGATIVE_FIXTURE_REL)
    provider_schema = _read_json(repo_root / V4_PROVIDER_SCHEMA_REL)
    validate_v4_response_contract(valid, provider_schema=provider_schema)
    try:
        validate_v4_response_contract(negative, provider_schema=provider_schema)
    except V4LocalUniquenessError:
        pass
    else:
        raise Wave2IntegrationError("V4 negative uniqueness fixture was accepted")
    return manifest


def provider_free_v4_dry_run(repo_root: Path) -> dict[str, Any]:
    manifest = validate_prefreeze_v4(repo_root)
    return {
        "schema": V4_DRY_RUN_SCHEMA,
        "status": "PASS_PROVIDER_FREE_SEMANTIC_EQUIVALENCE",
        "attempt_id": V4_ATTEMPT_ID,
        "manifest_digest": bytes_sha256(canonical_json_bytes(manifest)),
        "v1_v2_v3_seals_verified": True,
        "exact_six_keyword_derivation_verified": True,
        "local_canonical_deep_uniqueness_verified": True,
        "negative_fixture_rejected_locally": True,
        "valid_fixture_accepted": True,
        "ab_call_contract_symmetry_verified": True,
        "provider_calls": 0,
        "training_runs": 0,
        "candidate_qualifications": 0,
        "candidate_admissions": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
        "r1_worker_launch_authorized": False,
    }
