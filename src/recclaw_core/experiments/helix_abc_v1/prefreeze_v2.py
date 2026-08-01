"""Fail-closed Prefreeze V2 identity and provider-free validation.

Prefreeze V2 is an additive, pre-outcome amendment.  It seals the accepted V1
negative evidence and changes only the future R1 provider retry policy.  The
one diagnostic re-probe authorized by V2 is not an R1 proposal slot and never
uses that retry policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import bytes_sha256, canonical_json_bytes, canonical_value, sha256_digest
from .wave2_integration import GPT_5_4_MODEL_DIGEST, Wave2IntegrationError


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
