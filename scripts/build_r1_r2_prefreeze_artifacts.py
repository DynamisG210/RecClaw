#!/usr/bin/env python3
"""Build canonical R1/R2 prefreeze artifacts after the one-time probe."""

from __future__ import annotations

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
from recclaw_core.experiments.helix_abc_v1.wave2_integration import (  # noqa: E402
    BLOCKED_RECEIPT_SCHEMA,
    GPT_5_4_MODEL_DIGEST,
    PREFREEZE_SCHEMA,
    WAVE2_ACCEPTED_COMMIT,
    WAVE2_ACCEPTED_TREE,
    WAVE2_ACCEPTED_TREE_ARCHIVE_SHA256,
    WAVE2_GATE_RECEIPT_SHA256,
    dry_run_r1_r2_launcher,
    load_prefreeze_manifest,
    prefreeze_missing_fields,
    proposal_call_contract_digest,
    r1_prefreeze_ready_receipt,
)


DOC_ROOT = ROOT / "docs/research_line/vnext"
RESOURCE_ROOT = (
    ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
MANIFEST_PATH = DOC_ROOT / "R1_R2_PREFREEZE_MANIFEST.json"
READY_PATH = DOC_ROOT / "R1_PREFREEZE_READY_RECEIPT.json"
DRY_RUN_PATH = DOC_ROOT / "R1_R2_PREFREEZE_DRY_RUN_RECEIPT.json"
BLOCKED_PATH = DOC_ROOT / "PREFREEZE_BLOCKED_RECEIPT.json"
PROBE_PATH = DOC_ROOT / "FRESH_OPEN_SPEC_ENDPOINT_PROBE_RECEIPT_V1.json"
RELEASE_PATH = DOC_ROOT / "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V1.json"
IDENTITY_PATH = DOC_ROOT / "R1_R2_FRESH_IDENTITY_PLAN_V1.json"
POLICY_PATH = DOC_ROOT / "R1_R2_SCIENTIFIC_POLICY_V1.json"
RUNTIME_PATH = DOC_ROOT / "R1_R2_RUNTIME_DEPENDENCY_LOCK_V1.json"
PROMPT_PATH = RESOURCE_ROOT / "fresh_open_spec_proposal_prompt_v1.txt"
SCHEMA_PATH = RESOURCE_ROOT / "fresh_open_spec_proposal_response_v1.schema.json"
TOOL_PATH = RESOURCE_ROOT / "fresh_open_spec_tool_policy_v1.json"
SCHEDULE_PATH = RESOURCE_ROOT / "fresh_open_spec_call_schedule_v1.json"
FAILURE_PATH = RESOURCE_ROOT / "fresh_open_spec_failure_policy_v1.json"


def _repo_ref(path: Path, fragment: str | None = None) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return f"repo:{relative}" + (f"#{fragment}" if fragment else "")


def _raw_digest(path: Path) -> str:
    return bytes_sha256(path.read_bytes())


def _section_digest(value: dict[str, Any], *parts: str) -> str:
    current: Any = value
    for part in parts:
        current = current[part]
    return sha256_digest(current)


def _binding(
    path: Path,
    payload: dict[str, Any],
    *parts: str,
) -> tuple[str, str]:
    fragment = "/".join(parts)
    return _repo_ref(path, fragment), _section_digest(payload, *parts)


def _phase_identity(
    identity: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    db_key = "coordinator_db" if phase == "r1" else "db"
    names = {
        "lineage": "lineage",
        "seed": "seed_plan",
        "outcome_namespace": "outcome_namespace",
        "memory_namespace": "memory_namespace",
        "root": "root",
        "db": db_key,
        "candidate_namespace": "candidate_namespace",
        "package_namespace": "package_namespace",
    }
    result: dict[str, Any] = {}
    for manifest_name, resource_name in names.items():
        ref, digest = _binding(
            IDENTITY_PATH,
            identity,
            phase,
            resource_name,
        )
        result[f"{manifest_name}_ref"] = ref
        result[f"{manifest_name}_digest"] = digest
    if phase == "r1":
        for side in ("side_a", "side_b"):
            ref, digest = _binding(
                IDENTITY_PATH,
                identity,
                phase,
                side,
            )
            result[f"{side}_identity_ref"] = ref
            result[f"{side}_identity_digest"] = digest
    return result


def main() -> int:
    for canonical_path in (PROBE_PATH, RELEASE_PATH):
        canonical_path.write_bytes(
            canonical_json_bytes(json.loads(canonical_path.read_bytes()))
        )
    probe = json.loads(PROBE_PATH.read_bytes())
    if probe.get("physical_provider_calls") != 1:
        raise SystemExit("endpoint probe must prove exactly one physical call")
    probe_passed = probe.get("status") == "PASS"

    release = json.loads(RELEASE_PATH.read_bytes())
    identity = json.loads(IDENTITY_PATH.read_bytes())
    policy = json.loads(POLICY_PATH.read_bytes())
    runtime = json.loads(RUNTIME_PATH.read_bytes())

    manifest: dict[str, Any] = {
        "schema": PREFREEZE_SCHEMA,
        "accepted_wave2": {
            "commit": WAVE2_ACCEPTED_COMMIT,
            "git_tree": WAVE2_ACCEPTED_TREE,
            "tree_archive_sha256": WAVE2_ACCEPTED_TREE_ARCHIVE_SHA256,
            "gate_receipt_sha256": WAVE2_GATE_RECEIPT_SHA256,
        },
        "runtime_identity": {
            "runtime_ref": _repo_ref(RUNTIME_PATH, "runtime_identity"),
            "runtime_digest": _section_digest(runtime, "runtime_identity"),
            "dependency_lock_ref": _repo_ref(RUNTIME_PATH),
            "dependency_lock_digest": _raw_digest(RUNTIME_PATH),
        },
        "provider_identity": {
            "call_entrypoint_ref": (
                "git:56d3156f53d17d4d850368ce7a093fa368957d81:"
                "src/recclaw_core/experiments/helix_abc_v1/"
                "lab_api_broker.py:LabApiCanaryBrokerV1.call_with_session"
            ),
            "call_entrypoint_digest": (
                "bfcf24f562ee92f5b3570907b0a6322fb92e6de5872b6f59005484649967992e"
            ),
            "endpoint_ref": (
                _repo_ref(PROBE_PATH) + "#endpoint_digest"
            ),
            "endpoint_digest": probe["endpoint_digest"],
            "endpoint_support_status": (
                "VERIFIED_EXACT_GPT_5_4_FRESH_OPEN_SPEC_SCHEMA"
                if probe_passed
                else "BLOCKED_EXACT_FRESH_OPEN_SPEC_SCHEMA_UNSUPPORTED"
            ),
            "release_ref": _repo_ref(RELEASE_PATH),
            "release_digest": release["release_digest"],
            "model_name": "gpt-5.4",
            "model_digest": GPT_5_4_MODEL_DIGEST,
            "returned_model": probe["returned_model"],
            "authentication_status": (
                "VERIFIED"
                if probe_passed
                else probe["authentication_status"]
            ),
            "credential_config_digest": probe[
                "credential_config_digest"
            ],
            "credential_identity_digest": probe[
                "credential_identity_digest"
            ],
            "credential_identity_present": True,
            "schema_probe_receipt_ref": _repo_ref(PROBE_PATH),
            "schema_probe_receipt_digest": _raw_digest(PROBE_PATH),
            "schema_probe_status": probe["status"],
            "probe_call_count": probe["physical_provider_calls"],
        },
        "open_spec_contract": {
            "contract_ref": (
                "git:56d3156f53d17d4d850368ce7a093fa368957d81:"
                "src/recclaw_core/experiments/helix_abc_v1/"
                "vnext_contracts.py:OpenResearchSpecV1"
            ),
            "contract_digest": (
                "cf03a8e4cad845e10088f26dbd432a2d41c6785f4b2e4a36b096fdc2c958f56e"
            ),
            "producer_adapter_ref": (
                "git:56d3156f53d17d4d850368ce7a093fa368957d81:"
                "src/recclaw_core/experiments/helix_abc_v1/"
                "open_spec.py:project_open_producer_draft"
            ),
            "producer_adapter_digest": (
                "462c583bdf7505290830231f823d227cc1d5d8c9d0b94fdf0287b0e67e76e1fc"
            ),
            "resolver_ref": (
                "git:56d3156f53d17d4d850368ce7a093fa368957d81:"
                "src/recclaw_core/experiments/helix_abc_v1/"
                "open_spec.py:resolve_capability"
            ),
            "resolver_digest": (
                "462c583bdf7505290830231f823d227cc1d5d8c9d0b94fdf0287b0e67e76e1fc"
            ),
            "producer_roles": [
                "mechanism_composer",
                "lineage_refiner",
                "falsification_designer",
                "frontier_architect",
            ],
            "resolution_results": [
                "SEARCH_READY",
                "INNOVATION_REQUIRED",
                "DEFERRED_PROTOCOL_CHANGE",
                "UNSUPPORTED",
                "INVALID_SPEC",
            ],
            "required_resolution": "INNOVATION_REQUIRED",
            "fixed_catalog_candidate_allowed": False,
            "producer_rewrite_forbidden": True,
        },
        "shared_proposal_call": {
            "granularity": (
                "ONE_PREASSIGNED_PRODUCER_ROLE_ONE_PROPOSAL_PER_CALL"
            ),
            "token_budget": 6000,
            "call_count": 8,
            "expected_proposals_per_call": 1,
            "call_schedule_ref": _repo_ref(SCHEDULE_PATH),
            "call_schedule_digest": _raw_digest(SCHEDULE_PATH),
            "failure_rule_ref": _repo_ref(FAILURE_PATH),
            "failure_rule_digest": _raw_digest(FAILURE_PATH),
            "content_not_json_action": (
                "TERMINAL_CONSUME_PREASSIGNED_SLOT"
            ),
            "replacement_call": "FORBIDDEN",
            "schema_relaxation": "FORBIDDEN",
            "successful_response_selection": "FORBIDDEN",
            "prompt_ref": _repo_ref(PROMPT_PATH),
            "prompt_digest": _raw_digest(PROMPT_PATH),
            "tool_ref": _repo_ref(TOOL_PATH),
            "tool_digest": _raw_digest(TOOL_PATH),
            "response_contract_ref": _repo_ref(SCHEMA_PATH),
            "response_contract_digest": _raw_digest(SCHEMA_PATH),
            "no_retry": True,
            "retry_count": 0,
            "proposal_budget_per_side": 8,
            "shared_call_contract_digest": "",
            "side_a_call_contract_digest": "",
            "side_b_call_contract_digest": "",
        },
        "shared_implementation": {
            "implementer_ref": (
                "git:56d3156f53d17d4d850368ce7a093fa368957d81:"
                "recclaw_core.experiments.helix_abc_v1.innovation_spine:"
                "materialize_candidate_package"
            ),
            "implementer_digest": (
                "0c94ac83b03532a3da16615f17609735bda34f85ff434fd4bef3e6ec078db487"
            ),
            "qualifier_ref": (
                "git:56d3156f53d17d4d850368ce7a093fa368957d81:"
                "recclaw_core.experiments.helix_abc_v1."
                "innovation_recbole_adapter:"
                "MechanicalRecBoleAdapterV1.qualify"
            ),
            "qualifier_digest": (
                "38e9736b3276f55391a3a50da0475dfe825865f2b0e110c8dc19c113913d522e"
            ),
            "origin_blind_projection_ref": (
                "git:56d3156f53d17d4d850368ce7a093fa368957d81:"
                "recclaw_core.experiments.helix_abc_v1.innovation_spine:"
                "origin_blind_projection"
            ),
            "origin_blind_projection_digest": (
                "0c94ac83b03532a3da16615f17609735bda34f85ff434fd4bef3e6ec078db487"
            ),
            "manual_candidate_patch_forbidden": True,
            "qualification_evidence_class": "DEVELOPMENT_ONLY",
        },
        "r1_identity": _phase_identity(identity, phase="r1"),
        "r2_identity": _phase_identity(identity, phase="r2"),
        "evidence_policy": {
            "held_out_absent": True,
            "missingness_policy_ref": _repo_ref(
                POLICY_PATH,
                "missingness_policy",
            ),
            "missingness_policy_digest": _section_digest(
                policy,
                "missingness_policy",
            ),
            "threshold_policy_ref": _repo_ref(
                POLICY_PATH,
                "threshold_policy",
            ),
            "threshold_policy_digest": _section_digest(
                policy,
                "threshold_policy",
            ),
            "analysis_plan_ref": _repo_ref(
                POLICY_PATH,
                "analysis_plan",
            ),
            "analysis_plan_digest": _section_digest(
                policy,
                "analysis_plan",
            ),
            "qualification_gate_ref": _repo_ref(
                POLICY_PATH,
                "qualification_gate",
            ),
            "qualification_gate_digest": _section_digest(
                policy,
                "qualification_gate",
            ),
        },
        "r1_gate": {
            "requirements": {
                "scope": "PER_SIDE_EXCEPT_REAL_MECHANISM_OVERALL",
                "minimum_fresh_specs_per_side": 4,
                "minimum_producer_roles_per_side": 2,
                "minimum_qualified_per_side": 2,
                "minimum_real_mechanism_changes": 1,
                "accepted_change_kinds": [
                    "STRUCTURAL",
                    "INTERACTION",
                    "PROPAGATION",
                ],
                "negative_fixture_required": True,
            },
            "result_slots": {
                "fresh_spec_receipt_refs": [],
                "producer_role_receipt_refs": [],
                "qualified_capability_receipt_refs": [],
                "real_mechanism_change_receipt_refs": [],
            },
        },
    }
    call_digest = proposal_call_contract_digest(manifest)
    manifest["shared_proposal_call"][
        "shared_call_contract_digest"
    ] = call_digest
    manifest["shared_proposal_call"][
        "side_a_call_contract_digest"
    ] = call_digest
    manifest["shared_proposal_call"][
        "side_b_call_contract_digest"
    ] = call_digest

    MANIFEST_PATH.write_bytes(canonical_json_bytes(manifest))
    manifest_digest = bytes_sha256(MANIFEST_PATH.read_bytes())
    if probe_passed:
        load_prefreeze_manifest(
            MANIFEST_PATH,
            expected_digest=manifest_digest,
        )
        dry_run = dry_run_r1_r2_launcher(
            MANIFEST_PATH,
            expected_digest=manifest_digest,
        )
        ready = r1_prefreeze_ready_receipt(
            MANIFEST_PATH,
            expected_digest=manifest_digest,
        )
        DRY_RUN_PATH.write_bytes(canonical_json_bytes(dry_run))
        READY_PATH.write_bytes(canonical_json_bytes(ready))
        print(
            json.dumps(
                {
                    "dry_run_receipt_sha256": _raw_digest(DRY_RUN_PATH),
                    "manifest_sha256": manifest_digest,
                    "ready_receipt_sha256": _raw_digest(READY_PATH),
                    "shared_call_contract_digest": call_digest,
                    "status": "READY",
                },
                sort_keys=True,
            )
        )
        return 0

    if READY_PATH.exists() or DRY_RUN_PATH.exists():
        raise SystemExit(
            "blocked probe cannot coexist with READY or dry-run receipts"
        )
    missing = list(prefreeze_missing_fields(manifest))
    blocked_fields = [
        "provider_identity.authentication_status",
        "provider_identity.endpoint_support_status",
        "provider_identity.returned_model",
        "provider_identity.schema_probe_status",
    ]
    blocked = {
        "schema": BLOCKED_RECEIPT_SCHEMA,
        "status": "BLOCKED_PREFREEZE",
        "accepted_wave2_commit": WAVE2_ACCEPTED_COMMIT,
        "accepted_wave2_tree": WAVE2_ACCEPTED_TREE,
        "manifest_ref": MANIFEST_PATH.name,
        "manifest_digest": manifest_digest,
        "probe_receipt_ref": _repo_ref(PROBE_PATH),
        "probe_receipt_digest": _raw_digest(PROBE_PATH),
        "probe_classification": probe["classification"],
        "probe_http_status": probe.get("http_status"),
        "probe_error_detail_digest": probe.get("error_detail_digest"),
        "probe_request_digest": probe.get("request_digest"),
        "probe_outcome_digest": probe.get("broker_outcome_digest"),
        "unsupported_schema_keyword_digest": probe.get(
            "unsupported_schema_keyword_digest"
        ),
        "endpoint_digest": probe["endpoint_digest"],
        "credential_config_digest": probe["credential_config_digest"],
        "credential_identity_digest": probe["credential_identity_digest"],
        "provider_release_digest": probe["provider_release_digest"],
        "response_schema_digest": probe["response_schema_digest"],
        "blocked_fields": blocked_fields,
        "null_manifest_fields": missing,
        "physical_provider_calls": 1,
        "retry_count": 0,
        "schema_relaxation": "FORBIDDEN",
        "replacement_probe": "FORBIDDEN_WITHIN_THIS_PREFREEZE",
        "r1_prefreeze_ready_receipt_emitted": False,
        "r1_worker_launch_authorized": False,
        "side_effects": {
            "provider_calls": 1,
            "probe_private_roots_created": 1,
            "candidate_roots_created": 0,
            "databases_created": 0,
            "experiment_runs": 0,
            "fresh_r1_training_runs": 0,
            "verification_fixture_one_epoch_runs": 3,
            "candidate_admissions": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
        },
    }
    BLOCKED_PATH.write_bytes(canonical_json_bytes(blocked))
    print(
        json.dumps(
            {
                "blocked_receipt_sha256": _raw_digest(BLOCKED_PATH),
                "manifest_sha256": manifest_digest,
                "probe_receipt_sha256": _raw_digest(PROBE_PATH),
                "shared_call_contract_digest": call_digest,
                "status": "BLOCKED",
            },
            sort_keys=True,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
