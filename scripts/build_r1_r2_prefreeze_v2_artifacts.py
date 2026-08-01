#!/usr/bin/env python3
"""Prepare or finalize the additive Prefreeze V2 artifact set."""

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
)
from recclaw_core.experiments.helix_abc_v1.prefreeze_v2 import (  # noqa: E402
    ATTEMPT_ID,
    AUTH_REL,
    BLOCKED_REL,
    BLOCKED_SCHEMA,
    DRY_RUN_REL,
    MANIFEST_REL,
    POLICY_REL,
    PROBE_REL,
    READY_REL,
    V1_BLOCKED_REL,
    V1_BLOCKED_SHA256,
    V1_MANIFEST_REL,
    V1_MANIFEST_SHA256,
    V1_PROBE_REL,
    V1_PROBE_SHA256,
    expected_prefreeze_manifest,
    expected_reprobe_authorization,
    expected_retry_policy,
    provider_free_dry_run,
    validate_prefreeze_v2,
    verify_v1_seal,
)


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise SystemExit(f"existing artifact differs; refusing overwrite: {path.name}")
        return
    path.write_bytes(encoded)


def _prepare() -> int:
    verify_v1_seal(ROOT)
    for forbidden in (PROBE_REL, BLOCKED_REL, READY_REL):
        if (ROOT / forbidden).exists():
            raise SystemExit("V2 outcome artifact already exists; prepare is sealed")
    _write_once(ROOT / POLICY_REL, expected_retry_policy())
    _write_once(ROOT / MANIFEST_REL, expected_prefreeze_manifest(ROOT))
    _write_once(ROOT / AUTH_REL, expected_reprobe_authorization(ROOT))
    manifest = validate_prefreeze_v2(ROOT)
    dry_run = provider_free_dry_run(ROOT)
    print(
        json.dumps(
            {
                "attempt_id": ATTEMPT_ID,
                "authorization_sha256": bytes_sha256((ROOT / AUTH_REL).read_bytes()),
                "manifest_sha256": bytes_sha256((ROOT / MANIFEST_REL).read_bytes()),
                "policy_sha256": bytes_sha256((ROOT / POLICY_REL).read_bytes()),
                "provider_calls": dry_run["provider_calls"],
                "shared_call_contract_digest": manifest[
                    "future_r1_provider_call_contract"
                ]["shared_call_contract_digest"],
                "status": "PREPARED_PRE_OUTCOME",
            },
            sort_keys=True,
        )
    )
    return 0


def _finalize_blocked() -> int:
    manifest = validate_prefreeze_v2(ROOT)
    if (ROOT / READY_REL).exists():
        raise SystemExit("READY exists; refusing blocked finalization")
    probe_path = ROOT / PROBE_REL
    if not probe_path.is_file():
        raise SystemExit("V2 re-probe receipt is missing")
    probe = json.loads(probe_path.read_bytes())
    if canonical_json_bytes(probe) != probe_path.read_bytes():
        raise SystemExit("V2 re-probe receipt is not canonical JSON")
    required = {
        "status": "BLOCKED",
        "physical_provider_calls": 1,
        "retry_count": 0,
        "request_payload_digest": manifest["diagnostic_reprobe"][
            "request_payload_digest"
        ],
        "response_schema_digest": manifest["diagnostic_reprobe"][
            "response_schema_digest"
        ],
        "sensitive_values_persisted": False,
        "sensitive_headers_persisted": False,
        "research_candidates_generated": 0,
        "open_specs_projected": 0,
        "resolver_calls": 0,
        "candidate_roots_created": 0,
        "candidate_admissions": 0,
        "training_runs": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
    }
    for field, expected in required.items():
        if probe.get(field) != expected:
            raise SystemExit(f"V2 re-probe does not prove {field}")
    if probe.get("endpoint_digest") != manifest["diagnostic_reprobe"]["endpoint_digest"]:
        raise SystemExit("V2 re-probe endpoint identity changed")
    if probe.get("model_requested") != "gpt-5.4":
        raise SystemExit("V2 re-probe model identity changed")
    admitted_failures = {
        (
            "HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED",
            400,
            "REPEATED_DETERMINISTIC_INTERFACE_SCHEMA_REJECTION",
        ),
        (
            "HTTP_503_TRANSIENT_PROVIDER_SERVICE_UNAVAILABLE",
            503,
            "INCONCLUSIVE_TRANSIENT_PROVIDER_FAILURE",
        ),
    }
    observed_failure = (
        probe.get("classification"),
        probe.get("http_status"),
        probe.get("determinism_conclusion"),
    )
    if observed_failure not in admitted_failures:
        raise SystemExit("V2 re-probe failure is not an admitted exact blocker")

    blocked = {
        "schema": BLOCKED_SCHEMA,
        "status": "BLOCKED_PREFREEZE_V2",
        "attempt_id": ATTEMPT_ID,
        "manifest_ref": MANIFEST_REL.name,
        "manifest_digest": bytes_sha256((ROOT / MANIFEST_REL).read_bytes()),
        "retry_policy_ref": POLICY_REL.name,
        "retry_policy_digest": bytes_sha256((ROOT / POLICY_REL).read_bytes()),
        "reprobe_receipt_ref": PROBE_REL.name,
        "reprobe_receipt_digest": bytes_sha256(probe_path.read_bytes()),
        "v1_evidence": {
            "manifest_ref": V1_MANIFEST_REL.name,
            "manifest_digest": V1_MANIFEST_SHA256,
            "blocked_receipt_ref": V1_BLOCKED_REL.name,
            "blocked_receipt_digest": V1_BLOCKED_SHA256,
            "probe_receipt_ref": V1_PROBE_REL.name,
            "probe_receipt_digest": V1_PROBE_SHA256,
            "byte_identity_verified": True,
        },
        "classification": probe["classification"],
        "determinism_conclusion": probe["determinism_conclusion"],
        "http_status": probe["http_status"],
        "unsupported_schema_keyword_digest": probe[
            "unsupported_schema_keyword_digest"
        ],
        "endpoint_digest": probe["endpoint_digest"],
        "credential_config_digest": probe["credential_config_digest"],
        "credential_identity_digest": probe["credential_identity_digest"],
        "provider_release_digest": probe["provider_release_digest"],
        "response_schema_digest": probe["response_schema_digest"],
        "request_payload_digest": probe["request_payload_digest"],
        "physical_provider_calls_v2": 1,
        "retry_count_v2": 0,
        "physical_provider_calls_v1_plus_v2": 2,
        "provider_calls_must_stop": True,
        "third_probe_forbidden": True,
        "schema_change_forbidden": True,
        "model_or_endpoint_change_forbidden": True,
        "r1_prefreeze_ready_receipt_emitted": False,
        "r1_worker_launch_authorized": False,
        "side_effects_v2": {
            "provider_calls": 1,
            "research_candidates_generated": 0,
            "candidate_roots_created": 0,
            "candidate_admissions": 0,
            "training_runs": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
        },
    }
    _write_once(ROOT / BLOCKED_REL, blocked)
    print(
        json.dumps(
            {
                "blocked_receipt_sha256": bytes_sha256((ROOT / BLOCKED_REL).read_bytes()),
                "physical_provider_calls_v2": 1,
                "retry_count_v2": 0,
                "status": "BLOCKED_PREFREEZE_V2",
            },
            sort_keys=True,
        )
    )
    return 2


def _dry_run() -> int:
    receipt = provider_free_dry_run(ROOT)
    _write_once(ROOT / DRY_RUN_REL, receipt)
    print(
        json.dumps(
            {
                "dry_run_receipt_sha256": bytes_sha256(
                    (ROOT / DRY_RUN_REL).read_bytes()
                ),
                "provider_calls": 0,
                "training_runs": 0,
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "finalize-blocked", "dry-run"),
    )
    args = parser.parse_args()
    if args.action == "prepare":
        return _prepare()
    if args.action == "finalize-blocked":
        return _finalize_blocked()
    return _dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
