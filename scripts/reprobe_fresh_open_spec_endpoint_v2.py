#!/usr/bin/env python3
"""Execute the single authorized exact-payload Prefreeze V2 re-probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
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
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerError,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    LabApiCanaryBrokerV1,
    load_lab_api_credentials,
)
from recclaw_core.experiments.helix_abc_v1.prefreeze_v2 import (  # noqa: E402
    AUTH_REL,
    LOGICAL_CALL_ID,
    MODEL,
    PRIVATE_ROOT,
    PROBE_REL,
    PROBE_SCHEMA,
    PROBE_TOKEN_CEILING,
    RELEASE_REL,
    SCHEMA_REL,
    SENTINEL_REL,
    SESSION_ID,
    exact_probe_request_payload_digest,
    exact_probe_request_payload,
    schema_probe_prompt,
    validate_prefreeze_v2,
)


def _credential_identity(config_path: Path, *, base_url: str, api_key: str) -> dict[str, str]:
    config_digest = bytes_sha256(config_path.resolve().read_bytes())
    endpoint_digest = sha256_digest({"base_url": base_url.rstrip("/")})
    identity_digest = sha256_digest(
        {
            "scheme": "RECClaw credential config identity v1",
            "config_bytes_sha256": config_digest,
            "api_key_sha256": hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
            "endpoint_digest": endpoint_digest,
        }
    )
    return {
        "config_bytes_sha256": config_digest,
        "credential_identity_digest": identity_digest,
        "endpoint_digest": endpoint_digest,
    }


def _contains_unique_items(value: Any) -> bool:
    if isinstance(value, str):
        return "uniqueItems" in value
    if isinstance(value, dict):
        return any(_contains_unique_items(key) or _contains_unique_items(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_unique_items(child) for child in value)
    return False


def _zero_side_effects() -> dict[str, int]:
    return {
        "research_candidates_generated": 0,
        "open_specs_projected": 0,
        "resolver_calls": 0,
        "candidate_roots_created": 0,
        "candidate_admissions": 0,
        "training_runs": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
    }


def _failure_receipt_from_existing_call(
    *,
    manifest: dict[str, Any],
    private_root: Path,
) -> dict[str, Any]:
    """Recover a receipt from the one already-persisted physical call."""

    connection = sqlite3.connect(private_root / "broker.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT logical_call_id, proposal_generation_session_id, "
            "request_digest, status, error_type, error_detail_json, "
            "receipt_json, outcome_json FROM calls"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise SystemExit("V2 recovery requires exactly one persisted physical call")
    row = rows[0]
    if (
        row["logical_call_id"] != LOGICAL_CALL_ID
        or row["proposal_generation_session_id"] != SESSION_ID
        or row["status"] != "FAILED"
    ):
        raise SystemExit("V2 recovery call identity/status mismatch")
    release = json.loads((ROOT / RELEASE_REL).read_bytes())
    expected_envelope_digest = sha256_digest(
        {
            "expected_proposal_count": 1,
            "proposal_generation_session_id": SESSION_ID,
            "release_digest": release["release_digest"],
            "request_payload": exact_probe_request_payload(ROOT),
        }
    )
    if row["request_digest"] != expected_envelope_digest:
        raise SystemExit("V2 persisted call request identity mismatch")
    provider_receipt = json.loads(str(row["receipt_json"]))
    outcome = json.loads(str(row["outcome_json"]))
    error_detail = json.loads(str(row["error_detail_json"]))
    http_status = provider_receipt.get("http_status")
    repeated_unique_items = http_status == 400 and _contains_unique_items(error_detail)
    transient_503 = http_status == 503 and row["error_type"] == "HTTP_503"
    classification = (
        "HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED"
        if repeated_unique_items
        else (
            "HTTP_503_TRANSIENT_PROVIDER_SERVICE_UNAVAILABLE"
            if transient_503
            else str(outcome.get("failure_class") or row["error_type"])
        )
    )
    conclusion = (
        "REPEATED_DETERMINISTIC_INTERFACE_SCHEMA_REJECTION"
        if repeated_unique_items
        else (
            "INCONCLUSIVE_TRANSIENT_PROVIDER_FAILURE"
            if transient_503
            else "INCONCLUSIVE_DIFFERENT_FAILURE"
        )
    )
    diagnostic = manifest["diagnostic_reprobe"]
    sentinel = json.loads((ROOT / SENTINEL_REL).read_bytes())
    return {
        "schema": PROBE_SCHEMA,
        "status": "BLOCKED",
        "classification": classification,
        "determinism_conclusion": conclusion,
        "model_requested": MODEL,
        "returned_model": None,
        "authentication_status": "UNVERIFIED",
        "endpoint_digest": diagnostic["endpoint_digest"],
        "credential_config_digest": diagnostic["credential_config_digest"],
        "credential_identity_digest": diagnostic["credential_identity_digest"],
        "provider_release_digest": release["release_digest"],
        "response_schema_digest": diagnostic["response_schema_digest"],
        "sentinel_digest": sha256_digest(sentinel),
        "request_payload_digest": diagnostic["request_payload_digest"],
        "request_envelope_digest": row["request_digest"],
        "broker_outcome_digest": outcome.get("outcome_digest"),
        "provider_error_detail_digest": sha256_digest(error_detail),
        "response_digest": None,
        "http_status": http_status,
        "unsupported_schema_keyword_digest": (
            sha256_digest("uniqueItems") if repeated_unique_items else None
        ),
        "physical_provider_calls": 1,
        "retry_count": 0,
        "sensitive_values_persisted": False,
        "sensitive_headers_persisted": False,
        **_zero_side_effects(),
        "blocked_fields": [
            "endpoint_authentication",
            "exact_fresh_open_spec_schema_support",
            "returned_model",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-api-config", type=Path)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--recover-existing", action="store_true")
    args = parser.parse_args()

    manifest = validate_prefreeze_v2(ROOT)
    private_root = args.private_root.resolve()
    if private_root != PRIVATE_ROOT:
        raise SystemExit("V2 re-probe private-root identity mismatch")
    receipt_path = ROOT / PROBE_REL
    if receipt_path.exists():
        raise SystemExit("V2 re-probe receipt already exists; no second call")
    if args.recover_existing:
        if not private_root.is_dir():
            raise SystemExit("V2 recovery requires the existing private root")
        receipt = _failure_receipt_from_existing_call(
            manifest=manifest,
            private_root=private_root,
        )
        receipt_path.write_bytes(canonical_json_bytes(receipt))
        print(
            json.dumps(
                {
                    "classification": receipt["classification"],
                    "physical_provider_calls": receipt["physical_provider_calls"],
                    "receipt_sha256": bytes_sha256(receipt_path.read_bytes()),
                    "retry_count": receipt["retry_count"],
                    "status": receipt["status"],
                },
                sort_keys=True,
            )
        )
        return 2
    if private_root.exists():
        raise SystemExit("V2 re-probe private root already exists; no second call")
    if args.llm_api_config is None:
        raise SystemExit("--llm-api-config is required for the physical re-probe")
    if not (ROOT / AUTH_REL).is_file():
        raise SystemExit("V2 pre-outcome authorization is missing")

    config_path = args.llm_api_config.resolve()
    base_url, api_key = load_lab_api_credentials(config_path)
    credential = _credential_identity(config_path, base_url=base_url, api_key=api_key)
    del api_key
    diagnostic = manifest["diagnostic_reprobe"]
    for field, observed in (
        ("endpoint_digest", credential["endpoint_digest"]),
        ("credential_config_digest", credential["config_bytes_sha256"]),
        ("credential_identity_digest", credential["credential_identity_digest"]),
    ):
        if diagnostic[field] != observed:
            raise SystemExit(f"V2 re-probe identity mismatch: {field}")
    request_payload_digest = exact_probe_request_payload_digest(ROOT)
    if request_payload_digest != diagnostic["request_payload_digest"]:
        raise SystemExit("V2 re-probe physical request payload changed")

    sentinel = json.loads((ROOT / SENTINEL_REL).read_bytes())
    release = json.loads((ROOT / RELEASE_REL).read_bytes())
    broker: LabApiCanaryBrokerV1 | None = None
    physical_call_count = 0
    try:
        broker = LabApiCanaryBrokerV1(
            private_root,
            schema_path=ROOT / SCHEMA_REL,
            config_path=config_path,
            model=MODEL,
            max_total_tokens_per_call=6000,
            timeout_ms=900_000,
            release_manifest_path=ROOT / RELEASE_REL,
        )
        physical_call_count = 1
        result = broker.call_with_session(
            logical_call_id=LOGICAL_CALL_ID,
            proposal_generation_session_id=SESSION_ID,
            prompt=schema_probe_prompt(ROOT),
            expected_proposal_count=1,
            max_total_tokens=PROBE_TOKEN_CEILING,
        )
        if result.returned_model != MODEL:
            raise CanaryBrokerError("V2 re-probe returned a model other than exact gpt-5.4")
        if canonical_value(result.response) != canonical_value(sentinel):
            raise CanaryBrokerError("V2 re-probe response differed from the frozen sentinel")
        receipt = {
            "schema": PROBE_SCHEMA,
            "status": "PASS",
            "classification": "PASS_EXACT_GPT_5_4_ENDPOINT_AUTH_AND_V1_SCHEMA",
            "determinism_conclusion": "V1_FAILURE_NOT_REPEATED",
            "model_requested": MODEL,
            "returned_model": result.returned_model,
            "authentication_status": "VERIFIED",
            "endpoint_digest": credential["endpoint_digest"],
            "credential_config_digest": credential["config_bytes_sha256"],
            "credential_identity_digest": credential["credential_identity_digest"],
            "provider_release_digest": release["release_digest"],
            "response_schema_digest": bytes_sha256((ROOT / SCHEMA_REL).read_bytes()),
            "sentinel_digest": sha256_digest(sentinel),
            "request_payload_digest": request_payload_digest,
            "request_envelope_digest": result.request_digest,
            "response_digest": result.response_digest,
            "http_status": 200,
            "physical_provider_calls": physical_call_count,
            "retry_count": 0,
            "sensitive_values_persisted": False,
            "sensitive_headers_persisted": False,
            **_zero_side_effects(),
            "blocked_fields": [],
        }
        exit_code = 0
    except Exception as error:
        row = None
        if broker is not None:
            row = broker._connection.execute(  # noqa: SLF001 - private diagnostic DB
                "SELECT receipt_json, error_detail_json FROM calls WHERE logical_call_id = ?",
                (LOGICAL_CALL_ID,),
            ).fetchone()
        provider_receipt = json.loads(str(row["receipt_json"])) if row is not None else {}
        http_status = provider_receipt.get("http_status")
        error_detail = json.loads(str(row["error_detail_json"])) if row is not None else None
        repeated_unique_items = http_status == 400 and _contains_unique_items(error_detail)
        failure_class = (
            "HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED"
            if repeated_unique_items
            else (
                error.outcome.failure_class.value
                if isinstance(error, CanaryBrokerError) and error.outcome is not None
                else type(error).__name__
            )
        )
        receipt = {
            "schema": PROBE_SCHEMA,
            "status": "BLOCKED",
            "classification": failure_class,
            "determinism_conclusion": (
                "REPEATED_DETERMINISTIC_INTERFACE_SCHEMA_REJECTION"
                if repeated_unique_items
                else "INCONCLUSIVE_DIFFERENT_FAILURE"
            ),
            "model_requested": MODEL,
            "returned_model": None,
            "authentication_status": (
                "UNVERIFIED_AFTER_SCHEMA_REJECTION"
                if repeated_unique_items
                else "UNVERIFIED"
            ),
            "endpoint_digest": credential["endpoint_digest"],
            "credential_config_digest": credential["config_bytes_sha256"],
            "credential_identity_digest": credential["credential_identity_digest"],
            "provider_release_digest": release["release_digest"],
            "response_schema_digest": bytes_sha256((ROOT / SCHEMA_REL).read_bytes()),
            "sentinel_digest": sha256_digest(sentinel),
            "request_payload_digest": request_payload_digest,
            "request_envelope_digest": (
                error.outcome.request_envelope_digest
                if isinstance(error, CanaryBrokerError) and error.outcome is not None
                else None
            ),
            "broker_outcome_digest": (
                error.outcome.outcome_digest
                if isinstance(error, CanaryBrokerError) and error.outcome is not None
                else None
            ),
            "provider_error_detail_digest": sha256_digest(error_detail) if error_detail is not None else None,
            "response_digest": None,
            "http_status": http_status,
            "unsupported_schema_keyword_digest": (
                sha256_digest("uniqueItems") if repeated_unique_items else None
            ),
            "physical_provider_calls": physical_call_count,
            "retry_count": 0,
            "sensitive_values_persisted": False,
            "sensitive_headers_persisted": False,
            **_zero_side_effects(),
            "blocked_fields": [
                "endpoint_authentication",
                "exact_fresh_open_spec_schema_support",
                "returned_model",
            ],
        }
        exit_code = 2
    finally:
        if broker is not None:
            broker.close()

    receipt_path.write_bytes(canonical_json_bytes(receipt))
    print(
        json.dumps(
            {
                "classification": receipt["classification"],
                "physical_provider_calls": receipt["physical_provider_calls"],
                "receipt_sha256": bytes_sha256(receipt_path.read_bytes()),
                "retry_count": receipt["retry_count"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
