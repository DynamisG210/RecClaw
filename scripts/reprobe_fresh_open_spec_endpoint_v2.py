#!/usr/bin/env python3
"""Execute authorized exact-payload Prefreeze endpoint diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
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
    SQLITE_CALL_COLUMNS,
    V3_ATTEMPT_ID,
    V3_ATTEMPT_RECEIPT_REL,
    V3_ATTEMPT_RECEIPT_SCHEMA,
    V3_AUTH_REL,
    V3_LOGICAL_CALL_ID,
    V3_PRIVATE_ROOT,
    V3_SESSION_ID,
    V4_ATTEMPT_ID,
    V4_ATTEMPT_RECEIPT_REL,
    V4_ATTEMPT_RECEIPT_SCHEMA,
    V4_AUTH_REL,
    V4_LOGICAL_CALL_ID,
    V4_PRIVATE_ROOT,
    V4_PROVIDER_SCHEMA_REL,
    V4_RELEASE_REL,
    V4_SESSION_ID,
    exact_v4_probe_request_payload_digest,
    exact_probe_request_payload,
    exact_probe_request_payload_digest,
    schema_probe_prompt,
    validate_prefreeze_v2,
    validate_prefreeze_v3,
    validate_prefreeze_v4,
    v3_physical_root,
    v4_physical_root,
)
from recclaw_core.experiments.helix_abc_v1.v4_response_contract import (  # noqa: E402
    V4_UNIQUENESS_CONTRACT_DIGEST,
    V4LocalUniquenessError,
    validate_v4_response_contract,
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


def _main_v2() -> int:
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


def _validate_sqlite_schema(
    connection: sqlite3.Connection,
    *,
    expected_digest: str,
) -> str:
    """Read PRAGMA metadata before any Provider call and bind exact columns."""

    columns = tuple(
        str(row[1])
        for row in connection.execute("PRAGMA table_info(calls)").fetchall()
    )
    observed_digest = sha256_digest(
        {"table": "calls", "columns": list(columns)}
    )
    if columns != SQLITE_CALL_COLUMNS or observed_digest != expected_digest:
        raise SystemExit("V3 SQLite calls schema identity mismatch")
    return observed_digest


def _cause_type_names(error: BaseException) -> set[str]:
    names: set[str] = set()
    current: BaseException | None = error
    while current is not None and type(current).__name__ not in names:
        names.add(type(current).__name__)
        current = current.__cause__ or current.__context__
    return names


def _contains_overload(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return "overload" in lowered or "over capacity" in lowered
    if isinstance(value, dict):
        return any(
            _contains_overload(key) or _contains_overload(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_overload(child) for child in value)
    return False


def _classify_v3_failure(
    *,
    error: Exception,
    row: sqlite3.Row,
    semantic_after_success: bool,
) -> dict[str, Any]:
    provider_receipt = json.loads(str(row["receipt_json"]))
    error_detail = (
        json.loads(str(row["error_detail_json"]))
        if row["error_detail_json"] is not None
        else None
    )
    outcome = (
        json.loads(str(row["outcome_json"]))
        if row["outcome_json"] is not None
        else None
    )
    http_status = provider_receipt.get("http_status")
    error_type = row["error_type"]
    cause_names = _cause_type_names(error)
    unique_items = http_status == 400 and _contains_unique_items(error_detail)
    if isinstance(error, V4LocalUniquenessError):
        classification = "LOCAL_ARRAY_UNIQUENESS_CONTRACT_FAILURE"
        retry_eligible = False
    elif semantic_after_success:
        classification = "SEMANTIC_RESPONSE_CONTRACT_FAILURE"
        retry_eligible = False
    elif unique_items:
        classification = "HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED"
        retry_eligible = False
    elif http_status == 408:
        classification = "HTTP_408"
        retry_eligible = True
    elif http_status == 429:
        classification = "HTTP_429"
        retry_eligible = True
    elif http_status is not None and http_status >= 500:
        classification = f"HTTP_{http_status}_TRANSIENT_PROVIDER_ERROR"
        retry_eligible = True
    elif http_status in {401, 403}:
        classification = f"HTTP_{http_status}_AUTH"
        retry_eligible = False
    elif http_status is not None and 400 <= http_status < 500:
        classification = (
            "HTTP_400_SCHEMA_CONTRACT"
            if http_status == 400
            else "OTHER_DETERMINISTIC_HTTP_4XX"
        )
        retry_eligible = False
    elif error_type == "TIMEOUT":
        classification = "NETWORK_TIMEOUT"
        retry_eligible = True
    elif error_type == "TRANSPORT_ERROR" and cause_names.intersection(
        {"ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError"}
    ):
        classification = "NETWORK_RESET"
        retry_eligible = True
    elif error_type == "TRANSPORT_ERROR":
        classification = "OTHER_TRANSPORT_FAILURE"
        retry_eligible = False
    elif _contains_overload(error_detail):
        classification = "PROVIDER_OVERLOAD"
        retry_eligible = True
    elif error_type == "SCHEMA_VALIDATION_FAILURE":
        classification = "SCHEMA_VALIDATION_FAILURE"
        retry_eligible = False
    elif error_type == "RESPONSE_CONTRACT_ERROR":
        classification = (
            "CONTENT_NOT_JSON"
            if "JSONDecodeError" in cause_names
            else (
                "HTTP_200_RESPONSE_CONTRACT_VALUE_ERROR"
                if "ValueError" in cause_names
                else "JSON_OR_SEMANTIC_RESPONSE_CONTRACT_FAILURE"
            )
        )
        retry_eligible = False
    else:
        classification = "UNCLASSIFIED_TERMINAL_PROVIDER_FAILURE"
        retry_eligible = False
    return {
        "classification": classification,
        "retry_eligible": retry_eligible,
        "http_status": http_status,
        "provider_receipt_digest": provider_receipt.get("receipt_digest"),
        "broker_outcome_digest": (
            outcome.get("outcome_digest") if outcome is not None else None
        ),
        "provider_error_detail_digest": (
            sha256_digest(error_detail) if error_detail is not None else None
        ),
        "unsupported_schema_keyword_digest": (
            sha256_digest("uniqueItems") if unique_items else None
        ),
    }


def _attempt_chain_entry(
    *,
    entry: dict[str, Any],
    prior_attempt_digest: str | None,
) -> dict[str, Any]:
    preimage = {
        **entry,
        "prior_attempt_digest": prior_attempt_digest,
    }
    return {**preimage, "attempt_digest": sha256_digest(preimage)}


def _v3_top_receipt(
    *,
    manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
    status: str,
    final_classification: str,
    termination_reason: str,
    returned_model: str | None,
    response_digest: str | None,
) -> dict[str, Any]:
    success = status == "PASS"
    return {
        "schema": V3_ATTEMPT_RECEIPT_SCHEMA,
        "status": status,
        "attempt_id": V3_ATTEMPT_ID,
        "diagnostic_slot_id": (
            "PREFREEZE_V3_ENDPOINT_SCHEMA_DIAGNOSTIC"
        ),
        "model_requested": MODEL,
        "returned_model": returned_model,
        "authentication_status": "VERIFIED" if success else "UNVERIFIED",
        "endpoint_digest": manifest["exact_provider_contract"][
            "endpoint_digest"
        ],
        "credential_config_digest": manifest["exact_provider_contract"][
            "credential_config_digest"
        ],
        "credential_identity_digest": manifest[
            "exact_provider_contract"
        ]["credential_identity_digest"],
        "provider_release_digest": manifest["exact_provider_contract"][
            "provider_release_digest"
        ],
        "response_schema_digest": manifest["exact_provider_contract"][
            "response_schema_digest"
        ],
        "sentinel_digest": manifest["exact_provider_contract"][
            "sentinel_digest"
        ],
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "response_digest": response_digest,
        "physical_attempts": attempts,
        "physical_provider_calls": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "final_classification": final_classification,
        "termination_reason": termination_reason,
        "first_valid_response_accepted": success,
        "successful_response_selection": False,
        "extra_call_after_valid_response": False,
        "same_diagnostic_slot": True,
        "same_request_payload_and_envelope_digest": True,
        "sensitive_values_persisted": False,
        "sensitive_headers_persisted": False,
        **_zero_side_effects(),
        "blocked_fields": (
            []
            if success
            else [
                "endpoint_authentication",
                "exact_fresh_open_spec_schema_support",
                "returned_model",
            ]
        ),
        "r1_worker_launch_authorized": False,
    }


def _write_v3_checkpoint(receipt: dict[str, Any]) -> None:
    """Persist before any retry; failure propagates and prevents another call."""

    (ROOT / V3_ATTEMPT_RECEIPT_REL).write_bytes(
        canonical_json_bytes(receipt)
    )


def _main_v3() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-api-config", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = validate_prefreeze_v3(ROOT)
    private_root = args.private_root.resolve()
    if private_root != V3_PRIVATE_ROOT:
        raise SystemExit("V3 private-root identity mismatch")
    if private_root.exists():
        raise SystemExit("V3 private root already exists; no second V3 run")
    if (ROOT / V3_ATTEMPT_RECEIPT_REL).exists():
        raise SystemExit("V3 attempt receipt already exists; no second V3 run")
    if not (ROOT / V3_AUTH_REL).is_file():
        raise SystemExit("V3 pre-outcome authorization is missing")

    config_path = args.llm_api_config.resolve()
    base_url, api_key = load_lab_api_credentials(config_path)
    credential = _credential_identity(
        config_path,
        base_url=base_url,
        api_key=api_key,
    )
    del api_key
    contract = manifest["exact_provider_contract"]
    for field, observed in (
        ("endpoint_digest", credential["endpoint_digest"]),
        ("credential_config_digest", credential["config_bytes_sha256"]),
        (
            "credential_identity_digest",
            credential["credential_identity_digest"],
        ),
    ):
        if contract[field] != observed:
            raise SystemExit(f"V3 Provider identity mismatch: {field}")
    if exact_probe_request_payload_digest(ROOT) != contract[
        "request_payload_digest"
    ]:
        raise SystemExit("V3 physical request payload changed")

    sentinel = json.loads((ROOT / SENTINEL_REL).read_bytes())
    attempts: list[dict[str, Any]] = []
    prior_attempt_digest: str | None = None
    backoffs = (1000, 3000)
    for ordinal in (1, 2, 3):
        attempt_identity = manifest["bounded_retry"][
            "physical_attempt_identities"
        ][ordinal - 1]
        physical_root = v3_physical_root(ordinal)
        if ordinal > 1:
            time.sleep(backoffs[ordinal - 2] / 1000)
        started_ns = time.monotonic_ns()
        broker: LabApiCanaryBrokerV1 | None = None
        result = None
        try:
            broker = LabApiCanaryBrokerV1(
                physical_root,
                schema_path=ROOT / SCHEMA_REL,
                config_path=config_path,
                model=MODEL,
                max_total_tokens_per_call=6000,
                timeout_ms=900_000,
                release_manifest_path=ROOT / RELEASE_REL,
            )
            schema_digest = _validate_sqlite_schema(
                broker._connection,  # noqa: SLF001 - required schema proof
                expected_digest=manifest["bounded_retry"][
                    "sqlite_calls_schema_digest"
                ],
            )
            result = broker.call_with_session(
                logical_call_id=V3_LOGICAL_CALL_ID,
                proposal_generation_session_id=V3_SESSION_ID,
                prompt=schema_probe_prompt(ROOT),
                expected_proposal_count=1,
                max_total_tokens=PROBE_TOKEN_CEILING,
            )
            if result.returned_model != MODEL:
                raise CanaryBrokerError(
                    "V3 returned a model other than exact gpt-5.4"
                )
            if canonical_value(result.response) != canonical_value(sentinel):
                raise CanaryBrokerError(
                    "V3 response differed from the frozen sentinel"
                )
            broker._connection.row_factory = sqlite3.Row  # noqa: SLF001
            success_row = broker._connection.execute(  # noqa: SLF001
                "SELECT receipt_json FROM calls WHERE logical_call_id=?",
                (V3_LOGICAL_CALL_ID,),
            ).fetchone()
            if success_row is None:
                raise SystemExit("V3 successful call receipt row is missing")
            success_provider_receipt = json.loads(
                str(success_row["receipt_json"])
            )
            ended_ns = time.monotonic_ns()
            entry = _attempt_chain_entry(
                entry={
                    "ordinal": ordinal,
                    **attempt_identity,
                    "logical_call_id": V3_LOGICAL_CALL_ID,
                    "request_payload_digest": contract[
                        "request_payload_digest"
                    ],
                    "request_envelope_digest": result.request_digest,
                    "sqlite_calls_schema_digest": schema_digest,
                    "monotonic_start_ns": started_ns,
                    "monotonic_end_ns": ended_ns,
                    "classification": "PASS_EXACT_GPT_5_4_AUTH_SCHEMA",
                    "http_status": 200,
                    "retry_eligible": False,
                    "backoff_ms_after_attempt": 0,
                    "termination_reason": "FIRST_VALID_RESPONSE_ACCEPTED",
                    "returned_model": result.returned_model,
                    "response_digest": result.response_digest,
                    "provider_receipt_digest": success_provider_receipt[
                        "receipt_digest"
                    ],
                    "broker_outcome_digest": None,
                    "provider_error_detail_digest": None,
                },
                prior_attempt_digest=prior_attempt_digest,
            )
            attempts.append(entry)
            receipt = _v3_top_receipt(
                manifest=manifest,
                attempts=attempts,
                status="PASS",
                final_classification=entry["classification"],
                termination_reason=entry["termination_reason"],
                returned_model=result.returned_model,
                response_digest=result.response_digest,
            )
            _write_v3_checkpoint(receipt)
            print(
                json.dumps(
                    {
                        "classification": entry["classification"],
                        "physical_provider_calls": len(attempts),
                        "receipt_sha256": bytes_sha256(
                            (ROOT / V3_ATTEMPT_RECEIPT_REL).read_bytes()
                        ),
                        "retry_count": len(attempts) - 1,
                        "status": "PASS",
                    },
                    sort_keys=True,
                )
            )
            return 0
        except Exception as error:
            if broker is None:
                raise
            broker._connection.row_factory = sqlite3.Row  # noqa: SLF001
            row = broker._connection.execute(  # noqa: SLF001
                "SELECT logical_call_id, request_digest, response_digest, "
                "returned_model, status, error_type, error_detail_json, "
                "receipt_json, outcome_json FROM calls WHERE logical_call_id=?",
                (V3_LOGICAL_CALL_ID,),
            ).fetchone()
            if row is None:
                raise SystemExit(
                    "V3 local receipt failure stopped before retry"
                ) from error
            evidence = _classify_v3_failure(
                error=error,
                row=row,
                semantic_after_success=(result is not None),
            )
            retry = evidence["retry_eligible"] and ordinal < 3
            termination = (
                "RETRY_SCHEDULED"
                if retry
                else (
                    "TRANSIENT_ATTEMPTS_EXHAUSTED"
                    if evidence["retry_eligible"]
                    else "DETERMINISTIC_TERMINAL_FAILURE"
                )
            )
            ended_ns = time.monotonic_ns()
            entry = _attempt_chain_entry(
                entry={
                    "ordinal": ordinal,
                    **attempt_identity,
                    "logical_call_id": V3_LOGICAL_CALL_ID,
                    "request_payload_digest": contract[
                        "request_payload_digest"
                    ],
                    "request_envelope_digest": row["request_digest"],
                    "sqlite_calls_schema_digest": manifest[
                        "bounded_retry"
                    ]["sqlite_calls_schema_digest"],
                    "monotonic_start_ns": started_ns,
                    "monotonic_end_ns": ended_ns,
                    **evidence,
                    "backoff_ms_after_attempt": (
                        backoffs[ordinal - 1] if retry else 0
                    ),
                    "termination_reason": termination,
                    "returned_model": row["returned_model"],
                    "response_digest": row["response_digest"],
                },
                prior_attempt_digest=prior_attempt_digest,
            )
            attempts.append(entry)
            prior_attempt_digest = entry["attempt_digest"]
            status = "IN_PROGRESS" if retry else "BLOCKED"
            receipt = _v3_top_receipt(
                manifest=manifest,
                attempts=attempts,
                status=status,
                final_classification=entry["classification"],
                termination_reason=termination,
                returned_model=None,
                response_digest=None,
            )
            _write_v3_checkpoint(receipt)
            if not retry:
                print(
                    json.dumps(
                        {
                            "classification": entry["classification"],
                            "physical_provider_calls": len(attempts),
                            "receipt_sha256": bytes_sha256(
                                (ROOT / V3_ATTEMPT_RECEIPT_REL).read_bytes()
                            ),
                            "retry_count": len(attempts) - 1,
                            "status": "BLOCKED",
                        },
                        sort_keys=True,
                    )
                )
                return 2
        finally:
            if broker is not None:
                broker.close()
    raise SystemExit("V3 bounded retry loop reached an impossible state")


def _v4_top_receipt(
    *,
    manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
    status: str,
    final_classification: str,
    termination_reason: str,
    returned_model: str | None,
    response_digest: str | None,
) -> dict[str, Any]:
    success = status == "PASS"
    return {
        "schema": V4_ATTEMPT_RECEIPT_SCHEMA,
        "status": status,
        "attempt_id": V4_ATTEMPT_ID,
        "diagnostic_slot_id": "PREFREEZE_V4_SCHEMA_QUALIFICATION",
        "model_requested": MODEL,
        "returned_model": returned_model,
        "authentication_status": "VERIFIED" if success else "UNVERIFIED",
        "provider_schema_support_status": (
            "VERIFIED" if success else "UNVERIFIED"
        ),
        "local_semantic_equivalence_status": (
            "VERIFIED" if success else "UNVERIFIED"
        ),
        "endpoint_digest": manifest["exact_provider_contract"][
            "endpoint_digest"
        ],
        "credential_config_digest": manifest["exact_provider_contract"][
            "credential_config_digest"
        ],
        "credential_identity_digest": manifest[
            "exact_provider_contract"
        ]["credential_identity_digest"],
        "provider_release_digest": manifest["exact_provider_contract"][
            "provider_release_digest"
        ],
        "response_schema_digest": manifest["exact_provider_contract"][
            "response_schema_digest"
        ],
        "local_uniqueness_contract_digest": (
            V4_UNIQUENESS_CONTRACT_DIGEST
        ),
        "sentinel_digest": manifest["exact_provider_contract"][
            "sentinel_digest"
        ],
        "request_payload_digest": manifest["exact_provider_contract"][
            "request_payload_digest"
        ],
        "response_digest": response_digest,
        "physical_attempts": attempts,
        "physical_provider_calls": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "final_classification": final_classification,
        "termination_reason": termination_reason,
        "first_valid_response_accepted": success,
        "successful_response_selection": False,
        "extra_call_after_valid_response": False,
        "same_diagnostic_slot": True,
        "same_request_payload_and_envelope_digest": True,
        "sensitive_values_persisted": False,
        "sensitive_headers_persisted": False,
        **_zero_side_effects(),
        "candidate_qualifications": 0,
        "blocked_fields": (
            []
            if success
            else [
                "endpoint_authentication",
                "exact_gpt_5_4_returned_model",
                "v4_provider_schema_support",
                "local_semantic_equivalence",
            ]
        ),
        "r1_worker_launch_authorized": False,
    }


def _write_v4_checkpoint(receipt: dict[str, Any]) -> None:
    (ROOT / V4_ATTEMPT_RECEIPT_REL).write_bytes(
        canonical_json_bytes(receipt)
    )


def _refine_existing_v4_receipt(
    *,
    manifest: dict[str, Any],
    private_root: Path,
) -> int:
    """Refine the already-persisted terminal class without a Provider call."""

    receipt_path = ROOT / V4_ATTEMPT_RECEIPT_REL
    if not receipt_path.is_file():
        raise SystemExit("V4 refinement requires the existing attempt receipt")
    receipt = json.loads(receipt_path.read_bytes())
    attempts = receipt.get("physical_attempts")
    if (
        receipt.get("status") != "BLOCKED"
        or not isinstance(attempts, list)
        or len(attempts) != 1
    ):
        raise SystemExit("V4 refinement only admits the one-call blocked receipt")
    db_path = v4_physical_root(1) / "broker.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT error_type, error_detail_json, receipt_json FROM calls"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        raise SystemExit("V4 refinement requires exactly one SQLite call row")
    row = rows[0]
    detail = json.loads(str(row["error_detail_json"]))
    provider_receipt = json.loads(str(row["receipt_json"]))
    if (
        row["error_type"] != "RESPONSE_CONTRACT_ERROR"
        or detail.get("exception_type") != "ValueError"
        or provider_receipt.get("http_status") != 200
        or attempts[0].get("classification")
        != "JSON_OR_SEMANTIC_RESPONSE_CONTRACT_FAILURE"
    ):
        raise SystemExit("V4 refinement evidence does not match the admitted case")
    entry = dict(attempts[0])
    entry["classification"] = "HTTP_200_RESPONSE_CONTRACT_VALUE_ERROR"
    entry["provider_error_type"] = "RESPONSE_CONTRACT_ERROR"
    entry["provider_exception_type"] = "ValueError"
    preimage = dict(entry)
    preimage.pop("attempt_digest", None)
    entry["attempt_digest"] = sha256_digest(preimage)
    refined = {
        **receipt,
        "physical_attempts": [entry],
        "final_classification": "HTTP_200_RESPONSE_CONTRACT_VALUE_ERROR",
        "termination_reason": "DETERMINISTIC_TERMINAL_FAILURE",
        "response_contract_subtype_detail": (
            "UNAVAILABLE_BROKER_PERSISTED_EXCEPTION_TYPE_ONLY"
        ),
    }
    receipt_path.write_bytes(canonical_json_bytes(refined))
    print(
        json.dumps(
            {
                "classification": refined["final_classification"],
                "physical_provider_calls": 1,
                "receipt_sha256": bytes_sha256(receipt_path.read_bytes()),
                "retry_count": 0,
                "status": "BLOCKED_REFINED_FROM_EXISTING_SQLITE_ONLY",
            },
            sort_keys=True,
        )
    )
    return 2


def _main_v4() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-api-config", type=Path)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--refine-existing", action="store_true")
    args = parser.parse_args()

    manifest = validate_prefreeze_v4(ROOT)
    private_root = args.private_root.resolve()
    if private_root != V4_PRIVATE_ROOT:
        raise SystemExit("V4 private-root identity mismatch")
    if args.refine_existing:
        if not private_root.is_dir():
            raise SystemExit("V4 refinement requires the existing private root")
        return _refine_existing_v4_receipt(
            manifest=manifest,
            private_root=private_root,
        )
    if private_root.exists():
        raise SystemExit("V4 private root already exists; no second V4 run")
    if (ROOT / V4_ATTEMPT_RECEIPT_REL).exists():
        raise SystemExit("V4 attempt receipt already exists; no second V4 run")
    if not (ROOT / V4_AUTH_REL).is_file():
        raise SystemExit("V4 pre-outcome authorization is missing")
    if args.llm_api_config is None:
        raise SystemExit("--llm-api-config is required for the V4 Provider call")

    config_path = args.llm_api_config.resolve()
    base_url, api_key = load_lab_api_credentials(config_path)
    credential = _credential_identity(
        config_path,
        base_url=base_url,
        api_key=api_key,
    )
    del api_key
    contract = manifest["exact_provider_contract"]
    for field, observed in (
        ("endpoint_digest", credential["endpoint_digest"]),
        ("credential_config_digest", credential["config_bytes_sha256"]),
        (
            "credential_identity_digest",
            credential["credential_identity_digest"],
        ),
    ):
        if contract[field] != observed:
            raise SystemExit(f"V4 Provider identity mismatch: {field}")
    if exact_v4_probe_request_payload_digest(ROOT) != contract[
        "request_payload_digest"
    ]:
        raise SystemExit("V4 physical request payload changed")

    sentinel = json.loads((ROOT / SENTINEL_REL).read_bytes())
    provider_schema = json.loads((ROOT / V4_PROVIDER_SCHEMA_REL).read_bytes())
    attempts: list[dict[str, Any]] = []
    prior_attempt_digest: str | None = None
    backoffs = (1000, 3000)
    for ordinal in (1, 2, 3):
        attempt_identity = manifest["bounded_retry"][
            "physical_attempt_identities"
        ][ordinal - 1]
        physical_root = v4_physical_root(ordinal)
        if ordinal > 1:
            time.sleep(backoffs[ordinal - 2] / 1000)
        started_ns = time.monotonic_ns()
        broker: LabApiCanaryBrokerV1 | None = None
        result = None
        try:
            broker = LabApiCanaryBrokerV1(
                physical_root,
                schema_path=ROOT / V4_PROVIDER_SCHEMA_REL,
                config_path=config_path,
                model=MODEL,
                max_total_tokens_per_call=6000,
                timeout_ms=900_000,
                release_manifest_path=ROOT / V4_RELEASE_REL,
            )
            schema_digest = _validate_sqlite_schema(
                broker._connection,  # noqa: SLF001 - required schema proof
                expected_digest=manifest["bounded_retry"][
                    "sqlite_calls_schema_digest"
                ],
            )
            result = broker.call_with_session(
                logical_call_id=V4_LOGICAL_CALL_ID,
                proposal_generation_session_id=V4_SESSION_ID,
                prompt=schema_probe_prompt(ROOT),
                expected_proposal_count=1,
                max_total_tokens=PROBE_TOKEN_CEILING,
            )
            if result.returned_model != MODEL:
                raise CanaryBrokerError(
                    "V4 returned a model other than exact gpt-5.4"
                )
            if canonical_value(result.response) != canonical_value(sentinel):
                raise CanaryBrokerError(
                    "V4 response differed from the frozen sentinel"
                )
            validate_v4_response_contract(
                result.response,
                provider_schema=provider_schema,
            )
            broker._connection.row_factory = sqlite3.Row  # noqa: SLF001
            success_row = broker._connection.execute(  # noqa: SLF001
                "SELECT receipt_json FROM calls WHERE logical_call_id=?",
                (V4_LOGICAL_CALL_ID,),
            ).fetchone()
            if success_row is None:
                raise SystemExit("V4 successful call receipt row is missing")
            success_provider_receipt = json.loads(
                str(success_row["receipt_json"])
            )
            ended_ns = time.monotonic_ns()
            entry = _attempt_chain_entry(
                entry={
                    "ordinal": ordinal,
                    **attempt_identity,
                    "logical_call_id": V4_LOGICAL_CALL_ID,
                    "request_payload_digest": contract[
                        "request_payload_digest"
                    ],
                    "request_envelope_digest": result.request_digest,
                    "sqlite_calls_schema_digest": schema_digest,
                    "monotonic_start_ns": started_ns,
                    "monotonic_end_ns": ended_ns,
                    "classification": (
                        "PASS_EXACT_GPT_5_4_AUTH_PROVIDER_AND_LOCAL_SCHEMA"
                    ),
                    "http_status": 200,
                    "retry_eligible": False,
                    "backoff_ms_after_attempt": 0,
                    "termination_reason": (
                        "FIRST_VALID_LOCAL_EQUIVALENT_RESPONSE_ACCEPTED"
                    ),
                    "returned_model": result.returned_model,
                    "response_digest": result.response_digest,
                    "provider_receipt_digest": success_provider_receipt[
                        "receipt_digest"
                    ],
                    "broker_outcome_digest": None,
                    "provider_error_detail_digest": None,
                    "local_uniqueness_contract_digest": (
                        V4_UNIQUENESS_CONTRACT_DIGEST
                    ),
                    "local_uniqueness_status": "PASS",
                },
                prior_attempt_digest=prior_attempt_digest,
            )
            attempts.append(entry)
            receipt = _v4_top_receipt(
                manifest=manifest,
                attempts=attempts,
                status="PASS",
                final_classification=entry["classification"],
                termination_reason=entry["termination_reason"],
                returned_model=result.returned_model,
                response_digest=result.response_digest,
            )
            _write_v4_checkpoint(receipt)
            print(
                json.dumps(
                    {
                        "classification": entry["classification"],
                        "physical_provider_calls": len(attempts),
                        "receipt_sha256": bytes_sha256(
                            (ROOT / V4_ATTEMPT_RECEIPT_REL).read_bytes()
                        ),
                        "retry_count": len(attempts) - 1,
                        "status": "PASS",
                    },
                    sort_keys=True,
                )
            )
            return 0
        except Exception as error:
            if broker is None:
                raise
            broker._connection.row_factory = sqlite3.Row  # noqa: SLF001
            row = broker._connection.execute(  # noqa: SLF001
                "SELECT logical_call_id, request_digest, response_digest, "
                "returned_model, status, error_type, error_detail_json, "
                "receipt_json, outcome_json FROM calls WHERE logical_call_id=?",
                (V4_LOGICAL_CALL_ID,),
            ).fetchone()
            if row is None:
                raise SystemExit(
                    "V4 local receipt failure stopped before retry"
                ) from error
            evidence = _classify_v3_failure(
                error=error,
                row=row,
                semantic_after_success=(result is not None),
            )
            retry = evidence["retry_eligible"] and ordinal < 3
            local_contract_failure = isinstance(
                error,
                V4LocalUniquenessError,
            )
            termination = (
                "RETRY_SCHEDULED"
                if retry
                else (
                    "LOCAL_RESPONSE_CONTRACT_TERMINAL_FAILURE"
                    if local_contract_failure
                    else (
                        "TRANSIENT_ATTEMPTS_EXHAUSTED"
                        if evidence["retry_eligible"]
                        else "DETERMINISTIC_TERMINAL_FAILURE"
                    )
                )
            )
            ended_ns = time.monotonic_ns()
            entry = _attempt_chain_entry(
                entry={
                    "ordinal": ordinal,
                    **attempt_identity,
                    "logical_call_id": V4_LOGICAL_CALL_ID,
                    "request_payload_digest": contract[
                        "request_payload_digest"
                    ],
                    "request_envelope_digest": row["request_digest"],
                    "sqlite_calls_schema_digest": manifest[
                        "bounded_retry"
                    ]["sqlite_calls_schema_digest"],
                    "monotonic_start_ns": started_ns,
                    "monotonic_end_ns": ended_ns,
                    **evidence,
                    "backoff_ms_after_attempt": (
                        backoffs[ordinal - 1] if retry else 0
                    ),
                    "termination_reason": termination,
                    "returned_model": row["returned_model"],
                    "response_digest": row["response_digest"],
                    "local_uniqueness_contract_digest": (
                        V4_UNIQUENESS_CONTRACT_DIGEST
                    ),
                    "local_uniqueness_status": (
                        "FAIL" if local_contract_failure else "NOT_REACHED"
                    ),
                },
                prior_attempt_digest=prior_attempt_digest,
            )
            attempts.append(entry)
            prior_attempt_digest = entry["attempt_digest"]
            status = "IN_PROGRESS" if retry else "BLOCKED"
            receipt = _v4_top_receipt(
                manifest=manifest,
                attempts=attempts,
                status=status,
                final_classification=entry["classification"],
                termination_reason=termination,
                returned_model=None,
                response_digest=None,
            )
            _write_v4_checkpoint(receipt)
            if not retry:
                print(
                    json.dumps(
                        {
                            "classification": entry["classification"],
                            "physical_provider_calls": len(attempts),
                            "receipt_sha256": bytes_sha256(
                                (ROOT / V4_ATTEMPT_RECEIPT_REL).read_bytes()
                            ),
                            "retry_count": len(attempts) - 1,
                            "status": "BLOCKED",
                        },
                        sort_keys=True,
                    )
                )
                return 2
        finally:
            if broker is not None:
                broker.close()
    raise SystemExit("V4 bounded retry loop reached an impossible state")


def main() -> int:
    if "--v4" in sys.argv:
        sys.argv.remove("--v4")
        return _main_v4()
    if "--v3" in sys.argv:
        sys.argv.remove("--v3")
        return _main_v3()
    return _main_v2()


if __name__ == "__main__":
    raise SystemExit(main())
