"""Single-request laboratory API broker for Pilot and Main campaigns."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
import re
import signal
import socket
import sqlite3
import ssl
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urlerror
from urllib import request as urlrequest

import jsonschema

from .audit_snapshot import create_immutable_audit_snapshot
from .broker_process import (
    BrokerCallOutcomeV2,
    BrokerFailureClassV1,
    BrokerProcessExitReceiptV2,
    ProviderRequestConfirmationV1,
)
from .canonical import canonical_json_bytes, sha256_digest, validate_sha256
from .canary_broker import CanaryBrokerCallV1, CanaryBrokerError


@contextmanager
def _hard_transport_deadline(timeout_seconds: float):
    """Bound the complete open/read operation on the Linux campaign parent.

    ``urllib``'s timeout is a socket-operation timeout: a peer can keep the
    connection alive indefinitely by producing occasional bytes.  The
    campaign needs one wall-clock deadline around both ``urlopen`` and
    ``read`` so a transient Provider stall reaches the existing bounded retry
    loop instead of blocking round closure forever.
    """

    if (
        timeout_seconds <= 0.0
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def expire(_signum: int, _frame: object) -> None:
        raise TimeoutError("laboratory API hard wall-clock deadline exceeded")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(1e-6, previous_timer[0] - elapsed),
                previous_timer[1],
            )


class LabApiResponseFailureReasonV1(str, Enum):
    """Allowlisted, content-free HTTP-200 response failure reasons."""

    ENVELOPE_JSON_DECODE = "ENVELOPE_JSON_DECODE"
    ENVELOPE_SHAPE = "ENVELOPE_SHAPE"
    CHOICES_SHAPE = "CHOICES_SHAPE"
    CHOICE_SHAPE = "CHOICE_SHAPE"
    MESSAGE_SHAPE = "MESSAGE_SHAPE"
    MESSAGE_CONTENT_TYPE_OR_EMPTY = "MESSAGE_CONTENT_TYPE_OR_EMPTY"
    RESPONSES_STATUS = "RESPONSES_STATUS"
    RESPONSES_OUTPUT_SHAPE = "RESPONSES_OUTPUT_SHAPE"
    RESPONSES_MESSAGE_SHAPE = "RESPONSES_MESSAGE_SHAPE"
    RESPONSES_CONTENT_SHAPE = "RESPONSES_CONTENT_SHAPE"
    CONTENT_JSON_DECODE = "CONTENT_JSON_DECODE"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"
    PROPOSAL_COUNT = "PROPOSAL_COUNT"
    USAGE_SHAPE = "USAGE_SHAPE"
    TOKEN_USAGE_TYPE = "TOKEN_USAGE_TYPE"
    TOKEN_CEILING = "TOKEN_CEILING"
    RETURNED_MODEL_TYPE_OR_EMPTY = "RETURNED_MODEL_TYPE_OR_EMPTY"


def _provider_http_error_fields(raw: bytes, api_key: str = "") -> dict[str, Any]:
    """Retain bounded routing diagnostics and usage, not response content."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
    if not isinstance(payload, dict):
        return {}
    error = payload.get('error')
    if isinstance(error, str):
        error = {"message": error}
    if not isinstance(error, dict):
        error = {}
    fields = {
        'provider_error_' + name: (
            value.replace(api_key, '[REDACTED]') if api_key else value
        )[:512]
        for name in ('code', 'type', 'param', 'message')
        if isinstance(value := error.get(name), str)
    }
    if isinstance(payload.get("usage"), dict):
        fields["usage"] = {
            key: value for key, value in payload["usage"].items()
            if key in {"input_tokens", "prompt_tokens", "output_tokens",
                       "completion_tokens", "total_tokens", "cached_input_tokens"}
        }
    elif payload.get("usage") is not None:
        fields["provider_usage_incomplete"] = True
    if any(payload.get(key) for key in ("choices", "output", "content")):
        fields["provider_response_has_output"] = True
    return fields


def provider_http_failure_kind(http_status: int, detail: Mapping[str, Any]) -> str | None:
    """Recognize observed transient relay failures, including explicit routing faults."""
    message = " ".join(str(detail.get("provider_error_message") or "").lower().split())
    code = str(detail.get("provider_error_code") or "").lower()
    if http_status == 400 and (
        "unknown provider for model" in message
        or "no available channel" in message
        or "no channel available" in message
        or code in {"unknown_provider", "no_available_channel"}
    ):
        return "TRANSIENT_PROVIDER_ROUTING"
    if http_status == 503 and (
        "overloaded" in message or "temporarily unavailable" in message
    ):
        return "SERVICE_UNAVAILABLE"
    if http_status == 429 and "all credentials" in message and "cooling down" in message:
        return "CREDENTIAL_COOLDOWN"
    if http_status == 500 and (
        "upstream error: do request failed" in message
        or re.search(r"\bstream error:\s*stream id \d+;\s*protocol_error;\s*received from peer\b", message)
    ):
        return "UPSTREAM_TRANSPORT_FAILURE"
    if http_status == 504:
        return "GATEWAY_TIMEOUT"
    return None


def provider_http_failure_is_unbilled(http_status: int, detail: Mapping[str, Any]) -> bool:
    """A recognized pure error is unbilled; partial/positive usage remains charged."""
    usage = detail.get("usage")
    if isinstance(usage, Mapping):
        counts = (usage.get("input_tokens", usage.get("prompt_tokens")),
                  usage.get("output_tokens", usage.get("completion_tokens")))
        if not all(type(value) is int and value == 0 for value in counts):
            return False
        if "total_tokens" in usage and (
            type(usage["total_tokens"]) is not int or usage["total_tokens"] != 0
        ):
            return False
    if http_status == 400 and not re.search(
        r"\bunknown provider for model gpt-5\.6-terra(?![\w.-])",
        str(detail.get("provider_error_message") or "").lower(),
    ):
        # Other explicit routing faults can retry, but their price is not known.
        return False
    return (
        provider_http_failure_kind(http_status, detail) is not None
        and not detail.get("provider_response_has_output")
        and not detail.get("provider_usage_incomplete")
    )


class LabApiRequestAdmissionError(RuntimeError):
    """The request budget stopped this call before any HTTP work."""


class LabApiResponseContractError(ValueError):
    """Typed response failure whose public value is an allowlisted code only."""

    def __init__(
        self,
        reason: LabApiResponseFailureReasonV1,
        *,
        usage: Mapping[str, int] | None = None,
        response_metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.reason = reason
        self.usage = dict(usage or {})
        self.response_metadata = dict(response_metadata or {})
        super().__init__(reason.value)


def _provider_json_content(content: str) -> str:
    """Return the JSON payload after one endpoint-emitted reasoning prefix.

    Endpoint 1 can prepend a closed ``<think>...</think>`` block even when
    the request uses strict ``json_schema`` output.  Treat only that observed,
    leading wrapper as transport metadata; the remaining payload still goes
    through the unchanged JSON decode and schema validation path.
    """

    candidate = content.strip()
    think_open = "<think>"
    think_close = "</think>"
    if not candidate.startswith(think_open):
        return candidate
    close_index = candidate.find(think_close, len(think_open))
    if close_index < 0:
        return candidate
    return candidate[close_index + len(think_close) :].strip()


def validate_provider_strict_schema(
    schema: Mapping[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> None:
    """Validate the strict object rule required by the laboratory Provider."""

    properties = schema.get("properties")
    if schema.get("type") == "object":
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        property_names = set(str(item) for item in properties)
        required_names = (
            set(str(item) for item in required)
            if isinstance(required, list)
            else set()
        )
        if required_names != property_names:
            missing = sorted(property_names - required_names)
            extra = sorted(required_names - property_names)
            location = ".".join(path) or "$"
            raise CanaryBrokerError(
                "provider strict schema object is not closed at "
                f"{location}: missing_required={missing}, "
                f"unknown_required={extra}"
            )
        if schema.get("additionalProperties") is not False:
            location = ".".join(path) or "$"
            raise CanaryBrokerError(
                "provider strict schema object allows extra properties at "
                f"{location}"
            )
        for name, subschema in properties.items():
            if isinstance(subschema, Mapping):
                validate_provider_strict_schema(
                    subschema,
                    path=(*path, "properties", str(name)),
                )
    items = schema.get("items")
    if isinstance(items, Mapping):
        validate_provider_strict_schema(items, path=(*path, "items"))
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for index, branch in enumerate(branches):
                if isinstance(branch, Mapping):
                    validate_provider_strict_schema(
                        branch,
                        path=(*path, keyword, str(index)),
                    )


@dataclass(frozen=True, slots=True)
class LabApiBrokerReleaseV1:
    transport: str
    endpoint_digest: str
    model: str
    response_schema_digest: str
    request_mode: str
    temperature: float
    max_total_tokens_per_call: int
    timeout_ms: int
    retry_count: int
    reasoning_effort: str | None
    release_digest: str

    @classmethod
    def create(
        cls,
        *,
        base_url: str,
        model: str,
        response_schema_digest: str,
        max_total_tokens_per_call: int,
        timeout_ms: int,
        reasoning_effort: str | None = None,
        wire_api: str = "chat_completions",
    ) -> "LabApiBrokerReleaseV1":
        if wire_api == "chat_completions":
            transport = "HTTPS_CHAT_COMPLETIONS_V1"
            request_mode = "SINGLE_JSON_SCHEMA_NO_TOOLS"
        elif wire_api == "responses":
            transport = "HTTPS_RESPONSES_V1"
            request_mode = "SINGLE_RESPONSES_JSON_SCHEMA_NO_TOOLS"
        else:
            raise CanaryBrokerError("laboratory API wire API is invalid")
        payload = {
            "transport": transport,
            "endpoint_digest": sha256_digest(
                {"base_url": base_url.rstrip("/")}
            ),
            "model": model,
            "response_schema_digest": response_schema_digest,
            "request_mode": request_mode,
            "temperature": 0.0,
            "max_total_tokens_per_call": max_total_tokens_per_call,
            "timeout_ms": timeout_ms,
            "retry_count": 0,
            "reasoning_effort": reasoning_effort,
        }
        return cls(**payload, release_digest=sha256_digest(payload))

    def verify(self) -> None:
        validate_sha256(
            self.endpoint_digest, field_name="endpoint_digest"
        )
        validate_sha256(
            self.response_schema_digest,
            field_name="response_schema_digest",
        )
        payload = self.to_dict()
        expected = payload.pop("release_digest")
        if sha256_digest(payload) != expected:
            raise CanaryBrokerError("laboratory API release digest mismatch")
        if (
            (self.transport, self.request_mode)
            not in {
                (
                    "HTTPS_CHAT_COMPLETIONS_V1",
                    "SINGLE_JSON_SCHEMA_NO_TOOLS",
                ),
                (
                    "HTTPS_RESPONSES_V1",
                    "SINGLE_RESPONSES_JSON_SCHEMA_NO_TOOLS",
                ),
            }
            or self.temperature != 0.0
            or self.retry_count != 0
            or self.max_total_tokens_per_call < 1
            or self.timeout_ms < 1
        ):
            raise CanaryBrokerError("laboratory API release is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_digest": self.endpoint_digest,
            "max_total_tokens_per_call": self.max_total_tokens_per_call,
            "model": self.model,
            "release_digest": self.release_digest,
            "request_mode": self.request_mode,
            "response_schema_digest": self.response_schema_digest,
            "retry_count": self.retry_count,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "timeout_ms": self.timeout_ms,
            "transport": self.transport,
        }


@dataclass(frozen=True, slots=True)
class LabApiCallReceiptV1:
    logical_call_id: str
    proposal_generation_session_id: str
    request_digest: str
    response_digest: str | None
    release_digest: str
    status: str
    error_type: str | None
    error_detail_digest: str | None
    http_status: int | None
    latency_ms: int
    receipt_digest: str

    @classmethod
    def create(
        cls,
        *,
        logical_call_id: str,
        proposal_generation_session_id: str,
        request_digest: str,
        response_digest: str | None,
        release_digest: str,
        status: str,
        error_type: str | None,
        error_detail_digest: str | None,
        http_status: int | None,
        latency_ms: int,
    ) -> "LabApiCallReceiptV1":
        payload = {
            "logical_call_id": logical_call_id,
            "proposal_generation_session_id": (
                proposal_generation_session_id
            ),
            "request_digest": request_digest,
            "response_digest": response_digest,
            "release_digest": release_digest,
            "status": status,
            "error_type": error_type,
            "error_detail_digest": error_detail_digest,
            "http_status": http_status,
            "latency_ms": latency_ms,
        }
        return cls(**payload, receipt_digest=sha256_digest(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_detail_digest": self.error_detail_digest,
            "http_status": self.http_status,
            "latency_ms": self.latency_ms,
            "logical_call_id": self.logical_call_id,
            "proposal_generation_session_id": (
                self.proposal_generation_session_id
            ),
            "receipt_digest": self.receipt_digest,
            "release_digest": self.release_digest,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "status": self.status,
        }


def load_lab_api_credential_pairs(
    config_path: Path,
) -> tuple[tuple[str, str], ...]:
    """Read ordered endpoint/key pairs without persisting their secrets."""

    values: dict[str, list[str]] = {"base_url": [], "api_key": []}
    for raw_line in config_path.resolve().read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.partition("=")
        if separator and name.strip() in {"api_key", "base_url"}:
            parsed = ast.literal_eval(raw_value.strip())
            if isinstance(parsed, str):
                values[name.strip()].append(parsed)
    if not values["api_key"] or not values["base_url"]:
        raise CanaryBrokerError(
            "laboratory API config requires api_key and base_url"
        )
    if len(values["api_key"]) != len(values["base_url"]):
        raise CanaryBrokerError(
            "laboratory API config contains incomplete credential pairs"
        )
    pairs: list[tuple[str, str]] = []
    for base_url, api_key in zip(
        values["base_url"], values["api_key"], strict=True
    ):
        normalized_base_url = base_url.strip().rstrip("/")
        normalized_api_key = api_key.strip()
        if not normalized_api_key or not normalized_base_url:
            raise CanaryBrokerError(
                "laboratory API config contains an empty credential pair"
            )
        if not normalized_base_url.startswith("https://"):
            raise CanaryBrokerError(
                "laboratory API base_url must use HTTPS"
            )
        pairs.append((normalized_base_url, normalized_api_key))
    return tuple(pairs)


def load_lab_api_credentials(
    config_path: Path,
    credential_index: int = 0,
) -> tuple[str, str]:
    """Return one ordered credential pair; index zero is the default."""

    if not isinstance(credential_index, int) or credential_index < 0:
        raise CanaryBrokerError("laboratory API credential index is invalid")
    pairs = load_lab_api_credential_pairs(config_path)
    if credential_index >= len(pairs):
        raise CanaryBrokerError(
            "laboratory API credential index is outside the configured pairs"
        )
    return pairs[credential_index]


class LabApiCanaryBrokerV1:
    """Exactly one schema-bound API request per new logical call."""

    def __init__(
        self,
        private_root: Path,
        *,
        schema_path: Path,
        config_path: Path,
        model: str,
        max_total_tokens_per_call: int,
        reasoning_effort: str | None = None,
        wire_api: str = "chat_completions",
        credential_index: int = 0,
        timeout_ms: int = 900_000,
        release_manifest_path: Path | None = None,
        request_budget: Any = None,
    ) -> None:
        self.request_budget = request_budget
        self.private_root = private_root.resolve()
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.schema_path = schema_path.resolve()
        self.schema_bytes = self.schema_path.read_bytes()
        self.schema = json.loads(self.schema_bytes)
        jsonschema.validators.validator_for(self.schema).check_schema(
            self.schema
        )
        validate_provider_strict_schema(self.schema)
        self.schema_file_sha256 = hashlib.sha256(
            self.schema_bytes
        ).hexdigest()
        self.config_path = config_path.resolve()
        self.credential_index = credential_index
        self.credential_config_digest = hashlib.sha256(
            self.config_path.read_bytes()
        ).hexdigest()
        self.base_url, self._api_key = load_lab_api_credentials(
            self.config_path,
            credential_index=credential_index,
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        if wire_api not in {"chat_completions", "responses"}:
            raise CanaryBrokerError("laboratory API wire API is invalid")
        self.wire_api = wire_api
        self.max_total_tokens_per_call = max_total_tokens_per_call
        self.timeout_ms = timeout_ms
        computed_release = LabApiBrokerReleaseV1.create(
            base_url=self.base_url,
            model=model,
            response_schema_digest=self.schema_file_sha256,
            max_total_tokens_per_call=max_total_tokens_per_call,
            timeout_ms=timeout_ms,
            reasoning_effort=reasoning_effort,
            wire_api=wire_api,
        )
        if release_manifest_path is None:
            self.release = computed_release
        else:
            frozen = LabApiBrokerReleaseV1(
                **json.loads(
                    release_manifest_path.resolve().read_text(
                        encoding="utf-8"
                    )
                )
            )
            frozen.verify()
            if frozen.to_dict() != computed_release.to_dict():
                raise CanaryBrokerError(
                    "runtime laboratory API release differs from frozen bytes"
                )
            self.release = frozen
        self.release.verify()
        self.endpoint_digest = self.release.endpoint_digest
        self.credential_identity_digest = sha256_digest(
            {
                "credential_config_digest": self.credential_config_digest,
                "credential_index": self.credential_index,
                "endpoint_digest": self.endpoint_digest,
            }
        )
        (self.private_root / "LAB_API_BROKER_RELEASE_V1.json").write_bytes(
            canonical_json_bytes(self.release.to_dict()) + b"\n"
        )
        self.db_path = self.private_root / "broker.sqlite3"
        self._connection = sqlite3.connect(self.db_path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                logical_call_id TEXT PRIMARY KEY,
                request_digest TEXT NOT NULL,
                response_digest TEXT,
                response_json TEXT,
                input_tokens INTEGER NOT NULL,
                cached_input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                returned_model TEXT,
                status TEXT NOT NULL,
                error_type TEXT,
                error_detail_json TEXT,
                proposal_generation_session_id TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                closure_receipt_json TEXT,
                outcome_json TEXT,
                broker_release_digest TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._api_key = ""
        self._connection.close()

    def call_count(self) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(*) FROM calls WHERE status='SUCCESS'"
            ).fetchone()[0]
        )

    def create_audit_snapshot(
        self, snapshot_path: Path, *, audit_purpose: str
    ) -> Any:
        return create_immutable_audit_snapshot(
            writer_connection=self._connection,
            source_db_path=self.db_path,
            snapshot_path=snapshot_path,
            source_schema_identity=sha256_digest(
                {
                    "calls_table": "LAB_API_BROKER_CALLS_V1",
                    "release_digest": self.release.release_digest,
                }
            ),
            audit_purpose=audit_purpose,
        )

    def _stored(
        self,
        logical_call_id: str,
        request_digest: str,
        proposal_generation_session_id: str,
    ) -> CanaryBrokerCallV1 | None:
        self._connection.row_factory = sqlite3.Row
        row = self._connection.execute(
            "SELECT * FROM calls WHERE logical_call_id=?",
            (logical_call_id,),
        ).fetchone()
        if row is None:
            return None
        if (
            row["request_digest"] != request_digest
            or row["proposal_generation_session_id"]
            != proposal_generation_session_id
            or row["broker_release_digest"]
            != self.release.release_digest
        ):
            raise CanaryBrokerError(
                "logical API call identity differs from stored bytes"
            )
        if row["status"] != "SUCCESS":
            outcome = BrokerCallOutcomeV2(
                **json.loads(str(row["outcome_json"]))
            )
            receipt = BrokerProcessExitReceiptV2(
                **json.loads(str(row["closure_receipt_json"]))
            )
            raise CanaryBrokerError(
                "stored laboratory API call is a terminal failure",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                billed_tokens=int(row["total_tokens"]),
                wall_time_ms=int(row["latency_ms"]),
            )
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=request_digest,
            response_digest=str(row["response_digest"]),
            response=json.loads(str(row["response_json"])),
            input_tokens=int(row["input_tokens"]),
            cached_input_tokens=int(row["cached_input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            total_tokens=int(row["total_tokens"]),
            latency_ms=int(row["latency_ms"]),
            returned_model=str(row["returned_model"]),
        )

    def replay_stored(
        self,
        *,
        logical_call_id: str,
        proposal_generation_session_id: str,
        request_digest: str,
    ) -> CanaryBrokerCallV1:
        """Replay a sealed in-flight call without rebuilding its request bytes."""

        self._connection.row_factory = sqlite3.Row
        row = self._connection.execute(
            "SELECT * FROM calls WHERE logical_call_id=?",
            (logical_call_id,),
        ).fetchone()
        if row is None:
            raise CanaryBrokerError("sealed laboratory API call is missing")
        if (
            row["proposal_generation_session_id"]
            != proposal_generation_session_id
            or row["request_digest"] != request_digest
        ):
            raise CanaryBrokerError(
                "sealed laboratory API call identity differs from checkpoint"
            )
        if row["status"] != "SUCCESS":
            outcome = BrokerCallOutcomeV2(**json.loads(str(row["outcome_json"])))
            receipt = BrokerProcessExitReceiptV2(
                **json.loads(str(row["closure_receipt_json"]))
            )
            raise CanaryBrokerError(
                "stored laboratory API call is a terminal failure",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=0,
                input_tokens=0,
                output_tokens=0,
                billed_tokens=0,
                wall_time_ms=0,
            )
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=str(row["request_digest"]),
            response_digest=str(row["response_digest"]),
            response=json.loads(str(row["response_json"])),
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            returned_model=str(row["returned_model"]),
        )

    def sealed_request_identity(
        self,
        *,
        proposal_generation_session_id: str,
        prompt: str,
        expected_proposal_count: int,
        max_total_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Mapping[str, Any]:
        """Build the exact secret-free request identity consumed by the broker."""

        effective_ceiling = int(
            max_total_tokens
            if max_total_tokens is not None
            else self.max_total_tokens_per_call
        )
        if not 1 <= effective_ceiling <= self.max_total_tokens_per_call:
            raise CanaryBrokerError(
                "per-call token ceiling is outside the API release"
            )
        output_ceiling = int(
            max_output_tokens if max_output_tokens is not None else effective_ceiling
        )
        if not 1 <= output_ceiling <= effective_ceiling:
            raise CanaryBrokerError(
                "output token ceiling is outside the total token ceiling"
            )
        if self.wire_api == "responses":
            request_payload = {
                "model": self.model,
                "input": prompt,
                # Responses applies this ceiling to visible output plus hidden
                # reasoning, unlike the endpoint2 Chat Completions adapter.
                "max_output_tokens": output_ceiling,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "recclaw_campaign_proposals",
                        "strict": True,
                        "schema": self.schema,
                    }
                },
            }
            if self.reasoning_effort is not None:
                request_payload["reasoning"] = {
                    "effort": self.reasoning_effort
                }
        else:
            request_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_completion_tokens": output_ceiling,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "recclaw_campaign_proposals",
                        "strict": True,
                        "schema": self.schema,
                    },
                },
            }
            if self.reasoning_effort is not None:
                request_payload["reasoning_effort"] = self.reasoning_effort
        if self.request_budget is not None:
            if self.wire_api != "responses" or self.reasoning_effort != "medium":
                raise ValueError("E1 requires Responses with medium reasoning")
            # Match the E1 supervisor's tokenizer and framing headroom. UTF-8
            # bytes are not token counts and prematurely reject valid contexts.
            import tiktoken

            encoding = tiktoken.encoding_for_model("gpt-5")
            input_bound = len(encoding.encode(
                json.dumps(request_payload, ensure_ascii=False), disallowed_special=()
            )) + 1024
            allowance = min(output_ceiling, 16000, effective_ceiling - input_bound)
            if allowance < 1024:
                raise ValueError("E1 request exceeds per-call context budget; compact history")
            request_payload["max_output_tokens"] = allowance
        return {
            "expected_proposal_count": expected_proposal_count,
            "proposal_generation_session_id": proposal_generation_session_id,
            "request_payload": request_payload,
        }

    def call(
        self,
        *,
        logical_call_id: str,
        prompt: str,
        expected_proposal_count: int,
        max_total_tokens: int | None = None,
    ) -> CanaryBrokerCallV1:
        return self.call_with_session(
            logical_call_id=logical_call_id,
            proposal_generation_session_id=logical_call_id,
            prompt=prompt,
            expected_proposal_count=expected_proposal_count,
            max_total_tokens=max_total_tokens,
        )

    def call_with_session(
        self,
        *,
        logical_call_id: str,
        proposal_generation_session_id: str,
        prompt: str,
        expected_proposal_count: int,
        max_total_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> CanaryBrokerCallV1:
        effective_ceiling = int(
            max_total_tokens
            if max_total_tokens is not None
            else self.max_total_tokens_per_call
        )
        request_identity = self.sealed_request_identity(
            proposal_generation_session_id=proposal_generation_session_id,
            prompt=prompt,
            expected_proposal_count=expected_proposal_count,
            max_total_tokens=max_total_tokens,
            max_output_tokens=max_output_tokens,
        )
        request_payload = request_identity["request_payload"]
        request_digest = sha256_digest(request_identity)
        prior = self._stored(
            logical_call_id,
            request_digest,
            proposal_generation_session_id,
        )
        if prior is not None:
            return prior

        reservation_ceiling = effective_ceiling
        request_token_bound = getattr(self.request_budget, "request_token_bound", None)
        if callable(request_token_bound):
            reservation_ceiling = request_token_bound(
                request_payload, ceiling=effective_ceiling
            )

        request = urlrequest.Request(
            (
                f"{self.base_url}/responses"
                if self.wire_api == "responses"
                else f"{self.base_url}/chat/completions"
            ),
            data=canonical_json_bytes(request_payload),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        http_status: int | None = None
        envelope: Mapping[str, Any] | None = None
        try:
            reservation = (
                self.request_budget.reserve_call(self.model, request_digest, reservation_ceiling)
                if self.request_budget is not None else None
            )
        except Exception as error:
            raise LabApiRequestAdmissionError(str(error)) from error
        try:
            context = ssl.create_default_context()
            ignore_eof = getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0)
            if ignore_eof:
                context.options |= ignore_eof
            with _hard_transport_deadline(self.timeout_ms / 1000):
                with urlrequest.urlopen(
                    request,
                    timeout=self.timeout_ms / 1000,
                    context=context,
                ) as raw_response:
                    http_status = int(raw_response.status)
                    raw_bytes = raw_response.read()
            latency_ms = int((time.monotonic() - started) * 1000)
            try:
                envelope = json.loads(raw_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.ENVELOPE_JSON_DECODE
                ) from error
            if not isinstance(envelope, Mapping):
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.ENVELOPE_SHAPE
                )
            response, usage, returned_model = self._parse_response(
                envelope=envelope,
                expected_proposal_count=expected_proposal_count,
                effective_ceiling=effective_ceiling,
                output_ceiling=int(request_payload[
                    "max_output_tokens"
                    if self.wire_api == "responses"
                    else "max_completion_tokens"
                ]),
            )
        except urlerror.HTTPError as error:
            http_status = int(error.code)
            latency_ms = int((time.monotonic() - started) * 1000)
            raw_error_body = error.read(16_384)
            error_fields = _provider_http_error_fields(raw_error_body, self._api_key)
            receipt, outcome = self._persist_failure(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=(
                    proposal_generation_session_id
                ),
                request_digest=request_digest,
                error_type=f"HTTP_{error.code}",
                error_detail={"http_status": http_status, **error_fields},
                http_status=http_status,
                latency_ms=latency_ms,
            )
            release_external_failure = getattr(
                self.request_budget, "release_external_failure", None
            )
            if callable(release_external_failure):
                release_external_failure(
                    reservation,
                    http_status=http_status,
                    response_body=raw_error_body,
                )
            raise CanaryBrokerError(
                f"laboratory API returned HTTP {error.code}",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                wall_time_ms=latency_ms,
            ) from error
        except (TimeoutError, socket.timeout) as error:
            latency_ms = int((time.monotonic() - started) * 1000)
            receipt, outcome = self._persist_failure(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=(
                    proposal_generation_session_id
                ),
                request_digest=request_digest,
                error_type="TIMEOUT",
                error_detail={"exception_type": type(error).__name__},
                http_status=http_status,
                latency_ms=latency_ms,
            )
            raise CanaryBrokerError(
                "laboratory API call timed out",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                wall_time_ms=latency_ms,
            ) from error
        except (urlerror.URLError, ssl.SSLError, OSError) as error:
            latency_ms = int((time.monotonic() - started) * 1000)
            receipt, outcome = self._persist_failure(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=(
                    proposal_generation_session_id
                ),
                request_digest=request_digest,
                error_type="TRANSPORT_ERROR",
                error_detail={"exception_type": type(error).__name__},
                http_status=http_status,
                latency_ms=latency_ms,
            )
            raise CanaryBrokerError(
                "laboratory API transport failed",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                wall_time_ms=latency_ms,
            ) from error
        except LabApiResponseContractError as error:
            latency_ms = int((time.monotonic() - started) * 1000)
            response_envelope = envelope if isinstance(envelope, Mapping) else {}
            usage = error.usage or self._best_effort_usage(response_envelope)
            returned_model = response_envelope.get(
                "model", error.response_metadata.get("returned_model")
            )
            error_detail = {"reason_code": error.reason.value}
            error_detail.update(error.response_metadata)
            receipt, outcome = self._persist_failure(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=(
                    proposal_generation_session_id
                ),
                request_digest=request_digest,
                error_type="RESPONSE_CONTRACT_ERROR",
                error_detail=error_detail,
                http_status=http_status,
                latency_ms=latency_ms,
                usage=usage,
                returned_model=(
                    returned_model if isinstance(returned_model, str) else None
                ),
            )
            if (
                error.reason
                is LabApiResponseFailureReasonV1.CONTENT_JSON_DECODE
                and envelope is not None
            ):
                try:
                    self._write_content_json_decode_observation(
                        envelope=envelope,
                        logical_call_id=logical_call_id,
                        request_digest=request_digest,
                    )
                except OSError:
                    # This sidecar is audit-only. Its filesystem failure must
                    # not change the already-persisted typed Provider failure.
                    pass
            raise CanaryBrokerError(
                "laboratory API response violated the frozen contract",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                billed_tokens=int(usage.get("total_tokens", 0)),
                wall_time_ms=latency_ms,
            ) from error
        except jsonschema.ValidationError as error:
            latency_ms = int((time.monotonic() - started) * 1000)
            usage = self._best_effort_usage(envelope or {})
            returned_model = (envelope or {}).get("model")
            receipt, outcome = self._persist_failure(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=(
                    proposal_generation_session_id
                ),
                request_digest=request_digest,
                error_type="SCHEMA_VALIDATION_FAILURE",
                error_detail={
                    "reason_code": (
                        LabApiResponseFailureReasonV1.SCHEMA_VALIDATION.value
                    )
                },
                http_status=http_status,
                latency_ms=latency_ms,
                usage=usage,
                returned_model=(
                    returned_model if isinstance(returned_model, str) else None
                ),
            )
            raise CanaryBrokerError(
                "laboratory API payload failed schema validation",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                billed_tokens=int(usage.get("total_tokens", 0)),
                wall_time_ms=latency_ms,
            ) from error

        finally:
            # Absent usage retains the reservation unless the budget's HTTP
            # failure callback released it. Stored replay reserves nothing.
            if reservation is not None and isinstance(envelope, Mapping):
                measured_usage = envelope.get("usage")
                if isinstance(measured_usage, Mapping):
                    input_count = measured_usage.get("input_tokens")
                    output_count = measured_usage.get("output_tokens")
                    if all(type(value) is int and value >= 0
                           for value in (input_count, output_count)):
                        self.request_budget.settle_call(reservation, {
                            **dict(measured_usage),
                            "total_tokens": input_count + output_count,
                        })

        response_digest = sha256_digest(response)
        receipt = LabApiCallReceiptV1.create(
            logical_call_id=logical_call_id,
            proposal_generation_session_id=proposal_generation_session_id,
            request_digest=request_digest,
            response_digest=response_digest,
            release_digest=self.release.release_digest,
            status="SUCCESS",
            error_type=None,
            error_detail_digest=None,
            http_status=http_status,
            latency_ms=latency_ms,
        )
        self._connection.execute(
            """
            INSERT INTO calls(
                logical_call_id, request_digest, response_digest,
                response_json, input_tokens, cached_input_tokens,
                output_tokens, total_tokens, latency_ms, returned_model,
                status, error_type, error_detail_json,
                proposal_generation_session_id, receipt_digest,
                receipt_json, closure_receipt_json, outcome_json,
                broker_release_digest
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUCCESS', NULL, NULL,
                ?, ?, ?, NULL, NULL, ?
            )
            """,
            (
                logical_call_id,
                request_digest,
                response_digest,
                canonical_json_bytes(response).decode("utf-8"),
                usage["input_tokens"],
                usage["cached_input_tokens"],
                usage["output_tokens"],
                usage["total_tokens"],
                latency_ms,
                returned_model,
                proposal_generation_session_id,
                receipt.receipt_digest,
                canonical_json_bytes(receipt.to_dict()).decode("utf-8"),
                self.release.release_digest,
            ),
        )
        self._connection.commit()
        if self.wire_api == "responses":
            self._write_responses_usage_observation(
                logical_call_id=logical_call_id,
                request_digest=request_digest,
                returned_model=returned_model,
                usage=usage,
            )
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=request_digest,
            response_digest=response_digest,
            response=response,
            input_tokens=usage["input_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            latency_ms=latency_ms,
            returned_model=returned_model,
        )

    def _parse_response(
        self,
        *,
        envelope: Mapping[str, Any],
        expected_proposal_count: int,
        effective_ceiling: int,
        output_ceiling: int | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, int], str]:
        effective_output_ceiling = (
            int(output_ceiling)
            if output_ceiling is not None
            else effective_ceiling
        )
        wire_api = getattr(self, "wire_api", "chat_completions")
        choice: Mapping[str, Any] = {}
        if wire_api == "responses":
            response_status = envelope.get("status")
            if response_status != "completed":
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.RESPONSES_STATUS,
                    usage=self._best_effort_usage(envelope),
                    response_metadata={"response_status": str(response_status)},
                )
            output = envelope.get("output")
            if not isinstance(output, list):
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.RESPONSES_OUTPUT_SHAPE
                )
            messages = [
                item
                for item in output
                if isinstance(item, Mapping) and item.get("type") == "message"
            ]
            if len(messages) != 1:
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.RESPONSES_MESSAGE_SHAPE
                )
            parts = messages[0].get("content")
            if not isinstance(parts, list):
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.RESPONSES_CONTENT_SHAPE
                )
            output_text_parts = [
                part.get("text")
                for part in parts
                if isinstance(part, Mapping)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ]
            if len(output_text_parts) != 1:
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.RESPONSES_CONTENT_SHAPE
                )
            content = output_text_parts[0]
        else:
            choices = envelope.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.CHOICES_SHAPE
                )
            raw_choice = choices[0]
            if not isinstance(raw_choice, Mapping):
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.CHOICE_SHAPE
                )
            choice = raw_choice
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise LabApiResponseContractError(
                    LabApiResponseFailureReasonV1.MESSAGE_SHAPE
                )
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text") or item.get("content") or "")
                    for item in content
                    if isinstance(item, Mapping)
                )
        if not isinstance(content, str) or not content.strip():
            usage = self._best_effort_usage(envelope)
            response_metadata: dict[str, str] = {}
            returned_model = envelope.get("model")
            if isinstance(returned_model, str) and returned_model.strip():
                response_metadata["returned_model"] = returned_model
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason.strip():
                response_metadata["finish_reason"] = finish_reason
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.MESSAGE_CONTENT_TYPE_OR_EMPTY,
                usage=usage,
                response_metadata=response_metadata,
            )
        try:
            response = json.loads(_provider_json_content(content))
        except json.JSONDecodeError as error:
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.CONTENT_JSON_DECODE
            ) from error
        jsonschema.validate(response, self.schema)
        proposals = response.get("proposals")
        if (
            not isinstance(proposals, list)
            or len(proposals) != expected_proposal_count
        ):
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.PROPOSAL_COUNT
            )
        raw_usage = envelope.get("usage")
        if not isinstance(raw_usage, Mapping):
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.USAGE_SHAPE
            )
        try:
            input_tokens = int(
                raw_usage.get(
                    "prompt_tokens", raw_usage.get("input_tokens", 0)
                )
            )
            output_tokens = int(
                raw_usage.get(
                    "completion_tokens", raw_usage.get("output_tokens", 0)
                )
            )
            total_tokens = int(
                raw_usage.get(
                    "total_tokens", input_tokens + output_tokens
                )
            )
            prompt_details = raw_usage.get("prompt_tokens_details")
            if not isinstance(prompt_details, Mapping):
                prompt_details = raw_usage.get("input_tokens_details")
            cached_input_tokens = (
                int(prompt_details.get("cached_tokens", 0))
                if isinstance(prompt_details, Mapping)
                else int(raw_usage.get("cached_input_tokens", 0))
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.TOKEN_USAGE_TYPE
            ) from error
        output_details = raw_usage.get("output_tokens_details")
        usage = {
            "cached_input_tokens": cached_input_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        if isinstance(output_details, Mapping):
            usage["reasoning_output_tokens"] = int(
                output_details.get("reasoning_tokens", 0)
            )
        if (
            total_tokens < 1
            or total_tokens > effective_ceiling
            or output_tokens < 1
            or output_tokens > effective_output_ceiling
        ):
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.TOKEN_CEILING,
                usage=usage,
            )
        returned_model = envelope.get("model")
        if not isinstance(returned_model, str) or not returned_model.strip():
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.RETURNED_MODEL_TYPE_OR_EMPTY
            )
        return response, usage, returned_model

    @staticmethod
    def _best_effort_usage(envelope: Mapping[str, Any]) -> dict[str, int]:
        """Extract billed usage from an HTTP-200 response without masking its failure."""

        raw_usage = envelope.get("usage")
        if not isinstance(raw_usage, Mapping):
            return {}
        try:
            input_tokens = int(
                raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0))
            )
            output_tokens = int(
                raw_usage.get(
                    "completion_tokens", raw_usage.get("output_tokens", 0)
                )
            )
            total_tokens = int(
                raw_usage.get("total_tokens", input_tokens + output_tokens)
            )
            prompt_details = raw_usage.get("prompt_tokens_details")
            if not isinstance(prompt_details, Mapping):
                prompt_details = raw_usage.get("input_tokens_details")
            cached_input_tokens = (
                int(prompt_details.get("cached_tokens", 0))
                if isinstance(prompt_details, Mapping)
                else int(raw_usage.get("cached_input_tokens", 0))
            )
        except (TypeError, ValueError, OverflowError):
            return {}
        output_details = raw_usage.get("output_tokens_details")
        usage = {
            "cached_input_tokens": cached_input_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        if isinstance(output_details, Mapping):
            usage["reasoning_output_tokens"] = int(
                output_details.get("reasoning_tokens", 0)
            )
        return usage

    def _write_responses_usage_observation(
        self,
        *,
        logical_call_id: str,
        request_digest: str,
        returned_model: str,
        usage: Mapping[str, int],
    ) -> None:
        """Persist content-free Responses reasoning usage for token-quality audit."""

        payload = {
            "logical_call_id": logical_call_id,
            "release_digest": self.release.release_digest,
            "request_digest": request_digest,
            "returned_model": returned_model,
            "schema": "recclaw.responses-usage-observation.v1",
            "usage": dict(usage),
        }
        observation = {
            **payload,
            "observation_digest": sha256_digest(payload),
        }
        (self.private_root / "RESPONSES_USAGE_OBSERVATION_V1.json").write_bytes(
            canonical_json_bytes(observation) + b"\n"
        )

    def _write_content_json_decode_observation(
        self,
        *,
        envelope: Mapping[str, Any],
        logical_call_id: str,
        request_digest: str,
    ) -> None:
        """Persist content-free diagnostics without changing receipt identity."""

        choice: Mapping[str, Any] = {}
        if self.wire_api == "responses":
            output = envelope.get("output")
            if not isinstance(output, list):
                return
            messages = [
                item
                for item in output
                if isinstance(item, Mapping) and item.get("type") == "message"
            ]
            if len(messages) != 1:
                return
            parts = messages[0].get("content")
            if not isinstance(parts, list):
                return
            texts = [
                part.get("text")
                for part in parts
                if isinstance(part, Mapping)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ]
            if len(texts) != 1:
                return
            content = texts[0]
            content_container = "RESPONSES_OUTPUT_TEXT"
        else:
            choices = envelope.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                return
            raw_choice = choices[0]
            if not isinstance(raw_choice, Mapping):
                return
            choice = raw_choice
            message = choice.get("message")
            if not isinstance(message, Mapping):
                return
            content = message.get("content")
            content_container = "STRING"
            if isinstance(content, list):
                content_container = "TEXT_PART_LIST"
                content = "".join(
                    str(item.get("text") or item.get("content") or "")
                    for item in content
                    if isinstance(item, Mapping)
                )
        if not isinstance(content, str):
            return

        stripped = content.strip()
        normalized = _provider_json_content(content)
        content_bytes = content.encode("utf-8", errors="surrogatepass")
        normalized_bytes = normalized.encode("utf-8", errors="surrogatepass")
        wrapper_class = "NONE"
        if stripped.startswith("<think>"):
            wrapper_class = (
                "LEADING_CLOSED_THINK"
                if "</think>" in stripped[len("<think>") :]
                else "LEADING_UNCLOSED_THINK"
            )

        returned_model = envelope.get("model")
        if not isinstance(returned_model, str) or not returned_model.strip():
            returned_model = None
        finish_reason = (
            envelope.get("status")
            if self.wire_api == "responses"
            else choice.get("finish_reason")
        )
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            finish_reason = None
        usage = self._best_effort_usage(envelope)
        payload = {
            "content_container": content_container,
            "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "content_size_bytes": len(content_bytes),
            "finish_reason": finish_reason,
            "leading_wrapper_class": wrapper_class,
            "logical_call_id": logical_call_id,
            "normalized_candidate_sha256": hashlib.sha256(
                normalized_bytes
            ).hexdigest(),
            "normalized_candidate_size_bytes": len(normalized_bytes),
            "reason_code": (
                LabApiResponseFailureReasonV1.CONTENT_JSON_DECODE.value
            ),
            "release_digest": self.release.release_digest,
            "request_digest": request_digest,
            "returned_model": returned_model,
            "schema": "recclaw.lab-api-response-failure-observation.v1",
            "usage": usage or None,
        }
        observation = {
            **payload,
            "observation_digest": sha256_digest(payload),
        }
        observation_path = self.private_root / (
            "LAB_API_RESPONSE_FAILURE_OBSERVATION_V1_"
            f"{request_digest}.json"
        )
        observation_path.write_bytes(
            canonical_json_bytes(observation) + b"\n"
        )

    def _persist_failure(
        self,
        *,
        logical_call_id: str,
        proposal_generation_session_id: str,
        request_digest: str,
        error_type: str,
        error_detail: Mapping[str, Any],
        http_status: int | None,
        latency_ms: int,
        usage: Mapping[str, int] | None = None,
        returned_model: str | None = None,
    ) -> tuple[BrokerProcessExitReceiptV2, BrokerCallOutcomeV2]:
        usage = dict(usage or {})
        input_tokens = int(usage.get("input_tokens", 0))
        cached_input_tokens = int(usage.get("cached_input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", 0))
        error_detail_json = canonical_json_bytes(error_detail).decode(
            "utf-8"
        )
        receipt = LabApiCallReceiptV1.create(
            logical_call_id=logical_call_id,
            proposal_generation_session_id=proposal_generation_session_id,
            request_digest=request_digest,
            response_digest=None,
            release_digest=self.release.release_digest,
            status="FAILED",
            error_type=error_type,
            error_detail_digest=sha256_digest(error_detail),
            http_status=http_status,
            latency_ms=latency_ms,
        )
        error_bytes = error_detail_json.encode("utf-8")
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        failure_class = self._failure_class(
            error_type=error_type,
            http_status=http_status,
        )
        closure_receipt_payload = {
            "start_record_digest": request_digest,
            "spawn_succeeded": True,
            "pid_or_private_process_ref": None,
            "exit_code_or_NONE": http_status,
            "termination_signal_or_NONE": None,
            "timed_out": failure_class is BrokerFailureClassV1.TIMEOUT,
            "monotonic_end_ns": time.monotonic_ns(),
            "latency_ms": latency_ms,
            "stdout_artifact_ref": "NONE",
            "stdout_sha256": empty_sha256,
            "stdout_size_bytes": 0,
            "stdout_truncated": False,
            "stderr_artifact_ref": (
                f"sqlite:calls:{logical_call_id}:error_detail_json"
            ),
            "stderr_sha256": hashlib.sha256(error_bytes).hexdigest(),
            "stderr_size_bytes": len(error_bytes),
            "stderr_truncated": False,
            "provider_request_confirmation": (
                ProviderRequestConfirmationV1.RESPONSE_RECEIVED.value
                if http_status is not None
                else ProviderRequestConfirmationV1.UNKNOWN.value
            ),
            "returned_model_or_NONE": returned_model,
        }
        closure_receipt = BrokerProcessExitReceiptV2(
            **closure_receipt_payload,
            receipt_digest=sha256_digest(closure_receipt_payload),
        )
        outcome_payload = {
            "logical_call_id": logical_call_id,
            "proposal_generation_session_id": (
                proposal_generation_session_id
            ),
            "request_envelope_digest": request_digest,
            "receipt_digest": closure_receipt.receipt_digest,
            "status": "PROCESS_FAILURE",
            "failure_class": failure_class.value,
            "classifier_rule_id": f"LAB_API_{error_type}_V1",
            "supporting_artifact_ref": (
                f"sqlite:calls:{logical_call_id}:error_detail_json"
            ),
            "redacted_excerpt": None,
            "response_digest": None,
        }
        outcome = BrokerCallOutcomeV2(
            **outcome_payload,
            outcome_digest=sha256_digest(outcome_payload),
        )
        self._connection.execute(
            """
            INSERT INTO calls(
                logical_call_id, request_digest, response_digest,
                response_json, input_tokens, cached_input_tokens,
                output_tokens, total_tokens, latency_ms, returned_model,
                status, error_type, error_detail_json,
                proposal_generation_session_id, receipt_digest,
                receipt_json, closure_receipt_json, outcome_json,
                broker_release_digest
            ) VALUES (
                ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, 'FAILED',
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                logical_call_id,
                request_digest,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                total_tokens,
                latency_ms,
                returned_model,
                error_type,
                error_detail_json,
                proposal_generation_session_id,
                receipt.receipt_digest,
                canonical_json_bytes(receipt.to_dict()).decode("utf-8"),
                canonical_json_bytes(
                    closure_receipt.to_dict()
                ).decode("utf-8"),
                canonical_json_bytes(outcome.to_dict()).decode("utf-8"),
                self.release.release_digest,
            ),
        )
        self._connection.commit()
        return closure_receipt, outcome

    @staticmethod
    def _failure_class(
        *,
        error_type: str,
        http_status: int | None,
    ) -> BrokerFailureClassV1:
        if error_type == "SCHEMA_VALIDATION_FAILURE":
            return BrokerFailureClassV1.SCHEMA_VALIDATION_FAILURE
        if error_type == "RESPONSE_CONTRACT_ERROR":
            return BrokerFailureClassV1.CLI_CONTRACT_ERROR
        if error_type == "TIMEOUT":
            return BrokerFailureClassV1.TIMEOUT
        if error_type == "TRANSPORT_ERROR":
            return BrokerFailureClassV1.CONNECTIVITY_ERROR
        if http_status in {401, 403}:
            return BrokerFailureClassV1.AUTHENTICATION_ERROR
        if http_status == 404:
            return BrokerFailureClassV1.MODEL_UNAVAILABLE
        if http_status is not None and http_status >= 500:
            return BrokerFailureClassV1.PROVIDER_ERROR
        return BrokerFailureClassV1.CLI_CONTRACT_ERROR


__all__ = [
    "LabApiBrokerReleaseV1",
    "LabApiCallReceiptV1",
    "LabApiCanaryBrokerV1",
    "LabApiResponseContractError",
    "LabApiResponseFailureReasonV1",
    "load_lab_api_credential_pairs",
    "load_lab_api_credentials",
]
