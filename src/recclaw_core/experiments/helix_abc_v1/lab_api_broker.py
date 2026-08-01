"""Single-request laboratory API broker for Pilot and Main campaigns."""

from __future__ import annotations

import ast
import hashlib
import json
import socket
import sqlite3
import ssl
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


class LabApiResponseFailureReasonV1(str, Enum):
    """Allowlisted, content-free HTTP-200 response failure reasons."""

    ENVELOPE_JSON_DECODE = "ENVELOPE_JSON_DECODE"
    ENVELOPE_SHAPE = "ENVELOPE_SHAPE"
    CHOICES_SHAPE = "CHOICES_SHAPE"
    CHOICE_SHAPE = "CHOICE_SHAPE"
    MESSAGE_SHAPE = "MESSAGE_SHAPE"
    MESSAGE_CONTENT_TYPE_OR_EMPTY = "MESSAGE_CONTENT_TYPE_OR_EMPTY"
    CONTENT_JSON_DECODE = "CONTENT_JSON_DECODE"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"
    PROPOSAL_COUNT = "PROPOSAL_COUNT"
    USAGE_SHAPE = "USAGE_SHAPE"
    TOKEN_USAGE_TYPE = "TOKEN_USAGE_TYPE"
    TOKEN_CEILING = "TOKEN_CEILING"
    RETURNED_MODEL_TYPE_OR_EMPTY = "RETURNED_MODEL_TYPE_OR_EMPTY"


class LabApiResponseContractError(ValueError):
    """Typed response failure whose public value is an allowlisted code only."""

    def __init__(self, reason: LabApiResponseFailureReasonV1) -> None:
        self.reason = reason
        super().__init__(reason.value)


def validate_provider_strict_schema(
    schema: Mapping[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> None:
    """Validate the strict object rule required by the laboratory Provider."""

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
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
    ) -> "LabApiBrokerReleaseV1":
        payload = {
            "transport": "HTTPS_CHAT_COMPLETIONS_V1",
            "endpoint_digest": sha256_digest(
                {"base_url": base_url.rstrip("/")}
            ),
            "model": model,
            "response_schema_digest": response_schema_digest,
            "request_mode": "SINGLE_JSON_SCHEMA_NO_TOOLS",
            "temperature": 0.0,
            "max_total_tokens_per_call": max_total_tokens_per_call,
            "timeout_ms": timeout_ms,
            "retry_count": 0,
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
            self.transport != "HTTPS_CHAT_COMPLETIONS_V1"
            or self.request_mode != "SINGLE_JSON_SCHEMA_NO_TOOLS"
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


def load_lab_api_credentials(config_path: Path) -> tuple[str, str]:
    """Read the external credential source without persisting its secret."""

    values: dict[str, str] = {}
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
                values[name.strip()] = parsed
    api_key = str(values.get("api_key", "")).strip()
    base_url = str(values.get("base_url", "")).strip().rstrip("/")
    if not api_key or not base_url:
        raise CanaryBrokerError(
            "laboratory API config requires api_key and base_url"
        )
    if not base_url.startswith("https://"):
        raise CanaryBrokerError("laboratory API base_url must use HTTPS")
    return base_url, api_key


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
        timeout_ms: int = 900_000,
        release_manifest_path: Path | None = None,
    ) -> None:
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
        self.base_url, self._api_key = load_lab_api_credentials(config_path)
        self.model = model
        self.max_total_tokens_per_call = max_total_tokens_per_call
        self.timeout_ms = timeout_ms
        computed_release = LabApiBrokerReleaseV1.create(
            base_url=self.base_url,
            model=model,
            response_schema_digest=self.schema_file_sha256,
            max_total_tokens_per_call=max_total_tokens_per_call,
            timeout_ms=timeout_ms,
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
    ) -> CanaryBrokerCallV1:
        effective_ceiling = int(
            max_total_tokens
            if max_total_tokens is not None
            else self.max_total_tokens_per_call
        )
        if not 1 <= effective_ceiling <= self.max_total_tokens_per_call:
            raise CanaryBrokerError(
                "per-call token ceiling is outside the API release"
            )
        request_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": effective_ceiling,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "recclaw_campaign_proposals",
                    "strict": True,
                    "schema": self.schema,
                },
            },
        }
        request_identity = {
            "expected_proposal_count": expected_proposal_count,
            "proposal_generation_session_id": (
                proposal_generation_session_id
            ),
            "release_digest": self.release.release_digest,
            "request_payload": request_payload,
        }
        request_digest = sha256_digest(request_identity)
        prior = self._stored(
            logical_call_id,
            request_digest,
            proposal_generation_session_id,
        )
        if prior is not None:
            return prior

        request = urlrequest.Request(
            f"{self.base_url}/chat/completions",
            data=canonical_json_bytes(request_payload),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        http_status: int | None = None
        try:
            context = ssl.create_default_context()
            ignore_eof = getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0)
            if ignore_eof:
                context.options |= ignore_eof
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
            )
        except urlerror.HTTPError as error:
            http_status = int(error.code)
            latency_ms = int((time.monotonic() - started) * 1000)
            error.read(16_384)
            receipt, outcome = self._persist_failure(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=(
                    proposal_generation_session_id
                ),
                request_digest=request_digest,
                error_type=f"HTTP_{error.code}",
                error_detail={"http_status": http_status},
                http_status=http_status,
                latency_ms=latency_ms,
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
            receipt, outcome = self._persist_failure(
                logical_call_id=logical_call_id,
                proposal_generation_session_id=(
                    proposal_generation_session_id
                ),
                request_digest=request_digest,
                error_type="RESPONSE_CONTRACT_ERROR",
                error_detail={"reason_code": error.reason.value},
                http_status=http_status,
                latency_ms=latency_ms,
            )
            raise CanaryBrokerError(
                "laboratory API response violated the frozen contract",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                wall_time_ms=latency_ms,
            ) from error
        except jsonschema.ValidationError as error:
            latency_ms = int((time.monotonic() - started) * 1000)
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
            )
            raise CanaryBrokerError(
                "laboratory API payload failed schema validation",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                wall_time_ms=latency_ms,
            ) from error

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
    ) -> tuple[Mapping[str, Any], dict[str, int], str]:
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.CHOICES_SHAPE
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.CHOICE_SHAPE
            )
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
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.MESSAGE_CONTENT_TYPE_OR_EMPTY
            )
        try:
            response = json.loads(content)
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
            cached_input_tokens = (
                int(prompt_details.get("cached_tokens", 0))
                if isinstance(prompt_details, Mapping)
                else int(raw_usage.get("cached_input_tokens", 0))
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.TOKEN_USAGE_TYPE
            ) from error
        if total_tokens < 1 or total_tokens > effective_ceiling:
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.TOKEN_CEILING
            )
        usage = {
            "cached_input_tokens": cached_input_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        returned_model = envelope.get("model")
        if not isinstance(returned_model, str) or not returned_model.strip():
            raise LabApiResponseContractError(
                LabApiResponseFailureReasonV1.RETURNED_MODEL_TYPE_OR_EMPTY
            )
        return response, usage, returned_model

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
    ) -> tuple[BrokerProcessExitReceiptV2, BrokerCallOutcomeV2]:
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
            "returned_model_or_NONE": None,
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
                ?, ?, NULL, NULL, 0, 0, 0, 0, ?, NULL, 'FAILED',
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                logical_call_id,
                request_digest,
                latency_ms,
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
    "load_lab_api_credentials",
]
