"""Content-bound Broker process execution and diagnostic records for M6F."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .canonical import bytes_sha256, canonical_json_bytes, sha256_digest


CAPTURE_LIMIT_BYTES = 1_048_576
CAPTURE_TRUNCATION_MARKER = b"\n[RECCLAW_BROKER_CAPTURE_TRUNCATED_V1]\n"
REDACTED_EXCERPT_LIMIT_BYTES = 2_048

_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "LANG",
    "NO_COLOR",
    "PATH",
    "TEMP",
    "TMP",
    "WSLENV",
    "WSL_DISTRO_NAME",
    "WSL_INTEROP",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
)
_REDACTION_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)(\b)sk-[a-z0-9_-]{8,}\b"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+"),
)
_CLASSIFIER_RULES = (
    (
        "AUTHENTICATION_ERROR",
        "M6F_CLASSIFIER_AUTH_EXACT_V1",
        ("unauthorized", "authentication failed", "login required", "http 401"),
    ),
    (
        "CONNECTIVITY_ERROR",
        "M6F_CLASSIFIER_CONNECTIVITY_EXACT_V1",
        (
            "connection refused",
            "failed to connect",
            "network is unreachable",
            "dns resolution failed",
        ),
    ),
    (
        "MODEL_UNAVAILABLE",
        "M6F_CLASSIFIER_MODEL_EXACT_V1",
        ("model not found", "model is unavailable", "unsupported model"),
    ),
    (
        "PROVIDER_ERROR",
        "M6F_CLASSIFIER_PROVIDER_EXACT_V1",
        ("provider error", "http 500", "http 502", "http 503"),
    ),
    (
        "CLI_CONTRACT_ERROR",
        "M6F_CLASSIFIER_CLI_EXACT_V1",
        (
            "unknown argument",
            "unexpected argument",
            "unrecognized option",
            "invalid value for",
        ),
    ),
)


class BrokerFailureClassV1(str, Enum):
    SPAWN_FAILURE = "SPAWN_FAILURE"
    PROCESS_EXIT_FAILURE = "PROCESS_EXIT_FAILURE"
    TIMEOUT = "TIMEOUT"
    CLI_CONTRACT_ERROR = "CLI_CONTRACT_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    CONNECTIVITY_ERROR = "CONNECTIVITY_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SCHEMA_VALIDATION_FAILURE = "SCHEMA_VALIDATION_FAILURE"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    MALFORMED_EVENT_STREAM = "MALFORMED_EVENT_STREAM"
    UNKNOWN_PROCESS_FAILURE = "UNKNOWN_PROCESS_FAILURE"
    SUCCESS = "SUCCESS"


class ProviderRequestConfirmationV1(str, Enum):
    NOT_OBSERVED = "NOT_OBSERVED"
    CLI_SESSION_STARTED = "CLI_SESSION_STARTED"
    PROVIDER_REQUEST_CONFIRMED = "PROVIDER_REQUEST_CONFIRMED"
    RESPONSE_RECEIVED = "RESPONSE_RECEIVED"
    UNKNOWN = "UNKNOWN"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_durable_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _digest_record(value: Mapping[str, Any], digest_field: str) -> str:
    projection = dict(value)
    projection.pop(digest_field, None)
    return sha256_digest(projection)


@dataclass(frozen=True, slots=True)
class BrokerProcessReleaseV2:
    broker_release_id: str
    broker_executable_path: str
    broker_executable_sha256: str
    broker_cli_version: str
    argv_template_digest: str
    working_directory_policy_digest: str
    environment_allowlist_policy_digest: str
    login_mode: str
    model: str
    reasoning_effort: str
    sandbox_mode: str
    response_schema_digest: str
    request_encoder_digest: str
    response_parser_digest: str
    process_capture_policy_digest: str
    timeout_policy_digest: str
    error_classifier_digest: str
    redaction_policy_digest: str
    release_digest: str

    @classmethod
    def create(
        cls,
        *,
        executable: Path,
        cli_version: str,
        login_mode: str,
        model: str,
        reasoning_effort: str,
        sandbox_mode: str,
        response_schema_digest: str,
        timeout_ms: int,
        workspace: Path,
    ) -> "BrokerProcessReleaseV2":
        executable = executable.resolve()
        if not executable.is_file():
            raise FileNotFoundError("Broker executable is unavailable")
        payload = {
            "argv_template_digest": sha256_digest(
                {
                    "argv": [
                        "{executable}",
                        "exec",
                        "--ephemeral",
                        "--ignore-user-config",
                        "--ignore-rules",
                        "--skip-git-repo-check",
                        "--sandbox",
                        sandbox_mode,
                        "--json",
                        "--color",
                        "never",
                        "-m",
                        model,
                        "-c",
                        'model_reasoning_effort="{reasoning_effort}"',
                        "-c",
                        'service_tier="{service_tier}"',
                        "-C",
                        "{workspace}",
                        "--output-schema",
                        "{response_schema}",
                        "--output-last-message",
                        "{response_output}",
                        "-",
                    ]
                }
            ),
            "broker_cli_version": cli_version,
            "broker_executable_path": str(executable),
            "broker_executable_sha256": _file_sha256(executable),
            "broker_release_id": "RECCLAW_BROKER_PROCESS_RELEASE_V2",
            "environment_allowlist_policy_digest": sha256_digest(
                {
                    "allowlist": list(_ENVIRONMENT_ALLOWLIST),
                    "forced": {"NO_COLOR": "1"},
                    "policy": "EXACT_ALLOWLIST_V1",
                }
            ),
            "error_classifier_digest": sha256_digest(
                {
                    "closed_domain": [item.value for item in BrokerFailureClassV1],
                    "rules": [
                        {
                            "failure_class": failure_class,
                            "phrases": list(phrases),
                            "rule_id": rule_id,
                        }
                        for failure_class, rule_id, phrases in _CLASSIFIER_RULES
                    ],
                    "policy": "EVIDENCE_RULES_V1",
                }
            ),
            "login_mode": login_mode,
            "model": model,
            "process_capture_policy_digest": sha256_digest(
                {
                    "limit_bytes_per_stream": CAPTURE_LIMIT_BYTES,
                    "marker_sha256": bytes_sha256(CAPTURE_TRUNCATION_MARKER),
                    "separate_binary_streams": True,
                    "version": 1,
                }
            ),
            "reasoning_effort": reasoning_effort,
            "redaction_policy_digest": sha256_digest(
                {
                    "excerpt_limit_bytes": REDACTED_EXCERPT_LIMIT_BYTES,
                    "pattern_count": len(_REDACTION_PATTERNS),
                    "patterns": [
                        pattern.pattern for pattern in _REDACTION_PATTERNS
                    ],
                    "replacement": "[REDACTED]",
                    "version": 1,
                }
            ),
            "request_encoder_digest": sha256_digest(
                {"encoding": "UTF-8", "transport": "STDIN", "version": 2}
            ),
            "response_parser_digest": sha256_digest(
                {
                    "event_stream": "JSONL_UTF8_STRICT",
                    "response": "JSON_SCHEMA",
                    "version": 2,
                }
            ),
            "response_schema_digest": response_schema_digest,
            "sandbox_mode": sandbox_mode,
            "timeout_policy_digest": sha256_digest(
                {"kill_on_timeout": True, "timeout_ms": timeout_ms, "version": 1}
            ),
            "working_directory_policy_digest": sha256_digest(
                {"exact_resolved_workspace": str(workspace.resolve()), "version": 1}
            ),
        }
        return cls(**payload, release_digest=sha256_digest(payload))

    def verify(self) -> None:
        payload = self.to_dict()
        release_digest = payload.pop("release_digest")
        if sha256_digest(payload) != release_digest:
            raise ValueError("Broker release digest mismatch")
        if _file_sha256(Path(self.broker_executable_path)) != self.broker_executable_sha256:
            raise ValueError("Broker executable bytes do not match the release")

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BrokerRequestEnvelopeV2:
    logical_call_id: str
    proposal_generation_session_id: str
    request_digest: str
    request_size_bytes: int
    response_schema_digest: str
    model: str
    reasoning_effort: str
    sandbox_mode: str
    argv_digest: str
    cwd_digest: str
    environment_projection_digest: str
    timeout_ms: int
    broker_release_digest: str
    envelope_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BrokerProcessStartRecordV1:
    start_record_id: str
    request_envelope_digest: str
    spawn_attempt_ordinal: int
    monotonic_start_ns: int
    wall_clock_start: str
    process_contract_digest: str
    start_record_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BrokerProcessExitReceiptV2:
    start_record_digest: str
    spawn_succeeded: bool
    pid_or_private_process_ref: str | None
    exit_code_or_NONE: int | None
    termination_signal_or_NONE: int | None
    timed_out: bool
    monotonic_end_ns: int
    latency_ms: int
    stdout_artifact_ref: str
    stdout_sha256: str
    stdout_size_bytes: int
    stdout_truncated: bool
    stderr_artifact_ref: str
    stderr_sha256: str
    stderr_size_bytes: int
    stderr_truncated: bool
    provider_request_confirmation: str
    returned_model_or_NONE: str | None
    receipt_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BrokerCallOutcomeV2:
    logical_call_id: str
    proposal_generation_session_id: str
    request_envelope_digest: str
    receipt_digest: str
    status: str
    failure_class: str
    classifier_rule_id: str
    supporting_artifact_ref: str | None
    redacted_excerpt: str | None
    response_digest: str | None
    outcome_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BrokerConformanceReportV1:
    probe_id: str
    probe_shape: str
    treatment_free: bool
    search_round_opened: bool
    broker_release_digest: str
    request_envelope_digest: str
    receipt_digest: str
    outcome_digest: str
    actual_total_tokens: int
    verdict: str
    report_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CapturedProcessV2:
    request: BrokerRequestEnvelopeV2
    start: BrokerProcessStartRecordV1
    receipt: BrokerProcessExitReceiptV2
    outcome: BrokerCallOutcomeV2
    events: tuple[Mapping[str, Any], ...]
    response_bytes: bytes | None


class _BoundedCapture:
    def __init__(self, path: Path, limit: int) -> None:
        self.path = path
        self.limit = limit
        self.total_bytes = 0
        self.truncated = False
        self.error: BaseException | None = None

    def consume(self, stream: Any) -> None:
        try:
            with self.path.open("r+b", buffering=0) as handle:
                while True:
                    block = stream.read(64 * 1024)
                    if not block:
                        break
                    self.total_bytes += len(block)
                    remaining = max(0, self.limit - handle.tell())
                    if remaining:
                        handle.write(block[:remaining])
                    if len(block) > remaining:
                        self.truncated = True
                if self.truncated:
                    handle.write(CAPTURE_TRUNCATION_MARKER)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException as error:  # captured and re-raised by the owner thread
            self.error = error


def environment_projection() -> tuple[dict[str, str], str]:
    projected = {
        key: os.environ[key]
        for key in _ENVIRONMENT_ALLOWLIST
        if key in os.environ
    }
    projected["NO_COLOR"] = "1"
    digest_projection = {
        key: sha256_digest({"value": value})
        for key, value in sorted(projected.items())
    }
    return projected, sha256_digest(digest_projection)


def redact_excerpt(payload: bytes) -> str | None:
    if not payload:
        return None
    text = payload[:REDACTED_EXCERPT_LIMIT_BYTES].decode("utf-8", errors="replace")
    for pattern in _REDACTION_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text


def _event_projection(payload: bytes) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    events: list[Mapping[str, Any]] = []
    malformed = False
    if not payload:
        return (), False
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return (), True
    for line in text.splitlines():
        if not line.strip() or line.startswith("[RECCLAW_BROKER_CAPTURE_TRUNCATED"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed = True
            continue
        if not isinstance(event, Mapping):
            malformed = True
            continue
        events.append(event)
    return tuple(events), malformed


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("type") or event.get("event") or "")


def _provider_confirmation(
    events: Sequence[Mapping[str, Any]], response_present: bool
) -> ProviderRequestConfirmationV1:
    if response_present:
        return ProviderRequestConfirmationV1.RESPONSE_RECEIVED
    types = {_event_type(event).lower() for event in events}
    if any("turn.completed" in item or "response" in item for item in types):
        return ProviderRequestConfirmationV1.PROVIDER_REQUEST_CONFIRMED
    if any("thread.started" in item or "turn.started" in item for item in types):
        return ProviderRequestConfirmationV1.CLI_SESSION_STARTED
    return ProviderRequestConfirmationV1.NOT_OBSERVED


def _returned_model(events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in reversed(events):
        for key in ("returned_model", "model"):
            value = event.get(key)
            if value:
                return str(value)
    return None


def _specific_process_rule(
    stderr: bytes, events: Sequence[Mapping[str, Any]]
) -> tuple[BrokerFailureClassV1, str, str | None] | None:
    structured_text = canonical_json_bytes(list(events)).decode("utf-8", errors="replace")
    text = stderr.decode("utf-8", errors="replace")
    evidence = f"{structured_text}\n{text}".lower()
    for failure_class, rule_id, phrases in _CLASSIFIER_RULES:
        if any(phrase in evidence for phrase in phrases):
            return BrokerFailureClassV1(failure_class), rule_id, "stderr"
    return None


def classify_process_outcome(
    *,
    logical_call_id: str,
    proposal_generation_session_id: str,
    request_envelope_digest: str,
    receipt: BrokerProcessExitReceiptV2,
    stdout: bytes,
    stderr: bytes,
    events: Sequence[Mapping[str, Any]],
    malformed_events: bool,
    response_bytes: bytes | None,
    response_schema: Mapping[str, Any],
) -> BrokerCallOutcomeV2:
    failure_class: BrokerFailureClassV1
    rule_id: str
    supporting_ref: str | None = None
    excerpt: str | None = None
    response_digest: str | None = None
    if not receipt.spawn_succeeded:
        failure_class = BrokerFailureClassV1.SPAWN_FAILURE
        rule_id = "M6F_CLASSIFIER_SPAWN_V1"
    elif receipt.timed_out:
        failure_class = BrokerFailureClassV1.TIMEOUT
        rule_id = "M6F_CLASSIFIER_TIMEOUT_V1"
        supporting_ref = receipt.stderr_artifact_ref
        excerpt = redact_excerpt(stderr)
    elif receipt.exit_code_or_NONE not in (0, None):
        specific = _specific_process_rule(stderr, events)
        if specific is None:
            failure_class = BrokerFailureClassV1.PROCESS_EXIT_FAILURE
            rule_id = "M6F_CLASSIFIER_EXIT_CODE_ONLY_V1"
            supporting_ref = receipt.stderr_artifact_ref if stderr else receipt.stdout_artifact_ref
            excerpt = redact_excerpt(stderr or stdout)
        else:
            failure_class, rule_id, stream = specific
            supporting_ref = (
                receipt.stderr_artifact_ref
                if stream == "stderr"
                else receipt.stdout_artifact_ref
            )
            excerpt = redact_excerpt(stderr if stream == "stderr" else stdout)
    elif response_bytes is None:
        if malformed_events:
            failure_class = BrokerFailureClassV1.MALFORMED_EVENT_STREAM
            rule_id = "M6F_CLASSIFIER_MALFORMED_EVENT_V1"
            supporting_ref = receipt.stdout_artifact_ref
            excerpt = redact_excerpt(stdout)
        else:
            failure_class = BrokerFailureClassV1.EMPTY_RESPONSE
            rule_id = "M6F_CLASSIFIER_EMPTY_RESPONSE_V1"
            supporting_ref = receipt.stdout_artifact_ref
    else:
        try:
            response = json.loads(response_bytes)
            jsonschema.validate(response, response_schema)
        except (json.JSONDecodeError, UnicodeDecodeError, jsonschema.ValidationError):
            failure_class = BrokerFailureClassV1.SCHEMA_VALIDATION_FAILURE
            rule_id = "M6F_CLASSIFIER_RESPONSE_SCHEMA_V1"
        else:
            if malformed_events:
                failure_class = BrokerFailureClassV1.MALFORMED_EVENT_STREAM
                rule_id = "M6F_CLASSIFIER_MALFORMED_EVENT_V1"
                supporting_ref = receipt.stdout_artifact_ref
                excerpt = redact_excerpt(stdout)
            else:
                failure_class = BrokerFailureClassV1.SUCCESS
                rule_id = "M6F_CLASSIFIER_SUCCESS_V1"
                response_digest = sha256_digest(response)
    payload = {
        "classifier_rule_id": rule_id,
        "failure_class": failure_class.value,
        "logical_call_id": logical_call_id,
        "proposal_generation_session_id": proposal_generation_session_id,
        "receipt_digest": receipt.receipt_digest,
        "redacted_excerpt": excerpt,
        "request_envelope_digest": request_envelope_digest,
        "response_digest": response_digest,
        "status": (
            "SUCCESS"
            if failure_class is BrokerFailureClassV1.SUCCESS
            else "PROCESS_FAILURE"
        ),
        "supporting_artifact_ref": supporting_ref,
    }
    return BrokerCallOutcomeV2(**payload, outcome_digest=sha256_digest(payload))


class BrokerProcessRunnerV2:
    """Run one exact Broker process attempt and durably capture its evidence."""

    def __init__(
        self,
        *,
        private_root: Path,
        release: BrokerProcessReleaseV2,
        response_schema: Mapping[str, Any],
        timeout_ms: int,
    ) -> None:
        self.private_root = private_root.resolve()
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.release = release
        self.release.verify()
        self.response_schema = dict(response_schema)
        self.timeout_ms = timeout_ms

    def _recover_existing(
        self,
        *,
        call_root: Path,
        request: BrokerRequestEnvelopeV2,
        response_output: Path,
    ) -> CapturedProcessV2:
        stored_request = BrokerRequestEnvelopeV2(
            **json.loads((call_root / "request_envelope.json").read_text("utf-8"))
        )
        if stored_request.envelope_digest != request.envelope_digest:
            raise ValueError("Broker request envelope replay conflicts")
        start_path = call_root / "start_record.json"
        if not start_path.is_file():
            raise RuntimeError("Broker replay found no durable start record")
        start = BrokerProcessStartRecordV1(
            **json.loads(start_path.read_text("utf-8"))
        )
        stdout_path = call_root / "stdout.bin"
        stderr_path = call_root / "stderr.bin"
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
        events, malformed = _event_projection(stdout)
        response_bytes = response_output.read_bytes() if response_output.is_file() else None
        receipt_path = call_root / "exit_receipt.json"
        if receipt_path.is_file():
            receipt = BrokerProcessExitReceiptV2(
                **json.loads(receipt_path.read_text("utf-8"))
            )
        else:
            spawn_path = call_root / "spawn_record.json"
            spawn = (
                json.loads(spawn_path.read_text("utf-8"))
                if spawn_path.is_file()
                else {}
            )
            now_ns = time.monotonic_ns()
            receipt_payload = {
                "exit_code_or_NONE": None,
                "latency_ms": max(
                    0, (now_ns - start.monotonic_start_ns) // 1_000_000
                ),
                "monotonic_end_ns": now_ns,
                "pid_or_private_process_ref": spawn.get(
                    "pid_or_private_process_ref"
                ),
                "provider_request_confirmation": (
                    ProviderRequestConfirmationV1.UNKNOWN.value
                    if spawn
                    else ProviderRequestConfirmationV1.NOT_OBSERVED.value
                ),
                "returned_model_or_NONE": _returned_model(events),
                "spawn_succeeded": bool(spawn),
                "start_record_digest": start.start_record_digest,
                "stderr_artifact_ref": (
                    f"broker-private:calls/{request.envelope_digest}/stderr.bin"
                ),
                "stderr_sha256": bytes_sha256(stderr),
                "stderr_size_bytes": len(stderr),
                "stderr_truncated": (
                    CAPTURE_TRUNCATION_MARKER in stderr
                ),
                "stdout_artifact_ref": (
                    f"broker-private:calls/{request.envelope_digest}/stdout.bin"
                ),
                "stdout_sha256": bytes_sha256(stdout),
                "stdout_size_bytes": len(stdout),
                "stdout_truncated": (
                    CAPTURE_TRUNCATION_MARKER in stdout
                ),
                "termination_signal_or_NONE": None,
                "timed_out": False,
            }
            receipt = BrokerProcessExitReceiptV2(
                **receipt_payload, receipt_digest=sha256_digest(receipt_payload)
            )
            _write_durable_json(receipt_path, receipt.to_dict())
        outcome_path = call_root / "outcome.json"
        if outcome_path.is_file():
            outcome = BrokerCallOutcomeV2(
                **json.loads(outcome_path.read_text("utf-8"))
            )
        elif receipt.exit_code_or_NONE is None:
            payload = {
                "classifier_rule_id": (
                    "M6F_RECOVERY_SPAWN_WITHOUT_RECEIPT_V1"
                    if receipt.spawn_succeeded
                    else "M6F_RECOVERY_START_BEFORE_SPAWN_V1"
                ),
                "failure_class": BrokerFailureClassV1.UNKNOWN_PROCESS_FAILURE.value,
                "logical_call_id": request.logical_call_id,
                "proposal_generation_session_id": (
                    request.proposal_generation_session_id
                ),
                "receipt_digest": receipt.receipt_digest,
                "redacted_excerpt": None,
                "request_envelope_digest": request.envelope_digest,
                "response_digest": None,
                "status": "PROCESS_FAILURE",
                "supporting_artifact_ref": None,
            }
            outcome = BrokerCallOutcomeV2(
                **payload, outcome_digest=sha256_digest(payload)
            )
            _write_durable_json(outcome_path, outcome.to_dict())
        else:
            outcome = classify_process_outcome(
                logical_call_id=request.logical_call_id,
                proposal_generation_session_id=(
                    request.proposal_generation_session_id
                ),
                request_envelope_digest=request.envelope_digest,
                receipt=receipt,
                stdout=stdout,
                stderr=stderr,
                events=events,
                malformed_events=malformed,
                response_bytes=response_bytes,
                response_schema=self.response_schema,
            )
            _write_durable_json(outcome_path, outcome.to_dict())
        return CapturedProcessV2(
            request=request,
            start=start,
            receipt=receipt,
            outcome=outcome,
            events=events,
            response_bytes=response_bytes,
        )

    def execute(
        self,
        *,
        logical_call_id: str,
        proposal_generation_session_id: str,
        prompt: str,
        argv: Sequence[str],
        cwd: Path,
        response_output: Path,
    ) -> CapturedProcessV2:
        self.release.verify()
        if (
            not argv
            or Path(argv[0]).resolve()
            != Path(self.release.broker_executable_path).resolve()
        ):
            raise ValueError("Broker argv executable differs from the bound release")
        environment, environment_digest = environment_projection()
        request_bytes = prompt.encode("utf-8")
        request_payload = {
            "argv_digest": sha256_digest(list(argv)),
            "broker_release_digest": self.release.release_digest,
            "cwd_digest": sha256_digest({"cwd": str(cwd.resolve())}),
            "environment_projection_digest": environment_digest,
            "logical_call_id": logical_call_id,
            "model": self.release.model,
            "proposal_generation_session_id": proposal_generation_session_id,
            "reasoning_effort": self.release.reasoning_effort,
            "request_digest": bytes_sha256(request_bytes),
            "request_size_bytes": len(request_bytes),
            "response_schema_digest": self.release.response_schema_digest,
            "sandbox_mode": self.release.sandbox_mode,
            "timeout_ms": self.timeout_ms,
        }
        request = BrokerRequestEnvelopeV2(
            **request_payload, envelope_digest=sha256_digest(request_payload)
        )
        call_root = self.private_root / "calls" / request.envelope_digest
        if call_root.exists():
            return self._recover_existing(
                call_root=call_root,
                request=request,
                response_output=response_output,
            )
        call_root.mkdir(parents=True, exist_ok=True)
        _write_durable_json(call_root / "request_envelope.json", request.to_dict())
        stdout_path = call_root / "stdout.bin"
        stderr_path = call_root / "stderr.bin"
        for path in (stdout_path, stderr_path):
            with path.open("xb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        start_ns = time.monotonic_ns()
        start_payload = {
            "monotonic_start_ns": start_ns,
            "process_contract_digest": sha256_digest(
                {
                    "argv_digest": request.argv_digest,
                    "cwd_digest": request.cwd_digest,
                    "environment_projection_digest": request.environment_projection_digest,
                    "release_digest": self.release.release_digest,
                    "timeout_ms": self.timeout_ms,
                }
            ),
            "request_envelope_digest": request.envelope_digest,
            "spawn_attempt_ordinal": 1,
            "start_record_id": f"broker-start:{request.envelope_digest}",
            "wall_clock_start": _utc_now(),
        }
        start = BrokerProcessStartRecordV1(
            **start_payload, start_record_digest=sha256_digest(start_payload)
        )
        _write_durable_json(call_root / "start_record.json", start.to_dict())

        process: subprocess.Popen[bytes] | None = None
        timed_out = False
        spawn_error: OSError | None = None
        stdout_capture = _BoundedCapture(stdout_path, CAPTURE_LIMIT_BYTES)
        stderr_capture = _BoundedCapture(stderr_path, CAPTURE_LIMIT_BYTES)
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
            _write_durable_json(
                call_root / "spawn_record.json",
                {
                    "pid_or_private_process_ref": (
                        f"private-process:{request.envelope_digest}:{process.pid}"
                    ),
                    "start_record_digest": start.start_record_digest,
                },
            )
            assert process.stdout is not None and process.stderr is not None
            stdout_thread = threading.Thread(
                target=stdout_capture.consume, args=(process.stdout,), daemon=True
            )
            stderr_thread = threading.Thread(
                target=stderr_capture.consume, args=(process.stderr,), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            assert process.stdin is not None
            process.stdin.write(request_bytes)
            process.stdin.close()
            try:
                process.wait(timeout=self.timeout_ms / 1000)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
        except OSError as error:
            spawn_error = error
            with stderr_path.open("ab") as handle:
                handle.write(str(error).encode("utf-8", errors="replace"))
                handle.flush()
                os.fsync(handle.fileno())
            stderr_capture.total_bytes = stderr_path.stat().st_size
        finally:
            for thread in (stdout_thread, stderr_thread):
                if thread is not None:
                    thread.join()
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            for capture in (stdout_capture, stderr_capture):
                if capture.error is not None:
                    raise RuntimeError("Broker stream capture failed") from capture.error

        end_ns = time.monotonic_ns()
        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
        events, malformed_events = _event_projection(stdout_bytes)
        response_bytes = response_output.read_bytes() if response_output.is_file() else None
        exit_code = process.returncode if process is not None else None
        termination_signal = (
            -exit_code
            if exit_code is not None and exit_code < 0 and -exit_code in signal.valid_signals()
            else None
        )
        receipt_payload = {
            "exit_code_or_NONE": exit_code,
            "latency_ms": max(0, (end_ns - start_ns) // 1_000_000),
            "monotonic_end_ns": end_ns,
            "pid_or_private_process_ref": (
                f"private-process:{request.envelope_digest}:{process.pid}"
                if process is not None
                else None
            ),
            "provider_request_confirmation": _provider_confirmation(
                events, response_bytes is not None
            ).value,
            "returned_model_or_NONE": _returned_model(events),
            "spawn_succeeded": process is not None and spawn_error is None,
            "start_record_digest": start.start_record_digest,
            "stderr_artifact_ref": (
                f"broker-private:calls/{request.envelope_digest}/stderr.bin"
            ),
            "stderr_sha256": bytes_sha256(stderr_bytes),
            "stderr_size_bytes": stderr_capture.total_bytes,
            "stderr_truncated": stderr_capture.truncated,
            "stdout_artifact_ref": (
                f"broker-private:calls/{request.envelope_digest}/stdout.bin"
            ),
            "stdout_sha256": bytes_sha256(stdout_bytes),
            "stdout_size_bytes": stdout_capture.total_bytes,
            "stdout_truncated": stdout_capture.truncated,
            "termination_signal_or_NONE": termination_signal,
            "timed_out": timed_out,
        }
        receipt = BrokerProcessExitReceiptV2(
            **receipt_payload, receipt_digest=sha256_digest(receipt_payload)
        )
        _write_durable_json(call_root / "exit_receipt.json", receipt.to_dict())
        outcome = classify_process_outcome(
            logical_call_id=logical_call_id,
            proposal_generation_session_id=proposal_generation_session_id,
            request_envelope_digest=request.envelope_digest,
            receipt=receipt,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            events=events,
            malformed_events=malformed_events,
            response_bytes=response_bytes,
            response_schema=self.response_schema,
        )
        _write_durable_json(call_root / "outcome.json", outcome.to_dict())
        return CapturedProcessV2(
            request=request,
            start=start,
            receipt=receipt,
            outcome=outcome,
            events=events,
            response_bytes=response_bytes,
        )


__all__ = [
    "BrokerCallOutcomeV2",
    "BrokerConformanceReportV1",
    "BrokerFailureClassV1",
    "BrokerProcessExitReceiptV2",
    "BrokerProcessReleaseV2",
    "BrokerProcessRunnerV2",
    "BrokerProcessStartRecordV1",
    "BrokerRequestEnvelopeV2",
    "CapturedProcessV2",
    "ProviderRequestConfirmationV1",
    "classify_process_outcome",
    "environment_projection",
    "redact_excerpt",
]
