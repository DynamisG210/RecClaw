from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib import error as urlerror

import pytest

from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerError,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiCanaryBrokerV1,
    LabApiResponseFailureReasonV1,
)


SECRET = "v5-never-persist-secret"
CONTENT_MARKER = "v5-never-persist-provider-content"
ENDPOINT_LITERAL = "https://v5-private-endpoint.invalid/v1"


class _RawResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_RawResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _schema(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["proposals"],
                "properties": {
                    "proposals": {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(path: Path) -> Path:
    path.write_text(
        f'api_key = "{SECRET}"\nbase_url = "{ENDPOINT_LITERAL}"\n',
        encoding="utf-8",
    )
    return path


def _valid_envelope() -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"proposals": [{"marker": CONTENT_MARKER}]}
                    )
                }
            }
        ],
        "model": "gpt-5.4",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _body(value: Any) -> bytes:
    return json.dumps(value).encode("utf-8")


def _case_envelope_shape() -> bytes:
    return _body([CONTENT_MARKER])


def _case_choices_shape() -> bytes:
    envelope = _valid_envelope()
    envelope["choices"] = []
    return _body(envelope)


def _case_choice_shape() -> bytes:
    envelope = _valid_envelope()
    envelope["choices"] = [CONTENT_MARKER]
    return _body(envelope)


def _case_message_shape() -> bytes:
    envelope = _valid_envelope()
    envelope["choices"] = [{"message": CONTENT_MARKER}]
    return _body(envelope)


def _case_content_type() -> bytes:
    envelope = _valid_envelope()
    envelope["choices"] = [{"message": {"content": {"raw": CONTENT_MARKER}}}]
    return _body(envelope)


def _case_content_json_decode() -> bytes:
    envelope = _valid_envelope()
    envelope["choices"] = [
        {"message": {"content": "not-json-" + CONTENT_MARKER}}
    ]
    return _body(envelope)


def _case_schema_validation() -> bytes:
    envelope = _valid_envelope()
    envelope["choices"] = [
        {
            "message": {
                "content": json.dumps(
                    {"proposals": CONTENT_MARKER}
                )
            }
        }
    ]
    return _body(envelope)


def _case_proposal_count() -> bytes:
    envelope = _valid_envelope()
    envelope["choices"] = [
        {"message": {"content": json.dumps({"proposals": []})}}
    ]
    return _body(envelope)


def _case_usage_shape() -> bytes:
    envelope = _valid_envelope()
    envelope["usage"] = CONTENT_MARKER
    return _body(envelope)


def _case_token_usage_type() -> bytes:
    envelope = _valid_envelope()
    envelope["usage"] = {
        "prompt_tokens": CONTENT_MARKER,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    return _body(envelope)


def _case_token_ceiling() -> bytes:
    envelope = _valid_envelope()
    envelope["usage"] = {
        "prompt_tokens": 10,
        "completion_tokens": 500,
        "total_tokens": 510,
    }
    return _body(envelope)


CASES = (
    (b"not-json-" + CONTENT_MARKER.encode(), "ENVELOPE_JSON_DECODE", "RESPONSE_CONTRACT_ERROR"),
    (_case_envelope_shape(), "ENVELOPE_SHAPE", "RESPONSE_CONTRACT_ERROR"),
    (_case_choices_shape(), "CHOICES_SHAPE", "RESPONSE_CONTRACT_ERROR"),
    (_case_choice_shape(), "CHOICE_SHAPE", "RESPONSE_CONTRACT_ERROR"),
    (_case_message_shape(), "MESSAGE_SHAPE", "RESPONSE_CONTRACT_ERROR"),
    (_case_content_type(), "MESSAGE_CONTENT_TYPE_OR_EMPTY", "RESPONSE_CONTRACT_ERROR"),
    (_case_content_json_decode(), "CONTENT_JSON_DECODE", "RESPONSE_CONTRACT_ERROR"),
    (_case_schema_validation(), "SCHEMA_VALIDATION", "SCHEMA_VALIDATION_FAILURE"),
    (_case_proposal_count(), "PROPOSAL_COUNT", "RESPONSE_CONTRACT_ERROR"),
    (_case_usage_shape(), "USAGE_SHAPE", "RESPONSE_CONTRACT_ERROR"),
    (_case_token_usage_type(), "TOKEN_USAGE_TYPE", "RESPONSE_CONTRACT_ERROR"),
    (_case_token_ceiling(), "TOKEN_CEILING", "RESPONSE_CONTRACT_ERROR"),
)


@pytest.mark.parametrize(("raw_body", "reason_code", "error_type"), CASES)
def test_every_http_200_parse_failure_persists_allowlisted_reason_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_body: bytes,
    reason_code: str,
    error_type: str,
) -> None:
    attempts = 0

    def fake_urlopen(*_: object, **__: object) -> _RawResponse:
        nonlocal attempts
        attempts += 1
        return _RawResponse(raw_body)

    monkeypatch.setattr(
        "recclaw_core.experiments.helix_abc_v1."
        "lab_api_broker.urlrequest.urlopen",
        fake_urlopen,
    )
    private_root = tmp_path / "private"
    broker = LabApiCanaryBrokerV1(
        private_root,
        schema_path=_schema(tmp_path / "schema.json"),
        config_path=_config(tmp_path / "llm_api.toml"),
        model="gpt-5.4",
        max_total_tokens_per_call=100,
        timeout_ms=10_000,
    )
    try:
        with pytest.raises(CanaryBrokerError):
            broker.call_with_session(
                logical_call_id="v5-observability-slot",
                proposal_generation_session_id="v5-observability-session",
                prompt="do-not-persist-prompt-" + CONTENT_MARKER,
                expected_proposal_count=1,
                max_total_tokens=100,
            )
        row = broker._connection.execute(
            "SELECT status, error_type, error_detail_json FROM calls"
        ).fetchone()
        assert tuple(row) == (
            "FAILED",
            error_type,
            json.dumps(
                {"reason_code": reason_code},
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        assert attempts == 1
        assert reason_code in {item.value for item in LabApiResponseFailureReasonV1}
    finally:
        broker.close()
    persisted = b"".join(
        artifact.read_bytes()
        for artifact in private_root.rglob("*")
        if artifact.is_file()
    )
    for forbidden in (
        SECRET.encode(),
        CONTENT_MARKER.encode(),
        ENDPOINT_LITERAL.encode(),
        raw_body,
    ):
        assert forbidden not in persisted


def test_http_error_body_is_consumed_but_never_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_body = b'{"error":"' + CONTENT_MARKER.encode() + b'"}'

    def fail_urlopen(request: object, **_: object) -> object:
        raise urlerror.HTTPError(
            getattr(request, "full_url"),
            503,
            "provider text must not persist",
            {},
            io.BytesIO(provider_body),
        )

    monkeypatch.setattr(
        "recclaw_core.experiments.helix_abc_v1."
        "lab_api_broker.urlrequest.urlopen",
        fail_urlopen,
    )
    private_root = tmp_path / "private"
    broker = LabApiCanaryBrokerV1(
        private_root,
        schema_path=_schema(tmp_path / "schema.json"),
        config_path=_config(tmp_path / "llm_api.toml"),
        model="gpt-5.4",
        max_total_tokens_per_call=100,
    )
    try:
        with pytest.raises(CanaryBrokerError):
            broker.call(
                logical_call_id="v5-http-error",
                prompt="prompt-" + CONTENT_MARKER,
                expected_proposal_count=1,
            )
        row = broker._connection.execute(
            "SELECT error_type, error_detail_json FROM calls"
        ).fetchone()
        assert tuple(row) == ("HTTP_503", '{"http_status":503}')
    finally:
        broker.close()
    persisted = b"".join(
        artifact.read_bytes()
        for artifact in private_root.rglob("*")
        if artifact.is_file()
    )
    assert provider_body not in persisted
    assert CONTENT_MARKER.encode() not in persisted
    assert SECRET.encode() not in persisted


def test_success_path_result_and_persisted_response_are_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _valid_envelope()
    monkeypatch.setattr(
        "recclaw_core.experiments.helix_abc_v1."
        "lab_api_broker.urlrequest.urlopen",
        lambda *_args, **_kwargs: _RawResponse(_body(envelope)),
    )
    broker = LabApiCanaryBrokerV1(
        tmp_path / "private",
        schema_path=_schema(tmp_path / "schema.json"),
        config_path=_config(tmp_path / "llm_api.toml"),
        model="gpt-5.4",
        max_total_tokens_per_call=100,
    )
    try:
        result = broker.call(
            logical_call_id="v5-success",
            prompt="Return one proposal.",
            expected_proposal_count=1,
        )
        assert result.response == {
            "proposals": [{"marker": CONTENT_MARKER}]
        }
        assert result.returned_model == "gpt-5.4"
        assert result.total_tokens == 15
        row = broker._connection.execute(
            "SELECT status, response_json, error_detail_json FROM calls"
        ).fetchone()
        assert tuple(row) == (
            "SUCCESS",
            '{"proposals":[{"marker":"v5-never-persist-provider-content"}]}',
            None,
        )
    finally:
        broker.close()
