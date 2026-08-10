from __future__ import annotations

import json
import socket
import sqlite3
from pathlib import Path
from urllib import error as urlerror

import pytest

from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerError,
)
from recclaw_core.experiments.helix_abc_v1.broker_failure_closure import (
    BrokerFailureClosureV1,
)
from recclaw_core.experiments.helix_abc_v1.audit_snapshot import (
    association_free_neutral_audit,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiCanaryBrokerV1,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.status = 200
        self._bytes = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._bytes


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
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["mechanism_id"],
                            "properties": {"mechanism_id": {"type": "string"}},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(path: Path, *, api_key: str = "private-test-key") -> Path:
    path.write_text(
        f'api_key = "{api_key}"\n'
        'base_url = "https://laboratory.invalid/v1"\n',
        encoding="utf-8",
    )
    return path


def test_single_schema_request_and_create_once_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, object]] = []

    def fake_urlopen(request: object, **_: object) -> _Response:
        captured.append(
            json.loads(getattr(request, "data").decode("utf-8"))
        )
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"proposals": [{"mechanism_id": "m1"}]}
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
        )

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
        first = broker.call_with_session(
            logical_call_id="call-1",
            proposal_generation_session_id="session-1",
            prompt="Return one proposal.",
            expected_proposal_count=1,
            max_total_tokens=50,
            max_output_tokens=8,
        )
        replay = broker.call_with_session(
            logical_call_id="call-1",
            proposal_generation_session_id="session-1",
            prompt="Return one proposal.",
            expected_proposal_count=1,
            max_total_tokens=50,
            max_output_tokens=8,
        )
        assert first == replay
        assert broker.call_count() == 1
        assert len(captured) == 1
        payload = captured[0]
        assert payload["model"] == "gpt-5.4"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 8
        assert payload["response_format"]["type"] == "json_schema"
        assert "tools" not in payload
        assert "functions" not in payload
        snapshot = broker.create_audit_snapshot(
            tmp_path / "audit.sqlite3",
            audit_purpose="TEST_LAB_API_BROKER",
        )
        assert snapshot.audit_purpose == "TEST_LAB_API_BROKER"
        assert (tmp_path / "audit.sqlite3").is_file()
        state_path = tmp_path / "state.sqlite3"
        state = sqlite3.connect(state_path)
        state.execute("CREATE TABLE execution_claims(claim_id TEXT)")
        state.execute(
            "CREATE TABLE resource_ledger(dimension TEXT, quantity INTEGER)"
        )
        state.execute("CREATE TABLE rounds(status TEXT, terminal_class TEXT)")
        state.execute("CREATE TABLE scheduled_slots(slot_status TEXT)")
        state.commit()
        state.close()
        guard_path = tmp_path / "guard.sqlite3"
        guard = sqlite3.connect(guard_path)
        guard.execute("CREATE TABLE guard_calls(guard_call_id TEXT)")
        guard.commit()
        guard.close()
        projection = association_free_neutral_audit(
            state_snapshot=state_path,
            broker_snapshot=tmp_path / "audit.sqlite3",
            guard_snapshot=guard_path,
        )
        assert projection["broker_call_count_by_status"] == {"SUCCESS": 1}
        assert projection["broker_receipt_coverage_count"] == 1
    finally:
        broker.close()
    for artifact in private_root.rglob("*"):
        if artifact.is_file():
            assert b"private-test-key" not in artifact.read_bytes()


@pytest.mark.parametrize("returned_model", [None, "", "   ", 54, []])
def test_returned_model_must_be_explicit_non_empty_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returned_model: object,
) -> None:
    envelope: dict[str, object] = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"proposals": [{"mechanism_id": "m1"}]}
                    )
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    if returned_model is not None:
        envelope["model"] = returned_model

    monkeypatch.setattr(
        "recclaw_core.experiments.helix_abc_v1."
        "lab_api_broker.urlrequest.urlopen",
        lambda *_args, **_kwargs: _Response(envelope),
    )
    private_root = tmp_path / "private"
    broker = LabApiCanaryBrokerV1(
        private_root,
        schema_path=_schema(tmp_path / "schema.json"),
        config_path=_config(tmp_path / "llm_api.toml"),
        model="gpt-5.4-2026-03-05",
        max_total_tokens_per_call=100,
        timeout_ms=10_000,
    )
    try:
        with pytest.raises(CanaryBrokerError):
            broker.call_with_session(
                logical_call_id="strict-returned-model",
                proposal_generation_session_id="strict-model-session",
                prompt="Return one proposal.",
                expected_proposal_count=1,
                max_total_tokens=50,
            )
        row = broker._connection.execute(
            "SELECT status,error_type,error_detail_json,response_json,"
            "returned_model FROM calls"
        ).fetchone()
        assert tuple(row) == (
            "FAILED",
            "RESPONSE_CONTRACT_ERROR",
            '{"reason_code":"RETURNED_MODEL_TYPE_OR_EMPTY"}',
            None,
            None,
        )
    finally:
        broker.close()
    for artifact in private_root.rglob("*"):
        if artifact.is_file():
            content = artifact.read_bytes()
            assert b"private-test-key" not in content
            assert b"Return one proposal" not in content


def test_transport_failure_is_terminal_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0

    def fail_urlopen(*_: object, **__: object) -> object:
        nonlocal attempts
        attempts += 1
        raise urlerror.URLError("synthetic transport failure")

    monkeypatch.setattr(
        "recclaw_core.experiments.helix_abc_v1."
        "lab_api_broker.urlrequest.urlopen",
        fail_urlopen,
    )
    broker = LabApiCanaryBrokerV1(
        tmp_path / "private",
        schema_path=_schema(tmp_path / "schema.json"),
        config_path=_config(tmp_path / "llm_api.toml"),
        model="gpt-5.4",
        max_total_tokens_per_call=100,
        timeout_ms=10_000,
    )
    try:
        with pytest.raises(CanaryBrokerError) as failure:
            broker.call(
                logical_call_id="call-failure",
                prompt="Return one proposal.",
                expected_proposal_count=1,
            )
        assert failure.value.outcome is not None
        assert failure.value.receipt is not None
        assert failure.value.outcome.status == "PROCESS_FAILURE"
        closure = BrokerFailureClosureV1.create(
            round_id="round-lab-api-failure",
            receipt=failure.value.receipt,
            outcome=failure.value.outcome,
            physical_call_count=1,
            input_tokens=0,
            output_tokens=0,
            billed_tokens=0,
            wall_time_ms=failure.value.wall_time_ms,
        )
        assert closure.failure_class == "CONNECTIVITY_ERROR"
        assert closure.retry_count == 0
        assert attempts == 1
        row = broker._connection.execute(
            "SELECT status, error_type FROM calls"
        ).fetchone()
        assert tuple(row) == ("FAILED", "TRANSPORT_ERROR")
        with pytest.raises(CanaryBrokerError) as replay:
            broker.call(
                logical_call_id="call-failure",
                prompt="Return one proposal.",
                expected_proposal_count=1,
            )
        assert replay.value.outcome == failure.value.outcome
        assert replay.value.receipt == failure.value.receipt
        assert attempts == 1
    finally:
        broker.close()


def test_timeout_is_classified_as_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_urlopen(*_: object, **__: object) -> object:
        raise socket.timeout("synthetic timeout")

    monkeypatch.setattr(
        "recclaw_core.experiments.helix_abc_v1."
        "lab_api_broker.urlrequest.urlopen",
        fail_urlopen,
    )
    broker = LabApiCanaryBrokerV1(
        tmp_path / "private",
        schema_path=_schema(tmp_path / "schema.json"),
        config_path=_config(tmp_path / "llm_api.toml"),
        model="gpt-5.4",
        max_total_tokens_per_call=100,
        timeout_ms=10_000,
    )
    try:
        with pytest.raises(CanaryBrokerError) as failure:
            broker.call(
                logical_call_id="call-timeout",
                prompt="Return one proposal.",
                expected_proposal_count=1,
            )
        assert failure.value.receipt is not None
        assert failure.value.receipt.timed_out is True
        assert failure.value.outcome is not None
        closure = BrokerFailureClosureV1.create(
            round_id="round-lab-api-timeout",
            receipt=failure.value.receipt,
            outcome=failure.value.outcome,
            physical_call_count=1,
            input_tokens=0,
            output_tokens=0,
            billed_tokens=0,
            wall_time_ms=failure.value.wall_time_ms,
        )
        assert closure.failure_class == "TIMEOUT"
        row = broker._connection.execute(
            "SELECT status, error_type FROM calls"
        ).fetchone()
        assert tuple(row) == ("FAILED", "TIMEOUT")
    finally:
        broker.close()


def test_release_manifest_binds_public_transport_identity(
    tmp_path: Path,
) -> None:
    broker = LabApiCanaryBrokerV1(
        tmp_path / "private-1",
        schema_path=_schema(tmp_path / "schema.json"),
        config_path=_config(tmp_path / "llm_api.toml"),
        model="gpt-5.4",
        max_total_tokens_per_call=100,
        timeout_ms=10_000,
    )
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(broker.release.to_dict()), encoding="utf-8"
    )
    broker.close()
    second = LabApiCanaryBrokerV1(
        tmp_path / "private-2",
        schema_path=tmp_path / "schema.json",
        config_path=tmp_path / "llm_api.toml",
        model="gpt-5.4",
        max_total_tokens_per_call=100,
        timeout_ms=10_000,
        release_manifest_path=manifest,
    )
    try:
        connection = sqlite3.connect(second.db_path)
        try:
            assert connection.execute(
                "SELECT COUNT(*) FROM calls"
            ).fetchone()[0] == 0
        finally:
            connection.close()
    finally:
        second.close()
