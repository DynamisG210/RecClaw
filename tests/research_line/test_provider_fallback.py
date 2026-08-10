from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib import error as urlerror

import pytest

from recclaw_core.experiments.helix_abc_v1 import fresh_r1, lab_api_broker
from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerCallV1,
    CanaryBrokerError,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest


def _write_credential_sources(tmp_path: Path) -> tuple[Path, Path]:
    ordered = tmp_path / "ordered-llm-api.md"
    ordered.write_text(
        '\n'.join(
            (
                'base_url = "https://primary.fixture.invalid/v1"',
                'api_key = "primary-fixture-key"',
                'base_url = "https://fallback.fixture.invalid/v1"',
                'api_key = "fallback-fixture-key"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    single = tmp_path / "single-llm-api.md"
    single.write_text(
        'base_url = "https://single.fixture.invalid/v1"\n'
        'api_key = "single-fixture-key"\n',
        encoding="utf-8",
    )
    return ordered, single


def _fake_broker_class(
    events: list[dict[str, Any]],
    failures_by_index: Mapping[int, Mapping[str, Any]],
) -> type:
    class FakeBroker:
        def __init__(
            self,
            private_root: Path,
            *,
            schema_path: Path,
            config_path: Path,
            model: str,
            max_total_tokens_per_call: int,
            credential_index: int = 0,
            timeout_ms: int = 900_000,
            release_manifest_path: Path | None = None,
        ) -> None:
            del schema_path, max_total_tokens_per_call, timeout_ms
            del release_manifest_path
            self.private_root = Path(private_root)
            self.credential_index = credential_index
            self.model = model
            base_url, _ = lab_api_broker.load_lab_api_credentials(
                config_path,
                credential_index=credential_index,
            )
            self.endpoint_digest = sha256_digest({"base_url": base_url})
            self.credential_config_digest = hashlib.sha256(
                Path(config_path).read_bytes()
            ).hexdigest()
            self.credential_identity_digest = sha256_digest(
                {
                    "credential_config_digest": self.credential_config_digest,
                    "credential_index": credential_index,
                    "endpoint_digest": self.endpoint_digest,
                }
            )
            self.release = SimpleNamespace(
                release_digest=sha256_digest(
                    {"endpoint_digest": self.endpoint_digest}
                )
            )
            events.append(
                {
                    "credential_index": credential_index,
                    "model": model,
                    "stage": "init",
                }
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
            del expected_proposal_count, max_total_tokens, max_output_tokens
            request_digest = sha256_digest(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "proposal_generation_session_id": (
                        proposal_generation_session_id
                    ),
                }
            )
            event = {
                "credential_index": self.credential_index,
                "logical_call_id": logical_call_id,
                "request_digest": request_digest,
                "stage": "call",
            }
            events.append(event)
            failure = failures_by_index.get(self.credential_index)
            if failure is not None:
                error = CanaryBrokerError("fixture transport failure")
                error.fixture_failure = {
                    **failure,
                    "request_digest": request_digest,
                    "receipt_digest": sha256_digest(
                        {"request_digest": request_digest, "status": "FAILED"}
                    ),
                }
                raise error
            response = {"proposals": [{"fixture": "success"}]}
            return CanaryBrokerCallV1(
                logical_call_id=logical_call_id,
                request_digest=request_digest,
                response_digest=sha256_digest(response),
                response=response,
                input_tokens=11,
                cached_input_tokens=2,
                output_tokens=5,
                total_tokens=16,
                latency_ms=3,
                returned_model=self.model,
            )

        def close(self) -> None:
            return None

    return FakeBroker


def _run_bounded(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_path: Path,
    events: list[dict[str, Any]],
    failures_by_index: Mapping[int, Mapping[str, Any]],
    credential_schedule: tuple[int, ...] | None,
) -> fresh_r1.ProviderAttemptResult:
    monkeypatch.setattr(
        fresh_r1,
        "LabApiCanaryBrokerV1",
        _fake_broker_class(events, failures_by_index),
    )
    monkeypatch.setattr(
        fresh_r1,
        "_attempt_failure",
        lambda _private_root, error: error.fixture_failure,
    )
    return fresh_r1.bounded_provider_call(
        call_root=config_path.parent / "calls",
        schema_path=config_path,
        logical_call_id="fallback-logical-call",
        session_id="fallback-session",
        prompt="same frozen prompt",
        token_ceiling=6000,
        credential_config_path=config_path,
        credential_schedule=credential_schedule,
        maximum_physical_attempts=3,
        sleep=lambda _seconds: None,
    )


def test_ordered_loader_defaults_to_primary_and_preserves_single_config(
    tmp_path: Path,
) -> None:
    ordered, single = _write_credential_sources(tmp_path)

    assert len(lab_api_broker.load_lab_api_credential_pairs(ordered)) == 2
    primary_url, _ = lab_api_broker.load_lab_api_credentials(ordered)
    fallback_url, _ = lab_api_broker.load_lab_api_credentials(
        ordered,
        credential_index=1,
    )
    single_url, _ = lab_api_broker.load_lab_api_credentials(single)

    assert primary_url == "https://primary.fixture.invalid/v1"
    assert fallback_url == "https://fallback.fixture.invalid/v1"
    assert single_url == "https://single.fixture.invalid/v1"


def test_first_success_does_not_touch_fallback_and_keeps_model_fixed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ordered, _ = _write_credential_sources(tmp_path)
    events: list[dict[str, Any]] = []

    result = _run_bounded(
        monkeypatch,
        config_path=ordered,
        events=events,
        failures_by_index={},
        credential_schedule=(0, 1),
    )

    assert result.call is not None
    assert [event["credential_index"] for event in events] == [0, 0]
    assert {event["model"] for event in events if event["stage"] == "init"} == {
        "gpt-5.4"
    }
    assert len(result.attempts) == 1
    assert result.attempts[0]["credential_index"] == 0


@pytest.mark.parametrize(
    "failure",
    (
        {"http_status": 401, "failure_class": "PROVIDER_ERROR"},
        {"http_status": 403, "failure_class": "PROVIDER_ERROR"},
        {"http_status": 429, "failure_class": "PROVIDER_ERROR"},
        {"failure_class": "TIMEOUT"},
    ),
)
def test_retry_eligible_failure_switches_once_and_preserves_request_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: Mapping[str, Any],
) -> None:
    ordered, _ = _write_credential_sources(tmp_path)
    events: list[dict[str, Any]] = []

    result = _run_bounded(
        monkeypatch,
        config_path=ordered,
        events=events,
        failures_by_index={0: failure},
        credential_schedule=(0, 1),
    )

    assert result.call is not None
    assert [event["credential_index"] for event in events] == [0, 0, 1, 1]
    assert [attempt["credential_index"] for attempt in result.attempts] == [0, 1]
    assert len({attempt["request_digest"] for attempt in result.attempts}) == 1
    assert len({attempt["endpoint_digest"] for attempt in result.attempts}) == 2
    trace = json.dumps(result.attempts, sort_keys=True)
    assert "primary-fixture-key" not in trace
    assert "fallback-fixture-key" not in trace
    assert "https://primary.fixture.invalid" not in trace
    assert "https://fallback.fixture.invalid" not in trace


def test_reused_call_root_reads_the_current_failed_call_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ordered, _ = _write_credential_sources(tmp_path)
    schema_path = tmp_path / "proposal.schema.json"
    schema_path.write_text(
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
                            "properties": {
                                "mechanism_id": {"type": "string"}
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class ProviderResponse:
        status = 200

        def __enter__(self) -> "ProviderResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "proposals": [
                                            {"mechanism_id": "fixture"}
                                        ]
                                    }
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
            ).encode("utf-8")

    def fake_urlopen(request: object, **_kwargs: object) -> ProviderResponse:
        url = str(getattr(request, "full_url"))
        if "primary.fixture.invalid" in url:
            raise urlerror.HTTPError(
                url,
                401,
                "fixture primary failure",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"unauthorized"}'),
            )
        return ProviderResponse()

    monkeypatch.setattr(lab_api_broker.urlrequest, "urlopen", fake_urlopen)
    call_root = tmp_path / "reused-calls"
    results = []
    for index in (1, 2):
        results.append(
            fresh_r1.bounded_provider_call(
                call_root=call_root,
                schema_path=schema_path,
                logical_call_id=f"logical-call-{index}",
                session_id=f"session-{index}",
                prompt=f"frozen prompt {index}",
                token_ceiling=100,
                credential_config_path=ordered,
                credential_schedule=(0, 1),
                maximum_physical_attempts=2,
                sleep=lambda _seconds: None,
            )
        )

    assert all(result.call is not None for result in results)
    for result in results:
        assert len(result.attempts) == 2
        assert len(
            {attempt["request_digest"] for attempt in result.attempts}
        ) == 1


def test_non_retry_failure_does_not_switch_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ordered, _ = _write_credential_sources(tmp_path)
    events: list[dict[str, Any]] = []

    result = _run_bounded(
        monkeypatch,
        config_path=ordered,
        events=events,
        failures_by_index={
            0: {"http_status": 400, "failure_class": "CLI_CONTRACT_ERROR"}
        },
        credential_schedule=(0, 1),
    )

    assert result.call is None
    assert [event["credential_index"] for event in events] == [0, 0]
    assert len(result.attempts) == 1
    assert result.failure is not None
    assert result.failure["http_status"] == 400


def test_default_schedule_uses_only_single_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, single = _write_credential_sources(tmp_path)
    events: list[dict[str, Any]] = []

    result = _run_bounded(
        monkeypatch,
        config_path=single,
        events=events,
        failures_by_index={},
        credential_schedule=None,
    )

    assert result.call is not None
    assert [event["credential_index"] for event in events] == [0, 0]
