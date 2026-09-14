"""Real broker persistence with a valid strict schema and offline transport."""
import json

import pytest

from recclaw_core.experiments.helix_abc_v1 import lab_api_broker as lab
from recclaw_core.experiments.helix_abc_v1.canary_broker import CanaryBrokerError


@pytest.mark.parametrize("kind", ["schema", "json", "shape", "unknown", "success"])
def test_usage_and_replay(tmp_path, monkeypatch, kind):
    config = tmp_path / "api.toml"
    config.write_text('api_key = "offline-test-only"\nbase_url = "https://offline.invalid/v1"\n')
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({
        "type": "object", "additionalProperties": False, "required": ["proposals"],
        "properties": {"proposals": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }}},
    }))
    content = json.dumps({"proposals": [{"name": "mechanism"}]})
    if kind == "schema":
        content = json.dumps({"proposals": "invalid"})
    elif kind in {"json", "unknown"}:
        content = "{broken"
    envelope = {
        "choices": [{"message": {"content": content}}], "model": "gpt-5.6-terra",
        "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
    }
    if kind == "shape":
        envelope["choices"] = []
    if kind == "unknown":
        del envelope["usage"]
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(envelope).encode()

    def transport(*args, **kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(lab.urlrequest, "urlopen", transport)
    broker = lab.LabApiCanaryBrokerV1(
        tmp_path / "private", schema_path=schema, config_path=config,
        model="gpt-5.6-terra", max_total_tokens_per_call=100, timeout_ms=1000,
    )
    kwargs = dict(logical_call_id="fixture", prompt="offline fixture", expected_proposal_count=1)
    try:
        for _ in range(2):
            if kind == "success":
                assert broker.call(**kwargs).response["proposals"][0]["name"] == "mechanism"
            else:
                with pytest.raises(CanaryBrokerError) as caught:
                    broker.call(**kwargs)
                if len(calls) == 1 and kind != "unknown":
                    assert caught.value.billed_tokens == 22
        row = dict(broker._connection.execute("SELECT * FROM calls").fetchone())
        assert len(calls) == 1
        assert row["status"] == ("SUCCESS" if kind == "success" else "FAILED")
        if kind != "unknown":
            assert (row["input_tokens"], row["output_tokens"], row["total_tokens"]) == (17, 5, 22)
            assert row["returned_model"] == "gpt-5.6-terra"
        else:
            assert not row["total_tokens"]
    finally:
        broker.close()
