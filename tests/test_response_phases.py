import json
from pathlib import Path
import sys

import pytest

CORE = Path(__file__).resolve().parents[1] / "src/recrsi/native/task/search_source/isolated_graph_core/src"
sys.path.insert(0, str(CORE))
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiCanaryBrokerV1, LabApiResponseContractError,
)


def parse(messages):
    broker = object.__new__(LabApiCanaryBrokerV1)
    broker.wire_api = "responses"
    broker.schema = {"type": "object", "required": ["proposals"]}
    return broker._parse_response(
        envelope={"status": "completed", "output": messages,
                  "usage": {"input_tokens": 10, "output_tokens": 20}, "model": "fixture"},
        expected_proposal_count=2, effective_ceiling=100,
    )


def message(phase=None):
    result = {"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": json.dumps({"proposals": [{}, {}]})}]}
    if phase is not None:
        result["phase"] = phase
    return result


def test_explicit_final_excludes_commentary_and_preserves_usage():
    commentary = message("commentary")
    commentary["content"][0]["text"] = "not proposal JSON"
    result, usage, model = parse([commentary, {"type": "reasoning"}, message("final_answer")])
    assert len(result["proposals"]) == 2
    assert usage["total_tokens"] == 30


def test_legacy_single_message():
    assert len(parse([message()])[0]["proposals"]) == 2


@pytest.mark.parametrize("messages", [
    [message("final_answer"), message("final_answer")],
    [message(), message()],
    [message("commentary")],
    [message(), message("final_answer")],
])
def test_ambiguous_or_commentary_only_rejected(messages):
    with pytest.raises(LabApiResponseContractError):
        parse(messages)

