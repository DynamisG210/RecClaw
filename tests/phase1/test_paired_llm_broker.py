from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from recclaw_phase1.paired_llm_broker import (
    BrokerConfig,
    BrokerStore,
    canonical_request,
    process_request,
)


def _request(prompt: str, *, stream: bool = False) -> bytes:
    return json.dumps(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": stream,
        }
    ).encode()


class PairedLLMBrokerTests(unittest.TestCase):
    def config(self, root: Path) -> BrokerConfig:
        return BrokerConfig(
            store=BrokerStore(root / "broker.sqlite3"),
            client_token="client-secret-not-persisted",
            upstream_base_url="https://invalid.example/v1",
            upstream_api_key="upstream-secret-not-persisted",
            upstream_timeout_seconds=1,
            pair_wait_seconds=2,
        )

    def test_identical_sequential_pair_replays_one_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            calls = []

            def upstream(_config: BrokerConfig, body: bytes) -> dict[str, object]:
                calls.append(body)
                return {"status": 200, "body": b'{"choices":[{"message":{"content":"x"}}]}', "content_type": "application/json"}

            first, first_meta = process_request(
                config=config, pair_id="PAIR-42", arm="control", body=_request("same"), upstream=upstream
            )
            second, second_meta = process_request(
                config=config, pair_id="PAIR-42", arm="treatment", body=_request("same"), upstream=upstream
            )
            self.assertEqual(1, len(calls))
            self.assertEqual(first, second)
            self.assertEqual(0, first_meta["sequence_index"])
            self.assertEqual("IDENTICAL_PAIRED_REQUEST", second_meta["pair_relation"])

    def test_identical_concurrent_pair_waits_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            calls = []
            results = []
            errors = []

            def upstream(_config: BrokerConfig, body: bytes) -> dict[str, object]:
                calls.append(body)
                time.sleep(0.15)
                return {"status": 200, "body": b'{"choices":[]}', "content_type": "application/json"}

            def invoke(arm: str) -> None:
                try:
                    results.append(process_request(
                        config=config, pair_id="PAIR-43", arm=arm, body=_request("same"), upstream=upstream
                    )[0])
                except Exception as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            first = threading.Thread(target=invoke, args=("control",))
            second = threading.Thread(target=invoke, args=("treatment",))
            first.start()
            time.sleep(0.03)
            second.start()
            first.join()
            second.join()
            self.assertEqual([], errors)
            self.assertEqual(1, len(calls))
            self.assertEqual(2, len(results))
            self.assertEqual(results[0], results[1])

    def test_divergent_pair_calls_upstream_twice_and_records_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            calls = []

            def upstream(_config: BrokerConfig, body: bytes) -> dict[str, object]:
                calls.append(body)
                return {"status": 200, "body": body, "content_type": "application/json"}

            process_request(config=config, pair_id="PAIR-44", arm="control", body=_request("a"), upstream=upstream)
            _, metadata = process_request(
                config=config, pair_id="PAIR-44", arm="treatment", body=_request("b"), upstream=upstream
            )
            self.assertEqual(2, len(calls))
            self.assertEqual("DIVERGENT_REQUEST", metadata["pair_relation"])

    def test_secrets_are_not_written_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)

            def upstream(_config: BrokerConfig, _body: bytes) -> dict[str, object]:
                return {"status": 200, "body": b"{}", "content_type": "application/json"}

            process_request(config=config, pair_id="PAIR-42", arm="control", body=_request("safe"), upstream=upstream)
            raw = (root / "broker.sqlite3").read_bytes()
            self.assertNotIn(config.client_token.encode(), raw)
            self.assertNotIn(config.upstream_api_key.encode(), raw)

    def test_streaming_request_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            canonical_request(_request("stream", stream=True))


if __name__ == "__main__":
    unittest.main()
