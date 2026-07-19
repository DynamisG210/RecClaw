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
            "max_tokens": 4096,
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
                config=config, pair_id="AB002-SEED-42", arm="control", body=_request("same"), upstream=upstream
            )
            second, second_meta = process_request(
                config=config, pair_id="AB002-SEED-42", arm="treatment", body=_request("same"), upstream=upstream
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
                        config=config, pair_id="AB002-SEED-43", arm=arm, body=_request("same"), upstream=upstream
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

            process_request(config=config, pair_id="AB002-SEED-44", arm="control", body=_request("a"), upstream=upstream)
            _, metadata = process_request(
                config=config, pair_id="AB002-SEED-44", arm="treatment", body=_request("b"), upstream=upstream
            )
            self.assertEqual(2, len(calls))
            self.assertEqual("DIVERGENT_REQUEST", metadata["pair_relation"])

    def test_prompt_hash_pairing_survives_different_intervening_call_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            calls = []

            def upstream(_config: BrokerConfig, body: bytes) -> dict[str, object]:
                calls.append(body)
                return {"status": 200, "body": body, "content_type": "application/json"}

            control_a, _ = process_request(
                config=config,
                pair_id="AB002-SEED-47",
                arm="control",
                body=_request("a"),
                upstream=upstream,
            )
            control_b, _ = process_request(
                config=config,
                pair_id="AB002-SEED-47",
                arm="control",
                body=_request("b"),
                upstream=upstream,
            )
            treatment_b, meta_b = process_request(
                config=config,
                pair_id="AB002-SEED-47",
                arm="treatment",
                body=_request("b"),
                upstream=upstream,
            )
            treatment_a, meta_a = process_request(
                config=config,
                pair_id="AB002-SEED-47",
                arm="treatment",
                body=_request("a"),
                upstream=upstream,
            )
            self.assertEqual(2, len(calls))
            self.assertEqual(control_a, treatment_a)
            self.assertEqual(control_b, treatment_b)
            self.assertEqual("IDENTICAL_PAIRED_REQUEST", meta_a["pair_relation"])
            self.assertEqual("IDENTICAL_PAIRED_REQUEST", meta_b["pair_relation"])

    def test_secrets_are_not_written_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)

            def upstream(_config: BrokerConfig, _body: bytes) -> dict[str, object]:
                return {"status": 200, "body": b"{}", "content_type": "application/json"}

            process_request(config=config, pair_id="AB002-SEED-42", arm="control", body=_request("safe"), upstream=upstream)
            raw = (root / "broker.sqlite3").read_bytes()
            self.assertNotIn(config.client_token.encode(), raw)
            self.assertNotIn(config.upstream_api_key.encode(), raw)

    def test_streaming_request_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            canonical_request(_request("stream", stream=True))

    def test_provider_identity_usage_and_latency_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)

            def upstream(_config: BrokerConfig, _body: bytes) -> dict[str, object]:
                return {
                    "status": 200,
                    "body": b'{"id":"req-123","model":"gpt-5.4","usage":{"total_tokens":12}}',
                    "content_type": "application/json",
                    "provider_request_id": "req-123",
                    "returned_model": "gpt-5.4",
                    "usage_json": '{"total_tokens":12}',
                    "upstream_latency_ms": 7.5,
                }

            process_request(
                config=config,
                pair_id="AB002-SEED-45",
                arm="control",
                body=_request("metadata"),
                upstream=upstream,
            )
            import sqlite3

            with sqlite3.connect(root / "broker.sqlite3") as connection:
                row = connection.execute(
                    "SELECT provider_request_id, returned_model, usage_json, upstream_latency_ms "
                    "FROM requests"
                ).fetchone()
            self.assertEqual(("req-123", "gpt-5.4", '{"total_tokens":12}', 7.5), row)

    def test_single_fault_request_budget_exhaustion_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = BrokerConfig(
                **{
                    **self.config(root).__dict__,
                    "max_requests_per_arm_per_pair": 1,
                }
            )

            def upstream(_config: BrokerConfig, body: bytes) -> dict[str, object]:
                return {"status": 200, "body": body, "content_type": "application/json"}

            process_request(
                config=config,
                pair_id="AB002-SEED-46",
                arm="control",
                body=_request("first"),
                upstream=upstream,
            )
            with self.assertRaisesRegex(ValueError, "request budget exhausted"):
                process_request(
                    config=config,
                    pair_id="AB002-SEED-46",
                    arm="control",
                    body=_request("second"),
                    upstream=upstream,
                )


if __name__ == "__main__":
    unittest.main()
