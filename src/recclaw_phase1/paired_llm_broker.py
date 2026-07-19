"""OpenAI-compatible paired LLM broker for the AB-002 development experiment.

The broker is a fairness and provenance mechanism, not scientific authority.
It makes one upstream call when the two arms submit byte-equivalent canonical
requests at the same pair sequence. Divergent requests are sent separately and
recorded as divergent; they are never forced to share an answer.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest


PAIR_PATH = re.compile(
    r"^/v1/pairs/(?P<pair>[A-Za-z0-9_.-]{1,96})/"
    r"(?P<arm>control|treatment)/chat/completions$"
)
MAX_REQUEST_BYTES = 4 * 1024 * 1024


def canonical_request(body: bytes) -> tuple[bytes, str]:
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("chat completion request must be a JSON object")
    if value.get("stream") is True:
        raise ValueError("streaming is not supported by the paired broker")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return encoded, hashlib.sha256(encoded).hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


class BrokerStore:
    """Concurrency-safe request ledger stored outside the source checkout."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS arm_counters (
                    pair_id TEXT NOT NULL,
                    arm TEXT NOT NULL CHECK (arm IN ('control', 'treatment')),
                    next_index INTEGER NOT NULL,
                    PRIMARY KEY (pair_id, arm)
                );
                CREATE TABLE IF NOT EXISTS requests (
                    pair_id TEXT NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    arm TEXT NOT NULL CHECK (arm IN ('control', 'treatment')),
                    request_sha256 TEXT NOT NULL,
                    canonical_request BLOB NOT NULL,
                    state TEXT NOT NULL,
                    pair_relation TEXT NOT NULL,
                    upstream_status INTEGER,
                    response_body BLOB,
                    response_sha256 TEXT,
                    content_type TEXT,
                    error_class TEXT,
                    created_unix_ms INTEGER NOT NULL,
                    completed_unix_ms INTEGER,
                    PRIMARY KEY (pair_id, sequence_index, arm)
                );
                """
            )

    def allocate(
        self,
        *,
        pair_id: str,
        arm: str,
        request_sha256: str,
        canonical_body: bytes,
    ) -> dict[str, Any]:
        other_arm = "treatment" if arm == "control" else "control"
        now = int(time.time() * 1000)
        with _connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT next_index FROM arm_counters WHERE pair_id=? AND arm=?",
                (pair_id, arm),
            ).fetchone()
            index = int(row["next_index"]) if row else 0
            if row:
                connection.execute(
                    "UPDATE arm_counters SET next_index=? WHERE pair_id=? AND arm=?",
                    (index + 1, pair_id, arm),
                )
            else:
                connection.execute(
                    "INSERT INTO arm_counters(pair_id, arm, next_index) VALUES(?,?,?)",
                    (pair_id, arm, 1),
                )
            other = connection.execute(
                "SELECT * FROM requests WHERE pair_id=? AND sequence_index=? AND arm=?",
                (pair_id, index, other_arm),
            ).fetchone()
            relation = "FIRST_ARRIVAL"
            action = "CALL_UPSTREAM"
            if other is not None:
                if other["request_sha256"] == request_sha256:
                    relation = "IDENTICAL_PAIRED_REQUEST"
                    action = (
                        "REPLAY"
                        if other["state"] == "COMPLETE"
                        else "WAIT_FOR_PAIR"
                    )
                else:
                    relation = "DIVERGENT_REQUEST"
            connection.execute(
                """
                INSERT INTO requests(
                    pair_id, sequence_index, arm, request_sha256,
                    canonical_request, state, pair_relation, created_unix_ms
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    pair_id,
                    index,
                    arm,
                    request_sha256,
                    canonical_body,
                    "PENDING" if action == "WAIT_FOR_PAIR" else "UPSTREAM_OWNER",
                    relation,
                    now,
                ),
            )
            connection.execute("COMMIT")
        return {
            "sequence_index": index,
            "other_arm": other_arm,
            "action": action,
            "pair_relation": relation,
        }

    def paired_response(
        self, *, pair_id: str, sequence_index: int, other_arm: str
    ) -> dict[str, Any] | None:
        with _connect(self.path) as connection:
            row = connection.execute(
                "SELECT * FROM requests WHERE pair_id=? AND sequence_index=? AND arm=?",
                (pair_id, sequence_index, other_arm),
            ).fetchone()
        if row is None or row["state"] != "COMPLETE":
            return None
        return {
            "status": int(row["upstream_status"]),
            "body": bytes(row["response_body"]),
            "content_type": str(row["content_type"] or "application/json"),
        }

    def complete(
        self,
        *,
        pair_id: str,
        sequence_index: int,
        arm: str,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        with _connect(self.path) as connection:
            connection.execute(
                """
                UPDATE requests
                SET state='COMPLETE', upstream_status=?, response_body=?,
                    response_sha256=?, content_type=?, completed_unix_ms=?
                WHERE pair_id=? AND sequence_index=? AND arm=?
                """,
                (
                    status,
                    body,
                    hashlib.sha256(body).hexdigest(),
                    content_type,
                    int(time.time() * 1000),
                    pair_id,
                    sequence_index,
                    arm,
                ),
            )

    def copy_pair_completion(
        self,
        *,
        pair_id: str,
        sequence_index: int,
        arm: str,
        response: dict[str, Any],
    ) -> None:
        self.complete(
            pair_id=pair_id,
            sequence_index=sequence_index,
            arm=arm,
            status=int(response["status"]),
            body=bytes(response["body"]),
            content_type=str(response["content_type"]),
        )

    def fail(
        self, *, pair_id: str, sequence_index: int, arm: str, error_class: str
    ) -> None:
        with _connect(self.path) as connection:
            connection.execute(
                """
                UPDATE requests
                SET state='FAILED', error_class=?, completed_unix_ms=?
                WHERE pair_id=? AND sequence_index=? AND arm=?
                """,
                (error_class, int(time.time() * 1000), pair_id, sequence_index, arm),
            )


@dataclass(frozen=True)
class BrokerConfig:
    store: BrokerStore
    client_token: str
    upstream_base_url: str
    upstream_api_key: str
    upstream_timeout_seconds: float
    pair_wait_seconds: float


def call_upstream(config: BrokerConfig, canonical_body: bytes) -> dict[str, Any]:
    request = urlrequest.Request(
        f"{config.upstream_base_url.rstrip('/')}/chat/completions",
        data=canonical_body,
        headers={
            "Authorization": f"Bearer {config.upstream_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(
            request, timeout=config.upstream_timeout_seconds
        ) as response:
            return {
                "status": int(response.status),
                "body": response.read(),
                "content_type": response.headers.get_content_type(),
            }
    except urlerror.HTTPError as exc:
        return {
            "status": int(exc.code),
            "body": exc.read(),
            "content_type": exc.headers.get_content_type(),
        }


def process_request(
    *,
    config: BrokerConfig,
    pair_id: str,
    arm: str,
    body: bytes,
    upstream: Callable[[BrokerConfig, bytes], dict[str, Any]] = call_upstream,
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_body, request_sha256 = canonical_request(body)
    allocation = config.store.allocate(
        pair_id=pair_id,
        arm=arm,
        request_sha256=request_sha256,
        canonical_body=canonical_body,
    )
    index = int(allocation["sequence_index"])
    if allocation["action"] == "REPLAY":
        response = config.store.paired_response(
            pair_id=pair_id,
            sequence_index=index,
            other_arm=str(allocation["other_arm"]),
        )
        if response is None:
            raise RuntimeError("paired response disappeared after allocation")
        config.store.copy_pair_completion(
            pair_id=pair_id,
            sequence_index=index,
            arm=arm,
            response=response,
        )
        return response, allocation
    if allocation["action"] == "WAIT_FOR_PAIR":
        deadline = time.monotonic() + config.pair_wait_seconds
        while time.monotonic() < deadline:
            response = config.store.paired_response(
                pair_id=pair_id,
                sequence_index=index,
                other_arm=str(allocation["other_arm"]),
            )
            if response is not None:
                config.store.copy_pair_completion(
                    pair_id=pair_id,
                    sequence_index=index,
                    arm=arm,
                    response=response,
                )
                return response, allocation
            time.sleep(0.05)
        config.store.fail(
            pair_id=pair_id,
            sequence_index=index,
            arm=arm,
            error_class="PAIR_WAIT_TIMEOUT",
        )
        raise TimeoutError("timed out waiting for identical paired response")
    try:
        response = upstream(config, canonical_body)
    except Exception as exc:
        config.store.fail(
            pair_id=pair_id,
            sequence_index=index,
            arm=arm,
            error_class=type(exc).__name__,
        )
        raise
    config.store.complete(
        pair_id=pair_id,
        sequence_index=index,
        arm=arm,
        status=int(response["status"]),
        body=bytes(response["body"]),
        content_type=str(response.get("content_type") or "application/json"),
    )
    return response, allocation


class PairedBrokerHandler(BaseHTTPRequestHandler):
    server_version = "RecClawPairedBroker/0.2"

    @property
    def config(self) -> BrokerConfig:
        return self.server.broker_config  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging authorization headers or request bodies.
        super().log_message(format, *args)

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json_error(HTTPStatus.NOT_FOUND, "not found")
            return
        body = b'{"status":"ok"}'
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        matched = PAIR_PATH.fullmatch(self.path)
        if matched is None:
            self._json_error(HTTPStatus.NOT_FOUND, "unsupported broker path")
            return
        expected = f"Bearer {self.config.client_token}"
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, expected):
            self._json_error(HTTPStatus.UNAUTHORIZED, "invalid broker client token")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid content length")
            return
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request size rejected")
            return
        body = self.rfile.read(length)
        try:
            response, allocation = process_request(
                config=self.config,
                pair_id=matched.group("pair"),
                arm=matched.group("arm"),
                body=body,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except TimeoutError as exc:
            self._json_error(HTTPStatus.GATEWAY_TIMEOUT, str(exc))
            return
        except Exception as exc:
            self._json_error(HTTPStatus.BAD_GATEWAY, type(exc).__name__)
            return
        response_body = bytes(response["body"])
        self.send_response(int(response["status"]))
        self.send_header("Content-Type", str(response["content_type"]))
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("X-RecClaw-Request-Index", str(allocation["sequence_index"]))
        self.send_header("X-RecClaw-Pair-Relation", str(allocation["pair_relation"]))
        self.end_headers()
        self.wfile.write(response_body)


class PairedBrokerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: BrokerConfig) -> None:
        super().__init__(address, PairedBrokerHandler)
        self.broker_config = config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AB-002 paired LLM broker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--client-token-env", default="RECCLAW_BROKER_CLIENT_TOKEN")
    parser.add_argument("--upstream-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--upstream-timeout", type=float, default=300.0)
    parser.add_argument("--pair-wait", type=float, default=330.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client_token = os.environ.get(args.client_token_env, "")
    upstream_key = os.environ.get(args.upstream_api_key_env, "")
    if not client_token:
        raise SystemExit(f"missing client token env: {args.client_token_env}")
    if not upstream_key:
        raise SystemExit(f"missing upstream API key env: {args.upstream_api_key_env}")
    config = BrokerConfig(
        store=BrokerStore(args.db),
        client_token=client_token,
        upstream_base_url=str(args.upstream_base_url),
        upstream_api_key=upstream_key,
        upstream_timeout_seconds=max(1.0, float(args.upstream_timeout)),
        pair_wait_seconds=max(1.0, float(args.pair_wait)),
    )
    server = PairedBrokerServer((str(args.host), int(args.port)), config)
    print(json.dumps({
        "status": "READY",
        "host": args.host,
        "port": args.port,
        "db": str(args.db.resolve()),
        "secrets_recorded": False,
    }, sort_keys=True))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
