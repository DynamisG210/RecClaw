"""Bounded real Codex CLI broker for the development-only M5 Canary."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jsonschema

from .canonical import canonical_json_bytes, sha256_digest


class CanaryBrokerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CanaryBrokerCallV1:
    logical_call_id: str
    request_digest: str
    response_digest: str
    response: Mapping[str, Any]
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    returned_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cached_input_tokens": self.cached_input_tokens,
            "input_tokens": self.input_tokens,
            "latency_ms": self.latency_ms,
            "logical_call_id": self.logical_call_id,
            "output_tokens": self.output_tokens,
            "request_digest": self.request_digest,
            "response": dict(self.response),
            "response_digest": self.response_digest,
            "returned_model": self.returned_model,
            "total_tokens": self.total_tokens,
        }


def _windows_path(path: Path) -> str:
    resolved = path.resolve()
    text = resolved.as_posix()
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        suffix = text[6:].replace("/", "\\")
        return f"{drive}:\\{suffix}"
    return "\\\\wsl.localhost\\Ubuntu" + text.replace("/", "\\")


def _usage_candidates(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        keys = set(value)
        if {"input_tokens", "output_tokens"}.issubset(keys):
            found.append(value)
        for child in value.values():
            found.extend(_usage_candidates(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_usage_candidates(child))
    return found


class CodexCliCanaryBrokerV1:
    """One process call per upstream request, no retry and create-once replay."""

    def __init__(
        self,
        private_root: Path,
        *,
        schema_path: Path,
        codex_executable: Path,
        model: str,
        reasoning_effort: str,
        service_tier: str,
        max_total_tokens_per_call: int,
    ) -> None:
        self.private_root = private_root.resolve()
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.schema_path = schema_path.resolve()
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        jsonschema.validators.validator_for(self.schema).check_schema(self.schema)
        self.codex_executable = codex_executable.resolve()
        if not self.codex_executable.is_file():
            raise CanaryBrokerError("frozen Codex executable is unavailable")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.max_total_tokens_per_call = max_total_tokens_per_call
        self.workspace = (
            Path("/mnt/c/Users/gtrho/AppData/Local/Temp")
            / "RecClawM5Broker"
            / "workspace"
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_root = self.private_root / "outputs"
        self.output_root.mkdir(parents=True, exist_ok=True)
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
                input_tokens INTEGER,
                cached_input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms INTEGER,
                returned_model TEXT,
                status TEXT NOT NULL,
                error_type TEXT
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _stored(
        self, logical_call_id: str, request_digest: str
    ) -> CanaryBrokerCallV1 | None:
        row = self._connection.execute(
            "SELECT * FROM calls WHERE logical_call_id=?", (logical_call_id,)
        ).fetchone()
        if row is None:
            return None
        columns = [item[1] for item in self._connection.execute("PRAGMA table_info(calls)")]
        record = dict(zip(columns, row, strict=True))
        if record["request_digest"] != request_digest:
            raise CanaryBrokerError("logical call id binds different request bytes")
        if record["status"] != "SUCCESS":
            raise CanaryBrokerError("stored broker call is a terminal failure")
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=request_digest,
            response_digest=str(record["response_digest"]),
            response=json.loads(record["response_json"]),
            input_tokens=int(record["input_tokens"]),
            cached_input_tokens=int(record["cached_input_tokens"]),
            output_tokens=int(record["output_tokens"]),
            total_tokens=int(record["total_tokens"]),
            latency_ms=int(record["latency_ms"]),
            returned_model=str(record["returned_model"]),
        )

    def call(
        self,
        *,
        logical_call_id: str,
        prompt: str,
        expected_proposal_count: int,
    ) -> CanaryBrokerCallV1:
        request = {
            "expected_proposal_count": expected_proposal_count,
            "model": self.model,
            "prompt": prompt,
            "reasoning_effort": self.reasoning_effort,
            "response_schema_sha256": sha256_digest(self.schema),
            "service_tier": self.service_tier,
        }
        request_digest = sha256_digest(request)
        prior = self._stored(logical_call_id, request_digest)
        if prior is not None:
            return prior
        output_path = self.output_root / f"{logical_call_id}.json"
        if output_path.exists():
            raise CanaryBrokerError("uncommitted broker output path already exists")
        command = [
            str(self.codex_executable),
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "--color",
            "never",
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c",
            f'service_tier="{self.service_tier}"',
            "-C",
            _windows_path(self.workspace),
            "--output-schema",
            _windows_path(self.schema_path),
            "--output-last-message",
            _windows_path(output_path),
            "-",
        ]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "NO_COLOR": "1"},
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if completed.returncode != 0 or not output_path.is_file():
            self._connection.execute(
                """
                INSERT INTO calls(logical_call_id, request_digest, status, error_type)
                VALUES (?, ?, 'FAILED', ?)
                """,
                (
                    logical_call_id,
                    request_digest,
                    "PROCESS_FAILURE"
                    if completed.returncode != 0
                    else "MISSING_OUTPUT",
                ),
            )
            self._connection.commit()
            raise CanaryBrokerError(
                f"Codex broker call failed with process status {completed.returncode}"
            )
        response_bytes = output_path.read_bytes()
        try:
            response = json.loads(response_bytes)
            jsonschema.validate(response, self.schema)
        except Exception as error:
            raise CanaryBrokerError("Codex broker response failed the frozen schema") from error
        proposals = response["proposals"]
        if len(proposals) != expected_proposal_count:
            raise CanaryBrokerError("Codex broker returned the wrong proposal count")
        usage_records = _usage_candidates(events)
        if not usage_records:
            raise CanaryBrokerError("Codex JSON event stream omitted token usage")
        usage = usage_records[-1]
        input_tokens = int(usage.get("input_tokens", 0))
        cached_input_tokens = int(usage.get("cached_input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(
            usage.get("total_tokens", input_tokens + output_tokens)
        )
        if total_tokens <= 0 or total_tokens > self.max_total_tokens_per_call:
            raise CanaryBrokerError("Codex call exceeded or omitted its token debit")
        returned_model = self.model
        for event in reversed(events):
            candidates = []
            if isinstance(event, Mapping):
                candidates.extend(
                    str(value)
                    for key, value in event.items()
                    if key in {"model", "returned_model"} and value
                )
            if candidates:
                returned_model = candidates[0]
                break
        response_digest = sha256_digest(response)
        self._connection.execute(
            """
            INSERT INTO calls(
                logical_call_id, request_digest, response_digest, response_json,
                input_tokens, cached_input_tokens, output_tokens, total_tokens,
                latency_ms, returned_model, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUCCESS')
            """,
            (
                logical_call_id,
                request_digest,
                response_digest,
                canonical_json_bytes(response).decode("utf-8"),
                input_tokens,
                cached_input_tokens,
                output_tokens,
                total_tokens,
                latency_ms,
                returned_model,
            ),
        )
        self._connection.commit()
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=request_digest,
            response_digest=response_digest,
            response=response,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            returned_model=returned_model,
        )

    def call_count(self) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(*) FROM calls WHERE status='SUCCESS'"
            ).fetchone()[0]
        )


def original_canary_prompt(*, round_index: int, search_seed: int) -> str:
    return f"""You are the Original RecClaw proposal policy in a development-only recommender-system Canary.
Do not use tools or inspect files. Return JSON only through the supplied schema.
Protocol: ML-1M, frozen full-sort NDCG@10, unchanged protocol, one eventual execution.
Search seed: {search_seed}. Round: {round_index}.
Propose exactly four diverse runnable mechanism directions. You may choose only the closed
backbone/objective/sampler/axis values in the schema. Keep each hypothesis mechanistic,
state an expected signal and a concrete failure mode. This is proposal generation, not
scientific adjudication. Stay within search utility only."""


_ROLE_INSTRUCTIONS = {
    "mechanism_composer": "compose a coherent mechanism intervention",
    "lineage_refiner": "refine a plausible mechanism lineage without parameter-only tuning",
    "falsification_designer": "design a discriminative falsification candidate",
    "frontier_architect": "seek a structurally distinct frontier candidate",
}


def research_canary_prompt(
    *, role: str, round_index: int, search_seed: int
) -> str:
    try:
        instruction = _ROLE_INSTRUCTIONS[role]
    except KeyError as error:
        raise CanaryBrokerError("unknown Research Producer role") from error
    required_intent = "FALSIFICATION" if role == "falsification_designer" else "DISCOVERY"
    return f"""You are the {role} independent Producer in a development-only recommender-system Canary.
Do not use tools or inspect files. Return JSON only through the supplied schema.
Your role is to {instruction}. Protocol: ML-1M, frozen full-sort NDCG@10, unchanged.
Search seed: {search_seed}. Round: {round_index}. Return exactly one proposal and set
proposal_intent to {required_intent}. Use only the closed backbone/objective/sampler/axis
values. Optimize search utility: runnable probability, useful signal, frontier potential,
information gain, cost and blocker risk. Stay within search utility only."""


__all__ = [
    "CanaryBrokerCallV1",
    "CanaryBrokerError",
    "CodexCliCanaryBrokerV1",
    "original_canary_prompt",
    "research_canary_prompt",
]
