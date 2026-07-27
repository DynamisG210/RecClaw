"""Bounded real Codex CLI broker for the development-only M5 Canary."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .broker_process import (
    BrokerCallOutcomeV2,
    BrokerFailureClassV1,
    BrokerProcessExitReceiptV2,
    BrokerProcessReleaseV2,
    BrokerProcessRunnerV2,
)
from .canonical import canonical_json_bytes, sha256_digest
from .contracts import validate_no_research_evidence_authority_fields


class CanaryBrokerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        outcome: BrokerCallOutcomeV2 | None = None,
        receipt: BrokerProcessExitReceiptV2 | None = None,
        physical_call_count: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        billed_tokens: int = 0,
        wall_time_ms: int = 0,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.receipt = receipt
        self.physical_call_count = physical_call_count
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.billed_tokens = billed_tokens
        self.wall_time_ms = wall_time_ms


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
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_usage_candidates(child))
    return found


def _usage_projection(
    events: Any,
) -> tuple[int, int, int]:
    records = _usage_candidates(events)
    if not records:
        return 0, 0, 0
    usage = records[-1]
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    total_tokens = int(
        usage.get("total_tokens", input_tokens + output_tokens)
    )
    return input_tokens, output_tokens, total_tokens


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
        timeout_ms: int = 900_000,
        cli_version: str | None = None,
        login_mode: str = "CHATGPT",
        release_manifest_path: Path | None = None,
    ) -> None:
        self.private_root = private_root.resolve()
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.schema_path = schema_path.resolve()
        self.schema_bytes = self.schema_path.read_bytes()
        self.schema = json.loads(self.schema_bytes)
        self.schema_file_sha256 = hashlib.sha256(self.schema_bytes).hexdigest()
        jsonschema.validators.validator_for(self.schema).check_schema(self.schema)
        self.codex_executable = codex_executable.resolve()
        if not self.codex_executable.is_file():
            raise CanaryBrokerError("frozen Codex executable is unavailable")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.max_total_tokens_per_call = max_total_tokens_per_call
        self.timeout_ms = timeout_ms
        self.workspace = (
            Path("/mnt/c/Users/gtrho/AppData/Local/Temp")
            / "RecClawM6FBroker"
            / "workspace"
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.output_root = self.private_root / "outputs"
        self.output_root.mkdir(parents=True, exist_ok=True)
        if cli_version is None:
            version_process = subprocess.run(
                [str(self.codex_executable), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if version_process.returncode != 0 or not version_process.stdout.strip():
                raise CanaryBrokerError("Broker CLI version could not be resolved")
            cli_version = version_process.stdout.strip()
        computed_release = BrokerProcessReleaseV2.create(
            executable=self.codex_executable,
            cli_version=cli_version,
            login_mode=login_mode,
            model=model,
            reasoning_effort=reasoning_effort,
            sandbox_mode="read-only",
            response_schema_digest=self.schema_file_sha256,
            timeout_ms=timeout_ms,
            workspace=self.workspace,
        )
        if release_manifest_path is not None:
            frozen_release = BrokerProcessReleaseV2(
                **json.loads(
                    release_manifest_path.resolve().read_text(encoding="utf-8")
                )
            )
            frozen_release.verify()
            if frozen_release.to_dict() != computed_release.to_dict():
                raise CanaryBrokerError(
                    "runtime Broker release differs from package-owned bytes"
                )
            self.release = frozen_release
        else:
            self.release = computed_release
        self.process_runner = BrokerProcessRunnerV2(
            private_root=self.private_root,
            release=self.release,
            response_schema=self.schema,
            timeout_ms=timeout_ms,
        )
        (self.private_root / "BROKER_PROCESS_RELEASE_V2.json").write_bytes(
            canonical_json_bytes(self.release.to_dict()) + b"\n"
        )
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
                error_type TEXT,
                proposal_generation_session_id TEXT NOT NULL,
                request_envelope_digest TEXT NOT NULL,
                start_record_digest TEXT NOT NULL,
                exit_receipt_digest TEXT NOT NULL,
                exit_receipt_json TEXT NOT NULL,
                outcome_digest TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                broker_release_digest TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def create_audit_snapshot(
        self, snapshot_path: Path, *, audit_purpose: str
    ) -> Any:
        from .audit_snapshot import create_immutable_audit_snapshot

        return create_immutable_audit_snapshot(
            writer_connection=self._connection,
            source_db_path=self.db_path,
            snapshot_path=snapshot_path,
            source_schema_identity=sha256_digest(
                {
                    "calls_table": "BROKER_CALLS_V2",
                    "release_digest": self.release.release_digest,
                }
            ),
            audit_purpose=audit_purpose,
        )

    def _stored(
        self,
        logical_call_id: str,
        request_digest: str,
        proposal_generation_session_id: str,
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
        if (
            record["proposal_generation_session_id"]
            != proposal_generation_session_id
        ):
            raise CanaryBrokerError(
                "logical call id binds a different proposal session"
            )
        if record["broker_release_digest"] != self.release.release_digest:
            raise CanaryBrokerError(
                "stored broker call binds a different Broker release"
            )
        if record["status"] != "SUCCESS":
            outcome = BrokerCallOutcomeV2(**json.loads(record["outcome_json"]))
            receipt = BrokerProcessExitReceiptV2(
                **json.loads(record["exit_receipt_json"])
            )
            raise CanaryBrokerError(
                "stored broker call is a terminal failure",
                outcome=outcome,
                receipt=receipt,
                physical_call_count=1,
                input_tokens=int(record["input_tokens"] or 0),
                output_tokens=int(record["output_tokens"] or 0),
                billed_tokens=int(record["total_tokens"] or 0),
                wall_time_ms=receipt.latency_ms,
            )
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
        max_total_tokens: int | None = None,
    ) -> CanaryBrokerCallV1:
        return self.call_with_session(
            logical_call_id=logical_call_id,
            proposal_generation_session_id=logical_call_id,
            prompt=prompt,
            expected_proposal_count=expected_proposal_count,
            max_total_tokens=max_total_tokens,
        )

    def call_with_session(
        self,
        *,
        logical_call_id: str,
        proposal_generation_session_id: str,
        prompt: str,
        expected_proposal_count: int,
        max_total_tokens: int | None = None,
    ) -> CanaryBrokerCallV1:
        effective_token_ceiling = int(
            max_total_tokens
            if max_total_tokens is not None
            else self.max_total_tokens_per_call
        )
        if (
            effective_token_ceiling < 1
            or effective_token_ceiling > self.max_total_tokens_per_call
        ):
            raise CanaryBrokerError("per-call token ceiling is outside the release")
        request = {
            "expected_proposal_count": expected_proposal_count,
            "max_total_tokens": effective_token_ceiling,
            "model": self.model,
            "prompt": prompt,
            "reasoning_effort": self.reasoning_effort,
            "response_schema_sha256": self.release.response_schema_digest,
            "service_tier": self.service_tier,
        }
        request_digest = sha256_digest(request)
        prior = self._stored(
            logical_call_id,
            request_digest,
            proposal_generation_session_id,
        )
        if prior is not None:
            return prior
        output_path = self.output_root / (
            sha256_digest({"logical_call_id": logical_call_id}) + ".json"
        )
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
        captured = self.process_runner.execute(
            logical_call_id=logical_call_id,
            proposal_generation_session_id=proposal_generation_session_id,
            prompt=prompt,
            argv=command,
            cwd=self.workspace,
            response_output=output_path,
            broker_request_digest=request_digest,
        )
        outcome = captured.outcome
        if outcome.status != "SUCCESS":
            failed_input, failed_output, failed_total = _usage_projection(
                captured.events
            )
            self._connection.execute(
                """
                INSERT INTO calls(
                    logical_call_id, request_digest, status, error_type,
                    proposal_generation_session_id, request_envelope_digest,
                    start_record_digest, exit_receipt_digest, exit_receipt_json,
                    outcome_digest,
                    outcome_json, broker_release_digest, input_tokens,
                    output_tokens, total_tokens, latency_ms
                ) VALUES (?, ?, 'FAILED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    logical_call_id,
                    request_digest,
                    outcome.failure_class,
                    proposal_generation_session_id,
                    captured.request.envelope_digest,
                    captured.start.start_record_digest,
                    captured.receipt.receipt_digest,
                    canonical_json_bytes(captured.receipt.to_dict()).decode("utf-8"),
                    outcome.outcome_digest,
                    canonical_json_bytes(outcome.to_dict()).decode("utf-8"),
                    self.release.release_digest,
                    failed_input,
                    failed_output,
                    failed_total,
                    captured.receipt.latency_ms,
                ),
            )
            self._connection.commit()
            raise CanaryBrokerError(
                f"Codex broker call failed as {outcome.failure_class}",
                outcome=outcome,
                receipt=captured.receipt,
                physical_call_count=1,
                input_tokens=failed_input,
                output_tokens=failed_output,
                billed_tokens=failed_total,
                wall_time_ms=captured.receipt.latency_ms,
            )
        if captured.response_bytes is None:
            raise CanaryBrokerError(
                "successful Broker process omitted its response bytes",
                outcome=outcome,
            )
        response = json.loads(captured.response_bytes)
        jsonschema.validate(response, self.schema)
        proposals = response["proposals"]
        if len(proposals) != expected_proposal_count:
            failed_input, failed_output, failed_total = _usage_projection(
                captured.events
            )
            outcome = self._semantic_failure(
                captured=captured,
                failure_class=BrokerFailureClassV1.SCHEMA_VALIDATION_FAILURE,
                rule_id="M6F_CLASSIFIER_PROPOSAL_COUNT_V1",
            )
            self._persist_failure(
                logical_call_id=logical_call_id,
                request_digest=request_digest,
                proposal_generation_session_id=proposal_generation_session_id,
                captured=captured,
                outcome=outcome,
            )
            raise CanaryBrokerError(
                "Codex broker returned the wrong proposal count",
                outcome=outcome,
                receipt=captured.receipt,
                physical_call_count=1,
                input_tokens=failed_input,
                output_tokens=failed_output,
                billed_tokens=failed_total,
                wall_time_ms=captured.receipt.latency_ms,
            )
        usage_records = _usage_candidates(captured.events)
        if not usage_records:
            outcome = self._semantic_failure(
                captured=captured,
                failure_class=BrokerFailureClassV1.MALFORMED_EVENT_STREAM,
                rule_id="M6F_CLASSIFIER_USAGE_OMITTED_V1",
            )
            self._persist_failure(
                logical_call_id=logical_call_id,
                request_digest=request_digest,
                proposal_generation_session_id=proposal_generation_session_id,
                captured=captured,
                outcome=outcome,
            )
            raise CanaryBrokerError(
                "Codex JSON event stream omitted token usage",
                outcome=outcome,
                receipt=captured.receipt,
                physical_call_count=1,
                wall_time_ms=captured.receipt.latency_ms,
            )
        usage = usage_records[-1]
        input_tokens = int(usage.get("input_tokens", 0))
        cached_input_tokens = int(usage.get("cached_input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(
            usage.get("total_tokens", input_tokens + output_tokens)
        )
        if total_tokens <= 0 or total_tokens > effective_token_ceiling:
            outcome = self._semantic_failure(
                captured=captured,
                failure_class=BrokerFailureClassV1.CLI_CONTRACT_ERROR,
                rule_id="M6F_CLASSIFIER_TOKEN_DEBIT_V1",
            )
            self._persist_failure(
                logical_call_id=logical_call_id,
                request_digest=request_digest,
                proposal_generation_session_id=proposal_generation_session_id,
                captured=captured,
                outcome=outcome,
            )
            raise CanaryBrokerError(
                "Codex call exceeded or omitted its token debit",
                outcome=outcome,
                receipt=captured.receipt,
                physical_call_count=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                billed_tokens=total_tokens,
                wall_time_ms=captured.receipt.latency_ms,
            )
        returned_model = self.model
        for event in reversed(captured.events):
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
        response_digest = str(outcome.response_digest)
        self._connection.execute(
            """
            INSERT INTO calls(
                logical_call_id, request_digest, response_digest, response_json,
                input_tokens, cached_input_tokens, output_tokens, total_tokens,
                latency_ms, returned_model, status, proposal_generation_session_id,
                request_envelope_digest, start_record_digest, exit_receipt_digest,
                exit_receipt_json, outcome_digest, outcome_json,
                broker_release_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUCCESS', ?, ?, ?, ?, ?, ?, ?, ?)
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
                captured.receipt.latency_ms,
                returned_model,
                proposal_generation_session_id,
                captured.request.envelope_digest,
                captured.start.start_record_digest,
                captured.receipt.receipt_digest,
                canonical_json_bytes(captured.receipt.to_dict()).decode("utf-8"),
                outcome.outcome_digest,
                canonical_json_bytes(outcome.to_dict()).decode("utf-8"),
                self.release.release_digest,
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
            latency_ms=captured.receipt.latency_ms,
            returned_model=returned_model,
        )

    @staticmethod
    def _semantic_failure(
        *,
        captured: Any,
        failure_class: BrokerFailureClassV1,
        rule_id: str,
    ) -> BrokerCallOutcomeV2:
        payload = {
            "classifier_rule_id": rule_id,
            "failure_class": failure_class.value,
            "logical_call_id": captured.outcome.logical_call_id,
            "proposal_generation_session_id": (
                captured.outcome.proposal_generation_session_id
            ),
            "receipt_digest": captured.receipt.receipt_digest,
            "redacted_excerpt": None,
            "request_envelope_digest": captured.request.envelope_digest,
            "response_digest": None,
            "status": "PROCESS_FAILURE",
            "supporting_artifact_ref": captured.receipt.stdout_artifact_ref,
        }
        return BrokerCallOutcomeV2(
            **payload, outcome_digest=sha256_digest(payload)
        )

    def _persist_failure(
        self,
        *,
        logical_call_id: str,
        request_digest: str,
        proposal_generation_session_id: str,
        captured: Any,
        outcome: BrokerCallOutcomeV2,
    ) -> None:
        input_tokens, output_tokens, total_tokens = _usage_projection(
            captured.events
        )
        self._connection.execute(
            """
            INSERT INTO calls(
                logical_call_id, request_digest, status, error_type,
                proposal_generation_session_id, request_envelope_digest,
                start_record_digest, exit_receipt_digest, exit_receipt_json,
                outcome_digest,
                outcome_json, broker_release_digest, input_tokens,
                output_tokens, total_tokens, latency_ms
            ) VALUES (?, ?, 'FAILED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                logical_call_id,
                request_digest,
                outcome.failure_class,
                proposal_generation_session_id,
                captured.request.envelope_digest,
                captured.start.start_record_digest,
                captured.receipt.receipt_digest,
                canonical_json_bytes(captured.receipt.to_dict()).decode("utf-8"),
                outcome.outcome_digest,
                canonical_json_bytes(outcome.to_dict()).decode("utf-8"),
                self.release.release_digest,
                input_tokens,
                output_tokens,
                total_tokens,
                captured.receipt.latency_ms,
            ),
        )
        self._connection.commit()

    def call_count(self) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(*) FROM calls WHERE status='SUCCESS'"
            ).fetchone()[0]
        )

    def conformance_evidence(
        self, logical_call_id: str
    ) -> tuple[BrokerProcessExitReceiptV2, BrokerCallOutcomeV2]:
        row = self._connection.execute(
            "SELECT exit_receipt_json, outcome_json FROM calls "
            "WHERE logical_call_id=?",
            (logical_call_id,),
        ).fetchone()
        if row is None:
            raise CanaryBrokerError("Broker conformance call is not recorded")
        return (
            BrokerProcessExitReceiptV2(**json.loads(row[0])),
            BrokerCallOutcomeV2(**json.loads(row[1])),
        )


def original_canary_prompt(
    *,
    round_index: int,
    search_seed: int,
    phase_name: str = "Canary",
    catalog_projection: Mapping[str, Any] | Sequence[Mapping[str, Any]] = (),
    original_state: Mapping[str, Any] | None = None,
) -> str:
    catalog = json.dumps(
        (
            dict(catalog_projection)
            if isinstance(catalog_projection, Mapping)
            else list(catalog_projection)
        ),
        ensure_ascii=True,
        sort_keys=True,
    )
    state = json.dumps(
        dict(original_state or {}), ensure_ascii=True, sort_keys=True
    )
    return f"""You are the Original RecClaw proposal policy in a development-only recommender-system {phase_name}.
Do not use tools or inspect files. Return JSON only through the supplied schema.
Protocol: ML-1M, frozen full-sort NDCG@10, unchanged protocol, one eventual execution.
Search seed: {search_seed}. Round: {round_index}.
Propose exactly four diverse candidates from the exact executable catalog below.
Use each mechanism_id at most once. Preserve the Original policy's preference for
novel runnable families, avoid recently executed semantics, and use prior outcomes
without Research roles, Research Router scores, Meta policy, or Evidence authority.
Set parent_candidate_id only when the supplied Original state contains that exact ID.
Keep the mechanism hypothesis, competing hypothesis, predicted outcome signature and
failure mode consistent with the selected catalog entry. This is proposal generation,
not evidence adjudication.
Executable catalog: {catalog}
Original planner state: {state}"""


_ROLE_INSTRUCTIONS = {
    "mechanism_composer": (
        "choose a coherent interaction or representation mechanism whose "
        "components form one causal story"
    ),
    "lineage_refiner": (
        "choose the smallest mechanism-level change that isolates one causal "
        "difference from an established lineage, without parameter-only tuning"
    ),
    "falsification_designer": (
        "choose a discriminative candidate whose result would separate two "
        "competing mechanism explanations"
    ),
    "frontier_architect": (
        "choose an underexplored, structurally distinct mechanism family with "
        "credible upside under the fixed cost"
    ),
}

def research_canary_prompt(
    *,
    role: str,
    round_index: int,
    search_seed: int,
    phase_name: str = "Canary",
    memory_summary: Mapping[str, Any] | None = None,
    catalog_projection: Mapping[str, Any] | Sequence[Mapping[str, Any]] = (),
    policy_directive: Mapping[str, Any] | None = None,
    token_ceiling: int | None = None,
) -> str:
    try:
        instruction = _ROLE_INSTRUCTIONS[role]
    except KeyError as error:
        raise CanaryBrokerError("unknown Research Producer role") from error
    required_intent = str(
        (policy_directive or {}).get(
            "proposal_intent",
            (
                "FALSIFICATION"
                if role == "falsification_designer"
                else "DISCOVERY"
            ),
        )
    )
    prompt_feedback = dict(
        memory_summary
        or {
            "common_search_utility_slot": "ABSENT",
            "research_task_slot": "ABSENT",
        }
    )
    if set(prompt_feedback) != {
        "common_search_utility_slot",
        "research_task_slot",
    }:
        raise CanaryBrokerError(
            "Producer prompt requires the closed PromptFeedbackProjectionV2"
        )
    validate_no_research_evidence_authority_fields(prompt_feedback)
    memory_line = (
        "Prior compact Search Memory feedback: "
        + json.dumps(prompt_feedback, ensure_ascii=True, sort_keys=True)
    )
    catalog = json.dumps(
        (
            dict(catalog_projection)
            if isinstance(catalog_projection, Mapping)
            else list(catalog_projection)
        ),
        ensure_ascii=True,
        sort_keys=True,
    )
    directive = json.dumps(
        dict(policy_directive or {}), ensure_ascii=True, sort_keys=True
    )
    return f"""You are the {role} independent Producer in a development-only recommender-system {phase_name}.
Do not use tools or inspect files. Return JSON only through the supplied schema.
Your role is to {instruction}. Protocol: ML-1M, frozen full-sort NDCG@10, unchanged.
Search seed: {search_seed}. Round: {round_index}. Return exactly one proposal and set
proposal_intent to {required_intent}. Choose exactly one mechanism_id from the
executable catalog and keep every scientific field consistent with that exact
mechanism. Use parent_candidate_id only for an exact ID present in your role-scoped
memory. Do not invent a mechanism that the catalog cannot execute.
Optimize useful signal, frontier potential and information gain under the frozen
budget. Executability and mechanical cost are derived by the package runtime, not
self-reported by you. Stay within search utility only.
Pre-call policy directive: {directive}
Per-call total-token ceiling: {token_ceiling if token_ceiling is not None else 'RELEASE_DEFAULT'}
Executable catalog: {catalog}
{memory_line}"""


__all__ = [
    "CanaryBrokerCallV1",
    "CanaryBrokerError",
    "CodexCliCanaryBrokerV1",
    "original_canary_prompt",
    "research_canary_prompt",
]
