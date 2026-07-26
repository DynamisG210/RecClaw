from __future__ import annotations

import json
import hashlib
import os
import shutil
import signal
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.audit_snapshot import (  # noqa: E402
    association_free_neutral_audit,
    create_immutable_audit_snapshot,
    verify_immutable_snapshot,
)
from recclaw_core.experiments.helix_abc_v1.broker_failure_closure import (  # noqa: E402
    close_broker_failure,
)
from recclaw_core.experiments.helix_abc_v1.broker_process import (  # noqa: E402
    BrokerCallOutcomeV2,
    BrokerFailureClassV1,
    BrokerProcessReleaseV2,
    BrokerProcessRunnerV2,
    redact_excerpt,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerCallV1,
    CanaryBrokerError,
    CodexCliCanaryBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    IdempotencyConflict,
    OpenRoundCommand,
    SingleWriterExperimentStoreV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    BrokerRoundFailureError,
    ThreeArmPreCanaryOrchestratorV1,
)


SCHEMA = {
    "type": "object",
    "required": ["proposals"],
    "properties": {"proposals": {"type": "array"}},
    "additionalProperties": False,
}


def _fixture_child(mode: str, response_path: Path) -> int:
    if mode == "exit1_stderr":
        print("fixture terminal failure", file=sys.stderr, flush=True)
        return 1
    if mode == "exit1_stdout":
        print("fixture terminal failure", flush=True)
        return 1
    if mode == "timeout":
        time.sleep(5)
        return 0
    if mode == "signal":
        os.kill(os.getpid(), signal.SIGTERM)
    if mode == "empty":
        return 0
    if mode == "malformed":
        print("not-json", flush=True)
        response_path.write_text('{"proposals":[]}', encoding="utf-8")
        return 0
    if mode == "schema_invalid":
        print(json.dumps({"type": "turn.completed"}), flush=True)
        response_path.write_text('{"wrong":[]}', encoding="utf-8")
        return 0
    if mode == "success":
        print(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ),
            flush=True,
        )
        response_path.write_text('{"proposals":[]}', encoding="utf-8")
        return 0
    if mode == "oversized":
        os.write(sys.stdout.fileno(), b"x" * 1_100_000)
        return 1
    if mode == "invalid_utf8":
        os.write(sys.stdout.fileno(), b"\xff\xfe\xfd")
        return 1
    raise AssertionError(mode)


class BrokerProcessCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.wrapper = self.root / "fixture-broker"
        self.wrapper.write_text(
            "#!/bin/sh\nexec "
            + sys.executable
            + ' "$@"\n',
            encoding="utf-8",
        )
        self.wrapper.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def runner(self, name: str, *, timeout_ms: int = 2_000):
        workspace = self.root / f"workspace-{name}"
        workspace.mkdir(exist_ok=True)
        release = BrokerProcessReleaseV2.create(
            executable=self.wrapper,
            cli_version="fixture-broker 1",
            login_mode="NONE",
            model="fixture",
            reasoning_effort="none",
            sandbox_mode="read-only",
            response_schema_digest=sha256_digest(SCHEMA),
            timeout_ms=timeout_ms,
            workspace=workspace,
        )
        return (
            BrokerProcessRunnerV2(
                private_root=self.root / f"private-{name}",
                release=release,
                response_schema=SCHEMA,
                timeout_ms=timeout_ms,
            ),
            workspace,
        )

    def invoke(self, mode: str, *, timeout_ms: int = 2_000):
        runner, workspace = self.runner(mode, timeout_ms=timeout_ms)
        response = self.root / f"{mode}.response.json"
        captured = runner.execute(
            logical_call_id=f"fixture-{mode}",
            proposal_generation_session_id=f"session-{mode}",
            prompt="fixed diagnostic",
            argv=[
                str(self.wrapper),
                str(Path(__file__).resolve()),
                "--fixture-child",
                mode,
                str(response),
            ],
            cwd=workspace,
            response_output=response,
        )
        return captured

    def test_fake_process_capture_and_closed_classifier_domain(self):
        expected = {
            "exit1_stderr": BrokerFailureClassV1.PROCESS_EXIT_FAILURE,
            "exit1_stdout": BrokerFailureClassV1.PROCESS_EXIT_FAILURE,
            "timeout": BrokerFailureClassV1.TIMEOUT,
            "signal": BrokerFailureClassV1.PROCESS_EXIT_FAILURE,
            "empty": BrokerFailureClassV1.EMPTY_RESPONSE,
            "malformed": BrokerFailureClassV1.MALFORMED_EVENT_STREAM,
            "schema_invalid": BrokerFailureClassV1.SCHEMA_VALIDATION_FAILURE,
            "success": BrokerFailureClassV1.SUCCESS,
            "oversized": BrokerFailureClassV1.PROCESS_EXIT_FAILURE,
            "invalid_utf8": BrokerFailureClassV1.PROCESS_EXIT_FAILURE,
        }
        for mode, failure_class in expected.items():
            with self.subTest(mode=mode):
                captured = self.invoke(
                    mode, timeout_ms=100 if mode == "timeout" else 2_000
                )
                self.assertEqual(
                    captured.outcome.failure_class, failure_class.value
                )
                self.assertEqual(captured.start.spawn_attempt_ordinal, 1)
                self.assertEqual(
                    captured.receipt.start_record_digest,
                    captured.start.start_record_digest,
                )
                self.assertTrue(
                    Path(
                        self.root
                        / f"private-{mode}"
                        / "calls"
                        / captured.request.envelope_digest
                        / "exit_receipt.json"
                    ).is_file()
                )
        oversized = self.invoke("oversized")
        self.assertTrue(oversized.receipt.stdout_truncated)
        self.assertGreater(oversized.receipt.stdout_size_bytes, 1_048_576)
        invalid = self.invoke("invalid_utf8")
        self.assertIn("\ufffd", invalid.outcome.redacted_excerpt or "")
        signaled = self.invoke("signal")
        self.assertEqual(signaled.receipt.termination_signal_or_NONE, signal.SIGTERM)

    def test_spawn_failure_is_captured_without_retry(self):
        runner, workspace = self.runner("spawn")
        self.wrapper.chmod(0o644)
        response = self.root / "spawn.response.json"
        captured = runner.execute(
            logical_call_id="fixture-spawn",
            proposal_generation_session_id="session-spawn",
            prompt="fixed diagnostic",
            argv=[str(self.wrapper)],
            cwd=workspace,
            response_output=response,
        )
        self.assertEqual(
            captured.outcome.failure_class,
            BrokerFailureClassV1.SPAWN_FAILURE.value,
        )
        self.assertFalse(captured.receipt.spawn_succeeded)

    def test_public_excerpt_redacts_credentials_deterministically(self):
        payload = (
            b"Authorization: Bearer private-token "
            b"api_key=private-key password=private-password sk-private12345"
        )
        first = redact_excerpt(payload)
        second = redact_excerpt(payload)
        self.assertEqual(first, second)
        self.assertNotIn("private-token", first or "")
        self.assertNotIn("private-key", first or "")
        self.assertNotIn("private-password", first or "")
        self.assertNotIn("sk-private12345", first or "")

    def test_package_owned_release_matches_current_exact_policy(self):
        release_path = (
            SRC
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "resources"
            / "broker_process_release_v2.json"
        )
        frozen = json.loads(release_path.read_text(encoding="utf-8"))
        schema_path = (
            release_path.parent / "pilot_proposal_response_v1.schema.json"
        )
        self.assertEqual(
            hashlib.sha256(schema_path.read_bytes()).hexdigest(),
            frozen["response_schema_digest"],
        )
        computed = BrokerProcessReleaseV2.create(
            executable=Path(frozen["broker_executable_path"]),
            cli_version=frozen["broker_cli_version"],
            login_mode=frozen["login_mode"],
            model=frozen["model"],
            reasoning_effort=frozen["reasoning_effort"],
            sandbox_mode=frozen["sandbox_mode"],
            response_schema_digest=frozen["response_schema_digest"],
            timeout_ms=900_000,
            workspace=(
                Path("/mnt/c/Users/gtrho/AppData/Local/Temp")
                / "RecClawM6FBroker"
                / "workspace"
            ),
        )
        self.assertEqual(computed.to_dict(), frozen)

    def test_substituted_executable_bytes_fail_before_start(self):
        runner, workspace = self.runner("substitute")
        self.wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "executable bytes"):
            runner.execute(
                logical_call_id="fixture-substitute",
                proposal_generation_session_id="session-substitute",
                prompt="fixed diagnostic",
                argv=[str(self.wrapper)],
                cwd=workspace,
                response_output=self.root / "substitute.response.json",
            )
        self.assertFalse((self.root / "private-substitute" / "calls").exists())

    def test_restart_closes_start_and_spawn_ambiguity_without_respawn(self):
        for boundary in ("start_record.json", "exit_receipt.json", "outcome.json"):
            with self.subTest(boundary=boundary):
                name = "restart-" + boundary.split(".")[0]
                runner, workspace = self.runner(name)
                response = self.root / f"{name}.response.json"
                argv = [
                    str(self.wrapper),
                    str(Path(__file__).resolve()),
                    "--fixture-child",
                    "success",
                    str(response),
                ]
                from recclaw_core.experiments.helix_abc_v1 import broker_process

                original = broker_process._write_durable_json
                tripped = False

                def crash_after_write(path, value):
                    nonlocal tripped
                    original(path, value)
                    if path.name == boundary and not tripped:
                        tripped = True
                        raise RuntimeError("simulated crash")

                with mock.patch.object(
                    broker_process,
                    "_write_durable_json",
                    side_effect=crash_after_write,
                ):
                    with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                        runner.execute(
                            logical_call_id=name,
                            proposal_generation_session_id=name + "-session",
                            prompt="fixed diagnostic",
                            argv=argv,
                            cwd=workspace,
                            response_output=response,
                        )
                recovered = runner.execute(
                    logical_call_id=name,
                    proposal_generation_session_id=name + "-session",
                    prompt="fixed diagnostic",
                    argv=argv,
                    cwd=workspace,
                    response_output=response,
                )
                self.assertEqual(recovered.start.spawn_attempt_ordinal, 1)
                if boundary == "start_record.json":
                    self.assertEqual(
                        recovered.outcome.failure_class,
                        BrokerFailureClassV1.UNKNOWN_PROCESS_FAILURE.value,
                    )
                else:
                    self.assertEqual(
                        recovered.outcome.failure_class,
                        BrokerFailureClassV1.SUCCESS.value,
                    )

    def test_restart_after_spawn_before_receipt_never_respawns(self):
        runner, workspace = self.runner("restart-spawn")
        response = self.root / "restart-spawn.response.json"
        argv = [
            str(self.wrapper),
            str(Path(__file__).resolve()),
            "--fixture-child",
            "success",
            str(response),
        ]
        completed = runner.execute(
            logical_call_id="restart-spawn",
            proposal_generation_session_id="restart-spawn-session",
            prompt="fixed diagnostic",
            argv=argv,
            cwd=workspace,
            response_output=response,
        )
        call_root = (
            self.root
            / "private-restart-spawn"
            / "calls"
            / completed.request.envelope_digest
        )
        (call_root / "exit_receipt.json").unlink()
        (call_root / "outcome.json").unlink()
        recovered = runner.execute(
            logical_call_id="restart-spawn",
            proposal_generation_session_id="restart-spawn-session",
            prompt="fixed diagnostic",
            argv=argv,
            cwd=workspace,
            response_output=response,
        )
        self.assertEqual(
            recovered.outcome.failure_class,
            BrokerFailureClassV1.UNKNOWN_PROCESS_FAILURE.value,
        )
        self.assertEqual(recovered.start.spawn_attempt_ordinal, 1)

    def test_broker_v2_persists_complete_success_receipts_and_replays(self):
        fake_codex = self.root / "fake-codex"
        broker_root = self.root / "broker-v2"
        spawn_count_path = self.root / "success-spawn-count"
        expected_output = (
            broker_root
            / "outputs"
            / (
                sha256_digest({"logical_call_id": "success-call"})
                + ".json"
            )
        )
        fake_codex.write_text(
            f"""#!{sys.executable}
import json, pathlib, sys
if '--version' in sys.argv:
    print('fixture-codex 2')
    raise SystemExit(0)
count = pathlib.Path({str(spawn_count_path)!r})
count.write_text(count.read_text() + 'x' if count.exists() else 'x')
path = pathlib.Path({str(expected_output)!r})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('{{"proposals":[]}}', encoding='utf-8')
print(json.dumps({{"type":"turn.completed","input_tokens":1,"output_tokens":1,"total_tokens":2}}))
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        schema_path = self.root / "schema.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        broker = CodexCliCanaryBrokerV1(
            broker_root,
            schema_path=schema_path,
            codex_executable=fake_codex,
            model="fixture",
            reasoning_effort="none",
            service_tier="default",
            max_total_tokens_per_call=10,
            timeout_ms=2_000,
        )
        try:
            first = broker.call_with_session(
                logical_call_id="success-call",
                proposal_generation_session_id="success-session",
                prompt="fixed",
                expected_proposal_count=0,
            )
            replay = broker.call_with_session(
                logical_call_id="success-call",
                proposal_generation_session_id="success-session",
                prompt="fixed",
                expected_proposal_count=0,
            )
            receipt, outcome = broker.conformance_evidence("success-call")
            self.assertEqual(first, replay)
            self.assertEqual(outcome.status, "SUCCESS")
            self.assertEqual(receipt.exit_code_or_NONE, 0)
            self.assertEqual(broker.call_count(), 1)
            broker._connection.execute(
                "DELETE FROM calls WHERE logical_call_id='success-call'"
            )
            broker._connection.commit()
            with self.assertRaisesRegex(
                (CanaryBrokerError, ValueError),
                "conflict|exist|replay",
            ):
                broker.call_with_session(
                    logical_call_id="success-call",
                    proposal_generation_session_id="success-session",
                    prompt="different request",
                    expected_proposal_count=0,
                )
            recovered = broker.call_with_session(
                logical_call_id="success-call",
                proposal_generation_session_id="success-session",
                prompt="fixed",
                expected_proposal_count=0,
            )
            recovered_receipt, recovered_outcome = broker.conformance_evidence(
                "success-call"
            )
            self.assertEqual(first, recovered)
            self.assertEqual(receipt, recovered_receipt)
            self.assertEqual(outcome, recovered_outcome)
            self.assertEqual(spawn_count_path.read_text(), "x")
            self.assertEqual(broker.call_count(), 1)
        finally:
            broker.close()

    def test_research_session_failure_keeps_prior_call_resource_debits(self):
        class FailingUpstream:
            def __init__(self):
                self.count = 0

            def call_with_session(self, **_kwargs):
                self.count += 1
                if self.count == 3:
                    raise CanaryBrokerError(
                        "fixture failure",
                        physical_call_count=1,
                        input_tokens=3,
                        output_tokens=4,
                        billed_tokens=7,
                        wall_time_ms=11,
                    )
                return CanaryBrokerCallV1(
                    logical_call_id=f"prior-{self.count}",
                    request_digest="1" * 64,
                    response_digest=("%064x" % self.count),
                    response={"proposals": []},
                    input_tokens=10,
                    cached_input_tokens=0,
                    output_tokens=5,
                    total_tokens=15,
                    latency_ms=20,
                    returned_model="fixture",
                )

        broker = RealCanaryProposalBrokerV1.create(
            upstream=FailingUpstream(),
            template_path=(
                ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
            ),
        )
        with self.assertRaises(CanaryBrokerError) as raised:
            broker._research_upstream_calls(
                arm=ArmCode.B, round_index=1, search_seed=99
            )
        error = raised.exception
        self.assertEqual(error.physical_call_count, 3)
        self.assertEqual(error.input_tokens, 23)
        self.assertEqual(error.output_tokens, 14)
        self.assertEqual(error.billed_tokens, 37)
        self.assertEqual(error.wall_time_ms, 51)

    def test_broker_v2_failure_row_keeps_receipt_and_never_retries(self):
        fake_codex = self.root / "failing-codex"
        spawn_count_path = self.root / "failure-spawn-count"
        fake_codex.write_text(
            f"""#!{sys.executable}
import pathlib, sys
if '--version' in sys.argv:
    print('fixture-codex 2')
    raise SystemExit(0)
count = pathlib.Path({str(spawn_count_path)!r})
count.write_text(count.read_text() + 'x' if count.exists() else 'x')
print('fixture process failure', file=sys.stderr)
raise SystemExit(1)
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        schema_path = self.root / "failure-schema.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        broker = CodexCliCanaryBrokerV1(
            self.root / "broker-failure-v2",
            schema_path=schema_path,
            codex_executable=fake_codex,
            model="fixture",
            reasoning_effort="none",
            service_tier="default",
            max_total_tokens_per_call=10,
            timeout_ms=2_000,
        )
        try:
            arguments = {
                "logical_call_id": "failure-call",
                "proposal_generation_session_id": "failure-session",
                "prompt": "fixed",
                "expected_proposal_count": 0,
            }
            with self.assertRaises(CanaryBrokerError) as first:
                broker.call_with_session(**arguments)
            with self.assertRaises(CanaryBrokerError) as replay:
                broker.call_with_session(**arguments)
            self.assertEqual(
                first.exception.outcome, replay.exception.outcome
            )
            self.assertEqual(
                first.exception.receipt, replay.exception.receipt
            )
            self.assertEqual(first.exception.physical_call_count, 1)
            broker._connection.execute(
                "DELETE FROM calls WHERE logical_call_id='failure-call'"
            )
            broker._connection.commit()
            with self.assertRaisesRegex(ValueError, "logical call.*conflict"):
                broker.call_with_session(
                    **{**arguments, "prompt": "different request"}
                )
            with self.assertRaises(CanaryBrokerError) as recovered:
                broker.call_with_session(**arguments)
            self.assertEqual(first.exception.outcome, recovered.exception.outcome)
            self.assertEqual(first.exception.receipt, recovered.exception.receipt)
            self.assertEqual(spawn_count_path.read_text(), "x")
            connection = sqlite3.connect(broker.db_path)
            try:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM calls"
                    ).fetchone()[0]
                )
            finally:
                connection.close()
            self.assertEqual(count, 1)
        finally:
            broker.close()

    def test_broker_v2_response_failure_recovers_row_without_respawn(self):
        fake_codex = self.root / "semantic-failure-codex"
        broker_root = self.root / "broker-semantic-failure-v2"
        spawn_count_path = self.root / "semantic-failure-spawn-count"
        expected_output = (
            broker_root
            / "outputs"
            / (
                sha256_digest({"logical_call_id": "semantic-failure-call"})
                + ".json"
            )
        )
        fake_codex.write_text(
            f"""#!{sys.executable}
import json, pathlib, sys
if '--version' in sys.argv:
    print('fixture-codex 2')
    raise SystemExit(0)
count = pathlib.Path({str(spawn_count_path)!r})
count.write_text(count.read_text() + 'x' if count.exists() else 'x')
path = pathlib.Path({str(expected_output)!r})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('{{"proposals":[{{}}]}}', encoding='utf-8')
print(json.dumps({{"type":"turn.completed","input_tokens":1,"output_tokens":1,"total_tokens":2}}))
""",
            encoding="utf-8",
        )
        fake_codex.chmod(0o755)
        schema_path = self.root / "semantic-failure-schema.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        broker = CodexCliCanaryBrokerV1(
            broker_root,
            schema_path=schema_path,
            codex_executable=fake_codex,
            model="fixture",
            reasoning_effort="none",
            service_tier="default",
            max_total_tokens_per_call=10,
            timeout_ms=2_000,
        )
        arguments = {
            "logical_call_id": "semantic-failure-call",
            "proposal_generation_session_id": "semantic-failure-session",
            "prompt": "fixed",
            "expected_proposal_count": 0,
        }
        try:
            with self.assertRaises(CanaryBrokerError) as first:
                broker.call_with_session(**arguments)
            broker._connection.execute(
                "DELETE FROM calls WHERE logical_call_id='semantic-failure-call'"
            )
            broker._connection.commit()
            with self.assertRaisesRegex(ValueError, "logical call.*conflict"):
                broker.call_with_session(
                    **{**arguments, "expected_proposal_count": 1}
                )
            with self.assertRaises(CanaryBrokerError) as recovered:
                broker.call_with_session(**arguments)
            self.assertEqual(first.exception.outcome, recovered.exception.outcome)
            self.assertEqual(first.exception.receipt, recovered.exception.receipt)
            self.assertEqual(spawn_count_path.read_text(), "x")
            self.assertEqual(
                broker._connection.execute(
                    "SELECT COUNT(*) FROM calls"
                ).fetchone()[0],
                1,
            )
        finally:
            broker.close()


class BrokerFailureClosureAndAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = SingleWriterExperimentStoreV1(
            self.root / "state.sqlite3", self.root / "artifacts"
        )
        self.contract = default_experiment_contract()
        self.arm_ids = self.store.initialize_experiment(self.contract)
        self.budget = ResourceCeilingsV1(
            total_input_tokens=100,
            total_output_tokens=100,
            total_billed_token_debit=200,
            total_proposal_count=8,
            wall_time_ms=10_000,
            retry_debit=0,
            proposal_attempt_debit=8,
            ordinary_executions=1,
            common_validation_count=8,
            gpu_device_time_ms=10_000,
            gpu_cost_microunits=10_000,
        )
        self.genesis = sha256_digest(
            {
                "experiment_contract_digest": self.contract.identity_digest,
                "state": "GENESIS",
            }
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    @staticmethod
    def failure_records():
        from recclaw_core.experiments.helix_abc_v1.broker_process import (
            BrokerProcessExitReceiptV2,
        )

        receipt_payload = {
            "exit_code_or_NONE": 1,
            "latency_ms": 10,
            "monotonic_end_ns": 2,
            "pid_or_private_process_ref": "private-process:test",
            "provider_request_confirmation": "NOT_OBSERVED",
            "returned_model_or_NONE": None,
            "spawn_succeeded": True,
            "start_record_digest": "1" * 64,
            "stderr_artifact_ref": "broker-private:test/stderr.bin",
            "stderr_sha256": "2" * 64,
            "stderr_size_bytes": 12,
            "stderr_truncated": False,
            "stdout_artifact_ref": "broker-private:test/stdout.bin",
            "stdout_sha256": "3" * 64,
            "stdout_size_bytes": 0,
            "stdout_truncated": False,
            "termination_signal_or_NONE": None,
            "timed_out": False,
        }
        receipt = BrokerProcessExitReceiptV2(
            **receipt_payload, receipt_digest=sha256_digest(receipt_payload)
        )
        outcome_payload = {
            "classifier_rule_id": "M6F_CLASSIFIER_EXIT_CODE_ONLY_V1",
            "failure_class": "PROCESS_EXIT_FAILURE",
            "logical_call_id": "private-logical-call",
            "proposal_generation_session_id": "private-session",
            "receipt_digest": receipt.receipt_digest,
            "redacted_excerpt": "fixture failure",
            "request_envelope_digest": "4" * 64,
            "response_digest": None,
            "status": "PROCESS_FAILURE",
            "supporting_artifact_ref": receipt.stderr_artifact_ref,
        }
        outcome = BrokerCallOutcomeV2(
            **outcome_payload, outcome_digest=sha256_digest(outcome_payload)
        )
        return receipt, outcome

    def test_failure_closure_is_exactly_once_and_consumes_no_execution(self):
        opened = self.store.open_round(
            OpenRoundCommand(
                experiment_id=self.contract.experiment_id,
                arm_instance_id=self.arm_ids[ArmCode.A],
                arm_code=ArmCode.A,
                search_seed=42,
                round_index=1,
                budget_snapshot=self.budget,
                controller_state_before_digest=self.genesis,
                idempotency_key="m6f:open",
            )
        )
        receipt, outcome = self.failure_records()
        args = {
            "store": self.store,
            "experiment_id": self.contract.experiment_id,
            "search_seed": 42,
            "round_index": 1,
            "round_id": opened["round_id"],
            "controller_state_digest": self.genesis,
            "ceilings": self.budget,
            "receipt": receipt,
            "outcome": outcome,
            "physical_call_count": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "billed_tokens": 0,
            "wall_time_ms": receipt.latency_ms,
        }
        first = close_broker_failure(**args)
        replay = close_broker_failure(**args)
        self.assertEqual(first, replay)
        connection = sqlite3.connect(self.store.db_path)
        try:
            terminal = connection.execute(
                "SELECT terminal_class FROM rounds WHERE round_id=?",
                (opened["round_id"],),
            ).fetchone()[0]
            sessions = connection.execute(
                "SELECT SUM(quantity) FROM resource_ledger "
                "WHERE round_id=? AND dimension='PROPOSAL_GENERATION_SESSION'",
                (opened["round_id"],),
            ).fetchone()[0]
            execution_claims = connection.execute(
                "SELECT COUNT(*) FROM execution_claims"
            ).fetchone()[0]
            stopped = connection.execute(
                "SELECT COUNT(*) FROM arm_state WHERE state='STOPPED'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(terminal, "BROKER_PROCESS_FAILURE")
        self.assertEqual(sessions, 1)
        self.assertEqual(execution_claims, 0)
        self.assertEqual(stopped, 3)
        changed_payload = outcome.to_dict()
        changed_payload["failure_class"] = "UNKNOWN_PROCESS_FAILURE"
        changed_payload.pop("outcome_digest")
        changed = BrokerCallOutcomeV2(
            **changed_payload, outcome_digest=sha256_digest(changed_payload)
        )
        with self.assertRaises(IdempotencyConflict):
            close_broker_failure(**{**args, "outcome": changed})

    def test_orchestrator_process_failure_closes_before_guard_or_execution(self):
        receipt, outcome = self.failure_records()

        class FailingBroker:
            def generate(self, **_kwargs):
                raise CanaryBrokerError(
                    "fixture process failure",
                    outcome=outcome,
                    receipt=receipt,
                    physical_call_count=1,
                    wall_time_ms=receipt.latency_ms,
                )

        with ThreeArmPreCanaryOrchestratorV1(
            self.root / "orchestrator", broker=FailingBroker()
        ) as orchestrator:
            with self.assertRaises(BrokerRoundFailureError) as raised:
                orchestrator.run_fake_triplet(
                    search_seed=42, round_index=1, drafts=()
                )
            self.assertEqual(
                raised.exception.closure.round_terminal_class,
                "BROKER_PROCESS_FAILURE",
            )
            self.assertEqual(orchestrator.guard_ledger.count(), 0)
            connection = sqlite3.connect(orchestrator.store.db_path)
            try:
                terminal_rows = connection.execute(
                    "SELECT terminal_class, COUNT(*) FROM rounds "
                    "GROUP BY terminal_class"
                ).fetchall()
                execution_claim_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM execution_claims"
                    ).fetchone()[0]
                )
                session_debit = int(
                    connection.execute(
                        "SELECT SUM(quantity) FROM resource_ledger "
                        "WHERE dimension='PROPOSAL_GENERATION_SESSION'"
                    ).fetchone()[0]
                )
                stopped_slots = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM scheduled_slots "
                        "WHERE slot_status='NOT_STARTED_STOP'"
                    ).fetchone()[0]
                )
            finally:
                connection.close()
            self.assertEqual(
                terminal_rows, [("BROKER_PROCESS_FAILURE", 1)]
            )
            self.assertEqual(execution_claim_count, 0)
            self.assertEqual(session_debit, 1)
            self.assertGreater(stopped_slots, 0)

    def test_immutable_snapshots_create_no_sidecars_and_neutralize_association(self):
        state_snapshot = self.root / "snapshots" / "state.sqlite3"
        with self.store._lock:
            state_manifest = create_immutable_audit_snapshot(
                writer_connection=self.store._connection,
                source_db_path=self.store.db_path,
                snapshot_path=state_snapshot,
                source_schema_identity=self.store.migration_sha256,
                audit_purpose="TEST_STATE",
            )
        broker_path = self.root / "broker.sqlite3"
        broker = sqlite3.connect(broker_path)
        broker.execute(
            "CREATE TABLE calls(status TEXT, error_type TEXT, "
            "exit_receipt_digest TEXT)"
        )
        broker.execute(
            "INSERT INTO calls VALUES ('FAILED','PROCESS_EXIT_FAILURE',?)",
            ("a" * 64,),
        )
        broker.commit()
        broker_snapshot = self.root / "snapshots" / "broker.sqlite3"
        broker_manifest = create_immutable_audit_snapshot(
            writer_connection=broker,
            source_db_path=broker_path,
            snapshot_path=broker_snapshot,
            source_schema_identity="broker-test-v2",
            audit_purpose="TEST_BROKER",
        )
        guard_path = self.root / "guard.sqlite3"
        guard = sqlite3.connect(guard_path)
        guard.execute("CREATE TABLE guard_calls(guard_call_id TEXT)")
        guard.commit()
        guard_snapshot = self.root / "snapshots" / "guard.sqlite3"
        guard_manifest = create_immutable_audit_snapshot(
            writer_connection=guard,
            source_db_path=guard_path,
            snapshot_path=guard_snapshot,
            source_schema_identity="guard-test-v1",
            audit_purpose="TEST_GUARD",
        )
        broker.close()
        guard.close()
        for path, manifest in (
            (state_snapshot, state_manifest),
            (broker_snapshot, broker_manifest),
            (guard_snapshot, guard_manifest),
        ):
            verification = verify_immutable_snapshot(path, manifest)
            self.assertEqual(verification["integrity_check"], "ok")
            self.assertEqual(verification["sidecars_created"], 0)
            self.assertTrue(verification["sha256_match"])
        projection = association_free_neutral_audit(
            state_snapshot=state_snapshot,
            broker_snapshot=broker_snapshot,
            guard_snapshot=guard_snapshot,
        )
        encoded = json.dumps(projection).lower()
        for forbidden in (
            "arm_code",
            "arm_instance",
            "assignment_key",
            "candidate_id",
            "ndcg",
            "treatment",
        ):
            self.assertNotIn(forbidden, encoded)


if __name__ == "__main__" and "--fixture-child" in sys.argv:
    position = sys.argv.index("--fixture-child")
    raise SystemExit(
        _fixture_child(sys.argv[position + 1], Path(sys.argv[position + 2]))
    )
