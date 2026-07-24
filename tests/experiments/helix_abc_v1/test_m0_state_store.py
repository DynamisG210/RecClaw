from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from recclaw_core.experiments.helix_abc_v1 import (  # noqa: E402
    ArmCode,
    ResourceCeilingsV1,
    SingleWriterExperimentStoreV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    ClaimExecutionCommand,
    CloseRoundCommand,
    ConservativeRecoveryCommand,
    EXPECTED_TABLES,
    IdempotencyConflict,
    InvariantViolation,
    OpenRoundCommand,
    RegisterArtifactCommand,
    ResourceDebitV1,
    StopAndFillCommand,
)


class M0StateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.db_path = root / "state" / "experiment.sqlite3"
        self.artifact_root = root / "artifacts"
        self.store = SingleWriterExperimentStoreV1(
            self.db_path,
            self.artifact_root,
        )
        self.contract = default_experiment_contract()
        self.arm_ids = self.store.initialize_experiment(self.contract)
        self.budget = ResourceCeilingsV1(
            total_input_tokens=100,
            total_output_tokens=100,
            total_billed_token_debit=200,
            total_proposal_count=8,
            wall_time_ms=10000,
            retry_debit=2,
            proposal_attempt_debit=8,
            ordinary_executions=1,
            common_validation_count=8,
            gpu_device_time_ms=10000,
            gpu_cost_microunits=10000,
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

    def open_command(
        self,
        arm: ArmCode,
        *,
        seed: int = 42,
        index: int = 1,
        key: str | None = None,
        controller_digest: str | None = None,
    ) -> OpenRoundCommand:
        return OpenRoundCommand(
            experiment_id=self.contract.experiment_id,
            arm_instance_id=self.arm_ids[arm],
            arm_code=arm,
            search_seed=seed,
            round_index=index,
            budget_snapshot=self.budget,
            controller_state_before_digest=controller_digest or self.genesis,
            idempotency_key=key or f"open:{seed}:{index}:{arm.value}",
        )

    def close_command(
        self,
        round_id: str,
        *,
        key: str | None = None,
        terminal_class: str = "COMPLETED",
    ) -> CloseRoundCommand:
        return CloseRoundCommand(
            round_id=round_id,
            terminal_class=terminal_class,
            feedback_payload={
                "outcome_class": terminal_class,
                "search_update": "DEVELOPMENT_ONLY",
            },
            controller_state_after_digest=sha256_digest(
                {"round_id": round_id, "state": "AFTER"}
            ),
            resource_debits=(
                ResourceDebitV1("PROPOSAL_ATTEMPT", 1),
                ResourceDebitV1("PROPOSAL", 1),
            ),
            idempotency_key=key or f"close:{round_id}",
        )

    def test_migration_pragmas_integrity_and_exact_eight_tables(self) -> None:
        report = self.store.integrity_report()

        self.assertEqual(report["tables"], sorted(EXPECTED_TABLES))
        self.assertEqual(len(report["tables"]), 8)
        self.assertEqual(report["journal_mode"], "wal")
        self.assertEqual(report["synchronous"], 2)
        self.assertEqual(report["foreign_keys"], 1)
        self.assertEqual(report["user_version"], 1)
        self.assertEqual(report["integrity_check"], "ok")
        self.assertEqual(report["foreign_key_violations"], [])

        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO execution_claims (
                    round_id, claim_id, idempotency_key, payload_digest,
                    permit_digest, binding_digest, claim_state
                ) VALUES (?, ?, ?, ?, ?, ?, 'CLAIMED')
                """,
                ("missing", "claim", "key", "0" * 64, "1" * 64, "2" * 64),
            )
        connection.close()

    def test_open_is_create_once_and_payload_conflicts(self) -> None:
        command = self.open_command(ArmCode.A)
        first = self.store.open_round(command)
        replay = self.store.open_round(command)

        self.assertEqual(first, replay)
        with self.assertRaises(IdempotencyConflict):
            self.store.open_round(
                self.open_command(
                    ArmCode.A,
                    key=command.idempotency_key,
                    controller_digest="f" * 64,
                )
            )
        with self.assertRaisesRegex(InvariantViolation, "PLANNED"):
            self.store.open_round(self.open_command(ArmCode.A, key="other-open"))

        connection = sqlite3.connect(self.db_path)
        session_count = connection.execute(
            "SELECT SUM(quantity) FROM resource_ledger "
            "WHERE round_id = ? AND dimension = 'PROPOSAL_GENERATION_SESSION'",
            (first["round_id"],),
        ).fetchone()[0]
        connection.close()
        self.assertEqual(session_count, 1)

    def test_initial_research_memory_is_equal_and_open_binds_committed_state(self) -> None:
        connection = sqlite3.connect(self.db_path)
        memory_rows = connection.execute(
            """
            SELECT arm_code, search_memory_digest FROM arm_state
            WHERE experiment_id = ? AND search_seed = 42
            ORDER BY arm_code
            """,
            (self.contract.experiment_id,),
        ).fetchall()
        connection.close()

        self.assertIsNone(memory_rows[0][1])
        self.assertEqual(memory_rows[1][1], memory_rows[2][1])
        with self.assertRaisesRegex(InvariantViolation, "committed Arm state"):
            self.store.open_round(
                self.open_command(
                    ArmCode.A,
                    key="open:wrong-before-root",
                    controller_digest="f" * 64,
                )
            )

    def test_one_execution_claim_per_round_and_idempotent_replay(self) -> None:
        round_row = self.store.open_round(self.open_command(ArmCode.A))
        command = ClaimExecutionCommand(
            round_id=round_row["round_id"],
            permit_digest="1" * 64,
            binding_digest="2" * 64,
            idempotency_key="claim:a",
        )

        first = self.store.claim_execution(command)
        self.assertEqual(first, self.store.claim_execution(command))
        with self.assertRaisesRegex(InvariantViolation, "one execution claim"):
            self.store.claim_execution(
                ClaimExecutionCommand(
                    round_id=round_row["round_id"],
                    permit_digest="3" * 64,
                    binding_digest="4" * 64,
                    idempotency_key="claim:a:second",
                )
            )

    def test_concurrent_execution_claims_have_exactly_one_winner(self) -> None:
        round_row = self.store.open_round(self.open_command(ArmCode.A))
        commands = (
            ClaimExecutionCommand(
                round_id=round_row["round_id"],
                permit_digest="1" * 64,
                binding_digest="2" * 64,
                idempotency_key="claim:race:one",
            ),
            ClaimExecutionCommand(
                round_id=round_row["round_id"],
                permit_digest="3" * 64,
                binding_digest="4" * 64,
                idempotency_key="claim:race:two",
            ),
        )

        def attempt(command: ClaimExecutionCommand) -> str:
            try:
                self.store.claim_execution(command)
            except InvariantViolation:
                return "REJECTED"
            return "CLAIMED"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, commands))

        self.assertEqual(sorted(outcomes), ["CLAIMED", "REJECTED"])

    def test_one_feedback_and_triplet_barrier(self) -> None:
        rounds = {
            arm: self.store.open_round(self.open_command(arm))
            for arm in ArmCode
        }
        close_a = self.close_command(rounds[ArmCode.A]["round_id"])
        self.store.close_round(close_a)
        self.assertEqual(self.store.close_round(close_a)["status"], "CLOSED")
        self.store.close_round(self.close_command(rounds[ArmCode.B]["round_id"]))

        with self.assertRaisesRegex(InvariantViolation, "triplet barrier"):
            self.store.open_round(
                self.open_command(
                    ArmCode.A,
                    index=2,
                    controller_digest=close_a.controller_state_after_digest,
                )
            )

        self.store.close_round(self.close_command(rounds[ArmCode.C]["round_id"]))
        next_round = self.store.open_round(
            self.open_command(
                ArmCode.A,
                index=2,
                controller_digest=close_a.controller_state_after_digest,
            )
        )
        self.assertEqual(next_round["round_index"], 2)

        connection = sqlite3.connect(self.db_path)
        feedback_counts = connection.execute(
            """
            SELECT round_id, COUNT(*) FROM round_events
            WHERE event_type = 'ROUND_FEEDBACK'
            GROUP BY round_id
            """
        ).fetchall()
        connection.close()
        self.assertEqual(sorted(count for _, count in feedback_counts), [1, 1, 1])

    def test_stop_fills_planned_slots_and_prevents_future_open(self) -> None:
        opened = self.store.open_round(self.open_command(ArmCode.A, seed=43))
        command = StopAndFillCommand(
            experiment_id=self.contract.experiment_id,
            search_seed=43,
            current_round_index=1,
            reason="M0_FIXTURE_STOP",
            idempotency_key="stop:43",
        )

        first = self.store.stop_and_fill_remaining(command)
        self.assertEqual(first, self.store.stop_and_fill_remaining(command))
        self.assertEqual(first["slot_status_counts"]["NOT_STARTED_STOP"], 149)
        with self.assertRaisesRegex(InvariantViolation, "after stop"):
            self.store.open_round(self.open_command(ArmCode.B, seed=43))

        self.store.close_round(
            self.close_command(
                opened["round_id"],
                terminal_class="ABORTED",
            )
        )
        connection = sqlite3.connect(self.db_path)
        states = {
            row[0]
            for row in connection.execute(
                "SELECT state FROM arm_state WHERE search_seed = 43"
            )
        }
        connection.close()
        self.assertEqual(states, {"STOPPED"})

    def test_conservative_recovery_closes_open_and_ambiguous_claim_rounds(self) -> None:
        rounds = {
            arm: self.store.open_round(self.open_command(arm, seed=44))
            for arm in ArmCode
        }
        self.store.claim_execution(
            ClaimExecutionCommand(
                round_id=rounds[ArmCode.B]["round_id"],
                permit_digest="1" * 64,
                binding_digest="2" * 64,
                idempotency_key="claim:recovery:b",
            )
        )

        command = ConservativeRecoveryCommand(
            experiment_id=self.contract.experiment_id,
            search_seed=44,
            current_round_index=1,
            idempotency_key="recovery:44",
        )
        report = self.store.conservative_recovery(command)

        self.assertEqual(len(report["recovered_rounds"]), 3)
        self.assertEqual(
            self.store.get_execution_claim(rounds[ArmCode.B]["round_id"])[
                "claim_state"
            ],
            "START_AMBIGUOUS",
        )
        connection = sqlite3.connect(self.db_path)
        execution_debit = connection.execute(
            """
            SELECT SUM(quantity) FROM resource_ledger
            WHERE round_id = ? AND dimension = 'ORDINARY_EXECUTION'
            """,
            (rounds[ArmCode.B]["round_id"],),
        ).fetchone()[0]
        feedback_count = connection.execute(
            "SELECT COUNT(*) FROM round_events WHERE event_type = 'ROUND_FEEDBACK'"
        ).fetchone()[0]
        connection.close()
        self.assertEqual(execution_debit, 1)
        self.assertEqual(feedback_count, 3)
        self.assertEqual(self.store.conservative_recovery(command), report)

    def test_atomic_artifact_path_digest_and_idempotency(self) -> None:
        round_row = self.store.open_round(self.open_command(ArmCode.A))
        command = RegisterArtifactCommand(
            round_id=round_row["round_id"],
            artifact_type="M0_FIXTURE",
            relative_path="arm-a/round-1/fixture.json",
            producer="M0_TEST",
            idempotency_key="artifact:a:1",
        )
        payload = b'{"fixture":true}\n'

        first = self.store.register_artifact(command, payload)
        replay = self.store.register_artifact(command, payload)

        self.assertEqual(first, replay)
        self.assertEqual(first["sha256"], bytes_sha256(payload))
        self.assertEqual(
            bytes_sha256((self.artifact_root / first["relative_path"]).read_bytes()),
            first["sha256"],
        )
        with self.assertRaises(IdempotencyConflict):
            self.store.register_artifact(command, b'{"fixture":false}\n')
        with self.assertRaises(ValueError):
            self.store.register_artifact(
                RegisterArtifactCommand(
                    round_id=round_row["round_id"],
                    artifact_type="M0_FIXTURE",
                    relative_path="../escape",
                    producer="M0_TEST",
                    idempotency_key="artifact:escape",
                ),
                b"escape",
            )

    def test_resource_ceiling_is_enforced(self) -> None:
        round_row = self.store.open_round(self.open_command(ArmCode.A))
        command = self.close_command(round_row["round_id"])
        command = CloseRoundCommand(
            round_id=command.round_id,
            terminal_class=command.terminal_class,
            feedback_payload=command.feedback_payload,
            controller_state_after_digest=command.controller_state_after_digest,
            resource_debits=(ResourceDebitV1("PROPOSAL", 9),),
            idempotency_key=command.idempotency_key,
        )
        with self.assertRaisesRegex(InvariantViolation, "ceiling"):
            self.store.close_round(command)
        self.assertEqual(self.store.get_round(round_row["round_id"])["status"], "OPEN")

    def test_reinitialization_rejects_a_different_arm_instance_tuple(self) -> None:
        conflicting = dict(self.arm_ids)
        conflicting[ArmCode.A], conflicting[ArmCode.B] = (
            conflicting[ArmCode.B],
            conflicting[ArmCode.A],
        )

        with self.assertRaisesRegex(InvariantViolation, "Arm-instance tuple"):
            self.store.initialize_experiment(
                self.contract,
                arm_instance_ids=conflicting,
            )


if __name__ == "__main__":
    unittest.main()
