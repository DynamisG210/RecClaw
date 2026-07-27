from __future__ import annotations

import copy
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
    ProducerExecutionModeV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    ArmFilesystemCapabilityV1,
    ArmPrivateRootsV1,
    PreCanaryInvariantError,
    PrivateTreatmentAssignmentV1,
    RuntimeLayoutV1,
    ThreeArmFakeBrokerV1,
    ThreeArmPreCanaryOrchestratorV1,
    m4_budget,
    probe_uid_isolation,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (  # noqa: E402
    ProposalIntentV1,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    ClaimExecutionCommand,
    CloseRoundCommand,
    ConservativeRecoveryCommand,
    InvariantViolation,
    OpenRoundCommand,
    ResourceDebitV1,
    SingleWriterExperimentStoreV1,
    StopAndFillCommand,
)
from recclaw_core.helix.composition import SameSlateHelixSelectorV1  # noqa: E402
from recclaw_core.helix.contracts import (  # noqa: E402
    CandidateEnvelope,
    PortAdjudication,
    PortStage,
    PortStatus,
)
from recclaw_core.helix.fusion import DeterministicHelixFusionV1  # noqa: E402
from recclaw_core.helix.scientific_attribution import (  # noqa: E402
    ResearchTaskTypeV1,
)


FIXTURES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"


def program(name: str) -> dict[str, object]:
    document = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return copy.deepcopy(
        next(
            item["program"]
            for item in document["fixtures"]
            if item["anchor_name"] == name
        )
    )


def drafts() -> list[dict[str, object]]:
    specs = (
        ("DIRECTAU", "geometry", ProposalIntentV1.DISCOVERY.value, 0.88),
        ("LIGHTGCN", "propagation", ProposalIntentV1.DISCOVERY.value, 0.84),
        ("SGL", "self_supervision", ProposalIntentV1.FALSIFICATION.value, 0.82),
        ("ULTRAGCN", "architecture", ProposalIntentV1.DISCOVERY.value, 0.86),
    )
    return [
        {
            "mechanism_program": program(name),
            "mechanism_axis": axis,
            "proposal_intent": intent,
            "utility_features": {
                "runnable_probability": 0.9,
                "useful_signal": utility,
                "frontier_potential": utility,
                "information_gain": utility,
                "cost": 0.3,
                "blocker_risk": 0.1,
            },
        }
        for name, axis, intent, utility in specs
    ]


def genesis(contract) -> str:
    return sha256_digest(
        {
            "experiment_contract_digest": contract.identity_digest,
            "state": "GENESIS",
        }
    )


class M4AssignmentAndFilesystemTest(unittest.TestCase):
    def test_assignment_is_opaque_committed_and_neutral(self) -> None:
        assignment = PrivateTreatmentAssignmentV1.create(
            "HELIX-ABC-001", nonce="unit-nonce"
        )
        envelope = assignment.neutral_envelope().to_dict()
        serialized = canonical_json_bytes(envelope).decode("utf-8")
        self.assertEqual(len(set(envelope["opaque_instance_ids"])), 3)
        self.assertEqual(len(envelope["assignment_commitment"]), 64)
        self.assertNotIn('"arm"', serialized.lower())
        self.assertNotIn("evidence", serialized.lower())
        self.assertEqual(
            assignment.commitment,
            PrivateTreatmentAssignmentV1.create(
                "HELIX-ABC-001", nonce="unit-nonce"
            ).commitment,
        )

    def test_layout_has_private_roots_and_only_c_evidence_root(self) -> None:
        assignment = PrivateTreatmentAssignmentV1.create(
            "HELIX-ABC-001", nonce="layout"
        )
        with tempfile.TemporaryDirectory() as raw:
            layout = RuntimeLayoutV1.materialize(Path(raw) / "runtime", assignment)
            roots = [item.root.resolve() for item in layout.arm_roots]
            self.assertEqual(len(set(roots)), 3)
            self.assertTrue(all(layout.neutral_root.resolve() not in root.parents for root in roots))
            self.assertIsNone(layout.evidence_root(assignment.mapping[ArmCode.A]))
            self.assertIsNone(layout.evidence_root(assignment.mapping[ArmCode.B]))
            self.assertIsNotNone(layout.evidence_root(assignment.mapping[ArmCode.C]))

    def test_capability_rejects_traversal_absolute_symlink_and_hardlink(self) -> None:
        assignment = PrivateTreatmentAssignmentV1.create(
            "HELIX-ABC-001", nonce="capability"
        )
        with tempfile.TemporaryDirectory() as raw:
            layout = RuntimeLayoutV1.materialize(Path(raw) / "runtime", assignment)
            capability = ArmFilesystemCapabilityV1(
                layout.arm(assignment.mapping[ArmCode.A])
            )
            capability.write_bytes("candidate", "safe/value.bin", b"safe")
            self.assertEqual(
                capability.read_bytes("candidate", "safe/value.bin"), b"safe"
            )
            for bad in (
                "../memory/leak",
                "../../instances/sibling/raw/data",
                "/proc/self/environ",
            ):
                with self.subTest(path=bad), self.assertRaises(
                    PreCanaryInvariantError
                ):
                    capability.read_bytes("candidate", bad)
            base = layout.arm(assignment.mapping[ArmCode.A]).namespace("candidate")
            (base / "link").symlink_to(base / "safe", target_is_directory=True)
            with self.assertRaises(PreCanaryInvariantError):
                capability.read_bytes("candidate", "link/value.bin")
            os.link(base / "safe" / "value.bin", base / "hard.bin")
            with self.assertRaises(PreCanaryInvariantError):
                capability.read_bytes("candidate", "hard.bin")
            forged = list(capability.roots.namespaces)
            forged[0] = (
                forged[0][0],
                layout.arm(assignment.mapping[ArmCode.B]).namespace("cache"),
            )
            with self.assertRaises(PreCanaryInvariantError):
                ArmPrivateRootsV1(
                    opaque_instance_id=assignment.mapping[ArmCode.A],
                    root=capability.roots.root,
                    namespaces=tuple(forged),
                )

    @unittest.skipUnless(os.geteuid() == 0, "numeric UID isolation requires root")
    def test_kernel_uid_probe_denies_cross_arm_read_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            root.chmod(0o711)
            own = root / "own"
            sibling = root / "sibling"
            own.mkdir()
            sibling.mkdir()
            result = probe_uid_isolation(
                own_root=own, sibling_root=sibling, worker_uid=62001
            )
            self.assertEqual(
                result,
                {
                    "own_read": True,
                    "sibling_read_denied": True,
                    "sibling_write_denied": True,
                },
            )


class M4BrokerAndE2ETest(unittest.TestCase):
    def test_broker_preserves_equal_total_resources_and_physical_call_cost(self) -> None:
        broker = ThreeArmFakeBrokerV1.create()
        sessions = {
            arm: broker.generate(
                arm=arm,
                round_index=1,
                search_seed=42,
                drafts=drafts(),
                ceilings=m4_budget(),
            )
            for arm in ArmCode
        }
        totals = {
            (
                item.input_tokens,
                item.output_tokens,
                item.billed_tokens,
                item.proposal_count,
            )
            for item in sessions.values()
        }
        self.assertEqual(len(totals), 1)
        self.assertEqual(sessions[ArmCode.A].physical_call_count, 1)
        self.assertEqual(sessions[ArmCode.B].physical_call_count, 4)
        self.assertEqual(sessions[ArmCode.C].physical_call_count, 4)
        self.assertEqual(broker.bc_controller_identity_digest, broker.bc_controller_identity_digest)
        self.assertEqual(
            sessions[ArmCode.B].proposal_session_digest,
            sessions[ArmCode.C].proposal_session_digest,
        )
        self.assertEqual(
            sessions[ArmCode.B].route_trace_digest,
            sessions[ArmCode.C].route_trace_digest,
        )

    def test_fake_triplet_closes_one_round_execution_feedback_and_four_axes(self) -> None:
        source_before = sha256_digest(
            {
                path.relative_to(SRC).as_posix(): path.read_bytes().hex()
                for path in sorted(SRC.rglob("*.py"))
                if "__pycache__" not in path.parts
            }
        )
        with tempfile.TemporaryDirectory() as raw, ThreeArmPreCanaryOrchestratorV1(
            Path(raw) / "m4"
        ) as orchestrator:
            results = orchestrator.run_fake_triplet(
                search_seed=42, round_index=1, drafts=drafts()
            )
            replay = orchestrator.run_fake_triplet(
                search_seed=42, round_index=1, drafts=drafts()
            )
            self.assertEqual(results, replay)
            self.assertEqual(len(results), 3)
            self.assertTrue(all(item.terminal_class == "COMPLETED" for item in results))
            self.assertTrue(all(item.ordinary_execution_count == 1 for item in results))

            self.assertTrue(all(not item.training_backend_started for item in results))
            self.assertEqual(
                [item.evidence_port_status for item in results],
                ["NOT_ADJUDICATED", "NOT_ADJUDICATED", "ADJUDICATED"],
            )
            self.assertEqual([item.physical_call_count for item in results], [1, 4, 4])
            neutral = orchestrator.neutral_audit_projection(results)
            neutral_text = canonical_json_bytes(neutral).decode("utf-8").lower()
            for forbidden in (
                "arm",
                "controller",
                "evidence_port",
                "guard",
                "physical_call",
                "producer",
                "research",
                "treatment",
            ):
                self.assertNotIn(forbidden, neutral_text)
            self.assertTrue(neutral["triplet_closed"])
            axes = orchestrator.four_axis_totals(results)
            self.assertEqual(len(axes), 3)
            self.assertTrue(
                all(
                    set(value)
                    == {
                        "round_count",
                        "execution_count",
                        "token_count",
                        "gpu_cost_microunits",
                    }
                    for value in axes.values()
                )
            )
            self.assertEqual(
                {value["execution_count"] for value in axes.values()}, {1}
            )
            self.assertEqual(
                {value["token_count"] for value in axes.values()}, {400}
            )
            packet = orchestrator.seal_canary_review_packet(
                Path(raw) / "sealed-review", results
            )
            self.assertEqual(packet["verdict"], "READY_FOR_CANARY")
            with self.assertRaises(PreCanaryInvariantError):
                orchestrator.seal_canary_review_packet(
                    Path(raw) / "false-ready", ()
                )
            self.assertEqual(
                (Path(raw) / "sealed-review" / "sealed_treatment_mapping.json")
                .stat()
                .st_mode
                & 0o777,
                0o600,
            )
            neutral_packet = (
                Path(raw) / "sealed-review" / "neutral_audit_projection.json"
            ).read_text(encoding="utf-8")
            self.assertNotIn('"A"', neutral_packet)
            self.assertNotIn('"B"', neutral_packet)
            self.assertNotIn('"C"', neutral_packet)
            db = sqlite3.connect(orchestrator.store.db_path)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM rounds").fetchone()[0], 3)
            self.assertEqual(
                db.execute(
                    "SELECT COUNT(*) FROM round_events WHERE event_type='ROUND_CLOSED'"
                ).fetchone()[0],
                3,
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM execution_claims").fetchone()[0],
                3,
            )
            barrier = db.execute(
                "SELECT closed_bitmap, next_index_authorized FROM triplet_barrier "
                "WHERE search_seed=42 AND round_index=1"
            ).fetchone()
            db.close()
            self.assertEqual(barrier, (7, 1))
            c_id = orchestrator.assignment.mapping[ArmCode.C]
            self.assertIsNotNone(orchestrator.layout.evidence_root(c_id))
            self.assertIsNone(
                orchestrator.layout.evidence_root(
                    orchestrator.assignment.mapping[ArmCode.A]
                )
            )
        source_after = sha256_digest(
            {
                path.relative_to(SRC).as_posix(): path.read_bytes().hex()
                for path in sorted(SRC.rglob("*.py"))
                if "__pycache__" not in path.parts
            }
        )
        self.assertEqual(source_before, source_after)

    def test_guard_validation_queue_consumes_normal_rounds_without_llm(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw, ThreeArmPreCanaryOrchestratorV1(
            Path(raw) / "m4"
        ) as orchestrator:
            rounds = [
                orchestrator.run_fake_triplet(
                    search_seed=42,
                    round_index=round_index,
                    drafts=drafts(),
                )
                for round_index in (1, 2, 3)
            ]
            c_id = orchestrator.assignment.mapping[ArmCode.C]
            c_results = [
                next(
                    item
                    for item in triplet
                    if item.opaque_instance_id == c_id
                )
                for triplet in rounds
            ]
            self.assertGreater(c_results[0].physical_call_count, 0)
            self.assertEqual(
                [item.physical_call_count for item in c_results[1:]],
                [0, 0],
            )
            self.assertEqual(
                [item.proposal_count for item in c_results[1:]],
                [0, 0],
            )
            self.assertEqual(
                [item.ordinary_execution_count for item in c_results],
                [1, 1, 1],
            )
            self.assertEqual(
                tuple(
                    item.observation_seed
                    for item in orchestrator.guard_ledger.evidence_snapshot().observations
                ),
                ("2026", "2027", "2028"),
            )
            self.assertIsNone(
                orchestrator.research_task_queues[
                    ArmCode.C
                ].select_next(
                    allowed_types=frozenset(
                        {
                            ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE
                        }
                    )
                )
            )

    def test_post_training_learning_failure_preserves_completed_round_closure(
        self,
    ) -> None:
        class FailingLearningHookOrchestrator(
            ThreeArmPreCanaryOrchestratorV1
        ):
            def _after_research_close(self, **_kwargs) -> None:
                raise RuntimeError("simulated post-training learning failure")

        with tempfile.TemporaryDirectory() as raw, FailingLearningHookOrchestrator(
            Path(raw) / "m4"
        ) as orchestrator:
            with self.assertRaisesRegex(
                RuntimeError,
                "post-training learning failure",
            ):
                orchestrator.run_fake_triplet(
                    search_seed=42,
                    round_index=1,
                    drafts=drafts(),
                )
            rows = orchestrator.store._connection.execute(
                "SELECT arm_instance_id, status, terminal_class FROM rounds "
                "ORDER BY rowid"
            ).fetchall()
            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(
                all(
                    row["status"] == "CLOSED"
                    and row["terminal_class"] == "COMPLETED"
                    for row in rows
                )
            )
            self.assertIn(
                orchestrator.assignment.mapping[ArmCode.B],
                {row["arm_instance_id"] for row in rows},
            )

    def test_all_pre_blocked_closes_zero_execution_one_feedback_without_refresh(self) -> None:
        class BlockingPort:
            def pre_run(self, candidate):
                return PortAdjudication(
                    candidate_id=candidate.candidate_id,
                    stage=PortStage.PRE,
                    status=PortStatus.BLOCK,
                    protocol_status="CURRENT_PROTOCOL",
                    outcome_class="PRE_RUN_ONLY",
                    claim_ceiling="DEVELOPMENT_ONLY",
                    reason_codes=("REFERENCE_BLOCK",),
                    comparator_delta=None,
                    evidence_use="EXCLUDE_FROM_CURRENT_CLAIM",
                    recommended_validation="NONE",
                )

        slate = tuple(
            CandidateEnvelope(
                candidate_id=f"blocked-{index}",
                candidate_semantic_digest=sha256_digest(
                    {"semantic": index}
                ),
                opaque_arm_instance_id="opaque-c",
                common_status="COMMON_PASS",
                mechanism_program_digest=sha256_digest({"program": index}),
                common_plan_digest=sha256_digest({"plan": index}),
                action_family="RUN_OFFLINE_TOPN",
                planned_protocol={"protocol_id": "fixture"},
                target_model="CandidateModel",
                comparator="LightGCN",
                seed_ids=("2026",),
                purpose="development comparison",
            )
            for index in range(3)
        )
        selection = SameSlateHelixSelectorV1(
            DeterministicHelixFusionV1()
        ).select(slate, BlockingPort())
        self.assertIsNone(selection.selected_candidate)
        self.assertEqual(selection.terminal_status, "SLATE_EXHAUSTED")
        self.assertEqual(selection.inspected_candidate_ids, tuple(item.candidate_id for item in slate))
        self.assertEqual(selection.producer_refresh_count, 0)
        self.assertEqual(selection.extra_proposal_count, 0)

        with tempfile.TemporaryDirectory() as raw:
            store, contract, ids = M4StoreAdversarialTest()._store(Path(raw))
            try:
                opened = M4StoreAdversarialTest()._open(
                    store, contract, ids, ArmCode.C, "all-block-open"
                )
                store.close_round(
                    CloseRoundCommand(
                        round_id=opened["round_id"],
                        terminal_class="NO_EXECUTION",
                        feedback_payload={
                            "outcome_class": "ALL_PRE_BLOCKED",
                            "search_update": "NO_SEARCH_UPDATE",
                        },
                        controller_state_after_digest=genesis(contract),
                        resource_debits=(
                            ResourceDebitV1("PROPOSAL", 3),
                            ResourceDebitV1("COMMON_VALIDATION", 3),
                        ),
                        idempotency_key="all-block-close",
                    )
                )
                db = sqlite3.connect(store.db_path)
                claims = db.execute(
                    "SELECT COUNT(*) FROM execution_claims"
                ).fetchone()[0]
                feedbacks = db.execute(
                    "SELECT COUNT(*) FROM round_events WHERE event_type='ROUND_CLOSED'"
                ).fetchone()[0]
                db.close()
                self.assertEqual(claims, 0)
                self.assertEqual(feedbacks, 1)
            finally:
                store.close()


class M4StoreAdversarialTest(unittest.TestCase):
    def _store(self, root: Path):
        store = SingleWriterExperimentStoreV1(
            root / "state.sqlite3", root / "artifacts"
        )
        contract = default_experiment_contract()
        ids = store.initialize_experiment(contract)
        return store, contract, ids

    def _open(self, store, contract, ids, arm: ArmCode, key: str):
        return store.open_round(
            OpenRoundCommand(
                experiment_id=contract.experiment_id,
                arm_instance_id=ids[arm],
                arm_code=arm,
                search_seed=42,
                round_index=1,
                budget_snapshot=m4_budget(),
                controller_state_before_digest=genesis(contract),
                idempotency_key=key,
            )
        )

    def test_duplicate_open_and_two_client_claim_contention_have_one_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store, contract, ids = self._store(Path(raw))
            try:
                opened = self._open(store, contract, ids, ArmCode.A, "m4-open")
                self.assertEqual(
                    opened,
                    self._open(store, contract, ids, ArmCode.A, "m4-open"),
                )
                commands = (
                    ClaimExecutionCommand(
                        round_id=opened["round_id"],
                        permit_digest="1" * 64,
                        binding_digest="2" * 64,
                        idempotency_key="m4-claim-one",
                    ),
                    ClaimExecutionCommand(
                        round_id=opened["round_id"],
                        permit_digest="3" * 64,
                        binding_digest="4" * 64,
                        idempotency_key="m4-claim-two",
                    ),
                )

                def claim(command):
                    try:
                        store.claim_execution(command)
                    except InvariantViolation:
                        return "REJECTED"
                    return "CLAIMED"

                with ThreadPoolExecutor(max_workers=2) as pool:
                    outcomes = list(pool.map(claim, commands))
                self.assertEqual(sorted(outcomes), ["CLAIMED", "REJECTED"])
            finally:
                store.close()

    def test_process_kill_after_open_recovers_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            script = r"""
import os
from pathlib import Path
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode, default_experiment_contract
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import m4_budget
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.state_store import OpenRoundCommand, SingleWriterExperimentStoreV1
root=Path(os.environ["M4_KILL_ROOT"])
store=SingleWriterExperimentStoreV1(root/"state.sqlite3", root/"artifacts")
contract=default_experiment_contract()
ids=store.initialize_experiment(contract)
genesis=sha256_digest({"experiment_contract_digest": contract.identity_digest, "state": "GENESIS"})
store.open_round(OpenRoundCommand(contract.experiment_id, ids[ArmCode.A], ArmCode.A, 42, 1, m4_budget(), genesis, "kill-open"))
os._exit(137)
"""
            env = dict(os.environ)
            env["PYTHONPATH"] = str(SRC)
            env["M4_KILL_ROOT"] = str(root)
            completed = subprocess.run(
                [sys.executable, "-c", script], env=env, check=False
            )
            self.assertEqual(completed.returncode, 137)
            store = SingleWriterExperimentStoreV1(
                root / "state.sqlite3", root / "artifacts"
            )
            try:
                report = store.conservative_recovery(
                    ConservativeRecoveryCommand(
                        experiment_id="HELIX-ABC-001",
                        search_seed=42,
                        current_round_index=1,
                        idempotency_key="m4-recover-open",
                    )
                )
                self.assertEqual(
                    [item["terminal_class"] for item in report["recovered_rounds"]],
                    ["ABORTED_RECOVERY_BEFORE_EXECUTION"],
                )
            finally:
                store.close()

    def test_process_kill_after_claim_is_start_ambiguous_and_counts_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store, contract, ids = self._store(root)
            opened = self._open(store, contract, ids, ArmCode.A, "claim-kill-open")
            store.claim_execution(
                ClaimExecutionCommand(
                    round_id=opened["round_id"],
                    permit_digest="1" * 64,
                    binding_digest="2" * 64,
                    idempotency_key="claim-kill",
                )
            )
            store.close()
            recovered = SingleWriterExperimentStoreV1(
                root / "state.sqlite3", root / "artifacts"
            )
            try:
                report = recovered.conservative_recovery(
                    ConservativeRecoveryCommand(
                        experiment_id=contract.experiment_id,
                        search_seed=42,
                        current_round_index=1,
                        idempotency_key="claim-kill-recovery",
                    )
                )
                self.assertEqual(
                    [item["terminal_class"] for item in report["recovered_rounds"]],
                    ["ABORTED_RECOVERY_START_AMBIGUOUS"],
                )
                db = sqlite3.connect(root / "state.sqlite3")
                debit = db.execute(
                    "SELECT SUM(quantity) FROM resource_ledger "
                    "WHERE round_id=? AND dimension='ORDINARY_EXECUTION'",
                    (opened["round_id"],),
                ).fetchone()[0]
                db.close()
                self.assertEqual(debit, 1)
            finally:
                recovered.close()

    def test_stop_wins_against_unopened_siblings_and_prevents_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store, contract, ids = self._store(Path(raw))
            try:
                opened = self._open(store, contract, ids, ArmCode.A, "stop-open-a")
                store.stop_and_fill_remaining(
                    StopAndFillCommand(
                        experiment_id=contract.experiment_id,
                        search_seed=42,
                        current_round_index=1,
                        reason="M4_STOP_TEST",
                        idempotency_key="m4-stop",
                    )
                )
                with self.assertRaises(InvariantViolation):
                    self._open(store, contract, ids, ArmCode.B, "stop-open-b")
                store.close_round(
                    CloseRoundCommand(
                        round_id=opened["round_id"],
                        terminal_class="ABORTED",
                        feedback_payload={"outcome_class": "STOPPED"},
                        controller_state_after_digest=genesis(contract),
                        resource_debits=(ResourceDebitV1("PROPOSAL", 0),),
                        idempotency_key="stop-close-a",
                    )
                )
                db = sqlite3.connect(store.db_path)
                states = {
                    row[0]
                    for row in db.execute(
                        "SELECT state FROM arm_state WHERE search_seed=42"
                    )
                }
                remaining = db.execute(
                    "SELECT COUNT(*) FROM scheduled_slots WHERE search_seed=42 "
                    "AND round_index>1 AND slot_status!='NOT_STARTED_STOP'"
                ).fetchone()[0]
                db.close()
                self.assertEqual(states, {"STOPPED"})
                self.assertEqual(remaining, 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
