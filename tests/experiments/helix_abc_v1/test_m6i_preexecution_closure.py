from __future__ import annotations

import json
import tempfile
import unittest
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerError,
    PostProviderSemanticRejectionV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ArmCode,
    ResourceCeilingsV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
    ThreeArmPreCanaryOrchestratorV1,
    m4_budget,
)


class _SemanticRejectingBroker:
    def generate(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
        drafts: Sequence[Mapping[str, Any]],
        ceilings: ResourceCeilingsV1,
    ) -> None:
        del arm, round_index, search_seed, drafts, ceilings
        raise PostProviderSemanticRejectionV1(
            failure_class="RESEARCH_RESPONSE_SEMANTIC_REJECTION",
            cause=ValueError("provider parent identity is not canonical"),
        )

    def provider_usage_for_round(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
    ) -> dict[str, Any]:
        return {
            "billed_tokens": 470,
            "call_latencies_ms": [210, 220, 230, 240],
            "input_tokens": 250,
            "output_tokens": 220,
            "physical_call_count": 4,
            "proposal_count": 6,
            "response_digests": [
                sha256_digest(
                    {
                        "arm": arm.value,
                        "call": index,
                        "round_index": round_index,
                        "search_seed": search_seed,
                    }
                )
                for index in range(4)
            ],
            "wall_time_ms": 1100,
        }


class _UnexpectedRejectingBroker:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate(self, **_kwargs: Any) -> None:
        raise self.error


class M6IPreExecutionClosureTest(unittest.TestCase):
    def _orchestrator(
        self, root: Path, broker: Any
    ) -> ThreeArmPreCanaryOrchestratorV1:
        return ThreeArmPreCanaryOrchestratorV1(
            root,
            broker=broker,
            resource_ceilings=m4_budget(),
        )

    def test_semantic_rejection_closes_all_six_arm_orders(self) -> None:
        for order in permutations(tuple(ArmCode)):
            with self.subTest(order=tuple(item.value for item in order)):
                with tempfile.TemporaryDirectory() as temp:
                    orchestrator = self._orchestrator(
                        Path(temp), _SemanticRejectingBroker()
                    )
                    try:
                        results = orchestrator.run_fake_triplet(
                            search_seed=42,
                            round_index=1,
                            drafts=(),
                            execution_order=order,
                        )
                        self.assertEqual(
                            [item.terminal_class for item in results],
                            ["NO_EXECUTION"] * 3,
                        )
                        self.assertEqual(
                            [item.ordinary_execution_count for item in results],
                            [0, 0, 0],
                        )
                        self.assertEqual(
                            [item.evidence_port_status for item in results],
                            [
                                "NOT_CALLED_PRE_EXECUTION_REJECTION"
                            ]
                            * 3,
                        )
                        rows = orchestrator.store._connection.execute(
                            """
                            SELECT status, terminal_class FROM rounds
                            ORDER BY arm_code
                            """
                        ).fetchall()
                        self.assertEqual(
                            [tuple(row) for row in rows],
                            [("CLOSED", "NO_EXECUTION")] * 3,
                        )
                        self.assertEqual(
                            orchestrator.store._connection.execute(
                                "SELECT COUNT(*) FROM execution_claims"
                            ).fetchone()[0],
                            0,
                        )
                        barrier = orchestrator.store._connection.execute(
                            """
                            SELECT closed_bitmap, next_index_authorized
                            FROM triplet_barrier
                            WHERE search_seed = 42 AND round_index = 1
                            """
                        ).fetchone()
                        self.assertEqual(tuple(barrier), (7, 1))
                        for arm in ArmCode:
                            projection = (
                                orchestrator.integrated_state.round_projection(
                                    arm=arm,
                                    search_seed=42,
                                    round_index=1,
                                )
                            )
                            self.assertEqual(
                                projection["state"], "ROUND_TERMINAL"
                            )
                            self.assertEqual(
                                projection["observation_path"],
                                "NO_OBSERVATION",
                            )
                            self.assertEqual(
                                projection["meta_boundary_count"],
                                int(arm in {ArmCode.B, ArmCode.C}),
                            )
                        artifacts = list(
                            orchestrator.store.artifact_root.glob(
                                "pre_execution_rejections/*/"
                                "PRE_EXECUTION_REJECTION_V1.json"
                            )
                        )
                        self.assertEqual(len(artifacts), 3)
                        closure = json.loads(
                            artifacts[0].read_text(encoding="utf-8")
                        )
                        self.assertTrue(
                            closure["known_semantic_rejection"]
                        )
                        self.assertFalse(
                            closure["execution_claim_present"]
                        )
                        self.assertFalse(closure["training_started"])
                        self.assertFalse(closure["guard_called"])
                        self.assertFalse(
                            closure["search_memory_updated"]
                        )
                        self.assertFalse(
                            closure["meta_observation_updated"]
                        )
                        self.assertFalse(closure["frontier_updated"])
                        self.assertEqual(
                            closure["input_tokens_actual"], 250
                        )
                        self.assertEqual(
                            closure["wall_time_ms_actual"], 1100
                        )
                        self.assertEqual(
                            closure["allocation_ceiling_exceeded"],
                            [
                                "BILLED_TOKEN_DEBIT",
                                "INPUT_TOKEN",
                                "OUTPUT_TOKEN",
                                "PROPOSAL",
                                "PROPOSAL_ATTEMPT",
                                "WALL_TIME_MS",
                            ],
                        )
                        round_id = closure["round_id"]
                        ledger = dict(
                            orchestrator.store._connection.execute(
                                """
                                SELECT dimension, SUM(quantity)
                                FROM resource_ledger
                                WHERE round_id = ?
                                GROUP BY dimension
                                """,
                                (round_id,),
                            ).fetchall()
                        )
                        self.assertEqual(ledger["INPUT_TOKEN"], 200)
                        self.assertEqual(ledger["OUTPUT_TOKEN"], 200)
                        self.assertEqual(
                            ledger["BILLED_TOKEN_DEBIT"], 400
                        )
                        self.assertEqual(ledger["PROPOSAL"], 4)
                        self.assertEqual(
                            ledger["PROPOSAL_ATTEMPT"], 4
                        )
                        self.assertEqual(ledger["WALL_TIME_MS"], 1000)
                        self.assertEqual(ledger["ORDINARY_EXECUTION"], 0)
                        self.assertEqual(ledger["GPU_DEVICE_TIME_MS"], 0)
                    finally:
                        orchestrator.close()

    def test_unexpected_pre_execution_failure_closes_then_propagates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(
                Path(temp),
                _UnexpectedRejectingBroker(
                    RuntimeError("unexpected pre-execution failure")
                ),
            )
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "unexpected pre-execution failure"
                ):
                    orchestrator._run_arm(
                        arm=ArmCode.A,
                        search_seed=42,
                        round_index=1,
                        drafts=(),
                        ceilings=m4_budget(),
                    )
                row = orchestrator.store._connection.execute(
                    "SELECT status, terminal_class FROM rounds"
                ).fetchone()
                self.assertEqual(tuple(row), ("CLOSED", "NO_EXECUTION"))
                self.assertEqual(
                    orchestrator.store._connection.execute(
                        "SELECT COUNT(*) FROM execution_claims"
                    ).fetchone()[0],
                    0,
                )
            finally:
                orchestrator.close()

    def test_broker_error_without_process_receipt_closes_then_propagates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            orchestrator = self._orchestrator(
                Path(temp),
                _UnexpectedRejectingBroker(
                    CanaryBrokerError(
                        "provider transport identity unavailable",
                        physical_call_count=1,
                        input_tokens=11,
                        output_tokens=7,
                        billed_tokens=18,
                        wall_time_ms=90,
                    )
                ),
            )
            try:
                with self.assertRaisesRegex(
                    CanaryBrokerError,
                    "provider transport identity unavailable",
                ):
                    orchestrator._run_arm(
                        arm=ArmCode.B,
                        search_seed=42,
                        round_index=1,
                        drafts=(),
                        ceilings=m4_budget(),
                    )
                row = orchestrator.store._connection.execute(
                    "SELECT status, terminal_class FROM rounds"
                ).fetchone()
                self.assertEqual(tuple(row), ("CLOSED", "NO_EXECUTION"))
                artifact = next(
                    orchestrator.store.artifact_root.glob(
                        "pre_execution_rejections/*/"
                        "PRE_EXECUTION_REJECTION_V1.json"
                    )
                )
                closure = json.loads(
                    artifact.read_text(encoding="utf-8")
                )
                self.assertEqual(
                    closure["failure_class"],
                    "BROKER_ERROR_WITHOUT_PROCESS_RECEIPT",
                )
                self.assertFalse(
                    closure["known_semantic_rejection"]
                )
                self.assertEqual(closure["physical_call_count"], 1)
                self.assertEqual(closure["billed_tokens_actual"], 18)
            finally:
                orchestrator.close()


if __name__ == "__main__":
    unittest.main()
