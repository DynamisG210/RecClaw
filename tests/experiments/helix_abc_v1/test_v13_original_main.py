from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from recclaw_core.mechanism_space import compile_program
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    program_from_proposal,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.original_main import (
    ORIGINAL_MAIN_FILES,
    OriginalMainSourceReleaseV1,
    PinnedOriginalMainAdapterV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
    FakeProposalSessionV1,
)


ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = (
    ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
)


def action(
    candidate_id: str,
    *,
    priority: str,
    base_model: str = "LightGCN",
    parent_candidate_id: str | None = None,
    entrypoint_suffix: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "base_model": base_model,
        "candidate_id": candidate_id,
        "consumes": (),
        "entrypoint": (
            "recclaw.v13:"
            + (entrypoint_suffix or candidate_id.replace("-", "_"))
        ),
        "mechanism_semantics_digest": hashlib.sha256(
            candidate_id.encode()
        ).hexdigest(),
        "priority": priority,
        "runner_type": "model",
        "status": "implemented",
    }
    if parent_candidate_id is not None:
        payload["parent_candidate_id"] = parent_candidate_id
    return payload


class _UnusedUpstream:
    model = "unused"

    def call(self, **_kwargs):
        raise AssertionError("G3 adapter identity test must not call a Broker")


class V13PinnedOriginalMainTest(unittest.TestCase):
    def adapter(self, seed: int = 42) -> PinnedOriginalMainAdapterV1:
        return PinnedOriginalMainAdapterV1(ROOT, seed)

    def test_source_release_materializes_exact_pinned_main_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            release = OriginalMainSourceReleaseV1(
                repository_root=ROOT,
                materialization_root=Path(raw),
            )
            release.materialize()
            for relative, (blob_sha1, sha256) in ORIGINAL_MAIN_FILES.items():
                payload = (Path(raw) / relative).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), sha256)
                self.assertEqual(
                    subprocess.run(
                        ["git", "hash-object", "--stdin"],
                        cwd=ROOT,
                        input=payload,
                        stdout=subprocess.PIPE,
                        check=True,
                    ).stdout.decode().strip(),
                    blob_sha1,
                )
            module = release.load_agent_module()
            self.assertTrue(hasattr(module, "RecClawAgent"))
            self.assertTrue(hasattr(module, "AgentConfig"))

    def test_exact_original_refresh_schedule_and_force_refresh(self) -> None:
        adapter = self.adapter()
        self.assertTrue(adapter.refresh_required(1))
        adapter.install_proposals(
            round_index=1,
            proposals=({"candidate_id": "proposal-1"},),
        )
        self.assertFalse(adapter.refresh_required(2))
        self.assertFalse(adapter.refresh_required(3))
        self.assertTrue(adapter.refresh_required(4))
        adapter.install_proposals(
            round_index=4,
            proposals=({"candidate_id": "proposal-2"},),
        )
        adapter.load_original_state(force_proposal_refresh=True)
        self.assertTrue(adapter.refresh_required(5))

    def test_exact_plan_merges_registry_and_accepted_proposal_bonus(self) -> None:
        adapter = self.adapter()
        parent = action("parent", priority="low", entrypoint_suffix="parent")
        unrelated = action(
            "unrelated",
            priority="high",
            entrypoint_suffix="unrelated",
        )
        accepted = {
            **action(
                "accepted-child",
                priority="low",
                parent_candidate_id="parent",
                entrypoint_suffix="parent",
            ),
            "original_projection_kind": "ACCEPTED_PROPOSAL",
            "parameter_overrides": {},
            "proposal_type": "tuning",
            "proposal_validation_status": "accepted",
            "runnable_level": "config_only",
        }
        ranked = adapter.rank((parent, unrelated, accepted))
        self.assertEqual(ranked[0]["candidate_id"], "accepted-child")
        self.assertEqual(ranked[1]["candidate_id"], "unrelated")

    def test_pending_implemented_candidate_has_exact_original_priority(self) -> None:
        adapter = self.adapter()
        adapter.set_planner_config(max_pending_implemented=1)
        adapter.load_original_state(
            memory=(
                {
                    "candidate_id": "pending",
                    "event": "implementation_result",
                    "status": "implemented_and_runnable",
                },
            )
        )
        ranked = adapter.rank(
            (
                action("pending", priority="low"),
                action("ordinary", priority="high"),
            )
        )
        self.assertEqual(ranked[0]["candidate_id"], "pending")

    def test_algorithm_first_and_plateau_adjustments_are_live(self) -> None:
        adapter = self.adapter()
        adapter.set_planner_config(
            search_intensity="algorithm_first",
            plateau_window_metric_rows=2,
            plateau_family_overuse_window=2,
            max_same_family_repair_streak=2,
            plateau_weak_family_ceiling=1.0,
            anchor_families=["forced-family"],
        )
        adapter.load_original_state(
            memory=tuple(
                {
                    "candidate_id": "capped-family",
                    "decision": "revise",
                    "result": {"ndcg@10": 0.2},
                    "round_id": index,
                    "status": "success",
                }
                for index in range(1, 5)
            )
        )
        exact_agent = adapter._agent
        capped = exact_agent._algorithm_first_score_adjustment(
            {"candidate_id": "capped-family", "base_model": "LightGCN"}
        )
        forced = exact_agent._algorithm_first_score_adjustment(
            {"candidate_id": "forced-family", "base_model": "LightGCN"}
        )
        self.assertLess(capped, 0.0)
        self.assertGreater(forced, 0.0)
        ranked = adapter.rank(
            (
                {
                    **action("plain-root", priority="high"),
                    "runnable_level": "parameter_only",
                },
                {
                    **action("algorithm", priority="low"),
                    "runnable_level": "code_required",
                },
            )
        )
        self.assertEqual(ranked[0]["candidate_id"], "algorithm")

    def test_execution_signature_duplicate_and_validated_focus(self) -> None:
        duplicate_adapter = self.adapter()
        duplicate = action(
            "duplicate",
            priority="high",
            entrypoint_suffix="same",
        )
        signature = (
            "recclaw.v13:same::"
            + duplicate_adapter._agent._params_signature({})
        )
        duplicate_adapter.load_original_state(
            memory=(
                {
                    "candidate_id": "past",
                    "decision": "keep",
                    "execution_signature": signature,
                    "result": {"ndcg@10": 0.3},
                    "status": "success",
                },
            )
        )
        ranked = duplicate_adapter.rank(
            (
                duplicate,
                action("fresh", priority="low", entrypoint_suffix="fresh"),
            )
        )
        self.assertEqual(ranked[0]["candidate_id"], "fresh")

        focus_adapter = self.adapter()
        focus_adapter.load_original_state(
            memory=(
                {
                    "candidate_id": "validated",
                    "decision": "keep",
                    "result": {"ndcg@10": 0.4},
                    "round_id": 1,
                    "seed_validation": {"mean": 0.4, "status": "passed"},
                    "status": "success",
                },
            ),
            last_planner_action={"action": "tune_after_algorithm_success"},
        )
        focused = focus_adapter.rank(
            (
                action("unrelated", priority="high"),
                action(
                    "validated-child",
                    priority="low",
                    parent_candidate_id="validated",
                ),
            )
        )
        self.assertEqual(focused[0]["candidate_id"], "validated-child")

    def test_exact_reflect_and_remember_consume_keep_revise_and_crash(self) -> None:
        adapter = self.adapter()
        candidate = action("candidate", priority="high")
        adapter.rank((candidate,))

        def feedback(
            *,
            round_index: int,
            status: str,
            baseline_delta: float | None,
            history_delta: float | None,
            metric: float | None,
        ) -> dict[str, object]:
            metrics = {} if metric is None else {"ndcg@10": metric}
            return {
                "candidate_id": "candidate",
                "compare_baseline": {"delta": baseline_delta},
                "compare_history_best": {"delta": history_delta},
                "round_index": round_index,
                "search_outcome": {
                    "normalized_metrics": metrics,
                    "run_status": status,
                },
            }

        keep = adapter.close_round(
            feedback(
                round_index=1,
                status="SUCCESS",
                baseline_delta=0.1,
                history_delta=0.1,
                metric=0.4,
            )
        )
        revise = adapter.close_round(
            feedback(
                round_index=2,
                status="SUCCESS",
                baseline_delta=0.01,
                history_delta=0.0,
                metric=0.39,
            )
        )
        crash = adapter.close_round(
            feedback(
                round_index=3,
                status="FAILED",
                baseline_delta=None,
                history_delta=None,
                metric=None,
            )
        )
        self.assertEqual(
            [keep["decision"], revise["decision"], crash["decision"]],
            ["keep", "revise", "crash"],
        )
        self.assertEqual(
            [item["decision"] for item in adapter.state_projection()["executed"]],
            ["keep", "revise", "crash"],
        )

    def test_direct_reference_trace_equals_arm_a_non_projection_trace(self) -> None:
        actions = (
            action("candidate-high", priority="high"),
            action("candidate-low", priority="low"),
        )
        feedback = {
            "candidate_id": "candidate-high",
            "compare_baseline": {"delta": 0.05},
            "compare_history_best": {"delta": 0.05},
            "round_index": 1,
            "search_outcome": {
                "normalized_metrics": {"ndcg@10": 0.4},
                "run_status": "SUCCESS",
            },
        }

        with tempfile.TemporaryDirectory() as raw:
            release = OriginalMainSourceReleaseV1(
                repository_root=ROOT,
                materialization_root=Path(raw),
            )
            module = release.load_agent_module()
            reference = release.new_agent(search_seed=42, proposal_every=3)
            reference.registry = [
                PinnedOriginalMainAdapterV1._registry_candidate(item)
                for item in actions
            ]
            selected, params, _context = reference.plan()
            reference_decision = reference.reflect(
                selected,
                params,
                {"exit_code": 0},
                {"ndcg@10": 0.4, "status": "success"},
                {"delta": 0.05},
                {"delta": 0.05},
            )
            reference.remember(
                module.TrialRecord(
                    round_id=1,
                    candidate_id=str(selected["candidate_id"]),
                    params=params,
                    run_id="",
                    status="success",
                    result={"ndcg@10": 0.4, "status": "success"},
                    compare_baseline={"delta": 0.05},
                    compare_history_best={"delta": 0.05},
                    dimension_report={},
                    decision=reference_decision[0],
                    reason=reference_decision[1],
                    next_action=reference_decision[2],
                    execution_signature=str(
                        selected.get("execution_signature") or ""
                    ),
                )
            )

        arm_a = self.adapter()
        arm_order = arm_a.rank(actions)
        arm_transition = arm_a.close_round(feedback)
        reference_trace = {
            "budget_debits": {
                "ordinary_execution": 1,
                "proposal_calls": 1,
            },
            "decision": reference_decision[0],
            "feedback_consumption_count": 1,
            "selected_candidate_id": selected["candidate_id"],
            "stop": {
                "next_round_opened": False,
                "reason": "FROZEN_FINAL_ROUND",
            },
        }
        arm_trace = {
            "budget_debits": {
                "ordinary_execution": 1,
                "proposal_calls": 1,
            },
            "decision": arm_transition["decision"],
            "feedback_consumption_count": arm_transition[
                "feedback_consumption_count"
            ],
            "selected_candidate_id": arm_order[0]["candidate_id"],
            "stop": {
                "next_round_opened": False,
                "reason": "FROZEN_FINAL_ROUND",
            },
        }
        self.assertEqual(reference_trace, arm_trace)

    def test_v13_broker_factory_cannot_select_handwritten_adapter(self) -> None:
        broker = RealCanaryProposalBrokerV1.create_v13(
            upstream=_UnusedUpstream(),
            template_path=TEMPLATES,
            repository_root=ROOT,
            search_seed=9301,
        )
        self.assertIsInstance(
            broker.original_controller,
            PinnedOriginalMainAdapterV1,
        )
        self.assertNotIn(
            "OriginalRuntimeAdapterV1",
            type(broker.original_controller).__name__,
        )

    def test_v13_runtime_route_uses_pinned_original_plan(self) -> None:
        broker = RealCanaryProposalBrokerV1.create_v13(
            upstream=_UnusedUpstream(),
            template_path=TEMPLATES,
            repository_root=ROOT,
            search_seed=9301,
        )
        proposals = (
            {"mechanism_id": "BPR_MF", "original_priority": "low"},
            {"mechanism_id": "LIGHTGCN", "original_priority": "high"},
            {
                "mechanism_id": "LIGHTGCN_RESIDUAL",
                "original_priority": "medium",
            },
        )
        installed = tuple(
            {
                **proposal,
                "mechanism_program": program_from_proposal(proposal),
            }
            for proposal in proposals
        )
        broker.original_controller.install_proposals(
            round_index=1,
            proposals=installed,
        )
        programs = tuple(
            item["mechanism_program"] for item in installed
        )
        session = FakeProposalSessionV1(
            validation_programs=programs,
            ordered_programs=programs,
            selected_candidate_id="",
            physical_call_count=1,
            input_tokens=1,
            output_tokens=1,
            billed_tokens=2,
            proposal_count=3,
            proposal_session_digest=sha256_digest(installed),
            route_trace_digest=None,
            research_plan=None,
        )
        routed = broker.finalize_common_route(
            arm=ArmCode.A,
            round_index=1,
            session=session,
            common_eligible_candidate_ids=tuple(
                compile_program(program).candidate_id
                for program in programs
            ),
        )
        self.assertEqual(
            routed.selected_candidate_id,
            compile_program(programs[1]).candidate_id,
        )
        self.assertIsNotNone(routed.route_trace_digest)


if __name__ == "__main__":
    unittest.main()
