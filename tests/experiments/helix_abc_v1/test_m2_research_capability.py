from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
)
from recclaw_core.experiments.helix_abc_v1.controllers import (  # noqa: E402
    OriginalControllerV1,
)
from recclaw_core.experiments.helix_abc_v1.evidence import NullEvidencePortV1  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.research_capability import (  # noqa: E402
    ControlAblationBuilderV1,
    FixtureProducerBrokerV1,
    RepairEngineerV1,
    ResearchCapabilityError,
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedMetaPolicyUpdaterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (  # noqa: E402
    DISCOVERY_PRODUCERS,
    AgentizationVerdictV1,
    DevelopmentalMechanismBeliefV1,
    DiscoveryCreditV1,
    MetaVerdictV1,
    ProposalIntentV1,
    RouterHardGateReasonV1,
)
from recclaw_core.experiments.helix_abc_v1.research_controller import (  # noqa: E402
    ResearchLineControllerV1,
)
from recclaw_core.experiments.helix_abc_v1.research_quality_gate import (  # noqa: E402
    combine_research_quality_gate,
    run_agentization_gate,
    run_meta_gate,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (  # noqa: E402
    common_release_projection_digest,
    source_manifest_digest,
)


FIXTURES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"


def program(name: str) -> dict[str, object]:
    document = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return copy.deepcopy(
        next(item["program"] for item in document["fixtures"] if item["anchor_name"] == name)
    )


def ceilings() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=200,
        total_output_tokens=200,
        total_billed_token_debit=400,
        total_proposal_count=4,
        wall_time_ms=1000,
        retry_debit=0,
        proposal_attempt_debit=4,
        ordinary_executions=1,
        common_validation_count=4,
        gpu_device_time_ms=0,
        gpu_cost_microunits=0,
    )


def draft(name: str, axis: str, intent: str, utility: float = 0.8) -> dict[str, object]:
    return {
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


def mode_drafts() -> dict[ProducerExecutionModeV1, list[dict[str, object]]]:
    discovery = ProposalIntentV1.DISCOVERY.value
    falsification = ProposalIntentV1.FALSIFICATION.value
    return {
        ProducerExecutionModeV1.BATCHED_ROLE_PORTFOLIO_V1: [
            draft("BPR_MF", "objective", discovery, 0.75),
            draft("BPR_MF", "objective", discovery, 0.70),
            draft("BPR_MF", "objective", falsification, 0.72),
            draft("BPR_MF", "objective", discovery, 0.68),
        ],
        ProducerExecutionModeV1.NEUTRAL_MULTISAMPLE_CONTROL_V1: [
            draft("BPR_MF", "objective", discovery, 0.75),
            draft("LIGHTGCN", "propagation", discovery, 0.76),
            draft("BPR_MF", "objective", falsification, 0.72),
            draft("SGL", "self_supervision", discovery, 0.77),
        ],
        ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1: [
            draft("DIRECTAU", "geometry", discovery, 0.88),
            draft("LIGHTGCN", "propagation", discovery, 0.84),
            draft("SGL", "self_supervision", falsification, 0.82),
            draft("ULTRAGCN", "architecture", discovery, 0.86),
        ],
    }


def dispatch_all():
    broker = FixtureProducerBrokerV1()
    common = {
        "context": {"dataset": "fixture", "round_index": 1},
        "role_memory": {
            role: {"prior_round_digest": sha256_digest({"role": role})}
            for role in DISCOVERY_PRODUCERS
        }
        | {"neutral": {"prior_round_digest": sha256_digest({"role": "neutral"})}},
    }
    return {
        mode: broker.dispatch(
            session_id=f"m2-{mode.value.lower()}",
            mode=mode,
            drafts=drafts,
            context=common["context"],
            role_memory=common["role_memory"],
            seed=2026,
            ceilings=ceilings(),
        )
        for mode, drafts in mode_drafts().items()
    }


def belief(round_index: int = 1) -> DevelopmentalMechanismBeliefV1:
    return DevelopmentalMechanismBeliefV1(
        hypothesis_id=f"hyp-{round_index}",
        mechanism_axis="self_supervision",
        competing_hypotheses=("regularization_only", "augmentation_invariance"),
        predicted_outcome_signature="higher useful-signal under sparse histories",
        evidence_for=(f"development-observation-{round_index}",),
        evidence_against=(),
        unresolved_confounds=("capacity",),
        next_discriminative_test="matched augmentation ablation",
    )


class M2ResearchCapabilityTest(unittest.TestCase):
    def test_three_modes_share_resources_but_preserve_physical_call_treatment(self) -> None:
        sessions = dispatch_all()
        totals = {
            (
                item.proposal_count,
                item.input_tokens,
                item.output_tokens,
                item.billed_tokens,
                item.total_resource_envelope_digest,
            )
            for item in sessions.values()
        }
        self.assertEqual(len(totals), 1)
        self.assertEqual(
            sessions[ProducerExecutionModeV1.BATCHED_ROLE_PORTFOLIO_V1].physical_call_count,
            1,
        )
        self.assertEqual(
            sessions[
                ProducerExecutionModeV1.NEUTRAL_MULTISAMPLE_CONTROL_V1
            ].physical_call_count,
            4,
        )
        independent = sessions[
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ]
        self.assertEqual(independent.physical_call_count, 4)
        self.assertEqual(tuple(item.producer_role for item in independent.calls), DISCOVERY_PRODUCERS)
        self.assertEqual(len({item.physical_call_id for item in independent.calls}), 4)
        self.assertEqual(len({item.context_digest for item in independent.calls}), 4)
        self.assertEqual(len({item.memory_digest for item in independent.calls}), 4)
        self.assertTrue(all(len(item.candidate_ids) == 1 for item in independent.calls))

    def test_independent_call_lineage_and_scopes_are_complete(self) -> None:
        session = dispatch_all()[
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ]
        self.assertTrue(all(item.assigned_before_call for item in session.proposals))
        self.assertFalse(any(item.post_hoc_relabel for item in session.proposals))
        self.assertEqual(
            {item.producer_id for item in session.proposals}, set(DISCOVERY_PRODUCERS)
        )
        for call in session.calls:
            self.assertEqual(len(call.request_digest), 64)
            self.assertEqual(len(call.response_digest), 64)
            self.assertEqual(len(call.context_digest), 64)
            self.assertEqual(len(call.memory_digest), 64)
            self.assertEqual(len(call.rng_digest), 64)
        with self.assertRaises(TypeError):
            session.proposals[0].mechanism_program["candidate_id"] = "mutated"

    def test_falsification_slot_and_control_repair_credit_are_separate(self) -> None:
        proposals = dispatch_all()[
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ].proposals
        falsification = [
            item for item in proposals if item.proposal_intent is ProposalIntentV1.FALSIFICATION
        ]
        self.assertEqual(len(falsification), 1)
        control = ControlAblationBuilderV1().build(proposals[0])
        repair = RepairEngineerV1().build(proposals[0], "COMPILE_BLOCKER")
        self.assertEqual(
            control["discovery_credit"],
            DiscoveryCreditV1.NON_DISCOVERY_CONTROL.value,
        )
        self.assertEqual(
            repair["discovery_credit"],
            DiscoveryCreditV1.NON_DISCOVERY_REPAIR.value,
        )

    def test_router_consumes_duplicate_and_blocker_features_with_closed_trace(self) -> None:
        session = dispatch_all()[ProducerExecutionModeV1.BATCHED_ROLE_PORTFOLIO_V1]
        trace = StrongStaticRouterV1().route(session.proposals)
        self.assertEqual(trace.ordered_candidate_ids, tuple(item.candidate_id for item in session.proposals))
        reasons = {item.reason for item in trace.decisions}
        self.assertIn(RouterHardGateReasonV1.SEMANTIC_DUPLICATE, reasons)
        self.assertIsNotNone(trace.selected_candidate_id)
        self.assertEqual(len(trace.decisions), len(session.proposals))

        blocked = mode_drafts()[
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ]
        blocked[0]["utility_features"]["blocker_risk"] = 0.99
        blocked_session = FixtureProducerBrokerV1().dispatch(
            session_id="blocker-test",
            mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
            drafts=blocked,
            context={"round": 1},
            role_memory={role: {} for role in DISCOVERY_PRODUCERS},
            seed=1,
            ceilings=ceilings(),
        )
        blocked_trace = StrongStaticRouterV1().route(blocked_session.proposals)
        self.assertEqual(
            blocked_trace.decisions[0].reason,
            RouterHardGateReasonV1.BLOCKER_RISK_ABOVE_CEILING,
        )

    def test_mechanism_belief_has_exactly_eight_search_fields(self) -> None:
        item = belief()
        self.assertEqual(len(fields(item)), 8)
        self.assertEqual(
            tuple(field.name for field in fields(item)),
            (
                "hypothesis_id",
                "mechanism_axis",
                "competing_hypotheses",
                "predicted_outcome_signature",
                "evidence_for",
                "evidence_against",
                "unresolved_confounds",
                "next_discriminative_test",
            ),
        )
        self.assertEqual(item.authority, "NONE")
        self.assertEqual(item.evidence_class, "DEVELOPMENT_ONLY")

    def test_search_memory_is_single_writer_immutable_predecessor_chain(self) -> None:
        writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
        first = writer.commit(
            round_index=1,
            expected_predecessor_digest=None,
            beliefs=(belief(1),),
            route_trace_digest=sha256_digest({"route": 1}),
            feedback_projection={"useful_signal": 0.7},
        )
        second = writer.commit(
            round_index=2,
            expected_predecessor_digest=first.digest,
            beliefs=(belief(2),),
            route_trace_digest=sha256_digest({"route": 2}),
            feedback_projection={"useful_signal": 0.8},
        )
        self.assertEqual(second.predecessor_digest, first.digest)
        with self.assertRaises(ResearchCapabilityError):
            writer.commit(
                round_index=3,
                expected_predecessor_digest=first.digest,
                beliefs=(belief(3),),
                route_trace_digest=sha256_digest({"route": 3}),
                feedback_projection={"useful_signal": 0.9},
            )
        with self.assertRaises(ValueError):
            writer.commit(
                round_index=3,
                expected_predecessor_digest=second.digest,
                beliefs=(belief(3),),
                route_trace_digest=sha256_digest({"route": 3}),
                feedback_projection={"claim-ceiling": "forbidden"},
            )

    def test_agentization_gate_passes_independent_mode_on_frozen_fixture(self) -> None:
        sessions = dispatch_all()
        router = StrongStaticRouterV1()
        gate = run_agentization_gate(
            sessions,
            router=router,
            fixture_lineage_digest=sha256_digest(mode_drafts()),
        )
        self.assertEqual(
            gate.verdict, AgentizationVerdictV1.PASS_INDEPENDENT_MULTI_AGENT
        )
        self.assertEqual(
            gate.selected_mode,
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
        )
        self.assertTrue(gate.outcome_masked)

    def test_agentization_gate_rejects_controlled_contract_mismatch(self) -> None:
        sessions = dispatch_all()
        mode = ProducerExecutionModeV1.NEUTRAL_MULTISAMPLE_CONTROL_V1
        sessions[mode] = replace(sessions[mode], base_model_ref="different-model")
        with self.assertRaises(ValueError):
            run_agentization_gate(
                sessions,
                router=StrongStaticRouterV1(),
                fixture_lineage_digest=sha256_digest(mode_drafts()),
            )

    def test_agentization_gate_rejects_missing_role_scopes(self) -> None:
        drafts = mode_drafts()[
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ]
        sessions = dispatch_all()
        sessions[
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ] = FixtureProducerBrokerV1().dispatch(
            session_id="missing-role-memory",
            mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
            drafts=drafts,
            context={"round": 1},
            role_memory={},
            seed=2026,
            ceilings=ceilings(),
        )
        gate = run_agentization_gate(
            sessions,
            router=StrongStaticRouterV1(),
            fixture_lineage_digest=sha256_digest(mode_drafts()),
        )
        self.assertNotEqual(
            gate.verdict, AgentizationVerdictV1.PASS_INDEPENDENT_MULTI_AGENT
        )

    def test_versioned_meta_passes_replay_boundary_allowlist_and_collapse_checks(self) -> None:
        baseline = initial_research_policy()
        aggregate = {
            "producer_useful_rates": {
                "mechanism_composer": 0.8,
                "lineage_refiner": 0.7,
                "falsification_designer": 0.6,
                "frontier_architect": 0.75,
            },
            "mechanism_axis_gaps": (
                "objective",
                "propagation",
                "self_supervision",
                "geometry",
            ),
            "calibration_error": 0.1,
        }
        report = run_meta_gate(
            selected_mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
            baseline=baseline,
            updater=VersionedMetaPolicyUpdaterV1(),
            aggregate=aggregate,
            completed_round_index=1,
        )
        self.assertEqual(report.verdict, MetaVerdictV1.PASS_VERSIONED_META)
        self.assertTrue(report.deterministic_replay)
        self.assertFalse(report.single_producer_collapse)
        self.assertFalse(report.single_family_collapse)
        self.assertFalse(report.parameter_tuning_collapse)
        self.assertTrue(report.activated_policy_effect)
        with self.assertRaises(ResearchCapabilityError):
            VersionedMetaPolicyUpdaterV1().update(
                baseline,
                completed_round_index=1,
                aggregate=aggregate | {"candidate_ids": ("cand-secret",)},
            )

    def test_versioned_meta_policy_changes_next_round_dispatch_and_route(self) -> None:
        baseline = initial_research_policy()
        updated = VersionedMetaPolicyUpdaterV1().update(
            baseline,
            completed_round_index=1,
            aggregate={
                "producer_useful_rates": {
                    "mechanism_composer": 0.8,
                    "lineage_refiner": 0.7,
                    "falsification_designer": 0.6,
                    "frontier_architect": 0.75,
                },
                "mechanism_axis_gaps": (
                    "objective",
                    "propagation",
                    "self_supervision",
                    "geometry",
                ),
                "calibration_error": 0.1,
            },
        )
        drafts = mode_drafts()[
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ]
        broker = FixtureProducerBrokerV1()
        common = {
            "mode": ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
            "drafts": drafts,
            "context": {"round": 2},
            "role_memory": {role: {"round": 1} for role in DISCOVERY_PRODUCERS},
            "seed": 2028,
            "ceilings": ceilings(),
        }
        before = broker.dispatch(
            session_id="meta-before",
            policy_projection=baseline.to_dict(),
            **common,
        )
        after = broker.dispatch(
            session_id="meta-after",
            policy_projection=updated.to_dict(),
            **common,
        )
        self.assertEqual(sum(item.input_tokens for item in after.calls), 200)
        self.assertNotEqual(
            tuple(item.input_tokens for item in before.calls),
            tuple(item.input_tokens for item in after.calls),
        )
        router = StrongStaticRouterV1()
        self.assertNotEqual(
            router.route(before.proposals, baseline.to_dict()).policy_digest,
            router.route(after.proposals, updated.to_dict()).policy_digest,
        )

    def test_combined_quality_gate_passes_all_m2_progression_conditions(self) -> None:
        sessions = dispatch_all()
        agentization = run_agentization_gate(
            sessions,
            router=StrongStaticRouterV1(),
            fixture_lineage_digest=sha256_digest(mode_drafts()),
        )
        meta = run_meta_gate(
            selected_mode=agentization.selected_mode,
            baseline=initial_research_policy(),
            updater=VersionedMetaPolicyUpdaterV1(),
            aggregate={
                "producer_useful_rates": {
                    "mechanism_composer": 0.8,
                    "lineage_refiner": 0.7,
                    "falsification_designer": 0.6,
                    "frontier_architect": 0.75,
                },
                "mechanism_axis_gaps": (
                    "objective",
                    "propagation",
                    "self_supervision",
                    "geometry",
                ),
                "calibration_error": 0.05,
            },
            completed_round_index=1,
        )
        combined = combine_research_quality_gate(
            agentization=agentization,
            meta=meta,
            sessions=sessions,
        )
        self.assertEqual(combined.progression_status, "PASS")
        self.assertEqual(combined.producer_lineage_complete, 1.0)
        self.assertEqual(combined.post_hoc_relabel_count, 0)
        self.assertTrue(combined.falsification_slot_present)
        self.assertTrue(combined.control_repair_credit_separated)

    def test_original_arm_never_calls_research_components(self) -> None:
        original = OriginalControllerV1()
        proposals = original.propose(
            {
                "execution_mode": "M0_FIXTURE_ONLY",
                "fixture_proposals": ({"candidate_id": "a", "original_score": 1.0},),
            },
            {"space": "BL_ICF_MECHANISM_SPACE_V1"},
            {"proposal_count": 1},
        )
        self.assertEqual(proposals[0]["candidate_id"], "a")
        self.assertNotIn("research", str(original).lower())

    def test_research_line_runs_two_null_port_rounds_without_guard_package(self) -> None:
        self.assertIsNone(importlib.util.find_spec("recclaw_core.evidence_guard"))
        controller = ResearchLineControllerV1(
            producer_mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
            policy=initial_research_policy(),
            broker=FixtureProducerBrokerV1(),
            router=StrongStaticRouterV1(),
            memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        )
        port = NullEvidencePortV1()
        memory = {role: {} for role in DISCOVERY_PRODUCERS}
        predecessor = None
        policy_digests = []
        for round_index in (1, 2):
            plan = controller.plan_round(
                round_index=round_index,
                session_id=f"b-round-{round_index}",
                drafts=mode_drafts()[
                    ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
                ],
                context={"round_index": round_index},
                role_memory=memory,
                seed=2026 + round_index,
                ceilings=ceilings(),
            )
            self.assertIsNotNone(plan.selected_candidate_id)
            policy_digests.append(plan.policy_digest)
            adjudication = port.post_run({"candidate_id": plan.selected_candidate_id})
            self.assertEqual(adjudication.status.value, "NOT_ADJUDICATED")
            transition = controller.close_round(
                plan=plan,
                feedback_projection={
                    "candidate_id": plan.selected_candidate_id,
                    "outcome_class": "SYNTHETIC_USEFUL_SIGNAL",
                },
                beliefs=(belief(round_index),),
            )
            self.assertEqual(transition["feedback_consumption_count"], 1)
            self.assertEqual(plan.ordinary_execution_opportunities, 1)
            self.assertNotEqual(transition["memory_snapshot_digest"], predecessor)
            predecessor = transition["memory_snapshot_digest"]
            if round_index == 1:
                controller.apply_meta_update(
                    updater=VersionedMetaPolicyUpdaterV1(),
                    completed_round_index=round_index,
                    aggregate={
                        "producer_useful_rates": {
                            "mechanism_composer": 0.8,
                            "lineage_refiner": 0.7,
                            "falsification_designer": 0.6,
                            "frontier_architect": 0.75,
                        },
                        "mechanism_axis_gaps": (
                            "objective",
                            "propagation",
                            "self_supervision",
                            "geometry",
                        ),
                        "calibration_error": 0.1,
                    },
                )
        self.assertNotEqual(policy_digests[0], policy_digests[1])

    def test_b_and_c_research_controller_identity_is_exact_equal(self) -> None:
        def controller() -> ResearchLineControllerV1:
            return ResearchLineControllerV1(
                producer_mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
                policy=initial_research_policy(),
                broker=FixtureProducerBrokerV1(),
                router=StrongStaticRouterV1(),
                memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
            )

        self.assertEqual(controller().identity_digest, controller().identity_digest)

    def test_research_import_graph_has_no_guard_or_fusion_dependency(self) -> None:
        for name in (
            "research_contracts.py",
            "research_capability.py",
            "research_controller.py",
            "research_quality_gate.py",
        ):
            source = (
                SRC / "recclaw_core" / "experiments" / "helix_abc_v1" / name
            ).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            joined = " ".join(imports).lower()
            self.assertNotIn("guard", joined)
            self.assertNotIn("fusion", joined)

    def test_m1_common_runtime_release_remains_frozen(self) -> None:
        self.assertEqual(
            common_release_projection_digest(),
            "97c4247af6f0a2fb2fbccd64663e2a6ed25cc1bc98aab2f5630e6d41d6f67347",
        )
        self.assertEqual(
            source_manifest_digest(),
            "038ebf3186ff0c67d7ebe7f5d799ea2d0f4aa6ee141f6a42d9767a05fb01a5fd",
        )

    def test_fixed_seed_replay_is_byte_deterministic(self) -> None:
        first = dispatch_all()
        second = dispatch_all()
        self.assertEqual(
            {mode.value: session.to_dict() for mode, session in first.items()},
            {mode.value: session.to_dict() for mode, session in second.items()},
        )


if __name__ == "__main__":
    unittest.main()
