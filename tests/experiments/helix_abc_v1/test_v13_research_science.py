from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from recclaw_core.helix.contracts import CandidateEnvelope
from recclaw_core.helix.scientific_attribution import (
    NOT_AVAILABLE,
    SearchUtilityEventV2,
)
from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_runtime_profile,
    executable_mechanism,
    program_from_proposal,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerCallV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (
    ThreeArmPreCanaryOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (
    RealCanaryProposalBrokerV1,
    canary_budget,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV2,
    CandidateProposalV4,
    DevelopmentalMechanismBeliefV2,
    DiscoveryCreditV1,
    ProposalIntentV1,
    RouterHardGateReasonV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.research_science import (
    ControlAblationBuilderV2,
    LineageRecordV1,
    RepairEngineerV2,
)


ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = (
    ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
)
PROTOCOL_DIGEST = str(
    campaign_runtime_profile()["development_protocol_digest"]
)


class _UnusedUpstream:
    model = "unused"

    def call(self, **_kwargs):
        raise AssertionError("unit construction must not call a Provider")


class _ScopedFakeUpstream:
    model = "fixture"
    max_total_tokens_per_call = 1000

    def __init__(self) -> None:
        self.broker: RealCanaryProposalBrokerV1 | None = None

    def call(
        self,
        *,
        logical_call_id: str,
        prompt: str,
        expected_proposal_count: int,
        **_kwargs,
    ) -> CanaryBrokerCallV1:
        assert self.broker is not None
        if "original-" in logical_call_id:
            proposals = [
                {
                    **raw_proposal(
                        mechanism_id=mechanism_id,
                        intent="DISCOVERY",
                    ),
                    "original_priority": priority,
                    "status": "implemented",
                }
                for mechanism_id, priority in (
                    ("LIGHTGCN_RESIDUAL", "high"),
                    ("LIGHTGCN_RANK_AWARE", "high"),
                    ("LIGHTGCN_SHALLOW", "medium"),
                    ("LIGHTGCN_AUX_ALIGNMENT", "medium"),
                )
            ]
        else:
            role = logical_call_id.rsplit("-", 1)[-1]
            scope = self.broker._campaign_call_scopes[logical_call_id]
            intent = (
                "FALSIFICATION"
                if role == "falsification_designer"
                else "DISCOVERY"
            )
            proposals = [
                raw_proposal(
                    mechanism_id=scope[0],
                    intent=intent,
                )
            ]
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=sha256_digest({"prompt": prompt}),
            response_digest=sha256_digest(proposals),
            response={"proposals": proposals},
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=5,
            total_tokens=15,
            latency_ms=2,
            returned_model=self.model,
        )


def raw_proposal(
    *,
    mechanism_id: str,
    intent: str,
    parent_candidate_id: str | None = None,
) -> dict[str, object]:
    return {
        "candidate_label": mechanism_id,
        "competing_hypothesis": "the observed change comes from a confound",
        "failure_mode": "the predicted signature is absent",
        "mechanism_hypothesis": "the declared axis causes the metric change",
        "mechanism_id": mechanism_id,
        "parent_candidate_id": parent_candidate_id,
        "predicted_outcome_signature": "positive matched delta",
        "proposal_intent": intent,
        "utility_features": {
            "frontier_potential": 0.8,
            "information_gain": 0.8,
            "useful_signal": 0.8,
        },
    }


def call(raw: dict[str, object], suffix: str) -> CanaryBrokerCallV1:
    return CanaryBrokerCallV1(
        logical_call_id=f"call-{suffix}",
        request_digest=hashlib.sha256(f"request-{suffix}".encode()).hexdigest(),
        response_digest=hashlib.sha256(
            f"response-{suffix}".encode()
        ).hexdigest(),
        response={"proposals": [raw]},
        input_tokens=10,
        cached_input_tokens=0,
        output_tokens=5,
        total_tokens=15,
        latency_ms=2,
        returned_model="fixture",
    )


def lineage_record(
    *,
    proposal_candidate_id: str = "cand-parent000000000000000000",
    mechanism_id: str = "LIGHTGCN",
    metric: float = 0.30,
    run_status: str = "SUCCESS",
) -> LineageRecordV1:
    mechanism = executable_mechanism(mechanism_id)
    return LineageRecordV1(
        proposal_candidate_id=proposal_candidate_id,
        runtime_candidate_id=mechanism.candidate_id,
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program_digest=mechanism.mechanism_program_digest,
        mechanism_semantics_digest=mechanism.mechanism_semantics_digest,
        parent_candidate_id=None,
        protocol_digest=PROTOCOL_DIGEST,
        observation_seed="2026",
        run_status=run_status,
        metric_name="ndcg@10",
        metric_value=metric,
        result_digest=sha256_digest({"result": proposal_candidate_id}),
        round_index=1,
        mechanism_program=mechanism.mechanism_program,
    )


class V13ResearchScienceTest(unittest.TestCase):
    def broker(self) -> RealCanaryProposalBrokerV1:
        return RealCanaryProposalBrokerV1.create_v13(
            upstream=_UnusedUpstream(),
            template_path=TEMPLATES,
            repository_root=ROOT,
            search_seed=9301,
        )

    def test_all_four_producers_are_genuine_discovery_roles(self) -> None:
        broker = self.broker()
        rows = (
            ("mechanism_composer", "LIGHTGCN_RESIDUAL", "DISCOVERY"),
            ("lineage_refiner", "LIGHTGCN_SHALLOW", "DISCOVERY"),
            (
                "falsification_designer",
                "LIGHTGCN_AUX_ALIGNMENT",
                "FALSIFICATION",
            ),
            ("frontier_architect", "LIGHTGCN_RANK_AWARE", "DISCOVERY"),
        )
        proposals = [
            broker._proposal_from_call(
                arm=ArmCode.B,
                session_id="session",
                role=role,
                call=call(
                    raw_proposal(
                        mechanism_id=mechanism,
                        intent=intent,
                    ),
                    role,
                ),
                proposal=raw_proposal(
                    mechanism_id=mechanism,
                    intent=intent,
                ),
            )
            for role, mechanism, intent in rows
        ]
        self.assertTrue(all(isinstance(item, CandidateProposalV4) for item in proposals))
        self.assertTrue(
            all(
                item.discovery_credit is DiscoveryCreditV1.DISCOVERY
                for item in proposals
            )
        )
        falsification = proposals[2]
        self.assertIs(
            falsification.proposal_intent,
            ProposalIntentV1.FALSIFICATION,
        )
        self.assertIsNotNone(falsification.discriminative_plan)
        self.assertEqual(
            falsification.discriminative_plan.matched_control_plan.plan_status,
            "QUEUE_MATCHED_CONTROL",
        )
        self.assertEqual(
            falsification.utility_features.runnable_probability,
            1.0,
        )
        self.assertEqual(falsification.utility_features.blocker_risk, 0.0)

    def test_v13_static_router_path_runs_without_meta_runtime(self) -> None:
        upstream = _ScopedFakeUpstream()
        broker = RealCanaryProposalBrokerV1.create_v13(
            upstream=upstream,
            template_path=TEMPLATES,
            repository_root=ROOT,
            search_seed=9301,
        )
        upstream.broker = broker
        session = broker.generate(
            arm=ArmCode.B,
            round_index=1,
            search_seed=9301,
            drafts=(),
            ceilings=canary_budget(),
        )
        self.assertIsNone(broker.campaign_meta_runtime)
        self.assertEqual(session.physical_call_count, 4)
        self.assertTrue(
            all(
                isinstance(item, CandidateProposalV4)
                for item in session.research_proposals
            )
        )
        routed = broker.finalize_common_route(
            arm=ArmCode.B,
            round_index=1,
            session=session,
            common_eligible_candidate_ids=tuple(
                str(
                    compile_program(deep_thaw(program)).candidate_id
                )
                for program in session.validation_programs
            ),
        )
        self.assertIsNotNone(routed.research_plan)
        self.assertTrue(routed.selected_candidate_id)
        self.assertIsNotNone(routed.route_trace_digest)

    def test_missing_comparator_consumes_next_normal_round(self) -> None:
        upstream = _ScopedFakeUpstream()
        broker = RealCanaryProposalBrokerV1.create_v13(
            upstream=upstream,
            template_path=TEMPLATES,
            repository_root=ROOT,
            search_seed=9301,
        )
        upstream.broker = broker
        with tempfile.TemporaryDirectory() as raw:
            with ThreeArmPreCanaryOrchestratorV1(
                Path(raw) / "v13",
                broker=broker,
                resource_ceilings=canary_budget(),
            ) as orchestrator:
                first_triplet = orchestrator.run_fake_triplet(
                    search_seed=42,
                    round_index=1,
                    drafts=(),
                )
                first = first_triplet[1]
                queued = orchestrator.research_task_queues[
                    ArmCode.B
                ].select_next()
                self.assertIsNotNone(queued)
                self.assertEqual(
                    queued.task_type.value, "RUN_MATCHED_CONTROL"
                )
                second_triplet = orchestrator.run_fake_triplet(
                    search_seed=42,
                    round_index=2,
                    drafts=(),
                )
                second = second_triplet[1]
                self.assertEqual(first.physical_call_count, 4)
                self.assertEqual(second.physical_call_count, 0)
                self.assertEqual(first.broker_call_latencies_ms, (2, 2, 2, 2))
                self.assertGreaterEqual(
                    first.proposal_session_wall_time_ms, 2
                )
                self.assertGreaterEqual(
                    first.round_total_wall_time_ms,
                    first.proposal_session_wall_time_ms,
                )
                self.assertEqual(
                    orchestrator.research_task_queues[
                        ArmCode.B
                    ].select_next(),
                    None,
                )
                beliefs = broker.research_controllers[
                    ArmCode.B
                ].memory_writer.head.beliefs
                matched = [
                    item
                    for item in beliefs
                    if isinstance(item, DevelopmentalMechanismBeliefV2)
                    and item.comparator_delta != NOT_AVAILABLE
                ]
                self.assertEqual(len(matched), 1)
                self.assertTrue(matched[0].exact_comparator_candidate_id)

    def test_lineage_refiner_requires_exact_prior_parent(self) -> None:
        broker = self.broker()
        parent = lineage_record()
        broker.lineage_indexes[ArmCode.B].record(parent)
        raw = raw_proposal(
            mechanism_id="LIGHTGCN_RESIDUAL",
            intent="DISCOVERY",
            parent_candidate_id=parent.proposal_candidate_id,
        )
        proposal = broker._proposal_from_call(
            arm=ArmCode.B,
            session_id="session",
            role="lineage_refiner",
            call=call(raw, "exact-parent"),
            proposal=raw,
        )
        self.assertEqual(
            proposal.parent_candidate_id,
            parent.proposal_candidate_id,
        )
        missing = raw_proposal(
            mechanism_id="LIGHTGCN_RESIDUAL",
            intent="DISCOVERY",
            parent_candidate_id="cand-missing000000000000000",
        )
        with self.assertRaisesRegex(
            Exception,
            "absent from exact lineage",
        ):
            broker._proposal_from_call(
                arm=ArmCode.B,
                session_id="session",
                role="lineage_refiner",
                call=call(missing, "missing-parent"),
                proposal=missing,
            )

    def test_router_utility_floor_is_an_enforced_hard_gate(self) -> None:
        program = program_from_proposal({"mechanism_id": "LIGHTGCN"})
        proposal = CandidateProposalV2(
            candidate_id="cand-utilityfloor0000000000",
            producer_id="producer",
            producer_role="mechanism_composer",
            proposal_intent=ProposalIntentV1.DISCOVERY,
            discovery_credit=DiscoveryCreditV1.DISCOVERY,
            mechanism_axis="propagation",
            mechanism_program=program,
            utility_features=SearchUtilityFeaturesV1(
                runnable_probability=0.5,
                useful_signal=0.0,
                frontier_potential=0.0,
                information_gain=0.0,
                cost=0.0,
                blocker_risk=0.0,
            ),
            parent_candidate_id=None,
            assigned_before_call=True,
            post_hoc_relabel=False,
        )
        route = StrongStaticRouterV1().route((proposal,))
        self.assertFalse(route.decisions[0].allowed)
        self.assertIs(
            route.decisions[0].reason,
            RouterHardGateReasonV1.UTILITY_BELOW_FLOOR,
        )

    def test_router_features_change_from_exact_runtime_history(self) -> None:
        broker = self.broker()
        prior = lineage_record(
            mechanism_id="LIGHTGCN_RESIDUAL",
            metric=0.0,
            run_status="FAILED",
        )
        broker.lineage_indexes[ArmCode.B].record(prior)
        raw = raw_proposal(
            mechanism_id="LIGHTGCN_RESIDUAL",
            intent="DISCOVERY",
        )
        proposal = broker._proposal_from_call(
            arm=ArmCode.B,
            session_id="history",
            role="mechanism_composer",
            call=call(raw, "history"),
            proposal=raw,
        )
        self.assertTrue(proposal.feature_evidence.semantic_duplicate)
        self.assertEqual(proposal.feature_evidence.blocker_rate, 1.0)
        self.assertEqual(proposal.utility_features.runnable_probability, 0.0)
        self.assertEqual(proposal.utility_features.blocker_risk, 1.0)

    def test_belief_credit_requires_exact_matched_comparator(self) -> None:
        broker = self.broker()
        parent = lineage_record()
        broker.lineage_indexes[ArmCode.B].record(parent)
        raw = raw_proposal(
            mechanism_id="LIGHTGCN_AUX_ALIGNMENT",
            intent="FALSIFICATION",
            parent_candidate_id=parent.proposal_candidate_id,
        )
        proposal = broker._proposal_from_call(
            arm=ArmCode.B,
            session_id="session",
            role="falsification_designer",
            call=call(raw, "matched"),
            proposal=raw,
        )
        program = program_from_proposal(raw)
        compiled = compile_program(program)
        selected = CandidateEnvelope(
            candidate_id=str(compiled.candidate_id),
            candidate_semantic_digest=str(
                compiled.mechanism_semantics_digest
            ),
            opaque_arm_instance_id="opaque-b",
            common_status="COMMON_PASS",
            mechanism_program_digest=str(
                compiled.mechanism_program_digest
            ),
            common_plan_digest=sha256_digest({"plan": "matched"}),
            action_family="RUN_OFFLINE_TOPN",
            planned_protocol={"protocol_id": "v13"},
            target_model="LightGCN",
            comparator="LIGHTGCN",
            seed_ids=("2026",),
            purpose="matched comparison",
        )
        event = SearchUtilityEventV2(
            candidate_semantic_digest=str(
                compiled.mechanism_semantics_digest
            ),
            candidate_id=str(compiled.candidate_id),
            mechanism_axis=proposal.mechanism_axis,
            common_outcome_class="SUCCESS",
            runnable_observation="RUNNABLE",
            comparator_delta=0.02,
            metric_contract_digest=PROTOCOL_DIGEST,
            resource_cost_projection={"gpu": 1},
            typed_blocker_class="NONE",
            observation_seed="2026",
        )
        belief = ThreeArmPreCanaryOrchestratorV1._research_belief(
            None,
            selected=selected,
            event=event,
            proposal=proposal,
        )
        self.assertIsInstance(belief, DevelopmentalMechanismBeliefV2)
        self.assertEqual(belief.evidence_for, (f"development_observation:{event.digest}",))
        self.assertEqual(
            belief.exact_comparator_candidate_id,
            parent.proposal_candidate_id,
        )

        unmatched_raw = raw_proposal(
            mechanism_id="LIGHTGCN_RANK_AWARE",
            intent="FALSIFICATION",
        )
        unmatched = self.broker()._proposal_from_call(
            arm=ArmCode.B,
            session_id="unmatched",
            role="falsification_designer",
            call=call(unmatched_raw, "unmatched"),
            proposal=unmatched_raw,
        )
        unmatched_compiled = compile_program(
            deep_thaw(unmatched.mechanism_program)
        )
        no_match_event = SearchUtilityEventV2(
            candidate_semantic_digest=str(
                unmatched_compiled.mechanism_semantics_digest
            ),
            candidate_id=str(unmatched_compiled.candidate_id),
            mechanism_axis=unmatched.mechanism_axis,
            common_outcome_class="SUCCESS",
            runnable_observation="RUNNABLE",
            comparator_delta=NOT_AVAILABLE,
            metric_contract_digest=PROTOCOL_DIGEST,
            resource_cost_projection={"gpu": 1},
            typed_blocker_class="NONE",
            observation_seed="2026",
        )
        no_match_belief = ThreeArmPreCanaryOrchestratorV1._research_belief(
            None,
            selected=selected,
            event=no_match_event,
            proposal=unmatched,
        )
        self.assertEqual(no_match_belief.evidence_for, ())
        self.assertEqual(no_match_belief.evidence_against, ())
        self.assertEqual(
            no_match_belief.next_discriminative_task.task_status,
            "UNEXECUTED_NO_EXACT_CONTROL",
        )

    def test_control_and_repair_services_never_receive_discovery_credit(self) -> None:
        parent = lineage_record()
        broker = self.broker()
        broker.lineage_indexes[ArmCode.B].record(parent)
        control = ControlAblationBuilderV2().build(
            lineage=broker.lineage_indexes[ArmCode.B],
            primary_candidate_id="cand-primary000000000000000",
            parent_candidate_id=parent.proposal_candidate_id,
            changed_axis="objective",
            mechanism_hypothesis="objective changes ranking",
            protocol_digest=PROTOCOL_DIGEST,
        )
        repair = RepairEngineerV2().record(
            candidate_id="cand-primary000000000000000",
            blocker_code="IMPORT_ERROR",
            repair_status="REPAIRED",
        )
        self.assertEqual(control.plan_status, "MATCHED_COMPARATOR_AVAILABLE")
        self.assertEqual(repair.discovery_credit, "NON_DISCOVERY_REPAIR")


if __name__ == "__main__":
    unittest.main()
