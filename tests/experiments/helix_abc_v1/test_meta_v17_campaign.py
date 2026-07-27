from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from freeze_m6_meta_v17_pilot_contract import (  # noqa: E402
    build_activation_receipt,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerCallV1,
)
from run_m6_meta_v17_pilot import (  # noqa: E402
    descriptive_effect_summary,
    pilot_quality_review,
)

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    META_V17_RUNTIME_REPAIR_DIGEST_V1,
    MetaV17CampaignRuntimeV1,
    POLICY_BUNDLE_DIGEST_V17,
    PROMOTION_DECISION_DIGEST_V17,
    ROUTING_SOURCE_SHA256_RUNTIME_REPAIR_V1,
    ROUTING_SOURCE_SHA256_V17,
    SOURCE_MANIFEST_DIGEST_V17,
    meta_v17_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_pilot import (  # noqa: E402
    META_V17_PILOT_EXPERIMENT_ID,
    META_V17_PILOT_SEARCH_SEED,
    MetaV17PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (  # noqa: E402
    FixtureProducerBrokerV1,
    StrongStaticRouterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (  # noqa: E402
    DISCOVERY_PRODUCERS,
    ProposalIntentV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
    _load_templates,
    _program_from_proposal,
    _template_name,
)


FIXTURES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
CHECKPOINT = (
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "meta_vnext_policy_checkpoint_v17.json"
)


def program(name: str) -> dict[str, object]:
    document = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return copy.deepcopy(
        next(
            item["program"]
            for item in document["fixtures"]
            if item["anchor_name"] == name
        )
    )


def ceilings() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=60_000,
        total_output_tokens=20_000,
        total_billed_token_debit=80_000,
        total_proposal_count=4,
        wall_time_ms=1_500_000,
        retry_debit=0,
        proposal_attempt_debit=4,
        ordinary_executions=1,
        common_validation_count=4,
        gpu_device_time_ms=900_000,
        gpu_cost_microunits=250_000,
    )


def session(
    *,
    round_index: int = 1,
    anchors: tuple[str, ...] = ("BPR_MF", "LIGHTGCN", "SGL", "NGCF"),
):
    roles = tuple(DISCOVERY_PRODUCERS)
    axes = ("objective", "propagation", "self_supervision", "architecture")
    drafts = []
    for role, anchor, axis in zip(roles, anchors, axes, strict=True):
        drafts.append(
            {
                "mechanism_program": program(anchor),
                "mechanism_axis": axis,
                "proposal_intent": (
                    ProposalIntentV1.FALSIFICATION.value
                    if role == "falsification_designer"
                    else ProposalIntentV1.DISCOVERY.value
                ),
                "utility_features": {
                    "runnable_probability": 0.9,
                    "useful_signal": 0.8,
                    "frontier_potential": 0.8,
                    "information_gain": 0.8,
                    "cost": 0.3,
                    "blocker_risk": 0.1,
                },
            }
        )
    return FixtureProducerBrokerV1().dispatch(
        session_id=f"meta-v17-campaign-fixture-{round_index}",
        mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
        drafts=drafts,
        context={
            "round_index": round_index,
            "search_seed": META_V17_PILOT_SEARCH_SEED,
        },
        role_memory={
            role: {"prior_round_digest": sha256_digest({"role": role})}
            for role in roles
        },
        seed=META_V17_PILOT_SEARCH_SEED,
        ceilings=ceilings(),
    )


class RecipeUpstream:
    def __init__(self) -> None:
        self.calls: dict[str, CanaryBrokerCallV1] = {}

    def call(self, *, logical_call_id, prompt, expected_proposal_count):
        role = logical_call_id.rsplit("-", 1)[-1]
        catalog = json.loads(
            str(prompt)
            .split("Executable catalog: ", 1)[1]
            .split("\nPrior compact Search Memory feedback:", 1)[0]
        )
        mechanism_id = str(catalog["mechanisms"][0]["mechanism_id"])
        intent = (
            "CONTROL"
            if role == "falsification_designer"
            else "DISCOVERY"
        )
        proposal = {
            "candidate_label": role,
            "competing_hypothesis": "The matched parent explains the result.",
            "failure_mode": "neutral metric",
            "mechanism_hypothesis": (
                f"The {mechanism_id} mechanism may change ranking quality."
            ),
            "mechanism_id": mechanism_id,
            "parent_candidate_id": None,
            "predicted_outcome_signature": "bounded ranking signal",
            "proposal_intent": intent,
            "utility_features": {
                "frontier_potential": 0.7,
                "information_gain": 0.7,
                "useful_signal": 0.7,
            },
        }
        self.calls[logical_call_id] = CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=sha256_digest({"request": logical_call_id}),
            response_digest=sha256_digest({"response": logical_call_id}),
            response={"proposals": [proposal]},
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=5,
            total_tokens=15,
            latency_ms=1,
            returned_model="fake",
        )
        self.assert_expected = expected_proposal_count
        return self.calls[logical_call_id]


class MetaV17CampaignTest(unittest.TestCase):
    def runtime(self, *, task_scale: float = 0.2) -> MetaV17CampaignRuntimeV1:
        runtime = MetaV17CampaignRuntimeV1(
            checkpoint_path=CHECKPOINT,
            experiment_id=META_V17_PILOT_EXPERIMENT_ID,
            search_seed=META_V17_PILOT_SEARCH_SEED,
            scheduled_rounds=5,
            task_scale=task_scale,
            task_density=0.2843,
        )
        runtime.bind_instances(
            {
                ArmCode.A: "opaque-a",
                ArmCode.B: "opaque-b",
                ArmCode.C: "opaque-c",
            }
        )
        return runtime

    def test_imported_v17_implementation_sources_match_sealed_manifest(self):
        manifest = json.loads(
            (
                ROOT
                / "docs/research_line/meta_vnext/"
                "META_VNEXT_IMPLEMENTATION_MANIFEST_V1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["source_identity"]["v17_source_manifest_digest"],
            SOURCE_MANIFEST_DIGEST_V17,
        )
        active_runtime_sources = {
            "src/recclaw_core/experiments/helix_abc_v1/meta_vnext/contracts.py",
            "src/recclaw_core/experiments/helix_abc_v1/meta_vnext/evaluation.py",
            "src/recclaw_core/experiments/helix_abc_v1/meta_vnext/features.py",
            "src/recclaw_core/experiments/helix_abc_v1/meta_vnext/integration.py",
            "src/recclaw_core/experiments/helix_abc_v1/meta_vnext/learning.py",
        }
        for path in sorted(active_runtime_sources):
            self.assertEqual(
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                manifest["implementation_sources"][path],
            )
        routing_path = (
            "src/recclaw_core/experiments/helix_abc_v1/meta_vnext/routing.py"
        )
        self.assertEqual(
            manifest["implementation_sources"][routing_path],
            ROUTING_SOURCE_SHA256_V17,
        )
        self.assertEqual(
            hashlib.sha256((ROOT / routing_path).read_bytes()).hexdigest(),
            ROUTING_SOURCE_SHA256_RUNTIME_REPAIR_V1,
        )
        self.assertNotEqual(
            ROUTING_SOURCE_SHA256_RUNTIME_REPAIR_V1,
            ROUTING_SOURCE_SHA256_V17,
        )
        self.assertEqual(
            self.runtime().audit_projection()["runtime_repair_digest"],
            META_V17_RUNTIME_REPAIR_DIGEST_V1,
        )

    def test_activation_inputs_are_promoted_but_not_historically_applied(self):
        promotion = json.loads(
            (
                ROOT
                / "docs/research_line/meta_vnext/"
                "META_VNEXT_PROMOTION_DECISION_V17.json"
            ).read_text(encoding="utf-8")
        )
        closure = json.loads(
            (
                ROOT
                / "docs/research_line/meta_vnext/"
                "META_VNEXT_ACTIVATION_GATE_CLOSURE_V1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(promotion["decision_digest"], PROMOTION_DECISION_DIGEST_V17)
        self.assertEqual(promotion["decision"]["verdict"], "PROMOTE")
        self.assertFalse(promotion["activation_applied"])
        self.assertEqual(closure["verdict"], "PASS")
        self.assertTrue(closure["fresh_pilot_preconditions_satisfied"])
        self.assertFalse(closure["activation_applied"])

    def test_b_and_c_bind_same_active_checkpoint_and_initial_semantics(self):
        policies = MetaV17PilotStoreContractV1.create().arm_policies
        by_arm = {item.arm: item for item in policies}
        self.assertEqual(
            by_arm[ArmCode.B].non_guard_projection(),
            by_arm[ArmCode.C].non_guard_projection(),
        )
        self.assertEqual(
            by_arm[ArmCode.B].controller_policy_digest,
            meta_v17_research_control_policy().digest,
        )
        runtime = self.runtime()
        self.assertEqual(
            runtime.initial_semantic_state_projection(ArmCode.B),
            runtime.initial_semantic_state_projection(ArmCode.C),
        )

    def test_v17_winner_is_first_executable_candidate_for_b_and_c(self):
        runtime = self.runtime()
        fixture = session()
        b_route = runtime.route_session(
            arm=ArmCode.B,
            round_index=1,
            session=fixture,
            static_router=StrongStaticRouterV1(),
            research_policy=initial_research_policy(),
        )
        c_route = runtime.route_session(
            arm=ArmCode.C,
            round_index=1,
            session=fixture,
            static_router=StrongStaticRouterV1(),
            research_policy=initial_research_policy(),
        )
        self.assertEqual(b_route.mode, "META_VNEXT_V17_SLOW_PLUS_FAST")
        self.assertEqual(c_route.mode, b_route.mode)
        self.assertEqual(b_route.ranked_candidate_ids, c_route.ranked_candidate_ids)
        self.assertEqual(
            b_route.selected_candidate_id,
            b_route.ranked_candidate_ids[0],
        )
        self.assertIsNotNone(b_route.selected_candidate_semantics_digest)
        self.assertEqual(len(b_route.pool_candidate_semantics_digests), 4)
        self.assertEqual(len(b_route.pool_mechanism_axes), 4)
        self.assertTrue(b_route.task_context_supported)

    def test_ml1m_task_scale_uses_promoted_slow_policy_without_fast_residual(self):
        runtime = self.runtime(task_scale=1.0)
        route = runtime.route_session(
            arm=ArmCode.B,
            round_index=1,
            session=session(),
            static_router=StrongStaticRouterV1(),
            research_policy=initial_research_policy(),
        )
        support = runtime.task_support_projection()
        self.assertFalse(support["supported"])
        self.assertEqual(
            support["task_scale_support"],
            (0.02896, 0.417513333333333),
        )
        self.assertEqual(
            route.mode,
            "META_VNEXT_V17_SLOW_ONLY_FAST_OUT_OF_SUPPORT",
        )
        self.assertFalse(route.task_context_supported)
        self.assertIsNotNone(route.decision_digest)

    def test_fast_state_updates_only_from_selected_search_outcome(self):
        runtime = self.runtime()
        route = runtime.route_session(
            arm=ArmCode.B,
            round_index=1,
            session=session(),
            static_router=StrongStaticRouterV1(),
            research_policy=initial_research_policy(),
        )
        runtime.record_observation(
            arm=ArmCode.B,
            round_index=1,
            candidate_id=route.selected_candidate_id,
            runtime_candidate_id="bl1-runtime-id",
            run_status="SUCCESS",
            ndcg=0.034,
            wall_time_ms=10_000,
            source_search_utility_event_digest=sha256_digest(
                {"event": "selected-result"}
            ),
        )
        projection = runtime.audit_projection()
        self.assertEqual(projection["states"]["B"]["round_boundary"], 1)
        self.assertEqual(projection["states"]["C"]["round_boundary"], 0)
        self.assertEqual(len(projection["observations"]), 1)
        self.assertEqual(
            projection["observations"][0]["runtime_candidate_id"],
            "bl1-runtime-id",
        )
        serialized = json.dumps(projection["observations"], sort_keys=True).lower()
        self.assertNotIn("claim_ceiling", serialized)
        self.assertNotIn("evidence_admission", serialized)

    def test_five_round_fast_state_recovers_after_singleton_round(self):
        runtime = self.runtime()
        for round_index in (1, 2):
            route = runtime.route_session(
                arm=ArmCode.B,
                round_index=round_index,
                session=session(round_index=round_index),
                static_router=StrongStaticRouterV1(),
                research_policy=initial_research_policy(),
            )
            runtime.record_observation(
                arm=ArmCode.B,
                round_index=round_index,
                candidate_id=route.selected_candidate_id,
                run_status="SUCCESS",
                ndcg=0.1,
                wall_time_ms=10_000,
                source_search_utility_event_digest=sha256_digest(
                    {"event": f"round-{round_index}"}
                ),
            )
        singleton = runtime.route_session(
            arm=ArmCode.B,
            round_index=3,
            session=session(
                round_index=3,
                anchors=("LIGHTGCN",) * 4,
            ),
            static_router=StrongStaticRouterV1(),
            research_policy=initial_research_policy(),
        )
        self.assertEqual(
            singleton.mode,
            "STATIC_SINGLETON_INSUFFICIENT_META_POOL",
        )
        runtime.record_observation(
            arm=ArmCode.B,
            round_index=3,
            candidate_id=singleton.selected_candidate_id,
            run_status="SUCCESS",
            ndcg=0.1,
            wall_time_ms=10_000,
            source_search_utility_event_digest=sha256_digest(
                {"event": "round-3-singleton"}
            ),
        )
        for round_index in (4, 5):
            route = runtime.route_session(
                arm=ArmCode.B,
                round_index=round_index,
                session=session(round_index=round_index),
                static_router=StrongStaticRouterV1(),
                research_policy=initial_research_policy(),
            )
            runtime.record_observation(
                arm=ArmCode.B,
                round_index=round_index,
                candidate_id=route.selected_candidate_id,
                run_status="SUCCESS",
                ndcg=0.1 + round_index / 1000,
                wall_time_ms=10_000,
                source_search_utility_event_digest=sha256_digest(
                    {"event": f"round-{round_index}"}
                ),
            )
        projection = runtime.audit_projection()
        self.assertEqual(projection["states"]["B"]["round_boundary"], 5)
        self.assertEqual(
            projection["states"]["B"]["observed_response_rounds"],
            [1, 2, 4, 5],
        )

    def test_activation_receipt_is_future_only_and_schedule_bound(self):
        receipt = build_activation_receipt()
        preimage = dict(receipt)
        digest = preimage.pop("content_digest")
        self.assertEqual(sha256_digest(preimage), digest)
        self.assertTrue(receipt["activation_applied"])
        self.assertFalse(receipt["historical_activation_applied"])
        self.assertFalse(receipt["historical_campaign_backfill"])
        self.assertEqual(receipt["activation_boundary"], "NEXT_CAMPAIGN")
        self.assertEqual(receipt["rounds_per_arm"], 5)

    def test_quality_review_separates_engineering_from_information_value(self):
        rows = [
            {
                "opaque_instance_id": instance,
                "round_index": round_index,
                "run_status": "SUCCESS",
                "ndcg": 0.1,
            }
            for instance in ("opaque-a", "opaque-b", "opaque-c")
            for round_index in range(1, 6)
        ]
        routes = [
            {
                "mode": "META_VNEXT_V17_SLOW_PLUS_FAST",
                "pool_candidate_semantics_digests": [
                    f"{index + offset:064x}" for offset in range(4)
                ],
                "round_semantic_collision_count": 0,
                "selected_candidate_id": f"selected-{index}",
                "selected_candidate_semantics_digest": f"{index:064x}",
                "static_champion_candidate_id": f"static-{index}",
                "task_context_supported": True,
            }
            for index in range(1, 11)
        ]
        meta_audit = {
            "observations": [
                {"selected_axis": axis}
                for axis in (
                    "objective",
                    "propagation",
                    "architecture",
                    "objective",
                    "propagation",
                    "architecture",
                    "objective",
                    "propagation",
                    "architecture",
                    "objective",
                )
            ],
            "routes": routes,
            "states": {
                "B": {"round_boundary": 5},
                "C": {"round_boundary": 5},
            },
        }
        review = pilot_quality_review(
            rows=rows,
            meta_audit=meta_audit,
            expected_instance_ids={"opaque-a", "opaque-b", "opaque-c"},
            guard_call_count=10,
        )
        self.assertTrue(review["engineering_closed"])
        self.assertTrue(review["expansion_has_information_value"])

        unsupported_audit = copy.deepcopy(meta_audit)
        for route in unsupported_audit["routes"]:
            route["mode"] = "STATIC_OUT_OF_SUPPORT_META_FALLBACK"
            route["task_context_supported"] = False
        unsupported_review = pilot_quality_review(
            rows=rows,
            meta_audit=unsupported_audit,
            expected_instance_ids={"opaque-a", "opaque-b", "opaque-c"},
            guard_call_count=10,
        )
        self.assertTrue(unsupported_review["engineering_closed"])
        self.assertFalse(unsupported_review["expansion_has_information_value"])
        self.assertEqual(
            unsupported_review["quality"]["task_supported_route_fraction"],
            0.0,
        )

    def test_five_round_effect_summary_stays_development_only(self):
        rows = [
            {
                "candidate_id": f"{arm}-{round_index}",
                "ndcg": base + round_index / 1000,
                "opaque_instance_id": f"opaque-{arm.lower()}",
                "round_index": round_index,
                "run_status": "SUCCESS",
            }
            for arm, base in (("A", 0.10), ("B", 0.12), ("C", 0.125))
            for round_index in range(1, 6)
        ]
        result = descriptive_effect_summary(
            rows,
            {"A": "opaque-a", "B": "opaque-b", "C": "opaque-c"},
        )
        self.assertAlmostEqual(
            result["aggregate_descriptive"]["B_minus_A"],
            0.02,
        )
        self.assertAlmostEqual(
            result["aggregate_descriptive"]["C_minus_B"],
            0.005,
        )
        self.assertAlmostEqual(
            result["aggregate_descriptive"]["final_frontier_contrasts"][
                "B_minus_A"
            ],
            0.02,
        )
        self.assertAlmostEqual(
            result["aggregate_descriptive"]["final_frontier_contrasts"][
                "C_minus_B"
            ],
            0.005,
        )
        self.assertFalse(result["formal_inference"])
        self.assertFalse(result["main_evidence"])

    def test_provider_compatible_schema_binds_one_closed_recipe(self):
        schema = json.loads(
            (
                SRC
                / "recclaw_core"
                / "experiments"
                / "helix_abc_v1"
                / "resources"
                / "pilot_proposal_response_v3.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertNotIn("oneOf", json.dumps(schema))
        proposal = {
            "candidate_label": "graph propagation control",
            "expected_signal": "bounded ranking change",
            "failure_mode": "neutral result",
            "hypothesis": "LightGCN may improve graph collaborative filtering.",
            "proposal_intent": "DISCOVERY",
            "recipe": "LIGHTGCN",
            "utility_features": {
                "blocker_risk": 0.1,
                "cost": 0.2,
                "frontier_potential": 0.7,
                "information_gain": 0.7,
                "runnable_probability": 0.9,
                "useful_signal": 0.7,
            },
        }
        Draft202012Validator(schema).validate({"proposals": [proposal]})
        self.assertEqual(_template_name(proposal), "LIGHTGCN")
        materialized = _program_from_proposal(
            proposal,
            _load_templates(FIXTURES),
        )
        self.assertIn(
            "recipe=LIGHTGCN",
            materialized["program_payload"]["mechanism_explanation"],
        )

    def test_real_broker_routes_meta_only_after_common_eligibility(self):
        runtime = self.runtime()
        broker = RealCanaryProposalBrokerV1.create(
            upstream=RecipeUpstream(),
            template_path=FIXTURES,
            adaptive_memory=True,
            campaign_meta_runtime=runtime,
        )
        generated = broker.generate(
            arm=ArmCode.B,
            round_index=1,
            search_seed=META_V17_PILOT_SEARCH_SEED,
            drafts=(),
            ceilings=ceilings(),
        )
        self.assertEqual(generated.selected_candidate_id, "")
        self.assertIsNone(generated.research_plan)
        eligible = tuple(
            str(compile_program(deep_thaw(program)).candidate_id)
            for program in generated.validation_programs
        )
        routed = broker.finalize_common_route(
            arm=ArmCode.B,
            round_index=1,
            session=generated,
            common_eligible_candidate_ids=eligible,
        )
        self.assertEqual(
            routed.selected_candidate_id,
            str(
                compile_program(
                    deep_thaw(routed.ordered_programs[0])
                ).candidate_id
            ),
        )
        self.assertEqual(
            routed.ordered_programs[0],
            next(
                proposal.mechanism_program
                for proposal in routed.research_proposals
                if proposal.candidate_id
                == routed.research_plan.selected_candidate_id
            ),
        )


if __name__ == "__main__":
    unittest.main()
