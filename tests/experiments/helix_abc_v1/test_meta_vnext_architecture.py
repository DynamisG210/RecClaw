from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext import (  # noqa: E402
    RANK_FEATURE_NAMES_V1,
    ChangeClassV1,
    CompletePoolEpisodeV1,
    EpisodeSplitV1,
    HeldoutFreshnessV1,
    MetaVNextEvaluationError,
    MetaVNextLearningError,
    MetaVNextRouterV1,
    MetaVNextRoutingError,
    MetaVNextShadowRuntimeV1,
    PairwiseRidgeTrainerV1,
    PairwiseSlowPolicyV1,
    PromotionCriteriaV1,
    ResearchContextV1,
    ResearchValueWeightsV1,
    SearchValueObservationV1,
    advance_fast_without_observation,
    bind_three_arm_shadow,
    create_promotion_decision,
    evaluate_heldout_sequences,
    initialize_fast_residual,
    materialize_candidate_pool,
    static_champion_candidate,
    update_fast_residual,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (  # noqa: E402
    FixtureProducerBrokerV1,
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (  # noqa: E402
    CandidateProposalV2,
    DiscoveryCreditV1,
    ProposalIntentV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext.routing import (  # noqa: E402
    fast_prediction,
)


FIXTURES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
PRODUCERS = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)
ANCHORS = ("BPR_MF", "LIGHTGCN", "SGL", "ULTRAGCN")
AXES = ("objective", "propagation", "self_supervision", "architecture")


def anchor_program(name: str, learning_rate: float) -> dict[str, object]:
    document = json.loads(FIXTURES.read_text(encoding="utf-8"))
    program = copy.deepcopy(
        next(
            item["program"]
            for item in document["fixtures"]
            if item["anchor_name"] == name
        )
    )
    for component in program["program_payload"]["components"]:
        if component["slot_id"] == "TRAINING_PROCEDURE":
            component["parameters"]["learning_rate"] = learning_rate
    return program


def context(round_fraction: float = 0.2) -> ResearchContextV1:
    return ResearchContextV1(
        round_fraction=round_fraction,
        remaining_execution_fraction=0.8,
        remaining_token_fraction=0.75,
        remaining_gpu_fraction=0.9,
        starting_frontier=0.4,
        recent_frontier_gain=0.02,
        stagnation_fraction=0.3,
        axis_coverage=tuple((axis, 0.25) for axis in AXES),
        exact_duplicate_count=0,
        near_duplicate_count=1,
        blocker_count=0,
        lineage_depth=1,
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


def proposal_set(
    suffix: int,
    *,
    producers: tuple[str, ...] = PRODUCERS,
    tune_control: bool = False,
) -> tuple[
    tuple[CandidateProposalV2, ...],
    dict[str, dict[str, object]],
]:
    learning_rate = 0.001 + suffix / 100000.0
    parent = anchor_program("BPR_MF", learning_rate)
    utilities = (0.94, 0.82, 0.62, 0.76)
    intents = (
        ProposalIntentV1.CONTROL,
        ProposalIntentV1.DISCOVERY,
        ProposalIntentV1.FALSIFICATION,
        ProposalIntentV1.DISCOVERY,
    )
    proposals: list[CandidateProposalV2] = []
    parents: dict[str, dict[str, object]] = {}
    for index, (anchor, producer, axis, utility, intent) in enumerate(
        zip(ANCHORS, producers, AXES, utilities, intents, strict=True)
    ):
        candidate_id = f"cand-vnext-{suffix:03d}-{index}"
        program = anchor_program(anchor, learning_rate)
        if index == 0 and tune_control:
            for component in program["program_payload"]["components"]:
                if component["slot_id"] == "TRAINING_PROCEDURE":
                    component["parameters"]["learning_rate"] += 0.0002
        proposals.append(
            CandidateProposalV2(
                candidate_id=candidate_id,
                producer_id=producer,
                producer_role=PRODUCERS[index],
                proposal_intent=intent,
                discovery_credit=(
                    DiscoveryCreditV1.NON_DISCOVERY_CONTROL
                    if intent is ProposalIntentV1.CONTROL
                    else DiscoveryCreditV1.DISCOVERY
                ),
                mechanism_axis=axis,
                mechanism_program=program,
                utility_features=SearchUtilityFeaturesV1(
                    runnable_probability=0.95,
                    useful_signal=utility,
                    frontier_potential=utility,
                    information_gain=0.5 + index * 0.05,
                    cost=0.2 + index * 0.05,
                    blocker_risk=0.05,
                ),
                parent_candidate_id=f"parent-{suffix:03d}",
                assigned_before_call=True,
                post_hoc_relabel=False,
            )
        )
        parents[candidate_id] = copy.deepcopy(parent)
    return tuple(proposals), parents


def pool(suffix: int, *, producers: tuple[str, ...] = PRODUCERS):
    proposals, parents = proposal_set(suffix, producers=producers)
    router = StrongStaticRouterV1()
    materialized = materialize_candidate_pool(
        pool_id=f"vnext-pool-{suffix:03d}",
        proposals=proposals,
        parent_programs=parents,
        research_context=context(min(0.9, 0.1 + suffix / 100.0)),
        producer_invocation_digests=tuple(
            sha256_digest({"suffix": suffix, "producer": producer})
            for producer in producers
        ),
        pre_round_state_digest=sha256_digest({"pre-round": suffix}),
        candidate_order_policy_digest=sha256_digest("SOURCE_ORDER_V1"),
        static_router=router,
        lineage_depths={item.candidate_id: 1 for item in proposals},
    )
    return proposals, materialized


def episode(
    suffix: int,
    *,
    split: EpisodeSplitV1,
    group: str,
    lineage: str,
    sequence_index: int = 1,
    freshness: HeldoutFreshnessV1 = HeldoutFreshnessV1.DEVELOPMENT,
) -> CompletePoolEpisodeV1:
    _, candidate_pool = pool(suffix)
    frontier_by_axis = {
        "objective": 0.00,
        "propagation": 0.35,
        "self_supervision": 0.80,
        "architecture": 0.20,
    }
    observations = tuple(
        SearchValueObservationV1(
            candidate_semantics_digest=item.candidate_semantics_digest,
            source_search_utility_event_digest=sha256_digest(
                {"episode": suffix, "candidate": item.candidate_semantics_digest}
            ),
            frontier_value=frontier_by_axis[item.primary_mechanism_axis],
            discriminative_value=(
                0.20 if item.ablation_or_falsification else 0.0
            ),
            normalized_cost=item.static_utility_features.cost,
            blocker_loss=0.0,
            round_boundary=sequence_index,
        )
        for item in candidate_pool.eligible_candidates
    )
    return CompletePoolEpisodeV1(
        episode_id=f"episode-{suffix:03d}",
        episode_group_id=group,
        lineage_group_id=lineage,
        sequence_index=sequence_index,
        split=split,
        heldout_freshness=freshness,
        pool=candidate_pool,
        observations=observations,
    )


def fitted_policy() -> PairwiseSlowPolicyV1:
    weights = ResearchValueWeightsV1(
        frontier=1.0,
        discriminative=0.5,
        cost=0.15,
        blocker=1.0,
    )
    training = tuple(
        episode(
            suffix,
            split=EpisodeSplitV1.TRAIN,
            group=f"train-group-{suffix}",
            lineage=f"train-lineage-{suffix}",
        )
        for suffix in (1, 2, 3)
    )
    validation = tuple(
        episode(
            suffix,
            split=EpisodeSplitV1.VALIDATION,
            group=f"validation-group-{suffix}",
            lineage=f"validation-lineage-{suffix}",
        )
        for suffix in (11, 12)
    )
    return PairwiseRidgeTrainerV1().fit(
        training_episodes=training,
        validation_episodes=validation,
        value_weights=weights,
        parent_checkpoint_digest=sha256_digest("parent-checkpoint"),
        static_router_policy_digest=StrongStaticRouterV1().policy_digest,
    )


class MetaVNextFeatureAndChampionTests(unittest.TestCase):
    def test_score_schema_and_imports_exclude_identity_guard_and_fusion(self) -> None:
        forbidden_features = ("candidate", "digest", "producer", "guard", "evidence")
        self.assertFalse(
            any(
                token in feature_name.lower()
                for feature_name in RANK_FEATURE_NAMES_V1
                for token in forbidden_features
            )
        )
        package = (
            SRC
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "meta_vnext"
        )
        imported_modules: set[str] = set()
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(item.name for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)
        self.assertFalse(
            any(
                token in module.lower()
                for module in imported_modules
                for token in ("evidence_guard", "fusion")
            )
        )

    def test_materializes_real_parent_relative_change_classes(self) -> None:
        proposals, parents = proposal_set(21, tune_control=True)
        candidate_pool = materialize_candidate_pool(
            pool_id="change-class-pool",
            proposals=proposals,
            parent_programs=parents,
            research_context=context(),
            producer_invocation_digests=tuple(
                sha256_digest({"producer": item}) for item in PRODUCERS
            ),
            pre_round_state_digest=sha256_digest("pre-round"),
            candidate_order_policy_digest=sha256_digest("source-order"),
            static_router=StrongStaticRouterV1(),
        )
        classes = {
            item.primary_mechanism_axis: item.change_class
            for item in candidate_pool.eligible_candidates
        }
        self.assertEqual(
            classes["objective"], ChangeClassV1.PARAMETER_TUNING_ONLY
        )
        self.assertIn(
            classes["propagation"],
            {ChangeClassV1.MECHANISM_CHANGE, ChangeClassV1.ARCHITECTURE_REWRITE},
        )
        self.assertEqual(
            tuple(name for name, _ in candidate_pool.candidates[0].rank_features),
            RANK_FEATURE_NAMES_V1,
        )
        self.assertIn("axis_self_supervision", RANK_FEATURE_NAMES_V1)
        self.assertIn("intervention_x_stagnation", RANK_FEATURE_NAMES_V1)
        self.assertNotIn("stagnation_fraction", RANK_FEATURE_NAMES_V1)

    def test_static_champion_is_actual_runtime_router(self) -> None:
        proposals, candidate_pool = pool(22)
        runtime = StrongStaticRouterV1().route(proposals)
        champion = static_champion_candidate(candidate_pool)
        self.assertEqual(champion.candidate_id, runtime.selected_candidate_id)

    def test_duplicate_is_hard_rejected_but_source_pool_remains_bound(self) -> None:
        proposals, parents = proposal_set(24)
        duplicated = (
            *proposals[:3],
            replace(
                proposals[3],
                mechanism_program=proposals[1].to_dict()["mechanism_program"],
            ),
        )
        candidate_pool = materialize_candidate_pool(
            pool_id="duplicate-pool",
            proposals=duplicated,
            parent_programs=parents,
            research_context=context(),
            producer_invocation_digests=tuple(
                sha256_digest({"producer": item}) for item in PRODUCERS
            ),
            pre_round_state_digest=sha256_digest("duplicate-pre-round"),
            candidate_order_policy_digest=sha256_digest("source-order"),
            static_router=StrongStaticRouterV1(),
        )
        self.assertEqual(
            candidate_pool.rejected_candidate_ids,
            (proposals[3].candidate_id,),
        )
        self.assertEqual(len(candidate_pool.candidates), 3)
        self.assertEqual(
            candidate_pool.source_candidate_pool_digest,
            sha256_digest([item.to_dict() for item in duplicated]),
        )

    def test_producer_permutation_does_not_change_rank_features(self) -> None:
        _, original = pool(23)
        permuted_producers = (
            PRODUCERS[2],
            PRODUCERS[3],
            PRODUCERS[0],
            PRODUCERS[1],
        )
        _, permuted = pool(23, producers=permuted_producers)
        original_features = {
            item.mechanism_program_digest: item.rank_features
            for item in original.candidates
        }
        permuted_features = {
            item.mechanism_program_digest: item.rank_features
            for item in permuted.candidates
        }
        self.assertEqual(original_features, permuted_features)
        self.assertNotEqual(
            tuple(item.producer_id for item in original.candidates),
            tuple(item.producer_id for item in permuted.candidates),
        )


class MetaVNextLearningAndRoutingTests(unittest.TestCase):
    def test_pairwise_fit_is_deterministic_and_uses_validation(self) -> None:
        first = fitted_policy()
        second = fitted_policy()
        self.assertEqual(first.digest, second.digest)
        self.assertIn(
            first.selected_regularization,
            PairwiseRidgeTrainerV1().regularization_grid,
        )
        self.assertIn(
            first.selected_static_coefficient,
            PairwiseRidgeTrainerV1().static_coefficient_grid,
        )
        static_index = first.feature_names.index("strong_static_score")
        self.assertEqual(
            first.coefficients[static_index],
            first.selected_static_coefficient,
        )
        self.assertEqual(
            PairwiseSlowPolicyV1.from_dict(first.to_dict()).digest,
            first.digest,
        )
        self.assertTrue(
            all(
                "candidate" not in name and "digest" not in name
                for name in first.feature_names
            )
        )
        _, validation_pool = pool(13)
        decision = MetaVNextRouterV1().route(validation_pool, first)
        selected = next(
            item
            for item in validation_pool.candidates
            if item.candidate_id == decision.selected_candidate_id
        )
        self.assertEqual(selected.primary_mechanism_axis, "self_supervision")

    def test_training_rejects_task_group_leakage(self) -> None:
        weights = ResearchValueWeightsV1(1.0, 0.5, 0.1, 1.0)
        training = episode(
            31,
            split=EpisodeSplitV1.TRAIN,
            group="shared-group",
            lineage="train-lineage",
        )
        validation = CompletePoolEpisodeV1(
            episode_id="validation-leak",
            episode_group_id="shared-group",
            lineage_group_id="validation-lineage",
            sequence_index=1,
            split=EpisodeSplitV1.VALIDATION,
            heldout_freshness=HeldoutFreshnessV1.DEVELOPMENT,
            pool=training.pool,
            observations=training.observations,
        )
        with self.assertRaises(MetaVNextLearningError):
            PairwiseRidgeTrainerV1().fit(
                training_episodes=(training,),
                validation_episodes=(validation,),
                value_weights=weights,
                parent_checkpoint_digest=sha256_digest("parent"),
                static_router_policy_digest=StrongStaticRouterV1().policy_digest,
            )

    def test_fast_state_is_arm_seed_private_and_missing_is_reason_free(self) -> None:
        policy = fitted_policy()
        _, candidate_pool = pool(41)
        arm = sha256_digest("arm-c")
        seed = sha256_digest("seed-1")
        state = initialize_fast_residual(
            opaque_arm_instance_digest=arm,
            search_seed_digest=seed,
            policy=policy,
        )
        decision = MetaVNextRouterV1().route(
            candidate_pool, policy, fast_state=state
        )
        selected = next(
            item
            for item in candidate_pool.candidates
            if item.candidate_id == decision.selected_candidate_id
        )
        observation = SearchValueObservationV1(
            candidate_semantics_digest=selected.candidate_semantics_digest,
            source_search_utility_event_digest=sha256_digest("visible-event"),
            frontier_value=0.5,
            discriminative_value=0.2,
            normalized_cost=0.2,
            blocker_loss=0.0,
            round_boundary=1,
        )
        updated = update_fast_residual(
            state,
            policy=policy,
            candidate=selected,
            observation=observation,
            opaque_arm_instance_digest=arm,
            search_seed_digest=seed,
            round_boundary=1,
        )
        self.assertEqual(updated.round_boundary, 1)
        self.assertEqual(updated.visible_search_value_event_digests, (observation.digest,))
        with self.assertRaises(MetaVNextRoutingError):
            update_fast_residual(
                state,
                policy=policy,
                candidate=selected,
                observation=observation,
                opaque_arm_instance_digest=sha256_digest("another-arm"),
                search_seed_digest=seed,
                round_boundary=1,
            )
        missing = advance_fast_without_observation(
            state,
            opaque_arm_instance_digest=arm,
            search_seed_digest=seed,
            slow_policy_digest=policy.digest,
            round_boundary=1,
        )
        self.assertEqual(missing.visible_search_value_event_digests, ())
        self.assertNotIn("reason", missing.to_dict())
        later_observation = replace(
            observation,
            source_search_utility_event_digest=sha256_digest(
                "visible-event-after-gap"
            ),
            round_boundary=2,
        )
        recovered = update_fast_residual(
            missing,
            policy=policy,
            candidate=selected,
            observation=later_observation,
            opaque_arm_instance_digest=arm,
            search_seed_digest=seed,
            round_boundary=2,
        )
        self.assertEqual(recovered.round_boundary, 2)
        self.assertEqual(recovered.observed_responses[0][0], 2)
        self.assertEqual(
            recovered.visible_search_value_event_digests,
            (later_observation.digest,),
        )

    def test_fast_prototype_prior_is_available_before_first_outcome(self) -> None:
        policy = fitted_policy()
        _, candidate_pool = pool(42)
        state = initialize_fast_residual(
            opaque_arm_instance_digest=sha256_digest("cold-start-arm"),
            search_seed_digest=sha256_digest("cold-start-seed"),
            policy=policy,
        )
        candidate = candidate_pool.eligible_candidates[0]
        control_axis = next(
            item.primary_mechanism_axis
            for item in candidate_pool.eligible_candidates
            if item.matched_control
        )
        correction, uncertainty = fast_prediction(
            state,
            candidate,
            policy,
            neighbor_count=5,
            task_control_axis=control_axis,
        )
        self.assertEqual(state.observed_responses, ())
        self.assertNotEqual(correction, 0.0)
        self.assertGreaterEqual(uncertainty, 0.0)
        self.assertTrue(
            all(len(prototype) == 6 for prototype in policy.fast_response_prototypes)
        )

    def test_out_of_support_task_falls_back_to_static_champion(self) -> None:
        policy = fitted_policy()
        _, candidate_pool = pool(43)
        shifted_candidates = tuple(
            replace(
                candidate,
                rank_features=tuple(
                    (
                        name,
                        (
                            2.0
                            if name
                            in {
                                (
                                    f"axis_{candidate.primary_mechanism_axis}"
                                    "_x_task_scale"
                                ),
                                (
                                    f"axis_{candidate.primary_mechanism_axis}"
                                    "_x_task_density"
                                ),
                            }
                            else value
                        ),
                    )
                    for name, value in candidate.rank_features
                ),
            )
            for candidate in candidate_pool.candidates
        )
        shifted_pool = replace(
            candidate_pool,
            candidates=shifted_candidates,
        )
        decision = MetaVNextRouterV1().route(shifted_pool, policy)
        champion = static_champion_candidate(shifted_pool)
        self.assertEqual(
            decision.selected_candidate_semantics_digest,
            champion.candidate_semantics_digest,
        )
        self.assertTrue(
            all(score.uncertainty == 0.0 for score in decision.scored_candidates)
        )

    def test_four_producer_shadow_round_binds_same_slow_policy(self) -> None:
        policy = fitted_policy()
        proposals, parent_programs = proposal_set(45)
        drafts = [
            {
                "mechanism_program": item.to_dict()["mechanism_program"],
                "mechanism_axis": item.mechanism_axis,
                "proposal_intent": item.proposal_intent.value,
                "utility_features": item.utility_features.to_dict(),
                "parent_candidate_id": item.parent_candidate_id,
            }
            for item in proposals
        ]
        session = FixtureProducerBrokerV1().dispatch(
            session_id="meta-vnext-shadow",
            mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
            drafts=drafts,
            context={"round_index": 1},
            role_memory={producer: {} for producer in PRODUCERS},
            seed=2045,
            ceilings=ceilings(),
        )
        parent = next(iter(parent_programs.values()))
        runtime_parents = {
            item.candidate_id: copy.deepcopy(parent) for item in session.proposals
        }
        arm_b = sha256_digest("arm-b")
        arm_c = sha256_digest("arm-c")
        seed = sha256_digest("search-seed")
        binding, fast_state = bind_three_arm_shadow(
            policy=policy,
            arm_b_instance_digest=arm_b,
            arm_c_instance_digest=arm_c,
            search_seed_digest=seed,
        )
        result = MetaVNextShadowRuntimeV1(
            static_router=StrongStaticRouterV1(),
            meta_router=MetaVNextRouterV1(),
        ).plan_shadow_round(
            session=session,
            parent_programs=runtime_parents,
            research_context=context(),
            pre_round_state_digest=sha256_digest("pre-round-shadow"),
            candidate_order_policy_digest=sha256_digest("source-order-shadow"),
            policy=policy,
            fast_state=fast_state,
        )
        self.assertEqual(binding.slow_policy_digest, policy.digest)
        self.assertEqual(fast_state.slow_policy_digest, policy.digest)
        self.assertEqual(session.physical_call_count, 4)
        self.assertEqual(result.runtime_authority, "NONE")
        self.assertTrue(result.slow_decision.shadow_mode)
        self.assertTrue(result.fast_decision.shadow_mode)
        self.assertEqual(
            result.static_champion_candidate_id,
            StrongStaticRouterV1().route(session.proposals).selected_candidate_id,
        )

    def test_previously_unblinded_evaluation_cannot_promote(self) -> None:
        policy = fitted_policy()
        heldout = (
            episode(
                51,
                split=EpisodeSplitV1.PROMOTION_HELDOUT,
                group="heldout-group-1",
                lineage="heldout-lineage-1",
                freshness=HeldoutFreshnessV1.PREVIOUSLY_UNBLINDED,
            ),
            episode(
                52,
                split=EpisodeSplitV1.PROMOTION_HELDOUT,
                group="heldout-group-2",
                lineage="heldout-lineage-2",
                freshness=HeldoutFreshnessV1.PREVIOUSLY_UNBLINDED,
            ),
        )
        report = evaluate_heldout_sequences(
            episodes=heldout,
            policy=policy,
            router=MetaVNextRouterV1(),
            criteria=PromotionCriteriaV1(
                minimum_group_count=2,
                confidence_level=0.95,
                minimum_slow_net_improvement=0.001,
                minimum_fast_increment=0.001,
                cost_noninferiority_margin=0.05,
                blocker_noninferiority_margin=0.0,
            ),
            evaluation_id="unblinded-diagnostic",
        )
        repeated = evaluate_heldout_sequences(
            episodes=heldout,
            policy=policy,
            router=MetaVNextRouterV1(),
            criteria=report.criteria,
            evaluation_id="unblinded-diagnostic",
        )
        self.assertEqual(report.digest, repeated.digest)
        self.assertEqual(report.meta_promotion_recommendation, "INCONCLUSIVE")
        self.assertFalse(report.candidate_id_leakage_detected)
        self.assertIsNotNone(report.slow_ranker.all_candidate_value_rmse)
        self.assertIsNotNone(report.slow_plus_fast.all_candidate_value_rmse)
        with self.assertRaises(MetaVNextEvaluationError):
            create_promotion_decision(
                report=report,
                parent_policy_digest=sha256_digest("parent"),
                activation_boundary="NEXT_CAMPAIGN",
            )

    def test_heldout_task_group_leakage_is_rejected(self) -> None:
        policy = fitted_policy()
        development_episode = episode(
            1,
            split=EpisodeSplitV1.PROMOTION_HELDOUT,
            group=policy.development_group_ids[0],
            lineage="heldout-overlap",
            freshness=HeldoutFreshnessV1.FRESH_BLINDED,
        )
        with self.assertRaises(MetaVNextEvaluationError):
            evaluate_heldout_sequences(
                episodes=(development_episode,),
                policy=policy,
                router=MetaVNextRouterV1(),
                criteria=PromotionCriteriaV1(
                    minimum_group_count=2,
                    confidence_level=0.95,
                    minimum_slow_net_improvement=0.001,
                    minimum_fast_increment=0.001,
                    cost_noninferiority_margin=0.0,
                    blocker_noninferiority_margin=0.0,
                ),
                evaluation_id="leakage-check",
            )


if __name__ == "__main__":
    unittest.main()
