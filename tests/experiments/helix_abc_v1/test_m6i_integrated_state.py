from __future__ import annotations

import json
import random
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.compilation_cache import (
    compilation_cache_projection,
    compile_campaign_program,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.integrated_state_core import (
    CallSharingPolicyV1,
    CallSharingRegistryV1,
    CallSharingViolation,
    IntegratedCampaignStateCoreV1,
    ObservationPathV1,
    OwnershipViolation,
    ProposalSourceV1,
    ProviderRequestContextV1,
    RoundStateV1,
    RoundTransitionViolation,
    canonical_observation_path,
)
from recclaw_core.mechanism_space import compile_program


ROOT = Path(__file__).resolve().parents[3]
MATRIX = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "m6i_transition_matrix_v1.json"
)
FAILURES = ROOT / "tests" / "fixtures" / "m6i_failures"
ANCHORS = (
    ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
)


def digest(label: str) -> str:
    return sha256_digest({"label": label})


def core(experiment_id: str = "M6I-TEST") -> IntegratedCampaignStateCoreV1:
    item = IntegratedCampaignStateCoreV1(experiment_id=experiment_id)
    item.bind_arms(
        {
            ArmCode.A: f"{experiment_id}-opaque-a",
            ArmCode.B: f"{experiment_id}-opaque-b",
            ArmCode.C: f"{experiment_id}-opaque-c",
        }
    )
    return item


def provider_context(
    *,
    role: str = "mechanism_composer",
    round_index: int = 1,
    search_seed: int = 1,
) -> ProviderRequestContextV1:
    return ProviderRequestContextV1(
        model_release_digest=digest("model"),
        response_schema_digest=digest("schema"),
        temperature=0.0,
        timeout_policy_digest=digest("timeout"),
        producer_role=role,
        prompt_bytes_digest=digest("prompt"),
        complete_context_digest=digest("complete"),
        memory_view_digest=digest("memory"),
        meta_fast_state_digest=digest("meta"),
        lineage_view_digest=digest("lineage"),
        active_task_digest="ABSENT",
        research_task_queue_digest=digest("queue"),
        round_index=round_index,
        search_seed=search_seed,
        response_arm_neutral=True,
    )


def close_branch(
    item: IntegratedCampaignStateCoreV1,
    *,
    arm: ArmCode,
    search_seed: int,
    round_index: int,
    source: ProposalSourceV1,
    observation: ObservationPathV1,
) -> dict[str, object]:
    item.open_round(
        arm=arm, search_seed=search_seed, round_index=round_index
    )
    kwargs: dict[str, str] = {}
    if source is ProposalSourceV1.NORMAL_ROUTED_PROPOSAL:
        kwargs["route_digest"] = digest(
            f"route:{arm.value}:{search_seed}:{round_index}"
        )
    elif source is ProposalSourceV1.ACTIVE_BOUND_TASK:
        kwargs["active_task_digest"] = digest(
            f"task:{arm.value}:{search_seed}:{round_index}"
        )
    item.bind_proposal_source(
        arm=arm,
        search_seed=search_seed,
        round_index=round_index,
        source=source,
        **kwargs,
    )
    if observation is ObservationPathV1.NO_OBSERVATION:
        item.close_no_execution(
            arm=arm, search_seed=search_seed, round_index=round_index
        )
    else:
        item.select_candidate(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
            candidate_instance_id=(
                "cand-"
                + digest(
                    f"candidate:{arm.value}:{search_seed}:{round_index}"
                )
            ),
        )
        item.start_execution(
            arm=arm, search_seed=search_seed, round_index=round_index
        )
        item.close_result(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
            observation_path=observation,
        )
    item.terminalize(
        arm=arm,
        search_seed=search_seed,
        round_index=round_index,
        terminal_class=(
            "NO_EXECUTION"
            if observation is ObservationPathV1.NO_OBSERVATION
            else "COMPLETED"
        ),
    )
    return item.round_projection(
        arm=arm, search_seed=search_seed, round_index=round_index
    )


class M6IIdentityAndSharingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = core()
        self.b = self.state.owner(ArmCode.B)
        self.c = self.state.owner(ArmCode.C)
        self.a = self.state.owner(ArmCode.A)

    def test_arm_private_identical_contexts_never_cross_arm(self) -> None:
        registry = CallSharingRegistryV1(
            policy=CallSharingPolicyV1.ARM_PRIVATE
        )
        context = provider_context()
        b_physical, b_consumer, _ = registry.register_request(
            owner=self.b, context=context
        )
        c_physical, c_consumer, _ = registry.register_request(
            owner=self.c, context=context
        )
        self.assertNotEqual(b_physical.value, c_physical.value)
        self.assertNotEqual(b_consumer.value, c_consumer.value)
        audit = registry.audit_projection()
        self.assertEqual(audit["cross_arm_physical_identities"], 0)

    def test_feedback_classes_use_one_canonical_observation_taxonomy(
        self,
    ) -> None:
        self.assertEqual(
            canonical_observation_path(
                meta_update_allowed=True,
                search_feedback_class="ADMITTED_SEARCH_RESULT",
            ),
            ObservationPathV1.ADMITTED_OBSERVATION,
        )
        for feedback_class in (
            "COMMON_FAILED_EXECUTION",
            "DIAGNOSTIC_ONLY",
            "ENGINEERING_ONLY",
            "PROTOCOL_BRANCH_TASK",
        ):
            with self.subTest(feedback_class=feedback_class):
                self.assertEqual(
                    canonical_observation_path(
                        meta_update_allowed=False,
                        search_feedback_class=feedback_class,
                    ),
                    ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY,
                )
        for feedback_class in (
            "NO_SEARCH_UPDATE",
            "PRELIMINARY_SEARCH_SIGNAL",
        ):
            with self.subTest(feedback_class=feedback_class):
                self.assertEqual(
                    canonical_observation_path(
                        meta_update_allowed=False,
                        search_feedback_class=feedback_class,
                    ),
                    ObservationPathV1.WITHHELD_OBSERVATION,
                )

    def test_transition_matrix_matches_frozen_feedback_semantics(
        self,
    ) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        rows = {
            item["name"]: item["observation_path"]
            for item in matrix["result_paths"]
        }
        feedback_semantics = {
            "SUCCESSFUL_ADMITTED_RESULT": (
                True,
                "ADMITTED_SEARCH_RESULT",
            ),
            "REQUIRES_CONFIRMATION_PRELIMINARY_RESULT": (
                True,
                "PRELIMINARY_SEARCH_SIGNAL",
            ),
            "DIAGNOSTIC_ONLY": (False, "DIAGNOSTIC_ONLY"),
            "NOT_ADMISSIBLE": (False, "NO_SEARCH_UPDATE"),
            "PROTOCOL_BRANCH": (False, "PROTOCOL_BRANCH_TASK"),
            "QUARANTINE_OR_INCONCLUSIVE": (
                False,
                "DIAGNOSTIC_ONLY",
            ),
            "COMMON_EXECUTION_FAILURE": (
                False,
                "COMMON_FAILED_EXECUTION",
            ),
            "RESOURCE_CEILING_REJECTION": (
                False,
                "COMMON_FAILED_EXECUTION",
            ),
            "TRAINING_FAILURE": (
                False,
                "COMMON_FAILED_EXECUTION",
            ),
        }
        for scenario, (
            meta_update_allowed,
            feedback_class,
        ) in feedback_semantics.items():
            with self.subTest(scenario=scenario):
                self.assertEqual(
                    rows[scenario],
                    canonical_observation_path(
                        meta_update_allowed=meta_update_allowed,
                        search_feedback_class=feedback_class,
                    ).value,
                )

    def test_paired_exact_context_shares_only_physical_semantics(self) -> None:
        registry = CallSharingRegistryV1(
            policy=CallSharingPolicyV1.PAIRED_BC_EXACT_CONTEXT
        )
        context = provider_context()
        b_physical, b_consumer, b_decision = registry.register_request(
            owner=self.b, context=context
        )
        c_physical, c_consumer, c_decision = registry.register_request(
            owner=self.c, context=context
        )
        self.assertEqual(b_physical.value, c_physical.value)
        self.assertNotEqual(b_consumer.value, c_consumer.value)
        self.assertEqual((b_decision, c_decision), ("MISS", "HIT"))
        b_candidate = registry.register_candidate(
            owner=self.b,
            round_index=1,
            producer_role=context.producer_role,
            semantic_program_digest=digest("program"),
            local_parent_or_task_identity=None,
        )
        c_candidate = registry.register_candidate(
            owner=self.c,
            round_index=1,
            producer_role=context.producer_role,
            semantic_program_digest=digest("program"),
            local_parent_or_task_identity=None,
        )
        self.assertNotEqual(b_candidate.value, c_candidate.value)

    def test_each_complete_context_field_separates_paired_physical_call(self) -> None:
        base = provider_context()
        variants = {
            "model_release_digest": digest("different-model"),
            "response_schema_digest": digest("different-schema"),
            "temperature": 0.1,
            "timeout_policy_digest": digest("different-timeout"),
            "producer_role": "lineage_refiner",
            "prompt_bytes_digest": digest("different-prompt"),
            "complete_context_digest": digest("different-complete"),
            "memory_view_digest": digest("different-memory"),
            "meta_fast_state_digest": digest("different-meta"),
            "lineage_view_digest": digest("different-lineage"),
            "active_task_digest": digest("different-task"),
            "research_task_queue_digest": digest("different-queue"),
            "round_index": 2,
            "search_seed": 2,
        }
        for field_name, value in variants.items():
            with self.subTest(field_name=field_name):
                registry = CallSharingRegistryV1(
                    policy=CallSharingPolicyV1.PAIRED_BC_EXACT_CONTEXT
                )
                first, _, _ = registry.register_request(
                    owner=self.b, context=base
                )
                second, _, _ = registry.register_request(
                    owner=self.c,
                    context=replace(base, **{field_name: value}),
                )
                self.assertNotEqual(first.value, second.value)

    def test_a_cannot_consume_paired_research_call(self) -> None:
        registry = CallSharingRegistryV1(
            policy=CallSharingPolicyV1.PAIRED_BC_EXACT_CONTEXT
        )
        with self.assertRaisesRegex(CallSharingViolation, "Arm A"):
            registry.register_request(
                owner=self.a, context=provider_context()
            )

    def test_foreign_parent_is_rejected_before_lineage_resolution(self) -> None:
        registry = CallSharingRegistryV1()
        parent = registry.register_candidate(
            owner=self.c,
            round_index=1,
            producer_role="mechanism_composer",
            semantic_program_digest=digest("parent"),
            local_parent_or_task_identity=None,
        )
        with self.assertRaisesRegex(CallSharingViolation, "foreign Arm"):
            registry.register_candidate(
                owner=self.b,
                round_index=2,
                producer_role="lineage_refiner",
                semantic_program_digest=digest("child"),
                local_parent_or_task_identity=parent.value,
            )


class M6ICommonImmutableCompilationTest(unittest.TestCase):
    def test_campaign_cache_preserves_exact_compiler_reports(self) -> None:
        anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))[
            "fixtures"
        ]
        for fixture in anchors:
            with self.subTest(anchor=fixture["anchor_name"]):
                program = fixture["program"]
                uncached = compile_program(program)
                first = compile_campaign_program(program)
                second = compile_campaign_program(program)
                self.assertEqual(first.to_dict(), uncached.to_dict())
                self.assertIs(first, second)
        projection = compilation_cache_projection()
        self.assertEqual(projection["scope"], "COMMON_IMMUTABLE")
        self.assertEqual(
            projection["identity"], "EXACT_CANONICAL_PROGRAM_BYTES"
        )
        self.assertGreaterEqual(projection["hits"], len(anchors))

    def test_invalid_non_json_input_retains_original_error_report(self) -> None:
        invalid = {"not_json": object()}
        self.assertEqual(
            compile_campaign_program(invalid).to_dict(),
            compile_program(invalid).to_dict(),
        )


class M6IRoundStateMachineTest(unittest.TestCase):
    def test_machine_readable_matrix_covers_required_domains(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        self.assertEqual(len(matrix["proposal_paths"]), 9)
        self.assertEqual(len(matrix["result_paths"]), 10)
        self.assertEqual(len(matrix["meta_paths"]), 7)
        self.assertEqual(
            set(matrix["asserted_state_dimensions"]),
            {
                "controller_state",
                "meta_state",
                "search_memory",
                "lineage",
                "task_queue",
                "observed_frontier",
                "search_eligible_frontier",
                "confirmed_frontier",
                "resource_ledger",
                "round_state",
                "triplet_barrier",
            },
        )

    def test_every_proposal_and_result_branch_reaches_typed_terminal(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        round_index = 0
        for proposal in matrix["proposal_paths"]:
            round_index += 1
            with self.subTest(proposal_path=proposal["name"]):
                source = ProposalSourceV1(proposal["canonical_source"])
                no_execution = proposal["name"] in {
                    "ALL_PRE_CANDIDATES_BLOCKED",
                    "EMPTY_OR_INVALID_SLATE",
                    "PROVIDER_FAILURE_BEFORE_PROPOSAL",
                }
                projection = close_branch(
                    core(f"proposal-{round_index}"),
                    arm=(
                        ArmCode.A
                        if source
                        is ProposalSourceV1.ORIGINAL_CONTROLLER_PATH
                        else ArmCode.B
                    ),
                    search_seed=100,
                    round_index=round_index,
                    source=source,
                    observation=(
                        ObservationPathV1.NO_OBSERVATION
                        if no_execution
                        else ObservationPathV1.ADMITTED_OBSERVATION
                    ),
                )
                self.assertEqual(
                    projection["state"], RoundStateV1.ROUND_TERMINAL.value
                )
        for result in matrix["result_paths"]:
            round_index += 1
            with self.subTest(result_path=result["name"]):
                observation = ObservationPathV1(
                    result["observation_path"]
                )
                projection = close_branch(
                    core(f"result-{round_index}"),
                    arm=ArmCode.B,
                    search_seed=101,
                    round_index=round_index,
                    source=ProposalSourceV1.NORMAL_ROUTED_PROPOSAL,
                    observation=observation,
                )
                self.assertEqual(
                    projection["observation_path"], observation.value
                )
                self.assertEqual(projection["meta_boundary_count"], 1)

    def test_every_meta_branch_advances_exactly_once(self) -> None:
        matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        for round_index, branch in enumerate(matrix["meta_paths"], 1):
            with self.subTest(meta_path=branch["name"]):
                item = core(f"meta-{round_index}")
                projection = close_branch(
                    item,
                    arm=ArmCode.C,
                    search_seed=102,
                    round_index=round_index,
                    source=ProposalSourceV1(branch["proposal_source"]),
                    observation=ObservationPathV1(
                        branch["observation_path"]
                    ),
                )
                self.assertEqual(projection["meta_boundary_count"], 1)
                with self.assertRaisesRegex(
                    RoundTransitionViolation, "invalid from"
                ):
                    item.terminalize(
                        arm=ArmCode.C,
                        search_seed=102,
                        round_index=round_index,
                        terminal_class="DUPLICATE",
                    )

    def test_cross_arm_owner_read_fails_closed(self) -> None:
        item = core()
        with self.assertRaisesRegex(OwnershipViolation, "foreign Arm"):
            item.assert_owner(
                accessor_arm=ArmCode.B,
                owner_token=item.owner(ArmCode.C),
                operation="read-lineage",
            )
        self.assertEqual(item.audit_projection()["cross_arm_reads"], 1)


class M6IOrderAndRegressionTest(unittest.TestCase):
    @staticmethod
    def _run_schedule(
        orders: list[tuple[ArmCode, ArmCode, ArmCode]],
        *,
        experiment_id: str,
    ) -> dict[str, list[dict[str, object]]]:
        item = core(experiment_id)
        for round_index, order in enumerate(orders, 1):
            for arm in order:
                source = (
                    ProposalSourceV1.ORIGINAL_CONTROLLER_PATH
                    if arm is ArmCode.A
                    else (
                        ProposalSourceV1.ACTIVE_BOUND_TASK
                        if round_index % 3 == 0
                        else ProposalSourceV1.NORMAL_ROUTED_PROPOSAL
                    )
                )
                observation = (
                    ObservationPathV1.NO_OBSERVATION
                    if round_index % 5 == 0
                    else (
                        ObservationPathV1.WITHHELD_OBSERVATION
                        if arm is ArmCode.C and round_index % 2 == 0
                        else ObservationPathV1.ADMITTED_OBSERVATION
                    )
                )
                close_branch(
                    item,
                    arm=arm,
                    search_seed=200,
                    round_index=round_index,
                    source=source,
                    observation=observation,
                )
            item.close_triplet(search_seed=200, round_index=round_index)
        result: dict[str, list[dict[str, object]]] = {}
        for arm in ArmCode:
            result[arm.value] = [
                {
                    "proposal_source": item.round_projection(
                        arm=arm,
                        search_seed=200,
                        round_index=round_index,
                    )["proposal_source"],
                    "observation_path": item.round_projection(
                        arm=arm,
                        search_seed=200,
                        round_index=round_index,
                    )["observation_path"],
                    "meta_boundary_count": item.round_projection(
                        arm=arm,
                        search_seed=200,
                        round_index=round_index,
                    )["meta_boundary_count"],
                    "state": item.round_projection(
                        arm=arm,
                        search_seed=200,
                        round_index=round_index,
                    )["state"],
                }
                for round_index in range(1, len(orders) + 1)
            ]
        return result

    def test_all_six_orders_are_arm_local_state_invariant(self) -> None:
        all_orders = list(permutations(tuple(ArmCode)))
        reference = self._run_schedule(
            [all_orders[0]] * 12, experiment_id="six-reference"
        )
        for index, order in enumerate(all_orders):
            with self.subTest(order="-".join(item.value for item in order)):
                actual = self._run_schedule(
                    [order] * 12, experiment_id=f"six-{index}"
                )
                self.assertEqual(reference, actual)

    def test_500_randomized_multi_round_schedules_are_invariant(self) -> None:
        all_orders = list(permutations(tuple(ArmCode)))
        reference = self._run_schedule(
            [all_orders[0]] * 8, experiment_id="random-reference"
        )
        for seed in range(500):
            rng = random.Random(seed)
            orders = [rng.choice(all_orders) for _ in range(8)]
            actual = self._run_schedule(
                orders, experiment_id=f"random-{seed}"
            )
            self.assertEqual(reference, actual)

    def test_v19_v20_v21_permanent_regressions(self) -> None:
        v19 = json.loads(
            (FAILURES / "V19_FAILURE_RECORD.json").read_text(
                encoding="utf-8"
            )
        )
        v20 = json.loads(
            (FAILURES / "V20_FAILURE_RECORD.json").read_text(
                encoding="utf-8"
            )
        )
        v21 = json.loads(
            (FAILURES / "V21_FAILURE_RECORD.json").read_text(
                encoding="utf-8"
            )
        )
        hard_stop = json.loads(
            (
                FAILURES / "V21_CROSS_ARM_CONTAMINATION_HARD_STOP.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            v19["failure_class"],
            "META_WITHHELD_ROUND_BOUNDARY_NOT_ADVANCED",
        )
        self.assertEqual(
            v20["failure_class"],
            "META_ACTIVE_TASK_ROUND_BOUNDARY_NOT_ADVANCED",
        )
        self.assertEqual(
            v21["failure_class"],
            "RESEARCH_PARENT_OUTSIDE_EXACT_LINEAGE",
        )
        self.assertEqual(hard_stop["p0"], 1)
        self.assertEqual(
            hard_stop["hard_stop_class"],
            "CROSS_ARM_CONTAMINATION",
        )

        v19_projection = close_branch(
            core("V19-REPLAY"),
            arm=ArmCode.C,
            search_seed=v19["search_seed"],
            round_index=1,
            source=ProposalSourceV1.NORMAL_ROUTED_PROPOSAL,
            observation=ObservationPathV1.WITHHELD_OBSERVATION,
        )
        self.assertEqual(v19_projection["meta_boundary_count"], 1)

        v20_projection = close_branch(
            core("V20-REPLAY"),
            arm=ArmCode.B,
            search_seed=v20["search_seed"],
            round_index=1,
            source=ProposalSourceV1.ACTIVE_BOUND_TASK,
            observation=ObservationPathV1.NO_OBSERVATION,
        )
        self.assertIsNone(v20_projection["route_digest"])
        self.assertEqual(v20_projection["meta_boundary_count"], 1)

        state = core("V21-REPLAY")
        registry = CallSharingRegistryV1(
            policy=CallSharingPolicyV1.ARM_PRIVATE
        )
        c_context = provider_context(search_seed=v21["search_seed"])
        b_context = replace(
            c_context,
            lineage_view_digest=digest("B-private-lineage"),
            memory_view_digest=digest("B-private-memory"),
        )
        c_physical, c_consumer, _ = registry.register_request(
            owner=state.owner(ArmCode.C), context=c_context
        )
        b_physical, b_consumer, _ = registry.register_request(
            owner=state.owner(ArmCode.B), context=b_context
        )
        self.assertNotEqual(c_physical.value, b_physical.value)
        self.assertNotEqual(c_consumer.value, b_consumer.value)


if __name__ == "__main__":
    unittest.main()
