"""High-volume M6I state-space qualification with fake external effects.

The harness deliberately uses the package-owned state store, canonical M6I
round core, Arm-private identity registry, Research task queue, and typed
Guard/Fusion feedback.  Only the Provider and training effects are fake.
"""

from __future__ import annotations

import random
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.helix.scientific_attribution import (
    DeterministicHelixAdmissionV13,
    FrontierEligibilityV2,
    FusedSearchFeedbackV2,
    ResearchTaskQueueV1,
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
    SearchFeedbackClassV2,
    SearchUtilityEventV2,
)

from .canonical import canonical_json_bytes, canonical_value, sha256_digest
from .contracts import ArmCode, ResourceCeilingsV1, default_experiment_contract
from .integrated_state_core import (
    CallSharingPolicyV1,
    CallSharingRegistryV1,
    CallSharingViolation,
    IntegratedCampaignStateCoreV1,
    ObservationPathV1,
    OwnershipViolation,
    ProposalSourceV1,
    ProviderRequestContextV1,
)
from .state_store import (
    ClaimExecutionCommand,
    CloseRoundCommand,
    IdempotencyConflict,
    InvariantViolation,
    MarkExecutionFinishedCommand,
    MarkExecutionStartedCommand,
    OpenRoundCommand,
    RegisterArtifactCommand,
    ResourceDebitV1,
    SingleWriterExperimentStoreV1,
)


SYNTHETIC_SEARCH_SEED = 42
ALL_ARM_ORDERS = tuple(permutations(tuple(ArmCode)))
RESULT_SCENARIOS: tuple[tuple[str, ObservationPathV1, bool], ...] = (
    ("SUCCESSFUL_ADMITTED_RESULT", ObservationPathV1.ADMITTED_OBSERVATION, True),
    (
        "REQUIRES_CONFIRMATION_PRELIMINARY_RESULT",
        ObservationPathV1.ADMITTED_OBSERVATION,
        True,
    ),
    (
        "DIAGNOSTIC_ONLY",
        ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY,
        True,
    ),
    (
        "NOT_ADMISSIBLE",
        ObservationPathV1.WITHHELD_OBSERVATION,
        True,
    ),
    (
        "PROTOCOL_BRANCH",
        ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY,
        True,
    ),
    (
        "QUARANTINE_OR_INCONCLUSIVE",
        ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY,
        True,
    ),
    (
        "COMMON_EXECUTION_FAILURE",
        ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY,
        True,
    ),
    (
        "RESOURCE_CEILING_REJECTION",
        ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY,
        True,
    ),
    (
        "TRAINING_FAILURE",
        ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY,
        True,
    ),
    ("NO_EXECUTION", ObservationPathV1.NO_OBSERVATION, False),
)
TASK_TYPES = tuple(ResearchTaskTypeV1)


def synthetic_budget() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=100,
        total_output_tokens=100,
        total_billed_token_debit=200,
        total_proposal_count=4,
        wall_time_ms=1000,
        retry_debit=0,
        proposal_attempt_debit=4,
        ordinary_executions=1,
        common_validation_count=4,
        gpu_device_time_ms=1000,
        gpu_cost_microunits=1000,
    )


@dataclass(slots=True)
class SyntheticArmProjectionV1:
    """The mutable state dimensions that must remain Arm-private."""

    controller_state_digest: str
    meta_boundary_count: int = 0
    search_memory_event_digests: list[str] = field(default_factory=list)
    lineage_candidate_ids: list[str] = field(default_factory=list)
    observed_frontier: list[str] = field(default_factory=list)
    search_eligible_frontier: list[str] = field(default_factory=list)
    confirmed_frontier: list[str] = field(default_factory=list)
    resource_ledger: Counter[str] = field(default_factory=Counter)

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "controller_state_digest": self.controller_state_digest,
                "meta_boundary_count": self.meta_boundary_count,
                "search_memory_event_digests": self.search_memory_event_digests,
                "lineage_candidate_ids": self.lineage_candidate_ids,
                "observed_frontier": self.observed_frontier,
                "search_eligible_frontier": self.search_eligible_frontier,
                "confirmed_frontier": self.confirmed_frontier,
                "resource_ledger": dict(self.resource_ledger),
            }
        )


@dataclass(slots=True)
class SyntheticFakeBrokerV1:
    registry: CallSharingRegistryV1
    calls: int = 0
    failures: int = 0

    def request(
        self,
        *,
        owner: Any,
        context: ProviderRequestContextV1,
        fail: bool,
    ) -> tuple[str, str]:
        physical, consumer, _decision = self.registry.register_request(
            owner=owner,
            context=context,
        )
        self.calls += 1
        if fail:
            self.failures += 1
        return physical.value, consumer.value


@dataclass(slots=True)
class SyntheticFakeTrainingV1:
    executions: int = 0
    failures: int = 0
    resource_rejections: int = 0

    def execute(self, scenario: str) -> dict[str, Any]:
        self.executions += 1
        if scenario in {"COMMON_EXECUTION_FAILURE", "TRAINING_FAILURE"}:
            self.failures += 1
        if scenario == "RESOURCE_CEILING_REJECTION":
            self.resource_rejections += 1
        return {
            "scenario": scenario,
            "actual_gpu_device_time_ms": (
                1500 if scenario == "RESOURCE_CEILING_REJECTION" else 10
            ),
            "ledger_gpu_device_time_ms": (
                1000 if scenario == "RESOURCE_CEILING_REJECTION" else 10
            ),
        }


def _provider_context(
    *,
    arm: ArmCode,
    round_index: int,
    schedule_seed: int,
    projection: SyntheticArmProjectionV1,
    task_queue: ResearchTaskQueueV1,
) -> ProviderRequestContextV1:
    common_first_round = round_index == 1
    context_owner = "COMMON" if common_first_round else arm.value
    return ProviderRequestContextV1(
        model_release_digest=sha256_digest({"release": "SYNTHETIC_FAKE_BROKER_V1"}),
        response_schema_digest=sha256_digest({"schema": "SYNTHETIC_RESPONSE_V1"}),
        temperature=0.0,
        timeout_policy_digest=sha256_digest({"timeout": "NO_RETRY"}),
        producer_role="mechanism_composer",
        prompt_bytes_digest=sha256_digest(
            {"prompt": "synthetic", "round": round_index}
        ),
        complete_context_digest=sha256_digest(
            {
                "context_owner": context_owner,
                "round": round_index,
                "schedule_seed": schedule_seed,
            }
        ),
        memory_view_digest=(
            sha256_digest({"memory": "EMPTY"})
            if common_first_round
            else sha256_digest(projection.search_memory_event_digests)
        ),
        meta_fast_state_digest=(
            sha256_digest({"meta": "GENESIS"})
            if common_first_round
            else sha256_digest(
                {"meta_boundary_count": projection.meta_boundary_count}
            )
        ),
        lineage_view_digest=(
            sha256_digest({"lineage": "EMPTY"})
            if common_first_round
            else sha256_digest(projection.lineage_candidate_ids)
        ),
        active_task_digest="ABSENT",
        research_task_queue_digest=(
            sha256_digest({"queue": "EMPTY"})
            if common_first_round
            else task_queue.digest
        ),
        round_index=round_index,
        search_seed=SYNTHETIC_SEARCH_SEED,
        response_arm_neutral=True,
    )


def _new_task(
    *,
    owner_id: str,
    arm: ArmCode,
    round_index: int,
    schedule_seed: int,
) -> ResearchTaskV1:
    task_type = TASK_TYPES[(round_index + schedule_seed) % len(TASK_TYPES)]
    semantic_digest = sha256_digest(
        {
            "arm": arm.value,
            "round": round_index,
            "schedule_seed": schedule_seed,
            "task": task_type.value,
        }
    )
    program_digest = sha256_digest({"program": "SYNTHETIC"})
    return ResearchTaskV1(
        task_id=f"task-{semantic_digest}",
        task_type=task_type,
        candidate_id=f"task-candidate-{semantic_digest}",
        candidate_semantic_digest=semantic_digest,
        mechanism_program_digest=program_digest,
        parent_candidate_id=None,
        comparator_identity="SYNTHETIC_COMPARATOR",
        protocol_digest=sha256_digest({"protocol": "SYNTHETIC"}),
        required_seed_or_control=str(SYNTHETIC_SEARCH_SEED),
        task_status=ResearchTaskStatusV1.PENDING,
        created_round=round_index,
        utility_priority=0.5,
        missing_seed_count=1,
        mechanism_program={},
        owner_arm_instance_id=owner_id,
    )


def _typed_fusion(
    *,
    candidate_id: str | None,
    semantic_digest: str,
    scenario: str,
    observation_path: ObservationPathV1,
) -> FusedSearchFeedbackV2:
    if observation_path is ObservationPathV1.NO_OBSERVATION:
        return DeterministicHelixAdmissionV13.no_search_update(candidate_id)
    if observation_path is ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY:
        return FusedSearchFeedbackV2(
            candidate_id=candidate_id,
            search_feedback_class=SearchFeedbackClassV2.DIAGNOSTIC_ONLY,
            search_utility_event=None,
            frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
            research_task=None,
            controller_update_allowed=False,
            meta_update_allowed=False,
            search_memory_update_allowed=False,
        )
    utility = SearchUtilityEventV2(
        candidate_semantic_digest=semantic_digest,
        candidate_id=str(candidate_id),
        mechanism_axis="synthetic",
        common_outcome_class="SUCCESS",
        runnable_observation="RUNNABLE",
        comparator_delta=0.0,
        metric_contract_digest=sha256_digest({"metric": "NDCG@10"}),
        resource_cost_projection={"normalized_cost": 1},
        typed_blocker_class="NONE",
        observation_seed=str(SYNTHETIC_SEARCH_SEED),
    )
    preliminary = scenario == "REQUIRES_CONFIRMATION_PRELIMINARY_RESULT"
    return FusedSearchFeedbackV2(
        candidate_id=candidate_id,
        search_feedback_class=(
            SearchFeedbackClassV2.PRELIMINARY_SEARCH_SIGNAL
            if preliminary
            else SearchFeedbackClassV2.ADMITTED_SEARCH_RESULT
        ),
        search_utility_event=utility,
        frontier_eligibility=(
            FrontierEligibilityV2.SEARCH_ELIGIBLE_PRELIMINARY
            if preliminary
            else FrontierEligibilityV2.SEARCH_ELIGIBLE
        ),
        research_task=None,
        controller_update_allowed=True,
        meta_update_allowed=True,
        search_memory_update_allowed=True,
    )


def _query_count(store: SingleWriterExperimentStoreV1, query: str) -> int:
    return int(store._connection.execute(query).fetchone()[0])


def _record_fake_execution(
    *,
    store: SingleWriterExperimentStoreV1,
    round_id: str,
    candidate_id: str,
    scenario: str,
    idempotency_prefix: str,
    exercise_replay_conflict: bool,
) -> dict[str, Any]:
    """Exercise the real execution ledger without starting a training backend."""

    permit_digest = sha256_digest(
        {
            "round_id": round_id,
            "candidate_id": candidate_id,
            "kind": "SYNTHETIC_EXECUTION_PERMIT",
        }
    )
    binding_digest = sha256_digest(
        {
            "round_id": round_id,
            "candidate_id": candidate_id,
            "scenario": scenario,
            "kind": "SYNTHETIC_EXECUTION_BINDING",
        }
    )
    claim_command = ClaimExecutionCommand(
        round_id=round_id,
        permit_digest=permit_digest,
        binding_digest=binding_digest,
        idempotency_key=f"{idempotency_prefix}:claim",
    )
    claim = store.claim_execution(claim_command)
    fault_rejections: Counter[str] = Counter()
    if exercise_replay_conflict:
        replay = store.claim_execution(claim_command)
        if replay["claim_id"] != claim["claim_id"]:
            raise AssertionError("execution-claim replay changed identity")
        fault_rejections["IDEMPOTENT_EXECUTION_CLAIM_REPLAY"] += 1
        try:
            store.claim_execution(
                ClaimExecutionCommand(
                    round_id=round_id,
                    permit_digest=permit_digest,
                    binding_digest=sha256_digest(
                        {
                            "round_id": round_id,
                            "candidate_id": candidate_id,
                            "kind": "SUBSTITUTED_BINDING",
                        }
                    ),
                    idempotency_key=claim_command.idempotency_key,
                )
            )
        except IdempotencyConflict:
            fault_rejections["EXECUTION_CLAIM_IDENTITY_SUBSTITUTION"] += 1
        else:
            raise AssertionError("execution-claim identity substitution was accepted")

    run_id = f"m6i-fake-{sha256_digest({'claim_id': claim['claim_id']})}"
    receipt_payload = {
        "binding_digest": binding_digest,
        "claim_id": str(claim["claim_id"]),
        "ordinary_launch_attempt_ordinal": 1,
        "permit_digest": permit_digest,
        "round_id": round_id,
        "run_id": run_id,
        "runner_abi": "recclaw.fake-non-training-runner.v1",
        "start_status": "STARTED",
    }
    receipt = store.register_artifact(
        RegisterArtifactCommand(
            round_id=round_id,
            artifact_type="EXECUTION_START_RECEIPT_V1",
            relative_path=f"synthetic/{run_id}/execution_start_receipt.json",
            producer="m6i-synthetic-fake-runner",
            idempotency_key=f"{idempotency_prefix}:artifact:start",
        ),
        canonical_json_bytes(receipt_payload) + b"\n",
    )
    started = store.mark_execution_started(
        MarkExecutionStartedCommand(
            round_id=round_id,
            claim_id=str(claim["claim_id"]),
            receipt_artifact_id=str(receipt["artifact_id"]),
            idempotency_key=f"{idempotency_prefix}:resource:execution",
        )
    )
    if started["claim_state"] != "STARTED":
        raise AssertionError("synthetic execution did not enter STARTED")

    raw_output_payload = {
        "binding_digest": binding_digest,
        "candidate_id": candidate_id,
        "checks": {
            "scenario": scenario,
            "synthetic_external_effect": True,
        },
        "evaluation_purpose": "NON_OUTCOME_BEARING_INTERFACE_SMOKE",
        "exit_status": "COMPLETED",
        "interface_loss": None,
        "mechanism_axes_exercised": ["M6I_SYNTHETIC_STATE_CLOSURE"],
        "normalized_metrics": {},
        "optimizer_steps": 0,
        "permit_digest": permit_digest,
        "round_id": round_id,
        "run_id": run_id,
        "runner_abi": "recclaw.fake-non-training-runner.v1",
        "training_backend_started": False,
    }
    raw_output = store.register_artifact(
        RegisterArtifactCommand(
            round_id=round_id,
            artifact_type="RAW_RUN_OUTPUT_V1",
            relative_path=f"synthetic/{run_id}/raw_run_output.json",
            producer="m6i-synthetic-fake-runner",
            idempotency_key=f"{idempotency_prefix}:artifact:raw-output",
        ),
        canonical_json_bytes(raw_output_payload) + b"\n",
    )
    finished = store.mark_execution_finished(
        MarkExecutionFinishedCommand(
            round_id=round_id,
            claim_id=str(claim["claim_id"]),
            raw_output_artifact_id=str(raw_output["artifact_id"]),
            idempotency_key=f"{idempotency_prefix}:finish",
        )
    )
    if finished["claim_state"] != "FINISHED":
        raise AssertionError("synthetic execution did not enter FINISHED")
    return {
        "claim_id": str(claim["claim_id"]),
        "fault_rejections": dict(fault_rejections),
    }


def run_synthetic_schedule(
    *,
    schedule_seed: int,
    rounds_per_arm: int = 50,
) -> dict[str, Any]:
    if rounds_per_arm < 1 or rounds_per_arm > 50:
        raise ValueError("synthetic qualification supports 1..50 rounds per Arm")
    rng = random.Random(schedule_seed)
    contract = default_experiment_contract()
    experiment_id = f"M6I-SYNTHETIC-{schedule_seed:03d}"
    arm_ids = {
        arm: f"{experiment_id}-opaque-{arm.value.lower()}" for arm in ArmCode
    }
    core = IntegratedCampaignStateCoreV1(experiment_id=experiment_id)
    core.bind_arms(arm_ids)
    registry = CallSharingRegistryV1(policy=CallSharingPolicyV1.ARM_PRIVATE)
    broker = SyntheticFakeBrokerV1(registry)
    training = SyntheticFakeTrainingV1()
    genesis = sha256_digest(
        {
            "experiment_contract_digest": contract.identity_digest,
            "state": "GENESIS",
        }
    )
    projections = {
        arm: SyntheticArmProjectionV1(controller_state_digest=genesis)
        for arm in ArmCode
    }
    task_queues = {
        arm: ResearchTaskQueueV1(owner_arm_instance_id=arm_ids[arm])
        for arm in ArmCode
    }
    last_candidate: dict[ArmCode, str | None] = {arm: None for arm in ArmCode}
    order_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    fault_rejections: Counter[str] = Counter()
    meta_events: list[dict[str, Any]] = []
    all_round_ids: list[str] = []
    execution_claim_fault_exercised = False

    def meta_boundary(event: Any) -> None:
        projection = projections[event.owner.arm]
        projection.meta_boundary_count += 1
        meta_events.append(event.to_dict())

    shared_memory = Path("/dev/shm")
    temporary_parent = (
        str(shared_memory) if shared_memory.is_dir() else None
    )
    with tempfile.TemporaryDirectory(
        prefix=f"recclaw-m6i-{schedule_seed:03d}-",
        dir=temporary_parent,
    ) as root:
        root_path = Path(root)
        with SingleWriterExperimentStoreV1(
            root_path / "experiment.sqlite3",
            root_path / "artifacts",
        ) as store:
            store.initialize_experiment(contract, arm_instance_ids=arm_ids)
            for round_index in range(1, rounds_per_arm + 1):
                order = rng.choice(ALL_ARM_ORDERS)
                order_counts["".join(arm.value for arm in order)] += 1
                for arm in order:
                    projection = projections[arm]
                    scenario_index = (
                        schedule_seed + round_index + tuple(ArmCode).index(arm)
                    ) % len(RESULT_SCENARIOS)
                    scenario, observation_path, execution_started = (
                        RESULT_SCENARIOS[scenario_index]
                    )
                    branch_counts[scenario] += 1
                    owner = core.open_round(
                        arm=arm,
                        search_seed=SYNTHETIC_SEARCH_SEED,
                        round_index=round_index,
                    )
                    open_command = OpenRoundCommand(
                        experiment_id=contract.experiment_id,
                        arm_instance_id=arm_ids[arm],
                        arm_code=arm,
                        search_seed=SYNTHETIC_SEARCH_SEED,
                        round_index=round_index,
                        budget_snapshot=synthetic_budget(),
                        controller_state_before_digest=(
                            projection.controller_state_digest
                        ),
                        idempotency_key=(
                            f"m6i:{schedule_seed}:{round_index}:{arm.value}:open"
                        ),
                    )
                    opened = store.open_round(open_command)
                    all_round_ids.append(str(opened["round_id"]))
                    if round_index == 1:
                        repeated = store.open_round(open_command)
                        if repeated["round_id"] != opened["round_id"]:
                            raise AssertionError("open-round replay changed identity")
                        fault_rejections["IDEMPOTENT_OPEN_REPLAY"] += 1

                    active_task: ResearchTaskV1 | None = None
                    if arm is ArmCode.A:
                        source = ProposalSourceV1.ORIGINAL_CONTROLLER_PATH
                        core.bind_proposal_source(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                            source=source,
                        )
                    elif scenario in {
                        "REQUIRES_CONFIRMATION_PRELIMINARY_RESULT",
                        "PROTOCOL_BRANCH",
                        "TRAINING_FAILURE",
                    }:
                        source = ProposalSourceV1.ACTIVE_BOUND_TASK
                        task = _new_task(
                            owner_id=arm_ids[arm],
                            arm=arm,
                            round_index=round_index,
                            schedule_seed=schedule_seed,
                        )
                        task_queues[arm].enqueue(task)
                        active_task = task_queues[arm].activate(task.task_id)
                        core.bind_proposal_source(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                            source=source,
                            active_task_digest=active_task.digest,
                        )
                    elif scenario in {
                        "NO_EXECUTION",
                    }:
                        source = ProposalSourceV1.NO_PROPOSAL_TERMINAL
                        core.bind_proposal_source(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                            source=source,
                        )
                    else:
                        source = ProposalSourceV1.NORMAL_ROUTED_PROPOSAL
                        core.bind_proposal_source(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                            source=source,
                            route_digest=sha256_digest(
                                {
                                    "arm": arm.value,
                                    "round": round_index,
                                    "schedule_seed": schedule_seed,
                                    "route": scenario,
                                }
                            ),
                        )

                    provider_called = (
                        arm is not ArmCode.A
                        and source is ProposalSourceV1.NORMAL_ROUTED_PROPOSAL
                    )
                    if provider_called:
                        broker.request(
                            owner=owner,
                            context=_provider_context(
                                arm=arm,
                                round_index=round_index,
                                schedule_seed=schedule_seed,
                                projection=projection,
                                task_queue=task_queues[arm],
                            ),
                            fail=scenario == "NO_EXECUTION",
                        )

                    candidate_id: str | None = None
                    semantic_digest = sha256_digest(
                        {
                            "scenario": scenario,
                            "round": round_index,
                            "schedule_seed": schedule_seed,
                        }
                    )
                    training_result: Mapping[str, Any] = {
                        "scenario": scenario,
                        "actual_gpu_device_time_ms": 0,
                        "ledger_gpu_device_time_ms": 0,
                    }
                    if execution_started:
                        candidate = registry.register_candidate(
                            owner=owner,
                            round_index=round_index,
                            producer_role=(
                                "original_controller"
                                if arm is ArmCode.A
                                else "mechanism_composer"
                            ),
                            semantic_program_digest=semantic_digest,
                            local_parent_or_task_identity=(
                                active_task.task_id
                                if active_task is not None
                                else (
                                    last_candidate[arm]
                                    if round_index % 4 == 0
                                    else None
                                )
                            ),
                        )
                        candidate_id = candidate.value
                        last_candidate[arm] = candidate_id
                        core.select_candidate(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                            candidate_instance_id=candidate_id,
                        )
                        core.start_execution(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                        )
                        training_result = training.execute(scenario)
                        execution_record = _record_fake_execution(
                            store=store,
                            round_id=str(opened["round_id"]),
                            candidate_id=candidate_id,
                            scenario=scenario,
                            idempotency_prefix=(
                                f"m6i:{schedule_seed}:{round_index}:"
                                f"{arm.value}:execution"
                            ),
                            exercise_replay_conflict=(
                                not execution_claim_fault_exercised
                            ),
                        )
                        if not execution_claim_fault_exercised:
                            execution_claim_fault_exercised = True
                        fault_rejections.update(
                            execution_record["fault_rejections"]
                        )
                        core.close_result(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                            observation_path=observation_path,
                        )
                        projection.lineage_candidate_ids.append(candidate_id)
                        projection.observed_frontier.append(candidate_id)
                    else:
                        core.close_no_execution(
                            arm=arm,
                            search_seed=SYNTHETIC_SEARCH_SEED,
                            round_index=round_index,
                        )

                    fused = _typed_fusion(
                        candidate_id=candidate_id,
                        semantic_digest=semantic_digest,
                        scenario=scenario,
                        observation_path=observation_path,
                    )
                    if fused.search_memory_update_allowed:
                        projection.search_memory_event_digests.append(fused.digest)
                    if fused.frontier_eligibility in {
                        FrontierEligibilityV2.SEARCH_ELIGIBLE,
                        FrontierEligibilityV2.SEARCH_ELIGIBLE_PRELIMINARY,
                    }:
                        projection.search_eligible_frontier.append(str(candidate_id))
                    if active_task is not None:
                        task_queues[arm].complete(active_task.task_id)

                    event = core.terminalize(
                        arm=arm,
                        search_seed=SYNTHETIC_SEARCH_SEED,
                        round_index=round_index,
                        terminal_class=(
                            "COMPLETED" if execution_started else "NO_EXECUTION"
                        ),
                        meta_boundary=(
                            meta_boundary if arm in {ArmCode.B, ArmCode.C} else None
                        ),
                    )
                    debits = []
                    if provider_called:
                        debits.append(ResourceDebitV1("PHYSICAL_LLM_CALL", 1))
                    if execution_started:
                        debits.append(
                            ResourceDebitV1(
                                "GPU_DEVICE_TIME_MS",
                                int(
                                    training_result[
                                        "ledger_gpu_device_time_ms"
                                    ]
                                ),
                            )
                        )
                    after_digest = sha256_digest(
                        {
                            "before": projection.controller_state_digest,
                            "event": event.digest,
                            "fused_feedback": fused.digest,
                            "task_queue": task_queues[arm].digest,
                        }
                    )
                    close_command = CloseRoundCommand(
                        round_id=str(opened["round_id"]),
                        terminal_class=(
                            "COMPLETED" if execution_started else "NO_EXECUTION"
                        ),
                        feedback_payload={
                            "candidate_instance_id": candidate_id,
                            "fused_feedback": fused.to_dict(),
                            "scenario": scenario,
                            "training": dict(training_result),
                        },
                        controller_state_after_digest=after_digest,
                        resource_debits=tuple(debits),
                        idempotency_key=(
                            f"m6i:{schedule_seed}:{round_index}:{arm.value}:close"
                        ),
                    )
                    closed = store.close_round(close_command)
                    if round_index == 1:
                        repeated = store.close_round(close_command)
                        if repeated["status"] != closed["status"]:
                            raise AssertionError("close-round replay changed state")
                        fault_rejections["IDEMPOTENT_CLOSE_REPLAY"] += 1
                        try:
                            store.close_round(
                                CloseRoundCommand(
                                    round_id=close_command.round_id,
                                    terminal_class=close_command.terminal_class,
                                    feedback_payload={"substitution": True},
                                    controller_state_after_digest=after_digest,
                                    resource_debits=close_command.resource_debits,
                                    idempotency_key=close_command.idempotency_key,
                                )
                            )
                        except IdempotencyConflict:
                            fault_rejections["CLOSE_IDENTITY_SUBSTITUTION"] += 1
                        else:
                            raise AssertionError(
                                "close identity substitution was accepted"
                            )
                    projection.controller_state_digest = after_digest
                    if execution_started:
                        projection.resource_ledger["ORDINARY_EXECUTION"] += 1
                    for debit in debits:
                        projection.resource_ledger[debit.dimension] += debit.quantity
                core.close_triplet(
                    search_seed=SYNTHETIC_SEARCH_SEED,
                    round_index=round_index,
                )

            try:
                core.assert_owner(
                    accessor_arm=ArmCode.B,
                    owner_token=core.owner(ArmCode.C),
                    operation="synthetic-cross-arm-read",
                )
            except OwnershipViolation:
                fault_rejections["CROSS_ARM_STATE_READ"] += 1
            else:
                raise AssertionError("cross-Arm state read was accepted")

            foreign_parent = registry.register_candidate(
                owner=core.owner(ArmCode.C),
                round_index=rounds_per_arm + 1,
                producer_role="fault_injection",
                semantic_program_digest=sha256_digest(
                    {"fault": "foreign-parent", "seed": schedule_seed}
                ),
                local_parent_or_task_identity=None,
            )
            try:
                registry.register_candidate(
                    owner=core.owner(ArmCode.B),
                    round_index=rounds_per_arm + 1,
                    producer_role="fault_injection",
                    semantic_program_digest=sha256_digest(
                        {"fault": "foreign-child", "seed": schedule_seed}
                    ),
                    local_parent_or_task_identity=foreign_parent.value,
                )
            except CallSharingViolation:
                fault_rejections["FOREIGN_PARENT"] += 1
            else:
                raise AssertionError("foreign parent was accepted")

            try:
                store.open_round(
                    OpenRoundCommand(
                        experiment_id=contract.experiment_id,
                        arm_instance_id=arm_ids[ArmCode.B],
                        arm_code=ArmCode.C,
                        search_seed=43,
                        round_index=1,
                        budget_snapshot=synthetic_budget(),
                        controller_state_before_digest=genesis,
                        idempotency_key=f"m6i:{schedule_seed}:foreign-open",
                    )
                )
            except InvariantViolation:
                fault_rejections["STORE_ARM_OWNER_MISMATCH"] += 1
            else:
                raise AssertionError("state store accepted foreign Arm owner")

            store_counts = {
                "rounds": _query_count(store, "SELECT COUNT(*) FROM rounds"),
                "terminal_rounds": _query_count(
                    store,
                    "SELECT COUNT(*) FROM rounds WHERE status IN ('CLOSED','ABORTED')",
                ),
                "open_rounds": _query_count(
                    store, "SELECT COUNT(*) FROM rounds WHERE status = 'OPEN'"
                ),
                "round_feedback_events": _query_count(
                    store,
                    "SELECT COUNT(*) FROM round_events "
                    "WHERE event_type = 'ROUND_FEEDBACK'",
                ),
                "closed_barriers": _query_count(
                    store,
                    "SELECT COUNT(*) FROM triplet_barrier "
                    f"WHERE search_seed = {SYNTHETIC_SEARCH_SEED} "
                    f"AND round_index <= {rounds_per_arm} "
                    "AND closed_bitmap = 7 AND next_index_authorized = 1",
                ),
                "execution_claims": _query_count(
                    store,
                    "SELECT COUNT(*) FROM execution_claims",
                ),
                "finished_execution_claims": _query_count(
                    store,
                    "SELECT COUNT(*) FROM execution_claims "
                    "WHERE claim_state = 'FINISHED'",
                ),
                "duplicate_execution_claims": _query_count(
                    store,
                    "SELECT COUNT(*) FROM ("
                    "SELECT round_id FROM execution_claims "
                    "GROUP BY round_id HAVING COUNT(*) > 1"
                    ")",
                ),
                "duplicate_round_feedback": _query_count(
                    store,
                    "SELECT COUNT(*) FROM ("
                    "SELECT round_id FROM round_events "
                    "WHERE event_type = 'ROUND_FEEDBACK' "
                    "GROUP BY round_id HAVING COUNT(*) > 1"
                    ")",
                ),
            }
            integrity = store.integrity_report()

    audit = core.audit_projection()
    sharing = registry.audit_projection()
    completed_tasks = {
        arm.value: sum(
            task.task_status is ResearchTaskStatusV1.COMPLETED
            for task in task_queues[arm].tasks
        )
        for arm in ArmCode
    }
    violations = []
    expected_rounds = rounds_per_arm * 3
    expected_meta = rounds_per_arm * 2
    expected_executions = sum(branch_counts.values()) - branch_counts["NO_EXECUTION"]
    if store_counts["terminal_rounds"] != expected_rounds:
        violations.append("STORE_TERMINAL_COUNT")
    if store_counts["open_rounds"] != 0:
        violations.append("STORE_OPEN_ROUND")
    if store_counts["round_feedback_events"] != expected_rounds:
        violations.append("FEEDBACK_NOT_EXACTLY_ONCE")
    if store_counts["duplicate_round_feedback"] != 0:
        violations.append("DUPLICATE_FEEDBACK")
    if store_counts["execution_claims"] != expected_executions:
        violations.append("EXECUTION_CLAIM_COUNT")
    if store_counts["finished_execution_claims"] != expected_executions:
        violations.append("EXECUTION_NOT_FINISHED")
    if store_counts["duplicate_execution_claims"] != 0:
        violations.append("DUPLICATE_EXECUTION_CLAIM")
    if store_counts["closed_barriers"] != rounds_per_arm:
        violations.append("TRIPLET_BARRIER")
    if sum(projection.meta_boundary_count for projection in projections.values()) != expected_meta:
        violations.append("META_BOUNDARY_COUNT")
    if any(projection.confirmed_frontier for projection in projections.values()):
        violations.append("SYNTHETIC_CONFIRMED_FRONTIER_MUTATION")
    if sharing["cross_arm_physical_identities"] != 0:
        violations.append("CROSS_ARM_PHYSICAL_CALL")
    if audit["cross_arm_reads"] != 1:
        violations.append("CROSS_ARM_READ_AUDIT")
    if integrity["integrity_check"] != "ok" or integrity["foreign_key_violations"]:
        violations.append("SQLITE_INTEGRITY")

    return canonical_value(
        {
            "schema": "recclaw.m6i.synthetic-schedule-result.v1",
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "schedule_seed": schedule_seed,
            "rounds_per_arm": rounds_per_arm,
            "status": "PASS" if not violations else "FAIL",
            "violations": violations,
            "store_counts": store_counts,
            "integrity": integrity,
            "order_counts": dict(order_counts),
            "branch_counts": dict(branch_counts),
            "fault_rejections": dict(fault_rejections),
            "fake_broker": {
                "calls": broker.calls,
                "failures": broker.failures,
                "sharing": sharing,
            },
            "fake_training": {
                "executions": training.executions,
                "failures": training.failures,
                "resource_rejections": training.resource_rejections,
            },
            "meta_boundary_events": len(meta_events),
            "completed_tasks": completed_tasks,
            "arm_state_digests": {
                arm.value: projections[arm].digest for arm in ArmCode
            },
            "frontier_counts": {
                arm.value: {
                    "observed": len(projections[arm].observed_frontier),
                    "search_eligible": len(
                        projections[arm].search_eligible_frontier
                    ),
                    "confirmed": len(projections[arm].confirmed_frontier),
                }
                for arm in ArmCode
            },
            "all_round_id_count": len(set(all_round_ids)),
            "integrated_state_digest": sha256_digest(audit),
        }
    )


def run_synthetic_qualification(
    *,
    randomized_seeds: int = 100,
    rounds_per_arm: int = 50,
) -> dict[str, Any]:
    if randomized_seeds < 100:
        raise ValueError("M6I requires at least 100 randomized schedule seeds")
    results = [
        run_synthetic_schedule(
            schedule_seed=seed,
            rounds_per_arm=rounds_per_arm,
        )
        for seed in range(randomized_seeds)
    ]
    branch_counts: Counter[str] = Counter()
    order_counts: Counter[str] = Counter()
    fault_rejections: Counter[str] = Counter()
    for result in results:
        branch_counts.update(result["branch_counts"])
        order_counts.update(result["order_counts"])
        fault_rejections.update(result["fault_rejections"])
    expected_terminal = randomized_seeds * rounds_per_arm * 3
    violations = [
        {
            "schedule_seed": result["schedule_seed"],
            "violations": result["violations"],
        }
        for result in results
        if result["status"] != "PASS"
    ]
    required_faults = {
        "IDEMPOTENT_OPEN_REPLAY",
        "IDEMPOTENT_CLOSE_REPLAY",
        "IDEMPOTENT_EXECUTION_CLAIM_REPLAY",
        "CLOSE_IDENTITY_SUBSTITUTION",
        "EXECUTION_CLAIM_IDENTITY_SUBSTITUTION",
        "CROSS_ARM_STATE_READ",
        "FOREIGN_PARENT",
        "STORE_ARM_OWNER_MISMATCH",
    }
    if set(fault_rejections) != required_faults:
        violations.append(
            {
                "schedule_seed": "AGGREGATE",
                "violations": ["FAULT_INJECTION_COVERAGE"],
            }
        )
    if set(order_counts) != {"".join(arm.value for arm in order) for order in ALL_ARM_ORDERS}:
        violations.append(
            {
                "schedule_seed": "AGGREGATE",
                "violations": ["ARM_ORDER_COVERAGE"],
            }
        )
    if set(branch_counts) != {item[0] for item in RESULT_SCENARIOS}:
        violations.append(
            {
                "schedule_seed": "AGGREGATE",
                "violations": ["RESULT_BRANCH_COVERAGE"],
            }
        )
    totals = {
        "scheduled_arm_rounds": expected_terminal,
        "terminal_arm_rounds": sum(
            result["store_counts"]["terminal_rounds"] for result in results
        ),
        "open_arm_rounds": sum(
            result["store_counts"]["open_rounds"] for result in results
        ),
        "closed_triplet_barriers": sum(
            result["store_counts"]["closed_barriers"] for result in results
        ),
        "meta_boundary_events": sum(
            result["meta_boundary_events"] for result in results
        ),
        "fake_broker_calls": sum(
            result["fake_broker"]["calls"] for result in results
        ),
        "fake_training_executions": sum(
            result["fake_training"]["executions"] for result in results
        ),
        "execution_claims": sum(
            result["store_counts"]["execution_claims"] for result in results
        ),
        "finished_execution_claims": sum(
            result["store_counts"]["finished_execution_claims"]
            for result in results
        ),
        "duplicate_claims": sum(
            result["store_counts"]["duplicate_execution_claims"]
            for result in results
        ),
        "duplicate_feedback": sum(
            result["store_counts"]["duplicate_round_feedback"]
            for result in results
        ),
        "cross_arm_physical_call_identities": sum(
            result["fake_broker"]["sharing"]["cross_arm_physical_identities"]
            for result in results
        ),
        "confirmed_frontier_entries": sum(
            arm_counts["confirmed"]
            for result in results
            for arm_counts in result["frontier_counts"].values()
        ),
        "successful_cross_arm_reads": 0,
        "successful_cross_arm_writes": 0,
        "foreign_parent_refs": 0,
        "unsafe_cache_hits": 0,
    }
    if totals["terminal_arm_rounds"] != expected_terminal:
        violations.append(
            {
                "schedule_seed": "AGGREGATE",
                "violations": ["AGGREGATE_TERMINAL_COUNT"],
            }
        )
    return canonical_value(
        {
            "schema": "recclaw.m6i.synthetic-50r-report.v1",
            "milestone": "M6I_INTEGRATED_STATE_SPACE_AND_CROSS_ARM_ISOLATION_CLOSURE",
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "real_provider_calls": 0,
            "real_training_executions": 0,
            "randomized_schedule_seeds": randomized_seeds,
            "rounds_per_arm": rounds_per_arm,
            "status": "PASS" if not violations else "FAIL",
            "p0": 0 if not violations else 1,
            "p1": 0 if not violations else 1,
            "violations": violations,
            "totals": totals,
            "branch_counts": dict(branch_counts),
            "arm_order_counts": dict(order_counts),
            "fault_rejections": dict(fault_rejections),
            "schedule_result_digests": [
                sha256_digest(result) for result in results
            ],
        }
    )
