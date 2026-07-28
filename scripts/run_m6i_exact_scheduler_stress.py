#!/usr/bin/env python3
"""Stress the production scheduler with fake Provider and fake training effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import permutations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.campaign_dataset import (  # noqa: E402
    campaign_development_protocol,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (  # noqa: E402
    campaign_runtime_profile,
    executable_mechanism,
    program_from_proposal,
)
from recclaw_core.experiments.helix_abc_v1.broker_process import (  # noqa: E402
    BrokerCallOutcomeV2,
    BrokerProcessExitReceiptV2,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerCallV1,
    CanaryBrokerError,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.compilation_cache import (  # noqa: E402
    compile_campaign_program,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    MetaV19CampaignRuntimeV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    ThreeArmPreCanaryOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
    canary_budget,
)
from recclaw_core.helix.scientific_attribution import (  # noqa: E402
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
)


CHECKPOINT = (
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "meta_vnext_policy_checkpoint_v19.json"
)
TEMPLATES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "M6I_EXACT_SCHEDULER_STRESS.json"
)
ALL_ORDERS = tuple(permutations(tuple(ArmCode)))
SEARCH_SEED = 42
QUALIFIED_SOURCE_PATHS = (
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "campaign_runtime.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "broker_failure_closure.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "common_execution_guard.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "compilation_cache.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "integrated_state_core.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "precanary_orchestration.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "real_canary.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "meta_vnext_campaign.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "meta_vnext"
    / "features.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "materialization.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "research_capability.py",
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "research_quality_gate.py",
    SRC
    / "recclaw_core"
    / "helix"
    / "scientific_attribution.py",
    Path(__file__).resolve(),
)


def _temporary_root_parent() -> str | None:
    shared_memory = Path("/dev/shm")
    return str(shared_memory) if shared_memory.is_dir() else None


def _source_projection() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in QUALIFIED_SOURCE_PATHS
    }


def _proposal(mechanism_id: str, intent: str) -> dict[str, Any]:
    mechanism = executable_mechanism(mechanism_id)
    return {
        "candidate_label": mechanism_id,
        "composition": mechanism.prompt_projection()["composition"],
        "competing_hypothesis": "a competing mechanism explains the change",
        "failure_mode": "the declared signature is absent",
        "mechanism_hypothesis": "the declared mechanism changes ranking utility",
        "mechanism_id": mechanism.mechanism_id,
        "parent_candidate_id": None,
        "predicted_outcome_signature": "positive matched delta",
        "proposal_intent": intent,
        "utility_features": {
            "frontier_potential": 0.8,
            "information_gain": 0.8,
            "useful_signal": 0.8,
        },
    }


class M6IFakeProviderV1:
    """Deterministic external-effect adapter; it performs no network calls."""

    model = "m6i-deterministic-fake-provider"
    max_total_tokens_per_call = 1000

    def __init__(self) -> None:
        self.broker: RealCanaryProposalBrokerV1 | None = None
        self.call_count = 0
        self.injected_failure_count = 0

    def _logical_owner_context(
        self, logical_call_id: str
    ) -> tuple[ArmCode, int]:
        if self.broker is None:
            raise RuntimeError("fake Provider is not bound to its Broker")
        record = next(
            (
                item
                for item in reversed(
                    self.broker._call_registry.audit_records
                )
                if item.get("consumer", {}).get("value")
                and str(item["consumer"]["value"]) in logical_call_id
            ),
            None,
        )
        if record is None:
            raise AssertionError(
                "fake Provider call lacks its logical owner context"
            )
        return (
            ArmCode(str(record["owner"]["arm"])),
            int(record["consumer"]["round_index"]),
        )

    def _raise_typed_process_failure(
        self, *, logical_call_id: str, prompt: str
    ) -> None:
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        receipt_payload = {
            "start_record_digest": sha256_digest(
                {"logical_call_id": logical_call_id, "start": True}
            ),
            "spawn_succeeded": True,
            "pid_or_private_process_ref": "M6I_FAKE_PROCESS",
            "exit_code_or_NONE": 17,
            "termination_signal_or_NONE": None,
            "timed_out": False,
            "monotonic_end_ns": 1,
            "latency_ms": 1,
            "stdout_artifact_ref": "m6i://fake-provider/stdout",
            "stdout_sha256": empty_sha256,
            "stdout_size_bytes": 0,
            "stdout_truncated": False,
            "stderr_artifact_ref": "m6i://fake-provider/stderr",
            "stderr_sha256": empty_sha256,
            "stderr_size_bytes": 0,
            "stderr_truncated": False,
            "provider_request_confirmation": "CONFIRMED_SENT",
            "returned_model_or_NONE": self.model,
        }
        receipt = BrokerProcessExitReceiptV2(
            **receipt_payload,
            receipt_digest=sha256_digest(receipt_payload),
        )
        outcome_payload = {
            "logical_call_id": logical_call_id,
            "proposal_generation_session_id": (
                "M6I_FAKE_PROVIDER_FAILURE_SESSION"
            ),
            "request_envelope_digest": sha256_digest(
                {"prompt": prompt}
            ),
            "receipt_digest": receipt.receipt_digest,
            "status": "PROCESS_FAILURE",
            "failure_class": "PROCESS_EXIT_NONZERO",
            "classifier_rule_id": "M6I_TYPED_PROVIDER_FAILURE_V1",
            "supporting_artifact_ref": receipt.stderr_artifact_ref,
            "redacted_excerpt": "deterministic injected process failure",
            "response_digest": None,
        }
        outcome = BrokerCallOutcomeV2(
            **outcome_payload,
            outcome_digest=sha256_digest(outcome_payload),
        )
        self.injected_failure_count += 1
        raise CanaryBrokerError(
            "deterministic typed Provider process failure",
            outcome=outcome,
            receipt=receipt,
            physical_call_count=1,
            wall_time_ms=receipt.latency_ms,
        )

    def call(
        self,
        *,
        logical_call_id: str,
        prompt: str,
        expected_proposal_count: int,
        **_kwargs: Any,
    ) -> CanaryBrokerCallV1:
        if self.broker is None:
            raise RuntimeError("fake Provider is not bound to its Broker")
        owner_arm, round_index = self._logical_owner_context(
            logical_call_id
        )
        role = logical_call_id.rsplit("-", 1)[-1]
        if (
            owner_arm is ArmCode.B
            and round_index == 7
            and role == "mechanism_composer"
        ):
            self._raise_typed_process_failure(
                logical_call_id=logical_call_id,
                prompt=prompt,
            )
        if "original-" in logical_call_id:
            proposals = [
                {
                    **_proposal(mechanism_id, "DISCOVERY"),
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
            scope = self.broker._campaign_call_scopes[logical_call_id]
            proposal = _proposal(
                scope[0],
                (
                    "FALSIFICATION"
                    if role == "falsification_designer"
                    else "DISCOVERY"
                ),
            )
            if role == "lineage_refiner":
                consumer_value = next(
                    (
                        str(record["consumer"]["value"])
                        for record in reversed(
                            self.broker._call_registry.audit_records
                        )
                        if "consumer" in record
                        and str(record["consumer"]["value"])
                        in logical_call_id
                    ),
                    None,
                )
                if consumer_value is None:
                    raise AssertionError(
                        "lineage fake call lacks its consumer ownership record"
                    )
                owner_arm = next(
                    ArmCode(str(record["owner"]["arm"]))
                    for record in reversed(
                        self.broker._call_registry.audit_records
                    )
                    if record.get("consumer", {}).get("value")
                    == consumer_value
                )
                parent = self.broker.lineage_indexes[
                    owner_arm
                ].latest_success()
                proposal["parent_candidate_id"] = (
                    parent.proposal_candidate_id
                    if parent is not None
                    else None
                )
            proposals = [proposal]
        if len(proposals) != expected_proposal_count:
            raise AssertionError("fake Provider returned the wrong proposal count")
        self.call_count += 1
        return CanaryBrokerCallV1(
            logical_call_id=logical_call_id,
            request_digest=sha256_digest({"prompt": prompt}),
            response_digest=sha256_digest(proposals),
            response={"proposals": proposals},
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=5,
            total_tokens=15,
            latency_ms=1,
            returned_model=self.model,
        )


class M6IExactSchedulerV1(ThreeArmPreCanaryOrchestratorV1):
    """Configuration-only adapter around the production scheduler."""

    def __init__(
        self,
        *args: Any,
        meta_runtime: MetaV19CampaignRuntimeV1,
        **kwargs: Any,
    ) -> None:
        self.meta_runtime = meta_runtime
        self.injected_result_faults: Counter[str] = Counter()
        super().__init__(*args, **kwargs)
        self.meta_runtime.bind_instances(dict(self.assignment.arm_to_instance))

    def _common_execution_protocol(self) -> Any:
        return campaign_development_protocol()

    def _build_helix_raw(
        self,
        *,
        selected: Any,
        opaque_instance_id: str,
        common_result: Any,
        observation_seed: str,
    ) -> Any:
        raw = super()._build_helix_raw(
            selected=selected,
            opaque_instance_id=opaque_instance_id,
            common_result=common_result,
            observation_seed=observation_seed,
        )
        round_row = self.store.get_round(str(common_result.round_id))
        fault_by_round = {
            8: "COMMON_EXECUTION_FAILURE",
            9: "RESOURCE_CEILING_REJECTION",
            10: "TRAINING_FAILURE",
            11: "QUARANTINE_OR_INCONCLUSIVE",
        }
        fault = (
            fault_by_round.get(int(round_row["round_index"]))
            if str(round_row["arm_code"]) == ArmCode.B.value
            else None
        )
        if fault is None:
            return raw
        self.injected_result_faults[fault] += 1
        faulted_raw = raw.to_dict()
        faulted_raw.update(
            {
                "raw_result_digest": sha256_digest(
                    {
                        "base_raw_result_digest": raw.raw_result_digest,
                        "fault": fault,
                        "round_id": str(common_result.round_id),
                    }
                ),
                "run_status": fault,
                "normalized_metrics": {},
            }
        )
        return type(raw)(**faulted_raw)


def _count(connection: Any, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def _preload_task_branch_fixtures(
    orchestrator: M6IExactSchedulerV1,
    *,
    schedule_seed: int,
) -> dict[str, tuple[ArmCode, str, str]]:
    program = program_from_proposal(
        {"mechanism_id": "LIGHTGCN_RESIDUAL"}
    )
    compiled = compile_campaign_program(program)
    protocol_digest = str(
        campaign_runtime_profile()["development_protocol_digest"]
    )
    cases = (
        (
            ArmCode.B,
            ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE,
            "2027",
            ResearchTaskStatusV1.COMPLETED.value,
        ),
        (
            ArmCode.B,
            ResearchTaskTypeV1.RUN_ABLATION,
            "2026",
            ResearchTaskStatusV1.COMPLETED.value,
        ),
        (
            ArmCode.B,
            ResearchTaskTypeV1.REPAIR_IMPLEMENTATION,
            "repair",
            ResearchTaskStatusV1.COMPLETED.value,
        ),
        (
            ArmCode.B,
            ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC,
            "protocol-branch",
            ResearchTaskStatusV1.COMPLETED.value,
        ),
        (
            ArmCode.C,
            ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE,
            "2027",
            ResearchTaskStatusV1.CANCELLED.value,
        ),
    )
    fixtures: dict[str, tuple[ArmCode, str, str]] = {}
    for arm, task_type, required, expected_status in cases:
        owner = orchestrator.assignment.mapping[arm]
        task_id = sha256_digest(
            {
                "arm": arm.value,
                "schedule_seed": schedule_seed,
                "task_type": task_type.value,
                "fixture": "M6I_EXACT_TASK_BRANCH",
            }
        )
        task = ResearchTaskV1(
            task_id=task_id,
            task_type=task_type,
            candidate_id=str(compiled.candidate_id),
            candidate_semantic_digest=str(
                compiled.mechanism_semantics_digest
            ),
            mechanism_program_digest=str(
                compiled.mechanism_program_digest
            ),
            parent_candidate_id=None,
            comparator_identity="LIGHTGCN",
            protocol_digest=protocol_digest,
            required_seed_or_control=required,
            task_status=ResearchTaskStatusV1.PENDING,
            created_round=1,
            utility_priority=1.0,
            missing_seed_count=(
                1
                if task_type
                is ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE
                else 0
            ),
            mechanism_program=program,
            owner_arm_instance_id=owner,
        )
        orchestrator.research_task_queues[arm].enqueue(task)
        label = f"{arm.value}:{task_type.value}"
        fixtures[label] = (arm, task_id, expected_status)
    return fixtures


def run_schedule(
    schedule_seed: int, *, rounds_per_arm: int
) -> dict[str, Any]:
    rng = random.Random(schedule_seed)
    experiment_id = f"M6I-EXACT-SCHEDULER-{schedule_seed:03d}"
    meta_runtime = MetaV19CampaignRuntimeV1(
        checkpoint_path=CHECKPOINT,
        experiment_id=experiment_id,
        search_seed=SEARCH_SEED,
        scheduled_rounds=rounds_per_arm,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    upstream = M6IFakeProviderV1()
    broker = RealCanaryProposalBrokerV1.create_v13(
        upstream=upstream,
        template_path=TEMPLATES,
        repository_root=ROOT,
        search_seed=SEARCH_SEED,
        call_prefix=f"m6i-exact-{schedule_seed:03d}",
        phase_name="M6I exact scheduler stress",
        adaptive_memory=True,
        campaign_meta_runtime=meta_runtime,
    )
    upstream.broker = broker
    order_counts: Counter[str] = Counter()
    runtime_failure: dict[str, Any] | None = None
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"recclaw-m6i-exact-{schedule_seed:03d}-",
        dir=_temporary_root_parent(),
    ) as root:
        with M6IExactSchedulerV1(
            Path(root) / "runtime",
            broker=broker,
            resource_ceilings=canary_budget(),
            meta_runtime=meta_runtime,
            assignment_nonce=f"M6I-EXACT-SCHEDULER-{schedule_seed:03d}",
        ) as orchestrator:
            task_fixtures = _preload_task_branch_fixtures(
                orchestrator,
                schedule_seed=schedule_seed,
            )
            fault_path_checkpoints: list[dict[str, Any]] = []
            for round_index in range(1, rounds_per_arm + 1):
                order = (
                    ALL_ORDERS[
                        (schedule_seed + round_index - 1) % len(ALL_ORDERS)
                    ]
                    if round_index <= len(ALL_ORDERS)
                    else rng.choice(ALL_ORDERS)
                )
                order_counts["".join(arm.value for arm in order)] += 1
                try:
                    orchestrator.run_fake_triplet(
                        search_seed=SEARCH_SEED,
                        round_index=round_index,
                        drafts=(),
                        execution_order=order,
                    )
                    if 7 <= round_index <= 11:
                        controller = broker.research_controllers[ArmCode.B]
                        memory_head = controller.memory_writer.head
                        search_feedback = canonical_value(
                            broker._search_feedback.get(ArmCode.B, {})
                        )
                        common_slot = (
                            search_feedback.get(
                                "prompt_feedback_projection", {}
                            ).get("common_search_utility_slot")
                        )
                        fault_path_checkpoints.append(
                            {
                                "round_index": round_index,
                                "search_memory_head_round": (
                                    memory_head.round_index
                                    if memory_head is not None
                                    else None
                                ),
                                "belief_digests": (
                                    [
                                        sha256_digest(item.to_dict())
                                        for item in memory_head.beliefs
                                    ]
                                    if memory_head is not None
                                    else []
                                ),
                                "feedback_outcome_class": (
                                    common_slot.get("common_outcome_class")
                                    if isinstance(common_slot, dict)
                                    else None
                                ),
                            }
                        )
                except Exception as error:  # qualification evidence boundary
                    runtime_failure = {
                        "exception_class": type(error).__name__,
                        "message": str(error),
                        "order": "".join(arm.value for arm in order),
                        "round_index": round_index,
                        "traceback": traceback.format_exc(),
                    }
                    break
            store = orchestrator.store
            counts = {
                "rounds": _count(store._connection, "SELECT COUNT(*) FROM rounds"),
                "terminal_rounds": _count(
                    store._connection,
                    "SELECT COUNT(*) FROM rounds "
                    "WHERE status IN ('CLOSED','ABORTED')",
                ),
                "open_rounds": _count(
                    store._connection,
                    "SELECT COUNT(*) FROM rounds WHERE status = 'OPEN'",
                ),
                "execution_claims": _count(
                    store._connection,
                    "SELECT COUNT(*) FROM execution_claims",
                ),
                "unfinished_execution_claims": _count(
                    store._connection,
                    "SELECT COUNT(*) FROM execution_claims "
                    "WHERE claim_state != 'FINISHED'",
                ),
                "duplicate_execution_claims": _count(
                    store._connection,
                    "SELECT COUNT(*) FROM ("
                    "SELECT round_id FROM execution_claims "
                    "GROUP BY round_id HAVING COUNT(*) > 1"
                    ")",
                ),
                "duplicate_feedback": _count(
                    store._connection,
                    "SELECT COUNT(*) FROM ("
                    "SELECT round_id FROM round_events "
                    "WHERE event_type = 'ROUND_FEEDBACK' "
                    "GROUP BY round_id HAVING COUNT(*) > 1"
                    ")",
                ),
                "closed_triplet_barriers": _count(
                    store._connection,
                    "SELECT COUNT(*) FROM triplet_barrier "
                    "WHERE closed_bitmap = 7 AND next_index_authorized = 1",
                ),
            }
            state_audit = orchestrator.integrated_state.audit_projection()
            sharing_audit = broker.call_sharing_audit()
            meta_audit = meta_runtime.audit_projection()
            integrity = store.integrity_report()
            task_fixture_statuses = {
                label: {
                    "actual": next(
                        task.task_status.value
                        for task in orchestrator.research_task_queues[
                            arm
                        ].tasks
                        if task.task_id == task_id
                    ),
                    "expected": expected,
                }
                for label, (
                    arm,
                    task_id,
                    expected,
                ) in task_fixtures.items()
            }
            result_fault_counts = dict(
                orchestrator.injected_result_faults
            )
            meta_fault_records = {
                int(record["round_index"]): {
                    key: record.get(key)
                    for key in (
                        "frontier_value",
                        "observation_digest",
                        "observation_path",
                        "proposal_source",
                        "run_status",
                    )
                }
                for record in meta_audit["observations"]
                if record["arm"] == ArmCode.B.value
                and 7 <= int(record["round_index"]) <= 11
            }
    expected_rounds = rounds_per_arm * 3
    violations = []
    if runtime_failure is not None:
        violations.append(
            f"SCHEDULER_EXCEPTION:{runtime_failure['exception_class']}"
        )
    if counts["terminal_rounds"] != expected_rounds:
        violations.append("TERMINAL_ROUND_COUNT")
    if counts["open_rounds"] != 0:
        violations.append("OPEN_ROUNDS")
    if counts["unfinished_execution_claims"] != 0:
        violations.append("UNFINISHED_EXECUTION_CLAIMS")
    if counts["duplicate_execution_claims"] != 0:
        violations.append("DUPLICATE_EXECUTION_CLAIMS")
    if counts["duplicate_feedback"] != 0:
        violations.append("DUPLICATE_FEEDBACK")
    if counts["closed_triplet_barriers"] != rounds_per_arm:
        violations.append("TRIPLET_BARRIER")
    if len(meta_audit["observations"]) != rounds_per_arm * 2:
        violations.append("META_BOUNDARY_COUNT")
    if sharing_audit["cross_arm_physical_identities"] != 0:
        violations.append("CROSS_ARM_PHYSICAL_IDENTITY")
    if state_audit["cross_arm_reads"] != 0:
        violations.append("CROSS_ARM_STATE_ACCESS")
    if integrity["integrity_check"] != "ok" or integrity["foreign_key_violations"]:
        violations.append("STATE_STORE_INTEGRITY")
    if upstream.injected_failure_count != 1:
        violations.append("TYPED_PROVIDER_FAILURE_NOT_EXERCISED_ONCE")
    if any(
        item["actual"] != item["expected"]
        for item in task_fixture_statuses.values()
    ):
        violations.append("TASK_BRANCH_TERMINAL_STATUS")
    expected_result_faults = {
        name
        for fault_round, name in {
            8: "COMMON_EXECUTION_FAILURE",
            9: "RESOURCE_CEILING_REJECTION",
            10: "TRAINING_FAILURE",
            11: "QUARANTINE_OR_INCONCLUSIVE",
        }.items()
        if fault_round <= rounds_per_arm
    }
    if (
        set(result_fault_counts) != expected_result_faults
        or any(
            result_fault_counts[name] != 1
            for name in expected_result_faults
        )
    ):
        violations.append("RESULT_FAULT_BRANCH_COUNTS")
    checkpoints_by_round = {
        int(item["round_index"]): item
        for item in fault_path_checkpoints
    }
    if rounds_per_arm >= 7:
        provider_failure_meta = meta_fault_records.get(7)
        if (
            provider_failure_meta is None
            or provider_failure_meta["observation_path"] != "NO_OBSERVATION"
            or provider_failure_meta["proposal_source"]
            != "NO_PROPOSAL_TERMINAL"
            or provider_failure_meta["observation_digest"] is not None
            or float(provider_failure_meta["frontier_value"]) != 0.0
        ):
            violations.append("PROVIDER_FAILURE_META_NO_OBSERVATION")
    baseline_beliefs = (
        checkpoints_by_round.get(7, {}).get("belief_digests")
        if rounds_per_arm >= 7
        else None
    )
    for fault_round, fault_name in {
        8: "COMMON_EXECUTION_FAILURE",
        9: "RESOURCE_CEILING_REJECTION",
        10: "TRAINING_FAILURE",
        11: "QUARANTINE_OR_INCONCLUSIVE",
    }.items():
        if fault_round > rounds_per_arm:
            continue
        meta_record = meta_fault_records.get(fault_round)
        checkpoint = checkpoints_by_round.get(fault_round)
        if (
            meta_record is None
            or meta_record["observation_path"]
            != "DIAGNOSTIC_OR_ENGINEERING_ONLY"
            or meta_record["proposal_source"] != "NORMAL_ROUTED_PROPOSAL"
            or meta_record["observation_digest"] is not None
            or float(meta_record["frontier_value"]) != 0.0
            or meta_record["run_status"] != fault_name
        ):
            violations.append(
                f"RESULT_FAULT_META_DIAGNOSTIC:{fault_name}"
            )
        if (
            checkpoint is None
            or checkpoint["search_memory_head_round"] != fault_round
            or checkpoint["feedback_outcome_class"] != fault_name
            or checkpoint["belief_digests"] != baseline_beliefs
        ):
            violations.append(
                f"RESULT_FAULT_MEMORY_SEMANTICS:{fault_name}"
            )
    return canonical_value(
        {
            "schedule_seed": schedule_seed,
            "rounds_per_arm": rounds_per_arm,
            "forced_six_order_prefix": (
                rounds_per_arm >= len(ALL_ORDERS)
            ),
            "status": "PASS" if not violations else "FAIL",
            "violations": violations,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "order_counts": dict(order_counts),
            "provider_calls": upstream.call_count,
            "typed_provider_failures": upstream.injected_failure_count,
            "task_fixture_statuses": task_fixture_statuses,
            "result_fault_counts": result_fault_counts,
            "fault_path_semantics": {
                "search_memory_checkpoints": fault_path_checkpoints,
                "meta_records": meta_fault_records,
            },
            "runtime_failure": runtime_failure,
            "counts": counts,
            "sharing": {
                key: sharing_audit[key]
                for key in (
                    "candidate_instance_identities",
                    "consumer_logical_identities",
                    "cross_arm_physical_identities",
                    "physical_call_identities",
                    "policy",
                )
            },
            "meta_observation_count": len(meta_audit["observations"]),
            "state_audit_digest": sha256_digest(state_audit),
            "meta_audit_digest": sha256_digest(meta_audit),
            "integrity": integrity,
        }
    )


def run_stress(
    *,
    randomized_seeds: int,
    rounds_per_arm: int,
    workers: int = 1,
    seed_offset: int = 0,
) -> dict[str, Any]:
    if randomized_seeds < 1 or workers < 1 or seed_offset < 0:
        raise ValueError("stress seed/worker arguments must be positive")
    source_projection_before = _source_projection()
    schedule_seeds = tuple(
        range(seed_offset, seed_offset + randomized_seeds)
    )
    if workers == 1:
        results = [
            run_schedule(
                schedule_seed=seed,
                rounds_per_arm=rounds_per_arm,
            )
            for seed in schedule_seeds
        ]
    else:
        worker = partial(
            run_schedule,
            rounds_per_arm=rounds_per_arm,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(worker, schedule_seeds))
    failures = [
        {
            "schedule_seed": result["schedule_seed"],
            "violations": result["violations"],
        }
        for result in results
        if result["status"] != "PASS"
    ]
    order_counts: Counter[str] = Counter()
    for result in results:
        order_counts.update(result["order_counts"])
    if set(order_counts) != {
        "".join(arm.value for arm in order) for order in ALL_ORDERS
    }:
        failures.append(
            {
                "schedule_seed": "AGGREGATE",
                "violations": ["ARM_ORDER_COVERAGE"],
            }
        )
    source_projection_after = _source_projection()
    if source_projection_after != source_projection_before:
        failures.append(
            {
                "schedule_seed": "SOURCE_PROJECTION",
                "violations": ["SOURCE_CHANGED_DURING_QUALIFICATION"],
            }
        )
    return canonical_value(
        {
            "schema": "recclaw.m6i.exact-scheduler-stress.v1",
            "milestone": (
                "M6I_INTEGRATED_STATE_SPACE_AND_CROSS_ARM_ISOLATION_CLOSURE"
            ),
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "real_provider_calls": 0,
            "real_training_executions": 0,
            "source_projection": source_projection_before,
            "source_projection_digest": sha256_digest(
                source_projection_before
            ),
            "source_projection_unchanged": (
                source_projection_after == source_projection_before
            ),
            "scheduler_class": (
                "ThreeArmPreCanaryOrchestratorV1"
                "+M6IExactSchedulerV1_CONFIGURATION_ONLY"
            ),
            "canonical_core_class": "IntegratedCampaignStateCoreV1",
            "meta_runtime_class": "MetaV19CampaignRuntimeV1",
            "fake_runner_abi": "recclaw.fake-non-training-runner.v1",
            "fault_injection": {
                "typed_provider_process_failure": True,
                "active_task_pre_blocked": True,
                "validation_task": True,
                "ablation_task": True,
                "repair_task": True,
                "diagnostic_protocol_branch_task": True,
                "common_execution_failure_result": True,
                "resource_ceiling_rejection_result": True,
                "training_failure_result": True,
                "quarantine_or_inconclusive_result": True,
            },
            "randomized_schedule_seeds": randomized_seeds,
            "schedule_seed_first": schedule_seeds[0],
            "schedule_seed_last": schedule_seeds[-1],
            "qualification_workers": workers,
            "rounds_per_arm": rounds_per_arm,
            "status": "PASS" if not failures else "FAIL",
            "p0": 0 if not failures else 1,
            "p1": 0 if not failures else 1,
            "failures": failures,
            "totals": {
                "scheduled_arm_rounds": randomized_seeds * rounds_per_arm * 3,
                "terminal_arm_rounds": sum(
                    result["counts"]["terminal_rounds"] for result in results
                ),
                "open_arm_rounds": sum(
                    result["counts"]["open_rounds"] for result in results
                ),
                "execution_claims": sum(
                    result["counts"]["execution_claims"] for result in results
                ),
                "unfinished_execution_claims": sum(
                    result["counts"]["unfinished_execution_claims"]
                    for result in results
                ),
                "duplicate_claims": sum(
                    result["counts"]["duplicate_execution_claims"]
                    for result in results
                ),
                "duplicate_feedback": sum(
                    result["counts"]["duplicate_feedback"] for result in results
                ),
                "closed_triplet_barriers": sum(
                    result["counts"]["closed_triplet_barriers"]
                    for result in results
                ),
                "meta_boundary_events": sum(
                    result["meta_observation_count"] for result in results
                ),
                "fake_provider_calls": sum(
                    result["provider_calls"] for result in results
                ),
                "typed_provider_failures": sum(
                    result["typed_provider_failures"]
                    for result in results
                ),
                "injected_result_faults": dict(
                    sum(
                        (
                            Counter(result["result_fault_counts"])
                            for result in results
                        ),
                        Counter(),
                    )
                ),
                "cross_arm_physical_identities": sum(
                    result["sharing"]["cross_arm_physical_identities"]
                    for result in results
                ),
            },
            "arm_order_counts": dict(order_counts),
            "schedule_summaries": results,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomized-seeds", type=int, default=100)
    parser.add_argument("--rounds-per-arm", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_stress(
        randomized_seeds=args.randomized_seeds,
        rounds_per_arm=args.rounds_per_arm,
        workers=args.workers,
        seed_offset=args.seed_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "p0": report["p0"],
                "p1": report["p1"],
                "totals": report["totals"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
