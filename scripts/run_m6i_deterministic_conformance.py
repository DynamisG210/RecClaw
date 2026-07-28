#!/usr/bin/env python3
"""Generate deterministic M6I state, identity, and Arm-order evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import tempfile
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.integrated_state_core import (  # noqa: E402
    CallSharingPolicyV1,
    CallSharingRegistryV1,
    CallSharingViolation,
    IntegratedCampaignStateCoreV1,
    ObservationPathV1,
    ProposalSourceV1,
    ProviderRequestContextV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    ThreeArmPreCanaryOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (  # noqa: E402
    ProposalIntentV1,
)


MATRIX_PATH = (
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "m6i_transition_matrix_v1.json"
)
ANCHORS_PATH = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
FAILURE_ROOT = ROOT / "tests" / "fixtures" / "m6i_failures"
REPORT_ROOT = ROOT / "docs" / "research_line" / "continuous_program"
ALL_ORDERS = tuple(permutations(tuple(ArmCode)))


def digest(label: Any) -> str:
    return sha256_digest({"m6i": label})


def new_core(experiment_id: str) -> IntegratedCampaignStateCoreV1:
    core = IntegratedCampaignStateCoreV1(experiment_id=experiment_id)
    core.bind_arms(
        {
            arm: f"{experiment_id}-opaque-{arm.value.lower()}"
            for arm in ArmCode
        }
    )
    return core


def close_branch(
    core: IntegratedCampaignStateCoreV1,
    *,
    arm: ArmCode,
    round_index: int,
    source: ProposalSourceV1,
    observation: ObservationPathV1,
) -> dict[str, Any]:
    core.open_round(arm=arm, search_seed=42, round_index=round_index)
    kwargs: dict[str, str] = {}
    if source is ProposalSourceV1.NORMAL_ROUTED_PROPOSAL:
        kwargs["route_digest"] = digest(
            {"arm": arm.value, "round": round_index, "route": True}
        )
    if source is ProposalSourceV1.ACTIVE_BOUND_TASK:
        kwargs["active_task_digest"] = digest(
            {"arm": arm.value, "round": round_index, "task": True}
        )
    core.bind_proposal_source(
        arm=arm,
        search_seed=42,
        round_index=round_index,
        source=source,
        **kwargs,
    )
    if observation is ObservationPathV1.NO_OBSERVATION:
        core.close_no_execution(
            arm=arm, search_seed=42, round_index=round_index
        )
        terminal = "NO_EXECUTION"
    else:
        core.select_candidate(
            arm=arm,
            search_seed=42,
            round_index=round_index,
            candidate_instance_id=f"cand-{digest([arm.value, round_index])}",
        )
        core.start_execution(
            arm=arm, search_seed=42, round_index=round_index
        )
        core.close_result(
            arm=arm,
            search_seed=42,
            round_index=round_index,
            observation_path=observation,
        )
        terminal = "COMPLETED"
    core.terminalize(
        arm=arm,
        search_seed=42,
        round_index=round_index,
        terminal_class=terminal,
    )
    return core.round_projection(
        arm=arm, search_seed=42, round_index=round_index
    )


def state_machine_report() -> dict[str, Any]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    proposal_results = []
    for index, branch in enumerate(matrix["proposal_paths"], 1):
        source = ProposalSourceV1(branch["canonical_source"])
        no_execution = branch["name"] in {
            "ALL_PRE_CANDIDATES_BLOCKED",
            "EMPTY_OR_INVALID_SLATE",
            "PROVIDER_FAILURE_BEFORE_PROPOSAL",
        }
        projection = close_branch(
            new_core(f"M6I-PROPOSAL-{index}"),
            arm=(
                ArmCode.A
                if source is ProposalSourceV1.ORIGINAL_CONTROLLER_PATH
                else ArmCode.B
            ),
            round_index=index,
            source=source,
            observation=(
                ObservationPathV1.NO_OBSERVATION
                if no_execution
                else ObservationPathV1.ADMITTED_OBSERVATION
            ),
        )
        proposal_results.append(
            {
                "name": branch["name"],
                "terminal_state": projection["state"],
                "meta_boundary_count": projection["meta_boundary_count"],
            }
        )
    result_results = []
    for index, branch in enumerate(matrix["result_paths"], 1):
        projection = close_branch(
            new_core(f"M6I-RESULT-{index}"),
            arm=ArmCode.C,
            round_index=index,
            source=ProposalSourceV1.NORMAL_ROUTED_PROPOSAL,
            observation=ObservationPathV1(branch["observation_path"]),
        )
        result_results.append(
            {
                "name": branch["name"],
                "observation_path": projection["observation_path"],
                "terminal_state": projection["state"],
                "meta_boundary_count": projection["meta_boundary_count"],
            }
        )
    meta_results = []
    for index, branch in enumerate(matrix["meta_paths"], 1):
        projection = close_branch(
            new_core(f"M6I-META-{index}"),
            arm=ArmCode.B,
            round_index=index,
            source=ProposalSourceV1(branch["proposal_source"]),
            observation=ObservationPathV1(branch["observation_path"]),
        )
        meta_results.append(
            {
                "name": branch["name"],
                "terminal_state": projection["state"],
                "meta_boundary_count": projection["meta_boundary_count"],
            }
        )
    fixture_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(FAILURE_ROOT.glob("*.json"))
    }
    fixtures = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(FAILURE_ROOT.glob("*.json"))
    }
    fixture_labels = {
        "V19": fixtures["V19_FAILURE_RECORD.json"]["failure_class"],
        "V20": fixtures["V20_FAILURE_RECORD.json"]["failure_class"],
        "V21": fixtures["V21_FAILURE_RECORD.json"]["failure_class"],
        "V21_hard_stop": fixtures[
            "V21_CROSS_ARM_CONTAMINATION_HARD_STOP.json"
        ]["hard_stop_class"],
    }
    failures = []
    if any(item["terminal_state"] != "ROUND_TERMINAL" for item in proposal_results):
        failures.append("PROPOSAL_NON_TERMINAL")
    if any(item["terminal_state"] != "ROUND_TERMINAL" for item in result_results):
        failures.append("RESULT_NON_TERMINAL")
    if any(item["meta_boundary_count"] != 1 for item in meta_results):
        failures.append("META_BOUNDARY_NOT_EXACTLY_ONCE")
    return canonical_value(
        {
            "schema": "recclaw.m6i.state-machine-coverage.v1",
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "status": "PASS" if not failures else "FAIL",
            "p0": 0 if not failures else 1,
            "p1": 0 if not failures else 1,
            "failures": failures,
            "asserted_state_dimensions": matrix["asserted_state_dimensions"],
            "proposal_paths": proposal_results,
            "result_paths": result_results,
            "meta_paths": meta_results,
            "permanent_regression_fixtures": {
                "sha256": fixture_sha256,
                "actual_failure_labels": fixture_labels,
                "command_summary_label_discrepancy": {
                    "V19_command": "META_NO_OBSERVATION_BOUNDARY_NOT_ADVANCED",
                    "V19_fixture": fixture_labels["V19"],
                    "V20_command": (
                        "ACTIVE_TASK_WITHOUT_ROUTE_BOUNDARY_MISCLASSIFIED"
                    ),
                    "V20_fixture": fixture_labels["V20"],
                    "V21_command": "CONFIRMED_CROSS_ARM_RESEARCH_CALL_REUSE",
                    "V21_failure_fixture": fixture_labels["V21"],
                    "V21_hard_stop_fixture": fixture_labels["V21_hard_stop"],
                    "interpretation": (
                        "The command uses causal summaries; the sealed fixture "
                        "labels remain authoritative."
                    ),
                },
            },
        }
    )


def provider_context() -> ProviderRequestContextV1:
    return ProviderRequestContextV1(
        model_release_digest=digest("model"),
        response_schema_digest=digest("schema"),
        temperature=0.0,
        timeout_policy_digest=digest("timeout"),
        producer_role="mechanism_composer",
        prompt_bytes_digest=digest("prompt"),
        complete_context_digest=digest("complete-context"),
        memory_view_digest=digest("memory"),
        meta_fast_state_digest=digest("meta"),
        lineage_view_digest=digest("lineage"),
        active_task_digest="ABSENT",
        research_task_queue_digest=digest("queue"),
        round_index=1,
        search_seed=42,
        response_arm_neutral=True,
    )


def call_sharing_report() -> dict[str, Any]:
    core = new_core("M6I-CALL-SHARING")
    context = provider_context()
    private = CallSharingRegistryV1(CallSharingPolicyV1.ARM_PRIVATE)
    b_private = private.register_request(
        owner=core.owner(ArmCode.B), context=context
    )
    c_private = private.register_request(
        owner=core.owner(ArmCode.C), context=context
    )
    paired = CallSharingRegistryV1(
        CallSharingPolicyV1.PAIRED_BC_EXACT_CONTEXT
    )
    b_pair = paired.register_request(
        owner=core.owner(ArmCode.B), context=context
    )
    c_pair = paired.register_request(
        owner=core.owner(ArmCode.C), context=context
    )
    b_candidate = paired.register_candidate(
        owner=core.owner(ArmCode.B),
        round_index=1,
        producer_role=context.producer_role,
        semantic_program_digest=digest("program"),
        local_parent_or_task_identity=None,
    )
    c_candidate = paired.register_candidate(
        owner=core.owner(ArmCode.C),
        round_index=1,
        producer_role=context.producer_role,
        semantic_program_digest=digest("program"),
        local_parent_or_task_identity=None,
    )
    mutations = {
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
        "search_seed": 43,
    }
    mutation_results = {}
    for field_name, value in mutations.items():
        registry = CallSharingRegistryV1(
            CallSharingPolicyV1.PAIRED_BC_EXACT_CONTEXT
        )
        first = registry.register_request(
            owner=core.owner(ArmCode.B), context=context
        )[0]
        second = registry.register_request(
            owner=core.owner(ArmCode.C),
            context=replace(context, **{field_name: value}),
        )[0]
        mutation_results[field_name] = first.value != second.value
    rejected = {}
    try:
        paired.register_request(owner=core.owner(ArmCode.A), context=context)
    except CallSharingViolation:
        rejected["ARM_A_PAIRED_CALL"] = True
    else:
        rejected["ARM_A_PAIRED_CALL"] = False
    foreign_parent = paired.register_candidate(
        owner=core.owner(ArmCode.C),
        round_index=2,
        producer_role="lineage_refiner",
        semantic_program_digest=digest("foreign-parent"),
        local_parent_or_task_identity=None,
    )
    try:
        paired.register_candidate(
            owner=core.owner(ArmCode.B),
            round_index=2,
            producer_role="lineage_refiner",
            semantic_program_digest=digest("foreign-child"),
            local_parent_or_task_identity=foreign_parent.value,
        )
    except CallSharingViolation:
        rejected["FOREIGN_PARENT"] = True
    else:
        rejected["FOREIGN_PARENT"] = False
    checks = {
        "arm_private_identical_context_has_distinct_physical_ids": (
            b_private[0].value != c_private[0].value
        ),
        "arm_private_identical_context_has_distinct_consumer_ids": (
            b_private[1].value != c_private[1].value
        ),
        "paired_exact_context_shares_only_physical_id": (
            b_pair[0].value == c_pair[0].value
            and b_pair[1].value != c_pair[1].value
        ),
        "paired_candidate_instances_remain_distinct": (
            b_candidate.value != c_candidate.value
        ),
        "every_context_mutation_separates_physical_id": all(
            mutation_results.values()
        ),
        "arm_a_pairing_rejected": rejected["ARM_A_PAIRED_CALL"],
        "foreign_parent_rejected": rejected["FOREIGN_PARENT"],
    }
    return canonical_value(
        {
            "schema": "recclaw.m6i.call-sharing-conformance.v1",
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "active_runtime_policy": CallSharingPolicyV1.ARM_PRIVATE.value,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "p0": 0 if all(checks.values()) else 1,
            "p1": 0 if all(checks.values()) else 1,
            "checks": checks,
            "context_field_mutations": mutation_results,
            "private_audit": private.audit_projection(),
            "paired_audit": paired.audit_projection(),
        }
    )


def _drafts() -> list[dict[str, Any]]:
    anchors = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    programs = {
        item["anchor_name"]: item["program"] for item in anchors["fixtures"]
    }
    specs = (
        ("DIRECTAU", "geometry", ProposalIntentV1.DISCOVERY.value, 0.88),
        ("LIGHTGCN", "propagation", ProposalIntentV1.DISCOVERY.value, 0.84),
        ("SGL", "self_supervision", ProposalIntentV1.FALSIFICATION.value, 0.82),
        ("ULTRAGCN", "architecture", ProposalIntentV1.DISCOVERY.value, 0.86),
    )
    return [
        {
            "mechanism_program": copy.deepcopy(programs[name]),
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


def _stable_result_projection(
    orchestrator: ThreeArmPreCanaryOrchestratorV1,
    results: Any,
) -> dict[str, Any]:
    by_instance = {
        instance_id: arm.value
        for arm, instance_id in orchestrator.assignment.mapping.items()
    }
    unstable = {
        "broker_call_latencies_ms",
        "proposal_session_wall_time_ms",
        "training_wall_time_ms",
        "round_total_wall_time_ms",
    }
    return {
        by_instance[item.opaque_instance_id]: {
            key: value
            for key, value in item.to_dict().items()
            if key not in unstable
        }
        for item in results
    }


def _lightweight_schedule_projection(
    orders: list[tuple[ArmCode, ArmCode, ArmCode]],
    *,
    experiment_id: str,
) -> dict[str, Any]:
    core = new_core(experiment_id)
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
                core,
                arm=arm,
                round_index=round_index,
                source=source,
                observation=observation,
            )
        core.close_triplet(search_seed=42, round_index=round_index)
    return {
        arm.value: [
            {
                key: projection[key]
                for key in (
                    "proposal_source",
                    "observation_path",
                    "meta_boundary_count",
                    "state",
                )
            }
            for round_index in range(1, len(orders) + 1)
            for projection in (
                core.round_projection(
                    arm=arm,
                    search_seed=42,
                    round_index=round_index,
                ),
            )
        ]
        for arm in ArmCode
    }


def arm_order_report() -> dict[str, Any]:
    full_scheduler_results = {}
    reference: dict[str, Any] | None = None
    full_scheduler_invariant = True
    for order in ALL_ORDERS:
        label = "".join(arm.value for arm in order)
        with tempfile.TemporaryDirectory(prefix=f"m6i-order-{label}-") as root:
            with ThreeArmPreCanaryOrchestratorV1(
                Path(root) / "runtime",
                assignment_nonce="M6I-SIX-ORDER-CONFORMANCE",
            ) as orchestrator:
                results = orchestrator.run_fake_triplet(
                    search_seed=42,
                    round_index=1,
                    drafts=_drafts(),
                    execution_order=order,
                )
                projection = _stable_result_projection(orchestrator, results)
                full_scheduler_results[label] = {
                    "projection_digest": sha256_digest(projection),
                    "terminal_classes": {
                        arm: item["terminal_class"]
                        for arm, item in projection.items()
                    },
                    "integrated_state_digest": sha256_digest(
                        orchestrator.integrated_state.audit_projection()
                    ),
                    "open_rounds": int(
                        orchestrator.store._connection.execute(
                            "SELECT COUNT(*) FROM rounds WHERE status = 'OPEN'"
                        ).fetchone()[0]
                    ),
                }
                if reference is None:
                    reference = projection
                elif projection != reference:
                    full_scheduler_invariant = False
    lightweight_reference = _lightweight_schedule_projection(
        [ALL_ORDERS[0]] * 8,
        experiment_id="M6I-RANDOM-REFERENCE",
    )
    randomized_failures = []
    order_counts: dict[str, int] = {
        "".join(arm.value for arm in order): 0 for order in ALL_ORDERS
    }
    for seed in range(500):
        rng = random.Random(seed)
        orders = [rng.choice(ALL_ORDERS) for _ in range(8)]
        for order in orders:
            order_counts["".join(arm.value for arm in order)] += 1
        actual = _lightweight_schedule_projection(
            orders,
            experiment_id=f"M6I-RANDOM-{seed}",
        )
        if actual != lightweight_reference:
            randomized_failures.append(seed)
    failures = []
    if not full_scheduler_invariant:
        failures.append("FULL_SCHEDULER_ORDER_DEPENDENCE")
    if any(item["open_rounds"] for item in full_scheduler_results.values()):
        failures.append("FULL_SCHEDULER_OPEN_ROUND")
    if randomized_failures:
        failures.append("RANDOMIZED_ORDER_DEPENDENCE")
    return canonical_value(
        {
            "schema": "recclaw.m6i.arm-order-invariance.v1",
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "status": "PASS" if not failures else "FAIL",
            "p0": 0 if not failures else 1,
            "p1": 0 if not failures else 1,
            "failures": failures,
            "full_scheduler": {
                "orders_executed": len(ALL_ORDERS),
                "fake_broker": True,
                "fake_training": True,
                "invariant": full_scheduler_invariant,
                "results": full_scheduler_results,
            },
            "randomized_canonical_core": {
                "schedule_seeds": 500,
                "rounds_per_schedule": 8,
                "failures": randomized_failures,
                "order_counts": order_counts,
            },
        }
    )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    reports = {
        "M6I_STATE_MACHINE_COVERAGE.json": state_machine_report(),
        "M6I_CALL_SHARING_CONFORMANCE.json": call_sharing_report(),
        "M6I_ARM_ORDER_INVARIANCE_REPORT.json": arm_order_report(),
    }
    for filename, payload in reports.items():
        write_report(args.output_root / filename, payload)
    summary = {
        filename: {
            "status": payload["status"],
            "p0": payload["p0"],
            "p1": payload["p1"],
        }
        for filename, payload in reports.items()
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if all(item["status"] == "PASS" for item in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
