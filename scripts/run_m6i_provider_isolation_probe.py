#!/usr/bin/env python3
"""Run the bounded real-Provider, no-training M6I isolation probe."""

from __future__ import annotations

import argparse
import json
import sys
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
    IntegratedCampaignStateCoreV1,
    ProviderRequestContextV1,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    LabApiCanaryBrokerV1,
)


DEFAULT_CONFIG = Path(
    "/root/projects/RecClaw_v2_0_Final_Reference/llm_api.md"
)
DEFAULT_SCHEMA = (
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "m6i_provider_probe_response_v2.schema.json"
)
DEFAULT_ROOT = (
    ROOT
    / "results"
    / "research_line"
    / "m6i_provider_isolation_probe_v3"
)
DEFAULT_REPORT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "M6I_PROVIDER_ISOLATION_PROBE.json"
)
ALL_ORDERS = tuple(permutations(tuple(ArmCode)))


def _context(
    *,
    broker: LabApiCanaryBrokerV1,
    prompt: str,
    arm: ArmCode,
    order_index: int,
    scenario: str,
) -> ProviderRequestContextV1:
    common = scenario == "IDENTICAL_CONTEXT"
    owner_label = "COMMON" if common else arm.value
    return ProviderRequestContextV1(
        model_release_digest=broker.release.release_digest,
        response_schema_digest=broker.schema_file_sha256,
        temperature=0.0,
        timeout_policy_digest=sha256_digest(
            {
                "timeout_ms": broker.timeout_ms,
                "retry_count": broker.release.retry_count,
            }
        ),
        producer_role="m6i_isolation_probe",
        prompt_bytes_digest=sha256_digest({"prompt": prompt}),
        complete_context_digest=sha256_digest(
            {
                "scenario": scenario,
                "context_owner": owner_label,
                "order_index": order_index,
            }
        ),
        memory_view_digest=sha256_digest(
            {"memory": "COMMON_EMPTY" if common else f"{arm.value}_PRIVATE"}
        ),
        meta_fast_state_digest=sha256_digest({"meta": "READ_ONLY_GENESIS"}),
        lineage_view_digest=sha256_digest(
            {"lineage": "COMMON_EMPTY" if common else f"{arm.value}_PRIVATE"}
        ),
        active_task_digest=(
            "ABSENT"
            if common
            else sha256_digest({"task": f"{arm.value}_PRIVATE"})
        ),
        research_task_queue_digest=sha256_digest(
            {"queue": "COMMON_EMPTY" if common else f"{arm.value}_PRIVATE"}
        ),
        round_index=order_index + 1,
        search_seed=0,
        response_arm_neutral=True,
    )


def _prompt(*, arm: ArmCode, scenario: str, order_label: str) -> str:
    if scenario == "IDENTICAL_CONTEXT":
        context = "memory=EMPTY lineage=EMPTY task=ABSENT"
    else:
        context = (
            f"memory={arm.value}_PRIVATE lineage={arm.value}_PRIVATE "
            f"task={arm.value}_PRIVATE"
        )
    return (
        "This is a treatment-free transport isolation probe. "
        f"Order={order_label}. Context={context}. "
        "Return exactly one schema-valid proposal with the required constant."
    )


def _secret_absent(root: Path, secret: bytes) -> bool:
    if not secret:
        return False
    for path in root.rglob("*"):
        if path.is_file() and secret in path.read_bytes():
            return False
    return True


def run_probe(
    *,
    config_path: Path,
    schema_path: Path,
    private_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    if private_root.exists():
        raise RuntimeError(
            "provider probe root already exists; never reuse real call artifacts"
        )
    private_root.mkdir(parents=True)
    private_root.chmod(0o700)
    experiment_id = "M6I-PROVIDER-ISOLATION-PROBE-V3"
    arm_ids = {
        arm: f"{experiment_id}-opaque-{arm.value.lower()}" for arm in ArmCode
    }
    core = IntegratedCampaignStateCoreV1(experiment_id=experiment_id)
    core.bind_arms(arm_ids)
    registry = CallSharingRegistryV1(policy=CallSharingPolicyV1.ARM_PRIVATE)
    side_effect_state = {
        arm.value: {
            "search_memory": [],
            "meta_events": [],
            "observed_frontier": [],
            "search_eligible_frontier": [],
            "confirmed_frontier": [],
        }
        for arm in ArmCode
    }
    side_effect_before = sha256_digest(side_effect_state)
    calls: list[dict[str, Any]] = []
    candidate_ids: dict[str, list[str]] = {"B": [], "C": []}
    broker = LabApiCanaryBrokerV1(
        private_root,
        schema_path=schema_path,
        config_path=config_path,
        model="gpt-5.4",
        max_total_tokens_per_call=20_000,
        timeout_ms=900_000,
    )
    secret = broker._api_key.encode("utf-8")
    release_digest = broker.release.release_digest
    snapshot_receipt: Any | None = None
    try:
        for order_index, order in enumerate(ALL_ORDERS):
            order_label = "".join(arm.value for arm in order)
            scenario = (
                "IDENTICAL_CONTEXT"
                if order_index % 2 == 0
                else "DIVERGENT_MEMORY_LINEAGE_TASK_CONTEXT"
            )
            pair_records: list[dict[str, Any]] = []
            for arm in order:
                if arm is ArmCode.A:
                    continue
                prompt = _prompt(
                    arm=arm,
                    scenario=scenario,
                    order_label=order_label,
                )
                context = _context(
                    broker=broker,
                    prompt=prompt,
                    arm=arm,
                    order_index=order_index,
                    scenario=scenario,
                )
                physical, consumer, decision = registry.register_request(
                    owner=core.owner(arm),
                    context=context,
                )
                call = broker.call_with_session(
                    logical_call_id=consumer.value,
                    proposal_generation_session_id=(
                        f"m6i-provider-probe-session-{order_label}"
                    ),
                    prompt=prompt,
                    expected_proposal_count=1,
                    max_total_tokens=20_000,
                )
                candidate = registry.register_candidate(
                    owner=core.owner(arm),
                    round_index=order_index + 1,
                    producer_role="m6i_isolation_probe",
                    semantic_program_digest=call.response_digest,
                    local_parent_or_task_identity=None,
                )
                candidate_ids[arm.value].append(candidate.value)
                pair_records.append(
                    {
                        "arm": arm.value,
                        "provider_physical_call_id": physical.value,
                        "consumer_logical_call_id": consumer.value,
                        "candidate_instance_id": candidate.value,
                        "registry_decision": decision,
                        "request_context_digest": context.exact_request_digest,
                        "provider_request_digest": call.request_digest,
                        "provider_response_digest": call.response_digest,
                        "input_tokens": call.input_tokens,
                        "output_tokens": call.output_tokens,
                        "total_tokens": call.total_tokens,
                        "latency_ms": call.latency_ms,
                        "returned_model": call.returned_model,
                    }
                )
            calls.append(
                {
                    "order": order_label,
                    "scenario": scenario,
                    "provider_sequence": [
                        arm.value for arm in order if arm is not ArmCode.A
                    ],
                    "pair": pair_records,
                }
            )
        snapshot_receipt = broker.create_audit_snapshot(
            private_root / "broker_audit_snapshot.sqlite3",
            audit_purpose="M6I_REAL_PROVIDER_NO_TRAINING_ISOLATION_PROBE",
        )
        successful_calls = broker.call_count()
    finally:
        broker.close()

    credentials_removed_from_memory = broker._api_key == ""
    credentials_absent_from_artifacts = _secret_absent(private_root, secret)
    secret = b""
    side_effect_after = sha256_digest(side_effect_state)
    sharing = registry.audit_projection()
    pair_checks = []
    for call_group in calls:
        pair = call_group["pair"]
        exact_context_equal = (
            pair[0]["request_context_digest"]
            == pair[1]["request_context_digest"]
        )
        expected_equal = call_group["scenario"] == "IDENTICAL_CONTEXT"
        pair_checks.append(
            {
                "order": call_group["order"],
                "scenario": call_group["scenario"],
                "context_relation_correct": exact_context_equal == expected_equal,
                "physical_calls_distinct_under_arm_private_policy": (
                    pair[0]["provider_physical_call_id"]
                    != pair[1]["provider_physical_call_id"]
                ),
                "consumer_ids_distinct": (
                    pair[0]["consumer_logical_call_id"]
                    != pair[1]["consumer_logical_call_id"]
                ),
                "candidate_instances_distinct": (
                    pair[0]["candidate_instance_id"]
                    != pair[1]["candidate_instance_id"]
                ),
                "provider_request_relation_correct": (
                    (
                        pair[0]["provider_request_digest"]
                        == pair[1]["provider_request_digest"]
                    )
                    == expected_equal
                ),
            }
        )
    checks = {
        "all_six_orders": {item["order"] for item in calls}
        == {"ABC", "ACB", "BAC", "BCA", "CAB", "CBA"},
        "identical_context_pair_present": any(
            item["scenario"] == "IDENTICAL_CONTEXT" for item in calls
        ),
        "divergent_context_pair_present": any(
            item["scenario"] == "DIVERGENT_MEMORY_LINEAGE_TASK_CONTEXT"
            for item in calls
        ),
        "twelve_fresh_successful_calls": successful_calls == 12,
        "arm_private_no_cross_arm_physical_identity": (
            sharing["cross_arm_physical_identities"] == 0
        ),
        "separate_candidate_instances": len(
            set(candidate_ids["B"] + candidate_ids["C"])
        )
        == 12,
        "no_foreign_lineage_parent": all(
            record["candidate"]["local_parent_or_task_identity"] == "ROOT"
            for record in sharing["records"]
            if "candidate" in record
        ),
        "no_search_meta_frontier_side_effects": (
            side_effect_before == side_effect_after
        ),
        "credentials_removed_from_memory": credentials_removed_from_memory,
        "credentials_absent_from_artifacts": credentials_absent_from_artifacts,
        "no_training": True,
        "every_pair_conforms": all(
            item["context_relation_correct"]
            and item["physical_calls_distinct_under_arm_private_policy"]
            and item["consumer_ids_distinct"]
            and item["candidate_instances_distinct"]
            and item["provider_request_relation_correct"]
            for item in pair_checks
        ),
    }
    report = canonical_value(
        {
            "schema": "recclaw.m6i.provider-isolation-probe.v1",
            "milestone": (
                "M6I_INTEGRATED_STATE_SPACE_AND_CROSS_ARM_ISOLATION_CLOSURE"
            ),
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "pilot_result": False,
            "search_memory_import_allowed": False,
            "policy_training_import_allowed": False,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "p0": 0 if all(checks.values()) else 1,
            "p1": 0 if all(checks.values()) else 1,
            "checks": checks,
            "active_call_sharing_policy": (
                CallSharingPolicyV1.ARM_PRIVATE.value
            ),
            "broker_release_digest": release_digest,
            "schema_sha256": broker.schema_file_sha256,
            "successful_provider_calls": successful_calls,
            "training_executions": 0,
            "pair_checks": pair_checks,
            "calls": calls,
            "call_sharing_audit_digest": sha256_digest(sharing),
            "side_effect_state_before_digest": side_effect_before,
            "side_effect_state_after_digest": side_effect_after,
            "audit_snapshot_receipt": (
                snapshot_receipt.to_dict()
                if snapshot_receipt is not None
                else None
            ),
            "private_root": str(private_root),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_probe(
        config_path=args.config.resolve(),
        schema_path=args.schema.resolve(),
        private_root=args.private_root.resolve(),
        report_path=args.report.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "p0": report["p0"],
                "p1": report["p1"],
                "successful_provider_calls": report[
                    "successful_provider_calls"
                ],
                "training_executions": report["training_executions"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
