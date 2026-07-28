#!/usr/bin/env python3
"""Run the frozen five-round Meta V17 development Pilot exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_m6_pilot import (  # noqa: E402
    broker_export,
    collect_rows,
    file_sha256,
    pilot_environment_preflight,
    resource_audit,
    runtime_identity_audit,
    source_snapshot,
    write_json,
)

from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CodexCliCanaryBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
)
from recclaw_core.experiments.helix_abc_v1.m6e_conformance import (  # noqa: E402
    require_m6e_conformance_packet,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    MetaV17CampaignRuntimeV1,
    POLICY_BUNDLE_DIGEST_V17,
    PROMOTION_DECISION_DIGEST_V17,
    SOURCE_MANIFEST_DIGEST_V17,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_pilot import (  # noqa: E402
    META_V17_PILOT_ASSIGNMENT_NONCE,
    META_V17_PILOT_EXPERIMENT_ID,
    META_V17_PILOT_ROUNDS_PER_ARM,
    META_V17_PILOT_SEARCH_SEED,
    MetaV17PilotOrchestratorV1,
    MetaV17PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.pilot_analysis import (  # noqa: E402
    four_axis_frontiers,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    PrivateTreatmentAssignmentV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    pilot_budget,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (  # noqa: E402
    common_release_projection_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (  # noqa: E402
    TrainingExecutionPurposeV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    TRAINING_RUNNER_ABI,
    training_runtime_release_digest,
)


CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_DEVELOPMENT_PILOT_CONTRACT_V5.json"
)
ACTIVATION_RECEIPT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_NEXT_CAMPAIGN_ACTIVATION_RECEIPT_V5.json"
)
SEALED_PILOT_SEEDS = frozenset(range(9201, 9212))


def _verified_content_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    preimage = dict(record)
    expected = preimage.pop("content_digest")
    if sha256_digest(preimage) != expected:
        raise RuntimeError(f"content digest mismatch: {path}")
    return record


def verify_contract(contract_path: Path) -> dict[str, Any]:
    contract = _verified_content_record(contract_path)
    if (
        contract["status"] != "FROZEN_PRE_OUTCOME"
        or contract["record_schema"]
        != "recclaw.meta-v17-development-pilot-contract.v5"
        or contract["evidence_class"] != "DEVELOPMENT_ONLY"
        or contract["main_eligibility"] is not False
    ):
        raise RuntimeError("Meta V17 Pilot contract authority is invalid")
    for relative, expected_hash in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f"Meta V17 source mismatch: {relative}")
    exact_files = {
        Path(contract["activation"]["receipt_path"]): contract["activation"][
            "receipt_sha256"
        ],
        Path(contract["broker"]["response_schema_path"]): contract["broker"][
            "response_schema_sha256"
        ],
        Path(contract["broker"]["release_manifest_path"]): contract["broker"][
            "release_manifest_sha256"
        ],
        Path(contract["broker"]["codex_executable"]): contract["broker"][
            "codex_executable_sha256"
        ],
        Path(contract["broker"]["models_cache_path"]): contract["broker"][
            "models_cache_sha256"
        ],
        Path(contract["bl_icf"]["template_fixture_path"]): contract["bl_icf"][
            "template_fixture_sha256"
        ],
        Path(contract["meta"]["checkpoint_path"]): contract["meta"][
            "checkpoint_sha256"
        ],
        Path(contract["training"]["profile_path"]): contract["training"][
            "profile_sha256"
        ],
    }
    for path, expected_hash in exact_files.items():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Meta V17 external identity mismatch: {path}")

    receipt = _verified_content_record(Path(contract["activation"]["receipt_path"]))
    if (
        receipt["activation_applied"] is not True
        or receipt["activation_boundary"] != "NEXT_CAMPAIGN"
        or receipt["experiment_id"] != META_V17_PILOT_EXPERIMENT_ID
        or receipt["search_seed"] != META_V17_PILOT_SEARCH_SEED
        or receipt["rounds_per_arm"] != META_V17_PILOT_ROUNDS_PER_ARM
        or receipt["promotion_decision_digest"]
        != PROMOTION_DECISION_DIGEST_V17
        or receipt["policy_bundle_digest"] != POLICY_BUNDLE_DIGEST_V17
        or receipt["source_manifest_digest"] != SOURCE_MANIFEST_DIGEST_V17
        or receipt["historical_campaign_backfill"] is not False
    ):
        raise RuntimeError("Meta V17 activation receipt is not the exact future boundary")

    store = MetaV17PilotStoreContractV1.create()
    expected_pilot = {
        "experiment_id": store.experiment_id,
        "ordinary_execution_seed": store.ordinary_execution_seed,
        "rounds_per_arm": store.scheduled_slots_per_arm_seed,
        "search_seeds": list(store.search_seeds),
        "store_contract_identity_digest": store.identity_digest,
    }
    if contract["pilot"] != expected_pilot:
        raise RuntimeError("Meta V17 Pilot store identity mismatch")
    if SEALED_PILOT_SEEDS.intersection(store.search_seeds):
        raise RuntimeError("Meta V17 Pilot reuses a sealed seed")
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id,
        nonce=META_V17_PILOT_ASSIGNMENT_NONCE,
    )
    if contract["assignment"] != {
        "commitment": assignment.commitment,
        "opaque": True,
    }:
        raise RuntimeError("Meta V17 private assignment mismatch")
    if contract["budget_per_arm_round"] != pilot_budget().to_dict():
        raise RuntimeError("Meta V17 Pilot changed the common budget")
    if (
        contract["bl_icf"]["common_release_projection_digest"]
        != common_release_projection_digest()
        or contract["training"]["runner_abi"] != TRAINING_RUNNER_ABI
        or contract["training"]["runtime_release_digest"]
        != training_runtime_release_digest()
        or contract["training"]["execution_purpose"]
        != TrainingExecutionPurposeV1.PILOT.value
    ):
        raise RuntimeError("Meta V17 Pilot changed the common execution substrate")
    if contract["meta"]["policy_bundle_digest"] != POLICY_BUNDLE_DIGEST_V17:
        raise RuntimeError("Meta V17 policy bundle identity mismatch")
    if contract["candidate_schedule"]["logical_call_count"] != 45:
        raise RuntimeError("Meta V17 five-round candidate schedule changed")
    if contract["research"]["B"] != contract["research"]["C_non_guard"]:
        raise RuntimeError("B/C differ outside Evidence Guard")
    return contract


def descriptive_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    arm_to_instance: Mapping[str, str],
) -> dict[str, Any]:
    by_key = {
        (str(row["opaque_instance_id"]), int(row["round_index"])): row
        for row in rows
    }
    per_round = []
    confirmed_frontier: dict[str, float | None] = {
        "A": None,
        "B": None,
        "C": None,
    }
    for round_index in range(1, META_V17_PILOT_ROUNDS_PER_ARM + 1):
        metrics = {}
        candidates = {}
        for arm, instance in sorted(arm_to_instance.items()):
            row = by_key.get((instance, round_index))
            metrics[arm] = (
                float(row["ndcg"])
                if row is not None and isinstance(row.get("ndcg"), (int, float))
                else None
            )
            candidates[arm] = row["candidate_id"] if row is not None else None

        def delta(left: str, right: str) -> float | None:
            if metrics[left] is None or metrics[right] is None:
                return None
            return float(metrics[left]) - float(metrics[right])

        for arm, metric in metrics.items():
            if metric is not None:
                confirmed_frontier[arm] = (
                    metric
                    if confirmed_frontier[arm] is None
                    else max(float(confirmed_frontier[arm]), metric)
                )

        def frontier_delta(left: str, right: str) -> float | None:
            if (
                confirmed_frontier[left] is None
                or confirmed_frontier[right] is None
            ):
                return None
            return float(confirmed_frontier[left]) - float(
                confirmed_frontier[right]
            )

        per_round.append(
            {
                "candidate_ids": candidates,
                "confirmed_frontier_ndcg": dict(confirmed_frontier),
                "contrasts": {
                    "B_minus_A": delta("B", "A"),
                    "C_minus_A": delta("C", "A"),
                    "C_minus_B": delta("C", "B"),
                },
                "frontier_contrasts": {
                    "B_minus_A": frontier_delta("B", "A"),
                    "C_minus_A": frontier_delta("C", "A"),
                    "C_minus_B": frontier_delta("C", "B"),
                },
                "ndcg": metrics,
                "round_index": round_index,
            }
        )
    values = {
        arm: [
            float(item["ndcg"][arm])
            for item in per_round
            if item["ndcg"][arm] is not None
        ]
        for arm in ("A", "B", "C")
    }
    means = {
        arm: (
            sum(arm_values) / len(arm_values)
            if arm_values
            else None
        )
        for arm, arm_values in values.items()
    }

    def mean_delta(left: str, right: str) -> float | None:
        if means[left] is None or means[right] is None:
            return None
        return float(means[left]) - float(means[right])

    return {
        "aggregate_descriptive": {
            "B_minus_A": mean_delta("B", "A"),
            "C_minus_A": mean_delta("C", "A"),
            "C_minus_B": mean_delta("C", "B"),
            "final_confirmed_frontier": dict(confirmed_frontier),
            "final_frontier_contrasts": {
                "B_minus_A": (
                    float(confirmed_frontier["B"])
                    - float(confirmed_frontier["A"])
                    if confirmed_frontier["B"] is not None
                    and confirmed_frontier["A"] is not None
                    else None
                ),
                "C_minus_A": (
                    float(confirmed_frontier["C"])
                    - float(confirmed_frontier["A"])
                    if confirmed_frontier["C"] is not None
                    and confirmed_frontier["A"] is not None
                    else None
                ),
                "C_minus_B": (
                    float(confirmed_frontier["C"])
                    - float(confirmed_frontier["B"])
                    if confirmed_frontier["C"] is not None
                    and confirmed_frontier["B"] is not None
                    else None
                ),
            },
            "mean_ndcg": means,
        },
        "formal_inference": False,
        "interpretation": "FIVE_ROUND_DEVELOPMENT_SIGNAL_ONLY",
        "main_evidence": False,
        "per_round": per_round,
    }


def pilot_quality_review(
    *,
    rows: Sequence[Mapping[str, Any]],
    meta_audit: Mapping[str, Any],
    expected_instance_ids: set[str],
    guard_call_count: int,
) -> dict[str, Any]:
    grouped = {
        instance: [row for row in rows if row["opaque_instance_id"] == instance]
        for instance in expected_instance_ids
    }
    routes = list(meta_audit["routes"])
    observations = list(meta_audit["observations"])
    active_routes = [
        route
        for route in routes
        if route["mode"] == "META_VNEXT_V17_SLOW_PLUS_FAST"
    ]
    collision_count = sum(
        int(route["round_semantic_collision_count"]) for route in routes
    )
    proposal_denominator = 4 * 2 * META_V17_PILOT_ROUNDS_PER_ARM
    selected_semantics = {
        route["selected_candidate_semantics_digest"]
        for route in active_routes
        if route["selected_candidate_semantics_digest"] is not None
    }
    selected_axes = {
        observation["selected_axis"]
        for observation in observations
        if observation["selected_axis"] != "other"
    }
    pool_semantics = {
        digest
        for route in active_routes
        for digest in route["pool_candidate_semantics_digests"]
    }
    engineering = {
        "all_training_runs_successful": (
            len(rows) == 15
            and all(row["run_status"] == "SUCCESS" for row in rows)
        ),
        "five_rows_per_arm": (
            len(grouped) == 3
            and all(len(instance_rows) == 5 for instance_rows in grouped.values())
        ),
        "guard_pre_post_exact": guard_call_count == 10,
        "meta_observation_closure": len(routes) == 10 and len(observations) == 10,
        "meta_round_boundaries_closed": all(
            int(state["round_boundary"]) == 5
            for state in meta_audit["states"].values()
        ),
    }
    quality = {
        "active_meta_route_fraction": len(active_routes) / 10.0,
        "collision_rate": collision_count / proposal_denominator,
        "meta_override_count": sum(
            route["selected_candidate_id"]
            != route["static_champion_candidate_id"]
            for route in active_routes
        ),
        "pool_unique_semantics": len(pool_semantics),
        "selected_mechanism_axis_count": len(selected_axes),
        "selected_mechanism_axes": sorted(selected_axes),
        "selected_unique_semantics": len(selected_semantics),
        "task_supported_route_fraction": (
            sum(bool(route["task_context_supported"]) for route in routes)
            / 10.0
        ),
    }
    information_value = (
        quality["active_meta_route_fraction"] >= 0.6
        and quality["collision_rate"] <= 0.5
        and quality["meta_override_count"] >= 1
        and quality["pool_unique_semantics"] >= 4
        and quality["selected_unique_semantics"] >= 3
        and quality["task_supported_route_fraction"] == 1.0
    )
    return {
        "engineering_checks": engineering,
        "engineering_closed": all(engineering.values()),
        "expansion_has_information_value": information_value,
        "predeclared_quality_thresholds": {
            "active_meta_route_fraction_min": 0.6,
            "collision_rate_max": 0.5,
            "meta_override_count_min": 1,
            "pool_unique_semantics_min": 4,
            "selected_unique_semantics_min": 3,
            "task_supported_route_fraction_required": 1.0,
        },
        "quality": quality,
    }


def execute(contract_path: Path, output_root: Path) -> int:
    if output_root.exists():
        raise RuntimeError("Meta V17 Pilot output root already exists")
    output_root.mkdir(parents=True)
    contract = verify_contract(contract_path)
    write_json(
        output_root / "FROZEN_CONTRACT_IDENTITY.json",
        {
            "contract_content_digest": contract["content_digest"],
            "contract_sha256": file_sha256(contract_path),
        },
    )
    preflight = pilot_environment_preflight(contract)
    write_json(output_root / "ENVIRONMENT_PREFLIGHT.json", preflight)
    before = source_snapshot(contract)
    meta_runtime = MetaV17CampaignRuntimeV1(
        checkpoint_path=Path(contract["meta"]["checkpoint_path"]),
        experiment_id=META_V17_PILOT_EXPERIMENT_ID,
        search_seed=META_V17_PILOT_SEARCH_SEED,
        scheduled_rounds=META_V17_PILOT_ROUNDS_PER_ARM,
        task_scale=float(contract["meta"]["task_context"]["task_scale"]),
        task_density=float(contract["meta"]["task_context"]["task_density"]),
    )
    task_support = meta_runtime.task_support_projection()
    write_json(
        output_root / "META_V17_TASK_SUPPORT_PREFLIGHT.json",
        task_support,
    )
    if not task_support["supported"]:
        write_json(
            output_root / "META_V17_PILOT_FAILURE.json",
            {
                "authority": "NONE",
                "error_type": "MetaV17TaskOutOfSupport",
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
                "main_eligibility": False,
                "reason": "META_POLICY_TASK_CONTEXT_OUT_OF_SUPPORT",
                "task_support": task_support,
                "verdict": "NOT_READY",
            },
        )
        return 2
    upstream = CodexCliCanaryBrokerV1(
        output_root / "broker_private",
        schema_path=Path(contract["broker"]["response_schema_path"]),
        codex_executable=Path(contract["broker"]["codex_executable"]),
        model=contract["broker"]["model"],
        reasoning_effort=contract["broker"]["reasoning_effort"],
        service_tier=contract["broker"]["service_tier"],
        max_total_tokens_per_call=int(
            contract["broker"]["max_total_tokens_per_call"]
        ),
        cli_version=contract["broker"]["codex_cli_version"],
        login_mode=contract["broker"]["login_mode"],
        release_manifest_path=Path(contract["broker"]["release_manifest_path"]),
    )
    broker = RealCanaryProposalBrokerV1.create(
        upstream=upstream,
        template_path=Path(contract["bl_icf"]["template_fixture_path"]),
        call_prefix="m6-meta-v17-",
        phase_name="Meta V17 Pilot",
        adaptive_memory=True,
        campaign_meta_runtime=meta_runtime,
    )
    try:
        with MetaV17PilotOrchestratorV1(
            output_root / "runtime",
            broker=broker,
            meta_runtime=meta_runtime,
            project_root=ROOT,
            recbole_root=Path(contract["runtime"]["recbole_root"]),
            data_path=Path(contract["dataset"]["root"]).parent,
            python_executable=Path(contract["runtime"]["python"]),
        ) as orchestrator:
            initial_meta_projection = {
                arm.value: meta_runtime.initial_semantic_state_projection(arm)
                for arm in (ArmCode.B, ArmCode.C)
            }
            try:
                rounds = orchestrator.run_pilot()
            except Exception:
                write_json(
                    output_root / "IMMUTABLE_AUDIT_BUNDLE.json",
                    orchestrator.immutable_audit_bundle(
                        output_root / "audit_snapshots"
                    ),
                )
                write_json(
                    output_root / "sealed" / "META_V17_AUDIT.json",
                    meta_runtime.audit_projection(),
                    mode=0o600,
                )
                raise
            audit_bundle = orchestrator.immutable_audit_bundle(
                output_root / "audit_snapshots"
            )
            write_json(
                output_root / "IMMUTABLE_AUDIT_BUNDLE.json",
                audit_bundle,
            )
            state_db = (
                output_root / "audit_snapshots" / "neutral_state.audit.sqlite3"
            )
            broker_db = (
                output_root / "audit_snapshots" / "broker_state.audit.sqlite3"
            )
            guard_db = (
                output_root / "audit_snapshots" / "guard_state.audit.sqlite3"
            )
            audit = orchestrator.pilot_audit(state_db, guard_db)
            mapping = {
                arm.value: opaque
                for arm, opaque in orchestrator.assignment.arm_to_instance
            }
            meta_audit = meta_runtime.audit_projection()
            write_json(
                output_root / "sealed" / "ROUND_RESULTS.json",
                [
                    [result.to_dict() for result in triplet]
                    for triplet in rounds
                ],
                mode=0o600,
            )
            write_json(
                output_root / "sealed" / "TREATMENT_MAPPING.json",
                {
                    "assignment_commitment": orchestrator.assignment.commitment,
                    "mapping": mapping,
                    "nonce_digest": orchestrator.assignment.nonce_digest,
                },
                mode=0o600,
            )
            write_json(
                output_root / "sealed" / "META_V17_AUDIT.json",
                meta_audit,
                mode=0o600,
            )
            write_json(
                output_root / "NEUTRAL_AUDIT.json",
                audit_bundle["neutral_projection"],
            )
            resource = resource_audit(state_db, contract)
            runtime_identity = runtime_identity_audit(state_db)
            rows = collect_rows(
                state_db,
                output_root / "runtime" / "neutral" / "artifacts",
            )
        upstream.close()
        calls = broker_export(broker_db)
        after = source_snapshot(contract)
        quality = pilot_quality_review(
            rows=rows,
            meta_audit=meta_audit,
            expected_instance_ids=set(mapping.values()),
            guard_call_count=int(audit["guard_call_count"]),
        )
        effects = descriptive_effect_summary(rows, mapping)
        actual_broker_calls = sum(row["call_count"] for row in calls)
        broker_failures = sum(
            row["call_count"] for row in calls if row["status"] != "SUCCESS"
        )
        gates = {
            "barriers_closed": bool(audit["barriers_closed"]),
            "broker_call_count": actual_broker_calls,
            "broker_call_count_in_frozen_range": (
                int(contract["broker"]["expected_upstream_calls_min"])
                <= actual_broker_calls
                <= int(contract["broker"]["expected_upstream_calls_max"])
            ),
            "broker_failures": broker_failures,
            "budget_accounting_closed": bool(resource["closed"]),
            "execution_count": int(audit["execution_count"]),
            "feedback_count": int(audit["feedback_count"]),
            "guard_call_count": int(audit["guard_call_count"]),
            "initial_meta_semantics_equal": (
                initial_meta_projection["B"] == initial_meta_projection["C"]
            ),
            "meta_policy_bundle_exact": (
                meta_audit["policy_bundle_digest"] == POLICY_BUNDLE_DIGEST_V17
            ),
            "no_source_mutation": before == after,
            "round_count": int(audit["round_count"]),
            "runtime_identity_equal": bool(
                runtime_identity["A_B_C_common_runtime_identity_equal"]
            ),
            "state_store_integrity": (
                audit["state_store_integrity"]["integrity_check"] == "ok"
                and not audit["state_store_integrity"]["foreign_key_violations"]
            ),
        }
        engineering_pass = (
            all(
                (
                    gates["barriers_closed"],
                    gates["broker_call_count_in_frozen_range"],
                    gates["broker_failures"] == 0,
                    gates["budget_accounting_closed"],
                    gates["execution_count"] == 15,
                    gates["feedback_count"] == 15,
                    gates["guard_call_count"] == 10,
                    gates["initial_meta_semantics_equal"],
                    gates["meta_policy_bundle_exact"],
                    gates["no_source_mutation"],
                    gates["round_count"] == 15,
                    gates["runtime_identity_equal"],
                    gates["state_store_integrity"],
                    quality["engineering_closed"],
                )
            )
            and runtime_identity["closed_execution_counts"] == [5, 5, 5]
            and runtime_identity["runtime_binding_count"] == 15
        )
        issue_counts = {
            "P0": 0 if engineering_pass else 1,
            "P1": 0 if engineering_pass else 1,
            "P2": 0 if quality["expansion_has_information_value"] else 1,
        }
        expansion = (
            "ELIGIBLE_TO_PROPOSE_10_ROUND_IN_A_LATER_TURN"
            if engineering_pass and quality["expansion_has_information_value"]
            else "DO_NOT_EXPAND"
        )
        write_json(output_root / "BROKER_CALL_AUDIT.json", calls)
        write_json(
            output_root / "sealed" / "PILOT_ITT_ROWS.json",
            rows,
            mode=0o600,
        )
        write_json(
            output_root / "sealed" / "FOUR_AXIS_FRONTIERS.json",
            four_axis_frontiers(rows),
            mode=0o600,
        )
        write_json(output_root / "META_V17_PILOT_QUALITY_REVIEW.json", quality)
        write_json(output_root / "DESCRIPTIVE_EFFECT_SUMMARY.json", effects)
        write_json(
            output_root / "RUNTIME_IDENTITY_AUDIT.json",
            runtime_identity,
        )
        result = {
            "authority": "NONE",
            "contract_content_digest": contract["content_digest"],
            "engineering_pass": engineering_pass,
            "evidence_class": "DEVELOPMENT_ONLY",
            "expansion_recommendation": expansion,
            "formal_acceptance": False,
            "gates": gates,
            "issue_counts_are_local_not_independent": True,
            "local_issue_counts": issue_counts,
            "main_eligibility": False,
            "resource_audit": resource,
            "runtime_identity_audit": runtime_identity,
            "source_snapshot_digest": sha256_digest(after),
            "verdict": (
                "DEVELOPMENT_PILOT_COMPLETE"
                if engineering_pass
                else "NOT_READY"
            ),
        }
        write_json(output_root / "META_V17_PILOT_EXECUTION_RESULT.json", result)
        return 0 if engineering_pass else 2
    except Exception as error:
        try:
            upstream.close()
        except Exception:
            pass
        closure = getattr(error, "closure", None)
        write_json(
            output_root / "META_V17_PILOT_FAILURE.json",
            {
                "authority": "NONE",
                "broker_failure_closure_digest": (
                    closure.closure_digest if closure is not None else None
                ),
                "error_type": type(error).__name__,
                "error_message": str(error)[:1000],
                "evidence_class": "DEVELOPMENT_ONLY",
                "failure_class": (
                    closure.failure_class if closure is not None else None
                ),
                "formal_acceptance": False,
                "main_eligibility": False,
                "reason": (
                    "BROKER_PROCESS_FAILURE/COMMON_NO_EXECUTION"
                    if closure is not None
                    else "INTERNAL_META_V17_PILOT_FAILURE"
                ),
                "verdict": "NOT_READY",
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.contract.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
