#!/usr/bin/env python3
"""Freeze V17 activation and its one permitted five-round Pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from freeze_m6_pilot_v4_contract import file_sha256, git_value  # noqa: E402
from run_m6_meta_v17_pilot import (  # noqa: E402
    ACTIVATION_RECEIPT_PATH,
    CONTRACT_PATH,
    SEALED_PILOT_SEEDS,
)

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    POLICY_BUNDLE_DIGEST_V17,
    PROMOTION_DECISION_DIGEST_V17,
    SOURCE_MANIFEST_DIGEST_V17,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_pilot import (  # noqa: E402
    META_V17_PILOT_ASSIGNMENT_NONCE,
    META_V17_PILOT_EXPERIMENT_ID,
    META_V17_PILOT_ROUNDS_PER_ARM,
    META_V17_PILOT_SEARCH_SEED,
    MetaV17PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    PrivateTreatmentAssignmentV1,
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
    training_runtime_release,
)


STATIC_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6s"
    / "STATIC_DIAGNOSTIC_PILOT_CONTRACT_V2.json"
)
V1_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_DEVELOPMENT_PILOT_CONTRACT_V1.json"
)
V1_ACTIVATION_RECEIPT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_NEXT_CAMPAIGN_ACTIVATION_RECEIPT_V1.json"
)
V1_FAILURE_CLOSURE_PATH = (
    ROOT
    / "results"
    / "research_line"
    / "m6_meta_v17_pilot_9208_v6"
    / "PRE_OUTCOME_FAILURE_CLOSURE.json"
)
V2_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_DEVELOPMENT_PILOT_CONTRACT_V2.json"
)
V2_ACTIVATION_RECEIPT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_NEXT_CAMPAIGN_ACTIVATION_RECEIPT_V2.json"
)
V2_FAILURE_CLOSURE_PATH = (
    ROOT
    / "results"
    / "research_line"
    / "m6_meta_v17_pilot_9209_v7"
    / "PRE_OUTCOME_FAILURE_CLOSURE.json"
)
V3_SCHEMA_CONFORMANCE_PATH = (
    ROOT
    / "results"
    / "research_line"
    / "m6_meta_v17_schema_v3_conformance_probe_v2"
    / "CONFORMANCE_RESULT.json"
)
V3_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_DEVELOPMENT_PILOT_CONTRACT_V3.json"
)
V3_ACTIVATION_RECEIPT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_NEXT_CAMPAIGN_ACTIVATION_RECEIPT_V3.json"
)
V3_FAILURE_CLOSURE_PATH = (
    ROOT
    / "results"
    / "research_line"
    / "m6_meta_v17_pilot_9210_v8"
    / "PRE_OUTCOME_FAILURE_CLOSURE.json"
)
V4_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_DEVELOPMENT_PILOT_CONTRACT_V4.json"
)
V4_ACTIVATION_RECEIPT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "META_V17_NEXT_CAMPAIGN_ACTIVATION_RECEIPT_V4.json"
)
V4_FAILURE_CLOSURE_PATH = (
    ROOT
    / "results"
    / "research_line"
    / "m6_meta_v17_pilot_9211_v9"
    / "POST_OUTCOME_FAILURE_CLOSURE.json"
)
M6E_PACKET_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6e"
    / "M6E_TRAINING_RUNTIME_CONFORMANCE_PACKET.json"
)
M6F_RECORD_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6f"
    / "M6F_EXECUTION_RECORD.json"
)
M6F_AUDIT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6f"
    / "M6F_CLOSURE_INDEPENDENT_AUDIT.md"
)
TASK_AUTHORIZATION_PATH = ROOT / "RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md"
META_DOC_ROOT = ROOT / "docs" / "research_line" / "meta_vnext"
PROMOTION_PATH = META_DOC_ROOT / "META_VNEXT_PROMOTION_DECISION_V17.json"
ACTIVATION_CLOSURE_PATH = (
    META_DOC_ROOT / "META_VNEXT_ACTIVATION_GATE_CLOSURE_V1.json"
)
FINAL_EVIDENCE_PATH = META_DOC_ROOT / "META_VNEXT_FINAL_EVIDENCE_V1.json"
IMPLEMENTATION_MANIFEST_PATH = (
    META_DOC_ROOT / "META_VNEXT_IMPLEMENTATION_MANIFEST_V1.json"
)
REQUALIFICATION_PATH = (
    META_DOC_ROOT / "META_VNEXT_REQUALIFICATION_RECORD_V1.json"
)
CHECKPOINT_PATH = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "meta_vnext_policy_checkpoint_v17.json"
)
BROKER_RELEASE_PATH = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "broker_process_release_v2_pilot_schema_v3.json"
)
RESPONSE_SCHEMA = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "pilot_proposal_response_v3.schema.json"
)
MODELS_CACHE = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "pilot_v5_model_catalog_snapshot.json"
)
TEMPLATE_FIXTURE = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
TRAINING_PROFILE = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "pilot_training_profile_v1.json"
)


def _meta_source_files() -> set[str]:
    package_root = (
        ROOT / "src" / "recclaw_core" / "experiments" / "helix_abc_v1"
    )
    paths = {
        path.relative_to(ROOT).as_posix()
        for directory in ("meta_v2", "meta_vnext")
        for path in (package_root / directory).glob("*.py")
    }
    return paths


SOURCE_FILES = {
    "scripts/freeze_m6_meta_v17_pilot_contract.py",
    "scripts/run_m6_meta_v17_pilot.py",
    "scripts/run_m6_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/audit_snapshot.py",
    "src/recclaw_core/experiments/helix_abc_v1/broker_failure_closure.py",
    "src/recclaw_core/experiments/helix_abc_v1/broker_process.py",
    "src/recclaw_core/experiments/helix_abc_v1/canary_broker.py",
    "src/recclaw_core/experiments/helix_abc_v1/common_execution_guard.py",
    "src/recclaw_core/experiments/helix_abc_v1/materialization.py",
    "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_campaign.py",
    "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/pilot_analysis.py",
    "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py",
    "src/recclaw_core/experiments/helix_abc_v1/precanary_orchestration.py",
    "src/recclaw_core/experiments/helix_abc_v1/real_canary.py",
    "src/recclaw_core/experiments/helix_abc_v1/real_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/research_capability.py",
    "src/recclaw_core/experiments/helix_abc_v1/research_contracts.py",
    "src/recclaw_core/experiments/helix_abc_v1/research_controller.py",
    "src/recclaw_core/experiments/helix_abc_v1/runtime_contracts.py",
    "src/recclaw_core/experiments/helix_abc_v1/runtime_release.py",
    "src/recclaw_core/experiments/helix_abc_v1/state_store.py",
    "src/recclaw_core/experiments/helix_abc_v1/store_audit.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_execution_guard.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_filesystem.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_materialization.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_runtime_contracts.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_runtime_release.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_state_store.py",
    "src/recclaw_core/helix/composition.py",
    "src/recclaw_core/helix/contracts.py",
    "src/recclaw_core/helix/fusion.py",
    "src/recclaw_core/helix/guard_adapter.py",
    "src/recclaw_core/helix/ledger.py",
    "src/recclaw_core/helix/ports.py",
    "tests/experiments/helix_abc_v1/test_meta_v17_campaign.py",
} | _meta_source_files()


def candidate_schedule() -> list[dict[str, Any]]:
    roles = [
        "mechanism_composer",
        "lineage_refiner",
        "falsification_designer",
        "frontier_architect",
    ]
    return [
        {
            "round_index": round_index,
            "slots": {
                "A": ["original_search"],
                "B": roles,
                "C": roles,
            },
        }
        for round_index in range(1, META_V17_PILOT_ROUNDS_PER_ARM + 1)
    ]


def build_activation_receipt() -> dict[str, Any]:
    promotion = json.loads(PROMOTION_PATH.read_text(encoding="utf-8"))
    closure = json.loads(ACTIVATION_CLOSURE_PATH.read_text(encoding="utf-8"))
    if (
        promotion["decision"]["verdict"] != "PROMOTE"
        or promotion["decision_digest"] != PROMOTION_DECISION_DIGEST_V17
        or promotion["activation_applied"] is not False
        or closure["verdict"] != "PASS"
        or closure["activation_applied"] is not False
        or closure["activation_boundary"] != "NEXT_CAMPAIGN"
    ):
        raise RuntimeError("sealed Meta V17 evidence does not authorize activation")
    schedule = candidate_schedule()
    receipt = {
        "activation_applied": True,
        "activation_boundary": "NEXT_CAMPAIGN",
        "authority": "NONE",
        "candidate_schedule_digest": sha256_digest(schedule),
        "evidence_class": "DEVELOPMENT_ONLY",
        "experiment_id": META_V17_PILOT_EXPERIMENT_ID,
        "historical_activation_applied": False,
        "historical_campaign_backfill": False,
        "policy_bundle_digest": POLICY_BUNDLE_DIGEST_V17,
        "promotion_decision_digest": PROMOTION_DECISION_DIGEST_V17,
        "record_schema": "recclaw.meta-v17-next-campaign-activation-receipt.v5",
        "rounds_per_arm": META_V17_PILOT_ROUNDS_PER_ARM,
        "search_seed": META_V17_PILOT_SEARCH_SEED,
        "source_manifest_digest": SOURCE_MANIFEST_DIGEST_V17,
    }
    receipt["content_digest"] = sha256_digest(receipt)
    return receipt


def build_contract() -> dict[str, Any]:
    predecessor = json.loads(STATIC_CONTRACT_PATH.read_text(encoding="utf-8"))
    m6e = json.loads(M6E_PACKET_PATH.read_text(encoding="utf-8"))
    m6f = json.loads(M6F_RECORD_PATH.read_text(encoding="utf-8"))
    if m6e["verdict"] != "PASS" or m6e["P0"] or m6e["P1"]:
        raise RuntimeError("M6E is not passing")
    if m6f["verdict"] != "PASS" or m6f["audit"]["P0"] or m6f["audit"]["P1"]:
        raise RuntimeError("M6F is not passing")
    if sorted(SEALED_PILOT_SEEDS) != list(range(9201, 9212)):
        raise RuntimeError("sealed Pilot seed registry changed")

    store = MetaV17PilotStoreContractV1.create()
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id,
        nonce=META_V17_PILOT_ASSIGNMENT_NONCE,
    )
    release = training_runtime_release()
    broker_release = json.loads(BROKER_RELEASE_PATH.read_text(encoding="utf-8"))
    model_catalog = json.loads(MODELS_CACHE.read_text(encoding="utf-8"))
    source_files = {
        relative: file_sha256(ROOT / relative)
        for relative in sorted(SOURCE_FILES)
    }
    schedule = candidate_schedule()
    research_identity = {
        "controller": "ResearchLineControllerV1",
        "meta_algorithm": "META_VNEXT_V17_SLOW_PLUS_FAST",
        "meta_policy_bundle_digest": POLICY_BUNDLE_DIGEST_V17,
        "producer_mode": "BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1",
        "router_hard_gates": "StrongStaticRouterV1",
    }
    contract = {
        "activation": {
            "receipt_content_digest": json.loads(
                ACTIVATION_RECEIPT_PATH.read_text(encoding="utf-8")
            )["content_digest"],
            "receipt_path": str(ACTIVATION_RECEIPT_PATH),
            "receipt_sha256": file_sha256(ACTIVATION_RECEIPT_PATH),
        },
        "analysis": {
            "effect_interpretation": "FIVE_ROUND_DEVELOPMENT_SIGNAL_ONLY",
            "formal_inference": False,
            "four_axes": [
                "ROUND",
                "EXECUTION_COUNT",
                "BILLED_TOKEN",
                "GPU_NORMALIZED_COST",
            ],
            "primary_contrasts": ["B_MINUS_A", "C_MINUS_B"],
            "quality_review_required_before_expansion": True,
            "ten_round_auto_expansion": False,
        },
        "arm_composition": predecessor["arm_composition"],
        "assignment": {
            "commitment": assignment.commitment,
            "opaque": True,
        },
        "authority": "NONE",
        "bl_icf": {
            "common_release_projection_digest": (
                common_release_projection_digest()
            ),
            "template_fixture_path": str(TEMPLATE_FIXTURE),
            "template_fixture_sha256": file_sha256(TEMPLATE_FIXTURE),
        },
        "broker": {
            "adaptive_compact_search_memory": True,
            "broker_process_release_digest": broker_release["release_digest"],
            "codex_cli_version": broker_release["broker_cli_version"],
            "codex_executable": broker_release["broker_executable_path"],
            "codex_executable_sha256": broker_release[
                "broker_executable_sha256"
            ],
            "expected_upstream_calls_max": 41,
            "expected_upstream_calls_min": 25,
            "first_round_equal_memory_replay": True,
            "login_mode": broker_release["login_mode"],
            "max_total_tokens_per_call": 20000,
            "model": broker_release["model"],
            "models_cache_etag": model_catalog["etag"],
            "models_cache_mode": "FROZEN_SELECTED_MODEL_CATALOG_PROJECTION_V1",
            "models_cache_path": str(MODELS_CACHE),
            "models_cache_sha256": file_sha256(MODELS_CACHE),
            "reasoning_effort": broker_release["reasoning_effort"],
            "release_manifest_path": str(BROKER_RELEASE_PATH),
            "release_manifest_sha256": file_sha256(BROKER_RELEASE_PATH),
            "response_schema_path": str(RESPONSE_SCHEMA),
            "response_schema_sha256": file_sha256(RESPONSE_SCHEMA),
            "retries": 0,
            "sandbox": broker_release["sandbox_mode"],
            "service_tier": "default",
        },
        "budget_per_arm_round": pilot_budget().to_dict(),
        "candidate_schedule": {
            "logical_call_count": 45,
            "schedule": schedule,
            "schedule_digest": sha256_digest(schedule),
        },
        "common_runtime_authorization": {
            "path": str(TASK_AUTHORIZATION_PATH),
            "sha256": file_sha256(TASK_AUTHORIZATION_PATH),
        },
        "dataset": predecessor["dataset"],
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
        "guard_and_fusion": predecessor["guard_and_fusion"],
        "lineage_policy": {
            "historical_campaigns_rewritten": False,
            "main_memory_imported": False,
            "pilot_outputs_reusable_in_main": False,
            "v17_activation_boundary": "NEXT_CAMPAIGN",
        },
        "lineage_predecessors": {
            "m6f_audit_sha256": file_sha256(M6F_AUDIT_PATH),
            "m6f_evidence_commit": (
                "ad6c7b6e51ebabf5af4a43244d5769b0233f8603"
            ),
            "m6f_execution_record_sha256": file_sha256(M6F_RECORD_PATH),
            "static_diagnostic_v2_contract_sha256": file_sha256(
                STATIC_CONTRACT_PATH
            ),
            "v1_activation_receipt_sha256": file_sha256(
                V1_ACTIVATION_RECEIPT_PATH
            ),
            "v1_contract_sha256": file_sha256(V1_CONTRACT_PATH),
            "v1_pre_outcome_failure_closure_sha256": file_sha256(
                V1_FAILURE_CLOSURE_PATH
            ),
            "v2_activation_receipt_sha256": file_sha256(
                V2_ACTIVATION_RECEIPT_PATH
            ),
            "v2_contract_sha256": file_sha256(V2_CONTRACT_PATH),
            "v2_pre_outcome_failure_closure_sha256": file_sha256(
                V2_FAILURE_CLOSURE_PATH
            ),
            "v3_schema_conformance_sha256": file_sha256(
                V3_SCHEMA_CONFORMANCE_PATH
            ),
            "v3_activation_receipt_sha256": file_sha256(
                V3_ACTIVATION_RECEIPT_PATH
            ),
            "v3_contract_sha256": file_sha256(V3_CONTRACT_PATH),
            "v3_pre_outcome_failure_closure_sha256": file_sha256(
                V3_FAILURE_CLOSURE_PATH
            ),
            "v4_activation_receipt_sha256": file_sha256(
                V4_ACTIVATION_RECEIPT_PATH
            ),
            "v4_contract_sha256": file_sha256(V4_CONTRACT_PATH),
            "v4_post_outcome_failure_closure_sha256": file_sha256(
                V4_FAILURE_CLOSURE_PATH
            ),
        },
        "m6e": {
            "P0": int(m6e["P0"]),
            "P1": int(m6e["P1"]),
            "P2": int(m6e["P2"]),
            "conformance_packet_digest": m6e["content_digest"],
            "runtime_release_digest": m6e["training_runtime_release_digest"],
        },
        "m6f": {
            "P0": int(m6f["audit"]["P0"]),
            "P1": int(m6f["audit"]["P1"]),
            "P2": int(m6f["audit"]["P2"]),
            "audit_sha256": m6f["audit"]["sha256"],
            "broker_release_digest": m6f["broker_release"]["release_digest"],
            "verdict": m6f["verdict"],
        },
        "main_eligibility": False,
        "meta": {
            "activation_closure_sha256": file_sha256(
                ACTIVATION_CLOSURE_PATH
            ),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "checkpoint_sha256": file_sha256(CHECKPOINT_PATH),
            "final_evidence_sha256": file_sha256(FINAL_EVIDENCE_PATH),
            "implementation_manifest_sha256": file_sha256(
                IMPLEMENTATION_MANIFEST_PATH
            ),
            "policy_bundle_digest": POLICY_BUNDLE_DIGEST_V17,
            "promotion_decision_digest": PROMOTION_DECISION_DIGEST_V17,
            "promotion_decision_sha256": file_sha256(PROMOTION_PATH),
            "requalification_sha256": file_sha256(REQUALIFICATION_PATH),
            "source_manifest_digest": SOURCE_MANIFEST_DIGEST_V17,
            "task_context": {
                "interaction_count": 1000209,
                "item_count": 3883,
                "raw_density": 1000209 / (6040 * 3883),
                "task_density": (1000209 / (6040 * 3883)) / 0.15,
                "task_scale": 1.0,
                "user_count": 6040,
            },
        },
        "pilot": {
            "experiment_id": store.experiment_id,
            "ordinary_execution_seed": store.ordinary_execution_seed,
            "rounds_per_arm": store.scheduled_slots_per_arm_seed,
            "search_seeds": list(store.search_seeds),
            "store_contract_identity_digest": store.identity_digest,
        },
        "quality_thresholds": {
            "active_meta_route_fraction_min": 0.6,
            "collision_rate_max": 0.5,
            "pool_unique_semantics_min": 4,
            "selected_unique_semantics_min": 3,
        },
        "record_schema": "recclaw.meta-v17-development-pilot-contract.v5",
        "research": {
            "A": {
                "controller": "OriginalControllerV1",
                "meta": "NONE",
            },
            "B": research_identity,
            "C_non_guard": research_identity,
            "evidence_authority_inputs_to_meta": "FORBIDDEN",
        },
        "runtime": predecessor["runtime"],
        "sealed_pilot_seed_registry": sorted(SEALED_PILOT_SEEDS),
        "source": {
            "base_checkpoint_commit": git_value("rev-parse", "HEAD"),
            "base_checkpoint_tree": git_value("rev-parse", "HEAD^{tree}"),
            "files": source_files,
            "source_projection_digest": sha256_digest(source_files),
        },
        "status": "FROZEN_PRE_OUTCOME",
        "training": {
            "execution_purpose": TrainingExecutionPurposeV1.PILOT.value,
            "profile_path": str(TRAINING_PROFILE),
            "profile_sha256": file_sha256(TRAINING_PROFILE),
            "runner_abi": TRAINING_RUNNER_ABI,
            "runtime_release_digest": release.digest,
            "runtime_release_schema": release.record_type,
        },
    }
    contract["content_digest"] = sha256_digest(contract)
    return contract


def main() -> int:
    if ACTIVATION_RECEIPT_PATH.exists() or CONTRACT_PATH.exists():
        raise RuntimeError("refusing to overwrite frozen Meta V17 records")
    receipt = build_activation_receipt()
    ACTIVATION_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVATION_RECEIPT_PATH.write_bytes(canonical_json_bytes(receipt) + b"\n")
    contract = build_contract()
    CONTRACT_PATH.write_bytes(canonical_json_bytes(contract) + b"\n")
    print(
        json.dumps(
            {
                "activation_receipt_sha256": file_sha256(
                    ACTIVATION_RECEIPT_PATH
                ),
                "contract_content_digest": contract["content_digest"],
                "contract_path": str(CONTRACT_PATH),
                "contract_sha256": file_sha256(CONTRACT_PATH),
                "search_seed": META_V17_PILOT_SEARCH_SEED,
                "source_projection_digest": contract["source"][
                    "source_projection_digest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
