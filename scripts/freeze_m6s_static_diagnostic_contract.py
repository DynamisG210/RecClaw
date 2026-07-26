#!/usr/bin/env python3
"""Freeze the one-round development-only static diagnostic Pilot contract."""

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
from run_m6s_static_diagnostic_pilot import (  # noqa: E402
    CONTRACT_PATH,
    SEALED_PILOT_SEEDS,
    STATIC_DIAGNOSTIC_NONCE,
    StaticDiagnosticPilotStoreContractV1,
    static_policy_identity_digest,
)

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
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


V5_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "DEVELOPMENT_PILOT_CONTRACT_V5.json"
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
META_STOP_REPORT_PATH = ROOT / "META_V2_AUTONOMOUS_PROGRAM_STOP_REPORT.md"
V1_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6s"
    / "STATIC_DIAGNOSTIC_PILOT_CONTRACT_V1.json"
)
V1_FAILURE_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6s"
    / "M6S_STATIC_DIAGNOSTIC_V1_FAILURE_RECORD.json"
)
TASK_AUTHORIZATION_PATH = ROOT / "RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md"
MODELS_CACHE = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "pilot_v5_model_catalog_snapshot.json"
)
RESPONSE_SCHEMA = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "pilot_proposal_response_v1.schema.json"
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

ADDITIONAL_SOURCE_FILES = {
    "scripts/freeze_m6s_static_diagnostic_contract.py",
    "scripts/run_m6s_static_diagnostic_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/audit_snapshot.py",
    "src/recclaw_core/experiments/helix_abc_v1/broker_failure_closure.py",
    "src/recclaw_core/experiments/helix_abc_v1/broker_process.py",
    "src/recclaw_core/experiments/helix_abc_v1/research_capability.py",
    "src/recclaw_core/experiments/helix_abc_v1/research_contracts.py",
    "src/recclaw_core/experiments/helix_abc_v1/resources/broker_process_release_v2.json",
    "tests/experiments/helix_abc_v1/test_m6s_static_diagnostic.py",
}


def build_contract() -> dict[str, Any]:
    predecessor = json.loads(V5_CONTRACT_PATH.read_text(encoding="utf-8"))
    m6e = json.loads(M6E_PACKET_PATH.read_text(encoding="utf-8"))
    m6f = json.loads(M6F_RECORD_PATH.read_text(encoding="utf-8"))
    if m6e["verdict"] != "PASS" or m6e["P0"] != 0 or m6e["P1"] != 0:
        raise RuntimeError("refusing to freeze static diagnostic before M6E PASS")
    if (
        m6f["verdict"] != "PASS"
        or m6f["audit"]["P0"] != 0
        or m6f["audit"]["P1"] != 0
    ):
        raise RuntimeError("refusing to freeze static diagnostic before M6F PASS")

    store = StaticDiagnosticPilotStoreContractV1.create()
    if tuple(sorted(SEALED_PILOT_SEEDS)) != (
        9201,
        9202,
        9203,
        9204,
        9205,
        9206,
    ):
        raise RuntimeError("sealed Pilot seed registry changed")
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id, nonce=STATIC_DIAGNOSTIC_NONCE
    )
    release = training_runtime_release()
    source_paths = set(predecessor["source"]["files"]) | ADDITIONAL_SOURCE_FILES
    source_files = {
        relative: file_sha256(ROOT / relative)
        for relative in sorted(source_paths)
    }
    broker_release = json.loads(
        (
            ROOT
            / "src"
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "resources"
            / "broker_process_release_v2.json"
        ).read_text(encoding="utf-8")
    )
    model_catalog = json.loads(MODELS_CACHE.read_text(encoding="utf-8"))

    contract = {
        "analysis": {
            "effect_interpretation": "DESCRIPTIVE_SINGLE_ROUND_ONLY",
            "formal_inference": False,
            "four_axes": [
                "ROUND",
                "EXECUTION_COUNT",
                "BILLED_TOKEN",
                "GPU_NORMALIZED_COST",
            ],
            "purpose": "NON_META_RESEARCH_LINE_CHAIN_AND_SIGNAL_DIAGNOSTIC",
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
            "expected_upstream_calls": 5,
            "first_round_equal_memory_replay": True,
            "login_mode": broker_release["login_mode"],
            "max_total_tokens_per_call": 20000,
            "model": broker_release["model"],
            "models_cache_etag": model_catalog["etag"],
            "models_cache_mode": (
                "FROZEN_SELECTED_MODEL_CATALOG_PROJECTION_V1"
            ),
            "models_cache_path": str(MODELS_CACHE),
            "models_cache_sha256": file_sha256(MODELS_CACHE),
            "reasoning_effort": broker_release["reasoning_effort"],
            "response_schema_path": str(RESPONSE_SCHEMA),
            "response_schema_sha256": file_sha256(RESPONSE_SCHEMA),
            "retries": 0,
            "sandbox": broker_release["sandbox_mode"],
            "service_tier": "default",
        },
        "budget_per_arm_round": pilot_budget().to_dict(),
        "common_runtime_authorization": {
            "path": str(TASK_AUTHORIZATION_PATH),
            "sha256": file_sha256(TASK_AUTHORIZATION_PATH),
        },
        "dataset": predecessor["dataset"],
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
        "guard_and_fusion": predecessor["guard_and_fusion"],
        "lineage_policy": {
            "formal_meta_required_for_main": True,
            "main_memory_imported": False,
            "meta_v2_decision_rewritten": False,
            "pilot_outputs_reusable_in_main": False,
            "static_diagnostic_memory_imported_into_meta": False,
        },
        "lineage_predecessors": {
            "m6f_audit_sha256": file_sha256(M6F_AUDIT_PATH),
            "m6f_execution_record_sha256": file_sha256(M6F_RECORD_PATH),
            "meta_v2_stop_report_sha256": file_sha256(META_STOP_REPORT_PATH),
            "pilot_v5_contract_content_digest": predecessor["content_digest"],
            "pilot_v5_contract_sha256": file_sha256(V5_CONTRACT_PATH),
            "static_diagnostic_v1_contract_sha256": file_sha256(
                V1_CONTRACT_PATH
            ),
            "static_diagnostic_v1_failure_sha256": file_sha256(
                V1_FAILURE_PATH
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
        "pilot": {
            "experiment_id": store.experiment_id,
            "ordinary_execution_seed": store.ordinary_execution_seed,
            "rounds_per_arm": store.scheduled_slots_per_arm_seed,
            "search_seeds": list(store.search_seeds),
            "store_contract_identity_digest": store.identity_digest,
        },
        "record_schema": "recclaw.static-diagnostic-pilot-contract.v1",
        "research": {
            "agentization_gate": "PASS_INDEPENDENT_MULTI_AGENT",
            "formal_meta_required_for_main": True,
            "meta_activation": "NONE",
            "meta_mode": "STATIC_RESEARCH_ROUTER",
            "policy_label": "RESEARCH_STATIC_V2",
            "producer_mode": "BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1",
            "static_policy_identity_digest": static_policy_identity_digest(),
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
    if CONTRACT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite frozen contract: {CONTRACT_PATH}")
    contract = build_contract()
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_bytes(canonical_json_bytes(contract) + b"\n")
    print(
        json.dumps(
            {
                "content_digest": contract["content_digest"],
                "contract_path": str(CONTRACT_PATH),
                "contract_sha256": file_sha256(CONTRACT_PATH),
                "search_seed": contract["pilot"]["search_seeds"][0],
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
