#!/usr/bin/env python3
"""Freeze or verify the V13 Pilot contract without starting the Pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from recclaw_core.experiments.helix_abc_v1.campaign_dataset import (  # noqa: E402
    inspect_materialized_campaign_dataset,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v13 import (  # noqa: E402
    V13_PILOT_ASSIGNMENT_NONCE,
    V13_PILOT_ROUNDS_PER_ARM,
    V13_PILOT_SEARCH_SEED,
    V13PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (  # noqa: E402
    campaign_projection,
    campaign_runtime_profile,
    campaign_training_profile,
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    LabApiBrokerReleaseV1,
    load_lab_api_credentials,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    CHECKPOINT_SHA256_V18,
    POLICY_BUNDLE_DIGEST_V18,
    PROMOTION_DECISION_DIGEST_V18,
    MetaV18CampaignRuntimeV1,
)
from recclaw_core.experiments.helix_abc_v1.original_main import (  # noqa: E402
    ORIGINAL_MAIN_COMMIT,
    OriginalMainSourceReleaseV1,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    PrivateTreatmentAssignmentV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    pilot_budget,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    CAMPAIGN_TRAINING_RUNNER_ABI,
    campaign_training_runtime_release,
    validate_campaign_training_runtime_release,
)


RESOURCE_ROOT = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "research_line"
    / "v13_requalification"
    / "V13_FROZEN_PILOT_CONTRACT.json"
)
DEFAULT_LLM_CONFIG = Path(
    "/root/projects/RecClaw_v2_0_Final_Reference/llm_api.md"
)
DEFAULT_OUTPUT_ROOT = Path("/root/projects/RecClaw_campaign_pilot_9215_v13")
DEFAULT_SOURCE_DATASET = Path("/root/projects/RecBole/dataset/ml-1m")
DEFAULT_SEARCH_PARENT = Path("/root/projects/RecClaw_campaign_dataset_v1/search")
DEFAULT_HELDOUT_PARENT = Path(
    "/root/projects/RecClaw_campaign_dataset_v1/heldout"
)
DEFAULT_PYTHON = Path("/root/projects/RecClaw_m6_training_runtime_v2/bin/python")
DEFAULT_RECBOLE = Path("/root/projects/RecBole_m6_runtime")
G7_GATE = (
    ROOT
    / "docs"
    / "research_line"
    / "v13_requalification"
    / "V13_SCIENTIFIC_ATTRIBUTION_GATE.json"
)
META_CHECKPOINT = RESOURCE_ROOT / "meta_vnext_policy_checkpoint_v18.json"
META_PROMOTION = (
    ROOT
    / "docs"
    / "research_line"
    / "v13_requalification"
    / "META_VNEXT_PROMOTION_DECISION_V18.json"
)
BROKER_RELEASE = (
    RESOURCE_ROOT / "lab_api_broker_release_v1_v13_schema_v4.json"
)
PROPOSAL_SCHEMA = RESOURCE_ROOT / "campaign_proposal_response_v1.schema.json"
TRAINING_RELEASE = RESOURCE_ROOT / "training_runtime_release_v5.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(argument: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", argument],
        text=True,
    ).strip()


def _source_files() -> tuple[str, ...]:
    return (
        "recclaw_ext/models/composable_v2.py",
        "scripts/campaign_train_worker.py",
        "scripts/freeze_campaign_training_runtime_release_v5.py",
        "scripts/freeze_v13_pilot_contract.py",
        "scripts/run_v13_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/audit_snapshot.py",
        "src/recclaw_core/experiments/helix_abc_v1/broker_failure_closure.py",
        "src/recclaw_core/experiments/helix_abc_v1/broker_process.py",
        "src/recclaw_core/experiments/helix_abc_v1/campaign_dataset.py",
        "src/recclaw_core/experiments/helix_abc_v1/campaign_pilot_v13.py",
        "src/recclaw_core/experiments/helix_abc_v1/campaign_runtime.py",
        "src/recclaw_core/experiments/helix_abc_v1/canary_broker.py",
        "src/recclaw_core/experiments/helix_abc_v1/common_execution_guard.py",
        "src/recclaw_core/experiments/helix_abc_v1/contracts.py",
        "src/recclaw_core/experiments/helix_abc_v1/controllers.py",
        "src/recclaw_core/experiments/helix_abc_v1/lab_api_broker.py",
        "src/recclaw_core/experiments/helix_abc_v1/materialization.py",
        "src/recclaw_core/experiments/helix_abc_v1/m6e_conformance.py",
        "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_campaign.py",
        "src/recclaw_core/experiments/helix_abc_v1/meta_vnext_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/original_main.py",
        "src/recclaw_core/experiments/helix_abc_v1/pilot_analysis.py",
        "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py",
        "src/recclaw_core/experiments/helix_abc_v1/precanary_orchestration.py",
        "src/recclaw_core/experiments/helix_abc_v1/real_canary.py",
        "src/recclaw_core/experiments/helix_abc_v1/real_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/research_capability.py",
        "src/recclaw_core/experiments/helix_abc_v1/research_contracts.py",
        "src/recclaw_core/experiments/helix_abc_v1/research_controller.py",
        "src/recclaw_core/experiments/helix_abc_v1/research_science.py",
        "src/recclaw_core/experiments/helix_abc_v1/runtime_contracts.py",
        "src/recclaw_core/experiments/helix_abc_v1/runtime_release.py",
        "src/recclaw_core/experiments/helix_abc_v1/scientific_attribution_gate.py",
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
        "src/recclaw_core/helix/guard_adapter.py",
        "src/recclaw_core/helix/ledger.py",
        "src/recclaw_core/helix/ports.py",
        "src/recclaw_core/helix/scientific_attribution.py",
        "src/recclaw_evidence_guard/core_v1.py",
    )


def _dataset_projection() -> dict[str, Any]:
    audit = inspect_materialized_campaign_dataset(
        source_dataset_root=DEFAULT_SOURCE_DATASET,
        search_parent_root=DEFAULT_SEARCH_PARENT,
        heldout_parent_root=DEFAULT_HELDOUT_PARENT,
    )
    if not all(
        (
            audit["zero_overlap"],
            audit["all_source_rows_assigned_once"],
            audit["interaction_bytes_exact"],
            audit["metadata_bytes_exact"],
        )
    ):
        raise RuntimeError("V13 dataset materialization is not exact")
    search = DEFAULT_SEARCH_PARENT / "ml-1m"
    heldout = DEFAULT_HELDOUT_PARENT / "ml-1m"
    return {
        "counts": {
            "development_validation": audit["counts"]["dev"],
            "heldout_test": audit["counts"]["heldout"],
            "train": audit["counts"]["train"],
        },
        "heldout_dataset": heldout.as_posix(),
        "heldout_files": {
            "ml-1m.heldout.inter": file_sha256(
                heldout / "ml-1m.heldout.inter"
            )
        },
        "identity": "ML-1M",
        "manifest_digest": audit["manifest_digest"],
        "online_heldout_access": False,
        "search_dataset": search.as_posix(),
        "search_files": {
            name: file_sha256(search / name)
            for name in (
                "ml-1m.dev.inter",
                "ml-1m.item",
                "ml-1m.train.inter",
                "ml-1m.user",
            )
        },
        "search_parent": DEFAULT_SEARCH_PARENT.as_posix(),
        "source_dataset": DEFAULT_SOURCE_DATASET.as_posix(),
    }


def build_contract(
    *,
    llm_api_config: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    store_contract_factory: Any = V13PilotStoreContractV1.create,
    assignment_nonce: str = V13_PILOT_ASSIGNMENT_NONCE,
    search_seed: int = V13_PILOT_SEARCH_SEED,
    rounds_per_arm: int = V13_PILOT_ROUNDS_PER_ARM,
    meta_runtime_class: Any = MetaV18CampaignRuntimeV1,
    meta_checkpoint: Path = META_CHECKPOINT,
    meta_promotion: Path = META_PROMOTION,
    meta_checkpoint_sha256: str = CHECKPOINT_SHA256_V18,
    meta_policy_bundle_digest: str = POLICY_BUNDLE_DIGEST_V18,
    meta_promotion_digest: str = PROMOTION_DECISION_DIGEST_V18,
    broker_release_path: Path = BROKER_RELEASE,
    training_release_path: Path = TRAINING_RELEASE,
    scientific_gate_path: Path = G7_GATE,
    record_schema: str = "recclaw.v13-pilot-contract.v1",
    source_files: tuple[str, ...] | None = None,
    training_python: Path = DEFAULT_PYTHON,
    training_recbole_root: Path = DEFAULT_RECBOLE,
    training_data_path: Path = DEFAULT_SEARCH_PARENT,
    dataset_projection: dict[str, Any] | None = None,
    git_head_at_freeze: str | None = None,
) -> dict[str, Any]:
    version_label = record_schema.split(".")[1].upper()
    if output_root.exists():
        raise RuntimeError(f"{version_label} output root already exists")
    if any(
        (ROOT / "results" / "research_line").glob(f"*{search_seed}*")
    ):
        raise RuntimeError(
            f"{version_label} search seed already has result artifacts"
        )
    gate = json.loads(scientific_gate_path.read_text(encoding="utf-8"))
    if (
        gate["verdict"] != "PASS"
        or gate["p0"] != 0
        or gate["p1"] != 0
    ):
        raise RuntimeError("G7 Scientific Attribution Gate is not PASS")
    base_url, _api_key = load_lab_api_credentials(llm_api_config)
    broker_release = LabApiBrokerReleaseV1(
        **json.loads(broker_release_path.read_text(encoding="utf-8"))
    )
    broker_release.verify()
    expected_broker = LabApiBrokerReleaseV1.create(
        base_url=base_url,
        model="gpt-5.4",
        response_schema_digest=file_sha256(PROPOSAL_SCHEMA),
        max_total_tokens_per_call=20_000,
        timeout_ms=900_000,
    )
    if broker_release != expected_broker:
        raise RuntimeError(
            f"{version_label} laboratory API release is not exact"
        )
    runtime_failures = validate_campaign_training_runtime_release(
        data_path=training_data_path,
        python_executable=training_python,
        recbole_root=training_recbole_root,
    )
    if runtime_failures:
        raise RuntimeError(
            f"{version_label} training runtime is not qualified: "
            + ",".join(runtime_failures)
        )
    store = store_contract_factory()
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id,
        nonce=assignment_nonce,
    )
    profile = campaign_runtime_profile()
    projection = campaign_projection()
    release = campaign_training_runtime_release()
    original_release = OriginalMainSourceReleaseV1(
        repository_root=ROOT,
        materialization_root=ROOT / ".v13-identity-only-not-materialized",
    )
    meta_runtime = meta_runtime_class(
        checkpoint_path=meta_checkpoint,
        experiment_id=store.experiment_id,
        search_seed=search_seed,
        scheduled_rounds=rounds_per_arm,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    schedule = [
        {
            "A": (
                "ORIGINAL_REFRESH_4"
                if round_index in {1, 4}
                else "ORIGINAL_CACHED_SLATE"
            ),
            "B": [
                "mechanism_composer",
                "lineage_refiner",
                "falsification_designer",
                "frontier_architect",
            ],
            "C": [
                "mechanism_composer",
                "lineage_refiner",
                "falsification_designer",
                "frontier_architect",
            ],
            "ordinary_execution_opportunity_per_arm": 1,
            "round_index": round_index,
        }
        for round_index in range(1, rounds_per_arm + 1)
    ]
    source_files = {
        relative: file_sha256(ROOT / relative)
        for relative in (source_files or _source_files())
    }
    payload = {
        "record_schema": record_schema,
        "status": "FROZEN_PRE_OUTCOME",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
        "main_eligibility": False,
        "pilot_started": False,
        "output_root": output_root.as_posix(),
        "pilot": {
            "experiment_id": store.experiment_id,
            "ordinary_execution_seed": store.ordinary_execution_seed,
            "rounds_per_arm": store.scheduled_slots_per_arm_seed,
            "search_seeds": list(store.search_seeds),
            "store_contract_identity_digest": store.identity_digest,
        },
        "assignment": {
            "commitment": assignment.commitment,
            "opaque": True,
        },
        "three_arm_estimands": {
            "A": "PINNED_MAIN_ORIGINAL_PLUS_COMMON_BL_ICF_V2",
            "B": "RESEARCH_CAPABILITY_V13_PLUS_COMMON_BL_ICF_V2",
            "C": "B_PLUS_EVIDENCE_GUARD",
            "primary": ["B_MINUS_A", "C_MINUS_B"],
            "secondary": ["C_MINUS_A"],
        },
        "arm_policies": [item.to_dict() for item in store.arm_policies],
        "original": {
            "adapter": "PinnedOriginalMainAdapterV1",
            "main_commit": ORIGINAL_MAIN_COMMIT,
            "source_release_digest": original_release.identity_digest,
            "working_copy_scripts_agent_used": False,
        },
        "common_substrate": {
            "executable_mechanism_count": len(executable_mechanisms()),
            "executable_profile_digest": profile[
                "executable_profile_digest"
            ],
            "pilot_runtime_profile_digest": profile["profile_digest"],
            "main_runtime_profile_digest": profile["profile_digest"],
            "pilot_main_profile_identity_equal": True,
            "projection_digest": projection["projection_digest"],
            "search_space_id": projection["search_space_id"],
        },
        "candidate_schedule": {
            "guard_extra_proposals": 0,
            "guard_pre_fallback": "NEXT_FROM_ROUTER_FROZEN_SAME_SLATE",
            "schedule": schedule,
            "schedule_digest": sha256_digest(schedule),
        },
        "budget_per_arm_round": pilot_budget().to_dict(),
        "budget_semantics": {
            "max_ordinary_training_executions": 1,
            "one_search_round": (
                "ONE_NORMAL_CANDIDATE_EXECUTION_OPPORTUNITY_PLUS_FEEDBACK"
            ),
            "required_frontier_axes": [
                "ROUND",
                "EXECUTION_COUNT",
                "BILLED_TOKEN",
                "GPU_NORMALIZED_COST",
            ],
            "retries": 0,
        },
        "dataset": dataset_projection or _dataset_projection(),
        "meta": {
            "activation_boundary": "NEXT_CAMPAIGN",
            "checkpoint_path": meta_checkpoint.as_posix(),
            "checkpoint_sha256": file_sha256(meta_checkpoint),
            "expected_checkpoint_sha256": meta_checkpoint_sha256,
            "policy_bundle_digest": meta_policy_bundle_digest,
            "promotion_decision_digest": meta_promotion_digest,
            "promotion_record_path": meta_promotion.as_posix(),
            "promotion_record_sha256": file_sha256(meta_promotion),
            "control_policy_digest": meta_runtime.control_policy.digest,
            "fast_residual_use": "ONLY_WHEN_ACTUAL_TASK_CONTEXT_SUPPORTED",
            "task_context": {
                "task_density": 0.2843119865332499,
                "task_scale": 1.0,
            },
            "task_support_projection": meta_runtime.task_support_projection(),
        },
        "guard_and_fusion": {
            "b_port": "NullEvidencePortV1",
            "c_port": "EvidenceGuardPortV1",
            "admission": "DeterministicHelixAdmissionV13",
            "gate_path": scientific_gate_path.relative_to(ROOT).as_posix(),
            "gate_sha256": file_sha256(scientific_gate_path),
            "gate_result_digest": gate["gate_result_digest"],
            "guard_controls_search_memory_meta_frontier": True,
        },
        "broker": {
            **broker_release.to_dict(),
            "release_manifest_path": broker_release_path.as_posix(),
            "release_manifest_sha256": file_sha256(broker_release_path),
            "response_schema_path": PROPOSAL_SCHEMA.as_posix(),
            "response_schema_sha256": file_sha256(PROPOSAL_SCHEMA),
            "credential_source_external": True,
            "credential_persisted": False,
        },
        "training": {
            "release_digest": release.digest,
            "release_id": release.release_id,
            "release_manifest_path": training_release_path.as_posix(),
            "release_manifest_sha256": file_sha256(training_release_path),
            "runner_abi": CAMPAIGN_TRAINING_RUNNER_ABI,
            "profile": campaign_training_profile(),
            "python": training_python.as_posix(),
            "recbole_root": training_recbole_root.as_posix(),
        },
        "analysis": {
            "frontier_projections": [
                "OBSERVED",
                "SEARCH_ELIGIBLE",
                "CONFIRMED",
            ],
            "confirmed_requires_frozen_post_selection_evaluator": True,
            "pilot_computes_treatment_effect": False,
            "intent_to_treat_round_rows": True,
            "no_execution_rows_preserved": True,
        },
        "source": {
            "files": source_files,
            "manifest_digest": sha256_digest(source_files),
            "git_head_at_freeze": (
                git_head_at_freeze or _git_value("HEAD")
            ),
            "selected_source_manifest_is_authoritative": True,
        },
    }
    return {**payload, "content_digest": sha256_digest(payload)}


def verify_v13_pilot_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    preimage = dict(contract)
    expected = preimage.pop("content_digest")
    if sha256_digest(preimage) != expected:
        raise RuntimeError("V13 contract content digest mismatch")
    if (
        contract["record_schema"] != "recclaw.v13-pilot-contract.v1"
        or contract["status"] != "FROZEN_PRE_OUTCOME"
        or contract["pilot_started"] is not False
        or contract["broker"]["model"] != "gpt-5.4"
        or contract["common_substrate"]["executable_mechanism_count"] != 66
        or not contract["common_substrate"][
            "pilot_main_profile_identity_equal"
        ]
        or contract["meta"]["policy_bundle_digest"]
        != POLICY_BUNDLE_DIGEST_V18
        or contract["guard_and_fusion"]["gate_result_digest"]
        != json.loads(G7_GATE.read_text(encoding="utf-8"))[
            "gate_result_digest"
        ]
    ):
        raise RuntimeError("V13 contract invariant mismatch")
    expected_store = V13PilotStoreContractV1.create()
    if contract["pilot"] != {
        "experiment_id": expected_store.experiment_id,
        "ordinary_execution_seed": expected_store.ordinary_execution_seed,
        "rounds_per_arm": expected_store.scheduled_slots_per_arm_seed,
        "search_seeds": list(expected_store.search_seeds),
        "store_contract_identity_digest": expected_store.identity_digest,
    }:
        raise RuntimeError("V13 store contract identity mismatch")
    for relative, digest in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"V13 source identity mismatch: {relative}")
    exact = {
        Path(contract["broker"]["release_manifest_path"]): contract[
            "broker"
        ]["release_manifest_sha256"],
        Path(contract["broker"]["response_schema_path"]): contract[
            "broker"
        ]["response_schema_sha256"],
        Path(contract["meta"]["checkpoint_path"]): contract["meta"][
            "checkpoint_sha256"
        ],
        Path(contract["training"]["release_manifest_path"]): contract[
            "training"
        ]["release_manifest_sha256"],
    }
    for artifact, digest in exact.items():
        if file_sha256(artifact) != digest:
            raise RuntimeError(f"V13 release identity mismatch: {artifact}")
    if Path(contract["output_root"]).exists():
        raise RuntimeError("V13 Pilot output root is no longer fresh")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--llm-api-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG,
    )
    args = parser.parse_args()
    if args.verify is not None:
        contract = verify_v13_pilot_contract(args.verify.resolve())
        print(
            json.dumps(
                {
                    "content_digest": contract["content_digest"],
                    "pilot_started": contract["pilot_started"],
                    "verdict": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0
    contract = build_contract(llm_api_config=args.llm_api_config.resolve())
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("V13 frozen contract already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(contract) + b"\n")
    print(
        json.dumps(
            {
                "content_digest": contract["content_digest"],
                "path": output.as_posix(),
                "sha256": file_sha256(output),
                "verdict": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
