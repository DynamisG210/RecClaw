"""Q0 common-mode calibration and prior negative-Episode audit.

This module is deliberately DEVELOPMENT_ONLY.  It reuses the accepted
Producer/OpenSpec/Resolver, shared origin-blind Implementer, materializer,
Mechanical Qualifier, and matched RecBole runtime.  The parent-equivalent and
known-good arms are diagnostic instruments, never capability admissions.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import bytes_sha256, canonical_value, sha256_digest
from .campaign_runtime import executable_mechanisms
from .fresh_f1 import (
    F1_RECOVERY_ROOT,
    F1_ROOT,
    F1_V1_EXTERNAL_RECEIPT_SHA256,
    R2_EXTERNAL_RECEIPT_SHA256,
    R2_EXTERNAL_ROOT,
)
from .fresh_r1 import (
    AVAILABLE_DEPENDENCIES,
    BUDGET_LIMITS,
    IMPLEMENTATION_TOKEN_CEILING,
    MODEL,
    PROTOCOL_REQUIREMENTS,
    PROJECTS_ROOT,
    PYTHON_EXECUTABLE,
    RECBole_ROOT,
    SEARCH_DATASET_ROOT,
    EXPECTED_SEARCH_FILES,
    _git,
    _materialize_and_qualify,
    _physical_usage,
    _read_json,
    _shared_behavioral_unit_check,
    _shared_policy,
    _write_new_json,
    bounded_provider_call,
    render_implementation_prompt,
    run_development_training,
)
from .fresh_r2 import (
    R1_EXTERNAL_RECEIPT_SHA256,
    R1_EXTERNAL_ROOT,
    R1_REPO_RECEIPT_SHA256,
    _r2_bindings,
    _r2_environment,
    build_active_r2_profile,
    build_r1_registry,
    derive_fresh_r2_proposal_schema,
    load_registered_r1_artifacts,
    public_active_profile_catalog,
)
from .innovation_recbole_adapter import snapshot_candidate_tree
from .innovation_spine import build_shared_implementer_request
from .open_meta_f1 import build_f1_replay_dataset
from .open_spec import project_open_producer_draft, resolve_capability
from .v4_response_contract import validate_v4_response_contract
from .vnext_contracts import QualificationStatusV1


ACCEPTED_F1_COMMIT = "0041d1cd4dafb3e1a1e2aced97c6a4988db72fc8"
ACCEPTED_F1_PARENT = "3505885c69737064ba5bd59a5aa8c96e5309d15d"
ACCEPTED_F1_TREE = "a9395474954743ab88e3604156cdc2a812f99258"
Q0_BRANCH = "feat/research-line-quality-calibration"
Q0_RUN_IDENTITY = "q0-common-mode-calibration-v1"
Q0_CAMPAIGN_ID = "recclaw-q0-common-mode-calibration-v1"
Q0_ROOT = Path(
    os.environ.get(
        "RECCLAW_Q0_ROOT",
        "/root/projects/RecClaw_quality_calibration_runs/q0_common_mode_v1",
    )
)
Q0_CONTEXT_REF = "recclaw-q0-common-mode-calibration-context-v1"
Q0_CONTEXT_DIGEST = sha256_digest(
    {
        "accepted_f1_commit": ACCEPTED_F1_COMMIT,
        "campaign_id": Q0_CAMPAIGN_ID,
        "purpose": "DEVELOPMENT_ONLY_COMMON_MODE_CALIBRATION",
    }
)
Q0_TRAINING_SEED = 54102
Q0_QUALIFICATION_SEEDS = {
    "parent_equivalent_null": 54111,
    "known_good_reference": 54112,
    "frontier_candidate": 54113,
}
Q0_PROPOSAL_SEEDS = {
    "parent_equivalent_null": 54011,
    "known_good_reference": 54012,
    "frontier_candidate": 54013,
}
Q0_TIMEOUT_SECONDS = 1500
NULL_NDCG_ABS_TOLERANCE = 0.002
REFERENCE_MIN_NDCG = 0.10
REFERENCE_MAX_RUNTIME_RATIO = 3.0

F1_REPO_RECEIPT_SHA256 = (
    "f160436a9df1bb8d849849f6cb61e367dd54c62d4b5fcc68101a21c86a72b4b3"
)
F1_RECOVERY_REPO_RECEIPT_SHA256 = (
    "59d576b0bbc48fcef17d601acdd53c2205df8f775d5f58a01f5d7314ac8d27fe"
)
F1_RECOVERY_EXTERNAL_RECEIPT_SHA256 = (
    "fc4baf9e92b0a95669099ecf480be57274a6ad80abda394e7674b64315458d2b"
)
R2_REPO_RECEIPT_SHA256 = (
    "5ed2ebddc5f0724fd63849cdac9b3155f039192f2002b895d73d0c7cee665393"
)

KNOWN_GOOD_SOURCE = (
    PROJECTS_ROOT
    / "RecClaw_static_pilot/recclaw_ext/models/lightgcn_residual_norm.py"
)
KNOWN_GOOD_SOURCE_SHA256 = (
    "0b5bfe936d1bf306887467a96f0b5dd8f3e6d5ea3fb382f49aaba4a12b198a21"
)
KNOWN_GOOD_CANARIES = (
    (
        PROJECTS_ROOT
        / "RecClaw_static_pilot/results/research_line/"
        "campaign_readiness_canary_residual_norm_v5_final/"
        "CAMPAIGN_READINESS_RESIDUAL_NORM_V5_FINAL_RESULT.json",
        "ea21adebf30b6823e6699334941454c1862d4701248417a0a701899379f5ecf1",
        9317,
    ),
    (
        PROJECTS_ROOT
        / "RecClaw_static_pilot/results/research_line/"
        "campaign_readiness_canary_residual_norm_v6_final/"
        "CAMPAIGN_READINESS_RESIDUAL_NORM_V6_FINAL_RESULT.json",
        "1b2c6ace539bc36d037e62dadb4b4f3bf79713a975484a7b979913a8b26f1f29",
        9318,
    ),
)


class QualityCalibrationError(RuntimeError):
    """Q0 identity, contract, or orchestration failure."""


def _resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def _not_observed(*, reason: str, scope: str = "FULL_TRAINING") -> dict[str, str]:
    return {"status": "NOT_OBSERVED", "reason": reason, "scope": scope}


def _sealed_paths(repo_root: Path) -> dict[str, tuple[Path, str]]:
    return {
        "r1_repo": (
            repo_root
            / "docs/research_line/vnext/"
            "R1_FRESH_TRAINING_FILESYSTEM_FIX_V3_CANONICAL_RECEIPT.json",
            R1_REPO_RECEIPT_SHA256,
        ),
        "r1_external": (
            R1_EXTERNAL_ROOT / "R1_CANONICAL_RECEIPT.json",
            R1_EXTERNAL_RECEIPT_SHA256,
        ),
        "r2_repo": (
            repo_root
            / "docs/research_line/vnext/"
            "R2_FRESH_REGISTRY_EFFECT_CANONICAL_RECEIPT.json",
            R2_REPO_RECEIPT_SHA256,
        ),
        "r2_external": (
            R2_EXTERNAL_ROOT / "R2_CANONICAL_RECEIPT.json",
            R2_EXTERNAL_RECEIPT_SHA256,
        ),
        "f1_repo": (
            repo_root
            / "docs/research_line/vnext/F1_OPEN_META_CANONICAL_RECEIPT.json",
            F1_REPO_RECEIPT_SHA256,
        ),
        "f1_external": (
            F1_ROOT / "F1_CANONICAL_RECEIPT.json",
            F1_V1_EXTERNAL_RECEIPT_SHA256,
        ),
        "f1_recovery_repo": (
            repo_root
            / "docs/research_line/vnext/"
            "F1_OPEN_META_RUNTIME_RECOVERY_V2_CANONICAL_RECEIPT.json",
            F1_RECOVERY_REPO_RECEIPT_SHA256,
        ),
        "f1_recovery_external": (
            F1_RECOVERY_ROOT / "F1_RUNTIME_RECOVERY_CANONICAL_RECEIPT.json",
            F1_RECOVERY_EXTERNAL_RECEIPT_SHA256,
        ),
    }


def verify_q0_source_identity(
    repo_root: Path,
    *,
    require_fresh_root: bool,
) -> dict[str, Any]:
    """Verify the accepted F1 parent and immutable evidence inputs."""

    observed = {
        "branch": _git(repo_root, "branch", "--show-current"),
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "parent": _git(repo_root, "rev-parse", "HEAD^"),
        "head_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "python_sha256": bytes_sha256(PYTHON_EXECUTABLE.read_bytes()),
        "recbole_commit": _git(RECBole_ROOT, "rev-parse", "HEAD"),
        "recbole_tree": _git(RECBole_ROOT, "rev-parse", "HEAD^{tree}"),
    }
    expected = {
        "branch": Q0_BRANCH,
        "head": ACCEPTED_F1_COMMIT,
        "parent": ACCEPTED_F1_PARENT,
        "head_tree": ACCEPTED_F1_TREE,
        "recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
        "recbole_tree": "ca6386c4121ce2aae478ced7e136894ac1d7c218",
    }
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise QualityCalibrationError(
            "Q0 source identity mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    sealed_hashes: dict[str, str] = {}
    for name, (path, digest) in _sealed_paths(repo_root).items():
        observed_digest = bytes_sha256(path.read_bytes())
        if observed_digest != digest:
            raise QualityCalibrationError(f"sealed evidence byte drift: {name}")
        sealed_hashes[name] = observed_digest
    for name, digest in EXPECTED_SEARCH_FILES.items():
        if bytes_sha256((SEARCH_DATASET_ROOT / name).read_bytes()) != digest:
            raise QualityCalibrationError(f"search partition identity drift: {name}")
    canaries: list[dict[str, Any]] = []
    if bytes_sha256(KNOWN_GOOD_SOURCE.read_bytes()) != KNOWN_GOOD_SOURCE_SHA256:
        raise QualityCalibrationError("known-good source byte identity drift")
    for path, digest, seed in KNOWN_GOOD_CANARIES:
        if bytes_sha256(path.read_bytes()) != digest:
            raise QualityCalibrationError("known-good canary byte identity drift")
        canary = _read_json(path)
        if (
            canary.get("verdict") != "PASS"
            or canary.get("evidence_class") != "DEVELOPMENT_ONLY"
            or canary.get("formal_acceptance") is not False
            or canary.get("campaign_mechanism_id")
            != "LIGHTGCN_RESIDUAL_NORM"
            or canary.get("search_seed") != seed
            or canary.get("normalized_metrics", {}).get("ndcg@10") != 0.1108
        ):
            raise QualityCalibrationError("known-good canary contract drift")
        canaries.append(
            {
                "path": str(path),
                "sha256": digest,
                "search_seed": seed,
                "development_ndcg_at_10": 0.1108,
            }
        )
    if require_fresh_root and Q0_ROOT.exists():
        raise QualityCalibrationError(f"Q0 root already exists: {Q0_ROOT}")
    return canonical_value(
        {
            **observed,
            "accepted_parent_identity": expected,
            "sealed_receipt_hashes": sealed_hashes,
            "search_partition_files": EXPECTED_SEARCH_FILES,
            "known_good_source_sha256": KNOWN_GOOD_SOURCE_SHA256,
            "known_good_canaries": canaries,
            "initial_parent_worktree_clean_verified_before_branch_creation": True,
            "held_out_reads": 0,
        }
    )


def _validated_delegated_source_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the locally captured identity used by a relocated runtime."""

    identity = canonical_value(dict(value))
    expected = {
        "branch": Q0_BRANCH,
        "head": ACCEPTED_F1_COMMIT,
        "parent": ACCEPTED_F1_PARENT,
        "head_tree": ACCEPTED_F1_TREE,
        "recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
        "recbole_tree": "ca6386c4121ce2aae478ced7e136894ac1d7c218",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": identity.get(key)}
        for key, expected_value in expected.items()
        if identity.get(key) != expected_value
    }
    expected_sealed = {
        name: digest for name, (_path, digest) in _sealed_paths(Path(".")).items()
    }
    if identity.get("sealed_receipt_hashes") != expected_sealed:
        mismatches["sealed_receipt_hashes"] = "DRIFT"
    if identity.get("search_partition_files") != EXPECTED_SEARCH_FILES:
        mismatches["search_partition_files"] = "DRIFT"
    if identity.get("known_good_source_sha256") != KNOWN_GOOD_SOURCE_SHA256:
        mismatches["known_good_source_sha256"] = "DRIFT"
    if identity.get("held_out_reads") != 0:
        mismatches["held_out_reads"] = identity.get("held_out_reads")
    if mismatches:
        raise QualityCalibrationError(
            "delegated Q0 source identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return canonical_value(
        {
            **identity,
            "execution_relocation": "PREVALIDATED_LOCAL_IDENTITY_REMOTE_RUNTIME",
        }
    )


def _validated_runtime_environment_identity(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {"status": "LOCAL_RUNTIME"}
    identity = canonical_value(dict(value))
    mismatches: dict[str, Any] = {}
    for key, expected in {
        "recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
        "recbole_tree": "ca6386c4121ce2aae478ced7e136894ac1d7c218",
        "search_partition_files": EXPECTED_SEARCH_FILES,
        "held_out_reads": 0,
    }.items():
        if identity.get(key) != expected:
            mismatches[key] = {"expected": expected, "observed": identity.get(key)}
    if mismatches:
        raise QualityCalibrationError(
            "Q0 runtime environment identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return identity


def _run_index(receipt: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row["side"]), str(row["slot_id"]), str(row["run_kind"])): row
        for row in receipt["training"]["runs"]
    }


def _full_training_observability() -> dict[str, Any]:
    return {
        "initial_loss": _not_observed(
            reason="accepted worker retained only best_valid_result"
        ),
        "final_loss": _not_observed(
            reason="accepted worker retained only best_valid_result"
        ),
        "best_epoch": _not_observed(
            reason="accepted worker did not persist epoch index"
        ),
        "validation_curve": _not_observed(
            reason="accepted full runs emitted no TensorBoard event series"
        ),
        "score_statistics": _not_observed(
            reason="accepted full runs persisted only aggregate validation metrics"
        ),
        "embedding_statistics": _not_observed(
            reason="accepted full runs saved no model checkpoint or embedding probe"
        ),
        "gradient_statistics": _not_observed(
            reason="accepted full runs saved no gradient probe"
        ),
        "activation_statistics": _not_observed(
            reason="accepted full runs saved no activation probe"
        ),
        "throughput": _not_observed(
            reason="wall time exists but processed example count per epoch was not persisted"
        ),
        "peak_gpu_memory": _not_observed(
            reason="accepted worker did not persist allocator peak telemetry"
        ),
        "parent_relative_early_curve": _not_observed(
            reason="neither candidate nor matched parent persisted epoch curves"
        ),
    }


def audit_prior_evidence(repo_root: Path) -> dict[str, Any]:
    """Consume all accepted R1/R2 Episodes and resource-censored failures."""

    r1 = _read_json(R1_EXTERNAL_ROOT / "R1_CANONICAL_RECEIPT.json")
    r2 = _read_json(R2_EXTERNAL_ROOT / "R2_CANONICAL_RECEIPT.json")
    f1 = _read_json(F1_ROOT / "F1_CANONICAL_RECEIPT.json")
    recovery = _read_json(
        F1_RECOVERY_ROOT / "F1_RUNTIME_RECOVERY_CANONICAL_RECEIPT.json"
    )
    replay = build_f1_replay_dataset(
        r1_root=R1_EXTERNAL_ROOT, r2_root=R2_EXTERNAL_ROOT
    )
    runs = _run_index(r1)
    r1_records = {
        (side, str(row["logical_slot_id"])): row
        for side, rows in r1["side_records"].items()
        for row in rows
    }
    completed: list[dict[str, Any]] = []
    censored: list[dict[str, Any]] = []
    for row in replay["rows"]:
        if not row["selected_for_experiment"]:
            continue
        if row["campaign_family"] == "R1":
            _prefix, side, slot = row["audit_ref"].split("/")
            candidate = runs[(side, slot, "CANDIDATE")]
            control = runs[(side, slot, "MATCHED_CONTROL")]
            record = r1_records[(side, slot)]
            behavior = record["behavioral_mechanism_evidence"]
            capability_digest = record["capability_digest"]
            episode_digest = record.get("episode_digest")
        else:
            side = None
            slot = str(r2["selection"]["selected_slot"])
            candidate = r2["matched_control"]["candidate"]
            control = r2["matched_control"]["baseline"]
            behavior = _read_json(
                R2_EXTERNAL_ROOT / "qualification" / f"{slot}.json"
            )["behavioral_mechanism_evidence"]
            capability_digest = r2["selection"]["selected_capability_digest"]
            episode_digest = sha256_digest(r2["episode"])
        common = {
            "source": row["campaign_family"],
            "side": side,
            "slot_id": slot,
            "capability_digest": capability_digest,
            "candidate_parameter_count": behavior.get(
                "candidate_parameter_count", "NOT_OBSERVED"
            ),
            "baseline_parameter_count": behavior.get(
                "baseline_parameter_count", "NOT_OBSERVED"
            ),
            "candidate_wall_time_ms": candidate["wall_time_ms"],
            "control_wall_time_ms": control["wall_time_ms"],
            "mechanism_negative_evidence": False,
        }
        if row["episode_observed"]:
            completed.append(
                canonical_value(
                    {
                        **common,
                        "episode_digest": episode_digest,
                        "evidence_class": row["evidence_class"],
                        "mechanism_interpretation": row[
                            "mechanism_interpretation"
                        ],
                        "candidate_ndcg_at_10": candidate["metrics"]["ndcg@10"],
                        "control_ndcg_at_10": control["metrics"]["ndcg@10"],
                        "candidate_minus_control": row["ndcg_delta_audit_only"],
                    }
                )
            )
        else:
            censored.append(
                canonical_value(
                    {
                        **common,
                        "failure_type": (
                            "TIMEOUT"
                            if candidate.get("launcher_return_code") == 124
                            else candidate.get("worker_error_type")
                        ),
                        "resource_or_completion_probability_update": True,
                        "mechanism_effect_update": False,
                    }
                )
            )
    for source, candidate, control in (
        (
            "F1_V1",
            f1["matched_control"]["candidate"],
            f1["matched_control"]["baseline"],
        ),
        (
            "F1_RECOVERY_V2",
            recovery["runtime_recovery"]["candidate"],
            recovery["runtime_recovery"]["control"],
        ),
    ):
        censored.append(
            {
                "source": source,
                "failure_type": (
                    "TIMEOUT"
                    if candidate.get("launcher_return_code") == 124
                    else candidate.get("worker_error_type")
                ),
                "candidate_wall_time_ms": candidate["wall_time_ms"],
                "control_exit_status": control["exit_status"],
                "resource_or_completion_probability_update": True,
                "mechanism_effect_update": False,
                "mechanism_negative_evidence": False,
            }
        )
    r1_selected = next(
        row
        for row in completed
        if row["source"] == "R1"
        and row["capability_digest"]
        == r2["selection"]["selected_capability_digest"]
    )
    r1_location = next(
        item.evidence_locator
        for item in load_registered_r1_artifacts(repo_root)[0]
        if item.capability.digest == r2["selection"]["selected_capability_digest"]
    )
    r1_side, r1_slot = r1_location.split("/")
    r1_candidate_root = next(
        path
        for path in (R1_EXTERNAL_ROOT / r1_side / "candidates" / r1_slot).iterdir()
        if path.is_dir()
    )
    r2_candidate_root = next(
        path
        for path in (
            R2_EXTERNAL_ROOT
            / "execution/candidates"
            / r2["selection"]["selected_slot"]
        ).iterdir()
        if path.is_dir()
    )
    implementation_drift = canonical_value({
        "same_capability_digest": r2["selection"]["selected_capability_digest"],
        "r1_candidate_ndcg_at_10": r1_selected["candidate_ndcg_at_10"],
        "r2_candidate_ndcg_at_10": r2["matched_control"]["candidate"]["metrics"]["ndcg@10"],
        "r1_source_tree_digest": sha256_digest(
            {"files": snapshot_candidate_tree(r1_candidate_root)}
        ),
        "r2_source_tree_digest": sha256_digest(
            {"files": snapshot_candidate_tree(r2_candidate_root)}
        ),
        "interpretation": (
            "same semantic capability was freshly realized into materially different "
            "source and development behavior; this supports H1 without adjudicating effect"
        ),
    })
    implementation_drift["source_trees_equal"] = (
        implementation_drift["r1_source_tree_digest"]
        == implementation_drift["r2_source_tree_digest"]
    )
    if implementation_drift["r1_source_tree_digest"] == implementation_drift[
        "r2_source_tree_digest"
    ]:
        raise QualityCalibrationError("expected R1/R2 realization drift disappeared")
    if len(completed) != 8 or len(censored) != 6:
        raise QualityCalibrationError("Q0 evidence denominator drift")
    deltas = [float(row["candidate_minus_control"]) for row in completed]
    return canonical_value(
        {
            "schema": "recclaw.research-line.q0-prior-evidence-audit.v1",
            "typed_episode_count": len(completed),
            "r1_typed_episode_count": 7,
            "r2_typed_episode_count": 1,
            "resource_censored_count": len(censored),
            "resource_censored_breakdown": {
                "r1_oom": 1,
                "r1_timeout": 3,
                "f1_v1_timeout": 1,
                "f1_recovery_accelerator_error": 1,
            },
            "completed_episode_rows": completed,
            "resource_censored_rows": censored,
            "accepted_full_training_recipe": {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "epochs_requested": 100,
                "validation_metric": "NDCG@10",
            },
            "accepted_full_training_observability": _full_training_observability(),
            "completed_candidate_delta_summary": {
                "all_negative": all(value < 0 for value in deltas),
                "minimum": min(deltas),
                "maximum": max(deltas),
                "mean": sum(deltas) / len(deltas),
            },
            "implementation_drift": implementation_drift,
            "prior_hypothesis_support": {
                "H1_shared_implementer_or_recipe_defect": "STRONG_SUPPORT",
                "H2_complexity_resource_or_optimization_failure": "STRONG_SUPPORT",
                "H3_mechanism_active_but_hypothesis_wrong": (
                    "PLAUSIBLE_BUT_CONFOUNDED_BY_H1"
                ),
            },
            "failure_evidence_policy": (
                "RESOURCE_CENSORED_UPDATES_ONLY_RESOURCE_AND_COMPLETION_PROBABILITY"
            ),
            "resource_censored_mechanism_effect_updates": 0,
            "held_out_reads": 0,
        }
    )


def _target_semantics() -> dict[str, str]:
    mechanisms = {item.mechanism_id: item for item in executable_mechanisms()}
    names = {
        "parent_equivalent_null": "BPR_MF",
        "known_good_reference": (
            "LIGHTGCN__LGCN_RESIDUAL__LGCN_NORM_CONSTRAINT"
        ),
    }
    missing = [name for name in names.values() if name not in mechanisms]
    if missing:
        raise QualityCalibrationError(f"calibration target missing: {missing}")
    return {
        arm: mechanisms[name].mechanism_semantics_digest
        for arm, name in names.items()
    }


def build_prefrozen_manifest(
    *,
    active: Any,
    catalog: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    resources = _resource_root()
    targets = _target_semantics()
    return canonical_value(
        {
            "schema": "recclaw.research-line.q0-prefrozen-calibration-manifest.v1",
            "development_only": True,
            "campaign_id": Q0_CAMPAIGN_ID,
            "accepted_parent_commit": ACCEPTED_F1_COMMIT,
            "selection_frozen_before_provider_or_runtime_outcomes": True,
            "arm_order": (
                "parent_equivalent_null",
                "known_good_reference",
                "frontier_candidate",
            ),
            "arms": {
                "parent_equivalent_null": {
                    "producer_role": "falsification_designer",
                    "proposal_seed": Q0_PROPOSAL_SEEDS[
                        "parent_equivalent_null"
                    ],
                    "qualification_seed": Q0_QUALIFICATION_SEEDS[
                        "parent_equivalent_null"
                    ],
                    "selection_rule": (
                        "exact BPR_MF semantic parent; diagnostic instrument only"
                    ),
                    "target_semantics_digest": targets[
                        "parent_equivalent_null"
                    ],
                    "expected_resolution": "SEARCH_READY",
                    "expected_qualification": "PARENT_EQUIVALENT",
                    "registered_as_capability": False,
                },
                "known_good_reference": {
                    "producer_role": "lineage_refiner",
                    "proposal_seed": Q0_PROPOSAL_SEEDS[
                        "known_good_reference"
                    ],
                    "qualification_seed": Q0_QUALIFICATION_SEEDS[
                        "known_good_reference"
                    ],
                    "selection_rule": (
                        "before Q0 outcomes, choose the simple catalog capability with "
                        "two byte-bound successful DEVELOPMENT_ONLY real-GPU canaries at "
                        "different seeds and unchanged implementation source"
                    ),
                    "selected_mechanism": "LIGHTGCN_RESIDUAL_NORM",
                    "target_semantics_digest": targets[
                        "known_good_reference"
                    ],
                    "expected_resolution": "SEARCH_READY",
                    "expected_qualification": "STRUCTURAL_BEHAVIOR_ACTIVE",
                    "registered_as_capability": False,
                    "historical_canary_ndcg_at_10": 0.1108,
                    "fresh_normal_min_ndcg_at_10": REFERENCE_MIN_NDCG,
                },
                "frontier_candidate": {
                    "producer_role": "frontier_architect",
                    "proposal_seed": Q0_PROPOSAL_SEEDS["frontier_candidate"],
                    "qualification_seed": Q0_QUALIFICATION_SEEDS[
                        "frontier_candidate"
                    ],
                    "selection_rule": (
                        "one preassigned origin-blind frontier Producer call; the sole "
                        "valid response is the denominator and no success selection occurs"
                    ),
                    "target_semantics_digest": None,
                    "expected_resolution": "INNOVATION_REQUIRED",
                    "expected_qualification": "STRUCTURAL_BEHAVIOR_ACTIVE",
                    "registered_as_capability": False,
                },
            },
            "common_provider_contract": {
                "model": MODEL,
                "temperature": 0,
                "tools": [],
                "proposal_token_ceiling": 12_000,
                "implementation_token_ceiling": IMPLEMENTATION_TOKEN_CEILING,
                "producer_prompt_sha256": bytes_sha256(
                    (resources / "quality_calibration_producer_prompt_v1.txt").read_bytes()
                ),
                "implementer_prompt_sha256": bytes_sha256(
                    (resources / "quality_calibration_implementer_prompt_v1.txt").read_bytes()
                ),
                "tool_policy_sha256": bytes_sha256(
                    (resources / "fresh_open_spec_tool_policy_v1.json").read_bytes()
                ),
            },
            "common_runtime": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "held_out_reads": 0,
                "epochs": 100,
                "training_seed": Q0_TRAINING_SEED,
                "timeout_seconds_per_run": Q0_TIMEOUT_SECONDS,
                "run_order": (
                    "matched_bpr_control",
                    "parent_equivalent_null",
                    "known_good_reference",
                    "frontier_candidate",
                ),
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "negative_sampling": "uniform_one_negative_pairwise",
                "validation_metric": "NDCG@10",
                "early_stopping_patience": 10,
            },
            "prefrozen_thresholds": {
                "null_ndcg_absolute_tolerance": NULL_NDCG_ABS_TOLERANCE,
                "known_good_min_ndcg_at_10": REFERENCE_MIN_NDCG,
                "known_good_max_runtime_ratio_to_bpr": (
                    REFERENCE_MAX_RUNTIME_RATIO
                ),
                "frontier_negative_delta_threshold": (
                    -NULL_NDCG_ABS_TOLERANCE
                ),
            },
            "falsifiable_hypotheses": {
                "H1": {
                    "claim": "shared Implementer or training recipe has a common-mode defect",
                    "support_if": (
                        "parent-equivalent null deviates from matched BPR, or null is normal "
                        "but the byte-bound known-good target fails realization/runtime/score criteria"
                    ),
                    "weaken_if": (
                        "null matches BPR and known-good completes with active behavior and "
                        "prefrozen development score/runtime criteria"
                    ),
                },
                "H2": {
                    "claim": "complexity, resource, or optimization failure dominates",
                    "support_if": (
                        "any candidate is OOM/TIMEOUT/AcceleratorError or exceeds the frozen "
                        "completion/runtime boundary while its matched BPR control closes"
                    ),
                    "weaken_if": "all four common runtime runs close within the frozen ceiling",
                },
                "H3": {
                    "claim": "mechanism is active but the scientific hypothesis is wrong",
                    "support_if": (
                        "null and known-good are normal and the frontier mechanism is behaviorally "
                        "active, completes, and is worse than BPR beyond the frozen tolerance"
                    ),
                    "not_adjudicated_if": "H1 or H2 remains active",
                },
            },
            "active_profile_digest": active.profile_digest,
            "origin_blind_catalog_digest": sha256_digest(catalog),
            "protocol_ref": active.protocol_ref,
            "protocol_digest": active.protocol_digest,
            "forbidden": (
                "held_out_access",
                "candidate_specific_manual_patch",
                "outcome_conditioned_selection",
                "fallback_or_attempt_family",
                "static_source_reuse",
                "smoke_as_effect",
                "configuration_tuning_as_research",
                "capability_promotion",
            ),
        }
    )


def _arm_instruction(arm: str, targets: Mapping[str, str]) -> str:
    common = (
        " Use only exact compatibility tokens: "
        + ", ".join(PROTOCOL_REQUIREMENTS)
        + ". Use only exact dependency tokens: "
        + ", ".join(AVAILABLE_DEPENDENCIES)
        + ". Required budgets may not exceed "
        + json.dumps(BUDGET_LIMITS, sort_keys=True)
        + "."
    )
    if arm == "parent_equivalent_null":
        return (
            "Produce an EXPRESSIBLE diagnostic OpenSpec targeting semantic digest "
            + targets[arm]
            + ". Specify an exact mathematical re-expression of inherited BPR: identical "
            "user-item dot-product scores, pairwise BPR loss, full-sort scores, parameters, "
            "and optimizer-visible behavior. Empty capability diff/high-change arrays are "
            "required. It is a null instrument, not innovation."
            + common
        )
    if arm == "known_good_reference":
        return (
            "Produce an EXPRESSIBLE diagnostic OpenSpec targeting semantic digest "
            + targets[arm]
            + ". Precisely specify the known-good LIGHTGCN_RESIDUAL_NORM semantics: symmetric "
            "user-item graph propagation, layer fusion with an ego residual path, dot-product "
            "ranking, pairwise BPR training, and a small soft embedding-norm penalty. Empty "
            "capability diff/high-change arrays are required. This fresh realization is a "
            "diagnostic reference, not a capability claim."
            + common
        )
    if arm == "frontier_candidate":
        return (
            "Produce exactly one NOT_EXPRESSIBLE structural frontier mechanism for open "
            "pairwise recommendation research. It must be a substantive trainable model, "
            "interaction, representation, propagation, or objective change with non-empty "
            "capability diff/high-change arrays. Do not select or copy a profile entry."
            + common
        )
    raise QualityCalibrationError(f"unknown Q0 arm: {arm}")


def render_q0_producer_prompt(
    template: str,
    *,
    arm: str,
    role: str,
    seed: int,
    active: Any,
    catalog: Sequence[Mapping[str, str]],
    targets: Mapping[str, str],
) -> str:
    replacements = {
        "{{CAMPAIGN_ID}}": Q0_CAMPAIGN_ID,
        "{{CALIBRATION_ARM}}": arm,
        "{{LOGICAL_SLOT_ID}}": arm,
        "{{PROPOSAL_SEED}}": str(seed),
        "{{PRODUCER_ROLE}}": role,
        "{{ARM_INSTRUCTION}}": _arm_instruction(arm, targets),
        "{{PROTOCOL_REF}}": active.protocol_ref,
        "{{PROTOCOL_DIGEST}}": active.protocol_digest,
        "{{CONTEXT_REF}}": Q0_CONTEXT_REF,
        "{{CONTEXT_DIGEST}}": Q0_CONTEXT_DIGEST,
        "{{PROFILE_REF}}": active.profile_ref,
        "{{PROFILE_DIGEST}}": active.profile_digest,
        "{{PROFILE_CATALOG_JSON}}": json.dumps(
            catalog, sort_keys=True, separators=(",", ":")
        ),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if "{{" in rendered or "}}" in rendered:
        raise QualityCalibrationError("Q0 Producer prompt has unresolved placeholder")
    forbidden = (
        "side_a",
        "side_b",
        "outcome_digest",
        "episode_digest",
        "candidate_ndcg",
    )
    if any(value in rendered for value in forbidden):
        raise QualityCalibrationError("Q0 Producer prompt leaked origin/outcome")
    return rendered


def _structural_calibration_unit_check(
    evidence: dict[str, Any],
) -> Callable[[Any, Any, Any], None]:
    base = _shared_behavioral_unit_check(
        evidence, require_extra_parameters=False
    )

    def check(model: Any, config: Any, dataset: Any) -> None:
        base(model, config, dataset)
        evidence["probe_status"] = "PASS_STRUCTURAL_BEHAVIOR_ACTIVE"

    return check


def _parent_equivalent_unit_check(
    evidence: dict[str, Any],
) -> Callable[[Any, Any, Any], None]:
    def check(model: Any, config: Any, dataset: Any) -> None:
        import torch

        from recbole.data.interaction import Interaction
        from recbole.model.general_recommender.bpr import BPR

        overridden = tuple(
            name
            for name in ("calculate_loss", "predict", "full_sort_predict")
            if name in model.__class__.__dict__
        )
        if len(overridden) != 3:
            raise AssertionError("null must explicitly implement all behavioral equations")
        baseline = BPR(config, dataset).to(config["device"])
        with torch.no_grad():
            baseline.user_embedding.weight.copy_(model.user_embedding.weight)
            baseline.item_embedding.weight.copy_(model.item_embedding.weight)
        interaction = Interaction(
            {
                model.USER_ID: torch.tensor([1, 2], device=config["device"]),
                model.ITEM_ID: torch.tensor([1, 2], device=config["device"]),
                model.NEG_ITEM_ID: torch.tensor([2, 3], device=config["device"]),
            }
        )
        user_only = Interaction(
            {model.USER_ID: torch.tensor([1, 2], device=config["device"])}
        )
        model.eval()
        baseline.eval()
        with torch.no_grad():
            predict_delta = float(
                torch.max(
                    torch.abs(model.predict(interaction) - baseline.predict(interaction))
                ).item()
            )
            full_sort_delta = float(
                torch.max(
                    torch.abs(
                        model.full_sort_predict(user_only)
                        - baseline.full_sort_predict(user_only)
                    )
                ).item()
            )
        model.train()
        baseline.train()
        candidate_loss = model.calculate_loss(interaction)
        baseline_loss = baseline.calculate_loss(interaction)
        candidate_values = (
            candidate_loss if isinstance(candidate_loss, tuple) else (candidate_loss,)
        )
        baseline_values = (
            baseline_loss if isinstance(baseline_loss, tuple) else (baseline_loss,)
        )
        loss_delta = float(
            torch.abs(
                sum(value.reshape(()) for value in candidate_values).detach()
                - sum(value.reshape(()) for value in baseline_values).detach()
            ).item()
        )
        candidate_names = tuple(name for name, _value in model.named_parameters())
        baseline_names = tuple(name for name, _value in baseline.named_parameters())
        candidate_count = sum(value.numel() for value in model.parameters())
        baseline_count = sum(value.numel() for value in baseline.parameters())
        tolerance = 1e-7
        if (
            predict_delta > tolerance
            or full_sort_delta > tolerance
            or loss_delta > tolerance
            or candidate_names != baseline_names
            or candidate_count != baseline_count
        ):
            raise AssertionError("parent-equivalent null deviates from BPR")
        evidence.update(
            {
                "probe_status": "PASS_PARENT_EQUIVALENT",
                "behavioral_score_max_abs_delta": predict_delta,
                "behavioral_full_sort_max_abs_delta": full_sort_delta,
                "behavioral_loss_max_abs_delta": loss_delta,
                "candidate_parameter_count": candidate_count,
                "baseline_parameter_count": baseline_count,
                "extra_parameter_names": (),
                "overridden_behavioral_methods": overridden,
            }
        )

    return check


def _validate_arm_draft(
    arm: str,
    draft: Mapping[str, Any],
    *,
    target_digest: str | None,
) -> None:
    facts = draft["resolution_facts"]
    if arm in {"parent_equivalent_null", "known_good_reference"}:
        valid = (
            draft["current_profile_expressibility_claim"] == "EXPRESSIBLE"
            and facts["requested_current_semantics_digest"] == target_digest
            and facts["capability_diff"] == []
            and facts["high_change_dimensions"] == []
        )
    else:
        valid = (
            draft["current_profile_expressibility_claim"] == "NOT_EXPRESSIBLE"
            and facts["requested_current_semantics_digest"] is None
            and bool(facts["capability_diff"])
            and bool(facts["high_change_dimensions"])
        )
    if not valid:
        raise QualityCalibrationError(f"Provider violated frozen {arm} contract")


def _is_resource_failure(run: Mapping[str, Any]) -> bool:
    error_type = str(run.get("worker_error_type") or "")
    message = str(run.get("worker_error_message") or "")
    return (
        run.get("launcher_return_code") == 124
        or error_type in {"OutOfMemoryError", "AcceleratorError"}
        or "out of memory" in message.lower()
        or "cuda error" in message.lower()
    )


def _evaluate_q0(
    manifest: Mapping[str, Any],
    prior: Mapping[str, Any],
    arm_records: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    b_success = baseline.get("exit_status") == "SUCCESS" and "ndcg@10" in baseline.get(
        "metrics", {}
    )
    b_metric = float(baseline["metrics"]["ndcg@10"]) if b_success else None
    outcomes: dict[str, Any] = {}
    for arm in manifest["arm_order"]:
        record = arm_records[arm]
        run = record.get("training_run") or {}
        success = run.get("exit_status") == "SUCCESS" and "ndcg@10" in run.get(
            "metrics", {}
        )
        metric = float(run["metrics"]["ndcg@10"]) if success else None
        outcomes[arm] = canonical_value(
            {
                "qualification_status": record.get("qualification_status"),
                "training_success": success,
                "candidate_ndcg_at_10": metric,
                "matched_bpr_ndcg_at_10": b_metric,
                "candidate_minus_bpr": (
                    metric - b_metric
                    if metric is not None and b_metric is not None
                    else None
                ),
                "wall_time_ms": run.get("wall_time_ms"),
                "runtime_ratio_to_bpr": (
                    float(run["wall_time_ms"]) / float(baseline["wall_time_ms"])
                    if success and b_success
                    else None
                ),
                "resource_censored": _is_resource_failure(run),
                "mechanism_effect_update_allowed": success,
                "failure_updates_only_resource_or_completion_probability": (
                    not success
                ),
            }
        )
    null = outcomes["parent_equivalent_null"]
    reference = outcomes["known_good_reference"]
    frontier = outcomes["frontier_candidate"]
    null_normal = bool(
        b_success
        and null["training_success"]
        and abs(float(null["candidate_minus_bpr"])) <= NULL_NDCG_ABS_TOLERANCE
    )
    reference_normal = bool(
        reference["training_success"]
        and float(reference["candidate_ndcg_at_10"]) >= REFERENCE_MIN_NDCG
        and float(reference["runtime_ratio_to_bpr"]) <= REFERENCE_MAX_RUNTIME_RATIO
    )
    resource_failures = [
        arm for arm, outcome in outcomes.items() if outcome["resource_censored"]
    ]
    if _is_resource_failure(baseline):
        resource_failures.insert(0, "matched_bpr_control")
    frontier_negative = bool(
        frontier["training_success"]
        and float(frontier["candidate_minus_bpr"]) < -NULL_NDCG_ABS_TOLERANCE
    )
    if not null_normal and not null["resource_censored"]:
        decision = "H1_SHARED_LANE_OR_RECIPE_DIAGNOSIS"
        recommendation = (
            "Stop Idea/Meta expansion and locate the shared Implementer/training-recipe "
            "deviation exposed by the parent-equivalent null."
        )
    elif null_normal and not reference_normal and not reference["resource_censored"]:
        decision = "H1_PARENT_PRESERVING_REALIZATION_OR_ADAPTER_DIAGNOSIS"
        recommendation = (
            "Stop Idea/Meta expansion and repair parent-preserving realization or "
            "known-good adapter compatibility before another candidate campaign."
        )
    elif resource_failures:
        decision = "H2_RESOURCE_MODELING_AND_SCHEDULING"
        recommendation = (
            "Prioritize resource modeling and scheduling; do not convert censored runs "
            "into mechanism-negative evidence or create another recovery attempt family."
        )
    elif null_normal and reference_normal and frontier_negative:
        decision = "H3_IDEA_QUALITY_AND_MECHANISM_JUDGMENT"
        recommendation = (
            "Proceed next only to Idea-quality and mechanism-judgment redesign; the "
            "common lane and known-good reference calibrated normally while the active "
            "frontier mechanism was genuinely negative."
        )
    else:
        decision = "Q0_NO_COMMON_MODE_FAILURE_REPRODUCED"
        recommendation = (
            "Run one independently frozen Q0 replicate before any broader architecture "
            "expansion; this single development calibration does not justify a formal claim."
        )
    h1 = (
        "SUPPORTED"
        if decision.startswith("H1_")
        else "WEAKENED_BY_Q0_BUT_PRIOR_REALIZATION_DRIFT_REMAINS"
    )
    h2 = "SUPPORTED" if resource_failures else "NOT_REPRODUCED_IN_Q0"
    h3 = (
        "SUPPORTED_DEVELOPMENT_ONLY"
        if decision == "H3_IDEA_QUALITY_AND_MECHANISM_JUDGMENT"
        else "NOT_ADJUDICATED"
    )
    all_runs_closed = b_success and all(
        outcome["training_success"] for outcome in outcomes.values()
    )
    all_qualified = all(
        arm_records[arm].get("qualification_status") == "PASS"
        for arm in manifest["arm_order"]
    )
    gates = {
        "function_real_and_runnable": all_qualified and all_runs_closed,
        "end_to_end_result_chain_real_and_valid": all_runs_closed,
        "serves_open_algorithm_research_target": (
            all_qualified
            and arm_records["frontier_candidate"].get("resolution")
            == "INNOVATION_REQUIRED"
        ),
        "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": (
            all(
                record.get("fresh_provider_implementation") is True
                and record.get("manual_candidate_patches") == 0
                and record.get("static_source_reuse") is False
                for record in arm_records.values()
            )
            and all_runs_closed
        ),
    }
    return canonical_value(
        {
            "outcomes": outcomes,
            "matched_bpr_control": baseline,
            "null_normal": null_normal,
            "known_good_normal": reference_normal,
            "frontier_negative": frontier_negative,
            "resource_failure_arms": resource_failures,
            "hypothesis_support": {"H1": h1, "H2": h2, "H3": h3},
            "decision": decision,
            "unique_next_recommendation": recommendation,
            "core_gates": gates,
            "all_core_gates_pass": all(gates.values()),
            "scientific_claim_authorized": False,
            "formal_acceptance_self_approved": False,
            "prior_evidence_digest": sha256_digest(prior),
        }
    )


def offline_q0_check(repo_root: Path) -> dict[str, Any]:
    identity = verify_q0_source_identity(repo_root, require_fresh_root=False)
    prior = audit_prior_evidence(repo_root)
    artifacts, _receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    _current, _manifest, _next, _build, active = build_active_r2_profile(registry)
    catalog = public_active_profile_catalog(
        active, artifacts, seed=Q0_PROPOSAL_SEEDS["parent_equivalent_null"]
    )
    manifest = build_prefrozen_manifest(active=active, catalog=catalog)
    return canonical_value(
        {
            "status": "Q0_OFFLINE_READY",
            "identity": identity,
            "prior_evidence_digest": sha256_digest(prior),
            "prefrozen_manifest_digest": sha256_digest(manifest),
            "active_profile_digest": active.profile_digest,
            "catalog_entry_count": len(catalog),
            "held_out_reads": 0,
        }
    )


def run_quality_calibration(
    repo_root: Path,
    *,
    canonical_receipt_path: Path,
    delegated_source_identity: Mapping[str, Any] | None = None,
    runtime_environment_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    started_ns = time.monotonic_ns()
    identity = (
        verify_q0_source_identity(repo_root, require_fresh_root=True)
        if delegated_source_identity is None
        else _validated_delegated_source_identity(delegated_source_identity)
    )
    if delegated_source_identity is not None and Q0_ROOT.exists():
        raise QualityCalibrationError(f"Q0 root already exists: {Q0_ROOT}")
    runtime_identity = _validated_runtime_environment_identity(
        runtime_environment_identity
    )
    recbole_commit_identity = (
        str(runtime_identity["recbole_commit"])
        if "recbole_commit" in runtime_identity
        else None
    )
    if canonical_receipt_path.exists():
        raise QualityCalibrationError(
            f"Q0 canonical receipt already exists: {canonical_receipt_path}"
        )
    prior = audit_prior_evidence(repo_root)
    artifacts, _r1_receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    _current, _profile_manifest, _next, _build, active = build_active_r2_profile(
        registry
    )
    catalog = public_active_profile_catalog(
        active, artifacts, seed=Q0_PROPOSAL_SEEDS["parent_equivalent_null"]
    )
    prefrozen = build_prefrozen_manifest(active=active, catalog=catalog)

    Q0_ROOT.mkdir(parents=True)
    _write_new_json(Q0_ROOT / "RUN_IDENTITY.json", identity)
    prefrozen_digest = _write_new_json(
        Q0_ROOT / "PREFROZEN_CALIBRATION_MANIFEST.json", prefrozen
    )
    prior_digest = _write_new_json(
        Q0_ROOT / "PRIOR_EVIDENCE_AUDIT.json", prior
    )

    resources = _resource_root()
    schema = derive_fresh_r2_proposal_schema()
    schema_path = Q0_ROOT / "contracts/q0_proposal_response.schema.json"
    schema_digest = _write_new_json(schema_path, schema)
    producer_template_path = resources / "quality_calibration_producer_prompt_v1.txt"
    implementer_template_path = resources / "quality_calibration_implementer_prompt_v1.txt"
    implementation_schema_path = resources / "fresh_r1_implementation_response_v1.schema.json"
    tool_policy_path = resources / "fresh_open_spec_tool_policy_v1.json"
    producer_template = producer_template_path.read_text(encoding="utf-8")
    implementer_template = implementer_template_path.read_text(encoding="utf-8")
    targets = _target_semantics()
    bindings = canonical_value(
        {
            **_r2_bindings(active),
            "context_ref": Q0_CONTEXT_REF,
            "context_digest": Q0_CONTEXT_DIGEST,
        }
    )
    environment = _r2_environment(active)
    policy = _shared_policy(
        bytes_sha256(implementer_template_path.read_bytes()),
        bytes_sha256(tool_policy_path.read_bytes()),
    )
    arm_records: dict[str, dict[str, Any]] = {}
    live_specs: dict[str, Any] = {}

    # Freeze all three Producer/OpenSpec/Resolver results before any implementation,
    # qualification, or runtime outcome exists.
    for arm in prefrozen["arm_order"]:
        arm_plan = prefrozen["arms"][arm]
        prompt = render_q0_producer_prompt(
            producer_template,
            arm=arm,
            role=arm_plan["producer_role"],
            seed=arm_plan["proposal_seed"],
            active=active,
            catalog=catalog,
            targets=targets,
        )
        call = bounded_provider_call(
            call_root=Q0_ROOT / "provider/proposals" / arm,
            schema_path=schema_path,
            logical_call_id=f"{Q0_RUN_IDENTITY}:{arm}:proposal",
            session_id=f"{Q0_RUN_IDENTITY}:proposal-session",
            prompt=prompt,
            token_ceiling=12_000,
        )
        record: dict[str, Any] = {
            "arm": arm,
            "proposal_seed": arm_plan["proposal_seed"],
            "provider_attempts": call.attempts,
            "manual_candidate_patches": 0,
            "static_source_reuse": False,
        }
        if call.call is None:
            record.update(
                {
                    "stage": "PROPOSAL_PROVIDER_FAILURE",
                    "failure": call.failure,
                    "mechanism_negative_evidence": False,
                }
            )
            arm_records[arm] = record
            _write_new_json(Q0_ROOT / "arms" / arm / "proposal_record.json", record)
            continue
        validate_v4_response_contract(call.call.response, provider_schema=schema)
        draft = call.call.response["proposals"][0]
        if draft["producer_role"] != arm_plan["producer_role"]:
            raise QualityCalibrationError("Provider changed frozen producer role")
        target = arm_plan["target_semantics_digest"]
        _validate_arm_draft(arm, draft, target_digest=target)
        spec, facts = project_open_producer_draft(draft, bindings=bindings)
        resolution = resolve_capability(
            spec, resolution_facts=facts, environment=environment
        )
        if resolution.resolution.value != arm_plan["expected_resolution"]:
            raise QualityCalibrationError(f"unexpected Q0 resolution for {arm}")
        record.update(
            {
                "stage": "OPENSPEC_FROZEN",
                "proposal_response_digest": call.call.response_digest,
                "spec_ref": spec.spec_id,
                "spec_digest": spec.digest,
                "resolution": resolution.resolution.value,
                "resolution_digest": resolution.digest,
                "resolution_reason_codes": resolution.reason_codes,
                "calibration_instrument_reimplementation": arm
                != "frontier_candidate",
            }
        )
        live_specs[arm] = spec
        arm_records[arm] = record
        _write_new_json(
            Q0_ROOT / "arms" / arm / "open_spec.json",
            {
                "research_spec": spec.canonical_dict(),
                "resolution_facts": facts,
                "resolution": resolution.canonical_dict(),
            },
        )
        _write_new_json(Q0_ROOT / "arms" / arm / "proposal_record.json", record)

    # The same blind Implementer policy and prompt consume every frozen OpenSpec.
    for arm in prefrozen["arm_order"]:
        if arm not in live_specs:
            continue
        spec = live_specs[arm]
        request = build_shared_implementer_request(spec, policy=policy)
        prompt = render_implementation_prompt(implementer_template, request)
        call = bounded_provider_call(
            call_root=Q0_ROOT / "provider/implementations" / arm,
            schema_path=implementation_schema_path,
            logical_call_id=f"{Q0_RUN_IDENTITY}:{arm}:implementation",
            session_id=f"{Q0_RUN_IDENTITY}:shared-origin-blind-implementation-session",
            prompt=prompt,
            token_ceiling=IMPLEMENTATION_TOKEN_CEILING,
        )
        record = arm_records[arm]
        record["implementation_provider_attempts"] = call.attempts
        if call.call is None:
            record.update(
                {
                    "stage": "IMPLEMENTATION_PROVIDER_FAILURE",
                    "failure": call.failure,
                    "mechanism_negative_evidence": False,
                }
            )
            continue
        factory = (
            _parent_equivalent_unit_check
            if arm == "parent_equivalent_null"
            else _structural_calibration_unit_check
        )
        materialized, qualification, behavior = _materialize_and_qualify(
            repo_root=repo_root,
            side_root=Q0_ROOT / "arms" / arm,
            slot_id="candidate",
            seed=prefrozen["arms"][arm]["qualification_seed"],
            spec=spec,
            implementation=call.call.response["proposals"][0],
            implementation_prompt_digest=bytes_sha256(
                implementer_template_path.read_bytes()
            ),
            tool_policy_digest=bytes_sha256(tool_policy_path.read_bytes()),
            run_identity=Q0_RUN_IDENTITY,
            unit_check_factory=factory,
        )
        candidate_root = (
            Q0_ROOT
            / "arms"
            / arm
            / "candidates/candidate"
            / str(materialized.shared_request["blind_candidate_id"])
        )
        source_text = (candidate_root / "recclaw_ext/candidate.py").read_text(
            encoding="utf-8"
        )
        forbidden_static_imports = (
            "recclaw_ext.models",
            "executable_mechanism_catalog",
            "candidate_registry",
        )
        if any(token in source_text for token in forbidden_static_imports):
            raise QualityCalibrationError(
                f"{arm} implementation reused static catalog source"
            )
        _write_new_json(
            Q0_ROOT / "arms" / arm / "qualification.json",
            {**qualification.to_dict(), "behavioral_evidence": behavior},
        )
        record.update(
            {
                "stage": (
                    "QUALIFIED"
                    if qualification.receipt.status is QualificationStatusV1.PASS
                    else "QUALIFICATION_FAILURE"
                ),
                "qualification_status": qualification.receipt.status.value,
                "qualification_receipt_digest": qualification.receipt.digest,
                "behavioral_evidence": behavior,
                "candidate_package_digest": materialized.package.digest,
                "candidate_source_tree_digest": materialized.package.source_tree_digest,
                "candidate_root": str(candidate_root),
                "entrypoint": materialized.package.executable_entrypoint,
                "fresh_provider_implementation": True,
                "static_source_reuse": False,
                "manual_candidate_patches": 0,
            }
        )

    baseline_source = RECBole_ROOT / "recbole/model/general_recommender/bpr.py"
    baseline = run_development_training(
        repo_root=repo_root,
        side_root=Q0_ROOT / "execution",
        run_id="matched-bpr-control",
        seed=Q0_TRAINING_SEED,
        candidate_root=None,
        entrypoint="recbole.model.general_recommender.bpr:BPR",
        source_sha256=bytes_sha256(baseline_source.read_bytes()),
        run_identity=Q0_RUN_IDENTITY,
        authority="user-delegated-q0-common-mode-calibration",
        timeout_seconds=Q0_TIMEOUT_SECONDS,
        recbole_commit_identity=recbole_commit_identity,
    )
    for arm in prefrozen["arm_order"]:
        record = arm_records[arm]
        if record.get("qualification_status") != "PASS":
            record["training_run"] = {
                "exit_status": "NOT_RUN_QUALIFICATION_DID_NOT_PASS",
                "mechanism_negative_evidence": False,
            }
            continue
        candidate_root = Path(record["candidate_root"])
        source = candidate_root / "recclaw_ext/candidate.py"
        record["training_run"] = run_development_training(
            repo_root=repo_root,
            side_root=Q0_ROOT / "execution",
            run_id=arm.replace("_", "-"),
            seed=Q0_TRAINING_SEED,
            candidate_root=candidate_root,
            entrypoint=str(record["entrypoint"]),
            source_sha256=bytes_sha256(source.read_bytes()),
            run_identity=Q0_RUN_IDENTITY,
            authority="user-delegated-q0-common-mode-calibration",
            timeout_seconds=Q0_TIMEOUT_SECONDS,
            recbole_commit_identity=recbole_commit_identity,
        )
        _write_new_json(
            Q0_ROOT / "arms" / arm / "training_run.json",
            record["training_run"],
        )

    evaluation = _evaluate_q0(prefrozen, prior, arm_records, baseline)
    provider_records = list(arm_records.values())
    proposal_usage = _physical_usage(provider_records)
    implementation_usage = _physical_usage(
        [
            {"provider_attempts": row.get("implementation_provider_attempts", ())}
            for row in provider_records
        ]
    )
    receipt = canonical_value(
        {
            "schema": "recclaw.research-line.q0-quality-calibration-canonical-receipt.v1",
            "status": evaluation["decision"],
            "development_only": True,
            "attempt_identity": identity,
            "runtime_environment_identity": runtime_identity,
            "prefrozen_manifest": prefrozen,
            "prefrozen_manifest_file_sha256": prefrozen_digest,
            "prior_evidence_audit": prior,
            "prior_evidence_audit_file_sha256": prior_digest,
            "proposal_schema_sha256": schema_digest,
            "arm_records": arm_records,
            "evaluation": evaluation,
            "proposal_provider_usage": proposal_usage,
            "implementation_provider_usage": implementation_usage,
            "training": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "held_out_exposed": False,
                "epochs_requested_per_run": 100,
                "physical_runs_requested": 4,
                "new_recovery_attempt_families": 0,
                "manual_candidate_patches": 0,
            },
            "qualification_evidence_used_as_scientific": False,
            "parent_and_reference_registered_as_promising_capability": False,
            "scientific_effect_claim": False,
            "formal_scientific_experiment": False,
            "held_out_reads": 0,
            "wall_time_ms": max(
                1, (time.monotonic_ns() - started_ns) // 1_000_000
            ),
        }
    )
    external_sha256 = _write_new_json(
        Q0_ROOT / "Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json", receipt
    )
    repository_receipt = canonical_value(
        {
            **receipt,
            "external_receipt_ref": str(
                Q0_ROOT / "Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json"
            ),
            "external_receipt_sha256": external_sha256,
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


__all__ = [
    "Q0_ROOT",
    "QualityCalibrationError",
    "audit_prior_evidence",
    "build_prefrozen_manifest",
    "offline_q0_check",
    "render_q0_producer_prompt",
    "run_quality_calibration",
    "verify_q0_source_identity",
]
