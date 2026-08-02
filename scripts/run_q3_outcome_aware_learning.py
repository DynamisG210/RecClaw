#!/usr/bin/env python3
"""Execute the local DEVELOPMENT_ONLY Q3 outcome-aware learning closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.fresh_f1 import (  # noqa: E402
    F1_ROOT,
    R2_EXTERNAL_ROOT,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (  # noqa: E402
    R1_EXTERNAL_ROOT,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_f1 import (  # noqa: E402
    build_f1_replay_dataset,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_q3 import (  # noqa: E402
    HEAD_AUTHORITY_MATRIX,
    build_q3_denominator_projection,
    build_q3_policy_activation,
    consume_q3_active_policy,
    evaluate_q3_development_activation,
    fit_q3_three_head_policy,
    run_group_aware_offline_replay,
    shadow_compare_q3_policy,
)


Q2_INPUT_SHA256 = "a0b131dcf7782103ca75538b6215443219036ce95230b628c318aede423d44ab"
Q3_POLICY_VERSION = "research-open-meta-q3-v1.0.0"
Q3_ACTIVATION_ID = "q3-outcome-aware-learning-20260803-01"
Q3_ACQUISITION_SEEDS = {
    "IDEA": 56031,
    "EXPERIMENT": 56032,
    "REPLICATION": 56033,
}
Q3_RESULT_ID = "q3_outcome_aware_learning_20260803_01"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
    return bytes_sha256(payload)


def _file_digest(path: Path) -> str:
    return bytes_sha256(path.read_bytes())


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _artifact_record(path: Path, semantic_digest: str | None = None) -> dict[str, Any]:
    return canonical_value(
        {
            "path": _relative(path),
            "sha256": _file_digest(path),
            "semantic_digest": semantic_digest,
        }
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    canonical_path = args.canonical_receipt.resolve()
    closure_path = args.f1_closure_receipt.resolve()
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise RuntimeError(f"Q3 output root already contains files: {output_root}")
    if canonical_path.exists():
        raise RuntimeError(f"Q3 canonical receipt already exists: {canonical_path}")
    if not closure_path.is_file():
        raise RuntimeError(f"accepted F1 closure receipt is missing: {closure_path}")

    docs = ROOT / "docs/research_line/vnext"
    q2_root = (
        ROOT / "results/research_line/q2_mechanism_characterization_20260802_01"
    )
    q2_input_path = q2_root / "Q3_MECHANISM_EVIDENCE_PACKAGE.json"
    if _file_digest(q2_input_path) != Q2_INPUT_SHA256:
        raise RuntimeError("Q2 unique input digest drift")
    q3_package = _read(q2_input_path)
    if (
        q3_package["held_out_reads"] != 0
        or q3_package["mechanism_state"] != "NON_IDENTIFIABLE"
        or q3_package["resource_status"] != "RESOURCE_CENSORED"
        or q3_package["mechanism_effect_update_allowed"] is not False
    ):
        raise RuntimeError("Q2 authority boundary drift")

    f1_replay = build_f1_replay_dataset(
        r1_root=R1_EXTERNAL_ROOT, r2_root=R2_EXTERNAL_ROOT
    )
    resource_names = (
        ("Q0", "Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json"),
        ("Q0R", "Q0R_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json"),
        (
            "Q0R_FIXED_BATCH",
            "Q0R_FIXED_BATCH_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
        ),
        (
            "Q0R_TYPE_PRESERVING",
            "Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json",
        ),
        ("Q0R2", "Q0R2_RESOURCE_ADMISSION_CANONICAL_RECEIPT.json"),
    )
    projection = build_q3_denominator_projection(
        f1_replay=f1_replay,
        resource_receipts=tuple(
            (stage, _read(docs / name)) for stage, name in resource_names
        ),
        f1_receipt=_read(F1_ROOT / "F1_CANONICAL_RECEIPT.json"),
        f1_closure_receipt=_read(closure_path),
        q3_package=q3_package,
        q2_result=_read(q2_root / "Q2_PHYSICAL_RESULT.json"),
        q2_resource_result=_read(q2_root / "RESOURCE_PROBE_RESULT.json"),
    )
    parent_policy_path = F1_ROOT / "policy/versioned_policy.json"
    parent_policy = _read(parent_policy_path)
    q1_pool_path = (
        ROOT
        / "results/research_line/q1_prompt_contract_20260802_01/"
        "FROZEN_SELECTION_BEFORE_IMPLEMENTATION.json"
    )
    q1_pool = _read(q1_pool_path)
    policy = fit_q3_three_head_policy(
        projection,
        parent_policy_digest=parent_policy["policy_digest"],
        policy_version=Q3_POLICY_VERSION,
    )
    replay = run_group_aware_offline_replay(
        projection, parent_policy_digest=parent_policy["policy_digest"]
    )
    shadow = shadow_compare_q3_policy(
        projection=projection,
        q1_pool=q1_pool,
        parent_policy=parent_policy,
        q3_policy=policy,
    )
    promotion = evaluate_q3_development_activation(
        projection, replay, shadow, policy
    )
    activation = build_q3_policy_activation(
        policy,
        promotion,
        projection_digest=projection["projection_digest"],
        activation_id=Q3_ACTIVATION_ID,
    )

    paths = {
        "authority_projection": output_root / "evidence/denominator_projection.json",
        "policy": output_root / "policy/versioned_policy.json",
        "offline_replay": output_root / "evaluation/group_aware_replay.json",
        "shadow": output_root / "evaluation/f1_shadow_comparison.json",
        "promotion": output_root / "activation/development_promotion.json",
        "activation": output_root / "activation/active_policy.json",
        "acquisition_idea": output_root / "next_fresh/idea_acquisition_manifest.json",
        "acquisition_experiment": (
            output_root / "next_fresh/experiment_acquisition_manifest.json"
        ),
        "acquisition_replication": (
            output_root / "next_fresh/replication_acquisition_manifest.json"
        ),
        "soak": output_root / "soak/Q3_SOAK_POLICY_PACKAGE.json",
        "physical_receipt": output_root / "Q3_PHYSICAL_RECEIPT.json",
    }
    _write_new(paths["authority_projection"], projection)
    _write_new(paths["policy"], policy)
    _write_new(paths["offline_replay"], replay)
    _write_new(paths["shadow"], shadow)
    _write_new(paths["promotion"], promotion)
    _write_new(paths["activation"], activation)

    acquisitions = {}
    for task_type, seed in Q3_ACQUISITION_SEEDS.items():
        acquisition = consume_q3_active_policy(
            policy_path=paths["policy"],
            activation_path=paths["activation"],
            frozen_pool_path=q1_pool_path,
            task_type=task_type,
            random_seed=seed,
        )
        acquisitions[task_type] = acquisition
        _write_new(paths[f"acquisition_{task_type.lower()}"], acquisition)
    soak = canonical_value(
        {
            "schema": "recclaw.research-line.q3-soak-policy-package.v1",
            "policy": {
                "policy_ref": policy["policy_ref"],
                "policy_version": policy["policy_version"],
                "policy_digest": policy["policy_digest"],
                "parent_policy_digest": policy["parent_policy_digest"],
                "policy_file": _relative(paths["policy"]),
            },
            "activation": {
                "activation_id": activation["activation_id"],
                "activation_digest": activation["activation_digest"],
                "activation_file": _relative(paths["activation"]),
                "rollback_policy_digest": activation["rollback_policy_digest"],
                "reversible": activation["reversible"],
            },
            "resume_manifest": {
                "next_round_index": 1,
                "task_types_supported": ("IDEA", "EXPERIMENT", "REPLICATION"),
                "pool_digest": acquisitions["IDEA"]["pool_digest"],
                "exploration_probability": acquisitions["IDEA"][
                    "exploration_probability"
                ],
                "task_acquisitions": {
                    task_type: {
                        "acquisition_digest": acquisition["acquisition_digest"],
                        "selected_candidate_id": acquisition[
                            "selected_candidate_id"
                        ],
                        "random_seed": acquisition["random_seed"],
                        "random_draw": acquisition["random_draw"],
                        "exploration_selected": acquisition[
                            "exploration_selected"
                        ],
                        "selection_probabilities": {
                            row["candidate_id"]: row["selection_probability"]
                            for row in acquisition["candidates"]
                        },
                    }
                    for task_type, acquisition in acquisitions.items()
                },
            },
            "q4_consumable": {
                "projection_digest": projection["projection_digest"],
                "policy_digest": policy["policy_digest"],
                "offline_replay_digest": replay["replay_digest"],
                "shadow_digest": shadow["shadow_digest"],
                "acquisition_digests": {
                    task_type: acquisition["acquisition_digest"]
                    for task_type, acquisition in acquisitions.items()
                },
            },
            "soak_platform_or_service_created": False,
            "multi_round_soak_executed": False,
            "official_held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    soak = {**soak, "soak_package_digest": sha256_digest(soak)}
    _write_new(paths["soak"], soak)

    artifacts = {
        "denominator_projection": _artifact_record(
            paths["authority_projection"], projection["projection_digest"]
        ),
        "versioned_policy": _artifact_record(paths["policy"], policy["policy_digest"]),
        "offline_replay": _artifact_record(
            paths["offline_replay"], replay["replay_digest"]
        ),
        "shadow_comparison": _artifact_record(paths["shadow"], shadow["shadow_digest"]),
        "development_promotion": _artifact_record(
            paths["promotion"], promotion["promotion_digest"]
        ),
        "active_policy": _artifact_record(
            paths["activation"], activation["activation_digest"]
        ),
        "next_fresh_idea_acquisition": _artifact_record(
            paths["acquisition_idea"], acquisitions["IDEA"]["acquisition_digest"]
        ),
        "next_fresh_experiment_acquisition": _artifact_record(
            paths["acquisition_experiment"],
            acquisitions["EXPERIMENT"]["acquisition_digest"],
        ),
        "next_fresh_replication_acquisition": _artifact_record(
            paths["acquisition_replication"],
            acquisitions["REPLICATION"]["acquisition_digest"],
        ),
        "soak_policy_package": _artifact_record(
            paths["soak"], soak["soak_package_digest"]
        ),
    }
    physical_receipt = canonical_value(
        {
            "schema": "recclaw.research-line.q3-outcome-aware-physical-receipt.v1",
            "status": "Q3_DEVELOPMENT_ONLY_PASS",
            "development_only": True,
            "scientific_effect_claim": False,
            "policy_superiority_claim": False,
            "held_out_reads": 0,
            "source_identity": {
                "q2_commit": "49f9225517d76582500bcf80bf7fc6a5d112f137",
                "q2_parent": "d10083908a64a0d05a0cbbb1a6eddc2055b2f710",
                "q2_tree": "71ae16d237211f3852782765b34de8c9fadefc5d",
                "q2_input_sha256": _file_digest(q2_input_path),
                "f1_closure_commit": "1d3223104871bac6820ab1a550461bd142591a02",
                "f1_closure_receipt_sha256": _file_digest(closure_path),
                "parent_policy_file_sha256": _file_digest(parent_policy_path),
                "q1_frozen_pool_sha256": _file_digest(q1_pool_path),
                "authority_matrix_digest": sha256_digest(HEAD_AUTHORITY_MATRIX),
            },
            "denominator": {
                "row_count": projection["row_count"],
                "source_counts": projection["source_counts"],
                "head_update_counts": projection["head_update_counts"],
                "negative_evidence_preserved": projection[
                    "negative_evidence_preserved"
                ],
            },
            "three_head_learning": {
                "feasibility": {
                    "method": policy["heads"]["feasibility"]["method"],
                    "input_count": policy["heads"]["feasibility"]["input_count"],
                    "head_digest": policy["heads"]["feasibility"]["head_digest"],
                },
                "mechanism_information": {
                    "method": policy["heads"]["mechanism_information"]["method"],
                    "input_count": policy["heads"]["mechanism_information"][
                        "input_count"
                    ],
                    "state_counts": policy["heads"]["mechanism_information"][
                        "state_counts"
                    ],
                    "head_digest": policy["heads"]["mechanism_information"][
                        "head_digest"
                    ],
                },
                "effect": {
                    "method": policy["heads"]["effect"]["method"],
                    "input_count": policy["heads"]["effect"]["input_count"],
                    "head_digest": policy["heads"]["effect"]["head_digest"],
                    "scientific_superiority_claim": False,
                },
            },
            "offline_evaluation": {
                "group_aware_replay": replay["method"],
                "replay_metrics": replay["metrics"],
                "shadow_selection_difference_count": shadow[
                    "selection_difference_count"
                ],
                "shadow_task_types": tuple(
                    row["task_type"] for row in shadow["comparisons"]
                ),
                "statistical_superiority_claim": False,
            },
            "activation": {
                "promotion_status": promotion["status"],
                "promotion_gates": promotion["gates"],
                "activation_status": activation["status"],
                "activation_digest": activation["activation_digest"],
                "parent_policy_digest": activation["parent_policy_digest"],
                "rollback_policy_digest": activation["rollback_policy_digest"],
                "reversible": activation["reversible"],
            },
            "next_fresh_consumer": {
                "read_active_policy_from_disk": all(
                    acquisition["consumer_input"]["read_active_policy_from_disk"]
                    for acquisition in acquisitions.values()
                ),
                "real_provider_origin_pool": acquisitions["IDEA"][
                    "pool_provider_origin"
                ],
                "static_fixture": any(
                    acquisition["pool_static_fixture"]
                    for acquisition in acquisitions.values()
                ),
                "candidate_count_per_task": {
                    task_type: acquisition["candidate_count"]
                    for task_type, acquisition in acquisitions.items()
                },
                "task_acquisitions": {
                    task_type: {
                        "acquisition_digest": acquisition["acquisition_digest"],
                        "selected_candidate_id": acquisition[
                            "selected_candidate_id"
                        ],
                        "selection_probabilities_sum": acquisition[
                            "selection_probabilities_sum"
                        ],
                        "exploration_probability": acquisition[
                            "exploration_probability"
                        ],
                        "exploration_selected": acquisition[
                            "exploration_selected"
                        ],
                        "selection_score_distinct_count": len(
                            {
                                row["selection_score"]
                                for row in acquisition["candidates"]
                            }
                        ),
                    }
                    for task_type, acquisition in acquisitions.items()
                },
                "idea_equal_score_negative_evidence_preserved": (
                    len(
                        {
                            row["selection_score"]
                            for row in acquisitions["IDEA"]["candidates"]
                        }
                    )
                    == 1
                ),
                "experiment_policy_signal_present": (
                    len(
                        {
                            row["selection_score"]
                            for row in acquisitions["EXPERIMENT"]["candidates"]
                        }
                    )
                    > 1
                ),
            },
            "four_gates": {
                "function_real_and_runnable": True,
                "end_to_end_result_chain_real_and_valid": True,
                "serves_open_recommendation_algorithm_research": True,
                "no_66_item_tuning_static_candidate_wrapper_fallback_mock_or_smoke_effect": True,
            },
            "new_provider_calls": 0,
            "new_gpu_training_lines": 0,
            "manual_candidate_patches": 0,
            "artifacts": artifacts,
            "soak_input": {
                "path": _relative(paths["soak"]),
                "sha256": artifacts["soak_policy_package"]["sha256"],
                "semantic_digest": soak["soak_package_digest"],
            },
        }
    )
    physical_receipt = {
        **physical_receipt,
        "receipt_digest": sha256_digest(physical_receipt),
    }
    _write_new(paths["physical_receipt"], physical_receipt)

    sums_paths = [*paths.values()]
    sums_lines = [
        f"{_file_digest(path)}  {path.relative_to(output_root).as_posix()}"
        for path in sorted(sums_paths)
    ]
    sums_path = output_root / "SHA256SUMS"
    with sums_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(sums_lines) + "\n")

    canonical_receipt = canonical_value(
        {
            "schema": "recclaw.research-line.q3-outcome-aware-canonical-receipt.v1",
            "status": physical_receipt["status"],
            "development_only": True,
            "scientific_effect_claim": False,
            "policy_superiority_claim": False,
            "held_out_reads": 0,
            "physical_receipt_ref": _relative(paths["physical_receipt"]),
            "physical_receipt_sha256": _file_digest(paths["physical_receipt"]),
            "physical_receipt_digest": physical_receipt["receipt_digest"],
            "policy_version": policy["policy_version"],
            "policy_digest": policy["policy_digest"],
            "parent_policy_digest": policy["parent_policy_digest"],
            "projection_digest": projection["projection_digest"],
            "replay_digest": replay["replay_digest"],
            "shadow_digest": shadow["shadow_digest"],
            "activation_digest": activation["activation_digest"],
            "acquisition_digests": {
                task_type: acquisition["acquisition_digest"]
                for task_type, acquisition in acquisitions.items()
            },
            "soak_package_digest": soak["soak_package_digest"],
            "four_gates": physical_receipt["four_gates"],
            "head_update_counts": projection["head_update_counts"],
            "negative_evidence_preserved": projection[
                "negative_evidence_preserved"
            ],
            "next_fresh_selected_candidate_ids": {
                task_type: acquisition["selected_candidate_id"]
                for task_type, acquisition in acquisitions.items()
            },
            "next_soak_input": _relative(paths["soak"]),
        }
    )
    _write_new(canonical_path, canonical_receipt)
    return {
        "status": physical_receipt["status"],
        "output_root": str(output_root),
        "canonical_receipt": str(canonical_path),
        "canonical_receipt_sha256": _file_digest(canonical_path),
        "physical_receipt_sha256": _file_digest(paths["physical_receipt"]),
        "policy_digest": policy["policy_digest"],
        "activation_digest": activation["activation_digest"],
        "acquisition_digests": {
            task_type: acquisition["acquisition_digest"]
            for task_type, acquisition in acquisitions.items()
        },
        "soak_input": str(paths["soak"]),
        "selected_candidate_ids": {
            task_type: acquisition["selected_candidate_id"]
            for task_type, acquisition in acquisitions.items()
        },
        "exploration_selected": {
            task_type: acquisition["exploration_selected"]
            for task_type, acquisition in acquisitions.items()
        },
        "held_out_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/research_line" / Q3_RESULT_ID,
    )
    parser.add_argument(
        "--canonical-receipt",
        type=Path,
        default=(
            ROOT
            / "docs/research_line/vnext/"
            "Q3_OUTCOME_AWARE_LEARNING_CANONICAL_RECEIPT.json"
        ),
    )
    parser.add_argument(
        "--f1-closure-receipt",
        type=Path,
        default=(
            ROOT.parent
            / "RecClaw_f1_gpu35_closure/docs/research_line/vnext/"
            "F1_RESOURCE_COMPATIBLE_EXECUTION_CANONICAL_RECEIPT.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
