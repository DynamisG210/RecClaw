"""Fail-closed precommit seal for a genuinely unread V31 outer panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_value,
    sha256_digest,
)


OUTER_PANEL_SCHEMA_V31 = "recclaw.helix.outer-panel-precommit.v31"
OUTER_PANEL_REQUIRED_ROLES = (
    "train",
    "development_validation",
    "outer_heldout",
    "item_features",
    "user_features",
)
KNOWN_V29_ONLINE_INTERACTION_DIGESTS = frozenset(
    {
        "c84b1a4f6c6d974f32f126b173f11f7af8e12e1a143a50ac5a53e9945903491a",
        "631911b8e59d312e110ba7205151bcda3d52378ecbf510d48d8b9cc162956cc9",
        "4847d5cf5d3abb04b533baf0a2946bf26cd35c924234c7572093a37638a424a0",
    }
)
OUTER_EVALUATION_SEEDS_V31 = (55303, 55304, 55305, 55306, 55307)


class OuterPanelError(ValueError):
    pass


def _sha(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OuterPanelError(f"{name} must be a lowercase SHA256")
    return value


def seal_outer_panel_v31(
    *,
    dataset_id: str,
    files: Mapping[str, Path],
    provenance: Mapping[str, Any],
    custodian_attestation: Mapping[str, Any],
    evaluation_seeds: Sequence[int] = OUTER_EVALUATION_SEEDS_V31,
) -> dict[str, Any]:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise OuterPanelError("outer panel dataset_id is required")
    if tuple(sorted(files)) != tuple(sorted(OUTER_PANEL_REQUIRED_ROLES)):
        raise OuterPanelError("outer panel requires the exact five file roles")
    if custodian_attestation.get("search_access_history") != (
        "NEVER_EXPOSED_TO_RESEARCH_OR_HELIX"
    ):
        raise OuterPanelError("outer panel lacks an unread custodian attestation")
    if custodian_attestation.get("precommitted_before_arm_launch") is not True:
        raise OuterPanelError("outer panel was not precommitted before arm launch")
    custodian = custodian_attestation.get("custodian_ref")
    if not isinstance(custodian, str) or not custodian.strip():
        raise OuterPanelError("outer panel custodian_ref is required")
    _sha(provenance.get("source_snapshot_digest"), name="source_snapshot_digest")
    normalized_seeds = tuple(evaluation_seeds)
    if (
        len(normalized_seeds) != 5
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in normalized_seeds)
        or len(set(normalized_seeds)) != len(normalized_seeds)
    ):
        raise OuterPanelError("outer panel requires five unique integer seeds")

    file_rows: dict[str, dict[str, Any]] = {}
    for role in OUTER_PANEL_REQUIRED_ROLES:
        path = Path(files[role])
        if not path.is_file():
            raise OuterPanelError(f"outer panel file is missing: {role}")
        file_rows[role] = {
            "basename": path.name,
            "sha256": bytes_sha256(path.read_bytes()),
            "size_bytes": path.stat().st_size,
        }
    interaction_digests = {
        file_rows[role]["sha256"]
        for role in ("train", "development_validation", "outer_heldout")
    }
    if len(interaction_digests) != 3:
        raise OuterPanelError("outer train/dev/heldout files must be distinct")
    reused = interaction_digests & KNOWN_V29_ONLINE_INTERACTION_DIGESTS
    if reused:
        raise OuterPanelError("outer panel reuses a V29 online interaction file")

    payload = {
        "schema": OUTER_PANEL_SCHEMA_V31,
        "status": "PRECOMMITTED_UNREAD",
        "dataset_id": dataset_id.strip(),
        "panel_kind": "POST_SELECTION_CROSS_DATASET_REPLICATION",
        "files": file_rows,
        "provenance": canonical_value(dict(provenance)),
        "custodian_attestation": canonical_value(dict(custodian_attestation)),
        "access_contract": {
            "research_arm_online_access": "FORBIDDEN",
            "helix_arm_online_access": "FORBIDDEN",
            "selection_phase_access": "NONE",
            "post_selection_train_dev_access": "SELECTED_CANDIDATE_RETRAIN_ONLY",
            "outer_heldout_access": "FINAL_EVALUATOR_ONLY",
            "feedback_to_search": "FORBIDDEN",
        },
        "selection_contract": {
            "selected_candidates_per_arm": 1,
            "selection_metric": "DEVELOPMENT_VALIDATION_NDCG_AT_10",
            "selection_rule": "HIGHEST_OBSERVED_EXACT_CANDIDATE_SCORE",
            "tie_breakers": [
                "LOWER_FIRST_SUCCESS_ROUND",
                "LEXICOGRAPHIC_CANDIDATE_ID",
            ],
            "selection_must_precede_outer_access": True,
        },
        "evaluation_contract": {
            "metric": "NDCG@10",
            "candidate_universe": "FULL_SORT",
            "ordered_seed_panel": normalized_seeds,
            "epochs_requested": 100,
            "aggregation": "ARITHMETIC_MEAN_WITH_PAIRED_SEED_DELTAS",
            "report_all_failures": True,
        },
    }
    return {**canonical_value(payload), "manifest_digest": sha256_digest(payload)}


def validate_outer_panel_manifest_v31(manifest: Mapping[str, Any]) -> None:
    value = canonical_value(dict(manifest))
    if value.get("schema") != OUTER_PANEL_SCHEMA_V31:
        raise OuterPanelError("outer panel schema mismatch")
    if value.get("status") != "PRECOMMITTED_UNREAD":
        raise OuterPanelError("outer panel is not precommitted unread")
    declared = _sha(value.get("manifest_digest"), name="manifest_digest")
    payload = {key: item for key, item in value.items() if key != "manifest_digest"}
    if declared != sha256_digest(payload):
        raise OuterPanelError("outer panel manifest digest mismatch")
    if value.get("access_contract", {}).get("feedback_to_search") != "FORBIDDEN":
        raise OuterPanelError("outer panel feedback boundary is open")
    if value.get("selection_contract", {}).get(
        "selection_must_precede_outer_access"
    ) is not True:
        raise OuterPanelError("outer access is not post-selection")


__all__ = [
    "KNOWN_V29_ONLINE_INTERACTION_DIGESTS",
    "OUTER_EVALUATION_SEEDS_V31",
    "OUTER_PANEL_REQUIRED_ROLES",
    "OUTER_PANEL_SCHEMA_V31",
    "OuterPanelError",
    "seal_outer_panel_v31",
    "validate_outer_panel_manifest_v31",
]
