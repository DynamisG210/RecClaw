"""Development-only open Meta learner for Research Line vNext F1.

The learner consumes real R1/R2 engineering and TypedResearchEpisode evidence.
It learns pre-outcome research-direction reliability and experiment information
coverage.  Scientific outcomes remain INCONCLUSIVE/NOT_ADJUDICATED and are
never converted into a mechanism reward or a recommendation-performance claim.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_value, sha256_digest


F1_POLICY_MODE = "RESEARCH_LEARNED_VNEXT_DEVELOPMENT_ONLY"
F1_POLICY_VERSION = "research-open-meta-v1.0.0"
F1_POLICY_REF = "policy:research-open-meta-vnext:f1:v1"
F1_REPLAY_SCHEMA = "recclaw.research-line.vnext.open-meta.f1-replay.v1"
F1_POLICY_SCHEMA = "recclaw.research-line.vnext.open-meta.f1-policy.v1"

STATIC_DIRECTION_ORDER = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)
ALLOWED_DIRECTIONS = frozenset(STATIC_DIRECTION_ORDER)
ALLOWED_DIMENSIONS = frozenset(
    {
        "COMPOSITE_MECHANISM",
        "CORE_OBJECTIVE",
        "CORE_RELATION",
        "CORE_REPRESENTATION",
        "CUSTOM_EXECUTABLE_CAPABILITY",
        "INTERACTION_STRUCTURE",
        "MODEL_STRUCTURE",
        "PROPAGATION_MECHANISM",
        "TRAINING_PROCEDURE",
    }
)


class OpenMetaF1Error(RuntimeError):
    """A real-evidence, learner, replay, or activation invariant failed."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OpenMetaF1Error(f"JSON root must be an object: {path}")
    return value


def _attempt_tokens(attempts: object) -> int:
    if not attempts:
        return 0
    values = attempts if isinstance(attempts, list) else [attempts]
    return sum(int(item.get("billed_tokens", 0)) for item in values)


def _episode_semantics(episode: Mapping[str, Any] | None) -> dict[str, Any]:
    if episode is None:
        return {
            "episode_observed": False,
            "evidence_class": None,
            "failure_class": None,
            "mechanism_interpretation": None,
            "mechanism_negative_evidence": None,
        }
    expected = {
        "evidence_class": "INCONCLUSIVE_EXPERIMENT",
        "failure_class": "INCONCLUSIVE",
        "mechanism_interpretation": "NOT_ADJUDICATED",
        "mechanism_negative_evidence": False,
    }
    drift = {
        key: {"expected": expected_value, "observed": episode.get(key)}
        for key, expected_value in expected.items()
        if episode.get(key) != expected_value
    }
    if drift:
        raise OpenMetaF1Error(
            "TypedResearchEpisode scientific semantics drift: "
            + json.dumps(drift, sort_keys=True)
        )
    return {
        "episode_observed": True,
        **expected,
    }


def _training_by_subject(receipt: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for run in receipt["training"]["runs"]:
        grouped[(str(run["side"]), str(run["slot_id"]))][
            str(run["run_kind"])
        ] = run
    return grouped


def _ndcg_delta(runs: Mapping[str, Mapping[str, Any]]) -> float | None:
    control = runs.get("MATCHED_CONTROL")
    candidate = runs.get("CANDIDATE")
    if not control or not candidate:
        return None
    if control.get("exit_status") != "SUCCESS" or candidate.get("exit_status") != "SUCCESS":
        return None
    control_metric = control.get("metrics", {}).get("ndcg@10")
    candidate_metric = candidate.get("metrics", {}).get("ndcg@10")
    if control_metric is None or candidate_metric is None:
        return None
    return float(candidate_metric) - float(control_metric)


def build_f1_replay_dataset(
    *,
    r1_root: Path,
    r2_root: Path,
) -> dict[str, Any]:
    """Build one canonical replay dataset from all real R1/R2 evidence."""

    r1_receipt = _read_json(r1_root / "R1_CANONICAL_RECEIPT.json")
    r2_receipt = _read_json(r2_root / "R2_CANONICAL_RECEIPT.json")
    if (
        r1_receipt.get("status") != "R1_PASS"
        or r1_receipt.get("held_out_reads") != 0
        or r2_receipt.get("status") != "R2_ARCHITECTURE_EFFECT_PASS"
        or r2_receipt.get("held_out_reads") != 0
    ):
        raise OpenMetaF1Error("R1/R2 accepted evidence boundary is not intact")

    rows: list[dict[str, Any]] = []
    r1_runs = _training_by_subject(r1_receipt)
    for side in ("side_a", "side_b"):
        for record in r1_receipt["side_records"][side]:
            slot = str(record["logical_slot_id"])
            spec = _read_json(r1_root / side / "specs" / f"{slot}.json")
            facts = spec["resolution_facts"]
            budget = facts["required_budget"]
            qualification_path = r1_root / side / "qualifications" / f"{slot}.json"
            qualification = _read_json(qualification_path)
            episode_path = r1_root / side / "episodes" / f"{slot}.json"
            episode = _read_json(episode_path) if episode_path.is_file() else None
            runs = r1_runs.get((side, slot), {})
            candidate_run = runs.get("CANDIDATE")
            scientific = _episode_semantics(episode)
            row = {
                "audit_ref": f"r1/{side}/{slot}",
                "campaign_family": "R1",
                "replay_split": (
                    "SEARCH_TRAIN"
                    if side == "side_a"
                    else "DEVELOPMENT_VALIDATION"
                ),
                "direction": str(record["producer_role"]),
                "high_change_dimensions": tuple(
                    sorted(str(value) for value in facts["high_change_dimensions"])
                ),
                "required_implementation_tokens": int(
                    budget["implementation_token_ceiling"]
                ),
                "required_qualification_gpu_minutes": int(
                    budget["qualification_gpu_minutes"]
                ),
                "required_qualification_wall_minutes": int(
                    budget["qualification_wall_minutes"]
                ),
                "resolved_as_search_ready": False,
                "qualification_observed": True,
                "qualification_pass": record["qualification_status"] == "PASS",
                "qualification_failure_class": qualification["receipt"][
                    "failure_class"
                ],
                "qualification_missing_reason": None,
                "experiment_observed": bool(runs),
                "experiment_closed": scientific["episode_observed"],
                "runtime_failure": bool(candidate_run)
                and candidate_run.get("exit_status") != "SUCCESS",
                "proposal_billed_tokens": _attempt_tokens(record["provider_attempts"]),
                "implementation_billed_tokens": _attempt_tokens(
                    record["implementation_provider_attempts"]
                ),
                "candidate_runtime_wall_ms": (
                    int(candidate_run["wall_time_ms"]) if candidate_run else None
                ),
                "ndcg_delta_audit_only": _ndcg_delta(runs),
                "capability_ref": record.get("capability_ref"),
                "selected_for_experiment": bool(runs),
                **scientific,
            }
            rows.append(row)

    r2_episode = r2_receipt.get("episode")
    selected_slot = str(r2_receipt["selection"]["selected_slot"])
    for record in r2_receipt["proposal_records"]:
        slot = str(record["logical_slot_id"])
        spec = _read_json(r2_root / "specs" / f"{slot}.json")
        facts = spec["resolution_facts"]
        budget = facts["required_budget"]
        is_selected = slot == selected_slot
        baseline = r2_receipt["matched_control"]["baseline"] if is_selected else None
        candidate = r2_receipt["matched_control"]["candidate"] if is_selected else None
        runs = (
            {"MATCHED_CONTROL": baseline, "CANDIDATE": candidate}
            if is_selected
            else {}
        )
        scientific = _episode_semantics(r2_episode if is_selected else None)
        row = {
            "audit_ref": f"r2/{slot}",
            "campaign_family": "R2",
            "replay_split": "DEVELOPMENT_VALIDATION",
            "direction": str(record["producer_role"]),
            "high_change_dimensions": tuple(
                sorted(str(value) for value in facts["high_change_dimensions"])
            ),
            "required_implementation_tokens": int(
                budget["implementation_token_ceiling"]
            ),
            "required_qualification_gpu_minutes": int(
                budget["qualification_gpu_minutes"]
            ),
            "required_qualification_wall_minutes": int(
                budget["qualification_wall_minutes"]
            ),
            "resolved_as_search_ready": record.get("resolution") == "SEARCH_READY",
            "qualification_observed": is_selected,
            "qualification_pass": (
                r2_receipt["qualification"]["receipt"]["status"] == "PASS"
                if is_selected
                else None
            ),
            "qualification_failure_class": (
                r2_receipt["qualification"]["receipt"]["failure_class"]
                if is_selected
                else None
            ),
            "qualification_missing_reason": (
                None if is_selected else "NOT_SELECTED_BY_PREFROZEN_SLOT_ORDER"
            ),
            "experiment_observed": is_selected,
            "experiment_closed": scientific["episode_observed"],
            "runtime_failure": bool(candidate)
            and candidate.get("exit_status") != "SUCCESS",
            "proposal_billed_tokens": _attempt_tokens(record["provider_attempts"]),
            "implementation_billed_tokens": (
                int(r2_receipt["implementation_provider_usage"]["billed_tokens"])
                if is_selected
                else 0
            ),
            "candidate_runtime_wall_ms": (
                int(candidate["wall_time_ms"]) if candidate else None
            ),
            "ndcg_delta_audit_only": _ndcg_delta(runs),
            "capability_ref": record.get("resolved_capability_ref"),
            "selected_for_experiment": is_selected,
            **scientific,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda item: sha256_digest(item))
    if len(rows) != 20:
        raise OpenMetaF1Error(f"expected 20 R1/R2 denominator rows, got {len(rows)}")
    if sum(bool(row["episode_observed"]) for row in rows) != 8:
        raise OpenMetaF1Error("expected all seven R1 plus one R2 episodes")
    if any(row["direction"] not in ALLOWED_DIRECTIONS for row in rows):
        raise OpenMetaF1Error("replay contains an unsupported research direction")
    if any(
        not set(row["high_change_dimensions"]).issubset(ALLOWED_DIMENSIONS)
        for row in rows
    ):
        raise OpenMetaF1Error("replay contains an unknown high-change dimension")

    dataset = canonical_value(
        {
            "schema": F1_REPLAY_SCHEMA,
            "dataset_version": "1.0.0",
            "evidence_sources": {
                "r1_receipt": str(r1_root / "R1_CANONICAL_RECEIPT.json"),
                "r2_receipt": str(r2_root / "R2_CANONICAL_RECEIPT.json"),
            },
            "rows": rows,
            "row_count": len(rows),
            "episode_count": 8,
            "qualification_observed_count": sum(
                bool(row["qualification_observed"]) for row in rows
            ),
            "qualification_missingness": {
                "NOT_SELECTED_BY_PREFROZEN_SLOT_ORDER": sum(
                    row["qualification_missing_reason"]
                    == "NOT_SELECTED_BY_PREFROZEN_SLOT_ORDER"
                    for row in rows
                ),
                "R1_QUALIFICATION_FAILURE": sum(
                    row["campaign_family"] == "R1"
                    and row["qualification_pass"] is False
                    for row in rows
                ),
            },
            "held_out_reads": 0,
            "scientific_semantics": {
                "evidence_class": "INCONCLUSIVE_EXPERIMENT",
                "mechanism_interpretation": "NOT_ADJUDICATED",
                "mechanism_negative_evidence": False,
                "outcome_usage": "AUDIT_AND_UNCERTAINTY_ONLY_NOT_MECHANISM_REWARD",
            },
        }
    )
    return {**dataset, "dataset_digest": sha256_digest(dataset)}


def _selected_rows(
    dataset: Mapping[str, Any], splits: Sequence[str]
) -> list[Mapping[str, Any]]:
    allowed = set(splits)
    return [row for row in dataset["rows"] if row["replay_split"] in allowed]


def fit_open_meta_policy(
    dataset: Mapping[str, Any],
    *,
    splits: Sequence[str],
    policy_version: str,
) -> dict[str, Any]:
    """Fit a small empirical-Bayes research policy with current consumers."""

    rows = _selected_rows(dataset, splits)
    if not rows:
        raise OpenMetaF1Error("learner split is empty")
    direction_stats: list[dict[str, Any]] = []
    for direction in STATIC_DIRECTION_ORDER:
        group = [row for row in rows if row["direction"] == direction]
        qualification = [row for row in group if row["qualification_observed"]]
        experiments = [row for row in group if row["experiment_observed"]]
        passed = sum(row["qualification_pass"] is True for row in qualification)
        closed = sum(bool(row["experiment_closed"]) for row in experiments)
        qualification_mean = (passed + 1.0) / (len(qualification) + 2.0)
        closure_mean = (closed + 1.0) / (len(experiments) + 2.0)
        uncertainty = 1.0 / math.sqrt(len(qualification) + 1.0)
        score = 0.55 * qualification_mean + 0.25 * closure_mean + 0.20 * uncertainty
        direction_stats.append(
            {
                "direction": direction,
                "observed_rows": len(group),
                "qualification_observed": len(qualification),
                "qualification_pass": passed,
                "experiment_observed": len(experiments),
                "experiment_closed": closed,
                "qualification_posterior_mean": round(qualification_mean, 12),
                "experiment_closure_posterior_mean": round(closure_mean, 12),
                "uncertainty_bonus": round(uncertainty, 12),
                "score": round(score, 12),
            }
        )
    direction_stats.sort(key=lambda item: (-item["score"], item["direction"]))

    capability_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("capability_ref"):
            capability_rows[str(row["capability_ref"])].append(row)
    capability_stats: list[dict[str, Any]] = []
    for capability_ref, group in capability_rows.items():
        experiments = [row for row in group if row["experiment_observed"]]
        closed = sum(bool(row["experiment_closed"]) for row in experiments)
        failures = sum(bool(row["runtime_failure"]) for row in experiments)
        episodes = sum(bool(row["episode_observed"]) for row in group)
        reliability = (closed + 1.0) / (len(experiments) + 2.0)
        information_value = 1.0 / (episodes + 1.0)
        unresolved_attempt_gap = bool(experiments) and episodes == 0
        score = (
            0.60 * information_value
            + 0.25 * reliability
            + 0.15 * float(unresolved_attempt_gap)
        )
        capability_stats.append(
            {
                "capability_ref": capability_ref,
                "observed_rows": len(group),
                "experiment_observed": len(experiments),
                "experiment_closed": closed,
                "runtime_failures": failures,
                "episode_count": episodes,
                "unresolved_attempt_gap": unresolved_attempt_gap,
                "runtime_reliability_posterior_mean": round(reliability, 12),
                "information_value": round(information_value, 12),
                "score": round(score, 12),
            }
        )
    capability_stats.sort(key=lambda item: (-item["score"], item["capability_ref"]))

    audit_deltas = [
        float(row["ndcg_delta_audit_only"])
        for row in rows
        if row["ndcg_delta_audit_only"] is not None
    ]
    policy = canonical_value(
        {
            "schema": F1_POLICY_SCHEMA,
            "policy_ref": F1_POLICY_REF,
            "policy_version": policy_version,
            "policy_mode": F1_POLICY_MODE,
            "training_dataset_digest": dataset["dataset_digest"],
            "training_splits": tuple(sorted(set(splits))),
            "training_row_count": len(rows),
            "direction_stats": direction_stats,
            "direction_order": tuple(item["direction"] for item in direction_stats),
            "capability_stats": capability_stats,
            "decision_rule": {
                "idea": "EMPIRICAL_BAYES_RELIABILITY_WITH_UNCERTAINTY_ALL_DIRECTIONS_RETAINED",
                "experiment": "UNRESOLVED_ATTEMPT_GAP_THEN_MISSING_EPISODE_INFORMATION_VALUE_AND_RUNTIME_RELIABILITY",
                "candidate_identity_feature": False,
                "origin_feature": False,
                "held_out_feature": False,
                "scientific_metric_reward": False,
            },
            "scientific_audit": {
                "observed_ndcg_delta_count": len(audit_deltas),
                "mean_ndcg_delta": (
                    round(sum(audit_deltas) / len(audit_deltas), 12)
                    if audit_deltas
                    else None
                ),
                "interpretation": "INCONCLUSIVE_NOT_ADJUDICATED_NOT_A_POLICY_REWARD",
            },
            "held_out_reads": 0,
        }
    )
    return {**policy, "policy_digest": sha256_digest(policy)}


def capability_score(policy: Mapping[str, Any], capability_ref: str) -> float:
    for row in policy["capability_stats"]:
        if row["capability_ref"] == capability_ref:
            return float(row["score"])
    # A genuinely unseen registered capability has maximal information value
    # and neutral Beta(1,1) runtime reliability, but no observed unresolved
    # attempt gap.  This keeps known incomplete result chains actionable.
    return 0.60 + 0.25 * 0.5


def rank_search_ready_records(
    policy: Mapping[str, Any], records: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for record in records:
        if (
            record.get("resolution") != "SEARCH_READY"
            or not record.get("resolved_capability_ref")
        ):
            continue
        row = dict(record)
        row["learned_experiment_score"] = round(
            capability_score(policy, str(row["resolved_capability_ref"])), 12
        )
        ranked.append(row)
    ranked.sort(
        key=lambda item: (
            -item["learned_experiment_score"],
            str(item["resolved_capability_ref"]),
            str(item["logical_slot_id"]),
        )
    )
    return ranked


def shadow_evaluate_open_meta(
    dataset: Mapping[str, Any],
    shadow_policy: Mapping[str, Any],
) -> dict[str, Any]:
    validation = _selected_rows(dataset, ("DEVELOPMENT_VALIDATION",))
    direction_outcomes = {
        direction: [
            row
            for row in validation
            if row["direction"] == direction and row["qualification_observed"]
        ]
        for direction in STATIC_DIRECTION_ORDER
    }
    learned_order = tuple(shadow_policy["direction_order"])
    static_first = STATIC_DIRECTION_ORDER[0]
    learned_first = learned_order[0]

    all_capabilities = sorted(
        {
            str(row["capability_ref"])
            for row in dataset["rows"]
            if row.get("capability_ref")
        }
    )
    episode_counts = {
        capability: sum(
            row.get("capability_ref") == capability and row["episode_observed"]
            for row in dataset["rows"]
        )
        for capability in all_capabilities
    }
    static_capability = all_capabilities[0]
    learned_capability = sorted(
        all_capabilities,
        key=lambda ref: (-capability_score(shadow_policy, ref), ref),
    )[0]

    shadow = canonical_value(
        {
            "schema": "recclaw.research-line.vnext.open-meta.f1-shadow.v1",
            "evaluation_rule_frozen_pre_outcome": {
                "idea_control": "R2_FIXED_DIRECTION_ORDER",
                "experiment_control": "CANONICAL_REGISTERED_CAPABILITY_IDENTITY_ORDER",
                "idea_metrics": (
                    "selection_difference",
                    "observed_qualification_rate_by_selected_direction",
                ),
                "experiment_metrics": (
                    "selection_difference",
                    "prior_episode_coverage",
                    "missing_evidence_information_value",
                ),
                "superiority_required_for_development_promotion": False,
                "held_out_allowed": False,
            },
            "split": "DEVELOPMENT_VALIDATION",
            "validation_row_count": len(validation),
            "coverage": {
                "qualification_observed": sum(
                    bool(row["qualification_observed"]) for row in validation
                ),
                "episode_observed": sum(bool(row["episode_observed"]) for row in validation),
                "qualification_missing": sum(
                    not bool(row["qualification_observed"]) for row in validation
                ),
            },
            "idea": {
                "static_direction_order": STATIC_DIRECTION_ORDER,
                "learned_direction_order": learned_order,
                "selection_changed": learned_order != STATIC_DIRECTION_ORDER,
                "static_first_direction": static_first,
                "learned_first_direction": learned_first,
                "static_first_observed_qualification_rate": _observed_rate(
                    direction_outcomes[static_first]
                ),
                "learned_first_observed_qualification_rate": _observed_rate(
                    direction_outcomes[learned_first]
                ),
            },
            "experiment": {
                "static_selected_capability_ref": static_capability,
                "learned_selected_capability_ref": learned_capability,
                "selection_changed": learned_capability != static_capability,
                "static_prior_episode_count": episode_counts[static_capability],
                "learned_prior_episode_count": episode_counts[learned_capability],
                "learned_information_coverage_not_worse": (
                    episode_counts[learned_capability]
                    <= episode_counts[static_capability]
                ),
            },
            "uncertainty": {
                "small_sample": True,
                "counterfactual_outcomes_unobserved": True,
                "policy_superiority_established": False,
                "scientific_effect_established": False,
            },
            "held_out_reads": 0,
        }
    )
    return {**shadow, "shadow_digest": sha256_digest(shadow)}


def _observed_rate(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(
        sum(row["qualification_pass"] is True for row in rows) / len(rows),
        12,
    )


def evaluate_development_promotion(
    dataset: Mapping[str, Any],
    shadow: Mapping[str, Any],
    final_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen engineering-only promotion rule."""

    gates = {
        "all_r1_r2_denominator_rows_consumed": dataset["row_count"] == 20,
        "all_real_typed_episodes_consumed": dataset["episode_count"] == 8,
        "held_out_reads_zero": dataset["held_out_reads"] == 0,
        "scientific_semantics_preserved": dataset["scientific_semantics"] == {
            "evidence_class": "INCONCLUSIVE_EXPERIMENT",
            "mechanism_interpretation": "NOT_ADJUDICATED",
            "mechanism_negative_evidence": False,
            "outcome_usage": "AUDIT_AND_UNCERTAINTY_ONLY_NOT_MECHANISM_REWARD",
        },
        "origin_and_candidate_identity_not_features": (
            final_policy["decision_rule"]["origin_feature"] is False
            and final_policy["decision_rule"]["candidate_identity_feature"] is False
        ),
        "shadow_control_meaningful": (
            shadow["evaluation_rule_frozen_pre_outcome"]["idea_control"]
            == "R2_FIXED_DIRECTION_ORDER"
            and shadow["evaluation_rule_frozen_pre_outcome"]["experiment_control"]
            == "CANONICAL_REGISTERED_CAPABILITY_IDENTITY_ORDER"
        ),
        "learned_behavior_differs_from_static": (
            shadow["idea"]["selection_changed"]
            or shadow["experiment"]["selection_changed"]
        ),
        "information_coverage_not_worse": shadow["experiment"][
            "learned_information_coverage_not_worse"
        ],
        "versioned_policy_real": (
            final_policy["policy_mode"] == F1_POLICY_MODE
            and final_policy["policy_digest"]
            and final_policy["training_row_count"] == 20
        ),
    }
    passed = all(bool(value) for value in gates.values())
    promotion = canonical_value(
        {
            "schema": "recclaw.research-line.vnext.open-meta.f1-promotion.v1",
            "promotion_rule": "F1_DEVELOPMENT_ARCHITECTURE_EFFECT_V1",
            "promotion_rule_frozen_pre_shadow": True,
            "gates": gates,
            "status": (
                "DEVELOPMENT_ONLY_PROMOTION_PASS"
                if passed
                else "DEVELOPMENT_ONLY_PROMOTION_FAIL"
            ),
            "policy_superiority_claim": False,
            "scientific_effect_claim": False,
            "held_out_reads": 0,
        }
    )
    return {**promotion, "promotion_digest": sha256_digest(promotion)}


def build_policy_activation(
    policy: Mapping[str, Any], promotion: Mapping[str, Any], *, campaign_id: str
) -> dict[str, Any]:
    if promotion["status"] != "DEVELOPMENT_ONLY_PROMOTION_PASS":
        raise OpenMetaF1Error("failed policy cannot activate")
    activation = canonical_value(
        {
            "schema": "recclaw.research-line.vnext.open-meta.f1-activation.v1",
            "policy_ref": policy["policy_ref"],
            "policy_digest": policy["policy_digest"],
            "policy_version": policy["policy_version"],
            "policy_mode": policy["policy_mode"],
            "promotion_digest": promotion["promotion_digest"],
            "activation_boundary": "NEXT_FRESH_CAMPAIGN",
            "campaign_id": campaign_id,
            "development_only": True,
            "replaces_static_for_campaign": True,
            "scientific_effect_claim": False,
            "held_out_reads": 0,
        }
    )
    return {**activation, "activation_digest": sha256_digest(activation)}


__all__ = [
    "F1_POLICY_MODE",
    "F1_POLICY_REF",
    "F1_POLICY_VERSION",
    "OpenMetaF1Error",
    "STATIC_DIRECTION_ORDER",
    "build_f1_replay_dataset",
    "build_policy_activation",
    "capability_score",
    "evaluate_development_promotion",
    "fit_open_meta_policy",
    "rank_search_ready_records",
    "shadow_evaluate_open_meta",
]
