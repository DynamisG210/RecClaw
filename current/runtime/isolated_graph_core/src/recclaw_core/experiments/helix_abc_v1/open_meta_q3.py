"""Outcome-aware, three-head evidence projection for Research Line Q3.

This module is a minimal extension of the F1 open-Meta replay path.  It keeps
feasibility, mechanism-information, and development-effect authority separate
before any learner or acquisition rule can consume the evidence.  Candidate
identity is retained only for denominator audit and is never part of a head's
model input.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import bytes_sha256, canonical_value, sha256_digest


Q3_PROJECTION_SCHEMA = "recclaw.research-line.q3-denominator-projection.v1"
Q3_AUTHORITY_MATRIX_SCHEMA = "recclaw.research-line.q3-head-authority-matrix.v1"
Q3_ACQUISITION_MANIFEST_SCHEMA = "recclaw.research-line.q3-acquisition-manifest.v2"
Q3_SELECTION_SCORE_TIE_TOLERANCE = 1e-12
Q5_FEASIBILITY_SHRINKAGE_SUPPORT_THRESHOLD = 4.0
Q5_FROZEN_FEASIBILITY_PRIOR = {
    "source": "F1_STATIC_BETA_1_1_PRIOR",
    "mean": 0.5,
}
Q5_STAGE_CONDITIONAL_FEASIBILITY_STAGES = (
    "MATERIALIZE",
    "CONSTRUCT",
    "QUALIFY",
    "RESOURCE_ADMITTED",
    "FULL_EPISODE",
)

MECHANISM_STATE_ORDER = (
    "NOT_ASSESSED",
    "INACTIVE",
    "ACTIVE_SUPPORTED",
    "ACTIVE_CONTRADICTED",
    "NON_IDENTIFIABLE",
)
MECHANISM_STATES = frozenset(MECHANISM_STATE_ORDER)

HEAD_AUTHORITY_MATRIX = canonical_value(
    {
        "schema": Q3_AUTHORITY_MATRIX_SCHEMA,
        "development_only": True,
        "held_out_reads": 0,
        "scientific_effect_claim": False,
        "feasibility_authority": {
            "target": "COMPLETION_PROBABILITY_AND_RESOURCE_COST",
            "allowed": (
                "SUCCESS",
                "RIGHT_CENSORED",
                "RESOURCE_CENSORED",
                "RESOURCE_DEFERRED",
                "OOM",
                "TIMEOUT",
                "ACCELERATOR_ERROR",
                "RUNTIME_FAILURE",
                "QUALIFICATION_FAILURE",
                "PROTOCOL_NO_CHECKPOINT",
            ),
            "forbidden": (
                "NDCG_AS_COMPLETION_LABEL",
                "MECHANISM_STATE_AS_COMPLETION_LABEL",
                "CANDIDATE_IDENTITY_AS_FEATURE",
            ),
        },
        "mechanism_information_authority": {
            "target": (
                "PROBABILITY_OF_IDENTIFIABLE_MECHANISM_EVIDENCE_"
                "UNDER_THE_DECLARED_PROBE_DESIGN"
            ),
            "allowed": MECHANISM_STATE_ORDER,
            "forbidden": (
                "RESOURCE_STATUS_AS_MECHANISM_EFFECT",
                "NDCG_AS_MECHANISM_LABEL",
                "PROTOCOL_MISSINGNESS_AS_NEGATIVE_MECHANISM_LABEL",
                "CANDIDATE_IDENTITY_AS_FEATURE",
            ),
        },
        "effect_authority": {
            "target": "PARENT_RELATIVE_DEVELOPMENT_EFFECT",
            "allowed": ("FULL_COMPARABLE_FRESH_MATCHED_DEVELOPMENT_EPISODE",),
            "forbidden": (
                "RESOURCE_CENSORED",
                "RESOURCE_DEFERRED",
                "RIGHT_CENSORED",
                "PROTOCOL_MISSINGNESS",
                "PROBE_ONLY",
                "QUALIFICATION_ONLY",
                "NOT_ASSESSED",
                "NON_IDENTIFIABLE_WITHOUT_MATCHED_EFFECT",
            ),
            "uncertainty_rule": (
                "INCONCLUSIVE_OR_NOT_ADJUDICATED_EPISODES_RETAIN_OBSERVED_"
                "EFFECT_WITH_EXPLICIT_REDUCED_WEIGHT_AND_NO_SCIENTIFIC_CLAIM"
            ),
        },
        "q2_binding": {
            "feasibility_update": (
                "RESOURCE_CENSORED_AND_PROTOCOL_NO_CHECKPOINT_COMPONENTS_ONLY"
            ),
            "mechanism_information_update": (
                "NON_IDENTIFIABLE_LABEL_WITH_GENERAL_PROBE_DESIGN_FEATURES"
            ),
            "effect_update": False,
            "resource_status_changes_mechanism_effect": False,
        },
        "separation_invariant": (
            "EACH_HEAD_RECEIVES_ONLY_ITS_OWN_CANONICAL_INPUT_PROJECTION; "
            "AUDIT_IDENTITY_AND_OTHER_HEAD_FIELDS_ARE_EXCLUDED"
        ),
    }
)


class OpenMetaQ3Error(RuntimeError):
    """An evidence-authority or denominator invariant failed."""


def _authority(allowed: bool, reason: str) -> dict[str, Any]:
    return {"allowed": allowed, "reason": reason}


def _projection_row(
    *,
    row_id: str,
    source_stage: str,
    group_id: str,
    feasibility_input: Mapping[str, Any] | None,
    feasibility_reason: str,
    mechanism_input: Mapping[str, Any] | None,
    mechanism_reason: str,
    effect_input: Mapping[str, Any] | None,
    effect_reason: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    value = canonical_value(
        {
            "row_id": row_id,
            "source_stage": source_stage,
            "group_id": group_id,
            "head_authority": {
                "feasibility": _authority(
                    feasibility_input is not None, feasibility_reason
                ),
                "mechanism_information": _authority(
                    mechanism_input is not None, mechanism_reason
                ),
                "effect": _authority(effect_input is not None, effect_reason),
            },
            "head_inputs": {
                "feasibility": feasibility_input,
                "mechanism_information": mechanism_input,
                "effect": effect_input,
            },
            "audit_only_not_model_input": dict(audit),
        }
    )
    return {**value, "row_digest": sha256_digest(value)}


def _numeric_values(value: object, names: frozenset[str]) -> list[float]:
    found: list[float] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in names and isinstance(item, (int, float)) and not isinstance(
                item, bool
            ):
                number = float(item)
                if math.isfinite(number):
                    found.append(number)
            found.extend(_numeric_values(item, names))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_numeric_values(item, names))
    return found


def _first_value(value: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if value.get(name) is not None:
            return value[name]
    return None


def _failure_family(status: str, error_type: object) -> str | None:
    if status == "SUCCESS":
        return None
    text = f"{status} {error_type or ''}".upper()
    if "OOM" in text or "OUTOFMEMORY" in text or "OUT_OF_MEMORY" in text:
        return "OOM"
    if "ACCELERATOR" in text or "CUDA" in text:
        return "ACCELERATOR_ERROR"
    if "CENSOR" in text:
        return "RESOURCE_CENSORED"
    if "DEFER" in text:
        return "RESOURCE_DEFERRED"
    if "TIMEOUT" in text or "DEADLINE" in text:
        return "TIMEOUT"
    if "QUALIFICATION" in text:
        return "QUALIFICATION_FAILURE"
    return "RUNTIME_FAILURE"


def _normalize_resource_status(run: Mapping[str, Any]) -> str:
    raw = str(
        _first_value(
            run,
            ("exit_status", "status", "feasibility_status", "resource_disposition"),
        )
        or "RUNTIME_FAILURE"
    ).upper()
    if raw == "NOT_RUN_RESOURCE_DEFERRED":
        return "RESOURCE_DEFERRED"
    if raw in {"PASS", "COMPLETED", "COMPLETED_MATCHED_PAIR"}:
        return "SUCCESS"
    return raw


def _resource_head_input(
    run: Mapping[str, Any],
    *,
    resource_stage: str,
    physical_run: bool = True,
    protocol_missingness: str | None = None,
) -> dict[str, Any]:
    status = _normalize_resource_status(run)
    telemetry = run.get("resource_telemetry") or run.get("throughput_and_memory") or {}
    throughput = _numeric_values(telemetry, frozenset({"batches_per_second"}))
    completed_batches = _numeric_values(
        telemetry,
        frozenset(
            {
                "completed_train_batches",
                "completed_batches",
                "batch_count",
            }
        ),
    )
    peak_allocated = _numeric_values(
        telemetry, frozenset({"peak_allocated_mib"})
    )
    peak_reserved = _numeric_values(telemetry, frozenset({"peak_reserved_mib"}))
    wall_time_ms = _first_value(run, ("wall_time_ms", "elapsed_ms"))
    deadline_seconds = _first_value(
        run, ("resource_deadline_seconds", "deadline_seconds", "watchdog_seconds")
    )
    device = run.get("device_evidence") or {}
    error = run.get("failure") or {}
    error_type = run.get("worker_error_type") or error.get("error_type")
    censored = status in {
        "RIGHT_CENSORED",
        "RESOURCE_CENSORED",
        "RESOURCE_DEFERRED",
    }
    return canonical_value(
        {
            "resource_stage": resource_stage,
            "status": status,
            "completion_label": 1 if status == "SUCCESS" else 0,
            "observation_weight": 0.5 if censored else 1.0,
            "physical_run": physical_run,
            "censored": censored,
            "failure_family": _failure_family(status, error_type),
            "wall_time_ms": int(wall_time_ms) if wall_time_ms is not None else None,
            "censor_lower_bound_ms": (
                int(wall_time_ms) if censored and wall_time_ms is not None else None
            ),
            "resource_deadline_seconds": (
                int(deadline_seconds) if deadline_seconds is not None else None
            ),
            "prefix_batches_per_second": (
                round(sum(throughput) / len(throughput), 12)
                if throughput
                else None
            ),
            "completed_prefix_batches": (
                int(max(completed_batches)) if completed_batches else None
            ),
            "peak_allocated_mib": (
                round(max(peak_allocated), 12) if peak_allocated else None
            ),
            "peak_reserved_mib": (
                round(max(peak_reserved), 12) if peak_reserved else None
            ),
            "runtime_identity": {
                "runtime_binding_digest": run.get("runtime_binding_digest"),
                "runtime_release_digest": run.get("runtime_release_digest"),
                "device_name": device.get("cuda_device_name"),
                "torch_cuda_version": device.get("torch_cuda_version"),
            },
            "protocol_missingness": protocol_missingness,
        }
    )


def project_f1_replay_rows(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project the complete accepted R1/R2 denominator through Q3 authority."""

    if dataset.get("held_out_reads") != 0:
        raise OpenMetaQ3Error("R1/R2 replay attempted held-out use")
    rows: list[dict[str, Any]] = []
    for source in dataset["rows"]:
        feasibility_input: dict[str, Any] | None = None
        feasibility_reason = "NO_OBSERVED_RESOURCE_OR_COMPLETION_OUTCOME"
        if source["experiment_observed"]:
            status = (
                "SUCCESS"
                if source["experiment_closed"] and not source["runtime_failure"]
                else "RUNTIME_FAILURE"
            )
            feasibility_input = _resource_head_input(
                {
                    "exit_status": status,
                    "wall_time_ms": source.get("candidate_runtime_wall_ms"),
                },
                resource_stage="DEVELOPMENT_EXPERIMENT",
            )
            feasibility_reason = "OBSERVED_DEVELOPMENT_EXPERIMENT_COMPLETION"
        elif source["qualification_observed"]:
            status = (
                "SUCCESS" if source["qualification_pass"] else "QUALIFICATION_FAILURE"
            )
            feasibility_input = _resource_head_input(
                {"exit_status": status}, resource_stage="QUALIFICATION"
            )
            feasibility_reason = "OBSERVED_QUALIFICATION_COMPLETION"

        delta = source.get("ndcg_delta_audit_only")
        effect_allowed = bool(
            source.get("episode_observed")
            and source.get("experiment_closed")
            and not source.get("runtime_failure")
            and delta is not None
        )
        effect_input = (
            canonical_value(
                {
                    "effect_target": "PARENT_RELATIVE_DEVELOPMENT_NDCG_AT_10",
                    "parent_relative_effect": float(delta),
                    "evidence_weight": 0.5,
                    "evidence_class": source.get("evidence_class"),
                    "mechanism_interpretation": source.get(
                        "mechanism_interpretation"
                    ),
                    "comparability": "FULL_MATCHED_FRESH_DEVELOPMENT_EPISODE",
                    "research_features": {
                        "direction": source["direction"],
                        "high_change_dimensions": source["high_change_dimensions"],
                    },
                }
            )
            if effect_allowed
            else None
        )
        rows.append(
            _projection_row(
                row_id=str(source["audit_ref"]),
                source_stage=str(source["campaign_family"]),
                group_id=str(source.get("capability_ref") or source["audit_ref"]),
                feasibility_input=feasibility_input,
                feasibility_reason=feasibility_reason,
                mechanism_input=None,
                mechanism_reason="NO_AUTHORIZED_MECHANISM_PROBE_LABEL",
                effect_input=effect_input,
                effect_reason=(
                    "FULL_COMPARABLE_FRESH_MATCHED_DEVELOPMENT_EPISODE"
                    if effect_allowed
                    else "NO_COMPLETE_COMPARABLE_MATCHED_EFFECT"
                ),
                audit={
                    "candidate_identity_feature": False,
                    "source_audit_ref": source["audit_ref"],
                    "selected_for_experiment": source["selected_for_experiment"],
                },
            )
        )
    return rows


def _q0_runs(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = {
        str(name): value["training_run"]
        for name, value in receipt["arm_records"].items()
    }
    runs["matched_bpr_control"] = receipt["evaluation"]["matched_bpr_control"]
    outcomes = receipt["evaluation"].get("outcomes", {})
    for name, run in tuple(runs.items()):
        outcome = outcomes.get(name, {})
        if outcome.get("resource_censored"):
            runs[name] = {**run, "exit_status": "RESOURCE_CENSORED"}
    return runs


def _resource_receipt_runs(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    if "arm_records" in receipt:
        return _q0_runs(receipt)
    if "probe_runs" in receipt:
        return receipt["probe_runs"]
    if "fresh_results" in receipt:
        return receipt["fresh_results"]
    raise OpenMetaQ3Error("unsupported accepted resource receipt")


def project_resource_receipt_rows(
    source_stage: str, receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Project every arm in one accepted Q0/Q0R/Q0R2 denominator."""

    if receipt.get("held_out_reads") != 0:
        raise OpenMetaQ3Error(f"{source_stage} attempted held-out use")
    rows = []
    for arm, run in _resource_receipt_runs(receipt).items():
        feasibility_input = _resource_head_input(
            run,
            resource_stage=source_stage,
            physical_run=bool(run.get("physical_run_executed", True)),
        )
        rows.append(
            _projection_row(
                row_id=f"{source_stage.lower()}/{arm}",
                source_stage=source_stage,
                group_id=f"resource-family/{arm}",
                feasibility_input=feasibility_input,
                feasibility_reason="RESOURCE_OR_COMPLETION_AUTHORITY_ONLY",
                mechanism_input=None,
                mechanism_reason="RESOURCE_STATUS_HAS_NO_MECHANISM_AUTHORITY",
                effect_input=None,
                effect_reason="RESOURCE_ONLY_EVIDENCE_CANNOT_UPDATE_EFFECT",
                audit={"arm": arm, "candidate_identity_feature": False},
            )
        )
    return rows


def project_f1_outcome_rows(
    f1_receipt: Mapping[str, Any], closure_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Preserve F1 failures, deferred disposition, and completed matched pair."""

    if f1_receipt.get("held_out_reads") != 0 or closure_receipt.get(
        "held_out_reads"
    ) != 0:
        raise OpenMetaQ3Error("F1 evidence attempted held-out use")
    rows: list[dict[str, Any]] = []
    initial_pair = {
        "matched_bpr_control": f1_receipt["matched_control"]["baseline"],
        "selected_candidate": f1_receipt["matched_control"]["candidate"],
    }
    for arm, run in initial_pair.items():
        rows.append(
            _projection_row(
                row_id=f"f1-initial/{arm}",
                source_stage="F1_INITIAL",
                group_id="f1-selected-mechanism-family",
                feasibility_input=_resource_head_input(
                    run, resource_stage="F1_INITIAL_DEVELOPMENT"
                ),
                feasibility_reason="OBSERVED_F1_INITIAL_COMPLETION",
                mechanism_input=None,
                mechanism_reason="NO_AUTHORIZED_MECHANISM_PROBE_LABEL",
                effect_input=None,
                effect_reason="INCOMPLETE_MATCHED_PAIR_CANNOT_UPDATE_EFFECT",
                audit={"arm": arm, "candidate_identity_feature": False},
            )
        )

    closure_runs = closure_receipt["matched_full_runs"]
    control = closure_runs["matched_bpr_control"]
    realization = closure_runs["resource_compatible_realization"]
    control_ndcg = float(control["metrics"]["ndcg@10"])
    realization_ndcg = float(realization["metrics"]["ndcg@10"])
    episode = closure_receipt["episode"]
    for arm, run in closure_runs.items():
        effect_input = None
        effect_reason = "CONTROL_OR_NON_EFFECT_ROW"
        if arm == "resource_compatible_realization":
            if (
                control.get("exit_status") != "SUCCESS"
                or run.get("exit_status") != "SUCCESS"
                or not episode.get("experiment_executed")
            ):
                raise OpenMetaQ3Error("F1 closure is not a complete matched episode")
            effect_input = canonical_value(
                {
                    "effect_target": "PARENT_RELATIVE_DEVELOPMENT_NDCG_AT_10",
                    "parent_relative_effect": realization_ndcg - control_ndcg,
                    "control_metric": control_ndcg,
                    "candidate_metric": realization_ndcg,
                    "evidence_weight": 0.5,
                    "evidence_class": episode["evidence_class"],
                    "mechanism_interpretation": episode["mechanism_interpretation"],
                    "comparability": "FULL_MATCHED_FRESH_DEVELOPMENT_EPISODE",
                    "research_features": {
                        "mechanism_probe_state": "NOT_ADJUDICATED",
                        "resource_compatible_realization": True,
                    },
                }
            )
            effect_reason = "FULL_COMPARABLE_FRESH_MATCHED_DEVELOPMENT_EPISODE"
        rows.append(
            _projection_row(
                row_id=f"f1-resource-compatible/{arm}",
                source_stage="F1_RESOURCE_COMPATIBLE",
                group_id="f1-selected-mechanism-family",
                feasibility_input=_resource_head_input(
                    run, resource_stage="F1_RESOURCE_COMPATIBLE_DEVELOPMENT"
                ),
                feasibility_reason="OBSERVED_F1_RESOURCE_COMPATIBLE_COMPLETION",
                mechanism_input=None,
                mechanism_reason="EPISODE_IS_NOT_A_MECHANISM_PROBE",
                effect_input=effect_input,
                effect_reason=effect_reason,
                audit={"arm": arm, "candidate_identity_feature": False},
            )
        )

    deferred = str(closure_receipt["original_sealed_candidate_disposition"])
    rows.append(
        _projection_row(
            row_id="f1-original-sealed/resource-disposition",
            source_stage="F1_ORIGINAL_SEALED_DISPOSITION",
            group_id="f1-selected-mechanism-family",
            feasibility_input=_resource_head_input(
                {"status": deferred},
                resource_stage="F1_ORIGINAL_SEALED_DISPOSITION",
                physical_run=False,
            ),
            feasibility_reason="RESOURCE_DEFERRED_DISPOSITION_ONLY",
            mechanism_input=None,
            mechanism_reason="RESOURCE_DEFERRED_HAS_NO_MECHANISM_AUTHORITY",
            effect_input=None,
            effect_reason="RESOURCE_DEFERRED_CANNOT_UPDATE_EFFECT",
            audit={
                "original_sealed_candidate_disposition": deferred,
                "candidate_identity_feature": False,
            },
        )
    )
    return rows


def project_q2_evidence_row(
    q3_package: Mapping[str, Any],
    q2_result: Mapping[str, Any],
    resource_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Project Q2's resource and mechanism components without effect leakage."""

    if any(
        value.get("held_out_reads") != 0
        for value in (q3_package, q2_result, resource_result)
    ):
        raise OpenMetaQ3Error("Q2 evidence attempted held-out use")
    state = str(q3_package["mechanism_state"])
    if state not in MECHANISM_STATES:
        raise OpenMetaQ3Error(f"unsupported mechanism state: {state}")
    evidence = q2_result["mechanism_evidence"]
    target = evidence["target_conditioning"]
    parent = evidence["mechanism_off_parent_equivalence"]
    routing = evidence["routing_and_propagation"]
    loss = evidence["loss_participation"]
    discrimination = evidence["discriminative_prediction"]
    mechanism_input = canonical_value(
        {
            "mechanism_state": state,
            "identifiable_information_label": (
                1
                if state in {"INACTIVE", "ACTIVE_SUPPORTED", "ACTIVE_CONTRADICTED"}
                else 0
                if state == "NON_IDENTIFIABLE"
                else None
            ),
            "probe_design_features": {
                "declared_parent_surface_equivalence_verified": (
                    parent["status"] == "PASS"
                ),
                "target_conditioning_observed": bool(
                    target.get("target_input_observed_in_gate")
                ),
                "structural_routing_participation_observed": (
                    routing["status"] == "PASS" and loss["status"] == "PASS"
                ),
                "checkpoint_available_for_discriminative_probe": (
                    discrimination["assessment"]
                    != "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT"
                ),
                "full_ablation_executed": bool(
                    q3_package.get("full_ablation_executed", False)
                ),
            },
            "probe_cost": {
                "wall_time_ms": sum(
                    int(section.get("wall_time_ms", 0))
                    for section in resource_result["throughput_and_memory"].values()
                ),
                "resource_probe_executions": int(
                    q2_result["resource_probe_executions"]
                ),
            },
            "authority": "REAL_Q2_PROBE_NON_EFFECT",
        }
    )
    feasibility_input = _resource_head_input(
        {
            **resource_result,
            "status": q3_package["resource_status"],
            "throughput_and_memory": resource_result["throughput_and_memory"],
        },
        resource_stage="Q2_MECHANISM_PROBE_RESOURCE",
        protocol_missingness=str(q3_package["protocol_consumer_missingness"]),
    )
    return _projection_row(
        row_id="q2/selected-mechanism-probe",
        source_stage="Q2",
        group_id="target-conditioned-support-mechanism-family",
        feasibility_input=feasibility_input,
        feasibility_reason="Q2_RESOURCE_CENSORED_AND_PROTOCOL_MISSINGNESS_ONLY",
        mechanism_input=mechanism_input,
        mechanism_reason="Q2_REAL_PROBE_MECHANISM_STATE",
        effect_input=None,
        effect_reason="Q2_MECHANISM_EFFECT_UPDATE_FORBIDDEN",
        audit={
            "candidate_identity_feature": False,
            "mechanism_effect_update_allowed": q3_package[
                "mechanism_effect_update_allowed"
            ],
            "q2_physical_result_sha256": q3_package[
                "q2_physical_result_sha256"
            ],
        },
    )


def build_q3_denominator_projection(
    *,
    f1_replay: Mapping[str, Any],
    resource_receipts: Sequence[tuple[str, Mapping[str, Any]]],
    f1_receipt: Mapping[str, Any],
    f1_closure_receipt: Mapping[str, Any],
    q3_package: Mapping[str, Any],
    q2_result: Mapping[str, Any],
    q2_resource_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and seal one all-denominator, authority-annotated projection."""

    rows = project_f1_replay_rows(f1_replay)
    for source_stage, receipt in resource_receipts:
        rows.extend(project_resource_receipt_rows(source_stage, receipt))
    rows.extend(project_f1_outcome_rows(f1_receipt, f1_closure_receipt))
    rows.append(project_q2_evidence_row(q3_package, q2_result, q2_resource_result))
    rows.sort(key=lambda row: row["row_id"])
    row_ids = [row["row_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise OpenMetaQ3Error("projection row identities are not unique")
    source_counts = Counter(row["source_stage"] for row in rows)
    head_counts = {
        head: sum(
            bool(row["head_authority"][head]["allowed"]) for row in rows
        )
        for head in ("feasibility", "mechanism_information", "effect")
    }
    payload = canonical_value(
        {
            "schema": Q3_PROJECTION_SCHEMA,
            "dataset_version": "1.0.0",
            "development_only": True,
            "held_out_reads": 0,
            "scientific_effect_claim": False,
            "authority_matrix_digest": sha256_digest(HEAD_AUTHORITY_MATRIX),
            "row_count": len(rows),
            "source_counts": dict(sorted(source_counts.items())),
            "head_update_counts": head_counts,
            "negative_evidence_preserved": {
                "resource_censored_count": sum(
                    row["head_inputs"]["feasibility"] is not None
                    and row["head_inputs"]["feasibility"]["status"]
                    == "RESOURCE_CENSORED"
                    for row in rows
                ),
                "resource_deferred_count": sum(
                    row["head_inputs"]["feasibility"] is not None
                    and row["head_inputs"]["feasibility"]["status"]
                    == "RESOURCE_DEFERRED"
                    for row in rows
                ),
                "runtime_failure_count": sum(
                    row["head_inputs"]["feasibility"] is not None
                    and row["head_inputs"]["feasibility"]["failure_family"]
                    == "RUNTIME_FAILURE"
                    for row in rows
                ),
                "q2_non_identifiable_count": sum(
                    row["head_inputs"]["mechanism_information"] is not None
                    and row["head_inputs"]["mechanism_information"][
                        "mechanism_state"
                    ]
                    == "NON_IDENTIFIABLE"
                    for row in rows
                ),
            },
            "rows": rows,
        }
    )
    return {**payload, "projection_digest": sha256_digest(payload)}


def _beta_posterior(
    observations: Iterable[tuple[int, float]],
) -> dict[str, Any]:
    alpha = 1.0
    beta = 1.0
    count = 0
    effective_count = 0.0
    for label, weight in observations:
        if label not in {0, 1} or weight <= 0:
            raise OpenMetaQ3Error("invalid Beta-Binomial observation")
        alpha += weight * label
        beta += weight * (1 - label)
        count += 1
        effective_count += weight
    mean = alpha / (alpha + beta)
    variance = alpha * beta / (
        (alpha + beta) ** 2 * (alpha + beta + 1.0)
    )
    standard_deviation = math.sqrt(variance)
    return canonical_value(
        {
            "prior": "BETA_1_1",
            "observation_count": count,
            "effective_observation_count": round(effective_count, 12),
            "alpha": round(alpha, 12),
            "beta": round(beta, 12),
            "posterior_mean": round(mean, 12),
            "posterior_standard_deviation": round(standard_deviation, 12),
            "approximate_95_interval": (
                round(max(0.0, mean - 1.96 * standard_deviation), 12),
                round(min(1.0, mean + 1.96 * standard_deviation), 12),
            ),
        }
    )


def _support_shrinkage(
    posterior: Mapping[str, Any],
    *,
    prior_mean: float,
    support_threshold: float = Q5_FEASIBILITY_SHRINKAGE_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """Shrink sparse predictions toward the frozen lane prior."""

    effective_count = float(posterior["effective_observation_count"])
    support_weight = min(1.0, max(0.0, effective_count / support_threshold))
    shrunk_mean = support_weight * float(posterior["posterior_mean"]) + (
        1.0 - support_weight
    ) * prior_mean
    return canonical_value(
        {
            **posterior,
            "frozen_prior_mean": round(prior_mean, 12),
            "support_threshold": round(support_threshold, 12),
            "support_weight": round(support_weight, 12),
            "shrunk_posterior_mean": round(shrunk_mean, 12),
        }
    )


def project_stage_conditional_feasibility(
    observations: Sequence[Mapping[str, Any]],
    *,
    support_threshold: float = Q5_FEASIBILITY_SHRINKAGE_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """Project the existing feasibility lane into pre-decision stage factors."""

    if not observations:
        raise OpenMetaQ3Error("stage feasibility projection received no observations")
    rows_by_stage: dict[str, list[tuple[int, float]]] = {}
    features_by_stage: dict[str, set[str]] = {}
    for observation in observations:
        if observation.get("held_out_reads", 0) != 0:
            raise OpenMetaQ3Error("stage feasibility projection attempted held-out use")
        stages = observation.get("stages")
        if not isinstance(stages, Mapping):
            raise OpenMetaQ3Error("stage feasibility observation lacks stage rows")
        for stage, value in stages.items():
            stage_name = str(stage)
            if stage_name not in Q5_STAGE_CONDITIONAL_FEASIBILITY_STAGES:
                raise OpenMetaQ3Error(f"unknown feasibility stage: {stage_name}")
            if not isinstance(value, Mapping):
                raise OpenMetaQ3Error("stage feasibility row must be an object")
            label = value.get("completion_label")
            if label not in {0, 1}:
                raise OpenMetaQ3Error(
                    "stage feasibility completion_label must be 0 or 1"
                )
            weight = float(value.get("observation_weight", 1.0))
            if not math.isfinite(weight) or weight <= 0.0:
                raise OpenMetaQ3Error("stage feasibility observation weight is invalid")
            rows_by_stage.setdefault(stage_name, []).append((int(label), weight))
            visible = value.get("features_visible_before_stage", ())
            if not isinstance(visible, (list, tuple)):
                raise OpenMetaQ3Error("stage feature visibility must be a sequence")
            features_by_stage.setdefault(stage_name, set()).update(
                str(item) for item in visible
            )
    stage_stats = []
    for stage in Q5_STAGE_CONDITIONAL_FEASIBILITY_STAGES:
        rows = rows_by_stage.get(stage, [])
        observed = bool(rows)
        posterior = _support_shrinkage(
            _beta_posterior(rows),
            prior_mean=float(Q5_FROZEN_FEASIBILITY_PRIOR["mean"]),
            support_threshold=support_threshold,
        )
        stage_stats.append(
            {
                "stage": stage,
                "input_count": len(rows),
                "observed": observed,
                "observation_status": (
                    "OBSERVED" if observed else "FROZEN_PRIOR_NO_OBSERVATION"
                ),
                "conditional_probability": posterior["shrunk_posterior_mean"],
                "posterior": posterior,
                "features_visible_before_stage": tuple(
                    sorted(features_by_stage.get(stage, set()))
                ),
            }
        )
    full_probability = math.prod(
        float(value["conditional_probability"]) for value in stage_stats
    )
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q5-stage-conditional-feasibility-projection.v1",
            "authority_lane": "FEASIBILITY_COMPLETION_HEAD",
            "factorization": (
                "P(FULL_EPISODE)=P(MATERIALIZE)*P(CONSTRUCT|MATERIALIZE)*"
                "P(QUALIFY|CONSTRUCT)*P(RESOURCE_ADMITTED|QUALIFY)*"
                "P(FULL_EPISODE|RESOURCE_ADMITTED)"
            ),
            "stage_stats": stage_stats,
            "full_episode_probability": round(full_probability, 12),
            "frozen_prior": Q5_FROZEN_FEASIBILITY_PRIOR,
            "support_shrinkage": {
                "method": "LINEAR_EFFECTIVE_COUNT_TO_FROZEN_PRIOR",
                "support_threshold": round(support_threshold, 12),
            },
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "projection_digest": sha256_digest(payload)}


def _fit_feasibility_head(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inputs = [
        row["head_inputs"]["feasibility"]
        for row in rows
        if row["head_authority"]["feasibility"]["allowed"]
    ]
    observations = [
        (int(value["completion_label"]), float(value["observation_weight"]))
        for value in inputs
    ]
    global_posterior = _support_shrinkage(
        _beta_posterior(observations),
        prior_mean=float(Q5_FROZEN_FEASIBILITY_PRIOR["mean"]),
    )
    stage_stats = []
    for stage in sorted({str(value["resource_stage"]) for value in inputs}):
        group = [value for value in inputs if value["resource_stage"] == stage]
        posterior = _support_shrinkage(
            _beta_posterior(
                (int(value["completion_label"]), float(value["observation_weight"]))
                for value in group
            ),
            prior_mean=float(Q5_FROZEN_FEASIBILITY_PRIOR["mean"]),
        )
        successful_costs = sorted(
            int(value["wall_time_ms"])
            for value in group
            if value["completion_label"] == 1 and value["wall_time_ms"] is not None
        )
        censor_bounds = sorted(
            int(value["censor_lower_bound_ms"])
            for value in group
            if value["censor_lower_bound_ms"] is not None
        )
        stage_stats.append(
            {
                "resource_stage": stage,
                "posterior": posterior,
                "feature_contribution_vs_beta_prior": round(
                    float(posterior["shrunk_posterior_mean"])
                    - float(Q5_FROZEN_FEASIBILITY_PRIOR["mean"]),
                    12,
                ),
                "successful_cost_interval_ms": (
                    (min(successful_costs), max(successful_costs))
                    if successful_costs
                    else None
                ),
                "censored_cost_lower_bound_interval_ms": (
                    (min(censor_bounds), max(censor_bounds))
                    if censor_bounds
                    else None
                ),
            }
        )
    head = canonical_value(
        {
            "head": "FEASIBILITY_COMPLETION",
            "method": "CENSOR_WEIGHTED_BETA_BINOMIAL_BY_RESOURCE_STAGE",
            "stage_conditional_projection": (
                "P(MATERIALIZE)*P(CONSTRUCT|MATERIALIZE)*P(QUALIFY|CONSTRUCT)*"
                "P(RESOURCE_ADMITTED|QUALIFY)*P(FULL_EPISODE|RESOURCE_ADMITTED)"
            ),
            "frozen_prior": Q5_FROZEN_FEASIBILITY_PRIOR,
            "support_shrinkage": {
                "method": "LINEAR_EFFECTIVE_COUNT_TO_FROZEN_PRIOR",
                "support_threshold": Q5_FEASIBILITY_SHRINKAGE_SUPPORT_THRESHOLD,
            },
            "authority_matrix_digest": sha256_digest(HEAD_AUTHORITY_MATRIX),
            "input_count": len(inputs),
            "global_posterior": global_posterior,
            "stage_stats": stage_stats,
            "failure_family_counts": dict(
                sorted(
                    Counter(
                        value["failure_family"]
                        for value in inputs
                        if value["failure_family"] is not None
                    ).items()
                )
            ),
            "cost_interpretation": (
                "SUCCESS_COSTS_ARE_OBSERVED_INTERVALS; CENSORED_COSTS_ARE_"
                "LOWER_BOUNDS; RESOURCE_DEFERRED_HAS_NO_FABRICATED_WALL_TIME"
            ),
            "candidate_identity_feature": False,
            "effect_metric_feature": False,
        }
    )
    return {**head, "head_digest": sha256_digest(head)}


def _fit_mechanism_information_head(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    inputs = [
        row["head_inputs"]["mechanism_information"]
        for row in rows
        if row["head_authority"]["mechanism_information"]["allowed"]
    ]
    labeled = [
        value for value in inputs if value["identifiable_information_label"] is not None
    ]
    global_posterior = _beta_posterior(
        (int(value["identifiable_information_label"]), 1.0) for value in labeled
    )
    feature_stats = []
    feature_names = sorted(
        {
            name
            for value in labeled
            for name in value["probe_design_features"]
        }
    )
    for feature in feature_names:
        for feature_value in (False, True):
            group = [
                value
                for value in labeled
                if bool(value["probe_design_features"].get(feature))
                is feature_value
            ]
            if not group:
                continue
            posterior = _beta_posterior(
                (int(value["identifiable_information_label"]), 1.0)
                for value in group
            )
            feature_stats.append(
                {
                    "feature": feature,
                    "value": feature_value,
                    "posterior": posterior,
                    "feature_contribution_vs_beta_prior": round(
                        float(posterior["posterior_mean"]) - 0.5, 12
                    ),
                }
            )
    head = canonical_value(
        {
            "head": "MECHANISM_INFORMATION",
            "method": "BETA_BINOMIAL_WITH_BOOLEAN_PROBE_DESIGN_CONTRIBUTIONS",
            "authority_matrix_digest": sha256_digest(HEAD_AUTHORITY_MATRIX),
            "input_count": len(inputs),
            "labeled_input_count": len(labeled),
            "missing_assessment_count": len(inputs) - len(labeled),
            "state_counts": dict(
                sorted(Counter(value["mechanism_state"] for value in inputs).items())
            ),
            "global_posterior": global_posterior,
            "feature_stats": feature_stats,
            "prediction_target": (
                "IDENTIFIABLE_MECHANISM_EVIDENCE_UNDER_THE_GIVEN_PROBE_DESIGN_"
                "NOT_NDCG"
            ),
            "resource_status_feature": False,
            "candidate_identity_feature": False,
        }
    )
    return {**head, "head_digest": sha256_digest(head)}


def _weighted_effect_posterior(
    inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prior_weight = 1.0
    total_weight = sum(float(value["evidence_weight"]) for value in inputs)
    weighted_sum = sum(
        float(value["evidence_weight"])
        * float(value["parent_relative_effect"])
        for value in inputs
    )
    posterior_mean = weighted_sum / (prior_weight + total_weight)
    observed_mean = (
        weighted_sum / total_weight if total_weight > 0 else None
    )
    weighted_square_error = sum(
        float(value["evidence_weight"])
        * (float(value["parent_relative_effect"]) - posterior_mean) ** 2
        for value in inputs
    )
    prior_square_error = prior_weight * posterior_mean**2
    variance = (weighted_square_error + prior_square_error) / max(
        1.0, prior_weight + total_weight
    )
    posterior_standard_error = math.sqrt(variance / (prior_weight + total_weight))
    return canonical_value(
        {
            "prior": "ZERO_CENTERED_ONE_EFFECTIVE_OBSERVATION",
            "observation_count": len(inputs),
            "effective_observation_count": round(total_weight, 12),
            "observed_weighted_mean": (
                round(observed_mean, 12) if observed_mean is not None else None
            ),
            "posterior_mean": round(posterior_mean, 12),
            "posterior_standard_error": round(posterior_standard_error, 12),
            "approximate_95_interval": (
                round(posterior_mean - 1.96 * posterior_standard_error, 12),
                round(posterior_mean + 1.96 * posterior_standard_error, 12),
            ),
        }
    )


def _fit_effect_head(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inputs = [
        row["head_inputs"]["effect"]
        for row in rows
        if row["head_authority"]["effect"]["allowed"]
    ]
    if any(
        value["comparability"] != "FULL_MATCHED_FRESH_DEVELOPMENT_EPISODE"
        for value in inputs
    ):
        raise OpenMetaQ3Error("effect head received non-comparable evidence")
    direction_stats = []
    directions = sorted(
        {
            str(value["research_features"].get("direction"))
            for value in inputs
            if value["research_features"].get("direction") is not None
        }
    )
    for direction in directions:
        group = [
            value
            for value in inputs
            if value["research_features"].get("direction") == direction
        ]
        direction_stats.append(
            {"direction": direction, "posterior": _weighted_effect_posterior(group)}
        )
    head = canonical_value(
        {
            "head": "DEVELOPMENT_EFFECT",
            "method": "WEIGHTED_ZERO_CENTERED_NORMAL_SUMMARY",
            "authority_matrix_digest": sha256_digest(HEAD_AUTHORITY_MATRIX),
            "input_count": len(inputs),
            "global_posterior": _weighted_effect_posterior(inputs),
            "direction_stats": direction_stats,
            "inconclusive_episode_weight": 0.5,
            "scientific_superiority_claim": False,
            "resource_status_feature": False,
            "mechanism_probe_only_feature": False,
            "candidate_identity_feature": False,
        }
    )
    return {**head, "head_digest": sha256_digest(head)}


def fit_q3_three_head_policy(
    projection: Mapping[str, Any],
    *,
    parent_policy_digest: str,
    policy_version: str = "research-open-meta-q3-v1.0.0",
    excluded_group: str | None = None,
) -> dict[str, Any]:
    """Fit three transparent heads without forming a shared reward."""

    if projection.get("held_out_reads") != 0:
        raise OpenMetaQ3Error("projection attempted held-out use")
    rows = [
        row
        for row in projection["rows"]
        if excluded_group is None or row["group_id"] != excluded_group
    ]
    if not rows:
        raise OpenMetaQ3Error("three-head learner received no rows")
    feasibility = _fit_feasibility_head(rows)
    mechanism_information = _fit_mechanism_information_head(rows)
    effect = _fit_effect_head(rows)
    policy = canonical_value(
        {
            "schema": "recclaw.research-line.q3-three-head-policy.v1",
            "policy_ref": "policy:research-open-meta-vnext:q3:v1",
            "policy_version": policy_version,
            "policy_mode": "DEVELOPMENT_ONLY_OUTCOME_AWARE_THREE_HEAD",
            "parent_policy_digest": parent_policy_digest,
            "training_projection_digest": projection["projection_digest"],
            "training_row_count": len(rows),
            "training_group_count": len({row["group_id"] for row in rows}),
            "excluded_group": excluded_group,
            "heads": {
                "feasibility": feasibility,
                "mechanism_information": mechanism_information,
                "effect": effect,
            },
            "acquisition_rules": {
                "IDEA": {
                    "formula": (
                        "SCIENTIFIC_FALSIFIABILITY_X_MECHANISM_INFORMATION_"
                        "PROBABILITY_X_FEASIBILITY_PROBABILITY"
                    ),
                    "uses_effect_head": False,
                },
                "EXPERIMENT": {
                    "formula": (
                        "PROBE_DESIGN_COMPLETENESS_X_MECHANISM_INFORMATION_"
                        "PROBABILITY_X_FEASIBILITY_PROBABILITY"
                    ),
                    "uses_effect_head": False,
                },
                "REPLICATION": {
                    "formula": (
                        "EFFECT_INTERVAL_WIDTH_X_COMPARABILITY_X_"
                        "REPRODUCTION_VALUE"
                    ),
                    "uses_effect_mean_as_reward": False,
                },
            },
            "exploration": {
                "method": "PREFROZEN_EPSILON_GREEDY_WITH_RECORDED_THOMPSON_DRAWS",
                "probability": 0.15,
                "finite_pool_realized_fraction_may_differ": True,
            },
            "candidate_identity_feature": False,
            "origin_feature": False,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**policy, "policy_digest": sha256_digest(policy)}


def _stage_posterior(
    feasibility_head: Mapping[str, Any], resource_stage: str
) -> Mapping[str, Any]:
    for value in feasibility_head["stage_stats"]:
        if value["resource_stage"] == resource_stage:
            return value["posterior"]
    return feasibility_head["global_posterior"]


def _prediction_mean(posterior: Mapping[str, Any]) -> float:
    return float(
        posterior.get("shrunk_posterior_mean", posterior["posterior_mean"])
    )


def predict_q3_heads(
    policy: Mapping[str, Any], candidate_features: Mapping[str, Any]
) -> dict[str, Any]:
    """Return separate predictions, uncertainty, and feature contributions."""

    feasibility_head = policy["heads"]["feasibility"]
    resource_stage = str(candidate_features.get("resource_stage", "QUALIFICATION"))
    feasibility_posterior = _stage_posterior(feasibility_head, resource_stage)
    global_feasibility = feasibility_head["global_posterior"]
    feasibility = {
        "posterior_mean": round(_prediction_mean(feasibility_posterior), 12),
        "raw_posterior_mean": feasibility_posterior["posterior_mean"],
        "support_weight": feasibility_posterior.get("support_weight", 1.0),
        "frozen_prior_mean": feasibility_posterior.get(
            "frozen_prior_mean", Q5_FROZEN_FEASIBILITY_PRIOR["mean"]
        ),
        "approximate_95_interval": feasibility_posterior[
            "approximate_95_interval"
        ],
        "alpha": feasibility_posterior["alpha"],
        "beta": feasibility_posterior["beta"],
        "authority": "FEASIBILITY_COMPLETION_HEAD",
        "feature_contributions": {
            "resource_stage": round(
                _prediction_mean(feasibility_posterior)
                - _prediction_mean(global_feasibility),
                12,
            )
        },
    }

    mechanism_head = policy["heads"]["mechanism_information"]
    mechanism_global = mechanism_head["global_posterior"]
    design = candidate_features.get("probe_design_features", {})
    contributions = []
    means = [float(mechanism_global["posterior_mean"])]
    for feature in sorted(design):
        matching = next(
            (
                stat
                for stat in mechanism_head["feature_stats"]
                if stat["feature"] == feature
                and bool(stat["value"]) is bool(design[feature])
            ),
            None,
        )
        if matching is None:
            mean = 0.5
            contribution = 0.0
            evidence_basis = "UNSEEN_VALUE_BETA_1_1_PRIOR"
        else:
            mean = float(matching["posterior"]["posterior_mean"])
            contribution = matching["feature_contribution_vs_beta_prior"]
            evidence_basis = "OBSERVED_PROBE_DESIGN_VALUE"
        means.append(mean)
        contributions.append(
            {
                "feature": feature,
                "value": bool(design[feature]),
                "contribution_vs_beta_prior": contribution,
                "posterior_mean": mean,
                "evidence_basis": evidence_basis,
            }
        )
    mechanism_mean = sum(means) / len(means)
    mechanism_interval = mechanism_global["approximate_95_interval"]
    mechanism_information = {
        "posterior_mean": round(mechanism_mean, 12),
        "approximate_95_interval": mechanism_interval,
        "alpha": mechanism_global["alpha"],
        "beta": mechanism_global["beta"],
        "authority": "MECHANISM_INFORMATION_HEAD_NOT_EFFECT",
        "feature_contributions": contributions,
    }

    effect_head = policy["heads"]["effect"]
    effect_posterior = effect_head["global_posterior"]
    direction = candidate_features.get("direction")
    for stat in effect_head["direction_stats"]:
        if stat["direction"] == direction:
            effect_posterior = stat["posterior"]
            break
    interval = effect_posterior["approximate_95_interval"]
    effect = {
        "posterior_mean": effect_posterior["posterior_mean"],
        "approximate_95_interval": interval,
        "interval_width": round(float(interval[1]) - float(interval[0]), 12),
        "posterior_standard_error": effect_posterior["posterior_standard_error"],
        "authority": "DEVELOPMENT_EFFECT_HEAD_NO_SCIENTIFIC_CLAIM",
        "feature_contributions": {
            "direction_specific": any(
                stat["direction"] == direction for stat in effect_head["direction_stats"]
            )
        },
    }
    return canonical_value(
        {
            "feasibility": feasibility,
            "mechanism_information": mechanism_information,
            "effect": effect,
        }
    )


def run_group_aware_offline_replay(
    projection: Mapping[str, Any], *, parent_policy_digest: str
) -> dict[str, Any]:
    """Leave one evidence group out and score only authority-eligible labels."""

    groups = sorted({str(row["group_id"]) for row in projection["rows"]})
    predictions = []
    for group in groups:
        policy = fit_q3_three_head_policy(
            projection,
            parent_policy_digest=parent_policy_digest,
            policy_version="research-open-meta-q3-v1.0.0-offline",
            excluded_group=group,
        )
        for row in projection["rows"]:
            if row["group_id"] != group:
                continue
            candidate_features: dict[str, Any] = {}
            feasibility_input = row["head_inputs"]["feasibility"]
            mechanism_input = row["head_inputs"]["mechanism_information"]
            effect_input = row["head_inputs"]["effect"]
            if feasibility_input is not None:
                candidate_features["resource_stage"] = feasibility_input[
                    "resource_stage"
                ]
            if mechanism_input is not None:
                candidate_features["probe_design_features"] = mechanism_input[
                    "probe_design_features"
                ]
            if effect_input is not None:
                candidate_features["direction"] = effect_input[
                    "research_features"
                ].get("direction")
            predicted = predict_q3_heads(policy, candidate_features)
            labels = {
                "feasibility": (
                    feasibility_input["completion_label"]
                    if feasibility_input is not None
                    else None
                ),
                "mechanism_information": (
                    mechanism_input["identifiable_information_label"]
                    if mechanism_input is not None
                    else None
                ),
                "effect": (
                    effect_input["parent_relative_effect"]
                    if effect_input is not None
                    else None
                ),
            }
            errors = {}
            for head, label in labels.items():
                errors[head] = (
                    round(
                        abs(float(label) - float(predicted[head]["posterior_mean"])),
                        12,
                    )
                    if label is not None
                    else None
                )
            predictions.append(
                {
                    "left_out_group": group,
                    "row_id": row["row_id"],
                    "excluded_from_training": True,
                    "labels": labels,
                    "predictions": predicted,
                    "absolute_errors": errors,
                }
            )
    metrics = {}
    for head in ("feasibility", "mechanism_information", "effect"):
        values = [
            float(row["absolute_errors"][head])
            for row in predictions
            if row["absolute_errors"][head] is not None
        ]
        metrics[head] = {
            "scored_count": len(values),
            "mean_absolute_error": (
                round(sum(values) / len(values), 12) if values else None
            ),
        }
    replay = canonical_value(
        {
            "schema": "recclaw.research-line.q3-group-aware-offline-replay.v1",
            "method": "LEAVE_ONE_EVIDENCE_GROUP_OUT",
            "group_count": len(groups),
            "row_count": len(projection["rows"]),
            "metrics": metrics,
            "predictions": predictions,
            "small_sample": True,
            "statistical_superiority_claim": False,
            "official_held_out_reads": 0,
            "development_only": True,
        }
    )
    return {**replay, "replay_digest": sha256_digest(replay)}


def project_q1_frozen_pool(pool: Mapping[str, Any]) -> dict[str, Any]:
    """Project the real provider-origin Q1 pre-outcome pool for acquisition."""

    if (
        pool.get("held_out_reads") != 0
        or pool.get("implementation_or_qualification_outcomes_present_when_written")
        != 0
        or pool.get("outcome_fields_consumed") != []
    ):
        raise OpenMetaQ3Error("Q1 pool is not an outcome-blind frozen pool")
    candidates = []
    for pool_arm in sorted(pool["candidate_pools"]):
        for record in pool["candidate_pools"][pool_arm]:
            spec = record["research_spec"]
            score_features = record["preoutcome_score"]["features"]
            falsifiability_fields = (
                spec.get("falsifier"),
                spec.get("matched_control_requirement"),
                spec.get("expected_evidence"),
                spec.get("competing_explanation"),
            )
            probe_fields = falsifiability_fields + (
                spec.get("mechanism_off_definition"),
            )
            graded = "wedge_specificity" in score_features
            fallback_falsifiability = round(
                sum(bool(value) for value in falsifiability_fields)
                / len(falsifiability_fields),
                12,
            )
            scientific_falsifiability = float(
                score_features.get("scientific_falsifiability", fallback_falsifiability)
            )
            probe_design_completeness = (
                round(
                    (
                        float(score_features.get("wedge_specificity", 0.0))
                        + float(score_features.get("mechanism_off_executability", 0.0))
                        + sum(bool(value) for value in probe_fields) / len(probe_fields)
                    )
                    / 3.0,
                    12,
                )
                if graded
                else round(
                    sum(bool(value) for value in probe_fields) / len(probe_fields),
                    12,
                )
            )
            reproduction_value = (
                round(
                    (
                        float(score_features.get("pool_mechanism_novelty", 0.0))
                        + float(score_features.get("wedge_specificity", 0.0))
                        + float(score_features.get("executable_parent", 0.0))
                    )
                    / 3.0,
                    12,
                )
                if graded
                else round(
                    (
                        float(bool(score_features.get("discriminative_value")))
                        + float(bool(score_features.get("scientific_testability")))
                    )
                    / 2.0,
                    12,
                )
            )
            structural_values: dict[str, float] = {}
            selection_structure: float | None = None
            if graded:
                structural_values = {
                    "parent_family_novelty": float(
                        score_features.get("parent_family_novelty", 0.0)
                    ),
                    "causal_operator_novelty": float(
                        score_features.get("causal_operator_novelty", 0.0)
                    ),
                    "qualifier_risk": float(
                        score_features.get("qualifier_risk", 0.5)
                    ),
                    "resource_margin": float(
                        score_features.get("resource_margin", 0.0)
                    ),
                }
                selection_structure = round(
                    (
                        structural_values["parent_family_novelty"]
                        + structural_values["causal_operator_novelty"]
                        + structural_values["qualifier_risk"]
                        + structural_values["resource_margin"]
                        + float(score_features.get("wedge_specificity", 0.0))
                    )
                    / 5.0,
                    12,
                )
            candidates.append(
                canonical_value(
                    {
                        "candidate_id": str(record["preoutcome_score"]["spec_digest"]),
                        "pool_arm": pool_arm,
                        "slot": record["slot"],
                        "direction": record["producer_role"],
                        "resolution": record["resolution"]["resolution"],
                        "resolved_capability_ref": record["resolution"].get(
                            "resolved_current_capability_ref"
                        ),
                        "high_change_dimensions": record["resolution_facts"][
                            "high_change_dimensions"
                        ],
                        "resource_stage": (
                            "PRE_OUTCOME_ESTIMATE"
                            if graded and float(score_features.get("resource_margin", 0.0)) > 0.0
                            else "QUALIFICATION"
                        ),
                        "required_budget": record["resolution_facts"][
                            "required_budget"
                        ],
                        "scientific_falsifiability": scientific_falsifiability,
                        "probe_design_completeness": probe_design_completeness,
                        "comparability": float(
                            bool(spec.get("matched_control_requirement"))
                            and bool(spec.get("protocol_digest"))
                        ),
                        "reproduction_value": reproduction_value,
                        **(
                            {
                                "preoutcome_structural_features": structural_values,
                                "selection_structure": selection_structure,
                            }
                            if graded
                            else {}
                        ),
                        "probe_design_features": {
                            "declared_parent_surface_equivalence_verified": bool(
                                spec.get("parent_equivalence_preverified", False)
                            ),
                            "target_conditioning_observed": bool(
                                spec.get("target_conditioning_preverified", False)
                            ),
                            "structural_routing_participation_observed": bool(
                                spec.get("structural_routing_preverified", False)
                            ),
                            "checkpoint_available_for_discriminative_probe": bool(
                                spec.get("checkpoint_preverified", False)
                            ),
                            "full_ablation_executed": False,
                        },
                        "preoutcome_score_features": score_features,
                        "candidate_identity_feature": False,
                        "origin_feature": False,
                    }
                )
            )
    candidates.sort(key=lambda value: value["candidate_id"])
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q3-origin-blind-pool-projection.v1",
            "source_schema": pool["schema"],
            "source_selection_rule": pool["selection_rule"],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "provider_origin": "REAL_Q1_FROZEN_PRE_OUTCOME_POOL",
            "static_fixture": False,
            "outcome_fields_consumed": (),
            "held_out_reads": 0,
        }
    )
    return {**payload, "pool_digest": sha256_digest(payload)}


def _score_candidate(
    task_type: str,
    candidate: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> tuple[float, dict[str, Any], str]:
    feasibility = float(predictions["feasibility"]["posterior_mean"])
    mechanism = float(predictions["mechanism_information"]["posterior_mean"])
    if task_type == "IDEA":
        terms = {
            "scientific_falsifiability": float(
                candidate["scientific_falsifiability"]
            ),
            "mechanism_information_probability": mechanism,
            "feasibility_probability": feasibility,
        }
        if "selection_structure" in candidate:
            terms["selection_structure"] = float(candidate["selection_structure"])
        score = math.prod(terms.values())
        formula = (
            "SCIENTIFIC_FALSIFIABILITY_X_SELECTION_STRUCTURE_X_"
            "MECHANISM_INFORMATION_PROBABILITY_X_FEASIBILITY_PROBABILITY"
            if "selection_structure" in candidate
            else "SCIENTIFIC_FALSIFIABILITY_X_MECHANISM_INFORMATION_PROBABILITY_"
            "X_FEASIBILITY_PROBABILITY"
        )
    elif task_type == "EXPERIMENT":
        terms = {
            "probe_design_completeness": float(
                candidate["probe_design_completeness"]
            ),
            "mechanism_information_probability": mechanism,
            "feasibility_probability": feasibility,
        }
        score = math.prod(terms.values())
        formula = (
            "PROBE_DESIGN_COMPLETENESS_X_MECHANISM_INFORMATION_PROBABILITY_"
            "X_FEASIBILITY_PROBABILITY"
        )
    elif task_type == "REPLICATION":
        terms = {
            "effect_interval_width": float(predictions["effect"]["interval_width"]),
            "comparability": float(candidate["comparability"]),
            "reproduction_value": float(candidate["reproduction_value"]),
        }
        score = math.prod(terms.values())
        formula = "EFFECT_INTERVAL_WIDTH_X_COMPARABILITY_X_REPRODUCTION_VALUE"
    else:
        raise OpenMetaQ3Error(f"unsupported acquisition task: {task_type}")
    return round(score, 12), canonical_value(terms), formula


def build_q3_acquisition_manifest(
    *,
    policy: Mapping[str, Any],
    activation: Mapping[str, Any],
    frozen_pool: Mapping[str, Any],
    task_type: str,
    random_seed: int,
) -> dict[str, Any]:
    """Consume an active policy against a real full pool and select one item."""

    if (
        activation.get("status") != "ACTIVE_DEVELOPMENT_ONLY"
        or activation.get("policy_digest") != policy.get("policy_digest")
        or activation.get("held_out_reads") != 0
    ):
        raise OpenMetaQ3Error("consumer did not receive a valid active Q3 policy")
    pool = project_q1_frozen_pool(frozen_pool)
    epsilon = float(policy["exploration"]["probability"])
    rng = random.Random(random_seed)
    records = []
    for candidate in pool["candidates"]:
        predictions = predict_q3_heads(policy, candidate)
        score, terms, formula = _score_candidate(task_type, candidate, predictions)
        feasibility_draw = rng.betavariate(
            float(predictions["feasibility"]["alpha"]),
            float(predictions["feasibility"]["beta"]),
        )
        mechanism_draw = rng.betavariate(
            float(predictions["mechanism_information"]["alpha"]),
            float(predictions["mechanism_information"]["beta"]),
        )
        effect_draw = rng.gauss(
            float(predictions["effect"]["posterior_mean"]),
            max(1e-12, float(predictions["effect"]["posterior_standard_error"])),
        )
        records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_features": candidate,
                "head_predictions": predictions,
                "selection_score": score,
                "selection_score_terms": terms,
                "selection_score_formula": formula,
                "thompson_sample": {
                    "feasibility": round(feasibility_draw, 12),
                    "mechanism_information": round(mechanism_draw, 12),
                    "effect": round(effect_draw, 12),
                },
            }
        )
    records.sort(key=lambda value: (-value["selection_score"], value["candidate_id"]))
    if not records:
        raise OpenMetaQ3Error("acquisition pool is empty")
    top_score = float(records[0]["selection_score"])
    top_tie_records = [
        record
        for record in records
        if math.isclose(
            float(record["selection_score"]),
            top_score,
            rel_tol=0.0,
            abs_tol=Q3_SELECTION_SCORE_TIE_TOLERANCE,
        )
    ]
    top_tie_count = len(top_tie_records)
    top_tie_candidate_ids = tuple(
        record["candidate_id"] for record in top_tie_records
    )
    random_draw = rng.random()
    exploratory = random_draw < epsilon
    if exploratory:
        selected_index = rng.randrange(len(records))
        selected_by = "EXPLORATION"
    else:
        selected_index = rng.randrange(top_tie_count)
        selected_by = "UNIFORM_TIE" if top_tie_count > 1 else "LEARNED_SCORE"
    selected_id = records[selected_index]["candidate_id"]
    for index, record in enumerate(records):
        probability = epsilon / len(records)
        in_top_score_tie = index < top_tie_count
        if in_top_score_tie:
            probability += (1.0 - epsilon) / top_tie_count
        record["selection_probability"] = round(probability, 12)
        record["in_top_score_tie"] = in_top_score_tie
        record["selected"] = record["candidate_id"] == selected_id
        record["decision_reason"] = (
            "SELECTED_BY_PREFROZEN_EXPLORATION_DRAW"
            if record["selected"] and selected_by == "EXPLORATION"
            else "SELECTED_BY_UNIFORM_TOP_SCORE_TIE"
            if record["selected"] and selected_by == "UNIFORM_TIE"
            else "SELECTED_BY_TASK_SPECIFIC_POLICY_SCORE"
            if record["selected"]
            else "NOT_SELECTED_DUE_TO_PREFROZEN_EXPLORATION_DRAW"
            if exploratory
            else "NOT_SELECTED_UNIFORM_TOP_SCORE_TIE_DRAW"
            if in_top_score_tie
            else "NOT_SELECTED_LOWER_TASK_SPECIFIC_SCORE"
        )
    manifest = canonical_value(
        {
            "schema": Q3_ACQUISITION_MANIFEST_SCHEMA,
            "task_type": task_type,
            "policy_ref": policy["policy_ref"],
            "policy_version": policy["policy_version"],
            "policy_digest": policy["policy_digest"],
            "activation_digest": activation["activation_digest"],
            "parent_policy_digest": policy["parent_policy_digest"],
            "pool_digest": pool["pool_digest"],
            "pool_provider_origin": pool["provider_origin"],
            "pool_static_fixture": False,
            "candidate_count": len(records),
            "selection_budget": 1,
            "exploration_probability": epsilon,
            "random_seed": random_seed,
            "random_draw": round(random_draw, 12),
            "exploration_selected": exploratory,
            "selected_by": selected_by,
            "selected_candidate_id": selected_id,
            "top_score": round(top_score, 12),
            "top_score_tie_tolerance": Q3_SELECTION_SCORE_TIE_TOLERANCE,
            "top_score_tie_count": top_tie_count,
            "top_score_tie_candidate_ids": top_tie_candidate_ids,
            "candidates": records,
            "selection_probabilities_sum": round(
                sum(float(row["selection_probability"]) for row in records), 12
            ),
            "candidate_identity_feature": False,
            "future_effect_read": False,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**manifest, "acquisition_digest": sha256_digest(manifest)}


def consume_q3_active_policy(
    *,
    policy_path: Path,
    activation_path: Path,
    frozen_pool_path: Path,
    task_type: str,
    random_seed: int,
) -> dict[str, Any]:
    """Read the on-disk active policy and perform the normal pool acquisition."""

    def read_object(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise OpenMetaQ3Error(f"consumer JSON root is not an object: {path}")
        return value

    policy = read_object(policy_path)
    activation = read_object(activation_path)
    frozen_pool = read_object(frozen_pool_path)
    expected_policy_digest = policy.pop("policy_digest", None)
    observed_policy_digest = sha256_digest(policy)
    policy["policy_digest"] = expected_policy_digest
    if expected_policy_digest != observed_policy_digest:
        raise OpenMetaQ3Error("on-disk active policy semantic digest mismatch")
    expected_activation_digest = activation.pop("activation_digest", None)
    observed_activation_digest = sha256_digest(activation)
    activation["activation_digest"] = expected_activation_digest
    if expected_activation_digest != observed_activation_digest:
        raise OpenMetaQ3Error("on-disk activation semantic digest mismatch")
    manifest = build_q3_acquisition_manifest(
        policy=policy,
        activation=activation,
        frozen_pool=frozen_pool,
        task_type=task_type,
        random_seed=random_seed,
    )
    consumer_input = canonical_value(
        {
            "policy_file": str(policy_path),
            "policy_file_sha256": bytes_sha256(policy_path.read_bytes()),
            "activation_file": str(activation_path),
            "activation_file_sha256": bytes_sha256(activation_path.read_bytes()),
            "frozen_pool_file": str(frozen_pool_path),
            "frozen_pool_file_sha256": bytes_sha256(frozen_pool_path.read_bytes()),
            "read_active_policy_from_disk": True,
        }
    )
    payload = {**manifest, "consumer_input": consumer_input}
    payload_without_digest = {
        key: value for key, value in payload.items() if key != "acquisition_digest"
    }
    return {
        **payload_without_digest,
        "acquisition_digest": sha256_digest(payload_without_digest),
    }


def shadow_compare_q3_policy(
    *,
    projection: Mapping[str, Any],
    q1_pool: Mapping[str, Any],
    parent_policy: Mapping[str, Any],
    q3_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare Q3 and F1 controls on identical pools and unit budgets."""

    pool = project_q1_frozen_pool(q1_pool)
    comparisons = []
    parent_direction_order = {
        direction: index for index, direction in enumerate(parent_policy["direction_order"])
    }
    for task_type in ("IDEA", "EXPERIMENT"):
        candidates = []
        for candidate in pool["candidates"]:
            predictions = predict_q3_heads(q3_policy, candidate)
            score, terms, formula = _score_candidate(task_type, candidate, predictions)
            if task_type == "IDEA":
                parent_score = -float(
                    parent_direction_order.get(candidate["direction"], 999)
                )
            else:
                parent_score = 1.0 if candidate["resolved_capability_ref"] else 0.0
            candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "parent_score": parent_score,
                    "q3_score": score,
                    "q3_score_terms": terms,
                    "q3_score_formula": formula,
                    "head_authority_used": (
                        ("feasibility", "mechanism_information")
                    ),
                }
            )
        parent_selected = sorted(
            candidates, key=lambda value: (-value["parent_score"], value["candidate_id"])
        )[0]["candidate_id"]
        q3_selected = sorted(
            candidates, key=lambda value: (-value["q3_score"], value["candidate_id"])
        )[0]["candidate_id"]
        comparisons.append(
            {
                "task_type": task_type,
                "pool_digest": pool["pool_digest"],
                "candidate_count": len(candidates),
                "budget": 1,
                "parent_selected_candidate_id": parent_selected,
                "q3_selected_candidate_id": q3_selected,
                "selection_changed": parent_selected != q3_selected,
                "candidates": candidates,
            }
        )

    effect_rows = [
        row for row in projection["rows"] if row["head_authority"]["effect"]["allowed"]
    ]
    replication_candidates = []
    for row in effect_rows:
        effect_input = row["head_inputs"]["effect"]
        features = {
            "direction": effect_input["research_features"].get("direction"),
            "resource_stage": "DEVELOPMENT_EXPERIMENT",
            "probe_design_features": {},
            "comparability": 1.0,
            "reproduction_value": 1.0,
            "scientific_falsifiability": 1.0,
            "probe_design_completeness": 1.0,
        }
        predictions = predict_q3_heads(q3_policy, features)
        score, terms, formula = _score_candidate("REPLICATION", features, predictions)
        replication_candidates.append(
            {
                "candidate_id": row["row_id"],
                "parent_score": 0.0,
                "q3_score": score,
                "q3_score_terms": terms,
                "q3_score_formula": formula,
                "head_authority_used": ("effect_uncertainty", "comparability"),
            }
        )
    if replication_candidates:
        parent_selected = sorted(
            replication_candidates, key=lambda value: value["candidate_id"]
        )[0]["candidate_id"]
        q3_selected = sorted(
            replication_candidates,
            key=lambda value: (-value["q3_score"], value["candidate_id"]),
        )[0]["candidate_id"]
        comparisons.append(
            {
                "task_type": "REPLICATION",
                "pool_digest": sha256_digest(
                    sorted(value["candidate_id"] for value in replication_candidates)
                ),
                "candidate_count": len(replication_candidates),
                "budget": 1,
                "parent_selected_candidate_id": parent_selected,
                "q3_selected_candidate_id": q3_selected,
                "selection_changed": parent_selected != q3_selected,
                "candidates": replication_candidates,
            }
        )
    shadow = canonical_value(
        {
            "schema": "recclaw.research-line.q3-shadow-comparison.v1",
            "control_policy_digest": parent_policy["policy_digest"],
            "q3_policy_digest": q3_policy["policy_digest"],
            "same_pool_and_budget_within_each_task": True,
            "comparisons": comparisons,
            "authority_coverage": {
                "feasibility": q3_policy["heads"]["feasibility"]["input_count"],
                "mechanism_information": q3_policy["heads"][
                    "mechanism_information"
                ]["input_count"],
                "effect": q3_policy["heads"]["effect"]["input_count"],
            },
            "selection_difference_count": sum(
                bool(value["selection_changed"]) for value in comparisons
            ),
            "policy_superiority_claim": False,
            "scientific_effect_claim": False,
            "held_out_reads": 0,
            "development_only": True,
        }
    )
    return {**shadow, "shadow_digest": sha256_digest(shadow)}


def evaluate_q3_development_activation(
    projection: Mapping[str, Any],
    replay: Mapping[str, Any],
    shadow: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply contract/authority/replay gates before local activation."""

    rows = projection["rows"]
    authority_isolated = all(
        (row["head_inputs"][head] is not None)
        is bool(row["head_authority"][head]["allowed"])
        for row in rows
        for head in ("feasibility", "mechanism_information", "effect")
    )
    gates = {
        "all_denominator_rows_consumed": policy["training_row_count"]
        == projection["row_count"],
        "negative_evidence_preserved": (
            projection["negative_evidence_preserved"]["resource_censored_count"] > 0
            and projection["negative_evidence_preserved"][
                "resource_deferred_count"
            ]
            > 0
            and projection["negative_evidence_preserved"][
                "q2_non_identifiable_count"
            ]
            == 1
        ),
        "head_authority_isolated": authority_isolated,
        "three_real_head_models": set(policy["heads"])
        == {"feasibility", "mechanism_information", "effect"},
        "group_aware_replay_complete": replay["row_count"]
        == projection["row_count"],
        "shadow_same_pool_budget": shadow["same_pool_and_budget_within_each_task"],
        "task_specific_rules_separate": (
            policy["acquisition_rules"]["IDEA"]["uses_effect_head"] is False
            and policy["acquisition_rules"]["EXPERIMENT"]["uses_effect_head"]
            is False
            and policy["acquisition_rules"]["REPLICATION"][
                "uses_effect_mean_as_reward"
            ]
            is False
        ),
        "exploration_prefrozen_at_15_percent": policy["exploration"][
            "probability"
        ]
        == 0.15,
        "held_out_reads_zero": all(
            value["held_out_reads"] == 0
            for value in (projection, shadow, policy)
        )
        and replay["official_held_out_reads"] == 0,
        "development_only_no_scientific_claim": (
            policy["development_only"] is True
            and policy["scientific_effect_claim"] is False
        ),
    }
    status = (
        "DEVELOPMENT_ONLY_ACTIVATION_GATE_PASS"
        if all(gates.values())
        else "DEVELOPMENT_ONLY_ACTIVATION_GATE_FAIL"
    )
    promotion = canonical_value(
        {
            "schema": "recclaw.research-line.q3-development-promotion.v1",
            "gates": gates,
            "status": status,
            "policy_superiority_claim": False,
            "scientific_effect_claim": False,
            "held_out_reads": 0,
            "development_only": True,
        }
    )
    return {**promotion, "promotion_digest": sha256_digest(promotion)}


def build_q3_policy_activation(
    policy: Mapping[str, Any],
    promotion: Mapping[str, Any],
    *,
    projection_digest: str,
    activation_id: str,
) -> dict[str, Any]:
    if promotion["status"] != "DEVELOPMENT_ONLY_ACTIVATION_GATE_PASS":
        raise OpenMetaQ3Error("Q3 policy failed activation gates")
    activation = canonical_value(
        {
            "schema": "recclaw.research-line.q3-policy-activation.v1",
            "status": "ACTIVE_DEVELOPMENT_ONLY",
            "activation_id": activation_id,
            "activation_boundary": "NEXT_FRESH_POOL_ACQUISITION",
            "policy_ref": policy["policy_ref"],
            "policy_version": policy["policy_version"],
            "policy_digest": policy["policy_digest"],
            "parent_policy_digest": policy["parent_policy_digest"],
            "rollback_policy_digest": policy["parent_policy_digest"],
            "training_projection_digest": projection_digest,
            "promotion_digest": promotion["promotion_digest"],
            "reversible": True,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**activation, "activation_digest": sha256_digest(activation)}


__all__ = [
    "HEAD_AUTHORITY_MATRIX",
    "MECHANISM_STATES",
    "OpenMetaQ3Error",
    "Q5_FEASIBILITY_SHRINKAGE_SUPPORT_THRESHOLD",
    "Q5_FROZEN_FEASIBILITY_PRIOR",
    "Q5_STAGE_CONDITIONAL_FEASIBILITY_STAGES",
    "Q3_AUTHORITY_MATRIX_SCHEMA",
    "Q3_PROJECTION_SCHEMA",
    "build_q3_acquisition_manifest",
    "build_q3_denominator_projection",
    "build_q3_policy_activation",
    "consume_q3_active_policy",
    "evaluate_q3_development_activation",
    "fit_q3_three_head_policy",
    "predict_q3_heads",
    "project_f1_outcome_rows",
    "project_f1_replay_rows",
    "project_q1_frozen_pool",
    "project_q2_evidence_row",
    "project_resource_receipt_rows",
    "project_stage_conditional_feasibility",
    "run_group_aware_offline_replay",
    "shadow_compare_q3_policy",
]
