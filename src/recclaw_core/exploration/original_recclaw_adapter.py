"""Development-only adapters between Original RecClaw and Evidence Guard.

The functions in this module are deliberately side-effect free.  They project
an already selected Original RecClaw proposal or runner observation into the
frozen Evidence Guard API and translate the returned diagnostic into bounded
search feedback.  They do not authorize execution, mutate proposals, or write
accepted evidence or claim state.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from .evidence_guard import evaluate_evidence_guard


class OriginalRecClawAdapterError(ValueError):
    """Raised when an Original RecClaw fact packet cannot be projected safely."""


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OriginalRecClawAdapterError(
            "adapter inputs must contain finite canonical JSON values"
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OriginalRecClawAdapterError(f"{label} must be an object")
    result = {str(key): deepcopy(item) for key, item in value.items()}
    _json_bytes(result)
    return result


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OriginalRecClawAdapterError(f"{label} must be a non-empty string")
    return value.strip()


def _string_sequence(value: object, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OriginalRecClawAdapterError(f"{label} must be an array")
    result = [_nonempty(item, f"{label} item") for item in value]
    if not result:
        raise OriginalRecClawAdapterError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise OriginalRecClawAdapterError(f"{label} must contain unique values")
    return result


def _metric_projection(contract: Mapping[str, Any]) -> dict[str, str]:
    projection = _mapping(contract.get("metric_projection"), "metric projection")
    result = {
        _nonempty(claim_name, "metric projection claim name"): _nonempty(
            runner_name, "metric projection runner name"
        )
        for claim_name, runner_name in projection.items()
    }
    if not result:
        raise OriginalRecClawAdapterError("metric projection must not be empty")
    return result


def _planned_seed_ids(
    contract: Mapping[str, Any], packet: Mapping[str, Any]
) -> tuple[list[str], bool]:
    supplied = packet.get("planned_seed_ids")
    if supplied is not None:
        return _string_sequence(supplied, "planned_seed_ids"), True
    seed_policy = _mapping(contract.get("seed_policy"), "seed policy")
    default_seed = _nonempty(
        seed_policy.get("default_training_seed"),
        "seed policy default_training_seed",
    )
    return [default_seed], False


def _contract_parts(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checked = _mapping(contract, "experiment contract")
    claim = _mapping(checked.get("claim"), "experiment contract claim")
    protocol = _mapping(checked.get("protocol"), "experiment contract protocol")
    evidence = _mapping(
        checked.get("current_evidence"), "experiment contract current evidence"
    )
    if checked.get("authority") != "NONE":
        raise OriginalRecClawAdapterError("experiment contract authority must be NONE")
    if checked.get("evidence_class") != "DEVELOPMENT_ONLY":
        raise OriginalRecClawAdapterError(
            "experiment contract evidence_class must be DEVELOPMENT_ONLY"
        )
    if checked.get("formal_acceptance") is not False:
        raise OriginalRecClawAdapterError(
            "experiment contract formal_acceptance must be false"
        )
    return claim, protocol, evidence


def build_precheck_envelope(
    *,
    contract: Mapping[str, Any],
    candidate_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one real Original RecClaw proposal into a Guard pre-check."""

    claim, protocol, evidence = _contract_parts(contract)
    packet = _mapping(candidate_packet, "candidate packet")
    proposal = _mapping(packet.get("proposal"), "candidate proposal")
    candidate_id = _nonempty(packet.get("candidate_id"), "candidate_id")
    if proposal.get("candidate_id") != candidate_id:
        raise OriginalRecClawAdapterError(
            "candidate packet identity differs from proposal candidate_id"
        )

    projection = _mapping(
        packet.get("protocol_projection_evidence"),
        "candidate protocol projection evidence",
    )
    required_projection_facts = {
        "explicit_fixed_dataset",
        "explicit_full_sort_evaluation",
        "explicit_unchanged_split",
        "explicit_unchanged_training_sampling",
        "recbole_core_change_required_false",
        "parent_bound",
        "scaffold_bound",
        "training_seed_bound",
    }
    missing_projection_facts = sorted(
        name for name in required_projection_facts if projection.get(name) is not True
    )

    parent_id = proposal.get("parent_candidate_id")
    scaffold_id = proposal.get("scaffold_id")
    if not isinstance(parent_id, str) or not parent_id:
        missing_projection_facts.append("parent_candidate_id")
    if not isinstance(scaffold_id, str) or not scaffold_id:
        missing_projection_facts.append("scaffold_id")

    seed_ids, proposal_bound_seed = _planned_seed_ids(contract, packet)
    if not proposal_bound_seed:
        missing_projection_facts.append("planned_seed_ids")
    purpose = proposal.get("minimal_experiment") or proposal.get("hypothesis")
    purpose = _nonempty(purpose, "candidate proposal purpose")
    action_proposal = {
        "proposal_id": candidate_id,
        "action_family": "RUN_OFFLINE_TOPN",
        "target_claim_id": claim["claim_id"],
        "protocol_id": protocol["protocol_id"],
        "planned_protocol": deepcopy(protocol),
        "target_model": claim["target_model"],
        "comparator": claim["comparator"],
        "seed_count": len(seed_ids),
        "seed_ids": seed_ids,
        "purpose": purpose,
    }
    recommendation = (
        "REVISE_BEFORE_RUN" if missing_projection_facts else "FORWARD_TO_GUARD"
    )
    return {
        "record_type": "ORIGINAL_RECCLAW_EVIDENCE_GUARD_PRECHECK_ENVELOPE",
        "schema_version": "recclaw.exploration.original_recclaw_guard_precheck.v1",
        "candidate_id": candidate_id,
        "source_candidate_digest": _digest(packet),
        "source_round_id": packet.get("round_id"),
        "source_original_decision": packet.get("original_decision"),
        "source_parent_candidate_id": parent_id,
        "source_scaffold_id": scaffold_id,
        "protocol_projection_evidence": projection,
        "missing_material_fields": sorted(set(missing_projection_facts)),
        "adapter_recommendation": recommendation,
        "guard_request": {
            "mode": "observation",
            "claim": claim,
            "protocol": protocol,
            "current_evidence": evidence,
            "action_proposal": action_proposal,
            "observation": None,
        },
        "original_candidate_edited": False,
        "blocks_original_execution": False,
        "may_authorize_execution": False,
        "may_update_accepted_evidence_history": False,
        "may_update_claim_state": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def build_postcheck_envelope(
    *,
    contract: Mapping[str, Any],
    candidate_packet: Mapping[str, Any],
    precheck_envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one exact runner observation into a Guard post-check."""

    claim, protocol, evidence = _contract_parts(contract)
    packet = _mapping(candidate_packet, "candidate packet")
    precheck = _mapping(precheck_envelope, "precheck envelope")
    candidate_id = _nonempty(packet.get("candidate_id"), "candidate_id")
    if precheck.get("candidate_id") != candidate_id:
        raise OriginalRecClawAdapterError(
            "precheck and candidate packet identities differ"
        )
    runner = _mapping(packet.get("runner_observation"), "runner observation")
    raw_metrics = _mapping(runner.get("metrics"), "runner metrics")
    action_proposal = _mapping(
        _mapping(precheck.get("guard_request"), "precheck guard request").get(
            "action_proposal"
        ),
        "precheck action proposal",
    )
    observed_protocol = _mapping(
        runner.get("observed_protocol"), "observed protocol"
    )
    metric_projection = _metric_projection(contract)
    projected_metrics: dict[str, int | float] = {}
    for claim_name, runner_name in metric_projection.items():
        value = raw_metrics.get(runner_name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            projected_metrics[claim_name] = value

    seed_rows = runner.get("seed_runs")
    if seed_rows is None:
        seed_rows = [
            {
                "seed_id": action_proposal["seed_ids"][0],
                "run_id": runner.get("run_id"),
                "primary_artifact": runner.get("primary_artifact"),
            }
        ]
    if not isinstance(seed_rows, Sequence) or isinstance(seed_rows, (str, bytes)):
        raise OriginalRecClawAdapterError("runner seed_runs must be an array")
    projected_seed_runs: list[dict[str, str]] = []
    source_artifacts: list[dict[str, Any]] = []
    for index, raw_seed_row in enumerate(seed_rows):
        seed_row = _mapping(raw_seed_row, f"runner seed_runs[{index}]")
        artifact = _mapping(
            seed_row.get("primary_artifact"),
            f"runner seed_runs[{index}] primary artifact",
        )
        sha256 = _nonempty(
            artifact.get("sha256"),
            f"runner seed_runs[{index}] primary artifact sha256",
        )
        if len(sha256) != 64 or any(
            char not in "0123456789abcdef" for char in sha256
        ):
            raise OriginalRecClawAdapterError(
                "primary artifact sha256 must be a lowercase SHA-256"
            )
        projected_seed_runs.append(
            {
                "seed_id": _nonempty(
                    seed_row.get("seed_id"), f"runner seed_runs[{index}] seed_id"
                ),
                "run_id": _nonempty(
                    seed_row.get("run_id"), f"runner seed_runs[{index}] run_id"
                ),
                "artifact_sha256": sha256,
            }
        )
        source_artifacts.append(artifact)
    observation = {
        "observation_id": _nonempty(
            runner.get("observation_id"), "runner observation_id"
        ),
        "proposal_id": candidate_id,
        "claim_id": claim["claim_id"],
        "protocol_id": protocol["protocol_id"],
        "observed_protocol": observed_protocol,
        "target_model": claim["target_model"],
        "comparator": claim["comparator"],
        "seed_count": len(projected_seed_runs),
        "seed_runs": projected_seed_runs,
        "observation_kind": _nonempty(
            runner.get("observation_kind"), "runner observation_kind"
        ),
        "run_status": _nonempty(runner.get("run_status"), "runner run_status"),
        "artifact_identity_status": _nonempty(
            runner.get("artifact_identity_status"),
            "runner artifact_identity_status",
        ),
        "evidence_class": "DEVELOPMENT_ONLY",
        "metrics": projected_metrics,
        "notes": [
            f"original_decision={packet.get('original_decision')}",
            f"source_artifacts={len(source_artifacts)}",
        ],
    }
    return {
        "record_type": "ORIGINAL_RECCLAW_EVIDENCE_GUARD_POSTCHECK_ENVELOPE",
        "schema_version": "recclaw.exploration.original_recclaw_guard_postcheck.v1",
        "candidate_id": candidate_id,
        "source_candidate_digest": _digest(packet),
        "source_original_decision": packet.get("original_decision"),
        "source_original_interpretation": deepcopy(
            packet.get("original_interpretation")
        ),
        "source_runner_metrics": raw_metrics,
        "source_artifacts": source_artifacts,
        "source_blocker": deepcopy(runner.get("blocker")),
        "guard_request": {
            "mode": "observation",
            "claim": claim,
            "protocol": protocol,
            "current_evidence": evidence,
            "action_proposal": action_proposal,
            "observation": observation,
        },
        "original_observation_edited": False,
        "blocks_original_execution": False,
        "may_authorize_execution": False,
        "may_update_accepted_evidence_history": False,
        "may_update_claim_state": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def build_precheck_feedback(
    *,
    precheck_envelope: Mapping[str, Any],
    guard_result: Mapping[str, Any] | None,
    failure_kind: str = "GUARD_UNAVAILABLE",
) -> dict[str, Any]:
    """Translate pre-check diagnostics without changing the original decision."""

    envelope = _mapping(precheck_envelope, "precheck envelope")
    candidate_id = _nonempty(envelope.get("candidate_id"), "candidate_id")
    missing = list(envelope.get("missing_material_fields") or [])
    if guard_result is None:
        recommendation = "PRESERVE_ORIGINAL_DECISION_GUARD_UNAVAILABLE"
        reason_codes = [f"{_nonempty(failure_kind, 'failure_kind')}_FAIL_OPEN"]
        branch_required = False
    else:
        result = _mapping(guard_result, "guard result")
        legality = _mapping(result.get("action_legality"), "guard action legality")
        scope = _mapping(result.get("affected_claim_scope"), "guard claim scope")
        reason_codes = list(legality.get("reason_codes") or [])
        branch_required = scope.get("protocol_branch_required") is True
        if missing:
            recommendation = "REVISE_BEFORE_RUN"
        elif branch_required:
            recommendation = "ROUTE_TO_NEW_PROTOCOL_BRANCH"
        elif legality.get("development_verdict") == "LEGAL":
            recommendation = "PROTOCOL_COMPATIBLE_FOR_DEVELOPMENT"
        else:
            recommendation = "REVISE_BEFORE_RUN"
    return {
        "record_type": "ORIGINAL_RECCLAW_EVIDENCE_GUARD_PRECHECK_FEEDBACK",
        "schema_version": "recclaw.exploration.original_recclaw_guard_pre_feedback.v1",
        "candidate_id": candidate_id,
        "recommendation": recommendation,
        "missing_material_fields": missing,
        "protocol_branch_required": branch_required,
        "reason_codes": sorted(set(reason_codes)),
        "original_decision_preserved": True,
        "blocks_original_execution": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def build_postcheck_feedback(
    *,
    contract: Mapping[str, Any],
    postcheck_envelope: Mapping[str, Any],
    guard_result: Mapping[str, Any] | None,
    failure_kind: str = "GUARD_UNAVAILABLE",
) -> dict[str, Any]:
    """Translate post-check diagnostics into bounded next-round feedback."""

    envelope = _mapping(postcheck_envelope, "postcheck envelope")
    candidate_id = _nonempty(envelope.get("candidate_id"), "candidate_id")
    request = _mapping(envelope.get("guard_request"), "postcheck guard request")
    observation = _mapping(request.get("observation"), "postcheck observation")
    if guard_result is None:
        outcome_classification = "INSUFFICIENT_EVIDENCE"
        claim_ceiling = "NO_CLAIM_UPDATE"
        disposition = "GUARD_UNAVAILABLE_FAIL_OPEN"
        reason_codes = [f"{_nonempty(failure_kind, 'failure_kind')}_FAIL_OPEN"]
        memory_channel = "DO_NOT_INGEST_AS_RESEARCH_RESULT"
        branch_required = False
    else:
        result = _mapping(guard_result, "guard result")
        admission = _mapping(
            result.get("evidence_admissibility"), "guard evidence admissibility"
        )
        ceiling = _mapping(result.get("claim_ceiling"), "guard claim ceiling")
        scope = _mapping(result.get("affected_claim_scope"), "guard claim scope")
        disposition = str(admission.get("development_disposition"))
        claim_ceiling = str(ceiling.get("level"))
        reason_codes = list(admission.get("reason_codes") or [])
        branch_required = scope.get("protocol_branch_required") is True
        if branch_required or disposition == "EXCLUDE_FROM_CURRENT_CLAIM":
            outcome_classification = "PROTOCOL_MISMATCH"
            memory_channel = "PROTOCOL_BRANCH_FEEDBACK_ONLY"
        elif disposition == "RECORD_RUNTIME_BLOCKER_ONLY":
            outcome_classification = "RUNTIME_BLOCKER"
            memory_channel = "DIAGNOSTIC_FEEDBACK_ONLY"
        elif disposition in {
            "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL",
            "COUNT_AS_SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL",
        }:
            checked_contract = _mapping(contract, "experiment contract")
            baseline = _mapping(
                checked_contract.get("baseline"), "experiment contract baseline"
            )
            metric_name = str(
                _mapping(checked_contract.get("claim"), "claim")["metric"]
            )
            metric_value = observation.get("metrics", {}).get(metric_name)
            baseline_value = baseline.get("reference_metric_value")
            comparison_policy = _mapping(
                checked_contract.get("comparison_policy"), "comparison policy"
            )
            min_delta = comparison_policy.get("min_improvement_delta")
            if not isinstance(min_delta, (int, float)) or isinstance(min_delta, bool):
                raise OriginalRecClawAdapterError(
                    "comparison policy min_improvement_delta must be numeric"
                )
            if not isinstance(metric_value, (int, float)) or not isinstance(
                baseline_value, (int, float)
            ):
                outcome_classification = "INSUFFICIENT_EVIDENCE"
                memory_channel = "DO_NOT_INGEST_AS_RESEARCH_RESULT"
            elif metric_value - baseline_value > min_delta:
                outcome_classification = "VALID_IMPROVEMENT"
                memory_channel = "PRIMARY_RESEARCH_FEEDBACK"
            else:
                outcome_classification = "INFORMATIVE_NEGATIVE"
                memory_channel = "PRIMARY_RESEARCH_FEEDBACK"
        elif disposition == "RECORD_EXECUTABILITY_ONLY":
            outcome_classification = "INTEGRATION_SIGNAL_ONLY"
            memory_channel = "DIAGNOSTIC_FEEDBACK_ONLY"
        else:
            outcome_classification = "INSUFFICIENT_EVIDENCE"
            memory_channel = "DO_NOT_INGEST_AS_RESEARCH_RESULT"
    primary_search_memory = outcome_classification in {
        "VALID_IMPROVEMENT",
        "INFORMATIVE_NEGATIVE",
    }
    return {
        "record_type": "ORIGINAL_RECCLAW_EVIDENCE_GUARD_POSTCHECK_FEEDBACK",
        "schema_version": "recclaw.exploration.original_recclaw_guard_post_feedback.v1",
        "candidate_id": candidate_id,
        "outcome_classification": outcome_classification,
        "development_evidence_disposition": disposition,
        "claim_ceiling": claim_ceiling,
        "protocol_branch_required": branch_required,
        "reason_codes": sorted(set(reason_codes)),
        "next_memory_channel": memory_channel,
        "may_update_primary_search_memory": primary_search_memory,
        "may_update_diagnostic_memory": memory_channel in {
            "DIAGNOSTIC_FEEDBACK_ONLY",
            "PROTOCOL_BRANCH_FEEDBACK_ONLY",
        },
        "may_update_accepted_evidence_history": False,
        "may_update_claim_state": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


GuardEvaluator = Callable[..., Mapping[str, Any]]


def _evaluate_guard_request(
    request: Mapping[str, Any], evaluator: GuardEvaluator
) -> dict[str, Any]:
    checked = _mapping(request, "guard request")
    result = evaluator(
        claim=checked["claim"],
        protocol=checked["protocol"],
        current_evidence=checked["current_evidence"],
        action_proposal=checked["action_proposal"],
        observation=checked.get("observation"),
    )
    return _mapping(result, "guard result")


def evaluate_precheck_fail_open(
    *,
    contract: Mapping[str, Any],
    candidate_packet: Mapping[str, Any],
    evaluator: GuardEvaluator = evaluate_evidence_guard,
) -> dict[str, Any]:
    """Run the development pre-check without letting Guard failure block Original RecClaw."""

    started = time.perf_counter()
    envelope: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        envelope = build_precheck_envelope(
            contract=contract, candidate_packet=candidate_packet
        )
        result = _evaluate_guard_request(envelope["guard_request"], evaluator)
        feedback = build_precheck_feedback(
            precheck_envelope=envelope, guard_result=result
        )
    except Exception as exc:  # noqa: BLE001 - fail-open boundary is intentional
        failure = {"kind": type(exc).__name__, "message": str(exc)}
        candidate_id = str(candidate_packet.get("candidate_id") or "UNKNOWN")
        if envelope is None:
            feedback = {
                "record_type": "ORIGINAL_RECCLAW_EVIDENCE_GUARD_PRECHECK_FEEDBACK",
                "schema_version": "recclaw.exploration.original_recclaw_guard_pre_feedback.v1",
                "candidate_id": candidate_id,
                "recommendation": "PRESERVE_ORIGINAL_DECISION_GUARD_UNAVAILABLE",
                "missing_material_fields": [],
                "protocol_branch_required": False,
                "reason_codes": ["ADAPTER_OR_GUARD_ERROR_FAIL_OPEN"],
                "original_decision_preserved": True,
                "blocks_original_execution": False,
                "authority": "NONE",
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
            }
        else:
            feedback = build_precheck_feedback(
                precheck_envelope=envelope,
                guard_result=None,
                failure_kind="ADAPTER_OR_GUARD_ERROR",
            )
    return {
        "phase": "PRECHECK",
        "candidate_id": str(candidate_packet.get("candidate_id") or "UNKNOWN"),
        "guard_succeeded": failure is None,
        "guard_latency_ms": round((time.perf_counter() - started) * 1000, 6),
        "envelope": envelope,
        "guard_result": result,
        "feedback": feedback,
        "failure": failure,
        "original_decision_preserved_on_failure": True,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def evaluate_postcheck_fail_open(
    *,
    contract: Mapping[str, Any],
    candidate_packet: Mapping[str, Any],
    precheck_envelope: Mapping[str, Any],
    evaluator: GuardEvaluator = evaluate_evidence_guard,
) -> dict[str, Any]:
    """Run the development post-check without changing the original run outcome."""

    started = time.perf_counter()
    envelope: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        envelope = build_postcheck_envelope(
            contract=contract,
            candidate_packet=candidate_packet,
            precheck_envelope=precheck_envelope,
        )
        result = _evaluate_guard_request(envelope["guard_request"], evaluator)
        feedback = build_postcheck_feedback(
            contract=contract,
            postcheck_envelope=envelope,
            guard_result=result,
        )
    except Exception as exc:  # noqa: BLE001 - fail-open boundary is intentional
        failure = {"kind": type(exc).__name__, "message": str(exc)}
        candidate_id = str(candidate_packet.get("candidate_id") or "UNKNOWN")
        if envelope is None:
            feedback = {
                "record_type": "ORIGINAL_RECCLAW_EVIDENCE_GUARD_POSTCHECK_FEEDBACK",
                "schema_version": "recclaw.exploration.original_recclaw_guard_post_feedback.v1",
                "candidate_id": candidate_id,
                "outcome_classification": "INSUFFICIENT_EVIDENCE",
                "development_evidence_disposition": "GUARD_UNAVAILABLE_FAIL_OPEN",
                "claim_ceiling": "NO_CLAIM_UPDATE",
                "protocol_branch_required": False,
                "reason_codes": ["ADAPTER_OR_GUARD_ERROR_FAIL_OPEN"],
                "next_memory_channel": "DO_NOT_INGEST_AS_RESEARCH_RESULT",
                "may_update_primary_search_memory": False,
                "may_update_diagnostic_memory": False,
                "may_update_accepted_evidence_history": False,
                "may_update_claim_state": False,
                "authority": "NONE",
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_acceptance": False,
            }
        else:
            feedback = build_postcheck_feedback(
                contract=contract,
                postcheck_envelope=envelope,
                guard_result=None,
                failure_kind="ADAPTER_OR_GUARD_ERROR",
            )
    return {
        "phase": "POSTCHECK",
        "candidate_id": str(candidate_packet.get("candidate_id") or "UNKNOWN"),
        "guard_succeeded": failure is None,
        "guard_latency_ms": round((time.perf_counter() - started) * 1000, 6),
        "envelope": envelope,
        "guard_result": result,
        "feedback": feedback,
        "failure": failure,
        "original_observation_preserved": True,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def build_next_round_feedback_event(
    *,
    precheck_run: Mapping[str, Any],
    postcheck_run: Mapping[str, Any] | None,
    original_decision: object,
) -> dict[str, Any]:
    """Create the only Guard record exposed to the next Original RecClaw round."""

    pre = _mapping(precheck_run, "precheck run")
    pre_feedback = _mapping(pre.get("feedback"), "precheck feedback")
    candidate_id = _nonempty(pre.get("candidate_id"), "candidate_id")
    if postcheck_run is None:
        post_feedback = None
    else:
        post = _mapping(postcheck_run, "postcheck run")
        if post.get("candidate_id") != candidate_id:
            raise OriginalRecClawAdapterError(
                "precheck and postcheck candidate identities differ"
            )
        post_feedback = _mapping(post.get("feedback"), "postcheck feedback")

    pre_recommendation = _nonempty(
        pre_feedback.get("recommendation"), "precheck recommendation"
    )
    if pre_recommendation == "REVISE_BEFORE_RUN":
        next_iteration_effect = "REQUEST_PROPOSAL_REVISION"
        channel = "PROPOSAL_REVISION_FEEDBACK"
    elif pre_recommendation == "ROUTE_TO_NEW_PROTOCOL_BRANCH":
        next_iteration_effect = "ROUTE_TO_SEPARATE_PROTOCOL_BRANCH"
        channel = "PROTOCOL_BRANCH_FEEDBACK"
    elif post_feedback is None:
        next_iteration_effect = "NO_PRECHECK_MEMORY_UPDATE"
        channel = "TELEMETRY_ONLY"
    elif post_feedback.get("may_update_primary_search_memory") is True:
        next_iteration_effect = "INGEST_BOUNDED_RESEARCH_FEEDBACK"
        channel = "PRIMARY_RESEARCH_FEEDBACK"
    elif post_feedback.get("may_update_diagnostic_memory") is True:
        next_iteration_effect = "INGEST_DIAGNOSTIC_FEEDBACK_ONLY"
        channel = "DIAGNOSTIC_FEEDBACK"
    else:
        next_iteration_effect = "DO_NOT_INGEST_RESULT"
        channel = "NO_MEMORY_UPDATE"

    return {
        "event": "evidence_guard_development_feedback",
        "schema_version": "recclaw.exploration.original_recclaw_guard_memory_event.v1",
        "candidate_id": candidate_id,
        "original_decision": deepcopy(original_decision),
        "precheck_recommendation": pre_recommendation,
        "postcheck_outcome_classification": (
            post_feedback.get("outcome_classification") if post_feedback else None
        ),
        "development_evidence_disposition": (
            post_feedback.get("development_evidence_disposition")
            if post_feedback
            else None
        ),
        "claim_ceiling": post_feedback.get("claim_ceiling") if post_feedback else None,
        "protocol_branch_required": (
            post_feedback.get("protocol_branch_required")
            if post_feedback
            else pre_feedback.get("protocol_branch_required")
        ),
        "reason_codes": sorted(
            set(pre_feedback.get("reason_codes") or [])
            | set(post_feedback.get("reason_codes") or [] if post_feedback else [])
        ),
        "next_iteration_effect": next_iteration_effect,
        "memory_channel": channel,
        "expose_to_next_iteration": channel
        in {
            "PRIMARY_RESEARCH_FEEDBACK",
            "DIAGNOSTIC_FEEDBACK",
            "PROPOSAL_REVISION_FEEDBACK",
            "PROTOCOL_BRANCH_FEEDBACK",
        },
        "may_update_primary_search_memory": (
            post_feedback.get("may_update_primary_search_memory") is True
            if post_feedback
            else False
        ),
        "may_update_diagnostic_memory": (
            post_feedback.get("may_update_diagnostic_memory") is True
            if post_feedback
            else channel in {"PROPOSAL_REVISION_FEEDBACK", "PROTOCOL_BRANCH_FEEDBACK"}
        ),
        "original_candidate_edited": False,
        "may_authorize_execution": False,
        "may_update_accepted_evidence_history": False,
        "may_update_claim_state": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
